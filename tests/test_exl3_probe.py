"""CPU contracts only: real packed bit layout, asset identity and GPU admission."""
import fcntl
import hashlib
import json
import os

import numpy as np
import pytest

from bench import exl3_probe as probe


def matrix(k=128, n=128, bits=3.5):
    return {"name": "fixture", "k": k, "n": n, "bits": bits, "codebook": "mul1",
            **{name: {"path": name + ".bin", "sha256": "0" * 64, "shape": shape, "dtype": dtype}
               for name, shape, dtype in [("trellis", [k // 16, n // 16, int(bits * 16)], "int16"),
                                           ("suh", [k], "float16"), ("svh", [n], "float16")]}}


@pytest.mark.parametrize("bits", [2, 2.5, 3, 3.5, 4, 6])
def test_two_word_extraction_matches_bit_serial_ring(bits):
    packed = np.random.default_rng(17).integers(0, 65536, (3, int(bits * 16)), dtype=np.uint16)
    actual = probe.ring_windows(packed, bits)
    expected = np.zeros((3, 256), dtype=np.uint16)
    for tile_index, tile in enumerate(packed):
        words = [int(tile[i]) | (int(tile[i + 1]) << 16) for i in range(0, len(tile), 2)]
        ring = [(word >> b) & 1 for word in words for b in range(31, -1, -1)]
        end = 0
        for i in range(256):
            end += int(bits) + (int(bits * 2) % 2) * (i % 2)
            value = 0
            for position in range(end - 16, end):
                value = (value << 1) | ring[position % len(ring)]
            expected[tile_index, i] = value
    np.testing.assert_array_equal(actual, expected)


def test_inverse_tile_map_covers_every_fragment_once():
    inverse = probe.inverse_tile_indices()
    for thread in range(32):
        for fragment in range(8):
            row = (thread % 4) * 2 + (fragment & 1) + (8 if fragment & 2 else 0)
            column = thread // 4 + (8 if fragment & 4 else 0)
            assert inverse[row, column] == thread * 8 + fragment
    assert sorted(inverse.ravel()) == list(range(256))


def test_nmajor_repacking_preserves_each_complete_fractional_ring():
    original = np.random.default_rng(29).integers(-32768, 32768, (3, 5, 56), dtype=np.int16)
    before = original.copy()
    repacked = probe.nmajor_tiles(original)
    assert repacked.flags.c_contiguous and repacked.shape == (5, 3, 56)
    np.testing.assert_array_equal(repacked.transpose(1, 0, 2), original)
    np.testing.assert_array_equal(original, before)
    np.testing.assert_array_equal(probe.ring_windows(repacked, 3.5).transpose(1, 0, 2),
                                  probe.ring_windows(original, 3.5))


def test_mul1_lookup_matches_independent_oracle_for_every_codeword():
    from bench import exl3_reference
    lookup = probe.mul1_table()
    expected = exl3_reference.decode(np.arange(65536), "mul1")
    assert lookup.dtype == np.float16 and lookup.nbytes == 131072
    np.testing.assert_array_equal(lookup.view(np.uint16), expected.view(np.uint16))


def test_actual_down_projection_and_fractional_width_are_admitted():
    assert probe.validate_matrix(matrix(17408, 5120, 3.5)) == 7
    bad = matrix(); bad["trellis"]["shape"][-1] = 48
    with pytest.raises(ValueError, match="shape"):
        probe.validate_matrix(bad)
    for bits in (float("nan"), float("inf"), True, 3.25, 1.5, 7):
        with pytest.raises(ValueError):
            probe.bits_twice(bits)


def test_admission_checks_payload_bytes_and_finite_scales(tmp_path):
    entry = matrix()
    for name in ("trellis", "suh", "svh"):
        item = entry[name]
        payload = np.zeros(item["shape"], dtype=probe.DTYPES[item["dtype"]]).tobytes()
        (tmp_path / item["path"]).write_bytes(payload)
        item["sha256"] = hashlib.sha256(payload).hexdigest()
    probe.load_matrix(entry, tmp_path)
    path = tmp_path / entry["suh"]["path"]
    payload = np.full(128, np.inf, dtype=np.float16).tobytes()
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="digest"):
        probe.load_matrix(entry, tmp_path)
    entry["suh"]["sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="nonfinite"):
        probe.load_matrix(entry, tmp_path)
    path.write_bytes(payload[:-1])
    with pytest.raises(ValueError, match="size"):
        probe.load_matrix(entry, tmp_path)


def test_guard_rejects_unlocked_or_foreign_group(tmp_path, monkeypatch):
    lock = tmp_path / "gpu.lock"
    monkeypatch.setattr(probe, "LOCK", lock)
    lock.write_text(json.dumps({"group": os.getpgrp()}))
    with pytest.raises(RuntimeError, match="not held"):
        probe.require_guard()
    with lock.open("r+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX)
        probe.require_guard()
        owner.seek(0); owner.truncate(); owner.write(json.dumps({"group": -1})); owner.flush()
        with pytest.raises(RuntimeError, match="process group"):
            probe.require_guard()


def test_oracle_identity_checked_before_import(tmp_path):
    script = tmp_path / "oracle.py"
    script.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="digest"):
        probe.load_oracle(script)


def test_correctness_rejects_nonfinite_corrupted_and_zero_errors():
    assert probe.error_metrics(np.zeros(8), np.zeros(8))["pass"]
    assert not probe.error_metrics(np.ones(8), np.zeros(8))["pass"]
    assert not probe.error_metrics(np.full(8, np.nan), np.ones(8))["pass"]
    assert not probe.error_metrics(np.ones(8), np.ones(8) * 2)["pass"]


def test_correctness_poison_rejects_reused_nonwriting_and_partial_writing_paths():
    import torch
    cache = {}
    buffers = probe.operator_buffers(torch, 1, 128, 128, "cpu", cache, reuse=True)
    # A successful earlier variant left a perfectly correct answer in every buffer.
    for buffer in buffers:
        buffer.fill_(1)
    want = np.ones((1, 128))
    assert probe.error_metrics(buffers[-1].numpy(), want)["pass"]

    reused = probe.operator_buffers(torch, 1, 128, 128, "cpu", cache, reuse=True, poison=True)
    assert all(a is b for a, b in zip(buffers, reused))
    assert all(bool(torch.isnan(buffer).all()) for buffer in reused)
    # A missing output dispatch, and then a partial dispatch, must both be rejected.
    assert not probe.error_metrics(reused[-1].numpy(), want)["pass"]
    reused[-1][:, :64] = 1
    assert not probe.error_metrics(reused[-1].numpy(), want)["pass"]
    reused[-1].fill_(1)
    assert probe.error_metrics(reused[-1].numpy(), want)["pass"]
    # The next variant must re-poison the same storage, not just initialize it once.
    again = probe.operator_buffers(torch, 1, 128, 128, "cpu", cache, reuse=True, poison=True)
    assert all(bool(torch.isnan(buffer).all()) for buffer in again)


def test_fresh_correctness_buffers_are_poisoned_too():
    import torch
    buffers = probe.operator_buffers(torch, 1, 128, 128, "cpu", {}, poison=True)
    assert all(bool(torch.isnan(buffer).all()) for buffer in buffers)
