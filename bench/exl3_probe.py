"""Actual-weight EXL3 mul1 operator experiment, not a model runtime.

Default is CPU-only admission. --run requires the existing shared GPU supervisor.
The independent oracle is the pinned ExLlamaV3 tests/testlib/trellis.py supplied by
--oracle; it is loaded only after its digest is checked. No upstream CUDA extension
is imported. Complete-call timers include transforms; reuse variants retain scratch/output.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np

ORACLE_SHA256 = "072b5233f0e1ec39615a70416a865ca3fbf3043669aa7dc9c5597cd132e8d45d"
ORACLE_REVISION = "cdc785db89e263e65e5849cb53e1ada156411414"
LOCK = Path("/tmp/mac-kernels-gpu.lock")
DTYPES = {"int16": np.dtype("<i2"), "float16": np.dtype("<f2")}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bits_twice(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Invalid bitrate")
    twice = int(value * 2)
    if twice != value * 2 or not 4 <= twice <= 12:
        raise ValueError("Probe supports mul1 rates 2..6 in half-bit increments")
    return twice


def validate_matrix(matrix):
    k, n = matrix["k"], matrix["n"]
    if (type(k) is not int or type(n) is not int or min(k, n) <= 0
            or k % 128 or n % 128 or max(k, n) > 32768 or k * n > 200_000_000):
        raise ValueError("Bounded dimensions: positive multiples of 128, <=32768, KN<=200M")
    twice = bits_twice(matrix["bits"])
    if matrix.get("codebook") != "mul1":
        raise ValueError("Only the actual checkpoint's mul1 codebook is admitted")
    for key, shape, dtype in [("trellis", [k // 16, n // 16, twice * 8], "int16"),
                              ("suh", [k], "float16"), ("svh", [n], "float16")]:
        item = matrix[key]
        if item["shape"] != shape or item["dtype"] != dtype:
            raise ValueError(f"Invalid {key} tensor shape or dtype")
        sha = item["sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(f"Invalid {key} digest")
    return twice


def load_matrix(matrix, root):
    validate_matrix(matrix)
    arrays = {}
    for key in ("trellis", "suh", "svh"):
        item = matrix[key]
        path = (Path(root) / item["path"]).resolve()
        dtype = DTYPES[item["dtype"]]
        data = path.read_bytes()
        if len(data) != math.prod(item["shape"]) * dtype.itemsize:
            raise ValueError(f"{key} size mismatch")
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"{key} digest mismatch")
        arrays[key] = np.frombuffer(data, dtype=dtype).reshape(item["shape"]).copy()
        if key != "trellis" and not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} contains nonfinite scales")
    return arrays


def ring_windows(packed, bits):
    """CPU mirror of the shader's two-word extraction, for bit-layout contracts."""
    twice = bits_twice(bits)
    p = np.asarray(packed, dtype=np.int16).view(np.uint16)
    if p.shape[-1] != twice * 8:
        raise ValueError("Packed tile width does not match bitrate")
    words = p[..., 0::2].astype(np.uint32) | (p[..., 1::2].astype(np.uint32) << 16)
    i = np.arange(1, 257, dtype=np.uint32)
    end = i * (twice // 2) + (twice % 2) * (i // 2)
    start = (end + twice * 128 - 16) % (twice * 128)
    at, shift = start // 32, start % 32
    high = words[..., at] << shift
    # uint64 avoids an undefined 32-bit shift when shift == 0.
    low = (words[..., (at + 1) % words.shape[-1]].astype(np.uint64) >> (32 - shift)).astype(np.uint32)
    return ((high | low) >> 16).astype(np.uint16)


def inverse_tile_indices():
    r, c = np.indices((16, 16))
    return (((c & 7) * 4 + ((r & 7) >> 1)) * 8
            + (r & 1) + (r >> 3) * 2 + (c >> 3) * 4)


def nmajor_tiles(packed):
    """Lossless outer-axis transpose; ring bytes and quantized values do not change."""
    p = np.asarray(packed)
    if p.dtype != np.int16 or p.ndim != 3:
        raise ValueError("Expected [K/16,N/16,words] int16 tiles")
    return p.transpose(1, 0, 2).copy()


def mul1_table():
    """All codewords, computed on CPU once; checked against the independent oracle."""
    mixed = (np.arange(65536, dtype=np.uint64) * 0x83DCD12D) & 0xFFFFFFFF
    byte_sum = sum((mixed >> shift) & 255 for shift in (0, 8, 16, 24))
    inv, bias = np.array([0x1EEE, 0xC931], dtype=np.uint16).view(np.float16).astype(np.float64)
    return ((1024 + byte_sum).astype(np.float64) * inv + bias).astype(np.float16)


def require_guard():
    """Reject direct GPU dispatch and stale lock metadata."""
    with LOCK.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(stream, fcntl.LOCK_UN)
            raise RuntimeError("Shared GPU lock is not held")
        state = json.load(stream)
    if state.get("group") != os.getpgrp():
        raise RuntimeError("Run only inside the shared GPU supervisor's process group")


def load_oracle(path):
    if digest(path) != ORACLE_SHA256:
        raise ValueError("Independent oracle digest differs from the pinned upstream source")
    spec = importlib.util.spec_from_file_location("exl3_independent_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def operator_buffers(torch, rows, k, n, device, cache, *, reuse=False, poison=False):
    """Correctness calls poison even reused buffers, so a prior variant cannot supply the answer."""
    if reuse and rows in cache:
        buffers = cache[rows]
    else:
        buffers = (torch.empty((rows, k), dtype=torch.float32, device=device),
                   torch.empty((rows, n), dtype=torch.float32, device=device),
                   torch.empty((rows, n), dtype=torch.float16, device=device))
        if reuse:
            cache[rows] = buffers
    if poison:
        for buffer in buffers:
            buffer.fill_(float("nan"))
    return buffers


class Candidate:
    def __init__(self, torch, arrays, matrix):
        self.torch, self.k, self.n = torch, matrix["k"], matrix["n"]
        self.bits2 = validate_matrix(matrix)
        self.lib = torch.mps.compile_shader(Path(__file__).with_suffix(".metal").read_text())
        self.packed = torch.from_numpy(arrays["trellis"]).to("mps")
        began = time.perf_counter()
        nmajor = nmajor_tiles(arrays["trellis"])
        repack_ms = (time.perf_counter() - began) * 1000
        self.preparation = {"cpu_repack_ms": repack_ms, "nmajor_bytes": nmajor.nbytes,
                            "nmajor_sha256": hashlib.sha256(nmajor).hexdigest(), "lookup_bytes": 131072,
                            "scope": "Lossless tile-axis transpose and LUT prepared/uploaded outside timers; both original and transposed packed copies stay resident for controlled comparisons."}
        self.nmajor = torch.from_numpy(nmajor).to("mps")
        self.lut = torch.from_numpy(mul1_table()).to("mps")
        self.suh = torch.from_numpy(arrays["suh"]).to("mps")
        self.svh = torch.from_numpy(arrays["svh"]).to("mps")
        self.scratch = {}

    def __call__(self, x, mode="fused", reuse=False, poison=False):
        t = self.torch
        if (x.device.type != "mps" or x.dtype != t.float16 or not x.is_contiguous()
                or x.ndim != 2 or x.shape[1] != self.k or not 1 <= x.shape[0] <= 512):
            raise ValueError("Expected contiguous MPS half input, 1..512 rows")
        if mode not in ("fused", "reconstruct", "nmajor_arithmetic", "nmajor_lut"):
            raise ValueError("Unknown execution mode")
        rows = x.shape[0]
        if reuse and mode == "reconstruct":
            raise ValueError("Reusable output/scratch is implemented only for fused paths")
        xh, rotated, out = operator_buffers(t, rows, self.k, self.n, "mps", self.scratch,
                                            reuse=reuse, poison=poison)
        self.lib.exl3_input(xh, x, self.suh, self.k,
                            threads=(self.k, rows), group_size=(128, 1))
        if mode == "fused":
            self.lib.exl3_matvec(rotated, xh, self.packed, self.k, self.n, self.bits2,
                                 threads=(256, self.n, rows), group_size=(256, 1, 1))
        elif mode.startswith("nmajor_"):
            kernel = getattr(self.lib, "exl3_" + mode)
            kernel(rotated, xh, self.nmajor, self.lut, self.k, self.n, self.bits2,
                   threads=(256, self.n, rows), group_size=(256, 1, 1))
        else:
            weight = t.empty((self.k, self.n), dtype=t.float16, device="mps")
            if poison:
                weight.fill_(float("nan"))
            self.lib.exl3_unpack(weight, self.packed, self.k, self.n, self.bits2,
                                 threads=self.k * self.n, group_size=256)
            # This fallback adds half rounding at the transformed activation and GEMM output.
            rotated = (xh.half() @ weight).float()
        self.lib.exl3_output(out, rotated, self.svh, self.n,
                             threads=(self.n, rows), group_size=(128, 1))
        return out


def error_metrics(got, want):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    if got.shape != want.shape or not np.isfinite(got).all() or not np.isfinite(want).all():
        return {"pass": False, "finite": False}
    diff = got - want
    relative_rms = float(np.linalg.norm(diff) / max(np.linalg.norm(want), 1e-30))
    max_scaled = float(np.max(np.abs(diff)) / max(np.max(np.abs(want)), 1e-30))
    return {"pass": relative_rms < .0025 and max_scaled < .004, "finite": True,
            "relative_rms": relative_rms, "max_scaled": max_scaled}


def save(path, report):
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def run(args, manifest):
    require_guard()
    import torch
    torch.set_num_threads(1)
    oracle = load_oracle(args.oracle)
    # Exhaust every codeword, independently of the random/actual tensor fixture.
    np.testing.assert_array_equal(mul1_table().view(np.uint16),
                                  oracle.decode(np.arange(65536), "mul1").view(np.uint16))
    report = {"status": "running", "started": time.time(), "torch": torch.__version__,
              "manifest_sha256": digest(args.assets), "oracle_sha256": ORACLE_SHA256,
              "oracle_revision": ORACLE_REVISION, "source_sha256": digest(__file__),
              "shader_sha256": digest(Path(__file__).with_suffix(".metal")), "measurements": [],
              "scope": "Actual packed operators with synthetic activations; no full-model or GGUF speed claim.",
              "timing_scope": "Resident weights/pipeline; all calls include transforms and synchronization. fused_reuse and nmajor variants reuse scratch/output; other paths allocate each call. Python dispatch remains included.",
              "event_timing": "Not collected. Optional torch.mps.Event diagnostics were removed after the installed Torch 2.14.1 elapsed-time wait hung. Wall measurements are not pure GPU kernel durations.",
              "eviction_bytes": args.evict_mib * 2**20,
              "cache_treatment": ("Before every timed call, multiply a separate reusable FP32 MPS buffer and synchronize outside the stopwatch. This applies cache pressure; it does not prove a perfectly cold cache."
                                  if args.evict_mib else "No eviction treatment; repeated resident calls may benefit from cache.")}
    save(args.output, report)
    rng = random.Random(20261025)
    try:
        with torch.inference_mode():
            eviction = (torch.ones(args.evict_mib * 2**20 // 4, dtype=torch.float32, device="mps")
                        if args.evict_mib else None)
            for matrix in manifest["matrices"]:
                if args.matrix and matrix["name"] not in args.matrix:
                    continue
                arrays = load_matrix(matrix, args.assets.parent)
                cpu = {k: torch.from_numpy(v) for k, v in arrays.items()}
                weight64 = oracle.dequant(cpu["trellis"], cpu["suh"], cpu["svh"], matrix["bits"], "mul1")
                if not bool(torch.isfinite(weight64).all()):
                    raise ValueError("Oracle produced nonfinite reconstructed weights")
                candidate = Candidate(torch, arrays, matrix)
                resident = weight64.half().to("mps")
                for rows in args.rows:
                    # Exact dyadic half inputs shared with the actual GGUF baseline.
                    indices = torch.arange(rows * matrix["k"], dtype=torch.int64)
                    xcpu = (((17 * indices + 13) % 257 - 128).float() / 128).half().reshape(rows, matrix["k"])
                    want = (xcpu.double() @ weight64).numpy()
                    x = xcpu.to("mps")
                    variants = {"fused": ("fused", False), "fused_reuse": ("fused", True),
                                "nmajor_arithmetic": ("nmajor_arithmetic", True),
                                "nmajor_lut": ("nmajor_lut", True), "reconstruct": ("reconstruct", False)}
                    funcs = {name: lambda mode=mode, reuse=reuse: candidate(x, mode, reuse=reuse)
                             for name, (mode, reuse) in variants.items()}
                    funcs["resident_fp16"] = lambda: x @ resident
                    item = {"matrix": matrix["name"], "k": matrix["k"], "n": matrix["n"],
                            "bits": matrix["bits"], "rows": rows, "preparation": candidate.preparation,
                            "correctness": {}, "samples_ms": {}}
                    for name, fn in funcs.items():
                        if name in variants:
                            mode, reuse = variants[name]
                            result = candidate(x, mode, reuse=reuse, poison=True)
                        else:
                            result = torch.full((rows, matrix["n"]), float("nan"), dtype=x.dtype, device=x.device)
                            torch.mm(x, resident, out=result)
                        got = result.cpu().numpy()
                        item["correctness"][name] = error_metrics(got, want)
                    # Neither a fast incorrect operator nor an inaccurate diagnostic baseline earns timing.
                    if not all(v["pass"] for v in item["correctness"].values()):
                        report["measurements"].append(item)
                        raise AssertionError("Numerical gate failed")
                    # Zero input checks nonfinite scale/accumulator handling; the full random output above
                    # covers every channel, including the complete output Hadamard blocks.
                    for name, (mode, reuse) in variants.items():
                        zero = candidate(torch.zeros_like(x), mode, reuse=reuse, poison=True).cpu().numpy()
                        if not np.isfinite(zero).all() or np.any(zero):
                            raise AssertionError(f"{name} zero-input gate failed")
                    for fn in funcs.values():
                        fn(); torch.mps.synchronize()
                    samples = {name: [] for name in funcs}
                    for _ in range(args.rounds):
                        order = list(funcs); rng.shuffle(order)
                        for name in order:
                            if eviction is not None:
                                # An actual read/write of distinct storage, never part of the timed operator.
                                eviction.mul_(1.000001)
                            torch.mps.synchronize()
                            began = time.perf_counter()
                            out = funcs[name]()
                            torch.mps.synchronize()
                            samples[name].append((time.perf_counter() - began) * 1000)
                            del out
                    item.update(samples_ms=samples,
                                median_ms={k: statistics.median(v) for k, v in samples.items()},
                                live_gib=torch.mps.current_allocated_memory() / 2**30,
                                driver_gib=torch.mps.driver_allocated_memory() / 2**30)
                    report["measurements"].append(item); save(args.output, report)
                    print(json.dumps(item), flush=True)
                    del funcs, x, xcpu
                del candidate, resident, weight64, cpu, arrays
                torch.mps.synchronize(); torch.mps.empty_cache()
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"; report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished"] = time.time(); save(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, default=Path(__file__).with_name("exl3_reference.py"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--matrix", action="append")
    parser.add_argument("--rows", default="1,4")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--evict-mib", type=int, choices=(0, 128), default=0,
                        help="Untimed cache-pressure buffer before each sample; not a cold-cache guarantee")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    args.rows = [int(x) for x in args.rows.split(",")]
    if not args.rows or len(args.rows) > 6 or any(x not in (1, 4, 16, 128, 512) for x in args.rows) or not 3 <= args.rounds <= 20:
        parser.error("Rows from 1,4,16,128,512; at most six row counts; rounds 3..20")
    manifest = json.loads(args.assets.read_text())
    if not 1 <= len(manifest["matrices"]) <= 8:
        parser.error("Admit 1..8 matrices per bounded experiment")
    names = [m["name"] for m in manifest["matrices"]]
    if len(set(names)) != len(names) or (args.matrix and not set(args.matrix) <= set(names)):
        parser.error("Matrix names must be unique and selected names must exist")
    for matrix in manifest["matrices"]:
        load_matrix(matrix, args.assets.parent)
    if digest(args.oracle) != ORACLE_SHA256:
        parser.error("Oracle is not the pinned independent upstream reference")
    if not args.run:
        print(json.dumps({"status": "CPU admission only; no GPU dispatch", "matrices": names,
                          "rows": args.rows, "eviction_bytes": args.evict_mib * 2**20,
                          "oracle_sha256": ORACLE_SHA256})); return
    if args.output is None or args.output.exists():
        parser.error("--run requires a new --output receipt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args, manifest)


if __name__ == "__main__":
    main()
