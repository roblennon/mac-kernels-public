"""Bounded, warm-cache safetensors loading comparison; never loads a model.

Provide one checkpoint path. Selects a single I8/U8 tensor from its header,
preferably 128–256 MiB, and compares CPU mmap->MPS with MPS pread in ABBA
order, after one untimed warmup per loader. Runs only when invoked, through the project GPU lock and an external
timeout. No model construction, shader compilation, checkpoint writes, or
custom host aliasing. Exact CPU byte parity is checked outside timed loading.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
import statistics
import struct
import sys
import time

MIB = 1024**2
MAX_BYTES = 256 * MIB


def select_tensor(path: Path, name: str | None) -> tuple[str, dict]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Incomplete safetensors header")
        header_size = struct.unpack("<Q", raw)[0]
        if header_size > 16 * MIB or header_size + 8 > size:
            raise ValueError("Invalid or unexpectedly large safetensors header")
        header = json.loads(stream.read(header_size))
    candidates = []
    for key, info in header.items():
        if key == "__metadata__" or (name is not None and key != name):
            continue
        if info.get("dtype") not in ("I8", "U8"):
            continue
        begin, end = info["data_offsets"]
        nbytes = end - begin
        if 0 <= begin < end <= size - header_size - 8 and nbytes <= MAX_BYTES:
            candidates.append((nbytes, key, info))
    if not candidates:
        raise ValueError("No selected I8/U8 tensor at or below 256 MiB")
    preferred = [entry for entry in candidates if entry[0] >= 128 * MIB]
    _, key, info = max(preferred or candidates, key=lambda entry: (entry[0], entry[1]))
    return key, {**info, "absolute_offset": header_size + 8 + info["data_offsets"][0]}


def prewarm_bytes(path: Path, offset: int, count: int) -> None:
    """Touch only selected data with an at-most-8-MiB host buffer."""
    with path.open("rb", buffering=0) as stream:
        stream.seek(offset)
        buf = bytearray(min(8 * MIB, count))
        while count:
            nread = stream.readinto(memoryview(buf)[:min(len(buf), count)])
            if not nread:
                raise ValueError("Unexpected EOF while warming selected tensor")
            count -= nread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tensor", help="Optional exact I8/U8 tensor key")
    parser.add_argument("--cycles", type=int, default=3, choices=range(1, 11),
                        help="Measured ABBA cycles after warmup (default: 3; max: 10)")
    args = parser.parse_args()
    path = args.checkpoint.expanduser().resolve(strict=True)
    name, info = select_tensor(path, args.tensor)
    nbytes = info["data_offsets"][1] - info["data_offsets"][0]

    # Imports occur only after header size and tensor-size bounds are checked.
    import psutil
    import safetensors
    from safetensors import safe_open
    import torch

    if sys.platform != "darwin" or not torch.backends.mps.is_available():
        raise RuntimeError("This comparison requires macOS and available MPS")
    if psutil.virtual_memory().available < 4 * 1024**3:
        raise RuntimeError("Less than 4 GiB available; postpone this optional probe")

    process = psutil.Process()

    def snapshot() -> dict:
        return {
            "mps_active_mib": round(torch.mps.current_allocated_memory() / MIB, 2),
            "mps_driver_mib": round(torch.mps.driver_allocated_memory() / MIB, 2),
            "rss_mib": round(process.memory_info().rss / MIB, 2),
            "available_gib": round(psutil.virtual_memory().available / 1024**3, 2),
        }

    def release() -> None:
        torch.mps.synchronize()
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()

    print("CONFIG " + json.dumps({
        "checkpoint": str(path), "tensor": name, "shape": info["shape"],
        "dtype": info["dtype"], "tensor_mib": round(nbytes / MIB, 2),
        "torch": torch.__version__, "safetensors": safetensors.__version__,
        "cache_state": "selected bytes prewarmed; not a cold-load benchmark",
        "order": ["cpu_mmap_to_mps", "mps_pread", "mps_pread", "cpu_mmap_to_mps"],
        "cycles": args.cycles, "measured_trials_per_loader": 2 * args.cycles,
        "untimed_warmups_per_loader": 1, "external_timeout_required": True,
    }), flush=True)
    prewarm_bytes(path, info["absolute_offset"], nbytes)
    release()
    baseline = snapshot()
    rows = []
    modes = ("cpu_mmap_to_mps", "mps_pread")
    order = (modes[0], modes[1], modes[1], modes[0])
    plan = [(True, index, mode) for index, mode in enumerate(modes)]
    plan += [(False, index, mode) for index, mode in enumerate(order * args.cycles)]
    for warmup, index, mode in plan:
        if psutil.virtual_memory().available < 4 * 1024**3:
            raise RuntimeError("Headroom fell below 4 GiB; no further arms started")
        before = snapshot()
        start = None if warmup else time.perf_counter()
        if mode == "cpu_mmap_to_mps":
            with safe_open(path, framework="pt", device="cpu", backend="mmap") as handle:
                source = handle.get_tensor(name)
                tensor = source.to("mps")
                torch.mps.synchronize()
                del source
        else:
            with safe_open(path, framework="pt", device="mps", backend="pread") as handle:
                tensor = handle.get_tensor(name)
                torch.mps.synchronize()
        # The reader has closed. The returned tensor must retain valid storage.
        elapsed = None if warmup else time.perf_counter() - start
        del handle
        gc.collect()
        allocated = snapshot()
        if tensor.device.type != "mps" or list(tensor.shape) != info["shape"]:
            raise AssertionError("Loaded tensor has unexpected device or shape")
        expected_dtype = torch.int8 if info["dtype"] == "I8" else torch.uint8
        if tensor.dtype != expected_dtype:
            raise AssertionError("Loaded tensor dtype differs from packed source")
        copied_back = tensor.cpu()
        with safe_open(path, framework="pt", device="cpu", backend="mmap") as check:
            reference = check.get_tensor(name)
            equal = torch.equal(copied_back.view(torch.uint8), reference.view(torch.uint8))
            del reference
        del check, copied_back, tensor
        release()
        row = {"trial": index, "mode": mode,
               "exact_byte_parity_after_reader_close": equal,
               "before": before, "allocated": allocated, "released": snapshot()}
        if warmup:
            print("WARMUP " + json.dumps(row), flush=True)
        else:
            row.update(load_seconds=round(elapsed, 6),
                       effective_gib_s=round(nbytes / elapsed / 1024**3, 3),
                       cycle=index // len(order))
            rows.append(row)
            print("TRIAL " + json.dumps(row), flush=True)
        if not equal:
            raise AssertionError("Packed tensor bytes changed during loading")
    medians = {mode: statistics.median(row["load_seconds"] for row in rows if row["mode"] == mode)
               for mode in ("cpu_mmap_to_mps", "mps_pread")}
    ranges = {}
    for mode in modes:
        values = [row["load_seconds"] for row in rows if row["mode"] == mode]
        ranges[mode] = {"min": min(values), "median": statistics.median(values),
                        "max": max(values), "count": len(values)}
    print("RESULT " + json.dumps({
        "kind": "bounded_warm_load_probe", "all_exact": True,
        "median_seconds": medians, "load_seconds_min_median_max": ranges,
        "mmap_over_pread_ratio": medians["cpu_mmap_to_mps"] / medians["mps_pread"],
        "baseline": baseline, "released": snapshot(),
        "process_maxrss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MIB, 2),
        "scope_limit": "One tensor, warm file data; no full-model or cold-load speed claim",
    }), flush=True)


if __name__ == "__main__":
    main()
