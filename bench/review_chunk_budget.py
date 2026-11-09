"""Bounded synthetic probe of the existing Metal decoded-weight byte cap.

This changes lin._MAX_DECODE_BYTES only in this process. It does not implement
a combined weight/result budget. Run under the session watchdog and GPU lock.
Peak counters are sampled and are lower bounds, not hardware peak statistics.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import random
import statistics
import threading
import time

import torch

from comfy_kitchen.backends.mps import linear as lin


def emit(**data):
    print(json.dumps(data), flush=True)


def memory():
    return torch.mps.driver_allocated_memory(), torch.mps.current_allocated_memory()


def flush():
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()
    torch.mps.synchronize()


def observe_allocations(run, cap):
    """Untimed, same-thread snapshots while decode/GEMM results are still live.

    No synchronization is inserted inside the layer. The hooks record counters
    and shapes only, then release their local references normally. They cannot
    expose vendor-internal temporaries, hence these are still lower bounds.
    """
    original_linear = lin.torch.nn.functional.linear
    original_decode = lin.decode_int8
    baseline = memory()
    checkpoints = []

    def record(stage, **shapes):
        driver, live = memory()
        checkpoints.append(dict(stage=stage, driver_gib=driver / 2**30,
                                live_allocation_lower_bound_gib=live / 2**30, **shapes))

    def linear_observer(input, weight, bias=None):
        result = original_linear(input, weight, bias)
        record("after_gemm_before_release", input_shape=list(input.shape),
               decoded_weight_shape=list(weight.shape), result_shape=list(result.shape))
        return result

    def decode_observer(*args, **kwargs):
        result = original_decode(*args, **kwargs)
        record("after_decode_before_release", decoded_weight_shape=list(result.shape))
        return result

    try:
        lin.torch.nn.functional.linear = linear_observer
        lin.decode_int8 = decode_observer
        result = run(cap)
        record("returned_output_before_sync", result_shape=list(result.shape))
        torch.mps.synchronize()
        record("returned_output_after_sync", result_shape=list(result.shape))
    finally:
        lin.torch.nn.functional.linear = original_linear
        lin.decode_int8 = original_decode
    emit(event="allocation_observation", cap_mib=cap,
         baseline_driver_gib=baseline[0] / 2**30,
         baseline_live_gib=baseline[1] / 2**30,
         checkpoint_driver_peak_gib=max([baseline[0] / 2**30] +
                                       [row["driver_gib"] for row in checkpoints]),
         checkpoint_live_allocation_lower_bound_gib=max([baseline[1] / 2**30] +
                         [row["live_allocation_lower_bound_gib"] for row in checkpoints]),
         observed_gemm_calls=sum(row["stage"] == "after_gemm_before_release" for row in checkpoints),
         timing_valid=False, checkpoints=checkpoints)
    return result


class PeakPoller:
    def __init__(self, interval):
        self.interval = interval
        self.stop = threading.Event()
        self.baseline = memory()
        self.peak = list(self.baseline)
        self.samples = 1
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def read(self):
        current = memory()
        self.peak = [max(old, new) for old, new in zip(self.peak, current)]
        self.samples += 1

    def poll(self):
        while not self.stop.wait(self.interval):
            self.read()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join()
        self.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--n", type=int, default=21504)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--caps-mib", default="512,256,128")
    parser.add_argument("--poll-ms", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=100.0)
    parser.add_argument("--driver-cap-gib", type=float, default=3.5)
    args = parser.parse_args()
    m, k, n = args.m, args.k, args.n
    caps = [int(value) for value in args.caps_mib.split(",")]
    if (not 1 <= m <= 8192 or not 256 <= k <= 7168 or k % 256 or
            not 4 <= n <= 28672 or n % 4):
        raise ValueError("Use M<=8192, K<=7168 divisible by256, N<=28672 divisible by4")
    if not 1 <= args.rounds <= 5 or not caps or min(caps) < 64 or max(caps) > 512 or 512 not in caps:
        raise ValueError("Use 1–5 rounds and caps in [64,512] MiB, including512")
    if args.poll_ms < 1 or not 0 < args.max_seconds <= 110 or not 0 < args.driver_cap_gib < 4:
        raise ValueError("Use poll>=1ms, max-seconds<=110, driver cap<4GiB")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    if k >= lin._TRANSPOSED_DECODE_MIN_K:
        raise ValueError("Allocation-observation hooks require the nontransposed F.linear path")
    torch.set_num_threads(4)
    fraction = args.driver_cap_gib * 2**30 / torch.mps.recommended_max_memory()
    torch.mps.set_per_process_memory_fraction(fraction)
    generator = torch.Generator().manual_seed(20261003)
    rng = random.Random(20261003)
    original_cap = lin._MAX_DECODE_BYTES
    start = time.perf_counter()
    emit(event="metadata", torch=torch.__version__, macos=platform.mac_ver()[0],
         backend=lin.__file__, m=m, k=k, n=n, dtype="bf16", convrot=True,
         seed=20261003, rounds=args.rounds, caps_mib=caps, poll_ms=args.poll_ms,
         max_seconds=args.max_seconds, driver_cap_gib=args.driver_cap_gib,
         policy="existing decoded-weight-only cap; no combined-budget implementation",
         verification="up to64 seeded rows, all output channels; bounded CPU copy",
         memory_observation="timed thread polling plus separate untimed same-thread decode/GEMM checkpoints")
    results = {cap: [] for cap in caps}
    try:
        with torch.inference_mode():
            # Drop source tensors after transfer. No model files or full fp32 weights.
            q = torch.randint(-127, 128, (n, k), generator=generator, dtype=torch.int8).to("mps")
            scales = (torch.rand(n, generator=generator) * 0.0005 + 0.0001).to("mps")
            x = torch.randn(m, k, generator=generator).to(torch.bfloat16).to("mps")
            check_rows = sorted(set([0, min(1, m - 1), m // 4, m // 2, max(0, m - 2), m - 1] +
                                    rng.sample(range(m), min(m, 58))))
            indices = torch.tensor(check_rows, dtype=torch.long, device="mps")

            def run(cap):
                lin._MAX_DECODE_BYTES = cap * 2**20
                return lin.int8_linear(x, q, scales, convrot=True)

            # Warm each distinct chunk shape and save a small baseline on the CPU.
            reference = None
            for cap in caps:
                result = run(cap)
                torch.mps.synchronize()
                if cap == 512:
                    reference = result.index_select(0, indices).cpu().float()
                del result
                flush()
            assert reference is not None
            for round_index in range(args.rounds):
                order = list(caps)
                rng.shuffle(order)
                for cap in order:
                    if time.perf_counter() - start > args.max_seconds:
                        emit(event="time_budget_stop", elapsed_s=time.perf_counter() - start)
                        return
                    flush()
                    with PeakPoller(args.poll_ms / 1000) as poll:
                        tick = time.perf_counter()
                        result = run(cap)
                        torch.mps.synchronize()
                        ms = (time.perf_counter() - tick) * 1000
                    # This extra gather/copy is outside both timing and measured peak.
                    sample = result.index_select(0, indices).cpu().float()
                    diff = sample - reference
                    max_abs = float(diff.abs().max())
                    rel_l2 = float(diff.norm() / reference.norm().clamp_min(1e-30))
                    row_chunk = max(1, cap * 2**20 // (2 * k))
                    if row_chunk > 4:
                        row_chunk -= row_chunk % 4
                    chunk_rows = min(n, row_chunk)
                    entry = dict(event="arm", round=round_index, cap_mib=cap, ms=ms,
                                 baseline_driver_gib=poll.baseline[0] / 2**30,
                                 baseline_live_gib=poll.baseline[1] / 2**30,
                                 peak_driver_gib=poll.peak[0] / 2**30,
                                 peak_live_gib=poll.peak[1] / 2**30,
                                 peak_driver_delta_mib=(poll.peak[0] - poll.baseline[0]) / 2**20,
                                 peak_live_delta_mib=(poll.peak[1] - poll.baseline[1]) / 2**20,
                                 poll_samples=poll.samples, weight_chunk_rows=chunk_rows,
                                 chunks=math.ceil(n / row_chunk),
                                 decoded_weight_mib=chunk_rows * k * 2 / 2**20,
                                 gemm_result_mib=m * chunk_rows * 2 / 2**20,
                                 check_rows=len(check_rows), check_channels=n,
                                 max_abs_vs_512=max_abs, rel_l2_vs_512=rel_l2)
                    emit(**entry)
                    results[cap].append(entry)
                    del result, sample, diff
            for cap, rows in results.items():
                emit(event="summary", cap_mib=cap,
                     median_ms=statistics.median(row["ms"] for row in rows),
                     min_ms=min(row["ms"] for row in rows),
                     max_ms=max(row["ms"] for row in rows),
                     max_peak_driver_gib=max(row["peak_driver_gib"] for row in rows),
                     max_peak_live_gib=max(row["peak_live_gib"] for row in rows),
                     max_abs_vs_512=max(row["max_abs_vs_512"] for row in rows),
                     max_rel_l2_vs_512=max(row["rel_l2_vs_512"] for row in rows))
            # These hooked arms do not contribute to latency summaries above.
            for cap in caps:
                if time.perf_counter() - start > args.max_seconds:
                    emit(event="time_budget_stop", elapsed_s=time.perf_counter() - start)
                    return
                flush()
                result = observe_allocations(run, cap)
                sample = result.index_select(0, indices).cpu().float()
                diff = sample - reference
                emit(event="allocation_observation_accuracy", cap_mib=cap,
                     max_abs_vs_512=float(diff.abs().max()),
                     rel_l2_vs_512=float(diff.norm() / reference.norm().clamp_min(1e-30)))
                del result, sample, diff
            emit(event="complete", elapsed_s=time.perf_counter() - start)
    finally:
        lin._MAX_DECODE_BYTES = original_cap


if __name__ == "__main__":
    main()
