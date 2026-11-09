"""Bounded review benchmark: separate decode cost from small/large-M GEMM cost.

Run under bench/with_gpu.sh and the session watchdog. Does not load models or
change backend behavior. Synthetic INT8 ConvRot weights; at most one shape lives
on the GPU. Compilation and one warmup per operation are outside timings.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import statistics
import time

import torch

from comfy_kitchen.backends.mps import linear as lin
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation


def emit(**data):
    print(json.dumps(data), flush=True)


def timed(fn):
    torch.mps.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.mps.synchronize()
    duration = (time.perf_counter() - start) * 1000
    del result
    return duration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--tokens", default="1,16,64,256,4608")
    parser.add_argument("--shapes", default="4096,4096;6144,16384")
    parser.add_argument("--max-seconds", type=float, default=150)
    args = parser.parse_args()
    tokens = [int(v) for v in args.tokens.split(",")]
    shapes = [tuple(int(v) for v in item.split(",")) for item in args.shapes.split(";")]
    if not 1 <= args.rounds <= 20 or not tokens or min(tokens) < 1 or max(tokens) > 4608:
        parser.error("Use rounds in [1, 20] and token counts in [1, 4608]")
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 240:
        parser.error("max-seconds must be finite and in (0, 240]")
    if os.environ.get("COMFY_KITCHEN_MPS_COMPUTE_DTYPE", "float16") != "float16":
        parser.error("The decomposition requires COMFY_KITCHEN_MPS_COMPUTE_DTYPE unset or float16")
    if any(len(s) != 2 or min(s) < 1 or max(s) > 16384 or s[0] % 256 or
           s[0] * s[1] > 6144 * 16384 for s in shapes):
        raise ValueError("Shapes must be K,N <= 16384, K divisible by 256, K*N <= 100663296")
    if not torch.backends.mps.is_available():
        raise RuntimeError("This benchmark requires MPS")
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(20261003)
    rng = random.Random(20261003)
    start = time.perf_counter()
    emit(event="metadata", torch=torch.__version__, macos=platform.mac_ver()[0],
         backend=lin.__file__, decode_bytes=lin._MAX_DECODE_BYTES,
         transpose_min_k=lin._TRANSPOSED_DECODE_MIN_K,
         compute_dtype=os.environ.get("COMFY_KITCHEN_MPS_COMPUTE_DTYPE", "default"),
         rounds=args.rounds, tokens=tokens, shapes=shapes, seed=20261003,
         max_seconds=args.max_seconds)

    with torch.inference_mode():
        for k, n in shapes:
            # CPU oracle reuses these, avoiding a second full fp32 weight.
            q_cpu = torch.randint(-127, 128, (n, k), dtype=torch.int8, generator=generator)
            scales_cpu = torch.rand(n, generator=generator) * 0.0005 + 0.0001
            q, scales = q_cpu.to("mps"), scales_cpu.to("mps")
            exponents = torch.empty(n, dtype=torch.int32, device="mps")
            transposed = k >= lin._TRANSPOSED_DECODE_MIN_K
            resident = lin.decode_int8(q, scales, 0, n, torch.float16, transposed, exponents)
            down = (torch.randn(64, k, generator=generator) * 0.02).to("mps", torch.bfloat16)
            up = (torch.randn(n, 64, generator=generator) * 0.02).to("mps", torch.bfloat16)
            lora = (down, up, 0.8)
            for m in tokens:
                if time.perf_counter() - start > args.max_seconds:
                    emit(event="time_budget_stop", elapsed_s=time.perf_counter() - start)
                    return
                x_cpu = torch.randn(m, k, generator=generator).to(torch.bfloat16)
                x = x_cpu.to("mps")
                xc, _ = lin.prologue(x, 256, torch.float16)
                funcs = {
                    "full": lambda: lin.int8_linear(x, q, scales, convrot=True),
                    "decode": lambda: lin.decode_int8(q, scales, 0, n, torch.float16, transposed, exponents),
                    "resident_gemm": lambda: xc @ resident if transposed else torch.nn.functional.linear(xc, resident),
                    "prologue": lambda: lin.prologue(x, 256, torch.float16),
                    "lora_prep": lambda: lin._lora_gemm_operands(lora, 256, torch.float16),
                    "full_lora": lambda: lin.int8_linear(x, q, scales, convrot=True,
                                                         lora_down=down, lora_up=up, lora_scale=0.8),
                }
                for fn in funcs.values():
                    timed(fn)
                samples = {name: [] for name in funcs}
                for _ in range(args.rounds):
                    order = list(funcs)
                    rng.shuffle(order)
                    for name in order:
                        samples[name].append(timed(funcs[name]))
                # A manageable, independent fp32 CPU oracle over eight rows and 64 channels.
                check_m, check_n = min(m, 8), min(n, 64)
                h = _build_hadamard(256, device="cpu", dtype=torch.float32)
                xr = _rotate_activation(x_cpu[:check_m].float(), h, 256)
                reference = xr @ (q_cpu[:check_n].float() * scales_cpu[:check_n, None]).t()
                # Execute the timed shape and dtype, then sample its output.
                # Slicing inputs first would test a different dispatch path.
                actual = lin.int8_linear(x, q, scales, convrot=True)[:check_m, :check_n].float().cpu()
                rel_error = float(torch.linalg.vector_norm(actual - reference) /
                                  torch.linalg.vector_norm(reference).clamp_min(1e-30))
                emit(event="measurement", m=m, k=k, n=n, dtype="bf16", transposed=transposed,
                     median_ms={name: statistics.median(values) for name, values in samples.items()},
                     min_ms={name: min(values) for name, values in samples.items()},
                     max_ms={name: max(values) for name, values in samples.items()},
                     oracle_rel_l2=rel_error, oracle_rows=check_m, oracle_channels=check_n,
                     oracle_call="full_timed_shape_and_dtype",
                     driver_gb=torch.mps.driver_allocated_memory() / 1e9,
                     live_gb=torch.mps.current_allocated_memory() / 1e9)
                del x, x_cpu, xc, actual, reference, xr, funcs
            del q_cpu, scales_cpu, q, scales, resident, down, up, lora, exponents
            gc.collect()
            torch.mps.synchronize()
            torch.mps.empty_cache()
    emit(event="complete", elapsed_s=time.perf_counter() - start)


if __name__ == "__main__":
    main()
