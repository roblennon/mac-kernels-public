"""Microbenchmark: unmerged LoRA inside the mps backend's quantized linears (approach B).

For each shape and format: the base layer, base + LoRA through the kernels' lora_* arguments
(fused), and base + LoRA as a separate bf16 addmm on the finished output (unfused, the
side path a caller would run without kernel support).
Median and min over ROUNDS (default 15) interleaved rounds, bf16 in and out, ConvRot on.

Usage: python bench_lora_sidepath_kernel.py [M,K,N ...] (default the two DiT shapes)
"""
import os
import sys
import time

import torch
from comfy_kitchen.backends.eager import w4a8_int8 as E
from comfy_kitchen.backends.mps import linear, w4a8

dev = torch.device("mps")
RANK = int(os.environ.get("RANK", "64"))


def timeit(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


ROUNDS = int(os.environ.get("ROUNDS", "15"))


def interleaved(fns, rounds=None, warm=2, stat="median"):
    """Median (or min) ms per variant, the variants taking turns so drift hits all alike."""
    rounds = rounds or ROUNDS
    for f in fns.values():
        for _ in range(warm):
            f()
    times = {k: [] for k in fns}
    for _ in range(rounds):
        for k, f in fns.items():
            torch.mps.synchronize()
            t = time.perf_counter()
            f()
            torch.mps.synchronize()
            times[k].append(time.perf_counter() - t)
    pick = (lambda v: min(v)) if stat == "min" else (lambda v: sorted(v)[len(v) // 2])
    return {k: pick(v) * 1e3 for k, v in times.items()}


def separate_addmm(f, x, down, up, scale):
    """The base layer, then the LoRA as a separate GEMM pair in the activation dtype."""
    def run():
        out = f()
        out.view(-1, out.shape[-1]).addmm_(torch.nn.functional.linear(x, down), up.t(), alpha=scale)
        return out
    return run


def run(M, K, N, rank=RANK, formats=("INT8", "W6A8")):
    torch.manual_seed(0)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    down = (torch.randn(rank, K) * 0.02).to(dev, torch.bfloat16)
    up = (torch.randn(N, rank) * 0.02).to(dev, torch.bfloat16)
    q8 = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
    s8 = (torch.rand(N, device=dev) * 1e-3).float()
    q6, srel, sch, _, _ = E.quantize_w4a8_int8_weight((torch.randn(N, K) * 0.02).to(torch.bfloat16), bits=6)
    q6, srel, sch = q6.to(dev), srel.to(dev), sch.to(dev)

    calls = {
        "INT8": lambda **kw: linear.int8_linear(x, q8, s8, None, torch.bfloat16, convrot=True, **kw),
        "W6A8": lambda **kw: w4a8.w4a8_int8_linear(x, q6, srel, sch, None, None, None, 16, 256, torch.bfloat16, **kw),
    }
    lora = {"lora_down": down, "lora_up": up, "lora_scale": 0.8}
    for fmt, f in calls.items():
        if fmt not in formats:
            continue
        for stat in ("median", "min"):
            t = interleaved({
                "base": f,
                "fused": lambda f=f: f(**lora),
                "unfused": separate_addmm(f, x, down, up, 0.8),
            }, stat=stat)
            base, fused, unfused = t["base"], t["fused"], t["unfused"]
            print(f"| {M}x{K}x{N} r{rank} | {fmt} | {stat} | {base:.2f} | {fused:.2f} (+{fused - base:.2f}) "
                  f"| {unfused:.2f} (+{unfused - base:.2f}) |", flush=True)


if __name__ == "__main__":
    # M,K,N[,rank]; default the two DiT shapes at RANK
    shapes = [tuple(int(v) for v in a.split(",")) for a in sys.argv[1:]] or [(4608, 6144, 6144), (4608, 16384, 6144)]
    print(f"rank {RANK}, bf16, ConvRot 256, ms over {ROUNDS} interleaved rounds")
    print("| M x K x N rank | format | stat | base | + LoRA fused | + LoRA unfused |")
    print("|---|---|---|---|---|---|")
    for s in shapes:
        run(*s)
