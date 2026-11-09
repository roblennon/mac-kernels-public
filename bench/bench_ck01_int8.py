"""CK-01 microbenchmark: ConvRot INT8 linear on Apple GPUs, every path in one run.

Paths, bf16 in and out: resident dense bf16 and fp16 weights through F.linear; dequantize per
call then F.linear (what a host does without an INT8 kernel); comfy-kitchen's mps backend.
relerr is relative L2 against an fp32 F.linear over the dequantized weight on MPS.
Dense F.linear at wide K also pays the MPS transposed-weight cost (PyTorch report PT-01).

Usage: python bench_ck01_int8.py [M,K,N ...]
"""
import sys
import time

import torch
import torch.nn.functional as F

import comfy_kitchen as ck
from comfy_kitchen.backends.eager import quantization as Q
from comfy_kitchen.tensor.int8_utils import _build_hadamard

G = 256
dev = torch.device("mps")


def median_ms(fn, n=7, warm=2):
    for _ in range(warm):
        fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        r = fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1e3, r


def make(n, k):
    torch.manual_seed(0)
    w = torch.randn(n, k) * 0.02
    w = (w.reshape(n, k // G, G) @ _build_hadamard(G).T).reshape(n, k)  # stored rotated
    s = w.abs().amax(1, keepdim=True) / 127
    q = torch.round(w / s).clamp(-127, 127).to(torch.int8)
    return q.to(dev), s.to(dev)


def run(m, k, n):
    q, s = make(n, k)
    wd = Q.dequantize_int8_convrot_weight(q, s, G)  # fp32, original basis
    x = torch.randn(m, k, device=dev).to(torch.bfloat16)
    oracle = F.linear(x.float(), wd)
    wb, wh = wd.to(torch.bfloat16), wd.to(torch.float16)
    paths = {
        "dense bf16 F.linear": lambda: F.linear(x, wb),
        "dense fp16 F.linear": lambda: F.linear(x.half(), wh).to(torch.bfloat16),
        "dequantize per call": lambda: F.linear(x, Q.dequantize_int8_convrot_weight(q, s, G).to(torch.bfloat16)),
        "mps backend": lambda: ck.int8_linear(x, q, s, None, torch.bfloat16, convrot=True, convrot_groupsize=G),
    }
    print(f"\n== M={m} K={k} N={n}")
    for name, fn in paths.items():
        ms, out = median_ms(fn)
        rel = ((out.float() - oracle).norm() / oracle.norm()).item()
        print(f"RESULT {m},{k},{n} | {name:22s} | {ms:8.1f} ms | relerr {rel:.2e}", flush=True)
        del out
    del wd, wb, wh, oracle
    torch.mps.empty_cache()


if __name__ == "__main__":
    print("torch", torch.__version__, "kitchen", ck.__file__)
    shapes = [tuple(map(int, a.split(","))) for a in sys.argv[1:]] or [
        (4608, 6144, 6144), (4608, 6144, 16384), (4608, 16384, 6144)]
    for shape in shapes:
        run(*shape)
