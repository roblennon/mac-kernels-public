"""Microbenchmark: GGUF (GGML block-quantized) linear layers on Apple GPUs (MPS).

Per type, against the same decoded weight:
- dense fp16 / bf16 resident weight, bare F.linear (the ceiling);
- ComfyUI-GGUF's stock path on MPS: its torch dequantize per call (dequant.py, scales in
  fp16 as its default dequant_dtype=None computes them), cast to x's dtype, F.linear;
- comfy_kitchen mps gguf_linear: one-pass Metal decode + range-safe fp16 GEMM;
- the decode kernel alone.
relerr is relative L2 against a float64 CPU linear over gguf.quants' fp32 dequantization.

Weights are random legal blocks (every byte pattern is a legal block) with fp16 scales sized
so |w| is about 0.02-0.05, a DiT-like range. Inputs bf16, as ComfyUI computes on Apple GPUs.

Usage: python bench/bench_gguf_linear.py [--types Q4_K,Q8_0] [M,K,N ...]
Needs COMFY_GGUF_DIR (default vendor/ComfyUI-GGUF-fork) for the stock path.
"""
import argparse
import importlib.util
import os
import time

import gguf
import numpy as np
import torch
import torch.nn.functional as F

import comfy_kitchen as ck
from comfy_kitchen.backends.mps import gguf as mg

dev = torch.device("mps")
HERE = os.path.dirname(os.path.abspath(__file__))
GGUF_DIR = os.environ.get("COMFY_GGUF_DIR", os.path.join(HERE, "..", "vendor", "ComfyUI-GGUF-fork"))

# Byte offsets of the fp16 scale fields, and a magnitude giving |w| ~ 0.02-0.05.
F16_FIELDS = {
    "Q4_0": ([0], 5e-3), "Q4_1": ([0, 2], 3e-3), "Q5_0": ([0], 2.5e-3), "Q5_1": ([0, 2], 1.5e-3),
    "Q8_0": ([0], 3e-4), "Q2_K": ([80, 82], 1e-3), "Q3_K": ([108], 2e-4), "Q4_K": ([0, 2], 5e-5),
    "Q5_K": ([0, 2], 2.5e-5), "Q6_K": ([208], 3e-5),
    "IQ4_NL": ([0], 3e-4), "IQ4_XS": ([0], 1e-5),
}


def load_city96_dequant():
    spec = importlib.util.spec_from_file_location("city96_dequant", os.path.join(GGUF_DIR, "dequant.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sync():
    torch.mps.synchronize()


def timeit(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    sync()
    ts = []
    r = None
    for _ in range(n):
        t = time.perf_counter()
        r = fn()
        sync()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1e3, r


def blocks(name, n, k, seed=0):
    tid, qk, ts = mg.GGUF_TYPES[name]
    rng = np.random.default_rng(seed)
    nb = n * k // qk
    raw = rng.integers(0, 256, size=(nb, ts), dtype=np.uint8)
    offs, mag = F16_FIELDS[name]
    for off in offs:
        v = np.abs(rng.standard_normal(nb) * mag + mag).astype(np.float16)
        raw[:, off:off + 2] = v.view(np.uint8).reshape(nb, 2)
    return raw


def run(M, K, N, types, city):
    torch.manual_seed(0)
    x = torch.randn(M, K)
    xb, xh = x.to(dev, torch.bfloat16), x.to(dev, torch.float16)
    xd = x.double()
    print(f"\n== M={M} K={K} N={N}")
    print(f"  {'median ms':>10}  {'resident MB':>11}  {'relerr':>8}  path")
    dense_done = False
    for name in types:
        tid, qk, ts = mg.GGUF_TYPES[name]
        raw = blocks(name, N, K)
        ref_w = torch.from_numpy(gguf.quants.dequantize(raw.reshape(-1), gguf.GGMLQuantizationType(tid))).reshape(N, K)
        ref = xd @ ref_w.double().T

        def rel(r):
            return ((r.cpu().double() - ref).norm() / ref.norm()).item()

        rows = []
        if not dense_done:
            w16, wb = ref_w.to(dev, torch.float16), ref_w.to(dev, torch.bfloat16)
            ms, r = timeit(lambda: F.linear(xh, w16))
            rows.append(("dense fp16 weight, fp16 in (bare GEMM)", ms, w16.nbytes, rel(r)))
            ms, r = timeit(lambda: F.linear(xb, wb))
            rows.append(("dense bf16 weight, bf16 in", ms, wb.nbytes, rel(r)))
            del w16, wb
            dense_done = True
        wq = torch.from_numpy(raw).to(dev)
        if city is not None:
            qt = gguf.GGMLQuantizationType(tid)
            # ComfyUI-GGUF holds the tensor as its numpy reader shapes it: [N, K/qk*ts] bytes.
            wc = wq.reshape(N, K // qk * ts) if K % qk == 0 else wq.reshape(-1)

            def stock():
                w = city.dequantize(wc, qt, torch.Size((N, K)), dtype=None).to(torch.bfloat16)
                return F.linear(xb, w)

            ms, r = timeit(stock, n=5, warm=2)
            rows.append((f"{name}: ComfyUI-GGUF dequant per call + bf16 GEMM", ms, wq.nbytes, rel(r)))
        for xin, tag in ((xb, "bf16 in"), (xh, "fp16 in")):
            ms, r = timeit(lambda xin=xin: ck.gguf_linear(xin, wq, tid, (N, K)))
            rows.append((f"{name}: gguf_linear Metal decode + fp16 GEMM ({tag})", ms, wq.nbytes, rel(r)))
        ch = torch.empty(N, dtype=torch.int32, device=dev)
        transposed = K >= 12288
        ms, _ = timeit(lambda: mg._decode(wq.reshape(-1), name.lower(), K, 0, N, torch.float16, transposed, ch))
        rows.append((f"{name}: decode alone, row exponents included{' (transposed)' if transposed else ''}", ms, None, None))
        row_exp = getattr(mg._shaders(), f"gguf_row_exp_{name.lower()}")
        ms, _ = timeit(lambda: row_exp(ch, wq.reshape(-1), K, 0, threads=(32, N), group_size=(32, 1)))
        rows.append((f"{name}: row exponents alone", ms, None, None))
        for label, ms, b, err in rows:
            ms_s = f"{ms:10.2f}" if ms is not None else f"{'-':>10}"
            b_s = f"{b / 2**20:11.1f}" if b is not None else f"{'':>11}"
            e_s = f"{err:8.5f}" if err is not None else f"{'':>8}"
            print(f"  {ms_s}  {b_s}  {e_s}  {label}", flush=True)
        del wq


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="Q8_0,Q6_K,Q5_K,Q4_K,Q5_0,Q4_0,Q3_K,Q2_K")
    ap.add_argument("--no-stock", action="store_true")
    ap.add_argument("shapes", nargs="*")
    a = ap.parse_args()
    print("torch", torch.__version__, "comfy_kitchen", ck.__file__)
    city = None if a.no_stock else load_city96_dequant()
    shapes = [tuple(map(int, s.split(","))) for s in a.shapes] or [(4608, 6144, 6144)]
    for shape in shapes:
        run(*shape, a.types.split(","), city)
