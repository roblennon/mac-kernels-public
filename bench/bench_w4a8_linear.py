"""Microbenchmark: W4A8 / W6A8 (AsymW4A8Int8 storage) linear layers on Apple GPUs (MPS).

Compares dense resident fp16 / bf16 weights, eager's dequantize-per-call on MPS, eager's
INT8 GEMM on MPS (if it runs), and the mps backend's one-pass Metal decode + fp16 GEMM.
relerr is relative L2 against a float32 CPU linear over eager's dequantized weight.

Usage: python bench_w4a8_linear.py [M,K,N ...]
"""
import sys
import time

import torch
import torch.nn.functional as F

from comfy_kitchen.backends.eager import w4a8_int8 as E
from comfy_kitchen.backends.mps import w4a8 as mps_w4a8

dev = torch.device("mps")


def sync():
    torch.mps.synchronize()


def timeit(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    sync()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        r = fn()
        sync()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1e3, r


def nbytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def run(M, K, N):
    torch.manual_seed(0)
    w_true = (torch.randn(N, K) * 0.02).to(torch.bfloat16)
    x = torch.randn(M, K)
    xm16, xmb = x.to(dev, torch.float16), x.to(dev, torch.bfloat16)
    rows = []

    for label, kw in (("W4 codebook", dict(bits=4)), ("W6", dict(bits=6))):
        t0 = time.perf_counter()
        packed = E.quantize_w4a8_int8_weight(w_true, **kw)
        q, s_rel, s_ch, corr, cb = packed
        w_deq = E.dequantize_w4a8_int8_weight(q, s_rel, s_ch, cb, corr, output_dtype=torch.float32)
        ref = x @ w_deq.T
        print(f"  [{label}] quantized + oracle on CPU in {time.perf_counter() - t0:.1f}s")

        def rel(r):
            return ((r.float().cpu() - ref).norm() / ref.norm()).item()

        qm, sm, cm = q.to(dev), s_rel.to(dev), s_ch.to(dev)
        cbm = None if cb is None else cb.to(dev)
        packed_bytes = nbytes(qm, sm, cm, cbm)

        if label == "W4 codebook":  # dense baselines hold the same decoded weight
            w16, wb = w_deq.to(dev, torch.float16), w_deq.to(dev, torch.bfloat16)
            ms, r = timeit(lambda: F.linear(xm16, w16))
            rows.append(("dense fp16, resident", ms, nbytes(w16), rel(r)))
            ms, r = timeit(lambda: F.linear(xmb, wb))
            rows.append(("dense bf16, resident", ms, nbytes(wb), rel(r)))
            del w16, wb

        # MPS has no float8 arithmetic, so eager also gets a run with the same scales as fp32.
        sm32 = s_rel.float().to(dev)
        for scales, stag in ((sm, "e4m3 scales"), (sm32, "fp32 scales")):
            b = nbytes(qm, scales, cm, cbm)

            def eager_dequant_per_call(scales=scales):
                w = E.dequantize_w4a8_int8_weight(qm, scales, cm, cbm, None, output_dtype=torch.bfloat16)
                return F.linear(xmb, w)

            def eager_int8_gemm(scales=scales):
                return E.w4a8_int8_linear(xmb, qm, scales, cm, cbm, None, out_dtype=torch.bfloat16)

            for fn, what in ((eager_dequant_per_call, "eager dequant per call, bf16 GEMM"),
                             (eager_int8_gemm, "eager w4a8_int8_linear (A8 INT8 GEMM)")):
                try:
                    ms, r = timeit(fn, n=4, warm=1)
                    rows.append((f"{label}: {what}, {stag}", ms, b, rel(r)))
                except Exception as e:  # noqa: BLE001
                    rows.append((f"{label}: {what}, {stag}: FAILED {type(e).__name__}: {str(e)[:70]}", None, b, None))
        del sm32

        for xin, tag in ((xm16, "fp16 in"), (xmb, "bf16 in")):
            ms, r = timeit(lambda xin=xin: mps_w4a8.w4a8_int8_linear(xin, qm, sm, cm, cbm, out_dtype=xin.dtype))
            rows.append((f"{label}: mps Metal decode + fp16 GEMM ({tag})", ms, packed_bytes, rel(r)))

        bits = kw["bits"]
        levels = cbm if cbm is not None else mps_w4a8._uniform_levels(bits, dev)
        ms, _ = timeit(lambda: mps_w4a8._decode_w4a8(qm, sm, cm, levels, None, K, bits, 16, 0, N))
        rows.append((f"{label}: decode kernel alone", ms, None, None))
        del qm, sm, cm, cbm

    print(f"\n== M={M} K={K} N={N}")
    print(f"  {'median ms':>10}  {'resident MB':>11}  {'relerr':>8}  path")
    for name, ms, b, err in rows:
        ms_s = f"{ms:10.2f}" if ms is not None else f"{'-':>10}"
        b_s = f"{b / 2**20:11.1f}" if b is not None else f"{'':>11}"
        e_s = f"{err:8.5f}" if err is not None else f"{'':>8}"
        print(f"  {ms_s}  {b_s}  {e_s}  {name}")


if __name__ == "__main__":
    print("torch", torch.__version__)
    shapes = [tuple(map(int, s.split(","))) for s in sys.argv[1:]] or [(4096, 3072, 3072), (4096, 3072, 12288)]
    for shape in shapes:
        run(*shape)
