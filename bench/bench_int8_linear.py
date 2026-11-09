"""Microbenchmark: ways to run a ConvRot INT8 linear layer on Apple GPUs (MPS).

Oracle: float32 CPU result of x @ W_true.T where W_true is the un-rotated dequantized weight.
"""
import sys, time, torch
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation
from comfy_kitchen.backends.eager import quantization as Q

dev = torch.device("mps")
G = 256

def sync(): torch.mps.synchronize()

def timeit(fn, n=8, warm=3):
    for _ in range(warm): fn()
    sync(); ts = []
    for _ in range(n):
        t = time.perf_counter(); r = fn(); sync(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts)//2] * 1e3, r

SHADER = """
#include <metal_stdlib>
using namespace metal;
kernel void dequant_i8_f16(device half* out, constant char* w, constant float* scale,
                           constant uint& K, uint idx [[thread_position_in_grid]]) {
    out[idx] = half(float(w[idx]) * scale[idx / K]);
}
"""
lib = torch.mps.compile_shader(SHADER) if hasattr(torch.mps, "compile_shader") else None

def run(M, K, N):
    torch.manual_seed(0)
    w_true = torch.randn(N, K) * 0.02
    h = _build_hadamard(G)
    wq, ws = Q.quantize_int8_convrot_weight(w_true, G) if False else (None, None)
    # quantize rotated weight per-row (matches checkpoint layout: int8 [N,K] + per-row fp32 scale)
    w_rot = (w_true.reshape(N, K // G, G) @ h.T).reshape(N, K)
    ws = (w_rot.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    wq = torch.round(w_rot / ws).clamp(-127, 127).to(torch.int8)
    w_deq_true = ((wq.float() * ws).reshape(N, K // G, G) @ h).reshape(N, K)  # what the checkpoint encodes
    x = torch.randn(M, K)
    ref = x @ w_deq_true.T

    xm16, xmb = x.to(dev, torch.float16), x.to(dev, torch.bfloat16)
    wqm, wsm = wq.to(dev), ws.to(dev)
    w16 = w_deq_true.to(dev, torch.float16); wb = w_deq_true.to(dev, torch.bfloat16)
    wrot16 = (wq.float() * ws).to(dev, torch.float16)
    h16 = h.to(dev, torch.float16)
    res = {}
    def rel(r): return ((r.float().cpu() - ref).norm() / ref.norm()).item()

    res["A dense bf16 (weights resident, 2 B/w)"] = timeit(lambda: xmb @ wb.T)
    res["B dense fp16 (weights resident, 2 B/w)"] = timeit(lambda: xm16 @ w16.T)
    res["B2 rotate-x + fp16 rotated weight resident"] = timeit(lambda: _rotate_activation(xm16, h16, G) @ wrot16.T)
    def comfy_fallback():  # dequantize + un-rotate weight every call, like ComfyUI with int8 compute disabled
        w = Q.dequantize_int8_convrot_weight(wqm, wsm, G).to(torch.bfloat16)
        return xmb @ w.T
    try: res["C ComfyUI-style: dequant+unrotate weight per call, bf16"] = timeit(comfy_fallback, n=4, warm=1)
    except Exception as e: print("C failed", repr(e)[:200])
    def ours_v0():
        w = (wqm.to(torch.float16) * wsm.to(torch.float16))
        return _rotate_activation(xm16, h16, G) @ w.T
    res["D weight-only: rotate x, torch dequant per call, fp16"] = timeit(ours_v0)
    if lib is not None:
        out = torch.empty(N, K, dtype=torch.float16, device=dev)
        wsf = wsm.reshape(-1).contiguous()
        def ours_v1():
            lib.dequant_i8_f16(out, wqm, wsf, K)
            return _rotate_activation(xm16, h16, G) @ out.T
        try: res["E weight-only: Metal one-pass dequant, fp16"] = timeit(ours_v1)
        except Exception as e: print("E failed", repr(e)[:300])
    def native_int8():
        return Q.int8_linear(xmb, wqm, wsm, convrot=True, convrot_groupsize=G)
    try: res["F eager int8_linear (torch._int_mm)"] = timeit(native_int8, n=3, warm=1)
    except Exception as e: print("F failed:", repr(e)[:160])

    print(f"\n== M={M} K={K} N={N}  (int8 weight {N*K/1e6:.0f} MB, fp16 {2*N*K/1e6:.0f} MB)")
    base = res["A dense bf16 (weights resident, 2 B/w)"][0]
    for k, (ms, r) in res.items():
        print(f"  {ms:9.2f} ms  x{ms/base:5.2f}  relerr {rel(r):.4f}  {k}")

print("torch", torch.__version__)
shapes = [tuple(map(int, s.split(","))) for s in sys.argv[1:]] or [(1024, 3072, 3072), (4096, 3072, 12288)]
for shape in shapes:
    run(*shape)
