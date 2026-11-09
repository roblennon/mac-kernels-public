"""SDPA vs ComfyUI sub-quadratic attention on MPS at Wan self-attention shapes.

Checks correctness above 2^31 attention-matrix elements (ComfyUI #14837) against a query-chunked SDPA
reference whose chunks stay far below that limit, and reports time and peak driver memory.
"""
import os, sys, time, json, threading
CODE = os.environ.get("COMFY_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ComfyUI")
sys.path.insert(0, CODE); os.chdir(CODE)
sys.argv = [sys.argv[0]]
import comfy.options; comfy.options.enable_args_parsing()
import torch, torch.nn.functional as F
from comfy.ldm.modules.attention import attention_sub_quad

PEAK = {"gb": 0.0}
def _poll():
    while True:
        PEAK["gb"] = max(PEAK["gb"], torch.mps.driver_allocated_memory() / 2**30); time.sleep(0.05)
threading.Thread(target=_poll, daemon=True).start()

def chunked_ref(q, k, v, chunk=1024):
    return torch.cat([F.scaled_dot_product_attention(q[:, :, i:i + chunk], k, v) for i in range(0, q.shape[2], chunk)], dim=2)

def timed(fn, reps=2):
    fn(); torch.mps.synchronize()
    torch.mps.empty_cache(); PEAK["gb"] = torch.mps.driver_allocated_memory() / 2**30
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.mps.synchronize()
    return out, (time.perf_counter() - t) / reps, PEAK["gb"]

# (label, batch, heads, tokens, head_dim): Wan 5B 704x416x33, 5B 1280x704x81, 14B 640x384x33, 14B 832x480x81
SHAPES = [("5b-704x416x33", 1, 24, 2574, 128), ("5b-1280x704x81", 1, 24, 18480, 128),
          ("14b-640x384x33", 1, 40, 8640, 128), ("14b-832x480x81", 1, 40, 32760, 128)]
only = os.environ.get("ONLY")
with torch.inference_mode():
    for label, b, h, n, d in SHAPES:
        if only and only not in label:
            continue
        g = torch.Generator(device="mps").manual_seed(0)
        q, k, v = (torch.randn(b, h, n, d, device="mps", dtype=torch.float16, generator=g) for _ in range(3))
        ref, t_ref, m_ref = timed(lambda: chunked_ref(q, k, v), reps=1)
        out, t_sdpa, m_sdpa = timed(lambda: F.scaled_dot_product_attention(q, k, v))
        # sub-quad takes (b, n, h*d) and returns the same layout
        qs, ks, vs = (t.transpose(1, 2).reshape(b, n, h * d) for t in (q, k, v))
        sq, t_sq, m_sq = timed(lambda: attention_sub_quad(qs, ks, vs, h), reps=1)
        sq = sq.reshape(b, n, h, d).transpose(1, 2)
        res = dict(shape=label, elements=b * h * n * n, over_2_31=b * h * n * n > 2**31,
                   sdpa_s=round(t_sdpa, 3), sdpa_peak_gb=round(m_sdpa, 2), sdpa_maxdiff=float((out - ref).abs().max()),
                   subquad_s=round(t_sq, 3), subquad_peak_gb=round(m_sq, 2), subquad_maxdiff=float((sq - ref).abs().max()),
                   ref_chunked_s=round(t_ref, 3), ref_peak_gb=round(m_ref, 2), nan=bool(torch.isnan(out).any()))
        print("RESULT " + json.dumps(res), flush=True)
        del q, k, v, ref, out, sq, qs, ks, vs; torch.mps.empty_cache()
