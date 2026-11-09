"""Cost of fp32 upcast and of 2^31 chunking for SDPA on MPS at Wan self-attention shapes."""
import json, time
import torch, torch.nn.functional as F

SHAPES = [("5b-704x416x33", 24, 2574), ("5b-1280x704x81", 24, 18480), ("14b-832x480x81", 40, 32760)]

def timed(fn, reps=3):
    fn(); torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.mps.synchronize(); return out, (time.perf_counter() - t) / reps

def head_chunked(q, k, v, limit=2**31):
    h, n = q.shape[1], q.shape[2]
    hc = max(1, limit // (n * n))
    return torch.cat([F.scaled_dot_product_attention(q[:, i:i + hc], k[:, i:i + hc], v[:, i:i + hc]) for i in range(0, h, hc)], dim=1)

with torch.inference_mode():
    for label, h, n in SHAPES:
        g = torch.Generator(device="mps").manual_seed(0)
        q, k, v = (torch.randn(1, h, n, 128, device="mps", dtype=torch.float16, generator=g) for _ in range(3))
        reps = 3 if n < 20000 else 1
        o16, t16 = timed(lambda: F.scaled_dot_product_attention(q, k, v), reps)
        o32, t32 = timed(lambda: F.scaled_dot_product_attention(q.float(), k.float(), v.float()).half(), reps)
        oc, tc = timed(lambda: head_chunked(q, k, v), reps)
        print("RESULT " + json.dumps(dict(shape=label, fp16_s=round(t16, 4), fp32_upcast_s=round(t32, 4), chunked_fp16_s=round(tc, 4),
              chunks=-(-h // max(1, 2**31 // (n * n))), fp16_vs_fp32_maxdiff=float((o16 - o32).abs().max()), chunked_equal=bool(torch.equal(oc, o16)))), flush=True)
        del q, k, v, o16, o32, oc; torch.mps.empty_cache()
