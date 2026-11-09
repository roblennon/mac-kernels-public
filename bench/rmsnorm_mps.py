"""Wan VAE RMS_norm variants on MPS (fp16, channel-first)."""
import time, json, torch, torch.nn.functional as F
def timed(fn, reps=5):
    fn(); torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(reps): o = fn()
    torch.mps.synchronize(); return o, (time.perf_counter() - t) / reps
with torch.inference_mode():
    for shape in ((1, 256, 4, 352, 640), (1, 512, 4, 176, 320)):
        x = torch.randn(shape, device="mps", dtype=torch.float16); c = shape[1]
        g = torch.rand(c, 1, 1, 1, device="mps", dtype=torch.float16) + 0.5; s = c ** 0.5
        ref, t0 = timed(lambda: F.normalize(x, dim=1) * s * g + 0)
        o1, t1 = timed(lambda: F.normalize(x, dim=1) * (g * s))
        o2, t2 = timed(lambda: x * ((g * s) / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-12)))
        o3, t3 = timed(lambda: x * (torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-12).reciprocal() * (g * s)))
        print("RESULT " + json.dumps(dict(shape=shape, current_s=round(t0, 4), no_zero_add_s=round(t1, 4), fused_scale_s=round(t2, 4), recip_s=round(t3, 4),
              err1=float((o1 - ref).abs().max()), err2=float((o2 - ref).abs().max()), err3=float((o3 - ref).abs().max()))), flush=True)
