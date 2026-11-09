"""uni_pc corrector/predictor coefficients on CPU vs MPS, through the real ComfyUI sampler code with a dummy model.

No model weights; tiny tensors. Prints, per step, the solve() inputs b and outputs rhos for each device,
and their worst relative error against a float64 CPU run of the same sampler.
Env: STEPS (default 20), SHIFT (default 8), VARIANT (bh1|bh2).
"""
import os, sys, json
CODE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ComfyUI")
sys.path.insert(0, CODE)
import torch
import comfy.model_sampling as cms, comfy.samplers as cs
from comfy.extra_samplers import uni_pc

STEPS = int(os.environ.get("STEPS", 20)); SHIFT = float(os.environ.get("SHIFT", 8.0)); VARIANT = os.environ.get("VARIANT", "bh1")

class MS(cms.ModelSamplingDiscreteFlow, cms.CONST):
    pass
ms = MS(None); ms.set_parameters(shift=SHIFT, multiplier=1000)
sig = cs.calculate_sigmas(ms, "simple", STEPS + 1)
sig = torch.cat([sig[:-2], sig[-1:]])  # KSampler discards the penultimate sigma for uni_pc

def run(device, dtype):
    log = []
    real_solve = torch.linalg.solve
    def spy(R, b):
        r = real_solve(R, b); log.append(dict(b=b.cpu().tolist(), rhos=r.cpu().tolist())); return r
    torch.linalg.solve = spy
    try:
        g = torch.Generator().manual_seed(0)
        x0 = torch.randn(1, 4, 8, 8, generator=g, dtype=torch.float64)
        noise = torch.randn(1, 4, 8, 8, generator=g, dtype=torch.float64)
        # dummy denoiser: pulls toward a fixed x0, smooth in sigma (enough for a deterministic trajectory)
        def model(x, s, **kw):
            return x0.to(device, dtype) * torch.cos(20 * s.view(-1, 1, 1, 1)) + 0.3 * torch.sin(x)
        out = uni_pc.sample_unipc(model, (noise * sig[0]).to(device, dtype), sig.to(device, dtype).clone(), extra_args={}, disable=True, variant=VARIANT)
    finally:
        torch.linalg.solve = real_solve
    return log, out.cpu().double()

torch.set_default_dtype(torch.float64)  # uni_pc builds rks/b with torch.tensor(), so the reference needs float64 defaults
ref, out_ref = run("cpu", torch.float64)
torch.set_default_dtype(torch.float32)
res = {}
for dev in ("cpu", "mps"):
    if dev == "mps" and not torch.backends.mps.is_available():
        continue
    log, out = run(dev, torch.float32)
    worst_b = max(max(abs(a - r) / max(abs(r), 1e-12) for a, r in zip(l["b"], q["b"])) for l, q in zip(log, ref))
    worst_rho = max(max(abs(a - r) for a, r in zip(l["rhos"], q["rhos"])) for l, q in zip(log, ref))
    rel_out = float((out - out_ref).norm() / out_ref.norm())
    res[dev] = dict(worst_b_relerr=worst_b, worst_rho_abserr=worst_rho, final_relerr=rel_out,
                    max_abs_rho=max(abs(v) for l in log for v in l["rhos"]))
    print(dev, json.dumps(res[dev]))
print("ref max_abs_rho", max(abs(v) for l in ref for v in l["rhos"]))
print("first sigmas", [round(float(s), 5) for s in sig[:5]])
