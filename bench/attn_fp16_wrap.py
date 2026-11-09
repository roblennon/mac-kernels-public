"""Run a runner with bf16 SDPA calls on MPS computed in fp16 (ATTN_FP16=1) or untouched (ATTN_FP16=0), saving the final latent.

Usage: ATTN_FP16=1 python bench/attn_fp16_wrap.py bench/run_krea2.py
Only calls with heads * Nq * Nk >= 2^22 switch; a call whose q/k/v magnitude exceeds 60000 stays in bf16 (counted).
The latent from common_ksampler is saved to out/latents/attnfp16-<ATTN_FP16>-<runner>.pt.
"""
import os, runpy, sys
import torch

ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
ON = os.environ.get("ATTN_FP16", "1") == "1"
real_sdpa = torch.nn.functional.scaled_dot_product_attention
stats = {"fp16": 0, "kept_bf16_overflow": 0, "other": 0}
state = {"patched": False}

def _patch_ksampler():
    nodes = sys.modules.get("nodes")
    if nodes is None or state["patched"] or not hasattr(nodes, "common_ksampler"):
        return
    real = nodes.common_ksampler
    def wrapped(*a, **kw):
        out = real(*a, **kw)
        os.makedirs(f"{ROOT}/out/latents", exist_ok=True)
        torch.save(out[0]["samples"].cpu(), f"{ROOT}/out/latents/attnfp16-{int(ON)}-{os.path.basename(sys.argv[0])[:-3]}.pt")
        return out
    nodes.common_ksampler = wrapped; state["patched"] = True

def hook(q, k, v, attn_mask=None, *a, **kw):
    _patch_ksampler()
    if ON and q.device.type == "mps" and q.dtype == torch.bfloat16 and q.shape[-3] * q.shape[-2] * k.shape[-2] >= 2**22 \
            and (attn_mask is None or attn_mask.dtype == torch.bool):
        if float(torch.stack([q.abs().amax(), k.abs().amax(), v.abs().amax()]).amax()) < 60000:
            stats["fp16"] += 1
            return real_sdpa(q.half(), k.half(), v.half(), attn_mask, *a, **kw).to(torch.bfloat16)
        stats["kept_bf16_overflow"] += 1
    else:
        stats["other"] += 1
    return real_sdpa(q, k, v, attn_mask, *a, **kw)

torch.nn.functional.scaled_dot_product_attention = hook
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
print("ATTN_FP16_STATS", ON, stats, flush=True)
