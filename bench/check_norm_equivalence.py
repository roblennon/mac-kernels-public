"""Equivalence tier 2 for fused normalisation: teacher-forced logits of Qwen3-1.7B on fixed
token sequences, fp16 stock vs fp16 fused variants, each against a float32 reference computed in
a separate process. A rewrite passes if its error against float32 is no worse than stock fp16's.
Usage: --mode ref (fp32, writes reference) | --mode test (fp16 arms, reads reference)."""
import argparse, json
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
import transformers.models.qwen3.modeling_qwen3 as q3
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MODE = {"v": "stock"}
_orig = q3.Qwen3RMSNorm.forward
def _norm(self, x):
    if MODE["v"] in ("fused_fp16", "fused_fp16_rope"):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.variance_epsilon)
    if MODE["v"] == "fused_fp32":
        return (self.weight * F.rms_norm(x.float(), (x.shape[-1],), None, self.variance_epsilon).to(x.dtype))
    return _orig(self, x)
q3.Qwen3RMSNorm.forward = _norm
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fused_rope
_orig_rope = q3.apply_rotary_pos_emb
q3.apply_rotary_pos_emb = lambda q, k, c, s_, u=1: (fused_rope.apply_rotary_pos_emb(q, k, c, s_, u) if MODE["v"].endswith("rope") else _orig_rope(q, k, c, s_, u))
ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--out")
a = ap.parse_args()
g = torch.Generator().manual_seed(7)
ids = torch.randint(1000, 30000, (16, 96), generator=g)
dt = torch.float32 if a.mode == "ref" else torch.float16
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=dt).to("mps").eval()
@torch.no_grad()
def logits(fn=None):
    fn = fn or (lambda x: model(x).logits)
    out = []
    for i in range(0, 16, 4):
        out.append(fn(ids[i:i + 4].to("mps")).float().cpu())
    return torch.cat(out)
if a.mode == "ref":
    torch.save(logits(), a.ref); print("ref saved")
else:
    ref = torch.load(a.ref); res = {}
    compiled = torch.compile(lambda x: model(x).logits, dynamic=False)
    for v in ("stock", "fused_fp16", "fused_fp16_rope", "compiled"):
        MODE["v"] = "stock" if v == "compiled" else v
        L = logits(compiled if v == "compiled" else None); err = (L - ref).abs()
        res[v] = {"max_abs": float(err.max()), "mean_abs": float(err.mean()), "p999_abs": float(err.flatten().kthvalue(int(err.numel() * 0.999)).values),
                  "top1_agree_with_fp32": float((L.argmax(-1) == ref.argmax(-1)).float().mean())}
        print(v, {k: round(x, 4) for k, x in res[v].items()}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
