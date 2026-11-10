"""Thread experiment A: remove the small-kernel half of a decode step. Qwen3-1.7B on MPS, arms
interleaved in one process: stock; fused normalisation (F.rms_norm in place of Hugging Face's
Qwen3RMSNorm); a preallocated StaticCache; both. Outputs checked against stock.
"""
import argparse, json, statistics, time
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, StaticCache
import transformers.models.qwen3.modeling_qwen3 as q3
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

FUSED = {"on": False, "exact": False, "rope": False}
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fused_rope
_orig_rope = q3.apply_rotary_pos_emb
def _rope(q, k, cos, sin, unsqueeze_dim=1):
    if FUSED["rope"]:
        return fused_rope.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)
    return _orig_rope(q, k, cos, sin, unsqueeze_dim)
q3.apply_rotary_pos_emb = _rope
_orig = q3.Qwen3RMSNorm.forward
def _norm(self, x):
    if not FUSED["on"]:
        return _orig(self, x)
    if FUSED["exact"]:  # float32 inside the fused kernel, weight applied after: matches Hugging Face's arithmetic
        return self.weight * F.rms_norm(x.float(), (x.shape[-1],), None, self.variance_epsilon).to(x.dtype)
    return F.rms_norm(x, (x.shape[-1],), self.weight, self.variance_epsilon)
q3.Qwen3RMSNorm.forward = _norm

def sync(): torch.mps.synchronize()

@torch.no_grad()
def run(model, B, P, N, static, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, 30000, (B, P), generator=g).to("mps")
    kw = {}
    if static:
        kw["past_key_values"] = StaticCache(config=model.config, max_cache_len=P + N)
    out = model(ids, use_cache=True, cache_position=torch.arange(P, device="mps"), **kw)
    past = out.past_key_values; nxt = out.logits[:, -1].argmax(-1, keepdim=True); toks = [nxt]
    pos = P
    sync(); t = time.perf_counter()
    for _ in range(N - 1):
        out = model(nxt, past_key_values=past, use_cache=True, cache_position=torch.tensor([pos], device="mps"))
        past = out.past_key_values; nxt = out.logits[:, -1].argmax(-1, keepdim=True); toks.append(nxt); pos += 1
    sync(); dt = (time.perf_counter() - t) / (N - 1)
    return dt, torch.cat(toks, 1).cpu(), out.logits[:, -1].float().cpu()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="1,4,16"); ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
    arms = {"stock": (False, False, False, False), "fused_norm": (True, False, False, False),
            "fused_rope": (False, False, False, True), "norm_and_rope": (True, False, False, True)}
    res = {"torch": torch.__version__, "arms": list(arms), "cells": []}
    for name, (f, s, e, r) in arms.items():
        FUSED["on"], FUSED["exact"], FUSED["rope"] = f, e, r; run(model, 1, 8, 4, s, 0)
    for B in [int(b) for b in a.batches.split(",")]:
        cell = {"batch": B, "ms": {k: [] for k in arms}}
        ref_toks = ref_logits = None; checks = {}
        for r in range(a.reps):
            order = list(arms) if r % 2 == 0 else list(reversed(arms))
            for name in order:
                f, s, e, rp = arms[name]; FUSED["on"], FUSED["exact"], FUSED["rope"] = f, e, rp
                dt, toks, logits = run(model, B, 32, 32, s, seed=100 + B)
                cell["ms"][name].append(dt * 1e3)
                if r == 0:
                    if name == "stock": ref_toks, ref_logits = toks, logits
                    checks[name] = (toks, logits)
        for name, (toks, logits) in checks.items():
            checks[name] = {"token_agreement": float((toks == ref_toks).float().mean()),
                            "max_last_logit_diff": float((logits - ref_logits).abs().max())}
        cell["checks"] = checks
        cell["median_ms"] = {k: statistics.median(v) for k, v in cell["ms"].items()}
        base = cell["median_ms"]["stock"]
        print(f"B={B}", {k: f"{v:.1f} ms (x{base / v:.2f})" for k, v in cell["median_ms"].items()},
              {k: (round(c['token_agreement'], 3), round(c['max_last_logit_diff'], 3)) for k, c in checks.items()}, flush=True)
        res["cells"].append(cell)
        json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
