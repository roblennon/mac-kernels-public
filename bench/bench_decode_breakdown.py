"""Where does a Qwen3-1.7B decode step on MPS spend its time? Times the full step, then the
model's own linear layers alone in model order (weights streaming from memory, as in decode),
in nn.Linear layout and pre-transposed, and reports effective bandwidth against weight bytes.
"""
import argparse, json, statistics, time
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

def sync(): torch.mps.synchronize()

ap = argparse.ArgumentParser(); ap.add_argument("--dtype", default="float16"); ap.add_argument("--out", required=True)
a = ap.parse_args(); dt = getattr(torch, a.dtype)
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=dt).to("mps").eval()
lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
wbytes = sum(m.weight.numel() * m.weight.element_size() for m in lins)
allbytes = sum(p.numel() * p.element_size() for p in model.parameters())
res = {"dtype": a.dtype, "torch": torch.__version__, "n_linears": len(lins), "linear_weight_GB": wbytes / 1e9, "param_GB": allbytes / 1e9}

@torch.no_grad()
def step_time(B, n=24):
    ids = torch.randint(1000, 30000, (B, 32), device="mps")
    out = model(ids, use_cache=True); past = out.past_key_values; nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    for _ in range(3):
        out = model(nxt, past_key_values=past, use_cache=True); past = out.past_key_values; nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    sync(); t = time.perf_counter()
    for _ in range(n):
        out = model(nxt, past_key_values=past, use_cache=True); past = out.past_key_values; nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    sync(); return (time.perf_counter() - t) / n

@torch.no_grad()
def linears_time(B, layout, n=10):
    xs = {}
    for m in lins:
        xs.setdefault(m.in_features, torch.randn(B, m.in_features, device="mps", dtype=dt))
    if layout == "pre_t":
        wts = [m.weight.t().contiguous() for m in lins]
        fns = [(lambda x=xs[m.in_features], w=w: torch.mm(x, w)) for m, w in zip(lins, wts)]
    else:
        fns = [(lambda x=xs[m.in_features], m=m: F.linear(x, m.weight)) for m in lins]
    for f in fns: f()
    sync(); t = time.perf_counter()
    for _ in range(n):
        for f in fns: f()
    sync(); r = (time.perf_counter() - t) / n
    if layout == "pre_t": del wts; torch.mps.empty_cache()
    return r

res["cells"] = []
for B in (1, 4, 16):
    full = statistics.median(step_time(B) for _ in range(3))
    lin = statistics.median(linears_time(B, "linear") for _ in range(3))
    pre = statistics.median(linears_time(B, "pre_t") for _ in range(3))
    cell = {"batch": B, "full_step_ms": full * 1e3, "linears_only_ms": lin * 1e3, "linears_pre_t_ms": pre * 1e3,
            "linears_GBps": wbytes / lin / 1e9, "pre_t_GBps": wbytes / pre / 1e9, "full_step_GBps_params": allbytes / full / 1e9}
    res["cells"].append(cell)
    print({k: round(v, 2) if isinstance(v, float) else v for k, v in cell.items()}, flush=True)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "cells"})
json.dump(res, open(a.out, "w"), indent=1)
