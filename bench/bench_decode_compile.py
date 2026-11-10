"""Does torch.compile (Inductor's MPS backend) fuse the small-kernel half of a decode step?
Qwen3-1.7B fp16 with a StaticCache (fixed shapes), stock eager against compiled decode step.
Reports compile time, ms per step and greedy agreement."""
import argparse, json, statistics, time
import torch
from transformers import AutoModelForCausalLM, StaticCache
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--batch", type=int, default=1); a = ap.parse_args()
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
B, P, N = a.batch, 32, 32
def sync(): torch.mps.synchronize()
@torch.no_grad()
def decode(step_fn):
    g = torch.Generator().manual_seed(100 + B); ids = torch.randint(1000, 30000, (B, P), generator=g).to("mps")
    cache = StaticCache(config=model.config, max_cache_len=P + N)
    out = model(ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(P, device="mps"))
    nxt = out.logits[:, -1].argmax(-1, keepdim=True); toks = [nxt]; times = []
    for i in range(N - 1):
        sync(); t = time.perf_counter()
        logits = step_fn(nxt, cache, torch.tensor([P + i], device="mps"))
        nxt = logits[:, -1].argmax(-1, keepdim=True); sync(); times.append(time.perf_counter() - t); toks.append(nxt)
    return times, torch.cat(toks, 1).cpu()
def eager(nxt, cache, pos): return model(nxt, past_key_values=cache, use_cache=True, cache_position=pos).logits
res = {"torch": torch.__version__, "batch": B}
t_e, tok_e = decode(eager)
res["eager_ms"] = statistics.median(t_e[4:]) * 1e3
try:
    compiled = torch.compile(eager, dynamic=False)
    t0 = time.perf_counter(); t_c, tok_c = decode(compiled); res["first_compiled_pass_s"] = time.perf_counter() - t0
    t_c2, tok_c2 = decode(compiled)
    res["compiled_ms"] = statistics.median(t_c2[4:]) * 1e3
    res["compiled_first_steps_ms"] = [x * 1e3 for x in t_c[:3]]
    res["token_agreement"] = float((tok_c2 == tok_e).float().mean())
except Exception as e:
    res["compile_error"] = repr(e)[:800]
print(json.dumps(res, indent=1)); json.dump(res, open(a.out, "w"), indent=1)
