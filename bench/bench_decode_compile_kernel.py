"""Both decode levers together: torch.compile with a static cache (removes the small-kernel half of
a step) plus the multi-row Metal linear kernel at 2 to 16 rows (fixes the small-batch matrix path).
The kernel is registered as a custom op so Inductor treats it as opaque. Qwen3-1.7B fp16, one batch
size per process: eager, compiled, compiled+kernel. Reports ms per step and greedy agreement."""
import argparse, json, os, statistics, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, StaticCache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multirow_linear as mr
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

@torch.library.custom_op("mk::multirow_linear", mutates_args=())
def mk_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    x = x.contiguous()
    return mr.multirow_linear(x, w, b) if mr.supports(x, w, b) else F.linear(x, w, b)

@mk_linear.register_fake
def _(x, w, b):
    return x.new_empty((x.shape[0], w.shape[0]))

USE = {"kernel": False}
_orig = nn.Linear.forward
def _fwd(self, x):
    if USE["kernel"]:
        shp = x.shape; M = x.numel() // shp[-1]
        if 2 <= M <= 16:
            return torch.ops.mk.multirow_linear(x.reshape(M, shp[-1]), self.weight, self.bias).reshape(*shp[:-1], self.out_features)
    return _orig(self, x)
nn.Linear.forward = _fwd

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--batch", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="eager,compiled,compiled_kernel"); a = ap.parse_args()
    B, P, N = a.batch, 32, 32
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
    def sync(): torch.mps.synchronize()
    @torch.no_grad()
    def decode(step):
        g = torch.Generator().manual_seed(100 + B); ids = torch.randint(1000, 30000, (B, P), generator=g).to("mps")
        cache = StaticCache(config=model.config, max_cache_len=P + N)
        out = model(ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(P, device="mps"))
        nxt = out.logits[:, -1].argmax(-1, keepdim=True); toks = [nxt]; times = []
        for i in range(N - 1):
            sync(); t = time.perf_counter()
            logits = step(nxt, cache, torch.tensor([P + i], device="mps"))
            nxt = logits[:, -1].argmax(-1, keepdim=True); sync(); times.append(time.perf_counter() - t); toks.append(nxt)
        return statistics.median(times[4:]) * 1e3, torch.cat(toks, 1).cpu()
    def eager(nxt, cache, pos): return model(nxt, past_key_values=cache, use_cache=True, cache_position=pos).logits
    res = {"torch": torch.__version__, "batch": B, "ms": {}, "token_agreement_vs_eager": {}}
    ref = None
    for arm in a.arms.split(","):
        USE["kernel"] = arm.endswith("kernel")
        if arm == "eager":
            step = eager
        else:
            torch._dynamo.reset(); step = torch.compile(eager, dynamic=False)
            t0 = time.perf_counter(); decode(step); res.setdefault("compile_s", {})[arm] = time.perf_counter() - t0
        runs = [decode(step) for _ in range(2)]
        res["ms"][arm] = min(r[0] for r in runs)
        toks = runs[0][1]
        if ref is None: ref = toks
        res["token_agreement_vs_eager"][arm] = float((toks == ref).float().mean())
    base = res["ms"].get("eager")
    print(json.dumps({**res, "speedup_vs_eager": {k: round(base / v, 3) for k, v in res["ms"].items()} if base else None}))
    json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
