"""Second pass on the first-step-after-prefill penalty. What part of the prefill causes it, and is the extra time
CPU-side (issuing the step) or GPU-side (waiting on it)? Each case ends with one timed T step; the cache counter is
always rewound or refilled so the step sees position 4096 (or N for the short prefills)."""
import argparse, json, statistics, time
import torch
from transformers import AutoModelForCausalLM, StaticCache
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, default=4096); ap.add_argument("--mode", default="compiled")
ap.add_argument("--T", type=int, default=1); ap.add_argument("--reps", type=int, default=6)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sync = torch.mps.synchronize
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
P, T = a.ctx, a.T
g = torch.Generator().manual_seed(3)
lin_w = [m.weight for m in model.modules() if isinstance(m, torch.nn.Linear)]
res = {"ctx": P, "mode": a.mode, "T": T, "cases": {}}
with torch.no_grad():
    cache = StaticCache(config=model.config, max_cache_len=P + 32)
    ids = torch.randint(1000, 30000, (1, P), generator=g).to("mps")
    x = torch.randint(1000, 30000, (1, T), generator=g).to("mps")
    def step(x, cache): return model(x, past_key_values=cache, use_cache=True).logits
    fn = step if a.mode == "eager" else torch.compile(step, dynamic=False)
    def prefill(n=P, keep="all"):
        cache.reset()
        model(ids[:, :n], past_key_values=cache, use_cache=True, **({} if keep == "all" else {"logits_to_keep": 1}))
        sync()
    def rewind(n):
        for layer in cache.layers: layer.cumulative_length.sub_(n)
    def t_step():
        sync(); t = time.perf_counter(); fn(x, cache); t1 = time.perf_counter(); sync(); t2 = time.perf_counter()
        return {"total": (t2 - t) * 1e3, "issue": (t1 - t) * 1e3}
    xs = torch.randn(1, 2048, device="mps", dtype=torch.float16); x6 = torch.randn(1, 6144, device="mps", dtype=torch.float16)
    def weights_sweep():  # read every decoder linear weight once (about 2.8 GB, not cache-resident)
        for w in lin_w:
            if w.shape[0] == 151936: continue
            torch.nn.functional.linear(xs if w.shape[1] == 2048 else x6, w)
        sync()
    def t_sweep():
        sync(); t = time.perf_counter(); weights_sweep(); return {"total": (time.perf_counter() - t) * 1e3, "issue": 0.0}
    def bigalloc(nbytes=4096 * 151936 * 2):
        t = torch.empty(nbytes // 2, device="mps", dtype=torch.float16); t.fill_(1.0); sync(); del t
    def stepper(n):
        def f():
            return t_step()
        return f
    prefill(); fn(x, cache); sync()
    for _ in range(3): rewind(T); fn(x, cache); sync()
    for _ in range(2): weights_sweep()
    cases = {
        "steady_step": lambda: (rewind(T), t_step())[-1],
        "steady_sweep": lambda: t_sweep(),
        "prefill_then_sweep": lambda: (prefill(), t_sweep())[-1],
        "prefill_then_step": lambda: (prefill(), t_step())[-1],
        "prefill_sweep_then_step": lambda: (prefill(), weights_sweep(), t_step())[-1],
        "prefill_last_then_step": lambda: (prefill(keep="last"), t_step())[-1],
        "bigalloc1.2G_then_step": lambda: (rewind(T), bigalloc(), t_step())[-1],
        "bigalloc1.2G_emptycache_then_step": lambda: (rewind(T), bigalloc(), torch.mps.empty_cache(), t_step())[-1],
        "prefill1024_then_step": lambda: (prefill(1024), t_step())[-1],
        "prefill2048_then_step": lambda: (prefill(2048), t_step())[-1],
        "prefill2048last_then_step": lambda: (prefill(2048, "last"), t_step())[-1],
    }
    samples = {n: [] for n in cases}
    mems = {}
    for r in range(a.reps):
        for n, f in cases.items():
            samples[n].append(f())
            if r == 0: mems[n] = {"cur_MiB": torch.mps.current_allocated_memory() / 2**20, "drv_MiB": torch.mps.driver_allocated_memory() / 2**20}
        print("rep", r, {n: (round(s[-1]["total"], 1), round(s[-1]["issue"], 1)) for n, s in samples.items()}, flush=True)
    for n, s in samples.items():
        tot = [v["total"] for v in s]; iss = [v["issue"] for v in s]
        res["cases"][n] = {"median": statistics.median(tot), "issue_median": statistics.median(iss), "samples": tot, "issue_samples": iss}
    res["mem_after_case_rep0"] = mems
    print(a.mode, P, "T", T, {n: (round(v["median"], 1), round(v["issue_median"], 1)) for n, v in res["cases"].items()}, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
