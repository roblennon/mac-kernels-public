"""Why is the first step after a long prefill slow? One T step (compiled or eager), timed after different
things happen between the prefill and the step. Also a fixed non-model GPU workload timed the same way,
to separate 'the GPU is slow right now' from 'this model step is slow right now'."""
import argparse, json, statistics, time
import torch
from transformers import AutoModelForCausalLM, StaticCache
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, default=4096); ap.add_argument("--mode", default="compiled")
ap.add_argument("--T", type=int, default=1); ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--cases", default="all"); ap.add_argument("--out", required=True)
a = ap.parse_args()
sync = torch.mps.synchronize
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
P, T = a.ctx, a.T
g = torch.Generator().manual_seed(3)
W = torch.randn(6144, 2048, device="mps", dtype=torch.float16)
xw = torch.randn(1, 2048, device="mps", dtype=torch.float16)
def fixed_work():  # ~ one MLP-sized weight read, 84 times (roughly the 28 layers x 3 MLP mats)
    y = None
    for _ in range(84): y = torch.nn.functional.linear(xw, W)
    return y
res = {"ctx": P, "mode": a.mode, "T": T, "cases": {}}
with torch.no_grad():
    cache = StaticCache(config=model.config, max_cache_len=P + 32)
    ids = torch.randint(1000, 30000, (1, P), generator=g).to("mps")
    x = torch.randint(1000, 30000, (1, T), generator=g).to("mps")
    def step(x, cache): return model(x, past_key_values=cache, use_cache=True).logits
    fn = step if a.mode == "eager" else torch.compile(step, dynamic=False)
    def prefill(keep="all"):
        cache.reset()
        model(ids, past_key_values=cache, use_cache=True, **({} if keep == "all" else {"logits_to_keep": 1}))
        sync()
    def rewind(n):
        for layer in cache.layers: layer.cumulative_length.sub_(n)
    def t_step():
        sync(); t = time.perf_counter(); fn(x, cache); sync(); return (time.perf_counter() - t) * 1e3
    def t_fixed():
        sync(); t = time.perf_counter(); fixed_work(); sync(); return (time.perf_counter() - t) * 1e3
    prefill(); fn(x, cache); sync()
    for _ in range(3): rewind(T); fn(x, cache); sync()
    for _ in range(3): t_fixed()

    def mem(): return {"cur_MiB": torch.mps.current_allocated_memory() / 2**20, "drv_MiB": torch.mps.driver_allocated_memory() / 2**20}

    cases = {
        "steady_step": lambda: (rewind(T), t_step())[-1],
        "steady_fixed": lambda: t_fixed(),
        "prefill_all_then_step": lambda: (prefill("all"), t_step())[-1],
        "prefill_all_then_fixed": lambda: (prefill("all"), t_fixed())[-1],
        "prefill_last_then_step": lambda: (prefill("last"), t_step())[-1],
        "prefill_all_emptycache_then_step": lambda: (prefill("all"), torch.mps.empty_cache(), t_step())[-1],
        "prefill_all_fixed_then_step": lambda: (prefill("all"), t_fixed(), t_step())[-1],
        "prefill_all_sleep0.05_then_step": lambda: (prefill("all"), time.sleep(0.05), t_step())[-1],
        "prefill_all_sleep0.3_then_step": lambda: (prefill("all"), time.sleep(0.3), t_step())[-1],
        "sleep0.05_then_step": lambda: (rewind(T), time.sleep(0.05), t_step())[-1],
        "sleep0.3_then_step": lambda: (rewind(T), time.sleep(0.3), t_step())[-1],
        "sleep1_then_step": lambda: (rewind(T), time.sleep(1.0), t_step())[-1],
        "sleep3_then_step": lambda: (rewind(T), time.sleep(3.0), t_step())[-1],
        "sleep3_then_fixed": lambda: (time.sleep(3.0), t_fixed())[-1],
        "sleep3_then_fixed_then_step": lambda: (rewind(T), time.sleep(3.0), t_fixed(), t_step())[-1],
    }
    names = list(cases) if a.cases == "all" else a.cases.split(",")
    # interleave cases across reps so drift hits all cases alike
    samples = {n: [] for n in names}
    for r in range(a.reps):
        for n in names:
            samples[n].append(cases[n]())
        print("rep", r, {n: round(samples[n][-1], 1) for n in names}, flush=True)
    for n in names:
        s = samples[n]; res["cases"][n] = {"median": statistics.median(s), "min": min(s), "max": max(s), "samples": s}
    res["mem_after"] = mem()
    print(a.mode, P, "T", T, {n: round(v["median"], 2) for n, v in res["cases"].items()}, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
