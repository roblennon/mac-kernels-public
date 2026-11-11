"""Is the slow first weight pass after a long prefill a clock-state effect (a long compute-bound phase leaves the
memory system slow to ramp) or a memory-mapping effect (the prefill's footprint)? A bandwidth-bound sweep over
~2.8 GB of the model's linear weights is timed after different kinds of preceding GPU work."""
import json, statistics, sys, time
import torch
from transformers import AutoModelForCausalLM, StaticCache
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
sync = torch.mps.synchronize
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
P = 4096
g = torch.Generator().manual_seed(3)
lin_w = [m.weight for m in model.modules() if isinstance(m, torch.nn.Linear) and m.weight.shape[0] != 151936]
xs = torch.randn(1, 2048, device="mps", dtype=torch.float16); x6 = torch.randn(1, 6144, device="mps", dtype=torch.float16)
A = torch.randn(4096, 4096, device="mps", dtype=torch.float16)
def sweep():
    for w in lin_w: torch.nn.functional.linear(xs if w.shape[1] == 2048 else x6, w)
def t_sweep():
    sync(); t = time.perf_counter(); sweep(); sync(); return (time.perf_counter() - t) * 1e3
def gemm_for(sec):  # compute-bound, ~100 MB footprint
    t = time.perf_counter(); n = 0
    while time.perf_counter() - t < sec:
        for _ in range(4): A @ A
        sync(); n += 1
    return n
def touch(gb):  # bandwidth-bound, large fresh footprint, little compute
    b = torch.empty(int(gb * 2**30) // 2, device="mps", dtype=torch.float16); b.fill_(1.0); b.add_(1.0); sync(); del b
def stream_for(sec):  # bandwidth-bound over the weights themselves
    t = time.perf_counter()
    while time.perf_counter() - t < sec:
        sweep(); sync()
res = {}
with torch.no_grad():
    cache = StaticCache(config=model.config, max_cache_len=P + 32)
    ids = torch.randint(1000, 30000, (1, P), generator=g).to("mps")
    pre_t = []
    def prefill(n=P):
        cache.reset(); sync(); t = time.perf_counter()
        model(ids[:, :n], past_key_values=cache, use_cache=True); sync(); pre_t.append((n, time.perf_counter() - t))
    for _ in range(3): t_sweep()
    cases = {
        "steady": lambda: t_sweep(),
        "after_prefill4096": lambda: (prefill(), t_sweep())[-1],
        "after_prefill4096_2nd_sweep": lambda: (prefill(), sweep(), t_sweep())[-1],
        "after_prefill4096_sleep0.02": lambda: (prefill(), time.sleep(0.02), t_sweep())[-1],
        "after_gemm_0.3s": lambda: (gemm_for(0.3), t_sweep())[-1],
        "after_gemm_1.5s": lambda: (gemm_for(1.5), t_sweep())[-1],
        "after_touch_3GB": lambda: (touch(3), t_sweep())[-1],
        "after_stream_1.5s": lambda: (stream_for(1.5), t_sweep())[-1],
        "after_prefill4096_then_stream0.2s": lambda: (prefill(), stream_for(0.2), t_sweep())[-1],
    }
    samples = {n: [] for n in cases}
    for r in range(5):
        for n, f in cases.items(): samples[n].append(f())
        print("rep", r, {n: round(s[-1], 1) for n, s in samples.items()}, flush=True)
    res = {n: {"median": statistics.median(s), "samples": s} for n, s in samples.items()}
    res["prefill_seconds"] = pre_t
    print("SUMMARY", {n: round(v["median"], 1) for n, v in res.items() if isinstance(v, dict)}, "prefill_s", round(statistics.median(t for _, t in pre_t), 3), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
