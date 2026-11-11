"""Speculative decoding on MPS: what does verifying k drafted tokens cost? One forward of T = k + 1
new tokens against a static cache, batch 1, Qwen3-1.7B fp16, against a single-token step.
Speculative decoding pays when the tokens gained per verify (accepted drafts + 1) exceed
t(T) / t(1). Arm per process: eager, compiled, compiled_kernel (the multi-row kernel at 2 to 16 rows)."""
import argparse, json, os, statistics, sys, time
import torch
from transformers import AutoModelForCausalLM, StaticCache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_decode_compile_kernel import USE  # registers the custom op and the linear routing
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--T", default="1,2,3,5,9,13,17"); ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--repeat-first", action="store_true", help="measure the first T again at the end, as an A/A check")
ap.add_argument("--prompt-len", type=int, default=256)
ap.add_argument("--rewind", action="store_true", help="prefill once and rewind the cache counter before each call, "
                "instead of re-prefilling: the step right after a long prefill pays a one-off memory-clock penalty "
                "of about 30 ms on this M1 Max (2026-11-11 profile), so re-prefilling mis-times the steady step"); a = ap.parse_args()
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
P = a.prompt_len
USE["kernel"] = "kernel" in a.arm
if a.arm.endswith("smallq"):  # compiled + multirow kernel + the small-query attention kernel for T >= 4
    import smallq_hf; smallq_hf.enable(model)
def step(x, cache, pos): return model(x, past_key_values=cache, use_cache=True, cache_position=pos).logits
fn = step if a.arm == "eager" else torch.compile(step, dynamic=False)
g = torch.Generator().manual_seed(3)
res = {"arm": a.arm, "torch": torch.__version__, "prompt": P, "ms": {}, "compile_s": {}, "samples_ms": {}}
with torch.no_grad():
    cache = StaticCache(config=model.config, max_cache_len=P + 32)
    ids = torch.randint(1000, 30000, (1, P), generator=g).to("mps")
    def prefill():
        cache.reset()
        model(ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(P, device="mps"))
    def fresh():  # the static cache counts every update: re-fill the prompt, or with --rewind set the count back
        if a.rewind:
            for layer in cache.layers:
                layer.cumulative_length.fill_(P)
        else:
            prefill()
    if a.rewind:
        prefill()
    Ts = [int(t) for t in a.T.split(",")]
    for i, T in enumerate(Ts + (Ts[:1] if a.repeat_first else [])):
        key = f"{T}_end" if i == len(Ts) else T
        x = torch.randint(1000, 30000, (1, T), generator=g).to("mps"); pos = torch.arange(P, P + T, device="mps")
        fresh(); t0 = time.perf_counter(); fn(x, cache, pos); torch.mps.synchronize(); res["compile_s"][key] = time.perf_counter() - t0
        for _ in range(2): fresh(); fn(x, cache, pos)
        ts = []
        for _ in range(a.reps):
            fresh(); torch.mps.synchronize(); t = time.perf_counter(); fn(x, cache, pos); torch.mps.synchronize(); ts.append(time.perf_counter() - t)
        res["ms"][key] = statistics.median(ts) * 1e3; res["samples_ms"][key] = [t * 1e3 for t in ts]
        print(a.arm, "T", key, round(res["ms"][key], 2), "ms", "ratio to T=1:", round(res["ms"][key] / res["ms"][1], 2) if 1 in res["ms"] else "", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
