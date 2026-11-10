"""Decode-step latency rerun, Qwen3-1.7B fp16 on MPS. One process runs every arm in the order given
and keeps every sample; the driver runs several processes with rotated orders, and rerun_analyze.py
pairs arms within each process. Arms:
  eager_dynamic  transformers' default path: no compile, DynamicCache
  eager_static   no compile, StaticCache (the baseline the earlier numbers used)
  compiled_a/_b  torch.compile(dynamic=False) with a StaticCache, compiled separately: an A/A pair
  kernel_a/_b    compiled plus the multi-row kernel at 2 to 16 rows: an A/A pair
At batch 1 the kernel is never called, so all four compiled arms are nominally the same pipeline.
Each arm: one untimed decode (compiles where needed), then --repeats timed decodes of --steps
steps. Per decode the record keeps prefill time, every step time and the generated tokens."""
import argparse, json, os, subprocess, sys, time
import torch
from transformers import AutoModelForCausalLM, DynamicCache, StaticCache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_decode_compile_kernel import USE  # registers mk::multirow_linear and the nn.Linear routing
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ARMS = ("eager_dynamic", "eager_static", "compiled_a", "compiled_b", "kernel_a", "kernel_b")


def thermal():
    try:
        return subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception as e:  # recorded, never fatal
        return f"unavailable: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True); ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--steps", type=int, default=32); ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--order", default=",".join(ARMS)); ap.add_argument("--process", type=int, default=0)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    order = a.order.split(",")
    assert set(order) <= set(ARMS), order
    B, P, N = a.batch, a.prompt_len, a.steps
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
    g = torch.Generator().manual_seed(100 + B + P)
    ids = torch.randint(1000, 30000, (B, P), generator=g).to("mps")
    sync = torch.mps.synchronize

    def eager(x, cache, pos):
        return model(x, past_key_values=cache, use_cache=True, cache_position=pos).logits

    @torch.no_grad()
    def decode(step, dynamic):
        cache = DynamicCache(config=model.config) if dynamic else StaticCache(config=model.config, max_cache_len=P + N)
        sync(); t = time.perf_counter()
        out = model(ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(P, device="mps"))
        nxt = out.logits[:, -1].argmax(-1, keepdim=True); sync(); prefill = (time.perf_counter() - t) * 1e3
        toks, times = [nxt], []
        for i in range(N - 1):
            sync(); t = time.perf_counter()
            logits = step(nxt, cache, torch.tensor([P + i], device="mps"))
            nxt = logits[:, -1].argmax(-1, keepdim=True); sync()
            times.append((time.perf_counter() - t) * 1e3); toks.append(nxt)
        return prefill, times, torch.cat(toks, 1).cpu().tolist()

    res = {"torch": torch.__version__, "batch": B, "prompt_len": P, "steps": N, "repeats": a.repeats,
           "process": a.process, "order": order, "thermal_start": thermal(), "arms": {}}
    for arm in order:
        USE["kernel"] = arm.startswith("kernel")
        dynamic = arm == "eager_dynamic"
        if arm.startswith("eager"):
            step = eager
        else:
            torch._dynamo.reset(); step = torch.compile(eager, dynamic=False)
        t = time.perf_counter(); decode(step, dynamic); warm_s = time.perf_counter() - t
        runs = [decode(step, dynamic) for _ in range(a.repeats)]
        res["arms"][arm] = {"warm_s": warm_s, "prefill_ms": [r[0] for r in runs], "step_ms": [r[1] for r in runs],
                            "tokens": runs[0][2]}
        med = sorted(x for r in runs for x in r[1][4:])
        print(arm, f"median step {med[len(med) // 2]:.2f} ms, warm {warm_s:.1f} s", flush=True)
    USE["kernel"] = False
    res["thermal_end"] = thermal()
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
