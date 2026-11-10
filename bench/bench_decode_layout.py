"""End-to-end: does swapping every nn.Linear for mm on a pre-transposed weight speed up a real
model on MPS? Qwen3-1.7B (pinned revision), stock forward against patched forward in the same
process, interleaved, with output agreement checked.

HF_HOME=<hub parent> HF_HUB_OFFLINE=1 python3 bench/gpu_runner.py --monitor --seconds 290 --rss-gib 8 -- \
  .venv/bin/python bench/bench_decode_layout.py --dtype float16 --out f.json
"""
import argparse, json, statistics, time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

class Switch:
    on = False

def patch(model):
    n = 0
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            mod.register_buffer("w_t", mod.weight.detach().t().contiguous(), persistent=False)
            orig = mod.forward
            def fwd(x, mod=mod, orig=orig):
                if not Switch.on:
                    return orig(x)
                shp = x.shape
                y = torch.mm(x.reshape(-1, shp[-1]), mod.w_t)
                if mod.bias is not None:
                    y = y + mod.bias
                return y.reshape(*shp[:-1], y.shape[-1])
            mod.forward = fwd
            n += 1
    return n

def sync():
    torch.mps.synchronize()

@torch.no_grad()
def decode(model, batch, prompt_len, new_tokens, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, 30000, (batch, prompt_len), generator=g).to("mps")
    sync(); t0 = time.perf_counter()
    out = model(ids, use_cache=True)
    sync(); t_prefill = time.perf_counter() - t0
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    toks = [nxt]
    sync(); t1 = time.perf_counter()
    for _ in range(new_tokens - 1):
        out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        toks.append(nxt)
    sync(); t_decode = time.perf_counter() - t1
    return t_prefill, t_decode / (new_tokens - 1), torch.cat(toks, 1).cpu(), out.logits[:, -1].float().cpu()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float16"); ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--batches", default="1,4,8,16,24"); ap.add_argument("--prompt", type=int, default=32)
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=dt).to("mps").eval()
    n = patch(model); load = time.perf_counter() - t0
    res = {"torch": torch.__version__, "dtype": a.dtype, "linears_patched": n, "load_s": load, "cells": []}
    print("loaded", round(load, 1), "s; patched", n, "linears", flush=True)
    for arm in (False, True):  # warm both
        Switch.on = arm; decode(model, 1, 8, 4)
    for B in [int(b) for b in a.batches.split(",")]:
        cell = {"batch": B, "prompt": a.prompt, "new": a.new, "stock": [], "patched": []}
        for r in range(a.reps):
            for arm in ((False, True) if r % 2 == 0 else (True, False)):
                Switch.on = arm
                tp, td, toks, last = decode(model, B, a.prompt, a.new, seed=B)
                cell["patched" if arm else "stock"].append({"prefill_s": tp, "decode_s_per_step": td})
                if r == 0:
                    cell.setdefault("tokens", {})["patched" if arm else "stock"] = toks.tolist()
                    cell.setdefault("last_logits", {})["patched" if arm else "stock"] = last
        st, pt = cell["tokens"]["stock"], cell["tokens"]["patched"]
        agree = sum(a_ == b_ for sa, sb in zip(st, pt) for a_, b_ in zip(sa, sb)) / (len(st) * len(st[0]))
        ld = cell.pop("last_logits")
        cell["token_agreement"] = agree
        cell["first_step_note"] = "tokens compared greedily over all steps; early divergence propagates"
        md = lambda k, f: statistics.median(x[f] for x in cell[k])
        cell["summary"] = {"stock_ms_per_step": 1e3 * md("stock", "decode_s_per_step"),
                           "patched_ms_per_step": 1e3 * md("patched", "decode_s_per_step"),
                           "stock_prefill_ms": 1e3 * md("stock", "prefill_s"),
                           "patched_prefill_ms": 1e3 * md("patched", "prefill_s")}
        s = cell["summary"]
        print(f"B={B} decode ms/step stock {s['stock_ms_per_step']:.1f} patched {s['patched_ms_per_step']:.1f} "
              f"(x{s['stock_ms_per_step']/s['patched_ms_per_step']:.2f}); prefill {s['stock_prefill_ms']:.0f} vs "
              f"{s['patched_prefill_ms']:.0f} ms; token agreement {agree:.3f}", flush=True)
        del cell["tokens"]
        res["cells"].append(cell)
        json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
