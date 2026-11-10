"""Reviewer's end-to-end check of round 2 on Qwen3-1.7B fp16 decode (independent of bench_multirow.py).

Four arms interleaved in shuffled order per repeat: stock (nn.Linear's own forward), stock again,
patched (2 to 16 flattened rows go to multirow_linear with pick_config; the table's "linear" cells and
other row counts go to F.linear), patched again. Greedy decode, 32-token prompts from a different
slice of docs/findings.md than the builder's, 32 new tokens, ms per decode step with prefill excluded.
Also: the linears alone in model order (stock / patched / stock / patched, 10 passes each, twice),
greedy token agreement, and teacher-forced logits on stock's tokens (patched vs stock at the same
batch, stock at batch B vs stock at batch 1 per row).

HF_HOME=<cache> HF_HUB_OFFLINE=1 python3 bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10 -- <python> <this file> --batches 8,16 --seed N --out f.json
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
from bench import multirow_linear as mrl  # noqa: E402

REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
STATE = {"patched": False, "kernel_calls": 0}


def patch(model):
    lins = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    for m in lins:
        def fwd(x, m=m):
            if STATE["patched"]:
                x2 = x.reshape(-1, x.shape[-1])
                M = x2.shape[0]
                if 2 <= M <= 16 and mrl.supports(x2, m.weight, m.bias):
                    cfg = mrl.pick_config(M, m.in_features, m.out_features, x2.dtype)
                    if cfg is not None:
                        STATE["kernel_calls"] += 1
                        return mrl.multirow_linear(x2, m.weight, m.bias, config=cfg).view(*x.shape[:-1], m.out_features)
            return F.linear(x, m.weight, m.bias)
        m._review_fwd = fwd
    return lins


def set_arm(lins, patched):
    for m in lins:
        if patched:
            m.forward = m._review_fwd
        else:
            m.__dict__.pop("forward", None)  # nn.Linear's own forward
    STATE["patched"] = patched


@torch.no_grad()
def decode(model, ids, steps, forced=None):
    out = model(ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    toks, logits = [nxt], [out.logits[:, -1].float()]
    torch.mps.synchronize()
    t = time.perf_counter()
    for s in range(steps - 1):
        feed = forced[:, s:s + 1].to(ids.device) if forced is not None else nxt
        out = model(feed, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        toks.append(nxt)
        logits.append(out.logits[:, -1].float())
    torch.mps.synchronize()
    ms = (time.perf_counter() - t) / (steps - 1) * 1e3
    return torch.cat(toks, 1).cpu(), torch.stack(logits, 1).cpu(), ms


@torch.no_grad()
def linears_ms(lins, B, n=10):
    xs = {}
    for m in lins:
        xs.setdefault(m.in_features, torch.randn(B, m.in_features, device="mps", dtype=torch.float16))
    for m in lins:
        m(xs[m.in_features])
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        for m in lins:
            m(xs[m.in_features])
    torch.mps.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="8,16")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-dir", default="", help="save ids, stock tokens and teacher-forced logits for check_decode_ref.py")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
    lins = patch(model)
    text = (ROOT / "docs" / "findings.md").read_text()
    all_ids = tok(text, return_tensors="pt").input_ids[0]
    res = {"torch": torch.__version__, "model": REPO, "revision": REV, "seed": a.seed, "n_linears": len(lins),
           "batches": []}
    for B in [int(b) for b in a.batches.split(",")]:
        ids = torch.stack([all_ids[5000 + 89 * i: 5000 + 89 * i + 32] for i in range(B)]).to("mps")
        for p in (False, True):
            set_arm(lins, p)
            decode(model, ids, 4)
        arms = {"stock": False, "stock_again": False, "patched": True, "patched_again": True}
        ms = {k: [] for k in arms}
        toks = {}
        calls = 0
        for _ in range(a.reps):
            order = list(arms)
            rng.shuffle(order)
            for k in order:
                set_arm(lins, arms[k])
                STATE["kernel_calls"] = 0
                t, _, m = decode(model, ids, a.steps)
                ms[k].append(m)
                toks.setdefault(k, []).append(t)
                if arms[k]:
                    calls = STATE["kernel_calls"]
        T = toks["stock"][0]
        stock_repeat = all(torch.equal(t, T) for k in ("stock", "stock_again") for t in toks[k])
        agree = [(t == T).float().mean().item() for k in ("patched", "patched_again") for t in toks[k]]
        set_arm(lins, False)
        _, LB, _ = decode(model, ids, a.steps, forced=T)
        set_arm(lins, True)
        _, PB, _ = decode(model, ids, a.steps, forced=T)
        set_arm(lins, False)
        div = []
        for k in ("patched",):
            for i in range(B):
                neq = (toks[k][0][i] != T[i]).nonzero()
                if len(neq):
                    st = int(neq[0])
                    top = LB[i, st].topk(2).values
                    div.append({"row": i, "step": st, "stock_top2_gap": (top[0] - top[1]).item()})
        if a.save_dir:
            Path(a.save_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"ids": ids.cpu(), "T": T, "LB": LB.half(), "PB": PB.half()}, Path(a.save_dir) / f"decode-b{B}-s{a.seed}.pt")
        rows = []
        for i in range(B):
            _, L1, _ = decode(model, ids[i:i + 1], a.steps, forced=T[i:i + 1])
            rows.append({"row": i, "patchedB_vs_stockB": (PB[i] - LB[i]).abs().max().item(),
                         "stockB_vs_stock1": (LB[i] - L1[0]).abs().max().item()})
        lin = {"stock": [], "patched": []}
        for p in (False, True, False, True):
            set_arm(lins, p)
            lin["patched" if p else "stock"].append(linears_ms(lins, B))
        set_arm(lins, False)
        med = {k: statistics.median(v) for k, v in ms.items()}
        row = {"batch": B, "ms_per_step": ms, "median": med, "kernel_calls_per_generate": calls,
               "stock_tokens_repeat": stock_repeat, "token_agreement": agree,
               "speedup": med["stock"] / med["patched"],
               "speedup_cross": [med[s] / med[p] for s in ("stock", "stock_again") for p in ("patched", "patched_again")],
               "aa": med["stock_again"] / med["stock"], "bb": med["patched_again"] / med["patched"],
               "logit_rows": rows, "first_divergence": div,
               "logit_max": {k: max(r[k] for r in rows) for k in ("patchedB_vs_stockB", "stockB_vs_stock1")},
               "rows_where_patch_exceeds_own_stock_spread": sum(r["patchedB_vs_stockB"] > r["stockB_vs_stock1"] for r in rows),
               "linears_only_ms": lin}
        res["batches"].append(row)
        print(json.dumps({"B": B, "ms": {k: [round(x, 2) for x in v] for k, v in ms.items()}, "speedup": round(row["speedup"], 3),
                          "aa": round(row["aa"], 3), "bb": round(row["bb"], 3), "calls": calls, "agree": agree,
                          "logit_max": row["logit_max"], "lin": {k: [round(x, 2) for x in v] for k, v in lin.items()}}), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
