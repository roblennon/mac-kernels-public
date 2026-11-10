"""Reviewer's end-to-end checks on experiment B, Qwen3-1.7B fp16.

timing: three arms interleaved in one process: "native" (nn.Linear's own forward, the true
  stock), "stock" (the builder's wrapper with the patch off) and "patched".
spread: for every row of batch B, stock batch 1 on that row's prompt, fed stock batch B's
  tokens, against stock batch B and patched batch B fed the same tokens. This is the all-row
  version of the builder's row-0 batch-size spread.

HF_HOME=<cache> HF_HUB_OFFLINE=1 bench/gpu_runner.py --monitor --seconds 200 --rss-gib 10 -- <python> <this file> --out f.json
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from bench import bench_multirow as bm  # noqa: E402


def set_arm(lins, arm, saved):
    if arm == "native":
        for m in lins:
            m.__dict__.pop("forward", None)
    else:
        for m, f in zip(lins, saved):
            m.forward = f
    bm.Patch.enabled = arm == "patched"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--batches", default="1,8,16")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(bm.REPO, revision=bm.REV)
    model = AutoModelForCausalLM.from_pretrained(bm.REPO, revision=bm.REV, dtype=torch.float16).to("mps").eval()
    lins = bm.patch_linears(model)
    saved = [m.forward for m in lins]
    rng = random.Random(9)
    res = {"torch": torch.__version__, "steps": a.steps, "reps": a.reps, "batches": []}
    arms = ("native", "stock", "patched")
    for B in [int(b) for b in a.batches.split(",")]:
        ids = bm.prompts(tok, B).to("mps")
        for arm in arms:
            set_arm(lins, arm, saved)
            bm.generate(model, ids, 4)
        ms = {k: [] for k in arms}
        toks = {}
        for _ in range(a.reps):
            order = list(arms)
            rng.shuffle(order)
            for arm in order:
                set_arm(lins, arm, saved)
                t, _, m = bm.generate(model, ids, a.steps)
                ms[arm].append(m)
                toks.setdefault(arm, t)
        row = {"batch": B, "ms_per_step": ms, "median": {k: statistics.median(v) for k, v in ms.items()},
               "range": {k: [min(v), max(v)] for k, v in ms.items()},
               "native_tokens_equal_stock_wrapper": bool(torch.equal(toks["native"], toks["stock"]))}
        if B > 1:
            T = toks["native"]
            set_arm(lins, "native", saved)
            _, LB, _ = bm.generate(model, ids, a.steps, forced=T)
            set_arm(lins, "patched", saved)
            _, PB, _ = bm.generate(model, ids, a.steps, forced=T)
            set_arm(lins, "native", saved)
            per_row = []
            for i in range(B):
                _, L1, _ = bm.generate(model, ids[i:i + 1], a.steps, forced=T[i:i + 1])
                per_row.append({"row": i,
                                "stockB_vs_stock1": (LB[i] - L1[0]).abs().max().item(),
                                "patchedB_vs_stock1": (PB[i] - L1[0]).abs().max().item(),
                                "patchedB_vs_stockB": (PB[i] - LB[i]).abs().max().item()})
            row["spread_rows"] = per_row
            row["spread_max"] = {k: max(r[k] for r in per_row) for k in per_row[0] if k != "row"}
        res["batches"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("spread_rows", "ms_per_step")}), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
