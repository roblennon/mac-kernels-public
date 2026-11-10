"""Reviewer's check of the 16-row headline on Qwen3-1.7B's MLP shapes: the tuned config against the
other mmh configs the builder's sweeps ranked near the top, F.linear at 1 and 16 rows (streaming and
library references), each streamed (weights rotated through >= 256 MB) and cache-resident (one copy).
Interleaved in shuffled order, --reps samples per arm of about 3 GB of weight each.

python3 bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10 -- <python> <this file> --dtype D --shape S --rows M,... --seed N --out f.json
"""
import argparse
import json
import math
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

SHAPES = {"q17_up": (2048, 6144), "q17_down": (6144, 2048)}
DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}
CANDIDATES = ["mmh_r1_u1_s4_h1", "mmh_r2_u1_s8", "mmh_r2_u1_s4", "mmh_r2_u1_s8_h1", "mmh_r1_u1_s8_h1",
              "mmh_r2_u1_s8_f1_h1", "mmh_r2_u1_s4_h1"]


def sync():
    torch.mps.synchronize()


def timed(fn, n, iters):
    sync()
    t = time.perf_counter()
    for i in range(iters):
        fn(i % n)
    sync()
    return (time.perf_counter() - t) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--shape", default="q17_up")
    ap.add_argument("--rows", default="16")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dt = DT[a.dtype]
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    K, N = SHAPES[a.shape]
    nbytes = K * N * 2
    ncopy = max(4, math.ceil(256e6 / nbytes))
    ws = [torch.randn(N, K, device="mps", dtype=dt) / math.sqrt(K) for _ in range(ncopy)]
    iters = ncopy * max(1, math.ceil(3e9 / nbytes / ncopy))
    res = {"torch": torch.__version__, "dtype": a.dtype, "shape": a.shape, "seed": a.seed, "rows": []}
    x1 = torch.randn(1, K, device="mps", dtype=dt)
    for M in [int(m) for m in a.rows.split(",")]:
        x = torch.randn(M, K, device="mps", dtype=dt)
        tuned = mrl.pick_config(M, K, N, dt)
        names = list(dict.fromkeys(([tuned.name()] if tuned else []) + CANDIDATES))
        if dt == torch.bfloat16:
            names = [n for n in names if not n.endswith("_h1")]
        arms = {"lin": lambda i: F.linear(x, ws[i]), "lin_1row": lambda i: F.linear(x1, ws[i])}
        for n in names:
            c = mrl.Config.parse(n)
            arms[n] = lambda i, c=c: mrl.multirow_linear(x, ws[i], config=c)
            arms[n + "|cached"] = lambda i, c=c: mrl.multirow_linear(x, ws[0], config=c)
        for f in arms.values():
            for i in range(ncopy):
                f(i)
        sync()
        samples = {k: [] for k in arms}
        for _ in range(a.reps):
            order = list(arms)
            rng.shuffle(order)
            for k in order:
                samples[k].append(timed(arms[k], ncopy, iters))
        med = {k: statistics.median(v) for k, v in samples.items()}
        res["rows"].append({"M": M, "tuned": tuned.name() if tuned else None, "samples_us": samples, "median_us": med,
                            "min_us": {k: min(v) for k, v in samples.items()}})
        print(json.dumps({"M": M, "tuned": tuned.name() if tuned else None,
                          "med": {k: round(v, 1) for k, v in sorted(med.items(), key=lambda kv: kv[1])}}), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
