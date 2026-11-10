"""Reviewer's timing check of round 2 (independent of bench_multirow.py's timing code).

Per (dtype, shape, M): six arms interleaved in shuffled order, every sample a whole number of passes
over weight copies totalling at least 256 MB (2.5 GB for the head), about 3 GB of weight per sample:
  lin, lin2     F.linear twice (A/A)
  k, k2         the round-2 table's choice twice (B/B); where the table routes the cell to F.linear,
                the round-2 sweep winner is timed instead and the row says so
  r1            round 1's table choice for the same cell (commit a441c78), if it had one
  k_cached      the round-2 choice on one weight copy, re-used every call (cache-resident when it fits)
At 16 rows also lin32: F.linear at 32 rows on the same weights (Apple's kernel at its best tile).
Before the first GPU call, the process records the GPU's utilisation from ioreg (other GPU users).

python3 bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10 -- <python> <this file> --dtype D --cells SPEC --seed N --out f.json
SPEC: shape:M,M,...;shape:M,...
"""
import argparse
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
from bench import multirow_linear as mrl  # noqa: E402

SHAPES = {"q17_qo": (2048, 2048), "q17_kv": (2048, 1024), "q17_up": (2048, 6144), "q17_down": (6144, 2048),
          "q17_head": (2048, 151936), "q4b_up": (2560, 9728), "llama7b_up": (4096, 11008)}
DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}
R2DIR = Path(__file__).resolve().parents[1]


def round1_table():
    src = subprocess.run(["git", "show", "a441c78:bench/multirow_linear.py"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    ns = {}
    i = src.index("TUNED: dict")
    j = src.index("\n}\n", i) + 2
    exec("TUNED = " + src[src.index("{", i):j], ns)
    return ns["TUNED"]


def sweep_winner(dname, shape, M):
    p = R2DIR / f"tune-{dname}-{shape}.json"
    if p.exists():
        for r in json.loads(p.read_text())["rows"]:
            if r["M"] == M:
                return r["best"]
    return None


def gpu_util():
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


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
    ap.add_argument("--cells", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    util = [gpu_util() for _ in range(5)]
    dname, dt = a.dtype, DT[a.dtype]
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    r1 = round1_table()[dname]
    res = {"torch": torch.__version__, "dtype": dname, "seed": a.seed, "gpu_util_before": util, "rows": []}
    t0 = time.time()
    for spec in a.cells.split(";"):
        shape, ms = spec.split(":")
        K, N = SHAPES[shape]
        nbytes = K * N * 2
        ncopy = max(4, math.ceil((2.5e9 if N > 100000 else 256e6) / nbytes))
        ws = [torch.randn(N, K, device="mps", dtype=dt) / math.sqrt(K) for _ in range(ncopy)]
        iters = ncopy * max(1, math.ceil(3e9 / nbytes / ncopy))
        for M in [int(m) for m in ms.split(",")]:
            x = torch.randn(M, K, device="mps", dtype=dt)
            cfg = mrl.pick_config(M, K, N, dt)
            routed = cfg is not None
            if cfg is None:
                cfg = mrl.Config.parse(sweep_winner(dname, shape, M))
            old = r1.get((K, N), {}).get(M)
            arms = {"lin": lambda i: F.linear(x, ws[i]), "lin2": lambda i: F.linear(x, ws[i]),
                    "k": lambda i: mrl.multirow_linear(x, ws[i], config=cfg),
                    "k2": lambda i: mrl.multirow_linear(x, ws[i], config=cfg),
                    "k_cached": lambda i: mrl.multirow_linear(x, ws[0], config=cfg)}
            if old:
                oc = mrl.Config.parse(old)
                arms["r1"] = lambda i: mrl.multirow_linear(x, ws[i], config=oc)
            if M == 16:
                x32 = torch.randn(32, K, device="mps", dtype=dt)
                arms["lin32"] = lambda i: F.linear(x32, ws[i])
            for f in arms.values():
                for i in range(ncopy):
                    f(i)
            sync()
            # correctness of what is timed: each kernel arm against F.linear on copy 0
            ref = F.linear(x, ws[0]).float()
            yk = mrl.multirow_linear(x, ws[0], config=cfg).float()
            relk = ((yk - ref).norm() / ref.norm()).item()
            samples = {k: [] for k in arms}
            for _ in range(a.reps):
                order = list(arms)
                rng.shuffle(order)
                for k in order:
                    samples[k].append(timed(arms[k], ncopy, iters))
            med = {k: statistics.median(v) for k, v in samples.items()}
            row = {"shape": shape, "K": K, "N": N, "M": M, "config": cfg.name(), "table_routes_to_kernel": routed,
                   "round1_config": old, "copies": ncopy, "iters": iters, "samples_us": samples, "median_us": med,
                   "min_us": {k: min(v) for k, v in samples.items()},
                   "max_GBps_streamed": max(nbytes / v / 1e3 for k, vs in samples.items() if k not in ("k_cached",) for v in vs),
                   "rel_l2_vs_linear": relk,
                   "speedup": med["lin"] / med["k"], "aa": med["lin2"] / med["lin"], "bb": med["k2"] / med["k"],
                   "speedup_cross": [med[l] / med[k] for l in ("lin", "lin2") for k in ("k", "k2")]}
            if "lin32" in med:
                row["TFLOPS_lin32"] = 2 * 32 * N * K / med["lin32"] / 1e6
            row["TFLOPS_k"] = 2 * M * N * K / med["k"] / 1e6
            res["rows"].append(row)
            print(json.dumps({"shape": shape, "M": M, "cfg": cfg.name(), "routed": routed,
                              "us": {k: round(v, 1) for k, v in med.items()}, "sp": round(row["speedup"], 2),
                              "aa": round(row["aa"], 3), "bb": round(row["bb"], 3),
                              "maxGBps": round(row["max_GBps_streamed"]), "t": round(time.time() - t0)}), flush=True)
            Path(a.out).write_text(json.dumps(res, indent=1))
        del ws
        torch.mps.empty_cache()
    res["gpu_util_after"] = [gpu_util() for _ in range(3)]
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
