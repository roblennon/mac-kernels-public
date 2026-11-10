"""Reviewer's timing checks on experiment B.

Is 16 rows on the 2048->6144 projection really bound by arithmetic? Time the tuned kernel,
the mmr variant the README names, and F.linear with the weight streamed (rotated copies,
about 277 MB) and cache-resident (one copy), interleaved, and report achieved TFLOPS next to
F.linear's own TFLOPS at 32 rows, which bounds what the GPU can do on this shape.

Run through the GPU lock: bench/gpu_runner.py --monitor --seconds 200 --rss-gib 10 -- <python> <this file> --out f.json
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from bench import multirow_linear as mrl  # noqa: E402
from bench import bench_multirow as bm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()
    torch.manual_seed(5)
    rng = random.Random(5)
    rows = []
    for dt in (torch.float16,):
        for shape in ("q17_up", "q17_down"):
            K, N = bm.SHAPES[shape]
            ws, _ = bm.weight_copies(K, N, dt)
            nc = len(ws)
            iters = bm.iters_for(K, N, nc, 1.5e9)
            for M in (8, 16, 32):
                x = torch.randn(M, K, device="mps", dtype=dt)
                cfgs = {"tuned:" + mrl.pick_config(M, K, N, dt).name(): mrl.pick_config(M, K, N, dt),
                        "mmr_r2_u1_s4": mrl.Config("mmr", 2, 1, 4), "mma_r1_u1_s4": mrl.Config("mma", 1, 1, 4)}
                arms = {}
                for tag, c in cfgs.items():
                    if not c.fits(M):
                        continue
                    arms[tag + "|stream"] = lambda i, c=c: mrl.multirow_linear(x, ws[i], config=c)
                    arms[tag + "|cached"] = lambda i, c=c: mrl.multirow_linear(x, ws[0], config=c)
                arms["F.linear|stream"] = lambda i: F.linear(x, ws[i])
                arms["F.linear|cached"] = lambda i: F.linear(x, ws[0])
                samples = bm.interleaved(arms, nc, iters, a.reps, rng)
                med = {k: statistics.median(v) for k, v in samples.items()}
                flop = 2 * M * N * K
                row = {"dtype": str(dt), "shape": shape, "K": K, "N": N, "M": M, "copies": nc, "iters": iters,
                       "median_us": med, "TFLOPS": {k: flop / v / 1e6 for k, v in med.items()},
                       "GBps": {k: K * N * 2 / v / 1e3 for k, v in med.items()},
                       "memory_floor_us_at_400GBps": K * N * 2 / 400e3}
                rows.append(row)
                print(json.dumps({"shape": shape, "M": M, "us": {k: round(v, 1) for k, v in med.items()},
                                  "TFLOPS": {k: round(v, 2) for k, v in row["TFLOPS"].items()}}), flush=True)
            del ws
            torch.mps.empty_cache()
    Path(a.out).write_text(json.dumps({"torch": torch.__version__, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
