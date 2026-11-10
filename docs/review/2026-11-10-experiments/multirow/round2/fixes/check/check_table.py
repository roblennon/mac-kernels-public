"""Checker's correctness run on every config in the final TUNED table (all row counts, not only 2-16).

Each config is computed live from bench/multirow_linear.py's TUNED. For each one, in the given
dtype: every M from 1 to 32 that the config fits, on small and odd shapes (K of 8, 24, 264, 520,
2056; N of 1, 7, 33, 517, 1027), x and W as 16-byte-aligned views at a nonzero storage offset,
bias at an odd element offset, with and without bias. Criterion and view helpers are the review's
(../../review/check_kernel2.py): max-normalised error within F.linear's + u + K*2^-24, relative L2
within 1.25x F.linear's + sqrt(K)*2^-24, all finite.

python3 bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10 -- <python> <this file> --dtype D --out f.json
"""
import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("check_kernel2", HERE.parents[1] / "review" / "check_kernel2.py")
ck = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ck)
mrl = ck.mrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dt = ck.DT[a.dtype]
    names = sorted({c for t in mrl.TUNED.values() for row in t.values() for c in row.values() if c != mrl.LINEAR})
    cfgs = [mrl.Config.parse(n) for n in names]
    torch.manual_seed(23)
    shapes = [(8, 1), (24, 7), (264, 33), (520, 517), (2056, 1027)]
    started = time.time()
    cases, fails, runs = [], [], 0
    per_cfg = {n: {"runs": 0, "bias_runs": 0, "Ms": set(), "max_norm_ratio": 0.0} for n in names}
    for M in range(1, 33):
        for K, N in shapes:
            x = torch.randn(M, K).to(dt)
            w = (torch.randn(N, K) / math.sqrt(K)).to(dt)
            for bias in (False, True):
                b = (torch.randn(N) * 0.1).to(dt) if bias else None
                gx, gw, gb = ck.views(x, w, b)
                assert gx.storage_offset() > 0 and gw.storage_offset() > 0 and mrl.supports(gx, gw, gb)
                ys = {"F.linear": F.linear(gx, gw, gb).cpu()}
                for c in cfgs:
                    if not c.fits(M):
                        continue
                    ys[c.name()] = mrl.multirow_linear(gx, gw, gb, config=c).cpu()
                    runs += 1
                v = ck.judge(x, w, b, ys)
                for k, o in v.items():
                    p = per_cfg[k]
                    p["runs"] += 1
                    p["bias_runs"] += bias
                    p["Ms"].add(M)
                    p["max_norm_ratio"] = max(p["max_norm_ratio"], o["norm_ratio"])
                bad = [k for k, o in v.items() if not o["pass"]]
                cases.append({"M": M, "K": K, "N": N, "bias": bias, "configs": len(v), "fails": bad})
                fails += [{"M": M, "K": K, "N": N, "bias": bias, "cfg": k, **v[k]} for k in bad]
        ck.emit(event="table", dtype=a.dtype, M=M, runs=runs, fails=len(fails))
    for p in per_cfg.values():
        p["Ms"] = sorted(p["Ms"])
    res = {"torch": torch.__version__, "dtype": a.dtype, "configs": names, "runs": runs, "fails": fails,
           "per_config": per_cfg, "cases": cases, "seconds": time.time() - started}
    Path(a.out).write_text(json.dumps(res, indent=1))
    ck.emit(event="done", runs=runs, fails=len(fails), seconds=res["seconds"])


if __name__ == "__main__":
    main()
