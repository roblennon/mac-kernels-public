"""Static checks of the round-2 raw JSON (no GPU): bandwidth plausibility of every timed sample,
whether the layer runs timed the configs the final TUNED table holds, whether each claimed gain
clears both noise arms, and regressions at 1 to 8 rows against round 1.

python3 check_raw.py > check-raw.json   (run from this folder; reads ../*.json and ../../*.json)
"""
import ast
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.dirname(HERE)
R1 = os.path.dirname(R2)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(R1))))
SHAPES = {"q17_qo": (2048, 2048), "q17_kv": (2048, 1024), "q17_up": (2048, 6144), "q17_down": (6144, 2048),
          "q17_head": (2048, 151936), "q4b_up": (2560, 9728), "llama7b_up": (4096, 11008)}
LIMIT = 400.0  # GB/s, M1 Max peak


def final_tuned():
    src = open(os.path.join(REPO, "bench", "multirow_linear.py")).read()
    i = src.index("TUNED: dict")
    j = src.index("\n}\n", i) + 2
    body = src[src.index("{", i):j].replace("LINEAR", "'linear'")
    return ast.literal_eval(body)


def gbps(K, N, us):
    return K * N * 2 / us / 1e3


def bandwidth():
    """Every timed sample of every arm in every timing file: max GB/s, and arms above LIMIT."""
    out = {"files": {}, "over_limit": []}
    files = sorted(glob.glob(os.path.join(R2, "*.json")))
    for f in files:
        name = os.path.basename(f)
        if name.startswith(("correctness", "summary", "peak", "check")):
            continue
        d = json.load(open(f))
        best = 0.0
        for r in d.get("rows", []):
            K, N = r["K"], r["N"]
            for arm, xs in r.get("samples_us", {}).items():
                # probe arms that deliberately do not stream (l1*, now_*) are excluded
                if "|" in arm and arm.split("|")[1] not in ("base", "nomma", "nox_nomma"):
                    continue
                for v in xs:
                    g = gbps(K, N, v)
                    best = max(best, g)
                    if g > LIMIT:
                        out["over_limit"].append({"file": name, "shape": r.get("shape"), "M": r["M"], "arm": arm,
                                                  "us": v, "GBps": g})
        out["files"][name] = round(best, 1)
    return out


def layer_cells():
    cells = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(R2, "layers-run*.json"))):
        for r in json.load(open(f))["rows"]:
            r["_file"] = os.path.basename(f)
            cells[(r["dtype"], r["shape"], r["M"])].append(r)
    return cells


def config_match(cells, tuned):
    """Did the layer runs time the config the final table selects?"""
    mism = []
    for (dt, shape, M), rs in sorted(cells.items()):
        t = tuned[dt].get(SHAPES[shape], {})
        if M not in t:
            continue
        want = t[M]
        for r in rs:
            if want == "linear":
                continue
            if r["config"] != want:
                mism.append({"dtype": dt, "shape": shape, "M": M, "file": r["_file"], "timed": r["config"],
                             "final_table": want})
    return mism


def noise_bounds(cells, tuned):
    """Per cell: speedup range over processes, the worst noise deviation of either arm, and whether
    the slowest speedup clears 1 + noise. Also the B/B arm's own range, to see the kernel's spread."""
    rows = []
    for (dt, shape, M), rs in sorted(cells.items()):
        sp = [r["linear_over_multirow"] for r in rs]
        aa = [r["aa_ratio"] for r in rs]
        bb = [r["bb_ratio"] for r in rs]
        noise = max(max(abs(a - 1) for a in aa), max(abs(b - 1) for b in bb))
        # per-process speedup using every pairing of the two arms of each kind
        cross = []
        for r in rs:
            for la in ("F.linear", "F.linear_again"):
                for ka in ("multirow", "multirow_again"):
                    cross.append(r["median_us"][la] / r["median_us"][ka])
        t = tuned[dt].get(SHAPES[shape], {}).get(M)
        rows.append({"dtype": dt, "shape": shape, "M": M, "n_proc": len(rs), "config": rs[0]["config"],
                     "final": t, "speedup_min": min(sp), "speedup_max": max(sp), "cross_min": min(cross),
                     "cross_max": max(cross), "noise": noise, "clears": min(sp) > 1 + noise,
                     "kernel_med_us": [r["median_us"]["multirow"] for r in rs],
                     "linear_med_us": [r["median_us"]["F.linear"] for r in rs]})
    return rows


def round1_fastest():
    """Fastest kernel sample per (dtype, shape, M) from round 1's layers.json and its review reruns."""
    best = {}
    paths = [os.path.join(R1, "layers.json")] + sorted(glob.glob(os.path.join(R1, "review", "layers-rerun-*.json")))
    for p in paths:
        if not os.path.exists(p):
            continue
        for r in json.load(open(p))["rows"]:
            key = (r["dtype"], r["shape"], r["M"])
            v = min(r["samples_us"]["multirow"])
            best[key] = min(best.get(key, 1e18), v)
            lk = key + ("lin",)
            best[lk] = min(best.get(lk, 1e18), min(r["samples_us"]["F.linear"] + r["samples_us"].get("F.linear_again", [])))
    return best


def regressions(cells, r1):
    """Round 2 against round 1 at 1 to 8 rows (and 24, 32): fastest kernel sample and median ratio,
    normalised by F.linear's fastest sample in the same runs to remove the outside-load difference."""
    out = []
    for (dt, shape, M), rs in sorted(cells.items()):
        key = (dt, shape, M)
        if key not in r1 or M > 8 and M not in (24, 32):
            continue
        k2 = min(min(r["samples_us"]["multirow"] + r["samples_us"]["multirow_again"]) for r in rs)
        l2 = min(min(r["samples_us"]["F.linear"] + r["samples_us"]["F.linear_again"]) for r in rs)
        k1, l1 = r1[key], r1[key + ("lin",)]
        out.append({"dtype": dt, "shape": shape, "M": M, "r1_kernel_fastest": k1, "r2_kernel_fastest": k2,
                    "ratio": k2 / k1, "r1_linear_fastest": l1, "r2_linear_fastest": l2,
                    "ratio_normalised": (k2 / l2) / (k1 / l1), "r2_config": rs[0]["config"]})
    return out


def tune_winners(tuned):
    """Does TUNED hold the tune sweep's winner for each cell? (rows 2 to 16)"""
    diff = []
    for f in sorted(glob.glob(os.path.join(R2, "tune-*.json"))):
        d = json.load(open(f))
        dt = d["dtype"]
        for r in d["rows"]:
            t = tuned[dt].get((r["K"], r["N"]), {}).get(r["M"])
            if t is None:
                continue
            if t != r["best"] and t != "linear":
                diff.append({"file": os.path.basename(f), "M": r["M"], "best": r["best"], "table": t})
            # how close was the runner-up? margin of the winner over the second best median
            meds = sorted(v for k, v in r["median_us"].items() if k != "F.linear")
    return diff


def tune_margins():
    """Winner's median vs the second-best median in each tune cell (how arbitrary the pick is)."""
    m = []
    for f in sorted(glob.glob(os.path.join(R2, "tune-*.json"))):
        d = json.load(open(f))
        for r in d["rows"]:
            meds = sorted((v, k) for k, v in r["median_us"].items() if k != "F.linear")
            m.append((meds[1][0] / meds[0][0]) - 1)
    return {"cells": len(m), "median_margin": statistics.median(m), "within_1pct": sum(x < 0.01 for x in m),
            "within_3pct": sum(x < 0.03 for x in m)}


def decode_tables():
    out = []
    for f in sorted(glob.glob(os.path.join(R2, "decode-*.json"))):
        d = json.load(open(f))
        for b in d["batches"]:
            out.append({"file": os.path.basename(f), "batch": b["batch"], "ms": b["ms_per_step"],
                        "linears": b["linears_only_ms"], "calls": b["kernel_calls_per_generate"],
                        "agree": b["token_agreement_patched_vs_stock"][0], "logit_max": b["logit_max"]})
    return out


def decode_table_matches(tuned):
    out = []
    for f in sorted(glob.glob(os.path.join(R2, "decode-*.json"))):
        d = json.load(open(f))
        t = {dt: {tuple(map(int, k.split("x"))): {int(m): c for m, c in v.items()} for k, v in tt.items()}
             for dt, tt in d["tuned"].items()}
        out.append({"file": os.path.basename(f), "same_as_final": t == tuned})
    return out


def main():
    tuned = final_tuned()
    cells = layer_cells()
    r1 = round1_fastest()
    nb = noise_bounds(cells, tuned)
    res = {
        "bandwidth": bandwidth(),
        "layer_configs_not_in_final_table": config_match(cells, tuned),
        "tuned_not_sweep_winner": tune_winners(tuned),
        "tune_margins": tune_margins(),
        "noise": nb,
        "kept_cells_not_clearing_noise": [r for r in nb if r["final"] not in (None, "linear") and 2 <= r["M"] <= 16
                                          and not r["clears"]],
        "regressions_1_to_8_vs_round1": [r for r in regressions(cells, r1)],
        "decode": decode_tables(),
        "decode_table_is_final": decode_table_matches(tuned),
    }
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
