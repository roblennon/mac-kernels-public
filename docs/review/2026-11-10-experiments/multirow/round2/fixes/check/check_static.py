"""Static checks of the round-2 fixes (no GPU): frozen round-1 table, round-2 grid, table rule,
fallback rule and re-time data. Run from the repository root:

    python3 docs/review/2026-11-10-experiments/multirow/round2/fixes/check/check_static.py

Writes check-static.json next to this file. Reads committed sources with `git show`.
"""
import ast
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXES = HERE.parent
ROUND2 = FIXES.parent
REPO = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip())
sys.path.insert(0, str(REPO))

SHAPES = {"q17_qo": (2048, 2048), "q17_kv": (2048, 1024), "q17_up": (2048, 6144), "q17_down": (6144, 2048),
          "q17_head": (2048, 151936), "q4b_up": (2560, 9728), "llama7b_up": (4096, 11008)}
BY_KN = {v: k for k, v in SHAPES.items()}


def tuned_from_source(text):
    """The TUNED literal of a multirow_linear.py source, with LINEAR resolved to "linear"."""
    tree = ast.parse(text)
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == "TUNED":
            class Fix(ast.NodeTransformer):
                def visit_Name(self, n):
                    return ast.copy_location(ast.Constant("linear"), n) if n.id == "LINEAR" else n
            return ast.literal_eval(Fix().visit(node.value))
    raise SystemExit("no TUNED")


def git_show(rev, path):
    return subprocess.check_output(["git", "show", f"{rev}:{path}"], cwd=REPO, text=True)


def main():
    res = {}
    import bench.multirow_linear as mrl
    import bench.bench_multirow as bm

    # 1. Frozen round-1 table equals commit a441c78's TUNED.
    r1 = tuned_from_source(git_show("a441c78", "bench/multirow_linear.py"))
    res["round1_frozen_equals_a441c78"] = r1 == bm.ROUND1_TUNED
    # And it is the table in force between a441c78 and 1c48161 (only 1c48161 changed the file).
    log = subprocess.check_output(["git", "log", "--format=%h", "a441c78..1c48161", "--", "bench/multirow_linear.py"],
                                  cwd=REPO, text=True).split()
    res["commits_touching_mrl_after_a441c78_up_to_1c48161"] = log

    # 2. tune_grid_round2 equals the configs list of every round-2 tune file, in order.
    grid = {dt: [c.name() for c in bm.tune_grid_round2(dt)] for dt in ("float16", "bfloat16")}
    res["grid_sizes"] = {dt: len(g) for dt, g in grid.items()}
    mism = []
    for p in sorted(ROUND2.glob("tune-*.json")):
        d = json.loads(p.read_text())
        if d["configs"] != grid[d["dtype"]]:
            mism.append(p.name)
    res["tune_files_checked"] = len(list(ROUND2.glob("tune-*.json")))
    res["tune_files_grid_mismatch"] = mism
    rep = json.loads((FIXES / "tune-repro-float16-q17_kv.json").read_text())
    res["repro_file_grid_equal"] = rep["configs"] == grid["float16"]

    # 3. Live table: rows 1, 24, 32 unchanged from 1c48161; rows 2-16 equal selection.json; no LINEAR;
    #    candidate set and band constancy as the README states.
    prev = tuned_from_source(git_show("1c48161", "bench/multirow_linear.py"))
    sel = json.loads((FIXES / "selection.json").read_text())["choice"]
    cand = {(2, 3): {"dev_r2_u1_s4", "dev_r2_u1_s8", "mmh_r1_u1_s4", "mmh_r1_u1_s8", "mmh_r2_u1_s4", "mmh_r2_u1_s8"},
            (4, 8): {"mmh_r1_u1_s4", "mmh_r1_u1_s8", "mmh_r2_u1_s4", "mmh_r2_u1_s8"},
            (9, 16): {"mmh_r2_u1_s4", "mmh_r2_u1_s8"}}
    other_rows_changed, sel_mismatch, outside, nonconst, linear = [], [], [], [], []
    changed = 0
    configs_2_16, configs_all = set(), set()
    for dt, t in mrl.TUNED.items():
        assert set(t) == set(prev[dt]), "shape set changed"
        for kn, row in t.items():
            shape = BY_KN[kn]
            assert set(row) == set(prev[dt][kn])
            for M, c in row.items():
                configs_all.add(c)
                if c == mrl.LINEAR:
                    linear.append((dt, shape, M))
                if M in (1, 24, 32):
                    if c != prev[dt][kn][M]:
                        other_rows_changed.append((dt, shape, M))
                    continue
                configs_2_16.add(c)
                changed += c != prev[dt][kn][M]
                if sel[dt][shape][str(M)] != c:
                    sel_mismatch.append((dt, shape, M))
                band = next(b for b in cand if b[0] <= M <= b[1])
                if c not in cand[band]:
                    outside.append((dt, shape, M, c))
            for band in cand:
                if len({row[M] for M in range(band[0], band[1] + 1)}) != 1:
                    nonconst.append((dt, shape, band))
    res["table"] = {"rows_1_24_32_changed": other_rows_changed, "rows_2_16_vs_selection_mismatch": sel_mismatch,
                    "outside_candidate_set": outside, "band_not_constant": nonconst, "linear_entries": linear,
                    "cells_2_16_changed": changed, "configs_at_2_16": sorted(configs_2_16),
                    "configs_whole_table": sorted(configs_all - {mrl.LINEAR}), "n_configs_whole_table": len(configs_all - {mrl.LINEAR})}

    # 4. Re-derive the band choice independently: pooled per-cell rank-free score using every
    #    candidate's median relative to the cell's mean of logs (reference-free), from the same sources.
    spec = importlib.util.spec_from_file_location("select_table", FIXES / "select_table.py")
    st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
    rederive, margins = [], {}
    for dt in ("float16", "bfloat16"):
        for shape in SHAPES:
            data = list(st.cells(dt, shape))
            for (lo, hi), cands in st.BANDS:
                use = [med for _, M, med in data if lo <= M <= hi and all(c in med for c in cands)]
                score = {c: statistics.fmean(math.log(m[c]) - statistics.fmean(math.log(m[x]) for x in cands) for m in use)
                         for c in cands}
                best = min(score, key=score.get)
                srt = sorted(score.values())
                margins[f"{dt}/{shape}/{lo}-{hi}"] = {"best": best, "cells": len(use),
                                                       "runner_up_margin_pct": round(100 * (math.exp(srt[1] - srt[0]) - 1), 2)}
                if best != sel[dt][shape][str(lo)]:
                    rederive.append((dt, shape, lo, hi, best, sel[dt][shape][str(lo)]))
    res["selection_rederived_mismatch"] = rederive
    res["selection_margins"] = margins

    # 5. Re-time data: every changed cell at 4-16 and every 2-3 cell re-timed with the final config,
    #    the run recorded the final table, compare arm = previous config, sample counts.
    previous = json.loads((FIXES / "compare-previous.json").read_text())
    retime = {}
    run_meta = {}
    for p in sorted(FIXES.glob("retime-*.json")):
        d = json.loads(p.read_text())
        live = {dt: {f"{K}x{N}": {str(m): c for m, c in row.items()} for (K, N), row in t.items()} for dt, t in mrl.TUNED.items()}
        rec = {dt: {kn: {str(m): c for m, c in row.items()} for kn, row in t.items()} for dt, t in d["tuned"].items()}
        run_meta[p.name] = {"torch": d["torch"], "reps": d["reps"], "seed": d["seed"], "recorded_table_is_final": rec == live,
                            "rows": len(d["rows"])}
        for r in d["rows"]:
            retime.setdefault((r["dtype"], r["shape"], r["M"]), []).append((p.name, r))
    res["retime_runs"] = run_meta
    problems = []
    n_retimed = 0
    for dt, t in mrl.TUNED.items():
        for kn, row in t.items():
            shape = BY_KN[kn]
            for M in range(2, 17):
                c = row[M]
                was = prev[dt][kn][M]
                need = (c != was) or M <= 3
                runs = retime.get((dt, shape, M), [])
                if need:
                    n_retimed += 1
                    want = 3 if M <= 3 else 2
                    if len(runs) != want:
                        problems.append(("process_count", dt, shape, M, len(runs)))
                    for f, r in runs:
                        if r["config"] != c:
                            problems.append(("config", dt, shape, M, f, r["config"], c))
                        exp = previous[dt][shape][str(M)]
                        if r.get("compare_config") != (exp if exp != "linear" else None) and exp != "linear":
                            problems.append(("compare", dt, shape, M, f, r.get("compare_config"), exp))
                        if exp == "linear" and r.get("compare_config") is None:
                            problems.append(("compare_missing_for_former_linear", dt, shape, M, f))
                        nreps = len(r["samples_us"]["multirow"])
                        if nreps != (15 if M <= 3 else 7):
                            problems.append(("reps", dt, shape, M, f, nreps))
    res["retime_cells_needed"] = n_retimed
    res["retime_problems"] = problems
    # compare-previous.json vs the 1c48161 table (except the formerly linear cells)
    cp_diff = []
    for dt, t in prev.items():
        for kn, row in t.items():
            for M in range(2, 17):
                if row[M] != previous[dt][BY_KN[kn]][str(M)]:
                    cp_diff.append((dt, BY_KN[kn], M, row[M], previous[dt][BY_KN[kn]][str(M)]))
    res["compare_previous_vs_1c48161"] = cp_diff

    # 6. Speedups and noise, independently from raw medians of the samples.
    cells = []
    for (dt, shape, M), runs in sorted(retime.items()):
        for f, r in runs:
            s = r["samples_us"]
            med = {k: statistics.median(v) for k, v in s.items()}
            sp = med["F.linear"] / med["multirow"]
            cells.append({"dtype": dt, "shape": shape, "M": M, "file": f, "config": r["config"],
                          "speedup": sp, "speedup_worst_pair": min(med["F.linear"], med["F.linear_again"]) / max(med["multirow"], med["multirow_again"]),
                          "aa": med["F.linear_again"] / med["F.linear"], "bb": med["multirow_again"] / med["multirow"],
                          "prev_over_new": (med["compare"] / med["multirow"]) if "compare" in med else None,
                          "prev_over_new_vs_again": (med["compare"] / med["multirow_again"]) if "compare" in med else None,
                          "compare_config": r.get("compare_config"),
                          "max_GBps": max(r["GBps"].values()) if isinstance(r["GBps"], dict) else None})
    res["retime_min_speedup"] = min(c["speedup"] for c in cells)
    res["retime_min_speedup_cell"] = min(cells, key=lambda c: c["speedup"])
    res["retime_cells_below_1"] = [c for c in cells if c["speedup"] < 1]
    res["retime_cells_worst_pair_below_1"] = [{k: c[k] for k in ("dtype", "shape", "M", "file", "speedup", "speedup_worst_pair", "aa", "bb")}
                                              for c in cells if c["speedup_worst_pair"] < 1]
    res["retime_max_GBps"] = max(c["max_GBps"] or 0 for c in cells)
    # previous config faster than new in every process, beyond 2%
    by = {}
    for c in cells:
        if c["prev_over_new"] is not None and c["compare_config"] != c["config"]:
            by.setdefault((c["dtype"], c["shape"], c["M"], c["config"], c["compare_config"]), []).append(c["prev_over_new"])
    res["previous_faster_every_process"] = [{"cell": k, "prev_over_new": v} for k, v in by.items() if max(v) < 1.0]
    res["new_faster_every_process_by_5pct"] = len([1 for v in by.values() if min(v) > 1.05])
    pn = [x for v in by.values() for x in v]
    res["prev_over_new_all"] = {"n": len(pn), "median": statistics.median(pn), "min": min(pn), "max": max(pn),
                                "geomean": math.exp(statistics.fmean(math.log(x) for x in pn))}
    same = [c["prev_over_new"] for c in cells if c["compare_config"] == c["config"]]
    res["identical_compare_arm_ratio"] = {"n": len(same), "min": min(same) if same else None, "max": max(same) if same else None}

    # 7. Unchanged cells at 4-16: judged on round-2 layer runs of the same config.
    old = {}
    for p in sorted(ROUND2.glob("layers-run*.json")):
        for r in json.loads(p.read_text())["rows"]:
            old.setdefault((r["dtype"], r["shape"], r["M"]), []).append((p.name, r))
    unchanged = []
    for dt, t in mrl.TUNED.items():
        for kn, row in t.items():
            for M in range(4, 17):
                if row[M] == prev[dt][kn][M]:
                    runs = [r for f, r in old.get((dt, BY_KN[kn], M), []) if r["config"] == row[M]]
                    unchanged.append({"dtype": dt, "shape": BY_KN[kn], "M": M, "config": row[M], "processes": len(runs),
                                      "speedup": [round(r["linear_over_multirow"], 3) for r in runs],
                                      "noise": round(max(max(abs(r["aa_ratio"] - 1), abs(r["bb_ratio"] - 1)) for r in runs), 3) if runs else None})
    res["unchanged_cells"] = {"n": len(unchanged), "min_processes": min(u["processes"] for u in unchanged),
                              "min_speedup": min(min(u["speedup"]) for u in unchanged),
                              "below_1": [u for u in unchanged if min(u["speedup"]) < 1],
                              "below_1_plus_noise": [u for u in unchanged if min(u["speedup"]) < 1 + u["noise"]]}

    # 8. The six formerly routed cells: every process.
    former = [(dt, BY_KN[kn], M) for dt, t in prev.items() for kn, row in t.items() for M, c in row.items() if c == "linear"]
    res["former_linear"] = {f"{dt}/{s}/{M}": [{"file": c["file"], "config": c["config"], "speedup": round(c["speedup"], 3),
                                               "aa": round(c["aa"], 3), "bb": round(c["bb"], 3)}
                                              for c in cells if (c["dtype"], c["shape"], c["M"]) == (dt, s, M)]
                            for dt, s, M in former}

    # 9. Headline: fp16 gate/up and down at 16 and 12 rows.
    res["headline"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()} for c in cells
                       if c["dtype"] == "float16" and c["shape"] in ("q17_up", "q17_down") and c["M"] in (12, 16)]

    out = HERE / "check-static.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    short = {k: v for k, v in res.items() if k not in ("selection_margins", "headline", "former_linear", "unchanged_cells")}
    print(json.dumps(short, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
