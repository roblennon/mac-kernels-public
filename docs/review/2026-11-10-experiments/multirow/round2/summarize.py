"""Summarise round-2 layer runs: per cell, the range over processes of F.linear and kernel time,
GB/s, TFLOPS, the speedup and both noise arms; the F.linear fallback decision; model-order sums.

python3 summarize.py layers-run*.json > tables.md   (writes summary.json beside it)
Decision rule (Qwen3-1.7B shapes, rows 2 to 16): keep the kernel only if its slowest speedup over
the processes exceeds 1 + the largest noise-arm deviation (A/A or B/B) seen in that cell.
"""
import json
import sys
from collections import defaultdict

NAMES = {"q17_qo": "q/o 2048→2048", "q17_kv": "k/v 2048→1024", "q17_up": "gate/up 2048→6144",
         "q17_down": "down 6144→2048", "q17_head": "head 2048→151936", "q4b_up": "Qwen3-4B up 2560→9728",
         "llama7b_up": "Llama-7B up 4096→11008"}
PER_LAYER = {"q17_qo": 2, "q17_kv": 2, "q17_up": 2, "q17_down": 1}  # q,o; k,v; gate,up; down


def load(paths):
    cells = defaultdict(list)
    for p in paths:
        for r in json.load(open(p))["rows"]:
            cells[(r["dtype"], r["shape"], r["M"])].append(r)
    return cells


def rng(v, fmt="{:.1f}"):
    lo, hi = min(v), max(v)
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}–{fmt.format(hi)}"


def decide(rs):
    sp = [r["linear_over_multirow"] for r in rs]
    noise = max(max(abs(r["aa_ratio"] - 1), abs(r["bb_ratio"] - 1)) for r in rs)
    return min(sp) > 1 + noise, min(sp), noise


def main(paths):
    cells = load(paths)
    out = {"decisions": {}, "sums": {}}
    for dt in sorted({k[0] for k in cells}):
        print(f"\n### {dt}\n")
        print("| Shape | M | F.linear µs | kernel µs | fastest sample, F.linear / kernel µs | kernel GB/s | kernel TFLOPS | speedup | A/A | B/B | variant | routed |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for shape in NAMES:
            first = True
            for M in sorted({k[2] for k in cells if k[0] == dt and k[1] == shape}):
                rs = cells[(dt, shape, M)]
                med = lambda arm: [r["median_us"][arm] for r in rs]
                keep, lo, noise = decide(rs)
                routed = "kernel"
                if shape.startswith("q17") and 2 <= M <= 16:
                    out["decisions"].setdefault(dt, {}).setdefault(shape, {})[M] = {
                        "keep": keep, "min_speedup": lo, "noise": noise, "config": rs[0]["config"]}
                    routed = "kernel" if keep else "**F.linear**"
                elif M == 1:
                    routed = "F.linear (1 row)"
                fl = min(min(r["samples_us"]["F.linear"] + r["samples_us"]["F.linear_again"]) for r in rs)
                fk = min(min(r["samples_us"]["multirow"] + r["samples_us"]["multirow_again"]) for r in rs)
                print(f"| {NAMES[shape] if first else ''} | {M} | {rng(med('F.linear'))} | {rng(med('multirow'))} | {fl:.1f} / {fk:.1f} | "
                      f"{rng([r['GBps']['multirow'] for r in rs], '{:.0f}')} | {rng([r['TFLOPS']['multirow'] for r in rs], '{:.2f}')} | "
                      f"{rng([r['linear_over_multirow'] for r in rs], '{:.2f}x')} | {rng([r['aa_ratio'] for r in rs], '{:.2f}')} | "
                      f"{rng([r['bb_ratio'] for r in rs], '{:.2f}')} | `{rs[0]['config']}` | {routed} |")
                first = False
        # Model-order sums for Qwen3-1.7B: 28 layers x (2 qo + 2 kv + 2 up + 1 down) + head.
        for M in sorted({k[2] for k in cells if k[0] == dt and k[1] == "q17_up"}):
            try:
                per = {s: [r["median_us"] for r in cells[(dt, s, M)]] for s in list(PER_LAYER) + ["q17_head"]}
            except KeyError:
                continue
            if any(not v for v in per.values()):
                continue
            nproc = min(len(v) for v in per.values())
            for arm in ("F.linear", "multirow"):  # per process, in file order
                tot = []
                for i in range(nproc):
                    t = 28 * sum(n * per[s][i][arm] for s, n in PER_LAYER.items()) + per["q17_head"][i][arm]
                    tot.append(t / 1e3)
                out["sums"].setdefault(dt, {}).setdefault(str(M), {})[arm] = tot
        print(f"\nModel-order sum of the 197 linears, ms ({dt}; per process):\n")
        print("| M | F.linear | kernel (every cell on the kernel) |")  # range over the processes that ran every shape
        print("|---:|---:|---:|")
        for M, v in out["sums"].get(dt, {}).items():
            print(f"| {M} | {rng(v['F.linear'], '{:.2f}')} | {rng(v['multirow'], '{:.2f}')} |")
    with open("summary.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
