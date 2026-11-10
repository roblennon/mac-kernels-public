"""Summarise a decode rerun directory (rerun_all.sh output). Ratios are formed within each process,
where arms ran back to back, then summarised across processes: median, range, and a bootstrap 95%
interval over processes. A/A pairs give the noise floor; a gap smaller than the A/A spread is not
a finding. Writes summary.json and prints markdown tables."""
import argparse, glob, json, os, random, statistics as st
from collections import defaultdict

WARM_STEPS = 4  # the first steps of every timed decode are dropped, as in the earlier harness


def boot(xs, n=4000, seed=0):
    if len(xs) < 2:
        return (xs[0], xs[0]) if xs else (None, None)
    rnd = random.Random(seed); ms = sorted(st.median(rnd.choices(xs, k=len(xs))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def summ(xs):
    lo, hi = boot(xs)
    return {"median": st.median(xs), "min": min(xs), "max": max(xs), "ci95": [lo, hi], "n": len(xs)}


def fmt(s):
    return f"{s['median']:.3f} ({s['min']:.3f}–{s['max']:.3f}; CI {s['ci95'][0]:.3f}–{s['ci95'][1]:.3f}; n={s['n']})"


def decode_steps(d):
    out = defaultdict(lambda: defaultdict(list))  # (batch, P) -> name -> per-process ratios
    pairs = {"eager_dynamic/eager_static": ("eager_dynamic", "eager_static"),
             "eager_static/compiled": ("eager_static", "compiled_a"), "eager_dynamic/compiled": ("eager_dynamic", "compiled_a"),
             "compiled/kernel": ("compiled_a", "kernel_a"), "eager_static/kernel": ("eager_static", "kernel_a"),
             "eager_dynamic/kernel": ("eager_dynamic", "kernel_a"),
             "A/A compiled_a/compiled_b": ("compiled_a", "compiled_b"), "A/A kernel_a/kernel_b": ("kernel_a", "kernel_b")}
    ms = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(d, "step_*.json"))):
        r = json.load(open(f)); key = (r["batch"], r["prompt_len"])
        med = {arm: st.median([x for run in v["step_ms"] for x in run[WARM_STEPS:]]) for arm, v in r["arms"].items()}
        for arm, m in med.items():
            ms[key][arm].append(m)
        for name, (x, y) in pairs.items():
            if x in med and y in med:
                out[key][name].append(med[x] / med[y])
    return {f"B{b}_P{p}": {"ms": {a: summ(v) for a, v in ms[(b, p)].items()}, "ratios": {n: summ(v) for n, v in out[(b, p)].items()}}
            for (b, p) in sorted(out)}


def spec(d):
    res = {}
    for f in sorted(glob.glob(os.path.join(d, "spec_*.json"))):
        r = json.load(open(f)); t = res.setdefault(r["task"], {"ratios": defaultdict(list), "match": defaultdict(list),
                                                               "decode_s": defaultdict(list), "whole_s": defaultdict(list), "tokens": defaultdict(list)})
        arms = r["arms"]
        med = {a: {m: st.median(run[m] for run in v["runs"]) for m in ("decode_s", "whole_s")} for a, v in arms.items()}
        for a, v in med.items():
            t["decode_s"][a].append(v["decode_s"]); t["whole_s"][a].append(v["whole_s"])
            t["tokens"][a].append(arms[a]["runs"][0]["new_tokens"])
        base = arms.get("plain_a", {}).get("runs", [{}])[0].get("tokens")
        for a in arms:
            if base is not None:
                toks = arms[a]["runs"][0]["tokens"]
                first = next((i for i, (x, y) in enumerate(zip(base, toks)) if x != y), None)
                if first is None and len(base) != len(toks):
                    first = min(len(base), len(toks))
                t["match"][a].append(first)
            for m in ("decode_s", "whole_s"):
                if "plain_a" in med and a != "plain_a":
                    t["ratios"][f"{m} plain_a/{a}"].append(med["plain_a"][m] / med[a][m])
    return {k: {"ratios": {n: summ(v) for n, v in t["ratios"].items()}, "first_divergence_vs_plain": dict(t["match"]),
                "decode_s": {a: summ(v) for a, v in t["decode_s"].items()}, "whole_s": {a: summ(v) for a, v in t["whole_s"].items()},
                "new_tokens": dict(t["tokens"])} for k, t in res.items()}


def verify(d):
    out = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(d, "verify_*.json"))):
        r = json.load(open(f)); m = r["ms"]
        if "1" not in m:
            continue
        for T, v in m.items():
            if T.endswith("_end"):  # the first size measured again at the end: an A/A check
                out[r["arm"]][f"A/A T{T[:-4]} end/start"].append(v / m[T[:-4]])
            elif T != "1":
                out[r["arm"]][f"T{T}/T1"].append(v / m["1"])
    return {arm: {n: summ(v) for n, v in x.items()} for arm, x in out.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); a = ap.parse_args()
    s = {"decode_step": decode_steps(a.dir), "spec": spec(a.dir), "verify": verify(a.dir)}
    for f in ("rollback_*.json", "equiv_*.json"):
        for p in sorted(glob.glob(os.path.join(a.dir, f))):
            r = json.load(open(p)); r.pop("rounds", None)
            s.setdefault("checks", {})[os.path.basename(p)] = r
    json.dump(s, open(os.path.join(a.dir, "summary.json"), "w"), indent=1)
    print("## Decode step: ratio of times within a process (above 1 means the second arm is faster)\n")
    for k, v in s["decode_step"].items():
        print(f"### {k}\n"); [print(f"- {n}: {fmt(x)}") for n, x in v["ratios"].items()]; print()
    print("## Prompt lookup: plain_a time over arm time (above 1 means the arm is faster)\n")
    for k, v in s["spec"].items():
        print(f"### {k}\n"); [print(f"- {n}: {fmt(x)}") for n, x in v["ratios"].items()]
        print(f"- first divergence from plain_a (None = identical): {v['first_divergence_vs_plain']}\n")
    print("## Verify cost: T-token forward over one-token forward\n")
    for arm, v in s["verify"].items():
        print(f"### {arm}\n"); [print(f"- {n}: {fmt(x)}") for n, x in v.items()]; print()
    for name, r in s.get("checks", {}).items():
        print(f"- {name}: {r.get('verdict', {k: r[k].get('pass') for k in ('compiled', 'compiled_kernel') if k in r})}")


if __name__ == "__main__":
    main()
