"""Choose the tuned config for every cell at 2 to 16 rows from a small fixed candidate set.

The round-2 table took each cell's sweep winner, but the winner beat the runner-up by under 1% in
half the cells (REVIEW.md, finding 3). Here one rule covers every shape and dtype:

  rows 2-3   dev_r2_u1_s4, dev_r2_u1_s8, mmh_r1_u1_s4, mmh_r1_u1_s8, mmh_r2_u1_s4, mmh_r2_u1_s8
  rows 4-8   mmh_r1_u1_s4, mmh_r1_u1_s8, mmh_r2_u1_s4, mmh_r2_u1_s8
  rows 9-16  mmh_r2_u1_s4, mmh_r2_u1_s8

U = 1, no prefetch, no K split, fp32 fragment operands. Within a band, one config per (dtype,
shape): the candidate with the lowest geometric mean, over every cell of the band, of its median
time divided by the median of a fixed reference candidate in the same cell. Pooling a band's
cells (5 to 8 cells of 3 interleaved repeats each) is what makes the choice robust; the ordering
does not depend on which candidate is the reference.

Data: every streamed interleaved timing that measured all of a band's candidates in one cell:
the round-2 sweep (../tune-*.json), this round's rerun of the sweep (tune-repro-*.json), and
the review's config scans (../review/configs-*.json, streamed arms only).

python3 select_table.py   (no GPU; writes selection.json next to this file)
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND2 = HERE.parent
SHAPES = {"q17_qo": (2048, 2048), "q17_kv": (2048, 1024), "q17_up": (2048, 6144), "q17_down": (6144, 2048),
          "q17_head": (2048, 151936), "q4b_up": (2560, 9728), "llama7b_up": (4096, 11008)}
BANDS = [((2, 3), ["dev_r2_u1_s4", "dev_r2_u1_s8", "mmh_r1_u1_s4", "mmh_r1_u1_s8", "mmh_r2_u1_s4", "mmh_r2_u1_s8"]),
         ((4, 8), ["mmh_r1_u1_s4", "mmh_r1_u1_s8", "mmh_r2_u1_s4", "mmh_r2_u1_s8"]),
         ((9, 16), ["mmh_r2_u1_s4", "mmh_r2_u1_s8"])]


def cells(dtype, shape):
    """Yield (source, M, {config: median us}) for every timed cell of this dtype and shape."""
    for path in [ROUND2 / f"tune-{dtype}-{shape}.json", HERE / f"tune-repro-{dtype}-{shape}.json"]:
        if path.exists():
            for r in json.loads(path.read_text())["rows"]:
                yield path.name, r["M"], r["median_us"]
    for path in sorted((ROUND2 / "review").glob(f"configs-{dtype}-{shape}-p*.json")):
        for r in json.loads(path.read_text())["rows"]:
            yield path.name, r["M"], {k: v for k, v in r["median_us"].items() if "|" not in k}


def main():
    out = {"rule": __doc__.split("\n\n")[1:3], "bands": {f"{a}-{b}": c for (a, b), c in BANDS}, "choice": {},
           "scores": {}}
    for dtype in ("float16", "bfloat16"):
        for shape in SHAPES:
            data = list(cells(dtype, shape))
            for (lo, hi), cands in BANDS:
                use = [(src, M, med) for src, M, med in data if lo <= M <= hi and all(c in med for c in cands)]
                ref = cands[0]
                score = {c: math.exp(sum(math.log(med[c] / med[ref]) for _, _, med in use) / len(use)) for c in cands}
                best = min(cands, key=score.get)
                key = f"{dtype}/{shape}/{lo}-{hi}"
                out["scores"][key] = {"cells": len(use), "sources": sorted({s for s, _, _ in use}),
                                      "relative_to_best": {c: round(score[c] / score[best], 4) for c in cands}}
                for M in range(lo, hi + 1):
                    out["choice"].setdefault(dtype, {}).setdefault(shape, {})[str(M)] = best
                print(f"{key:28s} {best:14s} cells={len(use):2d} "
                      + " ".join(f"{c}={score[c] / score[best]:.3f}" for c in cands))
    (HERE / "selection.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
