"""Reviewer's checks on experiment B's kernel (bench/multirow_linear.py), not the builder's.

1. offsets: x and W as aligned views with a nonzero storage offset (the wrapper accepts them).
2. tuned: pick_config on every tuned (K, N) at every M from 1 to 32, fp16 and bf16, with and
   without bias, through the builder's own criterion (check_case).
3. sensitivity: subtle planted bugs the builder did not try (fp16 accumulator, a dropped last
   K chunk, bf16 rounding by truncation, a missing bias on one path).
4. nonfinite: inf and NaN inputs, against F.linear.

Run through the GPU lock: bench/gpu_runner.py --monitor --seconds 200 --rss-gib 10 -- <python> <this file> <part> --out f.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from bench import multirow_linear as mrl  # noqa: E402
from bench import bench_multirow as bm  # noqa: E402


def emit(**kw):
    print(json.dumps(kw), flush=True)


def part_offsets(res):
    torch.manual_seed(1)
    out = []
    for dt in (torch.float16, torch.bfloat16):
        for M in (2, 8, 16):
            K, N = 2048, 1027
            bigx = torch.randn(M + 5, K).to(dt)
            bigw = (torch.randn(N + 24, K) / math.sqrt(K)).to(dt)
            x, w = bigx[3:3 + M], bigw[16:16 + N]   # offsets 3K and 16K elements: 16-byte aligned
            gx, gw = bigx.to("mps")[3:3 + M], bigw.to("mps")[16:16 + N]
            assert gx.storage_offset() > 0 and gw.storage_offset() > 0 and mrl.supports(gx, gw)
            for cfg in bm.all_variants():
                if not cfg.fits(M):
                    continue
                y = mrl.multirow_linear(gx, gw, None, config=cfg).cpu().double()
                ref = x.double() @ w.double().t()
                rel = ((y - ref).norm() / ref.norm()).item()
                ok = rel < 1e-2
                out.append({"dtype": str(dt), "M": M, "cfg": cfg.name(), "rel_l2": rel, "ok": ok})
            emit(event="offsets", dtype=str(dt), M=M, fails=sum(not o["ok"] for o in out if o["M"] == M and o["dtype"] == str(dt)))
    res["offsets"] = {"runs": len(out), "fails": [o for o in out if not o["ok"]][:20],
                      "n_fail": sum(not o["ok"] for o in out)}


def part_tuned(res, shapes=None, Ms=range(1, 33)):
    torch.manual_seed(2)
    cases, fails = [], 0
    for dname, table in mrl.TUNED.items():
        dt = bm.DTYPES[dname]
        for (K, N) in table:
            if shapes is not None and (K, N) not in shapes:
                continue
            w = (torch.randn(N, K) / math.sqrt(K)).to(dt)
            rows = None
            if N > 20000:
                g = torch.Generator().manual_seed(1)
                rows = torch.cat([torch.randperm(N - 1, generator=g)[:4095], torch.tensor([N - 1])])
            for M in Ms:
                bias = M % 2 == 1
                x = torch.randn(M, K).to(dt)
                b = torch.randn(N).to(dt) if bias else None
                cfg = mrl.pick_config(M, K, N, dt)
                out = bm.check_case(x, w, b, [cfg], ref_rows=rows)
                c = list(out["configs"].values())
                ok = out["pass"] and len(c) == 1
                fails += not ok
                cases.append({"dtype": dname, "K": K, "N": N, "M": M, "bias": bias, "cfg": cfg.name(), "pass": ok,
                              "ours": c[0]["max_norm_err"] if c else None, "lin": out["F.linear"]["max_norm_err"],
                              "allow": out["allowance"]})
            emit(event="tuned", dtype=dname, K=K, N=N, fails=fails)
    res.setdefault("tuned", []).append({"shapes": None if shapes is None else sorted(shapes), "Ms": list(Ms), "cases": len(cases), "fails": fails, "failing": [c for c in cases if not c["pass"]]})


MUTATIONS = [
    # (name, config, old, new)
    ("dev fp16 accumulator", mrl.Config("dev", 2, 1, 4),
     "float acc[R][M];\n    for (int r = 0; r < R; ++r) for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;\n\n    for (uint k = lane; k < K8; k += 32 * U) {",
     "half acc_h[R][M];\n    float acc[R][M];\n    for (int r = 0; r < R; ++r) for (int m = 0; m < M; ++m) { acc[r][m] = 0.0f; acc_h[r][m] = 0.0h; }\n\n    for (uint k = lane; k < K8; k += 32 * U) {"),
    ("dev drops the last 8-element K chunk", mrl.Config("dev", 2, 1, 4),
     "for (uint k = lane; k < K8; k += 32 * U)", "for (uint k = lane; k < K8 - 1; k += 32 * U)"),
    ("mmr drops the last 4 x elements of K", mrl.Config("mmr", 1, 1, 4),
     "xq[u][mt][j] = (xok[mt][j] && kx < ke)", "xq[u][mt][j] = (xok[mt][j] && kx + 4 < ke)"),
    ("bf16 output rounded by truncation", mrl.Config("dev", 2, 1, 4),
     "u += 0x7fffu + ((u >> 16) & 1u);", "u += 0u;"),
    ("mma split path drops the bias", mrl.Config("mma", 1, 1, 4, 4),
     "if (has_bias != 0) s += C::one(bias[i % uint(N)]);", "if (has_bias != 0) s += 0.0f;"),
]


def part_sensitivity(res):
    src = mrl.SOURCE_PATH.read_text()
    out = []
    for name, cfg, old, new in MUTATIONS:
        assert src.count(old) == 1, name
        s = src.replace(old, new)
        if "fp16 accumulator" in name:
            # accumulate in half, then copy to float before the store
            s = s.replace("for (int r = 0; r < R; ++r) acc[r][m] += dot(xl, wl[r]) + dot(xh, wh[r]);\n            }\n        }\n    }\n    store_rows<C, M, R>(acc, y, bias, row0, N, has_bias != 0, lane);\n}\n\ntemplate <typename C, int M, int R, int U>\nkernel void mrl_tg",
                          "for (int r = 0; r < R; ++r) acc_h[r][m] += half(dot(xl, wl[r]) + dot(xh, wh[r]));\n            }\n        }\n    }\n    for (int r = 0; r < R; ++r) for (int m = 0; m < M; ++m) acc[r][m] = float(acc_h[r][m]);\n    store_rows<C, M, R>(acc, y, bias, row0, N, has_bias != 0, lane);\n}\n\ntemplate <typename C, int M, int R, int U>\nkernel void mrl_tg", 1)
            assert "acc_h[r][m] +=" in s
        lib = mrl._Library(s)
        for dt in ((torch.bfloat16,) if "bf16" in name else (torch.float16, torch.bfloat16)):
            for K, N, M in [(2056, 1027, 8), (6144, 2048, 16), (264, 33, 4)]:
                torch.manual_seed(K + M)
                x = torch.randn(M, K).to(dt)
                w = (torch.randn(N, K) / math.sqrt(K)).to(dt)
                b = torch.randn(N).to(dt) if "bias" in name else None
                o = bm.check_case(x, w, b, [cfg], lib=lib)
                c = list(o["configs"].values())[0]
                out.append({"mutation": name, "dtype": str(dt), "K": K, "N": N, "M": M, "caught": not o["pass"],
                            "ours_norm": c["max_norm_err"], "lin_norm": o["F.linear"]["max_norm_err"],
                            "allow": o["allowance"], "ours_rel": c["rel_l2"], "lin_rel": o["F.linear"]["rel_l2"]})
                emit(event="mutation", **out[-1])
    res["sensitivity"] = out


def part_nonfinite(res):
    out = []
    for dt in (torch.float16, torch.bfloat16):
        for cfg in (mrl.Config("dev", 2, 1, 4), mrl.Config("tg", 1, 2, 8), mrl.Config("mma", 1, 1, 4, 4),
                    mrl.Config("mmr", 2, 1, 4, 3, 1)):
            x = torch.randn(8, 2056, dtype=dt, device="mps")
            w = torch.randn(1027, 2056, dtype=dt, device="mps") / 45
            x[1, 5] = float("inf"); x[2, 7] = float("nan")
            big = torch.full((8, 2056), 60000.0 if dt == torch.float16 else 3e38, dtype=dt, device="mps")
            for tag, xx, ww in (("inf_nan", x, w), ("overflow", big, torch.ones_like(w))):
                y = mrl.multirow_linear(xx, ww, None, config=cfg).cpu().float()
                l = F.linear(xx, ww).cpu().float()
                same = bool(((torch.isnan(y) == torch.isnan(l)) & (torch.isinf(y) == torch.isinf(l))).all())
                out.append({"dtype": str(dt), "cfg": cfg.name(), "case": tag, "nonfinite_pattern_matches_F.linear": same})
                emit(event="nonfinite", **out[-1])
    res["nonfinite"] = out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    res = {"torch": torch.__version__}
    for p in a.parts.split(","):
        {"offsets": part_offsets, "tuned": lambda r: part_tuned(r, shapes={k for k in mrl.TUNED["float16"] if k[1] < 20000}),
         "tuned_head": lambda r: part_tuned(r, shapes={(2048, 151936)}, Ms=(2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16)), "sensitivity": part_sensitivity, "nonfinite": part_nonfinite}[p](res)
        Path(a.out).write_text(json.dumps(res, indent=1))
