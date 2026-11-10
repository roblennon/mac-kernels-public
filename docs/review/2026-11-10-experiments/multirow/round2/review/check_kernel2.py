"""Reviewer's correctness checks on round 2's kernel families (mmh, mmp, mmt), not the builder's.

Parts (one per process, each through the GPU lock):
  edges     every round-2 config (41: all in TUNED, the tune grid and the builder's variant list) on
            small and odd shapes: K of 8, 24, 40, 264, 2056 (K not a multiple of the 32-element
            block), N of 1, 5, 7, 13, 33, 1027, 517; x and W as 16-byte-aligned views with a
            nonzero storage offset; bias as a view at an odd element offset; M of 1, 2, 5, 8, 9,
            13, 16. Criterion: the builder's (max-normalised error within F.linear's + u + K*2^-24,
            relative L2 within 1.25x F.linear's + sqrt(K)*2^-24), computed here independently.
  special   products outside half range: fp16 inputs whose pairwise products overflow (|x w| >
            65504) or underflow (|x w| < 2^-24) in half precision while the sums stay finite and
            representable. Tells whether half-operand fragments (H = 1) multiply in fp32.
  nonfinite inf and NaN placed in x and W: the kernel's inf/NaN pattern must match F.linear's.
  tuned     pick_config at every tuned (K, N) for M = 1 to 32 (the head at 1 to 16, checked on its
            first and last 2048 rows), fp16 and bf16, with a bias, at the real shapes.
  routing   pick_config for every M from 1 to 32 on every tuned shape: which entry it takes.

python3 bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10 -- <python> <this file> <part> --dtype D --out f.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
from bench import multirow_linear as mrl  # noqa: E402

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}
UNIT = {torch.float16: 2.0 ** -11, torch.bfloat16: 2.0 ** -8}
R2 = ['mmh_r1_u1_s1', 'mmh_r1_u1_s4', 'mmh_r1_u1_s4_f1', 'mmh_r1_u1_s4_f1_h1', 'mmh_r1_u1_s4_h1', 'mmh_r1_u1_s8',
      'mmh_r1_u1_s8_f1', 'mmh_r1_u1_s8_f1_h1', 'mmh_r1_u1_s8_h1', 'mmh_r1_u1_s8_p2', 'mmh_r1_u1_s8_p2_h1',
      'mmh_r1_u2_s2_p3_f1', 'mmh_r2_u1_s2', 'mmh_r2_u1_s2_h1', 'mmh_r2_u1_s4', 'mmh_r2_u1_s4_f1', 'mmh_r2_u1_s4_f1_h1',
      'mmh_r2_u1_s4_h1', 'mmh_r2_u1_s4_p2', 'mmh_r2_u1_s4_p2_h1', 'mmh_r2_u1_s4_p3_h1', 'mmh_r2_u1_s8', 'mmh_r2_u1_s8_f1',
      'mmh_r2_u1_s8_f1_h1', 'mmh_r2_u1_s8_h1', 'mmh_r2_u2_s4', 'mmh_r2_u2_s4_h1', 'mmh_r2_u2_s8_f1_h1', 'mmh_r3_u1_s2_h1',
      'mmp_r1_u1_s8', 'mmp_r1_u1_s8_h1', 'mmp_r1_u2_s2_p3_f1', 'mmp_r2_u1_s1', 'mmp_r2_u1_s4', 'mmp_r2_u1_s4_h1',
      'mmt_r1_u2_s8', 'mmt_r1_u2_s8_h1', 'mmt_r2_u1_s4_h1', 'mmt_r2_u1_s4_p3', 'mmt_r2_u2_s4', 'mmt_r2_u2_s4_h1']
# extra K splits on small K: P large enough that the last split is short
R2 += ['mmh_r1_u1_s4_p7', 'mmp_r1_u1_s4_p5', 'mmt_r1_u1_s2_p5_h1']


def configs(dt):
    out = []
    for n in R2:
        c = mrl.Config.parse(n)
        if dt == torch.bfloat16 and c.H:
            continue  # H only changes the fp16 codec; identical code in bf16
        out.append(c)
    return out


def emit(**kw):
    print(json.dumps(kw), flush=True)


def views(x, w, b, off_x=8, off_w=16, off_b=3):
    """Device copies as views with nonzero storage offsets: x, W 16-byte aligned; bias odd."""
    M, K = x.shape
    N = w.shape[0]
    bx = torch.zeros(off_x + M * K, dtype=x.dtype, device="mps")
    bx[off_x:] = x.reshape(-1).to("mps")
    bw = torch.zeros(off_w + N * K, dtype=w.dtype, device="mps")
    bw[off_w:] = w.reshape(-1).to("mps")
    gx, gw = bx[off_x:].view(M, K), bw[off_w:].view(N, K)
    gb = None
    if b is not None:
        bb = torch.zeros(off_b + N + 5, dtype=b.dtype, device="mps")
        bb[off_b:off_b + N] = b.to("mps")
        gb = bb[off_b:off_b + N]
    return gx, gw, gb


def errs(got, ref, absref):
    got = got.cpu().double()
    mask = absref > 0
    n = ((got - ref).abs()[mask] / absref[mask]).max().item() if mask.any() else 0.0
    r = ((got - ref).norm() / ref.norm().clamp_min(1e-300)).item()
    return n, r


def judge(x, w, b, ys, rows=None):
    """ys: {name: device output}. Returns per-name verdicts with the builder's criterion."""
    K = x.shape[1]
    xd, wd = x.double(), w.double()
    if rows is not None:
        wd = wd[rows]
    ref = xd @ wd.t()
    absref = xd.abs() @ wd.abs().t()
    if b is not None:
        bd = b.double() if rows is None else b.double()[rows]
        ref, absref = ref + bd, absref + bd.abs()
    lin = ys.pop("F.linear")
    ln, lr = errs(lin, ref, absref)
    allow = UNIT[x.dtype] + K * 2.0 ** -24
    out = {}
    for k, y in ys.items():
        n, r = errs(y, ref, absref)
        fin = bool(torch.isfinite(y).all())
        out[k] = {"pass": fin and n <= ln + allow and r <= 1.25 * lr + math.sqrt(K) * 2.0 ** -24,
                  "norm_ratio": n / max(ln, 1e-30), "rel_ratio": r / max(lr, 1e-30), "finite": fin}
    return out


def part_edges(a, res):
    dt = DT[a.dtype]
    torch.manual_seed(5)
    shapes = [(8, 1), (8, 5), (24, 7), (40, 13), (264, 33), (2056, 1027), (520, 517)]
    Ms = [int(m) for m in a.rows.split(",")]
    cases, fails, runs = [], [], 0
    for M in Ms:
        for K, N in shapes:
            x = torch.randn(M, K).to(dt)
            w = (torch.randn(N, K) / math.sqrt(K)).to(dt)
            for bias in (False, True):
                b = (torch.randn(N) * 0.1).to(dt) if bias else None
                gx, gw, gb = views(x, w, b)
                assert gx.storage_offset() > 0 and gw.storage_offset() > 0 and mrl.supports(gx, gw, gb)
                ys = {"F.linear": F.linear(gx, gw, gb).cpu()}
                for c in configs(dt):
                    if not c.fits(M):
                        continue
                    ys[c.name()] = mrl.multirow_linear(gx, gw, gb, config=c).cpu()
                    runs += 1
                v = judge(x, w, b, ys)
                bad = [k for k, o in v.items() if not o["pass"]]
                worst = max(v.values(), key=lambda o: o["norm_ratio"])
                cases.append({"M": M, "K": K, "N": N, "bias": bias, "fails": bad,
                              "max_norm_ratio": worst["norm_ratio"],
                              "max_rel_ratio": max(o["rel_ratio"] for o in v.values())})
                fails += [{"M": M, "K": K, "N": N, "bias": bias, "cfg": k, **v[k]} for k in bad]
        emit(event="edges", dtype=a.dtype, M=M, runs=runs, fails=len(fails))
    res.update(cases=cases, fails=fails, runs=runs)


def part_special(a, res):
    """fp16 only. Product overflow: x = 256 on every k, W[n, k] = 256 * (-1)^k + n/64 on k = 0.
    Every product is +-65536 (overflows half), y[n] = 256 * n / 64 = 4n (exact in fp32).
    Product underflow: x = W = 2^-13, every product 2^-26 (below half's smallest subnormal),
    y = K * 2^-26."""
    dt = torch.float16
    out = {}
    for case in ("overflow", "underflow"):
        rows = []
        for M in (2, 9, 16):
            K, N = 264, 33
            if case == "overflow":
                x = torch.full((M, K), 256.0)
                w = torch.tensor([[256.0 * (-1) ** k for k in range(K)] for _ in range(N)])
                w[:, 0] += torch.arange(N) / 64.0
            else:
                x = torch.full((M, K), 2.0 ** -13)
                w = torch.full((N, K), 2.0 ** -13)
            x, w = x.to(dt), w.to(dt)
            ref = (x.double() @ w.double().t()).to(dt).double()
            gx, gw = x.to("mps"), w.to("mps")
            lin = F.linear(gx, gw).cpu().double()
            r = {"M": M, "expected_first_row": ref[0, :4].tolist(), "F.linear_max_abs_err": (lin - ref).abs().max().item(),
                 "configs": {}}
            for c in configs(dt):
                if not c.fits(M):
                    continue
                y = mrl.multirow_linear(gx, gw, config=c).cpu().double()
                r["configs"][c.name()] = {"max_abs_err": (y - ref).abs().nan_to_num(nan=1e30, posinf=1e30, neginf=1e30).max().item(),
                                          "finite": bool(torch.isfinite(y).all()), "sample": y[0, :4].tolist()}
            r["bad"] = sorted(k for k, v in r["configs"].items() if not v["finite"] or v["max_abs_err"] > max(1e-12, 2 * r["F.linear_max_abs_err"]))
            rows.append(r)
            emit(event="special", case=case, M=M, bad=r["bad"], linear_err=r["F.linear_max_abs_err"])
        out[case] = rows
    res.update(out)


def part_nonfinite(a, res):
    dt = DT[a.dtype]
    torch.manual_seed(6)
    rows = []
    for M in (3, 9, 16):
        K, N = 264, 33
        x = torch.randn(M, K).to(dt)
        w = (torch.randn(N, K) / 16).to(dt)
        x[1, 37] = float("inf")
        x[M - 1, 200] = float("nan")
        w[5, 100] = float("-inf")
        w[N - 1, 263] = float("nan")
        gx, gw = x.to("mps"), w.to("mps")
        lin = F.linear(gx, gw).cpu()
        pat = lambda y: (torch.isnan(y), torch.isposinf(y), torch.isneginf(y))
        lp = pat(lin)
        bad = []
        for c in configs(dt):
            if not c.fits(M):
                continue
            y = mrl.multirow_linear(gx, gw, config=c).cpu()
            yp = pat(y)
            if not all(torch.equal(p, q) for p, q in zip(lp, yp)):
                bad.append({"cfg": c.name(), "nan": int(yp[0].sum()), "lin_nan": int(lp[0].sum()),
                            "inf": int(yp[1].sum() + yp[2].sum()), "lin_inf": int(lp[1].sum() + lp[2].sum())})
        rows.append({"M": M, "bad": bad})
        emit(event="nonfinite", M=M, bad=len(bad))
    res["rows"] = rows


def part_tuned(a, res):
    dt = DT[a.dtype]
    torch.manual_seed(7)
    table = mrl.TUNED[a.dtype]
    cases, fails = [], []
    for (K, N) in table:
        head = N > 100000
        w = (torch.randn(N, K) / math.sqrt(K)).to(dt)
        b = (torch.randn(N) * 0.1).to(dt)
        gw, gb = w.to("mps"), b.to("mps")
        rows = torch.cat([torch.arange(2048), torch.arange(N - 2048, N)]) if head else None
        for M in (range(1, 17) if head else range(1, 33)):
            cfg = mrl.pick_config(M, K, N, dt)
            if cfg is None:
                cases.append({"K": K, "N": N, "M": M, "cfg": "linear"})
                continue
            x = torch.randn(M, K).to(dt)
            gx = x.to("mps")
            ys = {"F.linear": F.linear(gx, gw, gb).cpu(), cfg.name(): mrl.multirow_linear(gx, gw, gb).cpu()}
            if head:
                ys = {k: v[:, rows] for k, v in ys.items()}
            v = judge(x, w, b, ys, rows=rows)[cfg.name()]
            cases.append({"K": K, "N": N, "M": M, "cfg": cfg.name(), **v})
            if not v["pass"]:
                fails.append(cases[-1])
        emit(event="tuned", K=K, N=N, fails=len(fails))
        del w, gw
        torch.mps.empty_cache()
    res.update(cases=cases, fails=fails)


def part_routing(a, res):
    out = {}
    for dname, table in mrl.TUNED.items():
        dt = DT[dname]
        for (K, N), row in table.items():
            got = {}
            for M in range(1, 33):
                c = mrl.pick_config(M, K, N, dt)
                got[M] = "linear" if c is None else c.name()
            out[f"{dname}:{K}x{N}"] = got
    res["routing"] = out


def part_sensitivity(a, res):
    """Is this harness able to fail? Subtle mutations of the round-2 kernels, judged exactly as in
    `edges` (same shapes, offsets and criterion), and the share of runs bitwise equal to F.linear."""
    dt = DT[a.dtype]
    src = mrl.SOURCE_PATH.read_text()
    muts = {
        # mmt: the last tile's x zeroed (x check one tile short)
        "mmt_xcheck": ("const bool xok = k8 + 4 * u + xo < ke8;", "const bool xok = k8 + 4 * u + xo + 1 < ke8;"),
        # S > 1 reduction: bias added twice
        "mmh_bias_twice": ("v += red[(s2 * NR + nl) * MP + m];",
                           "v += red[(s2 * NR + nl) * MP + m] + ((hb && !split && s2 == 0) ? C::one(bias[n]) : 0.0f);"),
        # bf16 x: low/high halves of the x word swapped only for odd fragment rows
        "x2_bf16_swap": ("return float2(as_type<float>(hi ? (a & 0xffff0000u) : (a << 16)),",
                         "return float2(as_type<float>(hi ? (a << 16) : (a & 0xffff0000u)),"),
    }
    # mmh and mmp: the tail chunk skipped when it holds a single uint4 (the last 8 K elements)
    muts["mmh_drop_last8"] = ("    if (k8 < ke8) {  // at most one partial chunk per simdgroup",
                              "    if (k8 + 1 < ke8) {  // at most one partial chunk per simdgroup")
    targets = {"mmh_drop_last8": ["mmh_r2_u1_s4", "mmh_r1_u1_s8_h1"], "mmt_xcheck": ["mmt_r2_u1_s4_h1"],
               "mmh_bias_twice": ["mmh_r2_u1_s4"], "x2_bf16_swap": ["mmh_r2_u1_s4"]}
    torch.manual_seed(9)
    out = {}
    for name, (old, new) in muts.items():
        if name == "x2_bf16_swap" and dt != torch.bfloat16:
            continue
        assert src.count(old) == 1, name
        lib = mrl._Library(src.replace(old, new))
        caught, total = 0, 0
        for M in (2, 9, 16):
            for K, N in [(264, 33), (2056, 1027)]:
                x = torch.randn(M, K).to(dt)
                w = (torch.randn(N, K) / math.sqrt(K)).to(dt)
                b = (torch.randn(N) * 0.1).to(dt)
                gx, gw, gb = views(x, w, b)
                for cn in targets[name]:
                    c = mrl.Config.parse(cn)
                    ys = {"F.linear": F.linear(gx, gw, gb).cpu(), cn: mrl.multirow_linear(gx, gw, gb, config=c, lib=lib).cpu()}
                    v = judge(x, w, b, ys)[cn]
                    total += 1
                    caught += not v["pass"]
        out[name] = {"caught": caught, "cases": total}
        emit(event="sensitivity", mutation=name, caught=caught, cases=total)
    # bitwise equality share, unmutated
    eq = tot = 0
    for M in (2, 9, 16):
        x = torch.randn(M, 2056).to(dt).to("mps")
        w = (torch.randn(1027, 2056) / 45).to(dt).to("mps")
        lin = F.linear(x, w)
        for c in configs(dt):
            if c.fits(M):
                eq += bool(torch.equal(mrl.multirow_linear(x, w, config=c), lin))
                tot += 1
    out["bitwise_equal_share"] = eq / tot
    res["sensitivity"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("edges", "special", "nonfinite", "tuned", "routing", "sensitivity"))
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--rows", default="1,2,5,8,9,13,16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"torch": torch.__version__, "part": a.part, "dtype": a.dtype}
    t = time.time()
    {"edges": part_edges, "special": part_special, "nonfinite": part_nonfinite, "tuned": part_tuned,
     "routing": part_routing, "sensitivity": part_sensitivity}[a.part](a, res)
    res["seconds"] = time.time() - t
    Path(a.out).write_text(json.dumps(res, indent=1))
    emit(event="done", seconds=res["seconds"])


if __name__ == "__main__":
    main()
