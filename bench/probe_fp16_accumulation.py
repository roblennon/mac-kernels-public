"""Probe: how does the MPS fp16 GEMM accumulate, and how accurate is it at K = 16384?

Two questions from the 2026-11-09 pre-grant review.

1. Accumulation width. Does `F.linear` / `matmul` in float16 on MPS accumulate the dot
   product in fp16, or in fp32 or wider, rounding to fp16 once at the end? Two kinds of
   evidence, both against an fp64 CPU oracle and explicit simulations:
   - structured dot products whose fp16 operands are exact and whose result differs between
     fp16 accumulation (any of several summation orders) and fp32 accumulation;
   - random data: the share of outputs equal to the correctly rounded fp64 result. fp32
     accumulation leaves almost every output correctly rounded when cancellation is mild;
     fp16 accumulation in sequential, lane or block order is off by many units in the last
     place at K in the thousands (pairwise fp16 order stays within about 1 unit on
     same-sign data, so it is separated by the structured cases, not by this one).
2. Error at K = 16384 for the weight-only kernel's design: rows of the activation and of the
   decoded INT8 weight are divided by exact powers of two (magnitudes at most 1), cast to
   fp16, multiplied by the stock fp16 GEMM, and the exponents are restored. The scaling is
   exact; the cast to fp16 and the 11-bit fp16 GEMM output are not. This measures the
   per-element error of each stage against fp64, with the normalization emulated in numpy
   (arm "emulated") and through the vendored comfy-kitchen kernel itself (arm "kernel").
   The emulated arm also isolates each inexact step on the CPU in fp64: the activation
   cast alone, the weight cast alone, and the 11-bit output rounding alone.

Only float16 and float32 tensors are created; bf16-valued activations are represented as
float16 or float64 values with 8 significant bits, never as a bf16 tensor.

    bench/with_gpu.sh .venv/bin/python bench/probe_fp16_accumulation.py --out result.json

GPU memory stays near 1 GiB (the driver-allocated figure is recorded in the JSON); the whole
run takes well under a minute on an M1 Max, mostly CPU.
"""
import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
F16_MAX = 65504.0


def chip():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return platform.processor()


def jf(v):
    """A JSON-safe float: non-finite values become strings."""
    v = float(v)
    return v if math.isfinite(v) else str(v)


# ---------------------------------------------------------------- simulations

def rnd(a, acc):
    """Round float64 values to the accumulator format (nearest even), back in float64."""
    with np.errstate(over="ignore", invalid="ignore"):
        return a.astype(np.float16 if acc == "fp16" else np.float32).astype(np.float64)


def seq_sum(p, acc, axis_len=None):
    """Left-to-right sum along the last axis, rounding after every addition."""
    s = np.zeros(p.shape[:-1])
    with np.errstate(invalid="ignore", over="ignore"):
        for k in range(p.shape[-1]):
            s = rnd(s + p[..., k], acc)
    return s


def simulate(p, acc, order):
    """Sum exact products p [..., K] in `acc` precision using summation order `order`."""
    k = p.shape[-1]
    with np.errstate(invalid="ignore", over="ignore"):
        if order == "seq":
            return seq_sum(p, acc)
        if order == "pairwise":
            q = rnd(p, acc)
            while q.shape[-1] > 1:
                if q.shape[-1] % 2:
                    q = np.concatenate([q, np.zeros(q.shape[:-1] + (1,))], axis=-1)
                q = rnd(q[..., 0::2] + q[..., 1::2], acc)
            return q[..., 0]
        kind, width = order.split(":")
        width = int(width)
        pad = (-k) % width
        if pad:
            p = np.concatenate([p, np.zeros(p.shape[:-1] + (pad,))], axis=-1)
        if kind == "lanes":  # `width` strided accumulators (SIMD style), then combined
            q = p.reshape(p.shape[:-1] + (-1, width))
            lanes = np.zeros(q.shape[:-2] + (width,))
            for i in range(q.shape[-2]):
                lanes = rnd(lanes + q[..., i, :], acc)
            return seq_sum(lanes, acc)
        if kind == "blocks":  # contiguous blocks of `width`, each summed, then combined
            q = p.reshape(p.shape[:-1] + (-1, width))
            blocks = np.zeros(q.shape[:-1])
            for i in range(width):
                blocks = rnd(blocks + q[..., i], acc)
            return seq_sum(blocks, acc)
    raise ValueError(order)


ORDERS = ("seq", "pairwise", "lanes:2", "lanes:4", "lanes:8", "lanes:16", "blocks:64", "blocks:1024")


def all_sims(x, w):
    """{name: fp16 result} for every simulated GEMM. x, w are float64 arrays of exact fp16 values,
    broadcastable to [..., K]. Products of two fp16 values are exact in fp32 and fp64."""
    p = x * w
    out = {}
    with np.errstate(over="ignore", invalid="ignore"):
        p16 = rnd(p, "fp16")
        out["fp16_products+fp16_acc/seq"] = simulate(p16, "fp16", "seq")
        out["fp16_products+fp16_acc/pairwise"] = simulate(p16, "fp16", "pairwise")
        for order in ORDERS:
            out[f"fp16_acc/{order}"] = simulate(p, "fp16", order)
        for order in ORDERS:
            out[f"fp32_acc_then_round/{order}"] = rnd(simulate(p, "fp32", order), "fp16")
    return out


def same(a, b):
    """Bitwise-style equality for fp16 results held in float64: nan equals nan."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return (a == b) | (np.isnan(a) & np.isnan(b))


# ---------------------------------------------------------------- MPS variants

def mps_gemm(x16, w16, dev):
    """{variant: float64 numpy [M, N]} for the fp16 GEMM call forms the kernel uses."""
    x = torch.from_numpy(x16).to(dev)
    w = torch.from_numpy(w16).to(dev)
    wt = w.t().contiguous()
    res = {
        "F.linear(x, w)": F.linear(x, w),
        "x @ wt (contiguous)": x @ wt,
        "x @ w.t() (lazy)": x @ w.t(),
    }
    out = {k: v.cpu().numpy().astype(np.float64) for k, v in res.items()}
    assert all(v.dtype == torch.float16 for v in res.values())
    del x, w, wt, res
    torch.mps.empty_cache()
    return out


def oracle64(x16, w16):
    return (torch.from_numpy(x16.astype(np.float64)) @ torch.from_numpy(w16.astype(np.float64)).t()).numpy()


# ---------------------------------------------------------------- part 1a: structured

def patterns(k):
    """Structured dot products of length k >= 4. Every operand is an exact fp16 value and every
    product is exact in fp32. Returns {name: (x [k], w [k], exact result, what it separates)}."""
    pats = {}
    # 1 + (k-1) * 2^-12: each small product is a quarter of fp16's unit at 1.
    x = np.full(k, 2.0 ** -6); w = np.full(k, 2.0 ** -6); x[0] = w[0] = 1.0
    pats["absorb_small_after_one"] = (x, w, 1 + (k - 1) * 2.0 ** -12,
                                      "fp16 sequential accumulation stays at 1.0")
    # +p first half, -p second half; the partial sum reaches (k/2) * p, above 65504 when k >= 16.
    e = min(7, max(0, math.ceil((17 - math.log2(max(k // 2, 1))) / 2)))
    x = np.full(k, 2.0 ** e); w = np.full(k, 2.0 ** e); w[k // 2:] *= -1
    exact = (k // 2 - (k - k // 2)) * 4.0 ** e
    pats["partial_sum_over_65504_then_cancel"] = (x, w, exact,
                                                  "fp16 accumulation in contiguous order gives inf or nan")
    # [+2048, +0.25, -2048, +0.25] repeated: the 0.25 is absorbed next to 2048 in fp16 (unit 2).
    x = np.ones(k); w = np.tile([2048.0, 0.25, -2048.0, 0.25], k // 4 + 1)[:k].copy()
    pats["alternating_cancellation"] = (x, w, float(w.sum()), "fp16 sequential accumulation loses half of the small terms")
    # A single product of 65536 overflows fp16 by itself.
    x = np.zeros(k); w = np.zeros(k)
    x[0], w[0], x[1], w[1], x[k - 1], w[k - 1] = 256.0, 256.0, 1.5, 2.0, 256.0, -256.0
    pats["single_product_over_65504"] = (x, w, 3.0, "products rounded to fp16 give inf - inf = nan")
    # (1 + 2^-10)^2 - (1 + 2^-9) = 2^-20: needs the full 22-bit product.
    x = np.zeros(k); w = np.zeros(k)
    x[0], w[0], x[1], w[1] = 1 + 2.0 ** -10, 1 + 2.0 ** -10, 1.0, -(1 + 2.0 ** -9)
    pats["product_low_bits"] = (x, w, 2.0 ** -20, "products rounded to fp16 give 0")
    # 2^14 + (k-2) * 2^-11 - 2^14: each small product is a quarter of fp32's unit at 2^14.
    x = np.full(k, 2.0 ** -5); w = np.full(k, 2.0 ** -6)
    x[0], w[0], x[k - 1], w[k - 1] = 128.0, 128.0, 128.0, -128.0
    pats["fp32_absorb_small_after_16384"] = (x, w, (k - 2) * 2.0 ** -11,
                                             "fp32 sequential accumulation gives 0; exact needs wider or reordered sums")
    # Large terms of alternating sign on every even index, small terms on every odd index. Exact in
    # fp32 only for an order that keeps the two kinds apart (two strided accumulators, for example).
    x = np.full(k, 2.0 ** -5); w = np.full(k, 2.0 ** -6)
    x[::2] = 128.0; w[::4] = 128.0; w[2::4] = -128.0
    exact = float((x * w).sum())
    pats["fp32_interleaved_large_cancel"] = (x, w, exact,
                                             "every other product is +-16384; fp32 accumulation in most orders loses the small terms")
    return pats


def run_structured(shapes, dev):
    rows = []
    sim_cache = {}
    for m, k, n in shapes:
        for name, (xp, wp, exact, note) in patterns(k).items():
            if (name, k) not in sim_cache:
                sim_cache[(name, k)] = {s: float(v) for s, v in all_sims(xp, wp).items()}
            sims = sim_cache[(name, k)]
            x16 = np.tile(xp.astype(np.float16), (m, 1))
            w16 = np.tile(wp.astype(np.float16), (n, 1))
            assert np.array_equal(x16[0].astype(np.float64), xp) and np.array_equal(w16[0].astype(np.float64), wp)
            with np.errstate(over="ignore"):
                correct = float(np.float64(exact).astype(np.float16))
            got = mps_gemm(x16, w16, dev)
            fp16_vals = {v for s, v in sims.items() if not s.startswith("fp32")}
            fp32_vals = {v for s, v in sims.items() if s.startswith("fp32")}
            row = {
                "shape_MKN": [m, k, n], "pattern": name, "separates": note,
                "exact": jf(exact), "exact_rounded_to_fp16": jf(correct),
                "simulated": {s: jf(v) for s, v in sims.items()},
                "fp16_sims_all_wrong": all(not same(v, correct) for v in fp16_vals),
                "fp32_sims_all_correct": all(bool(same(v, correct)) for v in fp32_vals),
                "mps": {},
            }
            for variant, out in got.items():
                uniq, counts = np.unique(out, return_counts=True, equal_nan=True)
                row["mps"][variant] = {
                    "unique_values": [jf(u) for u in uniq[:8]],
                    "counts": [int(c) for c in counts[:8]],
                    "all_equal_exact_rounded": bool(same(out, correct).all()),
                    "matching_sims": [s for s, v in sims.items() if bool(same(out, v).all())],
                }
            rows.append(row)
            print(f"structured {m}x{k}x{n} {name}: exact16={correct} "
                  + " ".join(f"[{v['unique_values'][:3]}]" for v in row["mps"].values()), flush=True)
    return rows


# ---------------------------------------------------------------- part 1b: random data

def ulp16(r):
    a = np.maximum(np.abs(r), 2.0 ** -14)
    return 2.0 ** (np.floor(np.log2(a)) - 10)


def pct(a, q):
    return jf(np.percentile(a, q, method="higher")) if a.size else None


def round_stats(out, ref, sum_abs=None):
    """How close fp16 outputs `out` are to the fp64 reference, in fp16 units in the last place.
    With sum_abs = sum_k |x_k w_k|, also the error in units of that sum: an accumulator with
    p significant bits errs by up to about K * 2^-p of it, whatever the output magnitude."""
    with np.errstate(over="ignore", invalid="ignore"):
        correct = ref.astype(np.float16).astype(np.float64)
        err = np.abs(out - ref) / ulp16(ref)
    err = np.where(np.isfinite(err), err, np.inf)
    extra = {}
    if sum_abs is not None:
        with np.errstate(invalid="ignore"):
            r = np.abs(out - ref) / sum_abs
            ideal = np.abs(correct - ref) / sum_abs
        r = np.where(np.isfinite(r), r, np.inf)
        extra = {"max_err_over_sum_abs_products": jf(r.max()), "p99_9_err_over_sum_abs_products": pct(r, 99.9),
                 "median_err_over_sum_abs_products": pct(r, 50),
                 "max_err_over_sum_abs_products_if_correctly_rounded": jf(ideal.max())}
    return {
        **extra,
        "share_equal_to_rounded_fp64": jf(same(out, correct).mean()),
        "share_within_1_ulp_of_rounded_fp64": jf((np.abs(out - correct) <= ulp16(ref)).mean()),
        "max_err_ulp": jf(err.max()), "p99_9_err_ulp": pct(err, 99.9), "median_err_ulp": pct(err, 50),
        "nonfinite_outputs": int((~np.isfinite(out)).sum()),
    }


def random_inputs(dist, m, k, n, rng):
    if dist == "gauss":  # signed, heavy cancellation, outputs of order 1
        x = rng.standard_normal((m, k)); w = rng.standard_normal((n, k)) / math.sqrt(k)
    elif dist == "positive_uniform":  # no cancellation: the sum grows to about 0.56 K
        x = rng.uniform(0.5, 1.0, (m, k)); w = rng.uniform(0.5, 1.0, (n, k))
    elif dist == "normalized_signed":  # the kernel's domain: both operands in (-1, 1)
        x = rng.uniform(-1.0, 1.0, (m, k)); w = rng.integers(-127, 128, (n, k)) / 128.0
    elif dist == "cancel_random_large":
        # 3K/8 pairs of products +a*b and -a*b (magnitude up to 1024) at random positions, plus K/4
        # small products. The exact result is the sum of the small products (order 10), while the
        # partial sums of any generic order reach about 2^16: fp32 accumulation then usually errs
        # by several fp16 units (on small subsets an fp32 order can still land on the rounded
        # exact value); a wider accumulator returns the rounded exact value everywhere.
        h = 3 * k // 8
        perm = rng.permutation(k)
        a = (rng.uniform(-32.0, 32.0, (m, h))).astype(np.float16).astype(np.float64)
        b = (rng.uniform(-32.0, 32.0, (n, h))).astype(np.float16).astype(np.float64)
        x = rng.uniform(-1.0, 1.0, (m, k)); w = rng.uniform(-1.0, 1.0, (n, k))
        x[:, perm[:h]] = a; x[:, perm[h:2 * h]] = a
        w[:, perm[:h]] = b; w[:, perm[h:2 * h]] = -b
    else:
        raise ValueError(dist)
    return x.astype(np.float16), w.astype(np.float16)


def run_random(shapes, dev, seed, sim_side):
    rows = []
    for m, k, n in shapes:
        for dist in ("gauss", "positive_uniform", "normalized_signed", "cancel_random_large"):
            rng = np.random.default_rng([seed, m, k, n])
            x16, w16 = random_inputs(dist, m, k, n, rng)
            ref = oracle64(x16, w16)
            sum_abs = oracle64(np.abs(x16), np.abs(w16))
            got = mps_gemm(x16, w16, dev)
            rs, cs = min(m, sim_side), min(n, sim_side)
            sims = all_sims(x16[:rs, None, :].astype(np.float64), w16[None, :cs, :].astype(np.float64))
            row = {
                "shape_MKN": [m, k, n], "distribution": dist, "outputs": m * n,
                "ref_abs_max": jf(np.abs(ref).max()), "ref_abs_median": jf(np.median(np.abs(ref))),
                "sum_abs_products_median": jf(np.median(sum_abs)),
                "mps": {v: round_stats(o, ref, sum_abs) for v, o in got.items()},
                "mps_on_sim_subset": {v: round_stats(o[:rs, :cs], ref[:rs, :cs], sum_abs[:rs, :cs]) for v, o in got.items()},
                "mps_variants_bit_identical": bool(all(same(o, got["F.linear(x, w)"]).all() for o in got.values())),
                "sim_subset_outputs": rs * cs,
                "simulated": {},
            }
            sub = got["F.linear(x, w)"][:rs, :cs]
            for s, v in sims.items():
                st = round_stats(v, ref[:rs, :cs], sum_abs[:rs, :cs])
                st["share_of_mps_outputs_equal_to_this_sim"] = jf(same(sub, v).mean())
                row["simulated"][s] = st
            rows.append(row)
            a = row["mps"]["F.linear(x, w)"]
            print(f"random {m}x{k}x{n} {dist}: mps err/sumabs max {a['max_err_over_sum_abs_products']} exact-rounded {a['share_equal_to_rounded_fp64']:.6f} "
                  f"max {a['max_err_ulp']} ulp | fp16 seq sim "
                  f"{row['simulated']['fp16_acc/seq']['share_equal_to_rounded_fp64']:.4f} "
                  f"max {row['simulated']['fp16_acc/seq']['max_err_ulp']} ulp | fp32 seq sim "
                  f"{row['simulated']['fp32_acc_then_round/seq']['share_equal_to_rounded_fp64']:.4f} "
                  f"max {row['simulated']['fp32_acc_then_round/seq']['max_err_ulp']} ulp", flush=True)
    return rows


# ---------------------------------------------------------------- part 2: K = 16384 error

def err_stats(got, ref, extra_scale=None, bulk_rms=None):
    """Per-element error of `got` against the fp64 `ref`. `bulk_rms` is the RMS of the output
    with each activation row's largest element removed: the signal a dominant outlier can mask."""
    err = np.abs(got - ref)
    nz = ref != 0
    rel = err[nz] / np.abs(ref[nz])
    rms = math.sqrt(float((ref ** 2).mean()))
    out = {
        "rel_max": jf(rel.max()), "rel_p99_9": pct(rel, 99.9), "rel_p99": pct(rel, 99), "rel_median": pct(rel, 50),
        "err_over_output_rms_max": jf(err.max() / rms), "err_over_output_rms_p99_9": jf(np.percentile(err, 99.9) / rms),
        "rel_l2": jf(np.linalg.norm(got - ref) / np.linalg.norm(ref)),
        "nonfinite": int((~np.isfinite(got)).sum()),
    }
    if bulk_rms:
        out["err_over_bulk_output_rms_max"] = jf(err.max() / bulk_rms)
        out["err_over_bulk_output_rms_p99_9"] = jf(np.percentile(err, 99.9) / bulk_rms)
        out["err_rms_over_bulk_output_rms"] = jf(math.sqrt(float((err ** 2).mean())) / bulk_rms)
    if extra_scale is not None:  # error relative to sum_k |x_k w_k|, the scale of accumulation bounds
        r = err / extra_scale
        out["err_over_sum_abs_products_max"] = jf(r.max())
        out["err_over_sum_abs_products_p99_9"] = pct(r, 99.9)
    return out


def bf16_round(a):
    """Round float64 values to 8 significant bits (bf16's significand); exact powers of two kept."""
    mant, exp = np.frexp(a)
    return np.ldexp(np.rint(mant * 256.0) / 256.0, exp)


def activations(dist, m, k, rng):
    """(true activation in float64, description). All values fit the stated source dtype exactly."""
    g = rng.standard_normal((m, k))
    if dist == "gauss_fp16_values":
        return g.astype(np.float16).astype(np.float64), "N(0,1), fp16 values (11-bit significands)"
    if dist == "gauss_bf16_values":
        return bf16_round(g), "N(0,1), 8-bit significands as a bf16 activation would carry"
    if dist == "laplace_bf16_values":
        return bf16_round(rng.laplace(size=(m, k))), "Laplace(0,1), 8-bit significands"
    if dist == "outlier_16000_bf16_values":
        x = bf16_round(g)
        x[np.arange(m), rng.integers(0, k, m)] = 16000.0
        return bf16_round(x), "N(0,1) plus one value of 16000 per row (the H3 text-conditioning magnitude), 8-bit significands"
    if dist == "outlier_2p20_bf16_values":
        x = bf16_round(g)
        x[np.arange(m), rng.integers(0, k, m)] = 2.0 ** 20
        return x, "N(0,1) plus one value of 2^20 per row, 8-bit significands: the bulk sits below rowmax * 2^-14"
    raise ValueError(dist)


def bulk_output_rms(x_true, w_true):
    """RMS of x @ w^T with the largest-magnitude element of every activation row set to zero."""
    xb = x_true.copy()
    xb[np.arange(xb.shape[0]), np.abs(xb).argmax(axis=1)] = 0.0
    yb = (torch.from_numpy(xb) @ torch.from_numpy(w_true).t()).numpy()
    return math.sqrt(float((yb ** 2).mean()))


def int8_weight(n, k, rng):
    """INT8 weight with a per-row fp32 scale, as the checkpoints store it."""
    w = (rng.standard_normal((n, k)) * 0.02).astype(np.float32)
    scale = np.maximum(np.abs(w).max(axis=1, keepdims=True) / np.float32(127.0), np.float32(1e-30)).astype(np.float32)
    wq = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return wq, scale


def run_bound_emulated(m, k, n, dev, seed):
    """The kernel's steps 1 to 3 in numpy, with the MPS fp16 GEMM in the middle."""
    rows = []
    rng = np.random.default_rng([seed, 2, k])
    wq, scale = int8_weight(n, k, rng)
    # Decode: f = ceil_exp(scale) + 7, mant = scale * 2^-f, value = half(float(int8) * mant).
    _, e = np.frexp(scale)
    f = (e + 7).astype(np.int64)
    mant = np.ldexp(scale, -f).astype(np.float32)
    w16 = (wq.astype(np.float32) * mant).astype(np.float16)
    w_true = wq.astype(np.float64) * scale.astype(np.float64)
    w_cast_changed = float((w16.astype(np.float64) != wq.astype(np.float64) * mant.astype(np.float64)).mean())
    for dist in ("gauss_fp16_values", "gauss_bf16_values", "laplace_bf16_values",
                 "outlier_16000_bf16_values", "outlier_2p20_bf16_values"):
        x_true, desc = activations(dist, m, k, rng)
        # Prologue: E smallest with |x| < 2^E for the whole row; x' = x * 2^-E, then cast to fp16.
        _, row_exp = np.frexp(np.abs(x_true).max(axis=1, keepdims=True))
        xn = np.ldexp(x_true, -row_exp)
        x16 = xn.astype(np.float16)
        cast_changed = x16.astype(np.float64) != xn
        cast_zeroed = (x16 == 0) & (xn != 0)
        below_2m14 = np.abs(xn) < 2.0 ** -14
        y_true = torch.from_numpy(x_true) @ torch.from_numpy(w_true).t()
        y_true = y_true.numpy()
        bulk = bulk_output_rms(x_true, w_true)
        restore = np.ldexp(1.0, row_exp + f.reshape(1, -1))  # 2^(row_exp + ch_exp), exact
        c_ref = oracle64(x16, w16)  # exact product of the fp16 operands the GEMM actually sees
        # Each inexact step alone, in fp64 on the CPU. wn is the exactly scaled true weight.
        wn = np.ldexp(w_true, -f)
        mm = lambda a, b: (torch.from_numpy(np.asarray(a, dtype=np.float64)) @ torch.from_numpy(np.asarray(b, dtype=np.float64)).t()).numpy()
        act_only = mm(x16, wn)   # fp16 activation, exact weight, unrounded output
        wgt_only = mm(xn, w16)   # exact activation, fp16 weight, unrounded output
        c_exact = mm(xn, wn)     # exact operands: only the output is rounded below
        with np.errstate(over="ignore"):
            out_only = c_exact.astype(np.float16).astype(np.float64)
        sum_abs = (torch.from_numpy(np.abs(x16).astype(np.float64)) @ torch.from_numpy(np.abs(w16).astype(np.float64)).t()).numpy()
        with np.errstate(over="ignore"):
            ideal = c_ref.astype(np.float16).astype(np.float64)
        got = mps_gemm(x16, w16, dev)
        row = {
            "arm": "emulated", "shape_MKN": [m, k, n], "activation": dist, "activation_desc": desc,
            "row_exp_min_max": [int(row_exp.min()), int(row_exp.max())],
            "activation_cast_to_fp16": {
                "share_of_elements_changed": jf(cast_changed.mean()),
                "share_below_2^-14_after_scaling": jf(below_2m14.mean()),
                "share_flushed_to_zero": jf(cast_zeroed.mean()),
            },
            "weight_cast_to_fp16_share_changed": w_cast_changed,
            "bulk_output_rms_over_output_rms": jf(bulk / math.sqrt(float((y_true ** 2).mean()))),
            "gemm_output_abs_max": jf(np.abs(c_ref).max()), "gemm_output_abs_median": jf(np.median(np.abs(c_ref))),
            "share_of_gemm_outputs_below_fp16_normal_min": jf((np.abs(c_ref) < 2.0 ** -14).mean()),
            "stages": {
                "casts_only (exact GEMM of the fp16 operands, unrounded) vs truth": err_stats(c_ref * restore, y_true, bulk_rms=bulk),
                "casts + ideal 11-bit output (correctly rounded fp16 GEMM) vs truth": err_stats(ideal * restore, y_true, bulk_rms=bulk),
                "ideal 11-bit output vs exact GEMM of the fp16 operands": err_stats(ideal, c_ref, sum_abs),
            },
            "isolated_steps_vs_truth": {
                "scaling_only (exact operands, unrounded output; checks that the scaling is exact)": err_stats(c_exact * restore, y_true, bulk_rms=bulk),
                "activation_cast_only": err_stats(act_only * restore, y_true, bulk_rms=bulk),
                "weight_cast_only": err_stats(wgt_only * restore, y_true, bulk_rms=bulk),
                "output_rounding_only (exact operands, correctly rounded fp16 output)": err_stats(out_only * restore, y_true, bulk_rms=bulk),
            },
            "mps": {},
        }
        for variant, out in got.items():
            row["mps"][variant] = {
                "gemm_only (vs exact GEMM of the same fp16 operands)": err_stats(out, c_ref, sum_abs),
                "gemm_rounding": round_stats(out, c_ref),
                "end_to_end (restored, vs fp64 truth from unrounded inputs)": err_stats(out * restore, y_true, bulk_rms=bulk),
                "mps_minus_correctly_rounded_over_sum_abs_products_max": jf((np.abs(out - ideal) / sum_abs).max()),
            }
        rows.append(row)
        g = row["mps"]["x @ wt (contiguous)"]
        print(f"bound emulated {dist}: gemm-only rel max {g['gemm_only (vs exact GEMM of the same fp16 operands)']['rel_max']:.3e} "
              f"p99.9 {g['gemm_only (vs exact GEMM of the same fp16 operands)']['rel_p99_9']:.3e} | end-to-end rel max "
              f"{g['end_to_end (restored, vs fp64 truth from unrounded inputs)']['rel_max']:.3e} p99.9 "
              f"{g['end_to_end (restored, vs fp64 truth from unrounded inputs)']['rel_p99_9']:.3e}", flush=True)
    return rows


def run_bound_kernel(m, k, n, dev, seed, group):
    """The vendored comfy-kitchen kernel itself: fp16 activation in, fp32 and fp16 out."""
    import importlib.util
    pkg = REPO / "vendor" / "comfy-kitchen" / "comfy_kitchen"
    try:
        # Load this repository's vendored copy by path: an editable install of another checkout
        # may otherwise win the import.
        spec = importlib.util.spec_from_file_location("comfy_kitchen", pkg / "__init__.py",
                                                      submodule_search_locations=[str(pkg)])
        comfy_kitchen = importlib.util.module_from_spec(spec)
        sys.modules["comfy_kitchen"] = comfy_kitchen
        spec.loader.exec_module(comfy_kitchen)
        from comfy_kitchen.backends.mps.linear import int8_linear
        from comfy_kitchen.tensor.int8_utils import _build_hadamard
    except Exception as exc:  # recorded, not hidden
        return [{"arm": "kernel", "error": repr(exc)[:300]}], None
    import hashlib
    import subprocess
    ck = pkg.parent
    source = {"path": "vendor/comfy-kitchen/comfy_kitchen",
              "resolves_inside_this_repository": pkg.resolve().is_relative_to(REPO),
              "linear_py_sha256": hashlib.sha256((pkg / "backends" / "mps" / "linear.py").read_bytes()).hexdigest()}
    try:
        source["git_commit"] = subprocess.run(["git", "-C", str(ck), "rev-parse", "HEAD"], capture_output=True,
                                              text=True, timeout=20).stdout.strip() or None
        source["git_dirty_files"] = len(subprocess.run(["git", "-C", str(ck), "status", "--porcelain"],
                                                       capture_output=True, text=True, timeout=20).stdout.splitlines())
    except Exception as exc:
        source["git_error"] = repr(exc)[:200]
    rows = []
    rng = np.random.default_rng([seed, 3, k])
    wq, scale = int8_weight(n, k, rng)
    w_stored = wq.astype(np.float64) * scale.astype(np.float64)
    h = _build_hadamard(group).double().numpy()
    wqm, wsm = torch.from_numpy(wq).to(dev), torch.from_numpy(scale).to(dev)
    for convrot in (False, True):
        # With ConvRot the stored weight is in the rotated basis: W = stored @ H per group.
        w_true = (w_stored.reshape(n, k // group, group) @ h).reshape(n, k) if convrot else w_stored
        for dist in ("gauss_fp16_values", "outlier_16000_bf16_values"):
            x_true, desc = activations(dist, m, k, rng)
            x16 = x_true.astype(np.float16)  # the kernel arm feeds an fp16 tensor: the truth is that tensor
            x_true = x16.astype(np.float64)
            desc += "; rounded to fp16 for the fp16 input tensor"
            y_true = (torch.from_numpy(x_true) @ torch.from_numpy(w_true).t()).numpy()
            bulk = bulk_output_rms(x_true, w_true)
            xm = torch.from_numpy(x16).to(dev)
            row = {"arm": "kernel", "shape_MKN": [m, k, n], "convrot_group": group if convrot else None,
                   "activation": dist, "activation_desc": desc, "compute_dtype": "float16",
                   "bulk_output_rms_over_output_rms": jf(bulk / math.sqrt(float((y_true ** 2).mean()))), "out": {}}
            for out_dtype in (torch.float32, torch.float16):
                try:
                    y = int8_linear(xm, wqm, wsm, out_dtype=out_dtype, convrot=convrot, convrot_groupsize=group)
                    assert y.dtype == out_dtype
                    row["out"][str(out_dtype).replace("torch.", "")] = err_stats(y.cpu().numpy().astype(np.float64), y_true, bulk_rms=bulk)
                except Exception as exc:
                    row["out"][str(out_dtype).replace("torch.", "")] = {"error": repr(exc)[:300]}
            rows.append(row)
            o = row["out"]["float32"]
            print(f"bound kernel convrot={convrot} {dist}: " + (f"rel max {o['rel_max']:.3e} p99.9 {o['rel_p99_9']:.3e} "
                  f"rel_l2 {o['rel_l2']:.3e}" if "error" not in o else o["error"]), flush=True)
    return rows, source


# ---------------------------------------------------------------- summary

def summarize(result):
    """Counts and extremes computed from the rows above, so prose can quote them from this file."""
    s = {}
    sep = [r for r in result["structured"] if r["fp16_sims_all_wrong"] and r["fp32_sims_all_correct"]]
    cells = [v for r in sep for v in r["mps"].values()]
    s["structured_cases_where_every_fp16_sim_is_wrong_and_every_fp32_sim_is_right"] = {
        "shape_pattern_pairs": len(sep), "mps_results": len(cells),
        "mps_results_equal_to_exact_rounded": sum(c["all_equal_exact_rounded"] for c in cells),
        "mps_results_matching_any_fp16_sim": sum(any(not m.startswith("fp32") for m in c["matching_sims"]) for c in cells),
    }
    by_pat = {}
    for r in result["structured"]:
        if r["pattern"].startswith("fp32_"):
            continue
        d = by_pat.setdefault(r["pattern"], {"shapes": 0, "separating_shapes": 0, "separating_shapes_MKN": [],
                                             "mps_results_equal_to_exact_rounded": 0, "mps_results": 0,
                                             "mps_results_also_matching_an_fp16_sim": 0})
        d["shapes"] += 1
        if r["fp16_sims_all_wrong"] and r["fp32_sims_all_correct"]:
            d["separating_shapes"] += 1
            d["separating_shapes_MKN"].append(r["shape_MKN"])
        for c in r["mps"].values():
            d["mps_results"] += 1
            d["mps_results_equal_to_exact_rounded"] += c["all_equal_exact_rounded"]
            d["mps_results_also_matching_an_fp16_sim"] += any(not m.startswith("fp32") for m in c["matching_sims"])
    s["structured_fp16_vs_fp32_patterns_by_pattern"] = by_pat
    hard = [r for r in result["structured"] if r["pattern"].startswith("fp32_")]
    cells = [v for r in hard for v in r["mps"].values()]
    def fam(c):
        f32 = sorted({m.split("/")[1] for m in c["matching_sims"] if m.startswith("fp32")})
        return f32
    s["structured_fp32_defeating_matches"] = {
        "mps_results": len(cells),
        "matching_some_fp32_sim_order": sum(bool(fam(c)) for c in cells),
        "matching_fp32_seq": sum("seq" in fam(c) for c in cells),
        "matching_an_fp32_order_other_than_seq_only": sum(bool(set(fam(c)) - {"seq"}) and "seq" not in fam(c) for c in cells),
        "matching_no_simulated_order_at_all": sum(not c["matching_sims"] for c in cells),
        "matching_only_fp16_sims": sum(bool(c["matching_sims"]) and not fam(c) for c in cells),
        "per_result": [{"shape_MKN": r["shape_MKN"], "pattern": r["pattern"], "variant": v,
                        "value": c["unique_values"], "exact_rounded_to_fp16": r["exact_rounded_to_fp16"],
                        "exact": r["exact"], "matching_sims": c["matching_sims"]}
                       for r in hard for v, c in r["mps"].items()],
    }
    s["structured_cases_built_to_defeat_fp32_accumulation"] = {
        "shape_pattern_pairs": len(hard), "mps_results": len(cells),
        "mps_results_equal_to_exact_rounded": sum(c["all_equal_exact_rounded"] for c in cells),
        "mps_results_with_more_than_one_distinct_value_across_M_N": sum(len(c["unique_values"]) > 1 for c in cells),
    }
    rnd_rows = {}
    for dist in sorted({r["distribution"] for r in result["random"]}):
        rows = [r for r in result["random"] if r["distribution"] == dist]
        big = [r for r in rows if r["shape_MKN"][1] >= 4096]
        def worst(rows_, key, pick, where="mps"):
            vals = [v[key] for r in rows_ for v in r[where].values() if isinstance(v[key], float)]
            return pick(vals) if vals else None
        def sim(rows_, name, key, pick):
            vals = [r["simulated"][name][key] for r in rows_ if isinstance(r["simulated"][name][key], float)]
            return pick(vals) if vals else None
        rnd_rows[dist] = {
            "mps_min_share_equal_to_rounded_fp64_all_shapes": worst(rows, "share_equal_to_rounded_fp64", min),
            "mps_max_err_over_sum_abs_products_all_shapes": worst(rows, "max_err_over_sum_abs_products", max),
            "K>=4096": {
                "mps_min_share_equal_to_rounded_fp64": worst(big, "share_equal_to_rounded_fp64", min),
                "mps_max_share_equal_to_rounded_fp64": worst(big, "share_equal_to_rounded_fp64", max),
                "mps_subset_max_err_over_sum_abs_products": worst(big, "max_err_over_sum_abs_products", max, "mps_on_sim_subset"),
                "fp32_sims_max_err_over_sum_abs_products": max(
                    sim(big, f"fp32_acc_then_round/{o}", "max_err_over_sum_abs_products", max) for o in ORDERS),
                "fp16_sims_min_err_over_sum_abs_products": min(
                    (sim(big, f"fp16_acc/{o}", "max_err_over_sum_abs_products", min) or float("inf")) for o in ORDERS),
                "fp16_sims_max_share_equal_to_rounded_fp64": max(
                    sim(big, f"fp16_acc/{o}", "share_equal_to_rounded_fp64", max) for o in ORDERS),
                "fp32_sims_min_share_equal_to_rounded_fp64": min(
                    sim(big, f"fp32_acc_then_round/{o}", "share_equal_to_rounded_fp64", min) for o in ORDERS),
                "fp32_sims_max_share_equal_to_rounded_fp64": max(
                    sim(big, f"fp32_acc_then_round/{o}", "share_equal_to_rounded_fp64", max) for o in ORDERS),
            },
        }
    s["random"] = rnd_rows
    return s


# ---------------------------------------------------------------- main

SHAPES = [(1, 16, 1), (4, 64, 4), (3, 1023, 5), (64, 1024, 64), (512, 512, 512), (512, 4096, 512),
          (1, 16384, 1), (1, 16384, 512), (512, 16384, 1), (512, 16384, 512), (8, 32768, 8)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="JSON result file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim-side", type=int, default=32, help="simulate a sim-side x sim-side corner of each random GEMM")
    ap.add_argument("--bound-shape", type=int, nargs=3, default=[512, 16384, 512], metavar=("M", "K", "N"))
    ap.add_argument("--convrot-group", type=int, default=256)
    ap.add_argument("--skip-kernel", action="store_true", help="skip the vendored comfy-kitchen arm")
    args = ap.parse_args()
    if not torch.backends.mps.is_available():
        sys.exit("MPS is not available")
    dev = torch.device("mps")
    t0 = time.time()
    result = {
        "script": "bench/probe_fp16_accumulation.py",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "macos": platform.mac_ver()[0], "chip": chip(), "seed": args.seed,
        "oracle": "float64 CPU matmul of the same fp16 operands (products exact; sum error about K * 2^-53 of sum |x w|)",
        "simulations": "numpy; every addition rounded to the accumulator format, nearest even; orders: "
                       "seq, pairwise, lanes:L (L strided accumulators), blocks:B (contiguous blocks of B)",
        "shapes_MKN": SHAPES,
    }
    result["structured"] = run_structured(SHAPES, dev)
    result["random"] = run_random(SHAPES, dev, args.seed, args.sim_side)
    m, k, n = args.bound_shape
    result["bound_K16384"] = run_bound_emulated(m, k, n, dev, args.seed)
    if not args.skip_kernel:
        rows, source = run_bound_kernel(m, k, n, dev, args.seed, args.convrot_group)
        result["bound_K16384"] += rows
        result["comfy_kitchen_source"] = source
    result["summary"] = summarize(result)
    result["fp16_reference_points"] = {
        "half_ulp_relative_worst_case_2^-11": 2.0 ** -11, "fp16_max": F16_MAX,
        "fp32_sequential_accumulation_worst_case_K*2^-24_of_sum_abs_products": k * 2.0 ** -24,
    }
    result["mps_driver_allocated_bytes_at_end"] = int(torch.mps.driver_allocated_memory())
    result["seconds"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.out} in {result['seconds']} s")


if __name__ == "__main__":
    main()
