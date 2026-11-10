"""Experiment B of the mps-decode thread: a multi-row linear kernel for 2 to 16 rows.

Subcommands (run every one through the GPU lock):
  correctness  float64 CPU reference against the kernel and F.linear, random and real weights,
               plus planted bugs (source mutations) that the check must catch; planted_table
               mutates the family of every config in the tuned table
  tune         time every kernel variant per shape and row count, fp16 or bf16 (--grid round2
               adds the round-2 families mmh, mmp, mmt)
  layers       F.linear, the kernel, and two noise arms (F.linear against itself, the kernel
               against itself), weights rotated through copies larger than the on-die cache;
               --cells times chosen cells only, --compare adds an arm for another config
  model        Qwen3-1.7B fp16 decode, stock against patched, interleaved in one process (round 1)
  decode       round 2's end-to-end run: nn.Linear's own forward against the patch, per-row logits
  probe        diagnosis: kernels with loads made cache-resident, loads removed, or multiplies
               removed (round 2)

python3 bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10 -- <python> bench/bench_multirow.py layers --out f.json
Model runs need HF_HOME pointing at a cache holding Qwen/Qwen3-1.7B and HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import multirow_linear as mrl  # noqa: E402

REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SHAPES = {  # name: (K, N)
    "q17_qo": (2048, 2048), "q17_kv": (2048, 1024), "q17_up": (2048, 6144),
    "q17_down": (6144, 2048), "q17_head": (2048, 151936),
    "q4b_up": (2560, 9728), "llama7b_up": (4096, 11008),
}
QWEN17 = ["q17_qo", "q17_kv", "q17_up", "q17_down", "q17_head"]
ROWS = [1, 2, 4, 8, 12, 16, 24, 32]
UNIT_ROUNDOFF = {torch.float16: 2.0 ** -11, torch.bfloat16: 2.0 ** -8}
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def sync():
    torch.mps.synchronize()


def emit(**row):
    print(json.dumps(row), flush=True)


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))


# ---------------------------------------------------------------- correctness

def errors(got, ref, absref):
    """Max error normalised by sum |x||w| (+|b|), and relative L2 error."""
    got = got.cpu().double()
    mask = absref > 0
    norm = ((got - ref).abs()[mask] / absref[mask]).max().item() if mask.any() else 0.0
    rel = ((got - ref).norm() / ref.norm().clamp_min(1e-300)).item()
    return norm, rel


def check_case(x, w, b, configs, lib=None, ref_rows=None):
    """x, w, b are CPU tensors in the test dtype. Returns per-config verdicts against F.linear."""
    dt = x.dtype
    K = x.shape[1]
    xd, wd = x.double(), w.double()
    rows = ref_rows if ref_rows is not None else slice(None)
    ws = wd[rows]
    ref = xd @ ws.t()
    absref = xd.abs() @ ws.abs().t()
    if b is not None:
        ref = ref + b.double()[rows]
        absref = absref + b.double()[rows].abs()
    gx, gw = x.to("mps"), w.to("mps")
    gb = b.to("mps") if b is not None else None
    lin = F.linear(gx, gw, gb).cpu()[:, rows]
    lin_norm, lin_rel = errors(lin, ref, absref)
    allow = UNIT_ROUNDOFF[dt] + K * 2.0 ** -24
    out = {"F.linear": {"max_norm_err": lin_norm, "rel_l2": lin_rel}, "allowance": allow, "configs": {}}
    ok_all = True
    for cfg in configs:
        if not cfg.fits(x.shape[0]):
            continue  # exceeds 32 KB of threadgroup memory; never selected
        y = mrl.multirow_linear(gx, gw, gb, config=cfg, lib=lib).cpu()[:, rows]
        n, r = errors(y, ref, absref)
        finite = bool(torch.isfinite(y).all())
        ok = finite and n <= lin_norm + allow and r <= 1.25 * lin_rel + math.sqrt(K) * 2.0 ** -24
        ok_all &= ok
        out["configs"][cfg.name()] = {"max_norm_err": n, "rel_l2": r, "finite": finite, "pass": ok,
                                      "bitwise_equal_to_F.linear": bool(torch.equal(y, lin))}
    out["pass"] = ok_all
    return out


def all_variants(S=(4,)):
    """Every kernel template the tuned table can select, over its parameter range."""
    out = [mrl.Config(xs, R, U, s) for xs in ("dev", "tg") for R in (1, 2, 4) for U in (1, 2) for s in S]
    out += [mrl.Config("mma", R, U, s, P) for R in (1, 2) for U in (1, 2) for P in (1, 4) for s in S]
    out += [mrl.Config("mmr", R, U, s, P, F) for R in (1, 2, 4) for U in (1, 2) for P in (1, 3) for F in (0, 1)
            for s in S]
    out += [mrl.Config("mmr", 2, 1, s, 1, 0, 1) for s in S]  # half-precision fragments
    if S == (4,):
        out += [mrl.Config.parse(c) for c in ROUND2_VARIANTS]
        out += sorted({mrl.Config.parse(c) for t in mrl.TUNED.values() for row in t.values() for c in row.values()
                       if c != mrl.LINEAR} - set(out), key=lambda c: c.name())
    return out


# Round-2 families: every code path (S = 1 and S > 1, prefetch, K split, half and fp32 operands,
# U = 1 and 2, NS = 1 to 3) at least once.
ROUND2_VARIANTS = ["mmh_r1_u1_s1", "mmh_r2_u1_s4_h1", "mmh_r2_u2_s8_f1_h1", "mmh_r1_u2_s2_p3_f1",
                   "mmh_r2_u1_s4_p3_h1", "mmh_r3_u1_s2_h1", "mmh_r2_u1_s8_f1", "mmh_r1_u1_s8_h1", "mmh_r2_u2_s4",
                   "mmp_r2_u1_s4_h1", "mmp_r1_u2_s2_p3_f1", "mmp_r2_u1_s1",
                   "mmt_r2_u1_s4_h1", "mmt_r1_u2_s8_h1", "mmt_r2_u1_s4_p3"]


def real_weights():
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    index = json.loads(Path(hf_hub_download(REPO, "model.safetensors.index.json", revision=REV)).read_text())
    names = {"q17_qo": "model.layers.0.self_attn.q_proj.weight",
             "q17_kv": "model.layers.0.self_attn.k_proj.weight",
             "q17_up": "model.layers.0.mlp.up_proj.weight",
             "q17_down": "model.layers.0.mlp.down_proj.weight",
             "q17_head": "model.embed_tokens.weight"}  # tied head
    out = {}
    for key, name in names.items():
        path = hf_hub_download(REPO, index["weight_map"][name], revision=REV)
        with safe_open(path, framework="pt") as f:
            out[key] = f.get_tensor(name)
    return out


def cmd_correctness(a):
    parts = set(a.parts.split(","))
    torch.manual_seed(20261110)
    res = {"torch": torch.__version__, "criterion": (
        "per config: max over elements of |y - ref| / (sum_k |x||w| + |b|) <= the same for F.linear "
        "+ (u_out + K * 2^-24), and rel-L2 <= 1.25 * F.linear's + sqrt(K) * 2^-24"), "cases": []}
    started = time.time()
    failures = 0

    def record(tag, dt, M, K, N, bias, out):
        nonlocal failures
        failures += not out["pass"]
        worst = max(out["configs"].values(), key=lambda c: c["max_norm_err"])
        res["cases"].append({"source": tag, "dtype": str(dt).split(".")[-1], "M": M, "K": K, "N": N,
                             "bias": bias, **out})
        emit(event="case", source=tag, dtype=str(dt), M=M, K=K, N=N, bias=bias, ok=out["pass"],
             lin=out["F.linear"]["max_norm_err"], worst=worst["max_norm_err"], allow=out["allowance"])
        if len(res["cases"]) % 16 == 0:
            res["failures"] = failures
            dump(a.out, res)

    variants = all_variants()
    if "random" in parts:
        # Random inputs: odd N, K not a multiple of the 256-element tile, every M from 1 to 32.
        for dt in (torch.float16, torch.bfloat16):
            for K, N, Ms in [(2056, 1027, range(1, 33)), (264, 33, range(1, 33)), (8, 5, range(1, 33)),
                             (4104, 4099, (1, 2, 7, 16, 17, 32))]:
                for M in Ms:
                    for bias in (False, True):
                        x = torch.randn(M, K).to(dt)
                        w = (torch.randn(N, K) / math.sqrt(K)).to(dt)
                        b = torch.randn(N).to(dt) if bias else None
                        record("random", dt, M, K, N, bias, check_case(x, w, b, variants))
            # Threadgroup sizes other than 4 simdgroups, on one shape.
            for M in (3, 16):
                x = torch.randn(M, 2056).to(dt); w = (torch.randn(1027, 2056) / 45).to(dt)
                record("random_S2_S8", dt, M, 2056, 1027, False, check_case(x, w, None, all_variants(S=(2, 8))))
        emit(event="random_done", seconds=time.time() - started)
    if "real" in parts:
        # Real Qwen3-1.7B weights (layer 0 and the tied head), random activations of unit scale.
        # Every variant plus the tuned choice; the head (311M weights) runs only the tuned choice
        # and is compared on 8192 of its rows, including the last.
        weights = real_weights()
        for dt in (torch.float16, torch.bfloat16):
            for key, w32 in weights.items():
                w = w32.to(dt)
                N, K = w.shape
                rows = None
                if key == "q17_head":
                    g = torch.Generator().manual_seed(1)
                    rows = torch.cat([torch.randperm(N - 1, generator=g)[:8191], torch.tensor([N - 1])])
                for M in (ROWS if key != "q17_head" else (1, 2, 4, 8, 12, 16, 32)):
                    for bias in ((False, True) if key == "q17_kv" else (False,)):
                        x = torch.randn(M, K).to(dt)
                        b = (torch.randn(N) * 0.1).to(dt) if bias else None
                        tuned = [c for c in [mrl.pick_config(M, K, N, dt)] if c is not None]  # None: F.linear
                        if key == "q17_head" and not tuned:
                            continue
                        cfgs = tuned if key == "q17_head" else variants + [c for c in tuned if c not in variants]
                        record("qwen3-1.7b:" + key, dt, M, K, N, bias, check_case(x, w, b, cfgs, ref_rows=rows))
        emit(event="real_done", seconds=time.time() - started)
    planted = []
    if "planted" in parts:
        # Planted bugs: source mutations that skip or misalign one K tile. The check must fail each.
        src = mrl.SOURCE_PATH.read_text()
        mutations = [
            (mrl.Config("dev", 2, 1, 4), "for (uint k = lane; k < K8;", "for (uint k = lane + 32; k < K8;"),
            (mrl.Config("tg", 2, 1, 4), "for (uint k0 = 0; k0 < K8;", "for (uint k0 = TILE; k0 < K8;"),
            (mrl.Config("mma", 1, 1, 4), "for (uint k0 = kb; k0 < ke; k0 += KC)", "for (uint k0 = kb + KC; k0 < ke; k0 += KC)"),
            (mrl.Config("mmr", 1, 1, 4), "for (uint k = kb; k < ke; k += 32 * U)", "for (uint k = kb + 32 * U; k < ke; k += 32 * U)"),
            # Round 2. Gross: skip a chunk, a tile, or a simdgroup's partial sum; swap x halves.
            (mrl.Config("mmh", 2, 1, 4, 1, 0, 1), "uint k8 = kb8 + sg * 4 * U;", "uint k8 = kb8 + sg * 4 * U + STEP8;"),
            (mrl.Config("mmh", 2, 1, 4), "for (int s2 = 0; s2 < S; ++s2)", "for (int s2 = 1; s2 < S; ++s2)"),
            (mrl.Config("mmh", 2, 1, 4, 1, 0, 1), "return half2(hi ? ha.y : ha.x, hi ? hb.y : hb.x);",
             "return half2(hi ? ha.x : ha.y, hi ? hb.x : hb.y);"),
            (mrl.Config("mmp", 2, 1, 4, 1, 0, 1), "const uint kbase = blk * 32 + 8 * (fm / 2) + (fm & 1);",
             "const uint kbase = blk * 32 + 8 * (fm / 2);"),
            (mrl.Config("mmt", 2, 1, 4, 1, 0, 1), "for (uint t = 0; t < ntiles; ++t) {", "for (uint t = 1; t < ntiles; ++t) {"),
            # Subtle: drop the last partial chunk (8 of 2056 K elements here), drop the last
            # 8 elements of every x row in the packed layout.
            (mrl.Config("mmh", 2, 1, 4, 1, 0, 1), "if (k8 < ke8) {  // at most one partial chunk per simdgroup",
             "if (false) {"),
            (mrl.Config("mmp", 2, 1, 4, 1, 0, 1), "const uint lo = (m0 < uint(M) && k < K)", "const uint lo = (m0 < uint(M) && k + 8 < K)"),
        ]
        for cfg, old, new in mutations:
            assert src.count(old) == 1, old
            lib = mrl._Library(src.replace(old, new))
            for M in ((8,) if cfg.xsrc in ("dev", "tg", "mma", "mmr") else (8, 16)):
                x = torch.randn(M, 2056).to(torch.float16); w = (torch.randn(1027, 2056) / 45).to(torch.float16)
                out = check_case(x, w, None, [cfg], lib=lib)
                planted.append({"variant": cfg.name(), "M": M, "mutation": [old, new], "caught": not out["pass"], **out})
                emit(event="planted_bug", variant=cfg.name(), M=M, caught=not out["pass"],
                     err=list(out["configs"].values())[0]["max_norm_err"],
                     rel=list(out["configs"].values())[0]["rel_l2"], allow=out["allowance"])
    if "planted_table" in parts:
        # Planted bugs on every config the tuned table holds, in fp16 and bf16, with and without
        # bias, at the smallest and largest row count it is used at: each mutation of the config's
        # family must be caught. K = 2056 leaves a partial last chunk, so the tail mutation drops
        # 8 of 2056 K elements.
        src = mrl.SOURCE_PATH.read_text()
        family = {
            "dev": [("for (uint k = lane; k < K8;", "for (uint k = lane + 32; k < K8;")],
            "tg": [("for (uint k0 = 0; k0 < K8;", "for (uint k0 = TILE; k0 < K8;")],
            "mma": [("for (uint k0 = kb; k0 < ke; k0 += KC)", "for (uint k0 = kb + KC; k0 < ke; k0 += KC)")],
            "mmr": [("for (uint k = kb; k < ke; k += 32 * U)", "for (uint k = kb + 32 * U; k < ke; k += 32 * U)")],
            "mmh": [("uint k8 = kb8 + sg * 4 * U;", "uint k8 = kb8 + sg * 4 * U + STEP8;"),
                    ("if (k8 < ke8) {  // at most one partial chunk per simdgroup", "if (false) {")],
        }
        family["mmp"] = family["mmh"]
        used = {}
        for t in mrl.TUNED.values():
            for row in t.values():
                for M, c in row.items():
                    if c != mrl.LINEAR:
                        used.setdefault(c, set()).add(M)
        libs = {}
        for name, Ms in sorted(used.items()):
            cfg = mrl.Config.parse(name)
            for old, new in family[cfg.xsrc]:
                assert src.count(old) == 1, old
                lib = libs.setdefault((old, new), mrl._Library(src.replace(old, new)))
                for dt in (torch.float16, torch.bfloat16):
                    for M in sorted({min(Ms), max(Ms)}):
                        for bias in (False, True):
                            x = torch.randn(M, 2056).to(dt)
                            w = (torch.randn(1027, 2056) / 45).to(dt)
                            b = torch.randn(1027).to(dt) if bias else None
                            out = check_case(x, w, b, [cfg], lib=lib)
                            c = list(out["configs"].values())[0]
                            planted.append({"variant": name, "dtype": str(dt).split(".")[-1], "M": M, "bias": bias,
                                            "mutation": [old, new], "caught": not out["pass"], **out})
                            emit(event="planted_table", variant=name, dtype=str(dt), M=M, bias=bias,
                                 caught=not out["pass"], err=c["max_norm_err"], allow=out["allowance"],
                                 lin=out["F.linear"]["max_norm_err"])
    res["planted_bug"] = planted
    res["failures"] = failures
    res["seconds"] = time.time() - started
    dump(a.out, res)
    emit(event="done", failures=failures, cases=len(res["cases"]),
         planted_caught=all(p["caught"] for p in planted), seconds=res["seconds"])


# ---------------------------------------------------------------- timing

def weight_copies(K, N, dt, min_bytes=256e6, min_copies=4, pre_t=False):
    nbytes = K * N * 2
    n = max(min_copies, math.ceil(min_bytes / nbytes))
    ws = [torch.randn(N, K, device="mps", dtype=dt) / math.sqrt(K) for _ in range(n)]
    wts = [w.t().contiguous() for w in ws] if pre_t else None
    return ws, wts


def iters_for(K, N, ncopies, target_bytes=3e9):
    """Whole passes over the copies, at least one, about target_bytes of weight per sample."""
    return ncopies * max(1, math.ceil(target_bytes / (K * N * 2) / ncopies))


def per_call_us(fn, ncopies, iters):
    sync()
    t0 = time.perf_counter()
    for i in range(iters):
        fn(i % ncopies)
    sync()
    return (time.perf_counter() - t0) / iters * 1e6


def interleaved(arms, ncopies, iters, reps, rng):
    for f in arms.values():  # warm up, compile
        for i in range(ncopies):
            f(i)
    sync()
    samples = {k: [] for k in arms}
    for _ in range(reps):
        order = list(arms)
        rng.shuffle(order)
        for k in order:
            samples[k].append(per_call_us(arms[k], ncopies, iters))
    return samples


def tune_grid():
    """Every variant family with the parameter values that were competitive in first probes."""
    grid = [mrl.Config(xs, R, U, S) for xs in ("dev", "tg") for R in (1, 2, 4) for U in (1, 2) for S in (4, 8)]
    grid += [mrl.Config("mma", R, U, 4, P) for R in (1, 2) for U in (1, 2) for P in (1, 4)]
    grid += [mrl.Config("mmr", R, U, 4, P, F) for R in (1, 2, 4) for U in (1, 2) for P in (1, 2, 4) for F in (0, 1)]
    return grid


def tune_grid_round2(dtype):
    """Round 2: every config the round-1 table chose for 2 to 16 rows, plus the round-2 families
    at the settings that were competitive in the first 16-row sweeps."""
    names = {c for t in ROUND1_TUNED.values() for row in t.values() for m, c in row.items()
             if 2 <= m <= 16 and c != mrl.LINEAR}
    hs = (0, 1) if dtype == "float16" else (0,)
    for R in (1, 2):
        for S in (4, 8):
            for F in (0, 1):
                for H in hs:
                    names.add(mrl.Config("mmh", R, 1, S, 1, F, H).name())
    for H in hs:
        names |= {mrl.Config("mmh", 2, 2, 4, 1, 0, H).name(), mrl.Config("mmh", 2, 1, 2, 1, 0, H).name(),
                  mrl.Config("mmh", 2, 1, 4, 2, 0, H).name(), mrl.Config("mmh", 1, 1, 8, 2, 0, H).name(),
                  mrl.Config("mmp", 2, 1, 4, 1, 0, H).name(), mrl.Config("mmp", 1, 1, 8, 1, 0, H).name(),
                  mrl.Config("mmt", 1, 2, 8, 1, 0, H).name(), mrl.Config("mmt", 2, 2, 4, 1, 0, H).name()}
    return [mrl.Config.parse(n) for n in sorted(names)]


# The round-1 table (commit a441c78, bench/multirow_linear.py), frozen here so `tune --grid round2`
# rebuilds the grid round 2 was tuned on: every config round 1 chose for 2 to 16 rows.
ROUND1_TUNED: dict[str, dict[tuple[int, int], dict[int, str]]] = {
    "bfloat16": {
        (2048, 1024): {1: "dev_r4_u1_s8", 2: "tg_r1_u2_s4", 4: "dev_r2_u1_s4", 8: "mmr_r1_u1_s4_p4_f1", 12: "tg_r1_u2_s8", 16: "mma_r1_u1_s4_p4", 24: "mma_r1_u1_s4_p4", 32: "mma_r1_u1_s4_p4"},
        (2048, 2048): {1: "tg_r4_u1_s8", 2: "tg_r2_u2_s4", 4: "dev_r2_u2_s4", 8: "mmr_r1_u2_s4_f1", 12: "mma_r1_u1_s4_p4", 16: "mmr_r2_u1_s4_p4", 24: "mma_r2_u1_s4_p4", 32: "mma_r2_u1_s4_p4"},
        (2048, 6144): {1: "tg_r1_u2_s4", 2: "dev_r1_u1_s8", 4: "dev_r2_u1_s4", 8: "mmr_r1_u2_s4_f1", 12: "mma_r1_u1_s4", 16: "mma_r1_u1_s4", 24: "mma_r1_u1_s4", 32: "mma_r1_u1_s4"},
        (2048, 151936): {1: "tg_r2_u2_s8", 2: "tg_r1_u1_s4", 4: "tg_r2_u1_s8", 8: "mmr_r1_u1_s4_f1", 12: "mma_r1_u1_s4", 16: "mma_r1_u1_s4", 24: "mma_r1_u1_s4", 32: "mma_r1_u1_s4"},
        (2560, 9728): {1: "tg_r1_u2_s4", 2: "dev_r2_u1_s4", 4: "dev_r2_u1_s8", 8: "mmr_r1_u2_s4_f1", 12: "mma_r1_u1_s4_p4", 16: "mmr_r2_u1_s4", 24: "mma_r1_u1_s4_p4", 32: "mma_r1_u1_s4_p4"},
        (4096, 11008): {1: "tg_r1_u2_s8", 2: "dev_r1_u1_s8", 4: "dev_r2_u1_s4", 8: "mmr_r1_u2_s4_f1", 12: "mma_r1_u1_s4_p4", 16: "mma_r1_u1_s4_p4", 24: "mma_r1_u1_s4_p4", 32: "mma_r1_u1_s4"},
        (6144, 2048): {1: "dev_r1_u2_s4", 2: "dev_r2_u1_s8", 4: "dev_r2_u1_s4", 8: "mmr_r1_u2_s4_f1", 12: "mma_r1_u1_s4_p4", 16: "mma_r1_u1_s4_p4", 24: "mma_r1_u1_s4_p4", 32: "mma_r2_u1_s4_p4"},
    },
    "float16": {
        (2048, 1024): {1: "tg_r1_u2_s4", 2: "dev_r4_u1_s8", 4: "dev_r1_u1_s8", 8: "tg_r1_u1_s4", 12: "tg_r1_u1_s8", 16: "tg_r1_u1_s8", 24: "mma_r1_u1_s4_p4", 32: "mma_r1_u1_s4_p4"},
        (2048, 2048): {1: "tg_r2_u1_s4", 2: "tg_r1_u2_s8", 4: "dev_r2_u1_s4", 8: "mmr_r1_u2_s4_f1", 12: "tg_r1_u1_s8", 16: "mmr_r2_u1_s4_p4", 24: "mma_r1_u1_s4_p4", 32: "mma_r2_u1_s4_p4"},
        (2048, 6144): {1: "tg_r1_u2_s4", 2: "tg_r1_u2_s8", 4: "dev_r2_u1_s4", 8: "mmr_r1_u2_s4_f1", 12: "mmr_r2_u1_s4_p2", 16: "mma_r1_u1_s4", 24: "mma_r1_u1_s4", 32: "mma_r1_u1_s4"},
        (2048, 151936): {1: "dev_r4_u1_s8", 2: "mmr_r1_u1_s4_f1", 4: "dev_r2_u1_s4", 8: "mmr_r1_u1_s4", 12: "mmr_r2_u1_s4", 16: "mmr_r2_u1_s4", 24: "mma_r1_u1_s4", 32: "mma_r1_u1_s4"},
        (2560, 9728): {1: "dev_r1_u1_s8", 2: "dev_r2_u1_s8", 4: "dev_r2_u1_s4", 8: "mmr_r1_u2_s4_f1", 12: "mmr_r2_u1_s4", 16: "mmr_r2_u1_s4_p2", 24: "mma_r1_u1_s4", 32: "mma_r1_u1_s4"},
        (4096, 11008): {1: "dev_r2_u1_s4", 2: "dev_r2_u2_s8", 4: "tg_r2_u2_s4", 8: "mmr_r1_u2_s4_f1", 12: "mmr_r2_u1_s4_p2", 16: "mmr_r2_u1_s4", 24: "mma_r1_u1_s4_p4", 32: "mma_r1_u1_s4"},
        (6144, 2048): {1: "dev_r4_u1_s4", 2: "tg_r4_u1_s4", 4: "dev_r4_u1_s4", 8: "mmr_r2_u2_s4_p4", 12: "mma_r1_u1_s4_p4", 16: "mmr_r2_u1_s4_p4", 24: "mma_r1_u1_s4_p4", 32: "mma_r2_u1_s4_p4"},
    },
}


def cmd_tune(a):
    dt = DTYPES[a.dtype]
    rng = random.Random(7)
    torch.manual_seed(7)
    if a.configs:
        configs = [mrl.Config.parse(c) for c in a.configs.split(",")]
    else:
        configs = tune_grid_round2(a.dtype) if a.grid == "round2" else tune_grid()
    res = {"torch": torch.__version__, "dtype": a.dtype, "reps": a.reps, "configs": [c.name() for c in configs],
           "rows": []}
    started = time.time()
    for name in a.shapes.split(","):
        K, N = SHAPES[name]
        ws, _ = weight_copies(K, N, dt)
        iters = iters_for(K, N, len(ws), a.target_bytes)
        for M in [int(m) for m in a.rows.split(",")]:
            x = torch.randn(M, K, device="mps", dtype=dt)
            arms = {c.name(): (lambda i, c=c: mrl.multirow_linear(x, ws[i], config=c)) for c in configs if c.fits(M)}
            arms["F.linear"] = lambda i: F.linear(x, ws[i])
            samples = interleaved(arms, len(ws), iters, a.reps, rng)
            med = {k: statistics.median(v) for k, v in samples.items()}
            best = min((k for k in med if k != "F.linear"), key=med.get)
            row = {"shape": name, "K": K, "N": N, "M": M, "copies": len(ws), "iters": iters,
                   "median_us": med, "samples_us": samples, "best": best,
                   "best_GBps": K * N * 2 / med[best] / 1e3, "linear_over_best": med["F.linear"] / med[best]}
            res["rows"].append(row)
            emit(event="tune", shape=name, M=M, best=best, best_us=round(med[best], 1),
                 linear_us=round(med["F.linear"], 1), ratio=round(row["linear_over_best"], 2),
                 GBps=round(row["best_GBps"], 1), t=round(time.time() - started))
            dump(a.out, res)
        del ws
        torch.mps.empty_cache()
        dump(a.out, res)
    dump(a.out, res)


def cmd_layers(a):
    """Per layer shape and row count: F.linear, the kernel with its tuned variant, and two noise
    arms (F.linear against itself, the kernel against itself). Where the table routes a cell to
    F.linear, the kernel arm times the config given by --candidates (if any) instead. --cells
    limits each dtype and shape to the listed row counts; --compare adds an arm timing another
    config for a cell (for example the one the table held before)."""
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    cand = json.loads(Path(a.candidates).read_text()) if a.candidates else {}
    only = json.loads(Path(a.cells).read_text()) if a.cells else None
    comp = json.loads(Path(a.compare).read_text()) if a.compare else {}
    rows_list = [int(m) for m in a.rows.split(",")]
    res = {"torch": torch.__version__, "reps": a.reps, "seed": a.seed,
           "tuned": {dt: {f"{K}x{N}": v for (K, N), v in t.items()} for dt, t in mrl.TUNED.items()},
           "rows": []}
    started = time.time()
    for dname in a.dtypes.split(","):
        dt = DTYPES[dname]
        for name in a.shapes.split(","):
            K, N = SHAPES[name]
            ms = rows_list if only is None else [int(m) for m in only.get(dname, {}).get(name, [])]
            if not ms:
                continue
            ws, wts = weight_copies(K, N, dt, pre_t=a.pre_t)
            iters = iters_for(K, N, len(ws), a.target_bytes)
            for M in ms:
                x = torch.randn(M, K, device="mps", dtype=dt)
                cfg = mrl.pick_config(M, K, N, dt)
                routed = cfg is not None
                if cfg is None and cand.get(dname, {}).get(name, {}).get(str(M)):
                    cfg = mrl.Config.parse(cand[dname][name][str(M)])
                if cfg is None:
                    continue
                arms = {"F.linear": lambda i: F.linear(x, ws[i]),
                        "F.linear_again": lambda i: F.linear(x, ws[i]),
                        "multirow": lambda i: mrl.multirow_linear(x, ws[i], config=cfg),
                        "multirow_again": lambda i: mrl.multirow_linear(x, ws[i], config=cfg)}
                if a.pre_t:
                    arms["mm_pre_t"] = lambda i: torch.mm(x, wts[i])
                other = comp.get(dname, {}).get(name, {}).get(str(M))
                ocfg = mrl.Config.parse(other) if other and other != mrl.LINEAR else None
                if ocfg is not None:
                    arms["compare"] = lambda i: mrl.multirow_linear(x, ws[i], config=ocfg)
                samples = interleaved(arms, len(ws), iters, a.reps, rng)
                med = {k: statistics.median(v) for k, v in samples.items()}
                flop = 2 * M * N * K
                row = {"dtype": dname, "shape": name, "K": K, "N": N, "M": M, "config": cfg.name(),
                       "table_routes_to_kernel": routed,
                       "copies": len(ws), "weight_MB_total": len(ws) * K * N * 2 / 1e6, "iters": iters,
                       "median_us": med, "samples_us": samples,
                       "GBps": {k: K * N * 2 / v / 1e3 for k, v in med.items()},
                       "TFLOPS": {k: flop / v / 1e6 for k, v in med.items()},
                       "linear_over_multirow": med["F.linear"] / med["multirow"],
                       "aa_ratio": med["F.linear_again"] / med["F.linear"],
                       "bb_ratio": med["multirow_again"] / med["multirow"]}
                if ocfg is not None:
                    row["compare_config"] = ocfg.name()
                    row["compare_over_multirow"] = med["compare"] / med["multirow"]
                res["rows"].append(row)
                emit(event="layer", dtype=dname, shape=name, M=M, cfg=cfg.name(),
                     us={k: round(v, 1) for k, v in med.items()},
                     speedup=round(row["linear_over_multirow"], 2), aa=round(row["aa_ratio"], 2),
                     bb=round(row["bb_ratio"], 2), GBps=round(row["GBps"]["multirow"]),
                     TFLOPS=round(row["TFLOPS"]["multirow"], 2), t=round(time.time() - started))
            del ws, wts
            torch.mps.empty_cache()
            dump(a.out, res)
    dump(a.out, res)


# ---------------------------------------------------------------- probe (round 2 diagnosis)

def _edits(*pairs):
    def f(src):
        for old, new in pairs:
            assert src.count(old) == 1, old
            src = src.replace(old, new)
        return src
    return f


def _unroll_mmr(src):
    """Force full unrolling of every constant loop in the mmr variant (mmr_load and mrl_mmr)."""
    i = src.index("template <int M, int NS, int U>\ninline void mmr_load")
    return src[:i] + src[i:].replace("for (int ", "_Pragma(\"unroll\") for (int ")


_L1W = ("wrow[ns][kw / 8]", "wrow[ns][(kw / 8) & 7]")
_L1X = ("(xrow[mt][j] + kx)", "(xrow[mt][j] + (kx & 63))")
_NOMMA = ("++mt) simdgroup_multiply_accumulate(acc[ns][mt], A[ns], B[mt], acc[ns][mt]);",
          "++mt) acc[ns][mt].thread_elements()[0] += A[ns].thread_elements()[0] * B[mt].thread_elements()[1];")
MMR_PROBES = {  # name: source transformation of mrl_mmr
    "base": lambda s: s,
    "l1": _edits(_L1W, _L1X),           # weight and x loads both confined to a cache-resident 64-element window
    "l1w": _edits(_L1W),                # weight only
    "l1x": _edits(_L1X),                # x only
    "nomma": _edits(_NOMMA),            # each 8x8x8 multiply replaced by one FMA
    "l1_nomma": _edits(_L1W, _L1X, _NOMMA),
    "l1x_nomma": _edits(_L1X, _NOMMA),
    "unroll": _unroll_mmr,
}
# The same probes on the round-2 register-fed families (mmh, mmp), applied to frag_load / frag_mma.
_HL1W = ("wq[u][ns] = wok ? wrow[ns][kk] : uint4(0);", "wq[u][ns] = wok ? wrow[ns][kk & 3] : uint4(0);")
_HL1X = ("xq[u][mt][j] = xok ? xrow[mt][j][kk] : uint4(0);", "xq[u][mt][j] = xok ? xrow[mt][j][kk & 3] : uint4(0);")
_HNOMMA = ("                simdgroup_multiply_accumulate(acc[ns][mt], A[ns], B[mt], acc[ns][mt]);\n    }\n}",
           "                acc[ns][mt].thread_elements()[0] += float(A[ns].thread_elements()[0] * B[mt].thread_elements()[1]);\n    }\n}")
_HNOX = ("xq[u][mt][j] = xok ? xrow[mt][j][kk] : uint4(0);", "xq[u][mt][j] = uint4(kk + j);")
_HNOW = ("wq[u][ns] = wok ? wrow[ns][kk] : uint4(0);", "wq[u][ns] = uint4(kk + ns);")
MMH_PROBES = {"base": lambda s: s, "l1": _edits(_HL1W, _HL1X), "l1w": _edits(_HL1W), "nomma": _edits(_HNOMMA),
              "l1_nomma": _edits(_HL1W, _HL1X, _HNOMMA),
              "nox_nomma": _edits(_HNOX, _HNOMMA),   # weight stream only: no x loads, no multiplies
              "now_nomma": _edits(_HNOW, _HNOMMA)}   # x loads only


def cmd_probe(a):
    """Where does the 16-row time go? The mmr / mmh kernels with their loads made cache-resident
    (l1), their matrix multiplies replaced by one FMA (nomma), or loads removed (nox, now)."""
    rng = random.Random(13)
    torch.manual_seed(13)
    res = {"torch": torch.__version__, "reps": a.reps, "rows": []}
    # mmr / mmh probes on the large MLP shapes (the peak multiply rate is round2/peak_mma.py).
    src = mrl.SOURCE_PATH.read_text()
    libs = {name: mrl._Library(MMR_PROBES[name](src)) for name in a.probes.split(",") if name in MMR_PROBES}
    hlibs = {name: mrl._Library(MMH_PROBES[name](src)) for name in a.probes.split(",") if name in MMH_PROBES}
    dt = DTYPES[a.dtype]
    for shape in a.shapes.split(","):
        K, N = SHAPES[shape]
        ws, _ = weight_copies(K, N, dt)
        iters = iters_for(K, N, len(ws), a.target_bytes)
        for M in [int(m) for m in a.rows.split(",")]:
            x = torch.randn(M, K, device="mps", dtype=dt)
            arms = {"F.linear": lambda i: F.linear(x, ws[i])}
            for cname in a.configs.split(","):
                cfg = mrl.Config.parse(cname)
                for pname, lib in (hlibs if cfg.xsrc in ("mmh", "mmp") else libs).items():
                    if cfg.xsrc not in ("mmr", "mmh", "mmp") and pname != "base":
                        continue
                    arms[f"{cname}|{pname}"] = lambda i, c=cfg, l=lib: mrl.multirow_linear(x, ws[i], config=c, lib=l)
            samples = interleaved(arms, len(ws), iters, a.reps, rng)
            med = {k: statistics.median(v) for k, v in samples.items()}
            flop = 2 * M * N * K
            row = {"dtype": a.dtype, "shape": shape, "K": K, "N": N, "M": M, "copies": len(ws), "iters": iters,
                   "median_us": med, "samples_us": samples, "TFLOPS": {k: flop / v / 1e6 for k, v in med.items()},
                   "GBps": {k: K * N * 2 / v / 1e3 for k, v in med.items()}}
            res["rows"].append(row)
            emit(event="probe", shape=shape, M=M, us={k: round(v, 1) for k, v in med.items()})
            dump(a.out, res)
        del ws
        torch.mps.empty_cache()
    dump(a.out, res)


# ---------------------------------------------------------------- model

class Patch:
    enabled = False
    calls = 0


def patch_linears(model):
    import torch.nn as nn
    lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
    for m in lins:
        def forward(x, m=m):
            if Patch.enabled:
                x2 = x.reshape(-1, x.shape[-1])
                if 2 <= x2.shape[0] <= 16 and mrl.supports(x2, m.weight, m.bias):
                    cfg = mrl.pick_config(x2.shape[0], m.in_features, m.out_features, x2.dtype)
                    if cfg is not None:  # None: the tuned table routes this cell to F.linear
                        Patch.calls += 1
                        return mrl.multirow_linear(x2, m.weight, m.bias, config=cfg).view(*x.shape[:-1], m.out_features)
            return F.linear(x, m.weight, m.bias)
        m.forward = forward
    return lins


@torch.no_grad()
def generate(model, ids, steps, forced=None):
    """Greedy decode. Returns tokens [B, steps], per-step logits of the last step and of every step
    (float32 CPU), and decode ms per step (prefill excluded). With `forced`, feed those tokens."""
    if forced is not None:
        forced = forced.to(ids.device)
    out = model(ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    toks, logits = [nxt], [out.logits[:, -1].float()]
    sync()
    t0 = time.perf_counter()
    for s in range(steps - 1):
        feed = forced[:, s:s + 1] if forced is not None else nxt
        out = model(feed, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        toks.append(nxt)
        logits.append(out.logits[:, -1].float())
    sync()
    ms = (time.perf_counter() - t0) / (steps - 1) * 1e3
    return torch.cat(toks, 1).cpu(), torch.stack(logits, 1).cpu(), ms


def prompts(tok, B, length=32):
    text = Path(__file__).resolve().parents[1].joinpath("docs", "findings.md").read_text()
    ids = tok(text, return_tensors="pt").input_ids[0]
    return torch.stack([ids[2000 + 97 * i: 2000 + 97 * i + length] for i in range(B)])


@torch.no_grad()
def linears_ms(lins, B, dt, n=10):
    xs = {}
    for m in lins:
        xs.setdefault(m.in_features, torch.randn(B, m.in_features, device="mps", dtype=dt))
    fns = [(lambda x=xs[m.in_features], m=m: m(x)) for m in lins]
    for f in fns:
        f()
    sync()
    t = time.perf_counter()
    for _ in range(n):
        for f in fns:
            f()
    sync()
    return (time.perf_counter() - t) / n * 1e3


def cmd_model(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = torch.float16
    rng = random.Random(3)
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=dt).to("mps").eval()
    lins = patch_linears(model)
    wbytes = sum(m.weight.numel() * m.weight.element_size() for m in lins)
    res = {"torch": torch.__version__, "model": REPO, "revision": REV, "dtype": "float16",
           "n_linears": len(lins), "linear_weight_GB": wbytes / 1e9, "prompt_len": 32, "new_tokens": a.steps,
           "reps": a.reps, "batches": []}
    started = time.time()
    ref_b1 = None
    for B in [int(b) for b in a.batches.split(",")]:
        ids = prompts(tok, B).to("mps")
        for arm in ("stock", "patched"):  # warm up: allocator, and compile this batch's kernels
            Patch.enabled = arm == "patched"
            generate(model, ids, 4)
        Patch.enabled = False
        runs = {"stock": [], "patched": []}
        for _ in range(a.reps):
            order = ["stock", "patched"]
            rng.shuffle(order)
            for arm in order:
                Patch.enabled = arm == "patched"
                Patch.calls = 0
                toks, logits, ms = generate(model, ids, a.steps)
                runs[arm].append({"ms_per_step": ms, "toks": toks, "logits": logits, "calls": Patch.calls})
        Patch.enabled = False
        stock0 = runs["stock"][0]
        # Teacher-forced: patched model fed the stock tokens, so every step's logits share a context.
        Patch.enabled = True
        _, forced_logits, _ = generate(model, ids, a.steps, forced=stock0["toks"])
        Patch.enabled = False
        _, stock_forced, _ = generate(model, ids, a.steps, forced=stock0["toks"])
        # Linears alone in model order, both arms.
        lin_ms = {}
        for arm in ("stock", "patched", "stock", "patched"):
            Patch.enabled = arm == "patched"
            lin_ms.setdefault(arm, []).append(linears_ms(lins, B, dt))
        Patch.enabled = False

        def maxdiff(p, q):
            return (p - q).abs().max().item()

        agree = [(r["toks"] == stock0["toks"]).float().mean().item() for r in runs["patched"]]
        first_div = []
        for r in runs["patched"]:
            neq = (r["toks"] != stock0["toks"])
            first_div.append([int(row.nonzero()[0]) if row.any() else None for row in neq])
        row = {"batch": B,
               "ms_per_step": {k: [r["ms_per_step"] for r in v] for k, v in runs.items()},
               "median_ms_per_step": {k: statistics.median(r["ms_per_step"] for r in v) for k, v in runs.items()},
               "multirow_calls_per_generate": runs["patched"][0]["calls"],
               "token_agreement_vs_stock_rep0": agree,
               "first_divergent_step_per_row": first_div,
               "stock_token_agreement_run_to_run": [(r["toks"] == stock0["toks"]).float().mean().item()
                                                     for r in runs["stock"][1:]],
               "last_step_logit_maxdiff_patched_vs_stock_free_run": [maxdiff(r["logits"][:, -1], stock0["logits"][:, -1])
                                                                     for r in runs["patched"]],
               "last_step_logit_maxdiff_stock_run_to_run": [maxdiff(r["logits"][:, -1], stock0["logits"][:, -1])
                                                            for r in runs["stock"][1:]],
               "forced_logit_maxdiff_patched_vs_stock": {"last_step": maxdiff(forced_logits[:, -1], stock_forced[:, -1]),
                                                          "all_steps": maxdiff(forced_logits, stock_forced)},
               "forced_logit_maxdiff_stock_vs_stock": maxdiff(stock_forced, stock0["logits"]),
               "logit_abs_max": stock0["logits"].abs().max().item(),
               "linears_only_ms": lin_ms}
        # Near-ties: at each row's first divergent step, the stock top-1 minus top-2 logit gap
        # (the context up to that step is identical in both arms).
        margins = []
        for r, row_div in zip(runs["patched"][:1], first_div[:1]):
            for i, s_div in enumerate(row_div):
                if s_div is None:
                    continue
                top = stock0["logits"][i, s_div].topk(2).values
                ptop = r["logits"][i, s_div].topk(2).values
                margins.append({"row": i, "step": s_div, "stock_top2_gap": (top[0] - top[1]).item(),
                                "patched_top2_gap": (ptop[0] - ptop[1]).item(),
                                "stock_token": int(stock0["toks"][i, s_div]), "patched_token": int(r["toks"][i, s_div])})
        row["divergence_margins"] = margins
        row["tokens"] = {"stock_rep0": stock0["toks"].tolist(), "patched_rep0": runs["patched"][0]["toks"].tolist()}
        if B > 1:
            # Stock's own spread over all rows: the batch computed whole against in two halves,
            # all fed the same stock tokens (the halves take different matrix-kernel paths).
            h = B // 2
            _, l1, _ = generate(model, ids[:h], a.steps, forced=stock0["toks"][:h])
            _, l2, _ = generate(model, ids[h:], a.steps, forced=stock0["toks"][h:])
            halves = torch.cat([l1, l2])
            row["stock_spread_all_rows_whole_vs_halves"] = {"last_step": maxdiff(stock_forced[:, -1], halves[:, -1]),
                                                             "all_steps": maxdiff(stock_forced, halves)}
        if B == 1:
            ref_b1 = (ids, stock0["toks"], stock0["logits"])
        elif ref_b1 is not None:
            # Stock's own kernel-path spread: row 0 at batch B against the same prompt at batch 1,
            # both fed batch 1's stock tokens.
            ids1, toks1, logits1 = ref_b1
            forced = stock0["toks"].clone()
            forced[0] = toks1[0]
            _, lb, _ = generate(model, ids, a.steps, forced=forced)
            row["stock_spread_batchB_row0_vs_batch1"] = {"last_step": maxdiff(lb[0, -1], logits1[0, -1]),
                                                         "all_steps": maxdiff(lb[0], logits1[0])}
            Patch.enabled = True
            _, lp, _ = generate(model, ids, a.steps, forced=forced)
            Patch.enabled = False
            row["patched_batchB_row0_vs_stock_batch1"] = {"last_step": maxdiff(lp[0, -1], logits1[0, -1]),
                                                          "all_steps": maxdiff(lp[0], logits1[0])}
        res["batches"].append(row)
        emit(event="model", B=B, ms=row["median_ms_per_step"], agree=agree,
             forced=row["forced_logit_maxdiff_patched_vs_stock"],
             stock_rr=row["last_step_logit_maxdiff_stock_run_to_run"],
             spread=row.get("stock_spread_batchB_row0_vs_batch1"),
             spread_all=row.get("stock_spread_all_rows_whole_vs_halves"), margins=row["divergence_margins"],
             lin_ms={k: [round(x, 2) for x in v] for k, v in lin_ms.items()}, t=round(time.time() - started))
        dump(a.out, res)
    dump(a.out, res)


def cmd_decode(a):
    """Round 2 end to end: nn.Linear's own forward ("stock") against the patch, interleaved and
    shuffled; greedy token agreement; teacher-forced logits per row against stock at the same batch
    size and against stock at batch 1 (stock's own difference between batch sizes); the linears
    alone in model order."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = torch.float16
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=dt).to("mps").eval()
    lins = patch_linears(model)
    saved = [m.forward for m in lins]

    def set_arm(arm):
        for m, f in zip(lins, saved):
            if arm == "stock":
                m.__dict__.pop("forward", None)
            else:
                m.forward = f
        Patch.enabled = arm == "patched"

    res = {"torch": torch.__version__, "model": REPO, "revision": REV, "dtype": "float16", "seed": a.seed,
           "n_linears": len(lins), "prompt_len": 32, "new_tokens": a.steps, "reps": a.reps,
           "tuned": {d: {f"{K}x{N}": v for (K, N), v in t.items()} for d, t in mrl.TUNED.items()}, "batches": []}
    started = time.time()
    for B in [int(b) for b in a.batches.split(",")]:
        ids = prompts(tok, B).to("mps")
        for arm in ("stock", "patched"):
            set_arm(arm)
            generate(model, ids, 4)
        ms, toks, calls = {"stock": [], "patched": []}, {}, 0
        for _ in range(a.reps):
            order = ["stock", "patched"]
            rng.shuffle(order)
            for arm in order:
                set_arm(arm)
                Patch.calls = 0
                t, _, m = generate(model, ids, a.steps)
                ms[arm].append(m)
                toks.setdefault(arm, []).append(t)
                if arm == "patched":
                    calls = Patch.calls
        T = toks["stock"][0]
        row = {"batch": B, "ms_per_step": ms, "median_ms_per_step": {k: statistics.median(v) for k, v in ms.items()},
               "range_ms_per_step": {k: [min(v), max(v)] for k, v in ms.items()},
               "kernel_calls_per_generate": calls,
               "stock_tokens_repeat": all(torch.equal(t, T) for t in toks["stock"]),
               "token_agreement_patched_vs_stock": [(t == T).float().mean().item() for t in toks["patched"]]}
        # Free-running divergence: where each row first differs, with stock's top-2 gap there.
        div = []
        for i in range(B):
            neq = (toks["patched"][0][i] != T[i]).nonzero()
            if len(neq):
                div.append({"row": i, "step": int(neq[0])})
        # Teacher-forced on stock's tokens.
        set_arm("stock")
        _, LB, _ = generate(model, ids, a.steps, forced=T)
        set_arm("patched")
        _, PB, _ = generate(model, ids, a.steps, forced=T)
        for d in div:
            top = LB[d["row"], d["step"]].topk(2).values
            d["stock_top2_gap"] = (top[0] - top[1]).item()
        row["first_divergence"] = div
        per_row = []
        set_arm("stock")
        for i in range(B):
            r = {"row": i, "patchedB_vs_stockB": (PB[i] - LB[i]).abs().max().item()}
            if B > 1:
                _, L1, _ = generate(model, ids[i:i + 1], a.steps, forced=T[i:i + 1])
                r["stockB_vs_stock1"] = (LB[i] - L1[0]).abs().max().item()
                r["patchedB_vs_stock1"] = (PB[i] - L1[0]).abs().max().item()
            per_row.append(r)
        row["logit_rows"] = per_row
        row["logit_max"] = {k: max(r[k] for r in per_row) for k in per_row[0] if k != "row"}
        row["logit_abs_max"] = LB.abs().max().item()
        lin = {"stock": [], "patched": []}
        for arm in ("stock", "patched", "stock", "patched"):
            set_arm(arm)
            lin[arm].append(linears_ms(lins, B, dt))
        row["linears_only_ms"] = lin
        set_arm("stock")
        res["batches"].append(row)
        emit(event="decode", B=B, ms={k: [round(x, 2) for x in v] for k, v in ms.items()}, calls=calls,
             agree=row["token_agreement_patched_vs_stock"], logit_max=row["logit_max"], div=div,
             lin={k: [round(x, 2) for x in v] for k, v in lin.items()}, t=round(time.time() - started))
        dump(a.out, res)
    dump(a.out, res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("correctness"); c.add_argument("--out", required=True)
    c.add_argument("--parts", default="random,real,planted", help="any of random, real, planted, planted_table")
    t = sub.add_parser("tune"); t.add_argument("--out", required=True)
    t.add_argument("--dtype", default="float16"); t.add_argument("--shapes", default=",".join(SHAPES))
    t.add_argument("--rows", default=",".join(map(str, ROWS)))
    t.add_argument("--reps", type=int, default=3); t.add_argument("--target-bytes", type=float, default=1.5e9)
    t.add_argument("--configs", default="", help="comma-separated config names (default: the --grid)")
    t.add_argument("--grid", default="round1", choices=("round1", "round2"))
    l = sub.add_parser("layers"); l.add_argument("--out", required=True)
    l.add_argument("--dtypes", default="float16,bfloat16"); l.add_argument("--shapes", default=",".join(SHAPES))
    l.add_argument("--reps", type=int, default=7); l.add_argument("--target-bytes", type=float, default=3e9)
    l.add_argument("--rows", default=",".join(map(str, list(range(1, 17)) + [24, 32])))
    l.add_argument("--seed", type=int, default=11); l.add_argument("--pre-t", type=int, default=0)
    l.add_argument("--candidates", default="", help="JSON {dtype: {shape: {M: config}}} for cells routed to F.linear")
    l.add_argument("--cells", default="", help="JSON {dtype: {shape: [M, ...]}}: time only these cells")
    l.add_argument("--compare", default="", help="JSON {dtype: {shape: {M: config}}}: an extra arm per cell")
    pr = sub.add_parser("probe"); pr.add_argument("--out", required=True)
    pr.add_argument("--dtype", default="float16"); pr.add_argument("--shapes", default="q17_up,q17_down")
    pr.add_argument("--rows", default="16"); pr.add_argument("--reps", type=int, default=5)
    pr.add_argument("--target-bytes", type=float, default=1.5e9)
    pr.add_argument("--probes", default=",".join(MMR_PROBES));
    pr.add_argument("--configs", default="mmr_r2_u1_s4,mmr_r2_u1_s4_p4,mmr_r4_u1_s4_p4,mma_r1_u1_s4")
    dc = sub.add_parser("decode"); dc.add_argument("--out", required=True)
    dc.add_argument("--batches", default="1,4,8,12,16"); dc.add_argument("--steps", type=int, default=32)
    dc.add_argument("--reps", type=int, default=4); dc.add_argument("--seed", type=int, default=3)
    m = sub.add_parser("model"); m.add_argument("--out", required=True)
    m.add_argument("--batches", default="1,4,8,16"); m.add_argument("--steps", type=int, default=32)
    m.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(4)
    {"correctness": cmd_correctness, "tune": cmd_tune, "layers": cmd_layers, "model": cmd_model,
     "probe": cmd_probe, "decode": cmd_decode}[a.cmd](a)


if __name__ == "__main__":
    main()
