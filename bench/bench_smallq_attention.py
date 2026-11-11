"""Experiment E of the mps-decode thread: isolated attention timing for T = 1 to 16 new rows over a
long fp16 cache, stock torch SDPA against the small-query kernel (bench/smallq_attention.py).

Every timed sample is 28 attention calls, one per Qwen3-1.7B layer, each layer with its own K/V
cache tensors (at 4,096 keys that is about 470 MB, far beyond the on-die cache; one cache reused
in a loop would be cache-resident and invalid). The device is synchronised before and after each
sample. Arm order rotates every round; warm-up rounds are discarded; every sample is kept.

Arms:
  sdpa_gqa     F.scaled_dot_product_attention on the whole static cache with a [T, Lmax] boolean
               mask and enable_gqa=True
  sdpa_repeat  the same with K/V repeated to Hq heads first (transformers' repeat_kv path; the
               copy is inside the timing, as it is in the model)
  sdpa_chunk8  zero-kernel arm: T split into chunks of at most 8 rows, one sdpa_gqa call per chunk
               with that chunk's mask rows, so every call stays on the vector path
  kernel       smallq_attention(q, k, v, pos)
  kernel_dev   smallq_attention_dev(q, k, v, pos as a device tensor): chunks sized from Lmax, pos read
               on the GPU (grid --dev-arm only; the path the transformers integration uses)

Statistics. The arms in one round run back to back, so they share whatever the machine is doing
at that moment, and absolute times drift together within a process and between processes (the
same 4,096-key cell has measured 9.7 to 14.5 ms for stock at T = 8). The headline speed-up is
therefore the per-round paired ratio, arm / kernel in the same round, as a median with its
interquartile range; a ratio of two medians inherits the drift and is not used. Absolute ms are
reported as the range of per-process medians over several processes.

Subcommands (grid and tune touch the GPU: run them through the GPU lock, one JSON per process):
  grid     the arms over --lengths x --T at Qwen3-1.7B shapes (B 1, Hq 16, Hkv 8, D 128),
           Lmax = L_valid + 16
  tune     the kernel under several Config values (and sdpa_gqa as a yardstick) on a few cells
  combine  CPU only: a markdown table over one or more grid JSONs, per cell the paired ratio of
           each stock arm to the kernel (per-process medians and pooled median with IQR) and the
           range of per-process median ms

python3 bench/gpu_runner.py --monitor --seconds 540 --rss-gib 14 -- <python> \\
    bench/bench_smallq_attention.py grid --lengths 4096 --out f.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import smallq_attention as sq  # noqa: E402

LAYERS, B, HQ, HKV, D = 28, 1, 16, 8, 128
EXTRA = 16  # Lmax = L_valid + EXTRA


def sync():
    torch.mps.synchronize()


def caches(L_valid, seed=0):
    """28 distinct (k, v) pairs, [B, Hkv, Lmax, D] fp16; the slots past L_valid are zeros, as in a
    freshly allocated static cache."""
    g = torch.Generator(device="mps").manual_seed(seed)
    out = []
    for _ in range(LAYERS):
        k = torch.zeros(B, HKV, L_valid + EXTRA, D, dtype=torch.float16, device="mps")
        v = torch.zeros_like(k)
        k[:, :, :L_valid].normal_(generator=g)
        v[:, :, :L_valid].normal_(generator=g)
        out.append((k, v))
    sync()
    return out


def mask_for(T, pos, Lmax):
    return (torch.arange(Lmax, device="mps")[None, :] <= pos + torch.arange(T, device="mps")[:, None])


def repeat_kv(x, n):
    b, h, L, d = x.shape
    return x[:, :, None].expand(b, h, n, L, d).reshape(b, h * n, L, d)


def make_arms(T, L_valid, cfg=None, dev=False):
    pos = L_valid - T
    Lmax = L_valid + EXTRA
    q = torch.randn(B, HQ, T, D, device="mps", dtype=torch.float16)
    mask = mask_for(T, pos, Lmax)
    chunks = [(c, min(c + 8, T)) for c in range(0, T, 8)]
    G = HQ // HKV

    def sdpa_gqa(k, v):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)

    def sdpa_repeat(k, v):
        return F.scaled_dot_product_attention(q, repeat_kv(k, G), repeat_kv(v, G), attn_mask=mask)

    def sdpa_chunk8(k, v):
        outs = [F.scaled_dot_product_attention(q[:, :, a:b], k, v, attn_mask=mask[a:b], enable_gqa=True)
                for a, b in chunks]
        return outs[0] if len(outs) == 1 else torch.cat(outs, 2)

    def kernel(k, v):
        return sq.smallq_attention(q, k, v, pos, config=cfg)  # cfg None: pick_config

    arms = {"sdpa_gqa": sdpa_gqa, "sdpa_repeat": sdpa_repeat, "sdpa_chunk8": sdpa_chunk8, "kernel": kernel}
    if dev:
        pos_t = torch.tensor([pos], device="mps")
        arms["kernel_dev"] = lambda k, v: sq.smallq_attention_dev(q, k, v, pos_t, config=cfg)
    return arms


def time_layers(fn, kv):
    sync()
    t0 = time.perf_counter()
    for k, v in kv:
        fn(k, v)
    sync()
    return (time.perf_counter() - t0) * 1e3


def run_cell(arms, kv, rounds, warmup):
    names = list(arms)
    samples = {n: [] for n in names}
    for r in range(warmup + rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for n in order:
            ms = time_layers(arms[n], kv)
            if r >= warmup:
                samples[n].append(ms)
    return samples


def max_diff(arms, kv):
    """Max abs difference of each arm against sdpa_gqa on layer 0 (a sanity check, not the test)."""
    k, v = kv[0]
    ref = arms["sdpa_gqa"](k, v).float()
    return {n: (f(k, v).float() - ref).abs().max().item() for n, f in arms.items()}


def kernel_bytes(L_valid):
    """K and V read once per kv head, 28 layers."""
    return LAYERS * 2 * B * HKV * L_valid * D * 2


def summarize(samples):
    return {n: {"median_ms": statistics.median(s), "min_ms": min(s), "max_ms": max(s), "samples_ms": s}
            for n, s in samples.items()}


def quartiles(xs):
    """(q1, median, q3) of a non-empty list."""
    if len(xs) == 1:
        return xs[0], xs[0], xs[0]
    q1, q2, q3 = statistics.quantiles(xs, n=4, method="inclusive")
    return q1, q2, q3


def paired_ratios(samples, against="kernel"):
    """Per arm, the per-round ratios arm / `against` (sample r of every arm comes from round r)."""
    base = samples[against]
    return {n: [a / b for a, b in zip(s, base)] for n, s in samples.items() if n != against}


def paired_summary(samples, against="kernel"):
    out = {}
    for n, r in paired_ratios(samples, against).items():
        q1, med, q3 = quartiles(r)
        out[n] = {"median": med, "q1": q1, "q3": q3, "frac_ge_1_5": sum(x >= 1.5 for x in r) / len(r),
                  "ratios": r}
    return out


def env():
    return {"torch": torch.__version__, "machine": platform.machine(), "platform": platform.platform(),
            "layers": LAYERS, "B": B, "Hq": HQ, "Hkv": HKV, "D": D, "extra": EXTRA}


def cmd_grid(a):
    cfg = sq.Config.parse(a.config) if a.config else None
    result = {"cmd": "grid", "env": env(), "config": a.config or "pick_config",
              "rounds": a.rounds, "warmup": a.warmup, "cells": []}
    t_start = time.time()
    for L in a.lengths:
        kv = caches(L)
        for T in a.T:
            arms = make_arms(T, L, cfg, dev=a.dev_arm)
            diff = max_diff(arms, kv)
            s = run_cell(arms, kv, a.rounds, a.warmup)
            summ = summarize(s)
            nbytes = kernel_bytes(L)
            km = summ["kernel"]["median_ms"]
            cell = {"L_valid": L, "Lmax": L + EXTRA, "T": T, "pos": L - T, "max_abs_diff_vs_sdpa_gqa": diff,
                    "kernel_config": sq.pick_config(L, T).name() if cfg is None else cfg.name(),
                    "kernel_bytes": nbytes, "kernel_gbps": nbytes / (km * 1e-3) / 1e9, "arms": summ,
                    "paired_vs_kernel": paired_summary(s)}
            result["cells"].append(cell)
            print(json.dumps({"L": L, "T": T, **{n: round(v["median_ms"], 3) for n, v in summ.items()},
                              "gqa/kernel paired": round(cell["paired_vs_kernel"]["sdpa_gqa"]["median"], 3),
                              "kernel_gbps": round(cell["kernel_gbps"], 1)}), flush=True)
        del kv
        torch.mps.empty_cache()
    result["seconds"] = time.time() - t_start
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))


def cmd_tune(a):
    configs = [sq.Config.parse(c) for c in a.configs]
    result = {"cmd": "tune", "env": env(), "rounds": a.rounds, "warmup": a.warmup, "cells": []}
    for L in a.lengths:
        kv = caches(L)
        for T in a.T:
            base = make_arms(T, L)
            arms = {"sdpa_gqa": base["sdpa_gqa"]}
            for c in configs:
                arms[c.name()] = make_arms(T, L, c)["kernel"]
            s = run_cell(arms, kv, a.rounds, a.warmup)
            summ = summarize(s)
            result["cells"].append({"L_valid": L, "T": T, "arms": summ})
            best = min(configs, key=lambda c: summ[c.name()]["median_ms"])
            print(json.dumps({"L": L, "T": T, "sdpa_gqa": round(summ["sdpa_gqa"]["median_ms"], 3),
                              **{c.name(): round(summ[c.name()]["median_ms"], 3) for c in configs},
                              "best": best.name()}), flush=True)
        del kv
        torch.mps.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))


def cmd_combine(a):
    """Pool grid JSONs (one or more processes per length) into one markdown table on stdout."""
    cells = {}
    for f in a.files:
        d = json.loads(Path(f).read_text())
        for c in d["cells"]:
            cells.setdefault((c["L_valid"], c["T"]), []).append(c)
    stock = ["sdpa_gqa", "sdpa_repeat", "sdpa_chunk8"]
    arms = stock + ["kernel"]

    def rng(xs, fmt="{:.2f}"):
        lo, hi = fmt.format(min(xs)), fmt.format(max(xs))
        return lo if lo == hi else f"{lo}-{hi}"

    head = (["L_valid", "T", "procs"] + [f"{n} ms" for n in arms]
            + [f"{n}/kernel paired, median [IQR]" for n in stock]
            + ["gqa/kernel per-process paired medians", "kernel GB/s"])
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for (L, T), cs in sorted(cells.items()):
        row = [str(L), str(T), str(len(cs))]
        row += [rng([c["arms"][n]["median_ms"] for c in cs]) for n in arms]
        for n in stock:
            pooled = [x for c in cs for x in paired_ratios({m: c["arms"][m]["samples_ms"] for m in arms})[n]]
            q1, med, q3 = quartiles(pooled)
            row.append(f"{med:.2f} [{q1:.2f}-{q3:.2f}]")
        per = [quartiles(paired_ratios({m: c["arms"][m]["samples_ms"] for m in arms})["sdpa_gqa"])[1] for c in cs]
        row.append(", ".join(f"{x:.2f}" for x in per))
        row.append(rng([c["kernel_bytes"] / (c["arms"]["kernel"]["median_ms"] * 1e-3) / 1e9 for c in cs], "{:.0f}"))
        print("| " + " | ".join(row) + " |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    g.add_argument("--T", type=int, nargs="+", default=[1, 2, 4, 5, 8, 9, 12, 16])
    g.add_argument("--rounds", type=int, default=10)
    g.add_argument("--warmup", type=int, default=2)
    g.add_argument("--config", default=None, help="kernel Config name; default pick_config")
    g.add_argument("--dev-arm", action="store_true", help="also time the device-position entry")
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_grid)
    t = sub.add_parser("tune")
    t.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384])
    t.add_argument("--T", type=int, nargs="+", default=[1, 8, 16])
    t.add_argument("--configs", nargs="+", required=True)
    t.add_argument("--rounds", type=int, default=6)
    t.add_argument("--warmup", type=int, default=1)
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_tune)
    c = sub.add_parser("combine")
    c.add_argument("files", nargs="+", help="grid JSONs; several per length are pooled")
    c.set_defaults(fn=cmd_combine)
    a = p.parse_args()
    torch.manual_seed(0)
    a.fn(a)


if __name__ == "__main__":
    main()
