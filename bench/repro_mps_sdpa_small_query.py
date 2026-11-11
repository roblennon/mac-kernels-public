"""Standalone repro: on MPS, F.scaled_dot_product_attention with 9 query rows costs several times
what it costs with 8, at decode-style shapes over a long key length.

    python repro_mps_sdpa_small_query.py [--lengths 1024,4096,16384] [--T 1,4,8,9,12,16]
                                         [--rounds 9] [--warmup 2] [--dtype float16]
                                         [--variants gqa/none,gqa/bool,mha/none,mha/bool] [--json out.json]

Torch only, synthetic tensors. Shapes are one decode step of a 28-layer GQA model (Qwen3-1.7B):
B = 1, 16 query heads, 8 key/value heads, head dim 128, fp16. Each timed sample is 28 attention
calls, each on its own K/V tensors, because one K/V pair reused in a loop stays in the GPU's
cache at short key lengths and reads unrealistically fast. The device is synchronised before and
after each sample, and the output allocation is inside the timing.

Variants, each at every (L, T):
  gqa  / mha   enable_gqa=True on 8-head K/V, or K/V already expanded to 16 heads (made once,
               not timed) and enable_gqa=False
  none / bool  no mask (every key visible), or a [T, L] boolean mask in which query row i sees
               keys 0 .. L - T + i (the last T keys are the new tokens)
  chunk8       for T > 8 only: the same call split into query chunks of at most 8 rows (gqa,
               bool mask), as a workaround and to show that the extra cost is not in the work

Within a key length, every round runs every cell once, in an order rotated each round, so
background load hits all cells alike. The table reports per call (sample / 28): the median over
rounds, the min-max range, and the ratio to T = 8 of the same variant taken per round and then
the median (a paired ratio, robust to drift between rounds).
"""
import argparse
import json
import platform
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F

LAYERS, B, HQ, HKV, D = 28, 1, 16, 8, 128
DTYPE = torch.float16  # --dtype
DEV = "mps"


def sync():
    torch.mps.synchronize()


def make_tensors(L, T_max, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    kv = []
    for _ in range(LAYERS):
        k = torch.randn(B, HKV, L, D, device=DEV, dtype=DTYPE, generator=g)
        v = torch.randn(B, HKV, L, D, device=DEV, dtype=DTYPE, generator=g)
        kv.append((k, v))
    qs = [torch.randn(B, HQ, T_max, D, device=DEV, dtype=DTYPE, generator=g) for _ in range(LAYERS)]
    sync()
    return kv, qs


def bool_mask(T, L):
    rows = torch.arange(T, device=DEV)[:, None]
    cols = torch.arange(L, device=DEV)[None, :]
    return cols <= (L - T + rows)


def cell_fn(variant, T, L, kv, kv_mha, qs):
    heads, mask_kind, *rest = variant.split("/")
    chunk = bool(rest)
    mask = bool_mask(T, L) if mask_kind == "bool" else None
    pairs = kv if heads == "gqa" else kv_mha
    gqa = heads == "gqa"
    q_views = [q[:, :, :T] for q in qs]

    if not chunk:
        def run():
            for (k, v), q in zip(pairs, q_views):
                F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=gqa)
    else:
        bounds = [(s, min(s + 8, T)) for s in range(0, T, 8)]
        masks = [mask[s:e] for s, e in bounds]

        def run():
            for (k, v), q in zip(pairs, q_views):
                outs = [F.scaled_dot_product_attention(q[:, :, s:e], k, v, attn_mask=m, enable_gqa=gqa)
                        for (s, e), m in zip(bounds, masks)]
                torch.cat(outs, dim=2)
    return run


def time_ms(fn):
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return (time.perf_counter() - t0) * 1e3


def check_split(kv, qs, L):
    """T = 9 in one call against the same rows in two calls (8 + 1); outputs should agree."""
    k, v = kv[0]
    q = qs[0][:, :, :9]
    m = bool_mask(9, L)
    whole = F.scaled_dot_product_attention(q, k, v, attn_mask=m, enable_gqa=True)
    parts = torch.cat([F.scaled_dot_product_attention(q[:, :, :8], k, v, attn_mask=m[:8], enable_gqa=True),
                       F.scaled_dot_product_attention(q[:, :, 8:], k, v, attn_mask=m[8:], enable_gqa=True)], dim=2)
    return (whole.float() - parts.float()).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="1024,4096,16384")
    ap.add_argument("--T", default="1,4,8,9,12,16")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--variants", default="gqa/none,gqa/bool,mha/none,mha/bool")
    ap.add_argument("--json")
    a = ap.parse_args()
    global DTYPE
    DTYPE = getattr(torch, a.dtype)
    lengths = [int(x) for x in a.lengths.split(",")]
    Ts = [int(x) for x in a.T.split(",")]
    variants = a.variants.split(",")

    try:
        os_ver = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip()
    except OSError:
        os_ver = "?"
    print(f"torch {torch.__version__}, macOS {os_ver}, {platform.machine()}, "
          f"B={B} Hq={HQ} Hkv={HKV} D={D} {DTYPE}, {LAYERS} distinct K/V per sample, "
          f"{a.rounds} rounds after {a.warmup} warm-up")
    results = []
    for L in lengths:
        kv, qs = make_tensors(L, max(Ts), seed=L)
        kv_mha = [(k.repeat_interleave(HQ // HKV, dim=1).contiguous(),
                   v.repeat_interleave(HQ // HKV, dim=1).contiguous()) for k, v in kv] \
            if any(v.startswith("mha") for v in variants) else None
        sync()
        cells = [(var, T) for var in variants for T in Ts]
        if "gqa/bool" in variants:
            cells += [("gqa/bool/chunk8", T) for T in Ts if T > 8]
        fns = {c: cell_fn(c[0], c[1], L, kv, kv_mha, qs) for c in cells}
        samples = {c: [] for c in cells}
        for r in range(a.warmup + a.rounds):
            shift = r % len(cells)
            for c in cells[shift:] + cells[:shift]:
                t = time_ms(fns[c])
                if r >= a.warmup:
                    samples[c].append(t / LAYERS)
        err = check_split(kv, qs, L)
        for var in variants + ["gqa/bool/chunk8"]:
            for T in Ts:
                c = (var, T)
                if c not in samples:
                    continue
                ref = samples.get((var if "chunk8" not in var else "gqa/bool", 8))
                xs = samples[c]
                ratio = statistics.median(x / y for x, y in zip(xs, ref)) if ref else None
                results.append({"L": L, "variant": var, "T": T, "median_ms": statistics.median(xs),
                                "min_ms": min(xs), "max_ms": max(xs), "ratio_to_T8": ratio, "samples_ms": xs})
        print(f"\nL = {L}  (max |T=9 one call - two calls 8+1| = {err:.2e})")
        print(f"{'variant':<16}{'T':>4}{'median ms':>11}{'min-max ms':>18}{'/ T=8':>8}")
        for x in results:
            if x["L"] == L:
                print(f"{x['variant']:<16}{x['T']:>4}{x['median_ms']:>11.3f}"
                      f"{x['min_ms']:>9.3f}-{x['max_ms']:<8.3f}{x['ratio_to_T8']:>8.2f}")
        del kv, kv_mha, qs, fns
        torch.mps.empty_cache()
    print("\nms per attention call (sample / 28): median and min-max over rounds; "
          "/ T=8 = median per-round ratio to the same variant at T = 8 (chunk8: to gqa/bool at T = 8)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"torch": torch.__version__, "macos": os_ver, "dtype": a.dtype, "rounds": a.rounds,
                       "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
