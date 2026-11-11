# PyTorch issue: MPS scaled_dot_product_attention is 3x to 10x slower at 9 query rows than at 8

Prepared for [PyTorch issues](https://github.com/pytorch/pytorch/issues/new) (Bug Report; suggested labels `module: mps`, `module: performance`). Drafted November 11, 2026. Not filed. Measured on torch 2.15.0 only; main changed this code on October 24 and was not run.

## Duplicate triage (2026-11-11)

About 20 searches of pytorch/pytorch issues and PRs created since April 2026 (`gh api search/issues`): `sdpa_vector`, `sdpa_vector_2pass`, `kDecodeMaxQueries`, `PYTORCH_MPS_DISABLE_PREFILL_ATTENTION`, "mps scaled_dot_product_attention slow", "mps sdpa query length", "mps attention query length 8", "mps sdpa decode", "mps prefill attention", "mps sdpa performance", "mps speculative decoding", "mps assisted generation", "mps attention draft tokens", "mps sdpa gqa slow", "MPP attention", "mps attention regression 2.15", and (added after review) `mps sdpa "split-k" OR "split k"` and `author:Isalia20 sdpa created:>2026-10-20`. None reports a slowdown at 9 to 16 query rows. But one closed PR already implements the main fix suggested below. Nearest:

- [#201747](https://github.com/pytorch/pytorch/pull/201747) "[MPS] Split-K for SDPA prefill kernels", closed 2026-10-24 without landing. It is part of the same ghstack as #201745. It splits the prefill kernels over the key length when the grid is small: `kSplitTargetThreadgroups = 128`, `kSplitMinKBlocks = 8`, `kMaxSplits = 32`, grid `(nq, H, B * n_splits)`. At batch 1 with 16 heads and one query block, that gives up to 8 splits, which is exactly the 16-threadgroup case below. It is not in main: main's `Attention.mm` history has no commit from it, and main's prefill grid is still `(nq, H, B * slabs)` (`Attention.mm:285`). The PR has no review comments and no stated reason for closing. #201828 (prefill refactor), #201829 (staged scores pass) and #201830 (decode crossover) from the same stack also closed unlanded on 2026-10-24 and 2026-10-25. #201743, #201745 and #201765 landed.
- [#179294](https://github.com/pytorch/pytorch/issues/179294), open (2026-04-03): a plan for MPS SDPA (backward, call site, unused output). Does not cover small-query routing.
- [#201745](https://github.com/pytorch/pytorch/pull/201745), landed on main 2026-10-24 as `f6585f7f18`: moves all MPS attention to Metal and removes the MPSGraph fallback. It keeps the 8-row decode limit. Its benchmark table has no shape with 9 to 63 query rows over a long key length (the query lengths are 1, 2, 4, 7, 8 and then 64 and up), and it was measured on M5 Pro and M3 Ultra. Related, not a duplicate.
- [#201830](https://github.com/pytorch/pytorch/pull/201830) "Tune the two-pass SDPA decode crossover by row count", closed unlanded 2026-10-25, part of the same stack. It concerns the crossover inside the decode path (at most 8 rows), not the 8-row limit.
- [#202957](https://github.com/pytorch/pytorch/pull/202957) (open) and [#198564](https://github.com/pytorch/pytorch/pull/198564) (open): backend selection for the MPS kernels. They keep the same split at 8 rows.

**What main looks like** (pytorch/pytorch `7783ff41354c`, 2026-11-11, read with `gh api`). The limit is still 8: `kDecodeMaxQueries = 8` in `aten/src/ATen/native/mps/kernels/Attention.h:13`. `Attention.mm:403-411` sends any longer query to `sdpa_prefill_mps`. That prefill kernel still launches one threadgroup per (query block, head, batch, V/O slab) (`Attention.mm:285`), with no split over the key length. The MPSGraph fallback and `PYTORCH_MPS_DISABLE_PREFILL_ATTENTION` are gone, so main has no way back to the faster MPSGraph path measured below. The prefill kernels themselves changed in #201745 and #201765, so main's numbers at these shapes are unknown. The routing is the same.

The prefill path has served more than 8 rows since at least v2.13.0, so this is not a 2.15 regression.

## Issue text

---

**Title:** [MPS] `scaled_dot_product_attention` with 9 to 16 query rows over a long KV cache is 3x to 10x slower than with 8 rows

### 🐛 Describe the bug

On an M1 Max, an SDPA call with decode-style shapes and 9 query rows costs several times what the same call costs with 8 rows. Example shapes: batch 1, 16 query heads, 8 KV heads, head dim 128, fp16, a long key length, `enable_gqa=True`. In fp16 with a boolean mask, T = 9 costs 3.2x T = 8 at 1,024 keys, 6.1x at 4,096 keys and 7.5x at 16,384 keys. The cost then stays flat up to T = 16. At 16,384 keys, the T = 9 call reads the 67 MB of keys and values at about 8 GB/s, against about 265 GB/s at T = 1.

This is the shape of speculative decoding and other multi-token verify steps: a 7-to-15-token draft checked against a long cache. In a whole Qwen3-1.7B fp16 forward (28 layers, 4,096-token cache), going from 8 to 9 tokens added about 50 ms (26.8 to 76.7 ms). At a 16,384-token cache it added about 200 ms (48 to 250 ms). That setup was torch.compile with a static cache on transformers 5.18, stock SDPA for attention, and our own Metal kernel for the linear layers, so the absolute times will differ from a stock model. The jump from 8 to 9 rows comes from attention: everything else runs the same way at both lengths. An 8th draft token alone costs more than the whole 8-row verify, so at long context it pays to cap drafts at 7 tokens on MPS.

In v2.15.0, `_scaled_dot_product_attention_math_mps` sends T <= 8 to the vector kernels, which split the key length into 32 blocks from 1,024 keys up (`Attention.mm` [L809-L815](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/mps/operations/Attention.mm#L809-L815), [L858-L866](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/mps/operations/Attention.mm#L858-L866)). T > 8 goes to the prefill kernel ([L832-L856](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/mps/operations/Attention.mm#L832-L856)). It has 32-row query tiles for head dim 128, or 64-row tiles on the MPP path, and one threadgroup per (query tile, head, batch) ([L712-L714](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/mps/operations/Attention.mm#L712-L714)). At batch 1 with 16 heads that is 16 threadgroups. Each one walks the whole key length, and each query head re-reads the KV head it shares. My reading, not a profile, is that this is why the cost depends on the key length and not on T. On main, the 8-row limit and this routing are unchanged (`kernels/Attention.h:13`, `Attention.mm:403-411`).

Repro (standalone, torch only):

```python
import time, torch, torch.nn.functional as F

B, HQ, HKV, D, LAYERS = 1, 16, 8, 128, 28   # one decode step of a 28-layer GQA model
TS, ROUNDS = (1, 4, 8, 9, 12, 16), 15

def ms_per_call(fn):
    torch.mps.synchronize(); t0 = time.perf_counter(); fn(); torch.mps.synchronize()
    return (time.perf_counter() - t0) * 1e3 / LAYERS

def median(xs):
    return sorted(xs)[len(xs) // 2]

for L in (1024, 4096, 16384):
    # 28 distinct K/V pairs: a single pair reused in a loop stays in the GPU cache and times too fast
    kv = [(torch.randn(B, HKV, L, D, device="mps", dtype=torch.float16),
           torch.randn(B, HKV, L, D, device="mps", dtype=torch.float16)) for _ in range(LAYERS)]
    runs = {}
    for T in TS:
        q = torch.randn(B, HQ, T, D, device="mps", dtype=torch.float16)
        mask = torch.arange(L, device="mps")[None, :] <= (L - T + torch.arange(T, device="mps")[:, None])
        runs[T] = lambda q=q, mask=mask: [F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True) for k, v in kv]
    for _ in range(2):  # untimed warm-up of every cell
        for run in runs.values():
            ms_per_call(run)
    times = {T: [] for T in TS}
    for _ in range(ROUNDS):  # each round times every T once, so T=9 and T=8 share conditions
        for T, run in runs.items():
            times[T].append(ms_per_call(run))
    ratio = median([a / b for a, b in zip(times[9], times[8])])
    print(f"L={L}: " + ", ".join(f"T={T} {median(ts):.3f} ms" for T, ts in times.items()) + f"; T=9 / T=8 = {ratio:.1f}")
```

Its output on an M1 Max (torch 2.15.0). Three separate runs gave T = 9 / T = 8 of 3.2 to 3.3, 6.0 and 7.5:

```
L=1024: T=1 0.061 ms, T=4 0.103 ms, T=8 0.174 ms, T=9 0.564 ms, T=12 0.564 ms, T=16 0.568 ms; T=9 / T=8 = 3.3
L=4096: T=1 0.098 ms, T=4 0.200 ms, T=8 0.354 ms, T=9 2.129 ms, T=12 2.146 ms, T=16 2.141 ms; T=9 / T=8 = 6.0
L=16384: T=1 0.249 ms, T=4 0.593 ms, T=8 1.114 ms, T=9 8.346 ms, T=12 8.359 ms, T=16 8.370 ms; T=9 / T=8 = 7.5
```

A fuller script (link) rotates the order of the cells every round and reports paired ratios. It also covers no mask, MHA, other dtypes, `PYTORCH_MPS_DISABLE_PREFILL_ATTENTION=1` and chunking. Milliseconds per call below are medians of 9 rounds, each round 28 calls on distinct KV tensors. The ratio is per round, then the median, over two separate processes:

| keys | T = 1 | T = 4 | T = 8 | T = 9 | T = 12 | T = 16 | T = 9 / T = 8 |
|---|---|---|---|---|---|---|---|
| 1,024 | 0.061 | 0.104 | 0.173 | 0.558 | 0.566 | 0.564 | 3.17 to 3.25 |
| 4,096 | 0.095 | 0.195 | 0.348 | 2.118 | 2.118 | 2.122 | 6.10 |
| 16,384 | 0.253 | 0.591 | 1.107 | 8.363 | 8.395 | 8.384 | 7.54 |

fp16, `enable_gqa=True`, boolean mask. Other variants, as the T = 9 / T = 8 ratio:

- No mask: 3.1x, 5.7 to 5.8x and 6.3x at 1,024, 4,096 and 16,384 keys.
- KV expanded to 16 heads with `enable_gqa=False`: 3.3x, 6.1x and 2.6x. At 16,384 keys the vector path itself is slower here (3.3 ms at T = 8).
- bf16: 8.2x at 4,096 keys and 10.6x at 16,384.
- fp32: 2.4x at 4,096 keys and none at 16,384 keys. The fp16 T = 9 call is slower than the fp32 one: 2.12 against 1.30 ms at 4,096 keys, and 8.4 against 5.1 ms at 16,384. In v2.15.0, half and bfloat16 with T > 8 take the MPP prefill kernel when `has_mpp()` is true (macOS 26.2 or later, Metal 4, Apple7 or later). This machine runs macOS 27.0.1, so it probably takes that kernel. I did not confirm it with a GPU trace.
- `PYTORCH_MPS_DISABLE_PREFILL_ATTENTION=1` (2.15.0 falls back to MPSGraph): T = 9 takes 0.54 ms at 4,096 keys (1.6x T = 8) and 1.96 ms at 16,384 (1.7x). That is 3.9x and 4.3x faster than the prefill kernel at the same shapes. Main no longer has this fallback.
- Splitting the query into chunks of at most 8 rows, one call per chunk: T = 16 costs 2.0x T = 8, against 3.2x to 7.6x in one call. The outputs match the single call (max difference 1.2e-4 in fp16).

A related observation from the same runs: on the vector path, cost also grows with T while the KV bytes stay the same. At 16,384 keys, T = 8 costs 4.4x T = 1. The vector kernels launch per (head, query row, key block) ([L340](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/mps/operations/Attention.mm#L340)), so each block of keys and values is likely read once per query row and per query head. That is an inference.

Only one machine was measured. Numbers from M2 to M5 GPUs would show whether this holds there, especially where the MPP path runs on newer matrix units.

### Versions

- PyTorch 2.15.0 (pip wheel), Python 3.12.11
- macOS 27.0.1 (26A434), arm64
- Apple M1 Max, 32 GB, 32-core GPU

### Expected behavior

Cost should grow smoothly from T = 8 to T = 16 at a fixed key length. Splitting the call into chunks of 8 rows, or the 2.15.0 MPSGraph path, already gives about 1.2x to 2.3x T = 8 at T = 9 to 16.

### Suggested fix (a suggestion; maintainers know the tradeoffs better)

- Split the prefill kernel over the key length when its grid is small. #201747 ("Split-K for SDPA prefill kernels", closed without landing on 2026-10-24) already does this, and at batch 1 with 16 heads it would likely give this shape up to 8 splits. I have not run it. These numbers may help decide whether to revive it; is there a reason it was dropped from the stack?
- Or raise the decode limit above 8 rows (for example to 16 or 32), or send short queries over a long key length to the 2-pass vector kernel.
- Or add a small-query kernel for T up to about 16. It would split over the key length like the 2-pass kernel, but load each key/value block once for every query row and every query head that shares the KV head. That would also address the growth from T = 1 to T = 8.

A prototype of the third option exists outside PyTorch, as a custom Metal op ([link]). It is fp16 only, head dim 128, and tested against a float64 reference. On this M1 Max (2026-11-11), measured in isolation over 28 distinct caches, it is 5.6x faster than stock SDPA at T = 9 to 16 with 4,096 keys, 7x faster at 16,384 keys, and 1.65x faster at T = 8. It is slower than stock at T = 1 and 2 (0.5x to 0.75x). In the whole-model setup above (compiled, our linear kernel, transformers 5.18), swapping it in for stock SDPA makes a Qwen3-1.7B verify forward at T = 9 to 16 2.7x faster at a 4,096-token cache and 5.1x to 5.4x faster at 16,384. It shows the gap can be closed. It is not a drop-in patch.

---

## Notes (not part of the issue)

- **Evidence.** All 2026-11-11, under the GPU lock, raw output in the lane scratch directory (not in Git):
  - `bench/repro_mps_sdpa_small_query.py`, two full runs (fp16, 9 rounds each).
  - One run with `PYTORCH_MPS_DISABLE_PREFILL_ATTENTION=1`, one in fp32 and one in bf16 (4,096 and 16,384 keys, 5 rounds each).
  - The inline repro above, three runs after the review added a warm-up pass over every cell and per-round ratios (`inline_v2_run1..3.log`). The first version had no warm-up before L = 1,024, and its L = 1,024 ratio wandered from 2.5 to 3.2.
- **Whole-model numbers.** The 76.7, 26.8, 250 and 48 ms figures and the prototype figures come from the findings log entry "A small-query attention kernel removes the 9-row cliff" (2026-11-11). That run used `bench_verify_cost.py --rewind`, three processes per arm. The stock arm is `compiled_kernel`: torch.compile with a static cache, this repo's multi-row Metal linear kernel for 2 to 16 rows (`bench_decode_compile_kernel.py`), stock SDPA, and no `attention_mask`, so on transformers 5.18 it also pays the mask graph break in the companion transformers draft. Both apply equally at T = 8 and T = 9, which is why the issue leads with the difference.
- **#201747.** Found by the reviewer, not by my first searches. Its diff is saved in the lane scratch directory (`review/pr/pr201747.diff`). Running it would need a source build of that stack, which this lane did not do.
- **Before filing:**
  - Link the full repro and the prototype from a public location.
  - Run the inline repro on a nightly, since main removed the MPSGraph path and changed the prefill kernels.
  - Try a GPU trace (Instruments, or `torch.mps.profiler.metal_capture`) to name the kernel. One `metal_capture` attempt on 2026-11-11 wrote no trace.
- **Kept out of the issue text:**
  - The MHA anomaly at 16,384 keys with no mask. T = 1 took 4 to 5 ms with a wide spread, more than T = 4. It reproduced in both runs and is not understood.
  - The memory-clock penalty after a long prefill (findings log, 2026-11-11). It does not affect these steady-state timings.
