# Experiment B: a multi-row linear kernel for 2 to 16 rows

2026-11-10, M1 Max (32 GB, about 400 GB/s), macOS 27.0.1, PyTorch 2.15.0 (a scratch environment; the repository's `.venv` with 2.14.1 is unchanged and was used only to run the pytest file). Every GPU process ran through `bench/gpu_runner.py`. Thread: `docs/research/program/threads/mps-decode.md`, experiment B.

## Result

The kernel passes the gate and makes batched decode of Qwen3-1.7B (fp16) faster at batch 8 and 16. It does not reach the thread's prediction.

| Qwen3-1.7B fp16, decode ms per step (32-token prompt, 32 new tokens, greedy) | stock | patched | speedup |
|---|---:|---:|---:|
| batch 1 (rows = 1, so the patch routes to `F.linear`) | 22.15 / 22.19 | 22.50 / 22.67 | 0.98x |
| batch 4 | 27.19 / 27.87 | 26.29 / 26.86 | 1.03x to 1.04x |
| batch 8 | 36.20 / 36.41 | 27.76 / 28.16 | 1.29x to 1.30x |
| batch 16 | 42.23 / 41.71 | 33.77 / 33.45 | 1.25x |
| the 197 linears alone in model order, batch 16 | 26.4 to 27.2 | 18.0 to 19.6 | 1.4x |

Two runs (run1 / run2), each the median of 3 interleaved repeats, stock and patched in the same process. The thread predicted the 197 linears at batch 16 would fall from 26 ms to 14 ms or less and the full step from 42 ms to 30 ms or less. Measured: 18 to 19.6 ms and 33.5 ms. Both predictions miss. The reason is in the layer table: at 16 rows the kernel is limited by arithmetic issue rather than memory, and Apple's kernel is already fairly good on the 2048 to 6144 projection at exactly 16 rows (125 µs against our 129 µs).

The batch-1 slowdown (about 0.4 ms, 2%) is the Python routing check the patch adds to every linear call; it routes one-row calls back to `F.linear`. Done in C++ inside PyTorch's `Linear.mm` it would cost nothing.

## What was built

- `bench/multirow_linear.metal`: four kernel families computing `y[M,N] = x[M,K] @ W[N,K]^T (+ bias)` for M from 1 to 32, fp16 and bf16, with fp32 accumulation and the weight in `nn.Linear`'s [N, K] layout, unchanged. Each weight row crosses the memory bus once per call.
  - `dev`: each lane streams 16-byte weight chunks along K and applies each to all M rows; x read from device memory (it is cache-resident, at most 256 KB).
  - `tg`: the same, with a K tile of x staged in threadgroup memory.
  - `mma`: weight and x tiles staged in threadgroup memory as fp32, multiplied with simdgroup 8x8 matrix multiply-accumulate, the next weight tile loaded into registers during compute.
  - `mmr`: simdgroup 8x8 multiply-accumulate fed straight from device memory into fragment registers, with no threadgroup memory and no barriers. The k order inside each 32-element block is permuted, identically for W and x, so each lane's share of four fragments is one contiguous 16-byte weight load and two 8-byte x loads. This relies on the fragment register layout MLX's steel GEMM uses; the correctness suite confirms it on this machine.
  - `mma` and `mmr` can split K across threadgroups (P) with a small fixed-order reduce kernel.
- `bench/multirow_linear.py`: the wrapper (`multirow_linear`, `supports`), lazy per-(dtype, M, variant) compilation through `torch.mps.compile_shader`, and the tuned table (`TUNED`) choosing a variant per shape and row count. It refuses K not a multiple of 8, non-contiguous or misaligned inputs, other dtypes and M above 32; callers fall back to `F.linear`.
- `bench/bench_multirow.py`: subcommands `correctness`, `tune`, `layers`, `model`.
- `tests/test_multirow_linear.py`: 62 tests (every family, fp16 and bf16, M of 1, 2, 7, 16, 17, 32, with and without bias, K = 2056, N = 1027, the planted bug, refused inputs). They pass on torch 2.14.1 (the repository's venv) as well.

## Variants tried

Stated before each run where I made a prediction.

1. **Per-lane dot products only (`dev`, `tg`; R of 1, 2, 4 rows per simdgroup, U of 1 or 2 steps, 2, 4 or 8 simdgroups).** Prediction: about 300 GB/s at 16 rows on the 25 MB shapes, `dev` ahead of `tg`. Result (`firstpass-tune-dev-tg-float16.json`): 293 to 306 GB/s at 1 to 4 rows on the 25 MB shapes, but only 156 GB/s at 16 rows on the gate/up projection (161 µs, slower than `F.linear`'s 114 µs in that run). Each x element is reused only across one lane's R weight rows, so at 16 rows the kernel reads x from cache or threadgroup memory about 16/R times per weight byte. `tg` and `dev` traded wins; neither prediction held.
2. **`mma`, staged tiles.** 120 µs (211 GB/s) at 16 rows on gate/up, but capped near 210 GB/s even at 8 rows.
3. **`mmr`, register-fed fragments.** Best at 8 rows (260 to 315 GB/s on 25 to 90 MB shapes). At 16 rows it takes about 126 µs on gate/up whether the weight streams from memory or is a single cache-resident copy, so it is bound by arithmetic issue (393K 8x8x8 multiply-accumulates plus fragment assembly), not bandwidth. At 24 and 32 rows it runs 3 to 4 times slower than `mma` (probably register spills from its 3 or 4 accumulator fragments per weight tile); `mma` takes over there.
4. **Refinements to `mmr` that did not help:** prefetching the next step into registers (F) helps a little at 8 rows only; K steps of 4 or 8 blocks per iteration (U = 4, 8) no gain; half-precision fragment operands for fp16 (H) no consistent gain and slower in a cache-resident check. These three probes printed to the console and were not saved as JSON; the tuning sweep below re-measures F.
5. **Tuning sweep** (`tune-*.json`): 68 configurations per shape and row count (`dev`/`tg` R 1/2/4, U 1/2, S 4/8; `mma` R 1/2, U 1/2, P 1/4; `mmr` R 1/2/4, U 1/2, P 1/2/4, F 0/1), 3 interleaved repeats, every layer shape, all 8 row counts, fp16 and bf16, weights rotated through copies. The fastest per cell went into `TUNED`. Pattern: per-lane families at 1 to 4 rows, `mmr` at 8, `mmr` or `mma` at 12 to 16, `mma` at 24 and 32. Picking the minimum of 68 noisy medians flatters the tuned config inside the sweep; the layer table below re-measures with fresh repeats.

## Correctness

Criterion, per configuration: against a float64 CPU reference computed from the same 16-bit inputs, the largest elementwise error divided by `sum_k |x||w| + |b|` must be no more than `F.linear`'s own value plus `u + K * 2^-24` (u = 2^-11 for fp16, 2^-8 for bf16: one output rounding plus a worst-case fp32 sum of K terms), and the relative L2 error no more than 1.25 times `F.linear`'s plus `sqrt(K) * 2^-24`.

| Set | Cases | Kernel runs | Failures | Largest error ratio, ours / `F.linear` (max-normalised; rel-L2) |
|---|---:|---:|---:|---|
| Random inputs: (K, N) = (2056, 1027), (264, 33), (8, 5) at every M from 1 to 32, and (4104, 4099) at M 1, 2, 7, 16, 17, 32; with and without bias; fp16 and bf16; every variant, plus 2 and 8 simdgroups on one shape | 412 | 18,712 | 0 | 1.005; 1.00002 |
| Real Qwen3-1.7B weights (layer 0 q, k, up, down at M 1 to 32, k with and without a bias; tied head at M 1 to 32 on 8,192 rows including the last), fp16 and bf16, every variant plus the tuned choice | 94 | 3,639 | 0 | 1.003; 1.00004 |

The kernel's error is indistinguishable from `F.linear`'s: both round a nearly exact fp32 sum once. 59% of random-input kernel runs and 10% of real-weight runs produced output bitwise equal to `F.linear`'s. The largest excess over `F.linear` used 0.03% of the allowance.

**Planted bug** (`correctness-planted.json`): four source mutations, one per family, that skip or misalign one K tile. Every one is caught: normalised error 0.017 to 0.046 against an allowance of 0.0006 (F.linear 0.00003). The mutations are applied to the source text by the check and are not in the kernel file.

## Layer table

Median µs per call over 7 interleaved repeats in a shuffled order, weights rotated through at least 4 copies totalling at least 256 MB (2.5 GB for the head), about 3 GB of weight traffic per sample. Arms: `F.linear`, `torch.mm` on a pre-transposed weight, this kernel with its tuned variant, and `F.linear` a second time (A/A, the noise floor). GB/s is weight bytes over time. A/A ratios ranged from 0.94 to 1.15 across all 112 cells, mostly 0.99 to 1.01. `layers.json` has every sample.

**Gate** (thread: at least 1.3x over `F.linear` at M = 16 on most of the Qwen3-1.7B shapes):

| M = 16 | q/o 2048→2048 | k/v 2048→1024 | gate/up 2048→6144 | down 6144→2048 | head 2048→151936 | shapes passing |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 1.48x | 2.08x | 0.97x | 1.26x | 3.44x | 3 of 5 |
| bf16 | 1.43x | 2.06x | 1.29x | 1.69x | 3.32x | 4 of 5 |

Prediction before this run, from the tuning data (so not independent of it): fp16 1.4, 2.0, 1.0, 1.3, 3.4, passing narrowly at 3 of 5. It passed as predicted, narrowly, and the model was wired.

### fp16

| Shape | M | F.linear µs | mm pre-T µs | multirow µs | multirow GB/s | vs F.linear | vs mm | A/A | variant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-1.7B q/o 2048→2048 | 1 | 29.9 | 29.9 | 34.7 | 242 | 0.86x | 0.86x | 0.99 | `tg_r2_u1_s4` |
|  | 2 | 40.1 | 62.0 | 36.5 | 230 | 1.10x | 1.70x | 0.99 | `tg_r1_u2_s8` |
|  | 4 | 54.7 | 67.1 | 38.9 | 216 | 1.41x | 1.73x | 0.94 | `dev_r2_u1_s4` |
|  | 8 | 70.4 | 75.7 | 41.2 | 203 | 1.71x | 1.83x | 0.97 | `mmr_r1_u2_s4_f1` |
|  | 12 | 81.2 | 75.7 | 51.8 | 162 | 1.57x | 1.46x | 1.01 | `tg_r1_u1_s8` |
|  | 16 | 78.4 | 107.2 | 53.1 | 158 | 1.48x | 2.02x | 1.00 | `mmr_r2_u1_s4_p4` |
|  | 24 | 86.9 | 113.9 | 67.5 | 124 | 1.29x | 1.69x | 1.00 | `mma_r1_u1_s4_p4` |
|  | 32 | 85.1 | 111.8 | 81.4 | 103 | 1.05x | 1.37x | 0.99 | `mma_r2_u1_s4_p4` |
| Qwen3-1.7B k/v 2048→1024 | 1 | 18.0 | 17.2 | 22.0 | 191 | 0.82x | 0.78x | 0.97 | `tg_r1_u2_s4` |
|  | 2 | 28.0 | 53.5 | 24.4 | 172 | 1.15x | 2.19x | 1.00 | `dev_r4_u1_s8` |
|  | 4 | 35.7 | 58.1 | 26.2 | 160 | 1.36x | 2.22x | 0.95 | `dev_r1_u1_s8` |
|  | 8 | 38.8 | 65.7 | 35.0 | 120 | 1.11x | 1.88x | 1.07 | `tg_r1_u1_s4` |
|  | 12 | 50.2 | 60.0 | 31.0 | 135 | 1.62x | 1.93x | 0.99 | `tg_r1_u1_s8` |
|  | 16 | 73.4 | 101.9 | 35.3 | 119 | 2.08x | 2.88x | 1.00 | `tg_r1_u1_s8` |
|  | 24 | 75.6 | 103.1 | 46.7 | 90 | 1.62x | 2.21x | 1.00 | `mma_r1_u1_s4_p4` |
|  | 32 | 75.1 | 105.1 | 51.6 | 81 | 1.46x | 2.04x | 1.01 | `mma_r1_u1_s4_p4` |
| Qwen3-1.7B gate/up 2048→6144 | 1 | 76.4 | 76.9 | 81.8 | 308 | 0.93x | 0.94x | 1.01 | `tg_r1_u2_s4` |
|  | 2 | 84.7 | 108.8 | 84.2 | 299 | 1.01x | 1.29x | 0.99 | `tg_r1_u2_s8` |
|  | 4 | 126.2 | 135.1 | 98.5 | 256 | 1.28x | 1.37x | 1.03 | `dev_r2_u1_s4` |
|  | 8 | 160.0 | 141.6 | 89.7 | 281 | 1.78x | 1.58x | 1.00 | `mmr_r1_u2_s4_f1` |
|  | 12 | 126.4 | 148.7 | 113.7 | 221 | 1.11x | 1.31x | 1.02 | `mmr_r2_u1_s4_p2` |
|  | 16 | 125.0 | 147.9 | 129.0 | 195 | 0.97x | 1.15x | 1.00 | `mma_r1_u1_s4` |
|  | 24 | 145.7 | 167.5 | 158.8 | 159 | 0.92x | 1.05x | 0.99 | `mma_r1_u1_s4` |
|  | 32 | 147.7 | 172.8 | 180.1 | 140 | 0.82x | 0.96x | 1.00 | `mma_r1_u1_s4` |
| Qwen3-1.7B down 6144→2048 | 1 | 80.6 | 78.7 | 84.5 | 298 | 0.95x | 0.93x | 1.00 | `dev_r4_u1_s4` |
|  | 2 | 88.1 | 108.4 | 86.9 | 290 | 1.01x | 1.25x | 1.00 | `tg_r4_u1_s4` |
|  | 4 | 123.1 | 112.4 | 89.2 | 282 | 1.38x | 1.26x | 1.02 | `dev_r4_u1_s4` |
|  | 8 | 180.5 | 126.6 | 97.1 | 259 | 1.86x | 1.30x | 0.99 | `mmr_r2_u2_s4_p4` |
|  | 12 | 262.1 | 153.8 | 116.4 | 216 | 2.25x | 1.32x | 1.00 | `mma_r1_u1_s4_p4` |
|  | 16 | 144.5 | 172.7 | 114.3 | 220 | 1.26x | 1.51x | 1.00 | `mmr_r2_u1_s4_p4` |
|  | 24 | 188.4 | 215.8 | 154.3 | 163 | 1.22x | 1.40x | 1.00 | `mma_r1_u1_s4_p4` |
|  | 32 | 234.9 | 276.9 | 193.8 | 130 | 1.21x | 1.43x | 0.99 | `mma_r2_u1_s4_p4` |
| Qwen3-1.7B head 2048→151936 | 1 | 1733.2 | 5670.8 | 1788.5 | 348 | 0.97x | 3.17x | 1.01 | `dev_r4_u1_s8` |
|  | 2 | 1769.4 | 6485.7 | 1767.3 | 352 | 1.00x | 3.67x | 1.00 | `mmr_r1_u1_s4_f1` |
|  | 4 | 2052.3 | 4534.9 | 1774.3 | 351 | 1.16x | 2.56x | 1.00 | `dev_r2_u1_s4` |
|  | 8 | 3865.6 | 14810.4 | 1780.9 | 349 | 2.17x | 8.32x | 1.00 | `mmr_r1_u1_s4` |
|  | 12 | 7182.3 | 6017.1 | 2094.7 | 297 | 3.43x | 2.87x | 1.00 | `mmr_r2_u1_s4` |
|  | 16 | 7249.5 | 6121.3 | 2108.5 | 295 | 3.44x | 2.90x | 1.00 | `mmr_r2_u1_s4` |
|  | 24 | 7939.5 | 6791.3 | 3288.3 | 189 | 2.41x | 2.07x | 1.00 | `mma_r1_u1_s4` |
|  | 32 | 8018.3 | 6877.0 | 3974.8 | 157 | 2.02x | 1.73x | 1.01 | `mma_r1_u1_s4` |
| Qwen3-4B up 2560→9728 | 1 | 153.1 | 154.3 | 160.8 | 310 | 0.95x | 0.96x | 1.00 | `dev_r1_u1_s8` |
|  | 2 | 168.4 | 219.9 | 167.0 | 298 | 1.01x | 1.32x | 1.01 | `dev_r2_u1_s8` |
|  | 4 | 294.7 | 320.4 | 207.7 | 240 | 1.42x | 1.54x | 0.97 | `dev_r2_u1_s4` |
|  | 8 | 330.2 | 241.6 | 164.0 | 304 | 2.01x | 1.47x | 1.00 | `mmr_r1_u2_s4_f1` |
|  | 12 | 193.8 | 213.2 | 201.4 | 247 | 0.96x | 1.06x | 1.00 | `mmr_r2_u1_s4` |
|  | 16 | 202.1 | 221.4 | 214.8 | 232 | 0.94x | 1.03x | 1.00 | `mmr_r2_u1_s4_p2` |
|  | 24 | 285.6 | 254.9 | 325.7 | 153 | 0.88x | 0.78x | 1.00 | `mma_r1_u1_s4` |
|  | 32 | 274.9 | 298.3 | 367.9 | 135 | 0.75x | 0.81x | 1.00 | `mma_r1_u1_s4` |
| Llama-7B up 4096→11008 | 1 | 273.8 | 276.1 | 274.9 | 328 | 1.00x | 1.00x | 0.99 | `dev_r2_u1_s4` |
|  | 2 | 288.9 | 338.9 | 286.8 | 314 | 1.01x | 1.18x | 1.00 | `dev_r2_u2_s8` |
|  | 4 | 374.6 | 410.4 | 300.3 | 300 | 1.25x | 1.37x | 1.00 | `tg_r2_u2_s4` |
|  | 8 | 617.9 | 408.1 | 289.5 | 311 | 2.13x | 1.41x | 0.99 | `mmr_r1_u2_s4_f1` |
|  | 12 | 343.7 | 344.9 | 351.8 | 256 | 0.98x | 0.98x | 1.00 | `mmr_r2_u1_s4_p2` |
|  | 16 | 341.6 | 374.9 | 370.9 | 243 | 0.92x | 1.01x | 0.99 | `mmr_r2_u1_s4` |
|  | 24 | 1081.9 | 470.1 | 543.9 | 166 | 1.99x | 0.86x | 0.99 | `mma_r1_u1_s4_p4` |
|  | 32 | 1214.9 | 517.4 | 636.2 | 142 | 1.91x | 0.81x | 0.99 | `mma_r1_u1_s4` |

<details><summary>bf16</summary>

| Shape | M | F.linear µs | mm pre-T µs | multirow µs | multirow GB/s | vs F.linear | vs mm | A/A | variant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-1.7B q/o 2048→2048 | 1 | 32.0 | 31.0 | 38.1 | 220 | 0.84x | 0.81x | 1.00 | `tg_r4_u1_s8` |
|  | 2 | 56.2 | 67.1 | 40.4 | 208 | 1.39x | 1.66x | 1.00 | `tg_r2_u2_s4` |
|  | 4 | 58.5 | 76.5 | 48.6 | 173 | 1.20x | 1.57x | 1.07 | `dev_r2_u2_s4` |
|  | 8 | 69.9 | 91.5 | 43.8 | 191 | 1.60x | 2.09x | 0.96 | `mmr_r1_u2_s4_f1` |
|  | 12 | 87.9 | 100.3 | 57.8 | 145 | 1.52x | 1.74x | 0.99 | `mma_r1_u1_s4_p4` |
|  | 16 | 89.2 | 122.9 | 62.2 | 135 | 1.43x | 1.98x | 1.02 | `mmr_r2_u1_s4_p4` |
|  | 24 | 96.8 | 120.5 | 74.9 | 112 | 1.29x | 1.61x | 1.00 | `mma_r2_u1_s4_p4` |
|  | 32 | 100.2 | 136.0 | 88.9 | 94 | 1.13x | 1.53x | 1.01 | `mma_r2_u1_s4_p4` |
| Qwen3-1.7B k/v 2048→1024 | 1 | 19.3 | 18.5 | 23.4 | 179 | 0.83x | 0.79x | 0.98 | `dev_r4_u1_s8` |
|  | 2 | 32.1 | 51.5 | 24.2 | 173 | 1.32x | 2.12x | 1.01 | `tg_r1_u2_s4` |
|  | 4 | 36.9 | 55.6 | 25.0 | 168 | 1.48x | 2.23x | 0.95 | `dev_r2_u1_s4` |
|  | 8 | 39.2 | 62.9 | 32.9 | 127 | 1.19x | 1.91x | 1.14 | `mmr_r1_u1_s4_p4_f1` |
|  | 12 | 52.1 | 66.1 | 43.3 | 97 | 1.20x | 1.53x | 1.01 | `tg_r1_u2_s8` |
|  | 16 | 84.5 | 109.7 | 40.9 | 102 | 2.06x | 2.68x | 1.01 | `mma_r1_u1_s4_p4` |
|  | 24 | 89.5 | 125.1 | 50.8 | 83 | 1.76x | 2.46x | 1.01 | `mma_r1_u1_s4_p4` |
|  | 32 | 87.5 | 113.9 | 53.7 | 78 | 1.63x | 2.12x | 0.99 | `mma_r1_u1_s4_p4` |
| Qwen3-1.7B gate/up 2048→6144 | 1 | 78.3 | 78.5 | 82.4 | 306 | 0.95x | 0.95x | 1.00 | `tg_r1_u2_s4` |
|  | 2 | 95.3 | 109.6 | 82.9 | 303 | 1.15x | 1.32x | 1.00 | `dev_r1_u1_s8` |
|  | 4 | 106.3 | 136.9 | 88.7 | 284 | 1.20x | 1.54x | 1.03 | `dev_r2_u1_s4` |
|  | 8 | 160.9 | 155.9 | 89.9 | 280 | 1.79x | 1.73x | 1.01 | `mmr_r1_u2_s4_f1` |
|  | 12 | 155.1 | 183.1 | 121.1 | 208 | 1.28x | 1.51x | 0.99 | `mma_r1_u1_s4` |
|  | 16 | 155.6 | 183.5 | 121.0 | 208 | 1.29x | 1.52x | 1.00 | `mma_r1_u1_s4` |
|  | 24 | 218.7 | 243.2 | 158.8 | 158 | 1.38x | 1.53x | 1.00 | `mma_r1_u1_s4` |
|  | 32 | 241.5 | 266.8 | 185.7 | 136 | 1.30x | 1.44x | 1.00 | `mma_r1_u1_s4` |
| Qwen3-1.7B down 6144→2048 | 1 | 81.3 | 78.3 | 85.4 | 295 | 0.95x | 0.92x | 1.01 | `dev_r1_u2_s4` |
|  | 2 | 119.4 | 108.8 | 86.5 | 291 | 1.38x | 1.26x | 1.00 | `dev_r2_u1_s8` |
|  | 4 | 129.7 | 121.2 | 94.3 | 267 | 1.38x | 1.29x | 1.04 | `dev_r2_u1_s4` |
|  | 8 | 181.5 | 159.5 | 94.0 | 268 | 1.93x | 1.70x | 1.00 | `mmr_r1_u2_s4_f1` |
|  | 12 | 263.0 | 205.5 | 120.1 | 210 | 2.19x | 1.71x | 1.00 | `mma_r1_u1_s4_p4` |
|  | 16 | 203.9 | 230.6 | 120.5 | 209 | 1.69x | 1.91x | 1.01 | `mma_r1_u1_s4_p4` |
|  | 24 | 294.3 | 311.9 | 160.4 | 157 | 1.84x | 1.95x | 1.00 | `mma_r1_u1_s4_p4` |
|  | 32 | 265.8 | 309.0 | 199.4 | 126 | 1.33x | 1.55x | 1.00 | `mma_r2_u1_s4_p4` |
| Qwen3-1.7B head 2048→151936 | 1 | 1796.4 | 5717.9 | 1812.1 | 343 | 0.99x | 3.16x | 0.99 | `tg_r2_u2_s8` |
|  | 2 | 1789.6 | 6829.6 | 1806.5 | 344 | 0.99x | 3.78x | 1.01 | `tg_r1_u1_s4` |
|  | 4 | 2163.5 | 5680.5 | 1815.8 | 343 | 1.19x | 3.13x | 1.00 | `tg_r2_u1_s8` |
|  | 8 | 3793.6 | 14762.2 | 1807.7 | 344 | 2.10x | 8.17x | 1.00 | `mmr_r1_u1_s4_f1` |
|  | 12 | 8030.6 | 6895.3 | 2431.1 | 256 | 3.30x | 2.84x | 1.00 | `mma_r1_u1_s4` |
|  | 16 | 8096.0 | 6975.6 | 2435.1 | 256 | 3.32x | 2.86x | 1.00 | `mma_r1_u1_s4` |
|  | 24 | 9329.2 | 8177.3 | 3328.4 | 187 | 2.80x | 2.46x | 1.00 | `mma_r1_u1_s4` |
|  | 32 | 10497.1 | 9708.6 | 4205.7 | 148 | 2.50x | 2.31x | 1.01 | `mma_r1_u1_s4` |
| Qwen3-4B up 2560→9728 | 1 | 152.5 | 152.4 | 158.4 | 314 | 0.96x | 0.96x | 1.01 | `tg_r1_u2_s4` |
|  | 2 | 194.2 | 253.9 | 172.4 | 289 | 1.13x | 1.47x | 1.15 | `dev_r2_u1_s4` |
|  | 4 | 265.0 | 336.2 | 214.7 | 232 | 1.23x | 1.57x | 0.99 | `dev_r2_u1_s8` |
|  | 8 | 325.7 | 275.4 | 168.8 | 295 | 1.93x | 1.63x | 1.01 | `mmr_r1_u2_s4_f1` |
|  | 12 | 249.0 | 280.2 | 233.5 | 213 | 1.07x | 1.20x | 0.99 | `mma_r1_u1_s4_p4` |
|  | 16 | 295.8 | 335.0 | 251.8 | 198 | 1.17x | 1.33x | 1.00 | `mmr_r2_u1_s4` |
|  | 24 | 357.8 | 576.2 | 315.8 | 158 | 1.13x | 1.82x | 0.99 | `mma_r1_u1_s4_p4` |
|  | 32 | 484.5 | 758.0 | 393.2 | 127 | 1.23x | 1.93x | 1.00 | `mma_r1_u1_s4_p4` |
| Llama-7B up 4096→11008 | 1 | 276.8 | 280.0 | 284.9 | 316 | 0.97x | 0.98x | 1.01 | `tg_r1_u2_s8` |
|  | 2 | 304.1 | 312.4 | 284.4 | 317 | 1.07x | 1.10x | 1.00 | `dev_r1_u1_s8` |
|  | 4 | 344.2 | 367.7 | 290.1 | 311 | 1.19x | 1.27x | 1.00 | `dev_r2_u1_s4` |
|  | 8 | 610.5 | 447.3 | 287.0 | 314 | 2.13x | 1.56x | 1.00 | `mmr_r1_u2_s4_f1` |
|  | 12 | 494.4 | 531.6 | 397.6 | 227 | 1.24x | 1.34x | 0.99 | `mma_r1_u1_s4_p4` |
|  | 16 | 494.5 | 523.4 | 402.5 | 224 | 1.23x | 1.30x | 1.00 | `mma_r1_u1_s4_p4` |
|  | 24 | 1186.4 | 1161.8 | 532.8 | 169 | 2.23x | 2.18x | 1.00 | `mma_r1_u1_s4_p4` |
|  | 32 | 1188.7 | 1218.6 | 648.9 | 139 | 1.83x | 1.88x | 0.99 | `mma_r1_u1_s4` |

</details>

Reading the table:

- At 1 row the kernel is 0.82x to 1.0x of `F.linear`, which in 2.15.0 already routes one row to PyTorch's GEMV. One-row calls are left to `F.linear`.
- 2 to 8 rows is where the kernel wins: 0.99x to 2.2x, ties only at 2 rows on the largest shapes, and 230 to 350 GB/s on shapes of 25 MB and more.
- At 12 to 16 rows the kernel holds 195 to 300 GB/s on large shapes, but Apple's kernel recovers at exactly 12 and 16 rows on the 2048→6144, 2560→9728 and 4096→11008 projections in fp16, where the kernel is 0.92x to 1.11x. In bf16 the kernel wins every cell from 4 rows up (1.07x to 3.32x).
- The head is the largest single gain: `F.linear` at 12 to 32 rows takes 7.2 to 10.5 ms per call against 2.1 to 4.2 ms.
- `F.linear`'s time for some cells changes between processes while the A/A arm within a process stays flat. The gate/up projection at 4 rows measured 98.6, 155.4 and 106.1 µs in three tuning runs and 126.2 µs here. Per-op tables are good for ranking variants; model-order sums predict decode better (next section).

## End to end, Qwen3-1.7B fp16

Each `nn.Linear`'s forward was replaced by a function that sends a flattened row count of 2 to 16 to the kernel (tuned variant) and everything else to `F.linear`. All 197 linears were routed at batch 4, 8 and 16 (6,107 kernel calls per 32-token generation, 197 x 31 decode steps); prefill (32 x batch rows) stays on `F.linear`. Prompts are 32-token windows of `docs/findings.md`. Raw: `model-qwen3-1.7b-fp16-run1.json`, `-run2.json` (run2 adds the spread and near-tie checks below).

Prediction before the model run, summing the layer table over the model's 197 linears: linears at batch 4 from 17.6 to 13.4 ms, batch 8 from 24.0 to 13.8 ms, batch 16 from 26.8 to 17.5 ms; full step at batch 16 from 42 to about 33 ms. Measured in model order: batch 4 from 14.3-14.6 to 13.7-13.9 ms (the layer table overstated `F.linear` at 4 rows, see above), batch 8 from 22.8-23.0 to 14.7-15.1 ms, batch 16 from 26.4-27.2 to 18.0-19.6 ms. Full step at batch 16: 33.5 ms, as predicted.

Outputs against stock:

| Batch | Greedy token agreement | Logit max difference, patched vs stock, teacher-forced on stock's tokens (last step; all 32 steps) | Stock's own spread | Stock run-to-run |
|---:|---:|---|---|---|
| 1 | 100% | 0; 0 (same kernels) | | 0 |
| 4 | 100% | 0.023; 0.043 | row 0 at batch 4 vs batch 1: 0.027; 0.031. Whole batch vs two halves: 0; 0 | 0 |
| 8 | 98.8% | 0.031; 0.078 | row 0 at batch 8 vs batch 1: 0.027; 0.031. Halves: 0; 0 | 0 |
| 16 | 94.7% | 0.031; 0.110 | row 0 at batch 16 vs batch 1: 0.025; 0.039. Batch 16 vs two batches of 8: 0.039; 0.178 | 0 |

Logits reach 29 to 44 in magnitude, where one fp16 step is 0.016 to 0.031, so the differences are one to seven fp16 steps. Stock is bitwise repeatable, so the spread reference is stock computed at a different batch size, which changes its matrix kernel. On the same row, patched against stock batch 1 (0.031 to 0.039 over all steps) matches stock batch B against stock batch 1 (0.031 to 0.039). Over all rows at batch 16, patched (0.110) is inside stock's own whole-against-halves spread (0.178). At batch 4 and 8 stock's halves are bitwise equal to the whole batch, so there the only stock reference is the batch-1 comparison, and patched's all-row maximum (0.078 at batch 8) is about two and a half times it.

Token agreement below 100% is one row each time, free-running greedy diverging at a near-tie and then following a different continuation. At batch 8, row 3 step 6: stock's top two logits are exactly tied. At batch 16, row 7 step 2: stock's top two are 0.016 apart (one fp16 step) and tied under the patch. Stock's own tokens repeat exactly run to run.

## What failed

- The thread's predictions: 197 linears at batch 16 at 14 ms or less (measured 18.0 to 19.6) and a 30 ms step (33.5).
- The gate's spirit on the largest Qwen3-1.7B projections in fp16: gate/up 0.97x and down 1.26x at 16 rows. The gate passed on q/o, k/v and the head.
- 16 rows is arithmetic-bound in `mmr`: about 126 µs on the 2048→6144 shape, against a memory floor of about 75 µs.
- 24 and 32 rows in fp16 on the two Qwen up-projections: 0.75x to 0.92x.
- One row: 0.82x to 1.0x of PyTorch 2.15's GEMV path.
- Batch 1 with the patch: 2% slower from the Python routing check.

## What was not tested

- Speculative-decoding verification, short prefills, and the 17 to 32 row band in a model; only batched greedy decode.
- bf16 end to end; Qwen3-4B and Llama-7B end to end (layer shapes only, synthetic weights).
- Bias timing (bias is covered by correctness only; Qwen3 has none).
- The `F.linear` fallback paths the wrapper takes (K not a multiple of 8, non-contiguous or misaligned inputs) in a model; they are exercised only by refusal tests.
- Row counts between the tuned ones (3, 5 to 7, 9 to 11, 13 to 15) were checked for correctness but not timed; they use the nearest tuned row count's variant.
- Other chips. The tuned table and the fragment layout are M1 Max results.
- Sustained thermal behaviour, and GPU use by other processes outside the lock.

## Files

- `bench/multirow_linear.metal`, `bench/multirow_linear.py`, `bench/bench_multirow.py`, `tests/test_multirow_linear.py`
- `correctness-random.json`, `correctness-real.json`, `correctness-planted.json`
- `firstpass-tune-dev-tg-float16.json` (first sweep, per-lane families only); `tune-float16-{a,b,c}.json`, `tune-bfloat16-{a1,a2,b,c}.json` (the sweep behind `TUNED`)
- `layers.json`
- `model-qwen3-1.7b-fp16-run1.json`, `model-qwen3-1.7b-fp16-run2.json`

Reproduce with `bench/bench_multirow.py` subcommands through `bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10`; model runs need `HF_HOME` pointing at a cache with Qwen/Qwen3-1.7B (revision 70d244cc) and `HF_HUB_OFFLINE=1`.

## Corrections after review (2026-11-10, lead agent)

The skeptical review in `REVIEW.md` reproduced the batch-8 and batch-16 gains and found nothing fatal. It changes how this README should be read in five places:

- **Why the prediction was missed.** Not a hardware limit. At 16 rows this kernel reaches about 3.1 TFLOPS; Apple's kernel does the same 2048→6144 multiply at 32 rows at 5.4 to 5.5 TFLOPS, and the memory floor is about 63 to 72 µs. A 16-row kernel near 75 to 80 µs is plausible, against about 130 µs today. The limit is this kernel's efficiency at 16 rows.
- **The gate, as the thread wrote it** (beat Apple's kernel by 1.3x on the layer sweep): weighted by weight bytes, fp16, the 28 layers' linears gain 1.34x at 4 rows, 1.68x at 8 and 1.27x at 16. At 16 rows the output head supplies 55% of the saving. Batch 8 is where the kernel helps the layers most.
- **Noise.** The A/A arm covered `F.linear` only. Across four processes the kernel's own time varied (gate/up at 16 rows, 129 to 157 µs), and the down projection at 16 rows lands on both sides of 1.3x.
- **Batch 4 and batch 1.** No clear step gain at batch 4, and no established slowdown at batch 1 (0.998x against `nn.Linear`'s own forward).
- **Batch-8 logits.** Stock's own difference between batch sizes (up to 0.245 per row) is larger than the patch's difference from stock (0.078). The earlier caveat resolves in the kernel's favour.
