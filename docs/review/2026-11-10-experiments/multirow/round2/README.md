# Experiment B, round 2: making 9 to 16 rows fast on the large MLP shapes

2026-11-10, M1 Max (32 GPU cores, about 400 GB/s), PyTorch 2.15.0 (scratch environment), every GPU process through `bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10`. Round 1 is `../README.md` and its review `../REVIEW.md`.

## Result

A new register-fed kernel family, `mmh`, takes 16 rows of Qwen3-1.7B's gate/up and down projections from round 1's 114 to 157 µs to about 85 to 95 µs in a quiet process, roughly 4.2 to 4.7 TFLOPS. The best samples are 82 to 95 µs. The same kernel is also the fastest choice at 2 to 8 rows on most shapes. End to end, Qwen3-1.7B fp16 decode at batch 16 drops from 40.7–42.3 ms to 27.7–28.5 ms per step (1.48x), and batch 12 gains 1.50x.

| Qwen3-1.7B fp16, ms per decode step (32-token prompt, 32 new tokens, greedy; two processes, each 4 interleaved repeats; range over all 8 samples per arm) | stock (`nn.Linear`) | patched | median speedup |
|---|---:|---:|---:|
| batch 1 (one row: the patch routes to `F.linear`) | 21.0–22.2 | 21.5–22.4 | 0.98x to 1.00x |
| batch 4 | 26.5–27.5 | 25.3–26.7 | 1.03x to 1.04x |
| batch 8 | 35.4–36.5 | 26.2–28.0 | 1.33x to 1.35x |
| batch 12 | 41.6–42.9 | 27.4–28.8 | 1.50x to 1.51x |
| batch 16 | 40.7–42.3 | 27.7–28.5 | 1.47x to 1.48x |
| the 197 linears alone in model order, batch 16 | 26.1–27.5 | 14.1–14.3 (run 1), 16.2 (run 2) | 1.69x to 1.86x |

**The prediction check, plainly:**

- **The batch-16 linears at 15 ms or less: did not hold reliably.** One process measured 14.1 and 14.3 ms, the other 16.2 and 16.2 ms. Both samples of each process agree with each other, so this is a difference between processes, not noise within one. With one process on each side of the line, I cannot claim it.
- **The batch-16 step at 30 ms or less: held.** 27.7 to 28.5 ms in all eight patched samples across both processes. Round 1 measured 33.5 ms.

**What sets the remaining limit.** At 16 rows, `mmh` no longer waits on arithmetic. Its multiplies plus fragment building take about 75 µs with every load cache-resident. Its weight stream alone, with no x loads and no multiplies, takes 89 to 90 µs. That is the same rate as round 1's one-row `dev` and `tg` kernels in the same process (about 280 GB/s), while `F.linear`'s one-row path reached 82 to 85 µs there. Getting from about 90 µs to the 75 to 85 µs the task asked for now means streaming the weight faster, not multiplying faster. Section 4 has the probe.

**Measurement caveat.** Another workload used the GPU during this round's layer runs: 74 to 80% GPU utilization appeared between my processes while I waited for the lock, and within a repeat every arm slowed together. Layer-table medians for the 25 MB shapes are therefore noisy. For example, gate/up at 16 rows, fp16, has kernel medians of 118 to 131 µs across four processes, but a fastest sample of 95 µs, and 87 to 103 µs in the tuning and probe processes. Speedups are interleaved ratios and hold up better than absolute times. The tables report both the median range and the fastest sample. The noise arms (A/A and B/B) decide fallbacks.

## What changed in the code

- `bench/multirow_linear.metal`: three new families for 9 to 16 rows. They also turned out best at 2 to 8 rows.
  - `mmh`: register-fed 8x8 fragments in a new k order, `k = 8p + 2b + e` inside each 32-element block. A lane's two elements of a weight fragment become one 32-bit word of the 16-byte chunk it loads: no instruction for fp16 (half operands, fp32 accumulators) and two integer operations for bf16. x costs one select per element. The main loop has no bounds checks; one partial chunk per simdgroup is checked. S simdgroups of a threadgroup share the same weight rows and split K by interleaved chunks, then add their 8x8 tiles in threadgroup memory. Options: U (32-element blocks per step), F (register prefetch), P (K split across threadgroups, with round 1's reduce kernel), H (half operands, fp16 only).
  - `mmp`: `mmh` with x first repacked into fragment order by a small kernel (`mrl_xpack`), so x fragments need no instruction either.
  - `mmt`: the weight staged through threadgroup memory as raw 16-bit, double-buffered, one barrier per K tile; x fragments from device memory.
  - Every helper of the three is force-inlined and every constant loop unrolled.
- `bench/multirow_linear.py`: the new families, `TUNED` rebuilt for rows 2 to 16 on all seven shapes, `"linear"` entries that route a cell to `F.linear` (`pick_config` returns `None` and `multirow_linear` calls `F.linear`). A row count not in the table now takes the nearest tuned row count with the same number of 8-row tiles, so a 16-row choice is never reused at 17 rows. `default_config` uses `mmh` at 9 to 16 rows, replacing `mma` with P = 4, which the review found slower than `F.linear` there.
- `bench/bench_multirow.py`: `tune --grid round2` (56 configs fp16, 40 bf16: every config round 1 chose for 2 to 16 rows, plus the new families), `layers` with two noise arms (`F.linear` against itself, the kernel against itself), any row list, and TFLOPS. Also `probe` (the diagnosis edits) and `decode` (the end-to-end run below). `correctness` covers the new families and every tuned config, with seven new planted bugs.
- `tests/test_multirow_linear.py`: the four new families in the parametrised suite, two planted bugs in `mmh` (one gross, one dropping the last 8 of 2056 K elements), and the `"linear"` routing. 113 tests pass on the repository's `.venv` (torch 2.14.1).

## 1. Diagnosis: why round 1 ran at about 3.1 TFLOPS at 16 rows

The diagnosis rests on two probe runs of round 1's kernel, made before any new kernel existed. One caveat on timing: this file was first written after the first design sweeps (section 2) had run. The predictions at the end of this section are the ones I worked from before timing each design, but they were not saved to a file beforehand, so treat them as reconstructed rather than pre-registered. The probes run round 1's kernel with source edits that remove one cost at a time (`bench/bench_multirow.py probe`; raw data in `probe-a.json` and `probe-b.json`).

### What the hardware allows

`peak_mma.py` (raw: `peak-mma.json`) runs nothing but 8x8x8 multiply-accumulates on register-resident operands, with long launches (150 to 280 ms each, so the clock has ramped):

| operands, accumulator | 2 accumulators | 4 | 8 |
|---|---:|---:|---:|
| fp32 × fp32 → fp32 | 5.9 to 6.4 TFLOPS | 7.4 to 8.0 | 7.1 to 7.9 |
| fp16 × fp16 → fp32 | 6.2 to 6.5 | 7.5 to 8.0 | 7.9 to 8.9 |

So the 393,216 multiplies of a 16-row pass over gate/up (2048→6144) cost at least 45 µs with half operands and about 51 µs with fp32 operands, if nothing else were issued. The memory floor is 63 µs at 400 GB/s. The fastest any of our kernels has streamed a 25 MB weight is about 300 to 310 GB/s, which is 82 to 84 µs. `F.linear` at one row reaches 76 to 85 µs depending on the process. (`probe-a.json` also holds a first, short-launch peak measurement. Its launches were 1 to 10 ms, too short for the clock to ramp, so `peak-mma.json` supersedes it.)

### Where round 1's 16-row time goes

Round 1's register-fed variant `mmr_r2_u1_s4` at 16 rows covers 16 weight rows × 16 input rows per simdgroup: two weight tiles × two row tiles, four 8x8 accumulators. Each weight fragment feeds two multiplies and each x fragment two. Per simdgroup and per 32-element K block, it issues 2 weight loads (16 bytes per lane) and 4 x loads (8 bytes), each with a bounds compare and select. It also issues 32 conversions from 16-bit to fp32 (16 for the weight and 16 for x), 32 fragment element writes and 16 multiplies.

`probe-b.json`, gate/up 2048→6144, fp16, 16 rows, 3 interleaved repeats:

| variant of `mmr_r2_u1_s4` | µs | what it measures |
|---|---:|---|
| as built | 127 | |
| loads confined to a cache-resident 64-element window (l1) | 119 | arithmetic plus issue, no memory traffic |
| each 8x8x8 multiply replaced by one FMA (nomma) | 106 | memory path plus fragment building |
| both (l1_nomma) | 54 | fragment building, loads and loop alone |
| nomma, with x alone kept cache-resident | 105 | whether x re-reads matter |
| P = 4 (four times the simdgroups): as built / l1 / nomma / l1_nomma | 127 / 120 / 109 / 58 | whether occupancy matters |

What this says, cause by cause:

- **Instruction mix and fragment assembly.** Everything except the multiplies (l1_nomma) costs 54 µs, about as much as the multiplies themselves at peak (45 to 51 µs). Both run on the same issue slots, so they add: 54 + 51 to 65 ≈ 119 µs, which is what l1 measures. This is what round 1 called arithmetic-bound. About half of it is not arithmetic but feeding the matrix unit: conversions, element moves, checked loads.
- **Occupancy.** At P = 1 gate/up runs 384 simdgroups, three per scheduler. Quadrupling them with a K split changes nothing (127 against 127, l1_nomma 54 against 58), so the kernel is not latency-starved. It is limited by issue throughput.
- **Memory path.** Even with no multiplies, the round-1 load pattern streams only about 240 GB/s (nomma, 106 µs). Re-reading x from cache is not the reason: keeping x resident leaves nomma at 105 µs.
- **Register pressure.** Doubling the weight tiles per simdgroup (R = 4, eight accumulators) makes it 2.6 times slower (328 µs), and forcing full unrolling does not help (325). At 24 and 32 rows (three and four row tiles), `mmr` runs at 615 µs against `mma`'s 186. The pattern fits a register budget being exceeded and spilling, not a loop-structure problem. (Learned later: the round-2 kernel at R = 2 already reports a pipeline limit of 448 threads per threadgroup, Metal's error when 512 were requested, a sign that the compiler allocates many registers per thread even there.)
- **Threadgroup memory traffic (`mma`).** The staged variant converts the weight to fp32 in threadgroup memory, which doubles the bytes. Its simdgroups each own 8 weight rows, so per 8-wide K step it loads 3 fragments from threadgroup memory for 2 multiplies, plus the staging writes. It lands on the same plateau (122 µs), by a different route.
- **Rows covered by a simdgroup's tiles.** All 16 input rows, in both `mmr` (2 row tiles) and `mma`. The row coverage was not the problem. Reuse was already 2 multiplies per weight fragment. The cost per fragment was the problem.

In one sentence: at 16 rows, round 1 spends as many issue cycles building fragments as it spends multiplying, and the multiplies alone already take 45 to 51 µs. The overhead doubling that pushes it to about 125 µs, roughly 3.1 TFLOPS.

### Predictions, stated before each design was timed

1. **`mmh`: one-word weight fragments, half operands, unchecked main loop, K split inside the threadgroup.** Changing the k order inside each 32-element block from round 1's `k = 8p + 4e + b` to `k = 8p + 2b + e` makes a lane's two elements of weight fragment b one 32-bit word of the 16-byte chunk it already loads. For fp16 the fragment is then that word reinterpreted (no instruction). For bf16 it is two integer operations. x costs one select per element. Bounds checks move to a single tail chunk. Several simdgroups share the same weight rows and split K, adding their tiles in threadgroup memory at the end, which raises parallelism without a second kernel. Prediction: overhead from about 54 to about 25 µs, so issue ≈ 45 + 25 ≈ 70 µs, below the memory path. The kernel then becomes stream-bound at about 280 to 300 GB/s: **80 to 90 µs on gate/up and down at 16 rows**, and about the same at 9 to 15 rows (the same number of multiplies, since 9 rows still need two row tiles).
2. **`mmp`: `mmh` with x repacked into fragment order by a small extra kernel.** Removes the x selects (a few µs of issue) but adds a dispatch of about 5 µs. Prediction: **no better than `mmh`**.
3. **`mmt`: weight staged through threadgroup memory, 16-bit, double-buffered, one barrier per tile, x fragments from device memory.** Adds a threadgroup write and read of every weight byte and a barrier per tile, for better-coalesced device reads. Prediction: **slower than `mmh`, about 100 to 110 µs**.
4. **Larger K step (U = 2) and register prefetch (F = 1)** on `mmh`. If `mmh` is stream-bound, more bytes in flight could help a little. Prediction: **within ±3%**.
5. **Four weight tiles per simdgroup (R = 4)** on `mmh`. It halves x loads per multiply, but if round 1's cliff was register spills, it stays. Prediction: **slower**.

## 2. Designs tried

Raw data: `design-sweep-1.json` (before the inlining fix), `design-sweep-2.json` and `design-sweep-3.json` (after), fp16, 3 interleaved repeats, weights rotated through 11 copies (277 MB).

| Design | 16 rows, gate/up µs | 16 rows, down µs | 9 rows, gate/up / down µs | Prediction | Outcome |
|---|---:|---:|---:|---|---|
| round 1 `mmr_r2_u1_s4`, `mma_r1_u1_s4` (same processes) | 118, 117 | 191, 203 | | | |
| `F.linear` (same processes) | 118 to 132 | 152 to 158 | 190 / 205 | | |
| **`mmh`, R = 1 or 2, U = 1, S = 4 or 8** | **87 to 93** | **92 to 95** | **90 / 91** | 80 to 90 | held, at the top of the range |
| `mmp` (x repacked first) | 94 to 98 | 99 to 101 | | no better than `mmh` | held: 5 to 8 µs slower (the extra dispatch) |
| `mmt` (weight staged, double-buffered) | 121 to 128 | 167 to 265 | | 100 to 110 | worse than predicted |
| `mmh` with U = 2 | 96 to 99 | 96 to 99 | | within ±3% | slightly worse (+5 to 10%) |
| `mmh` with prefetch (F = 1) | 92 to 96 | 92 to 94 | | within ±3% | held: no gain on average |
| `mmh` with R = 3, 4 or 8 | 245 to 424 | 250 to 480 | 245 / 278 | slower | held: the cliff stays |
| half against fp32 fragment operands (H) | within 2 to 3% either way | | | | no consistent difference |

Every design was correct (smoke check first, then the full suite in section 5), so all are kept and the tuner chooses between them. `mmt` and `mmp` are each picked in only a few cells. `mmp` in `TUNED`: q/o at 12 rows bf16, down at 12 rows fp16, and the head at 11 to 16 rows bf16 and 16 rows fp16.

**The R = 3 and above cliff.** With more than 4 accumulator tiles per simdgroup (three or more weight tiles × two row tiles), `mmh` runs 2.7 to 5 times slower whatever the other settings. Round 1's `mmr` hits the same wall at R = 4, and at 24 or 32 rows. Forcing inlining and unrolling did not move it. The peak-rate probe runs 8 accumulators at full speed when the operands are constants, so the limit is register-assembled fragments plus loads at that tile count, almost certainly spills. I did not find a way past it. More weight rows per x fragment, which would halve x loads, is therefore off the table on this compiler.

## 3. Hypotheses checked against the references

- **Apple** ("Discover Metal enhancements for A14 Bionic", Apple tech talk 10858, 2020; the only Apple material on `simdgroup_matrix` found) introduces 8x8 `simdgroup_multiply_accumulate`. Its one example stages both tiles in threadgroup memory and loads fragments with `simdgroup_load`, with one accumulator per simdgroup. It gives no tuning guidance. I took the API from it, and the decision to feed fragments from registers was mine (round 1).
- **MLX steel GEMM** (mlx 0.32.2 headers, read from a local install: `steel/gemm/mma.h`, `BaseMMAFrag::get_coord` and `BlockMMA`). It confirmed the fragment layout `mmh` relies on: lane l holds row `(l/4 & 4) + (l/2) % 4`, columns `(l/4 & 2) * 2 + (l % 2) * 2` and the next one. MLX stages both operands in threadgroup memory in their native 16-bit type and converts to fp32 fragments on load. Each simdgroup owns TM × TN fragments of a BM × BN block, so M is padded up to the block. The block sizes are chosen in MLX's C++ dispatch, which the installed package does not include.
- **MLX `gemv_wide`** (same install, `gemv.h`; the kernel MLX issue 3967 is about). Each threadgroup streams a block of weight rows once and applies every loaded 4-element weight chunk to `vecs_per_tg` input rows. `k_lanes` lanes reduce K for one weight row with shuffles, and the loads are unrolled 8 deep to keep them in flight. There are no simdgroup matrices: it is round 1's `dev` design, with a deeper unroll. Row counts above `vecs_per_tg` are handled by re-walking the same weight rows from cache.
- **MLX issue 3967** (read on GitHub). MLX enables `gemv_wide` only on Apple GPU generation 15 (M3) and later. As the issue summarises the code's rationale, pre-M3 parts are limited by load issue rate rather than bandwidth. The reporter measured 1.10 to 2.94 times over steel GEMM on an M1 Pro at M = 2 to 8 (f16, K = 2048). The maintainers closed it as won't-fix to avoid shape- and machine-specific heuristics.
- **What I took from them.** The MLX rationale agrees with the round-1 diagnosis: on M1 the cost is issuing loads and the instructions around them, not bandwidth and not multiplies. `gemv_wide`'s per-lane dot products are the family round 1 already had (`dev`, `tg`), and round 1 showed they run out of x-read bandwidth past about 8 rows. So for 9 to 16 rows the answer had to stay on the matrix unit and cut the instructions that feed it, which is what `mmh` does. `gemv_wide`'s 8-deep load unroll is the idea behind trying U = 2 and F = 1 here; neither helped `mmh`.

## 4. Second probe: what limits `mmh`

`probe-c.json` and `probe-d.json`, the same edits applied to `mmh` (fp16, 3 to 5 interleaved repeats):

| 16 rows, gate/up / down µs | `mmh_r2_u1_s4` | `mmh_r2_u1_s8` | round 1 `mmr_r2_u1_s4` |
|---|---:|---:|---:|
| as built | 93 / 100 | 94 / 96 | 125 / 206 |
| loads cache-resident (l1) | 74 / 76 | 75 / 75 | 122 / 170 |
| no multiplies (nomma) | 92 / 97 | 92 / 95 | 107 / 152 |
| neither (l1_nomma): fragment building, loads, loop | 37 / 35 | 35 / 31 | 57 / 103 |

In `probe-d.json`, `mmh_r1_u1_s8_h1` at 16 rows takes 94 / 101 µs as built, 93 / 96 µs as a weight stream only (no x loads, no multiplies) and 44 / 40 µs for x loads only. At one row in the same process, the weight-stream-only edit of `mmh` takes 89 to 91 µs on both shapes, round 1's `dev_r2_u1_s4` 89 to 90, `tg_r1_u2_s4` 90, and `F.linear` 82 to 85.

- The non-multiply overhead fell from 54 to 57 µs to 31 to 37 µs, as predicted.
- With loads cache-resident, the kernel now takes about 75 µs, so issue no longer limits it.
- With multiplies removed it still takes 92 to 98 µs, the same as as built: the kernel is now limited by how fast its loads stream the weight.
- That streaming rate (about 280 GB/s here) is what every custom kernel in this repository reaches on a 25 MB weight in this process. `F.linear`'s one-row path is 6 to 10% faster.
- The half-operand `nomma` edit is pathological (460 to 940 µs) at two row tiles, so it is left out of the table. It replaces the multiply with a half-precision FMA on extracted elements, not something the real kernel does.

## 5. Correctness

The criterion is round 1's. Against a float64 CPU reference from the same 16-bit inputs, each config's largest error divided by `sum_k |x||w| + |b|` must not exceed `F.linear`'s by more than `u + K·2^-24`. Its relative L2 error must not exceed 1.25 times `F.linear`'s plus `sqrt(K)·2^-24`.

| Set | Cases | Kernel runs | Failures | Largest ratio to `F.linear` (max-normalised; rel-L2) |
|---|---:|---:|---:|---|
| Random (`correctness-random.json`): (K, N) = (2056, 1027), (264, 33), (8, 5) at every M from 1 to 32, (4104, 4099) at M 1, 2, 7, 16, 17, 32; with and without bias; fp16 and bf16; every round-1 variant, 15 round-2 variants covering every code path, and every config in the final `TUNED` (35) | 412 | 33,808 | 0 | 1.005; 1.00005 (all runs together) |
| Real Qwen3-1.7B weights (`correctness-real.json`): layer 0 q, k, up, down at M 1 to 32 (k with and without a bias), the tied head at M 1 to 32 on 8,192 rows, every variant plus the tuned choice | 94 | 6,574 | 0 | |
| Planted bugs (`correctness-planted.json`): round 1's four, plus seven on the new families, each at 8 and 16 rows | 18 | | all caught | |

- Every one of the 35 tuned configs ran in fp16 and bf16, with and without bias, on random inputs. The tuned choice per cell also ran on the real weights.
- Each round-2 family on its own (`mmh`, `mmp`, `mmt`, half or fp32 operands) matches `F.linear`'s error: median and maximum ratio 1.00 on both arms. Half operands cost nothing in precision, because a product of two halves is exact in the fp32 accumulator. 51% of all kernel runs were bitwise equal to `F.linear`.
- **Planted bugs on the new families.** The gross ones each land at 0.035 to 0.20 normalised error against an allowance of 0.0006: skip one K chunk in `mmh`, drop one simdgroup's partial sum in the threadgroup reduction, swap the x halves in the half-operand fragment, misalign the packed x in `mmp`. Skipping the first `mmt` tile gives NaN. Two subtle ones are also caught. Dropping the last partial chunk (8 of 2056 K elements) lands at 0.0075 and 0.0135, 12 and 22 times the allowance. Dropping the last 8 elements of every packed x row lands at 0.009.
- The pytest file (`tests/test_multirow_linear.py`, 113 tests) passes on the repository's `.venv` (torch 2.14.1). The scratch 2.15.0 environment has no pytest, and this round downloads nothing. The 2.15.0 evidence is the three JSON runs above.

## 6. Layer table

Median µs per call over 7 interleaved repeats in shuffled order, weights rotated through at least 256 MB of copies (2.5 GB for the head), about 3 GB of weight traffic per sample. Arms: `F.linear`, `F.linear` again (A/A), the kernel with its tuned variant, the kernel again (B/B). GB/s is weight bytes over time. TFLOPS is 2·M·N·K over time.

- **Processes.** The four per-layer shapes (q/o, k/v, gate/up, down) ran in four separate processes per dtype (`layers-run{1..4}-{dtype}-a.json`). The head, Qwen3-4B up and Llama-7B up ran in two (`layers-run{1,2}-{dtype}-b.json`).
- **Ranges.** Each cell gives the range over processes. "Fastest sample" is the fastest single sample of either arm of that kind, over all processes, the least affected by the outside load described under Result.
- **Variant column.** The config the round-2 sweep chose. Where the routed column says F.linear, the table now sends that cell to `F.linear`.

**Headline: the MLP projections at 8 to 16 rows, fp16.** gate/up is 2048→6144 (25.2 MB), down is 6144→2048. The memory floor is 63 µs at 400 GB/s.

| Rows | gate/up: `F.linear` µs | gate/up: kernel µs | fastest sample, `F.linear` / kernel | TFLOPS at fastest | speedup | down: `F.linear` µs | down: kernel µs | fastest sample | TFLOPS at fastest | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 163–283 | 87–110 | 159 / 82 | 2.4 | 1.88x–2.57x | 210–294 | 92–120 | 177 / 86 | 2.3 | 1.89x–2.45x |
| 9 | 178–242 | 85–124 | 176 / 83 | 2.7 | 1.96x–2.49x | 208–269 | 92–127 | 203 / 88 | 2.6 | 1.98x–2.28x |
| 10 | 135–202 | 96–126 | 122 / 91 | 2.8 | 1.41x–1.60x | 233–263 | 95–120 | 224 / 87 | 2.9 | 2.17x–2.44x |
| 11 | 159–193 | 108–160 | 133 / 90 | 3.1 | 1.21x–1.50x (routed to `F.linear`, see 7) | 243–283 | 90–135 | 240 / 86 | 3.2 | 2.08x–2.70x |
| 12 | 122–198 | 88–143 | 116 / 84 | 3.6 | 1.28x–1.58x | 283–294 | 106–128 | 263 / 93 | 3.2 | 2.30x–2.67x |
| 13 | 115–193 | 84–120 | 114 / 83 | 3.9 | 1.38x–1.60x | 298–305 | 96–115 | 279 / 86 | 3.8 | 2.64x–3.17x |
| 14 | 154–166 | 98–115 | 122 / 87 | 4.1 | 1.44x–1.57x | 304–336 | 88–112 | 296 / 86 | 4.1 | 2.92x–3.68x |
| 15 | 172–185 | 109–132 | 115 / 85 | 4.4 | 1.39x–1.62x | 318–354 | 89–123 | 317 / 86 | 4.4 | 2.82x–3.58x |
| 16 | 140–195 | 118–131 | 119 / 95 | 4.3 | 1.11x–1.49x | 147–208 | 87–118 | 145 / 85 | 4.7 | 1.46x–1.90x |

In bf16 the same cells run 1.41x to 3.24x over `F.linear`, with fastest kernel samples of 85 to 107 µs (full table below).

**Against round 1, fastest kernel sample per cell** (round 1: `../layers.json` and the review's three reruns; round 2: all runs here). At 9 to 16 rows on every shape the ratio is 0.59 to 0.87: gate/up at 16 rows 124.5 → 94.6 µs, down 113.3 → 85.1, q/o 51.3 → 36.6, k/v 34.4 → 23.5, head 2099 → 1741, Qwen3-4B up 212 → 167, Llama-7B up 361 → 279. At 2 to 8 rows nothing got slower beyond the noise. 37 of the 42 cells at 2, 4 and 8 rows are equal or faster (0.74 to 1.00). The five above 1.00 are 1.006 to 1.026: the three cells now routed to `F.linear` at 2 rows (gate/up, down and head in fp16), bf16 gate/up at 4 rows and bf16 Qwen3-4B up at 8 rows. Each is inside its cell's A/A and B/B spread.

**At 16 rows on all five Qwen3-1.7B shapes** (the thread's gate, beat `F.linear` by 1.3x; range over processes):

| 16 rows | q/o | k/v | gate/up | down | head |
|---|---:|---:|---:|---:|---:|
| fp16 | 1.93x–2.09x | 2.84x–2.98x | 1.11x–1.49x | 1.46x–1.90x | 3.93x–4.27x |
| bf16 | 2.09x–2.26x | 2.85x–3.19x | 1.58x–1.75x | 1.93x–2.29x | 4.62x–4.81x |

The per-layer gate/up at 16 rows in fp16 is the one cell whose range reaches below 1.3x (one of four processes at 1.11x). Its fastest samples are 119 µs for `F.linear` and 95 µs for the kernel (1.25x), and the tuning and probe processes measured 87 to 103 µs. In bf16 every shape passes in every process.

**Model-order sum of the 197 linears from the layer table** (28 × (2 q/o + 2 k/v + 2 gate/up + 1 down) + head; the two processes that ran every shape): fp16 at 16 rows 15.99 to 16.38 ms against 31.9 to 32.0 ms for `F.linear`. At 8 rows 13.44 to 13.58 against 25.2 to 27.5. At 12 rows 14.17 to 14.55 against 30.2 to 31.4. The decode run's own model-order timing (section 8) is the better predictor, because its `F.linear` times are not inflated by the outside load.

Full tables, both dtypes, every row count 1 to 16 plus 24 and 32 (raw samples in `layers-run*.json`, summary in `summary.json`, generated by `summarize.py`):

<details><summary>bfloat16</summary>

### bfloat16

| Shape | M | F.linear µs | kernel µs | fastest sample, F.linear / kernel µs | kernel GB/s | kernel TFLOPS | speedup | A/A | B/B | variant | routed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| q/o 2048→2048 | 1 | 30.4–32.0 | 36.1–38.1 | 29.7 / 35.2 | 220–233 | 0.22–0.23 | 0.83x–0.85x | 1.00–1.01 | 0.98–1.01 | `tg_r4_u1_s8` | F.linear (1 row) |
|  | 2 | 50.1–60.8 | 38.3–40.2 | 48.6 / 35.2 | 209–219 | 0.42–0.44 | 1.25x–1.58x | 0.88–1.17 | 0.91–1.00 | `dev_r2_u1_s8` | kernel |
|  | 3 | 49.6–60.6 | 35.9–42.6 | 48.4 / 34.8 | 197–233 | 0.59–0.70 | 1.36x–1.42x | 0.99–1.02 | 0.99–1.02 | `dev_r1_u1_s8` | kernel |
|  | 4 | 47.4–65.7 | 37.0–44.4 | 46.8 / 35.6 | 189–227 | 0.76–0.91 | 1.28x–1.48x | 0.96–1.04 | 0.98–1.12 | `mmh_r2_u1_s4` | kernel |
|  | 5 | 47.8–66.0 | 36.7–45.8 | 46.5 / 35.5 | 183–229 | 0.92–1.14 | 1.28x–1.44x | 0.99–1.03 | 0.97–1.00 | `mmh_r2_u2_s4` | kernel |
|  | 6 | 51.6–70.4 | 36.2–40.6 | 49.9 / 35.5 | 207–232 | 1.24–1.39 | 1.38x–1.73x | 0.89–1.02 | 0.96–1.02 | `mmh_r1_u1_s8` | kernel |
|  | 7 | 54.6–73.7 | 36.8–43.1 | 53.1 / 35.6 | 195–228 | 1.36–1.60 | 1.48x–1.71x | 0.98–1.02 | 1.00–1.02 | `mmh_r1_u1_s4` | kernel |
|  | 8 | 60.6–73.2 | 36.2–38.9 | 59.1 / 35.4 | 216–232 | 1.73–1.85 | 1.63x–1.88x | 1.00–1.21 | 0.99–1.03 | `mmh_r2_u1_s4_f1` | kernel |
|  | 9 | 66.3–79.6 | 39.7–49.0 | 65.2 / 38.3 | 171–212 | 1.54–1.90 | 1.62x–1.71x | 0.98–1.10 | 1.00–1.03 | `mmh_r2_u1_s8_f1` | kernel |
|  | 10 | 72.8–85.3 | 41.7–48.9 | 71.2 / 40.5 | 171–201 | 1.71–2.01 | 1.66x–1.74x | 0.92–1.02 | 0.99–1.00 | `mmh_r2_u2_s4` | kernel |
|  | 11 | 78.1–88.7 | 38.6–46.5 | 76.7 / 37.7 | 180–217 | 1.98–2.39 | 1.89x–2.03x | 0.98–1.02 | 0.99–1.01 | `mmh_r2_u1_s4` | kernel |
|  | 12 | 85.5–88.6 | 44.4–47.9 | 84.3 / 43.0 | 175–189 | 2.10–2.27 | 1.85x–1.93x | 1.00 | 1.00–1.01 | `mmp_r2_u1_s4` | kernel |
|  | 13 | 91.5–94.5 | 39.2–42.1 | 90.8 / 38.4 | 199–214 | 2.59–2.78 | 2.23x–2.34x | 0.98–1.01 | 0.98–1.03 | `mmh_r2_u1_s4_f1` | kernel |
|  | 14 | 97.6–100.5 | 40.2–43.0 | 96.4 / 39.0 | 195–208 | 2.73–2.92 | 2.34x–2.43x | 1.00–1.07 | 0.98–1.10 | `mmh_r2_u1_s8` | kernel |
|  | 15 | 103.9–111.8 | 39.7–46.6 | 103.0 / 38.6 | 180–211 | 2.70–3.17 | 2.40x–2.61x | 0.99–1.03 | 0.95–1.01 | `mmh_r1_u1_s4` | kernel |
|  | 16 | 89.4–99.0 | 39.9–45.9 | 87.8 / 38.5 | 183–210 | 2.92–3.37 | 2.09x–2.26x | 0.98–1.01 | 1.00–1.01 | `mmh_r2_u1_s4_f1` | kernel |
|  | 24 | 97.0–107.7 | 74.7–84.6 | 95.0 / 73.7 | 99–112 | 2.38–2.69 | 1.27x–1.30x | 1.00 | 0.99–1.01 | `mma_r2_u1_s4_p4` | kernel |
|  | 32 | 98.0–108.9 | 85.9–94.5 | 95.6 / 82.8 | 89–98 | 2.84–3.12 | 1.11x–1.15x | 0.99–1.01 | 0.97–1.05 | `mma_r2_u1_s4_p4` | kernel |
| k/v 2048→1024 | 1 | 19.2–20.2 | 23.1–24.9 | 17.7 / 22.3 | 168–181 | 0.17–0.18 | 0.77x–0.85x | 0.99–1.03 | 0.99–1.05 | `dev_r4_u1_s8` | F.linear (1 row) |
|  | 2 | 31.9–36.6 | 23.7–26.1 | 30.6 / 22.1 | 161–177 | 0.32–0.35 | 1.34x–1.40x | 0.98–1.02 | 0.99–1.02 | `dev_r1_u1_s8` | kernel |
|  | 3 | 37.2–43.3 | 23.7–27.0 | 35.8 / 22.6 | 156–177 | 0.47–0.53 | 1.55x–1.62x | 0.99–1.03 | 0.99–1.01 | `dev_r2_u1_s8` | kernel |
|  | 4 | 32.0–39.0 | 24.1–27.1 | 31.0 / 23.3 | 155–174 | 0.62–0.70 | 1.30x–1.44x | 1.00–1.08 | 0.99–1.03 | `dev_r2_u1_s4` | kernel |
|  | 5 | 31.8–38.4 | 24.3–27.9 | 30.9 / 23.1 | 150–173 | 0.75–0.86 | 1.30x–1.38x | 0.99–1.01 | 1.00–1.02 | `dev_r2_u1_s8` | kernel |
|  | 6 | 33.9–43.4 | 25.2–28.1 | 32.8 / 23.8 | 149–166 | 0.90–1.00 | 1.34x–1.55x | 0.94–1.01 | 1.01–1.02 | `mmh_r1_u1_s4` | kernel |
|  | 7 | 35.7–44.5 | 24.4–27.0 | 34.5 / 23.0 | 155–172 | 1.09–1.20 | 1.46x–1.65x | 0.94–1.03 | 1.00–1.07 | `mmh_r2_u1_s8` | kernel |
|  | 8 | 37.5–43.2 | 24.5–26.3 | 36.1 / 22.9 | 159–171 | 1.28–1.37 | 1.53x–1.67x | 0.99–1.01 | 0.99–1.05 | `mmh_r2_u1_s8` | kernel |
|  | 9 | 41.6–50.0 | 27.3–32.3 | 40.2 / 26.5 | 130–154 | 1.17–1.38 | 1.49x–1.59x | 0.97–1.02 | 1.00–1.05 | `mmh_r1_u1_s4` | kernel |
|  | 10 | 45.9–49.8 | 26.1–28.7 | 43.8 / 24.3 | 146–161 | 1.46–1.61 | 1.73x–1.76x | 0.98–1.00 | 0.96–1.01 | `mmh_r2_u1_s8` | kernel |
|  | 11 | 48.5–52.9 | 25.9–27.8 | 47.7 / 24.8 | 151–162 | 1.66–1.78 | 1.85x–1.90x | 0.99–1.01 | 0.99–1.00 | `mmh_r2_u1_s8` | kernel |
|  | 12 | 52.6–54.9 | 26.8–28.8 | 50.9 / 25.0 | 146–156 | 1.75–1.88 | 1.90x–1.98x | 0.98–1.01 | 0.96–1.01 | `mmh_r2_u1_s8_f1` | kernel |
|  | 13 | 55.6–58.8 | 26.4–28.5 | 54.0 / 24.8 | 147–159 | 1.91–2.07 | 2.06x–2.11x | 0.99–1.00 | 0.99–1.01 | `mmh_r1_u1_s8` | kernel |
|  | 14 | 57.4–59.3 | 26.5–27.4 | 56.2 / 24.7 | 153–158 | 2.15–2.21 | 2.15x–2.20x | 0.99–1.00 | 1.00–1.04 | `mmh_r2_u1_s8_f1` | kernel |
|  | 15 | 61.1–62.7 | 25.9–27.1 | 59.5 / 24.0 | 155–162 | 2.32–2.43 | 2.31x–2.36x | 0.99–1.01 | 0.99–1.02 | `mmh_r2_u1_s8` | kernel |
|  | 16 | 84.0–90.9 | 26.6–29.8 | 83.3 / 24.8 | 141–158 | 2.25–2.52 | 2.85x–3.19x | 0.99–1.03 | 0.93–1.01 | `mmh_r1_u1_s8` | kernel |
|  | 24 | 87.9–93.9 | 48.9–54.7 | 87.1 / 46.9 | 77–86 | 1.84–2.06 | 1.72x–1.82x | 0.98–1.01 | 0.99–1.00 | `mma_r1_u1_s4_p4` | kernel |
|  | 32 | 87.3–92.9 | 53.9–59.8 | 86.1 / 53.0 | 70–78 | 2.24–2.49 | 1.55x–1.63x | 0.99–1.02 | 0.99–1.01 | `mma_r1_u1_s4_p4` | kernel |
| gate/up 2048→6144 | 1 | 78.1–81.0 | 81.2–87.4 | 77.1 / 80.6 | 288–310 | 0.29–0.31 | 0.93x–0.96x | 0.99–1.00 | 0.99–1.01 | `tg_r1_u2_s4` | F.linear (1 row) |
|  | 2 | 94.5–121.7 | 82.6–109.8 | 93.6 / 81.0 | 229–305 | 0.46–0.61 | 1.11x–1.17x | 1.00–1.08 | 0.98–1.01 | `mmh_r2_u1_s4` | kernel |
|  | 3 | 103.4–139.7 | 83.0–95.1 | 101.4 / 81.9 | 264–303 | 0.79–0.91 | 1.24x–1.47x | 0.99–1.16 | 0.97–1.12 | `mmh_r2_u1_s2` | kernel |
|  | 4 | 124.4–145.5 | 92.3–119.8 | 102.1 / 87.3 | 210–273 | 0.84–1.09 | 1.21x–1.42x | 0.94–1.15 | 0.94–1.05 | `mmh_r1_u1_s4` | kernel |
|  | 5 | 165.1–172.3 | 108.9–145.9 | 143.1 / 90.8 | 172–231 | 0.86–1.16 | 1.17x–1.55x | 0.93–1.13 | 0.82–1.06 | `mmh_r1_u1_s8` | **F.linear** |
|  | 6 | 214.3–236.5 | 119.2–141.2 | 190.2 / 97.3 | 178–211 | 1.07–1.27 | 1.57x–1.87x | 0.90–0.98 | 0.87–1.01 | `mmh_r2_u1_s2` | kernel |
|  | 7 | 164.2–238.4 | 91.6–119.7 | 147.6 / 85.8 | 210–275 | 1.47–1.92 | 1.79x–2.10x | 0.95–1.08 | 0.89–1.01 | `mmh_r2_u1_s4` | kernel |
|  | 8 | 187.8–235.3 | 95.6–117.8 | 168.5 / 85.2 | 214–263 | 1.71–2.11 | 1.85x–2.05x | 0.92–1.08 | 0.85–1.05 | `mmh_r1_u1_s8` | kernel |
|  | 9 | 179.9–262.9 | 88.9–131.1 | 178.3 / 86.3 | 192–283 | 1.73–2.55 | 2.01x–2.15x | 0.91–1.03 | 0.98–1.07 | `mmh_r2_u1_s2` | kernel |
|  | 10 | 127.0–198.9 | 86.4–141.5 | 126.4 / 85.4 | 178–291 | 1.78–2.91 | 1.41x–1.52x | 0.90–1.00 | 0.95–1.00 | `mmh_r2_u1_s4` | kernel |
|  | 11 | 152.8–235.4 | 86.7–115.4 | 152.3 / 85.9 | 218–290 | 2.40–3.19 | 1.76x–2.04x | 0.97–1.06 | 0.97–1.12 | `mmh_r2_u1_s4` | kernel |
|  | 12 | 172.9–247.1 | 107.1–154.2 | 152.7 / 86.0 | 163–235 | 1.96–2.82 | 1.44x–1.63x | 1.08–1.15 | 0.89–1.10 | `mmh_r2_u1_s8` | kernel |
|  | 13 | 203.6–246.7 | 117.3–144.3 | 175.8 / 107.0 | 174–215 | 2.27–2.79 | 1.60x–1.80x | 0.94–1.08 | 0.98–1.14 | `mmh_r2_u1_s8` | kernel |
|  | 14 | 187.2–244.5 | 107.7–134.9 | 170.3 / 91.6 | 187–234 | 2.61–3.27 | 1.61x–1.85x | 0.93–1.12 | 0.89–1.06 | `mmh_r2_u1_s4` | kernel |
|  | 15 | 207.9–231.9 | 120.1–127.2 | 180.5 / 96.8 | 198–210 | 2.97–3.14 | 1.68x–1.82x | 0.99–1.12 | 0.95–1.11 | `mmh_r2_u1_s4` | kernel |
|  | 16 | 212.1–254.8 | 127.6–151.2 | 187.9 / 104.6 | 166–197 | 2.66–3.15 | 1.58x–1.75x | 0.95–1.17 | 0.91–1.06 | `mmh_r2_u1_s8` | kernel |
|  | 24 | 224.0–263.8 | 165.9–190.3 | 220.8 / 162.2 | 132–152 | 3.17–3.64 | 1.35x–1.39x | 0.98–1.04 | 0.95–1.09 | `mma_r1_u1_s4` | kernel |
|  | 32 | 251.6–275.2 | 203.4–209.3 | 243.3 / 186.1 | 120–124 | 3.85–3.96 | 1.24x–1.32x | 0.97–1.06 | 0.99–1.06 | `mma_r1_u1_s4` | kernel |
| down 6144→2048 | 1 | 81.9–83.4 | 87.1–90.5 | 79.4 / 83.9 | 278–289 | 0.28–0.29 | 0.92x–0.95x | 0.99–1.03 | 1.00–1.04 | `dev_r1_u2_s4` | F.linear (1 row) |
|  | 2 | 118.8–160.3 | 85.4–108.3 | 117.6 / 85.0 | 232–295 | 0.46–0.59 | 1.37x–1.48x | 0.98–1.02 | 1.00–1.05 | `mmh_r2_u1_s4` | kernel |
|  | 3 | 121.0–170.8 | 85.6–102.0 | 119.5 / 84.0 | 247–294 | 0.74–0.88 | 1.41x–1.67x | 0.98–1.06 | 0.97–1.03 | `mmh_r1_u1_s8` | kernel |
|  | 4 | 114.2–192.2 | 84.7–121.0 | 113.0 / 83.7 | 208–297 | 0.83–1.19 | 1.35x–1.62x | 0.96–1.02 | 1.00–1.07 | `mmh_r1_u1_s8` | kernel |
|  | 5 | 121.9–214.5 | 85.6–131.8 | 120.3 / 84.1 | 191–294 | 0.95–1.47 | 1.42x–1.78x | 0.98–1.10 | 0.99–1.17 | `mmh_r1_u1_s8` | kernel |
|  | 6 | 228.8–279.2 | 114.4–138.9 | 180.8 / 103.4 | 181–220 | 1.09–1.32 | 1.86x–2.20x | 0.79–1.01 | 0.89–1.04 | `mmh_r2_u1_s8_f1` | kernel |
|  | 7 | 176.4–253.1 | 93.9–117.1 | 172.0 / 89.9 | 215–268 | 1.50–1.88 | 1.88x–2.16x | 0.96–1.02 | 0.97–1.14 | `mmh_r1_u1_s4` | kernel |
|  | 8 | 189.5–271.4 | 90.7–120.1 | 180.2 / 86.9 | 210–278 | 1.68–2.22 | 1.96x–2.56x | 0.97–1.03 | 0.86–1.15 | `mmh_r1_u1_s4_f1` | kernel |
|  | 9 | 211.5–268.7 | 90.9–116.0 | 205.8 / 88.6 | 217–277 | 1.95–2.49 | 2.14x–2.33x | 0.97–1.05 | 0.96–1.01 | `mmh_r1_u1_s8` | kernel |
|  | 10 | 224.8–250.7 | 92.2–121.2 | 223.2 / 90.8 | 208–273 | 2.08–2.73 | 2.07x–2.44x | 0.99–1.06 | 0.97–1.00 | `mmh_r2_u1_s4` | kernel |
|  | 11 | 248.6–277.0 | 96.7–141.4 | 242.4 / 92.7 | 178–260 | 1.96–2.86 | 1.93x–2.57x | 1.01–1.05 | 0.93–1.05 | `mmh_r1_u1_s4` | kernel |
|  | 12 | 281.3–296.3 | 105.7–122.7 | 264.6 / 94.0 | 205–238 | 2.46–2.86 | 2.34x–2.77x | 0.99–1.02 | 0.97–1.12 | `mmh_r2_u1_s8_f1` | kernel |
|  | 13 | 302.4–328.9 | 107.4–123.4 | 287.6 / 92.0 | 204–234 | 2.65–3.05 | 2.49x–2.92x | 0.93–1.02 | 0.87–1.21 | `mmh_r2_u1_s4` | kernel |
|  | 14 | 334.3–341.3 | 111.5–136.4 | 313.2 / 95.7 | 184–226 | 2.58–3.16 | 2.50x–3.00x | 1.00–1.01 | 0.90–1.08 | `mmh_r2_u1_s8_f1` | kernel |
|  | 15 | 339.0–353.0 | 104.5–127.9 | 320.8 / 89.8 | 197–241 | 2.95–3.61 | 2.67x–3.24x | 0.95–1.03 | 1.04–1.14 | `mmh_r2_u1_s4_f1` | kernel |
|  | 16 | 212.7–233.8 | 94.0–112.2 | 203.1 / 89.1 | 224–268 | 3.59–4.28 | 1.93x–2.29x | 1.00–1.14 | 0.86–1.08 | `mmh_r2_u1_s4_f1` | kernel |
|  | 24 | 291.3–333.8 | 157.6–194.5 | 290.3 / 156.8 | 129–160 | 3.10–3.83 | 1.71x–1.86x | 0.98–1.01 | 0.99–1.04 | `mma_r1_u1_s4_p4` | kernel |
|  | 32 | 261.4–299.2 | 198.8–234.9 | 260.3 / 197.5 | 107–127 | 3.43–4.05 | 1.27x–1.33x | 0.99–1.00 | 0.98–1.10 | `mma_r2_u1_s4_p4` | kernel |
| head 2048→151936 | 1 | 1730.7–1766.6 | 1783.7–1792.8 | 1723.7 / 1731.0 | 347–349 | 0.35 | 0.97x–0.99x | 1.00–1.02 | 0.99–1.01 | `tg_r2_u2_s8` | F.linear (1 row) |
|  | 2 | 1776.3–1802.1 | 1706.1–1733.8 | 1749.2 / 1695.3 | 359–365 | 0.72–0.73 | 1.04x | 1.00–1.01 | 1.00–1.01 | `mmh_r2_u1_s8_f1` | kernel |
|  | 3 | 2048.9–2054.3 | 1729.5–1869.2 | 2025.1 / 1696.5 | 333–360 | 1.00–1.08 | 1.10x–1.18x | 0.99–1.02 | 0.95–0.99 | `mmh_r2_u1_s8_f1` | kernel |
|  | 4 | 2152.1–2355.0 | 1752.5–1820.7 | 2145.1 / 1727.4 | 342–355 | 1.37–1.42 | 1.23x–1.29x | 0.94–1.01 | 0.97–0.99 | `mmh_r2_u2_s4` | kernel |
|  | 5 | 2411.4–2427.4 | 1729.0–1755.1 | 2392.1 / 1708.0 | 355–360 | 1.77–1.80 | 1.38x–1.39x | 1.00 | 0.99 | `mmh_r2_u2_s4` | kernel |
|  | 6 | 2981.6–2992.3 | 1789.5–1819.6 | 2936.5 / 1721.2 | 342–348 | 2.05–2.09 | 1.64x–1.67x | 0.99–1.01 | 0.98–0.99 | `mmh_r1_u1_s8` | kernel |
|  | 7 | 3266.0–3272.9 | 1761.4–1775.7 | 3245.4 / 1715.0 | 350–353 | 2.45–2.47 | 1.84x–1.85x | 1.00 | 0.99–1.01 | `mmh_r2_u1_s8_f1` | kernel |
|  | 8 | 3800.5–4134.9 | 1774.7–1874.2 | 3745.9 / 1724.3 | 332–351 | 2.66–2.81 | 2.03x–2.33x | 0.92–1.01 | 0.95–1.00 | `mmh_r2_u1_s4_f1` | kernel |
|  | 9 | 4281.2–4321.3 | 1788.9–1841.4 | 4242.5 / 1730.9 | 338–348 | 3.04–3.13 | 2.35x–2.39x | 1.00 | 0.97–1.00 | `mmh_r2_u1_s8` | kernel |
|  | 10 | 8178.6–8521.2 | 1822.5–1836.0 | 7973.6 / 1746.9 | 339–341 | 3.39–3.41 | 4.49x–4.64x | 1.00–1.02 | 0.99 | `mmh_r2_u1_s4` | kernel |
|  | 11 | 8308.5–8367.4 | 1771.1–1788.3 | 7929.2 / 1752.4 | 348–351 | 3.83–3.87 | 4.68x–4.69x | 0.97–1.00 | 1.00–1.01 | `mmp_r1_u1_s8` | kernel |
|  | 12 | 8331.6–8483.7 | 1806.8–1850.9 | 7991.3 / 1747.0 | 336–344 | 4.03–4.13 | 4.58x–4.61x | 1.01–1.02 | 0.99–1.01 | `mmp_r1_u1_s8` | kernel |
|  | 13 | 8483.3–8495.2 | 1815.5–1821.5 | 7985.0 / 1749.1 | 342–343 | 4.44–4.46 | 4.66x–4.67x | 0.98–1.01 | 1.00–1.02 | `mmp_r1_u1_s8` | kernel |
|  | 14 | 8109.8–8294.5 | 1797.6–1851.9 | 8004.5 / 1745.4 | 336–346 | 4.70–4.85 | 4.48x–4.51x | 0.99–1.01 | 0.96–0.99 | `mmh_r2_u1_s8` | kernel |
|  | 15 | 8280.5–8545.9 | 1845.9–1847.4 | 7987.4 / 1769.4 | 337 | 5.05–5.06 | 4.48x–4.63x | 1.00–1.02 | 0.98–0.99 | `mmp_r1_u1_s8` | kernel |
|  | 16 | 8354.5–8757.4 | 1806.6–1820.6 | 8048.6 / 1755.5 | 342–344 | 5.47–5.51 | 4.62x–4.81x | 0.98–1.02 | 1.00 | `mmh_r2_u1_s8` | kernel |
|  | 24 | 9904.5–9988.1 | 3400.0–3429.1 | 9361.2 / 3313.2 | 181–183 | 4.36–4.39 | 2.89x–2.94x | 0.99–1.00 | 0.99–1.00 | `mma_r1_u1_s4` | kernel |
|  | 32 | 10286.5–10533.9 | 4214.7–4218.7 | 9998.5 / 4178.0 | 148 | 4.72–4.73 | 2.44x–2.50x | 0.99–1.00 | 1.00–1.01 | `mma_r1_u1_s4` | kernel |
| Qwen3-4B up 2560→9728 | 1 | 154.1–155.4 | 162.8–164.1 | 150.5 / 159.4 | 303–306 | 0.30–0.31 | 0.95x | 0.99–1.00 | 0.99–1.01 | `tg_r1_u2_s4` | F.linear (1 row) |
|  | 2 | 185.2–185.8 | 183.6–185.3 | 167.9 / 160.1 | 269–271 | 0.54 | 1.00x–1.01x | 0.99–1.00 | 0.95–1.04 | `mmh_r1_u1_s8` | kernel |
|  | 3 | 221.2–261.6 | 190.0–200.6 | 177.4 / 167.5 | 248–262 | 0.74–0.79 | 1.16x–1.30x | 0.90–1.02 | 1.03–1.08 | `mmh_r2_u1_s8_f1` | kernel |
|  | 4 | 213.2–327.5 | 166.4–210.0 | 189.2 / 160.5 | 237–299 | 0.95–1.20 | 1.28x–1.56x | 0.94–0.98 | 0.92–1.02 | `mmh_r2_u1_s4` | kernel |
|  | 5 | 209.4–215.7 | 161.7–164.0 | 204.2 / 159.5 | 304–308 | 1.52–1.54 | 1.29x–1.32x | 0.98–1.05 | 1.00–1.03 | `mmh_r1_u1_s4` | kernel |
|  | 6 | 295.8–303.9 | 183.3–198.2 | 252.2 / 161.5 | 251–272 | 1.51–1.63 | 1.49x–1.66x | 0.98–1.07 | 0.95–1.03 | `mmh_r1_u1_s8` | kernel |
|  | 7 | 330.3–385.0 | 194.2–195.0 | 296.9 / 162.6 | 255–256 | 1.79–1.80 | 1.70x–1.97x | 0.97–1.09 | 0.98–1.11 | `mmh_r2_u1_s8` | kernel |
|  | 8 | 367.3–400.4 | 191.2–223.5 | 334.3 / 166.5 | 223–261 | 1.78–2.08 | 1.79x–1.92x | 0.99–1.16 | 1.01 | `mmh_r2_u1_s2` | kernel |
|  | 9 | 374.1–374.5 | 170.8–171.6 | 363.4 / 161.3 | 290–292 | 2.61–2.62 | 2.18x–2.19x | 0.99–1.01 | 1.00–1.01 | `mmh_r2_u1_s8` | kernel |
|  | 10 | 286.0–292.7 | 197.0–216.2 | 247.2 / 167.2 | 230–253 | 2.30–2.53 | 1.35x–1.45x | 1.00–1.04 | 0.93–0.98 | `mmh_r2_u1_s8` | kernel |
|  | 11 | 256.0–332.1 | 177.5–268.6 | 247.8 / 174.0 | 185–281 | 2.04–3.09 | 1.24x–1.44x | 0.99–1.02 | 0.81–1.04 | `mmh_r2_u1_s4` | kernel |
|  | 12 | 258.0–351.6 | 175.7–225.5 | 247.2 / 167.5 | 221–283 | 2.65–3.40 | 1.47x–1.56x | 0.96–0.97 | 0.98–0.99 | `mmh_r2_u1_s8` | kernel |
|  | 13 | 288.5–292.7 | 188.1–215.2 | 253.7 / 168.6 | 231–265 | 3.01–3.44 | 1.36x–1.53x | 0.95–1.02 | 0.95–0.96 | `mmh_r2_u1_s8` | kernel |
|  | 14 | 328.3–393.5 | 206.4–217.9 | 299.2 / 173.1 | 229–241 | 3.20–3.38 | 1.59x–1.81x | 0.98–1.01 | 0.88–0.95 | `mmh_r2_u1_s8` | kernel |
|  | 15 | 333.5–469.2 | 241.0–243.7 | 301.2 / 174.8 | 204–207 | 3.07–3.10 | 1.37x–1.95x | 1.01–1.06 | 0.84–1.04 | `mmp_r2_u1_s4` | kernel |
|  | 16 | 323.7–336.8 | 196.2–201.7 | 294.2 / 169.4 | 247–254 | 3.95–4.06 | 1.60x–1.72x | 1.04–1.19 | 0.90–1.19 | `mmh_r2_u1_s8` | kernel |
|  | 24 | 361.4–380.0 | 346.6–355.5 | 346.7 / 309.4 | 140–144 | 3.36–3.45 | 1.02x–1.10x | 1.03–1.10 | 0.97–0.99 | `mma_r1_u1_s4_p4` | kernel |
|  | 32 | 485.7–520.6 | 409.0–415.1 | 476.3 / 380.7 | 120–122 | 3.84–3.90 | 1.19x–1.25x | 1.00–1.02 | 0.96–1.01 | `mma_r1_u1_s4_p4` | kernel |
| Llama-7B up 4096→11008 | 1 | 275.8–278.8 | 284.1–286.4 | 269.5 / 279.3 | 315–317 | 0.31–0.32 | 0.96x–0.98x | 0.98–1.00 | 0.99–1.00 | `tg_r1_u2_s8` | F.linear (1 row) |
|  | 2 | 307.1–323.4 | 280.0–286.8 | 299.6 / 272.7 | 314–322 | 0.63–0.64 | 1.10x–1.13x | 0.96–1.01 | 0.99 | `mmh_r2_u1_s8_f1` | kernel |
|  | 3 | 334.6–390.4 | 283.8–311.1 | 328.8 / 277.4 | 290–318 | 0.87–0.95 | 1.18x–1.25x | 1.00–1.03 | 0.99–1.00 | `mmh_r1_u1_s8_f1` | kernel |
|  | 4 | 390.2–485.3 | 292.0–311.3 | 345.0 / 275.9 | 290–309 | 1.16–1.24 | 1.34x–1.56x | 1.02–1.05 | 0.97–0.99 | `mmh_r2_u1_s8` | kernel |
|  | 5 | 447.9–451.7 | 302.5–356.8 | 388.0 / 277.4 | 253–298 | 1.26–1.49 | 1.27x–1.48x | 0.97–1.24 | 0.82–0.97 | `mmh_r1_u1_s4` | kernel |
|  | 6 | 529.7–543.6 | 306.7–321.9 | 476.8 / 277.5 | 280–294 | 1.68–1.76 | 1.65x–1.77x | 0.99 | 0.92–0.97 | `mmh_r1_u1_s8` | kernel |
|  | 7 | 585.9–587.0 | 293.3–300.5 | 549.4 / 275.4 | 300–307 | 2.10–2.15 | 1.95x–2.00x | 0.96–1.01 | 0.96–1.04 | `mmh_r2_u1_s8_f1` | kernel |
|  | 8 | 631.5–671.1 | 284.8–303.5 | 618.3 / 277.8 | 297–317 | 2.38–2.53 | 2.21x–2.22x | 0.98–1.04 | 0.99–1.03 | `mmh_r2_u1_s4` | kernel |
|  | 9 | 686.6–714.5 | 301.8–340.0 | 679.4 / 293.9 | 265–299 | 2.39–2.69 | 2.10x–2.28x | 0.99–1.00 | 0.94–0.99 | `mmh_r2_u1_s8_f1` | kernel |
|  | 10 | 442.5–456.9 | 309.3–328.8 | 413.6 / 286.0 | 274–292 | 2.74–2.92 | 1.39x–1.43x | 1.04–1.06 | 0.94–1.02 | `mmh_r2_u1_s8` | kernel |
|  | 11 | 414.5–548.4 | 285.3–339.2 | 413.4 / 281.1 | 266–316 | 2.92–3.48 | 1.45x–1.62x | 0.87–1.01 | 0.98–1.00 | `mmp_r1_u1_s8` | kernel |
|  | 12 | 516.1–521.1 | 307.6–308.4 | 496.6 / 288.8 | 292–293 | 3.51–3.52 | 1.67x–1.69x | 0.99 | 0.98–1.02 | `mmh_r2_u1_s8` | kernel |
|  | 13 | 512.1–515.5 | 292.5–302.5 | 492.9 / 285.1 | 298–308 | 3.88–4.01 | 1.70x–1.75x | 0.99–1.01 | 1.01–1.02 | `mmp_r1_u1_s8` | kernel |
|  | 14 | 499.4–540.6 | 291.8–296.1 | 492.6 / 283.8 | 305–309 | 4.26–4.33 | 1.71x–1.83x | 0.95–1.00 | 1.00–1.01 | `mmp_r1_u1_s8` | kernel |
|  | 15 | 508.3–541.3 | 300.4–319.3 | 498.2 / 288.7 | 282–300 | 4.24–4.50 | 1.69x–1.70x | 1.00–1.12 | 1.00–1.04 | `mmh_r2_u1_s4` | kernel |
|  | 16 | 510.5–527.2 | 304.1–329.8 | 496.1 / 286.7 | 273–297 | 4.37–4.74 | 1.60x–1.68x | 0.99–1.03 | 0.91–0.98 | `mmh_r2_u1_s4` | kernel |
|  | 24 | 1200.6–1276.7 | 579.4–592.9 | 1192.6 / 528.0 | 152–156 | 3.65–3.74 | 2.07x–2.15x | 0.96–1.01 | 0.94–0.96 | `mma_r1_u1_s4_p4` | kernel |
|  | 32 | 1260.5–1270.3 | 654.8–699.0 | 1210.7 / 647.4 | 129–138 | 4.13–4.41 | 1.82x–1.92x | 1.00 | 0.97–1.02 | `mma_r1_u1_s4` | kernel |

Model-order sum of the 197 linears, ms (bfloat16; per process):

| M | F.linear | kernel (every cell on the kernel) |
|---:|---:|---:|
| 1 | 11.34–11.41 | 12.26–12.62 |
| 2 | 15.63–17.75 | 12.44–13.55 |
| 3 | 16.11–19.91 | 12.12–13.62 |
| 4 | 18.12–21.07 | 13.65–14.48 |
| 5 | 22.58–23.34 | 15.14–15.44 |
| 6 | 27.96–29.01 | 15.46–16.02 |
| 7 | 25.63–28.13 | 13.47–14.87 |
| 8 | 26.75–28.04 | 13.56–14.54 |
| 9 | 30.15–31.64 | 15.22–15.97 |
| 10 | 29.16–31.30 | 13.61–15.51 |
| 11 | 34.44–36.47 | 15.03–15.39 |
| 12 | 34.92–37.96 | 16.19–17.76 |
| 13 | 36.78–37.79 | 15.21–16.14 |
| 14 | 37.79–40.09 | 14.94–16.31 |
| 15 | 39.25–39.51 | 15.53–16.47 |
| 16 | 38.53–38.78 | 16.67–17.03 |
| 24 | 43.84–44.63 | 25.72–26.83 |
| 32 | 42.21–44.92 | 29.04–30.76 |

</details>

<details><summary>float16</summary>

### float16

| Shape | M | F.linear µs | kernel µs | fastest sample, F.linear / kernel µs | kernel GB/s | kernel TFLOPS | speedup | A/A | B/B | variant | routed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| q/o 2048→2048 | 1 | 30.5–32.4 | 35.2–39.8 | 29.5 / 33.7 | 211–238 | 0.21–0.24 | 0.81x–0.87x | 0.97–1.01 | 0.97–1.01 | `tg_r2_u1_s4` | F.linear (1 row) |
|  | 2 | 38.2–46.7 | 35.6–42.0 | 36.8 / 34.3 | 200–236 | 0.40–0.47 | 1.07x–1.11x | 0.97–1.01 | 0.99–1.01 | `tg_r1_u2_s4` | kernel |
|  | 3 | 38.5–53.5 | 36.1–43.8 | 37.8 / 35.2 | 191–233 | 0.57–0.70 | 1.07x–1.22x | 0.98–1.03 | 0.98–1.01 | `mmh_r1_u1_s4_f1` | kernel |
|  | 4 | 41.1–55.1 | 36.3–42.2 | 40.1 / 34.9 | 199–231 | 0.80–0.93 | 1.10x–1.31x | 1.00–1.04 | 0.98–1.01 | `dev_r2_u1_s4` | kernel |
|  | 5 | 52.6–68.1 | 36.7–42.8 | 44.4 / 34.3 | 196–228 | 0.98–1.14 | 1.40x–1.61x | 0.91–1.02 | 0.95–1.17 | `mmh_r1_u1_s8_f1_h1` | kernel |
|  | 6 | 50.3–78.6 | 36.3–44.8 | 48.3 / 34.8 | 187–231 | 1.12–1.39 | 1.39x–1.81x | 0.96–1.03 | 0.99–1.03 | `mmh_r1_u1_s4_h1` | kernel |
|  | 7 | 52.6–68.4 | 36.5–42.0 | 50.6 / 34.5 | 200–230 | 1.40–1.61 | 1.41x–1.70x | 0.99–1.02 | 0.94–1.00 | `mmh_r1_u1_s8` | kernel |
|  | 8 | 57.3–89.7 | 35.6–44.2 | 56.6 / 34.9 | 190–236 | 1.52–1.89 | 1.52x–2.03x | 1.01–1.14 | 0.87–1.02 | `mmh_r1_u1_s8_h1` | kernel |
|  | 9 | 62.3–89.8 | 36.7–47.6 | 61.2 / 36.0 | 176–228 | 1.59–2.06 | 1.68x–1.89x | 0.97–1.06 | 0.99–1.04 | `mmh_r1_u1_s8_h1` | kernel |
|  | 10 | 68.3–74.6 | 37.6–42.6 | 67.6 / 36.8 | 197–223 | 1.97–2.23 | 1.75x–1.84x | 0.98–1.01 | 0.99–1.03 | `mmh_r2_u1_s8` | kernel |
|  | 11 | 74.3–86.1 | 36.9–46.2 | 73.0 / 35.7 | 182–227 | 2.00–2.50 | 1.86x–2.01x | 0.94–1.00 | 0.96–1.04 | `mmh_r1_u1_s8` | kernel |
|  | 12 | 80.9–89.1 | 37.7–43.2 | 79.7 / 36.2 | 194–222 | 2.33–2.67 | 2.02x–2.16x | 0.99–1.01 | 0.97–1.02 | `mmh_r1_u1_s4` | kernel |
|  | 13 | 87.9–95.7 | 38.3–44.6 | 85.2 / 36.8 | 188–219 | 2.44–2.85 | 2.14x–2.30x | 1.00 | 0.98–1.03 | `mmh_r2_u1_s4` | kernel |
|  | 14 | 94.1–98.0 | 37.9–41.7 | 92.3 / 36.6 | 201–221 | 2.82–3.10 | 2.31x–2.48x | 0.98–1.01 | 0.99–1.01 | `mmh_r1_u1_s4_h1` | kernel |
|  | 15 | 98.4–106.6 | 37.7–44.5 | 97.6 / 36.3 | 188–223 | 2.83–3.34 | 2.39x–2.61x | 1.00–1.01 | 1.00–1.04 | `mmh_r1_u1_s8_h1` | kernel |
|  | 16 | 79.2–86.3 | 38.0–44.0 | 76.9 / 36.6 | 191–221 | 3.05–3.54 | 1.93x–2.09x | 0.99–1.00 | 0.98–1.02 | `mmh_r1_u1_s4_h1` | kernel |
|  | 24 | 89.1–96.2 | 67.8–79.6 | 88.0 / 66.2 | 105–124 | 2.53–2.97 | 1.21x–1.32x | 1.00–1.02 | 0.95–1.05 | `mma_r1_u1_s4_p4` | kernel |
|  | 32 | 85.6–96.2 | 82.4–88.6 | 85.1 / 80.8 | 95–102 | 3.03–3.26 | 1.03x–1.09x | 0.95–1.03 | 0.99–1.03 | `mma_r2_u1_s4_p4` | kernel |
| k/v 2048→1024 | 1 | 19.3–20.6 | 23.7–26.8 | 18.1 / 22.1 | 156–177 | 0.16–0.18 | 0.77x–0.81x | 1.00–1.03 | 0.97–1.02 | `tg_r1_u2_s4` | F.linear (1 row) |
|  | 2 | 26.8–29.1 | 23.0–23.4 | 25.2 / 22.1 | 179–182 | 0.36 | 1.16x–1.24x | 0.95–1.00 | 0.98–1.01 | `dev_r2_u1_s8` | kernel |
|  | 3 | 28.2–31.4 | 24.0–25.8 | 26.4 / 22.2 | 162–175 | 0.49–0.52 | 1.17x–1.31x | 0.99–1.07 | 0.94–1.05 | `dev_r2_u1_s8` | kernel |
|  | 4 | 28.5–34.7 | 23.9–26.8 | 27.8 / 22.6 | 157–176 | 0.63–0.70 | 1.19x–1.30x | 0.94–1.06 | 0.97–1.08 | `dev_r2_u1_s4` | kernel |
|  | 5 | 30.7–38.7 | 24.6–25.8 | 29.5 / 22.5 | 163–171 | 0.81–0.85 | 1.25x–1.50x | 0.99–1.02 | 0.98–1.01 | `mmh_r1_u1_s8_f1` | kernel |
|  | 6 | 34.9–49.4 | 24.5–26.7 | 33.0 / 22.2 | 157–171 | 0.94–1.03 | 1.42x–1.85x | 0.90–1.20 | 0.98–1.05 | `mmh_r1_u1_s8_f1` | kernel |
|  | 7 | 31.3–36.9 | 24.1–25.1 | 30.4 / 22.6 | 167–174 | 1.17–1.22 | 1.29x–1.52x | 0.93–1.03 | 1.01–1.06 | `mmh_r1_u1_s8_f1` | kernel |
|  | 8 | 32.0–33.4 | 23.2–24.8 | 31.5 / 22.4 | 169–181 | 1.35–1.44 | 1.33x–1.38x | 0.98–1.00 | 0.99–1.00 | `mmh_r1_u1_s8_f1_h1` | kernel |
|  | 9 | 35.7–50.7 | 24.5–30.2 | 34.1 / 23.2 | 139–171 | 1.25–1.54 | 1.44x–1.68x | 0.99–1.00 | 0.97–1.02 | `mmh_r2_u1_s8` | kernel |
|  | 10 | 40.4–44.4 | 24.1–27.0 | 39.2 / 23.5 | 155–174 | 1.55–1.74 | 1.60x–1.68x | 0.99–1.02 | 0.96–1.07 | `mmh_r1_u1_s8` | kernel |
|  | 11 | 45.0–48.6 | 24.0–27.0 | 44.2 / 23.2 | 155–175 | 1.71–1.93 | 1.78x–1.91x | 0.95–1.00 | 0.96–1.04 | `mmh_r2_u1_s8` | kernel |
|  | 12 | 49.6–51.7 | 25.1–26.5 | 48.2 / 23.4 | 158–167 | 1.90–2.00 | 1.95x–2.00x | 1.00–1.01 | 1.00–1.03 | `mmh_r1_u1_s8` | kernel |
|  | 13 | 53.1–54.5 | 23.9–25.8 | 52.6 / 23.6 | 163–175 | 2.12–2.28 | 2.10x–2.22x | 0.98–1.00 | 0.98–1.02 | `mmh_r2_u1_s8` | kernel |
|  | 14 | 59.4–64.7 | 24.9–27.0 | 58.3 / 23.8 | 155–168 | 2.17–2.35 | 2.34x–2.39x | 0.96–1.02 | 0.97–1.05 | `mmh_r2_u1_s8_h1` | kernel |
|  | 15 | 59.8–62.3 | 25.3–26.0 | 58.9 / 24.3 | 162–166 | 2.42–2.49 | 2.30x–2.45x | 0.97–1.01 | 1.00–1.04 | `mmh_r2_u1_s8_f1` | kernel |
|  | 16 | 73.3–74.6 | 24.9–26.2 | 72.5 / 23.5 | 160–169 | 2.56–2.70 | 2.84x–2.98x | 0.99–1.01 | 0.99–1.04 | `mmh_r1_u1_s8_h1` | kernel |
|  | 24 | 75.9–80.9 | 47.6–50.1 | 74.4 / 45.2 | 84–88 | 2.01–2.11 | 1.59x–1.68x | 0.94–1.00 | 0.99–1.03 | `mma_r1_u1_s4_p4` | kernel |
|  | 32 | 76.1–79.0 | 52.0–55.4 | 75.4 / 50.9 | 76–81 | 2.42–2.58 | 1.41x–1.47x | 1.00–1.02 | 0.96–1.00 | `mma_r1_u1_s4_p4` | kernel |
| gate/up 2048→6144 | 1 | 76.9–77.5 | 82.3–85.9 | 75.7 / 81.0 | 293–306 | 0.29–0.31 | 0.90x–0.94x | 0.99–1.01 | 0.99–1.00 | `tg_r1_u2_s4` | F.linear (1 row) |
|  | 2 | 85.1–112.5 | 82.6–110.1 | 83.7 / 81.5 | 229–305 | 0.46–0.61 | 1.01x–1.07x | 0.95–1.02 | 0.97–1.03 | `mmh_r1_u1_s8_h1` | **F.linear** |
|  | 3 | 106.9–138.5 | 96.1–117.5 | 85.9 / 83.7 | 214–262 | 0.64–0.79 | 1.11x–1.23x | 0.94–1.29 | 0.93–1.12 | `mmh_r2_u1_s8_f1_h1` | **F.linear** |
|  | 4 | 108.9–151.6 | 85.4–113.9 | 95.4 / 82.2 | 221–295 | 0.88–1.18 | 1.24x–1.33x | 0.92–1.09 | 0.94–1.03 | `mmh_r2_u1_s4_f1_h1` | kernel |
|  | 5 | 149.2–206.3 | 96.7–130.7 | 136.8 / 85.2 | 193–260 | 0.96–1.30 | 1.54x–1.79x | 0.91–1.08 | 0.93–1.04 | `mmh_r1_u1_s4_h1` | kernel |
|  | 6 | 126.7–227.1 | 83.5–116.9 | 126.0 / 81.4 | 215–301 | 1.29–1.81 | 1.52x–1.98x | 0.84–1.02 | 0.98–1.00 | `mmh_r2_u1_s4` | kernel |
|  | 7 | 139.7–228.4 | 83.4–123.7 | 138.8 / 82.6 | 203–302 | 1.42–2.11 | 1.67x–2.10x | 0.97–1.06 | 0.86–1.06 | `mmh_r2_u1_s2` | kernel |
|  | 8 | 162.7–283.0 | 86.8–109.9 | 159.0 / 82.4 | 229–290 | 1.83–2.32 | 1.88x–2.57x | 0.93–0.99 | 0.93–1.08 | `mmh_r2_u1_s8_f1` | kernel |
|  | 9 | 178.2–242.4 | 85.2–123.7 | 176.2 / 83.1 | 203–296 | 1.83–2.66 | 1.96x–2.49x | 0.99–1.13 | 0.94–1.29 | `mmh_r1_u1_s8_h1` | kernel |
|  | 10 | 135.0–201.7 | 95.9–125.6 | 122.2 / 91.1 | 200–262 | 2.00–2.62 | 1.41x–1.60x | 0.95–1.03 | 1.01–1.22 | `mmh_r2_u1_s4_f1` | kernel |
|  | 11 | 158.9–192.8 | 108.3–159.6 | 133.4 / 90.4 | 158–232 | 1.73–2.56 | 1.21x–1.50x | 0.93–1.22 | 0.85–0.96 | `mmh_r2_u1_s4` | **F.linear** |
|  | 12 | 121.8–197.8 | 87.9–143.3 | 116.1 / 84.1 | 176–286 | 2.11–3.44 | 1.28x–1.58x | 0.99–1.02 | 0.84–1.13 | `mmh_r2_u1_s8` | kernel |
|  | 13 | 115.1–192.9 | 83.6–120.4 | 114.0 / 83.0 | 209–301 | 2.72–3.91 | 1.38x–1.60x | 0.85–0.99 | 1.00–1.16 | `mmh_r2_u1_s8_h1` | kernel |
|  | 14 | 154.1–165.9 | 98.3–115.3 | 122.3 / 86.7 | 218–256 | 3.06–3.58 | 1.44x–1.57x | 0.97–1.02 | 0.97–1.08 | `mmh_r2_u1_s8_h1` | kernel |
|  | 15 | 172.2–184.7 | 108.8–132.5 | 115.0 / 85.2 | 190–231 | 2.85–3.47 | 1.39x–1.62x | 0.90–0.99 | 0.89–1.09 | `mmh_r2_u1_s8` | kernel |
|  | 16 | 140.3–194.7 | 118.4–130.7 | 118.6 / 94.6 | 193–212 | 3.08–3.40 | 1.11x–1.49x | 0.91–1.05 | 0.97–1.10 | `mmh_r1_u1_s4_h1` | kernel |
|  | 24 | 158.7–243.9 | 161.2–222.0 | 148.1 / 157.8 | 113–156 | 2.72–3.75 | 0.98x–1.10x | 0.87–1.13 | 0.94–1.08 | `mma_r1_u1_s4` | kernel |
|  | 32 | 161.9–188.5 | 189.7–229.3 | 147.8 / 178.3 | 110–133 | 3.51–4.24 | 0.80x–0.97x | 0.92–1.03 | 0.93–1.05 | `mma_r1_u1_s4` | kernel |
| down 6144→2048 | 1 | 80.7–86.6 | 84.3–91.8 | 78.9 / 82.5 | 274–299 | 0.27–0.30 | 0.92x–1.00x | 0.98–1.00 | 0.98–1.00 | `dev_r4_u1_s4` | F.linear (1 row) |
|  | 2 | 88.4–146.5 | 84.2–115.4 | 86.6 / 83.5 | 218–299 | 0.44–0.60 | 1.04x–1.32x | 0.89–1.01 | 0.99–1.07 | `mmh_r1_u1_s8_h1` | **F.linear** |
|  | 3 | 94.8–157.2 | 85.2–121.9 | 93.8 / 83.8 | 207–296 | 0.62–0.89 | 1.11x–1.31x | 1.00–1.08 | 0.97–1.02 | `mmh_r1_u1_s4_h1` | kernel |
|  | 4 | 108.5–196.9 | 85.3–117.5 | 107.0 / 83.5 | 214–295 | 0.86–1.18 | 1.27x–1.68x | 0.92–1.00 | 0.98–1.04 | `mmh_r1_u1_s4` | kernel |
|  | 5 | 126.1–208.4 | 85.2–128.1 | 124.1 / 83.9 | 196–295 | 0.98–1.48 | 1.46x–1.81x | 0.99–1.08 | 0.89–1.02 | `mmh_r1_u1_s8` | kernel |
|  | 6 | 145.3–238.8 | 88.1–118.9 | 142.3 / 83.7 | 212–286 | 1.27–1.71 | 1.65x–2.01x | 0.96–1.12 | 0.97–1.19 | `mmh_r2_u1_s4` | kernel |
|  | 7 | 201.1–224.6 | 102.2–115.8 | 177.5 / 88.3 | 217–246 | 1.52–1.72 | 1.83x–2.12x | 0.94–1.08 | 0.97–1.16 | `mmh_r1_u1_s8_f1_h1` | kernel |
|  | 8 | 210.3–294.1 | 92.5–120.1 | 177.5 / 85.7 | 210–272 | 1.68–2.18 | 1.89x–2.45x | 0.96–1.04 | 0.95–1.11 | `mmh_r2_u1_s8_f1` | kernel |
|  | 9 | 208.5–268.6 | 91.5–127.4 | 202.6 / 87.6 | 198–275 | 1.78–2.47 | 1.98x–2.28x | 0.94–1.12 | 0.85–1.04 | `mmh_r1_u1_s8` | kernel |
|  | 10 | 232.6–262.6 | 95.2–120.0 | 223.9 / 86.5 | 210–264 | 2.10–2.64 | 2.17x–2.44x | 0.99–1.06 | 0.91–1.01 | `mmh_r2_u1_s8_f1_h1` | kernel |
|  | 11 | 243.2–283.2 | 90.2–134.9 | 239.7 / 85.7 | 187–279 | 2.05–3.07 | 2.08x–2.70x | 0.93–1.00 | 0.78–1.03 | `mmh_r2_u1_s8_f1_h1` | kernel |
|  | 12 | 282.7–294.3 | 105.9–127.9 | 263.2 / 93.0 | 197–238 | 2.36–2.85 | 2.30x–2.67x | 0.99–1.02 | 0.94–1.13 | `mmp_r2_u1_s4_h1` | kernel |
|  | 13 | 297.7–304.7 | 96.1–114.6 | 278.8 / 86.1 | 220–262 | 2.85–3.40 | 2.64x–3.17x | 0.98–1.04 | 0.95–1.03 | `mmh_r2_u1_s4` | kernel |
|  | 14 | 304.2–335.8 | 87.7–111.5 | 295.6 / 85.8 | 226–287 | 3.16–4.02 | 2.92x–3.68x | 0.99–1.03 | 0.90–1.27 | `mmh_r1_u1_s8_h1` | kernel |
|  | 15 | 317.7–353.7 | 88.7–123.4 | 316.8 / 86.4 | 204–284 | 3.06–4.26 | 2.82x–3.58x | 0.97–1.00 | 0.94–1.15 | `mmh_r1_u1_s8_h1` | kernel |
|  | 16 | 146.6–208.3 | 87.1–118.4 | 144.8 / 85.1 | 212–289 | 3.40–4.62 | 1.46x–1.90x | 0.95–1.05 | 0.95–1.04 | `mmh_r2_u1_s8_f1_h1` | kernel |
|  | 24 | 205.8–237.9 | 166.5–203.7 | 188.2 / 155.2 | 124–151 | 2.97–3.63 | 1.16x–1.25x | 0.92–1.02 | 0.96–1.04 | `mma_r1_u1_s4_p4` | kernel |
|  | 32 | 236.5–265.5 | 204.9–227.7 | 232.1 / 192.0 | 111–123 | 3.54–3.93 | 1.15x–1.24x | 1.00–1.03 | 0.97–1.05 | `mma_r2_u1_s4_p4` | kernel |
| head 2048→151936 | 1 | 1731.1–1787.2 | 1771.0–1811.5 | 1705.6 / 1758.3 | 344–351 | 0.34–0.35 | 0.98x–0.99x | 0.99 | 1.00 | `dev_r4_u1_s8` | F.linear (1 row) |
|  | 2 | 1818.9–1880.2 | 1840.5–1904.7 | 1762.7 / 1727.7 | 327–338 | 0.65–0.68 | 0.99x | 0.98–1.02 | 0.96–1.05 | `mmh_r1_u1_s4_f1` | **F.linear** |
|  | 3 | 1832.7–2110.4 | 1743.2–1995.7 | 1770.4 / 1713.4 | 312–357 | 0.94–1.07 | 1.05x–1.06x | 0.97–1.00 | 1.00–1.01 | `mmh_r1_u1_s8_f1` | kernel |
|  | 4 | 2135.4–2160.6 | 1800.9–1804.9 | 2074.9 / 1730.8 | 345–346 | 1.38 | 1.18x–1.20x | 1.00–1.02 | 0.99–1.01 | `mmh_r1_u1_s8` | kernel |
|  | 5 | 2422.7–2485.7 | 1726.6–1965.7 | 2403.7 / 1705.7 | 317–360 | 1.58–1.80 | 1.26x–1.40x | 1.00 | 0.91–0.99 | `mmh_r2_u2_s4_h1` | kernel |
|  | 6 | 2903.9–2965.5 | 1743.3–1787.7 | 2894.9 / 1707.3 | 348–357 | 2.09–2.14 | 1.62x–1.70x | 1.00 | 0.99–1.01 | `mmh_r1_u1_s8_f1_h1` | kernel |
|  | 7 | 3342.3–3384.8 | 1750.5–1771.6 | 3320.0 / 1698.3 | 351–356 | 2.46–2.49 | 1.89x–1.93x | 1.00 | 1.00–1.01 | `mmh_r2_u2_s4` | kernel |
|  | 8 | 3834.8–3913.2 | 1791.8–1837.7 | 3802.4 / 1702.4 | 339–347 | 2.71–2.78 | 2.13x–2.14x | 1.02–1.04 | 0.99 | `mmh_r1_u1_s8_f1_h1` | kernel |
|  | 9 | 4374.3–4408.0 | 1793.9–1834.6 | 4325.2 / 1730.5 | 339–347 | 3.05–3.12 | 2.40x–2.44x | 1.00 | 1.00–1.02 | `mmh_r2_u1_s8_h1` | kernel |
|  | 10 | 7642.6–7680.8 | 1806.8–1812.5 | 7154.5 / 1754.1 | 343–344 | 3.43–3.44 | 4.23x–4.24x | 0.99–1.01 | 1.01–1.03 | `mmh_r2_u1_s2` | kernel |
|  | 11 | 7334.6–7609.6 | 1747.8–1838.6 | 7123.2 / 1710.3 | 338–356 | 3.72–3.92 | 4.14x–4.20x | 0.99–1.01 | 0.99–1.00 | `mmh_r2_u1_s8_h1` | kernel |
|  | 12 | 7418.6–7792.8 | 1789.5–1803.3 | 7168.9 / 1739.8 | 345–348 | 4.14–4.17 | 4.11x–4.35x | 1.01–1.02 | 1.00–1.01 | `mmh_r1_u1_s4` | kernel |
|  | 13 | 7459.4–7718.7 | 1776.9–1791.5 | 7118.5 / 1732.6 | 347–350 | 4.52–4.55 | 4.16x–4.34x | 0.98–0.99 | 1.00–1.02 | `mmh_r2_u1_s8` | kernel |
|  | 14 | 7577.7–7631.3 | 1793.9–1815.3 | 7197.6 / 1744.0 | 343–347 | 4.80–4.86 | 4.20x–4.22x | 1.01–1.02 | 0.99–1.00 | `mmh_r2_u1_s8_h1` | kernel |
|  | 15 | 7629.9–7633.4 | 1815.6–1900.6 | 7177.1 / 1733.1 | 327–343 | 4.91–5.14 | 4.02x–4.20x | 1.00–1.01 | 0.96–1.04 | `mmh_r2_u1_s8_f1_h1` | kernel |
|  | 16 | 7193.8–7497.4 | 1755.8–1830.5 | 7166.7 / 1741.1 | 340–354 | 5.44–5.67 | 3.93x–4.27x | 1.00–1.02 | 0.99–1.01 | `mmp_r1_u1_s8` | kernel |
|  | 24 | 8194.2–8698.9 | 3290.7–3294.5 | 7750.5 / 3260.3 | 189 | 4.53–4.54 | 2.49x–2.64x | 0.95–1.00 | 1.02 | `mma_r1_u1_s4` | kernel |
|  | 32 | 8438.8–8452.0 | 4009.4–4040.3 | 7955.7 / 3956.0 | 154–155 | 4.93–4.97 | 2.09x–2.11x | 0.99–1.00 | 0.99–1.02 | `mma_r1_u1_s4` | kernel |
| Qwen3-4B up 2560→9728 | 1 | 153.1–154.9 | 161.4–161.5 | 149.7 / 158.1 | 308–309 | 0.31 | 0.95x–0.96x | 0.99–1.00 | 0.99–1.01 | `dev_r1_u1_s8` | F.linear (1 row) |
|  | 2 | 164.3–199.2 | 164.8–203.4 | 160.5 / 160.8 | 245–302 | 0.49–0.60 | 0.98x–1.00x | 1.00–1.01 | 0.99–1.00 | `dev_r2_u1_s8` | kernel |
|  | 3 | 167.3–181.0 | 165.7–166.0 | 164.6 / 159.2 | 300–301 | 0.90 | 1.01x–1.09x | 1.01–1.03 | 0.97–1.00 | `mmh_r1_u1_s8_h1` | kernel |
|  | 4 | 186.9–195.2 | 159.8–162.4 | 185.4 / 157.6 | 307–312 | 1.23–1.25 | 1.15x–1.22x | 1.00 | 0.99–1.00 | `mmh_r2_u1_s8_f1_h1` | kernel |
|  | 5 | 221.3–307.2 | 164.8–188.0 | 217.5 / 158.9 | 265–302 | 1.32–1.51 | 1.34x–1.63x | 0.99–1.00 | 1.00–1.14 | `mmh_r1_u1_s8_h1` | kernel |
|  | 6 | 252.3–260.7 | 159.6–161.3 | 249.1 / 157.1 | 309–312 | 1.85–1.87 | 1.58x–1.62x | 0.99–1.00 | 1.00–1.01 | `mmh_r2_u1_s8_f1_h1` | kernel |
|  | 7 | 286.3–364.8 | 165.3–175.3 | 285.1 / 160.1 | 284–301 | 1.99–2.11 | 1.73x–2.08x | 0.88–1.00 | 0.99–1.22 | `mmh_r2_u1_s2` | kernel |
|  | 8 | 327.6–369.6 | 162.1–225.6 | 325.2 / 157.6 | 221–307 | 1.77–2.46 | 1.64x–2.02x | 0.97–1.00 | 0.86–0.98 | `mmh_r1_u1_s8_f1_h1` | kernel |
|  | 9 | 390.8–446.1 | 185.9–196.5 | 361.2 / 162.1 | 254–268 | 2.28–2.41 | 2.10x–2.27x | 0.89–1.00 | 0.98–1.07 | `mmh_r2_u1_s8_f1` | kernel |
|  | 10 | 224.5–231.7 | 190.7–203.7 | 200.9 / 164.8 | 245–261 | 2.45–2.61 | 1.10x–1.22x | 1.07–1.08 | 0.87–1.05 | `mmh_r2_u1_s8_h1` | kernel |
|  | 11 | 220.7–285.0 | 175.7–216.1 | 195.0 / 163.7 | 230–283 | 2.54–3.12 | 1.26x–1.32x | 0.98–1.05 | 0.98–1.04 | `mmh_r1_u1_s8` | kernel |
|  | 12 | 202.2–243.7 | 163.9–180.7 | 188.5 / 161.3 | 276–304 | 3.31–3.65 | 1.23x–1.35x | 0.97–1.00 | 0.98–1.02 | `mmh_r2_u1_s2` | kernel |
|  | 13 | 209.3–296.3 | 198.7–231.1 | 184.6 / 162.2 | 216–251 | 2.80–3.26 | 1.05x–1.28x | 0.96–1.04 | 0.89–0.94 | `mmh_r1_u1_s4` | kernel |
|  | 14 | 261.3–305.5 | 192.5–233.3 | 218.0 / 165.3 | 214–259 | 2.99–3.62 | 1.31x–1.36x | 0.95–0.97 | 0.89–0.96 | `mmh_r2_u1_s8_h1` | kernel |
|  | 15 | 209.6–269.1 | 172.7–239.8 | 201.7 / 163.4 | 208–288 | 3.12–4.33 | 1.12x–1.21x | 0.99–1.08 | 0.97–0.99 | `mmh_r2_u1_s8_f1` | kernel |
|  | 16 | 262.9–278.8 | 199.1–209.6 | 201.3 / 166.6 | 238–250 | 3.80–4.00 | 1.25x–1.40x | 1.06–1.08 | 1.02–1.05 | `mmh_r2_u1_s4_h1` | kernel |
|  | 24 | 279.2–315.8 | 318.8–346.5 | 276.8 / 317.7 | 144–156 | 3.45–3.75 | 0.88x–0.91x | 0.97–1.00 | 1.01–1.05 | `mma_r1_u1_s4` | kernel |
|  | 32 | 277.7–340.1 | 377.3–402.0 | 274.7 / 364.0 | 124–132 | 3.96–4.22 | 0.74x–0.85x | 0.92–1.11 | 1.01–1.07 | `mma_r1_u1_s4` | kernel |
| Llama-7B up 4096→11008 | 1 | 266.9–275.0 | 279.2–281.9 | 265.1 / 274.0 | 320–323 | 0.32 | 0.96x–0.98x | 1.00–1.01 | 1.00–1.01 | `dev_r2_u1_s4` | F.linear (1 row) |
|  | 2 | 331.0–340.7 | 284.7–305.0 | 284.4 / 272.8 | 296–317 | 0.59–0.63 | 1.09x–1.20x | 0.91–1.10 | 1.02–1.07 | `mmh_r1_u1_s8` | kernel |
|  | 3 | 340.2–444.8 | 301.8–302.6 | 301.7 / 272.6 | 298–299 | 0.89–0.90 | 1.12x–1.47x | 0.88–0.98 | 0.95–1.01 | `mmh_r1_u1_s8_f1_h1` | kernel |
|  | 4 | 440.6–472.7 | 279.3–313.9 | 345.5 / 272.4 | 287–323 | 1.15–1.29 | 1.51x–1.58x | 0.99–1.08 | 0.95–1.02 | `mmh_r1_u1_s4_h1` | kernel |
|  | 5 | 448.0–511.1 | 315.2–321.2 | 426.1 / 281.7 | 281–286 | 1.40–1.43 | 1.39x–1.62x | 1.00–1.04 | 0.94–0.96 | `mmh_r2_u1_s8_h1` | kernel |
|  | 6 | 531.9–532.8 | 302.4–313.6 | 503.2 / 274.4 | 288–298 | 1.73–1.79 | 1.70x–1.76x | 1.04–1.24 | 1.00–1.01 | `mmh_r2_u1_s8` | kernel |
|  | 7 | 578.0–582.4 | 307.7 | 537.8 / 280.3 | 293 | 2.05 | 1.88x–1.89x | 0.98–1.02 | 1.00–1.03 | `mmh_r2_u1_s4_h1` | kernel |
|  | 8 | 626.2–661.9 | 279.4–308.8 | 608.5 / 271.7 | 292–323 | 2.34–2.58 | 2.14x–2.24x | 0.96–0.99 | 0.96–1.02 | `mmh_r2_u1_s8_f1` | kernel |
|  | 9 | 698.0–701.6 | 296.8–302.4 | 671.9 / 278.1 | 298–304 | 2.68–2.73 | 2.31x–2.36x | 1.00–1.01 | 1.00–1.19 | `mmh_r2_u1_s8_h1` | kernel |
|  | 10 | 321.0–351.5 | 286.6–315.7 | 314.6 / 278.3 | 286–315 | 2.86–3.15 | 1.11x–1.12x | 1.01 | 1.00–1.01 | `mmh_r2_u1_s8` | kernel |
|  | 11 | 400.7–430.4 | 312.1–331.4 | 313.0 / 278.3 | 272–289 | 2.99–3.18 | 1.28x–1.30x | 0.80–1.01 | 0.99–1.09 | `mmh_r2_u1_s8_f1_h1` | kernel |
|  | 12 | 392.2–447.6 | 335.1–342.0 | 346.8 / 281.5 | 264–269 | 3.16–3.23 | 1.17x–1.31x | 0.98–0.99 | 0.90–1.05 | `mmh_r2_u1_s8_f1_h1` | kernel |
|  | 13 | 403.3–430.1 | 295.3–338.5 | 340.5 / 278.1 | 266–305 | 3.46–3.97 | 1.27x–1.37x | 0.87–0.95 | 0.90–1.02 | `mmh_r1_u1_s4` | kernel |
|  | 14 | 374.0–392.4 | 304.4–308.3 | 343.9 / 280.1 | 293–296 | 4.10–4.15 | 1.21x–1.29x | 1.03 | 1.00–1.04 | `mmh_r1_u1_s8_h1` | kernel |
|  | 15 | 351.5–399.0 | 284.1–323.7 | 340.5 / 274.5 | 279–317 | 4.18–4.76 | 1.23x–1.24x | 1.00–1.12 | 0.98–0.99 | `mmh_r2_u1_s8_h1` | kernel |
|  | 16 | 391.1–396.1 | 295.7–358.2 | 338.4 / 278.9 | 252–305 | 4.03–4.88 | 1.11x–1.32x | 0.93–1.10 | 0.93–0.98 | `mmh_r2_u1_s8_f1` | kernel |
|  | 24 | 1127.3–1156.0 | 574.0–586.9 | 1091.1 / 526.5 | 154–157 | 3.69–3.77 | 1.96x–1.97x | 0.98–0.99 | 0.91–1.00 | `mma_r1_u1_s4_p4` | kernel |
|  | 32 | 1233.5–1250.7 | 663.3–694.7 | 1198.1 / 616.2 | 130–136 | 4.15–4.35 | 1.80x–1.86x | 0.98–1.00 | 0.96–0.97 | `mma_r1_u1_s4` | kernel |

Model-order sum of the 197 linears, ms (float16; per process):

| M | F.linear | kernel (every cell on the kernel) |
|---:|---:|---:|
| 1 | 11.26–11.43 | 12.40–12.77 |
| 2 | 13.25–16.36 | 12.53–14.75 |
| 3 | 14.52–18.46 | 13.18–15.19 |
| 4 | 18.54–19.68 | 14.33–14.81 |
| 5 | 19.91–22.36 | 13.51–15.20 |
| 6 | 20.05–27.65 | 12.48–15.23 |
| 7 | 22.52–27.70 | 13.20–15.38 |
| 8 | 25.16–27.45 | 13.44–13.58 |
| 9 | 27.10–28.75 | 12.99–13.63 |
| 10 | 28.72–30.51 | 14.02–14.50 |
| 11 | 30.42–33.44 | 14.00–18.01 |
| 12 | 30.19–31.36 | 14.17–14.55 |
| 13 | 30.25–34.01 | 13.02–14.58 |
| 14 | 34.22–34.89 | 13.54–14.51 |
| 15 | 36.29–36.90 | 15.20–16.21 |
| 16 | 31.94–31.99 | 15.99–16.38 |
| 24 | 33.38–38.53 | 25.15–28.01 |
| 32 | 35.11–35.28 | 28.66–30.05 |

</details>

24 and 32 rows still use round 1's `mma` choices, which this round did not retune (out of scope). Qwen3-4B up and gate/up remain below `F.linear` there in fp16, as in round 1.

## 7. The tuned table

`TUNED` in `bench/multirow_linear.py` now holds, for every row count from 2 to 16 on all seven shapes and both dtypes, the fastest config of the round-2 sweep (`tune-{dtype}-{shape}.json`, 3 interleaved repeats, 56 candidates in fp16 and 40 in bf16). Rows 1, 24 and 32 keep round 1's entries. Picking the minimum of many noisy medians flatters the winner, so the layer runs re-measured every chosen config against `F.linear` with fresh repeats and both noise arms.

**Fallback rule**, applied to the Qwen3-1.7B shapes at 2 to 16 rows: keep the kernel only if its slowest speedup over all layer processes exceeds 1 plus the largest A/A or B/B deviation in that cell. Six cells fail and now route to `F.linear` (`"linear"` in the table):

| dtype | shape | rows | slowest speedup | noise bound |
|---|---|---:|---:|---:|
| fp16 | gate/up | 2 | 1.01 | 0.05 |
| fp16 | gate/up | 3 | 1.11 | 0.29 |
| fp16 | gate/up | 11 | 1.21 | 0.22 |
| fp16 | down | 2 | 1.04 | 0.11 |
| fp16 | head | 2 | 0.99 | 0.05 |
| bf16 | gate/up | 5 | 1.17 | 0.18 |

The 2-row cells are genuine ties: the kernel and `F.linear` both stream the weight once. The 3, 5 and 11-row cells are probably real gains, at 1.11 to 1.21 at worst and about 1.5 typically, hidden by this round's outside GPU load. The rule is deliberately strict, and a quieter rerun of the layer table would likely restore them. The other 144 cells keep the kernel.

## 8. End to end, Qwen3-1.7B fp16

`bench_multirow.py decode`, two processes (`decode-qwen3-1.7b-fp16-run1.json`, `-run2.json`). Stock is `nn.Linear`'s own forward, not a wrapper. The patched arm sends every linear call with 2 to 16 flattened rows to `pick_config`, and on to the kernel unless the table says `"linear"`. At batch 4, 8, 12 and 16 that is 6,107 kernel calls per 32-token generation, 197 × 31 decode steps; prefill stays on `F.linear`. Arms are shuffled and interleaved, 4 repeats each. The step table is under Result.

| Batch | Greedy token agreement with stock | Patched B vs stock B, teacher-forced, max over rows and 32 steps | Stock's own difference: stock B vs stock 1 on the same row, max (range over rows) | Rows where patched-vs-stock exceeds that row's own stock difference |
|---:|---:|---:|---:|---|
| 1 | 100% | 0 (same kernels) | | |
| 4 | 100% | 0.035 | 0.043 (0.031–0.043) | 1 of 4, by 0.003 |
| 8 | 100% | 0.066 | 0.245 (0.031–0.245) | none |
| 12 | 93.0% (one row) | 0.096 | 0.135 (0.031–0.135) | none |
| 16 | 94.7% (one row) | 0.158 | 0.156 (0.031–0.156) | 5 of 16, by 0.002 to 0.018 |

- **Scale.** Logits reach 31 to 44 in magnitude, where one fp16 step is 0.016 to 0.031. Every excess in the last column is less than one fp16 step at that magnitude.
- **Repeatability.** Both processes gave identical tokens and logit differences. Stock repeats bit for bit.
- **The diverging row.** At batch 12 and 16 it is row 7 at step 2, where stock's top two logits are 0.0156 apart (one fp16 step). It is the same near-tie round 1 found at batch 16. After it, free-running greedy follows a different continuation.
- **Compared with round 1.** At batch 16 the all-row patched-vs-stock maximum (0.158) is larger than round 1's (0.110) but equals stock's own batch-size difference (0.156).

**Linears alone in model order** (stock / patched, two samples per process): batch 4 14.2–14.6 / 13.4–14.8, batch 8 22.8–23.9 / 13.7–15.9, batch 12 27.4–28.5 / 14.1–15.8, batch 16 26.1–27.5 / 14.1–14.3 and 16.2.

## 9. Prediction check

| Prediction | Measured | Held? |
|---|---|---|
| Diagnosis-stage: `mmh` at 80 to 90 µs on gate/up and down at 16 rows | 87 to 95 µs in tuning, probe and sweep processes; 82 to 95 µs fastest layer samples; 87 to 131 µs layer medians under outside load | roughly, at the top of the range |
| `mmp` no better than `mmh` | 5 to 8 µs slower | yes |
| `mmt` 100 to 110 µs | 121 to 128 (gate/up), 167 to 265 (down) | no, worse |
| U = 2 and F = 1 within ±3% | F: yes; U = 2: 5 to 10% slower | partly |
| R ≥ 3 slower | 2.7 to 5 times slower | yes |
| Thread: batch-16 linears at 15 ms or less | 14.1–14.3 ms in one process, 16.2 ms in the other | **not reliably**: one process each side |
| Thread: batch-16 step at 30 ms or less | 27.7–28.5 ms in both processes | **yes** |

## 10. What failed, and what was not tested

- **The 75 to 85 µs target at 9 to 16 rows on gate/up and down** was reached only in single samples (82 to 95 µs). Typical quiet-process medians are 85 to 95 µs. The remaining limit is how fast the kernel streams the weight: about 280 GB/s for every custom kernel here, against `F.linear`'s one-row path at about 300 to 310.
- **The linears prediction**: see 9.
- **Six cells** fall back to `F.linear` under a strict noise rule (section 7).
- **More weight rows per simdgroup** (R ≥ 3) is blocked by a register cliff.
- **`mmt`** (staged, double-buffered) is correct but rarely useful.
- **The layer table's absolute times** for the 25 MB shapes are inflated by outside GPU load in all four processes. Speedups, fastest samples and the tuning and probe processes are the more reliable figures.
- **Not tested.** 17 to 23 rows and 25 to 31 rows (they use the 24 and 32-row entries; covered by correctness only). bf16 end to end. Qwen3-4B and Llama-7B end to end. Speculative-decoding verification. Bias timing (Qwen3 has none). Other chips. The pytest file under torch 2.15.0.

## 11. Files

In this folder:

- `README.md` (this file)
- Diagnosis: `peak_mma.py`, `peak-mma.json`, `probe-a.json` (its peak section is superseded), `probe-b.json`, `probe-c.json`, `probe-d.json`
- Designs: `design-sweep-1.json`, `design-sweep-2.json`, `design-sweep-3.json`
- Tuning: `tune-{float16,bfloat16}-{q17_qo,q17_kv,q17_up,q17_down,q17_head,q4b_up,llama7b_up}.json`
- Correctness: `correctness-random.json`, `correctness-real.json`, `correctness-planted.json`
- Layer table: `layers-run{1,2,3,4}-{float16,bfloat16}-a.json`, `layers-run{1,2}-{float16,bfloat16}-b.json`, `summarize.py`, `summary.json`, `tables.md`
- End to end: `decode-qwen3-1.7b-fp16-run1.json`, `decode-qwen3-1.7b-fp16-run2.json`

Code: `bench/multirow_linear.metal`, `bench/multirow_linear.py`, `bench/bench_multirow.py`, `tests/test_multirow_linear.py`.

Reproduce: every command goes through `bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10 -- <python> bench/bench_multirow.py <subcommand>`. The subcommands are `probe`, `tune --grid round2 --dtype D --shapes S --rows 2,...,16`, `correctness --parts random|real|planted`, `layers --dtypes D --shapes S --seed N` and `decode --seed N`. `decode` and `correctness --parts real` need `HF_HOME` pointing at a cache with Qwen/Qwen3-1.7B (revision 70d244cc) and `HF_HUB_OFFLINE=1`.

