# Round 2 fixes: reproducible tuning, a robust table, no lost cells

2026-11-10, M1 Max, PyTorch 2.15.0 (scratch environment) for every GPU timing and correctness run, the repository's `.venv` (torch 2.14.1) for pytest. Every GPU process ran through `bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10`. This answers `../REVIEW.md`: its must-fix (1) and should-fix items 2 and 3. Nothing is committed.

## Result

Nothing here needs a decision. One judgment call is stated under section 3 so it can be overruled.

- **The tuning command runs again and rebuilds the recorded grid.** Round 1's table is now a literal in `bench/bench_multirow.py`. `tune --grid round2` builds exactly the 56 fp16 and 40 bf16 configs recorded in all fourteen `../tune-*.json` files, and the README's reproduce command ran through the lock for k/v in fp16.
- **The tuned table now comes from one rule for every shape and dtype.** At 2 to 16 rows each cell takes one of six plain configs, chosen per (dtype, shape, row band) from time pooled over the whole band. At 9 to 16 rows that is always R = 2, U = 1, S = 4 or 8. 168 of the 210 cells changed.
- **No cell routes to `F.linear` any more.** The fallback rule now sends a cell to `F.linear` only when the kernel is slower beyond the noise in every process. Under it, all six former `"linear"` cells return to the kernel. All 210 cells at 2 to 16 rows on all seven shapes beat `F.linear` in every process that timed them, by 1.01x to 4.8x. The slowest is the head at 2 rows, 1.01x to 1.02x.
- **The 16-row MLP cells improve as the review predicted.** In fp16 at 16 rows, gate/up is 9 to 13% faster than the previous R = 1 pick in the same processes, and runs 1.48x to 1.53x over `F.linear`. Down runs 1.70x.
- **Correctness holds.** The float64 suite covers every config in the final table, in fp16 and bf16, with and without bias: 0 failures in 34,526 kernel runs. 158 planted bugs were all caught, including 140 that mutate the family of each of the table's 18 configs. pytest: 115 passed.
- **End to end, Qwen3-1.7B fp16 decode (`model` subcommand, two processes).** Batch 16 runs 1.49x to 1.50x over stock, batch 12 1.53x to 1.55x and batch 8 1.34x to 1.37x. The 197 linears alone at batch 16 took 13.8 to 14.8 ms in all four samples, inside the thread's 15 ms line.

## What changed

- `bench/bench_multirow.py`
  - `ROUND1_TUNED` is the round-1 table copied from commit `a441c78` as a literal. Before, it was a copy of the live `TUNED` taken at import.
  - `tune_grid_round2` skips `"linear"` entries. `all_variants` already did, and `pick_config` maps them to `None`, so no table entry reaches `Config.parse` any more.
  - `layers` has two new options. `--cells` times only the listed cells. `--compare` adds an arm that times another config in the same cell, here the config the table held before.
  - `correctness` has a new part, `planted_table`. It applies each family's mutations to every config in the tuned table: fp16 and bf16, with and without bias, at the smallest and largest row count the config is used at.
  - The stray duplicated comment in `cmd_probe` is gone (review finding 8).
- `bench/multirow_linear.py`: `TUNED` at rows 2 to 16 is rebuilt by the rule below, and its comment now describes that rule. Rows 1, 24 and 32 are unchanged.
- `tests/test_multirow_linear.py`: two new tests. Every table entry parses or is `"linear"`. The round-2 grid is built from the frozen round-1 table, with the recorded sizes (56 and 40), even when the live table holds a `"linear"` cell.
- `bench/multirow_linear.metal`: unchanged.

## 1. The tuning reproduces (must-fix 1)

- **Static check.** `tune_grid_round2("float16")` and `tune_grid_round2("bfloat16")` return config lists identical, in order, to the `configs` field of all seven `../tune-float16-*.json` and seven `../tune-bfloat16-*.json` files.
- **The README's command, run.** `tune --grid round2 --dtype float16 --shapes q17_kv --rows 2,...,16` ran through the lock in 46 s and timed all 56 configs at 15 row counts (`tune-repro-float16-q17_kv.json`). Before the fix, the review saw this command fail with `KeyError: 'r'`.
- **What the rerun says about the old picks.** Only 4 of the 15 cells chose the same winner as the original sweep. The original winner ranked as low as 19th of 56 in the rerun, up to 21% slower than the new winner. Per-cell sweep winners do not repeat, which is review finding 3 seen directly.

## 2. How the table is chosen now (should-fix 3)

`select_table.py` writes `selection.json`. The rule:

| Row band | Candidates (all U = 1, no prefetch, no K split, fp32 fragment operands) |
|---|---|
| 2 to 3 | `dev_r2_u1_s4`, `dev_r2_u1_s8`, `mmh_r1_u1_s4`, `mmh_r1_u1_s8`, `mmh_r2_u1_s4`, `mmh_r2_u1_s8` |
| 4 to 8 | `mmh_r1_u1_s4`, `mmh_r1_u1_s8`, `mmh_r2_u1_s4`, `mmh_r2_u1_s8` |
| 9 to 16 | `mmh_r2_u1_s4`, `mmh_r2_u1_s8` |

Each (dtype, shape, band) takes one config: the candidate with the lowest geometric mean, over every cell in the band, of its median time relative to the other candidates in the same cell. The data is every streamed, interleaved measurement that timed all of a band's candidates in one cell: the round-2 sweeps, this round's k/v rerun, and the review's config scans of gate/up and down.

Why this set:

- **R = 2 at 9 to 16 rows.** This is the review's evidence. Cache-resident, R = 2 takes 65 to 67 µs at 16 rows, while R = 1 needs about 75 µs, close to the weight stream itself. Streamed, every R = 2 config beat the R = 1 pick by 1 to 13%.
- **No R = 1 at 9 to 16 rows. Within one band, choose only between S = 4 and S = 8.** The pooled sweeps put S = 4 and S = 8 within 0 to 12% of each other. The pooled score settles which one each shape gets, from 8 to 16 cells per band instead of one.
- **No prefetch (F), half operands (H), U = 2, K split, `mmp` or `mmt`.** Round 2 found each of these no better than plain `mmh` on average: within noise for F and H, slower for U = 2, `mmp` and `mmt` (README section 2). Every one is another way for a per-cell choice to pick noise.
- **`dev` only at 2 and 3 rows.** Below 4 rows on the small shapes, the per-lane `dev` kernel is as fast as `mmh` (pooled within 0 to 3%). Its cost grows with the row count, so it is not a candidate above 3 rows.

The resulting table holds six configs at 2 to 16 rows. Of the 210 cells, 168 changed. Across the re-timed changed cells, the new pick ran between 0.4% slower and 1% faster than the previous pick timed in the same process (geometric mean per dtype and band). The headline cells gained, for example fp16 gate/up at 16 rows (+9 to 13%) and fp16 down at 12 rows (+8 to 12%, previously `mmp`).

## 3. The fallback rule, one rule for every shape (should-fix 2)

`summarize_fixes.py` applies this rule to all 210 cells (2 to 16 rows, seven shapes, two dtypes). A cell goes to `F.linear` only when the kernel's largest speedup over the processes that timed it is below 1 − noise. Noise is the largest deviation of any A/A or B/B ratio from 1 in those processes. In words: the kernel must be slower beyond the noise in every process. A tie keeps the kernel, because a tie costs nothing and the 2- and 3-row ties sit where speculative decoding verifies.

- Changed cells and every 2- and 3-row cell are judged on this round's re-time. Each of the 35 unchanged cells is judged on round 2's own layer runs of the same config, from two to four processes.
- **No cell meets the rule** (`linear_cells.json` is empty). The table holds no `"linear"` entry at 2 to 16 rows.

**The six formerly routed cells**, re-timed with `F.linear` interleaved against the kernel and both noise arms:

| Cell | Config now | Processes | `F.linear` µs | kernel µs | fastest, `F.linear` / kernel | speedup | A/A | B/B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 down, 2 rows | `dev_r2_u1_s8` | 3 | 97.6–99.0 | 88.8–90.8 | 90.0 / 85.7 | 1.09–1.10x | 0.99–1.10 | 0.99–1.02 |
| fp16 gate/up, 2 rows | `mmh_r2_u1_s8` | 3 | 87.4–95.6 | 84.7–93.0 | 83.2 / 82.1 | 1.03x | 0.98–1.01 | 0.99–1.00 |
| fp16 gate/up, 3 rows | `mmh_r2_u1_s8` | 3 | 104.0–122.1 | 87.6–96.6 | 94.4 / 84.4 | 1.19–1.26x | 0.98–1.01 | 0.96–1.02 |
| fp16 gate/up, 11 rows | `mmh_r2_u1_s4` | 2 | 170.0–183.0 | 112.8–135.3 | 147.0 / 97.0 | 1.35–1.51x | 1.06–1.10 | 1.00–1.04 |
| fp16 head, 2 rows | `mmh_r1_u1_s8` | 3 | 1794–1811 | 1752–1792 | 1758 / 1710 | 1.01–1.02x | 0.99–1.00 | 0.98–1.02 |
| bf16 gate/up, 5 rows | `mmh_r1_u1_s4` | 2 | 107.0–110.5 | 85.5–90.1 | 104.8 / 84.0 | 1.23–1.25x | 0.97–1.03 | 0.98–1.01 |

`F.linear` on down at 2 rows sat at about 98 µs in all three processes this time, its fast state. The kernel still won by 9 to 10%.

**Judgment call, stated so it can be overruled.** In six bf16 cells the previous config was faster than the new one in every process: down at 12 rows by 5 to 8% (`mmh_r2_u1_s8_f1`), Llama-7B up at 13 and 14 rows by 2 to 3% (`mmp_r1_u1_s8`), and the head at 3, 4 and 7 rows by 1 to 2%. I kept the rule's choice rather than add exceptions, for two reasons. First, the comparison arm is itself noisy. In the eight cells where the previous config was the same as the new one, the two arms of identical code disagreed by up to 9% (0.91 to 1.02). Second, exceptions would bring back exactly the per-cell picks this fix removes. None of the six cells is fp16, and the end-to-end model is fp16.

## 4. The re-timed cells

- **Which cells.** Every changed cell at 4 to 16 rows: 147 cells, two processes per dtype, 7 interleaved repeats. Every 2- and 3-row cell, changed or not: 28 cells, three processes, 15 repeats. Each cell has two `F.linear` arms, two kernel arms, and the previous config as a fifth arm. Weights rotate through at least 256 MB of copies, as in round 2. Raw data is in `retime-4to16-{float16,bfloat16}-p{1,2}.json` and `retime-2to3-p{1,2,3}.json`. The full table is in `tables.md`, and every cell's verdict is in `summary.json`.
- **Lock contention.** No process waited on the lock. Each ran in about 35 to 60 s.

**2 and 3 rows**, the verification sizes. Every cell beats `F.linear` in all three processes:

| 2 / 3 rows | q/o | k/v | gate/up | down | head | Qwen3-4B up | Llama-7B up |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16, speedup range | 1.08–1.15 / 1.19–1.30 | 1.14–1.25 / 1.25–1.27 | 1.03 / 1.19–1.26 | 1.09–1.10 / 1.28–1.33 | 1.01–1.02 / 1.02–1.03 | 1.03–1.04 / 1.05–1.10 | 1.02–1.06 / 1.11 |
| bf16, speedup range | 1.39–1.53 / 1.52–1.63 | 1.44–1.53 / 1.51–1.79 | 1.23–1.26 / 1.35–1.53 | 1.38–1.53 / 1.49–1.69 | 1.01–1.02 / 1.16–1.17 | 1.05–1.12 / 1.05–1.11 | 1.07–1.10 / 1.16–1.19 |

- On the large fp16 shapes (gate/up, head, Qwen3-4B up, Llama-7B up), the 2-row gain is small, 1 to 6%, because both paths stream the weight once.
- Qwen3-4B up at 2 rows in fp16 (`dev_r2_u1_s8`) is one of the cells review finding 2 flagged as kept without clearing the old rule. Its round-2 range was 0.98x to 1.00x. Here it is 1.03x to 1.04x in all three processes, with noise of 0.014.

**The MLP projections at 8 to 16 rows, fp16.** Down at 13 rows kept its config, so it is judged on round 2's four layer processes.

| Rows | gate/up config | gate/up speedup | gate/up kernel µs (fastest) | previous / new | down config | down speedup | down kernel µs (fastest) | previous / new |
|---:|---|---:|---:|---:|---|---:|---:|---:|
| 8 | `mmh_r2_u1_s4` | 1.90–2.05x | 95.0–99.0 (89.3) | 0.95–0.99 | `mmh_r1_u1_s8` | 1.99–2.06x | 93.2–96.6 (86.0) | 1.00 |
| 9 | `mmh_r2_u1_s4` | 2.15–2.22x | 92.5–95.6 (85.3) | 1.08–1.13 | `mmh_r2_u1_s4` | 2.19–2.21x | 91.7–95.3 (87.0) | 1.00 |
| 10 | `mmh_r2_u1_s4` | 1.59–1.63x | 94.4–112.6 (88.5) | 1.03–1.10 | `mmh_r2_u1_s4` | 2.51–2.53x | 89.0–89.6 (86.1) | 0.99–1.00 |
| 11 | `mmh_r2_u1_s4` | 1.35–1.51x | 112.8–135.3 (97.0) | was `F.linear` | `mmh_r2_u1_s4` | 2.51–2.71x | 89.8–102.4 (88.1) | 0.95–0.99 |
| 12 | `mmh_r2_u1_s4` | 1.49–1.53x | 102.9–111.3 (91.3) | 0.96–1.02 | `mmh_r2_u1_s4` | 2.83–2.94x | 89.5–93.2 (86.8) | 1.08–1.12 |
| 13 | `mmh_r2_u1_s4` | 1.43–1.55x | 105.0–107.8 (91.1) | 1.01–1.02 | `mmh_r2_u1_s4` | 2.64–3.17x | 96.1–114.6 (86.1) | unchanged |
| 14 | `mmh_r2_u1_s4` | 1.41–1.50x | 89.6–112.4 (87.8) | 0.99–1.00 | `mmh_r2_u1_s4` | 3.04–3.25x | 91.7–103.0 (87.0) | 1.00–1.02 |
| 15 | `mmh_r2_u1_s4` | 1.47–1.53x | 99.0–110.1 (92.6) | 0.99–1.00 | `mmh_r2_u1_s4` | 3.53–3.55x | 88.6–89.4 (85.8) | 1.03 |
| 16 | `mmh_r2_u1_s4` | 1.48–1.53x | 92.9–115.8 (89.4) | 1.09–1.13 | `mmh_r2_u1_s4` | 1.70x | 88.7–97.3 (86.6) | 0.99 |

"previous / new" is the previous table's config over the new one, timed in the same process. Above 1 means the new config is faster.

- gate/up at 16 rows now clears the thread's 1.3x line in both processes, at 1.48x to 1.53x. In round 2's layer runs it was the one cell that fell below 1.3x in one process (1.11x).
- Absolute medians on gate/up still swing by 10 to 25% between processes, as they did in round 2. The interleaved ratios and fastest samples are the steadier figures. Down at 9 to 16 rows was steady, at 86 to 88 µs fastest.

**16 rows, every shape** (speedup range over processes): fp16 q/o 2.03–2.09x, k/v 2.78–2.84x, gate/up 1.48–1.53x, down 1.70x, head 4.10–4.20x, Qwen3-4B up 1.22–1.29x, Llama-7B up 1.28x. bf16: 2.24–2.30x, 3.23–3.24x, 1.75x, 2.09–2.32x, 4.54–4.80x, 1.60–1.72x, 1.60–1.68x.

## 5. Correctness

Same criterion as rounds 1 and 2. Against a float64 CPU reference from the same 16-bit inputs, each config's largest error divided by `sum_k |x||w| + |b|` must not exceed `F.linear`'s by more than `u + K·2^-24`. Its relative L2 error must not exceed 1.25 times `F.linear`'s plus `sqrt(K)·2^-24`.

| Set | Scope | Cases | Kernel runs | Result |
|---|---|---:|---:|---|
| Random (`correctness-random.json`) | (K, N) = (2056, 1027), (264, 33), (8, 5) at every M from 1 to 32, (4104, 4099) at M 1, 2, 7, 16, 17, 32. Every round-1 variant, round 2's 15 path-covering variants, and all 18 configs in the final table. fp16 and bf16, with and without bias. | 412 | 28,912 | 0 failures. Largest error ratio to `F.linear` 1.005. 59% of runs bitwise equal to `F.linear`. |
| Real Qwen3-1.7B weights (`correctness-real-planted.json`) | Layer 0 q, k (with and without bias), up and down at M 1 to 32, and the tied head on 8,192 rows. Every variant plus the tuned choice. | 94 | 5,614 | 0 failures. Largest ratio 1.003. |
| Planted bugs, round 2's set | 18 mutations across all families | 18 | | all caught |
| Planted bugs on every table config (`planted_table`) | Each of the 18 configs in the final table, under its family's mutations (`mmh`: skip a K chunk, and drop the last partial chunk, 8 of 2056 K elements). fp16 and bf16, with and without bias, at the smallest and largest row count the config is used at. | 140 | | all caught |

- **Coverage.** Every final-table config ran in both dtypes, with and without bias, in both the random and real sets: 72 of 72 (config, dtype, bias) combinations each.
- **The weakest catch.** The subtle tail-drop mutation lands 1.41 to 18.9 times above its allowance (`F.linear`'s error plus `u + K·2^-24`). The 1.41 is a bf16 case.
- **pytest.** `tests/test_multirow_linear.py` on the repository's `.venv` (torch 2.14.1): 115 passed, 113 existing and 2 new.

## 6. End to end, Qwen3-1.7B fp16

`bench_multirow.py model --batches 4,8,12,16`, two processes (`model-qwen3-1.7b-fp16-p{1,2}.json`). The run used 32-token prompts, 32 new tokens and greedy decoding. Each process ran 3 interleaved, shuffled repeats per arm.

- **Stock arm.** Here stock is the same wrapper with the patch disabled, which calls `F.linear`. Round 2's own table used the `decode` subcommand, whose stock arm is `nn.Linear`'s forward. The two harnesses are close but not identical.
- **Kernel calls.** The patched arm made 6,107 kernel calls per generation at every batch size: no cell is routed away.

| Batch | stock ms/step (all 6 samples) | patched ms/step | speedup of medians, per process | linears alone, stock / patched ms (4 samples) | round 2, `decode` subcommand |
|---:|---:|---:|---:|---:|---:|
| 4 | 26.4–27.2 | 25.0–26.8 | 1.03x, 1.06x | 14.2–14.6 / 13.4–14.0 | 1.03–1.04x |
| 8 | 35.1–38.2 | 25.7–28.2 | 1.34x, 1.37x | 22.6–23.4 / 13.7–14.4 | 1.33–1.35x |
| 12 | 41.1–44.6 | 26.6–28.0 | 1.55x, 1.53x | 27.2–27.4 / 13.8–14.0 | 1.50–1.51x |
| 16 | 41.0–43.0 | 27.1–29.4 | 1.49x, 1.50x | 26.1–28.0 / 13.8–14.8 | 1.47–1.48x |

- **The batch-16 linears at 15 ms or less now holds.** All four patched samples across both processes were 13.8 to 14.8 ms. In round 2, two of four samples were 16.2 ms.
- **The batch-16 step stays under 30 ms.** It was 27.1 to 29.4 ms in all six patched samples.

**Tokens and logits.**

- **Repeatability.** Both processes produced identical tokens and logit differences, and stock repeated bit for bit.
- **Batch 4.** Greedy agreement is 100%. The teacher-forced patched-against-stock maximum is 0.035, the same as round 2.
- **Batch 8, 12 and 16.** In each, one row diverges: row 7 at step 2, the same prompt each time. Token agreement is 89.5%, 93.0% and 94.7%. At that step, stock's top two logits are 0.0156 apart, one fp16 step, and the patched top two are an exact tie. This is the near-tie that rounds 1 and 2 reported at batch 12 and 16. It now also flips at batch 8, where round 2 kept 100%.
- **Teacher-forced logit differences** (patched against stock on stock's tokens, maximum over rows and 32 steps): 0.109 at batch 8, 0.090 at batch 12 and 0.078 at batch 16. Logits reach 43 in magnitude, where one fp16 step is 0.031.
  - Against stock's own whole-batch against two-halves spread at batch 12 and 16 (0.164 and 0.178), the patch is smaller.
  - At batch 8, stock's two halves agree exactly with the whole batch, so that yardstick is 0. Round 2 measured stock at batch 8 against stock at batch 1 at up to 0.245, and the review found the patch closer to a float32 reference than stock at batch 8.
  - I did not rerun the float32 reference.

## 7. Not done here

- **Review finding 4.** `bench_verify_cost.py`, `bench_prompt_lookup.py` and `bench_decode_compile_kernel.py` import this table. At 2 and 3 rows they now run the kernel on fp16 gate/up, down and the head, where the committed round-2 table sent those cells to `F.linear`. Their committed numbers need a rerun with this table.
- **Review findings 5 to 7** (README wording on the outside workload, the linears prediction, the float32 logit reference) concern round 2's README. I did not edit it.
- **Not tested.** bf16 end to end, other models end to end, and other chips. Rows 1, 24 and 32 are unchanged and were not re-timed. The six bf16 cells in the judgment call above keep the rule's choice.

## Files

In this folder:

- `README.md`: this file.
- Selection: `select_table.py` → `selection.json`.
- Re-time inputs: `cells-4to16.json`, `cells-2to3.json`, and `compare-previous.json` (the previous table, or for formerly routed cells the config round 2 timed there).
- Re-time results: `retime-4to16-{float16,bfloat16}-p{1,2}.json` and `retime-2to3-p{1,2,3}.json`. `summarize_fixes.py` turns them into `summary.json`, `tables.md` and `linear_cells.json`.
- Tuning reproduction: `tune-repro-float16-q17_kv.json`.
- Correctness: `correctness-random.json` and `correctness-real-planted.json`.
- End to end: `model-qwen3-1.7b-fp16-p1.json` and `model-qwen3-1.7b-fp16-p2.json`.

Reproduce. Every GPU command goes through `bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10 -- <python> bench/bench_multirow.py <subcommand>`:

- `layers --dtypes D --cells fixes/cells-4to16.json --compare fixes/compare-previous.json --seed N`
- `layers --dtypes float16,bfloat16 --cells fixes/cells-2to3.json --compare fixes/compare-previous.json --reps 15 --seed N`
- `correctness --parts random`
- `correctness --parts real,planted,planted_table`
- `model --batches 4,8,12,16`

`correctness --parts real` and `model` need `HF_HOME` pointing at a cache with Qwen/Qwen3-1.7B (revision 70d244cc) and `HF_HUB_OFFLINE=1`. `select_table.py` and `summarize_fixes.py` need no GPU.
