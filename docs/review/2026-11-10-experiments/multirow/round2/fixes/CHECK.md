# Check of the round-2 fixes

2026-11-10. Checks that the fixes described in `README.md` in this folder answer `../REVIEW.md`'s must-fix (1) and should-fix items 2 and 3. I read the diff of the three bench and test files against commit `9293a94`, this folder's README, `select_table.py`, `summarize_fixes.py` and the raw re-time files. I then reran the tuning command, correctness on every config in the final table, pytest, and decode at batch 8 and 16, each through `bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10` with no lock wait. GPU work used PyTorch 2.15.0 (scratch environment); pytest used the repository's `.venv` (torch 2.14.1). My scripts and output are in `check/`. I did not edit the builder's files or the kernel.

## Verdict

The fixes hold. Nothing here is fatal and nothing must be fixed before committing.

- The tuning command runs from the code and rebuilds the recorded grid in both dtypes.
- The frozen round-1 table is identical to commit `b64a958`'s.
- The new table follows the stated rule in every one of its 210 cells at 2 to 16 rows.
- All six former `"linear"` cells are back on the kernel, and the re-time data supports each one.
- Every config in the final table is correct in both dtypes, with and without bias, on edge shapes and on every tuned shape.
- The batch-8 and batch-16 decode speedups reproduce in fresh processes.

One claim does not reproduce. The README says that the 197 linears at batch 16 now take 15 ms or less ("now holds"). In my two fresh processes, three of four samples were above it, at 15.6 to 15.9 ms. That claim should be restated as a range before it goes into `docs/findings.md`. The rest are wording points.

| Severity | Count |
|---|---:|
| fatal | 0 |
| must-fix | 0 |
| should-fix | 1 |
| minor | 3 |

## Should-fix

### 1. "The batch-16 linears at 15 ms or less now holds" does not hold in fresh processes

The README's result list and section 6 say that all four patched samples were 13.8 to 14.8 ms, and contrast this with round 2. I ran two fresh processes: `decode`, round 2's harness, and `model`, the builder's harness. Both time the linears with the same `linears_ms` function.

| Process | Batch-16 linears, patched ms | Batch-8 linears, patched ms |
|---|---|---|
| Builder, `model`, 2 processes | 13.8–14.8 (4 samples) | 13.7–14.4 |
| Checker, `decode` (`check/decode-b8-b16.json`) | 15.59, 15.90 | 16.03, 14.98 |
| Checker, `model` (`check/model-b8-b16.json`) | 15.73, 14.50 | 14.49, 15.28 |

Across the builder's and my processes, 5 of 8 batch-16 samples are at or under 15 ms. That is about the same rate as round 2 and the review (7 of 10). The step times did not move, so this is noise in the linears-alone timing, not a regression. Fix: report the range (13.8 to 15.9 ms over eight samples in four processes) and drop "now holds" and "which round 2 could not".

## Minor

2. **The judgment call's criterion is not stated.** The README says the previous config was faster "in every process" in six bf16 cells, and that none of those cells is fp16. The six come from a stricter test: the previous config was faster beyond that cell's noise (A/A and B/B) in every process. By the plain reading, faster in every process, 45 re-timed cells qualify. Twenty-four of them are fp16, and seventeen are faster by more than 1% in every process, seven of those fp16 (for example fp16 k/v at 16 rows, 0.981 and 0.989). None of this changes the conclusion. Identical arms disagree by up to 9% (0.906 to 1.023 over 23 process-cells, which I confirmed), and the new table is on balance equal or faster: previous over new has a geometric mean of 1.005 and a median of 1.002 over 355 process-cells. State the criterion in one clause.
3. **The 2 to 3 row picks rest on few cells.** The README says pooling uses "8 to 16 cells per band instead of one". That holds at 9 to 16 rows. At 4 to 8 rows it is 5 cells, or 10 for k/v. At 2 to 3 rows it is 2 cells of 3 repeats each, or 4 for k/v. There, the runner-up was within 1% in 7 of 14 (dtype, shape) pairs, so the 2 to 3 row choices are still close to coin tosses among near-equal configs. That is harmless here. The re-time with 15 repeats in three processes shows every 2 to 3 row cell beating `F.linear`, and the new pick is not slower than the previous one (geometric mean 1.008 fp16, 0.999 bf16). The README should give the real cell counts.
4. **Docs to update at commit.** The committed `docs/findings.md` entry says the tuning issues "are being fixed". The round-2 `README.md` still describes six `"linear"` cells and the per-cell sweep winners as the table. The re-runs from review finding 4 are still pending, as the README says. None of this blocks the commit, but the findings entry should be updated in the same commit.

## What I checked

### Must-fix 1: the tuning reproduces

`check/check_static.py` → `check/check-static.json`, no GPU:

- `ROUND1_TUNED` equals the `TUNED` literal parsed from `git show b64a958:bench/multirow_linear.py`. `9293a94` is the only commit after `b64a958` that touches that file, so this is the table round 2's grid was built from.
- `tune_grid_round2` returns 56 fp16 and 40 bf16 configs. They are identical, in order, to the `configs` field of all 14 `../tune-*.json` files and of `tune-repro-float16-q17_kv.json`.
- `all_variants`, `planted_table` and `layers --compare` skip `"linear"` before `Config.parse`, and `pick_config` maps it to `None`. No other caller parses table entries.

On the GPU, I ran `tune --grid round2 --dtype bfloat16 --shapes q17_down --rows 2,9,16`, a dtype and shape the builder did not rerun (`check/tune-repro-bfloat16-q17_down.json`). It timed all 40 configs plus `F.linear`, its grid matches `../tune-bfloat16-q17_down.json`, and it took 6 s. At 9 and 16 rows its winner was `mmh_r2_u1_s4`, the table's pick for that band.

Differential check: the two new pytest tests run against the `9293a94` code fail with `KeyError: 'r'` (1 failed, 114 passed). They pass on the fixed code.

### Should-fix 3: the table comes from one rule

- `TUNED` at 2 to 16 rows equals `selection.json` in all 210 cells. Rows 1, 24 and 32 are unchanged from `9293a94`. There are no `"linear"` entries.
- Each (dtype, shape, band) holds one config. Every config is inside its band's candidate set, and every 9 to 16 row cell has R = 2. The table holds 6 configs at 2 to 16 rows and 18 overall. 168 of 210 cells changed.
- I re-derived the band choice with a reference-free score (each candidate's log time relative to the cell's mean over candidates). It picks the same config in all 42 (dtype, shape, band) groups.
- The re-time files recorded the final table (all 7 files). Every changed cell at 4 to 16 rows has 2 processes and 7 repeats. Every 2 to 3 row cell has 3 processes and 15 repeats. Each timed the final config, and its compare arm is the `9293a94` config (for former `"linear"` cells, the config round 2 timed there). There are 175 cells in all, with no gaps.
- Headline cells: fp16 gate/up at 16 rows is 1.48x and 1.53x over `F.linear`, and the previous config over the new is 1.09 and 1.13. fp16 down at 12 rows: previous over new is 1.12 and 1.08. fp16 down at 16 rows: 1.70x. These match the README.

### Should-fix 2: no cell lost to `F.linear`, one rule for all shapes

- All 355 re-timed process-cells have the kernel's median below `F.linear`'s. The lowest is bf16 head at 2 rows, 1.009x. Only one process-cell drops below 1 under the worst pairing of noise arms: fp16 head at 2 rows, 0.998, a tie the rule keeps on the kernel.
- The 35 unchanged cells at 4 to 16 rows are judged on round 2's layer runs of the same config, 2 to 4 processes each. Their lowest speedup is 1.11x.
- The six former `"linear"` cells, from the raw files:

| Cell | Config | Speedup per process | A/A | B/B |
|---|---|---|---|---|
| fp16 down, 2 rows | `dev_r2_u1_s8` | 1.090, 1.087, 1.100 | 0.99–1.10 | 0.99–1.02 |
| fp16 gate/up, 2 rows | `mmh_r2_u1_s8` | 1.028, 1.029, 1.033 | 0.98–1.01 | 0.99–1.00 |
| fp16 gate/up, 3 rows | `mmh_r2_u1_s8` | 1.187, 1.261, 1.263 | 0.98–1.01 | 0.96–1.02 |
| fp16 gate/up, 11 rows | `mmh_r2_u1_s4` | 1.352, 1.508 | 1.06–1.10 | 1.00–1.04 |
| fp16 head, 2 rows | `mmh_r1_u1_s8` | 1.011, 1.024, 1.022 | 0.99–1.00 | 0.98–1.02 |
| bf16 gate/up, 5 rows | `mmh_r1_u1_s4` | 1.251, 1.225 | 0.97–1.03 | 0.98–1.01 |

- The review's six other-shape cells that failed the old rule all clear 1 in every process now. Examples: fp16 Qwen3-4B up at 2 rows 1.03 to 1.04x; at 3 rows, now `dev_r2_u1_s8`, 1.05 to 1.10x.
- No sample in the re-time implies more than 360 GB/s, so the weights stream rather than sitting in cache.

### Correctness on every table config

| Check | Scope | Result |
|---|---|---|
| `check/check_table.py` (new; the review's criterion and view helpers) | All 18 configs in the final `TUNED`, every M from 1 to 32, (K, N) = (8, 1), (24, 7), (264, 33), (520, 517), (2056, 1027). x and W as aligned views at a nonzero storage offset; bias at an odd offset. With and without bias. fp16 and bf16. | 5,760 runs per dtype, 0 failures. Largest error ratio to `F.linear` 1.005 (fp16) and 1.000 (bf16). |
| The review's `tuned` part, rerun on the new table | `pick_config` on every tuned (K, N), M 1 to 32 (head 1 to 16), with bias, real shapes | 208 cells per dtype, all on the kernel, 0 failures |
| The review's `routing` part | `pick_config` for M 1 to 32 on every tuned shape | No `"linear"` anywhere; every M above 16 goes to `mma` |
| Builder's `correctness --parts real,planted,planted_table`, rerun | Real Qwen3-1.7B weights and all planted bugs | 94 real cases, 0 failures. All 158 planted bugs caught, including 140 on the 18 table configs. |
| pytest, repository `.venv` | `tests/test_multirow_linear.py` | 115 passed |

### End to end, Qwen3-1.7B fp16, fresh processes

| Batch | Harness | Stock ms/step (median) | Patched ms/step (median) | Speedup | Builder |
|---:|---|---:|---:|---:|---:|
| 8 | `decode` | 36.35 | 27.07 | 1.34x | 1.34x, 1.37x |
| 8 | `model` | 35.60 | 26.61 | 1.34x | |
| 16 | `decode` | 41.68 | 28.20 | 1.48x | 1.49x, 1.50x |
| 16 | `model` | 41.26 | 27.82 | 1.48x | |

- 6,107 kernel calls per generation at both batch sizes. Tokens and logits match the README. Row 7 diverges at step 2 at batch 8 and 16, where stock's top two logits are 0.0156 apart and the patched ones tie. Agreement is 89.5% and 94.7%. The teacher-forced maximum is 0.109 at batch 8 and 0.078 at batch 16.
- The `decode` harness gives the yardstick the `model` harness lacked at batch 8. Stock at batch 8 against stock at batch 1 differs by up to 0.245. Patched at batch 8 against stock at batch 1 differs by 0.197. The new batch-8 flip is a near-tie, not a precision loss.

## Files in `check/`

- `check_static.py` → `check-static.json`: frozen table, grid, table rule, re-derived selection, re-time coverage, speedups, previous-against-new, unchanged cells.
- `check_table.py` → `check-table-{float16,bfloat16}.json`: every final-table config on edge shapes.
- `check-tuned-{float16,bfloat16}.json`, `check-routing.json`: `../review/check_kernel2.py tuned|routing` on the new table.
- `correctness-real-planted-rerun.json`: `bench_multirow.py correctness --parts real,planted,planted_table`.
- `tune-repro-bfloat16-q17_down.json`: `bench_multirow.py tune --grid round2 --dtype bfloat16 --shapes q17_down --rows 2,9,16`.
- `decode-b8-b16.json` (`decode --batches 8,16 --seed 41`) and `model-b8-b16.json` (`model --batches 8,16`).

Reproduce: run each GPU command as `bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10 -- <python> <script> ...` from the repository root. `check_static.py` needs no GPU. The decode, model and `real` runs need `HF_HOME` pointing at a cache with Qwen/Qwen3-1.7B (revision 70d244cc) and `HF_HUB_OFFLINE=1`.
