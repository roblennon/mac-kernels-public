# Review of experiment B, round 2: `mmh` and the 9 to 16 row band

2026-11-10. Skeptical review of the round-2 build described in `README.md` in this folder. I read the diff of the four bench and test files, the README and every raw JSON file here. I then reran correctness, layer timings and end-to-end decode in fresh processes, each through `bench/gpu_runner.py --monitor --seconds 280`, on the same M1 Max with the same PyTorch 2.15.0 environment. My scripts and their output are in `review/`. I did not edit the builder's files.

## Verdict

The result holds. I found nothing fatal. The new kernels are correct on every case I could build. The end-to-end gains reproduce in three fresh processes against `nn.Linear`'s own forward, with tight noise arms. The 16-row speedup on the two MLP projections is real.

One thing must be fixed before committing. The tuning command that produced `TUNED` now crashes, and if it ran it would build a different grid, so the table cannot be reproduced from the code being committed. Four should-fix items concern the tuned table itself: some of its choices are noise, and the `F.linear` fallback discards real gains in at least two cells.

| Severity | Count |
|---|---:|
| fatal | 0 |
| must-fix | 1 |
| should-fix | 4 |
| minor | 4 |

### What reproduced

| Claim | Builder | Reviewer, fresh processes |
|---|---|---|
| Batch-16 step, fp16 (stock → patched) | 40.7–42.3 → 27.7–28.5 ms, 1.47–1.48x | 39.8–41.5 → 26.9–28.4 ms, 1.47–1.49x (three processes, 4 interleaved arms each, A/A 1.001–1.010, B/B 1.005–1.008) |
| Batch-8 step | 35.4–36.5 → 26.2–28.0 ms, 1.33–1.35x | 34.7–35.9 → 25.7–26.8 ms, 1.35–1.37x |
| Batch-16 linears alone, model order | 14.1–14.3 and 16.2 ms | 13.9, 14.2, 14.3, 14.3, 14.8 and 16.5 ms (three processes) |
| gate/up and down at 16 rows, fp16, quiet process | 85–95 µs | gate/up 87.5 µs (tuned), 82.6 µs (best `mmh`); down 85.9 µs |
| Round 1 at the same cells | 114–157 µs | round-1 config in the same processes: gate/up 166–169 µs median (121 fastest), down 117–126 µs |
| Correctness, round-2 families | 0 failures | 0 failures in 30,016 more runs (below) |

Sources: `review/decode-p1.json`, `review/decode-p2.json`, `review/configs-float16-*.json`, `review/timing-*.json`. The first decode process's output was overwritten when I reran it with logit saving; its linears figures (13.9 and 14.2 ms at batch 16) are from that run's console output.

## Must-fix

### 1. The tuning that produced `TUNED` cannot be rerun from the committed code

`bench_multirow.py` builds the round-2 grid from `ROUND1_TUNED`, documented as "the round-1 table, kept so the round-2 grid includes every config round 1 chose". It is computed at import as a copy of `mrl.TUNED`, which is now the round-2 table. Two consequences:

- `tune --grid round2` crashes. `TUNED` now holds `"linear"` entries, and `tune_grid_round2` passes them to `Config.parse`. I ran the README's own reproduce command through the lock: `KeyError: 'r'` in `Config.parse` (console output; nothing written).
- Filtering out `"linear"` would make it run, but on a different grid: round 2's picks instead of round 1's. The 56-config fp16 and 40-config bf16 grids recorded in `tune-*.json` would not be rebuilt.

Fix: freeze round 1's table as a literal (or read it from commit `b64a958`), and skip `LINEAR` wherever table entries are parsed into configs.

## Should-fix

### 2. Two cells routed to `F.linear` are clear kernel wins

The fallback rule keeps the kernel only if its slowest speedup across processes beats 1 plus the largest A/A or B/B deviation in that cell. Applied to layer runs that were noisy (see 5), the rule removes real gains. In my fresh processes, fp16, with the cell's sweep winner on both kernel arms:

| Cell, fp16 | Builder: worst speedup / noise | Reviewer, process 1 | Reviewer, process 2 | Round 1 sent this cell to |
|---|---|---|---|---|
| down, 2 rows | 1.04 / 0.11 | 1.26x (`F.linear` 127 µs, kernel 101) | 1.22x (122, 100) | kernel, `tg_r4_u1_s4` (99–105 µs here) |
| gate/up, 11 rows | 1.21 / 0.22 | 1.30x | 1.39x | nearest-row kernel |
| gate/up, 3 rows | 1.11 / 0.29 | 1.05x | 1.26x | nearest-row kernel |
| gate/up, 2 rows | 1.01 / 0.05 | 1.02x | 1.06x | kernel |
| head, 2 rows | 0.99 / 0.05 | 1.03x | | kernel |

- **Down at 2 rows.** `F.linear` at 2 rows on 6144→2048 is bimodal. Its samples sit at 87 to 91 µs in two of the builder's four layer processes, and at 115 to 221 µs in the other two and in both of mine. In every process the kernel's median is below `F.linear`'s: 84 to 115 µs, against 88 to 146 µs. Routing this cell to `F.linear` makes it 20 to 25% slower than round 1 whenever `F.linear` is in its slow state, and no faster otherwise.
- **gate/up at 11 rows** is a 1.3 to 1.4x gain. Its neighbours at 10 and 12 rows stay on the kernel with similar spreads.
- **gate/up at 2 and 3 rows and the head at 2 rows** are ties or small gains. Routing them either way costs little.
- **The rule was applied to the Qwen3-1.7B shapes only.** On the other two shapes, six kept cells fail the same rule (`review/check-raw.json`, `kept_cells_not_clearing_noise`). For example, fp16 Qwen3-4B up at 2 rows (`dev_r2_u1_s8`) measured 0.98 to 1.00x in the builder's runs and 1.03x in mine.

Why it matters: 2 and 3 rows are the verification sizes that `bench_verify_cost.py` and `bench_prompt_lookup.py` use (see 4). Put down at 2 rows and gate/up at 11 rows back on the kernel. Then either apply the rule to every shape, or state that it covers Qwen3-1.7B only.

### 3. Many tuned picks are noise, including the headline cell

In the tune sweep, the winner's median beats the runner-up's by a median of 0.9%. In 108 of 210 cells the margin is under 1%, and in 181 it is under 3% (`review/check-raw.json`, `tune_margins`). With 3 repeats in one process, most per-cell choices among the top `mmh` configs are a coin toss. That would not matter if the configs were interchangeable, but at 16 rows they are not:

| fp16, µs (median of 9 interleaved samples) | gate/up 16 rows, process 1 / 2 | down 12 rows, process 1 / 2 |
|---|---:|---:|
| tuned choice | `mmh_r1_u1_s4_h1`: 105.7 / 87.5 | `mmp_r2_u1_s4_h1`: 115.9 / 93.0 |
| best of 7 `mmh` candidates | `mmh_r2_u1_s8(_h1)`: 93.3 / 82.6 | `mmh_r2_u1_s4`, `_s8_f1_h1`: 95.3 / 86.1 |
| tuned choice, one cache-resident copy | 99.9 / 75.2 | 118.1 / 72.8 |
| R = 2, F = 0 configs, one cache-resident copy | 83.0–92.1 / 64.9–66.4 | 88.7–114.9 / 66.8–67.3 |
| `F.linear` at 16 / 12 rows | 142.7 / 115.6 | 349.3 / 260.6 |

(`review/configs-float16-q17_up-p{1,2}.json`, `review/configs-float16-q17_down-p{1,2}.json`)

- In both processes, the tuned gate/up 16-row config was the slowest of the seven `mmh` candidates. Every R = 2 config was faster, by 1 to 13%. The tuned down 12-row config (`mmp`) was the slowest candidate in one process and third-slowest in the other.
- With R = 2, gate/up at 16 rows runs at 82.6 µs in a quiet process: 4.9 TFLOPS, and inside the 75 to 85 µs the task asked for. The README reports that target as reached "only in single samples". With a better pick, it is reached at the median.
- The builder's diagnosis holds for R = 2. Cache-resident, it takes 65 to 67 µs; streamed, 82 to 88 µs, close to `F.linear`'s one-row stream (76 to 80 µs). The weight stream is the limit. The R = 1 tuned config needs 75 µs even cache-resident, about the time of the stream itself, so its issue cost is close to being the limit again.

Fix: choose from a small set of robust configs (for example R = 2, S = 4 or 8, U = 1 at 9 to 16 rows). Re-time only the cells where a different family wins by more than the noise.

### 4. Round 2 changes committed code in another thread, and that code's numbers were measured mid-edit

`bench_decode_compile_kernel.py` (commit `b30d5ed`) calls `multirow_linear` with the default `pick_config`. `bench_verify_cost.py` and `bench_prompt_lookup.py` (`7e9a42e`, `81e4482`) import its routing. Committing round 2 therefore changes what those benchmarks run, including the `"linear"` cells at 2 and 3 rows, the verification sizes. Their committed results (10:04, 10:14 and 11:01) were produced while the round-2 builder was editing `bench/multirow_linear.py` and `.metal` in the same working tree (final edits at 10:14 and 10:52). So it is not known which table those runs used.

I reran `bench_verify_cost.py` with the round-2 table (`review/verify-*.json`, one process per arm). With the kernel, T = 2, 3, 5, 9 and 13 cost 23.1, 23.6, 25.8, 27.9 and 29.2 ms, against 28.5, 27.5, 27.1, 48.1 and 47.7 ms compiled without it. The committed run had 18.6 and 21.5 ms at T = 2, with and without the kernel. The kernel still helps verification, but my absolute times are 10 to 32% above the committed ones in both arms, so the two runs cannot be compared directly. Note this in the decode thread, and rerun those benchmarks after round 2 is committed.

### 5. The "outside workload" explanation for noisy layer runs is not established

The README attributes noisy layer medians to another workload, based on 74 to 80% GPU utilisation seen between processes. That reading cannot separate the two. ioreg's utilisation is a smoothed average, and right after each of my processes ended it read 76 to 93% before falling to 0. My processes all started at 0% (`gpu_util_before` in `review/timing-*.json`), and all of them still had bursts in which every arm slowed together, for example gate/up at 16 rows in `review/timing-float16-p1.json`, samples 2 to 4. Jobs from the other session go through the same lock, so they cannot overlap these runs. The noise is real either way, and the builder's response (interleaved ratios, fastest samples, both noise arms) is the right one. But the README should not name a cause it did not observe.

## Minor

6. **The linears prediction is a little better than "not reliably".** Across my three processes, five of six batch-16 samples were 13.9 to 14.8 ms. One was 16.5 ms, in a stretch where a stock sample in the same process also jumped to 31.8 ms. The builder's two processes gave 14.1, 14.3, 16.2 and 16.2. Over all ten samples, seven are at or under 15 ms. Switching the gate/up 16-row entry to an R = 2 config (finding 3) would take about 0.25 ms more off the batch-16 linears.

7. **The logit comparison should use a float32 reference, not stock at another batch size.** On my prompts (a different slice of `docs/findings.md`), the batch-8 patched-vs-stock maximum is 0.371, twice stock's own batch-8 vs batch-1 spread (0.180), in 3 of 8 rows. On its own that would look like a precision loss. Against a float32 run of the same model on the same tokens (`review/decode-fp32-reference.json`), the patch is closer to the truth than stock at batch 8: max 0.154 against 0.413, mean 0.0068 against 0.0090. At batch 16 the means are equal (0.0087 against 0.0089), and the patch's max is a little higher (0.374 against 0.281). The one diverging greedy row in each batch splits at an exact tie in stock's logits (gap 0.0). Stock-at-another-batch-size is a noisy proxy that can point either way.

8. **Small code issues.** `cmd_probe` has a stray duplicated comment ("# 2. mmr probes on the large MLP shapes."). The README's model-order sums from the layer table (fp16 at 16 rows, 15.99 to 16.38 ms) depend on cells timed in the noisy processes; the decode run's own linears timing is the better figure, as the README says.

9. **Disclosed limitations confirmed.** The predictions in section 1 were reconstructed rather than saved beforehand, as the builder says. The pytest file was not run on 2.15.0. On the repository's `.venv` (torch 2.14.1), I got 113 passed.

## Correctness: what I checked

All in `review/check_kernel2.py`, one part per process:

| Check | Scope | Result |
|---|---|---|
| Edge shapes (`edges`) | All 44 round-2 configs (41 from `TUNED`, the tune grids and the builder's list, plus three with P = 5 or 7 on small K). K of 8, 24, 40, 264, 2056 and 520. N of 1, 5, 7, 13, 33, 517 and 1027. Every M from 1 to 32, with and without bias. x and W as 16-byte-aligned views at a nonzero storage offset; bias at an odd element offset. fp16 and bf16. Builder's criterion, computed independently. | 19,712 fp16 and 10,304 bf16 kernel runs, 0 failures; largest ratio to `F.linear`'s error 1.000 on both arms |
| Is the harness able to fail? (`sensitivity`) | Subtle mutations: the `mmh` tail chunk dropped (last 8 of 264 or 2056 K elements), `mmt`'s x check one chunk short, bias added twice in the threadgroup reduction, bf16 x halves swapped | Every mutation caught in every case (24 of 24 fp16, 30 of 30 bf16); only 4.5% (fp16) and 41% (bf16) of clean runs are bitwise equal to `F.linear` |
| Half-operand products (`special`) | fp16 x = 256 with alternating ±256 weights: each product overflows half, and the sum is 0. x = w = 2^-13: each product underflows half, and the sum is 66·2^-24. All configs at 2, 9 and 16 rows. | All exact and finite: H = 1 multiplies in fp32, as the README says |
| Non-finite inputs (`nonfinite`) | inf and NaN in x and W, including in the last x row and last weight row (the clamped ones) | inf/NaN pattern identical to `F.linear` for every config |
| Every tuned cell (`tuned`) | `pick_config` on every tuned (K, N), M from 1 to 32 (head 1 to 16, checked on its first and last 2048 rows), with bias, fp16 and bf16 | 410 kernel cells, 0 failures; 6 cells correctly routed to `F.linear` |
| Routing (`routing`) | `pick_config` for M from 1 to 32 on every tuned shape | No round-2 family is chosen above 16 rows (the register cliff). `"linear"` only at the six documented cells. |

The builder's own files check out: 412 random cases and 33,808 runs, plus 94 real-weight cases and 6,574 runs, all with 0 failures, and all 18 planted bugs caught.

## Timing: what I checked

- **Streamed, not cached.** No sample in any timing file, the builder's or mine, implies more than 400 GB/s. The highest are 367 GB/s in the builder's files and 361 GB/s in mine, both on the 622 MB head (`review/check-raw.json`, `bandwidth`). Each kernel arm in my runs also has a one-copy arm. At 1 to 8 rows that arm is 25 to 45% faster (e.g. 49 against 82 µs on gate/up at 2 rows), so the rotated arms really stream.
- **The layer runs timed the final table.** Every layer row used the config the final `TUNED` holds, every `TUNED` entry from 2 to 16 rows is the sweep winner, and both decode runs recorded the final table.
- **Both noise arms.** My A/A and B/B arms deviate by up to 8% on q/o, k/v and the head, and by up to 29% on the 25 to 90 MB shapes, the same spread the builder saw. Taking the worst pairing of the two arms of each kind, every kept MLP cell at 8 to 16 rows clears 1.29x in every process except one: fp16 gate/up at 16 rows. There, one process gave 1.33x on the first arms but 1.03x on the worst pairing, because one kernel arm drew a slow stretch (`review/timing-float16-p2.json`). In the quieter config-scan processes the same cell was 1.32x and 1.35x with the tuned config, and 1.40 to 1.53x with R = 2. This is the one gate cell that still sits near the thread's 1.3x line, as the builder also reports.
- **Regressions at 1 to 8 rows and on other shapes.** Against round 1's config timed in the same process (`r1` arm, `review/timing-*.json`), round 2 is equal or faster on every cell I timed at 2, 4 and 8 rows. Examples: down at 8 rows, 92 against 114 µs; q/o at 8 rows, 39 against 45. The table entries for 1, 24 and 32 rows are unchanged. The only regressions are the `"linear"` cells in finding 2. The Qwen3-4B up, Llama-7B up and head cells I timed (2 to 16 rows) agree with the builder's tables within noise.

## Deviation from the run rules

`review/check_decode_ref.py` loads Qwen3-1.7B in float32, and its process peaked at 11.7 GiB. I ran it with `--rss-gib 12` instead of 10. There was no memory pressure and no swap growth. Every other process used `--rss-gib 10`. I also ran the other thread's `bench/bench_verify_cost.py` through the lock, read-only, writing its output to `review/`.

## Files in `review/`

- `check_raw.py` → `check-raw.json`: static checks of every raw JSON (bandwidth, config match, sweep winners and margins, noise rule on every shape, round-1 comparison).
- `check_kernel2.py` → `check-{edges,nonfinite,tuned,sensitivity}-{float16,bfloat16}.json`, `check-special.json`, `check-routing.json`.
- `check_timing2.py` → `timing-float16-p{1,2,3}.json`, `timing-bfloat16-p1.json`: six arms per cell (`F.linear` ×2, tuned ×2, one-copy, round-1 config, plus `F.linear` at 32 rows for 16-row cells).
- `check_configs.py` → `configs-float16-{q17_up,q17_down}-p{1,2}.json`: tuned against six other `mmh` configs at 9, 12 and 16 rows, streamed and cache-resident.
- `check_decode.py` → `decode-p{1,2}.json`; `check_decode_ref.py` → `decode-fp32-reference.json`.
- `verify-{compiled,compiled_kernel}-*.json`: `bench/bench_verify_cost.py` with the round-2 table.

Reproduce: `bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10 -- <python> docs/review/2026-11-10-experiments/multirow/round2/review/<script> ...`. Each script's docstring gives its arguments. The decode scripts need `HF_HOME` pointing at a cache with Qwen/Qwen3-1.7B (revision 70d244cc) and `HF_HUB_OFFLINE=1`.
