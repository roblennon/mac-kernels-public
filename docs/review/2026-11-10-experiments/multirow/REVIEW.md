# Review of experiment B: the multi-row linear kernel

2026-11-10. Skeptical review of the build described in `README.md` in this folder. I read the kernel, the wrapper, the benchmark, the tests and every raw JSON file, then reran checks of my own through `bench/gpu_runner.py` on the same machine and the same PyTorch 2.15.0 environment. My scripts and their output are in `review/`. I did not edit the builder's files.

## Verdict

The result holds. I found nothing fatal. The kernel is correct on every case I could construct. The decode gains at batch 8 and 16 reproduce in a fresh process against `nn.Linear`'s own forward. The layer table reproduces in three more processes.

Two things in the write-up need correcting before anyone acts on it. The main one is why the result missed the thread's prediction. The README says 16 rows hit an arithmetic limit. That is true of this kernel, but the GPU itself can go much faster, so the miss is not a hard limit. The second is that the gate passes at 16 rows mostly because of the output head. On the per-layer matrices that hold most of the weight bytes, the gain at 16 rows is under 1.3x.

| Severity | Count |
|---|---:|
| fatal | 0 |
| must-fix | 1 |
| should-fix | 4 |
| minor | 5 |

## Must-fix

### 1. The reason given for missing the prediction is wrong as stated

The README says that at 16 rows the kernel is "bound by arithmetic issue rather than memory". It adds that Apple's kernel is "already about as fast" on the 2048→6144 projection. A reader would conclude that 16 rows cannot go much faster. The data says otherwise.

Evidence (`review/check-timing.json`, fp16, streamed weights compared with one cache-resident copy, interleaved, 7 repeats):

| 2048→6144, 16 rows | streamed µs | cached µs | TFLOPS |
|---|---:|---:|---:|
| tuned `mma_r1_u1_s4` | 131.0 | 131.8 | 3.07 |
| `mmr_r2_u1_s4` | 137.5 | 126.6 | 2.93 to 3.18 |
| `F.linear` at 16 rows | 132.6 | 113.1 | 3.04 |
| `F.linear` at **32** rows, same shape | 150.5 | 145.8 | **5.35 to 5.52** |
| tuned `mma` at 32 rows | 184.4 | 179.0 | 4.37 to 4.50 |

- The builder is right that 16 rows is not limited by memory: streamed and cached times are the same.
- It is not a hardware arithmetic limit either. On the same shape, Apple's kernel does 32 rows at 5.4 to 5.5 TFLOPS. At that rate, 16 rows would take about 73 µs. The memory floor is 63 µs at 400 GB/s, or about 72 µs at the 350 GB/s this kernel itself reaches at 1 to 8 rows.
- So a 16-row kernel near 75 to 80 µs is plausible, against 125 to 131 µs today. The limit is this kernel's own cost per instruction at 16 rows, which the README calls "fragment assembly". On down at 6144→2048 the gap looks the same.
- Why it matters: the gate/up and down projections hold 75% of each layer's weight bytes. Moving them from about 130 µs to about 80 µs at 16 rows would bring the batch-16 linears close to the thread's 14 ms prediction. The README should say the prediction was missed because of this kernel's efficiency at 16 rows, and that the hardware has headroom.

## Should-fix

### 2. The gate is described more generously than the thread wrote it, and the batch-16 gain is mostly the head

The thread says: "If it cannot beat Apple's kernel by 1.3 times on the layer-level sweep, stop before wiring it into a model." The README restates this as "at least 1.3x over `F.linear` at M = 16 on most of the Qwen3-1.7B shapes", then counts shapes, giving 3 of 5 in fp16. The passing shapes are the two small attention matrices and the head. The failing shapes are gate/up (0.97x) and down (1.26x), and those hold 75% of each layer's bytes.

The same layer table, summed in model order with each matrix weighted by how often it appears (`layers.json`, fp16):

| Rows | 28 layers' linears | head | total | head's share of the saving |
|---:|---|---|---|---:|
| 4 | 15.6 → 11.7 ms (1.34x) | 2.05 → 1.77 | 1.31x | 7% |
| 8 | 20.1 → 12.0 ms (1.68x) | 3.87 → 1.78 | 1.74x | 20% |
| 16 | 19.5 → 15.4 ms (**1.27x**) | 7.25 → 2.11 | 1.53x | **55%** |

The gate passes clearly at 8 rows. At 16 rows it passes only because `F.linear` handles the 151,936-row head badly. In bf16 the per-layer linears do reach 1.52x at 16 rows. The README should quote the thread's gate as written, give the weighted view, and say plainly that batch 8 is where the kernel helps the model's layers most.

### 3. The A/A arm bounds the noise of `F.linear` only, not of the kernel

`F.linear_again` is compared with `F.linear`. Nothing measures how much the kernel's own time varies, and on some cells it varies more than `F.linear` does. Across the builder's layer run and my three reruns of the same subcommand (`review/layers-rerun-{1,2,3}.json`, fp16):

| Cell | `F.linear` µs, 4 processes | kernel µs, 4 processes | speedup range |
|---|---|---|---|
| gate/up, 12 rows | 126.4, 127.2, 128.0, 133.4 | 113.7, 120.1, 128.0, 140.1 | 0.95 to 1.11 |
| gate/up, 16 rows | 125.0, 127.7, 138.3, 127.4 | 129.0, 140.0, 156.8, 137.2 | 0.88 to 0.97 |
| down, 16 rows | 144.5, 149.7, 147.2, 145.8 | 114.3, 114.4, 114.5, 114.6 | 1.26 to 1.31 |
| kv, 4 rows | 35.7, 32.6, 32.0, 37.3 | 26.2, 26.5, 25.8, 26.2 | 1.23 to 1.42 |

Within the builder's own process, the kernel's samples on gate/up at 16 rows ran from 124.5 to 160.3 µs. The gate count of 3 of 5 in fp16 holds in all four processes. But the down cell lands on both sides of 1.3. Add a second kernel arm (B/B), or report the range across processes for the gate cells.

### 4. The batch-4 gain is inside the noise, and so is the batch-1 slowdown

- **Batch 4.** In run 2, stock ran 26.98 to 29.40 ms and patched 26.20 to 27.62 ms. Run 1 overlaps the same way. The "1.03x to 1.04x" step gain is not separable from run-to-run noise. The linear layers alone show a steady 4% (14.3–14.6 to 13.7–13.9 ms). Report this as no clear step gain at batch 4.
- **Batch 1.** I timed three arms interleaved (`review/check-model.json`): `nn.Linear`'s own forward at 22.30 ms, the builder's wrapper with the patch off at 22.08 ms, and with it on at 22.34 ms. That is 0.998x against the true stock forward. The 2% slowdown, and its cause in the routing check, is not established.
- Batch 8 and 16 are clearly real (below).

### 5. The batch-8 logit caveat resolves in the kernel's favour; the README should say so

The README says that at batch 8, patched's all-row logit difference (0.078) is "about two and a half times" stock's only reference. That reference covered row 0 only. I compared every row (`review/check-model.json`): stock at batch B against stock at batch 1 on the same prompt, all fed the same tokens.

| Batch | stock B vs stock 1, per row (max) | patched B vs stock B (max) | patched B vs stock 1 (max) |
|---:|---|---:|---:|
| 8 | 0.031 to 0.245 (0.245) | 0.078 | 0.264 |
| 16 | 0.031 to 0.156 (0.156) | 0.110 | 0.203 |

Stock's own difference between batch sizes is larger than the patch's difference from stock. Replace the caveat with this table.

## Minor

6. **The planted bugs are gross, and the max-normalised tolerance is loose.** The allowance, `u + K·2^-24`, follows from the format and is a fair bound. But it is about 17 times `F.linear`'s own max-normalised error (6.1e-4 against 3.5e-5 at K = 2056, fp16). The builder's four mutations each skip an eighth of K and land 30 to 75 times over the allowance, so they do not show sensitivity near the line. I tried subtler ones (`review/check-kernel-a.json`). Dropping the last 8 K elements (dev), dropping the last 4 x elements (mmr), and dropping the bias on the split-reduce path were all caught. bf16 rounding by truncation was also caught, but only by the relative-L2 arm; the max-normalised arm alone missed it. An fp16 accumulator was caught in fp16, again only by relative L2. A half-precision accumulator in the bf16 path passes both arms (1.0 to 1.1 times `F.linear`'s error). It is harmless for precision, but half overflows at 65,504 where bf16 does not. Keep the relative-L2 arm, and note that it carries the detection.
7. **The "126 µs cached" claim had no saved data.** It does now: `review/check-timing.json` confirms 126.6 µs for `mmr_r2_u1_s4` with the weight cached.
8. **Shapes outside the tuned table were never timed.** `patch_linears` sends any linear with 2 to 16 rows to the kernel. Every Qwen3-1.7B shape is tuned, but other models fall back to `default_config`, which uses `mma` with P = 4 above 8 rows. On the 2048→6144 projection, `mma` is 0.88x to 0.97x of `F.linear` at 16 rows. Gate any reuse on a timing check.
9. **The `F.linear` times in the tuning files drift between processes** (down at 16 rows: 183.8 and 250.8 µs in two tuning runs, against 144.5 to 149.7 µs in four layer runs). The layer harness itself reproduces, so the gate is not affected. Treat the tuning files' `F.linear` columns as unreliable.
10. **Small errors.** The `.metal` header says x stays in cache at up to 256 KB, but at 32 rows and K = 6144, x is 393 KB. The pytest file ran only on torch 2.14.1; the 2.15.0 evidence is the JSON correctness runs.

## What I checked and found sound

- **Kernel indexing, read line by line.** Rows past N are clamped for loads and never written. The `mmr` k permutation is the same for W and x (lane `fm`/`fn` → k = 8p + 4e + b on both sides), and its 8-byte x loads are aligned because K % 8 == 0 and every split starts on a 32-element block. The K-split ranges never come out empty. Bias is added exactly once on both the split and the unsplit paths. bf16 output rounds to nearest-even and handles NaN and infinity correctly.
- **Correctness of the configs the model actually uses.** `pick_config` at every M from 1 to 32 on every tuned shape, fp16 and bf16, with a bias on odd M: 384 cases, 0 failures. The head at 2, 3, 5 to 7, 9 to 11 and 13 to 16 rows: 24 cases, 0 failures (`review/check-kernel-tuned*.json`).
- **Aligned views with a nonzero storage offset** for both x and W: 270 runs, all correct (`torch.mps.compile_shader` respects the offset).
- **inf, NaN and fp16 overflow** give the same non-finite pattern as `F.linear` in every family.
- **3-D and non-contiguous inputs** are refused by `supports()`. The model patch flattens to 2-D first, and anything else falls back to `F.linear`.
- **Memory traffic.** The weights rotate through at least 256 MB of copies, more than five times the 48 MB system-level cache. The head uses 2.5 GB. The highest bandwidth in any sample in any tuning or layer file is 364 GB/s (`F.linear`, head, 1 row). The kernel's highest is 352 GB/s. Nothing exceeds about 400 GB/s.
- **Interleaving.** All arms run in one process, in a shuffled order each repeat, with each arm warmed over every weight copy first.
- **End-to-end fairness.** Both arms run in the same process with the same prompts, after a warm-up, with a sync around the timed loop, and with prefill excluded from both. The builder's "stock" arm, which goes through the wrapper, matches `nn.Linear`'s own forward to within noise, and its tokens are identical.
- **Reproduction** (`review/check-model.json`, 4 interleaved repeats). Batch 8: 36.15 → 28.04 ms, 1.29x. Batch 16: 41.55 → 31.98 ms, 1.30x. The README reports 1.29–1.30x and 1.25x.
- **Token checks.** The batch-8 divergence happens where stock's top two logits are exactly tied (gap 0.0). The batch-16 divergence happens at a gap of 0.0156, one fp16 step. Stock repeats itself bit for bit.
- **Counts.** 506 cases and 22,351 kernel runs, 0 failures, all four planted bugs caught. These match the JSON.

## Files

All under `review/`, run through `bench/gpu_runner.py --monitor --seconds 200 --rss-gib 10` with the PyTorch 2.15.0 environment:

- `check_kernel.py`: storage offsets, tuned configs at every M, subtle planted bugs, non-finite inputs. Output in `check-kernel-a.json`, `check-kernel-tuned.json` and `check-kernel-tuned-head.json`.
- `check_timing.py`: streamed against cached weights at 8, 16 and 32 rows, with TFLOPS. Output in `check-timing.json`.
- `check_model.py`: native, wrapper-stock and patched decode interleaved, plus the all-row spread across batch sizes. Output in `check-model.json`.
- `layers-rerun-{1,2,3}.json`: the builder's own `layers` subcommand, fp16, on the four per-layer Qwen3-1.7B shapes, in three separate processes.
