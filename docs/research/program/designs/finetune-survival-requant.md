# Fine-tune survival under requantization: does one signal-to-noise number predict what survives?

Status: queued, 2026-11-09. Search gate passed. Step 1 is runnable and judgeable from "Step 1, exactly" together with its definitions (A2) and reasons (A1).

Scope: steps 1a and 2 use round-to-nearest (RTN) quantizers; steps 1b and 3 use Rob's merge path, which re-derives scales and rounds stochastically with a seed. GPTQ, AWQ and K-quants are untested.

Idea id: `finetune-survival-requant`. Forgotten knowledge returns after 4-bit rounding, and adapter merges into packed files sometimes work and sometimes do not. Published explanations count weights crossing a rounding boundary; this card tests whether a layer-level signal-to-noise number, S_x, predicts survival better.

## Hypothesis

For trained forgetting edits rounded by RTN from a high-precision base, the retained fraction R of the bf16 effect rises with S_x measured on forget inputs. S_x ranks R better than crossing fraction, edit norm, bf16 output change, bf16 logit margin, or two published weights-only statistics. This is wrong if any of those rivals ranks R as well as S_x does.

## Theory

For a weight spread evenly within a bin of width s and a change d much smaller than s, RTN gives Q(w+d) − Q(w) = s·sign(d) with probability |d|/s, else zero: mean d, variance |d|·s − d². This is classical dither (Schuchman 1964; Gray and Stockham 1993, seen as listings only). Over a layer, for edit D and input x:

S_x = ||D x||² / Σ_ij |D_ij| s_ij x_j²

For a rank-k edit with per-weight RMS d, input width n, and x inside the edit's row space, S_x scales as (d/s)(n/k), up to order-one constants: for a Gaussian rank-1 edit with x = v it is about (π/4)(d/s)n. Step 1 uses the exact per-weight model, not this scaling.

On planted matrices this is algebra (regime A, an off-grid base): a reviewer's numpy run (896 × 4864, 4-bit RTN, group 64) matched within about 15%, and a shuffled twin lost about 4,800 times. In regime B, an on-grid base with frozen scales, any |d| < s/2 rounds back. Re-derived absmax scales move every weight in a group (about 85% of stored values changed in that run), so crossing fraction uses integer codes with scales frozen from W. The open question is whether trained edits follow S_x and beat rivals.

Stochastic rounding (SR) changes the constants. A target at fractional position f inside a bracket of width Δ gets variance f(1−f)Δ². For f spread evenly that averages Δ²/6, twice the RTN figure of Δ²/12. For a small offset |δ| from an existing level the variance is about |δ|·Δ, the dither form, so S_x keeps its shape on SR paths.

Rob's merge path (A1) dequantizes in the LoRA compute dtype, fp16 on this Mac, adds the patch, and requantizes with seeded SR; at d/s = 0.01 the edit's RMS is about one fp16 ulp of a typical weight. Requantize refits the fp8 group scales and the fp32 per-row s_channel without the 6-bit scale search, so stored bytes change even at D = 0, and a new s_channel moves a row's whole grid where s_rel is unchanged. Decoded levels are uneven at 6 bits and a Lloyd-Max codebook at 4, so "s" is not one number: the card uses Δ, the float width of the bracket around each weight, and the effective step ratio r (A2).

## What is already known

Verdict: partly open. Every paper below explains survival per weight; none sums sub-bin changes over a layer, uses rank or inputs, or correlates a predictor with retention.

- [Zhang et al.](https://arxiv.org/abs/2410.16454), 2024-10: forgetting reverts at 4 bits; per-weight bin crossing; GPTQ and AWQ behave like RTN.
- [Abitante et al.](https://arxiv.org/abs/2602.13151), 2026-02: LoRA unlearning survives 4-bit RTN; no per-rank result, update size unmeasured.
- [MANSU](https://arxiv.org/abs/2605.15138), 2026-05: per-parameter magnitude floor; no correlation, no LoRA arm.
- [FOM-UL](https://arxiv.org/abs/2609.10439), 2026-09: bin-change fraction per method (0.7% global to 5.6% selected layers, 4-bit), untested against retention.
- [Scale-QLoRA](https://arxiv.org/abs/2609.04526), 2026-09: naive RTN merges keep about 15% of the gain; re-selected block scales (30.2% of gpt-oss blocks) carry over 99.99% of the displacement; a weights-only ratio ||Q(W+Δ)−W||/||Δ|| tracks retention across two grids (no correlation).
- [DAQ](https://arxiv.org/abs/2603.22324), 2026-03: FP8 erases small SFT deltas; sign preservation rate of ΔW as the objective.
- Abstract only: [QUAIL](https://arxiv.org/abs/2601.15538), [DurableUn](https://arxiv.org/abs/2605.02196), [sharp basins](https://arxiv.org/abs/2609.27355), [optimizer choice](https://arxiv.org/abs/2510.00761), [TULIP](https://arxiv.org/abs/2609.34591).

Search gate, 2026-11-09: 18 searches, about 15 fetches; read in full except parts of Scale-QLoRA and Abitante. Adjacent: [task vector quantization](https://arxiv.org/abs/2503.06921), [Delta-CoMe](https://arxiv.org/abs/2406.08903), [CACHE-UK](https://arxiv.org/abs/2607.28292) (snippet), [layer SNR](https://arxiv.org/abs/2102.06365); forum reports of 1.5 to 3 times fp8 LoRA strength loss (anecdote). OpenReview not reached.

Left open: a rank-aware, input-aware survival statistic tested against rivals on trained edits, and a strength threshold for diffusion LoRAs merged into packed weights.

## Prediction (written before any run)

Anything between a predicted value and its kill line is labelled inconclusive in advance.

- Step 1b: P-1b to P-1d and their kill lines live only under "Judging" in "Step 1, exactly". P-1a (most groups change step) is withdrawn as near-certain from fp32 round-off in s_channel.
- Step 2: across all arm × α × seed points, Spearman(R, S_x) ≥ 0.7, and it exceeds every rival's Spearman by at least 0.15.
- Step 2, secondary: where arms' RMS ranges overlap, the rank-1 LoRA keeps at least 0.3 more R than the Adam full-matrix arm.
- Step 2, generality: for low-rank arms, S_x on natural activations is at least 10 times below S_x on forget inputs. This decides how general any claim can be.
- Step 3: the strength where merged renders keep half the side-path effect is within a factor of 2 of the 3a prediction.

## Design

**Step 1 (one slice, CPU).** 1a tests the instrument on planted effects; 1b runs Rob's W6A8 and W4A8 requantize on real Qwen weights. See "Step 1, exactly".

**Step 2 (headline; two MPS slices).** Same model, unless the pilot fails.

Pilot (about 5 minutes). Commit a country-to-capital list, about 300 pairs from a public list with source and licence, before any model sees it. Keep facts the base answers correctly in bf16 and under every quantizer below; minimum 50 forget, 100 retain, 50 paraphrased forget. Otherwise switch to `Qwen/Qwen2.5-1.5B-Instruct` (memory fit not checked) or another fact type.

Training (slice 2a). Edit down_proj of layers 8 to 11 with gradient-difference loss, stopped at bf16 forget accuracy ≤ 10% with retain ≥ 90% of base. Four arms: rank-1 LoRA (Adam), rank-8 LoRA (Adam), full-matrix SGD, full-matrix Adam. LoRA is two hand-written factor matrices (no peft). Three seeds, one process per run. Each delta's singular spectrum and effective rank, (Σσ²)²/Σσ⁴, not its arm label, is the predictor.

Scaling. Five log-spaced α values per arm, up to 2, chosen from each delta's per-weight RMS so every pair of arms overlaps over at least a factor of 2 in RMS.

Quantizers. All linear layers of base and edit alike: RTN 4-bit group 64 with re-derived scales, the same with scales frozen from the base, and simulated NF4 block 64.

Evaluation (slice 2b). One process per quantizer, arms and seeds interleaved; each bf16 evaluation runs twice for the MPS noise floor. R = (quantized base − quantized edit forget accuracy) / (bf16 base − bf16 edit). R is not reported for a point whose bf16 effect is under 3 times its spread across seeds.

Predictors, saved before quantized evaluation. S_x uses the down_proj input at each forget fact's last prompt token, signal and noise each summed over layers 8 to 11 and facts. Rivals: crossing fraction on integer codes; ||D x|| / (s·√n); bf16 logit margin on forget facts; ||D||_F / s; Scale-QLoRA's displacement ratio ||Q(W+D)−Q(W)||/||D||; DAQ's sign preservation rate of D. S_x is also computed on 512 natural activations.

Deciding metric: Spearman of R against each predictor across all points, per quantizer, with a bootstrap over facts for intervals on each difference.

**Step 3 (Rob's ground).** 3a, CPU, one slice. For Krea 2 Turbo INT8 ConvRot with the rwbflat and darkbrush LoRAs, and H3 W6A8 with the 4-step Turbo LoRA, compute per-layer S_x on each LoRA's input directions at strengths {0.1, 0.2, 0.4, 0.8, 1.0}, using model P2 in the ConvRot basis, the fraction of re-scaled groups, and a predicted half-effect strength. Use the compute dtype Rob's merges ran in (A1). First confirm the Krea INT8 ConvRot path has a merge-then-requantize mode.

The H3 Turbo LoRA is high-rank (qkv rank 384), and its merge noise was about 30 times its own per-layer contribution, yet it worked (`docs/findings.md`, 2026-10-02). Decided in advance: summed S_x from 3a below 0.3 at strength 1.0 counts against S_x on this path; above 3 is consistent; between is no claim.

3b, renders, only if 3a puts a half-effect strength inside the grid (otherwise widen the grid or use a 4-bit path). Krea, 512 px, 8 steps, 8 prompts × 4 seeds; base, side path and merge per strength, interleaved. Metric: DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`, cached) shift of the merge projected on the side-path shift, over that shift. H3 video renders need Rob's approval.

**Step 4 (later, cloud, Rob's login).** Public (base, fine-tune) pairs, real bitsandbytes NF4, GPTQ and K-quants, an 8B model on MUSE or WMDP, and a sparse full-rank arm in the MANSU style.

## Step 1, exactly

This section is the only one a runner reads from. Reasons are in A1; definitions in A2.

**Inputs.** `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775` (main on 2026-11-09). Only `config.json` and `model.safetensors` (988,097,824 bytes). Download as a pre-step, outside the timed processes, and check the file's SHA-256 against the Hub's linked etag, `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`; stop on a mismatch. Tensors `model.layers.{8,10,12}.mlp.down_proj.weight`, shape [896, 4864].

**Code.** This repository's vendor/comfy-kitchen at b5dd528 and vendor/ComfyUI at 04162521. Use the repository `.venv` (torch 2.14.1, numpy 2.5.3, safetensors 0.8.0) with PYTHONPATH set to this repository's vendor/comfy-kitchen and vendor/ComfyUI. Record `comfy_kitchen.__file__` and the git HEAD of its checkout in env.json; stop if HEAD is not b5dd528. After import, call `comfy_kitchen.disable_backend("mps")` and keep every tensor on CPU. Nothing reports which backend served a call, so log the module of `registry.get_implementation(name, kwargs=...)` (`from comfy_kitchen.registry import registry`) for `quantize_w4a8_int8_weight` and `dequantize_w4a8_int8_weight` with the layout's kwargs (expected: eager).

**Formats.** Q0 = `QuantizedTensor.from_float(W, "AsymW4A8Int8Layout", **kw)`, with `QuantizedTensor` from `comfy_kitchen.tensor` and W the bf16 tensor as loaded; and every requantize is `Q0.requantize_from_float(...)`. W6A8: `bits=6`, `group_size=32`, `scale_dtype=torch.float8_e4m3fn`, `convrot_groupsize=256`. W4A8: `bits=4`, `group_size=16`, `codebook=True`, same scale dtype and ConvRot group. Other arguments take their defaults (`scale_search=True` at 6 bits). Record whether the W4A8 kurtosis gate chose the fixed LUT or a fit.

**Edits.** Per layer, four edits, edit i using `numpy.random.default_rng(b + i)`, with b = 20261109 for layers 8 and 10 and b = 20261209 for layer 12 (fresh seeds for the repeat; see the layer rule): i = 0 rank-1 at d/s = 0.01, i = 1 rank-1 at 0.1, i = 2 full-rank at 0.01, i = 3 full-rank at 0.1. Rank-1: draw u (length 896) then v (length 4864), standard normal, unit-normalise each, D = c·u vᵀ. Full-rank: draw G with shape [896, 4864], D = c·G. c sets the per-weight RMS of D to (d/s)·Δ̄ in 1b (Δ̄ per format and layer, A2), or (d/s) times the RMS RTN step in 1a. Built in fp64, cast to fp32.

**Inputs x.** Gaussian set: `numpy.random.default_rng(k).standard_normal((256, 4864))`, fp64, with k = 7 for layers 8 and 10 and k = 8 for layer 12. Row-space input, rank-1 edits only: the single vector x = v of that edit. For per-weight metrics, x̃ is x rotated with comfy-kitchen's `_build_hadamard(256)` built in fp64.

**Process 1, step 1a** (layer 8, under the same runner with `--monitor --seconds 60 --rss-gib 2`). Gates 1 and 2 in numpy fp64 on all four edits; gate 3 in torch fp32 with comfy-kitchen's rotation. Everything in 1a except G3 works in the original basis.

- G1, identity quantizer: Σ||noise·x||² ≤ 1e-20·Σ||Dx||².
- G2, injected Gaussian noise with variance s_g²/12 per RTN group: 20 draws (seeds 100 to 119) on the Gaussian set; mean measured S_x within 2% of predicted; draw-to-draw SD printed.
- G3, rotation: rotated W times rotated x equals W·x on the whole output (896 rows × 256 x), ||a − b|| / ||b|| ≤ 1e-5; max absolute error printed.

Then, reported only: own numpy RTN (4-bit, levels −8 to 7, group 64, s = absmax/7) on the four edits, regime A (frozen and re-derived scales) and regime B, against the dither prediction Σ_x Σ_ij (|D_ij|s_ij − D_ij²)x_j², with the |D|s form beside it.

**Processes 2 to 7, step 1b**, one per format × layer, in order W6A8 L8, W4A8 L8, W6A8 L10, W4A8 L10, then W6A8 L12, W4A8 L12. Each runs under `bench/gpu_runner.py --monitor --seconds 300 --rss-gib 2`, with `MK_GPU_LOCK` set to a lock file of its own, not the GPU lock.

1. Q0 = quantize(bf16 W).
2. For D in [0, edit 0, edit 1, edit 2, edit 3], run arms R, S16 and S32. deq32(Q) means `Q.to(torch.float32).dequantize()`, the fp32 decode whatever Q's stored orig_dtype.
   - R: `requantize_from_float(deq32(Q0) + D, stochastic_rounding=0)`.
   - S16, the shipped chain on this Mac and the judged arm: `Q0.to(torch.float16).dequantize()` (it returns fp16), add D cast to fp16, then `requantize_from_float(fp16 tensor, scale="recalculate", stochastic_rounding=seed)`.
   - S32: as S16 with fp32 in place of fp16 (the explicit-CPU-device chain).
   - S16 and S32 use seeds seed0 to seed0+4, with seed0 = ComfyUI `string_to_seed("model.layers.{L}.mlp.down_proj.weight")`, copied from `comfy/utils.py` and cited.
3. W6A8 at D = 0 only: R-search, which is R with `scale_search=True`.

Loop over D outermost, so D = 0 for every arm runs first. Within each D the order is R, S16 seeds seed0 to seed0+4, S32 seeds seed0 to seed0+4, then at D = 0 only R-search (W6A8) and C1's repeat of S16 seed0. That gives 13 requantizes at D = 0 for W6A8 (12 for W4A8) and 11 at each edit: 57 and 56 in all.

**Timing.** All times run from `time.monotonic()` at the first line of the script, so the load, Q0 quantize and edit construction count. When the D = 0 pass ends at elapsed time T0, t is that pass's time (each requantize with its dequantize, rotations and metrics) over its requantize count. If T0 + 44t exceeds 240 s, cut S16 and S32 to three seeds for the four edits (7 requantizes each) and record it; if T0 + 28t still exceeds 240 s, stop and re-plan. A deadline at 270 s elapsed writes the CSV rows done so far and marks the process truncated in env.json; a truncated process makes no claim for its format and layer.

**Checks**, which stop the slice before predictions are read. Tolerances are norm-relative over the whole tensor, ||a − b|| / ||b||, with max absolute error printed.

- C1: S16 at D = 0 with seed0 run twice is byte-identical.
- C2: rotate(deq32(Q0)) matches the direct decode, the int8 levels from `_dequant_int4_grouped_to_int8` times s_channel in the rotated basis, within 1e-6.
- C3: for S16 and S32 on the Gaussian set, mean measured noise energy over SR seeds is within a bar of the P1 noise energy for every D, including D = 0. The bar is max(5%, 3·SD/√n), with SD the seed-to-seed SD of measured noise energy and n the seed count, both printed. A fixed 5% could stop the slice for sampling spread, not a broken instrument, where noise is concentrated (A1). Row-space numbers are printed, not checked.
- C4: G1 to G3 passed.

**Judging.** Predictions are judged on arm S16 and Gaussian inputs. R, S32 and the row-space input are reported only. Across SR seeds, S_x and noise ratios are ratios of means (mean signal over mean noise energy), to match C3; P-1c is the exception, as its row says. Each judged cell (prediction × format × layer × D) gets an interval, mean ± 2·SD/√n of the per-seed value; seeds pair across D, since every D uses the same SR seeds. A cell inside its band passes. It kills only when the value and its whole interval lie beyond the kill line; otherwise it is inconclusive, as the frame requires for a difference inside the seed spread. A prediction is killed in a format and layer if any judged cell kills, and passes only if all pass.

| | Prediction | Kill line | Where judged |
|---|---|---|---|
| P-1b, re-grid account | noise energy at D = 0 is at least half that of each d/s = 0.01 edit | below a quarter of either; then the re-grid account of the H3 merge error is wrong | every layer, both formats |
| P-1c | in the shift ≥ 1 stratum, mean Ẽ² is 0.67 to 1.5 times mean Δ⁰²/6 (mean over seeds of the per-seed ratio) | outside 0.4 to 2.5 | every layer, both formats, at each of the five D separately where that D's stratum holds at least 1,000 weights in every seed |
| P-1d | measured S_x over P2 S_x is 0.75 to 1.33 for all four edits | outside 0.5 to 2; then step 3a cannot use P2 and must run the real requantize per strength | every layer, both formats |

Expected before the run: the W4A8 shift ≥ 1 stratum will hold under 1,000 weights, so P-1c is in practice a W6A8 test (A1); a W4A8 stratum that fills is judged as written, and smaller strata are reported only. P-1c is judged per D, not pooled, because the stratum's population can differ between D = 0 and the d/s = 0.1 edits (a default the researcher may change before the run). Fractions of changed s_rel bytes, changed s_channel rows and shift ≥ 0.25 weights at D = 0 are a diagnostic, not a test.

Layer rule: layers 8 and 10 are the first observation; layer 12, with fresh edit and x seeds, is the unseen repeat. Step 1 is one slice run in two stages: stage 1 is process 1 and the four L8 and L10 processes, ending with their verdicts written into the slice plan under a date; stage 2 is the two L12 processes. A verdict that recurs on layer 12 may be called a result. If layer 12 disagrees with 8 and 10, or is inconclusive, the write-up opens with "unconfirmed observation".

**Outputs.** Slice plan first, at `docs/research/program/slices/finetune-survival-step1.md`, copying the Judging table and the expected W4A8 P-1c outcome. Results in `docs/research/program/slices/finetune-survival-step1/`: `step1a.json`, one `step1b_<format>_L<layer>.csv` per process (columns in A2), `env.json` (package versions, backend, comfy-kitchen import path and HEAD, ComfyUI commit, model revision and file SHA-256, CPU, wall time, peak RSS and truncation flag per process), and logs.

## Checks before believing anything

- Self-comparison: C1; the D = 0 merge as noise floor; one image arm rendered twice with one seed gives the same PNG hash.
- Planted effects independent of the theory: 1a's gates; a closed-form logit-margin edit read back unquantized (step 2); an fp32 merge without requantize giving R near 1 (step 3).
- The dither match is a result, not a gate; a mismatch is reported, not fixed away.

## The control that could embarrass it

Crossing fraction and the other rivals, scored on the same trained points. If any of them ranks R as well as S_x, the extra machinery adds nothing. For renders, merge a random matrix with the LoRA's norm and rank; if side-against-merge PSNR is again about 17.8 dB, that figure was only re-drawn rounding noise.

## Confounds a hostile reader would raise

- Larger LoRA updates (Abitante's confound): α overlaps RMS ranges; the decision is a rank correlation.
- Optimizer sets full-matrix rank: SGD and Adam arms, with measured effective rank.
- Shallower rank-1 forgetting: R is normalised per point; small effects are excluded.
- Sharp basins: not addressed beyond the optimizer split.
- ConvRot basis: predicted S_x works in the rotated basis; measured S_x is output-level.
- SR against RTN, fp16 against fp32: R, S16 and S32 run side by side.

## Kill criterion

- Step 1b: the kill lines in the Judging table of "Step 1, exactly".
- Step 2: Spearman(R, S_x) below 0.4, or any rival beats S_x with a bootstrap interval on the difference above zero.
- Step 2 secondary: the rank-1 arm beats the Adam full arm in R by less than 0.1 where RMS overlaps.
- Step 3: the half-effect strength misses the prediction by more than a factor of 3, or the H3 condition above.

## Cost on this hardware

Estimates, not measurements.

- Step 1: a 1 GB download (33 GiB free on 2026-11-09), then seven CPU processes under 2 GiB each, capped at 60 s + 6 × 300 s, about 31 minutes plus launch overhead (stage 1 at most about 21, stage 2 about 10).
- Step 2: slice 2a, pilot plus 12 training runs, about 25 MPS minutes; slice 2b, about 20 minutes. Under 4 GiB per process.
- Step 3a: about 20 CPU minutes; files assumed present. 3b: 11 conditions × 32 images at about 30 s (`docs/findings.md`) is about 3 GPU hours, 6 or more slices, via the Krea server (`docs/how-to-run.md`). Keep free disk above 15 GB.
- Step 4: cloud time, Rob's approval.

## What a yes or a no changes

A yes gives merge tools a number to warn on, favours concentrated edits over inflated ones, and tells Rob whether his 6-bit merge should default to the side path. A no supports per-weight magnitude and the MANSU-style floor.

## Why it might carry to larger models

The formula has no model-size term, but wider layers help low-rank edits only when relevant inputs lie in the edit's row space; step 2's natural-activation S_x is the evidence.

## Receipts to keep

Slice plans, revision hashes and file SHA-256s, vendor commits and import paths, quantizer code, per-point CSVs, spectra, prompt lists, eval logs, check outputs, image hashes and DINOv3 embeddings.

## Review record

Hostile review, 2026-11-09, queue-after-fixes: all points fixed (1a as unit test, step 2 as headline, rivals by Spearman, SGD and Adam arms, frozen-scale crossing, bands in advance, 3a gates 3b, pilot, interleaving).

Cold reads 1 to 3, 2026-11-09: from not plannable to runnable cold; fixed layers, SR path and merge dtype (S16 judged), strata, P-1a withdrawn, bracket convention, predictions moved into "Step 1, exactly", timing, tolerances, download, backend and checks.

Cold read 4 fixes, 2026-11-09: no blocks; four result-changing decisions made. T̃ drives Δ⁰, f⁰, Δ', f' at every D (stored levels only for Δ̄); P-1c judged per D on the mean of per-seed ratios; C3's bar widens with seed spread; layer 12 gets fresh seeds; kill-side values inside the seed spread are inconclusive, per the frame. Also fixed: timing projection, arm order, API calls, backend record, 1a flags and basis, ratio-of-means S_x, energy columns, K = 896 list, cost (about 31 minutes, two stages).

## Appendix A1: step 1 sources and reasons

- Merge path. In vendor/ComfyUI, `comfy/model_patcher.py` `patch_weight_to_device` takes the patch dtype from `lora_compute_dtype(device_to)` and passes `seed=string_to_seed(key)`; `comfy/ops.py` `set_weight` calls `requantize_from_float(..., scale="recalculate", stochastic_rounding=seed)`. `should_use_fp16` is true for MPS, false for a CPU device, so the dtype is fp16 here. In comfy-kitchen, `tensor/base.py` `_handle_to` rewrites `orig_dtype` on a cast and dequantize returns it, hence deq32 (Q0's orig_dtype is bf16). `tensor/w4a8_int8.py` `requantize_kwargs` forces `scale_search=False` and reuses the codebook; `backends/eager/w4a8_int8.py` fits group scales by ALS, sets s_channel = max|fitted weight|/127 and s_rel = group scale / s_channel, rounds by SR between bracketing decoded levels, and offsets the seed per 862-row chunk.
- Which arm is shipped. S16 matches a ComfyUI merge on this Mac, S32 one sent to an explicit CPU device. Default for step 3: S16, until Rob confirms the device.
- Q0 from bf16. How the H3 file was made is unknown, and `docs/findings.md` says the quantization study fed fp32; bf16 changes Q0's grid slightly, not the mechanism.
- Bracket convention (A2). It follows `_assign_grid`, so Δ⁰ is the bracket the quantizer actually draws in. It matters: it moves Δ̄, and so every edit's size c, by several percent, because 4-bit codebook gaps widen toward the edges and the 6-bit int8 grid alternates gaps.
- P-1c on W4A8. With the 16-level codebook, |T̃|/Δ⁰ is at most about 5.3, and one e4m3 byte step of s_rel near 64 to 128 moves r by 6 to 12.5%, so shift ≥ 1 needs a change of two or more bytes (round 3 reader's arithmetic from the code). The stratum will usually hold under 1,000 weights. On W6A8, |T̃|/Δ⁰ reaches tens, so the stratum fills.
- Import path and backend. The repository `.venv` links to the sibling checkout's `.venv`, whose `.pth` imports that checkout's vendor/comfy-kitchen; both were identical at b5dd528 on 2026-11-09, and PYTHONPATH removes the dependence. Importing comfy_kitchen registers MPS; disabling it keeps this CPU slice off the GPU.
- Layers. K must divide by the ConvRot group; K = 896 fails at the shipped 256, so every projection whose input width is 896 (q, k, v, o, gate and up) is out, and down_proj (K = 4864 = 19 × 256) passes. 8 and 10 are in step 2's range; 1a runs on 8 so layer 12 stays unopened. Confirm 896 and 4864 from `config.json`.
- Formats, arms, seeds. Group sizes from `docs/findings.md`; whether the H3 file used scale search is not checked, and R-search bounds the effect. R is RTN so the Δ²/6 against Δ²/12 factor is measured; five SR seeds give C3 an expectation. seed0 is non-zero (1613083851, 2147278447, 1370232815 for L8, L10, L12). One global scale per edit matches the theory's per-weight d.
- Inputs. Gaussian x: step 1 tests the noise account; real activations are step 2's job.
- Gate and check sizes. On 256 Gaussian x, noise energy has about 0.3% relative SD per draw, so 2% and 5% bars are sound; on one direction it is about 4.7% and they would fail falsely 2 to 6% of the time, so gates and checks use the Gaussian set. The 0.3% assumes noise spread over all weights; at D = 0, above all for W4A8 S16, it may sit in a few re-gridded groups, hence C3's widening bar.
- Tolerances. Level 0 returns from the fp32 Hadamard round trip as about 1e-9, so elementwise relative tests fail; all are norm-relative.
- Timing. Metric work counts because S16 adds fp16 CPU rotations. The round 3 reader puts the H3 merge, eager on CPU in fp16, at about 2 s per 37M-element layer, so about 0.25 s here. Projecting from process start, with D = 0's finished work counted, keeps each process inside the runner's 300 s kill.
- Regime B. Heavy-tailed rank-1 entries cross half a step at d/s = 0.1, so expect S_x near, not exactly, 1. Reported only.

## Appendix A2: step 1 definitions

All per-weight quantities are in the ConvRot-rotated basis (group 256), where comfy-kitchen rounds. x̃ is x rotated by the same Hadamard, so W·x = W̃·x̃.

- Decoded grid. For row i and group g, levels are L = clamp(round(ℓ·s_rel), −127, 127) in int8 units (`_grid_levels`), with ℓ ∈ {−31 … 31} (W6A8) or the 16-entry codebook (W4A8). The float level is L·s_channel_i. Brackets are taken between distinct decoded values; the count of groups with repeated levels goes in env.json.
- Target T̃_ij: the merged weight as the quantizer sees it, built as the code builds it: `rotate_int8_convrot_weight` in the arm's dtype, then `.float()`. This holds at every D, D = 0 included, so for S16 at D = 0 the target is the fp16 round trip of Q0's decode. The code divides it by s_channel' and rounds against int8 levels; the card compares the undivided T̃ with float levels, L·s_channel for Q0's grid and L'·s_channel' for the new one, which is the same comparison.
- Δ⁰_ij and f⁰_ij: width of the bracket of Q0's grid that holds T̃_ij, and T̃'s fractional position in it. The bracket is [level[lo], level[lo+1]] with lo = clamp(searchsorted(levels, T̃, right=False) − 1, 0, last − 1), as in `_assign_grid`: a target on a level takes the bracket below it, and only the bottom level takes the bracket above. Δ' and f' are the same on the requantized grid. All four use T̃, so P1 and P2 keep the SR spread of the round-trip offsets the quantizer really draws (from stored levels, f' would be 0 wherever the grid did not move, and C3 could stop falsely). The exception is Δ̄, the RMS of Δ⁰ over Q0's stored levels (same bracket rule), per format and layer: it fixes the edit size c once, the same for every arm.
- Outside the grid: a target beyond the outermost decoded level is clamped to the edge level deterministically. Its noise is v = (T̃ − edge)², in P1 and P2 alike. The count goes in the CSV.
- Changed scale: s_rel byte differs (per group); s_channel differs (per row); r = s_rel'·s_channel' / (s_rel·s_channel) per group.
- Expected level shift, both formats: shift_ij = |T̃_ij|·|r − 1| / Δ⁰_ij, how far the grid moves at the target, in local brackets. Strata: below 0.25, 0.25 to 1, at least 1. Noise ratio per stratum: mean Ẽ² over mean Δ⁰²/6 (SR arms) or Δ⁰²/12 (R).
- Decoded values changed: decoded int8 levels, overall and where the s_rel byte is unchanged; raw codes are not compared across grids.
- Noise: E = deq32(Q') − deq32(Q0) − D in the original basis, with D the fp32 edit as built, so fp16 rounding of D counts as noise; Ẽ rotated with the fp64 Hadamard. Noise energy is Σ_x ||E x||².
- Measured S_x = Σ_x ||D x||² / Σ_x ||E x||².
- P1 (instrument, SR arms only): per in-grid weight v = f'(1−f')Δ'², the exact expectation given the new grid. Its noise energy, which C3 compares, is Σ_x Σ_ij v_ij x̃_j².
- P2 (re-grid model; uses Q0's grid, the target and r): per in-grid weight, v = Δ⁰²/6 (SR) or Δ⁰²/12 (R) where shift ≥ 0.25, since a grid move of a quarter bracket or more re-draws the fractional position; otherwise f⁰(1−f⁰)Δ⁰² (SR) or the squared nearest-level error on Q0's grid (R). Predicted S_x = Σ_x ||D x||² / Σ_x Σ_ij v_ij x̃_j². P2-srel, the pre-registered form, applies the same rule with "s_rel byte changed" in place of shift ≥ 0.25 and is reported beside it.
- CSV columns: format, layer, arm, edit, d_over_s, seed, codebook_kind, frac_srel_changed, frac_schannel_rows_changed, frac_shift_ge025, frac_decoded_changed, frac_decoded_changed_srel_same, n_outside_grid, signal_energy_gauss, noise_energy_gauss, noise_energy_p1_gauss (blank for R), noise_energy_p2_gauss, noise_energy_p2srel_gauss, n_shift_lt025, n_shift_025_1, n_shift_ge1, ratio_shift_lt025, ratio_shift_025_1, ratio_shift_ge1, sx_measured_gauss, sx_p1_gauss (blank for R), sx_p2_gauss, sx_p2srel_gauss, sx_measured_row, sx_p2_row, wall_s, peak_rss_mb. The sx_* columns are blank at D = 0, where the signal is zero; the energy columns let C3 and P-1b be audited there from the raw CSV.
