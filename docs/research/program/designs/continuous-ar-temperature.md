# What temperature does in continuous-token autoregressive models: the innovations test, then a two-sided feedback map

Status: queued, 2026-11-09. Search gate passed the same day; neither stage is published. Revised after a hostile review and after the search; records at the end.

MAR, VibeVoice and Kyutai's CALM (the base of Pocket TTS) generate continuous tokens: a backbone produces a state, and a small diffusion head turns noise into the next token. Each paper defines temperature differently and sets it by grid search. This card asks why lower temperature helps. Stage 1 runs real tokens backwards through the head under teacher forcing and measures how wide the recovered noise is. Stage 2 lets the emitted token differ from the fed-back token, from the same noise draw. Only stage 2 can say which reading of temperature holds.

## Hypothesis

Stage 1. On MAR-B, guidance off, at the trained mask ratios 0.75 and 0.95, the per-dimension RMS of real-token innovations, divided by the RMS of the model's own τ = 1 samples under the same inversion, lies in 0.95 ± 0.03. The prediction fails if this ratio is above 1, or within 1 ± 0.01, at either ratio. Any other value is reported as "prediction missed".

Stage 2. On MAR-B, some arm whose fed-back token differs from its emitted token sits below the diagonal frontier at its own recall, by more than twice the noise floor. The diagonal frontier is Fréchet distance against recall, interpolated across a fine sweep of ordinary temperature. This is wrong if no such arm exists.

## Theory

A deterministic head maps noise to token invertibly given the backbone state. Inverting a real token with real context recovers the noise the model would have needed. For a perfect model these innovations are standard normal.

Four readings of why low temperature helps, not mutually exclusive. (1) Recalibration: the head is too wide. (2) Exposure bias: low temperature keeps the free-running context close to training data. (3) Metric preference: FID-like metrics reward precision over recall, so colder scores better even with a calibrated head. Matched recall controls this. (4) Parallel factorisation: below num_iter 256, MAR samples several tokens at once, independently, and low temperature may compensate. A num_iter 64 against 256 arm controls this. (5) Sampler noise: eMIGM finds an ODE sampler at τ = 1 far better than DDPM at τ = 1, so part of the gain may undo stochastic-sampler error, not head width. The DDIM knob tests this.

An RMS below 1 fits reading 1 but does not show that sampling at that radius helps. The predictive test is in step 2a: does a sampler built on the fitted innovation prior match the fine-diagonal optimum at matched recall?

Assumptions. Inversion bias on real tokens is measurable: the step-count check and the self-sample floor test this. Teacher-forced inversion does not depend on num_iter, since it is one forward pass at a fixed mask ratio. The backbone was trained on clean context, so cold context is also off-distribution; hence the map goes both ways.

## Temperature definition

Per the reviewer's reading of MAR's code, MAR scales the per-step DDPM noise by τ, not the initial noise. Inversion needs a deterministic sampler, so this card uses a different knob: DDIM with initial noise scaled by τ, 100 head steps. Its own best τ comes from a fine diagonal sweep, never from MAR's 0.95 (README, guidance off).

Step 1b is a gate. Generate with both knobs at τ = 1.0 and 0.8 and compare the DINOv3 trend. If the trends differ by more than the seed spread, stop and redefine the knob. Add one point: DDIM at τ = 1.0 against MAR's DDPM at 0.95, which tests reading 5 directly.

## What is already known

Searched 2026-11-09 (gate below). Neither stage is published. One known result changes how stage 1 is read.

- [Pasini et al.](https://arxiv.org/abs/2411.18447), read in full. Emits a clean embedding and feeds back one with small added noise (k near 0.02), in a model trained with noise augmentation: the split is half occupied, in the hot direction. None of its 24 citers decouples on a frozen model (titles, two abstracts).
- [Diffusion Forcing](https://arxiv.org/abs/2407.01392), [Kyutai CALM](https://arxiv.org/abs/2509.06926), [MELLE](https://arxiv.org/abs/2407.08551), xAR, reAR, RAE-AR: training-time context noise; at inference the emitted token is fed back.
- [VideoMAR](https://arxiv.org/abs/2506.14168), abstract only: a progressive temperature schedule at inference reduces accumulated error. This is reading 2 used as a fix, without the emit/feedback split.
- Inverted noise from real images is under-dispersed in image diffusion models. [There and Back Again](https://arxiv.org/abs/2410.23530) (first half read) finds DDIM-inverted latents with per-pixel std of about 0.34 to 0.96, and blames inversion error in the first steps, not miscalibration. [Spectral Collapse](https://arxiv.org/abs/2602.13303) (abstract and summary) also finds low variance. The bias runs in the same direction as the stage 1 prediction.
- [Bjare et al.](https://arxiv.org/abs/2508.05306): a MAR-style head on music latents, PF-ODE from data to noise for NLL, one step ahead. No statistics of the recovered noise, no temperature. The closest tool to stage 1.
- [eMIGM](https://arxiv.org/abs/2503.07197), relevant part read: on a MAR-style model, DPM-Solver at τ = 1 gives FID 6.6 where DDPM at τ = 1 gives 10.6 (8 mask steps). Part of what temperature fixes may be the stochastic sampler itself. Tuned DDPM was not compared.
- [GIVT](https://arxiv.org/abs/2312.02116) uses variance scaling; [Tencent CALM](https://arxiv.org/abs/2510.27688) has likelihood-free temperature; [DiTAR](https://arxiv.org/abs/2502.03930) injects noise partway along the ODE. All set it by sweep. NextStep-1 and VibeVoice: no temperature analysis found.

## Prediction

Written before any run. Guesses, not measurements.

- Round trip on toy and MAR-B: relative error below 1%.
- Innovations RMS ratio (real over self-sample) at 0.75 and 0.95: 0.95 ± 0.03. Ratios 0.25 and 0.5 reported with no prediction.
- Slice B: cold feedback at τ_e = 0.95 gives lower KID than the diagonal (observation only).
- Step 2a: a cold or mean arm sits below the diagonal frontier on MAR-B. On Pocket TTS, trained on noised context, default or hotter backbone feedback wins.

Fixed wording if refuted: "prediction failed: innovations calibrated (or wider than normal) on MAR-B"; "prediction failed: hot feedback won on MAR-B"; "prediction failed: no decoupled arm beat the diagonal frontier"; "prediction failed: cold feedback won on Pocket TTS".

## Design

Models and data. MAR-B (208M; head 6 blocks by 1,024 per the README) and its KL-16 tokenizer, official release (mirror: huggingface.co/jadechoghari/mar). Record hashes and sizes. Images: full-size Imagenette, centre-cropped to 256, labels mapped to ImageNet IDs. Generate only those 10 classes. The reference set is every Imagenette validation image that exists (count not checked; the reviewer recalls about 3.9k). Overlap with ImageNet train is not checked, so results stay provisional until ImageNet validation (gated, needs Rob's login).

Arms interleaved by batch, shared seeds. num_iter is 64 for Mac generation because 256 does not fit a slice.

**Step 1, slice A (30 minutes): plumbing and innovations.** Every process under 300 s, appending to one results file.
- 1a, toy on CPU. A small MLP diffusion head conditioned on a class label, trained on per-class Gaussians. Its own samples must invert to RMS 1.00 ± 0.01. Planted miscalibration: data shrunk toward each class mean by 0.9 must invert to RMS 0.90 ± 0.01.
- 1b, the temperature gate. 32 images per knob and τ, two seeds.
- 1c, 128 held-out images, teacher-forced at mask ratios 0.75 and 0.95, plus 0.25 and 0.5 labelled untrained. Invert every masked token. Record per-dimension variance, RMS, kurtosis and cross-token correlation, bootstrapped over images.
- 1c inversion floor (added after the search). DDIM inversion of real data is known to under-disperse. So also invert MAR-B's own τ = 1.0 deterministic samples with the same steps and context. The stage 1 band applies to the ratio of real-token RMS to this floor.
- 1c checks. Invert the same tokens at 50, 100, 250 and 1,000 head steps; RMS must stay within 0.005. Repeat on images downsampled to 128 and back, to show how much resolution alone moves the RMS. Plumbing: tokens generated at τ = 0.8 by deterministic DDIM, inverted with their own context, return 0.80 ± 0.01.
- 1d, subject gate (go or no-go for the stage 1 verdict and for slice B; tests the model and data, not the code). Three parts, read before any innovation ratio is scored. (i) Base task in reach: on 1b's own images (64 per knob and τ), nearest-class-centroid accuracy on DINOv3 features against the Imagenette reference must be at least 70 percent (chance is 10 percent) for both DDIM τ = 1.0 and MAR's DDPM 0.95, guidance off, num_iter 64. (ii) Context is used: on 1c's images at mask ratios 0.75 and 0.95, the head mean's per-token MSE to the real token with the true context must be at most 0.9 times the same MSE with context taken from a different image of another class; otherwise the innovations measure the marginal, not a conditional head. (iii) Room to resolve: the bootstrap 95 percent interval half-width of the real-over-floor RMS ratio must be at most 0.01 at both trained ratios, since the band is ±0.03 and the fail zone 1 ± 0.01. The 70 percent and 0.9 bars are stated choices; MAR-B's accuracy at num_iter 64 with a DDIM head is not checked. On a miss: (i) with DDPM passing and DDIM failing means the knob is broken, which is the 1b gate's stop; both failing means a pipeline or label-mapping fault, one fix, then park the card. (ii) failing stops stage 1. (iii) failing allows one rerun with more images inside the slice; still wide, stage 1 is reported as unresolved, with no verdict. Cost: (i) and (iii) are CPU arithmetic on features and innovations already planned; (ii) adds one teacher-forced forward pass per batch, inside slice A, with counts cut by the timing rule below if needed.
- The first process times one image and one inversion batch. If counts do not fit, cut them and log a dated change.

**Step 1, slice B (30 minutes): one feedback row.** τ_e = 0.95, num_iter 64, 128 images per arm, two seed sets. Arms: diagonal, τ_f = 0.75, head mean, Pasini k = 0.02. Metric: KID on DINOv3 features, equal generated and reference counts. A go or no-go observation for step 2a.

**Power check (one slice).** Diagonal at 0.95 on two seed sets, plus 0.85 and 1.05. If the 0.85 against 1.05 gap is under four times the seed spread, park step 2 until cloud approval.

**Step 2a, Mac-sized map (several slices).** Diagonal from 0.85 to 1.05 in steps of 0.025, plus the fitted-prior sampler. At τ_e equal to the diagonal optimum: τ_f ∈ {mean, τ_e − 0.2, τ_e + 0.1, Pasini k = 0.02}. Controls: cold feedback from an independent draw; a linear shrink of the emitted token toward the head mean; one num_iter 256 arm. Two seed sets. At least 5,000 images per arm if it fits, else KID. Analysis: fit Fréchet distance against recall along the diagonal; score each arm by its distance below that curve at its own recall, paired bootstrap over seed blocks. Exploratory: the best decoupled arm at the shipped guidance (cfg 2.9, linear, per the reviewer's reading of the README).

**Step 2b, cloud, needs Rob's approval.** Full grid with Inception FID, precision and recall on MAR-B and MAR-L. The author estimated 15 to 25 T4-hours, unmeasured.

**Step 3, Pocket TTS.** First read its tech report and code (gated; Rob must accept terms). Confirm head type, step count, where temperature enters, and whether there are two context paths. With one path, use one feedback route. With a one-step non-invertible head, drop speech innovations. Then 200 LibriSpeech test-clean sentences, three seeds: WER by local Whisper large-v3-turbo, F0 and duration variance across seeds, and WER in the last 2 seconds against the first 2.

## Checks before believing anything

Zero checks. The two-call sampler with τ_f = τ_e matches the one-call sampler bit for bit. Two disjoint image halves give RMS values inside each other's interval; this measures sampling noise only, and the step-count check covers inversion bias. Two seed sets of one diagonal arm give the noise floor; a gap under twice that is inconclusive.

Planted effects. The toy's 0.9 shrink is the test that the statistic detects a miscalibrated head. The τ = 0.8 round trip is plumbing. For the image metric, add Gaussian noise of known σ to real latents and measure the shift; the same noise on one arm's emitted tokens must reproduce it within spread.

## The control that could embarrass it

Hot feedback beats cold on a model never trained with context noise. Independent-draw cold feedback matches same-noise feedback. A linear shrink matches the two-call method. A finer diagonal point matches every decoupled arm. All four are in step 2a.

## Confounds

Guidance is off for every claim. MAR trains on mask ratios from a truncated normal with minimum 0.7 (reviewer's reading of mar.py), so 0.25 and 0.5 decide nothing. Resolution has the downsample control. Small-sample Fréchet bias has the power check. Blandness is handled by matched recall. MPS nondeterminism is handled by running the zero check on the same device and dtype.

## Kill criterion

Stage 1 stops if the toy shrink misses by more than 0.01, the step-count check drifts more than 0.005, or the temperature gate fails and cannot be fixed, each after one fix. Stage 2 parks if the power check fails, and dies if no decoupled arm beats the frontier by twice the noise floor and no speech arm gives more diversity at equal WER.

## Cost on this hardware

- Step 1: two 30-minute slices; the first process measures MPS time per image.
- Downloads: MAR-B, KL-16, full-size Imagenette, DINOv3 features. Sizes not checked; confirm disk stays above 15 GB.
- Step 2a: several slices, sized by the power check. A full grid is about a Mac day (reviewer's estimate), so it goes to cloud.
- Step 3: CPU hours (author's estimate).

## What a yes or a no changes

If the fitted-prior sampler matches the fine-diagonal optimum, practitioners calibrate without a sweep. If a cold arm beats the frontier on a frozen model, diffusion-head models get a sampler change costing one extra head call per token.

## How far it carries

The result applies to guidance-free MAR-B and its training recipe. Calibration and exposure bias depend on backbone and training choices (context noise, mask range, guidance) that differ across MAR, NextStep-1 and CALM. Untested: models above MAR-L, flow and consistency heads, video, RVQ codecs, guided sampling.

## Receipts to keep

Slice plans with dated changes, toy, step-count and floor tables, the 1b gate, per-token innovations, the bit-identity log, raw per-arm and per-seed metrics with features, and versions, hashes and the image list.

## Search gate, 2026-11-09

Queries: 22 web searches ran; 1 was refused when the shared budget ran out again. Semantic Scholar: all 24 citers of Pasini by title, and 807 citers of MAR by title keyword (one page; later pages rate-limited). Topics: emit/feedback decoupling, mean feedback, inference-only context noise in speech, Gaussianity of DDIM and PF-ODE inversion, diffusion-head temperature calibration, self-forcing for MAR, and temperature in GIVT, NextStep-1, VibeVoice and Pocket TTS.

Closest works, with how much was read (findings are under "What is already known"):
- [There and Back Again](https://arxiv.org/abs/2410.23530), Oct 2024, rev. Jan 2026: first half of the HTML.
- [Bjare et al.](https://arxiv.org/abs/2508.05306), Aug 2025: method sections.
- [eMIGM](https://arxiv.org/abs/2503.07197), Mar 2025: sampler section.
- [Spectral Collapse](https://arxiv.org/abs/2602.13303), Feb 2026: abstract and a summary of most of the text.
- [VideoMAR](https://arxiv.org/abs/2506.14168), Jun 2025, and [Parallel Rollout Approximation](https://arxiv.org/abs/2606.27978), Jun 2026: abstracts only.

Still open: MAR citers beyond the first page; the VideoMAR schedule itself; whether CALM's Gaussian temperature or DiTAR's report a precision-recall analysis; a tuned-DDPM against ODE comparison on MAR. None of these blocks step 1.

## Review record

- No scoop search (must-fix): fixed on 2026-11-09; see the search gate.
- Fixed, must-fix: stage 1 could not fail (trained ratios, band prediction); stage 1 cannot pick the reading (claim removed, readings added, predictive test in 2a); temperature definition (own best τ, 1b gate); planted effect (step-count check, toy 0.9 shrink on per-class Gaussians); 160 px Imagenette (full size, downsample control); underpowered metric (power check, KID fallback); grid artefact (fine diagonal frontier).
- Fixed, should-fix and minor: num_iter 64 plus a 256 arm; refutation wording in advance; Pocket TTS read first; scope cut to MAR-B; asset sizes in step 1; simpler first test split into two slices.
- After the search, 2026-11-09: added the 1c inversion floor, reading 5, and the DDIM-versus-DDPM point in 1b.
- 2026-11-10: subject-property gate review, after copies-as-evidence 1a ran a model that got no greedy base-task answer right (one template inaccurate, the other off-format). The power check already gates stage 2's room (bar and park), and 1b gates the knob's trend; nothing gated whether MAR-B at this setup makes class-recognisable images, whether teacher-forced inversion is conditional, or whether 128 images resolve the ±0.03 band. Added 1d with bars, stops and no extra slice. Risk the stage 1 test would have been void without it: medium, since MAR-B is a working model but the DDIM knob, num_iter 64 and the Imagenette mapping are untested here.

The reviewer's own three queries were refused by the budget. It fetched MAR's code and README directly; facts from them are marked as its reading.
