# Paired-metric floor: park as research, fold the calibration into the next INT8 fidelity card

Status: parked, 2026-11-09. Search gate still open: all 3 queries on 2026-11-09 were refused (budget spent), so no independent scooping search has run.

Merges "Divergence vs damage (round 1, rechecked)" and "Noise floor of paired metrics and sampler self-consistency (merged cluster)".

People judge a quantized image model by comparing its picture with the full-precision picture from the same seed (PSNR, SSIM, LPIPS). The question was how much of that distance is harmless trajectory divergence rather than damage. Others have already made and measured that claim, so this card is not a research study. It is parked. Its design below is the calibration arm that the next INT8 fidelity card should borrow. That arm answers one in-house question: whether Rob can report his Metal INT8 kernel as "matching" a reference on a few-step model, using a same-seed number.

## Why parked

- The thesis belongs to Lee, and a benign-perturbation null for a diffusion transformer to Albaghdadi (below).
- The repo already compares Metal with stock at the same seed for five formats: Qwen GGUF Q4_K_M 27.24 dB and Wan 24.10 dB (`docs/findings.md` lines 546 and 560), Krea NVFP4 32.12 dB (line 586), Krea W4A4/INT8 34.34 dB and MXFP8 36.88 dB (line 620). Repeated sampled latents for Qwen GGUF were bit-exact (line 558). `docs/RESULTS.md` line 19 says the GGUF paths are close but not identical and explains why. The one missing cell is weight-only INT8 on Qwen-Image-2.1-Turbo, Metal against stock, next to a floor.
- The text conditioning for new prompts needs the 9.4 GB text encoder, above the 8 GiB cap. Whether 8 distinct prompts are already cached for this model is not checked.

## Hypothesis (for the fold-in)

On Qwen-Image-2.1-Turbo at 512, 8 steps, the benign rungs give a same-seed LPIPS at least half of what an INT8-sized weight error gives. The benign rungs are the stock dequantize path on the same INT8 file, and a bf16-sized per-row scale gain. The INT8-sized error is a per-element perturbation of the INT8 codes whose RMS matches INT8's own rounding error. If the benign rungs sit far below it, raw same-seed numbers are fair for INT8 on this model.

The INT8-sized rung is a proxy for INT8 against bf16. It runs inside the 8 GiB cap because it needs no bf16 weights. It assumes that two independent per-element errors of equal RMS move the sampler about equally. Step 2 tests that assumption against a real bf16 arm.

## Theory

A sampler is a chain of steps. A small early change can grow into a different but equally good image, and a same-seed metric counts that as loss. Two directions compete. Few-step models take large steps, so growth per step may be larger. Fewer steps also mean fewer chances to grow. Neither is checked. If amplification is roughly linear in weight error, LPIPS rises with the RMS weight error along one slope, whatever the perturbation's structure. Then the hypothesis is decided by the ratio of RMS errors before any image is made. The run looks for where that breaks: a per-row error and a per-element error of equal RMS may not move the sampler equally. The few-step direction is tested only by step 3, which runs the same ladder on the many-step base model. A steeper slope on Turbo favours "few-step amplifies more". A shallower one favours "fewer chances to amplify".

One repo data point: Turbo INT8 against its own bf16 read 41.3 dB PSNR, SSIM 0.997, on one image at 1024 (`docs/findings.md` line 316).

## What is already known

Novelty verdicts from agents stay provisional (FRAME).

- [Lee, Stepwise Asymmetric Quantization for Flow Matching Models](https://doi.org/10.5281/zenodo.19496643), Zenodo, 2026-04-10. Read in full in the earlier recheck. States the divergence-versus-damage thesis with FLUX.1-dev GGUF data. No benign floor, no seed ceiling.
- [Albaghdadi, Calibrating a Biomechanical Acceptance Test for Post-Training Quantization of a Video-Diffusion Transformer](https://doi.org/10.5281/zenodo.23143508), Zenodo, 2026-10-04. Record page re-read 2026-11-09. Null from 57 bf16 render pairs on a 5.77B video DiT, each differing only in attention kernel, a one-ULP weight dither, or platform. Same-seed PSNR null median 34.2 dB; full W8A8 sits 2.4 to 3.0 dB below. No LPIPS or step count on the page. Its benign rungs are this card's S and step 2 dither, on another model and modality.
- [Gandhi, Asaria and Salomone, INT8 and GGUF PTQ of Ideogram 4.0](https://arxiv.org/abs/2606.12280), arXiv, 2026-06-10 (v2 2026-06-12). HTML skimmed. Same-seed against FP8, 48 steps, 300 prompts, one seed: INT8 W8A8 18.53 dB PSNR, LPIPS 0.243. No floor, rerun check or seed ceiling.
- [The FID Lottery](https://huggingface.co/papers/2606.20536), arXiv 2606.20536, 2026-06-18. Most of the page read. FID seed variance on ImageNet SiT; no paired metrics, no quantization.
- Also seen: [Sathyanarayanan](https://arxiv.org/abs/2609.23125) and [Kaplan](https://arxiv.org/abs/2609.18005) for LLMs (not re-opened).

## Checked facts

- The INT8 Turbo file stores 192 INT8 linear weights, each with an fp32 per-row scale of shape [out, 1]; 73 other tensors stay bf16. Read from the file header on 2026-11-09. So a scale gain of (1 ± 2^-8) is applied exactly, with no bf16 rounding on top.
- The bf16 Turbo transformer and the INT8 conversion are both on the external data volume (seen 2026-11-09).
- `lpips` is not installed in the repo venv. Download size of the package and AlexNet weights: not checked.
- PartiPrompts licence: not checked. Check before committing the prompt list.

## Prediction (written before any run)

- Fresh-process rerun of the Metal path: bit-identical, LPIPS 0.
- CPU script: the per-row gain rung's RMS weight error is below INT8's rounding RMS by a factor of 4 or more. If the script says otherwise, rewrite these predictions before slice 1b.
- Dose ladder: same-seed LPIPS rises monotonically with RMS weight error.
- INT8-matched code-flip rung F: median PSNR 38 to 45 dB against M1, in line with the 41.3 dB data point.
- Stock path S: median PSNR 38 to 45 dB against M1, and its LPIPS at least half of F's. Metal and stock compute in different orders, and INT8 is already close to bf16, so the hypothesis may hold. Confidence is low.
- Seed ceiling C: median LPIPS above 0.4.

## Design

Model: Qwen-Image-2.1-Turbo (official 8-step checkpoint), the repo's INT8 ConvRot conversion. 512 squared, 8 steps, euler, cfg 1, saved sigmas, initial noise drawn on CPU, conditioning cached once and shared by all arms, fp16 VAE for every arm. No fallback model. Wan at one frame would answer a different question, so a failed memory probe parks the card.

Prompts: 8 from a fixed public list, committed before the run.

Decisive metric: same-seed LPIPS (AlexNet, `lpips` v0.1 weights) against M1, as paired per-prompt values. PSNR reported beside it.

**Before any GPU time (CPU, a few minutes).** Stream both files tensor by tensor. Compute, per layer and overall, the RMS relative weight error of INT8 against bf16, of a per-element bf16 half-ULP dither, and of the per-row gain. Write down each rung's RMS in INT8-rounding units and the LPIPS ratios the linear model predicts. Also list which prompts already have cached conditioning at this model's settings.

**Slice 1a, one lab slice (about 20 device-minutes).** First action: one image per path (Metal, stock) to read peak memory. Qwen-Image 2.1 INT8 at 512 measured 8.1 GB, and stock at 512 is not measured. If either path exceeds 8 GiB, stop and park. Then:

- M1 reference and M2 fresh-process rerun, 8 prompts.
- Dose ladder on 4 prompts, Metal path, INT8 codes changed by ±1 at random on a fraction p of elements (clipped to ±127). RMS error equals sqrt(p) code steps. INT8 rounding RMS is about 0.29 steps (one over root 12). So p = 1/192, 1/48, 1/12 and 1/3 give 0.25, 0.5, 1 and 2 times INT8's own error. A fifth rung uses ±2 codes at p = 1/3 for 4 times.

This slice already tells us whether the sampler is deterministic, the LPIPS slope against weight error, and the LPIPS of an INT8-sized error.

**Slice 1b.** S on 8 prompts. Per-row gain rung D, scale times (1 ± 2^-8), random sign per row, 4 independent dither seeds, 8 prompts. C, seed + 1, 8 prompts.

**Slice 1c.** F, the p = 1/12 rung, 4 flip seeds, 8 prompts.

At most 5 images per process. Arms interleaved by prompt in a shuffled order committed beforehand.

Decision rule, fixed now. For each prompt, take S minus the mean of D over its 4 seeds. Bootstrap a 95% interval over prompts for the median. The margin m is half the median D. If the interval lies inside ±m, "Metal matches stock within a bf16-sized floor". If it lies above +m, Metal and stock differ by more than the floor. Otherwise, inconclusive. Report the S median against the range of D medians across seeds. The hypothesis is tested the same way on S and D against F, with the margin set at half of F's median.

**Step 2, needs Rob to raise the memory cap** (bf16 transformer about 14 GB by parameter count). Add bf16 reference, INT8 Metal against bf16, and the per-element half-ULP dither on bf16 weights, which is the benign rung the reviewer preferred and which cannot run on INT8 codes. This tests whether F predicts real INT8 against bf16. Report the dither rung's median PSNR beside Albaghdadi's 34.2 dB null, as an outside reference, not a target. A mixed bf16-and-INT8 model was considered to stay under the cap and rejected: its reference is not bf16 either.

**Step 3, optional.** The slice 1a ladder on 4 prompts on Qwen-Image 2.1 (many-step), to set the few-step direction. A CUDA rung on Modal only if a write-up needs it, with Rob's approval.

## Checks before believing anything

- Self-comparison: M2 against M1 must read zero. If not, stop: nondeterminism is the floor. Also decode one cached latent twice and confirm identical PNGs.
- Planted effect, in weight space: the dose ladder must rise monotonically. Record the slope. If it does not rise, the instrument cannot see weight changes through this sampler, and nothing else is read.
- Scorer unit test (30 seconds, CPU): Gaussian pixel noise at sigma 0.01 and 0.04 gives PSNR within 0.2 dB of 20 log10(1/sigma). This tests the arithmetic only.
- Spread: D and F each have 4 seeds, so each floor has its own spread. A difference inside that spread is inconclusive.

## The control that could embarrass it

D or S reads as large as C. Then the instrument cannot separate a harmless change from a different image, and same-seed numbers on this model say nothing at 8 prompts.

## Confounds a hostile reader would raise

- Conditioning, noise, VAE or drift differ between arms: shared once, fp16 VAE everywhere, arms interleaved, host state logged.
- The per-row gain is not typical bf16 error: correct. It is correlated across a row and sits at the worst-case size. It stays as a labelled rung; the per-element rung moves to step 2.
- Code flips are not INT8 rounding: they match its RMS but not its distribution. Step 2 checks the proxy.
- Eight prompts are too few: an unconfirmed observation until it recurs on new prompts and seeds.

## Kill criterion

As research, already killed by prior art. Retire the calibration too if D and S both sit under 10% of F's LPIPS; then cards can use raw same-seed LPIPS with the seed ceiling. A standalone note is worth writing only if the floor covers half or more of the INT8 gap on two few-step image models.

## Cost on this hardware

Metal images take about 25 s at 512 (8 steps at 2.4 s measured on Qwen-Image 2.1, plus decode); stock about 40 s (not measured at 512); about 45 s load per process. Slice 1a, 38 images: about 20 device-minutes. Slice 1b, 48 images: about 23. Slice 1c, 32 images: about 14. CPU script: minutes. Disk under 100 MB. Step 2: about 15 GB process memory, several slices. Step 3: not estimated.

## What a yes or a no changes

Yes: Rob reports INT8 as matching within numerical noise on this model, and image cards report distances in floor units. No: raw same-seed LPIPS is fair for INT8 here. Slice 1b settles "Metal matches stock" for weight-only INT8. Neither branch is news for the field.

## Why it should carry to larger models, and what it leaves untested

The procedure carries to any size; the numbers do not, since a floor belongs to one checkpoint, step count, sampler and resolution. Untested: many-step models (until step 3), video, other samplers, CUDA, and whether the proxy F tracks real INT8 error (until step 2). The mechanism it could reject: that same-seed LPIPS depends only on RMS weight error, whatever the error's structure.

## Receipts to keep

Prompt list, arm order and seeds; the CPU RMS table; hashed PNGs and latents; one JSONL line per image (arm, prompt, seeds, peak memory, host state, s/step); the M2 byte comparison; the ladder slope; the per-arm table with intervals; the scorer revision.

## Search gate, 2026-11-09

Queries run: 0. Queries refused: 3 of 3 (the shared web-search budget was spent again before this pass started). No arXiv, OpenReview or Semantic Scholar query was tried, since the refusal forbids routing around the budget.

Instead, four cited records were re-read directly (Lee, Albaghdadi, the Ideogram INT8 paper, the FID Lottery); what changed is folded into "What is already known". None of them is a new closest work, and none measures a benign floor on a few-step image model.

Design change: step 2 reports its dither rung beside Albaghdadi's 34.2 dB null.

Still open: the whole scooping search (benign same-seed floors on few-step image models, LPIPS against weight-RMS dose curves, Metal or MLX INT8 fidelity). The gate stays until it runs with a fresh budget.

Confirmed by a second reader, 2026-11-09.

## Review record

Hostile review, 2026-11-09, verdict park. Must-fixes, all addressed: step 1 could not test the hypothesis without bf16 (INT8-RMS-matched flip rung added, step 2 checks it); the overlap rule favoured "matches" with one dither draw (four seeds, paired differences, written margin, inconclusive on straddle); the planted effect only tested PSNR arithmetic (weight-space dose ladder; pixel noise is now a scorer unit test); benign rung size was a free choice (CPU RMS script first; the per-element half-ULP dither moves to step 2 because INT8 codes cannot express it). Should-fixes, agreed: Metal against stock is mostly answered (prior numbers cited, card parked); theory direction ungrounded (competing prediction, step 3); memory fit optimistic (probe first, failure parks). Minor: three of four unchecked facts checked; PartiPrompts licence and download size still open.

The reviewer's six scooping queries were refused for budget, as were this pass's three; see the search gate.
