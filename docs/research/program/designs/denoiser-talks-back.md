# The denoiser talks back: what an image model's text stream knows beyond the image stream

Status: queued, 2026-11-09. The novelty search gate passed on 2026-11-09: partly scooped, core question still open (see "Search gate, 2026-11-09").

Long shot (round 2 shortlist S09). Trained readers already pull unstated image details out of the text-side tokens of diffusion transformers. In a single-stream model like Krea 2 that is close to guaranteed: each text token's attention output mixes image-token values. This card asks two sharper questions. Does the text stream hold seed-specific image information that the image stream at the same block and step does not already expose to a linear reader? And is it readable with nothing fitted, in the model's own conditioning space? Steps 1 and 2 use a trained linear probe, so their contribution is the per-block, per-step profile, not training-free reading. Only step 3 tests that.

## Hypothesis

In Krea 2 Turbo, at block 14 and step 2 of 8, pooled text-token states predict a seed-specific image property beyond what pooled image-token states at the same point predict. This is wrong if the incremental R² of text over image, at that pre-registered cell, sits inside its own permutation null.

## Theory

Text tokens attend selectively over image tokens, so an attention-weighted summary can keep what a fixed image pool throws away. Part of any positive result may be pooling, not a "register" effect; the 4×4 grid pool narrows that. 2607.19139 showed causally that template tokens take object identity from the image latents. In Krea 2 the text passes through `txtfusion` and `txtmlp`, then shares weights with image tokens in 28 single-stream blocks (`docs/krea2-blocks-and-variety.md`).

## What is already known

- [Learning to Read the Contextual Tokens in Diffusion Transformers](https://arxiv.org/abs/2610.06844), 2026-10-05. A trained Q-Former over all layers' text-side outputs, into a frozen LLM, reads unstated, seed-varying details on SD3.5 and FLUX.2. Its Appendix B.1.1 is the closest result to this card: the contextual reader beats the same reader on the output latent at every timestep, and beats a VLM on the predicted clean image before about 40% of denoising. That is a nonlinear reader against the latent, not against image tokens at the same block. No per-layer profile, linear probe, fit-free readout or code.
- [Diagnosing and Correcting Concept Omission in MM-DiTs](https://arxiv.org/abs/2605.14270) (OSI), ICML 2026, v3 2026-07-31. Linear probes on concept text-token keys, per head, layer and timestep, predict whether a concept appears in the final image (FLUX.1-Dev, SD3.5; top head 91%; middle layers and intermediate timesteps best). No image-token baseline; seeds per prompt not stated. Code is public.
- [Text Template Tokens Are Implicit Semantic Registers](https://arxiv.org/abs/2607.19139), 2026-07. Causal, Krea-2-Turbo in the appendix. No probe or readout.
- [NitroFusion2](https://arxiv.org/abs/2611.05648), 2026-11-03. On Z-Image-Turbo, a single-stream model, a discriminator reading text-stream states beats one reading image-stream states for adversarial distillation. A usefulness test, not an information measure.
- [T2I Lens](https://arxiv.org/abs/2605.24624), [Contextual-space repulsion](https://arxiv.org/abs/2603.28762) (early contextual-token edits set per-sample diversity, also on Turbo models), [Prompt Reinjection](https://arxiv.org/abs/2602.06886), [Patchscopes](https://arxiv.org/abs/2401.06102), [Probe-Select](https://arxiv.org/abs/2603.02829).

What is done: text-side states read seed-specific image content, trained (Dahary) and linear (OSI, binary targets, dual-stream models). What is still open: text over image tokens at the same block and step, on a single-stream model, and a fit-free readout.

## Prediction

Written before any run. Targets are final-image properties centred within each prompt, so prompt content explains nothing. All numbers are cross-validated R² read at step 2 (100 prompts × 4 seeds), not step 1.

- Pre-registered cell: block 14 output, sampler step 2 of 8, all text tokens mean-pooled, DINOv3 PC1. The deciding quantity is ΔR² = R²(text + image) − R²(image), with the image features being 4×4 grid-pooled image tokens at the same block and step.
- I predict ΔR² clears the 95th percentile of its permutation null, at about 0.03 to 0.08. My credence in this is about one in three. My best single guess is that the image stream already exposes almost everything.
- Image-stream R² alone at the cell: at least 0.4. Last block, last step: at least 0.7. These are positive references, not the claim.
- Suffix versus content tokens: no reliable difference. Krea passes only about five suffix tokens to the DiT.
- Qwen-Image 2.1, a second architecture: ΔR² at the matched cell inside its null. Its stream layout is not checked.
- Step 3a, forward caption match: 4-way seed matching above chance (25%) at block 14, at 35% or more. Under 30% is a null.
- Step 1 makes no R² prediction. At its sample size it is a pipeline and envelope check only.

## Design

**Model and route.** Krea 2 Turbo, 512×512, euler simple, CFG 1, 8 steps, `bench/run_krea2.py`. Encoder `qwen3vl_4b_bf16` (8.3 GB, about 7.7 GiB). DiT: the 7.1 GB `krea2_turbo_nvfp4` file first, since it already completed an isolated captured-conditioning replay with normal pressure (`docs/findings.md`, 2026-10-04); the 13 GB int8 file only if the pre-flight shows it fits. Prompts are encoded once, in their own process, through `bench/krea_conditioning.py` and `bench/conditioning_cache.py`; renders replay the cache.

**Memory.** The lab guard enforces group RSS, not Metal memory. Every process logs both. If encoding breaks 8 GiB RSS, stop and get Rob's explicit exception or another route.

**Targets and prompts.** DINOv3 ViT-L/16 (local) embeds each image. Targets: top 3 principal components of within-prompt-centred embeddings, mean Lab a* and b*, left-right luminance balance. PC1 decides. Prompts are under-specified (car colour, dog breed, background), committed before the run, plus one empty prompt.

**Capture.** Hooks save states at the input to block 0 and outputs of blocks 0, 7, 14, 21, 27, at steps 1, 2, 4. Text pools: all tokens, the trailing `<|im_end|>\n<|im_start|>assistant\n` suffix, and content. Krea's encoder strips the system and user-opening prefix (ComfyUI `comfy/text_encoders/krea2.py`); take exact positions from the tokenizer. Image tokens mean-pooled and 4×4-pooled; noise pooled to 8×8. All fp16.

All arms come from the same forward passes. Generation order is shuffled and logged.

**Analysis.** Centre states and targets within each prompt. Leave-one-prompt-out cross-validation. Inside each fold: fit PCA (k = 8 per feature set), then ridge with alpha from inner CV. ΔR² compares the joint model with the image-only model in the same folds. The null for the pre-registered cell is 500 within-prompt permutations of the targets through the identical pipeline. Any claim made across the grid (6 depths × 3 steps × 3 text pools × 5 targets) uses a max-statistic null: the 95th percentile of the grid-wide maximum across the same 500 permutations. Every other cell is exploratory and labelled so. Added after the search gate, exploratory: ΔR² of text over the 4×4-pooled block-0 input image tokens (the patch-embedded latent), a linear version of Dahary's latent-reader comparison, so the gap between "beyond the latent" and "beyond same-block image tokens" is measured from the same captures.

**Steps.**

0. CPU only. Simulate the analysis on Gaussian data at the real width (6144), k = 8, and the n of steps 1 and 2. Plant a direction at ΔR² = 0.20 and 0.05. Record its share of total state variance, whether it enters the top 8 components, and the spread of recovered ΔR². This sets the tolerance. If it is wider than ±0.1 at step 2's n, enlarge step 2 or change k first.
1. One slice. Pre-flight A encodes all prompts and records RSS, Metal memory and time. Pre-flight B renders 2 images and records load time, seconds per image and memory. Then render at most 6 prompts × 4 seeds plus 2 repeats, as many per process as fit in 300 s, with loads counted against the 30 minutes. If that does not fit, drop to 256 px or 5 prompts and log it. Alone it tells us whether the route fits the envelope, whether rendering is deterministic, whether the checks pass, and what step 2 costs. It reads no R² as evidence.
1b. Subject gate, before the deciding analysis. The test needs two things of the subject, not the instrument: seeds of one prompt must give different final images, and the step-2 image stream must leave room above it for a text increment of the predicted size. Neither is checked yet; Turbo distillation can collapse seed variety (see `variety-census.md`). (a) Read from step 1's images, first: the within-prompt difference between seeds' DINOv3 embeddings is above the re-render difference on at least 5 of 6 prompts. (b) Read from step 2's baseline arm, first: render step 2's first 20 prompts × 4 seeds as their own block, and compute image-only R² for PC1 at the pre-registered cell and at last block, last step; no text-stream number is computed until (b) is read. Bars: last block, last step at least 0.5; the pre-registered cell at most 0.9, which leaves at least 0.1 of variance, above the 0.08 top of the predicted ΔR² band. There is no lower bar on the cell: a low image-only R² leaves more room for text, not less. Step 0 must show R² spread within ±0.15 at n = 20 prompts, else make the block larger. If (a) fails, change prompts toward more open subjects once, then park. If the last-block reference fails, stop: targets or hooks are broken, or there is nothing to predict. If the cell is over 0.9, re-register at step 1 or block 7 in writing (both are in the same captures, so no new renders), drop the 20 gate prompts from the deciding analysis and replace them with 20 new prompts committed before rendering, or park. If the gate passes, the 20 prompts stay in step 2. Cost: no renders beyond step 2's own unless the cell is re-registered.
2. 100 new prompts × 4 seeds on Krea 2; the pre-registered cell decides. Then Qwen-Image 2.1 (6.8 GB int8 file), with sampler and steps taken from the repo's Qwen runner and written here first.
3a. Forward caption match, fit-free, only if step 2 passes. Caption each final image. Send the 4 captions through Qwen3-VL-4B, `txtfusion` and `txtmlp`, the model's own path with nothing fitted. Mean-pool, centre within prompt, and match each seed's centred text state at blocks 0 and 14 to its own caption by cosine. The captioner is not chosen and its download is not checked.
3b. Inverse map, only if 3a beats chance. Fit a ridge map per block from block-k text states (single-step forwards on 2,000 captions) to Qwen3-VL-4B tap layer 20 (one of the 12 taps Krea uses: 2, 5, …, 35). Patch the mapped state at layer 20 into a "Describe this:" prompt and generate 30 tokens. Taps 8 and 35 are exploratory. Positive control for the map: add the block-k difference between two known captions to a text-stream state and check the readout moves toward the planted caption. A blind judge (local Qwen3.5-4B) matches readouts to captions 4-way.
4. Trained-reader comparison on the same states. Dahary released no code, so this means reimplementing their reader. Probably needs cloud time and Rob's approval.

## Checks before believing anything

- Self-comparison, hook placement: pre-block-0 text states must be bitwise identical across seeds of one prompt, so their within-prompt R² is exactly 0.
- Self-comparison, full pipeline: 2 (prompt, seed) pairs are rendered twice. Run "text-stream features minus their re-render copy" through the same CV and null; it must read zero within the measured render-to-render noise. Record whether MPS rendering is bitwise deterministic. If it is not, that difference is the noise floor.
- Planted effect: tolerance and plant scale come from step 0, not from a guess. A plant at 0.05 must land inside the null.
- Positive reference: image tokens at the last block of the last step must reach high R². If not, the targets or hooks are broken.

## Controls

The main control is the image stream at the same block and step, built into ΔR². The pooled-noise probe stays only as a floor at step 1, block 0. It is linear on raw noise, so beating it later says little.

## Confounds

- *Pooling discards spatial detail, so text wins by default.* The 4×4 pool narrows this. A positive ΔR² is reported as "beyond a 4×4-pooled linear image reader", not "beyond the image stream".
- *The readout paraphrases the prompt.* All 4 seeds share the prompt, so a paraphrase matches at chance.
- *One distilled checkpoint.* Claims stay per checkpoint.

## Kill criterion

- Step 0: if the plant cannot be recovered within ±0.1 at any affordable n, stop and write up why.
- Step 1: if encoding or rendering cannot fit 8 GiB RSS and no exception is granted, park.
- Step 1b: the bars and stops are in the step. Step 2 renders no prompts past its first 20 until 1b passes.
- Step 2: if ΔR² at the pre-registered cell sits inside its null on Krea 2, stop. Report the grid as exploratory.
- Step 3a: if matching sits within noise of 25%, do not build the map. Write up that the content is not aligned with caption directions in the model's own conditioning space, which is narrower than "absent from the encoder's space".
- Step 3b: a null counts only if the positive control passes.

## Cost on this hardware

- Step 0: CPU minutes.
- Step 1: one 30-minute slice including loads, encode pre-flight and render pre-flight. Under 0.5 GB disk. No downloads.
- Step 1b: no extra renders; it reads step 1's images and step 2's first 20 prompts. 80 replacement renders only if the cell is re-registered.
- Step 2: the reviewer's rough figure, counting reloads, is 4.5 to 5 guarded hours per model, 10 or more slices. Recompute from step 1. About 2 GB disk.
- Step 3a: minutes of compute plus captioning; captioner download not checked.
- Step 3b: about 2.5 device-hours for paired forwards, not re-checked. Keep free disk above 15 GB.
- Step 4: cloud, not estimated; needs Rob's approval.

## What a yes changes and what a no changes

A positive ΔR² is a per-block map of what the text stream holds beyond the image stream, separating register storage from copying. A yes at 3a is a fit-free monitor of what the model has committed to at step 2 of 8. A no at step 2 says trained readers of a single-stream model may be reading copied image statistics.

## Why it should carry to larger models, and what it leaves untested

Krea 2, Qwen-Image, FLUX.2 and Lumina all put LLM-encoded text into joint attention, but their layouts differ, so Krea's result does not transfer by default. Untested: dual-stream models until Qwen-Image is checked, other samplers, non-distilled checkpoints, and any causal claim.

## Receipts to keep

Committed plan, prompts, seeds and order log. Step 0 outputs. Pre-flight memory and timings. PNGs, embeddings and state arrays with SHA-256 hashes. Check outputs. One CSV of R², ΔR² and nulls per cell. Model hashes, repo commit, torch version. Step 3 captions, readouts and judge decisions verbatim.

## Search gate, 2026-11-09

Queries: 18 web searches and 2 arXiv API listing queries ran; none was refused. A Semantic Scholar fetch was rate-limited (HTTP 429). Wordings covered probing, registers, lenses, seed selection, Krea and Qwen-Image interpretability, OpenReview, GitHub and Hugging Face papers.

Closest works, and how much was read:

- [2610.06844](https://arxiv.org/abs/2610.06844), Dahary et al., 2026-10-05. Full HTML read, appendices included; figures not viewed. No project page or code found.
- [2605.14270](https://arxiv.org/abs/2605.14270), Baek et al., ICML 2026, v3 2026-07-31. Probe sections read; repo not opened.
- [2611.05648](https://arxiv.org/abs/2611.05648), NitroFusion2, 2026-11-03. Abstract and the stream ablation (Section 4.3) read.
- [2605.24624](https://arxiv.org/abs/2605.24624), 2026-05-23. Abstract only.

Verdict: partly scooped; the gaps named in "What is already known" are still open. The design gained one exploratory arm (text over the latent, beside Dahary's comparison) and a note on step 4.

Still open: whether OSI used several seeds per prompt; Dahary's figure values; a citing-papers check on Semantic Scholar once its rate limit clears.

## Review record

Hostile review, 2026-11-09, verdict queue-after-fixes.

All must-fixes done: cached-conditioning encode with memory logged; ΔR² over 4×4-pooled image features decides; one pre-registered cell plus a max-statistic null; step 0 sets the tolerance; fit-free caption match (3a) gates the map, with a positive control. Should-fixes done: suffix tokens from `krea2.py`, re-render self-comparison, image stream as main control, novelty search (see "Search gate, 2026-11-09"). Minor fixes done: Qwen-Image as a second architecture; step 2 cost recomputed after step 1.

Reviewer's findings: arXiv 2610.06844 and 2607.19139 (Li et al., 2026-07-21, v2 07-30) confirmed real by direct fetch. Local checks: the Krea int8, Krea nvfp4, Qwen3-VL-4B bf16, Qwen-Image 2.1 int8 and DINOv3 files are on disk; the guard caps group RSS; Krea's encoder strips the template prefix.

Base-property review, 2026-11-10. Prompted by copies-as-evidence step 1a, where every measurement check passed but the model got no greedy base-task answer right (one template inaccurate, the other off-format), so nothing was tested. This card had no gate confirming that Krea 2 Turbo varies across seeds or that the step-2 image stream holds seed content off floor and ceiling; the only positive reference ran inside step 2. Added step 1b with bars, stops and cost. Risk that step 2 would have been void without it: medium. Seed-variety collapse and image R² at the cell are not checked.

2026-11-10, cross-family gate review (Sol): dropped the 10% seed-variance share and the 0.1 lower bar on the cell, tied the 0.9 upper bar to the predicted ΔR² band, and moved part (b) from a separate 80-render gate set into step 2's first 20 prompts, read before any text-stream number.
