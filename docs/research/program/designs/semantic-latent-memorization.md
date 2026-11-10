# Semantic latents and the memorization threshold

Status: queued, 2026-11-09. Step 0 needs Rob. Search gate passed on the second pass: 10 searches ran, nothing found measures the threshold across input spaces at matched information; two last searches (REPA, other wordings) were refused and stay owed.

Merges "Does a semantic latent lower the amount of data a diffusion model needs before it stops memorizing, or does it only hide the copying?" and the disjoint-halves cluster.

Diffusion models trained on disjoint halves of a dataset copy their own halves when data is scarce, and agree for the same noise once data is plentiful. This card asks whether that threshold moves when the model works in frozen self-supervised features (the RAE recipe) instead of pixels, once the two spaces carry the same amount of information about the image. It matters where thresholds bind: fine-tuning and small-data domains such as medical images or personal LoRAs. Large base models train far above any plausible threshold.

## Hypothesis

At matched reconstruction error, equal parameters, equal latent shape and a matched stage of training, a 6M-parameter diffusion model at 32 px trained on frozen DINOv3 patch features reaches cross-half agreement at a training-set size at least 4 times smaller than the same model on pixels low-passed to the same error. The excess over what a Gaussian model of each space predicts is the part attributed to the encoder. The claim is wrong if the bootstrap interval on the threshold ratio sits below 2, and inconclusive if it crosses 2.

## Theory

Three readings predict different results. (a) Geometry: the space makes the data more Gaussian, and Wang, Zavatone-Veth and Pehlevan's random-matrix account of cross-split agreement then predicts the shift from the spectra alone. (b) Imported knowledge: the encoder carries structure from its own training set, so trained models beat the Gaussian prediction. (c) Compression: a lossy space has fewer distinct images to copy. Bubeck et al. report that stronger latent compression suppresses memorization. Reading (c) is not interesting here, so the design removes it by matching information.

Assumptions: an upsampled 32 px face still gives DINOv3 features with semantic content (Step 1 probes this). DINOv3 may have seen CelebA, so reading (b) is reported as "not separable from encoder exposure" until Step 4.

## What is already known

About 14 queries ran earlier, cut short by the search budget; the first 2026-11-09 gate added reads, the second added 10 searches (below).

- [Bubeck et al., What Makes a Good Medical Image Tokenizer?](https://arxiv.org/abs/2609.24691), 2026-09-21. The abstract describes 30 tokenizer configurations from ten families on twelve medical datasets. Memorization is mild and "further suppressed by stronger latent space compression". No semantic tokenizer, size sweep or disjoint halves.
- [Wang, Zavatone-Veth, Pehlevan](https://arxiv.org/abs/2602.02908), ICML 2026. Disjoint halves from 300 to 30,000 images, pixel space, explained through Gaussian statistics. Appendices not read, so a latent experiment there is not ruled out.
- [Favero et al.](https://arxiv.org/abs/2505.16959), 2025. Onset time grows with dataset size; supplies the 1/3 copy criterion.
- [Pai, Stancevic, Ma, Ambrogioni](https://arxiv.org/abs/2610.33337), 2026-10-23. Scaling law for "model information" across model and data size, from models on independent training sets, with a fitted critical size N_c(P). Pixel EDM on ImageNet, N from 10 to 100,000. The closest threshold framework; no latent arm seen.
- [Li et al., LIRF](https://arxiv.org/abs/2509.19903), 2025. Plain SiT-B/2 on FFHQ 100 to 2k in a DINO-aligned VAE latent gets no better recall than the standard latent (0.237 against 0.245 at 2k). Not matched, no copy metric, but weak prior evidence against the hypothesis.
- [Tong et al., Scaling Text-to-Image Diffusion Transformers with RAEs](https://arxiv.org/abs/2601.16208), 2026-01-22. Fine-tuning on BLIP-3o 60k, VAE models overfit after 64 epochs (training loss near zero); RAE models stay stable through 256. No copy metric, no size sweep. Their untested explanation is RAE's wider tokens (1152 against under 100), the shape confound this card fixes. Nearest result to the hypothesis.
- [Gu et al., On Memorization in Diffusion Models](https://arxiv.org/abs/2310.02664), TMLR 2025. EMM (largest N still memorized), pixel EDM on CIFAR-10: about 1k at 32 px, above 4k at 16 px, about 8k at 8 px. Lower dimension needs more data, against the direction of Bubeck's result. No latent arm.
- [Maillard and Goldt](https://arxiv.org/abs/2605.21402), 2026-05-20, abstract only. In linear models, agreement emerges once N is linear in input dimension. Here d = 3,072 in every arm, so shifts should come from the spectrum.
- Also: [Kadkhodaie et al.](https://arxiv.org/abs/2310.02557) (the original pixel disjoint-halves test, abstract only), [Zheng et al., RAE](https://arxiv.org/abs/2510.11690) (the recipe, judged by FID), [Zhang et al.](https://arxiv.org/abs/2310.05264) (transition depends on capacity relative to N), [Pu et al.](https://arxiv.org/abs/2512.23628) (longer 3D latents memorize less).

Left open after the second pass: the threshold across input spaces at matched information and latent shape, with a frozen self-supervised latent. Tong et al. make it timely.

## Prediction

Step 1 computes the Gaussian threshold ratio from each space's spectra and commits it before any model trains. Then:

- Planted ends: in every space, the exact memorizer reads copy rate above 95% and D above 0.9; the Gaussian denoiser per half reads copy rate below 2% at every N, and its D falls as N grows. The mixed denoiser reads its planted copy share within 5 points.
- At N = 256 per twin, all spaces copy: copy rate above 50%.
- Under the hypothesis, at N = 2,048 and matched relative loss gap, the pixel-matched arm's copy rate is at least 20 points higher than the DINOv3 arm's. If it is not, Step 2 counts against the hypothesis.
- Step 3: the DINOv3 to pixel-matched threshold ratio is at least 4 and exceeds the Gaussian ratio by more than the A against A′ spread.

## Design

**Data.** CelebA aligned, from the ungated Hugging Face mirror `flwrlabs/celeba`, config `img_align+identity+attr` (identity and 40 attributes; 25 parquet shards of about 0.4 to 0.5 GB, about 11.7 GB in all, per the Hub listing today). Stream one shard at a time, center crop to 140 px, resize to 32 px, keep uint8 (about 0.6 GB), delete the shard. Fix a held-out set of 10,000 images for PCA, probes, decoders and reference distances. Twin halves are drawn disjoint, without replacement, from the rest.

**Spaces, all 8x8 tokens of 48 numbers, whitened the same way.** Each space is rotated by a per-token PCA fitted on held-out data and whitened per channel. For 48-dimensional pixel patches this rotation is invertible.
- Pixel: 32x32x3, patch 4.
- DINOv3: `facebook/dinov3-vitl16-pretrain-lvd1689m` on the 32 px image upsampled to 128 px; 8x8x1024 patch tokens projected to 48.
- Pixel-matched: pixels projected onto the top k components of a global pixel PCA, rebuilt, then patched. k is set so the reconstruction error brackets the DINOv3 arm's error (below), at two or three values.
- Later: reconstruction autoencoders at two or three bottleneck widths that bracket the same error; random-weight DINOv3 with the same projection.

**Information matching.** A small decoder from DINOv3-48 to pixels, trained on held-out images, gives a reconstruction error on other held-out images. It bounds the loss from above, since the decoder is imperfect, so the match errs against the hypothesis.

**Model.** One DiT-style transformer, about 6M parameters, 6 blocks, width 256, flow matching with x-prediction, Euler sampler, 32 steps, fixed noise bank, the same code for every space.

**Arms per (space, N).** Twin A and twin B on disjoint halves. Twin A′ on A's half with another seed. Spaces alternate within each slice.

**Readouts.** In each space's own coordinates and in one shared judge, SSCD on pixels, decoding where needed and round-tripping training images the same way.
- D: mean squared distance between A and B samples for the same noise, divided by the mean between A samples for different noise.
- Copy rate: share of 512 samples whose nearest-neighbour ratio d1/d2 to the model's own half is below 1/3. Secondary, from Bubeck et al.: share whose top-1 SSCD similarity exceeds the 95th percentile of held-out-to-train similarity, so held-out data reads 5% by construction.
- Train and held-out denoising loss at every checkpoint, and the relative gap (held-out minus train, over held-out).

**The one metric that decides.** N*, the size at which D first falls below 0.5, interpolated on log N, in each space. The decision is the ratio N*(pixel-matched) / N*(DINOv3), with a bootstrap interval over seeds.

## Steps

0. **Rob's ticket.** DINOv3 is gated on the Hub: Rob accepts the licence. The weights are 1,212,559,808 bytes per the Hub listing. SSCD's size is not checked. Re-check disk before each download: 33 GB free today, floor 15 GB, peak need about 3 GB with streamed shards.
1. **Instruments and theory, mostly CPU, one to two slices.** Encode the held-out set and 2 × 4,096 training images with DINOv3, timed and cached. Fit projections. Semantic check: linear probes for a few attributes and identity retrieval, DINOv3-48 against pixel-PCA-48; if DINOv3 does not clearly win, retry at 64 px, then drop the arm. Train the DINOv3 decoder and fit the pixel-matched ladder. Plant pixel near-copies (1 px shift, JPEG, mild noise, colour jitter) among held-out images, encode them in every space, and check each detector flags them at SSCD's rate. Then run three closed-form denoisers through the real sampler and readouts in every space, N from 256 to 8,192 in 2x steps, so the curves bracket N = d = 3,072 (Maillard and Goldt) with room: the empirical-optimal denoiser per half (an exact memorizer), the Gaussian denoiser per half (small ridge below full rank), and a mix that copies with probability 0.25. The Gaussian curves give the predicted ratio, written down before Step 2. Informative alone: a Gaussian ratio of 4 or more means geometry explains the headline.

2. **First training, two slices.** A preflight times training steps, sampling and readouts. Slice 2b runs first: N = 256, twins A and B in each space, 6 models. Slice 2a: N = 2,048, twins A, B and A′, in pixel, pixel-matched (middle rung) and DINOv3, about 120 s each, 9 models. In each slice the pixel-matched twins train first and the other spaces then alternate. Checkpoints at four points per run. Read copy rate and D against step, comparing spaces at matched relative loss gap. If the preflight is slow, train 100 s per model; never drop N = 2,048.

   **Subject gate, read from the baseline run, first.** The planted checks test the readouts, not whether a trained model copies at all within the training the envelope allows. An undertrained model can read low copy rate and low D at every N, which would look like generalization. The gate reads Step 2's own pixel-matched twins as they finish, before the other spaces train. It screens that the trainer reaches memorization; it cannot certify the DINOv3 arm. Bars: at N = 256, copy rate at least 50% in both twins and D at least 0.7; at N = 2,048, copy rate at least 25%, so the 20-point gap has room to show. If N = 256 fails, retry those twins once at the longest training that fits one process (under 300 s including load, sampling and readouts) and use that length for every Step 2 model; if it still fails, park the card, because the trainer cannot reach memorization inside the envelope and multi-process training needs Rob. If only N = 2,048 fails, train the pixel-matched twins at 1,024, then 512, read the 20-point prediction at the largest N that clears 25%, and record the change before the other spaces train. The DINOv3 arm's competence comes from its own loss curves, not from this gate: a lower DINOv3 copy rate counts as generalization only at checkpoints where its relative loss gap matches the pixel-matched arm's, and if the two arms' gaps never overlap, the pair reads inconclusive, neither for nor against. Whether 120 s reaches memorization at N = 256 is known only once the preflight has run.
3. **Grid, needs Rob's approval for many Mac slices or cloud.** N in 2x steps around the transition seen in Steps 1 and 2, 3 seeds per cell, spaces pixel, pixel-matched ladder, autoencoder ladder, DINOv3, random-weight DINOv3. Train to held-out loss plateau. One capacity check: repeat one N cell at about 3 times the parameters. One whitened against unwhitened check per space at one N.
4. **Exploratory, later.** Hidden copying: compare decoded samples with decoded training reconstructions and originals, with the decoder's SSCD score on held-out reconstructions as the floor. A small non-web dataset (synthetic renders, or Rob's photos with his consent) to separate reading (b) from encoder exposure.

## Checks before believing anything

The zero is A against A′, with at least 3 seeds by Step 3; a space gap inside that spread is inconclusive. A same-noise resample must give D below 1e-6; that only shows determinism. The planted ends and the 0.25 mixture must read correctly in every space, through the real sampler, before trained models are read. Pixel-planted near-copies must be flagged at matched rates across spaces. In Step 3, duplicate 10 images 20 times each in one half and confirm every space and judge flags them.

## The controls that could embarrass it

The Gaussian denoiser: if trained models do not beat its ratio by more than the A against A′ spread, reading (a) explains the shift. The pixel-matched ladder: if the advantage disappears at matched error, it was compression. The random-weight encoder: if it moves the threshold as far, the shift is not learned knowledge.

## Kill criterion

Fix the instruments if a planted end, the mixture or the pixel-planted copies read wrong. Drop the DINOv3 arm if it fails the semantic probes at 32 and 64 px. Kill the claim if the bootstrap interval on N*(pixel-matched) / N*(DINOv3) sits below 2. Report reading (a) instead of (b) if the trained ratio sits within the spread of the Gaussian ratio.

## Cost on this hardware

Step 0: a licence click; 1.2 GB plus 11.7 GB streamed. Step 1: encoding time not measured, the rest minutes on CPU plus a small decoder on MPS. Step 2, including the subject gate, which adds no runs: two slices, about 18 and 12 GPU-minutes of training, sampling not measured, disk about 2 GB. Step 3: not measured; Step 2 throughput sets it.

## What a yes or a no changes

Yes: beyond geometry and at matched information, a semantic latent lowers the data needed before copying stops, which matters for small-data fine-tuning. No: semantic latents change geometry, as Wang et al. predict, and nothing more.

## Why it should carry, and what it leaves untested

This is a result about a 6M model at 32 px, the weakest case for transfer in FRAME. The capacity check tests one step up. Untested: text conditioning, larger models, higher resolution, less uniform data, speech.

## Receipts to keep

Timestamped plan with the Gaussian prediction; hashed ID lists; projections, probe scores, decoder error and k values; checkpoint hashes, loss logs, noise bank; raw per-sample distances and D; every check's output; commit, torch version, DINOv3 hash.

## Search gate, 2026-11-09

First pass: all 6 queries refused. 18 direct fetches instead: Wang et al. v2, about 200k of 243k characters (pixels only; memorization up to about n = 1,000, no split difference above 3,000); Bubeck et al. in full; Pai et al. except Appendix D; Li et al.'s main tables; abstracts of [Merger and Goldt](https://arxiv.org/abs/2606.14390) and [ICR](https://arxiv.org/abs/2606.09718) (no latent comparison); citers of Wang et al. and Favero et al.

Second pass: 12 attempted, 10 ran, 2 refused (a REPA query and its one retry; the per-turn budget was spent, so searching stopped). They covered five of the reviewer's six, plus medical, EMM and nearest-neighbour wordings. None measures a threshold across input spaces.

- Tong et al., about 100k of 117k characters via a summarizer.
- Gu et al., about 100k of 125k characters via a summarizer. Design change: the 32 px pixel EMM of about 1k is a sanity check on the pixel arm's N*.
- Maillard and Goldt, abstract. Design change: Step 1's sweep runs to 8,192.
- [Ganguli, Hunt and Kamb](https://icml.cc/virtual/2026/83844), ICML 2026 workshop abstract: memorization when local observations carry more than log N of information. Per pixel, no latent arm.
- Search snippets only: [Dar et al.](https://arxiv.org/abs/2402.01054) (medical latent diffusion copies patients, no space comparison); [Zhang et al.](https://arxiv.org/abs/2512.20963), ICLR 2026 (internal codes, not input space); [Pavlova and Wei](https://arxiv.org/abs/2506.07841), 2025 (disjoint CelebA subsets, pixel); [SVG](https://arxiv.org/abs/2510.15301) (quality only).

Still open: the REPA query, speech and privacy wordings, SSCD in latents, RAE's citers. Read Pavlova and Wei's CelebA protocol for a pixel reference curve. Compare N* with Pai et al.'s N_c when Step 3 reports.

## Review record

- Must-fix, information not matched: DINOv3 decoder error and a bracketing pixel-PCA ladder in Step 1, autoencoder ladder in Step 3.
- Must-fix, metric differs: same whitening for all arms, pixel-planted copies through every encoder, SSCD as shared judge.
- Must-fix, training stage: checkpoints, curves against step, matched relative loss gap, a 20-point prediction that can fail.
- Must-fix, self-comparison tautological: A against A′ is the zero; planted ends added. The low end is the Gaussian denoiser per half, because one model fitted to the full pool reads D = 0 by construction.
- Must-fix, not ready: Step 0 ticket; DINOv3 gated, size from the Hub, ungated CelebA mirror named, SSCD size not checked.
- Should-fix, all done: prediction from the Gaussian ratio; Gaussian sweep is Step 1; N* with bootstrap; semantic probes with a 64 px fallback; hidden copying moved to Step 4; budget split over two slices; scale stated with a capacity check.
- Minor, done: encoder exposure reported as not separable; Bubeck description corrected.
- 2026-11-10, base-task check after the lab's case 002 result: no gate confirmed that a model trained for Step 2's length copies at N = 256 or still copies at N = 2,048, so an undertrained floor could have read as generalization and counted against the hypothesis. Added the subject gate at the end of Step 1 with bars, a retry, a park rule and its cost. The semantic probe's "clearly win" still has no number.
- 2026-11-10, cross-family gate review (Sol): the gate now reads Step 2's own pixel-matched twins first (slice 2b first) instead of four separate models, and is scoped to trainer function. The DINOv3 arm's competence is read from its own loss curves at matched relative loss gap, inconclusive when the gaps never overlap.

Reviewer's six owed searches: five ran in the second gate pass; SSCD copy detection in latents is still owed.
