# Backward error for samplers: a calibrated, band-resolved noise test for approximate samplers

Status: queued, 2026-11-09 (revised after hostile review). Search gate passed: 22 keyword searches over three passes found no scoop; the watermark line closed in the third pass.

Covers "Backward error for samplers" and "A whiteness test for samplers" (the second is a musing at the end).

A fast sampler (fewer steps, caching, a distilled student, low-bit weights) produces images. The full model's deterministic ODE maps noise to images one-to-one, so each fast image is what the full model would make from some other noise, its "equivalent noise". If the fast sampler is right, those noises look like the full model's own. The question is whether this is a usable instrument at 100 to 300 images that sees what an image-spectrum test or feature metric at the same n cannot.

## Hypothesis

Two claims, judged separately. Claim I, the instrument works: on a guidance-free pixel model, a converged inversion of the model's own outputs comes back Gaussian in every frequency band, two independent reference sets compare as equal, and graded planted damage is detected. Claim II, it is worth having: at matched sample size, the noise test detects some planted damage or real sampler that the image-spectrum test does not.

## Theory

Wilkinson's backward error judges an algorithm by the input change that makes its output exact; here the input is the noise. If the reference flow T is a bijection, a sampler with output law q matches the reference law p exactly when T^-1 of its outputs is standard Gaussian, and TV and KL are preserved by T.

Resolution comes from a CPU simulation of the Gaussian null on 32x32x3 noise (1,000 draws). With equal-radius eighths, band coefficient counts are 21, 76, 128, 176, 220, 246, 120 and 37. At 100 images the two-sample ratio standard error (SE) is 2.5% in band 1, and D_max passes 3% a third of the time, so the original 3% threshold sat inside the null. With eight equal-count bands, SE is about 1.0% per band at 100 images and 0.58% at 300. The null 95th percentile of D_max is 2.8% at 100 and 1.6% at 300. Against N(0,I), the one-sample 95th percentile of the band maximum is 1.9% at 100 and 1.2% at 300. These assume Gaussian noise; the permutation test does not.

Assumptions: the inverter is close to the true inverse, and inversion error is similar across arms, so it cancels (measured, not assumed). A sampler can pass every statistic and still be wrong.

## What is already known

From 21 earlier queries, 8 sources opened, and the direct fetches in the search gate. Novelty is provisional (FRAME).

- [Anil et al.](https://arxiv.org/abs/2510.02692), 2025-10, first half read. Lemma 4.2 gives the TV identity. They invert data, not approximate-sampler outputs.
- [Bomatter, Geary, Gouk, The Signal in the Noise](https://arxiv.org/abs/2605.22496), 2026-05, read in full. Inverted-noise Anderson-Darling and spectral flatness for OOD detection at 32x32. No samplers.
- [Staniszewski, Kucinski, Deja, There and Back Again](https://arxiv.org/abs/2410.23530), v5 2026-01, first half read. DDIM inversion of a model's own samples is not Gaussian; patch correlation 0.382 against 0.039 on ADM-32. Appendix, read 2026-11-09: correlation plateaus with steps (ADM-64 0.133 at 100, 0.095 at 1,000); forward-noising only the first step brings it to about 0.055. No Heun, no frequency analysis, no samplers. The main risk to Claim I.
- [Daras et al.](https://arxiv.org/abs/2302.09057), 2023-02, read in full. Covers most of the whiteness idea.
- [Li and Bai, EMPURPLE](https://www.preprints.org/manuscript/202610.0546), 2026-10, abstract only (via Crossref). Not a scoop; its typical-set test passes 4-step DDIM-inverted noise (99.92% inside, 99.95% for real noise). A norm test is that blind; hence bands.
- Abstract only: [Spectral Collapse](https://arxiv.org/abs/2602.13303), 2026-02, excess low frequency in inverted latents, scored by a norm KS test; [Diffusion Noise Feature](https://arxiv.org/abs/2312.02625), a fake-image detector. No samplers.

Three search passes found no scoop and two cautions (gate section).

## Prediction (written before any run)

Step 1, EDM CIFAR-10 class-conditional, no guidance. SE is taken from the committed null simulation at the arm size actually run.

- Round trip with known noise: relative L2 error under 1%. 128 against 256 inversion steps differ by under 0.1%.
- Floor: against N(0,I), the band maximum sits below its null 95th percentile, and patch correlation is near the real-noise value. A low-confidence bet against Staniszewski's DDIM plateau.
- Self-comparison A against B: permutation p above 0.05.
- Amplitude plants (top image band of B's outputs scaled by 0.97, 0.93, 0.85): the image-spectrum test detects a smaller attenuation than the noise test. This is the image test's home ground.
- Phase plants (phase jitter in the top image band, magnitudes unchanged, three levels): the image-spectrum test is blind by construction. The noise test detects the strongest level and DINOv3 KID does not. Low confidence.
- Heun 18 steps (35 evaluations, EDM's default): inconclusive at this n. Low confidence.
- Heun 9 evaluations: confirmed, with the largest deviation a deficit in band 7 or 8. Sign and band are committed guesses.

Outcome table, per arm:

| Result | Reading |
|---|---|
| p < 0.05 and D_max ≥ 3% | Effect confirmed (an observation until it recurs on fresh seeds) |
| p < 0.05 and D_max < 3% | Real but below what matters: inconclusive |
| p ≥ 0.05, every band interval inside ±3% | No effect that matters |
| p ≥ 0.05, some band interval reaches 3% | Inconclusive; needs more images |
| 9-evaluation arm confirmed, deviation in another band or sign | Claim I holds; the band guess is reported as wrong |

Claim II is confirmed only if, on a phase plant or a real arm, the noise test is confirmed where the image-spectrum test reads "no effect" at the same n. A tie everywhere refutes it.

Step 2: 4-bit weights at full steps keep every band under 3%. The cache arm lands between full steps and 9 evaluations. 4-bit has larger same-seed LPIPS than the 9-evaluation arm but a smaller D_max.

## Design

Bands: eight fixed equal-count radial bands of the unshifted 32x32 FFT, 128 coefficients each, ordered by radius with ties broken by flat index. Band power averages over coefficients and the 3 channels.

Decisive metric: D_max, the largest absolute value over the bands of (mean band power, test arm) / (mean band power, reference A) minus 1. Its null is a label permutation test (pool both arms, shuffle labels 10,000 times, recompute D_max), exact for a maximum over fixed bands. Per-band bootstrap intervals are reported separately. A plant is detected at p < 0.05; the smallest level each statistic detects is recorded.

The same scorer runs on the images (the image-spectrum test) and on DINOv3 features as unbiased KID with a permutation null. Fréchet distance is not used: at 100 images its covariance is rank-deficient. Secondary: kurtosis, Anderson-Darling, spectral CV, patch correlation, sliced Wasserstein.

Inverter: EDM probability-flow ODE, Heun, rho=7, 128 steps, sigma 0.002 to 80; equivalent noise is x(80)/80. Every sampler's last step lands at sigma 0, and the inverter treats that image as the state at sigma 0.002, the same rule for every arm and plant. No forward-noising fix; it would hide deviations.

Model: `edm-cifar10-32x32-cond-vp.pkl` from the NVlabs EDM release (CC BY-NC-SA 4.0), loaded with that repository's code, hash recorded. Parameter count, FLOPs and MPS support not checked.

**Step 1, slice 1a: port, floor, power.**

1. EDM's sampler runs in float64, which MPS lacks. Port to float32. On 4 images, compare against CPU float64; above 0.1% relative difference, run that path on CPU. A batch rerun in a fresh process must be bit-identical.
2. Timing probe: one batch, logging evaluations per second and measured FLOPs per evaluation.
3. CPU null simulation at the chosen n, committed with the plan.
4. Subject gate, before any arm beyond 16 images. (a) Generate and invert the first 16 images of A (seeds 0 to 15) and compare each equivalent noise with its seed noise. Bar: median relative L2 error under 1%, and per-band power of the recovered noise within 0.5% of the seed noise in every band, half the 1.0% SE at 100 images, so inversion error cannot by itself fill the 3% decision margin. If it fails, rerun those 16 at 256 steps; if that also fails, park the card, since nothing the arms show could be told from inverter error. (b) Run F9 on the same 16 seeds and invert. Bar: in some band the same-seed equivalent-noise power differs from the seed noise by at least 3%. If no band does, F9 has no deviation to find and "Stop if F9 reads no effect" would kill on an absent effect: swap F9 for Heun at 5 evaluations (step 2's arm), check it against the same bar, and recommit the F9 prediction to that arm before slice 1b; if 5 evaluations also fails, Claim II has no real-sampler arm and rests on the phase plants alone, said in the plan. Cost: about 16 x 510 plus 16 x 264, roughly 12,400 evaluations, a quarter of arm A at 100 images; the (a) images are A's first 16, so only (b) is extra. Runs after the timing probe inside slice 1a, state on disk per process.
5. Arms, interleaved by batch in shuffled order, equal per class: A, Heun 128 steps, seeds 0 onward; B, seeds 1000 onward. Invert both. Apply the six plants to B's images and invert each. Pipeline-only noise plant: 32 images, seeds 2000 onward, band 8 of the starting noise scaled to 0.80, recovered within 3 SE. 16 images of A inverted at 256 steps. 16 images of A inverted with the first inversion step replaced by forward noising, a diagnostic only: if the floor fails and this passes, the failure sits near sigma 0, not in the flow.

**Slice 1b: real samplers.** F35 (seeds 3000 onward) and F9 (seeds 4000 onward). Regenerate both from their equivalent noise with the reference; if reconstruction error differs between arms by more than the effect, that arm is not read.

Arm size is 300 if the probe says it fits, else 100. Evaluations per image: A and B 510 each, plants 255 each, F35 290, F9 264, reconstruction 510; plus 16 x 511 and 32 x 510. If slice 1a does not fit 30 device-minutes at 100 images, drop the 0.97 and weakest phase plants, then split it in two. In slice 1b, drop reconstruction on F35 before cutting n. State goes to disk so no process passes 300 s.

**Step 2, two or three slices.** Same-seed LPIPS for every arm. Heun at 5 and 17 evaluations. Cache arm: on every second step, the predictor call D(x_{i+1}, t_{i+1}) reuses the previous corrector output D(x', t_{i+1}). 4-bit arm: per-output-channel round-to-nearest on conv and linear weights, with the first conv, last conv and embeddings in fp32. Gate: if the 4-bit images look broken in a 16-image grid, or image-spectrum D_max exceeds 20%, the named-pair prediction is void. No student of this checkpoint was found; [consistency_models_cifar10](https://github.com/openai/consistency_models_cifar10) (JAX, Apache-2.0) ships its own EDM teacher with distilled students, usable as a pair with their own floor (conditioning not checked).

**Step 3, raised cap.** SiT-XL/2 at 256, no guidance; about 675M parameters plus the SD VAE (sizes not checked). Latent bands are not image frequency bands, so report latent-band deviations as such, and name an image band only after testing the mapping.

**Step 4, exploratory, Rob's approval.** Qwen-Image 2.1 against Turbo (lineage not checked) and against INT8 and W4A8 with Rob's kernels, under guidance. Its own floor first. Likely days; not estimated.

## Checks before believing anything

- Self-comparison: p above 0.05. One failure means one rerun on fresh seeds; a second stops the card.
- Planted effect: the image-space plants are the real check, scored by detection and by whether the largest deviation lands in the top bands. The noise plant checks the pipeline only; on CPU, band-scaled Gaussian noise must be recovered within 1 SE.
- Convergence and the float32 port, as above.

## The control that could embarrass it

The image-spectrum test is in the headline. If it ties or wins everywhere at matched n, Claim II fails. Staniszewski's floor failing for a converged solver is the second; the first-step diagnostic says where.

## Confounds

Inversion error cancels in the two-sample design and is bounded by round trip, convergence and reconstruction. Eight bands are handled by testing the maximum. A deviation from the reference law is not damage by itself (Vouitsis); same-seed LPIPS and the images decide. Amplitude plants favour the image test, phase plants exclude it; both are reported.

## Kill criterion

Stop if the self-comparison fails twice or the pipeline plant is not recovered. Stop if the floor fails and the 0.85 plant is missed. Stop if F9 reads "no effect". Downgrade to "known identity, no new tool" if the image-spectrum test ties or wins everywhere.

## Cost on this hardware

Step 1: two slices of about 30 device-minutes each if the probe allows (not measured), under 1 GB disk, the EDM pickle and code. Step 2: two or three slices, plus `lpips` and AlexNet weights. Step 3: about 119 GFLOPs per SiT-XL/2 forward (DiT paper), several device-hours, about 3 GB of downloads. Step 4: not estimated, likely days, memory above 8 GiB.

## What a yes or a no changes

Both yes: speed-up papers get a few-hundred-image test with a stated floor that sees damage an image spectrum misses. Claim I only: a calibrated floor correcting Staniszewski for converged solvers, no reason to prefer inversion. Floor fails: their warning extends to converged solvers.

## What it says about larger models

The identity holds for any deterministic flow, but inversion error grows under guidance and latent bands lose their frequency meaning. A yes covers pixel-space, guidance-free models only.

## Receipts to keep

The plan and null simulation; pickle hash and code revision; the port check; every noise, image and equivalent noise, hashed; error checks; plant and per-band tables with p-values; the timing probe.

## Musing: the whiteness half

Daras covers the martingale property; the per-band increment ratio under guidance belongs in MUSINGS.

## Search gate, 2026-11-09

First pass: 4 queries refused. By direct fetch: Staniszewski's appendix and his 16 citers; none evaluates samplers by inverted noise ([SSI-DM](https://arxiv.org/abs/2602.02193), withdrawn; [Compression Asymmetry](https://arxiv.org/abs/2607.09784) quantizes stored noise).

Second pass: 10 queries run, 2 refused; sampler error by inversion, quantization, caching, students, latent goodness-of-fit. No scoop.

- [Vouitsis et al.](https://arxiv.org/abs/2411.08954), 2024-11, abstract read. Students with lower PF-ODE error give worse samples, so a deviation is not damage by itself.
- [Tang et al., Is Your Diffusion Sampler Actually Correct?](https://proceedings.mlr.press/v306/tang26ad.html), ICML 2026, abstract read. Same question for discrete samplers by exact oracle, no inversion; framing.
- [FLD](https://arxiv.org/abs/2410.02004), 2024-10, abstract read: a flow metric said to stabilize at a few hundred images. Possible extra control; code not checked.
- Not scoops: [Kurscheidt and Vergari](https://openreview.net/forum?id=f27xCLe8z7), ICML 2026, latent goodness-of-fit on data; [TCEC](https://arxiv.org/abs/2508.12094), SeaCache, FreqCa, forward only.

Third pass: 12 queries run, none refused; two citation-list fetches (Semantic Scholar, an aggregator) returned HTTP 429. Covered watermarks, numerical backward error, students through the teacher ODE, flow push-back tests elsewhere, spectrum two-sample tests, Daras follow-ups, few-step frequency error, GitHub. No work inverts approximate-sampler outputs through a reference flow and tests the noise.

- Watermarks, closed: [Tree-Ring](https://arxiv.org/abs/2305.20030), 2023-05, and successors ([rectified-flow limits](https://arxiv.org/abs/2504.03850), 2025-04; [OSI](https://arxiv.org/abs/2602.09494), 2026-02; [FARI](https://arxiv.org/abs/2607.26723), 2026-07), search summaries only. They detect a pattern planted in the noise by a Fourier test on inverted noise; higher-order inverters detect better. No sampler evaluation; support for bands and a Heun inverter.
- Name clash: [Gao et al.](https://arxiv.org/abs/2304.11446), 2023-04, abstract read. Backward error analysis for a step schedule; no inversion. Cite to avoid confusion.
- Band guess: a frequency-aware schedule paper, seen only on an aggregator (no arXiv id), claims pixel-distance schedules lose high frequencies at 5 to 15 evaluations on CIFAR. That would support the F9 band 7 or 8 deficit; unverified, prediction unchanged.
- Not scoops, summaries only: [Corvi et al.](https://arxiv.org/abs/2304.06408), 2023-04, generated images differ in mid-high radial spectra (supports the image control); [Qi et al.](https://arxiv.org/abs/2407.14041), 2024-07, inversion stability for seed choice.

Still open: the citation lists of Anil and Daras (429; keyword searches found nothing). No released dataset or better control turned up, so the design is unchanged.

## Review record

- Must-fix, all accepted: image-space plants replace the circular noise plant; thresholds moved outside the null (equal-count bands, 300 images, CPU-simulated); outcome table with an inconclusive zone, F9 committed, Claims I and II separated; power against the image spectrum and KID is the headline, with Heun 18 and phase plants so amplitude plants do not rig it.
- Should-fix and minor, all accepted: permutation for D_max; checks and drop order costed; KID for Fréchet; larger-model claim narrowed; scoop check; cache and 4-bit arms defined; one sigma convention.
- The reviewer confirmed the checkpoint, the 18-step default at FID 1.79 and the licence in the NVlabs README; MPS is not mentioned.
- 2026-11-10, subject-gate check after copies-as-evidence step 1a (model got no greedy base-task answer right, one template inaccurate, the other off-format; hypothesis never tested). The card had round trip and floor as predictions but no stop before the arms, and no check that F9 deviates at all before its "no effect" kill. Added slice 1a item 4: a round-trip bar on A's first 16 images (park on failure) and an F9 same-seed deviation bar (swap to 5 evaluations on failure). Risk the test would have been void without it: low; the checkpoint is a published model and the port check already existed, but an absent F9 deviation would have killed the card on nothing. F9's real deviation size not checked.
