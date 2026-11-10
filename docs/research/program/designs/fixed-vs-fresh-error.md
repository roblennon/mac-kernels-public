# Fixed error versus fresh noise: is weight-rounding error in a diffusion sampler a per-sample bias?

Status: queued (diffusion half only; the language half is parked), 2026-11-09. Search gate passed: the wider sweep ran in full (12 queries, none refused) and found no scoop; one design change followed (guidance-residual split, see Analysis).

Merges the ledger ideas `bias-or-noise` and "Fixed error versus fresh noise". Revised after a hostile review. Designed, not run.

A 4-bit diffusion model applies the same wrong weights to a slowly changing input at every step. Sampler-side fixes treat the error left after a calibrated mean as independent across steps; QuAKE and TCEC say so in their text (quotes below), and Q-Drift and AccuQuant are named by the earlier recheck. ASSERT states the persistence mechanism and leaves the cross-step measurement for future work. This card is the confirming measurement of that published idea, plus one test of whether the persistent part costs image quality. It is a measurement note, not a new mechanism.

## Hypothesis

In a flow-matching transformer with weight-only 4-bit weights, a large share of the velocity error persists across steps within each sample, and the per-timestep, per-channel conditional mean that QuAKE, PTQD and TCEC remove (an affine fit of the error on the output) does not remove it. At equal final-latent L2 error, persistent error costs more semantic image quality than fresh error. It is wrong if the residual after mean removal is close to uncorrelated beyond neighbouring steps, or if persistent and fresh error of equal final L2 do equal damage.

## Theory

Later steps see a drifted latent and can partly correct fresh error injected early. A persistent error pushes the same way every step, so they cannot. That is the mechanism under test.

Matching is the trap. A size rule e·sqrt(T/k) equates sizes only if the fixed error is perfectly persistent, so it builds in the hypothesis, and final-latent L2 is the sum it targets, so L2 cannot also judge damage. This revision calibrates each arm's scale to equal measured final-latent L2 and judges damage on something else: DINOv3 feature distance to the clean image, and error energy by frequency band.

## What is already known

From earlier searches (about 62 queries, often rate-limited) and the 2026-11-09 gate below.

- [ASSERT, arXiv 2609.00955](https://arxiv.org/abs/2609.00955), read in full earlier. Fixed analog weight noise gives correlated per-step errors; sampler noise helps. Leaves the cross-step cosine for future work. Old U-Nets, FID only. A DATE 2026 paper on a sampler called EtaMix, apparently from the same group, uses the same early-stochasticity idea (search snippet only).
- [QuAKE, arXiv 2609.21407](https://arxiv.org/abs/2609.21407), method and calibration appendix read. W4A4 (SVDQuant) on FLUX.1-dev, Sana-0.6B, PixArt-α and PixArt-Σ. Observation model fhat_i = γ_i f_i + ξ_i + ε_i (Eq. 12–13), statistics fitted per timestep and per latent channel with positions pooled (Appendix C). Below Eq. 15: "We assume that the process and observation noises are mutually and temporally independent." Its temporal model is for the clean signal, not the error.
- [TCEC / Error Propagation, arXiv 2508.12094](https://arxiv.org/abs/2508.12094), assumptions and setup read. States the per-step error is independent of other steps; models error as K_t ⊙ output with K_t per timestep, per channel. W8A8, W4A4, W4A8 only; no weight-only arm; no cross-step correlation measured.
- [Q-Drift, arXiv 2603.18095](https://arxiv.org/abs/2603.18095), first 100k characters read. One scalar residual variance per step; SVDQuant W3A4 and MixDQ W4A8 on six models including Sana. Its "independent auxiliary perturbations across steps" describe its own correction, not a measurement.
- [Stochastically Perturbed Weights, arXiv 2609.08412](https://arxiv.org/abs/2609.08412): in weather models, redrawing weight noise changed little.
- [PulseQuant, arXiv 2609.33384](https://arxiv.org/abs/2609.33384): on Wan, local error barely predicts final error.
- [When Guidance Goes Off-Scale, arXiv 2608.19644](https://arxiv.org/abs/2608.19644), summary of the full text read. One weight-noise draw held fixed for all steps on DiT-XL/2 and PixArt. Each branch stays near clean, but the guidance residual (their difference) rotates and shrinks; retuning the guidance scale closes most of the FID gap. No cross-step correlation, no fresh-noise arm.
- [TMPDiff, arXiv 2603.14062](https://arxiv.org/abs/2603.14062), summary of the full text read. On Sana and PixArt-Σ at W4A4, per-step error contributions add roughly linearly in the final-latent L2 norm (Pearson r about 0.98). Norms add linearly when errors align and in quadrature when they are independent, so this is an indirect hint of persistence; the paper does not test it.
- [Croci and Giles, arXiv 2010.16225](https://arxiv.org/abs/2010.16225), abstract read. In low-precision heat-equation solvers, round-to-nearest error accumulates systematically while stochastic-rounding error is mean-independent in time: the same contrast for arithmetic rounding, not fixed weights.

Open as far as checked: the measured step-by-step Gram matrix on any diffusion transformer, the three-way split, and any weight-only arm. Every sampler-fix paper found uses weight-and-activation formats. QuAKE tests Sana-0.6B, the model this card uses, so its W4A4 numbers give a direct reference point.

## Arms

Every arm perturbs only the linear weights inside transformer blocks. PyTorch on MPS.

1. Clean, replayed (self-comparison).
2. RTN 4-bit: symmetric round-to-nearest, group 32, fp16 scales. The deployed case.
3. Frozen stochastic rounding (SR): one SR draw on the 4-bit grid, kept for all steps.
4. Fresh SR: a new SR draw every step (and, in step 2, every k steps).
5. Fixed Gaussian: one draw, per-layer norm equal to that layer's RTN 4-bit error.
6. Fresh Gaussian: same norm, new draw every step.

Arms 3 and 4 are the main pair: real rounding error with identical marginals, differing only in persistence. The Gaussian arms are secondary. Gaussian noise is generated per layer on the fly from a seeded generator, x·(W+dW)ᵀ = x·Wᵀ + x·dWᵀ, so no weight copy is held. Floor/ceil alternation is dropped (see the review record).

Seeds come from a committed table. Arms interleave by sample: one process handles all arms for one or two samples, in a rotating order.

Model: `Efficient-Large-Model/Sana_600M_512px_diffusers`, downloaded with `variant="fp16"` and `allow_patterns` so the fp32 copies are skipped. Pin the revision hash in the committed plan. Euler flow matching, 20 steps, guidance 4.5, conditional and unconditional errors kept apart.

## Analysis

For each sample s and step t, the velocity error is e(s,t), the arm's velocity minus the clean velocity on the same saved input. Split the 16 samples into a fitting half and a held-out half (8 and 8), then swap. On the fitting half, fit the population mean in QuAKE's form: per timestep and channel, positions pooled, as an affine fit e = (γ−1)·v + ξ on the clean velocity v (Eq. 12). The affine term matters: a gain error on a slowly changing per-sample velocity would look like persistent per-sample error, and the fitted mean already removes it. A constant-only mean and a per-timestep, per-position mean are secondary variants. (Changed 2026-11-09 after reading QuAKE's Eq. 12.) Subtract it from the held-out half to get the residual r(s,t).

- Population share: energy removed by the mean, over total error energy.
- Persistent share: the average over step pairs t ≠ t' of ⟨r(s,t), r(s,t')⟩, over the average |r(s,t)|². This is the part of residual energy shared across steps within a sample.
- Fresh share: the remainder.
- Residual cosine against step distance 1, 5 and 10.
- Baseline: the same statistics with samples shuffled between steps, which gives the level a population-only structure would produce.
- Guidance residual: the same split on the error of (conditional − unconditional) velocity, reported beside the two branches. 2608.19644 finds this difference is the fragile channel under fixed weight noise, so persistence there matters more than in either branch. Reported, not a scored prediction. (Added 2026-11-09 from the search gate.)

## Checks before believing anything

- Self-comparison: clean replay reads zero, or a recorded MPS noise floor that every share must clear.
- Planted split: add to the saved clean velocities three parts of known energy share: a per-timestep, per-channel affine term in v (0.3), a per-sample vector held fixed across steps (0.4), and fresh noise (0.3). At 8 fitting and 8 held-out samples, the split must recover each share within ±0.05. If it fails, raise the sample count before any real arm is read.
- Pipeline checks, true by construction and not counted as predictions: fresh Gaussian and fresh SR have adjacent-step residual cosine within ±0.1 of the shuffled baseline; fixed Gaussian and frozen SR have high adjacent-step cosine.
- Step 2: the closed-loop scale calibration must bring each arm's final-latent L2 within 5% of the target before damage is compared.

## Predictions (written before any run)

Step 1 (open loop, numbers are commitments to be scored, not estimates from data):
- P1. RTN 4-bit, after per-timestep, per-channel mean removal: persistent share at least 0.3, and residual cosine at step distance 10 at least 0.2.
- P2. Frozen SR and RTN 4-bit: persistent shares within 0.1 of each other. Persistence comes from fixed weights, not from the shape of rounding.
- P3. Per-position mean removal leaves a persistent share at least 0.2. The per-sample part is not a positional artefact.

Step 2:
- P4. On the 4-step `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`, the persistent share is at most half of 20-step Sana's (the judges' sharper version). Confound: Sprint is distilled and runs at 1024 px, so a difference is suggestive, not causal.
- P5. Closed loop at equal final-latent L2: frozen SR's DINOv3 distance to the clean image is at least 1.3 times fresh SR's, and a larger share of its error energy sits in the lowest frequency band.
- P6. The conditional-mean correction (QuAKE's Eq. 33 form, used by PTQD and D2-DPM), per timestep and channel, fitted on separate calibration prompts and applied in closed loop to RTN 4-bit, removes less than 60% of its DINOv3 damage.

An effect that matters: a frozen-to-fresh DINOv3 ratio of 1.3 or more, outside the spread of clean-against-clean seeds at fixed noise.

## Steps

**Step 1 (one slice, open loop).** A 60-second timing probe first. One process encodes 8 committed prompts and saves embeddings (text encoder only). One process runs the clean trajectory for 8 prompts × 2 seeds, saving input and velocity at each step: 16 × 20 × 2 = 640 forwards. Replay processes push the saved inputs through arms 1 to 6: 6 × 640 = 3,840 forwards. About 4,500 forwards in all. Each process stays under 300 s and 8 GiB (transformer about 1.2 GB fp16, plus one RTN and one frozen-SR copy). If one forward takes more than 0.3 s, the Gaussian arms move to step 2. Step 1 alone scores P1 to P3, which decide whether step 2 is worth running.

**Step 2 (local slices).** 2a: Sprint replay for P4 (whether it shares Sana's text encoder is not checked). 2b: a short closed-loop pilot finds, for each arm, the scale on its error at which final-latent L2 matches RTN 4-bit; the scale is reported. 2c: closed loop at matched L2, 32 prompts × 2 seeds, frozen SR, fresh SR redrawn every k ∈ {1, 2, 5, 10}, RTN 4-bit with and without the QuAKE-style correction; VAE decode, DINOv3 distance, band energies. GGUF Q4_K and the repo's W4A8 codebook join here.

## The control that could embarrass it

Fresh SR at equal final L2. If it does as much semantic damage as frozen SR, persistence adds nothing beyond total size.

## Confounds a hostile reader would raise

- Gaussian is not rounding error: the main pair is SR.
- Open loop does not predict damage (PulseQuant): P5 and P6 are closed loop.
- Choosing k afterwards: the k set is fixed here.

## Kill criterion

Stop after step 1 if RTN 4-bit's persistent share after mean removal is below 0.15 and its step-distance-10 cosine is below 0.1. That outcome is written up as the first direct support for the independence assumption on weight-only formats. Stop after step 2 if the frozen-to-fresh DINOv3 ratio sits inside the clean seed spread.

## Cost on this hardware

- Step 1: about 4,500 Sana forwards at an unmeasured speed; the probe settles it. Downloads: transformer about 1.2 GB and text encoder about 5.2 GB. Saved tensors under 100 MB. Free disk 33 GiB on 2026-11-09, about 26 GiB after.
- Step 2: 3 to 5 slices; VAE about 1.25 GB more; Sprint's download not checked.

## What a yes or a no changes

Yes: weight-only 4-bit error has a per-sample persistent part the calibrated mean only partly removes, and it costs semantic quality. That shows sampler-fix authors a ceiling, not that their W4A4 methods lose quality for this reason. No: the independence assumption gets its first direct support on weight-only formats. Either way it is a short measurement note with saved error tensors.

## Carry to larger models, and what stays untested

Only the direction carries: a fixed error on a slowly changing input persists at any size. Its damage depends on error relative to signal, which shrinks as models grow. Untested: Sana 1.6B (the named next check), 14B video models, W4A4, samplers other than Euler.

## Language half: parked

Whether a fixed weight error causes more loops than one redrawn every token needs its own card; the full test is roughly 90 hours. Its first slice: clean, fixed and fresh k = 1 on 20 problems at 2,000 tokens, one seed, timing first, with fresh decode noise drawn at the layer output via dW·x ~ N(0, σ²|x|²).

## Receipts to keep

Plan, seed table, revision hashes, torch version, prompts, timing probe, check readings, per-arm error tensors and shares; in step 2, calibration scales, images and DINOv3 distances.

## Search gate, 2026-11-09

First pass: 6 queries ran, 3 refused (shared budget); fetches read four paper texts and four citation lookups. Closest then: QuAKE, Q-Drift, TCEC (above); [CentriQ](https://arxiv.org/abs/2610.06260) (October 2026, abstract) removes a per-token mean inside the network, not cross-step error. ASSERT and QuAKE had no indexed citers; none of Q-Drift's five citers measures cross-step error; "Pipis" matched nothing; the DRY loop paper (2608.22761) has no fixed-versus-fresh arm; SR turned up only training-time use and robustness ensembles ([arXiv 2312.00105](https://arxiv.org/abs/2312.00105)). Design change then: the mean removed became QuAKE's affine conditional mean.

Second pass, same day: 12 queries ran, none refused, covering EtaMix, cross-step error correlation and cosine, analog weight noise, SR at inference, dither, ODE rounding, weight-only Sana, timestep-varying rounding and weather-model perturbation.

New closest: [2608.19644](https://arxiv.org/abs/2608.19644) (August 2026, full-text summary), [TMPDiff](https://arxiv.org/abs/2603.14062) (March 2026, full-text summary, two passes), [Croci and Giles](https://arxiv.org/abs/2010.16225) (October 2020, abstract). From snippets only, none measures cross-step error: [Sampling-Aware Quantization](https://openaccess.thecvf.com/content/CVPR2026/html/Zeng_Sampling-Aware_Quantization_for_Diffusion_Models_CVPR_2026_paper.html) (CVPR 2026), [Q&C](https://arxiv.org/abs/2503.02508) (activation cosines), [PTQD](https://arxiv.org/abs/2305.10657) and TAC-Diffusion (residual as calibrated noise). Dither and SR searches found only training and signal-processing uses.

Still open: EtaMix's full text (the DATE 2026 PDF returned 404; snippets say it tapers sampler noise from stochastic to deterministic, which ASSERT already covers).

Design change: the guidance-residual split (see Analysis). TMPDiff's linear-norm additivity is a known result to compare against: a low persistent share in step 1 would leave it needing another explanation.

Scooped: no. Nobody found measures the cross-step error Gram matrix, splits persistent from fresh error, or runs a weight-only arm.

## Review record

A hostile review raised five must-fixes and five should-fixes; all were acted on.

- Circular metric under sqrt(T/k) matching: arms are now calibrated to equal measured final L2, and damage is judged on DINOv3 distance and band energies.
- Floor/ceil alternation: the arithmetic was wrong (two-step average only ties RTN, adjacent cosine is −0.5 by construction), so the arm and the "free fix" claim are gone.
- Predictions true by construction are relabelled pipeline checks; P1 to P6 can fail.
- The planted check now recovers a three-part plant through the full split at 8 + 8 samples.
- The language half is parked behind the reviewer's pilot (about 90 hours for the full test).
- Frozen versus fresh SR became the main pair; Gaussian arms are secondary; RTN 4-bit is the deployed case.
- Step-1 cost recounted at about 4,500 forwards. The output-space noise identity goes to the language pilot, since positions in one diffusion forward share dW.
- A yes is framed as a measurement note; the larger-model claim keeps only the direction, with Sana 1.6B as the next check.
- Minor: revisions pinned, fp16-only download, QuAKE's assumption quoted (done 2026-11-09).
- ASSERT holds the mechanism, so this is at best the confirming measurement. The scooping searches ran in the search gate above.
