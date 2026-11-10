# The squared-error-optimal guidance weight, measured on held-out data

Status: queued, 2026-11-09. The search gate is cleared: the second pass ran all 12 queries with none refused, and nothing found scoops the card (see "Search gate, 2026-11-09").

A diffusion model trained with squared error should output a conditional mean. Then the best linear mix of its conditional and unconditional outputs puts weight exactly 1 on the conditional one: no guidance. This card measures, per noise level and with forward passes only, how much loss a rescaled guidance weight would remove, on the model's training distribution and on held-out data, to separate model error from distribution shift.

## Hypothesis

On held-out COCO images and captions, SD 1.5 has a guidance weight other than 1 that removes at least 1% of its conditional loss, at three or more adjacent noise levels. On images and alt-text from its own training source the saving is smaller. This is wrong if the saving stays under 1% at every level on both sets.

## Theory

Let v be the per-sample training target at noise level t (the noise ε for SD 1.5, the velocity for a flow model). Let v_c and v_u be the model's conditional and unconditional outputs, and d = v_c − v_u. Define, as averages over all samples at one level:

- w*(t) = E⟨v − v_u, d⟩ / E‖d‖², the one weight per level that minimises the loss of v_u + w·d.
- D(t) = E‖d‖² / E‖v − v_c‖², the size of the caption's effect relative to the conditional loss.
- R(t) = (w* − 1)²·D(t), the share of the conditional loss that w* removes. This follows because the loss is quadratic in w.
- g*(t) = E⟨v, v_c⟩ / E‖v_c‖², a scalar gain, reported as a side quantity.

If v_c = E[v | x_t, c], the residual v − v_c is orthogonal to every function of (x_t, c), including v_u. Then w* = 1 and R = 0 exactly, however poor v_u is. Because E[v | x_t, c] is the ideal conditional target, averaging the per-sample v gives an unbiased numerator with no sampling.

Why R decides. At very high and very low noise the caption barely changes the output, so E‖d‖² nears zero and w* becomes a ratio of two small numbers. R stays bounded and says how much loss is at stake.

Assumptions. (1) The identity holds on the training distribution only; off it, any imperfect model can give w* ≠ 1. (2) Memorised pairs pull the in-distribution w* toward 1. (3) Guided sampling visits states the forward process never produces (Galashov et al.), so one-step regression may miss most of guidance's effect.

## What is already known

This rests on a colleague's search (about 18 queries), a full read of Galashov et al., and two search passes on 2026-11-09 (below). Nothing found reports w* or R from real targets on the forward process. The nearest is PathGuide, which fits a per-interval weight to data under the sampler's own rollout, on toys and MNIST only.

- [PathGuide](https://arxiv.org/abs/2608.29107), 2026-08, full HTML read via two fetch summaries. Its per-interval weight is a one-dimensional least-squares ratio of the same form as w*. The target is the exact conditional field, rebuilt from real endpoint samples. But it is tested against chosen test functions under the rollout law, not the forward process. Experiments use a 2-D two-class mixture and MNIST, and report no per-interval values. The weight is clipped below at 1. The clip binds in 53% of updates, and relaxing it to 0 worsens KL, W2 and MMD in every flow tested. That bears on assumption (3): on the rollout the unclipped optimum often sits below 1, while this card predicts about 1 on the forward process.
- [Kong et al.](https://arxiv.org/abs/2310.07972), ICLR 2024, read via fetch summaries. SD 2.1 on filtered COCO val2017 captions, with code released (github.com/kxh001/Info-Decomp). Fig. 5 plots conditional and unconditional MMSE against log-SNR for 10 images. There is no table of the gap. They compare two per-level estimators only qualitatively: the loss gap ("standard") and E‖d‖² ("orthogonal", which they call "more stable"). The loss gap equals (2w* − 1)·E‖d‖². So their standard minus orthogonal estimate is 2(w* − 1)·E‖d‖², and the sign of w* − 1 was within their reach but went unreported.

- [Galashov et al., Learn to Guide Your Diffusion Model](https://arxiv.org/abs/2510.00815), 2025-10, read in full. Appendix C.1 derives the identity and reports, with no numbers, levels or held-out data, that loss-fitted weights came out as zero guidance. Their ReLU head would hide w* < 1.
- [Pokle, Galashov et al., Adversarial Learning of CFG Schedules](https://arxiv.org/abs/2608.14038), 2026-08, most of the HTML read via a fetch summary. Section 2 repeats the argument for flow matching (fitting the weight to the loss "will necessarily" give no guidance) as theory only, with no measurement. Experiments use custom MMDiT models on MS-COCO 2014.
- [Improving CFG of Flow Matching via Manifold Projection](https://arxiv.org/abs/2601.21892), 2026-01, now read in full via fetch summaries. Section 3.2 defines a pointwise optimal scale with the ideal velocity, calls it intractable, and never estimates it; the appendix only sweeps fixed scales. This card's w*(t) is the level-wide version, estimable because the per-sample target is unbiased for the ideal one.
- [CFG-Zero*](https://arxiv.org/abs/2503.18886), 2025-03, read by the colleague, uses model outputs only, never real targets. A different estimator.
- Abstracts only: [MMG](https://arxiv.org/abs/2509.20609) turns the loss gap into bits per level; [Kynkäänniemi et al.](https://arxiv.org/abs/2404.07724) and [autoguidance](https://arxiv.org/abs/2406.02507) supply the "guidance repairs under-fit" view; [Tang et al.](https://arxiv.org/abs/2502.12154) and [Chen et al.](https://arxiv.org/abs/2501.15420) fold the guided target into training.

- Schedules judged by samples, not by loss on real targets (abstracts only): [C²FG](https://arxiv.org/abs/2603.08155), 2026-03, bounds the score gap per timestep and decays the weight; [Papalampidi et al.](https://arxiv.org/abs/2509.16131), 2025-09, greedy per-step search on evaluator feedback; [2606.24025](https://arxiv.org/abs/2606.24025), 2026-06, distribution matching on SD-XL and COCO.

What looks open: a held-out and an in-distribution measurement of w*(t) and R(t) on a public text-to-image model, per level, unclipped, with intervals.

## Prediction

Written 2026-11-09, before any run. One prediction, for step 2 on SD 1.5.

- R(t) stays under 1% at every level, on both the in-distribution and the COCO arm.
- Where D(t) is above its floor, w* lies between 0.97 and 1.10, and any departure is above 1 and larger on COCO than in-distribution.
- g* stays within 1 ± 0.02.

I give this about 60%, mostly on Galashov's remark.

Inconclusive: an R interval straddling 1%, a |w* − 1| between 0.03 and 0.05, or an arm difference inside the half-split spread.

Outcomes and decisions.
- R < 1% everywhere on both arms: stop after step 2, skip step 4, write a short negative.
- R ≥ 1% with w* > 1, on COCO only: the effect is distribution shift. Run the caption-wording arms in step 2b.
- R ≥ 1% with w* > 1 on the in-distribution arm too: model error. Run step 2b and consider step 3.
- w* < 1 at three adjacent levels: check for memorisation and over-fitting before any write-up.

## Design

Estimator. Encode each image to latents. At each level, draw a noise sample and run the model with the caption and with the empty prompt in one batch of two. Compute the four inner products and two losses per sample in float32 on the device, move them to the CPU, and accumulate in float64 there. w*, D, R and g* are ratios of image-level sums. Intervals come from 2,000 bootstrap resamples over images, with one image's draws kept together. Levels are integer timesteps from SD 1.5's trained range of 0 to 999. The slice plan lists them with their log-SNR values taken from the scheduler.

Denominator floor. w* is reported, and checked against any gate, only at levels where D(t) ≥ 0.01 and its interval excludes 0.01. Elsewhere the card reports R alone.

Also reported: the two-coefficient fit a·v_c + b·v_u, and the distribution of per-image w* at each level, so a pooled 1 hiding opposite signs would show. E‖d‖² per level is plotted against C²FG's claimed decay, a free comparison with a published result (added 2026-11-09). The per-level loss gap E‖v − v_u‖² − E‖v − v_c‖² is also reported. It equals (2w* − 1)·E‖d‖², so it is Kong et al.'s standard estimator, and E‖d‖² is their orthogonal one. Both are plotted against log-SNR, so they compare directly with their Fig. 5 (added 2026-11-09, second pass).

Reference code (added 2026-11-09, second pass). Before writing the SD forward pass, read Kong et al.'s released code for its noising and empty-prompt handling. Reuse it where it fits, and note any difference. An optional 1b cross-check: on 10 of their COCO100-IT pairs with SD 2.1 base (download size not checked), the per-level loss gap should match their Fig. 5 in shape. Their filter keeps captions that name an annotated object. That gives a better-aligned COCO subset at no extra download.

**Step 1, one lab slice.**
- 1a, analytic toy (CPU, under 5 minutes). A 2-D, 4-component Gaussian mixture. Its posterior means E[v | x_t, c] and E[v | x_t] have closed forms, so the reference denoiser is exact. Gate: w* = 1 within its bootstrap interval at every level above the floor, on 20,000 fresh samples, three seeds. A failure here is a code bug, because no trained model is involved. Fix it until it passes.
- Planted effect, inside the model: replace the analytic conditional output with D_u + α(D_c − D_u), α = 0.8. True w* = 1.25 and true R = 0.0625·D. Both must fall inside their bootstrap intervals.
- Observation only: a trained class-conditional MLP with 10% label dropout, converged and at 250 steps. Not a gate.
- 1b, SD 1.5 (MPS, processes under 300 s appending to one file). `stable-diffusion-v1-5/stable-diffusion-v1-5`, fp16, 512×512 (about 2 GB, not checked). 32 COCO val2017 images with one caption each, fixed seed, fetched singly. 6 levels, 2 draws: 384 UNet calls (speed not measured). Add 32 in-distribution images if reachable and time allows. This is an unconfirmed observation; its main output is interval width per level.
- 1c, subject gate (added 2026-11-10), computed from the 1b sums with no extra UNet calls. The toy and the plant test the code. This tests whether SD 1.5 on COCO gives the hypothesis room to show. Bar, all three on the COCO images: (i) at three or more adjacent levels of the six, D(t) ≥ 0.01 with its interval excluding 0.01; (ii) at those levels the loss gap E‖v − v_u‖² − E‖v − v_c‖² is positive with its interval excluding 0, so the caption measurably helps; (iii) the R half-width projected to the step-2 image count is at most 0.3 percentage points at those levels, and that count fits step 2's device budget. If (i) or (ii) fails, the caption barely moves the output on these levels and R ≥ 1% would need |w* − 1| near 1 or more, so a negative would be empty: stop, re-pick the six levels from the scheduler's mid range once, and if it still fails, park the card. If only (iii) fails, re-plan step 2's image count or park; do not run step 2 at a size that cannot tell 0.5% from 1%. Cost: CPU arithmetic on stored sums, under a minute, inside 1b's slice (30 GPU-minutes, 300 s per process, under 8 GiB; SD 1.5 fp16 memory and UNet speed on MPS not measured).

**Step 2, the full measurement** (2 to 3 slices).
- In-distribution arm: image and alt-text pairs from SD 1.5's training source, LAION-Aesthetics per its model card (not re-checked), perhaps via Re-LAION; access not checked. Without it, drop "under-use" and report only a held-out optimal weight.
- COCO arm. 12 levels, 4 draws, image count set from 1b so the w* half-width is at most 0.02 above the floor.
- Estimator check: shuffled captions push w* toward 0 by orthogonality. Pass band |w*| < 0.1 above the floor. It tests wiring, not the hypothesis.

**Step 2b, caption wording** (only if step 2 finds R ≥ 1% somewhere). Arms: the COCO caption, a length-matched paraphrase with no new information (isolates wording), and a dense caption fitted to CLIP's 77 tokens (isolates information). Record token counts. DOCCI's human dense captions may replace a VLM and serve as the post-cutoff arm; not checked. Whether Qwen3.5-4B takes images is not checked. Choose and date-check the post-cutoff source in this plan.

**Step 3, a flow model** (parked until Rob approves a memory cap above 8 GiB). Qwen-Image 2.1, already on disk, rather than SDXL, to save about 7 GB of disk (size not checked). Its training data is not public, so only held-out readings are possible.

**Step 4, constructive** (only if step 2 finds R ≥ 1% and max w* > 2). One-slice pilot: 100 COCO prompts, matched seeds, interleaved arms w = 1, w*(t) and constant 7.5, CLIP score only (download size not checked). Expand only if the schedule recovers a clear share of CFG's gain, scoring with KID on DINOv3 features.

## Checks before believing anything

- Self-comparison: split each image's draws into two halves and compute w* and R on each. The difference must sit inside its bootstrap spread. Rerunning one chunk must agree to 1e-3 in w*.
- Planted effect: the analytic α = 0.8 plant above. The post-hoc rescale v_u + (v_c − v_u)/k on stored outputs stays as a unit test of the arithmetic only.
- The analytic toy gate above, at levels above the floor.

## The control that could embarrass it

The in-distribution arm. If w* is 1 there and off 1 on COCO, "caption under-use" fails and only wording shift stands. Step 4 is the control for "guidance repairs it".

## Confounds

A bad unconditional branch: handled by the projection argument in-distribution and by the arm contrast off it. Memorised pairs: the post-cutoff arm. CFG-Zero*: shown on the same data. Sampling-time states: step 4.

## Kill criterion

Stop if R(t) < 1% at every level on both step 2 arms. Write a short, precise negative. A failing analytic toy is a bug to fix, not a reason to stop.

## Cost on this hardware

| Step | Device time | Wall time | Disk | Downloads |
| --- | --- | --- | --- | --- |
| Gate before step 2 | none | 30 min | none | none |
| 1 | under 30 min (UNet speed not measured) | about 1 hour | about 3 GB | SD 1.5 fp16, COCO captions, 32 images |
| 2 | 1.5 to 3 h, in slices | 1 to 2 days | under 1 GB | LAION subset, if reachable |
| 2b | about 1 h | 1 day | small | perhaps a VLM or DOCCI |
| 3 | several hours | 2 to 3 days | none | needs memory approval |
| 4 pilot | one slice | half a day | small | CLIP |

No cloud spend in any step.

## What a yes or a no changes

The payoff is one number for theory work on guidance: per level, how much loss a rescaled weight removes, in and out of distribution. R ≈ 0 turns Galashov's remark into a measured fact and points theory toward off-forward-process states. R > 0 on COCO only says guidance partly corrects wording shift. No claim for samplers or trainers unless step 4 recovers a large share of CFG's gain.

## Carrying to larger models

The estimator carries to any squared-error-trained denoiser. Untested: distilled models, video and audio, anything beyond SD 1.5 and perhaps Qwen-Image.

## Receipts to keep

The slice plan and gate record; the per-sample sums file (image id, arm, timestep, seed) and its code; image and caption lists with sources, dates and token counts; model hashes; all check readouts; result tables with intervals and the bootstrap seed.

## Search gate, 2026-11-09

First pass: 6 of 12 queries ran; 6 were refused when the shared search budget ran out. It also screened the citation lists of Galashov et al. and 2601.21892, and found PathGuide and [PMC-CFG](https://arxiv.org/abs/2609.24287), 2026-09. Both were read as abstracts only. PMC-CFG uses model outputs only.

Second pass: 12 queries ran, none refused. They covered the six refused topics: distribution shift and held-out prompts; the "perfect score needs no guidance" argument; the per-level loss gap on SD and COCO; least-squares weights in flow matching; caption information per level; and the orthogonality argument. They also used other communities' wording: diffusion zero-shot classifiers, regressing clean data on both outputs, under-fitting and model error, OpenReview, GitHub notebooks, and forward process against sampling trajectory.

New closest works (none scoops the card):
- PathGuide, now read in full. Details are above. It is the nearest method, but it uses the rollout law, toys and MNIST, and a weight clipped at 1.
- Kong et al., read via summaries. It is the nearest measurement: per-level conditional and unconditional MMSE on SD 2.1 and COCO, with released code. Its two estimators recover the sign of w* − 1, but the paper never reports their difference. This changed the design (see Design).
- [Diffusion Classifier](https://arxiv.org/abs/2303.16203), ICCV 2023, search snippets only. It computes per-prompt ε errors on SD 2.0 over sampled timesteps, and its repository stores them. This is reusable tooling, not a measurement of w*.
- Screened by snippet, no overlap: [2509.22728](https://arxiv.org/abs/2509.22728), [2510.01378](https://arxiv.org/abs/2510.01378) (abstract read), [2511.14075](https://arxiv.org/abs/2511.14075), [2502.07849](https://arxiv.org/abs/2502.07849), [2602.00716](https://arxiv.org/abs/2602.00716), [2502.02150](https://arxiv.org/abs/2502.02150). They cover a quality-supervised scale, selective under-fitting, sampling-error correction, high-dimensional CFG theory, sampling distortion, and learned guidance fields.

Still open, and not blocking: Kong et al.'s Info-Decomp code has not been opened. Diffusion Classifier's timestep figure is unread. Neither pass searched OpenReview directly, beyond one search through a web engine.

## Review record

Hostile review, 2026-11-09, verdict queue-after-fixes. All must-fix items are addressed: R(t) decides; an analytic toy gate with the plant inside the model; one prediction with inconclusive bands; an in-distribution arm; dense captions moved to 2b; prior art redone. All should-fix items are addressed: the shuffled control is an estimator check; step 4 is a KID pilot; step 1b is an observation; per-image w* is reported. The minor items are handled too: CPU float64 accumulation and listed integer timesteps. The reviewer's own searches never ran. Its worry that "2601.21892 defines the same w*" was checked: that scale is pointwise and never estimated.

Subject-gate check, 2026-11-10, after copies-as-evidence step 1a ran on a model that got no greedy base-task answer right (one template inaccurate, the other off-format). This card gated only the instrument (analytic toy, plant, shuffled captions) and left 1b as an observation with no bar or stop. Added step 1c: from the 1b sums, D(t) above its floor and a positive loss gap at three or more adjacent levels, and a projected R half-width of at most 0.3 points, else re-pick levels once, re-size, or park. Risk that step 2 would have been void without it: medium. D(t) for SD 1.5 on COCO is not checked; if it sits near the floor, R ≥ 1% is out of reach and a negative says nothing.
