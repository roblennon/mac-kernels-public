# Positive-only self-training: which pass rate is worth the most per attempt?

Status: parked, 2026-11-09. Search gate still open: every WebSearch query was refused again on 2026-11-09, so only known pages were read. Unpark when step 0 (CPU only) or a completed search gives the 1/e claim a model to stand on.

Idea title in the ledger: "a pass or fail is worth H(r) bits, so positive-only self-training should peak near a 37% pass rate".

Rejection fine-tuning keeps a model's checker-accepted attempts and trains on them. The card asks which pass rate r makes each attempt worth the most. Candidates: 1/e, 0.5 (the RL default) and about 0.84 (the 85% rule).

## Why this is parked

First, the old step 1 metric built its own answer: one small positive-only step on a two-outcome softmax changes log p(correct) by lr · r(1-r)², which peaks at r = 1/3. A step 1 peak "near 1/e" would have been an artefact.

Second, the KL step was not a derivation. KL(pi+ || pi) = log(1/r) is a distance per round, not a movement per accepted sample. One explicit full-fit model (each accepted sample buys a fixed KL amount, a round fully fits pi+, gain 1-r) gives gain per attempt proportional to r(1-r)/log(1/r), which rises to r = 1 with no interior peak. No explicit model found gives 1/e. Derived here, not simulated.

Third, the failure claim leaned on the label-bit ratio (3.4x), an argument the card said it had dropped.

What survives is a measurement: does positive-only training that fits each round peak at 0.5 like RL? In small steps, algebra says it matches policy gradient on accuracy, so a surprise can only come from the full-fit regime, which step 0 tests on a CPU. No device time until step 0 shows a peak away from 0.5.

## Hypothesis (reframed)

Under a fixed attempt budget, with r held fixed, the held-out accuracy gain per attempt of positive-only training that fits the accepted set each round peaks at a pass rate different from 0.5. Its 90% interval excludes 0.5. The shape contest is r(1-r) against r·log(1/r) against an 85%-rule curve against a curve rising to the top bin, decided on accuracy. A separate claim: with failures (NFT loss) beats positive-only by at least 1.3x per attempt at r = 0.9. The earlier 2x threshold was a guess, and it is now marked as one.

## Theory, derived here, not checked in the literature

Two-outcome softmax with correct-answer logit θ and p = σ(θ) = r. The positive-only loss is summed over accepted samples and divided by attempts. Its expected update is Δθ = lr · r(1-r).

- Change in log p: lr · r(1-r)². Peak at 1/3. Normalised: 0.55, 0.86, 0.99, 0.84, 0.43, 0.15, 0.06 at bins 0.1, 0.2, 0.37, 0.5, 0.7, 0.84, 0.9.
- Change in pass probability p: lr · r²(1-r)². Peak at 0.5.
- REINFORCE with the exact mean baseline has the same expected update. Its per-sample variance is r(1-r)(1-2r)². Positive-only has r(1-r)³. The ratio is (1-2r)²/(1-r)²: 0.79 at r = 0.1, 0 at 0.5, 64 at 0.9. So in the two-outcome toy, adding failures raises the noise at high r. Multi-outcome is not derived.

## What is already known

Read by direct fetch on 2026-11-09; keyword search was still refused. Nothing read gives an optimal acceptance rate for positive-only training, and nothing read predicts 1/e.

- [Huang et al., the sharpening mechanism](https://arxiv.org/abs/2412.01951), Dec 2024, theory sections read in full. SFT-Sharpening's sample cost scales with coverage C_cov = E[1/π(y*|x)], roughly 1/r, and is minimax optimal when coverage is good. A cost that falls as r rises gives no interior peak, which matches the card's own full-fit model. Their reward is the model's own likelihood, not an external checker.
- RL side: [Bae et al.](https://arxiv.org/abs/2504.03380) bound improvement below by the variance of task success, which favours intermediate r. [Foster et al.](https://arxiv.org/abs/2502.12272) pick problems by success variance for PPO and VinePPO. [Gao et al.](https://arxiv.org/abs/2510.01135) sweep target r in {0, 0.25, 0.5, 0.75, 1} for on-policy GRPO on Qwen3-4B-Base: 0.5 is best (46.1 average), with 0.25 and 0.75 close (about 45.0 and 44.8). None has an SFT arm.
- Positive-only in practice: [RAFT++](https://arxiv.org/abs/2504.11343) nearly matches GRPO on Qwen2.5-Math-7B (56.1 against 56.3) but loses entropy faster; no breakdown by r. [ReST-EM](https://arxiv.org/abs/2312.06585) caps correct solutions per problem so easy problems do not dominate; its gains are largest on problems the base model solves 25 to 75% of the time, a split of the evaluation, not of training. [Expert iteration](https://arxiv.org/abs/2403.04642) roughly matches PPO in samples. [DART-Math](https://arxiv.org/abs/2407.13690) gives hard queries more correct samples and beats vanilla rejection tuning by 3 to 4.5 points, but it distils from a stronger model and does not count per attempt.
- [NFT](https://arxiv.org/abs/2505.18116) equals GRPO when strictly on-policy and weights prompts by 1-r or sqrt((1-r)/r); no breakdown by r. [The 85% rule](https://www.nature.com/articles/s41467-019-12552-4) is about SGD on a binary classifier.
- Abstracts only: [negative reinforcement](https://arxiv.org/abs/2506.01347), [expert failures](https://arxiv.org/abs/2504.13145), [AdaRFT](https://arxiv.org/abs/2504.05520) (an RL curriculum aimed at "challenging but solvable"), and [self-improving transformers](https://arxiv.org/abs/2502.01612) (correctness-filtered self-training on arithmetic, the nearest neighbour to step 1's task).

## Prediction (written before any run)

- Step 0, tabular simulation: in the small-step setting, the accuracy curve peaks at 0.5 and the log-probability curve at 1/3, matching the algebra. In the full-fit setting I have no derived peak, so the prediction is "not at 1/e". The r(1-r)/log(1/r) model above predicts a peak at the top bin.
- Step 3 known comparator: Gao et al.'s RL sweep puts 0.5 first and 0.25 a little ahead of 0.75. Positive-only is compared with that ordering.
- Step 1, one fixed small rate: the primary metric (expected change in held-out pass probability) peaks at 0.5 ± 0.1. The secondary log-probability readout peaks at 0.33 ± 0.1. The REINFORCE arm's mean gain equals positive-only within noise at every bin; its spread follows the variance ratio above, at least in sign.
- Effect that matters: a result decides only if the 90% bootstrap interval for the peak excludes at least one competitor.

## Design

**Step 0 (CPU seconds; fits every lab limit; the real first test).** An exact tabular simulation of step 3's loop: many prompts, each a softmax over 50 outcomes with a known passing set, with a shared-parameter variant so transfer exists. Arms: positive-only, random-accept, REINFORCE, NFT. Fixed attempt budgets. Per round, an exact fit to pi+ or k SGD steps, k from 1 to 100. Metric: held-out accuracy gain per attempt, swept over r. Pre-registered gate: if no setting peaks below 0.45 on accuracy per attempt, the 1/e claim is dropped and the card stays parked as "restates Bae et al. and PCL for positive-only training". Step 0 also runs the analysis pipeline at step 1's cell sizes and pre-registers the minimum seed count that gives a decisive peak interval.

**Task and model (steps 1 to 3).** Integer multiplication with an exact checker, answers in reversed digit order, from a seeded generator; nothing is downloaded. A character-level decoder of about 5M parameters (6 layers, width 256, about 16 tokens), trained from scratch in PyTorch on MPS. Small transformers struggle with multi-digit multiplication without a scratchpad, so the first 300-second chunk measures the reachable difficulty and caps it there.

**Step 1 (one lab slice, only after step 0 passes its gate).** Bin within one difficulty level (say 3x3), choosing operands to spread r; uncovered bins are reported missing. Bins: 0.1, 0.2, 0.37, 0.5, 0.7, 0.9. The 0.84 bin overlaps 0.9 at this sample size and is dropped; the 85% curve stays separable through 0.7 and 0.9. Each prompt's r is estimated from 64 fresh samples, drawn apart from the training attempts. Each (bin, seed) draws 64 prompts x 8 attempts = 512, shared by all arms. Arms: positive-only, random-accept, and REINFORCE with a per-prompt mean baseline. Each arm takes one step at one fixed small learning rate, which must pass the halving check in every bin. Primary metric: expected change in held-out pass probability on the same difficulty. It is computed as the model's exact-match probability of the correct answer, not by sampling. Log-probability is a secondary readout, next to its own derived peak of 1/3. A side study sweeps four rates, picks the best on a validation split, and reports held-out gain per cell; it does not decide. Order is randomised within each seed. One probe cell is timed first; an oversized grid splits across slices, never cutting seeds.

**Step 2 (one slice).** Add the NFT arm. Add a temperature control at fixed difficulty, moving r with temperature from 0.7 to 1.3. Samples at T ≠ 1 are off-policy and sharpen or flatten the model, so a random-accept arm at each temperature is subtracted as its null. Importance weights back to T = 1 are a second readout.

**Step 3 (many slices; needs Rob's approval for device time).** The full loop. A controller holds r at each bin; a fixed attempt budget runs in 300-second resumable chunks; 6 bins x 3 arms x the seeds from step 0. Primary metric: held-out exact-match accuracy gain per attempt. Cost unmeasured; the author guessed an hour per run.

**Step 4 (needs Rob's login and approval).** Qwen2.5-0.5B or Qwen2.5-1.5B with LoRA on GSM8K, at bins 0.2, 0.37, 0.5 and 0.9. Fit in 8 GiB and model availability are not checked; a free tier is the fallback.

## Checks before believing anything

Learning rate 0 must change every readout by exactly zero. Two independent attempt draws in one cell give the noise floor; a bin difference inside it is inconclusive. Random-accept must read zero within noise.

Planted effects: step 0 must recover the analytic one-step peaks (0.5 and 1/3) within 0.05. In step 1, per-prompt losses are reweighted by a narrow known function of r centred at 0.65; binning, selection and the bootstrap must recover 0.65 within 0.05, and each candidate shape when it is the planted weight. Halving the fixed rate must halve the primary readout within 15%.

## The control that could embarrass it

A shape contest decided on accuracy: r(1-r), the RL default, against r·log(1/r), the 85% curve, and a curve rising to the top bin (the coverage cost in Huang et al. and the card's full-fit model). If r(1-r) wins by bootstrap, positive-only training just follows the RL result; if the rising curve wins, it follows sharpening theory. Random-accept is a null that must read zero, not a competitor.

## Confounds a hostile reader would raise

- Difficulty and r move together: handled by within-difficulty bins and the temperature control.
- Learning rate favours an arm: one fixed rate decides; best rate is chosen on a validation split.
- Noisy r selection: 64 separate samples per prompt; realised r logged.
- The toy says nothing about LLMs: true until step 4, and said in the first line of any write-up.
- Normalisation per attempt or per accepted sample: a constant rescaling; reported both ways in step 3.

## Kill criterion

- Step 0: no setting peaks below 0.45 on accuracy per attempt, so the 1/e claim is dropped.
- Step 3: the positive-only peak's 90% interval contains 0.5, or r(1-r) beats r·log(1/r) by bootstrap, so the claim is dropped.
- The spread of bin means is under twice the noise floor, so the curve claim is dropped.
- With failures gains under 1.3x at r = 0.9, so the failure claim is dropped.

## Cost on this hardware

Step 0: CPU minutes. Step 1: one chunk for reachable difficulty, up to two for pretraining, one timed probe cell, then the grid, split if it overruns 30 minutes; under 1 GB disk. Steps 3 and 4: not measured.

## What a yes or a no changes, and what it leaves untested

A step 0 peak away from 0.5 earns device time. A peak at 0.5 says positive-only training follows the RL default: useful, expected. The experiment uses none of Rob's assets (low-bit kernels, packed trainer, steering tools), so a better-resourced group has every advantage; that is part of why it is parked. The peak depends on metric, update regime and difficulty transfer, none of which carries from a 5M-parameter toy.

## Receipts to keep

Dated step plans; commit, PyTorch version, checkpoint hash; step 0 outputs and all check readings; raw JSONL per run (seed, bin, target and realised r, arm, rate, attempts, readouts); bootstrap script and intervals; chunk timings.

## Search gate, 2026-11-09

Queries run: 0. Refused: 4 of 4, for the shared per-turn WebSearch budget (optimal pass rate in rejection fine-tuning; sharpening and coverage sample complexity; DART-Math difficulty ablations; ReST-EM and STaR theory). The ten further queries were not tried after the refusal, and keyword search through arXiv, Semantic Scholar, OpenReview, Hugging Face or GitHub was not used to get around it. A Semantic Scholar citation listing for Huang et al. returned 429.

Read instead, by known URL (about 14 fetches): Huang et al. theory and ReST-EM method and difficulty sections from the PDFs; Gao et al.'s sweep section; RAFT++, DART-Math and NFT through a model summary of the full HTML; abstracts for Foster, Bae, AdaRFT, [RFT scaling](https://arxiv.org/abs/2308.01825), expert iteration and self-improving transformers. An arXiv cs.LG listing page (9 Nov 2026, first 50 titles) had nothing on topic.

Closest now: Huang et al. (Dec 2024; theory, cost falls with r), Gao et al. (Oct 2025; the empirical sweep, RL only), ReST-EM (Dec 2023; positive-only, gains by difficulty). No scoop in what was read.

Design change from this pass: the shape contest gains a fourth curve rising to the top bin, and step 3 is compared with Gao et al.'s ordering.

Still open: novelty is unchecked because no keyword search ran. Owed next pass: the four queries above plus ten in other wording, led by the cross-entropy method, whose elite fraction is the same knob as r (keep the accepted fraction, refit fully); its guidance is unread. Then success-rate curricula, expert iteration by difficulty, STaR theory, Gao et al. Appendix D, and self-improving transformers on multiplication.

## Review record

- Fatal, step 1 metric builds in a 1/3 peak: agreed by algebra. Primary metric is held-out pass probability in steps 1 and 3; log-probability is secondary.
- Must-fix, 1/e is not a derivation: agreed; the main reason for parking. Reframed as a measurement gated on step 0.
- Must-fix: step 0 added with a gate at 0.45; one fixed rate decides, best rate reported apart; 3.4x and 2x removed, 1.3x kept; step 1 bins within one difficulty, with a per-temperature null.
- Should-fix: step 0 sizes seeds; 64 samples per r; 0.84 bin dropped; planted peak at 0.65 and each shape; shape contest on accuracy with random-accept as a null; difficulty capped and one probe cell timed first.
- Should-fix, scooping undone: still undone; see the search gate.
- Minor: normalisation kept for step 3 only; step 4 named (availability not checked); toy stays toy.
