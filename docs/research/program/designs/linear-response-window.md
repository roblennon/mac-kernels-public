# The linear-response window: over fifteen decades of perturbation size, when does training answer in proportion

Status: queued, 2026-11-09. Search gate cleared 2026-11-09: twelve searches ran, none found the prediction-space curve below 1e-5 against injection time (see Search gate). Revised after a hostile review (record at the end). Designed, not run.

If you nudge a model's weights by a tiny amount at some step and finish training, do the final predictions change in proportion, or land where another seed would? Attribution methods such as MAGIC differentiate through training, which works only where training responds in proportion. Sensitivity is known to fade with the time of the perturbation in weight space, and a one-ulp change at init gives near seed-level churn in long runs. Open: prediction response from 1e-15 to 1e-5, a mechanism that predicts where the window opens, and whether attribution accuracy tracks it. Bit-exact fp64 CPU replay reaches those sizes.

## Hypothesis

For each injection step t there is a threshold eps*(t). Below it, final predictions change in proportion to the perturbation; above it, by about a seed-to-seed gap. The threshold rises with t faster than the shrinking number of remaining steps explains, so there is a distinct early phase. "Fewer steps remain to amplify" is the rival, and the design measures it.

## Theory

A small difference at step t is multiplied at each later step by the update's Jacobian, so it grows to about eps times G(t), the gain from t to the end. The response stays proportional until the amplified difference reaches the scale where the landscape bends. If that scale is the seed gap C, eps*(t) is about C / G(t); if the landscape bends earlier, eps* sits lower. G(t) is the finite-time Lyapunov growth that Herrmann et al. measure in weight space. A forward-mode Jacobian-vector product through training measures it independently of the replays, so the story becomes a number that can miss.

Assumptions, said out loud:
- Early training has a large gain. Saturating at 1e-12 needs about 23 e-folds. A small model may not have that. This is the weakest link.
- Weights can land in permuted basins while predictions barely move.
- Adam adds its own nonlinearity, so its epsilon is a named variable.
- fp64 CPU training is bit-exact with one thread and a fixed data order. The self-comparison tests this.

## What is already known

- [Kwok et al., The Butterfly Effect](https://arxiv.org/abs/2506.13234), ICML 2025, read in full. Weight space only, down to 1e-5; the threshold depends on spawn time; no prediction measure, no attribution.
- [Summers and Dinneen](https://arxiv.org/abs/2103.04514), ICML 2021, key sections read. One ulp on one weight at init gives nearly full seed variability (CIFAR-10, Penn Treebank); linear models barely move; noise switched on after 50 of 500 epochs still does.
- [Jordan, Calibrated Chaos](https://arxiv.org/abs/2304.01910), v4 2024-06, main text and appendix D read. One weight times 1.001 at init gives less churn than a new seed in short runs (7.0-7.5% against 8.5%), the same by 128 epochs.
- [Ly and Gong, Fractal basin geometry](https://arxiv.org/abs/2510.05606), 2025-10, v2 2026-09, most read. Init-only, 10 decades down to the last bit: churn is flat, no proportional regime.
- [Herrmann, Granz and Landgraf](https://proceedings.neurips.cc/paper_files/paper/2022/hash/222a2a46018a1e7b55ba48ba11932d04-Abstract-Conference.html), NeurIPS 2022, abstract and results read. Finite-time Lyapunov exponents positive early, falling to 0; size 0.1 at init, weight space, small MLPs; smaller sizes drowned in numerical error.
- [Deng et al., How Faithful Is Trajectory-Based Data Attribution?](https://arxiv.org/abs/2605.18814), 2026-05, most read. Attribution error grows with steps remaining; a closed-form proxy tracks it (Spearman 0.83 to 0.89). Binary removal, no size sweep, no chaos framing.
- [MAGIC](https://arxiv.org/abs/2504.16430), read in full: a few hundred warm-up iterations, never where smoothness begins. [Metagradient Descent](https://arxiv.org/abs/2503.13751), mostly read: metasmoothness at one size.
- [Schioppa et al.](https://arxiv.org/abs/2305.16971), NeurIPS 2023, theory and experiments read. Parameter divergence after upweighting is bounded by, and observed to follow, an exponential in the integrated learning rate, steeper at first (BERT, ResNet); influence accuracy fades over retraining time. Weight space; the sizes tried are in figures only.
- [Zhou et al., two-phase learning dynamics](https://arxiv.org/abs/2505.13900), 2025-05, read via summary of the full text. One size (1e-7) injected at varied t0; disagreement and loss barrier appear only for injections before a transition (about 2,500 iterations, VGG-16). No size sweep, no mechanism.
- [Liu and Li, minibatch perturbations in AdamW](https://arxiv.org/abs/2608.19762), 2026-08, main text read. Linearised joint state (weights and both Adam moments) predicts loss response over 8 to 32 steps; tangent error grows as size squared over about one decade (1/32 to 1/4). No injection-time sweep, no chaos framing.
- Abstracts only: [Ni and Talnikar](https://arxiv.org/abs/1905.04561) (linear range of one step), [Frankle et al.](https://arxiv.org/abs/1912.05671), [Fort et al.](https://arxiv.org/abs/2010.15110), [Mlodozeniec et al.](https://arxiv.org/abs/2506.12965), [Bae et al.](https://arxiv.org/abs/2209.05364), [Speck and Shepard](https://arxiv.org/abs/2608.19168), [Chen et al.](https://arxiv.org/abs/2609.15044).

Net: saturation at init in long runs, prediction churn at one tiny size by injection time, and exponential weight divergence in integrated learning rate are known. Open: the prediction window across sizes below 1e-5 against injection time, E(t), the mechanism, and the attribution link across sizes.

## Prediction (written before any run)

Two recipes, run as interleaved arms: no warmup, and a 100-step linear warmup. Sizes are relative to the weight norm.
- Step 0, no warmup: the response is flat within a factor of 2 from 1e-13 to 1e-2, within a factor of 3 of C. This is the less likely half: Summers and Dinneen and Ly and Gong see it in long runs, Jordan sees less churn in short ones, and 2,000 steps is short.
- Step 50, both recipes: log-log slope 1.0 plus or minus 0.1 from the rounding floor to at least 1e-5.
- Decisive number: the early-phase excess E(50), the shift log10 eps*(50) minus log10 eps*(0), less what remaining steps predict (see Design). At least 2 decades with no warmup; at least 1 decade smaller with warmup.
- Mechanism: eps*(t) within 1 decade of C / G(t) at t = 0 and 50, both recipes.
- Attribution (step 2): Spearman above 0.9 in windows after 10% of training, below 0.3 in the first 2%.

## Design

Model: a 2-layer decoder-only transformer in the lab's own code. d_model 192, 6 heads, MLP 768, context 64, byte vocabulary, tied embeddings, about 0.95M parameters. If a 2,000-step run exceeds 250 seconds, fall back to d_model 128, MLP 512 (437,760 parameters, about 49 s per 1,000 steps by the reviewer's timing) and note it with a date. Optimizer: the lab's own functional AdamW, so replays and Jacobian products share one code path. Learning rate 1e-3, betas (0.9, 0.95), eps 1e-8, weight decay 0.1, cosine decay to 10% over 2,000 steps, batch 16 windows of 64 bytes. fp64 on CPU, one thread, deterministic algorithms.

Data: `TinyStoriesV2-GPT4-valid.txt` from `roneneldan/TinyStories`, 22.5 MB. Train on the first 90% of bytes; 2,048 held-out positions from the rest. Batch order is drawn once and saved. Every 50 steps a checkpoint saves weights, both Adam moments, the step count and the data cursor; replays start from it.

Perturbation: at step t, add eps times ||w|| times a seeded unit Gaussian direction to the weights only.

Output measure: R(eps, t), the RMS difference in final log-probabilities over held-out positions and all 256 tokens, which ignores the shift a softmax discards. Secondary: logit RMS, mean KL, top-1 disagreement, raw weight distance, and weight distance after Hungarian alignment of MLP units. The seed ceiling C is R between runs with different init seeds.

Threshold: eps*(t), per direction, is the largest eps for which R/eps stays within a factor of 2 of its value at the bottom of the range at every smaller size. Below the rounding floor is censored. The interval is the range across directions widened by half the grid step each side. A shift whose interval covers a decision boundary is inconclusive.

Remaining-steps control: measure eps*(0) on runs of 500, 1,000 and 2,000 steps with the same schedule shape and fit log10 eps*(0) against the run's integrated learning rate, not its step count. Schioppa et al. find divergence exponential in integrated learning rate, and with warmup the first 50 steps carry little of it, so a step count would overstate the rival in one arm. The slope b gives the predicted shift as b times the integrated learning rate over [0, t]. E(t) is the measured shift minus that.

Second threshold estimator: below eps*, the nonlinear remainder R minus eps times the forward-mode response should scale as eps squared (Liu and Li fit an exponent near 2 over one decade). The size where that fit breaks is reported beside the factor-of-2 rule; a gap over one grid step is flagged.

Step 0, no compute, gate: passed 2026-11-09 (see Search gate).

Step 1a, one slice. Timing, the checks below, then both recipes at injection steps 0 and 50, 2 directions, sizes 1e-15, 1e-13, ..., 1e-1: 64 replays, 4 references, 4 zero-size replays, per recipe one init-seed and one data-order run, and two one-ulp single-weight changes at step 0 per recipe (the Summers and Dinneen point). Alone it says whether step 0 is saturated or linear below 1e-5, gives a coarse shift, and shows what warmup does.

**Gate 1a: does this model learn, and is the window inside the grid (added 2026-11-10).** The checks below test replay and the metric, not whether this run has a seed gap to saturate into or enough gain to reach it. The design names the gain the weakest link. Read before the other replays, no-warmup recipe first:
- Learns: held-out loss of the reference run at step 2,000 at least 0.5 nat per byte below the unigram byte-frequency loss of the training bytes. If not, the model has nothing to be chaotic about: stop, raise learning rate or steps once, else park.
- Room: C at least 1e4 times R at size 1e-13 injected at step 50. If not, there are too few decades between floor and seed gap to place a threshold: park.
- Reachable: one direction at step 50, size 1e-1, gives R at least C/3. If not, eps*(50) is censored above and E(50) is undefined: add sizes 0.3 and 1 at step 50; if still below C/3, park the early-phase and mechanism tests at this size and report only the step-0 curve.
Cost: the reference, init-seed, 1e-13 and 1e-1 runs are already in 1a; the gate reorders them, plus two replays if the fallback fires. About 5 processes, inside the slice. The bars are design choices; values not checked.

Step 1b, one or two slices. A third direction, the even decades, half-decade sizes near each threshold, two more seeds per recipe for C, the run-length arm, and forward-mode G(0) and G(50) per recipe and direction, with the tangent carried through both Adam moments (Liu and Li show the moments shape the delayed response), in 500-step segments with primal and tangent saved between processes. E(50) and the mechanism test are read only after 1b.

Step 2, two or three slices, attribution. Six windows over training. In each, remove 30 distinct batches, replay, and record the change in held-out loss. For 5 batches per window, also replay at weights 0.5, 1.5 and 2 to test proportionality. The first-order estimate is a real metagradient: one reverse-mode pass through the replay, recomputing between checkpoints and saving the adjoint at segment ends. On 5 batches it must match a forward-mode product to 1e-8 relative. Each window gets a Spearman correlation over its 30 removals with a bootstrap interval. Baseline predictor: Deng et al.'s closed-form error proxy, computed per window; the window must explain more than it does to count. This tests the assumption MAGIC rests on, not MAGIC's code.

Step 3, several slices, variants one at a time: Adam eps 1e-6, learning rate 3e-4, run lengths 1,000 and 4,000. Run length says whether the window opens at a fixed step count (MAGIC's reading) or a fixed fraction. New seeds turn the observation into a result.

Step 4, optional: step-1a curves in fp32 on Metal. A larger model needs cloud and Rob's approval; not designed here.

## Checks before believing anything

- Self-comparison: two reference runs in separate processes give identical SHA-256 weight hashes; an eps-0 replay from each checkpoint reproduces that hash and R reads exactly 0.0. Anything else stops the slice.
- Planted, metric only: perturb the final weights with no further training; R matches eps times the forward-mode product to 1% from 1e-12 to 1e-4.
- Planted linear response through training: a bigram softmax table trained on the same bytes by the same AdamW code, Adam eps 1e-3. Its replay response matches eps times its forward-mode product through training to 1% wherever that product exceeds 1e-13, and its eps* is 1e-3 or higher.
- Planted saturation: the data-order run and the eps-1 replay at step 0 give R within the spread of C.
- Estimator: on 200 synthetic curves with known thresholds and the observed direction spread, the rule recovers the threshold within half a grid step at least 95% of the time.
- Rounding floor: sizes 1e-15 and 1e-14 show where the response leaves the eps line. Only sizes above it count.

## The control that could embarrass it

Remaining steps: E(50) near zero means fewer steps left to amplify explains the window. Weights versus predictions: proportional predictions at step 0 with saturated aligned weight distance means "chaotic in weights, linear in predictions". A flat step-0 curve far below C is not chaos.

## Confounds a hostile reader would raise

Rounding noise: measured floor. No-warmup built to be chaotic: warmup arm. Random directions are not data: step 2 uses real removals. One direction drives it: all are reported.

## Kill criterion

- Early phase dead: the E(50) interval lies below 1 decade in both recipes. Below 1 decade with warmup only is reported as "warmup removes the early phase".
- Saturated chaos at init dead: no-warmup step-0 slope within 0.2 of 1 over 6 or more decades above the rounding floor.
- Mechanism wrong: eps* misses C / G(t) by over 1 decade at either step in either recipe.
- Attribution link dead: Spearman above 0.9 in every window, the first included. Then step 3 is not run.

## Cost on this hardware

Step 1a: about 80 CPU processes, roughly 200 s each (scaled, unmeasured), 30 minutes at 8 in parallel, under 1 GB. Step 1b: about 100 replays plus 8 forward-mode runs. Step 2: about 240 replays plus one reverse-mode run. Step 3: about 300 replays. Step 4: under 30 GPU-minutes. No cloud spend.

## What a yes or a no changes

A yes gives attribution a domain of validity, a mechanism that locates it, and a test of MAGIC's rule. A no on the early phase says the threshold is just remaining steps; a no on the mechanism says gain alone cannot locate the window.

## Why it should carry to larger models, and what it leaves untested

Sensitivity fading with time is seen on BERT and OLMo fine-tuning, and G(t) can be estimated at scale. Untested: models above 1M parameters, real tokenizers, long training, data-parallel noise.

## Receipts to keep

Hashes, zero-size values, planted tables; one CSV row per replay; G(t); per-window Spearman with intervals; batch order, commit, torch version, data SHA-256, plots from a committed script. The write-up says "unconfirmed observation" until step 3 reproduces it on new seeds.

## Search gate, 2026-11-09

First pass: 4 searches refused (shared budget); papers read by direct fetch, plus citation lists for MAGIC, the Butterfly paper and Summers and Dinneen.

Second pass: 12 searches run, 0 refused, covering growth rates, spawn time, attribution fragility, metasmoothness and 2026 work, plus fp64 sweeps, attribution noise floors, tangent-linear validity (the weather-forecasting term), first-order validity sweeps and Herrmann et al. follow-ups.

New closest: Liu and Li (2026-08, main text), Schioppa et al. (2023-05, theory and experiments), Zhou et al. (2025-05, full-text summary); the rest of Deng et al.: nonlinearity grows with learning rate, no size sweep, no code link seen. Abstracts only: [FTLEs predict final loss](https://arxiv.org/abs/2405.00743), [last-bit sensitivity in CNNs](https://arxiv.org/abs/2206.02001), [Lyapunov exponent against learning rate](https://arxiv.org/abs/2506.08523). None sweeps size below 1e-5 against injection time in prediction space or tests a gain-based threshold. Not scooped.

Design changes: the remaining-steps control uses integrated learning rate; an eps-squared remainder estimator; the tangent carries the Adam moments. No released dataset or published curve to compare against beyond the one-ulp point.

Still open: GitHub code search; a full citation list for Herrmann et al.; Liu and Li's appendix on temporal ablations (not read).

## Review record

Fixed: E(50) and a G(t) mechanism decide; run-length arm; planted effects through replay; six attribution windows with a real metagradient; about 1M parameters; log-probability RMS; warmup arm; per-direction intervals; self-comparison; measured rounding floor. Partly agreed: attribution moves to step 2, behind step 1. Prior art: direct reads, then twelve searches on 2026-11-09; gate cleared. Reviewer's timing, synthetic data: 120,576 parameters at about 30 s per 1,000 steps; 437,760 at about 49 s.
- 2026-11-10, base-task check: no step confirmed the model learns, that C stands well above the rounding floor, or that a size inside the grid saturates at step 50; the planted checks test replay and the metric. Added gate 1a (loss 0.5 nat per byte below unigram; C at least 1e4 times R at 1e-13; R at 1e-1 at step 50 at least C/3, else extend to size 1, else park the E(50) and mechanism tests). Risk of a void test without it: medium.
