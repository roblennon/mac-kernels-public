# Is the layer-duplication effect just a bigger step?

Status: queued, 2026-11-09 (revised after hostile review). Search gate passed in a second pass the same day: 12 searches ran, none refused, and none found duplication compared with one-pass amplification on a frozen unshared LLM.

Round 2 shortlist S08. Merges "bigger step" and "The missing control in layer repetition".

Running a block of middle layers twice in a frozen LLM doubles the step along its update and lets it read its own output. Nobody has measured which carries the duplication effect, gain or loss. A one-pass arm that multiplies the update by a constant costs no extra compute and separates them.

## Hypothesis

H1, the original claim at full strength. Amplification picks its own best span and scale a on dev. Duplication picks its own best span on dev. On held-out items, the best amplification reaches 50% or more of the best duplication change. The author expects this to be false, P = 0.15. A true H1 is the outcome that would surprise the author.

H2, the decomposition on duplication's span. Duplication equals amplification at a = 2 plus a unit-step re-read (A4), exactly at the span output. For gains, A4 carries at least half the change. For losses, A2(2) carries at least half (overshoot). The parts need not add on the metric, so the interaction is reported and may dominate.

## Theory

Let F(x) = Span(x) − x. Duplication gives x + F(x) + F(x + F(x)) = [x + 2F(x)] + [F(x + F(x)) − F(x)]. Relative to A0, the residual change of A1 is that of A2(2) plus that of A4. This holds for the residual at the span output only. Later layers are nonlinear, so metric effects do not add. The interaction is I = Δ_A1 − Δ_A2(2) − Δ_A4.

TFLT blames naive-loop collapse on over-integration. mergekit #198 practitioners reported, informally, that scaling the update above 1 broke output. Jastrzebski et al. report "representation explosion" from naive layer sharing. Chen et al. (EMNLP 2023 Findings) found a step of β above 1 holds up only for modest cuts in a trained shared model. All point to amplification hurting, so H1 is the surprise and H2-for-losses is close to expected.

Assumptions: the span update F is the right unit (per-layer amplification is step 2). Option scoring without generation tracks benchmark changes; step 1 speaks only to scored multiple choice on one base checkpoint. TFLT says naive looping degrades performance almost universally, so a loss at 1.7B is likelier than a gain; losses are analysed, not discarded.

## What is already known

21 keyword searches over two passes and citation-list reads found no one-pass amplification run against duplication. The nearest precedent is amplification standing in for repeats in trained weight-shared models (Chen et al.). Novelty holds for frozen unshared LLMs.

- [Boosting Inference Efficiency: parameter-shared PLMs](https://arxiv.org/abs/2310.12818) (Chen et al.), EMNLP 2023 Findings, main text and Table 1 read. At inference, multiplies a shared layer's update by β in 1.0 to 3.0 so fewer iterations cover the same Euler "time". On their own shared BERT-large and GPT-2-large it holds to 16 of 24 iterations and collapses at 12 (GPT-2 perplexity 33.0 to 105.1); on unshared-equivalent GPT-2 it fails. Aimed at speed; never compares repeats with a bigger step on a frozen unshared LLM.

- [Training-Free Looped Transformers](https://arxiv.org/abs/2605.23872) (TFLT), 2026-05, read in full across two passes. Loops a frozen mid-stack block with damping α in (0, 1]. Naive K = 4 looping on Qwen3-0.6B raised lambada perplexity from about 13 to about 1054. No α above 1, no one-pass amplification.
- [From Growing to Looping](https://arxiv.org/abs/2602.16490), 2026-02, nearly full: self-trained 360M and 1.7B models, no scaling control.
- [mergekit #198](https://github.com/arcee-ai/mergekit/issues/198), 2024, 100k of 153k characters: settled on down_proj × sqrt(1/2) plus Q/K × 1/sqrt(2); never benchmarked against a scaled once-run model.
- [CircuitProbe](https://arxiv.org/abs/2604.00716), 2026-04, full: gains below 3B (Qwen2.5-3B +4.8%), losses at 7B. No scaling control, no held-out split.
- [RYS-II](https://dnhkng.github.io/posts/rys-ii/), 2026-03, full: block sweep on Qwen3.5-27B, no scaling arm.
- [Jastrzebski et al.](https://arxiv.org/abs/1710.04773), ICLR 2018, full text: unrolled ResNet blocks explode activations; a block-output scale tames them. Scaling serves looping, never replaces it.
- [Lad et al., Stages of Inference](https://arxiv.org/abs/2406.19384), 2024-06 (v3 2025-06), full text: duplicating late five-layer blocks lowers output entropy ("sharpening") and lifts some scores. No scaling arm; amplification could reproduce sharpening.
- [Layer Duplication in LLMs](https://aclanthology.org/2025.findings-emnlp.967/), full: no scaling control. [Layers as Painters](https://arxiv.org/abs/2407.09298), 2024-07, main text: middle-layer repeat is "catastrophic"; only rescaling is averaging. [CoLa](https://arxiv.org/abs/2507.07996), 2025-07, main text: per-input skip and repeat search; no scaling. [Recirculation](https://arxiv.org/abs/2608.17981), 2026-08, most of the text: scaled deep-to-shallow feedback; looping baseline gave no robust gain.
- [Inner Loop Inference](https://arxiv.org/abs/2602.14759), 2026-02, main text: loops with convex averaging; gains within error; a matched-norm noise control usually hurts (precedent for R). No amplification.
- [SelfMorph](https://arxiv.org/abs/2610.38382), 2026-10, most of the text: per-input REPEAT or BRANCH on 4B to 27B; REPEAT alone mixed; no gain arm.
- [There Is No Free Loop](https://arxiv.org/abs/2610.42030), 2026-10, 100k of 116k characters: loops beat equal-parameter and equal-FLOP twins only at 0.8B (48.3% vs 44.2%); its training-free loop uses damped 1/K steps. [Agreement rather than normalisation](https://arxiv.org/abs/2610.31943), 2026-10, main text: "overshoot" means loop count over trained budget, not step size. Neither has an arm with a > 1.
- [RYS on small language models](https://john-broadway.github.io/rys/), 2026-10-30, full page: duplication sweep, 21 models from 135M to 32B, one 49-question probe for search and report. A re-test erased the Llama-3.2-1B gain (GSM8K 34% base, 23% with the config). No scaling control. Supports the held-out split.
- [miqu-1-120b discussion 4](https://huggingface.co/wolfram/miqu-1-120b/discussions/4), 2024-04, full thread: the origin of the mergekit recipe, judged by eye; nobody tries scale above 1 in one pass. Model cards zero or halve o_proj and down_proj on copies, also without a one-pass control.
- Adjacent, abstracts or summaries only: [Loop the Loopies](https://arxiv.org/abs/2607.16051) (2026-07, trained loops), [Prune&Comp](https://arxiv.org/abs/2507.18212) (AAAI 2026, magnitude compensation after pruning), [Activation Scaling](https://arxiv.org/abs/2410.04962) (EMNLP 2024, learned steering scalars).

If a later search finds amplification run against duplication, rewrite this card as a replication.

## Prediction

Written 2026-11-09, before any run.

- Step 1b, Qwen3-1.7B-Base: duplication's best span gives a held-out change with CI excluding zero, P ≈ 0.6 per task, more often a loss than a gain.
- H1 true, P = 0.15, tested only if the duplication change is a gain.
- H2 on losses: A2(2) carries at least half, P = 0.6.
- H2 on gains, if a confirmed gain exists: A4 carries at least half, P = 0.5.
- The random-direction control R hurts less than A2(2), P = 0.6.
- Interaction dominates on at least one task (|I| above both |Δ_A2(2)| and |Δ_A4|, CI excluding zero), P = 0.3.

Size that matters: 0.03 nats held-out mean change in the deciding metric, or 1 accuracy point.

## Design

Arms on one contiguous span i..j:
- A0 baseline.
- A1 duplicate: x1 = x + F(x), out = x1 + F(x1).
- A2 amplify: out = x + a·F(x), a in {1.1, 1.25, 1.5, 2}.
- A4 unit-step re-read: out = x + F(x + F(x)).
- R random direction: out = x + F(x) + r, fixed-seed r per token with ‖r‖ = ‖F(x)‖.
- Step 2 only: A3 (TFLT's damped scheme, matched exactly), A5 (mergekit recipe), norm-matched A2.

One pass per span serves all arms. Run layers before i once, run the span on x, run it again on x + F(x) (giving both A1 and A4), then run the remaining layers once per arm output.

Scoring. ARC is letter-scored: log-softmax of the A/B/C/D letter log-probs. Arithmetic: each candidate's summed token log-prob, log-softmaxed over four candidates. Deciding metric: mean paired change from A0 in the correct option's normalised log-prob. Accuracy with McNemar is secondary. Next-token entropy at the answer position is logged per arm (added 2026-11-09 after Lad et al.), so "duplication sharpens and amplification sharpens the same way" can be checked directly. Batches are right-padded; causal attention keeps earlier positions untouched, and only real positions are read.

Statistics. Decide on differences, not ratios. H1 tests Δ_A2* − 0.5·Δ_A1* > 0 (each arm's own dev winner). H2 tests Δ_A4 − 0.5·Δ_A1 for gains and 0.5·Δ_A1 − Δ_A2(2) for losses. Paired item bootstrap, 10,000 resamples, seed 0. Shares are descriptive, computed only when Δ_A1's CI excludes zero and |Δ_A1| ≥ 0.03 nats. Otherwise step 1 reports a duplication result only. Between 1a and 1b, the dev SD of paired differences gives a power estimate, written into the plan.

Tasks: ARC-Challenge test (allenai/ai2_arc, 1,172 items; download size not checked), 300 dev / 872 held-out. Synthetic arithmetic from seed 0 (two- and three-digit add and subtract, four candidates), 300 / 300.

Steps:

0. Base-task gate, before any arm runs (added 2026-11-10). A0 only, on the 300 dev items of each task, with the scoring above. A task passes when A0 accuracy's 95% Wilson interval lies above 30% (chance is 25%) and its point estimate is below 95%, and the mean normalised log P(correct) lies between −1.29 nats (0.1 above the uniform −1.386) and −0.05 nats, so a 0.03-nat change has room in both directions. ARC letter scoring on a 1.7B base model may sit at chance, and four-candidate arithmetic may sit at ceiling; neither is checked. On a fail, change the task once and re-gate: ARC switches to summed option-text log-prob; arithmetic switches to three-digit items whose distractors differ from the answer in one digit. A task that fails twice is dropped; if both are dropped, park the card. Cost: one forward per item, 600 items, run inside the 1a pilot process; about a minute (not checked), under 300 s and 8 GiB with the about 4 GB weights.
1a. One slice, dev only, on the tasks that passed step 0. Qwen3-1.7B-Base, bf16, mlx-lm in a uv environment. Weights about 4 GB, 28 layers; confirm both. Span length 4, starts 4, 6, …, 22. All arms on all spans. Each config is its own process under 300 s, writes JSONL, resumes. A 60-second pilot measures throughput and peak memory. If the projection exceeds 25 device-minutes, thin in order: starts every 3 layers, dev to 150 items per task, drop a = 1.1. Output: dev maps, winners, power estimate. Alone, an unconfirmed observation.
1b. One slice, held-out. Each task's duplication winner and amplification winner (span, a). Arms A0, A1, A2 at the winning a and at 2, A4, R. Decide H1 and H2.
2. Several slices. Qwen2.5-3B first, where CircuitProbe reports a gain (bf16 about 6 GB; pilot measures memory; 8-bit if over 8 GiB). Then Qwen3-4B-Base (about 8 GB bf16, so 8-bit) and Llama-3.2-3B (about 6.4 GB bf16; gated, Rob must accept the license; gating and size of mlx-community conversions not checked). Add A3, A5, norm-matched A2, per-layer amplification, a multi-hop set, split seeds 1 and 2.
3. IFEval and GSM8K with generation on a 7B to 8B instruct model in 4-bit. Needs longer limits or cloud.
4. One 27B to 32B model in 4-bit. Needs a raised memory limit.

Steps 3 and 4 are where the RYS and frankenmerge audience cares. Rob is asked before 1a whether he will approve them if step 2 confirms an effect; a "no" caps this card at a small-model note.

## Checks before believing anything

Noise floor: the paired item bootstrap. A0 against A0 guards hook and determinism bugs only, since bf16 MLX in one process is probably bit-stable.

Identity: A0 twice in separate processes, A2 at a = 1, and A1 with the copy's update zeroed all match A0 per item within 1e-4 nats. A batched item matches the same item scored alone within 1e-3.

Planted: (1) capture F(x) with the activation tools, compute x + 2F(x) by hand, match A2's post-span residual. (2) Add a known logit bias to the correct option; the analysis recovers it within its CI. (3) A synthetic arm A0 + 0.5·(A1 − A0) per item makes the difference test read zero.

Inconclusive: any Δ or difference whose CI includes zero. Step-1 results stay unconfirmed observations until step 2 repeats them.

## The control that could embarrass it

R, for the overshoot story: if a random push of equal norm hurts as much as a = 2, the loss decomposition says nothing about over-integration. A2 at its own best, for the re-read story: if it matches duplication, the "thinks twice" reading loses its evidence.

## Confounds a hostile reader would raise

- Winner's curse: winners chosen on dev, judged on held-out.
- Unequal search: amplification gets four values of a. Both own winners and A2(2) on duplication's span are reported.
- Compute: A1 and A4 add one span pass; A2 and R add none.
- Norm: later RMSNorm partly undoes norm changes. R controls size; post-span norms are logged.

## Kill criterion

Stop after 1b if no task gives a duplication change with CI excluding zero and size at least 0.03 nats; write a short note. Stop the programme if, after step 2, duplication has no confirmed held-out effect of either sign on at least two of three models. Drop the overshoot reading if R hurts as much as A2(2) in 1b and again in step 2.

## Cost on this hardware

- Gate: passed 2026-11-09 (12 searches).
- Step 1a: target 25 device-minutes or less, set by the pilot. A FLOP count of about 2M forward tokens gives about 11 minutes at peak and plausibly 25 to 40 at realistic efficiency, hence the thinning rule. Downloads: about 4 GB weights, ARC, the mlx-lm environment (not checked). Disk under 6 GB.
- Step 1b: smaller than 1a.
- Step 2: 4 to 8 slices; downloads about 4 to 6 GB each (not checked).
- Steps 3 and 4: not estimated; step 4 needs about 16 to 20 GB (not checked).

## What a yes and a no change

A true H1 is the citable surprise: one scalar per span replaces extra layers, and "under-stepped middle layers" becomes a training question. A gain carried by A4 separates step from re-read, supporting looped-depth work. Losses carried by A2(2) but not R turn TFLT's over-integration account into a one-pass test: useful, expected. Losses R matches mean the decomposition is uninformative there. Interaction-dominated means ill-posed at the metric level: a paragraph. A held-out duplication null says hobby gains look like selection noise at this scale.

## What carries to larger models

Step 1 covers one base checkpoint without generation. The arms port to any residual network; the result may not. CircuitProbe reports the duplication sign flipping between 3B and 7B. Untested until steps 3 and 4: 7B and above, instruct models, generation, mixture-of-experts, hybrid attention.

## Receipts to keep

- `runs/layer-repeat/plan.md` committed before 1a, changes dated underneath, including the power estimate before 1b.
- Per-config JSONL: item id, arm, span, a, option log-probs, correct index, post-span norm, answer-position entropy.
- Split files, arithmetic generator, random-direction seed.
- Identity and planted outputs with tolerances; pilot log; lockfile; model revision hashes.
- Analysis script and table: Δ per arm, H1 and H2 differences, I, with CIs; dev maps.

## Search gate, 2026-11-09

Second pass. Queries: 12 run, 0 refused: the four gate phrasings (residual-branch scaling at inference; looping ablations with step above 1; middle layers as Euler steps; boosting intermediate layers) plus frankenmerge and passthrough scale (two), one-pass extrapolation, attention and MLP output scaling, DenseFormer and LayerScale post-hoc gains, RYS wording, activation scaling, pruning compensation, enlarged ODE steps. Ten pages read.

Closest new works, linked above: Chen et al. 2023 (main text, Table 1), the RYS small-model sweep (full), the miqu-1-120b thread (full), the two 2026-10 loop papers (full or nearly). Only Chen et al. runs an amplified step in place of repeats, in trained shared models, for speed. Gate passed; the card stays a new test.

Design unchanged. Chen et al. supports a grid ending at 2 and A2(2) hurting; the RYS re-test supports the held-out split. Their GPT-2 numbers do not transfer to scored multiple choice on Qwen3.

Still open, not blocking: the last 16k characters of There Is No Free Loop; a citation-list read of Chen et al.

First pass: 4 searches refused (budget spent); 26 direct fetches instead, covering Jastrzebski, Lad, Painters, CoLa, Inner Loop Inference, Recirculation, SelfMorph and the rest of TFLT, plus five Semantic Scholar citation lists (12 abstracts read). Added answer-position entropy.

## Review record

Hostile review, 2026-11-09, verdict "queue after fixes". Must-fixes applied: amplification picks its own (span, a) on dev, surprise stated at P = 0.15; exactness limited to the span-output residual, I reported, A3 moved to step 2; difference tests decide, shares descriptive and gated on |Δ_A1| ≥ 0.03; losses analysed as the overshoot test, Qwen3-1.7B-Base kept for step 1; prior-art gap made a gate. Should-fixes applied: R in step 1, norm-matched A2 in step 2; step 1 split into 1a and 1b with a thinning order; carry section rewritten, Rob's approval for steps 3 and 4 asked up front. Minor: determinism-only checks stated; memory figures marked. The reviewer's searches ran in the gate's second pass.

Base-task gate check, 2026-11-10, after copies-as-evidence step 1a's model got no greedy answer right (one template inaccurate, the other off-format). The card had identity and planted checks but nothing confirming Qwen3-1.7B-Base sits above chance and below ceiling on letter-scored ARC or four-candidate arithmetic. Added step 0: A0-only dev gate with accuracy and log-prob bars, one task change on a fail, park if both tasks fail. Risk the hypothesis test would have been void without it: medium. Nothing else changed.
