# Noise margin: does a transformer's response to noise bend away from its own linearization, and does Gaussian noise price real shortcuts?

Status: queued, 2026-11-09 (revised after hostile review). Search gate passed on a second pass the same day (11 queries, none refused). Step 2 is partly scooped for activation quantization (Sathyanarayanan 2026: matched Gaussian badly underprices it), so that arm is now a replication and positive control; the knee and the other shortcut classes stay open.

Inference shortcuts add error inside the network, and each is measured alone today. This card asks two things. Does a trained transformer have a noise margin: an amplitude below which its response to injected noise follows its own linearization, and above which damage grows faster? And is Gaussian noise of matched norm the right noise model for each shortcut class? The output is a table per layer and per shortcut class. It does not claim calibration replaces simulating a shortcut, which costs about the same forward passes.

## Hypothesis

In a trained transformer, single-layer damage KL_l(e) follows the linearized model at small relative amplitude e and departs upward above a knee e*. Separately, for at least two of four shipped shortcuts, the shortcut's KL is within a factor of 2 of KL from Gaussian noise of the same per-token norm at most layers. Each part can fail on its own.

## Theory

Von Neumann (1956): a chain of noisy stages works at any depth if each stage shrinks small deviations. In a network, shrinking at small amplitude is already part of the linear response. A margin is a change of regime: past some amplitude, errors leave the region that later readers treat as the same, and damage grows faster than linear.

KL is itself nonlinear in the size of the output perturbation, so a ratio of KLs can change with e in a network with no margin. The fix is a linear null. Push the same noise through the Jacobian with a forward-mode derivative, add it to the clean logits, and put that through the same KL. The deciding quantity is D(e) = KL_real(e) / KL_lin(e). The meter's curvature is in both, so D = 1 without a regime change.

Assumption: RMSNorm makes per-token residual norm a sensible unit, but Qwen residuals carry massive dimensions and a sink token, so the unit is reported two ways.

## What is already known

Novelty rests on a colleague's 19-query search with 7 sources opened, plus the 11-query search gate below. Three works found there change the picture for Step 2.

- [Sathyanarayanan, "Perplexity Cost Understates What Activation Quantisation Breaks"](https://arxiv.org/abs/2609.23125), 2026-09-19, about 100k of 128k characters of the HTML read through a summarizer. Fake-quantizes the residual stream at one block (shallow, middle or deep) in 12 models including Qwen3-1.7B, against Gaussian of matched per-channel magnitude. Gaussian reproduces only 0.173 of quantization's excess perplexity at 4 bits, 0.020 at 3 bits; quantized-to-Gaussian perplexity ratio is 1.00 at 8 bits with per-token scaling and rises to about 300 at 2 bits. Randomizing only the sign of the quantization error is nearly as harmless as Gaussian. No KL, no full layer sweep, no matmul-input, weight, KV or skip arms. Code released. For residual-stream activation quantization, Step 2's question is answered: Gaussian badly underprices it below 8 bits.
- [LeVi et al., "You Had One Job" (TAQ)](https://arxiv.org/abs/2511.06516), 2025-11-09, v4 2026-06-28, about 100k of 130k characters read through a summarizer. TAQ-KL adds uniform noise at a 4-bit step size to one layer's hidden states and scores the layer by output KL, then gives the top quarter 8 bits. This is nearly g_l. Never validated against real per-layer damage; fixed amplitude. Step 2's Spearman test supplies that validation.
- [Kennedy and Kennedy, "Quantization Error Is Spectrally Flat"](https://arxiv.org/abs/2609.33923), 2026-09-27, abstract. A random Gaussian probe estimates weight-quantization error size; propagated, it tracks the GPTQ objective (Spearman 0.81 to 0.83). Error size, not output damage or pricing.

- [Chen, Beyer, Zhu, Chen](https://arxiv.org/abs/2609.11716), 2026-09-10, body read (about 200k of 225k characters). New quantization error opposes inherited error, so relative hidden error stops growing in later layers of pretrained Qwen3; random init grows as a power law. W4, A4, W4A4, KL; no matched Gaussian, linear null, KV or pruning. Rare failures (about 0.28% of positions). Not a check, since injected noise does not depend on the block input.
- [Janiak, Heimersheim et al.](https://arxiv.org/abs/2409.17113), 2024, method read. Interpolating residuals between two prompts gives plateaus and a sharp jump in trained Qwen2 and OLMo, near-linear curves at random init. No noise, no linear null. Closest prior evidence for a knee.
- Abstracts only: [Ferreira da Silva and Heimersheim](https://arxiv.org/abs/2606.24964), 2026-06-23, plateaus, feature directions less robust than random. [Pandey et al.](https://arxiv.org/abs/2608.20988), 2026-08-21, Jacobian-scaled Gaussian noise as training. [Labiosa et al.](https://arxiv.org/abs/2608.03842), 2026-08-04, input perturbations. [Lad et al.](https://arxiv.org/abs/2406.19384), 2024, layer deletion. [Quant-Noise](https://arxiv.org/abs/2004.07320), 2020. [Li et al.](https://arxiv.org/abs/2609.27996), 2026-09-23, norm-scaled residual covert codes; a snippet says KL grows about quadratically with strength, unchecked. [Hill](https://arxiv.org/abs/2607.12266), 2026-07-14, per-layer quantization effects explain 85 to 93% of subset-loss variance.
- [Nettasinghe and Joseph](https://arxiv.org/abs/2603.12541), 2026-03-13, methods read. A 32-dimensional linear surrogate matches GPT-2-large's layerwise steering gain (Spearman 0.99 or more) for ε up to 0.5. Steering directions, no knee sought.

No search found one measure validated across pruning, quantization and KV damage.

Open, as far as checked: an amplitude knee measured against a linear null; matched-noise pricing of matmul-input activation quantization, weight quantization, KV-cache quantization and block skip; and a per-layer validation of a TAQ-KL-like score against several real shortcuts.

## Prediction

Written before any run. Amplitudes e = 0.5, 2, 8, 32% of residual norm. Intervals are bootstrap intervals over passages.
- Step 1, trained Qwen3-0.6B: D_l(0.5%) has an interval containing 1 at every layer. At the median layer, D_l is at least 1.5, with its interval above 1.25, at 8% or 32%. The knee e* (smallest amplitude where D's interval lies wholly above 1.25) is 2% or 8%.
- Step 1, secondary: R(e) = KL(noise after all blocks) / KL(noise after last quarter), real and linearized. Real R over linearized R is at least 2 at the highest amplitude where last-quarter KL stays below 1 nat per token.
- Step 1b, random controls: open prediction. Known results to read against, not predictions: Janiak et al. find near-linear curves at random init, and Chen finds power-law error growth there.
- Step 2: Q_l = KL(shortcut) / KL(matched-norm Gaussian at the same place) lies in 0.5 to 2 at 80% or more of layers for at least two of four shortcuts. Secondary: Spearman between g_l and shortcut KL is at least 0.8 for three of four and beats the best rival by 0.15.
- Effects that matter: D of 1.5 or more; Q within a factor of 2. Q within 2 at 50 to 80% of layers is partial.
- Added 2026-11-09 after the search, before any run: the residual-stream replication arm (Design, Step 2) gives Q above 2 at A4 at most layers, matching Sathyanarayanan. Matmul-input A4 is not predicted from it, since the error passes through the linear layers before reaching the residual.

## Design

Data: `Salesforce/wikitext`, `wikitext-103-raw-v1`. Validation passages pick the massive dimensions. 32 test passages for Step 1 and a disjoint 32 held out for recurrence, 256 tokens each, saved by index. WikiText is likely in Qwen3's pretraining data. Every arm sees the same passages, so exposure shifts the shared clean baseline and favours no noise arm.

Precision: fp32 throughout; bf16 rounding (about 0.4%) sits inside the amplitude range.

Noise: after block l, add e · (‖h‖/√d) · z per token, z standard normal. A second scale excludes the massive dimensions from ‖h‖ (list fixed on validation passages before the run). Position 0 gets no noise in main arms; a separate arm injects there only. KL is broken down by position bucket. Two disjoint seed sets of 2 seeds.

Linear null: torch.func.jvp of the logits with respect to a scalar s scaling the injected noise, at s = 0. One derivative per arm and seed covers all amplitudes. Forward-mode AD on MPS is not checked; fallback is a central difference at e = 0.01%, checked against fp64 on CPU for one passage.

Readouts: mean and per-token KL(clean ‖ noisy), plus relative error of the final normed hidden state, real and linearized. Added 2026-11-09: the share of tokens whose own D exceeds 1.25, so a knee carried by a rare tail (Chen saw about 0.28% of positions escape) is not averaged away.

Memory and order: 4 passages (1,024 tokens) per chunk. Clean logits per chunk are about 0.6 GB in fp32 at a vocabulary near 152k. They are computed once and held while every arm for that chunk runs in shuffled order, so arms interleave and clean is never recomputed. Clean residuals at every block are held too (about 0.1 GB), so a layer-l arm runs only blocks l to 27 and the head. KL streams to disk per chunk.

**Subject gate (Step 0, inside Step 1's slice, before any other arm).** The checks below test the meter, not whether this model's response to this grid can show a knee. On 8 validation passages (not test passages), one seed, layers 0, 7, 14, 21 and 27, all four amplitudes, real and linearized, measure mean KL per token. Also take the KL between the fp32 clean logits and a clean fp64 forward on CPU for one of these passages (the fp64 CPU path the derivative fallback already uses); that is the numerical floor of this computation. Pass needs all three. (a) Floor: at every gate layer, real and linearized KL at the lowest amplitude are above zero and at least 100 times that fp32-versus-fp64 KL. The e = 0 replay is not the bound, since it can be exactly zero; it stays under the self-comparison check. (b) Room: at 4 or more of the 5 gate layers, at least 3 amplitudes pass the floor and keep real KL below the 1-nat cutoff. Otherwise the kill criterion reads "no margin" on one or two usable points. (c) Resolution: at 8%, the passage-bootstrap 95% interval of D at the middle layer, projected to Step 1's 32 test passages by square-root scaling (gate half-width × √(8/32)), has half-width 0.1 or less, so 1 and 1.25 can be told apart; that is a gate half-width of 0.2 or less. If (a) or (b) fails with every failing layer in one direction, shift the whole grid once by a factor of 4 (down if KL saturates, up if it sits on the floor) and rerun the gate. If gate layers fail in opposite directions, keep the grid and add one amplitude a factor of 4 beyond each failing end (0.125% for saturating layers, 128% for floor-bound ones), read each layer only at the amplitudes that pass its own floor and cutoff, and rerun the gate. If it fails again, park the card with the gate numbers. If only (c) fails, use 64 test passages when the projection to 64 meets 0.1 and the timing pass allows; otherwise park. The kill criterion is not applied to a grid that has not passed. Cost: about 5 layers × 4 amplitudes × 2 passes × 2 chunks, roughly 80 to 120 forward-equivalents (estimate, not measured), plus one fp64 CPU forward (memory not checked), one process under 300 s with load. It is charged to Step 1's 30 device-minutes.

**Step 1 (one slice).** Trained `Qwen/Qwen3-0.6B`. Single-layer arms: 28 layers × 4 amplitudes × 2 seed sets, plus linearized passes. Depth arms: all blocks and last 7, × 4 amplitudes, real and linearized. Position-0 arm and massive-dimension variant at 2% and 8%. All checks below. A 60-second timing pass projects time; a process projecting above 250 s is split by amplitude band, and if the slice projects above 30 device-minutes, cut to the 14 even layers. Step 1 alone says whether a knee exists against the linear null.

**Step 1b (one slice).** Same arms on the held-out passages. Same arms on two random copies of the config: default init (std 0.02), and a matched control whose matrix entries are permuted and whose block outputs are rescaled so per-layer residual norm and update-to-residual ratio match the trained model. Trained-versus-random is read against the matched control.

**Step 2 (one or two slices; novelty search done).** Shipped shortcuts at one block l at a time: per-token absmax fake-quantization of the inputs to block l's linear layers after the norm at A8, A6, A4 (where Rob's W4A8 path rounds); weight-only RTN 4-bit group 64, and GGUF Q4_0 and Q3_K through a reference dequantizer; 4-bit KV cache; block skip. Each gets a Gaussian arm matched to its per-token change in the residual after block l. Main metric: Q_l. Rank analysis: Spearman over layers between g_l = KL / e² at 2% and each shortcut's KL. Rivals: layer index, residual norm, block-skip KL, block influence (1 − cosine of block input and output), leave-one-out cross-shortcut prediction. Spearman on all layers and with the first two and last two dropped, with passage bootstrap intervals. Token-shift "stale updates" and adjacent-token merging are dropped: neither ships for LLMs, and merging leaks future tokens. Added after the search: (a) a replication arm, residual stream fake-quantized after block l at 4 and 3 bits per token against Gaussian of matched per-channel magnitude, as in Sathyanarayanan; the positive control that Q detects a known mispricing; (b) a sign-randomized copy of each shortcut's error, which splits Q into magnitude and structure; (c) TAQ-KL as a named rival to g_l.

**Step 3 (Rob's approval, higher memory limit).** Repeat on `Qwen/Qwen3-1.7B`, `Qwen/Qwen2.5-0.5B`, `HuggingFaceTB/SmolLM2-1.7B`, one download at a time. 1.7B in fp32 needs about 7 GB of weights. Sizes not checked.

**Step 4 (Rob's approval, hours).** The same table on Wan 2.1 1.3B or Krea 2 Turbo with held-out denoising error in place of KL. Shortcuts: Rob's W4A8 kernel, caching across denoising steps, token merging.

## Checks before believing anything

- Hook validation: assert each injected perturbation's per-token norm equals the intended norm to 1e-5 relative; log and assert layer and positions.
- Self-comparison: KL below 1e-7 nats per token at e = 0. Two disjoint seed sets on one arm agree within the passage interval. At e = 0.05%, real and linearized KL agree within 2%, which checks the derivative.
- Planted meter effect: Gaussian noise of σ = 0.1 on the final logits; expected KL to second order is 0.5 σ² (1 − Σp²) per token; the mean lands within 10%.
- Planted knee: a model copy clamps the perturbation (noisy minus clean residual) to 4% of residual norm after each block. The knee finder must place the departure at the first amplitude above 4%, and report no knee on the linearized model.
- Rank bookkeeping: a fake shortcut of Gaussian noise times a known per-layer factor must rank against g_l times that factor with Spearman of at least 0.95.

## The control that could embarrass it

The linear null: if D's interval contains 1 at every amplitude under the 1-nat cutoff, the knee was the meter. The matched random control: the same D curve means architecture, not learning. The rivals: if another shortcut's profile predicts as well as g_l, then g_l only measures "some layers are fragile". Until Step 3, any result is an unconfirmed observation on one checkpoint.

## Kill criterion

Stop after Step 1 if D's interval contains 1 at every amplitude below the 1-nat cutoff for at least 24 of 28 layers. That is a clean "no margin beyond linear response", written up. Stop after Step 2 if Q falls outside 0.5 to 2 at most layers for all four shortcuts, or a rival predicts as well as g_l. Then the write-up says which shortcut classes Gaussian noise misprices and by how much.

## Cost on this hardware

Estimates, not measured. Each step opens with a 60-second timing pass that also records revision hashes and file sizes.
- Step 1: roughly 900 to 1,200 forward-equivalents of 1,024 tokens, counting a forward-mode derivative as about three. Device time not checked. Memory 3 to 5 GB. About 1.5 GB download (not checked); disk has about 31 GB free.
- Step 1b: about three times Step 1, likely two slices. Step 2: one or two slices. Step 3: a few device-hours, about 9 GB downloaded in turn. Step 4: not checked.

## What a yes changes and what a no changes

A yes gives a measurable limit of linear response to noise, and names the shortcut classes Gaussian noise prices correctly. A no says response is linear up to breakdown, or each class needs its own error model.

## Why it should carry to larger models

Untested. Plateau work reports sharper stable regions with size, a weak hint only. Nothing above 2B, and no diffusion transformer until Step 4.

## Receipts to keep

The plan committed before Step 1; passage indices, hashes, seeds, massive-dimension list, versions; raw per-token KL, real and linearized (npz); check results; D, R, Q and Spearman tables with intervals; run script and log.

## Search gate, 2026-11-09

First pass: all 5 queries refused for budget; Chen to Quant-Noise above were read by direct fetch. It added the per-token tail readout.

Second pass: 11 queries run, 0 refused, standard mode: the five owed, then six in analog-hardware, mixed-precision, KV-cache and interpretability wording. Closest works and how much was read are listed under "What is already known". Snippets only: Zlokapa et al. 2022 (von Neumann threshold, constructed networks), NORA, EDGE-LLM, arXiv 2302.01382 (said to find per-tensor Gaussian noise weak at ordering sensitivity; abstract silent) and arXiv 2604.13440.

Verdict: partly scooped. Gaussian underpricing of residual-stream activation quantization is known, so that arm is now replication and positive control. Nothing found measures a knee against a linear null, prices weight, KV or skip shortcuts against matched noise per layer, or validates a TAQ-KL-like score across shortcuts. Gate cleared for Step 2 with the three added arms.

Still open: the unread last fifth of Sathyanarayanan and LeVi (appendices may hold per-layer tables); 2302.01382 and 2604.13440 in full; HAWQ, UNIQ, ShortGPT not reread.

## Review record

Hostile review, all items done: JVP linear null; Chen dropped as a check; real shortcuts only; stronger rivals, bootstrap, Q as the main Step 2 test; sink and massive dimensions; permuted-and-rescaled random control; held-out passages and disjoint seeds; chunked memory plan; Step 1 as the simpler experiment; novelty search gating Step 2; planted knee. The reviewer's three searches failed on budget, superseded by the search gate.

2026-11-10, subject-readiness review after the copies-as-evidence case 002 lesson (model got no greedy answer right, one template inaccurate, the other off-format; hypothesis never tested): every check here tested the meter; none confirmed the amplitude grid gives readable, unsaturated KL on Qwen3-0.6B with D resolvable at 1.25. Added Step 0 range gate with bars and a park rule. Risk without it judged medium: a saturated or floor-bound grid would have fired the kill criterion as a false "no margin". Qwen3-0.6B's actual KL at these amplitudes is not checked.

2026-11-10, cross-family gate review (Sol): resolution bar now projects the 8-passage half-width to the 32 test passages (0.2 at the gate), the floor is an fp32-versus-fp64 KL instead of a possibly zero e = 0 replay, and opposite-direction layer failures extend the grid at both ends instead of shifting it. The gate stays a separate pre-step, since no existing baseline run covers noise arms on non-test passages.
