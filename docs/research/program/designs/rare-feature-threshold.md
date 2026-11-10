# Does beta2 set the rarest fact a model can keep? (rare-feature-threshold)

Status: queued, 2026-11-09. Gate 0 (prior-art search) passed on 2026-11-09 (12 queries, none refused): no source states a beta2-dependent cutoff, but the decay fix has close precedent and recommender work goes the other way (see the search gate). Step 0 may start.

Merged cluster: "Rare features and the cost of weight decay."

Weight decay hurts rare tokens most. Under AdamW, a rarely touched embedding row gets a fixed-size kick per touch, whatever its gradient, and decay pulls it back every step. The balance gives a cutoff touch rate that moves with beta2 and has no learning rate in it. Revised after a hostile review (record at the end).

## Hypothesis

A planted fact whose trigger has its own embedding row is kept when its touch rate p (exposures per step) is above a cutoff p_c. When touches are sparse (p well below 1-beta2) and the run has reached balance, p_c scales with the per-touch kick computed below. Between beta2 0.95 and 0.999 that predicts a shift of about 0.74 decades. This is wrong if p_c does not move with beta2 (the static account of Pinto et al.), or if it stays put when the embeddings are exempt from decay.

## Theory

Treat a trigger row as a weight w, touched with probability p per step and pulled back by lr*lambda*w every step. With beta1 0.9 and one isolated touch, the total update over the momentum tail is lr*K, where K = S/sqrt(1-beta2) and S = 0.1/(1 - 0.9/sqrt(beta2)). S is 1.31 at beta2 0.95, 1.05 at 0.99 and 1.00 at 0.999. The kick does not depend on |g|. Average drift p*lr*K equals decay lr*lambda*w at w* = p*K/lambda. The fact is kept if w* exceeds w_need, the row size the shared layers need to read it. So p_c = w_need*lambda/K, and lr cancels.

Three conditions follow, and each arm must meet all three for its cutoff to test the law:
- Sparse at the cutoff: the touch interval k_c = 1/p_c must be well above 1/(1-beta2). This reduces to lambda*w_need < S*sqrt(1-beta2), which is 0.29 at beta2 0.95, 0.105 at 0.99 and 0.032 at 0.999.
- Balanced: many touches per tau = 1/(lr*lambda). At the cutoff that number is w_need/(lr*K), so lr*K must be well below w_need.
- Long enough: the run lasts several tau, so the row is not a sawtooth that decay erodes between rare touches.

With the old settings (lambda 0.1, tau 3,333, 12,000 steps), the 0.999 arm fails one of these for any plausible w_need, so step 0 now picks the pair and settings.

When touches are frequent, the second moment tracks p*g^2 and p_c = (w_need*lambda)^2: no beta2, Pinto's exponent.

Other assumptions: epsilon far below sqrt(v_hat) on trigger rows; similar w_need across facts; shared layers cannot keep the fact alone; fp32 optimizer state. The author's numpy toy gave 0.67 decades for 0.95 against 0.999.

## What is already known

That sparse gradients call for beta2 near 1 is old: [Kingma and Ba](https://arxiv.org/abs/1412.6980) say so in sections 3 and 5, as a reason to correct bias, with no decay and no cutoff. That Adam scales up rare rows' updates is known ([Stollenwerk and Stollenwerk](https://arxiv.org/abs/2502.08441), dense regime, no decay on embeddings; Kunstner et al., arXiv 2402.19449, on heavy-tailed classes). Any novelty is the balance against decay, the cutoff's beta2 dependence and lr independence. The fix is not new: frequency-aware per-row coefficients exist in Caffe2's counter-based decay, Pinterest's frequency-adaptive learning rate and AdamAR (links below). AdamAR decays stale rows harder, against one-epoch overfitting in CTR models; TRD decays them less, so TRD must report held-out loss too.

[Pinto et al.](https://arxiv.org/abs/2410.11985), theory read: their cutoff, n_k at or below about lambda_W*lambda_H*N^2, is Corollary 4.6 of Dang et al. (ICML 2024) on global minima of an unconstrained-features model. No optimizer, so no beta2 and no lr; lambda squared only when both decays are equal. Read by the colleague: [Chang et al.](https://arxiv.org/abs/2406.11813) (exposure-interval threshold, decay and beta2 not varied), [Kosson et al.](https://arxiv.org/abs/2305.17212) (dense equilibrium, beta2 drops out). Abstracts: [McClendon et al.](https://arxiv.org/abs/2610.00423) (AdamW growth against decay for massive activations, lambda swept, beta2 not mentioned); [Li et al.](https://arxiv.org/abs/2110.04844) (counter-based per-token learning rates under SGD), precedent that makes the TRD fix a variant, not a new idea.

**Gate 0 (no compute), passed 2026-11-09.** No paper found states the beta2-dependent cutoff; details in the search gate.

## Prediction (written before any run)

Unit: p_half, the touch rate at which a rung of facts recovers half its loss, in log10. Delta(a, b) = log10 p_half(a) - log10 p_half(b).

Hand arithmetic, not simulated: in the sparse, balanced limit Delta(0.95, 0.999) = 0.74, Delta(0.95, 0.99) = 0.25 and Delta(0.99, 0.999) = 0.48 decades. Step 0 replaces these with exact values for the chosen settings, and that file is the committed prediction for step 1.

For the bigram model in step 1:
- Delta(0.95, 0.999) lies between the step-0 value minus 0.25 and plus 0.15. The minus side is the committed allowance for dilution by the shared readout.
- Each arm's measured k_c is within 0.3 decades of the step-0 value, given the pilot's w_need.
- Exempt-decay arm: p_half falls by more than 0.5 decades, or no cutoff on the ladder.
- Touch-rate-scaled decay arm: p_half falls by more than 0.5 decades, at background loss within 1% of the 0.95 arm.

What each outcome means, fixed now:
- In the band, checks passed, all arms in window: an unconfirmed observation for the law. Go to step 2.
- Delta under 0.2 with all arms in window: the beta2 law is killed for this model.
- Delta from 0.2 to the band's low edge: beta2 matters, but the per-row balance is not the whole mechanism. Do not run the GPT step. Next is the lambda slope in the bigram model.
- Delta above the band: the law is wrong in the other direction. Report it.
- Any arm outside its sparse-and-balanced window, or failing the plateau check: design failed for that arm. Not a kill and not a confirmation.

## Design

**Step 0: exact scalar simulation (CPU, seconds).** Run torch.optim.AdamW on independent scalar weights, one per fact, each with logistic loss on logit w - w_need, zero gradient between touches, and step 1's exact touch schedule. This is the real recurrence, with bias correction, epsilon and decoupled decay. Output per arm: the recovered-fraction curve, k_c, Delta, and flags for the three conditions above. Search lr, lambda and run length for a setting where every arm passes. If none fits a slice for 0.999, step 1 runs 0.95 against 0.99 and the 0.999 arm moves to its own costed step. Commit the curves before step 1.

**Step 1: bigram softmax model (one slice).** Embedding width 64, one shared linear layer to 256 outputs, softmax. Untied, fp32, CPU unless the timing probe favours MPS. Background is an order-1 Markov chain over 256 common tokens with Zipf-shaped rows, from a fixed seed. Each fact is a dedicated trigger T followed by a common target Y. Ladder: fixed exposure intervals k from 16 to 16,384 at sqrt(2) spacing (21 rungs), 24 facts per rung, phases staggered at random. Thirty-two control triggers are never shown. Rungs and targets are assigned at random per seed.

A 2,000-step pilot at lambda 0 logs the trigger-row norms of kept facts as an estimate of w_need. It feeds step 0, which then fixes lr, lambda and run length.

**Gate 1a: can the model keep a fact, and does G95 lose some (added 2026-11-10).** Step 0 and the planted checks test the arithmetic and the pipeline, not whether this network learns T to Y or whether decay removes facts inside the ladder. Two bars, read before any other arm runs:
- Pilot (lambda 0): median r at least 0.8 on rungs k = 16 to 64 (72 facts). If not, the model cannot learn the base task: stop, raise lr or width once, else park.
- G95 seed 0 runs first, at step 0's settings. Rung means must reach at least 0.8 at the frequent end and at most 0.2 at the rare end, with the 0.5 crossing at least two rungs from either end. If not, the cutoff is off the ladder: re-run step 0 with lambda or run length moved by the miss, retry once, else park. No other arm starts until this passes.
Cost: the pilot is already planned; the gate adds only the reordering. Both fit one slice if the timing probe shows a run under 300 s; not measured.

Arms, interleaved: G95, G99, G999 at seed 0; G95 seed 0 rerun; seed 1 of the three; X95 (embeddings exempt from decay); TRD (per-row decay scaled by touch rate, beta2 0.95); seed 2 of the three; L95 and L999 (lazy moments, below). Processes checkpoint to stay under 300 s. Arms that do not fit move to the next slice.

**Lazy-moment arms (added 2026-11-09, from the search).** Sparse embedding optimizers (torch SparseAdam, LazyAdam) update moments only on touched rows, so there is no momentum tail and v does not decay between touches: the kick is about lr, with no 1/sqrt(1-beta2). L95 and L999 use masked moments plus the same every-step decay. The balance account predicts Delta(L95, L999) near 0, and a gap between dense and lazy at the same beta2 of log10 K: about 0.77 decades at 0.95 and 1.50 at 0.999 (hand arithmetic; step 0 simulates both and replaces these). Pinto's static account predicts no dense-lazy gap, so this pair separates the accounts at one lr. Step 0 runs the lazy recurrence too.

**Metric that decides.** Per fact, L = -log p(Y | T) every 500 steps, averaged over the last fifth. r = 1 - L / L_ctrl, with L_ctrl the control triggers' mean loss. p_half is the 0.5 crossing of rung means, interpolated in log10 p; also report the 10-90% width.

**Step 2 (GPT, three or four slices).** The old card's 2-layer, width-64 GPT on the same data, with the pair and settings step 0 says fit, probably 0.95 against 0.99 with lambda lowered and lr raised. Prediction: step 0 plus step-1 dilution. Add E99 (higher beta2 on embeddings only).

**Step 3.** Lambda slope (near 1 sparse, near 2 dense), 4x lower lr at the same tau multiple (shift under 1.5x), SGD with momentum, lambda 0, onset snapshots. Bigram first.

**Step 4 (cloud, Rob's approval).** TinyStories, 10 to 30M GPT. Data size not checked.

**Desk check before step 4.** For one public model, count the sparse vocabulary tokens: a token is touched per step with probability about 1 - exp(-f*B), so it is sparse only when its frequency f is below about (1-beta2)/B. With B in the millions, only the rarest tokens qualify. The old Pythia step is dropped, since nearly all its tokens are dense. Pythia's settings were not checked.

## Checks before believing anything

- Self-comparison: the same-seed rerun of G95 must give identical per-fact losses on CPU, or the size of MPS nondeterminism is stated.
- Noise floor: the spread of p_half across three G95 seeds, plus a bootstrap over facts within each rung (1,000 resamples) on every p_half and on Delta. Read Delta only if the floor is under 0.1 decades and the Delta interval is narrower than 0.3. A difference inside the floor is inconclusive.
- Planted touch-rate effect: a group of 12 facts per rung, from k = 64 up, exposed every k/4 steps but labelled at rung k. Its p_half must sit 0.60 ± 0.15 decades from the main group's in every run. Its crossing must be at least two rungs from either end of the ladder.
- Planted beta2-sized effect: the p_half pipeline, run on step-0 and synthetic curves with known Delta (0, 0.5) plus observed-size noise, returns each within 0.05 decades.
- Plateau: p_half moves under 0.1 decades over the last fifth. Smallest sqrt(v_hat) on trigger rows stays well above epsilon (1e-10).

## The control that could embarrass it

Exempt the embeddings from decay (X95). If the cutoff stays within 0.15 decades of G95, the shared layer reads the random initial rows and the balance is not the mechanism. A Delta near zero with every arm in window means Pinto's static account wins. X95 does not separate the two accounts, since Pinto's cutoff also falls when row decay is removed; Delta, the dense-lazy gap, the lr test and the lambda slope do (added 2026-11-09). If G999 and L999 sit within 0.2 decades of each other with both in window, the momentum-tail kick is not the mechanism.

## Confounds a hostile reader would raise

Beta2 also changes the shared layer (E99 in step 2). Facts are read at varying times since last touch (staggered phases, averaged reads). Arms reach balance at different speeds (equal tau multiples, plateau check, window flags). A bigram model is not language (steps 2 and 4).

## Kill criterion

After both planted checks pass and the floor is under 0.1 decades: Delta under 0.2 with every arm in its sparse-and-balanced window kills the beta2 law. Later, a lambda slope outside 0.6 to 2.2, or a 4x learning-rate change that moves p_half more than 2.5x, kills it.

## Cost on this hardware

Gate 0 is search only; step 0 takes seconds on CPU. Step 1: per-step time not measured; the timing probe counts evaluation time, and the plan states per-run wall time against the 300 s cap before running. Up to 12 runs in one slice, under 100 MB, no downloads. Step 2: the old 10 ms per GPT step was a guess; at 25 ms a 12,000-step run takes 300 s, so checkpoints and three or four slices. Step 4 needs Rob's approval.

## What a yes or a no changes

A yes says that where touches are sparse, beta2 and decay, not learning rate, set how much of the tail survives under dense AdamW, and that lazy-moment optimizers behave differently. The citable part is the law and the dense-lazy gap; touch-rate-scaled decay is a variant of known per-row schemes, and recommender evidence (AdamAR) says protecting stale rows can cost generalization. A no says the static account suffices.

## Why it should carry to larger models

Only where touches are sparse: small-batch training and fine-tuning, recommender ID embeddings, the rarest LLM tokens. Large-batch pretraining is dense, where beta2 drops out, and LLM facts rarely live in single rows. Untested: tied embeddings, schedules, bf16 state.

## Receipts to keep

Gate-0 search log; step-0 curves before step 1; seeds and assignments; per-fact loss CSVs; timing probe; every check; p_half, Delta and width with code, torch version, device and git hash. Step 1 can reject the beta2 law only for dedicated rows in a small softmax network.

## Search gate, 2026-11-09

First pass: 0 queries, 4 refused (shared budget); read Kingma and Ba, Pinto et al. (theory), McClendon et al. and Li et al. 2021 (abstracts) by direct fetch.

Second pass: 12 queries run, 0 refused, in recommender, optimizer-library, LLM rare-token and equilibrium-theory wording.

Closest works:
- [AdamAR](https://arxiv.org/abs/2511.06374) (Nov 2025, ICLR 2026), read except the appendices. Per-row decay min(1, alpha * interval since last touch), every step, derived as inversely proportional to frequency. No beta2 or momentum analysis, no equilibrium norm, no cutoff. Code and public datasets (iPinYou, Avazu, Amazon).
- [Pinterest FAL](https://arxiv.org/abs/2505.05605) (May 2025), method read. Dense Adam (0.9, 0.999), learning rate scaled by log frequency, no embedding decay, no beta2 analysis.
- [Coupled Adam](https://arxiv.org/abs/2502.08441) (Feb 2025, v3 Aug 2025), most read. E[v] proportional to unigram probability; beta2 fixed at 0.95; no embedding decay; no cutoff.
- Caffe2 counter-based decay (PyTorch PR 66177) and PyTorch SparseAdam docs: snippets only.
- Abstracts, not close: Han et al. and Zhao et al. (Feb 2026; the second finds decoupled decay blocks neural collapse, which weakens a static account for AdamW), Glentis et al. (May 2026).

Verdict: not scooped. The fix has precedent and an opposite-direction rival. Design changes: lazy-moment arms; TRD reports held-out loss.

Still open, none blocking step 0: McClendon's full text, Kunstner et al., AdamAR's appendices, FBGEMM's current decay modes. Later, not costed: dense AdamW at two beta2 values on Avazu, with AdamAR as baseline.

## Review record

- Must-fix, all agreed: 0.999 cannot be sparse and balanced at tau 3,333 (window conditions, step 0 picks settings, "design failed" outcome, GPT moved to step 2); 0.85 ignored momentum (0.74 by hand, step 0 commits the value); gap between kill and band now assigned, dilution allowance 0.25; prior-art search made gate 0 (passed in the second pass, not scooped, lazy arms added); LLM relevance narrowed to sparse regimes, Pythia step replaced by a desk count.
- Should-fix: three-seed floor plus bootstrap (agreed); beta2-sized plant runs on curves with known Delta (partly agreed); scalar, bigram, GPT order (agreed).
- Minor: timing probe counts evaluation; same-seed rerun added; TinyStories size and Pythia's settings still unchecked.
- 2026-11-10, base-task check: no step confirmed that the bigram model learns T to Y or that G95 loses facts inside the ladder; only the scalar simulation and planted checks on the pipeline. Added gate 1a (pilot r at least 0.8 on k 16 to 64; G95 seed 0 first, crossing two rungs inside the ladder; retry once, else park). Risk of a void test without it: medium.
