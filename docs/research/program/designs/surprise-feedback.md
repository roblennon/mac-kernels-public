# Surprise feedback: tell the model how wrong it just was (round 2 S12)

Status: queued, search gate cleared 2026-11-09 (twelve keyword queries ran; no controlled transformer version of the idea found). Revised after a hostile review and two search passes, all recorded at the end. Designed, not run.

Merges the ledger ideas "Surprise feedback" (round 2, S12) and "Tell the model how wrong it just was: prediction error as an input to the next position".

Rocki (2016) gave an LSTM its own surprisal at each step and reported a gain. Does a transformer's mid-layer exit error at t-1, fed in at t, lower loss once extra parameters and exit loss, a plain previous-position pathway, and the model already having the signal are ruled out?

## Hypothesis

A small causal transformer fed, above layer m at t, its layer-m exit head's error about x_t reaches lower loss than the same model given a shifted prediction or a matched token-shift layer. It is wrong if it does not beat both beyond paired seed spread.

## Theory

The exit head at t-1 gave a distribution p over the next token. Once x_t arrives, two signals are free: the surprisal s = -log p(x_t), and the innovation e = E[x_t] - sum_v p(v) E[v], with E the token embedding. Adding g·(P e + embed(bucket(s))) at t, with a gate g starting at zero, keeps the graph feed-forward: training stays parallel and KV-cache inference stays exact.

E[x_t] is already present at t, and the subtracted term depends only on layer m at t-1. So the innovation is close to RWKV token shift, the speedrun's smear and Canon layers (both read in code, see below). The error framing earns its keep only if the subtraction adds something those lack. The scalar surprisal is the part attention plausibly cannot rebuild cheaply, since it needs a log-sum-exp at t-1; step 0.5 tests that premise.

## What is already known

- [Rocki 2016](https://arxiv.org/abs/1608.06027), read in full. Previous-step surprisal enters every LSTM gate; 1.39 BPC on enwik8 against 1.45 for its own LSTM. No transformer, seeds or matched controls; none of its 14 citers is a transformer error-input paper.
- [LIFT](https://arxiv.org/abs/2609.38149), 2026-09-29, read in full, the closest work. Each token gets a state built from the previous position's next-token distribution (a teacher's top-k softmax), fused at the input; at inference the model feeds back its own state. At 135M, C4 perplexity is 30.65 against 32.43 token-matched and 31.04 compute-matched. Trained on its own states with no teacher, it ties the compute-matched Transformer (+0.01 ± 0.05, three seeds); the authors put 98% of the gain on the teacher. No error term, surprisal or token-shift control; Rocki is not cited. [Code released](https://github.com/dortirosh1/LIFT). This is arm P with a teacher.
- [Asvin and Lindsey](https://arxiv.org/abs/2605.25459), 2026-05-25, sections 2.4 to 2.5 and appendix C via fetch summary. Llama-3.1-70B represents the incoming token's surprise under its own earlier prediction (layer 21, binning and PCA, no R²); steering it changes output entropy. Base models have it too, in a rotated subspace. At scale the model already computes what C would hand it.
- [T2MLR](https://arxiv.org/abs/2607.15178), 2026-07-16, read in full. A middle layer's output at t-1 enters an earlier layer at t through zero-initialised gates; Jacobi-trained at 2 to 4 times the wall clock. Perplexity 22.64 to 21.54 at 135M, yet a compute-matched Transformer wins on zero-shot accuracy.
- [KalmanNet](https://arxiv.org/abs/2107.10043), 2021, feature section read. A GRU is fed the innovation y_t - ŷ_{t|t-1} or the plain difference y_t - y_{t-1}; the better input depended on the problem. C against S, in filtering.
- [Mesa-optimization](https://arxiv.org/abs/2309.05858) and [von Oswald et al.](https://arxiv.org/abs/2212.07677) (2022-12-15, section 2 read): transformers build error-like quantities internally; one linear attention layer can carry a regression residual. The strongest argument for a null on the regression task.
- [Titans](https://arxiv.org/abs/2501.00663) (mostly read) gates memory writes by surprise, never an input; [DeltaNet](https://arxiv.org/abs/2406.06484), [TTT](https://arxiv.org/abs/2407.04620) and [MesaNet](https://arxiv.org/abs/2506.05233) (abstracts) do online regression inside the layer.
- Read by the idea's author: [Mixture of Inputs](https://arxiv.org/abs/2505.14827) (2025-05-20, training-free distribution feedback at generation), [WAVE](https://arxiv.org/abs/2410.03159) (moving-average error term in attention for time series), [Analog Bits](https://arxiv.org/abs/2208.04202) (diffusion self-conditioning), [Feedback Transformer](https://arxiv.org/abs/2002.09402) (representations, not error). [PredNet](https://arxiv.org/abs/1605.08104) and [Surprisal-Triggered Conditional Computation](https://arxiv.org/abs/2006.01659) (abstracts) do not feed error forward.
- Code read: modded-nanogpt has smear (a gated copy of the previous token's embedding) and hashed bigram and trigram embeddings, no prediction or error feedback. Canon layers ([Physics of LMs 4.1](https://ssrn.com/abstract=5240330), NeurIPS 2025) are a residual depthwise causal convolution, kernel 4 by default.
- modded-nanogpt [PR 205](https://github.com/KellerJordan/modded-nanogpt/pull/205) (test-time training) is closed unmerged. [PR 378](https://github.com/KellerJordan/modded-nanogpt/pull/378) (copy feature) is an open Track 1 entry reporting val loss 3.27548 ± 0.00057 over 8 runs.
- Adjacent, not error inputs: [Latent Recurrent Transformer](https://arxiv.org/abs/2605.26797) (2026-05-26, abstract) carries a hidden state to the next token at about 2x compute; [PonderLM](https://arxiv.org/abs/2505.20674) (2025-05-27, abstract) feeds a probability-weighted embedding back at the same position with extra passes; [Alpay et al. 2018](https://www.esann.org/sites/default/files/proceedings/legacy/es2018-120.pdf) (pages 1 to 2) use surprisal to gate RNN updates, not as input.

No work found feeds surprisal or innovation into a transformer. LIFT without a teacher and T2MLR both lose their edge against a compute-matched baseline; that is the bar C must clear.

## Predictions (written before any run)

- Step 0.5 probe: in the trained control, a linear probe above layer 2 at t recovers layer-2 exit surprisal at t-1 with R² below 0.5. Above 0.8, I predict a null for C against P and S.
- Switching chains, C against A: a pass (prior about 0.8), a plumbing check only.
- Switching chains, the claim that matters: C beats both P and S beyond paired seed spread. Prior about 0.3.
- P against A: small or none, since LIFT's self-fed states only tied a compute-matched baseline.
- No-switch chains: C minus A inside A's spread; a gain here flags capacity or optimisation, not error.
- In-context regression: C beats P and S by under 10 percent of A's excess squared error, since mesa-optimization suggests the model builds the residual itself. Prior of a gain beyond spread: about 0.25.
- Step 3: 0.005 to 0.02 nats over the auxiliary-loss-only arm. That straddles the 0.01 kill line: about a coin flip.

## Design

**Step 0, no device time.** Done 2026-11-09: no controlled transformer version found (see search gate). Before step 2, re-check the citers of Surprisal-Triggered Conditional Computation, the one list that was rate-limited.

**Step 0.4, outside any GPU slice.** Generate held-out sets (2,000 sequences per task, hashes recorded) and precompute exact Bayes losses in vectorised numpy: a posterior over the switch time with Dirichlet-multinomial counts on each side, O(T²) per sequence. Time 50 sequences first; split across processes if the full set would pass 300 seconds.

**Step 1, three or four slices.** Data generated in code. Switching Markov chains: 64 symbols, length 512, rows from Dirichlet(0.3), one switch uniform in 128 to 384. No-switch chains, same settings. In-context linear regression: dimension 8, 32 (x, y) pairs, fresh weights per sequence, noise 0.1; the exit head predicts y and the residual is fed forward. Settings are chosen, not tuned; changes are logged here with a date.

Model: 4 layers, width 128, 4 heads, exit head after layer 2, exit loss weight 0.3 in every arm, exit distribution detached. Arms:

- A, control: feedback modules present, input zeroed.
- C, error: innovation plus bucketed surprisal, gated.
- P, shifted prediction: g·P(sum_v p_{t-1}(v) E[v]) with no subtraction of E[x_t] and no surprisal; same gate, parameters and exit loss. This is LIFT's channel with the model's own exit head in place of a teacher; check its fusion against LIFT's released code (Eq. 5) before step 3.
- S, token shift: a Canon-style residual depthwise causal convolution, kernel 4, over layer-2 outputs, padded to C's parameter count. (Was width 2; changed 2026-11-09 to match the form that works in practice.)
- D, shuffled error from another sequence: a sanity check.
- Planted effects: B, exact Bayes surprisal fed through C's channel; O1, the true switch flag; O½, the flag delivered with probability 0.5.

Slice 1a is the go/no-go: a 30-second timing probe, then A on three seeds. A's excess loss over Bayes, over the 64 positions before the switch, must fall below 0.1 nats on every seed. Record each run's induction step, where that excess first halves. If a seed fails, cut to 16 states or chain two processes from a checkpoint, log it, and repeat 1a.

Slice 1a also gates headroom (added 2026-11-10), from the same three A runs at no extra cost. The deciding window is switch+1 to switch+32, so A must leave room there: A's mean excess over Bayes in that window must be at least 0.05 nats on every seed and at least four times the across-seed standard deviation. If it fails, run B on one seed inside the same slice (about 2.5 minutes, well inside 30 GPU-minutes); if B does not beat A by more than A's seed range, park the switching task, log it, and do not spend slice 1b on it. The regression task gets the same two gates at the head of slice 1d, with A's three seeds run first: A's excess squared error over the Bayes posterior-mean predictor (ridge with the weight prior's variance), over pairs 9 to 32, must be below half the excess of predicting zero (the model does the task) and at least four times A's seed standard deviation (room for C to help). If either fails, drop C, P and S on regression and log it. Both bars are design choices, not measured; the post-switch excess A will actually leave is not checked.

Slice 1b: C, P, S on switching chains, three seeds each, interleaved. Slice 1c: B, O1, O½, D, one seed each, plus C and A on no-switch chains. Slice 1d: A, C, P, S on regression, three seeds. If a run takes over 2.2 minutes, each slice splits in two. Seeds are paired: same data order in every arm and same initial weights for shared parameters. Each run is one process under 300 seconds.

The deciding metric is the per-seed paired difference, C minus P and C minus S, in mean excess loss over Bayes at positions switch+1 to switch+32. Report every seed, the induction step, and curves at half and full steps.

**Step 0.5, no training.** On trained A, C and P, fit ridge probes on the residual stream at layers 3 and 4, position t, for layer-2 exit surprisal and innovation at t-1. Report held-out R². Repeat on a small public LM if the lab has one (not checked), after fitting a mid-depth linear exit head; there, probe for the incoming token's surprise under the final head too, the quantity Asvin and Lindsey found at 70B, so the small-model R² sits next to a known large-model result. (Added 2026-11-09.)

**Step 2, two or three slices, only if C beats P and S.** Surprisal only, innovation only, exit not detached; a naive delta-rule layer in place of layer 3's attention, with and without feedback. A text8 anchor, not a Rocki replication (Rocki used enwik8): measure a 1-layer, width-512 LSTM's MPS throughput first, and drop the anchor or move it to the cloud stage if a run cannot get below 2.0 BPC within four chained processes.

**Step 3, many slices, only if step 2 holds.** GPT, 6 layers, width 384, GPT-2 tokenizer, FineWeb shards from the modded-nanogpt download (confirmed at run time). Copy feature and exit loss in every arm. Arms: control, error, shifted prediction, and a compute-matched control (A on extra tokens equal to C's measured FLOP overhead; added 2026-11-09, since LIFT and T2MLR tie or lose against one); 5 seeds. Gates before asking for slices: a written token budget per run; a measured 300-second throughput probe; document-boundary masking, reporting the share of tokens past position 200; and restart determinism (interrupted and uninterrupted runs give identical losses after restoring optimiser, data cursor and RNG). Then Rob decides whether a coin flip on the kill is worth the measured hours.

**Step 4, cloud, Rob's login and approval.** Port to modded-nanogpt Track 1 and count steps to target under the repo's significance rule. The baseline already carries smear and bigram embeddings, so it has a previous-position pathway built in. PR 378's ±0.00057 over 8 runs shows the speedrun resolves effects well under 0.01 nats; size the run count from it.

## Checks before believing anything

Self-comparison: on CPU, C with its gate fixed at zero and A's seed gives identical logits for 50 steps, and the analysis of A against itself reads exactly zero.

Planted effect, file level: 0.05 nats added to A's post-switch losses in a copy of its eval file must read back within 0.002, pre-switch unchanged. Arithmetic only.

Planted effect, training level: before C is read, the analysis must rank O1 above O½ above A, and B above A, each beyond A's spread. B is the ceiling for a surprisal input. If the ranking fails, step 1 says nothing.

Bayes check: with no switch, Bayes loss approaches the entropy rate, and no trained arm beats Bayes. Causality check: changing x_{t+1} leaves outputs up to t unchanged.

## The control that could embarrass it

Arms P and S: a same-size previous-position pathway with no error in it. If either matches C, the gain is token shift, not feedback. At step 3, the compute-matched arm, and a late-position gain that vanishes with the copy feature on (feedback rediscovering copying).

## Confounds a hostile reader would raise

Extra parameters and deep supervision: A carries the same modules, zeroed, and every arm has the exit loss. Previous-token shortcut: P and S. The induction plateau: slice 1a and the recorded induction step; if C mainly moves the transition earlier, that is reported as a training-speed effect. Leakage: the causality check. Tuning: one learning rate, chosen on A. A fading head start: curves at half and full steps. Compute: FLOPs logged, and a compute-matched arm at step 3. A rigged toy: the switching result is only a plumbing check.

## Kill criterion

Step 1: C fails to beat both P and S with the paired difference the same sign on all three seeds and its mean outside A's range. Or C's no-switch gain matches P's. Or probe R² is above 0.8 and C matches P. Step 3: under 0.01 nats across 5 seeds, or no win over the compute-matched control, or no larger late-position gain, or a gain that shrinks along the curve. A controlled transformer version turning up in the owed citer list also stops it.

## Cost on this hardware

Steps 0, 0.4, 0.5: no GPU time. Step 1: about 30 runs at an estimated 2.5 minutes, three or four slices; disk under 100 MB; system Python here has no torch (checked). Step 2: two or three slices; text8 adds about 131 MB. Step 3: the author's unmeasured 1.5 hours per run makes 20 runs about 30 hours; token disk not checked; free disk was 33 GiB on 2026-11-09. Step 4: the author's unmeasured 10 to 20 free-tier GPU hours.

## What a yes or a no changes

A yes against P and S says the subtraction matters and gives a near-free input channel in an ordinary forward pass. A no with B and O passing is a citable null; a high probe R² says why; a no from the delta-rule arm says modern sequence layers cover it.

## Carry to larger models, and what stays untested

A step 3 yes is a small-model observation. Nothing above about 30M parameters runs before step 4; the idea's author put a win at scale at 10 to 15 percent. Untested: long contexts, other modalities, fine-tuning from a checkpoint.

## Receipts to keep

Commit hash and dated changes; generator code, seeds, held-out hashes; Bayes timing; gate-zero, causality and planted-effect logs; one JSON per run (arm, seed, induction step, wall time, FLOPs, per-position excess loss); probe R² tables; step 0 queries.

## Search gate, 2026-11-09

First pass: five WebSearch queries refused (budget spent); about twenty fetches instead (Semantic Scholar citer lists, arXiv pages, KalmanNet, modded-nanogpt and PhysicsLM4 code) found LIFT, T2MLR, KalmanNet and [CoCoMix](https://arxiv.org/abs/2502.08524) (2025-02-12, method read; predicted concepts as extra positions, no error input), and changed S to Canon shape and added step 3's compute-matched arm.

Second pass: twelve WebSearch queries ran, none refused, worded for ML, predictive-coding, forecasting, in-context-regression and nanoGPT communities (error or surprisal as input, self-conditioning, early-exit logits fed forward, soft-embedding feedback, residual tokens). WAVE's four citers are all time-series forecasting. The citer list for Surprisal-Triggered Conditional Computation returned HTTP 429.

New closest, linked above: Asvin and Lindsey (2026-05-25, sections 2.4 to 2.5 and appendix C via fetch summary), Latent Recurrent Transformer (2026-05-26, abstract), PonderLM (2025-05-27, abstract), Alpay et al. (2018, pages 1 to 2), LIFT's appendix D. Set aside after abstracts: Kalman world models (arXiv 2603.13423) and Equilibrium Transformers (arXiv 2511.21882), neither feeds error to the next position.

Verdict: not scooped; LIFT partly answers arm P, and Asvin and Lindsey weaken the case at scale. Changes: step 0.5 now probes incoming-token surprise on a public LM; arm P is checked against LIFT's code. Gate cleared. Still owed before step 2: the rate-limited citer list.

## Review record

- Hostile review, all must-fix, should-fix and minor items done: the switching result became a plumbing check with no-switch and regression tasks added; P and S became the embarrassing control; slice 1a, paired seeds and step 0.5 were added; B, O1 and O½ must rank first; step 3 got its gates; the text8 anchor got a drop rule; PR facts were corrected. Verdict: queue after fixes.
- The reviewer's six searches did not run (budget spent); the search gate above replaced them.
- 2026-11-10, base-task check after the lab's copies-as-evidence step 1a (model got no greedy answer right, one template inaccurate, the other off-format; hypothesis never tested). Slice 1a already gated A learning the chain before the switch, with a stop, and B, O1 and O½ must rank before C is read. Missing: a gate that A leaves room in the post-switch window the decision reads, and any base-task gate on regression. Added both to step 1 with bars and stops; no extra slice. Risk the test would have been void without them: medium, mainly for regression.
