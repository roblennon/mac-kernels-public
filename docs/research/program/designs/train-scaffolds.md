# Train with a lookup table, ship without: catalyst or capacity?

Status: queued, 2026-11-09, search gate cleared: 12 searches, none refused, found nobody removing an n-gram table during training. Step 3 gains a MeCo-style arm. Step 1 starts on the lab's case 001.

Round 2 shortlist S03, "Train-time scaffolds removed before shipping". This card covers only the n-gram table; the attention-hint half has its own card, not yet written.

Hashed n-gram tables speed up language-model training (about 7 of 40 seconds in a modded-nanogpt record; Engram uses one at 27B), but they are large to serve. If the table is a catalyst, it can be faded out before training ends with the gain kept. If it is capacity, the gain leaves with it, and speedrun records should count it as parameters. As far as the search below goes, nobody has measured which.

## Hypothesis

A small GPT trained with a hashed bigram table, whose gate fades to zero during the stable learning-rate phase, ends with the table removed at a lower validation loss than the same GPT never given a table. Separately, a learned table retains more than a content-free table of the same size on the same schedule. The hypothesis is wrong if the annealed arm is inconclusive against no table, or if the content-free table retains as much.

## Theory

Catalyst reading: the table hands layer 0 local-context features from the first step, so later layers start on longer-range structure sooner, and the backbone rebuilds the local features while the gate fades. Capacity reading: the table stores per-n-gram vectors the backbone never learns.

Evidence is indirect. MeCo (below) shows a training-only input dropped at cooldown can leave a gain behind. Evidence on tables concerns models that keep them. For: Engram's layer 5 aligns with its baseline's layer 12 by CKA; an image study found a frozen random table nearly as good as a learned one. Against: that study found disabling the pathway catastrophic, and Engram keeps 29 to 44% of factual-benchmark performance with the table cut at inference.

Assumptions: the backbone needs learning rate left to take over, so the fade must end before cooldown; a tiny model says something about the sign of the effect at larger sizes (the weakest assumption).

## What is already known

Sources come from direct reads, citation lists and 12 keyword searches (see the search gate section). None anneals or removes an n-gram table during training and measures what remains.

- [DroPE](https://arxiv.org/abs/2512.12167), 2025-12, read in large part. Drops RoPE for the last 2B of 16B tokens, matches the baseline. Positions only.
- [HASTE](https://arxiv.org/abs/2505.16792), 2025-05, read. Diffusion teacher alignment cut abruptly; gains persist. A teacher and a loss term, not an input feature.
- [Engram](https://arxiv.org/html/2601.07372v2), 2026-01, read in large part. Suppresses the table at inference only, as a stress test.
- [modded-nanogpt PR 360](https://github.com/KellerJordan/modded-nanogpt/pull/360), read. 84,602,880 rows before layer 0 behind a gate, worth +7.05 s. Nobody anneals it, drops it late, or evaluates without it; commenters dispute whether it counts as parameters. The [record table](https://github.com/KellerJordan/modded-nanogpt), read 2026-11-09, has bigram entries from record 62 (2026-01) to 92 (2026-08), none removing the table.
- [Engram in image generation](https://arxiv.org/abs/2605.13179), 2026-05, abstract. A frozen N(0,1) table costs only 0.10 FID; disabling the pathway is catastrophic. No removal during training.
- [Token-Superposition Training](https://arxiv.org/abs/2605.06546), 2026-05, method and results read. The nearest catalyst result: bagged token embeddings for the first 20 to 40% of steps, then switched off abruptly; the gain persists at 270M to 10B. A different mechanism, and a precedent for step 3's abrupt cut.
- [COUNT](https://arxiv.org/abs/2610.20456), 2026-10, abstract. Precomputed n-gram counts, no learned parameters, kept at inference; at 1B matches a model with 57% more compute.
- [Low-Bit N-Gram Embeddings](https://arxiv.org/abs/2610.31646), 2026-10, abstract. 4-bit tables keep over 99% of the gain. Weakens the "awkward to serve" motivation, not the catalyst question.
- [MeCo, metadata conditioning then cooldown](https://arxiv.org/html/2501.01956v3), 2025-01 (ICML 2025), method and ablations read. The nearest precedent for a training-only input: URLs prepended, dropped for the last 10% of steps. At 1.6B on 160B tokens, evaluated without metadata, 56.7 average against 55.7 standard; never dropping gives 50.3. Hashed URLs match real ones. Context tokens, not a table, removed inside cooldown.
- [Annealable soft-prior transformers](https://arxiv.org/abs/2609.10287), 2026-09, abstract via an aggregator. Faded attention biases keep recall accuracy at gate 0 (about 0.734 against 0.095 unfaded); a hard switch does not. Synthetic tasks, no table; mostly for the attention-hint card.
- [modded-nanogpt PR 201](https://github.com/KellerJordan/modded-nanogpt/pull/201), 2026-01, read. The first bigram record: 5 × vocab rows, zero-initialised, added at every layer, no gate, -5.1 s. Never removed.
- Search summaries only, none removes a table: [SCONE](https://arxiv.org/abs/2502.01637) (2025-02) serves its n-gram embeddings off-accelerator; [CoCoNTs](https://arxiv.org/abs/2407.02819) (2024-07) uses n-gram statistics as a loss; [PIQL](https://arxiv.org/abs/2605.07799) (2026-05, tabular) reconstructs privileged inputs at inference.
- Abstracts: [formal-language pre-pretraining](https://arxiv.org/abs/2502.19249) (2025-02) competes for saved tokens; [N-gram Is Back](https://arxiv.org/abs/2210.14431) (2022) keeps its n-gram model; [share-then-unshare](https://arxiv.org/abs/2110.03848) (2021) scaffolds weights.

Not checked: Canon layers, re-parameterisation, auxiliary heads, full texts of Engram's citers and MoME; LUPI at abstract level only.

## Prediction

Written before any run. Numbers are guesses, not results.

Falsifiable, at step 2:
1. The annealed arm, table removed, beats no table by the decision rule, and retains 30 to 60% of the kept-table gain.
2. The learned table, annealed, beats the content-free table, annealed, by the decision rule.

Sanity expectations, not tests: kept beats none; removing the table at the end with no further training is worse than annealing.

Decision rule: the paired mean per-token validation-loss difference exceeds 0.005 nats, its 95% bootstrap interval excludes zero, and it exceeds twice the no-table seed standard deviation (crude with 3 seeds; the write-up says so). Anything less is inconclusive.

## Design

Data (steps 1b onward): FineWeb 10B in GPT-2 tokens, modded-nanogpt's shards (Hugging Face `kjj0/fineweb10B-gpt2`; repo id and shard size to confirm on download). GPT-2 vocabulary keeps speedrun comparability; the output head then dominates compute in small models.

Model: nanoGPT-style decoder on MPS, tied embeddings, RoPE, context 512. Rows indexed by a hash of (previous, current token) are added to the residual stream before layer 0, times a gate g(t). Table and backbone use separate random generators; same seed means same init and data order, so per-token differences are paired.

Sizes, estimated from layers and width, not measured:

| Step | Backbone | Non-embedding | Embedding (tied) | Table |
|---|---|---|---|---|
| 1b | 4 layers, width 256 | about 3.1M | about 12.9M | 131,072 × 256, about 34M |
| 2 | 6 layers, width 384 | about 10.6M | about 19.3M | 131,072 × 384, about 50M |

Table rows per training token is reported as a covariate in every step.

Schedule, identical for every arm: short warmup, constant learning rate to 80% of steps, linear cooldown to the end. The gate is 1 until 50%, fades linearly to 0 by 70%, and stays 0.

Branches: the kept-table run is checkpointed at 50%, then branches into kept and annealed, which share everything before the branch. "Removed at the end, no further training" needs no run: it is the kept arm evaluated at gate 0.

Content-free table (the control): same rows, width, optimizer, gate and fade, but indexed by a hash of (training step, position), redrawn every step, so it carries nothing about the tokens. Step 3's frozen random table indexed by the bigram hash still delivers bigram identity, so it is not this control.

Deciding metric: final per-token validation loss with gate 0, on a fixed 2M-token slice of the validation shard, paired by seed against no table.

Secondary, not part of the falsification: mean loss gain per token in three buckets (trigram ending at the target seen in tokens consumed; unseen; bigram copyable from earlier in the window), with token shares. The seen set is a sorted uint64 array on disk, queried in chunks, under 8 GiB at 300M tokens.

Runs proceed round-robin across arms and seeds in chunks under 300 s, checkpointing between chunks.

**Step 1a: the lab's case 001 (added 2026-11-09).** A 3.2M-parameter character transformer on tinyshakespeare on MPS, scored by the lab as charged seconds to validation loss 1.70 (about 32 s baseline; noise about 2% for same-trajectory and 5% for trajectory-changing comparisons). With 65 characters no hash is needed: an exact bigram table of 4,225 rows, learned from zero, added before layer 0 behind the gate. Arms: no table; kept (table on); annealed, the fade ending before the baseline's time to target and validation read at gate 0, so the clock stops only once the table is gone; content-free annealed; gate held at 0. Fading changes the trajectory, so the 5% floor is the bar. A learned table is not a candidate-owned input. A count-initialised variant passes the lab's causality checks if counted only from characters already consumed, in order; counted from text not yet reached, it must be declared as a candidate-owned input in this card and the lab contract. The lab's planned split of validation loss into lookup-predictable and novel characters, computed outside the clock, stands in for the trigram buckets. The table is too small to test the serving argument, but the step reads the sign of catalyst against capacity cheaply.

Mapping onto the lab's candidate interface (prepare, advance, predict; the clock charges preparation and learning; the contract owns the validation split), fixed 2026-11-09 after the lab's showrunner described it:

- The fade schedule is counted in updates, not seconds, so a faster arm does not fade sooner. It ends at 60% of the no-table baseline's median update count to loss 1.70, measured in the lab's existing baseline receipts: 536 over contract version 2 arms, so the fade ends at update 322. It starts at update 230, keeping the 50-to-70 shape of the main design. The gate is 1 before 230, falls linearly to 0 at 322, and stays 0.
- `predict` for the annealed and content-free annealed arms always runs with the gate at 0, from the first update. The contract's clock therefore stops only when the table-free model reaches 1.70, whatever the gate is during learning. The kept arm predicts with the gate as trained. The gate-held-at-0 arm is the no-table model plus unused parameters, a check that the extra parameters alone change nothing.
- A learned table starts at zero in `prepare`, which costs nothing worth charging. A count-initialised variant is built in `prepare` from the training split only, in data order, and its counting time is charged. It is not run in step 1a.
- Step 1a prediction, entered in the lab's plan ledger before any run, in the lab's own outcome (charged seconds to 1.70, as a ratio to the no-table baseline): kept, below 0.95. Annealed, between the kept ratio and 1.0, recovering 30 to 60% of the kept arm's saving, the same retained share claim 1 makes at step 2. Content-free annealed, within the 5% floor of 1.0. Gate held at 0, within the 5% floor of 1.0. These numbers are the card's guesses carried over from step 2, not measurements. If kept does not beat the 5% floor, the table does not help at this scale, and step 1a reports that and stops.
- Step 1a does not need the lookup-predictable and novel loss split. The split would only explain where a gain comes from.
- Case 001 keeps a constant learning rate and never cools down. So "the fade ends before cooldown" always holds, and step 1a cannot test how a fade interacts with cooldown; that stays with step 2.
- The gate-held-at-0 arm must reproduce the no-table baseline bit for bit, since training on case 001 is deterministic per seed. That replaces the 5% floor for this arm. The table must be initialised outside the baseline's initialisation loop, with its own random stream, or it would shift every later weight.
- The annealed prediction (a ratio near 0.91 to 0.955) mostly falls where the lab's verdict reads inconclusive. An inconclusive annealed arm at step 1a is not a no: step 2's kill rule does not apply to step 1a, and the step reads only the sign.

**Step 1b: FineWeb harness (one slice: 30 GPU-minutes, processes under 300 s, under 8 GiB).** Does the table help at GPT-2 vocabulary, and how large are seed spread and MPS noise? No retained share. A 60-second probe sets the per-run budget (estimated 2 to 6M tokens, not measured). Ten runs of about 160 s: no table, seeds 1 to 3; no table, seed 1 again; gate held at 0, seed 1; kept, seeds 1 to 3; planted handicap, seeds 1 and 2. If kept does not beat none, step 2 raises tokens before any removal claim. Step 1b reports kept minus none, paired over seeds 1 to 3, as an input to step 2's token budget, not as a bar: nothing shows that a gain at 2 to 6M tokens predicts the gain at 300M.

**Step 2: the deciding run.** Planning figure 300M tokens, revised from step 1. Arms, 3 seeds: no table; kept; annealed (branch); content-free annealed; about 3.5 full runs per seed. Estimated 4 to 10 hours per run, so 2 to 5 days of Mac time; needs Rob's approval, recomputed from step 1 before asking.

**Subject gate (step 2, read from the baseline run, first).** The removal test needs a kept-table gain large enough that a 30% retained share can clear the decision rule. The no-table and kept arms, all 3 seeds, run first; they are needed anyway, and the branches fork from kept's 50% checkpoint. Bar: kept minus none at the end of those runs, paired over seeds 1 to 3 with the table on, is at least T/0.3, where T is the decision threshold (the larger of 0.005 nats and twice the no-table seed standard deviation); with T at 0.005 that is about 0.017 nats, and the 95% bootstrap interval excludes zero. If kept falls short, stop before the annealed and content-free branches and report park, not kill: an annealed arm read against a gain that small would be inconclusive by construction.

**Step 3.** Identity-without-learned-content arm; abrupt cut at 50% with training continued; a fade over the last 10% before cooldown; a two-point learning-rate check; two more backbone widths for key arms; RoPE dropped late as an exploratory arm; a MeCo-style arm, gate 1 until cooldown starts at 80% and then cut to 0, MeCo's working schedule, to read the annealed arm against. Same approval as step 2.

**Step 4 (optional).** One confirming pair at about 124M on Modal, Kaggle or Colab. Needs Rob's login and approval; may cost money.

## Checks before believing anything

Rerun floor: the same-seed no-table rerun measures MPS nondeterminism (embedding backward and scatter-add may not be deterministic on MPS; not checked).

Self-comparison: the gate-0 arm against seed-1 no table must stay within the rerun floor at every logged step. This tests RNG separation and the hook.

Planted effect, training level: the planted arm trains with no table on fewer tokens, chosen from the no-table loss curve for an expected gap of about 0.01 nats. The paired analysis must recover it with the right sign within a factor of two; if it calls it inconclusive, step 2 needs more seeds or tokens.

Planted effect, evaluation level (unit test): on one checkpoint, set p' = (1 − ε)p + ε·onehot(target) on unseen-bucket tokens only, for a mean change of 0.005 to 0.02 nats. The harness must report the closed-form change within 1% there and zero elsewhere.

## The control that could embarrass it

The content-free table, annealed. If it retains as much as the learned table, the gain comes from a side pathway and extra trainable parameters during training, not from what the table carries.

## Confounds a hostile reader would raise

- *The table adds parameters.* The metric reads the model with the table removed, at equal backbone and steps.
- *Learning rate favours one arm.* Step 2 uses one rate tuned for neither; step 3 checks two.
- *The fade lands in cooldown.* It ends at 70%; cooldown starts at 80%.
- *Small-model artefact.* Step 3 tests only the sign of the trend across widths.

## Kill criterion and gates

Before step 2: the rerun floor, self-comparison and both planted checks pass. Before step 2's removal arms: the subject gate passes on step 2's own no-table and kept arms; a failure there parks, it does not kill. The search gate passed on 2026-11-09.

Kill if, at step 2 with 3 seeds, the annealed arm is inconclusive against no table, or the content-free table retains as much as the learned one. Park if the kept table never beats no table at an affordable budget. A failed check means fix the harness, not kill.

## Cost on this hardware

- Step 1a: a few minutes per seed across five arms at about 32 s each. Step 1b: 30 GPU-minutes, about 45 minutes wall, about 400 MB download, under 3 GB of checkpoints.
- Step 2: 2 to 5 days of Mac time (estimate); three more shards; seen set up to about 2.4 GB on disk.
- Steps 3 and 4: recomputed later.

## What a yes or a no changes

A yes makes lookup tables training accelerators with nothing to serve. A no means the speedrun's table gain is capacity and should count as parameters. A content-free win means the pathway, not its content, does the work.

## Why it should carry to larger models, and what it leaves untested

It may not. Engram's 27B results concern a model that keeps its table; DroPE's concern positions. The sweep tests only the sign of the trend, on token budgets thousands of times smaller than real runs, with a table about 650 times smaller than the speedrun's. The mechanism step 2 can reject is "the table's content transfers into the backbone". Larger tables, longer training, other n-gram orders and downstream tasks stay untested.

## Receipts to keep

The slice file with this prediction, committed before the run under `docs/research/program/slices/`; training script, configs, seeds, commit hash and per-step loss CSVs; rerun-floor, self-comparison and planted-effect outputs; per-token validation losses, bucket counts and the seen-set build script; measured tokens per second, peak memory and rows per training token; final backbone weights with checksums.

## Search gate, 2026-11-09

First pass: six queries refused. Direct reads (record table, PR 360, Engram's and DroPE's citers by title, Engram section 6.3) found Token-Superposition, COUNT, Low-Bit N-Gram and [Frozen Memory Is Not Enough](https://arxiv.org/abs/2608.17050) (2026-08, abstract: content and addressing both matter).

Second pass: 12 queries run, 0 refused. Gate topics plus other communities' wording: metadata conditioning, auxiliary inputs dropped in pretraining, n-gram teachers decayed away, random against learned tables.

New closest works, summarised above: MeCo (2025-01, method and ablations read), annealable soft-prior transformers (2026-09, abstract), PR 201 (2026-01, read). Also seen, tables kept: Lngram (2605.24869), Gengram (2601.22203), MoME (2609.15126), abstracts.

Verdict: no scoop. MeCo supplies a removal schedule to compare against.

Still open: the items under "Not checked". None blocks step 1a.

Design changes: step 1a (first pass) moves the first signal to case 001; step 3 (this pass) adds a MeCo-style cut at cooldown start, and its fade-against-cut contrast now has conflicting precedents (Token-Superposition, soft-prior study).

## Review record

Hostile review, all must-fixes adopted: harness slice first; content-free control, random bigram table to step 3; buckets out of the falsification; rerun floor, 3 seeds, paired bootstrap; 0.01-nat plant; fade ends before cooldown; step 2 cost stated. Should-fixes adopted: hint half split off, RoPE drop exploratory, sizes split by embedding, forked branches, the scoop gate (cleared 2026-11-09). FineWeb GPT-2 kept over TinyStories for speedrun comparability. The reviewer's six searches were also refused.
- 2026-11-09 night: step 1a mapped onto the lab's prepare/advance/predict interface (fade counted in updates; table-free predict for annealed arms; count-init charged in prepare; step 1a prediction stated in the lab's outcome).
- 2026-11-09 night: fade fixed at updates 230 to 322 from the lab's version 2 baseline receipts (536 median updates to 1.70); gate-0 arm must be bit-identical to baseline; step 1a inconclusive is not a no. From the lab showrunner's fit check for Q002.
- 2026-11-10: base-property check after the lab's case 002 ran on a model that got no greedy answer right (one template inaccurate, the other off-format). Step 1a already stops if kept misses the 5% floor; the park rule had no number. Added a room gate: kept-minus-none at least the decision threshold over 0.3 (about 0.017 nats), read at step 1b and again on step 2's no-table and kept arms before the removal branches run; failure parks. Whether the gain at 2 to 6M tokens predicts the gain at 300M is not checked.
- 2026-11-10, cross-family gate review (Sol): dropped the step 1b T/0.3 veto on starting step 2, since nothing shows short-budget gains predict 300M gains; step 1b's gain now only sizes step 2's budget. The bar stays as the subject gate, read from step 2's no-table and kept arms first, before the removal branches.
- 2026-11-10, lab framing of step 1a as Q002 (accepted): the annealed arms run only if the kept arm's confirmed median ratio is at or below 0.83, since the lab's threshold confirms only about 9 to 10 percent trajectory-changing gains; otherwise step 1a parks as subject present, too little room. Step 1a reads only the sign and cannot name catalyst, capacity or pathway, because the content-free arm's per-step redraw confounds noise with content; step 3's identity arm is where that is separated.

## Result of step 1a, 2026-11-10 (lab Q002, case 001)

Stopped by the card's own rule: no demonstrated kept-table gain at this power. The kept-table arm's probe read 0.925 of baseline; its confirmation was inconclusive, with paired ratios 0.843, 0.880, 0.994, 1.019 and 0.948 (median 0.948). Per seed, the table saved 0 to 18 percent of updates and cost 1 to 3 percent per update on MPS. The annealed, content-free and gate-held-at-0 arms were not run on the device. 506 guarded seconds. After Astra's skeptic review the lab adopted this statement: on the four-layer, width-256 character transformer on Tiny Shakespeare on MPS, the kept 4,225-row table had a median charged-time ratio of 0.9478 against the baseline; five paired ratios ranged from 0.8427 to 1.0194, four below one; under the frozen threshold-v2 rule the result is inconclusive, and the confirmation missed the 0.83 room bar, so step 1a parks. It establishes neither a table-free benefit nor a catalyst or capacity explanation. Caveat: this programme's kernel subagent took the shared GPU lock before blocks 3, 4 and 5; block 4 cleared its threshold by 0.569 s, so contention could have changed whether all five ratios fell below one. The observed median saving was 5.2 percent, with two pairs above 11 percent. Reopening needs a new design sized before any run. The lab's outcome record is the source for this reading.
