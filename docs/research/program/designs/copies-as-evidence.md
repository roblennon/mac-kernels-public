# Copies in context counted as evidence: does a replayed error hurt fresh answers more than a summary of it?

Status: queued, 2026-11-09. Designed, not run; no code exists. Search gate passed; revised after a hostile review and five cold-planning reviews. The loop-share question moves to card B, not yet written.

Sinha et al. showed that errors visible in earlier turns lower later accuracy, and harnesses usually show history verbatim. If a model treats each copy of an error as fresh evidence that it errs, a replayed error should hurt more than one error plus a note saying how often it appeared, and collapsing repeats in agent history is a cheap fix.

## Hypothesis

In Qwen3-1.7B, on a key-value-sum task after Sinha, an error turn that appears 4 times (once, then 3 verbatim replays) lowers log P(correct answer) on a fresh item by at least 0.3 nats more than the same error turn shown once with a note saying it appeared 4 times, at matched token length: C − R ≥ 0.3, each arm name standing for its mean log P(correct). "Step 1, exactly > Decision rules" confirms, kills or parks it.

## Theory

Softmax attention sums over positions, so four copies of an error turn draw about four times the attention of one unless a head de-duplicates. The rival account is pattern continuation (Xu et al. 2022): repeated text makes itself likelier to repeat. That predicts a lift for the copied string and no extra harm to a fresh item, so scoring only fresh items separates the two. Hu et al. (2026) show repetition effects saturate, so the claim covers small counts only.

Assumptions: the count note is attended (the probe checks only retrieval); teacher-forced scores move with greedy answers (recorded, not gated). Thinking, which Sinha found mitigates self-conditioning, is an untested moderator.

## What is already known

- [Sinha et al. (ICLR 2026)](https://arxiv.org/abs/2509.09677): injected errors in prior turns of a key-value-sum task lower later accuracy, with a dose-response, across Qwen3 4B to 32B and Gemma3. Scale does not fix it; thinking mitigates it. No replayed-versus-collapsed contrast in the part read.
- [Xu et al. (NeurIPS 2022)](https://arxiv.org/abs/2206.02369), abstract: the more a sentence repeats in context, the likelier it is generated again.
- [Hu et al. (ICML 2026 poster)](https://icml.cc/media/icml-2026/Slides/65762.pdf): one assertion repeated up to 1000 times on six models including Qwen3-4B; the logit gap converges. They score the asserted answer, Xu's regime.
- [arXiv 2601.06189](https://arxiv.org/abs/2601.06189): paraphrased copies of one document flip retrieval-QA answers about 6 to 12 points more often than distinct documents. No verbatim or count arm.
- [Chen et al., DeLoop (arXiv 2610.19376)](https://arxiv.org/abs/2610.19376), first 100,000 characters: agent loops rise after history compression. It and OpenClaw and pydantic-deep-agents block or warn on loops; nothing found collapses repeats in history.

## Prediction (written before any run)

Slice 1a, Qwen3-1.7B, log P(correct answer) at the scored turn. A difference "X − Y" is the paired mean over scenarios; positive means Y is worse. Size that matters: 0.3 nats per comparison.

- Copy-counting predicts C − R ≥ 0.3, |R − D| < 0.3 (copies weigh like distinct errors), |C − S| < 0.15 (the note does little by itself) and |N − Q| < 0.15 (a replayed correct turn does no harm).
- Pattern continuation predicts R, C and S within 0.15 nats of each other on the fresh item, with log P(E's wrong number) highest in R.
- Role swap (secondary): if C − RS is under half of C − R, a self-recognition reading survives, and the harness fix changes from de-duplicating to relabelling.

## Design

**Task.** After Sinha: 50 keys with values sit in the system prompt, and each independent turn asks for the sum of 3 keys. The history has 20 turn slots (a user and an assistant message each), then the scored turn, whose answer appears nowhere in history. Two templates: T1 natural language, T2 terse.

**Arms.** Each scenario draws 4 special slots P1 < P2 < P3 < P4; the other 16 hold the same correct turns in every arm. E is the error turn; K is E's question with the true sum, so N − S and Q − R differ only in answer digits. K2 to K4 are further correct turns; E2 to E4 are their questions with wrong sums.

| Arm | P1 | P2 to P4 | Role |
|---|---|---|---|
| N | K | filler | comparator for Q |
| S | E | filler | one error, no note |
| C | E with the count note | filler, shortened by the note's length | collapsed |
| R | E | E, E, E | replayed |
| D | E | E2, E3, E4 | four distinct errors |
| D0 | K | K2, K3, K4 | four distinct correct turns, gate comparator |
| Q | K | K, K, K | replayed correct turn |
| S′ | as S, with the 16 correct turns reshuffled and the second filler text | | wording check |
| RS | as R, with the role-swap system sentence | | secondary |

D0, not N, is the gate comparator, since D0 − D differs only in four answers. The count note sits on E's own assistant message, where a de-duplicating harness would write it; a user-role note is a later arm. Paraphrased replay (P) waits for step 2.

**The one metric that decides:** C − R in nats of log P(correct), paired by scenario, with a paired percentile bootstrap 95% interval over scenarios, stratified by template when pooled.

**Toy side-study (slice 1b, cannot rescue the claim).** Does a small transformer trained on episodes with an error rate θ and verbatim replays treat copies as evidence about θ when Bayes does not? Appendix A.

## Step 1, exactly

A runner reads only this section and its appendices; where another section differs, this one wins. "Cold read" numbers were measured by a planning reviewer and are asserted again in the CPU tests.

**Inputs.** `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (current main, checked 2026-11-09) and its tokenizer. The repo is 4,079,450,110 bytes (cold read) and not in the local Hugging Face cache (2026-11-09), so it is downloaded before the run, outside the GPU lock. The `.venv` has torch 2.14.1, transformers 5.18.0 and numpy 2.5.3 (checked 2026-11-09). Code goes in `docs/research/program/slices/copies-as-evidence/`: `gen.py` (keys, scenarios, rendering), `score.py` (forwards, prefix sharing, spans, branches), `run_1a.py --shard k`, `checks.py` and `toy/`.

**Values.** Each key's value is drawn uniformly from 37 to 99, with replacement, so true sums lie in 111 to 297. A wrong sum is the true sum plus an offset drawn uniformly from the 20 values ±1 to ±10 (101 to 307) and equals no other answer in the scenario. The scored answer differs from every number shown in history. Any violated rule redraws only the failing draw, continuing the same stream.

**Keys.** Random 4-letter lowercase strings. Qwen's pre-tokenizer glues a preceding "+" or "(" onto letters (cold read: into about 35% and 88% of keys in T2 contexts). A key qualifies if, in every context, its 4 letters are covered by exactly 3 tokens holding no other character except one leading space. Contexts: the dictionary line and each key position in each template's question and probe, measured by tokenizing the whole rendered message, so probes name keys with the same tokens as history. `gen.py` enumerates all 456,976 strings once on CPU and saves the list with its SHA-256 (keys joined by newlines); 0.697% qualify, 3,183 keys, as measured by the lab's build (cold reads had estimated 0.75 to 0.8%; the binding context is T2's probe position after "count("). Each dictionary is 50 distinct keys from that list.

**Key triples.** No key repeats within a turn; keys may repeat across turns. Every question triple in a scenario is unique as a set, apart from E's replays and K. E's keys appear in no history turn except E's replays and K; the scored turn's keys appear in no history turn. Reason: E's question then appears once in S and C and 4 times in R, so a "4" from C can only come from the note.

**Positions.** P1 to P4 are one of the 715 sets of 4 slots from 2 to 17 with no two adjacent, enumerated in lexicographic order and drawn uniformly, shared across a scenario's arms. The last 3 history slots are always shared correct turns.

**Rendering.** Wording is in Appendix B. The chat template runs with `enable_thinking=False`; the generation prompt ends in an empty think block and history assistant turns have none (cold read). Sequences are joined from token-id segments only at message boundaries, each message tokenized whole. E plus the count note is one assistant message (in T1 E's final "." and the note's newline merge into one token), so every arm carries only canonical tokenizations. The note's length L is tokens(E plus note) − tokens(E) (cold read: 14 in T1, 15 in T2).

**Length.** All arms have the same token count up to the scored turn (asserted; a failing scenario is redrawn). A filler turn is "Please continue." plus a neutral assistant text cut, at a word boundary or else a token, to a question turn's length (cold read: 22 assistant tokens in T1, 12 in T2). In C each of the three fillers loses floor(L/3) tokens and the last also the remainder (T1: 18, 18, 16; T2: 7, 7, 7); that they read differently is accepted. The system prompt's last sentence is the role-swap sentence in RS (15 tokens, cold read) and elsewhere the neutral sentence cut mid-clause to the same count.

**Scored span.** The answer prefix is forced (T1: "The sum is" and the separate space token; T2: nothing). Log P(correct) is the sum of log-probabilities of the 3 digit tokens, one per digit (cold read), terminator excluded.

**Count probe.** A branch on C, S and R replaces the scored turn with the probe (Appendix B) and scores the digits 1 to 5 after its forced prefix. The choice is the argmax among those five; the full-vocabulary rank of "4" is logged. C is "available" in a template if it chooses 4 in at least 2/3 of that template's C scenarios and in at most 1/3 of its S scenarios.

**Greedy decode.** 6 tokens after the forced prefix, argmax on the fp32 head so decode and scorer read the same logits. The record is the first 3-digit number.

**Precision.** bf16 weights, thinking off, batch size 1, no padding. At scored positions the final hidden state is cast to fp32 and the head and log-softmax run in fp32 against an fp32 copy of the tied head on the device (151,936 × 2,048 × 4 bytes, about 1.24 GB), because bf16 logits of size 16 to 32 sit on a 0.125 grid. No fp16. The M1 Max has no native bf16 arithmetic, so speed is measured.

**Seeds.** Every stream is `numpy.random.default_rng([20261110, tag, *indices, 9])`; the final 9 stops collisions, because SeedSequence ignores trailing zeros. Tags: 1 scenario (j, k), drawn in this order: dictionary keys, values, E's and the scored triples, other triples, positions, wrong sums, redraws, S′'s shuffle; 2 execution order (j, k), a permutation of the 9 arms plus R+plant, which shards without it skip; 3 bootstrap (j); 4 toy map f; 5 toy training (layers, s); 6 toy eval. j is 0 for T1 and 1 for T2; k indexes scenarios within a template. Step 2 uses base 20261111.

**CPU unit tests first.** Value, wrong-sum, key and triple rules; equal lengths across arms; segment assembly equal to `apply_chat_template` for scenarios whose fillers are cut at word boundaries; the cold-read numbers above; the span finder on both templates; distinct seed tuples; the toy's Bayes, self-comparison and plant checks (Appendix A).

**Memory and the runner.** Every device process sets `HF_HUB_OFFLINE=1`, calls `torch.mps.set_per_process_memory_fraction(8 GiB / torch.mps.recommended_max_memory())` and logs both numbers (cold read: 24.96 GiB, so 0.3205), and runs as `python3 bench/gpu_runner.py --monitor --seconds 300 --rss-gib 8 -- .venv/bin/python <script>` (`bench/with_gpu.sh` cannot pass these bounds; this takes the same lock). The MPS fraction is the cap; `--rss-gib` is a backstop.

Load with `from_pretrained(revision=..., dtype=torch.bfloat16, device_map="mps")`, straight to the device, not via CPU and `.to("mps")`, which holds the mmap'd file and a CPU copy at once. transformers 5.18 refuses any `device_map`, "mps" included, unless `accelerate` is installed (the lab's builder found this in `integrations/accelerate.py`; an earlier reading of this card was wrong), so case 002's isolated environment carries `accelerate` 1.15.0 for this load path. Then build the fp32 head from `model.lm_head.weight`, drop CPU references and run gc.

Memory cap: Rob raised the per-process cap for case 002 to 10 GiB on 2026-11-10, after the first pilot reached 8 GiB in the library reference's fp32 upcast (lab slice 04-case-002-cap10). Budget (estimates, against the original 8 GiB = 8.59 GB): 3.44 GB of bf16 weights (the file's 4.06 GB counts the tied head twice), the 1.24 GB head, the Python and PyTorch runtime (roughly 0.5 to 1 GB), and about 130 MB of KV cache per arm (about 1.3 GB for one scenario's 10), so about 7 GB at steady state. The library reference peaks near 1.6 GB on top of weights and head: at about 1,050 tokens its bf16 logits are 0.32 GB, and the loss's fp32 upcast and log-softmax add about 0.64 GB each. So it runs before any arm cache exists; then its outputs are deleted, gc runs and `torch.mps.empty_cache()` releases the allocator's blocks. Only the current scenario's caches are kept. The backstop may also count file pages resident during the load, so headroom is smaller than this sum. The pilot logs RSS after load, after the reference and after `empty_cache()`, plus `peak_group_rss_gib` and `torch.mps.driver_allocated_memory()`. If the backstop or the cap trips, stop and ask Rob rather than raise it.

**Forward and branches.** Arms share slots 1 to P1 − 1 and often more. Each arm's tail runs, in the stream-2 order, from a clone of the longest cached prefix it shares with an arm already run, cut back with `DynamicCache.crop` (present in 5.18). Every arm is cached to the end of history; branches fork there, since the probe replaces the scored turn. Each branch adds its user message, the assistant header with the empty think block, and the forced prefix: the scored answer; E's wrong number in the scored position (every arm); the probe (C, S, R); the greedy decode. Logits only at scored positions.

**Shards.** Scenarios run in the order (k = 0, T1), (k = 0, T2), (k = 1, T1), and so on.
1. Pilot, one process: k = 0 and 1 of each template (rerun by shard 1; pilot results are not among the 120), all arms and branches, R+plant at a provisional b = 1.0 (timing does not depend on b), S scored once without cache reuse, the library reference. Records load time (guard start to model ready), seconds per arm and per branch, the 90th-percentile seconds per scenario of main-shard work with the first scenario's warm-up included (p90) and of R+plant alone (p90_plant), memory, decoded spans, and the unbiased R logits that set the final b.
2. Go or stop, on CPU. Shard 1 holds floor((300 − load − 30) / (p90 + p90_plant)) scenarios; later main shards hold floor((300 − load − 30) / p90), each planned at 300 s; the determinism shard is planned at load + 4 × p90 + 30 s; the pilot counts at its measured runner wall time. Go if pilot + main shards + determinism ≤ 1,800 s for 120 scenarios and shard 1 holds at least 4. Otherwise drop, recomputing p90 from the logged per-arm and per-branch times after each: RS; the probe on R; the greedy decode except on S, C and R. Never dropped: the other arms, the probe on C and S, the checks, 60 scenarios per template. If it still does not fit, stop and ask Rob for the minutes the sum needs. A cold read's unmeasured estimate (p90 about 10 to 15 s) puts the run over 1,800 s even without RS, so this stop is likely and designed.
3. Main shards append JSONL after each scenario and resume from the last complete one; shard 1 carries R+plant.
4. Determinism shard: a new process re-scores k = 0 and 1 of each template.

**Budget.** 30 guarded GPU-minutes of runner wall time for slice 1a, pilot and checks inside. Slice 1b has its own 30 (Appendix A).

**Checks, run and read first.** A failing check stops the run; the cause is fixed and Rob is asked before anything is read. Path noise on the answer digits is reported, not a stop (see Identity).
- Library reference (pilot, S and R): call the model with `labels` masked to -100 except at the scored and forced positions, and compare its per-token log-probabilities with the scorer's. It is a bug check: an off-by-one, a wrong row or a wrong position ruins tokens the model is confident about, while bf16 rounding barely moves them (the change in log p is about (1 − p) times the logit error). Pass: at every forced or scored position where the reference gives the target probability at least 0.9, the two agree within 0.05 nats, with at least 3 such positions per template across the pilot, counted as in the identity check below. The floor is 0.9 here, not 0.5 as in the identity check, because the reference computes its logits in bf16 while the scorer uses an fp32 head. With bf16 logit error of about 0.125 to 0.2, rounding moves log p by about (1 − p) times that, up to about 0.1 nats near p = 0.5 but at most about 0.02 at p of 0.9 or more, while a crop or index bug moves it by many nats (17 in the lab's planted test). Raised from 0.5 on 2026-11-10 after the second pilot, before any main shard: the pilot's three failures were single positions at p between 0.51 and 0.58 with gaps of 0.085 to 0.099 nats, and every position above 0.9 agreed within about 0.0005. The change is to a bug check, not to the decisive metric or its decision rules. The summed difference on the 3 answer digits is reported per scenario, with its mean and spread, and does not stop the run.
- Identity (pilot): S without cache reuse against S from the cache path, per-token differences logged. Same bug-check rule as the library reference: every confident position (uncached target probability at least 0.5) within 0.05 nats; a failure stops the run. Forced positions include the generation prompt (assistant header and the empty think block) as well as the answer prefix: they sit after the cache branch point, so they exercise the crop and position code under test. T2's answer prefix is empty, so T2 relies on the think-block tokens. The minimum of 3 confident positions applies per template across its pilot scenarios, not per scenario, because a crop, index or position bug sits in shared code and shows in every scenario of a template. A scenario with fewer than 3 confident positions is counted as unchecked and reported; the run stops only if a confident position fails, or if a template's pilot scenarios hold fewer than 3 confident positions in total. The summed difference on the 3 answer digits is bf16 path noise, not a bug: the full prefill and the cached tail use different matmul shapes, and the card's own rounding bound allows about 0.125 nats per token per path, so about 0.5 per scenario across two paths. On CPU the lab's builder measured 0.54 on one scenario, almost all on one low-probability digit. It is reported per scenario, as a mean and a spread, beside every result. Every arm is scored through the same cache path, so this noise enters each arm in the same way and the paired bootstrap over scenarios carries it. Changed 2026-11-10, before any scoring run, because the earlier 0.1-nat stop sat below the card's own rounding bound.
- Plant (shard 1): R+plant subtracts a fixed b > 0 from the correct digit's logit at each answer position, as a hook on the fp32 head output, leaving the KV cache untouched (a positive bias would cap the shift at −log P(correct)). An independent locator finds the positions by matching the answer's token ids at the end of the sequence. After the pilot, b is bisected on its R logits to a mean expected shift of 0.5 ± 0.01 nats and frozen to 4 decimals; each scenario's expected shift comes from its unbiased fp32 logits. Pass: R − (R+plant) within 0.05 nats of the expected mean shift. This checks only the readout.
- Span: decoded scored tokens are the 3 answer digits for k = 0 to 9 of each template, decoded on CPU from the main shards' JSONL and rendered ids.
- Determinism: re-scored scenarios match to the bit; otherwise the largest difference is the floor, reported with every result.
- Wording (S′ against S), pooled like the decisive metric, per-template values recorded: pass if pooled |S − S′| < 0.1 nats. Whether its interval covers zero is reported, not gated, since S′ changes wording and order. The wording spread, the baseline's own spread in the frame's sense, is the larger absolute end of the pooled interval.
- Greedy record: per-arm exact-match accuracy, and whether its arm order agrees with log P. Not a gate.

**Bootstrap.** Paired percentile, 10,000 resamples drawn within template from stream 3 (j); one set of resample indices per template serves every contrast (D0 − D, C − R, S − S′, S − R, N − Q and the rest). A pooled value is the mean of the two template means.

**Decision rules, in this order.**
1. Probe: C is void in any template where it is not available. If void in both, step 1 decides nothing and S − R is reported as "collapse without a count".
2. Gate: pooled D0 − D ≥ 0.3 with its interval above zero. Otherwise the verdict is "parked until 4B" and the rest is descriptive, since errors were not shown to hurt this model.
3. Decisive: pooled C − R over templates where C is available, with its interval. When C is available in only one template, the pooled value and its interval are that template's own C − R and bootstrap.
   - Supported: at least 0.3, interval above zero, and the point estimate above the wording spread. Reason: the spread is already an interval's far end and C − R's own noise is covered by its interval clearing zero, so comparing its lower end would count noise twice. "Supported, unconfirmed until step 2"; "holds in both" only if each template's C − R is at least 0.3. If one template falls short or C was void in one: "template-dependent, unconfirmed".
   - Killed: the interval's upper end is below 0.3, excluding an effect of the size that matters. The toy cannot save it.
   - Inconclusive, in every other case: a point below 0.3 whose interval reaches 0.3, an interval touching zero, or a C − R point estimate inside the wording spread. The frame calls a difference inside the baseline's spread inconclusive, not a kill. Step 2 then reruns with more scenarios, on Rob's minutes. Step 2 does not pool with step 1a. It is a fresh test on new scenarios (seed base 20261111), with a third template and the paraphrase arm, judged on its own by these same rules. Pooling after an inconclusive look would let the result decide when to stop, which inflates false positives, and step 2's design differs from 1a's in any case. Step 1a's numbers are reported beside step 2's, never combined.
4. Correct replay, read only if S − R has its interval above zero: if N − Q does too and is at least half of S − R (a default), the claim becomes "repetition disrupts" and the fix is retested on that footing.
5. Role swap, deciding nothing: C − RS against C − R, as under "Prediction".

**Outputs.** Before the run, a dated plan in the slice folder. Then: code with git hash, seeds, the key-list hash, rendered templates, pilot timings and memory, per-scenario per-arm JSONL (log P(correct), log P(E's wrong number), probe log-probabilities and rank, greedy output, token counts), check outputs, the bootstrap table, toy receipts. Last, a results note naming the mechanism this could reject and the claims left untested.

## Controls and confounds

If R sits near C while D falls, the model de-duplicates. Q against N guards against "any replay hurts". The note may read as a warning; C − S bounds that. Teacher forcing is not sampling; the greedy record is a weak check, and card B samples.

## Cost on this hardware

Slice 1a: 30 guarded GPU-minutes, which may not hold the full design (see go or stop); about 7 GB of device memory against an 8 GiB cap (see "Memory and the runner"); a 4.08 GB download, with 33 GiB disk free on 2026-11-09. Slice 1b: 30 guarded GPU-minutes, under 200 MB of outputs. Step 3: Qwen3-4B needs about 8 GB of bf16 weights, so a raised cap or cloud, with Rob's approval.

## Steps

1. Slice 1a (decisive), then slice 1b (toy). See "Step 1, exactly".
2. Recurrence: rerun slice 1a on new scenarios (seed base 20261111) with a third template, and add the paraphrased-replay arm P (Appendix B). Until then, any slice 1a result is an unconfirmed observation.
3. Scale: `Qwen/Qwen3-4B`, thinking on and off, the model Sinha tested (not Qwen3.5-4B, a different multimodal architecture).

## What a yes changes and what a no changes

A yes says verbatim history costs accuracy beyond the facts it carries, so harnesses could collapse repeated failures into one copy plus a count; for reasoning-model harnesses it waits on step 3. A no says history format does not matter for this effect at 1.7B; larger models, thinking, real agent tasks and mixture-of-experts stay untested. An inconclusive result says neither.

## Search gate, 2026-11-09

17 WebSearch calls on self-conditioning, duplicate-token heads, agent memory, double-counted evidence, observation masking and history editing. Also read, abstract or summary only: [Falck et al. (ICML 2024)](https://arxiv.org/abs/2406.00793), nothing on duplicates; [Wang et al., More Memory, Worse Agents (arXiv 2610.27406)](https://arxiv.org/abs/2610.27406); [Bouleimen, Pagan, Hannák on illusory truth in LLM agents](https://chatpaper.com/chatpaper/paper/353020) (no arXiv ID found); [Lindenbauer et al., The Complexity Trap](https://arxiv.org/abs/2508.21433); [OpenClaw loop detection](https://docs.openclaw.ai/tools/loop-detection). Still open: DeLoop's appendices F and J and the full text of 2610.27406 could hold a history-collapse ablation.

## Review record

**Hostile review, 2026-11-09.** Queue after fixes: metric on fresh items, D, R and Q compete, one decisive contrast, plant, span and self-comparison checks, loop share to card B, search as a gate.

**Cold-planning reviews, rounds 1 to 4, 2026-11-09.** Toy specified; arms rebuilt on 4 shared slots; length, note, span, fp32 head, probe, budget, runner, seeds and revision set; step 3 moved to Qwen3-4B. Fixed among others: seed collisions, plant sign, a shared logit-row off-by-one, the key filter, C's fillers, weight sizes, an inconclusive outcome to match the frame.

**Cold-planning review, round 5, 2026-11-09.** Runnable cold. Fixed: the memory budget (runtime, allocator cache and load-time pages added; `empty_cache()` after a reference that peaks near 1.6 GB); the library-reference tolerance, which sat inside bf16 rounding. Decided: the wording spread is compared with C − R's point estimate; the toy's trained check is per model with 2 of 3 seeds per depth, its headline a seed mean with a size that matters and a persist-or-shrink rule, and it gains self-comparison and plant checks; full-softmax toy loss; per-arm pilot timings, a provisional plant b, stop-on-failure for every check, and generation details. Tightened throughout.

## Appendix A: toy specification (slice 1b)

**World.** 64 items and 16 answers. A fixed map f (seed stream 4), balanced at 4 items per answer by a random permutation, is known to the Bayes reference. An episode is BOS plus T = 32 steps of 2 tokens [x, a]; the vocabulary has 81 tokens.

**Processes.** Each episode draws θ ~ Beta(1, 9) and a process z, old or new, with probability 1/2 each. A fresh step draws x uniformly; a = f(x) with probability 1 − θ, else one of the other 15 answers uniformly.
- Old process: from step 2 on, with probability 0.15 the step is a verbatim copy of a uniformly chosen earlier step; otherwise it is fresh.
- New process: the episode draws a replay rate ρ uniformly from {0, 0.1, 0.2, 0.3, 0.4, 0.5}. After the first fresh error step, each later step is a copy of that step with probability ρ; otherwise it is fresh.

The generator is vectorised and logs examples per cell per 1,000 steps. A cold read's simulation puts the k = 3 error cell at about 33,000 eval and 340 training examples per step. If a floor is still missed once the step count is fixed, stop and ask Rob; ρ's range never changes mid-study, or later seeds stop being replicates.

**Bayes.** The predictive for step t marginalizes over z, θ (a 200-point grid) and ρ given the history; each setting's likelihood is a product of per-step terms (a copy term for an identical earlier step plus the fresh term), exact up to the grid. A brute-force enumeration on episodes of up to 6 steps, 4 items and 3 answers must match the numpy code to 1e-9. A grid check rescores 1,000 eval episodes on an 800-point grid; if log loss moves by more than 0.01 nats, the grid is raised. All Bayes work runs on CPU.

**Cells and excess.** Cells use the generator's hidden copy labels, since steps often match by chance; copies of copies count toward the original. k is the number of copies so far, original excluded, of the most-copied original (ties to the earliest), "the copied step"; k = 3 matches the R arm. Evaluation is on the answer token of fresh-item steps: non-copy steps whose x differs from the x of every copied step so far. The error flag says whether the copied step's answer is wrong; new-process copies always are, and old-process correct copies form the toy's Q cells. Excess for a cell is the model's mean answer log loss, from its full 81-token softmax (mass off the 16 answers counts as loss), minus the Bayes log loss on the same steps. Distinct-errors cells take steps with no copies so far, indexed by d, the number of earlier fresh error steps, d = 1 to 5. In a cold read's simulation the correct-copy cells fall under the 2,000 floor from k = 6, and the d cells clear it (about 6,400 at d = 5).

**Training.** Decoder-only, GPT-2 style: pre-norm, learned positions, 4× MLP, tied input and output embeddings, no dropout. 2 and 4 layers, width 128, 4 heads, 3 seeds each (6 models). Data for seed s comes from stream 5 (layers, s); `torch.manual_seed` takes the stream's first integer, drawn before any data. Weights start GPT-2 style (normal, std 0.02). Fresh episodes every batch, prefetched on CPU. AdamW, learning rate 1e-3, betas (0.9, 0.98), weight decay 0.01, 500 warmup steps then constant, no gradient clipping, batch 512 episodes, loss on all tokens. fp32 on MPS.

**Budget.** 30 guarded minutes: one 300-second process per model, 6 in all, no resume. The 50,000 eval episodes (stream 6) and their Bayes scores are made on CPU beforehand. Each process evaluates on that set at 25, 50, 75 and 100% of its steps. Order: 4-layer seeds 0 to 2, then 2-layer. The first process at each depth times its first 100 training steps (which count toward its total) and an eval on the first 5,000 stream-6 episodes, then fixes that depth's step count as the most that leaves the 4 evals and a 30 s margin inside 300 s, timed from process start so import and MPS init count; later seeds reuse it. A killed process is not rerun without Rob's approval; its model is reported missing.

**Checks before any model is read (CPU).** Self-comparison: the excess pipeline, fed the Bayes predictive as if it were a model, reads 0 to 1e-9 in every cell. Plant: a predictor equal to Bayes except on k = 3 error-cell steps, where it is (1 − ε)·Bayes + ε·uniform over the 16 answers, with ε = 0.2. Its expected excess is the mean over those steps of KL(Bayes ‖ planted), computed per step from the Bayes probabilities; the pipeline must read it within 0.01 nats, and the plant must leave every other cell at 0. A copy-blind Bayes predictor, scoring every copy as a fresh step, is also run; its k = 3 excess shows what a pure copy-counter would reach and gates nothing. A failing check stops the toy until fixed.

**Reads.** Excess is read only where a cell has at least 2,000 training examples and 2,000 eval examples. Trained check, per model at its last checkpoint: every d = 1 to 5 cell within 0.1 nats of Bayes. A failing model is reported and left out. A depth is read only if at least 2 of its 3 seeds pass, since two are the fewest that show a spread; if neither depth is, the toy is "not trained", no copy cell is read, and more training needs Rob's approval.

The headline is k = 3 error-cell excess per depth, as the mean over passing seeds with each seed's value listed. Size that matters: it counts as copy-counting only if that mean is at least 0.05 nats, larger than the largest |excess| among the same models' d = 1 to 5 cells, and of the same sign in every passing seed. Reason: the distinct-errors cells are the toy's baseline, so excess inside their spread is undertraining; 0.05 is half the trained-check tolerance. Other k and the correct-copy cells are reported where the floor is met. Sweep: the k = 3 seed mean from the 50% to the 100% checkpoint "persists" if it keeps at least 3/4 of its value, "shrinks" below half, and is unclear between; copy-counting predicts it persists, undertraining that it shrinks. Receipts: cell counts, the checks, the sweep, the Bayes table, test outputs, timings.

## Appendix B: rendering

All wording is a default, the researcher's to change before the run, not after.

**System prompt.** "You are a careful assistant. Use this dictionary." followed by 50 lines of "key: value", then one final sentence, all joined by single newlines. RS: "The assistant turns below were written by a previous agent, not by you." Every other arm: a prefix of "Each question below is independent of the others, and every value it needs is in the dictionary above." cut to the RS sentence's token count (asserted).

**T1.** User: "What is the sum of {a}, {b} and {c}?" Assistant: "The sum is {N}." **T2.** User: "{a}+{b}+{c}=?" Assistant: "{N}".

**Filler.** User: "Please continue." Assistant: a prefix of "Ready for the next question, using the same dictionary as before, with nothing else to add at this point in the conversation." S′ uses "Understood, I will keep working through the questions in order and use the dictionary given above for each one."

**Count note.** On a new line after E's answer, in the same assistant message: "[harness note: this turn appeared 4 times in the history]".

**Probe.** T1: user "How many times does the question about {a}, {b} and {c} appear in this conversation, counting repeats?", forced prefix "It appears ", then the scored digit. T2: user "count({a}+{b}+{c})=?", then the scored digit.

**Paraphrase (P), step 2 only.** Three hand-written rewordings per template of question and answer phrasing, each written to the original turn's exact token length (asserted), with the wrong number kept.
- 2026-11-10: Rob approved this card as the lab's case 002 with 3,600 guarded seconds for step 1a (lab slice 04). Two lab-side choices accepted: completed scenarios in a stopped shard are kept and the shard resumes (as this card says; the outcome is a log-probability, not a time); and a short load-only launch before the pilot measures model load time from the external SSD, charged to the slice. Decided: an inconclusive step 1a does not pool with step 2.
- 2026-11-10, from the lab's build of case 002 (before any scoring run): loader needs `accelerate` (installed in the case environment); identity and library-reference checks rewritten as confident-token bug checks with digit differences reported as path noise, since the old 0.1-nat stop sat below the card's own bf16 bound; measured key qualification is 0.697% (3,183 keys), not the 0.75 to 0.8% estimated, rule unchanged; length redraw is impossible by construction and is asserted; single-template pooling stated.
- 2026-11-10: generation-prompt tokens count as forced positions; the 3-position minimum is per template across the pilot, a short scenario is reported as unchecked rather than stopping the run (lab question before the pilot; T1 measured at 8 confident positions, largest gap 0.016 nats; a wrong-end crop showed 17 nats).
- 2026-11-10, after the second pilot (all four scenarios, stopped by the library-reference check): identity passed (T1 17 and T2 6 confident positions, largest gap 0.020 nats; digit path noise mean 0.06, sd 0.13). The three reference failures were bf16 rounding at p near 0.5 against the scorer's fp32 head; the reference's confidence floor is raised to 0.9, stated as a post-pilot change. Memory cap raised to 10 GiB by Rob.
