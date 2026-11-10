# Do a model's own visible errors raise its next error, and where does the falling agent hazard come from?

Status: queued, 2026-11-09. Revised after a hostile review (record at the end). Search gate passed 2026-11-09 (20 queries, none refused): partly scooped. Role relabeling and identical-versus-distinct repeated errors now have published results; own-versus-other-agent attribution on next-step accuracy, with a count regressor, is still open.

Ledger idea `agent-hazard-mixture`, merged with "Repeating a failed tool call gets more likely each time it is repeated". Designed, not run.

Visible own errors cannot change time to first error, since none are in view before it, so the card carries two claims with separate verdicts. Claim A is the main one: cheap, and its attribution half is still open after the search. Claim B's mixture half rests on METR's public runs file, which the search found (repeats per task still to confirm).

## Claim A: visible errors and who made them

On a step-checkable task, each earlier error still in context lowers the chance the next step is right. Sinha et al. already show planted errors lower accuracy; the new part is attribution. The effect should be larger when the errors sit in the model's own assistant turns than when the identical turns carry another agent's name, and wrong arithmetic unrelated to the task should not produce it. If another agent's errors hurt just as much, this is in-context copying of an error rate, and "self-conditioning" is the wrong name.

The same mechanism, in agent form: the chance of issuing an identical failed tool call again rises with the number of identical failed copies in view. That is step 1b.

## Claim B: the falling population hazard

Weibull fits to METR pass/fail data give a shape below 1. Ord and frailty theory say a mix of task difficulties produces that with no change inside any run. Two things would add something: a direct measurement of within-run hazard on clean histories, free of survival selection, per difficulty level; and a frailty re-analysis of real repeated attempts per task (step 4). If the runs file turns out to hold one run per model and task, the mixture half is cut and only the within-run measurement stays.

This card makes no claim about METR horizons unless step 4 runs on task-failure data.

## What is already known

Updated by the 2026-11-09 search gate below.

- [Sinha et al., "The Illusion of Diminishing Returns"](https://arxiv.org/abs/2509.09677), ICLR 2026 (v3 March 2026). Running sum, errors planted at a chosen rate, Qwen3 4B to 32B, local scoring; thinking removes the effect. Only the error rate varies: no other-agent or user-role arm, no count or recency manipulation, no irrelevant-error control. [Code](https://github.com/long-horizon-execution/measuring-execution) and [data](https://huggingface.co/datasets/arvindh75/Long-Horizon-Execution) exist, contents not read.
- [Chen et al., "The Self-Correction Illusion"](https://arxiv.org/abs/2606.05976), June 2026. One byte-identical planted wrong claim, moved from the agent's own thought block to a user, tool or system-memory role, raises explicit flagging by 23 to 93 points. Final-answer accuracy shows no significant change. A duplicate inside the thought block adds little. One error per trial, no count, outcome is flagging rather than next-step accuracy.
- [Noël, "An Error in the Context Is a Demonstration"](https://neurips.cc/virtual/2026/171281), NeurIPS 2026 SLM-Agents workshop. Teacher-forced ledger audits with injected errors, scoring only logically independent later steps. Repeating one wrong rule increases damage; several different wrong rules do not. Detect-only guardrails do nothing; redacting in place still leaves damage. No attribution arm.
- [Cheng et al., "Contextual Drag"](https://arxiv.org/abs/2602.04288), Feb 2026. One or two failed drafts in context cut accuracy 10 to 20 points across 11 models. Drafts come from the evaluated model or others, without a controlled comparison.
- [Ord, half-life](https://www.tobyord.com/writing/half-life) and [hazard rates decline](https://www.tobyord.com/writing/hazard-rates-for-ai-agents-decline); [Hamilton, "Peto's paradox"](https://gushamilton.github.io/lab/2026/01/23/petos-paradox-ai-agents), Weibull shapes about 0.6 to 0.9 on METR data; [Kwa et al. (METR)](https://arxiv.org/abs/2503.14499). The mixture argument is stated, not tested; no METR or Epoch reply and no frailty re-analysis turned up. Vaupel et al. 1979 on frailty, title only.
- [Moss, "METR's data can't distinguish between trajectories"](https://www.lesswrong.com/posts/sBEzomgnYJmYHki9T/updated-metr-s-data-can-t-distinguish-between-trajectories), Feb 2026. Bayesian IRT on METR data with a random difficulty per task; no hazard or frailty analysis. Stan code at [JonasMoss/metr-stats](https://github.com/JonasMoss/metr-stats).
- METR's [eval-analysis-public](https://github.com/METR/eval-analysis-public) ships `runs.jsonl` per report with `task_id`, `alias`, `score_binarized`, `human_minutes`. Kwa et al. state about 8 runs per agent and task.
- [ReInBench](https://arxiv.org/abs/2610.36471), 2026-10, abstract only. About half of error events in real traces self-recover.

## Prediction (written before any run)

Non-thinking mode; any write-up says so in its first line.

- Own history: the teacher-forced log-probability of the locally correct answer falls with visible-error count, with an interval excluding zero. On greedy accuracy, the ledger author's number stands: at least 0.3 log-odds more error per visible error.
- Label-only role swap: slope at most half the own-history slope. Irrelevant errors: slope inside the permutation null. These are this card's numbers.
- Repair rate (true sum instead of the local update) is higher under role swap than under own history.
- Repeated failed call: the probability of the identical call rises with k from 1 to 6. Collapsing the copies to one plus "failed k times" lowers it at k = 4. The ledger's size, at least 10 points per copy, is kept for greedy reads. k distinct failed calls stay flat, matching Noël's one-rule-versus-many result.
- Claim B: on clean histories within one difficulty level, the per-step error rate does not fall with step index (coefficient zero or positive). Sinha's clean arm suggests it rises.

## Design

Task: Sinha's running sum, built from their released code. Record every difference from their generator and scorer. One key per turn. Difficulty knob: digits per value.

Scoring is local. Turn t is correct if it equals the last visible sum plus the new value, whether or not that visible sum was right. Greedy answers are logged as local-correct, global-correct (repaired), or other.

Model: `Qwen/Qwen3.5-4B` through `mlx-community/Qwen3.5-4B-MLX-4bit` (about 3.03 GB per the reviewer), one fixed serving stack.

**Step 0 (CPU, about one hour, before any GPU).** Searches are done. Load METR's `runs.jsonl` and count rows per `task_id` and `alias`. If there is one row per pair, cut step 4 and the mixture wording now. Then a 5-minute setup: load the model with the pinned loader, confirm text-only generation with thinking off (flag name not checked), record versions. The base repo is tagged image-text-to-text; fallbacks are `mlx-vlm` text-only, then `transformers` on MPS in bf16 under 8 GiB (not checked).

**Step 1 (one slice, about 25 GPU-minutes).**

1. Timing probe, 60 seconds: one teacher-forced forward pass and one short greedy answer at 20 turns of context.
2. Base-rate gate, rule fixed now, run before any arm. On 40 clean (count-0) own histories per digits-per-value level, starting at the smallest, read greedy error at turn 21 and mean teacher-forced log P(locally correct answer) per digit. Pick the level whose greedy error lies in 15% to 40%. Bar to go on: some level has greedy error between 10% and 50% and mean log P(correct) per digit above -1.0 nat (uniform guessing is about -2.3). If no level clears the bar, stop: do not run the arms. If the model is at ceiling on the smallest level, add turns or digits once; if it is near chance on the smallest level, try thinking-off at the next Qwen3.5 size once, then park the card. Cost: about 5 GPU-minutes inside this slice, sized from the timing probe, each process under 300 s and 8 GiB.
3. Histories: 20 turns, padded to matched token length, query at turn 21. The most recent planted error always sits at turn 20, and older ones are added behind it at turns pre-registered in the seed table. Counts: 0, 1, 2, 4, 8 for own history; 0, 2, 8 for the other arms. A recency arm puts a single error at turn 5, 12 or 20.
4. Arms, shuffled per block. Own history. Label-only swap: the exact assistant-turn layout with each turn prefixed "Agent B:" and a one-line handover. Transcript swap: the same turns quoted inside a user message, which is Chen et al.'s user-role condition measured on next-step accuracy instead of flagging. Irrelevant: the same number of wrong sums in an unrelated side note, with all task turns correct.
5. Deciding metric: the slope of teacher-forced log-probability of the locally correct answer on count, own minus label-only swap, with a bootstrap over histories. One forward pass per history, and arms that share a prefix reuse a prompt cache. Greedy accuracy and the three-way outcome are the secondary read. Ownership counts as supported only if the label-only swap shows the gap. The transcript swap shows what formatting alone does.
6. Step index and context length are constant, so step 1 fits count only.

Histories per cell are set from the probe; a greedy-only design was near 1.5 standard errors on the reviewer's rough sum, hence the continuous metric.

Each process runs one block, appends JSONL and exits under 300 s and 8 GiB.

**Step 1b (same harness, a few minutes).** A short tool-loop transcript in which one tool call has failed k times, k = 0 to 6. Metric: teacher-forced probability of the identical call next. Controls: k distinct failed calls (a rise means recency or a general stuck state), and one copy plus "failed k times". Payoff: a harness rule to collapse repeated failed calls. Gate first, on 20 transcripts at k = 1: the mean teacher-forced probability of the identical call must lie between 0.02 and 0.8, so a rise with k has room to show. Outside that band, change the transcript once (error message wording, tool set); if it still fails, drop step 1b.

**Step 2 (local slices).** Natural 20-step runs at three difficulty levels. Fit count, step index and context length, which now vary. For Claim B, score planted clean histories teacher-forced at every step, so none drops out after a first error; a step-level logistic per level with a random effect per history separates within-run change from item frailty, which a Weibull on first-error times cannot. Add one thinking-on cell (count 0 against 8, own history) and Sinha's deletion and sliding-window arms.

**Step 3 (conditional, needs Rob's approval for its length).** 200-step runs as an out-of-sample test of the frozen step-2 fit. Each process carries one segment and saves the history (and prompt cache, if the loader can) for a deterministic resume. Slice count from step-1 timing; the ledger's 30 to 40 hours is not checked.

**Step 4 (CPU, only if step 0 finds repeats).** The runs are pass/fail with no failure time, so the fit is a logistic of success on log human minutes with a random effect per task, estimated from the repeats; the task variance is the frailty. Compare against Moss's IRT fit, reusing his Stan code. This is the mixture result. Synthetic pooled Weibull fits are then a methods check only.

## Checks before believing anything

- Permutation null: give clean histories random fake count labels, padded to the real cells' lengths, and fit the same regression 200 times. The slope must sit inside its interval around zero. Also split-half base rate on count 0, and ten re-runs with identical outputs and log-probabilities.
- Planted effect on real outputs: corrupt real count-0 greedy outputs by a known logistic law in a fake count, run the full pipeline, and recover the lift; for the deciding metric, recover a known shift added to real log-probabilities through the same bootstrap.
- Planted frailty: a scripted stand-in with known per-history hazard spread; the step-2 model must recover the spread and a zero step coefficient, and zero variance from a zero-spread run.
- The scorer passes 50 hand-made turns, including repaired cases.

## Kill criterion

Claim A: drop it if the own-history slope's interval includes zero at step 1 and at a second model size. Drop "own" if the label-only swap slope lies within the null's spread of the own slope. Step 1b: drop the copy-strength reading if distinct failed calls rise at the same rate as identical ones. Claim B: drop "within-run hazard does not fall" if the step coefficient on clean histories is negative with an interval excluding zero within a level. Drop the mixture half if step 0 finds no repeated-attempt data.

## Cost on this hardware

Step 0: an hour of CPU, a 3 GB download (33 GiB free). Steps 1 and 1b: one slice. Step 2: 3 to 6 slices. Step 3: unknown. Step 4: an afternoon.

## What a yes or a no changes

Claim A yes: errors hurt more because they look like the model's own, so relabelling failed turns has a measured reason, and step 1b gives a rule for repeated tool calls. No on the swap: the model copies whatever error rate is in view, a different fix. Claim B with step 4: the METR hazard shape split into task mix and within-run change; without it, one step coefficient on one synthetic task.

## Untested

Sinha reports the effect persists with scale. Untested here: thinking beyond one cell, real agents, larger models, METR tasks.

## Receipts to keep

Plan, seed table, base-rate rule, versions, model hash, the diff against Sinha's code, raw JSONL, null, planted and determinism readings, fits with intervals, the frozen step-2 fit with its hash.

## Search gate, 2026-11-09

Ran 20 standard web searches, none refused, about 15 fetches and three GitHub repository searches (no hits). Semantic Scholar rate-limited one call; OpenReview served a verification page. Beyond the owed list, queries used other communities' words: role relabeling, choice-supportive bias, multi-agent handoff errors, error contagion, contextual drag, tool-call loops, induction-head copying, Weibull mixtures under unobserved heterogeneity.

Closest works and how much was read (findings are under "What is already known"):

- [Chen et al., arXiv 2606.05976](https://arxiv.org/abs/2606.05976), June 2026 (v2 July): abstract and method section. Closest to the swap arms.
- [Noël, NeurIPS 2026 workshop](https://neurips.cc/virtual/2026/171281): listing abstract only. Closest to step 1b.
- [Cheng et al., arXiv 2602.04288](https://arxiv.org/abs/2602.04288), Feb 2026: abstract and two-thirds of the HTML.
- [Moss, LessWrong](https://www.lesswrong.com/posts/sBEzomgnYJmYHki9T/updated-metr-s-data-can-t-distinguish-between-trajectories), Feb 2026, and METR's runs README: summary and field list. The runs file is over 10 MB and was not opened.
- Abstract only: [Bogdanov et al., arXiv 2609.29508](https://arxiv.org/abs/2609.29508), Aug 2026, discrete-time hazard on 84,540 agent trajectories, no frailty term, event a reward claim; [Wang et al., arXiv 2610.27406](https://arxiv.org/abs/2610.27406), Oct 2026, errors reproduced through agent memory.

Design changes: step 0 checks the runs file; step 4 is a random-effect logistic on pass/fail repeats, compared against Moss; the transcript swap is Chen's user role on a new outcome; step 1b's distinct-call prediction follows Noël. The write-up cites both and leads with the agent-name swap and the count slope, which neither covers.

Still open: whether `runs.jsonl` holds repeats per pair; Noël's full text (a slope per repeated copy would scoop step 1b's headline); Chen's appendix on final-answer accuracy per role; a Google study on models defending answers framed as their own, seen only via a blog summary.

## Review record

Must-fix, all fixed: the two halves never met (split into Claims A and B; METR wording only if step 4 runs); scoring rule unstated (local scoring, three-way outcome, repair rate, Sinha's code); recency confound (newest error at turn 20, recency arm, pre-registered turns); step 1 underpowered (teacher-forced log-probability decides, base-rate rule); self-comparison not computable (permutation null, split-half, determinism). Weibull confounds within-run hazard with frailty: fixed differently, by teacher-forced scoring of clean histories at every step with a per-history random effect, since a logistic on steps before the first error still selects easy items.

Should-fix: planted check moved onto real outputs; mixture half cut unless step 0 finds repeats; tool-call idea added as step 1b; swap split into label-only and transcript variants; one thinking-on cell, scope stated first; prior art checked by the 2026-11-09 search gate. Minor, fixed: step 1 fits count only; loader setup with fallbacks; 200-step runs segmented. The reviewer's pre-search read of Sinha et al. is folded into "What is already known".

2026-11-10, base-task check (after copies-as-evidence step 1a ran at chance): the base-rate pick had no stop, so step 1 could run its arms on a model at chance or at ceiling. Added a gate in step 1 item 2 (count-0 greedy error 10% to 50% and log P(correct) per digit above -1.0 nat at some level, else stop, one retry, then park) and a k = 1 headroom gate to step 1b. Whether Qwen3.5-4B with thinking off is above chance on the running sum is not checked.
