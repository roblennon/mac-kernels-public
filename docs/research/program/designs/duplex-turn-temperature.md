# Late turn-taking in duplex speech models: sampler, stimulus, or model?

Status: queued, 2026-11-09. Search gate passed 2026-11-09 (22 queries, none refused); partly anticipated, not scooped.

A full-duplex speech model such as Moshi decides every 80 ms frame whether to keep padding or start a word. An October 2026 paper found that two PersonaPlex-7B instances in self-play take the floor late and almost never start in the partner's last 120 ms. The authors read this as reacting instead of predicting turn ends. That run sampled text at temperature 0.7, and sampling below 1 makes a rare per-frame event rarer. But PersonaPlex reports 70 to 170 ms Full-Duplex-Bench latency on human-recorded turns, so the stimulus (model speech against human speech), or the metric, may matter more than the sampler. Which causes the lateness: the logits, the sampler, or the stimulus?

## Hypothesis

Read straight from Moshi's logits at temperature 1, the per-frame start probability after a user's turn ends implies a median start under 250 ms. Sampling at deployed settings delays that median. The claim covers the median only. The early-start deficit (about 1% of transfers in the last 120 ms) is predicted not to be a temperature artefact under the PyTorch defaults. If the T=1 logits imply a median above 400 ms on human-recorded turns, the temperature account is wrong for Moshi.

## Theory

Moshi's text stream emits PAD while silent. One frame before each word it emits an end-of-padding token (EPAD), so the start decision is close to a binary PAD-versus-EPAD choice. In the PyTorch code PAD id is 3 and EPAD id is 0. If the model gives a start probability h in a frame, temperature T turns it into h^(1/T) / (h^(1/T) + (1-h)^(1/T)). For small h and T below 1 this shrinks h, compounding frame after frame. For a sharp hazard (h near 0.5 or above) T below 1 barely delays the start, so the effect needs a diffuse hazard, which step 1 measures.

Truncation is harsher. The MLX sampler defaults to 0.8 with top-p 0.95, so a frame where PAD holds over 95% of tempered mass cannot start. The PyTorch defaults (text 0.7, top-k 25) barely truncate a binary choice.

Assumptions, said out loud:

- The start decision sits in the text stream, and audio follows by a fixed lag. Audio-only starts (breath, backchannel) were not checked.
- PersonaPlex is a fine-tune of the Moshiko weights, so it likely shares the PAD/EPAD scheme. That is inferred, not checked.

A toy (not a result): tempering a human-like onset hazard at T=0.7 moves the median from about 200 ms to 280 to 520 ms, but leaves 8 to 10% of starts in the last 160 ms, against about 1% in the paper. Temperature likely explains part of the median and little of the missing early starts. Step 1 recomputes this as `toy_tempered_hazard.py`.

## What is already known

Verdict: partly open. The search gate below found no work that varies temperature and measures per-transfer start timing in a duplex model, or reads the start hazard from logits.

- [Moshi (arXiv 2410.00037)](https://arxiv.org/abs/2410.00037), Table 9, self-play: total gap 12.5 s at T=0.8, 7.5 s at T=0.9, 4.5 s at T=1.0, against 4.2 s ground truth. Overlap 1.2, 2.2, 4.1 s against 3.3 s. Conditional perplexity rises from 41.9 to 79.3. Forcing EPAD makes Moshi start at once. So "lower temperature lengthens gaps on Moshi, and T=1 costs content quality" is published, in aggregate.
- [Coupled but Late (arXiv 2610.08683)](https://arxiv.org/abs/2610.08683): PersonaPlex-7B self-play median 400 to 560 ms against 137 ms for Switchboard, few transfers in the final 120 ms. Delaying one channel shifts the response equally, read as reactive. Text 0.7, audio 0.8, confirmed; top-k not stated; temperature never varied. Humans put about 10% of transfers in the last 120 ms. IPUs merge across gaps under 180 ms; agent speech is cut at 0.01 RMS.
- PersonaPlex: the model card (nvidia/personaplex-7b-v1, gated) gives 0.170 s smooth turn-taking and 0.240 s interruption; the [paper (arXiv 2602.06053)](https://arxiv.org/abs/2602.06053) gives 0.070 s, with Moshi at 0.265 s. Neither states decoding. [Full-Duplex-Bench v1](https://arxiv.org/abs/2503.04721) uses human Candor audio, and its [code](https://github.com/DanielLin94144/Full-Duplex-Bench/blob/main/v1_v1.5/evaluation/eval_smooth_turn_taking.py) reports the mean of first ASR word start minus user end, takeovers only, negatives clamped to zero. So 170 against 480 ms differs in metric as well as stimulus.
- [Riera et al. (arXiv 2605.20356)](https://arxiv.org/abs/2605.20356) and Kyutai's [`padding_bonus`](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/tts.py): PAD-logit bias already exists. Riera et al. run two Moshi instances in self-play at three unstated bias levels and report no gap timings.
- [Ohashi et al. (arXiv 2606.11167)](https://arxiv.org/abs/2606.11167): RL cuts latency on Full-Duplex-Bench v1. [Chang et al. (arXiv 2606.11386)](https://arxiv.org/abs/2606.11386), abstract: PersonaPlex is slow to switch to listening when interrupted ("state inertia"); an earlier note here had the direction wrong.
- [ASPIRin (arXiv 2604.10065)](https://arxiv.org/abs/2604.10065), April 2026: on Moshi, sums text logits into a binary speak-or-silent state and trains timing with GRPO apart from content. Step 2's decoupling, done by training; no temperature analysis or sampler-only baseline. Moshi on Full-Duplex-Bench smooth turn-taking: 0.265 s, or 0.161 s with a 3 s prompt delay.
- [TurnBench (arXiv 2608.25218)](https://www.alphaxiv.org/abs/2608.25218), September 2026: on smooth transfers humans start a median 151 ms before the turn ends; the best detector (VAP) decides 368 ms after.

Open: the per-transfer start distribution read from the logits, especially in the last 120 ms; whether the stimulus moves timing more than temperature does; and a decoupled start temperature measured on timing and content together. The contribution is that diagnosis, on PersonaPlex's published lateness. The PAD-bias arm is a known baseline, and training-time decoupling is ASPIRin's; the sampler-only version is open.

## Prediction (written before any run)

- Decisive (step 1): on Moshiko MLX q4 with human-recorded turns, the median start offset implied by T=1 logits, plus the measured text-to-audio lag, is under 250 ms.
- The share of T=1 start mass in the partner's last 120 ms is under 3%, and in the last 160 ms under 4% (reactive reading survives at T=1).
- The T=1 median moves more between human and model-generated stimuli than between T=0.7 and T=1.0. Low confidence.
- Expected from Table 9, not a test: the T=0.7 top-k 25 median is at least 1.4 times the T=1 median.
- Step 2: a start temperature of 1.0 with words at 0.7 brings timing at least as close to human as coupled T=1, with content no worse than coupled T=0.7 beyond the seed noise floor.
- A difference under one frame (80 ms) does not matter.

## Design

**Step 0 (no compute).** Mostly done in the search gate; NVIDIA's decoding is unreported. Left: copy the IPU and floor-transfer rule from 2610.08683 into code, and find out whether the Full-Duplex-Bench v1 Candor smooth-turn segments can be downloaded.

**Model.** `kyutai/moshiko-mlx-q4`: `model.q4.safetensors` 4.81 GB, Mimi `tokenizer-e351c8d8-checkpoint125.safetensors` 0.38 GB, `tokenizer_spm_32k_3.model`; 5.19 GB in all. `moshi_mlx` in a uv environment, version recorded.

**Stimuli.** 60 texts committed first: 30 questions, 30 statements, 20 with a 400 ms internal pause. Four renderings: TTS voice A, TTS voice B (noise floor), Rob reading the texts (human arm), and a model-generated arm close to self-play (Kyutai TTS, which shares Mimi; runnable size not checked). If obtainable, the Full-Duplex-Bench v1 Candor segments form a second human arm (added after the search). Turn ends come from the paper's IPU and floor-transfer rule, applied identically to every arm. Under that rule offsets sit on the same scale as the paper's 137 ms and 400 to 560 ms; the 250 and 400 ms thresholds are re-derived if the rule differs.

**Step 1a (one slice, TTS voice A).** Moshi speaks a greeting, sampled once; the KV cache after it is checkpointed (or re-prefilled, with that time counted). The user turn plays. Moshi's text is forced to PAD; its audio is sampled by the depth transformer at the deployed audio temperature. Save text logits from 2 s before to 3 s after each end. Compute the per-frame start probability for T ∈ {0.5, 0.7, 0.85, 1.0}, each with no truncation, top-k 25 and top-p 0.95, and turn each into a start distribution by survival products. On 10 turns, also run digital-silence audio codes and the planted delay below.

**Subject gate (read from the step 1a run, first: its first 10 turns, before the other turns).** The test needs Moshi to take the floor under this harness, with a hazard that is neither absent nor instant. Forcing text to PAD through the greeting and user turn may push it off-distribution, and nothing so far shows q4 Moshi starts after TTS turns at all. On the first 10 turns, measure: (a) free run at default MLX decoding, one seed per turn: does a word (EPAD then text) start within 3 s after the turn end? (b) From the saved T=1 logits: cumulative start probability by 3 s after the end, and the share of start mass before the end. The computed distribution in (b) decides; one sampled run per turn is too noisy to veto (a subject that starts within 3 s on 70% of turns would pass 8 of 10 only 38% of the time). Bar: (a) is a gross-functionality check only: free run produces at least one word start after a turn end across the 10 turns, so the harness and model can speak at all; (b) cumulative T=1 start probability at least 0.5 by the end of the window on at least 8 of 10 turns, so the median exists inside the window, and under 0.5 of start mass falls before the end on at least 8 of 10. Also report the free-run start count and peak per-frame h; if h exceeds 0.5 on most turns, the hazard is sharp, the temperature arms are expected to read null (see Theory), and the card says so before step 1b. If (b) fails only because start mass has not reached 0.5 by 3 s, re-save logits over a longer window on the same 10 turns (length fixed and recorded before the rerun) and re-read (b) before declaring the subject absent. If (a) fails, or (b) still fails: stop. Check the forced-PAD prefix against free run first, then try a different greeting. If it still fails, park the card; a model that does not take the turn cannot test the sampler. Cost: inside the step 1a slice (10 turns plus 10 free runs, under 30 GPU-minutes, each process under 300 s, peak under 7 GiB); a longer-window rerun is a second pass over the same 10 turns. Per-turn frame time not checked yet.

**Step 1b (two to four slices).** Voice B, human and model-generated arms, arms interleaved by turn. Free-run check: 20 turns × 5 seeds at T=1.

**Metric that decides:** median T=1 implied offset on human turns, bootstrap 95% interval over turns.

**Reported with it:** last-120 and 160 ms start shares; starts in internal pauses, separately; the projection test (hazard in the 480 ms before true ends against before internal pauses); peak hazard; text-to-audio lag.

**Step 2 (several slices, sampled).** A two-stage sampler: PAD-or-start at its own temperature, then the word at 0.7. Arms: start temperature 0.5, 0.7, 0.85, 1.0; coupled 0.7 top-k 25; coupled 1.0; MLX default; PAD-bias baseline at three values. Teacher-force to 1 s before each end, sample 3 s, 5 seeds per turn per arm, arms interleaved. Content per arm: Whisper large-v3-turbo WER of Moshi's audio against its own text stream, and perplexity of that text under Qwen3.5-4B given the user turn. Timing and content are judged together.

**Step 3 (needs Rob).** PersonaPlex: an ungated MLX conversion, `aufklarer/PersonaPlex-7B-MLX-4bit` (about 4.9 to 5.8 GB, Swift runtime), fits; its card says 4-bit garbles output and 8-bit needs about 11 GB, and logit access is not checked. Otherwise the 16.74 GB bf16 file, which breaks the disk floor and memory limit. Then two-instance self-play with the paper's rules. **Step 4 (later).** Compare the best arm with Ohashi et al.'s and ASPIRin's RL on Full-Duplex-Bench v1; checkpoint availability not checked.

## Checks before believing anything

- Self-comparison: split-half and voice-A against voice-B median differences must sit inside the bootstrap interval. A primary difference smaller than these is called inconclusive. In step 2, two identical arms with different seeds must read zero on timing and content.
- Planted effects through the model: delay the user audio by 240 ms (3 frames) while keeping the original turn-end label. The implied distribution must shift by +240 ms within one frame. This is also the paper's delay test, so it doubles as a replication on Moshi. In step 2, 400 ms of spliced silence before a known onset reads +400 ± 40 ms.
- Free-run match: the 100 sampled T=1 starts must match the computed distribution across the whole histogram (two-sample KS and decile comparison), not only the median.
- Known result: on the Candor arm at default decoding, Full-Duplex-Bench's own metric should land near Moshi's published 0.265 s (bf16; q4 may differ). A large miss means the harness differs, and is reported before any temperature claim.
- Sanity: non-PAD mass at the start frame is almost all EPAD (effective m under 2); if not, the pipeline is broken. Gaps lengthen from T=1.0 to 0.8, as in Table 9.

## The control that could embarrass it

The T=1 logits on human turns. A median above 400 ms means the model is late on its own. If hazard before internal pauses equals hazard before true ends, the model responds to silence, not to turn ends. If the stimulus arms differ more than the temperatures do, the lateness in self-play is about model speech, and that becomes the headline.

## Confounds

Moshi is not PersonaPlex (step 3). 4-bit weights shift logits; a q8 or bf16 check waits for more memory. The text decision is not audio onset; the lag is measured.

## Kill criterion

Drop the temperature account for Moshi if the T=1 median on human turns exceeds 400 ms, or if moving only the start temperature from 0.7 to 1.0 changes the step-2 median by under 15%. A no on TTS turns alone is not a kill. Between 250 and 400 ms is inconclusive. A decoupled arm that wins on timing but loses on content is a no for the fix.

## Cost on this hardware

- Step 1a: 5.19 GB download; free disk about 31 GB, re-measured at run time. Under 30 device minutes; frame rate measured on the first turn, turns cut if needed. Stop if peak memory exceeds 7 GiB. Each process under 300 s, one file per turn, resumable. Logits about 0.5 GB.
- Step 1b: Rob's time to record 60 short texts.
- Step 2: several slices, sized from step 1. Whisper and Qwen3.5-4B run after, not alongside, Moshi.
- Step 3: the 4-bit MLX conversion fits; 8-bit or bf16 need a higher memory limit.

## What a yes or a no changes

A yes gives a free fix checked on timing and content, and forces the "react, not project" reading to be redone with decoding controlled. A stimulus finding says self-play benchmarks measure the partner's speech too. A no puts the lateness in the model, supporting training fixes such as RL.

## What carries to larger models, and what stays untested

Better models may have sharper hazards, shrinking the effect; peak hazard and median are reported so readers can judge transfer. Untested: PersonaPlex until step 3, true self-play, models without an EPAD-style token.

## Receipts to keep

Texts, recordings, turn ends; versions, hashes, sampler settings; per-turn PAD, EPAD and top-50 logits; the toy's output dated before the run; check outputs; per-arm timing and content numbers; peak memory and frame rate.

## Search gate, 2026-11-09

Queries: 22 web searches, none refused: the six owed, plus sixteen worded for other communities (temperature ablations, reactive against predictive timing, GitHub issues, logit probes, Talking Turns, Switchboard gap comparisons, PAD bias, TTS against human stimuli, OpenReview, survival analysis, FDB v3, dyadic evaluation, anticipation, quants). Also a Semantic Scholar query (105 hits, top 20 scanned) and an arXiv API query (no hits).

Read in full or by section: Coupled but Late (6 October 2026; settings, rules, results), ASPIRin (11 April 2026; method, Table 1), Riera et al. (19 May 2026; setup, metrics), the Full-Duplex-Bench v1 paper and its scoring script, the PersonaPlex paper (February 2026; Table 2, data). Abstract only: [DyaFDB (2610.08125)](https://arxiv.org/abs/2610.08125), 6 October 2026, two-model self- and cross-play, models and timing not in the abstract; Chang et al.; TurnBench.

Outcome: partly anticipated (Moshi Table 9 in aggregate, ASPIRin for decoupling by training), not scooped. Changes: a Candor human arm and a known-result check; step 0 mostly closed; step 3 uses an ungated quant; Chang corrected; ASPIRin added to step 4.

Still open: DyaFDB's full text; whether the Candor segments are downloadable; NVIDIA's decoding; whether PersonaPlex keeps EPAD; logit access in the Swift port; its per-step time (68 ms in a blog, 158 ms on the card).

## Review record

Hostile review 2026-11-09, verdict queue-after-fixes. Must-fixes, all made: Table 9 anticipates Moshi; 170 ms card latency against self-play; no content metric; pre-end metric unlike the paper's; planted effects tested only arithmetic. Five should-fixes and three minor points, all addressed.

Base-task gate review 2026-11-10, after copies-as-evidence step 1a's model got no greedy answer right (one template inaccurate, the other off-format): the card checked the instrument (planted delay, EPAD sanity) but never whether q4 Moshi takes the turn under forced PAD. Added a gate at the start of step 1a: free-run starts in at least 8 of 10 turns, a T=1 median inside the 3 s window, and most start mass after the end; stop or park otherwise. Risk without it: medium.

2026-11-10, cross-family gate review (Sol): free-run starts demoted from an 8-of-10 veto (38% pass at a 70% start rate, checked) to a gross-functionality check, with the computed T=1 distribution deciding whether the median exists. A longer logit window on the same 10 turns is now tried before the subject is declared absent.

Scooping search: the first six queries were refused (budget spent); all were rerun on 2026-11-09 with sixteen more, see the search gate.
