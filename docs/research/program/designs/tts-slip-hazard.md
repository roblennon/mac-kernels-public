# Do speech models lose their place at a constant rate? Hazard curves for TTS skips and repeats

Status: queued, 2026-11-09 (revised after hostile review, and after Sol's cross-family review: sizing and a budget stop rule). Search gate passed on 2026-11-09: 12 searches found no hazard-shape, chunked-control or slip-temperature-law study; Step 3 was changed for a known low-temperature result.

Ledger idea `tts-slip-hazard`, merged with "Ten thousand takes of one sentence" and the speech half of the Arrhenius idea. Designed, not run.

Autoregressive text-to-speech models sometimes skip, repeat, garble, or stop early, and LACI shows Qwen3-TTS error rising with length. The usual reading is drift: state degrades as the utterance grows, fixed by longer training clips. The other account is rare random slips at a constant rate per second, as a tracking loop loses lock in noise. Those are fixed by detect-and-retry, and they give a length-free number: slips per hour.

A rising error with length does not by itself favour drift, because a constant-rate slip that loses the rest of the clip also raises mean WER with length. As arithmetic only: suppose all of LACI's 2.7 percent mean WER on short prompts came from rest-of-clip failures at a constant rate, and a typical short prompt is 300 words (an assumption; LACI's bucket is "under 500"). Then the same rate gives roughly 12 percent mean WER at 1,500 words, against LACI's 16.6. The length curve cannot separate the accounts; the hazard's shape over position can.

## Hypothesis

Under the shipped sampler, and at lengths where LACI sees error rise, the hazard of the first slip does not rise with position in the utterance. The hypothesis is wrong if the hazard in the last third of long clips is clearly above the hazard in the first third (ratio above 1.5), or if long clips slip clearly more than the same text cut into pieces.

Secondary: for skips and garbles, log slip rate is linear in 1/T, and beats T and log T on held-out temperatures. Repeats and run-ons are expected to rise as T falls (see What is already known) and are fitted apart.

## Theory

Attention tracks a reading position, and sampling noise jitters it each frame. In phase-locked loop theory (Viterbi, 1963), loss of lock is a rare escape: near-Poisson, with a rate falling exponentially in signal-to-noise. Leaving the top softmax token goes roughly as exp(-gap/T), so log rate should be linear in 1/T.

Assumptions: temperature changes noise but not which contexts the model visits, so the 1/T line may bend; and pooling texts of different difficulty fakes a falling hazard, so fits are stratified by passage.

## What is already known

- [Taming Long-form Text-to-Speech (LACI), arXiv 2609.16989](https://arxiv.org/abs/2609.16989), 2026-09-15. Live slip detection from alignment heads, rollback and reseed. For Qwen3-TTS-0.6B, Table 2 reports mean / worst-of-N WER over seeds: 2.7 / 5.4 under 500 words, 16.6 / 35.2 at 1,500 words or more. Table 3 covers Qwen3-TTS-1.7B-CustomVoice. No temperature sweep (10 seeds per prompt at one sampler), no position or hazard analysis, no chunked baseline (checked in the HTML twice on 2026-11-09). Two lines bear on this card (Section 3.1 and 5, read 2026-11-09): only skips (10+ words) and hallucinations (20+ words) grow with length, while "other errors occur at the same rate in short- and long-form"; and VoxCPM2 failures beyond 1,300 words resist regeneration, which the authors call a coherence limit, a drift-like sign in another model. Code promised "upon publication". It ties some failures to long reference audio in voice cloning, which this design avoids with a preset speaker.
- [Qwen3-TTS technical report, arXiv 2601.15621](https://arxiv.org/abs/2601.15621). Its last pretraining stage raises the maximum token length from 8,192 to 32,768 and upsamples long speech. It gives no utterance duration distribution and no default sampler. Its long-speech test set runs 200 to 2,000 words (Table 10: 2.8 percent long-en WER for the 12Hz 1.7B), so the makers treat 2,000 words as in range. The duration that 32,768 tokens buys was not checked.
- [Reliable Neural-Codec TTS by ASR Self-Verification and Distillation, arXiv 2606.18323](https://arxiv.org/abs/2606.18323), 2026-06-16, HTML read 2026-11-09. Best-of-N with ASR checks on Llasa, Orpheus and CSM-1B (base failures 0.8 to 5.8 percent on LibriSpeech). Nothing by length or position, no temperature, no independence test.
- [Very Attentive Tacotron, arXiv 2410.22179](https://arxiv.org/abs/2410.22179) (NAACL 2025), read by a colleague. Error rises sharply beyond the training length.
- [Attention-Constrained Inference, arXiv 2404.19723](https://arxiv.org/abs/2404.19723), [attention guidance, arXiv 2509.19852](https://arxiv.org/abs/2509.19852) and [AlignDPO, arXiv 2609.12855](https://arxiv.org/abs/2609.12855) (2026-09-11, abstract). Alignment fixes for skips and repeats; aggregate rates on short sentences only.
- Temperature. [DASB, arXiv 2406.14294](https://arxiv.org/abs/2406.14294), Appendix H: for discrete-token TTS, dWER against temperature is roughly convex; at low temperature an early error is harder to recover from. [VALL-E 2, arXiv 2406.05370](https://arxiv.org/abs/2406.05370) (2024, abstract): small top-p causes infinite loops. [ECCD, arXiv 2608.00722](https://arxiv.org/abs/2608.00722) (2026-08-01, Table 1): CosyVoice2 test-en WER rose from 2.98 to 4.35 percent at T = 0.75. Total error is not monotone in T; loops likely rise as T falls.
- [MagpieTTS-LF, arXiv 2606.18485](https://arxiv.org/abs/2606.18485) (2026-06-16, HTML read). Stateful chunked generation; blames Qwen3-TTS errors on chunk boundaries and lost context. Overall WER only, no length buckets, no error position, T fixed at 0.7. Its Long-Form HiFiTTS benchmark has no data link.

Nothing found measures where in an utterance slips occur, how first-slip hazard moves with position, a like-for-like chunked control, or a per-type temperature law.

## Prediction

Written before any run.
- Step 1 decides on one number, H: first-slip hazard in the last third of long clips over the first third, per reference word, stratified by passage. Slip account: H between 0.67 and 1.5. Drift: H above 1.5.
- Step 2: R, long-arm hazard over pieces-arm hazard, lies between 0.67 and 1.5.
- Step 3: for skips plus garbles, log rate against 1/T beats the T and log T forms on held-out temperatures, and a fit on T at or above 1.0 predicts the rate at T = 0.7 within a factor of 3, erring low. Repeats plus run-ons do not fall with T; their rate at T = 0.7 is at or above their rate at T = 1.0.
- Step 4: within-text Weibull shape between 0.8 and 1.25; one failed draw does not raise the next draw's failure chance by more than 1.5 times.

## Definitions, fixed now

- **Slip types.** Skip: 3 or more consecutive reference words deleted. Repeat: 3 or more inserted words copying earlier text. Garble: 3 or more reference words where the character match is under 50 percent and mean CTC confidence is below the 1st percentile of real speech. Run-on: 3 or more inserted words that copy nothing, or audio continuing over 2 seconds past the last reference word. Early stop counts as a skip of the rest. Types are reported apart; H and R use all types, and skips plus repeats alone.
- **Estimator.** First-slip hazard per reference word. Exposure is the reference position reached at the first slip or the clip's end; each piece is censored on its own, by the same rule. Bootstrap unit: the passage.
- **Exclusion windows.** The first and last 5 reference words of each generation. H and R are reported raw and trimmed.
- **Seeds.** Seeds in each arm are independent replicates. No pairing across arms.

## Design

Model: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, preset English speaker "Aiden". The shipped sampler, confirmed from the checkpoint's `generation_config.json` on 2026-11-09: temperature 0.9, top_k 50, top_p 1.0, repetition penalty 1.05, subtalker temperature 0.9 and top_k 50. It also sets `max_new_tokens` 8192, which at about 12 tokens per second (if one token per frame) caps a clip near 11 minutes, roughly 1,600 to 1,800 words. Long arms raise this cap and log it, so a truncated clip is censored, never scored as an early stop. Every claim names its sampler.

Texts: consecutive public-domain book text, a list committed before Step 1. Real-speech checks join LibriSpeech test-clean recordings within a chapter.

Detector: `facebook/wav2vec2-base-960h`, character CTC with no language model (Whisper fills in skipped words). Greedy decode with timestamps, align to the reference by edit distance. LACI's thresholds (10 and 20 words) are a secondary count.

Workers exit before 280 seconds; a resume file records progress.

**Step 0, slice 1 (the first step; fits 30 GPU-minutes, 300 s per process, 8 GiB).**
1. Search gate: passed 2026-11-09 (Appendix B).
2. The model loads on MPS with sdpa or eager attention inside one 300-second process. Measure generation seconds per audio second and peak memory. If it cannot load, park the card.
3. Measure the longest clip one process can generate. If under about 1,500 words, build a save-and-resume harness (KV cache and RNG state to disk). It passes only if resumed tokens match an uninterrupted run exactly.
4. Detector validation: identity, planted skips and repeats, real-speech false alarms by position.
5. A pilot on 5 passages, never reused: slip rate per type under the shipped sampler and at T = 1.15 with top_k 50 kept.
6. Base-task gate, before any sizing. The hazard test assumes the model, as run here on MPS, speaks the text correctly most of the time, so that a slip is a rare departure from good output and not the normal state. Measure, under the shipped sampler, detector WER on 40 pieces of about 25 words from the pilot passages, beside the detector's WER on LibriSpeech test-clean. Bar: piece WER at most 10 percent and at most 3 times the real-speech WER (LACI reports 2.7 percent for this checkpoint under 500 words; the slack is for detector and device differences), and fewer than half of the pilot's long clips slip in their first third. If piece WER fails, listen to 5 pieces: broken audio parks the card as an MPS port problem; intelligible audio the detector misreads sends the card back to detector validation. If the first-third bar fails, the long clips leave no room for a last-third hazard to show; Step 1 uses the longest length at which it passes, or the card parks. Cost: about 1,000 words, roughly 7 audio minutes, inside Step 0's slice if generation is near real time (not checked); otherwise a second 30-minute slice.

Its outputs (throughput, maximum length, detector recovery, false alarms, base-task gate, pilot rates) size Steps 1 to 3. Nothing is sized until item 6 passes.

**Power, worked out before any run.** For a hazard ratio near 1 to have a 95 percent interval inside [0.67, 1.5], the standard error of its log must be about 0.2. Poisson counts give that with about 50 events in each compared group. This is arithmetic, not a measurement. Each step runs slices until it reaches its event target, and whether that target is affordable is settled by the stop rule in "What each step decides, and the stop rule" below.

**Step 1 (one arm, the opener).** Long clips at about 1,500 words, or the longest the harness allows, under the shipped sampler. Plot the Nelson-Aalen cumulative first-slip hazard against reference position, per passage, beside real speech through the detector. Decide on H. Stop when the first and last thirds each hold 50 events, or after 10 slices; at the cap, H is inconclusive and the slip rate is reported as an upper bound. Which design runs (first slips, all slips, or a raised temperature) is fixed by the stop rule below before Step 1 starts. If the long arm cannot reach LACI's range, report "no drift up to N words" only.

**Step 2 (pieces control).** Rules out text that gets harder late in passages. The same passages, cut at sentence boundaries into pieces of about 25 words, generated separately and joined, interleaved with new long runs. Decide on R. Report per-third ratios: third k of the long clip against the pieces covering the same words. Same stop rule.

**Step 3 (temperature law, needs Rob's approval for the device time).** Long arm, shipped top_k and repetition penalty, T at 0.7, 0.85, 1.0, 1.15 and 1.3, interleaved. Poisson fits of rate on 1/T, T and log T, compared by leave-one-temperature-out log-likelihood, run separately for skips plus garbles and for repeats plus run-ons (changed after the search: DASB and ECCD show total error rising at low T, so a pooled fit would mix two opposite trends). The headline is the extrapolation from T of 1.0 and above to T = 0.7. It needs at least 10 events at T = 0.7 to pin the rate within a factor of 3. One untruncated arm at T = 1.15 is reported apart. The control from the Arrhenius ledger entry: on shipped-sampler trajectories, compute in one pass the probability mass outside the top token at each frame, rescaled to each T. If that predicts the rate ordering as well as sampling, sampling is unnecessary.

**Step 4 (Weibull and independence).** 20 passages, 50 seeds each, at one committed sampler. Shared Weibull shape with a scale per passage. Test overdispersion across passages and dependence across draws. Add a LACI-style alignment-head detector, checked against CTC. Add clips beyond the 32,768-token context and report inside and outside apart.

**Step 5 (cloud; needs Rob's login and approval).** Qwen3-TTS-12Hz-1.7B and one non-Qwen model (Orpheus or CSM-1B).

## What each step decides, and the stop rule

Step 0 decides nothing about slips. Step 1 alone decides whether, under the named sampler and at the tested length, the slip hazard in the last third of long clips differs from the first third. H wholly above 1.5 refutes the hypothesis without naming the cause; text that gets harder late gives the same H, and Step 2 separates that from drift. H inside [0.67, 1.5] supports the slip account for this checkpoint, sampler and length only. Stopped at its cap, Step 1 gives only an upper bound on slips per hour.

**Sizing.** Take a constant hazard λ per reference word, a clip of L words with thirds of ℓ = (L − 10)/3 scored words, and x = λℓ expected slips per third. Each clip gives at most one first slip, and a clip that slipped early has left the last third, so the first-slip design needs

N₁ = 50 / [e^(−2x)(1 − e^(−x))] long generations,

never fewer than 338 (Appendix A). Device-minutes for N clips are D = N·L·a·(ρ·g + c)/60, with a the audio seconds per reference word, g and c the generation and detector seconds per audio second, and ρ the save-and-resume overhead (1 without the harness). Listening is N·f/3 minutes, f being flags per clip at 20 seconds each. Step 0 supplies λ under the shipped sampler, λ′ at T = 1.15, and a, g, ρ, c and f. If the pilot sees no slips in W words, any design needs at least 50·W generated words (Appendix A).

Budget per step: 10 slices of 30 device-minutes (300) and 2 hours of listening. At real-time generation that buys about 30 clips of 1,500 words (Appendix A), so the first-slip design fits only if generation here is about ten times faster than real time, which is not checked.

**Cheaper designs.**
- *All slips.* A recurrent-event Poisson model (passage strata, errors clustered by clip) counts every slip; exposure pauses 20 words after each and ends at an early stop. It needs N = 50/x clips, about 150/λ generated words. A covariate for "this clip has already slipped" keeps slips that cause slips from posing as drift. It saves much when slips are common, little when rare (Appendix A).
- *Raised temperature.* Skips plus garbles at T = 1.15, top_k 50 kept; cost falls by the pilot's ratio k = λ′/λ. The claim is then about T = 1.15. Carrying it to the shipped sampler rests on an assumption that is part of the claim: temperature scales the hazard by the same factor at every position. Step 3 tests the size of that factor, not its evenness, so a small shipped-sampler arm runs beside it.

**Stop rule, applied once after Step 0.** At the pilot's point estimates, cost the first-slip design under the shipped sampler, then all slips under the shipped sampler, then all slips at T = 1.15. Run the first that fits the budget, and record which. At half budget, if the projected count at the cap is under 35 per compared third, stop and report inconclusive. If none fits, Steps 1 to 4 do not run: the card is parked with the pilot's rate per type, or its upper bound, in slips per hour, and the device time a deciding run would need goes to Rob as a request. Step 2 is sized the same way.

## Checks before believing anything

Self-comparison at full size: resample the arm under test into two pseudo-arms, each the size of a real arm, and compute H (later R) between them, 1,000 times. That spread is the noise floor; a difference inside it is inconclusive. The detector run twice on the same audio must give identical slip times.

Planted effects made by the TTS: in a known 20 percent of clips, the input text has a 3 to 5 word span removed or duplicated; score against the unedited reference. The detector must recover 20 percent within 3 points, onsets within 0.5 seconds. Audio splices are secondary, as the detector may key on splice artifacts.

End-to-end planted drift: text edits at twice the base rate in the last third must give H of about 2. Simulated censored times with Weibull shape 1.0 and 2.0 must be recovered.

Real-speech false alarms must not vary by position beyond the noise floor.

Every flag counted in H or R is listened to, as a 20-second window around it.

## Kill criterion

H with its interval wholly above 1.5 at in-range lengths, or Step 2's R wholly above 1.5, or within-text Weibull shape above 1.3, kills the slip account for this model and sampler; calling the cause drift needs Step 2. Under the all-slips design, H is the intensity ratio with the already-slipped covariate in the model, against the same thresholds. If log rate of skips plus garbles against 1/T loses to either other form on held-out temperatures, the temperature law is dead. A U-shape in total error is already known and kills nothing.

## Cost on this hardware

Step 0: one slice, about 3.5 GB of downloads. Steps 1 and 2: at most 300 device-minutes and 2 hours of listening each, run only if the stop rule finds a design that fits. Steps 3 and 4: sized by the same formula. Step 5: cloud, not estimated. Keep tokens, delete WAVs if needed; free disk stays above 15 GB (about 33 GB on 2026-11-09).

## What a yes or a no changes

A yes makes failure a rate: predict any length from short tests, size best-of-N or rollback, report slips per hour beside WER. A no is direct evidence of degradation inside the range, favouring longer training clips. A temperature law gives an accelerated test for rare failures.

## What carries to larger models, and what stays untested

Position tracking is shared by larger codec decoders, and LACI sees the length effect at 1.7B too. This card tests one 0.6B checkpoint and one named sampler. Untested: other families, non-English, voice cloning, flow-matching models, other truncation settings.

## Receipts to keep

Each slice plan committed before its run. Per-clip records (passage, arm, seed, sampler, slip type and position, censoring, alignment), listening verdicts, check results, versions, revision hashes, and the analysis script with raw output.

## Review record

- Must-fix, 30-second clips too short for drift: fixed (about 1,500-word arm, resume harness).
- Must-fix, underpowered Step 1, garble at raised T, weak temperature law: fixed (event targets, garble class, held-out comparison, logit control).
- Should-fix, loose estimator, unlike pieces arm, generous self-comparison, easy planted effect: fixed (Definitions, Checks).
- Should-fix, prior art thin: fixed. Search gate passed 2026-11-09; LACI figures sourced from Table 2.
- Should-fix, MPS unproven: fixed by the Step 0 load gate.
- Minor: 1.7B claim kept (LACI Table 3); pilot leakage fixed; one-arm opener adopted as Step 1.

Reviewer's notes: the model card confirms "Aiden", with CUDA code and nothing on MPS; nothing is cached locally.

- 2026-11-09, Sol's cross-family review ([queue review](../reviews/2026-11-09-queue-review-sol.md)), "move down":
  - No affordable path to the event target, with attrition, ASR and listening cost: accepted; the floor is 338 long generations, not 100. Added the generation and device-minute formula from Step 0's quantities (Appendix A), with detector time and a 2-hour listening budget.
  - Step 0 decides nothing scientific: accepted. What Step 1 decides alone is now stated.
  - Added a stop rule at 300 device-minutes per step, two cheaper designs (all slips; T = 1.15 with its sampler link as part of the claim), and a park-and-report outcome. Search gate moved to Appendix B.
- 2026-11-10, base-task gate check (lesson from copies-as-evidence case 002, where the model was at chance and the test was void): the stop rule already gated slip rate against budget, but nothing confirmed the model produces correct speech on MPS before slips are counted, nor that long clips leave room in the last third. Added Step 0 item 6 with a WER bar, a first-third bar and park or redirect outcomes. Risk without it: medium.

## Appendix A: sizing arithmetic

Constant hazard λ per scored word, thirds of ℓ words, x = λℓ. A generation's first slip falls in the first third with probability p₁ = 1 − e^(−x) and in the last third with p₃ = e^(−2x)(1 − e^(−x)). Setting the derivative of p₃ to zero gives e^(−x) = 2/3, so p₃ is at most 4/27 at x = ln 1.5 ≈ 0.41. Counting all slips, each third holds about N·x events, so N = 50/x. Rounded up, 50 events in the last third need:

| x (slips per third) | first-slip design, 50/p₃ | all-slips design, 50/x |
|---|---|---|
| 0.02 | 2,629 | 2,500 |
| 0.05 | 1,134 | 1,000 |
| 0.1 | 642 | 500 |
| 0.2 | 412 | 250 |
| 0.41 | 338 | 124 |
| 1.0 | 585 | 50 |

Total generated words for the all-slips design are about 3·50/λ = 150/λ, whatever the clip length, which is why rare slips stay expensive under any counting. With zero pilot events in W words, the Poisson 95 percent upper bound is λ < 3/W, so 150/λ > 50·W. All of this assumes the null (flat hazard); under drift the last third gains events and the first third becomes the scarcer, so the plan checks both thirds against the target.

Budget illustration, not a measurement: at L = 1,500, a = 0.4 (about 150 words a minute, the rate the 11-minute cap implies) and g = 1, one clip is about 10 audio minutes and so about 10 device-minutes before detection; 300 device-minutes buy about 30 clips.

## Appendix B: search gate, 2026-11-09

Two passes: 5 searches refused and 15 direct fetches, then 12 searches in TTS, LLM survival-analysis, phase-locked-loop and reliability wording (error position, sampler loops, temperature against WER, chunked against whole-passage, LACI follow-ups, failures per hour, error accumulation). Read: LACI Sections 3.1 and 5, DASB Appendix H, ECCD Table 1, MagpieTTS-LF in full; abstracts of AlignDPO, VALL-E 2 and Diagnose-Then-Refine (2608.28970, prosody only).

No scoop. Nearest: LACI (length, not position), MagpieTTS-LF (chunking, no position), survival analysis of multi-turn LLM dialogue (turns, not speech). No released long-form set with a data link. Design change: Step 3 fits the temperature law per slip type; Step 1 reports by type. Still open, low risk: papers citing LACI (none found), Interspeech 2026 proceedings, OpenReview.
