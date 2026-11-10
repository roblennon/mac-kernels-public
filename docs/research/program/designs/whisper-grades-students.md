# Is Whisper grading its own students? Evaluator-training coupling in TTS word error rate

Status: queued, 2026-11-09. Search gate passed on the third pass (12 queries, none refused). Partly scooped: a 2024 German study ran the label swap with the labeler and one outside judge but never analysed the interaction; its published means lean our way. Step 1 and the English, multi-judge step 3 remain open.

Ledger idea `whisper-grades-students`. Designed, not run. Revised after a hostile review; see "Review record".

Many text-to-speech papers report intelligibility as word error rate (WER) under Whisper large-v3. The largest open TTS corpora, Emilia above all, were transcribed and filtered by a Whisper-family model. If a TTS model trained on Whisper's transcripts gets an easier score from a Whisper judge, the standard number is partly circular, and gaps between Emilia-trained and human-transcript-trained models are biased in a known direction. A yes would change the protocol. A clean no would let the field keep it with a citation behind it.

## Hypothesis

A TTS model trained on Whisper-family transcripts gains more under a Whisper judge, relative to an outside judge, than the same architecture trained on human transcripts does. The gain is ordered by lineage: largest for the exact labeler, then the protocol judge, then a Whisper student, and absent for outside judges. The hypothesis is wrong if the interaction sits inside the noise floor, or if outside judges show the same gain.

## Theory

Training labels carry the labeler's habits: its spelling, its dropped disfluencies, the words it hears in place of rare ones. Where a label disagrees with the audio, the TTS model learns speech that sounds like the label, and a same-family judge is the likeliest listener to hear it that way. Filtering adds a second route: Emilia-Pipe keeps clips Whisper already finds easy.

Assumptions said out loud. Label and selection routes separate only by training (step 3). Outside judges are outside only if their cards say so; each is checked before step 1. Domain match is a rival explanation, so LibriSpeech-trained judges never decide. Judge skill may scale rather than add, so the deciding metric uses log-odds.

## What is already known

One paper ran our step 3a almost exactly, in German and without the analysis. Nothing found tests lineage with an interaction, an interval or more than one outside judge, or on Emilia-scale English TTS.

- [Comparing Human-Labeled and AI-Labeled Speech Datasets for TTS](https://papers.academic-conferences.org/index.php/icair/article/view/3030), Wirth and Peinl, ICAIR 2024. VITS on Thorsten Voice (German, one speaker, about 11 h, hand-labelled), trained on human, Whisper-large-v3 and phone-ASR labels, one run each. Text from 12 German test sets, judged by Whisper-large-v3 (the labeler) and NVIDIA's STT De Conformer-Transducer Large. Mean WER, human vs Whisper labels: 14.64 vs 16.21 under Whisper, 11.62 vs 14.24 under the Conformer. The authors conclude AI labels are fine and do not test lineage. Our arithmetic on their tables: the Whisper-label penalty is 1.57 points under the labeler and 2.62 under the outside judge, log-odds 0.12 vs 0.23, so I_log ≈ +0.11; per test set, 9 of 12 in that direction. The largest gaps are on M-AILABS and Thorsten, audiobook and in-domain sets, so domain may carry part. One seed, no intervals, one outside judge with unchecked training data.
- [Best-of-N TTS Evaluation is Confounded by ASR Family Alignment](https://arxiv.org/abs/2607.08256), 2026-07, ICML 2026 audio workshop. On LibriSpeech-PC test-clean (1,127 utterances), a Distil-Whisper verifier recovers 26.0% of oracle headroom under faster-whisper large-v3 against 7.9% for wav2vec2; under wav2vec2-lv60 the same-family edge is 26.1% vs 18.2%. They name lineage-level coupling and leave it to future work. Their TTS, F5-TTS, is Emilia-trained ([paper](https://arxiv.org/html/2410.06885)), so their larger Whisper-side edge may carry our effect, unseparated (our reading).
- [Measuring the Verifier's Floor](https://arxiv.org/html/2610.31470v1), 2026-10-22. Whisper large-v3-turbo, wav2vec2-large and OWSM v3.1 miss 33–67% of very rare words in human speech; F5-TTS and Kokoro. No discussion of TTS training labels.
- F5-TTS Table 8 gives the one public number near ours: on test-clean, Whisper/HuBERT WER is 2.23/1.89 for human audio and 2.11/1.81 for Emilia-trained F5-TTS (32 NFE, BigVGAN), ratios 1.18 and 1.17. No visible Emilia-specific Whisper advantage, but HuBERT is in-domain and there is no human-transcript-trained arm.
- [Preference Leakage](https://arxiv.org/abs/2502.01534). The same effect in text LLMs; template for step 3.
- [Emilia](https://arxiv.org/html/2407.05361). Labels with Whisper-medium through WhisperX.
- [Granary](https://arxiv.org/html/2505.13404) (read by the reviewer). Pseudo-labels from Whisper-large-v3. Parakeet-tdt-0.6b-v3's card lists 660,000 Granary hours against 10,000 human hours, so Parakeet v3 is mostly a Whisper student.
- [ZipVoice](https://arxiv.org/html/2506.13053) (read by the reviewer). Scores test-clean with a HuBERT ASR and Seed-TTS test-en with Whisper-large-v3. WER 1.64 (Emilia model) vs 1.69 (LibriTTS model). The checkpoints also differ in updates, batch size, token type and reported NFE.

## Prediction

Written before any run. Priors, not results.

**Deciding metric.** For source s and judge j, lo(s, j) is the log-odds that a reference word is wrong (substitution or deletion; insertions reported apart). g(s) = lo(s, outside panel mean) − lo(s, Whisper large-v3). I_log = g(Emilia model) − g(LibriTTS model) on Seed-TTS test-en, with a paired bootstrap over utterances. Absolute I in WER points is secondary, always printed beside the base-WER gap between the two models.

- Step 1: I_log positive, absolute I between 0.3 and 1.0 points. Step 1 is an observation that motivates step 3, not evidence for lineage.
- Dose ordering, I_log with each rung in the Whisper slot: labeler (Whisper-medium) ≥ protocol judge (large-v3) ≥ Whisper student (Parakeet v3) > outside, near zero. It counts against lineage if the outside rung's gain matches the Whisper rungs, or the rungs do not separate. Gaps among the three Whisper rungs are weak evidence; they share training data.
- Domain alone predicts I_log positive under in-domain judges (wav2vec2-960h, HuBERT-ls960, OWSM v3.1) on test-clean, and under any Common Voice-trained judge, OWSM included, on Seed-TTS, and near zero under the outside panel on Seed-TTS. Lineage predicts positive under the outside panel on both.
- Human recordings do not enter I. They give each judge's odds multiplier against large-v3 on real speech; it should roughly match the multiplier on the LibriTTS model.
- Step 3: absolute I between 0.2 and 0.8 points for the label swap; selection carries under half. Wirth and Peinl's means give about 1.0 point (0.11 log-odds) for 3a's analogue, unreplicated and partly domain-driven; this prior stays.

## Design

**Audio sets.** Main: Seed-TTS test-en, 1,000 Common Voice utterances per the [seed-tts-eval README](https://github.com/BytedanceSpeech/seed-tts-eval) (licence not stated there), where TTS papers actually judge with Whisper-large-v3. Second: the LibriSpeech-PC test-clean 4 to 10 s subset (1,127 utterances), judged by HuBERT in the protocol.

**TTS models.** `k2-fsa/ZipVoice`: `zipvoice` (Emilia) and `zipvoice_libritts`, 0.49 GB each. Same NFE, guidance and sampler for both, logged. Prompt text is the dataset's ground truth, never a recognizer output, in both arms. Step 2 adds F5-TTS, StyleTTS 2 and Kokoro.

**Judges by rung.** Each judge's label source is checked on its card before step 1 and logged.
- Labeler: Whisper-medium MLX (1.52 GB).
- Protocol: `Systran/faster-whisper-large-v3` (3.09 GB), the protocol's own implementation, with MLX large-v3 (cached) as a cross-check.
- Whisper student: Parakeet-TDT 0.6B v3 int8 (cached).
- Outside panel, deciding: one CTC model fine-tuned only on human-labelled speech that is neither audiobook nor Common Voice, such as TED-LIUM or VoxPopuli (chosen in step 0a), plus a second judge of the same kind if one exists, so the panel is not a single model. Seed-TTS test-en is Common Voice, so Common Voice judges are in-domain there. `espnet/owsm_v3.1_ebf` (4.07 GB) trains on CommonVoice and LibriSpeech ([paper](https://arxiv.org/html/2401.16658)), so it moved to in-domain on 2026-11-09. If a judge fails its check, the panel shrinks and the write-up says so.
- In-domain, never deciding: OWSM v3.1 (above), wav2vec2-large-960h-lv60-self (1.26 GB) and HuBERT-large-ft-ls960, the test-clean protocol judge (size not checked).

One Whisper `EnglishTextNormalizer` for everything. Greedy, temperature 0, no prompt for Whisper judges.

**Step 0a, no device time. Done 2026-11-09** (see "Search gate"). Partly scooped by Wirth and Peinl; the card narrows rather than drops. Left over, cheap: citers of Wirth and Peinl, and the training data of their Conformer judge.

**Step 0b, one slice: generation.** ZipVoice lists CPU and ONNX paths; MPS not checked. Time 10 clips per checkpoint on each path that works, extrapolate, then generate in 300 s shards, cache and hash every clip. If generation exceeds a slice, cut utterances first, seeds second, and record the cut here.

**Step 0c: power pilot.** On 50 Seed-TTS utterances, run the Emilia model's two seeds through the full analysis and read the SE of I. Size N so the 95% half-width on absolute I is 0.5 points or less. If that needs every utterance, keep two seeds for the Emilia model only and one for the LibriTTS model, rather than shrink N. If no N that fits three slices reaches 0.5 points, park step 1 and record it as unable to test, not null; this sizing decision is the resolution bar, and the subject gate does not repeat it. Time each judge including model load.

**Step 0d: Subject gate, on 0c's pilot.** The plants test the instrument; this tests the subjects. On the 0c clips plus the 50 human recordings, before step 1:
- Each ZipVoice checkpoint has WER under 15% under every deciding judge (large-v3 and each outside judge), and fewer than 3 of 50 clips at 50% WER or worse under all judges. Above that, generation failures swamp label habits.
- Each outside judge's WER on the human recordings is under 10% and at most 3× large-v3's. A judge that cannot transcribe clean Common Voice audio cannot show a Whisper-specific advantage.
- Error mass: the pilot error rate, extrapolated to the N that 0c chose, gives at least 100 substitution-plus-deletion errors per model under large-v3. Fewer, and a 0.3 to 1.0 point I is a handful of words.

If a checkpoint fails, recheck prompt text, NFE and sampler once, then park step 1. If an outside judge fails, drop it; with no outside judge left, park, since the panel decides. If error mass fails, park step 1 and record it as unable to test, not null. Cost: none beyond 0c's slice, since it reuses 0c's clips and judge passes; the human recordings add one pass per judge on 50 clips, inside the 300 s, 8 GiB per-process limits. Bars are set before the pilot; no TTS or judge WER on Seed-TTS has been measured here.

**Step 1, one to three slices by measured timing. Runs only if 0d passes.** Seed-TTS test-en at that N, then test-clean if time remains. Sources: human, ZipVoice-Emilia, ZipVoice-LibriTTS, interleaved within each judge pass. One process per judge per shard, under 300 s and 8 GiB.

**Step 2, device only.** Full sets, extra TTS models, word-frequency bins, adjudication as in the verifier's-floor paper, and whether any pairwise ranking flips across judges.

**Step 3, needs Rob's approval.** 3a: Wirth and Peinl's design in English, made decisive: a VITS-scale model on a single-speaker human-transcribed set, trained on the same clips with human vs Whisper-medium labels, two seeds per arm, judged by every rung with I_log and a bootstrap, not one outside judge and a table of means. Their +0.11 log-odds, 9 of 12 sets, is the number to beat or explain. Cost not checked; 300 s step-time pilot first. 3b: ZipVoice on the LibriTTS recipe in a 2×2, labels {book text, Whisper-medium} × selection {all, Emilia-style filter}, the filter choosing the same clips for both label arms; likely cloud.

## Checks before believing anything

- **Noise floor.** Seed 0 against seed 1 of the Emilia model, analysed as two models, must give I_log inside its interval around zero. Human recordings split in halves compare different utterances, so they only check that judge multipliers are stable.
- **Planted label effect, signed.** In the Emilia model's large-v3 transcripts, replace words that were correct with other dictionary words at 1 per 100 reference words. That raises its large-v3 error by exactly that rate, lowers g(Emilia), and must lower absolute I by 1.0 point. The expected I_log shift is computed from the planted counts beforehand and must be recovered. A 0.3-point plant tests power; if missed, step 1 is called underpowered, not null.
- **Planted acoustic effect.** In a fixed 10% of Emilia-model clips, overwrite one word with noise at its forced-alignment span. Every judge's error must rise by about one word per clip, and I_log must stay inside the noise floor, since the change hits all judges alike.

## The control that could embarrass it

The outside panel on Seed-TTS test-en. If the LibriTTS model shows the same Whisper advantage as the Emilia model under human-labelled, non-audiobook judges, the effect is a TTS-wide habit, not lineage.

## Confounds a hostile reader would raise

- *The checkpoints differ in data, updates, batch and token type.* True; token type bears on rare words, the proposed mechanism. Step 1 is an observation; step 3 holds audio fixed.
- *Domain match.* In-domain judges never decide; the main set is not LibriVox.
- *Scale, not lineage.* Log-odds metric, base-WER gap printed beside I.
- *The human side was recognizer-filtered too.* Confirmed: the [LibriTTS paper](https://ar5iv.labs.arxiv.org/html/1904.02882) ran Google Cloud Speech-to-Text and a modified YouTube Auto-sync aligner, and kept only sentences that aligned with zero edit distance. That is a non-Whisper recognizer, so it biases toward "easy for ASR" in general, not toward Whisper. Step 3 holds selection fixed.
- *Normalization or insertions.* One normalizer, top error words per judge, insertions apart, identical silence trimming.

## Kill criterion

Stop after 0a if prior work answers it. Park, not stop, if 0c finds no N that resolves 0.5 points or step 0d fails. Stop after step 1 if both label plants are recovered at the right sign and size, I_log sits inside the noise floor, and Whisper and outside rungs do not separate. Stop after step 3 if the label-swap I is below 0.2 points or covers zero across seeds and no ranking flips in step 2. Then the protocol is fair in this respect, and that is the paper.

## Cost on this hardware

Step 0a: none. Step 0b: one slice; generation speed is what it measures. Steps 0c and 1: about 11 GB of known downloads plus HuBERT, the second outside judge and Seed-TTS, sizes not checked. With 31 GB free and a 15 GB floor that is close; delete each judge's weights after its passes if needed. Judge time comes from step 0c, likely one to three slices. Step 2: about 6 GB more, several slices. Step 3: Rob's approval; 3b likely cloud.

## What a yes and a no change

A yes means TTS papers should report WER under at least one outside judge plus a human floor, and Emilia-vs-LibriTTS comparisons need re-reading. It extends Preference Leakage to speech with a causal design. A no, with plants recovered, is a citable clearance.

## Why it should carry to larger models

The mechanism lives in the data pipeline, not model size. Untested: models above about 1B parameters, other languages (Wirth and Peinl hint German carries), the pseudo-labelled share of data, WavLM speaker similarity, and CLIP judges of captioner-labelled images.

## Receipts to keep

In the lab's run folder for this card: the 0a query log; utterance IDs, seeds, prompt map; model and judge hashes with label-source notes; clip hashes and raw transcripts; normalizer version, per-word error table, bootstrap script and outputs for I_log, absolute I, dose ordering and all three plants; noise-floor readings and timing logs.

## Search gate, 2026-11-09

Passes one and two: queries run 0, refused 6 for budget; direct fetches only.

Pass three: queries run 12, refused 0. Wordings covered pseudo-label and same-labeler bias, ASR-for-TTS training labels, Seed-TTS-eval judge choice, preference leakage in speech, circular labelling and judging, multi-ASR disagreement, Emilia label noise, ASR-filtered selection, and corpora with both human and Whisper transcripts. A Semantic Scholar citation fetch returned HTTP 429.

New closest works, pass three:
- [Wirth and Peinl](https://papers.academic-conferences.org/index.php/icair/article/view/3030), ICAIR 2024 (PDF created 2024-12-03). Full text read, both results tables. Partial scoop; see "What is already known".
- [LibriTTS](https://ar5iv.labs.arxiv.org/html/1904.02882), 2019. Data-processing section read. Corrects the confound entry.
- [PashtoTTS-Bench](https://arxiv.org/abs/2605.26978), 2026-05-26. Abstract only. Uses an "independent" ASR judge without checking lineage. Not a scoop.
- Abstracts or snippets only, not scoops: [Raon-OpenTTS](https://arxiv.org/abs/2605.20830) (2026-05), [2605.12107](https://arxiv.org/abs/2605.12107) (speech enhancement), FlexSpeech [2505.05159](https://arxiv.org/pdf/2505.05159) (Emilia label noise, qualitative).

Earlier closest works still stand: 2607.08256 (2026-07-09) and 2610.31470 (2026-10-22), main sections read; F5-TTS, data and Appendix B.6 read; [2606.18323](https://arxiv.org/abs/2606.18323), Whisper round-trip self-verification, partly read; 2608.10606, 2510.06927, 2608.00545, abstracts only.

Design changes: earlier passes made OWSM v3.1 in-domain and restricted the outside panel to non-audiobook, non-Common Voice judges. Pass three makes step 3a a multi-judge, multi-seed English replication of Wirth and Peinl, and names LibriTTS's real recognizer.

Still open: citers of Wirth and Peinl (Semantic Scholar rate-limited); the training data of their Conformer judge; OpenReview and Hugging Face papers listings; Seed-TTS test-en licence; a second outside judge. None blocks step 0b.

## Review record

- Must-fix, all addressed: neutral panel; the real protocol; planted-effect sign, plus an acoustic plant; scale artifact (log-odds I_log decides); cost (generation is step 0b); no search (step 0a, done 2026-11-09).
- Should-fix, all addressed: pilot-sized N; matched NFE and sampler, token type named; ground-truth prompt text; human-side selection left to step 3; four-rung dose ordering.
- Minor: step 3 split into 3a and 3b; "grows with" dropped; seed pair is the noise floor.
- 2026-11-10, base-task gate pass (after copies-as-evidence step 1a ran a model that got no greedy base-task answer right: one template inaccurate, the other off-format): no subject-side gate existed; 0c measured SE but had no bar or stop, and nothing checked TTS intelligibility, outside-judge competence on Common Voice audio, or error mass. Added step 0d with bars and a park rule. Risk the step 1 test was void without it: medium.
- 2026-11-10, cross-family gate review (Sol): the resolution bar moved out of step 0d into 0c's sizing decision, where N was already chosen, so it is read once with no second run or approval; 0d is renamed Subject gate and keeps the intelligibility, judge-competence and error-mass bars.

The reviewer's four searches were refused for budget; its fetches are folded into "What is already known".
