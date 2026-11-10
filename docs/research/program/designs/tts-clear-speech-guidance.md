# The guidance scale is a clear-speech dial: guided flow-matching TTS over-articulates the way people do for a listener

Status: queued, 2026-11-09. Search gate passed on the sixth pass: no work measures vowel area or SRT against guidance scale. The text/prompt split of guidance is not new (MegaTTS 3, DualSpeech, Zheng and Maleki), so the novelty is the clarity measurement, not the decomposition. Two cited-by lists stayed rate-limited; recheck them when writing `plan.md`.

F5-TTS samples with classifier-free guidance and tends to sound like careful reading. Does more guidance widen the vowel space beyond the talker's own reading, does the text part cause it, and does it help recognisers in noise? If yes, guidance is already the clarity control recent papers add by training or steering.

## Hypothesis

Two claims, each with its own kill line.

- Mechanism claim. In F5-TTS, raising guidance widens the corner-vowel triangle (/i/, /ɑ/, /u/) in F1–F2 space, and the widening comes from the text residual, not the prompt residual. At s = 2 the area exceeds the prompt talker's own read-speech area.
- Practical claim. Raising guidance from s = 0.5 to s = 2 lowers the babble level at which speech recognisers get half the words right (SRT50) by at least 1 dB. This is machine intelligibility only.

## Theory

F5-TTS guides as v = v_c + s·(v_c − v_u), with s = `cfg_strength` (default 2). The reviewer confirmed this in `cfm.py` on GitHub; the agent rechecks the installed source.

The unconditional output averages over every text, so at a vowel it should be centralised. Guidance pushes each vowel away from that centre, expanding the vowel space, the main acoustic mark of human clear speech. F5-TTS fixes duration before sampling, so guidance cannot slow speech down.

Per the reviewer's reading of `cfm.py`, training drops text and audio together (cond_drop_prob 0.2) or audio alone (audio_drop_prob 0.3), never text alone. So (no text, audio) is out of distribution. The agent confirms this in source. Residuals use trained branches only:

- Text residual: R_text = v(text, no audio) − v(no text, no audio).
- Prompt residual: R_prompt = v(text, audio) − v(text, no audio).

They sum to F5-TTS's full residual. "Prompt", not "speaker": the audio prompt also carries rate, prosody and recording conditions. Arms start from v(text, audio) and add s times R_text + R_prompt (full, F5-TTS's own), R_text alone (text-only, three branches per step), or R_prompt alone (prompt-only, two branches). Any arm using (no text, audio), such as Shi et al.'s four-branch split, is exploratory only.

Untested assumption: the effect survives the Vocos vocoder.

## What is already known

No paper varies guidance against vowel area, formants or SRT. Akti and Waibel's arXiv author listing (19 entries, newest 2026-10-06) has no guidance paper.

Guidance split into text and prompt parts from trained branches is prior art, measured only on SIM, WER and MOS:

- [Jiang et al., MegaTTS 3](https://arxiv.org/abs/2502.18924), 2025. Separate α_txt and α_spk, branches (all), (text, no speaker), (none): the same split as R_text and R_prompt here. Appendix K sweeps α_txt 1 to 6 at α_spk 2.5 and describes, without numbers, pronunciation moving from improper (1 to 1.5) to the speaker's accent (1.5 to 2.5) to standard (above 4). Nearest to the mechanism claim; qualitative.
- [Yang et al., DualSpeech](https://arxiv.org/abs/2408.14423), Interspeech 2024. Separate text and speaker scales; text-heavy (4, 1) against speaker-heavy (1, 4) raises PMOS 3.36 to 3.83 and drops SMOS 3.74 to 2.35. No acoustic analysis.
- [Zheng and Maleki, Selective CFG](https://arxiv.org/abs/2509.19668), v2 2026-03-24. On F5-TTS, "input_text" is this card's prompt-only arm; "def_text" uses full guidance early and prompt-only after flow time 0.08 (about step 9 of 32). SIM and WER only.
- [Wang, Sun and Baayen, Not Flat, But Dissociated](https://papers.cool/venue/wang26ea@interspeech_2026@ISCA), Interspeech 2026 (abstract only). Four older non-guided systems have vowel spaces 9 to 30% of the human baseline. A strong prior against "s = 2 exceeds the talker's reading".
- Practitioner claim, untested: a ComfyUI F5-TTS node's docs say cfg_strength 3 to 5 gives "crisper, more articulated speech".

Earlier reads, still current:

- [Akti and Waibel, Loud and Clear](https://arxiv.org/abs/2610.07647), 2026. Activation steering of Qwen3-TTS (autoregressive): vowel space area and Whisper-medium WER in babble at 1, 5, 10 dB, no SRT. Area falls for German (9.05 to 4.00).
- [Akti and Waibel, Synthesizing the Lombard Effect](https://arxiv.org/abs/2606.23176), Interspeech 2026. Trained articulation and effort controls in Matcha-TTS; mean vowel dispersion rises with articulation; Whisper-medium WER at 1, 5, 10 dB.
- [Akti and Waibel, Zero-Shot Lombard](https://arxiv.org/abs/2601.12966), SLT 2026. F5-TTS with style-embedding directions; Whisper large-v3 WER in babble; no vowel measure.
- [Shi et al., Joint Residual Reweighting](https://arxiv.org/abs/2606.25672), 2026. Four branches, S using the (no text, audio) branch; F5-TTS at (g_T, g_S, g_I) = (2, 3, 4.5); SIM and WER only.
- [Chen et al., F5-TTS](https://arxiv.org/abs/2410.06885), 2024. Default strength 2, sway −1; guidance not ablated.

None of the Akti and Waibel papers mentions guidance.

Required before step 1, written into `plan.md` with a date and a verdict: the two cited-by lists still open in the search gate section below.

A hit that measures vowel area or SRT against guidance scale moves this card to parked.

## Prediction

Written 2026-11-09, before any run.

- Full-guidance area, matched correct tokens: grows at least 10% from s = 0.5 to s = 2 in each voice.
- At s = 2 the area exceeds the talker's own default read-speech area beyond the band. This is the prediction most likely to fail, given Wang, Sun and Baayen's 9 to 30% for older systems. Whether it reaches the talker's enunciated area is reported, not predicted.
- Text-only at s = 2 gives at least twice the area growth of prompt-only at s = 2.
- Pitch range (5th to 95th percentile of F0, semitones) grows by at least 1 semitone from s = 0.5 to s = 2 (secondary).
- Late-only guidance (flow time t > 0.3) gives at least two-thirds of full growth; early-only less than one-third.
- SRT50 improves 1 to 2 dB from s = 0.5 to s = 2, same sign under both recognisers.

## Design

Model: F5-TTS v1 Base (`F5TTS_v1_Base`), PyTorch on MPS, Vocos vocoder, 32 steps; size not checked. Phoneme check: `facebook/wav2vec2-xlsr-53-espeak-cv-ft` emissions through `torchaudio.functional.forced_align` with the known phonemes; the posterior at each target vowel judges it. Formants and F0: `praat-parselmouth`. Voice prompts: 5–8 s clips with transcripts, from Expresso default read speech in step 1 and LibriSpeech test-clean after (sizes not checked), half female, listed in the plan.

Every arm of an item reuses the same initial noise; arms run interleaved. Each process saves and exits under 300 s and a driver resumes; the slice budget includes the model-load time each process pays.

Deciding metric: per voice, the ratio of corner-vowel triangle area (Hz, F1–F2 at the vowel's middle third, median over tokens per vowel) between two arms, on tokens judged correct in both. Analysis: paired bootstrap over items and seeds, 95% interval of the log ratio per voice.

**Step 1 (one lab slice; no cloud).** Words heed, hod, who'd in fixed carrier frames ("Say ___ again", "I said ___ twice", and so on). The /æ/ word "had" is dropped: the triangle does not use it. Two voices (one female, one male), two seeds. Arms per item: s = 0; full at s = 0.5 and s = 2; text-only s = 2; prompt-only s = 2; and full s = 2 on a second noise draw (the null). That is 2 voices × 3 words × F frames × 2 seeds × 6 arms. F is set from the pilot below, starting at 8 (576 generations).

Human reference, CPU only (changed 2026-11-09): the two step-1 voices come from [Expresso](https://speechbot.github.io/expresso) (4 North American speakers, CC BY-NC 4.0, read styles include "default" and about 116 minutes of "enunciated"). Prompts are cut from default read speech. Each talker's stressed /i/, /ɑ/, /u/ are aligned in both styles: a reading floor and a clear-speech ceiling from the same person, and the fraction of that gap s = 2 closes is reported. Sentence pairing and download size not checked; fallback is LibriSpeech reading. Word contexts differ from hVd, reported as a limit.

Pilot first: 10 items through every arm, recording generation and load time. Estimate the band's half-width from the pilot's null arm; if over 5%, raise F and log it. If the run then exceeds 20 device-minutes, run one voice per slice; one voice still answers every step-1 question.

Step-1 outcomes fixed in advance. Growth under 5% in both voices with a band under 5%: killed. Growth inside the band: inconclusive. s = 2 area not above the talker's reading, or most growth between s = 0 and 0.5: downgraded to "less mumbling". Prompt-only at least 80% of text-only growth: mechanism killed.

**Step 2 (2–3 slices).** Flow-time sweep and wider coverage. The plan lists each step's flow time under the sway setting used (paper: −1; code default not checked). Early-only and late-only arms are matched on guidance integrated over flow time; matched step count is also reported. One no-sway run repeats the flow-time arms. Five voices, 40 Harvard sentences, plus the step-1 set, with text-only and prompt-only arms. Zheng and Maleki's def_text (full early, prompt-only after flow time 0.08) runs as one more arm. Exploratory: Shi et al.'s four-branch split at their published F5-TTS weights (2, 3, 4.5), labelled as out of distribution. Mean vowel dispersion is reported beside area, to compare with Synthesizing the Lombard Effect.

**Step 3 (3–5 slices).** Machine intelligibility in noise. 100 Harvard sentences, 5 voices, s ∈ {0.5, 1, 2, 3}. Babble from 8 summed LibriSpeech talkers not used as prompts. SNRs −5 to 10 dB, including 1, 5 and 10. Recognisers: Whisper large-v3-turbo and Parakeet, both local. Logistic fit per voice and arm; report SRT50 and dB per unit of s, plus WER at 1, 5 and 10 dB so the result lines up with Akti and Waibel's tables (changed 2026-11-09).

**Step 4 (needs Rob).** A second guided model, E2-TTS from the same code base, which the carry-over claim requires. Optional: a word-recognition-in-babble listening test, the only route to a human claim; with one listener it is a pilot.

## Checks before believing anything

Determinism: regenerate 5 items at s = 2 on the same noise and record whether MPS output is bit-identical, and if not, by how much.

Paired null: full s = 2 on noise draw A against draw B, same items, same estimator. Its interval must contain 1; its half-width is the inconclusive band. The band is conservative.

Plant 1, formant scaling: Praat "Change gender" at ratio 1.00 is the plant's own null, then at 1.05. The planted audio goes through the whole pipeline, phoneme check included. The area ratio of 1.05 against 1.00 must be 1.1025 ± 0.03.

Plant 2, the confound: apply to s = 0.5 outputs a tilt change and added noise sized to span the measured s = 0.5 to s = 2 difference, formants unchanged. Area must stay inside the band; tokens passing the phoneme check are reported.

Step 3 plant: +2 dB to one arm's speech after mixing must move SRT50 by 2.0 ± 0.5 dB.

## The control that could embarrass it

Both in step 1: prompt-only guidance from trained branches (if it matches text-only, this is sharpening), and the talker's own reading (if s = 2 does not exceed it, this is less mumbling).

## Confounds a hostile reader would raise

- Low-guidance mispronunciations: only tokens correct in both arms count; dropped counts reported.
- Tracker bias: plant 2, plus 5000 Hz and 5500 Hz formant ceilings, result must hold under both.
- Slower speech: duration is fixed; vowel durations reported. Louder or brighter: level normalised, tilt reported.
- Prompt drift: within-voice comparison; speaker similarity in step 2. One recogniser's prior: two recognisers.

## Kill criterion

Mechanism claim dies if prompt-only gives at least 80% of text-only growth, or if s = 2 area does not exceed the talker's own read-speech area. Practical claim dies if SRT50 moves less than 1 dB from s = 0.5 to s = 2, or if Whisper and Parakeet disagree in sign. A result inside the band at two voices is inconclusive, not a kill. The flow-time result can stand alone.

## Cost on this hardware

Step 1: up to 25 device-minutes including loads (speed not measured); about 1.5 hours wall time; 3–4 GB downloads plus Expresso, not checked. Step 2: about 2,000 generations, 2–3 slices. Step 3: about 2,000 generations plus recognition, 3–5 slices. Step 4: E2-TTS download. Disk stays under 10 GB.

## What a yes changes and what a no changes

Yes: a free setting that helps recognisers in noise, and a dB baseline that trained and steered clarity controls must beat. No: guidance in speech is sharpening or de-mumbling.

## Why it should carry to larger models, and what it leaves untested

It carries only to models whose unconditional branch drops text, and a general statement needs E2-TTS to agree. Untested: unguided autoregressive TTS, CosyVoice2, other languages, larger checkpoints, human listeners, and listeners with hearing loss.

## Receipts to keep

In the lab repository under `results/tts-clear-speech-guidance/`:

`plan.md` (committed before step 1, dated prior-art verdict and changes underneath); `env.json` (versions, hashes, guidance formula, dropout settings, sway coefficient, step times); `gen/` and `noise/` by voice, item, seed, arm; `align.csv` and `formants.csv` per token; `checks.json`; `area_by_voice.csv` and `srt50.csv` with intervals and scripts. The first run is an "unconfirmed observation" until it recurs on new voices and seeds.

## Search gate, 2026-11-09

Earlier passes today: at least 16 web searches, all refused by a shared budget; the five "earlier reads" were fetched directly. Cited-by found nothing new (Zero-Shot Lombard's list came back empty and incomplete).

Sixth pass: 11 web searches run, 0 refused, worded for TTS, phonetics and hearing communities (guidance with vowel space, hyperarticulation, "over-enunciated", Lombard and clear speech, intelligibility in noise, pitch variance, vowel space in neural TTS, hearing loss), plus Expresso, Shi et al.'s title and F5-TTS issues. Six `gh` searches of F5-TTS issues found nothing on articulation.

Read: MegaTTS 3 (v4 2025-03-28, full HTML with Appendix K); Zheng and Maleki (v2 2026-03-24, method and results); DualSpeech (v2 2024-08-27, formula and tables); Wang, Sun and Baayen (abstract). Zheng and Maleki's cited-by (Semantic Scholar) holds three accent-guidance papers: [Accent Analogy Guidance](https://arxiv.org/abs/2609.29123) (2026-09-24) and [Closed-Loop Dual-Reference Guidance](https://arxiv.org/abs/2610.29351) (2026-10-21), abstracts read, measure accent, language ID, GOP, SIM and WER, no vowel or noise measure; [Phrase-Localized Language-Contrastive Guidance](https://arxiv.org/abs/2609.01016) (EMNLP 2026) not opened. Abstract only, not relevant: [learnable CFG null embeddings](https://arxiv.org/abs/2609.25411) (2026-09-21), [F5-TTS without CFG](https://arxiv.org/abs/2504.20334) (2025; checkpoint promised, release not confirmed).

Verdict: partly scooped on the decomposition, not on the claim.

Still open: cited-by for Shi et al. and Zero-Shot Lombard (HTTP 429 again), Google Scholar cited-by for the four original papers, and whether Expresso's two read styles share sentences.

## Review record

Hostile review, 2026-11-09, verdict "queue after fixes". All fixes applied: residuals from trained branches only, "speaker" renamed "prompt", four-branch split exploratory; text-only, prompt-only and the talker's own reading moved into step 1 with outcomes fixed in advance; scoop check made a gate; kill criterion split into mechanism and practical claims; determinism repeat, same-estimator null and pilot-sized band (the band is called conservative, since two noise draws cannot be truly paired); ratio-1.00 plant null and a tilt-and-noise plant; per-step flow times with matched arms and a no-sway run; claims narrowed to machine intelligibility, E2-TTS required for carry-over.
