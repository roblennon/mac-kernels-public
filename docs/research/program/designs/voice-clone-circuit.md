# Voice cloning as a per-phoneme lookup into the prompt

Status: queued (Part A only; Part B parked), 2026-11-09. Search gate passed: 12 queries, none refused, nothing answers either claim. Design changed: a concentration arm joins the mass arm (see Search gate).

Zero-shot TTS models copy a voice from a few seconds of audio. Either they read the prompt as a global voice summary, or they look up, sound by sound, how this speaker said this phone, as kNN-VC does by hand. This card tests which on a released model, with no training. The payoff is interpretability and prompt-selection guidance. A head ablation is not a cloning lock on open weights, since anyone can undo it.

## Hypothesis

In F5-TTS, target frames read speaker identity from a few prompt frames of the same phone, not from the prompt spread out. The heads that do this matching carry more of the voice than heads that simply attend heavily to the prompt. Moving one phone class of the prompt to another speaker moves that class of the output toward that speaker, and leaves other classes mostly in place.

## Theory

Zero-shot TTS trains by filling masked audio from the rest of the same utterance, so one speaker sits in context and target. A content-addressed copy would key on phonetic and acoustic similarity together, as kNN-VC does. So the question is whether attention over the prompt is local and content-matched, or broad.

Assumptions, said out loud:

- In F5 the prompt mel enters only at prompt positions, so attention is close to the only route from prompt to distant target frames. A head that reads the prompt at all will matter when removed. That is why the decisive comparison is against heads matched on attention mass, not against random heads.
- Forced alignments on 24 kHz frames (about 93.75 per second) are precise enough to label phones. Not checked.

## What is already known

Step 0 ran (see Search gate). No paper found analyses attention heads of a zero-shot TTS model for speaker copying, or tests prompt content phone by phone.

- [Pouw et al.](https://arxiv.org/abs/2604.06356), 2026-04-07, read. Induction-head ablation removes in-context TTS in a speech LM; studies speaking rate, pitch range, intensity. No timbre or same-phone analysis.
- [Baas et al., kNN-VC](https://arxiv.org/abs/2305.18975), 2023-05-30, read. The per-phone lookup built by hand; short references miss phones.
- [Diwan, Harwath, Choi](https://arxiv.org/abs/2610.40505), 2026-10-29, about 80% read. Residual-stream features on six voice-cloning TTS models; no heads, no reference phone content.
- [Chen et al., F5-TTS](https://arxiv.org/abs/2410.06885), 2024-10-09, method read. Prompt enters only as kept mel frames; early flow steps carry the prompt.
- [Lee, Shin, Lee, TruS](https://arxiv.org/abs/2601.20481), 2026-01-28, method read. Training-free speaker unlearning on F5-TTS by steering FFN outputs away from the opt-out speaker, layers and steps chosen per speaker. Closest work on the unlearning payoff; no attention or prompt-content analysis.
- [Li et al., Time-Layer Adaptive Alignment](https://arxiv.org/abs/2511.09995), 2025-11 (v3 2026-03), method read. Pooled-feature alignment (CKNNA) on a flow-matching TTS: speaker information is concentrated in early, noisy steps and is weakest in the final layers. Supports the sampling-path census and layer matching. No attention analysis.
- [Li et al., Text Template Tokens Are Implicit Semantic Registers](https://arxiv.org/abs/2607.19139), v2 2026-07-30, read in part. Image DiT, not speech: the heads with the most attention mass on the prompt are largely causally inert; pruning a quarter of them cost 1.4 GenEval points. This changes the design (below).
- [Zhao et al.](https://arxiv.org/abs/2506.14226), 2025-06-17, results read. Phoneme coverage of the synthesized text, not the prompt, moves verification EER. Not this question.
- [CDFSE](https://thuhcsi.github.io/interspeech2022-cdfse-tts), Interspeech 2022, project page only. A designed per-phoneme speaker embedding; phone-local detail helps when built in, not shown to be learned.
- Summaries only, none on heads or prompt phones: [retrieval VC](https://arxiv.org/abs/2305.07204) (2023), [Koriagin et al.](https://arxiv.org/abs/2606.10029) (2026-06), [GUARD](https://neurips.cc/virtual/2026/160176) (2026), [CloneShield](https://arxiv.org/abs/2505.19119) (2025-05), Lenglet et al., Computer Speech & Language 2026 (probes; full text embargoed).

Left open: which heads carry timbre in zero-shot TTS, and whether prompt content matters phone by phone. No prompt-selection study by phonetic coverage of the prompt was found.

## Prediction (written before any run)

G is the mean right-speaker minus wrong-speaker similarity on the baseline. Effects are shares of G. W is the WER rise from 16 NFE to 4 NFE on the same pairs, a reference amount of audio damage. WER effects are shares of W.

- Census: in at least 5 heads, attention from target frames of phone p to prompt frames of p is at least 3 times the base rate. Those heads' attention over the prompt has lower entropy and higher top-8 mass than the median head.
- Ablation: the top 18 heads by phone-match score (about 5% of 352) lower similarity by at least 10% of G more than the stronger of two layer-matched control sets: the 18 heads with the most target-to-prompt attention mass, and the 18 with the most concentrated prompt attention (top-8 mass). Their WER cost is under half their similarity cost, in shares of W and G.
- Transplant: replacing every class-p segment in the prompt with a donor's class-p audio moves output class p toward the donor by at least 15% of G more than it moves other classes.

## Outcome table (fixed before the run)

"Inconclusive" means a readout falls between its kill line and its prediction, or inside the baseline's spread.

| Census | Ablation vs mass | Transplant | Conclusion |
|---|---|---|---|
| local | beats the stronger control by ≥10% of G | class-specific ≥15% of G | per-phone lookup, supported |
| local | ties (inside spread), or shares ≥12 of 18 heads with that control | class-specific | lookup, carried by the heads that read the prompt most or most sharply; report which |
| any | any | shift under 5% of G, or other classes move as much | no practical per-phone lookup; prompt content matters globally |
| broad | loses or ties | any | global summary |
| any other cell, or any readout in its middle zone | | | inconclusive |

The transplant alone decides the practitioner claim. The lookup claim needs the transplant plus the census or the ablation. Top versus random shows only that some heads read the prompt.

## Design

**Models and data.** F5-TTS v1 Base, `model_1250000.safetensors` (1.35 GB, CC-BY-NC-4.0, non-commercial; record hash). Config (reviewer-checked): depth 22, 16 heads, 24 kHz, hop 256, trained on Emilia_ZH_EN; LibriSpeech overlap unlikely, not checked. Vocos mel-24khz vocoder. LibriSpeech dev-clean for head selection, test-clean for evaluation; the split is fixed now. Montreal Forced Aligner, english_us_arpa. Verifiers microsoft/wavlm-base-plus-sv and speechbrain/spkrec-ecapa-voxceleb. WER from local Whisper large-v3-turbo (MLX). Sizes of vocoder, verifiers, aligner and datasets not checked.

**Per-class identity readout.** Utterance verifiers fail on short fragments, so identity is read per frame. For each generated frame of class p: the mean distance in WavLM features to its k nearest real held-out frames of p from the right speaker, minus the same to a wrong speaker. Averaged per class. Classes are voiced (vowels, diphthongs, nasals) and chosen before generation by availability.

**Decisive metric (mechanism).** The similarity drop from the top-18 phone-match set minus the larger drop of the two control sets (mass, concentration), layer-matched, in shares of G, bootstrapped over speakers. The concentration arm was added after Step 0: in an image DiT the heads with the most prompt mass were largely inert, so a mass arm alone may be easy to beat. If the mass arm sits inside the random sets' spread, say so.

**Analysis.** Paired per pair, arms interleaved within each pair under the same noise seed. Intervals by bootstrap over speakers. A difference inside the baseline's seed-to-seed spread is inconclusive. Census and selection on dev-clean speakers only.

**Implementation notes.** Patch F5's fused attention to compute weights and reduce them on the fly to per-head scalars (phone-match ratio, entropy, top-8 mass, total mass). Ablate in both guidance passes. 16 NFE with CFG and sway sampling for every arm except the W reference. Processes stay under 300 s and append to JSONL.

**Step 0 (no device time).** A recorded prior-art search of at least 12 queries: head and attention analysis of zero-shot TTS; reference-audio or prompt selection and phonetic coverage; speaker similarity against prompt content; cloning prevention and speaker unlearning on F5, E2, CosyVoice and MaskGCT. Record queries, dates and hits here. A paper that already answers either claim sends the card back for review.

**Step 1 (one lab slice, 30 device minutes).**
1. Setup, about 10 minutes: install MFA, align 5 utterances, time one generation and one full score (two verifiers, frame readout, Whisper). Size pairs from that, cutting pairs, never arms.
2. Capability gate, before any planted check or head is touched, at most 8 device minutes. On 12 dev-clean speakers, one prompt-target pair each, generate the 16-NFE baseline at two seeds with stock (unpatched) attention. It passes only if all of these hold: (a) baseline WER under 10%, and MFA aligns at least 90% of generated utterances, since the census and the frame readout need phone labels on generated audio; (b) both verifiers score the right speaker above the wrong one in at least 11 of 12 pairs; (c) frame-readout G, averaged over pairs, has a bootstrap-over-speakers lower bound above zero in at least 4 voiced classes; (d) the seed-to-seed difference in mean G is under 5% of G, so a 10%-of-G ablation effect sits at least twice outside the spread. If (a) or (b) fails, check prompt text and formatting once against F5's own inference script, then stop and park the card: the model is not cloning on this setup. If (c) or (d) fails, the frame readout cannot resolve the predicted effects; stop, and send the readout (k, feature layer, class set) back for review before any census or ablation. Record the gate's numbers whatever the outcome.
3. Checks below, including the planted census and planted readout checks, on real audio.
4. Census on dev-clean pairs, twice: on the real sampling path (the flow times the 16-NFE sway schedule visits, conditional pass) and teacher-forced. Select heads on the sampling-path version, and report the census by flow-time bin, since speaker information concentrates in early steps (Li et al., 2511.09995). Split-half stability: overlap of the top 18 heads across two disjoint speaker halves.
5. Pilot ablation on test-clean pairs: baseline at two seeds, phone-match top 18, mass top 18, concentration top 18, three random sets of 18, and the 4-NFE W reference. Every Step 1 ablation number is labelled "unconfirmed observation".

Step 1 alone says whether prompt attention is local or broad, and whether the readouts can see the predicted effect sizes.

**Step 2 (one slice).** Transplant test on test-clean speakers. For each speaker and chosen class p, a prompt with every class-p segment replaced by a donor's class-p audio, and a control prompt with the same duration of a different class replaced. Score output class p and the other classes with the frame readout. No internals needed.

**Step 3 (2 to 4 slices).** Full ablation: top 1%, 2%, 5% by phone match, mass and concentration, five random sets per size, lowest-scoring heads, zero- and mean-ablation, two seeds. Then a secondary missing-phone test: equal-duration, equal-SNR prompts with and without a class, scored by difference in differences.

**Step 4 (later).** Repeat on a second model with no separate speaker-embedding input; E2-TTS is the candidate (not checked). On CosyVoice-family models, which also take a speaker embedding (reviewer's memory, not checked), the prediction is that the global route takes over and phone-match heads matter less.

## Checks before believing anything

- Self-comparison: same-seed baseline reruns, and zero-head ablation, must be bit-identical to the baseline; if MPS is not deterministic, check on CPU fp32 and use the two-seed spread as the MPS tolerance. Patched attention must match the original within 1e-5 in fp32. Rescoring a file must give the same numbers.
- Planted, census: score kNN-VC's own neighbour choices on the same pairs with the census code, a known lookup, before and after acoustic matching. Score synthetic attention with a known same-phone ratio (for example 1, 3 and 6 times base rate); the score must come back at that ratio.
- Planted, similarity pipeline: prompts that splice 0, 25, 50 and 75% of another speaker's audio, generated and scored before any head is touched. The drop must be monotonic and roughly proportional.
- Planted, frame readout: on real held-out audio, swap a known share of a speaker's class-p frames for another speaker's. The readout must recover a 15%-of-G shift. If it cannot, Step 2 does not run.
- Planted, hook: ablating all 16 heads of one layer through the hook must equal zeroing that layer's output-projection columns by hand.
- W must sit outside the baseline's seed spread, or the WER unit is unusable and a degraded-vocoder reference replaces it.

## The control that could embarrass it

The attention-mass and concentration arms. If the heads that look at the prompt most, or most sharply, carry as much voice as the heads that match phones, the phone match is a side effect of reading the prompt. In the transplant, the other-class control: if every class shifts toward the donor as much as class p, the prompt is read globally.

## Confounds a hostile reader would raise

Acoustic similarity is answered by the kNN-VC control, broken audio by W and by logging wrong-speaker similarity, a fooled verifier by requiring the second verifier to agree in sign, double dipping by the dev/test split. If F5's character embeddings carry the match, that is still a lookup, and the write-up says which.

## Kill criterion

The practitioner claim dies if the class-p transplant shift is under 5% of G above the control. The lookup claim dies if that happens, or if the census is broad (phone match under 1.5 times base rate in every head) and the phone-match set does not beat the mass set. Middle zones are inconclusive, not partial support.

## Cost on this hardware

Step 1: at most 30 device minutes, about 2 hours wall time; downloads about 2 to 3 GB (not checked); outputs under 200 MB of FLAC. Step 2: one similar slice. Step 3: 2 to 3 device hours over several slices, about 1 GB of FLAC, sampled and pruned after scoring. Step 4: 1 to 2 GB more. The 15 GB disk floor holds throughout.

## What a yes or a no changes

A yes tells practitioners to build prompts for phonetic coverage, shows an in-context copy circuit outside text, and names a head-level place to try speaker unlearning in hosted models, beside TruS's FFN steering on the same model. A no says learned models do not rebuild kNN-VC, and prompt content matters only globally.

## Why it should carry to larger models

F5 and E2 share the prompt-and-fill objective and the in-sequence prompt. Untested: whether larger models spread the copy over more heads, and whether codec language models do the same. A finding on one checkpoint stays a finding on one checkpoint.

## Part B, parked

Whether the circuit appears above a training-speaker count moves to its own card, parked. Its first step is a capability gate: one 20M model at the largest speaker count must clone held-out speakers. It needs Rob's approval for device time (8 to 10 hours per run, not checked), and may give a smooth curve, a non-finding. Nearest work read: [Breslow et al.](https://arxiv.org/abs/2609.14011), no speech.

## Receipts to keep

Plan and outcome table, search log, hashes, pair lists and splits, per-head census scores, per-pair JSONL, check results, head lists, timing log, code.

## Search gate, 2026-11-09

First pass: 4 queries, all refused (budget spent); direct fetches only. Second pass: 12 queries, none refused, covering head and attention analysis of F5 and zero-shot TTS, speaker information by layer and flow step, prompt selection and phonetic coverage, speaker unlearning, copy heads for timbre, phone-level speaker representations, head ablation in diffusion transformers, prompt-target overlap, anti-cloning, and Lenglet et al. New closest works are listed above with dates and depth read. None answers either claim. Scoop: no.

Design changes: concentration arm, decisive metric against the stronger control, census by flow-time bin.

Still open: whether TruS's chosen layers and steps on F5 line up with census heads; E2-TTS head work, not searched by name. No released dataset or known result to compare against was found. Gate passed.

## Review record

- Must-fix, all addressed: layer-matched mass arm as the decisive comparison (top-vs-random foregone); local-versus-broad framing with a kNN-VC positive control; circular plant replaced by synthetic-attention, kNN-VC and splice-prompt plants; outcome table with inconclusive zones and WER in shares of W; Step 0 search gate; frame-level per-class readout with its own plant.
- Should-fix, all addressed: transplant as Step 2, deciding the practitioner claim (missing phoneme to Step 3); second seed, third random set, "unconfirmed" label, CPU fp32 fallback; dev/test split with split-half stability; payoff restated, CosyVoice prediction stated; census on the sampling path.
- Minor: Part B parked with a capability gate; setup sub-step and licence noted.

Reviewer's six scooping queries were refused; fetches confirmed 2604.06356, 2610.40505, 2609.14011 and the F5 config, checkpoint size and licence.
- 2026-11-10, capability-gate pass (after copies-as-evidence step 1a ran on a model at chance): the card had planted checks on the instruments but no check that stock F5 actually clones these speakers with a frame-level G large and steady enough for 10% and 15% of G to show. Added Step 1.2, a capability gate with WER, alignment, verifier, per-class G and seed-spread bars and a stop. Whether F5 v1 clones LibriSpeech speakers on this setup: not checked. Nothing else changed.
