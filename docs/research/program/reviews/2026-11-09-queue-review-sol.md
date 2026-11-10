# Queue review by Sol, 2026-11-09

A cross-family review of [QUEUE.md](../QUEUE.md) and its first four cards, by GPT-6.1 Sol (OpenAI, via the Codex CLI, read-only sandbox, high reasoning effort), run after the cards had passed their Claude hostile reviews and search gates. The prompt asked whether a fresh agent could run card 1 cold, for the most serious flaw in cards 2 to 4, for misplaced cards, for missed prior art, and for its own top three. Its text follows unedited. What was done about it is recorded in the cards' review records and in the queue.

---

## 1. The first card

**No: a fresh agent could implement a pilot, but could not run the specified comparison without making scientific and engineering decisions.**

- **Models:** Both model IDs are valid, but model revisions and the Transformers version are missing.
- The processors must differ because small uses 80 mel bins and turbo uses 128. ([Small configuration](https://huggingface.co/openai/whisper-small/blob/main/config.json), [turbo configuration](https://huggingface.co/openai/whisper-large-v3-turbo/blob/main/config.json))
- **Implementation:** The mask hook remains an instruction to investigate, without a checked implementation, attention backend, mask shape, or numerical bias.
- Attention capture and masking must use a consistent backend across arms, with timing measured under that backend.
- **Data:** Seed 0 does not identify the sample without a dataset revision, ordering, duration calculation, and selection procedure.
- The energy detector lacks thresholds, frame sizes, and a rule translating its boundary into encoder positions.
- “Padding” could mean silence after speech or synthetic padding after the recording ends, and those are different interventions.
- Sw-diff cannot assign an eligible donor to targets shorter than five seconds when donors come from the specified two-to-eight-second pool.
- The mean replacement must exclude donors still speaking at each replaced position.
- Sw-sim can introduce donor speech near the boundary when its donor is longer.
- Digital-zero and pink-noise durations are unspecified, and peak normalization needs an explicit zero-input exception.
- **Metric:** Word normalization and the treatment of overlapping repeated 5-grams are unspecified.
- Cutting to 100 speech clips undermines the two-percentage-point equivalence claim because even zero failures in 100 trials leaves a roughly three-point one-sided 95% upper bound.
- A nonsignificant difference does not establish equivalence within two points.
- **Controls:** C may be impossible when the pad carries more attention mass than all speech frames combined.
- Matching average mass also leaves head-specific and decoding-step-specific disruption unmatched.
- Dropping C removes the proposed basis for interpreting A2 as evidence for an attention sink.
- A3 need not collapse because the retained pad states can already contain speech information from the bidirectional encoder.
- CPU-fp32 versus MPS-fp16 disagreement measures a backend-and-precision intervention, rather than the repeatability floor of the experimental backend.
- “Outputs must change clearly” is not the frame’s required recovery of a known-size planted effect.
- **Arithmetic:** The main design requires **2,170 decodes per model**: 1,400 speech decodes, 20 A3 decodes, 700 nonzero non-speech decodes, and 50 digital-zero decodes.
- That is 4,340 decodes across both models before the pilot and validation checks, rather than approximately 3,600.
- The approximately 2.6 GB saved-state estimate is reasonable if all saved states are fp16.
- **Limits:** Twenty-five-clip shards do not establish compliance with 300 seconds, especially when interventions cause 200-token overruns.
- The card needs load-inclusive deadlines, completed-arm checkpoints, a timeout policy, and peak-memory accounting.
- The pilot must include masked and looping cases before it can justify the 30-minute budget.

## 2. Each of cards 2 to 4

**Card 2, tts-slip-hazard — move down.**

Its most serious flaw is that the event requirement has no demonstrated affordable acquisition path.

Fifty first-third events plus fifty last-third events require at least 100 long generations because each generation contributes only one first slip.

Survivor attrition makes the requirement substantially larger, and ASR plus listening adds further cost.

Step 0 establishes feasibility, but it does not settle the scientific question promised by the queue’s ranking rule.

**Card 3, img-subliminal — move down.**

Its most serious flaw is the assertion that greater transfer to the same-initialization student rules out ordinary learning of visible changes.

Two generators can differ in their sensitivity to exactly the same visible statistical shift because their representations and optimization geometry differ.

Classifier filtering and the proposed detectability probes do not eliminate that alternative.

A positive result could establish generator-to-generator transfer, but the claimed distinction between hidden transfer and ordinary distribution learning remains unidentified.

**Card 4, rare-exploit-onset — move down and remove the advertised safety bound.**

Its most serious flaw is the conclusion that a clean N-step pilot can bound the initial exploit probability as approximately 1/N².

In the hidden-only bandit, observing no exploit in NGB draws has probability `(1 − p₀)^(NGB)`, giving an upper bound of approximately `3/(NGB)` at 95% confidence.

Faster amplification after an exploit appears cannot strengthen evidence from never sampling it.

A finite-regime optimizer study may survive, but that deployment-safety conclusion does not.

## 3. The order as a whole

- **14, cross-stack-numerics, is too low** because it extends an existing working probe, fits the owned hardware, and produces reusable numerical facts without training a model.
- **13, finetune-survival-requant, is too low** because it addresses an actual local merge path and compares its predictor against explicit practical rivals.
- **5, copies-as-evidence, belongs nearer the front** because its main contrast uses forward passes and directly tests a possible harness intervention.
- **9, train-scaffolds, is too high as research** because compatibility with the lab’s existing case does not justify a deciding experiment estimated at days of Mac time.
- **10, guidance-optimal-weight, is too high** because its distribution-shift interpretation depends on an unchecked training-source dataset, while its likely null mostly quantifies an existing theoretical expectation.

The top ten do not repeat one scientific hypothesis.

They repeatedly make one operational bet: a cheap instrument check or toy result will justify a much longer route to a citable finding.

Cards 2, 3, 4, and 9 particularly weaken the promise that the first slice already decides something scientifically useful.

A well-calibrated null on a tiny model is not automatically a publishable contribution.

## 4. Anything already published that the cards miss

**[WhisperKit: On-device Real-time ASR with Billion-Scale Transformers](https://arxiv.org/html/2507.10860v1)**, July 2025, explicitly describes caching encoder outputs for zero-padded audio blocks.

It obtains this through self-distillation with block-diagonal encoder attention, so it does not settle replacement on unchanged public checkpoints.

It nevertheless directly narrows card 1’s caching novelty and supplies a necessary comparison.

**[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)**, ICLR 2024, establishes the broader phenomenon of preserving attention targets despite their limited semantic importance.

It is missing mechanism background for card 1, rather than a Whisper-specific scoop.

I have not verified an exact publication that settles the remaining three cards.

## 5. My own top three from the 30, and why

1. **14 — Cross-stack numerics.** It has the strongest combination of existing implementation, affordable decisive measurements, and results other Mac researchers could reuse and cite.
2. **1 — Whisper silence, after repair.** Interventions on deployed public checkpoints can answer a concrete mechanism question without training, provided the donor rules, controls, equivalence analysis, and budget are fixed.
3. **5 — Copies as evidence.** It offers a cheap, direct behavioral contrast with a practical consequence, provided any effect survives new templates, sensible filler controls, and a second checkpoint.
