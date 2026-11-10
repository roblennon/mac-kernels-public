# Experiment queue

The experiments worth running first, in order, each with a design card under [designs](designs/). The order was written 2026-11-09 by the lead agent, from 33 cards that each had a real prior-art search, a hostile review and a revision. The card is the source of truth; this file only orders them and says why.

Since then, the lab ran card 1's step 1a as its case 002 on 2026-11-10. It decided nothing: every measuring check passed, but the model could not do the base task, so the hypothesis was never tested (the card's "Result of step 1a" has the reading). Card 1 was redesigned to start with a capability step, and the other queued cards gained a gate on their subject. What that taught is in [LESSONS.md](LESSONS.md); new cards start from [the template](designs/TEMPLATE.md).

## Focus, from 2026-11-10

Rob redirected the programme on 2026-11-10 toward training, fine-tuning, inference, attention, context and gradients, and away from behaviour. The behavioural cards are set aside below. The thread now leading is not a card: [decode and small-batch speed on MPS](threads/mps-decode.md), opened from a lab result and measured the same day ([findings](../../findings.md)). The cards below are ordered for that focus; the reasons under each were written for the earlier order and still describe the card.

## Needs Rob

Nothing below runs past these without him. The lab asks him for its own approvals; this list is so they are visible in one place.

- [Semantic latents and the memorization threshold](designs/semantic-latent-memorization.md): Accept the DINOv3 licence on the Hub.
- [Train with a lookup table, ship without: catalyst or capacity?](designs/train-scaffolds.md): Approve days of Mac time for step 2, the deciding run.
- Whether these research files go public.

## How the order was chosen

Four things, in this order of weight. A first step that runs on the Mac inside the lab's limits (30 GPU-minutes, 300 seconds per process, 8 GiB) and already decides something. What a yes would change, and whether it carries to larger models. Whether it sits on ground only Rob has: his kernels and formats, his voice work, his steering tools. And spread, so the top of the queue is not one bet made many times, which is the warning in [the musings](MUSINGS.md#a-sceptical-reader-on-the-whole-notebook).

A reviewer from a different model family (GPT-6.1 Sol) read the queue and its top cards cold ([its review](reviews/2026-11-09-queue-review-sol.md)). It found the top of the first draft made one operational bet over and over: a cheap first slice that only showed feasibility, followed by a long road to a citable result. The order below follows most of its advice. Cards whose first slice decides something moved up; cards whose first slice only shows feasibility moved down. The top cards were then repaired against its list and checked by a fresh agent who tried to plan each one cold.

The self-rated chances on the cards are low on purpose: most say 2 out of 5. A null written up honestly is a deliverable, and several cards are built to return a tight null.

Each card opens with a status line, shortened here. "Search gate" means the card's prior-art search had not passed when this was written; run it first, because the card may park itself.

## For the lab

The lab's case 001 measures charged seconds to a target loss on a small character model, and the lab has already run questions on it. Card 14, removable training scaffolds, fits case 001 as it stands; its step 1a is drafted there as Q002, not yet opened. Case 002 ran card 1's step 1a. Every other card has a non-speed outcome and needs a new case with its own contract, noise floor and planted-effect check; each card describes both checks. Bind the lab's plan to the card's commit, and put the card's predictions into the lab's plan ledger before any run.

## The queue

1. **[How do Apple's stock GEMMs add? A summation-order and accumulator census on the M1 Max](designs/cross-stack-numerics.md)**  
   Status: queued, 2026-11-09. Search gate passed (Appendix C). (more in the card)  
   Why here: Extends a probe that already runs in this repository, needs no model, and produces facts every other Mac result in this queue rests on: how Apple GPU and CPU stacks round and add inside a matmul. Citation value is narrow (an Apple column beside published NVIDIA and AMD arithmetic models), but the cross-family reviewer ranked it first for being the most certain to finish and be reused.

2. **[Fine-tune survival under requantization: does one signal-to-noise number predict what survives?](designs/finetune-survival-requant.md)**  
   Status: queued, 2026-11-09. Search gate passed. (more in the card)  
   Why here: Answers a practical question about Rob's own merge path whatever happens to the law: which fine-tunes survive rounding to 4 bits, and whether one number per layer predicts it against explicit rival predictors. Moved up on the cross-family reviewer's advice.

3. **[Noise margin: does a transformer's response to noise bend away from its own linearization, and does Gaussian noise price real shortcuts?](designs/noise-margin-gain-table.md)**  
   Status: queued, 2026-11-09 (revised after hostile review). Search gate passed on a second pass the same day (11 queries, none refused). (more in the card)  
   Why here: Step 1 builds the instrument that eight separate ideas need (a per-layer noise tolerance with a linearized control), on a small public model. If one calibration prices many shortcuts, format choice becomes a lookup.

4. **[Fixed error versus fresh noise: is weight-rounding error in a diffusion sampler a per-sample bias?](designs/fixed-vs-fresh-error.md)**  
   Status: queued (diffusion half only; the language half is parked), 2026-11-09. Search gate passed: the wider sweep ran in full (12 queries, none refused) and found no scoop; one design change followed (guidance-residual split, see Analysis).  
   Why here: Rob's kernels and formats are the subject. It measures whether 4-bit error behaves as a per-sample bias across sampling steps, the thread seven lenses reached on their own, and either answer is a writable note.

5. **[Train with a lookup table, ship without: catalyst or capacity?](designs/train-scaffolds.md)**  
   Status: queued, 2026-11-09, search gate cleared: 12 searches, none refused, found nobody removing an n-gram table during training. Step 3 gains a MeCo-style arm. (more in the card)  
   Why here: The card that fits the lab's existing case 001 as it stands; its step 1a is drafted there as Q002, not yet opened. As research it sits lower: its deciding run is days of Mac time and needs Rob's approval.

6. **[Does beta2 set the rarest fact a model can keep? (rare-feature-threshold)](designs/rare-feature-threshold.md)**  
   Status: queued, 2026-11-09. Gate 0 (prior-art search) passed on 2026-11-09 (12 queries, none refused): no source states a beta2-dependent cutoff, but the decay fix has close precedent and recommender work goes the other way (see the search gate). (more in the card)  
   Why here: A bold law (weight decay and beta2 set the rarest fact a model can keep), cheap to test exactly in a scalar simulation first. Weak case for large models, since the sparse regime is rare in big-batch pretraining.

7. **[Does a token cost the same energy whatever it says? Content-dependent joules, and renumbering hidden units to save power](designs/content-energy.md)**  
   Status: queued, 2026-11-09. The search gate passed on 2026-11-09 (see "Search gate"); (more in the card)  
   Why here: Whether a token costs different energy depending on what it says, on the Mac. Step 1 is an instrument check other energy work can reuse; later steps have weak priors.

8. **[How fast does a rare exploit grow from its base rate? Adam's onset exponent moves from 1/2 to 1 as visible-reward gradients fill the shared weights](designs/rare-exploit-onset.md)**  
   Status: queued, 2026-11-09 (revised after hostile review; deployment-safety bound removed after cross-family review the same day). (more in the card)  
   Why here: CPU-only numpy and decisive for its narrowed optimizer claim. Moved down after the cross-family review found its advertised safety bound was wrong (no exploits in N draws bounds the starting probability at about 3/N, not 1/N squared); the card now claims a finite-regime optimizer result only.

9. **[Backward error for samplers: a calibrated, band-resolved noise test for approximate samplers](designs/sampler-backward-error.md)**  
   Status: queued, 2026-11-09 (revised after hostile review). Search gate passed: 22 keyword searches over three passes found no scoop; the watermark line closed in the third pass.  
   Why here: A reference-free test of fast samplers by inverting their outputs and checking the noise. Cheap on a small public checkpoint; slice 1a settles a real disagreement with published DDIM results.

10. **[The squared-error-optimal guidance weight, measured on held-out data](designs/guidance-optimal-weight.md)**  
   Status: queued, 2026-11-09. The search gate is cleared: the second pass ran all 12 queries with none refused, and nothing found scoops the card (see "Search gate, 2026-11-09").  
   Why here: Forward passes only, with an analytic toy that catches estimator bugs first. Moved down: its distribution-shift reading depends on access to a training-source dataset that is not checked, and the likeliest result is a precise negative that mostly confirms theory.

11. **[Is the layer-duplication effect just a bigger step?](designs/layer-repeat-step-size.md)**  
   Status: queued, 2026-11-09 (revised after hostile review). Search gate passed in a second pass the same day: 12 searches ran, none refused, and none found duplication compared with one-pass amplification on a frozen unshared LLM.  
   Why here: Separates the two things layer duplication does (a bigger step and a second read) with exact bookkeeping. Cheap, but the headline is judged unlikely; the useful outcome is probably the overshoot account.

12. **[The search cutoff: which entropy sets the cost of language-model tree search (renyi-search-cutoff)](designs/renyi-search-cutoff.md)**  
   Status: queued, 2026-11-09. Rebuilt after a hostile review. (more in the card)  
   Why here: A theory-backed second number next to perplexity for search. Rebuilt after a fatal objection into a narrower per-sentence prediction; CPU only.

13. **[The linear-response window: over fifteen decades of perturbation size, when does training answer in proportion](designs/linear-response-window.md)**  
   Status: queued, 2026-11-09. Search gate cleared 2026-11-09: twelve searches ran, none found the prediction-space curve below 1e-5 against injection time (see Search gate). (more in the card)  
   Why here: How small a weight nudge must be for training to answer in proportion. No GPU needed, so it can run beside GPU cards; weak carry to large models.

14. **[What temperature does in continuous-token autoregressive models: the innovations test, then a two-sided feedback map](designs/continuous-ar-temperature.md)**  
   Status: queued, 2026-11-09. Search gate passed the same day; neither stage is published. (more in the card)  
   Why here: What temperature means in continuous-token generators, by inverting real tokens through the head. Half the idea is occupied by a 2026 paper.

15. **[Surprise feedback: tell the model how wrong it just was (round 2 S12)](designs/surprise-feedback.md)**  
   Status: queued, search gate cleared 2026-11-09 (twelve keyword queries ran; no controlled transformer version of the idea found). Revised after a hostile review and two search passes, all recorded at the end. (more in the card)  
   Why here: Feed a model its own prediction error as an input. A build, self-contained on synthetic data; the lead's prior is low after the review.

16. **[Semantic latents and the memorization threshold](designs/semantic-latent-memorization.md)**  
   Status: queued, 2026-11-09. Step 0 needs Rob. (more in the card)  
   Why here: Whether a semantic latent lowers the data a diffusion model needs before it stops memorizing. Blocked on Rob accepting the DINOv3 licence.

17. **[Voice cloning as a per-phoneme lookup into the prompt](designs/voice-clone-circuit.md)**  
   Status: queued (Part A only; Part B parked), 2026-11-09. (more in the card)  
   Why here: Training-free on a released TTS model: does it clone a voice by looking up matching sounds in the prompt? Active field with two 2026 neighbours, so it comes after the cards above.

18. **[Does a fine-tune leak through pictures that never show it? Hidden trait transfer in image generators](designs/img-subliminal.md)**  
   Status: queued, 2026-11-09. Search gate passed in a second pass the same day: 12 searches ran and none found an image generator trained as student on a fine-tuned teacher's concept-free samples. (more in the card)  
   Why here: The long shot. The first step is a CPU-sized MNIST diffusion toy and a positive result would be widely cited, but the cross-family review showed that greater transfer to a same-initialisation student does not by itself separate hidden transfer from ordinary learning of visible changes. The card was revised for that; it moved down until its first slice can identify what it claims.

## Set aside (behavioural)

Set aside on 2026-11-10 at Rob's direction. Two hold a speed angle worth salvaging: whisper-silence (if the padding's content does not matter, short clips spend encoder compute on up to 30 s of padding) and denoiser-talks-back (if the text stream barely depends on the seed, it can be cached across seeds and steps).

- [Copies in context counted as evidence: does a replayed error hurt fresh answers more than a summary of it?](designs/copies-as-evidence.md)



- [Do a model's own visible errors raise its next error, and where does the falling agent hazard come from?](designs/agent-hazard-mixture.md)

- [What Whisper reads from its padding (whisper-silence)](designs/whisper-silence.md)

- [Do speech models lose their place at a constant rate? Hazard curves for TTS skips and repeats](designs/tts-slip-hazard.md)

- [How many people can a fast image model draw? (variety-census)](designs/variety-census.md)

- [Is Whisper grading its own students? Evaluator-training coupling in TTS word error rate](designs/whisper-grades-students.md)

- [The guidance scale is a clear-speech dial: guided flow-matching TTS over-articulates the way people do for a listener](designs/tts-clear-speech-guidance.md)

- [Late turn-taking in duplex speech models: sampler, stimulus, or model?](designs/duplex-turn-temperature.md)

- [Does the language model's left-to-right reading of a picture shape its binding errors? Wrap-around pairs first, re-ordered image tokens second](designs/raster-binding.md)

- [Binding by position: does a position-tagged bag of words tell FLUX.1 which colour goes with which object?](designs/dit-positional-binding.md)

- [The denoiser talks back: what an image model's text stream knows beyond the image stream](designs/denoiser-talks-back.md)

## Parked

Kept because the reasoning is worth having; each card says what would unpark it.

- **[Paired-metric floor: park as research, fold the calibration into the next INT8 fidelity card](designs/paired-metric-floor.md)**. parked, 2026-11-09. Search gate still open: all 3 queries on 2026-11-09 were refused (budget spent), so no independent scooping search has run.

- **[Nested bit-planes as precision on demand (merged cluster)](designs/nested-bitplanes.md)**. parked, 2026-11-09. A hostile review found the headline experiment gives no benefit on this hardware and is largely published. (more in the card)

- **[Positive-only self-training: which pass rate is worth the most per attempt?](designs/self-training-37.md)**. parked, 2026-11-09. Search gate still open: every WebSearch query was refused again on 2026-11-09, so only known pages were read. (more in the card)
