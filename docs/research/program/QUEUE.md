# Experiment queue

The experiments worth running first, in order, each with a design card under [designs](designs/). Written 2026-11-09 by the lead agent, from 33 cards that each had a real prior-art search, a hostile review and a revision. Nothing here has run. The card is the source of truth; this file only orders them and says why.

## How the order was chosen

Four things, in this order of weight. A first step that runs on the Mac inside the lab's limits (30 GPU-minutes, 300 seconds per process, 8 GiB) and already decides something. What a yes would change, and whether it carries to larger models. Whether it sits on ground only Rob has: his kernels and formats, his voice work, his steering tools. And spread, so the top of the queue is not one bet made many times, which is the warning in [the musings](MUSINGS.md#a-sceptical-reader-on-the-whole-notebook).

A reviewer from a different model family (GPT-6.1 Sol) read the queue and its top cards cold ([its review](reviews/2026-11-09-queue-review-sol.md)). It found the top of the first draft made one operational bet over and over: a cheap first slice that only showed feasibility, followed by a long road to a citable result. The order below follows most of its advice. Cards whose first slice decides something moved up; cards whose first slice only shows feasibility moved down. The top cards were then repaired against its list and checked by a fresh agent who tried to plan each one cold.

The self-rated chances on the cards are low on purpose: most say 2 out of 5. A null written up honestly is a deliverable, and several cards are built to return a tight null.

Each card opens with a status line. "Search gate" means the card's prior-art search had not passed when this was written; run it first, because the card may park itself.

## For the lab

The lab's only case so far measures seconds to a target loss on a small character model. Card 14, removable training scaffolds, fits that case as it stands, so it is the natural first end-to-end test of the lab. Every other card has a non-speed outcome and needs a new case with its own contract, noise floor and planted-effect check; each card describes both checks. Put the card's predictions into the lab's plan ledger before any run.

## The queue

1. **[Copies in context counted as evidence: does a replayed error hurt fresh answers more than a summary of it?](designs/copies-as-evidence.md)**  
   Status: redesigned after step 1a, 2026-11-10, and revised after a review of the redesign the same day; before step 0 runs it awaits this redesign's commit, the lab's plan rebound to that commit (see "Before any plan"), and Rob's allowances and envelopes for step 0 and step 1a (see "Budget"). T2's wording is decided (see "Before step 0"). Rob's answers on Qwen3-4B and Qwen3.5-4B are needed only before a second ladder starts, not before step 0 (see "Candidates, in order"). Step 1 is now step 0 (a capability step that picks a task, model and protocol the model can do, and checks that it reads the count note) and then step 1a on what step 0 picks. Step 1a ran in the lab on 2026-11-10 (case 002) and decided nothing: with thinking off the 1.7B model got no greedy answer right (T1 was inaccurate, T2 mostly broke format), and the count probe failed, so the hypothesis was not tested; see "Result of step 1a" at the end. Earlier status: queued, 2026-11-09; search gate passed; revised after a hostile review and five cold-planning reviews. The loop-share question moves to card B, not yet written.  
   Why here: Forward passes only on a public 1.7B model, one decisive contrast with a planted effect, and a practical consequence: whether agent harnesses should collapse repeated failures into one copy and a count. It is also the cheapest test of the question the whole-notebook critic named for a year of work, how much long-horizon decline comes from the decoding loop rather than the model. Its search gate came back clean, and a cross-family reviewer put it in its own top three.

2. **[How do Apple's stock GEMMs add? A summation-order and accumulator census on the M1 Max](designs/cross-stack-numerics.md)**  
   Status: queued, 2026-11-09. Search gate passed (Appendix C). Revised after a hostile review and four cold reads the same day. Designed, not run. Merges "Exact integers inside float GEMMs" and "What fp16 arithmetic means on each stack"; the divergence law gets its own card, not yet written.  
   Why here: Extends a probe that already runs in this repository, needs no model, and produces facts every other Mac result in this queue rests on: how Apple GPU and CPU stacks round and add inside a matmul. Citation value is narrow (an Apple column beside published NVIDIA and AMD arithmetic models), but the cross-family reviewer ranked it first for being the most certain to finish and be reused.

3. **[What Whisper reads from its padding (whisper-silence)](designs/whisper-silence.md)**  
   Status: queued, 2026-11-09. Designed, not run; no code exists yet. Step 1 is 1a (pilot and checks, two slices) and 1b (the deciding run, several slices). "Step 1, exactly" is the run plan; where the rest of the card seems to differ, it governs. Merged from four ledger ideas.  
   Why here: Training-free on public checkpoints, and it decides between three accounts of why Whisper stops: a stop signal in the padding, an attention sink, or nothing. It sits on a failure people who rely on local speech tools meet (text invented over silence). It moved from first to third after the cross-family review found it was not yet runnable cold; the card has been repaired against that list.

4. **[Fine-tune survival under requantization: does one signal-to-noise number predict what survives?](designs/finetune-survival-requant.md)**  
   Status: queued, 2026-11-09. Search gate passed. Step 1 is runnable and judgeable from "Step 1, exactly" together with its definitions (A2) and reasons (A1).  
   Why here: Answers a practical question about Rob's own merge path whatever happens to the law: which fine-tunes survive rounding to 4 bits, and whether one number per layer predicts it against explicit rival predictors. Moved up on the cross-family reviewer's advice.

5. **[Noise margin: does a transformer's response to noise bend away from its own linearization, and does Gaussian noise price real shortcuts?](designs/noise-margin-gain-table.md)**  
   Status: queued, 2026-11-09 (revised after hostile review). Search gate passed on a second pass the same day (11 queries, none refused). Step 2 is partly scooped for activation quantization (Sathyanarayanan 2026: matched Gaussian badly underprices it), so that arm is now a replication and positive control; the knee and the other shortcut classes stay open.  
   Why here: Step 1 builds the instrument that eight separate ideas need (a per-layer noise tolerance with a linearized control), on a small public model. If one calibration prices many shortcuts, format choice becomes a lookup.

6. **[Fixed error versus fresh noise: is weight-rounding error in a diffusion sampler a per-sample bias?](designs/fixed-vs-fresh-error.md)**  
   Status: queued (diffusion half only; the language half is parked), 2026-11-09. Search gate passed: the wider sweep ran in full (12 queries, none refused) and found no scoop; one design change followed (guidance-residual split, see Analysis).  
   Why here: Rob's kernels and formats are the subject. It measures whether 4-bit error behaves as a per-sample bias across sampling steps, the thread seven lenses reached on their own, and either answer is a writable note.





8. **[Do speech models lose their place at a constant rate? Hazard curves for TTS skips and repeats](designs/tts-slip-hazard.md)**  
   Status: queued, 2026-11-09 (revised after hostile review, and after Sol's cross-family review: sizing and a budget stop rule). Search gate passed on 2026-11-09: 12 searches found no hazard-shape, chunked-control or slip-temperature-law study; Step 3 was changed for a known low-temperature result.  
   Why here: Rob's own ground and likely to change practice if true: a constant slip rate would make long-form TTS failure predictable from short tests. Moved down because its first slice shows feasibility but decides nothing; the deciding run needs about a hundred long generations or more, and the card now says how many and when to stop.

9. **[Does a fine-tune leak through pictures that never show it? Hidden trait transfer in image generators](designs/img-subliminal.md)**  
   Status: queued, 2026-11-09. Search gate passed in a second pass the same day: 12 searches ran and none found an image generator trained as student on a fine-tuned teacher's concept-free samples. Revised after a hostile review, then again after a cross-family review (Sol) that found the identification broken; the record is at the end.  
   Why here: The long shot. The first step is a CPU-sized MNIST diffusion toy and a positive result would be widely cited, but the cross-family review showed that greater transfer to a same-initialisation student does not by itself separate hidden transfer from ordinary learning of visible changes. The card was revised for that; it moved down until its first slice can identify what it claims.

10. **[How fast does a rare exploit grow from its base rate? Adam's onset exponent moves from 1/2 to 1 as visible-reward gradients fill the shared weights](designs/rare-exploit-onset.md)**  
   Status: queued, 2026-11-09 (revised after hostile review; deployment-safety bound removed after cross-family review the same day). Search gate passed on the second pass, 2026-11-09: the update-form half of the rho switch is published (DP-AdamBC), but no stochastic onset exponent in p0 and no rho crossover in RL turned up.  
   Why here: CPU-only numpy and decisive for its narrowed optimizer claim. Moved down after the cross-family review found its advertised safety bound was wrong (no exploits in N draws bounds the starting probability at about 3/N, not 1/N squared); the card now claims a finite-regime optimizer result only.

11. **[How many people can a fast image model draw? (variety-census)](designs/variety-census.md)**  
   Status: queued, 2026-11-09 (revised after hostile review). Search gate passed in a third pass the same day: 12 searches ran, none was refused, and none found an identity count from a distilled or quantized image model.  
   Why here: Counts how many distinct people a one-step model draws, and asks whether 4-bit rounding changes who gets drawn. Rob's formats again, and a concrete harm (sameness) a person can see.

12. **[Backward error for samplers: a calibrated, band-resolved noise test for approximate samplers](designs/sampler-backward-error.md)**  
   Status: queued, 2026-11-09 (revised after hostile review). Search gate passed: 22 keyword searches over three passes found no scoop; the watermark line closed in the third pass.  
   Why here: A reference-free test of fast samplers by inverting their outputs and checking the noise. Cheap on a small public checkpoint; slice 1a settles a real disagreement with published DDIM results.

13. **[Voice cloning as a per-phoneme lookup into the prompt](designs/voice-clone-circuit.md)**  
   Status: queued (Part A only; Part B parked), 2026-11-09. Search gate passed: 12 queries, none refused, nothing answers either claim. Design changed: a concentration arm joins the mass arm (see Search gate).  
   Why here: Training-free on a released TTS model: does it clone a voice by looking up matching sounds in the prompt? Active field with two 2026 neighbours, so it comes after the cards above.

14. **[Train with a lookup table, ship without: catalyst or capacity?](designs/train-scaffolds.md)**  
   Status: queued, 2026-11-09, search gate cleared: 12 searches, none refused, found nobody removing an n-gram table during training. Step 3 gains a MeCo-style arm. Step 1 starts on the lab's case 001.  
   Why here: The card that fits the lab's existing case 001 as it stands, so the lab can use it as an end-to-end test of itself. As research it sits lower: its deciding run is days of Mac time and needs Rob's approval.

15. **[The squared-error-optimal guidance weight, measured on held-out data](designs/guidance-optimal-weight.md)**  
   Status: queued, 2026-11-09. The search gate is cleared: the second pass ran all 12 queries with none refused, and nothing found scoops the card (see "Search gate, 2026-11-09").  
   Why here: Forward passes only, with an analytic toy that catches estimator bugs first. Moved down: its distribution-shift reading depends on access to a training-source dataset that is not checked, and the likeliest result is a precise negative that mostly confirms theory.

16. **[Is the layer-duplication effect just a bigger step?](designs/layer-repeat-step-size.md)**  
   Status: queued, 2026-11-09 (revised after hostile review). Search gate passed in a second pass the same day: 12 searches ran, none refused, and none found duplication compared with one-pass amplification on a frozen unshared LLM.  
   Why here: Separates the two things layer duplication does (a bigger step and a second read) with exact bookkeeping. Cheap, but the headline is judged unlikely; the useful outcome is probably the overshoot account.

17. **[Do a model's own visible errors raise its next error, and where does the falling agent hazard come from?](designs/agent-hazard-mixture.md)**  
   Status: queued, 2026-11-09. Revised after a hostile review (record at the end). Search gate passed 2026-11-09 (20 queries, none refused): partly scooped. Role relabeling and identical-versus-distinct repeated errors now have published results; own-versus-other-agent attribution on next-step accuracy, with a count regressor, is still open.  
   Why here: Whether errors in an agent's own turns hurt more than the same errors labelled as another agent's. Partly scooped by two 2026 papers that the card now cites; the remaining contrast is cheap.

18. **[Does beta2 set the rarest fact a model can keep? (rare-feature-threshold)](designs/rare-feature-threshold.md)**  
   Status: queued, 2026-11-09. Gate 0 (prior-art search) passed on 2026-11-09 (12 queries, none refused): no source states a beta2-dependent cutoff, but the decay fix has close precedent and recommender work goes the other way (see the search gate). Step 0 may start.  
   Why here: A bold law (weight decay and beta2 set the rarest fact a model can keep), cheap to test exactly in a scalar simulation first. Weak case for large models, since the sparse regime is rare in big-batch pretraining.

19. **[Is Whisper grading its own students? Evaluator-training coupling in TTS word error rate](designs/whisper-grades-students.md)**  
   Status: queued, 2026-11-09. Search gate passed on the third pass (12 queries, none refused). Partly scooped: a 2024 German study ran the label swap with the labeler and one outside judge but never analysed the interaction; its published means lean our way. Step 1 and the English, multi-judge step 3 remain open.  
   Why here: Whether TTS models trained on Whisper-labelled data get an easier score from Whisper judges. Step 1 can only observe; the causal answer needs a later step.

20. **[The guidance scale is a clear-speech dial: guided flow-matching TTS over-articulates the way people do for a listener](designs/tts-clear-speech-guidance.md)**  
   Status: queued, 2026-11-09. Search gate passed on the sixth pass: no work measures vowel area or SRT against guidance scale. The text/prompt split of guidance is not new (MegaTTS 3, DualSpeech, Zheng and Maleki), so the novelty is the clarity measurement, not the decomposition. Two cited-by lists stayed rate-limited; recheck them when writing `plan.md`.  
   Why here: Whether guidance makes synthetic speech over-articulate the way people do for a listener. Waits on its search gate, because one group published three 2026 clarity papers with these measures.

21. **[Late turn-taking in duplex speech models: sampler, stimulus, or model?](designs/duplex-turn-temperature.md)**  
   Status: queued, 2026-11-09. Search gate passed 2026-11-09 (22 queries, none refused); partly anticipated, not scooped.  
   Why here: Whether late turn-taking in a duplex speech model is a sampler artefact. Partly anticipated by the model's own paper; the open value needs Rob to record stimuli.

22. **[Does the language model's left-to-right reading of a picture shape its binding errors? Wrap-around pairs first, re-ordered image tokens second](designs/raster-binding.md)**  
   Status: queued, 2026-11-09 (revised after hostile review). Search gate cleared: a third pass ran the 12 owed searches with none refused and found no scoop; one small design addition (absolute-position split) followed.  
   Why here: Whether a vision-language model confuses colours of objects that sit near each other in the token string rather than in the picture. Cheap on a 0.5B model; the prior favours a null.

23. **[Binding by position: does a position-tagged bag of words tell FLUX.1 which colour goes with which object?](designs/dit-positional-binding.md)**  
   Status: queued, 2026-11-09 (revised after hostile review; search gate passed 2026-11-09, not scooped)  
   Why here: Whether position tags in a diffusion transformer prompt do binding work. Cheap first slice with a clean stop; claim limited to one model family.

24. **[The search cutoff: which entropy sets the cost of language-model tree search (renyi-search-cutoff)](designs/renyi-search-cutoff.md)**  
   Status: queued, 2026-11-09. Rebuilt after a hostile review. Search gate passed on the second pass the same day: no scoop of the per-sentence predictor, but password research already has a sampled per-target rank estimator, now added as a second control.  
   Why here: A theory-backed second number next to perplexity for search. Rebuilt after a fatal objection into a narrower per-sentence prediction; CPU only.

25. **[The denoiser talks back: what an image model's text stream knows beyond the image stream](designs/denoiser-talks-back.md)**  
   Status: queued, 2026-11-09. The novelty search gate passed on 2026-11-09: partly scooped, core question still open (see "Search gate, 2026-11-09").  
   Why here: Whether a diffusion transformer's text tokens carry seed-specific image information. Two gates before GPU time and a heavy deciding step.

26. **[The linear-response window: over fifteen decades of perturbation size, when does training answer in proportion](designs/linear-response-window.md)**  
   Status: queued, 2026-11-09. Search gate cleared 2026-11-09: twelve searches ran, none found the prediction-space curve below 1e-5 against injection time (see Search gate). Revised after a hostile review (record at the end). Designed, not run.  
   Why here: How small a weight nudge must be for training to answer in proportion. No GPU needed, so it can run beside GPU cards; weak carry to large models.

27. **[What temperature does in continuous-token autoregressive models: the innovations test, then a two-sided feedback map](designs/continuous-ar-temperature.md)**  
   Status: queued, 2026-11-09. Search gate passed the same day; neither stage is published. Revised after a hostile review and after the search; records at the end.  
   Why here: What temperature means in continuous-token generators, by inverting real tokens through the head. Half the idea is occupied by a 2026 paper.

28. **[Surprise feedback: tell the model how wrong it just was (round 2 S12)](designs/surprise-feedback.md)**  
   Status: queued, search gate cleared 2026-11-09 (twelve keyword queries ran; no controlled transformer version of the idea found). Revised after a hostile review and two search passes, all recorded at the end. Designed, not run.  
   Why here: Feed a model its own prediction error as an input. A build, self-contained on synthetic data; the lead's prior is low after the review.

29. **[Does a token cost the same energy whatever it says? Content-dependent joules, and renumbering hidden units to save power](designs/content-energy.md)**  
   Status: queued, 2026-11-09. The search gate passed on 2026-11-09 (see "Search gate"); Steps 2 and 3 now wait only on Step 1. Revised after a hostile review (record at the end).  
   Why here: Whether a token costs different energy depending on what it says, on the Mac. Step 1 is an instrument check other energy work can reuse; later steps have weak priors.

30. **[Semantic latents and the memorization threshold](designs/semantic-latent-memorization.md)**  
   Status: queued, 2026-11-09. Step 0 needs Rob. Search gate passed on the second pass: 10 searches ran, nothing found measures the threshold across input spaces at matched information; two last searches (REPA, other wordings) were refused and stay owed.  
   Why here: Whether a semantic latent lowers the data a diffusion model needs before it stops memorizing. Blocked on Rob accepting the DINOv3 licence.

## Parked

Kept because the reasoning is worth having; each card says what would unpark it.

- **[Paired-metric floor: park as research, fold the calibration into the next INT8 fidelity card](designs/paired-metric-floor.md)**. parked, 2026-11-09. Search gate still open: all 3 queries on 2026-11-09 were refused (budget spent), so no independent scooping search has run.

- **[Nested bit-planes as precision on demand (merged cluster)](designs/nested-bitplanes.md)**. parked, 2026-11-09. A hostile review found the headline experiment gives no benefit on this hardware and is largely published. The one open part needs a different design and a prior-art search first. The search gate is still open: on 2026-11-09 every search query was refused again because the shared budget ran out. The record is at the end.

- **[Positive-only self-training: which pass rate is worth the most per attempt?](designs/self-training-37.md)**. parked, 2026-11-09. Search gate still open: every WebSearch query was refused again on 2026-11-09, so only known pages were read. Unpark when step 0 (CPU only) or a completed search gives the 1/e claim a model to stand on.
