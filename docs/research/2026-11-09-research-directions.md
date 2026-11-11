# Research directions, 2026-11-09

This is the map of an idea search run on 2026-11-09, before any experiment. It says what was looked at, how, what the search found already done, which threads kept coming back, and where the queue of experiments came from. The experiments themselves are in [the queue](program/QUEUE.md), each with a design card under [designs](program/designs/). Every one of the 314 ideas is listed, with its claim and the work its author found, in [the idea ledger](2026-11-09-idea-ledger.md). The wilder thoughts that are not experiments yet are in [the musings notebook](program/MUSINGS.md).

## What needs Rob

Update, 2026-11-10: the lab ran the first card and it decided nothing, because the model could not do the base task; the card was redesigned and every queued card now has a subject gate. The decisions now waiting on Rob are listed in [the tracker](program/TASKS.md#waiting-on-rob). The original line follows. Nothing here needs a decision before the lab can start. The queue is ordered so its first experiments run inside the lab's current limits, on the Mac, with no login and no spend.

Three things only Rob can give will open up later parts of the queue. The gaming PC's GPU model decides whether any CUDA control runs at home. A Kaggle or Colab login decides whether the training-from-scratch steps run on free T4s or stay small. And several speech cards end in a listening test, where Rob's own ears are the instrument; the cards say where.

## Why this exists

Rob asked for "novel insights" across "the full surface area" the funders cover, testable on a Mac Studio, a 2017 gaming PC and free cloud tiers, and likely to "generalize well to larger models". He asked for findings, not only CUDA ideas ported to Metal, and for work that would "advance the state of the art". Later he changed the shape of this pass: queue the ideas and design the experiments, and run nothing until his research lab is built ([the frame](program/FRAME.md) records both).

## How the ideas were made

Three rounds, forty-four thinking lenses, 314 ideas.

The first round used nine lenses (compression, cross-modal, human-scale, inversion, measurement, outsider, physics, redundancy, small science) and produced 71 ideas. Nearly all of them were about quantization and kernels, because that is what this repository already does. Fourteen survived a first check, but that check ran without keyword search, because the search budget ran out.

The second round did two things. It re-checked those fourteen with real searches, each agent running dozens of queries and opening the closest papers. And it ran eight new lenses where quantization was not allowed to be the subject: anomalies, architecture, generative modelling, inference algorithms, interpretability, learning, origins of past breakthroughs, and speech. That produced 48 more ideas and a shortlist of twelve.

The third round ran twenty-seven lenses chosen to be unlike each other: borrowed theory (rate-distortion, statistical physics, information geometry, dynamical systems, numerical analysis, control, neuroscience and biology, old engineering tricks), disputes in the literature (folklore nobody measured, contradictions between papers, beliefs turned backwards, future-work sections, a view back from 2031), Rob's own ground (his results, his instruments, the Mac as a measuring device, what a person with no boss can do, a human in the loop), and areas the first two rounds avoided (tiny-model science, data, data as the subject, building something new, learning beyond pretraining, multimodal understanding, agents and long horizons, speech as a modelling problem). A critic read the first pass and named six regions still missing; six fill lenses covered them. That round produced 195 ideas.

Then sixteen judges, two per slice of the ideas, nominated the strongest. One judge in each pair chose for reputation: would a researcher at a serious lab repeat this to a colleague the same week? The other chose as a hostile reviewer: is the prediction real, could the control embarrass it, is it already published? A separate reader merged the duplicates across lenses. A fifth reader read every lens's own account of how it thought and wrote the sceptical essay that opens the musings notebook.

Thirty-three ideas went on to design cards. Each card had its own real prior-art search, was written by one agent, attacked by a second agent told to break it, and revised by a third.

## How much to trust the novelty verdicts

Only two sets of verdicts rest on real searches: the fourteen round-one rechecks and the thirty-three design cards. The novelty claims in the rest of the ledger come from the idea's own author, often during throttled searches, and a missing reference there means "not searched", not "open". Every verdict is an agent's reading. Before Rob publishes any thread, a person should read the three closest papers on its card.

## What the searches found already done

The round-one rechecks were humbling, and that is useful. Five of the fourteen came back mostly done.

- How often community quantized files are broken is already counted: one audit found problems in about 14 percent of 185 repositories, 27 percent weighted by downloads, and a 2026 paper covers it.
- Whether quantization brings back concepts that were erased from a diffusion model has two 2026 submissions saying it does not, on SD 1.5.
- A fixed-width version of WaterSIC's per-column bit allocation is mostly covered: a 2026 paper finds rotation leaves only about 0.04 bit of gain over GPTQ. What is left is a decode-cost question on GPUs without matrix units, which the M1 Max is.
- An additive-noise law for logit damage was rejected by a 2026 paper across 642 damaged conditions.
- A null result for searched kernels has Metal-specific prior work already.

The other nine came back partly open, usually narrower than first proposed: the step-to-step correlation of quantization error in diffusion, the noise floor of same-seed image metrics, a LoRA folklore audit, mixed-precision retrieval, the Muon quantization puzzle, sibling-model deltas, a slow axis in speech models, how training's ending affects stored bits, and Whisper's invented text under quantization. Each of those has its prior art listed in [the ledger](2026-11-09-idea-ledger.md#round-1-ideas-rechecked-with-real-search-round-2).

The lesson the searches taught: in 2026 the obvious version of almost any quantization idea is taken, often by a low-visibility Zenodo record or an OpenReview submission. What stays open is the control nobody ran, the measurement a paper named as future work, or the same question on a model family nobody tested.

## Threads that kept coming back

The merge reader found about fifty clusters of ideas that were the same idea in different words. The most useful finding is not any single cluster but the ten threads that lenses with nothing in common reached on their own. Each is a hidden theory: if it holds, several separate problems are one problem. What follows is the merge reader's account, shortened. These are hypotheses, not findings.

**Rounding error is a bias, not noise.** A quantized weight adds the same error at every step, token and sample. So its damage builds like a bias: it may pull generation into loops, leave a signature a linear probe can read, and get learned by adapters trained on top. If so, re-randomizing the rounding across steps, or subtracting one anchor correction, is a general lever. This sits squarely on Rob's kernels.

**One noise margin per layer.** Lenses from physics, control, old engineering and psychophysics each reached a per-layer tolerance that would price quantization, caching, token merging and activation rounding together. One lens predicted the opposite: robustness is several separate traits. Either answer settles something the field assumes.

**Rare failures follow exponential laws in temperature.** Skips, loops, stops, turn-taking and falling apart were predicted, by six lenses, to be rare events whose rate goes as exp(-gap/T). If so, reliability can be measured by accelerated testing at high temperature, the way hardware is qualified, and some behaviours blamed on the model are sampler artefacts.

**Models count copies in their context as new evidence.** Attention sums duplicates. Repeated failed calls, repeated claims, stale observations and generated speech may reinforce themselves for that reason. Sycophancy, agent loops, context rot and voice drift may share one mechanism, and a toy model with an exact Bayes answer can measure it.

**Fixed grids leave errors locked to the grid phase.** The patch grid in a diffusion transformer, image tiling in vision-language models, Whisper's frame hop and audio sub-frame shifts each showed errors that move with grid position, not content. The cheap common fix is to randomize the phase per step and average.

**Generators commit early.** Flow-matching speech fixes word timing early; autoregressive image models may plan layout in the first rows; prose habits are chosen at the last prompt token. Guidance, precision and compute could be scheduled by commitment point instead of spread evenly.

**The measurement stack is as large as the effect.** Stack defaults, trajectory noise, the recognizer used for scoring, and arithmetic differences often match the quantization and acceleration differences being compared. A shared noise-floor protocol is a methods contribution in itself, and it tells Rob which of his own comparisons are real.

**One scalar calibration absorbs several distortions.** The best weight on a model's conditional prediction may be above one; a quantized model may be too flat; critical temperature may sit below one. Many sampling knobs could be set by measurement rather than swept.

**Fine-tunes are low-dimensional shifts in a shared basis.** Subliminal trait transfer, LoRA acting like a steering vector, sibling deltas and merging of reward-trained adapters may be one phenomenon. Kilobyte personal adapters and audits for hidden traits would rest on one measurement.

**What is learned early, or with temporary help, persists.** Learning order and early scaffolds may fix internal solutions that two models with equal loss do not share. Training-speed records would then need a second metric.

## A critic's warning, and how the queue answers it

The reader who went through every lens's own account was not kind, and was right on several counts. Its full essay opens [the musings notebook](program/MUSINGS.md#a-sceptical-reader-on-the-whole-notebook). Its main points:

The forty-four lenses share one voice: take an exact theorem from another field, line it up against one published number, report a fit. The 314 ideas are far less independent than the count suggests; one cross-pollination lens found 71 of them rest on about five hidden quantities. A portfolio drawn naively would be one bet made many times.

The shared taste is a referee's, not an inventor's. Almost every idea is defined by a gap between other people's papers. Only one lens asked what should exist.

With 330 hypotheses drawn from the same few public models, a few close numerical matches are expected by chance. Only one lens proposed committing predictions by hash before looking at data, and that standard should apply everywhere.

Round two banned quantization as a subject, which was the one area where Rob has working code, measured results and puzzles nobody else holds. Data and instruments only Rob has (his labelled projects, his kept-versus-discarded image log, his voice lab) are the most defensible advantage in the notebook and got the least weight.

The notebook is strangely innocent about security: mismatched chat templates, sampler order, a 26-byte update that adds 29 points and subliminal transfer are all treated as curiosities rather than attack surfaces.

The critic's candidate for a year-long question is how much of a generative model's decline over a long horizon comes from the decoding loop rather than the model: temperature, errors that repeat every step, the prompt leaving a finite window, the transcript read as evidence about its own author. It has an exact null and can be tested on toys. The case against: it is where well-funded labs already are, it needs long generations and many seeds, and the likeliest answer is "partly both", which is hard to publish.

The queue answers this in four ways. It puts quantization and Rob's own ground back in (bias versus noise, fine-tune survival under requantization, nested bit-planes, cross-stack numerics, the energy of a token on the Mac). It tried to include builds, and here it only partly succeeded: removable training scaffolds and surprise feedback survived as queued builds, emit-hot-feed-cold sampling survives as the second stage of the continuous-token card, and nested bit-planes was parked after its review found the headline gave no benefit on this hardware. The queue is still mostly measurement, and the critic's point stands as a gap for the next round. It gives the security reading a card (subliminal transfer through images). And every card writes its prediction down as a number before any run, with a planted-effect check, as the frame requires; pinning those predictions with a hash in the lab before the first run would answer the critic's point about chance matches.

## Shared instruments

The merge reader also found fourteen experiments that would each settle several ideas at once. They matter for planning, because the lab gets more out of building one good instrument than out of thirty one-off scripts. The design cards say which instrument each one would use.

1. A tensor bank with a noise-injection gain table on Wan 2.1 1.3B, Krea 2 Turbo and Qwen2.5-0.5B: save layer inputs and outputs once, inject noise per layer, predict damage from real 4-bit weights, cache lag, token merging and activation rounding.
2. A fixed-versus-fresh error harness: matched-norm real 4-bit, fixed Gaussian, fresh per-step noise and floor/ceil alternation, on one few-step diffusion model and one small language model.
3. A many-takes speech corpus: about 30,000 clips from two or three open autoregressive TTS models across temperatures and lengths, force-aligned.
4. A Pythia temperature atlas: long samples across a fine temperature grid with single-token insertions.
5. A layer-intervention harness on frozen 0.5B to 8B models: duplicate, scale, skip and extrapolate every span.
6. A disjoint-halves image training family on CelebA-64 and CIFAR across latent spaces and objectives.
7. A LoRA grid on one image model and one TTS model across training-set size, steps, rank and storage bits.
8. A cross-stack numerics and sampler probe across MPS, MLX, llama.cpp Metal and Vulkan, WebGPU and CUDA.
9. A Whisper probing bench across sizes, precisions, sub-frame shifts, padding and normalization.
10. A speaker-count and pre-pretraining TTS family at fixed hours.
11. A small-GPT testbed with planted facts, controlled data and seeds crossed by initialization and data order.
12. A nested bit-plane diffusion file driven through Rob's Metal kernels and packed trainer.
13. A diversity census on few-step image models: 10,000 seeds per prompt family.
14. A synthetic agent-loop bench with injectable stale observations, rival strings and repeated failed calls.

## The queue at a glance

The ordered list and the reasons are in [the queue](program/QUEUE.md). Thirty cards are queued and three are parked. None was dropped outright: the prior-art searches narrowed several (agent-hazard-mixture, content-energy, noise-margin-gain-table, renyi-search-cutoff, rare-exploit-onset, among others) and the cards now cite the overlapping work instead of claiming it.

The top of the queue, in one line each:

1. Whether a replayed error in an agent's context hurts fresh answers more than a one-line summary of it.
2. How Apple's stock matmuls round and add, which every Mac measurement here rests on.
3. What Whisper reads from its padding: a stop signal, an attention sink, or nothing.
4. Whether one number per layer predicts which fine-tunes survive rounding to 4 bits.
5. Whether a transformer's response to injected noise bends away from its own linear prediction, and whether one per-layer number prices many inference shortcuts.

A reviewer from a different model family read the first draft of the order and moved cards whose first slice only shows feasibility below cards whose first slice decides something; [its review](program/reviews/2026-11-09-queue-review-sol.md) is kept.

Spread across the thirty:

| Area | Cards (queue position) | Kinds |
|---|---|---|
| speech | [whisper-silence](program/designs/whisper-silence.md) (3), [tts-slip-hazard](program/designs/tts-slip-hazard.md) (8), [voice-clone-circuit](program/designs/voice-clone-circuit.md) (13), [whisper-grades-students](program/designs/whisper-grades-students.md) (19), [tts-clear-speech-guidance](program/designs/tts-clear-speech-guidance.md) (20), [duplex-turn-temperature](program/designs/duplex-turn-temperature.md) (21) | audit, measurement, mechanism |
| image generation | [img-subliminal](program/designs/img-subliminal.md) (9), [fixed-vs-fresh-error](program/designs/fixed-vs-fresh-error.md) (6), [guidance-optimal-weight](program/designs/guidance-optimal-weight.md) (15), [variety-census](program/designs/variety-census.md) (11), [sampler-backward-error](program/designs/sampler-backward-error.md) (12), [dit-positional-binding](program/designs/dit-positional-binding.md) (23), [denoiser-talks-back](program/designs/denoiser-talks-back.md) (25), [continuous-ar-temperature](program/designs/continuous-ar-temperature.md) (27), [semantic-latent-memorization](program/designs/semantic-latent-memorization.md) (30) | measurement, mechanism |
| text models, inference | [copies-as-evidence](program/designs/copies-as-evidence.md) (1), [noise-margin-gain-table](program/designs/noise-margin-gain-table.md) (5), [finetune-survival-requant](program/designs/finetune-survival-requant.md) (4), [layer-repeat-step-size](program/designs/layer-repeat-step-size.md) (16), [agent-hazard-mixture](program/designs/agent-hazard-mixture.md) (17), [renyi-search-cutoff](program/designs/renyi-search-cutoff.md) (24) | law, measurement, mechanism |
| text models, training | [rare-exploit-onset](program/designs/rare-exploit-onset.md) (10), [train-scaffolds](program/designs/train-scaffolds.md) (14), [rare-feature-threshold](program/designs/rare-feature-threshold.md) (18), [linear-response-window](program/designs/linear-response-window.md) (26), [surprise-feedback](program/designs/surprise-feedback.md) (28) | law, measurement, mechanism, method |
| vision-language | [raster-binding](program/designs/raster-binding.md) (22) | measurement |
| Mac systems and numerics | [cross-stack-numerics](program/designs/cross-stack-numerics.md) (2), [content-energy](program/designs/content-energy.md) (29) | measurement |

Every card had a first step sized to one lab slice when the queue was written; the redesigned first card now needs two allowances, a capability step and then its test. Most first steps need nothing but the Mac. The ones that need Rob say so in their status line: semantic-latent-memorization needs a licence click, duplex-turn-temperature's open half needs him to record speech, and several later steps need cloud time or a longer unattended run.

## What was set aside, and why

The ledger holds every idea. These are the notable ones that did not get a card, with the reason, so a later reader does not have to rediscover the reasoning.

- Synthetic or cipher pre-pretraining for speech alignment, and a public TTS speedrun built on it: two lenses converged on it, but it needs two to three weeks of free T4 time and the baseline may not reach intelligibility in budget. A strong candidate for when Rob has a GPU budget.
- Speech re-coded for screen-reader speeds: a real accessibility goal, but the decisive test is a listening study with practised screen-reader users, which cannot be set up from here. Worth raising with Rob as a human-scale project for the Fellowship.
- Re-reading a prompt under the causal mask, and image tokens read before the question in vision-language models: the prompt-repetition paper names the first as future work, and the second may be folklore; prior art was barely checked.
- Predictive-coding depth (spending upper layers only where the model was surprised): a 2026 paper reports adaptive-depth gains in text are small; its exit head is reused in the surprise-feedback card.
- The memorization clock for LoRA (steps per image): judges ranked it low on surprise; it fits as an arm of the shared LoRA grid.
- The annealing tax and cross-fitted distillation: reasonable, but both judges ranked them below the cards that made it.
- Long reasoning as a symptom, with quantized models thinking longer: interesting and cheap, and close to the stop-decision thread; a candidate for the next round of cards.
- Nucleation theory for grokking time, Kibble-Zurek scaling in diffusion sampling, a cache-resident weight-shared model, and several others were killed by their own lens from memory alone, without a paper opened. The musings notebook lists them for a second look.

## Where this sits next to the Fellowship

The funder research is in [the funder notes](../review/2026-11-09-funder-research.md). The Mozilla Fellowship letter of intent is due 2026-12-17 and scores people-centred, open work with a path to adoption. Several queued threads read naturally in that frame: local speech tools that fail people who depend on them (Whisper inventing text over silence, TTS falling apart on long passages), honest measurement of what quantization does to models people run at home, and audits for hidden traits passed between open models. The application itself has to be Rob's own writing; nothing here drafts it.

## Source and inference

Statements about what a paper shows are agents' reports from their reading, with the link given so a person can check. The threads and the critic's essay are interpretation. Nothing in this document is an experimental result; no experiment ran in this pass. One experiment has run since, in the lab: the first card's step 1a on 2026-11-10, which decided nothing; its card holds the result.
