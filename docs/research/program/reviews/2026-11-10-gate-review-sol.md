# Gate review by Sol, 2026-11-10

A cross-family review of the subject gates added to the 30 queued cards on 2026-11-10, by GPT-6.1 Sol (OpenAI, via the Codex CLI, read-only sandbox, high reasoning effort), run under Rob's standing rule that any gate able to block work gets a different-family review for needless slowness before it is adopted. It read `raw/gates-summary.md` (git-ignored) and the cards. Its text follows unedited, except that local absolute paths in its links were made relative. What was done about it is in the cards' review records and in [the gate batch](../GATE-BATCH.md).

---

## 1. Gates that are needlessly strict, slow or fragile

- **[Copies-as-evidence](../designs/copies-as-evidence.md:76):** Four separate 11-of-16 note-reading requirements create a substantial false-stop risk.
  With independent cells, a subject that reads each note correctly 75% of the time passes all four only **43%** of the time.
  Pool balanced count-probe trials within each template, reserve one bounded confirmation for borderline results, and remove the rule that one manipulation failure ends the model’s ladder.
  Run D0 capability first, before building C′, wording-spread analysis and the full six-arm gate.

- **[Voice-clone-circuit](../designs/voice-clone-circuit.md:59):** The frame readout defines right-speaker distance minus wrong-speaker distance, which should be negative for successful cloning, while the gate requires positive G.
  Reverse that distance difference explicitly before adopting the gate.
  An individually 90%-reliable verifier passes 11-of-12 only **66%** of the time.
  Use the deciding frame readout for capability, with disagreement between the two utterance verifiers triggering diagnosis rather than automatic parking.

- **[TTS-clear-speech-guidance](../designs/tts-clear-speech-guidance.md:78):** Four tokens per cell make one failure worth 25 percentage points, and the phoneme posterior has no numerical cutoff.
  Fix the cutoff from real speech first, merge this gate into the existing pilot, and size by usable paired-token count.
  Drop the four-token IQR veto; retain a readable vowel triangle and enough valid pairs for the area estimate.

- **[Duplex-turn-temperature](../designs/duplex-turn-temperature.md:57):** A subject that starts within three seconds on 70% of turns passes 8-of-10 only **38%** of the time.
  Make free-run starts a gross-functionality check, while the computed distribution determines whether the median exists.
  Allow the stated longer-window fallback before declaring the subject absent.

- **[Rare-exploit-onset](../designs/rare-exploit-onset.md:75):** Requiring every pilot onset to be uncensored contradicts the design’s later allowance for censored observations.
  Even eight individually 95%-reliable runs all finish only **66%** of the time.
  Gate on whether usable onset estimates and the planned fit fit the budget.
  Replace the arbitrary 55–85% visible-accuracy band with mixed rewards and measurable visible gradients, which the rho calculation actually needs.

- **[Rare-feature-threshold](../designs/rare-feature-threshold.md:63):** Requiring 0.8 and 0.2 endpoints plus two-rung margins can reject a broad but measurable transition.
  Require a bracketed, estimable 0.5 crossing instead.
  Resolve “72 facts”: the stated five rungs contain 120 facts.

- **[Surprise-feedback](../designs/surprise-feedback.md:64):** Four times an SD estimated from three seeds is an unstable veto.
  Keep the learning and absolute-headroom checks, and let the planned paired comparisons determine resolution.
  Do not require building the Bayes-feedback intervention merely to certify the control.

- **[Noise-margin-gain-table](../designs/noise-margin-gain-table.md:60):** Requiring a 0.1 interval half-width on eight passages overdemands pilot precision.
  Under ordinary square-root scaling, a pilot half-width of **0.2** projects to **0.1** on 32 passages.
  Project to the actual test size before parking, and replace the single global grid-shift rule when different layers fail in opposite directions.

- **[Cross-stack-numerics](../designs/cross-stack-numerics.md:73):** One exact cell currently disqualifies an entire backend–dtype arm.
  Classify cells separately and continue wherever non-exact, deterministic outputs exist.
  Make the repeat call explicit rather than adding another full dry run.

- **[Layer-repeat-step-size](../designs/layer-repeat-step-size.md:79), [agent-hazard-mixture](../designs/agent-hazard-mixture.md:57), and [raster-binding](../designs/raster-binding.md:60):** Accuracy ceilings or required baseline errors can reject subjects whose decisive continuous scores still have room to move.
  Remove those vetoes and assess room in log probability or L.
  For raster-binding, demonstrate partner-colour pull in the logits rather than requiring at least 5% argmax swaps.

- **[Denoiser-talks-back](../designs/denoiser-talks-back.md:55):** Low image-only R² does not prevent text from adding predictive information.
  Drop the 0.1 lower bound and the requirement that seed variance constitute 10% of total variance.
  Keep detectable seed variation and an upper bound tied to the meaningful ΔR².
  Its 80 additional renders are two slices, not a cheap queue-screening gate.

- **[Rényi-search-cutoff](../designs/renyi-search-cutoff.md:77):** A two-bit median prediction gap is stronger than the one-bit deciding improvement requires.
  Loosen the bar to the intended comparison’s actual attainable improvement.
  Reconcile the two-seed throughput probe with its four-seed gate calculation.

## 2. Gates that are too weak to catch the case 002 failure for their card


  Report all-prompt coverage and require adequate usable replies before calculating the abstention contrast.
  Apply the existing truncation remedy before this gate, and validate revised scorer wording on fresh replies.

- **[Noise-margin-gain-table](../designs/noise-margin-gain-table.md:60) and [fixed-vs-fresh-error](../designs/fixed-vs-fresh-error.md:86):** A signal requirement of at least 100 times a zero replay floor permits zero signal.
  Require a positive signal above a numerical bound derived from the actual computation, while retaining the later residual check.

- **[Variety-census](../designs/variety-census.md:73):** LFW false matches need not represent blurry generated portraits.
  Treat the gate’s LFW pass as provisional and move the already-required domain-matched calibration before committing the large teacher sample.
  Twenty matching pairs are not twenty independent observations; a small image cluster can supply them.

- **[Semantic-latent-memorization](../designs/semantic-latent-memorization.md:72):** Memorization in one pixel-matched space does not establish that the same training budget makes the semantic-space model competent.
  Use the already-planned per-space loss curves and matched loss gaps before interpreting reduced copying as generalization.
  A plain-pixel preliminary check can screen trainer functionality, but cannot certify the eventual matched representation.

- **[Img-subliminal](../designs/img-subliminal.md:72):** Generated-digit scores cannot establish capability if either classifier misreads real digits.
  Establish the classifiers’ real-MNIST confusion floors before using the 2% base-confusion veto.
  Reuse the existing classifier setup rather than adding another generated-image gate.

## 3. Gates to drop entirely, because the card’s own design or an earlier check already covers the subject

- **[Content-energy](../designs/content-energy.md:64):** Drop the added gate as a separate decision layer.
  Existing P/H checks establish instrument function, and Step 1 already decides the real GEMM contrast.
  Reconcile the added 1% threshold with the existing 2% kill line in that single decision.

- **[Train-scaffolds](../designs/train-scaffolds.md:95):** Drop the short-budget T/0.3 veto on proceeding to the longer-budget assessment.
  The card explicitly lacks evidence that gains at 2–6M tokens predict gains at 300M.
  Keep the existing affordable-benefit assessment and the room check on Step 2’s own required baseline arms.

- **[Guidance-optimal-weight](../designs/guidance-optimal-weight.md:76):** Drop the positive-caption-benefit prerequisite and the D≥0.01 prerequisite for reporting R.
  The card already restricts unstable w* reporting by D while allowing R elsewhere.
  Algebraically, D=0.005 and w*=3 give R=2%, so the new gate can reject the effect sought.
  A negative caption loss gap can likewise be model error worth measuring.
  Keep the resolution check.

- **Rare-exploit-onset:** Remove the duplicate split-half calculation from the subject gate and consume the existing 1a check.
  Likewise, merge **whisper-grades-students’** resolution bar into its existing 0c sizing decision.
  Neither needs a second execution or approval.

## 4. The batch

**Measure the cheapest prerequisites first, but do not build 30 harnesses to rank 30 cards.**

- **Tier A:** Start with case 002’s existing-record tally and exact rendered system text.
  Follow with the bounded role-swap diagnostic on about 20 paired old-T2 scenarios, including complete-neutral controls and a D0 capability readout.
  Then run rare-feature-threshold’s learning pilot → scalar settings → G95.
  Run rare-exploit-onset’s ordinary CPU first stage next, consuming its checks once.

- **Tier A, deferred within the batch:** Surprise-feedback has no downloads, but still needs its trainer, Bayes calculation and approximately 15–20 GPU-minutes for both tasks.
  Start with switching A only.
  Add regression after that first result rather than making both tasks the admission price.

- **Tier B:** After download approval, consider SmolLM2-only Rényi statistics and Whisper-small’s stock baseline on the shared test-clean subset.
  Finetune-survival-requant should start with base-fact competence, before implementing three quantizers and training probes.
  A small download alone does not make linear-response-window or MNIST diffusion ready.

- **Tier C:** Leave out the audio stacks, FLUX dictionary, Sana, semantic memorization, denoiser’s 80 renders, FineWeb training, and full energy instrumentation.
  Cross-stack-numerics can move earlier only if a genuinely small probe subset can reuse existing code.

For gates embedded in required baselines, run the baseline once, read the gate first, and retain the outputs.
A universal separate gate batch would spend more than it saves on several cards.

## 5. Anything else that would make the gate batch waste the researcher’s time

- Treat missing code, failed capability and insufficient resolution as different queue states.
- Count implementation time, model loads, downloads and researcher labeling alongside device minutes.
- Run one GPU process at a time on this machine.
- Resolve unspecified thresholds and contradictory counts before execution, without expanding every card into another review project.
- Reuse compatible downloads and environments, while checking actual files rather than cache-directory names.
- Keep failed gate values visible, including legitimate nulls, rather than repeatedly adjusting subjects until they pass.
- Predeclare bounded adjustments and record amendments as LESSONS requires.
- Keep the existing 300-second and 8-GiB limits unless the researcher approves an exception.
- None of the suggested runs, downloads or edits was performed for this review.
