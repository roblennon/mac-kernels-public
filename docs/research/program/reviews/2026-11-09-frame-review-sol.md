# Review of the programme frame, 2026-11-09

Reviewer: GPT-6.1 Sol at medium reasoning, through the Codex command line, read-only. The frame was written by Claude Fable 5.2, so this is a different-family review. The reviewer saw `FRAME.md`, `TASKS.md`, the research statement and the funder research.

## What was done with it

Each point was weighed against what it costs on a run this size. Most landed, some in a narrower form.

- **Check the measuring code before trusting it.** Taken. Every experiment now runs an identity case that must read zero and an injected effect that must be detected, before the real run is read.
- **Reserve a confirmation run.** Taken as a labelling rule, since the budget cannot promise a confirmation for every thread. A first run is an observation. It becomes a result only when it recurs on inputs it has not seen and on more seeds.
- **Freeze the decision rule first.** Taken. The slice plan is committed before the run and later changes are added below it, never edited in.
- **Matched comparisons and a noise floor.** Taken. It was already the practice in this repo's benchmarks and is now written into the protocol.
- **Separate what a small test can reject from what it says about big models.** Taken. Each slice file ends with that statement.
- **Kernels are sequencing, not out of scope.** Taken. Reworded.
- **"Run it" was cited without a quote.** The words are in Rob's handover message. The quote is now in the frame.
- **Coverage across modalities should be checked, not just permitted.** Taken. The directions document carries a coverage table.
- **"Up to four" allows zero experiments.** Changed to at least one, at most four.
- **Exact deadline, enforced GPU limits, disk headroom.** Taken.
- **The Modal volume already costs money.** True and already parked for Rob. This run cannot delete it, because nothing was asked of that account.

## The review as returned

The frame could produce a convincing false finding while satisfying every stated safeguard. Prediction first prevents hindsight only if the plan is frozen; a control can share the treatment’s bug; a kill number can be arbitrary; and a skeptic can review the same mistaken implementation without detecting it. The research statement already documents relevant failures: training images called validation, dtype confounds, thin variance, and missing gradient-fidelity evidence.

I would add these five requirements, ranked by likely effect on the outcome:

1. **Verify the measurement independently before interpreting it.** Require a trusted reference on tractable inputs, an identity/no-effect case, and an injected effect the measurement must detect. For agent-written code, the skeptic must inspect executable code and raw outputs, not just the experimental narrative. A second model family helps, but does not establish independence of evidence.

2. **Reserve a confirmation run.** One seed is exploration. A surviving effect must recur on untouched inputs and additional seeds, within a budget fixed beforehand. Repeating timings on the same generated sample does not establish quality robustness. If confirmation cannot fit, label the result an unconfirmed observation.

3. **Freeze the complete decision rule before execution.** Record the primary metric, evaluation data, baseline, meaningful effect threshold, repetition count, exclusions, and stopping rule in a dated commit. Retain all attempts and amendments. Otherwise, four slices containing many prompts, metrics, and code revisions become an undisclosed search for a winner.

4. **Require matched comparisons and uncertainty.** Hold dtype, inputs, precision policy, and workload constant; synchronize GPU timings; separate compilation and warm-up; interleave comparison order across fresh processes. Require observed gains to exceed measured variability. The repo’s documented 2–4% baseline drift can swallow its smallest claimed effects.

5. **Separate falsification from generalization.** State exactly which mechanism a small experiment can reject and which larger-model claim remains untested. Where feasible, check a second model or size. A positive result on one checkpoint must not acquire broader scope through the write-up.

The reading of the ask mostly fits, but changes its emphasis. “New kernels” is explicitly requested; excluding kernel construction is reasonable for this first phase, but should be described as sequencing rather than outside the ask. Conversely, “Run it” is invoked as authorization without appearing in the quoted evidence. The full-surface search also risks staying quantization-led: permitting another subject is weaker than checking coverage across the modalities and programme interests requested. Documenting all funding options is part of the quoted ask; the background file may satisfy that, but completion should explicitly point to it.

The done points are partly checkable. Files, links, numbers, and reviews are observable. “Every idea considered” needs a ledger; “up to four” permits zero experiments; “cheapest experiment that would settle it” overstates what a pilot can establish. The checkpoint is planned but absent from the done list. Define what constitutes an adequate search and distinguish “unknown” from “not novel.”

The likeliest false stated assumption is that a few Mac-hours establish something about large models. Billion-parameter checkpoints alone do not justify transfer. The missing assumption is that the evaluation measures the intended scientific property independently of implementation artifacts.

The budget is insufficiently protective. “About twelve hours” lacks an exact deadline. The GPU wrapper serializes jobs, but monitoring and time limits are optional; the frame never requires them. One GPU job can still consume the owner’s shared memory and responsiveness. Require enforced resource limits and live disk headroom covering caches, unpacking, outputs, and swap. Repeatedly deleting and downloading models also wastes time. The parked Modal storage charge already contradicts any practical reading of “no paid compute” as zero ongoing cost.
