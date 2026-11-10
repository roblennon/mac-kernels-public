# Research programme: frame and plan

Written 2026-11-09 by the agent holding the helm, while Rob was away. Rob has not confirmed this frame yet. Anything here that he overrules gets changed here first, because every reviewer and worker reads this file before anything else.

## The ask, in Rob's words

From the request that started this work:

> We aren't going to apply to any just yet today. But do document all of these, and lets keep in mind the Dec 17 Fellowship. But before we attempt that, we need to flex our minds a bit. [...] I want you to get creative, think of novel insights, question assumptions, deconstruct using multiple mental models, really go at the full surface area of what they programs cover. For resources, we only have my Mac studio, a 2017-era gaming PC, and what we can access for free on Modal, Kaggle, and other such sites. So it may make sense to really focus on things that are able to be tested with what we have access to, and hopefully generalize well to larger models. This can be LLMs, multi-modal, image/video, speech synthesis, etc.

> new work, new kernels, new machine learning findings of all sorts [...] Not just taking some pre-existing idea from CUDA and doing it on Metal (although I do want to do a lot of that too because I like to run stuff on my Mac) [...] push things further [...] make a name for ourselves and publish something that advances the state of the art.

From the follow-up that handed over the helm:

> Good. Keep at it. From here on, you have the helm. [...] SLICE PLAN -> Re-review with fresh eyes -> IMPROVED SLICE PLAN -> Code it/Run it/Do it/Research it -> Optional sub-agent skeptical review of results -> Next Slice [...] Take no shortcuts. [...] Work slow, bit by bit [...] State hypotheses and assumptions out loud, then question them. Or have a subagent question them.

An earlier correction that still governs how research is done here: "do your research and actually read things, don't just theorize."

## Amendment, 2026-11-09 evening: ideas now, experiments later

Rob read the first report and changed the shape of this pass:

> I actually want you to focus on queuing up the ideas but you don't need to run experiments until the lab is fully constructed. let this pass be one of creativity, research, musing, wild thoughts, hidden theories, and otherwise be the prep work to make sure that when we do run experiments, they're the best ones we could run

This replaces item 3 of Done and lifts the cap on idea rounds. No experiment runs in this pass. The work is now wider thinking and deeper preparation: more idea rounds from angles the first two did not take, theories worked through to predictions, and for the strongest threads an experiment design good enough that the lab can pick it up cold. The section on how an experiment earns belief stays, as the standard those designs are written to.

Done, as amended:

1. The directions document, as before, now covering every round.
2. `QUEUE.md`: the experiments worth running first, in order, with the reason for the order.
3. One design card per queued experiment under `designs/`. A card states the hypothesis, the theory behind it, what is already known with links, the prediction as a number, the control that could embarrass it, the planted-effect check, the confounds a hostile reader would raise, the cost on this hardware, and what a yes and a no would each change. Each card has been attacked by a skeptic before it is queued.
4. `MUSINGS.md`: the wilder thoughts and half-theories that are not experiments yet, kept because a later reader may see what to do with them.
5. The checkpoint note.

The 08:00 stop stays. Device time is no longer needed, so the GPU rules above apply to nothing in this pass.

## How the ask is read

Rob asked for ideas worth publishing, checked against what already exists, and sized to hardware he owns. He did not ask for an application, and he did not ask for a product. Running a cheap first experiment is read as inside the ask, because "testable with what we have" is the selection rule he gave and "Run it" is in the loop he described. New kernels are part of what he asked for, and they come later in the sequence: a kernel is worth writing once a measurement says what it should do. A paper draft or a new subsystem waits for him. Those get proposed in the report, not started.

The second handover message reads as written for a consolidation project (it talks about preserving craft, tooling and data that are already mapped). What carries over to this work is the method: small slices, a skeptic on each, durable notes, assumptions said out loud.

## What kind of work this is

Personal research before any publication. Nobody is waiting on it and nothing here can break a user. The cost of a mistake is wasted days, or worse, a finding Rob repeats in public that turns out to be an artifact. So the proof that matters is the scientific kind: a prediction written down before the run, a control that could embarrass the result, and a reviewer told to break it.

## Done

Three things Rob can open when he is back.

1. `docs/research/2026-11-09-research-directions.md` holds every idea considered, the closest existing work with links and dates, what is still open, and the cheapest experiment that would settle it. Each novelty verdict says whether it rests on a real literature search or only on papers a reviewer already knew.
2. A short ranked list of the threads worth Rob's time, with the reason for the order.
3. For at least one and at most four of those threads, a first experiment that actually ran on this Mac, with its numbers, its control, and a skeptic's review, each in its own file under `slices/`.

4. A checkpoint note saying what was learned, what was killed, what is parked, and where the funder research already sits (`docs/review/2026-11-09-funder-research.md`).

A thread killed by its own first experiment counts as done. A null result written up honestly is a deliverable.

## Budget

This is an unattended run, so these limits are stops, not suggestions.

- Two idea-search rounds in total. The second is running. No third round without Rob.
- At most four experiment slices. A fifth is proposed, not started.
- The run stops at 08:00 on 2026-11-10, local time, whatever state it is in, and reports.
- No paid compute. Modal launches are blocked in this session anyway. Free CPU and the Mac's GPU only.
- Disk: the Mac had about 30 GB free at the start. Free space stays above 15 GB at all times, counting caches, outputs and swap. A model that more than one slice needs is kept until the last of them closes.
- One GPU job at a time, through `bench/with_gpu.sh`, which takes the lock other sessions on this Mac also take. Other agent sessions share this machine today, including the one building Rob's research lab project, so a look at the process list comes before any device run, every device run carries the runner's deadline and memory bounds, and experiments here prefer CPU or a few minutes of GPU over anything long.

## Neighbours

Rob's separate research lab project (a sibling checkout named `model-research-lab`) is being built in another session. It will be the long-term home for experiments like these. This programme does not write there. What it produces is meant to be picked up there: the directions document is a list of candidate research questions, and each slice file is written so a cold reader can rerun it. That project's own rule is worth borrowing here: a measurement counts only with its receipt, and a result inside the session's own noise is called inconclusive.

## Undo

Everything is a new file in git or a model in the Hugging Face cache. Reverting the commits and deleting the downloaded models undoes all of it. Nothing is pushed.

## Plan

The work is cut so that each piece can be checked by someone who was not there when it was made.

**Slice A: finish the idea search.** Round one produced 71 ideas, nearly all about quantization, and its novelty checks ran without any keyword search because the search budget ran out. Round two repeats the checks with real searches and adds a round where quantization and kernels may not be the main subject. Output: the directions document.

**Slice B: choose what to test.** Rank the surviving ideas by a rule stated before looking at the list: how cheaply one experiment settles the question, how surprising a yes would be, how much a yes would change what people build, and whether the result would carry to larger models. A skeptic reviews the ranking before any experiment starts. Output: `PORTFOLIO.md`.

**Slices C1 to C4: first experiments.** One file each under `slices/`. The file is written in this order: hypothesis, the assumptions it rests on, the prediction with a number, the control, the kill number, and only then the method. A skeptic reads the plan before the run and the result after it. The plan review can run while the code is being written, since most of what it finds folds into the next step.

**Slice D: checkpoint.** What was learned, what was killed, what is parked, and what an honest Fellowship letter could claim. Handoff notes for Rob's agentware project.

## How an experiment earns belief

A reviewer from another model family read this frame and pointed out that every safeguard in it could be satisfied by a false finding ([the review](reviews/2026-11-09-frame-review-sol.md)). What follows is the answer.

The plan for a slice is committed before its run. It names the one metric that decides, the data, the baseline, the size of effect that would matter, how many repeats, and when to stop. Anything changed afterwards is added underneath with a date, so a reader can see what was planned and what was improvised.

Before the real run is read, the measuring code has to pass two checks. Comparing a thing with itself must read zero, or the session's noise. An effect of known size planted on purpose must be found at about that size. A measurement that fails either is fixed before anything is interpreted.

Comparisons hold everything else still: same inputs, same data type, same seeds, arms interleaved, timings synchronised. A difference inside the spread of the baseline against itself is called inconclusive.

A first run is an observation. It is called a result only after it recurs on inputs it has not seen and on more seeds. Where the budget does not stretch to that, the write-up says "unconfirmed observation" in its first line.

Each slice file ends by saying which mechanism the experiment could have rejected, and which claim about larger models it leaves untested. A finding on one checkpoint stays a finding on one checkpoint.

The skeptic reads the code and the raw output, not the summary.

## What is deliberately not in the plan

Writing any part of an application. Building new Metal kernels (later in the sequence, see above). Experiments that need the gaming PC, whose GPU is still unknown. Anything on Kaggle or Colab, which need Rob's login. A paper draft. These are real and they are parked in `TASKS.md`.

## Assumptions, said out loud

Each of these could be wrong, and the plan review is asked to attack them.

- A first experiment that fits in a few hours on one Mac can say something true about large models. This is the weakest assumption in the programme. It holds best for training-free measurements on public checkpoints of a billion parameters or more, and worst for tiny models trained from scratch.
- Reviewers from the same model family as the author share its blind spots. Where a second family is reachable from this machine, experiment results go to it.
- Novelty verdicts from agents are provisional even with search. A thread Rob wants to publish needs one more pass by a person reading the three closest papers.
- The measuring code measures the intended property and not an artifact of how it was written. The identity and planted-effect checks exist because this cannot be assumed.
- The gaming PC is assumed to have an 8 GB card that cannot do fp16. Nothing depends on it yet.
