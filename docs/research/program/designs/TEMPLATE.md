# Title: the question, in plain words (short-id)

Status: one line, at most about 300 characters. Where the card stands now: queued, parked, handed to the lab, or run, and what it waits on. History goes in the review record, not here.

Needs Rob:
- Each thing only Rob can give, one per line, with the step it blocks (an allowance, a download, a licence, a login, his ears, a call he reserved). Write "nothing" when nothing is waiting on him.

Lab: the lab case, the card commit the lab's plan is bound to, and the last result in a few words. Or "not handed over".

<!--
How to use this file. Copy it to designs/<short-id>.md and replace every paragraph with the card's own text. Keep the headings and their order: build_queue.py reads the header lines and warns about a queued card with no "Step 1, exactly" section or no subject gate. The card holds the current design, not its history. When a review changes it, rewrite the affected text in place and add one line to the review record. Read LESSONS.md before writing or revising a card.
-->

## The question and why it matters

What we want to know, who would act differently on the answer, and why it is worth a Mac's time now. A few sentences.

## Hypothesis

One falsifiable claim, with the direction and size of the effect, and what would make it wrong.

## Theory and assumptions

The mechanism the hypothesis assumes and the rival accounts that predict something else. List the assumptions out loud, and mark the weakest one.

## What is known

The closest prior work, with dates and how much of each source was read (full text, method, abstract). Then the verdict on novelty, and say whether it rests on a real search: how many queries ran, how many were refused, and what was not checked. A verdict from memory or from refused searches is provisional and says so.

## Prediction

Written before any run. The numbers you expect for each arm and each check, and which of them are tests and which are sanity expectations. Guesses are labelled as guesses.

## Step 1, exactly

This section is the run plan, and where the rest of the card seems to differ, it governs. A fresh agent with no other context should be able to run step 1 and judge it from this section and the appendices it names. Keep it under about 2,500 words; derivations, power calculations and file layouts go to appendices.

Inputs come first: the model and its exact checkpoint, the data, the seeds, the hardware limits it runs under. Then the five parts below, in this order.

### Subject gate

Before the step that tests the hypothesis, confirm the model, task, instrument or data has the property the test depends on, with room for the effect to show, against a bar written in advance, with a stop. Name the property, the measurement, the numeric bar (above chance and below ceiling, or whatever "room" means here), what happens on a miss (one retry or fallback, then park), and what it costs. A planted-effect check does not count, because it tests the instrument, not the subject. Use the name "Subject gate" so the queue script and the lab's intake can find it.

### Instrument checks

Checks on the measuring code, each with a numeric pass line and what a failure means (fix the harness, not kill the card):

- Self-comparison or noise floor: the thing compared with itself reads zero, or the session's own noise, measured.
- Planted effect: an effect of known size, planted on purpose, is found at about that size and with the right sign.
- Tolerances: derive the numeric format's own error first (rounding in the data type, accumulation order, nondeterministic kernels), then set each tolerance above it. Say what the bound is and where it comes from.

### The hypothesis test

The arms, the one metric that decides, how many repeats, the pairing, and the order in which arms run.

### Decision rules

In order, so exactly one applies to any outcome: supported, wrong (kill), and inconclusive. Inconclusive is named with its own condition (for example, a difference inside the baseline's spread against itself), and the card says what an inconclusive result leads to. Fix the statistical method and its level here.

### Budget

Device time, wall time, memory, disk and downloads, with what is measured and what is estimated. Name the allowance that must come from Rob, if any, and the stop when it runs out.

## Later steps

Each later step in a paragraph: what it adds, its own gate, and its cost. They are plans, not commitments; each is rewritten before it runs.

## Confounds

What a hostile reader would raise, and how the design answers each, or says it does not.

## Kill criterion

What result ends the line of work, and what result only parks it. A failed check means fix the harness, not kill.

## Cost

The whole card, step by step, on this hardware.

## What a yes and a no change

Who acts differently on each outcome, including an honest null.

## What carries to larger models

What the result would say beyond this checkpoint, and what it leaves untested. A finding on one checkpoint stays a finding on one checkpoint.

## Receipts

What to keep from each run: the plan committed before the run, code and commit hash, configs and seeds, raw outputs, check outputs, versions.

## Result of step N

One section per lab run, added after it, with the date, the lab case and the card commit it ran against. This section is the single source for any reading of that run. Messages and other files point here instead of restating the numbers. If a reading here is corrected, the correction lists every place the old reading went, so each can be fixed.

## Review record

One dated line per review round, at most two sentences: what was checked and what changed. Detail belongs in the text it changed.

- YYYY-MM-DD: who reviewed (hostile, cold read, cross-family), what changed.

## Appendices

Derivations, prior work as read, the search log, power and memory arithmetic, implementation detail.

## Questions for every attack and every cold read

These stay in this file; leave them out of the card. A reviewer checks what the artifact puts in front of it, so every review prompt asks all six.

1. The subject. Does the model, task, instrument or data have the property this test depends on, with room for the effect to show? What cheap pilot proves it, against what bar, and where is the stop?
2. The instrument. Does it measure the intended property? Do self-comparison and the planted effect pass, and are the tolerances above the format's own error?
3. The decision. Are the rules fixed before the run, in order, with an inconclusive outcome, so that one rule applies to any result?
4. The scoop. Did a real search run, and does any close work already answer the question?
5. Runnable cold. Could a fresh agent with only this card run step 1 and judge it without asking anything?
6. The plan change. Was any threshold changed after its check failed? If so, are the failing values recorded, and did the lab's skeptic look before the plan was rebound?

A review loop ends on a clean read, not on a fix. If a read finds gaps and they are fixed, a fresh reader reads the card again before the loop closes.
