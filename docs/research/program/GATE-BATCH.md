# Gate batch 1, 2026-11-10

*Revised later on 2026-11-10: Rob redirected the programme toward training, inference and the technical machinery, away from behaviour. Item 1 (the role-swap check) belongs to a behavioural card and is withdrawn. The decode-speed thread ([threads/mps-decode.md](threads/mps-decode.md)) now comes first; items 2 to 4 follow it.*

Rob chose to measure the subject gates before running more full experiments. A subject gate asks, before an experiment's real test, whether its model or data can show the effect at all. This brief is what the lab runs first. It was shaped by a review from a different model family ([Sol's gate review](reviews/2026-11-10-gate-review-sol.md)), whose main advice was not to build thirty harnesses just to rank thirty cards. So the batch is small. For the other cards, the gate is read first from the run the card already plans, when that card comes up.

## The batch, in order

**1. The role-swap check (copies-as-evidence).** Case 002's R and RS arms carry identical histories and differ only in the system prompt's last sentence. RS ends with a complete sentence; every other arm ends with a 15-token fragment, "...and every value it needs is". In the terse template (T2), that one sentence moved the answer format from 13 of 60 scenarios to 53 of 60. The check runs 20 fresh T2 scenarios in the old wording, with histories D0 and R, under four system endings of equal token length: the fragment, a complete neutral sentence, the RS sentence, and a complete "written by you" sentence. Each gets one scored branch and one 6-token greedy decode, forward only. The prediction, the decision it informs and the cost are written in [the card](designs/copies-as-evidence.md), under "The T2 wording decision waits on the role-swap check". About 5 GPU-minutes; Qwen3-1.7B and the case code are already on the machine. It runs first because it may show that case 002's terse-template failure came from our control, not from the model, and it decides whether the card's planned wording change is needed at all.

**2. rare-feature-threshold, the pilot and its first measured run (CPU).** Does a small bigram model trained with AdamW keep rare facts down to a frequency where the predicted cutoff can be seen, with room on either side? Read first from the pilot of 2,000 steps, then step 0's scalar settings, then the G95 seed-0 run. Synthetic data, nothing to download. About 1 CPU-minute for the pilot and 2 to 5 for the G95 run; most of the cost is writing the harness, a few hours. [Card](designs/rare-feature-threshold.md), "Subject gate".

**3. rare-exploit-onset, the CPU bandit stage.** Do the visible task's gradients reach the adapter, and do rare-exploit onsets land inside the step budget often enough to fit? Read first from the rho measurement the design already plans, then eight short pilot runs; the split-half check reuses step 1a's own. Synthetic data, nothing to download. About 5 to 10 CPU-minutes once the bandit code exists. [Card](designs/rare-exploit-onset.md), "Subject gate (1b, before any 1b slope run)".

**4. surprise-feedback, the switching task only (MPS).** Does the control arm learn the switching chain, and does it leave at least 0.05 nats of excess loss over the exact Bayes answer after a switch, on every seed? Read first from arm A's three seeds. Synthetic data. About 8 GPU-minutes plus a 30-second timing probe. The regression task waits until this result is in, and its room bar must be written before it runs. [Card](designs/surprise-feedback.md), "Subject gate, read from the baseline run, first".

## What the batch needs

- From Rob, through the lab: device time for items 1 and 4, about 15 GPU-minutes in all, and CPU time for items 2 and 3. No downloads.
- From the lab: a small addition to case 002's code for item 1 (the system ending as a parameter, and two new endings). For items 2 to 4, new harnesses, which are most of the cost.

## Recording the outcome

Each gate ends in one of three states, kept apart in the queue: **no code yet**, **subject absent** (the gate ran and failed), and **subject present but too little room or resolution**. A failed gate is a result. Its values are kept, and the subject is not adjusted until it passes. The card's result section is the one place the reading lives.

## What comes after

Once the batch is in, the queue is reordered by which cards have a subject. The next candidates, each needing a small download Rob would approve:

- renyi-search-cutoff, with SmolLM2-135M alone first;
- whisper-silence, its stock baseline on the LibriSpeech test-clean subset already on the machine;
- finetune-survival-requant, base-fact competence on Qwen2.5-0.5B before any quantizer is built.

Everything heavier waits for its turn in the queue, and reads its gate from its own first run.
