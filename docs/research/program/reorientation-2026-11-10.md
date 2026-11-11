# Reorientation, 2026-11-10

Rob paused this session and the lab after the first card came back void, and asked for an assessment: what was done, how it went, where it is headed, and what needs cleaning up, capturing or refining. This is that report. The evidence behind it is in four sweep reports kept in the git-ignored `raw/` folder (`reorient_stale.md`, `reorient_learnings.md`, `reorient_process.md`, `reorient_avenues.md`).

## What needs Rob

**1. Where to go next.** My recommendation: before running any more full experiments, run the cheap subject gates as one batch in the lab, and the five-minute role-swap check below. Every queued card now has a gate that asks whether its model or data can show the effect at all, but the gates were designed, not measured. Measuring the cheap ones first (several are CPU-only or a few GPU-minutes) tells us which cards have a subject before we spend an hour on any of them. The queue would then be reordered by which subjects exist. The alternative is to resume the queue as it stands, starting with the redesigned first card, which needs two new allowances from Rob (1,200 and 2,400 guarded seconds).

**2. A frame for this phase.** The frame's goal was met on 2026-11-09. Handing cards to the lab and reading results has no budget or stop of its own. Rob can set one, or keep it attended only, as now.

**3. Whether the research files go public.** The repository is public once pushed. The new documents name Rob's private projects and machines. Fifteen commits sit unpushed. The raw agent output that names them most is git-ignored either way.

Smaller items waiting on Rob are listed in [the tracker](TASKS.md#waiting-on-rob): the Qwen3-4B download (only if the 1.7B ladder fails), the DINOv3 licence, recorded stimuli, listening sessions, days of Mac time for train-scaffolds step 2, a Kaggle or Colab login, the gaming PC's GPU, and rotating the Modal token.

## What was done

Over 2026-11-09 and the morning of 2026-11-10:

- Three idea rounds through forty-four lenses produced 314 ideas, kept in [the ledger](../2026-11-09-idea-ledger.md), with [a directions document](../2026-11-09-research-directions.md) and [a musings notebook](MUSINGS.md).
- Thirty-three design cards, each with a prior-art search, a hostile review, a revision and a search gate. A queue of thirty, reordered after a review by GPT-6.1 Sol. The top five were cold-read by fresh agents until the first read clean.
- The lab took two cards. It framed train-scaffolds step 1a as its Q002 on case 001 (drafted, not opened). It built the first card, copies-as-evidence, as case 002, Rob approved it, and it ran on 2026-11-10.
- Case 002 decided nothing. Every measuring check passed, including a planted effect recovered to within 0.0000017 nats. But with thinking off, Qwen3-1.7B got no greedy answer right: one template was inaccurate, the other broke format, and the probe meant to confirm the manipulation failed. So the hypothesis was never tested.
- The card was redesigned around a capability step, and a sweep found that 27 of the 29 other cards had no check on the subject either. Each now has one.

## How it went

The machinery worked. The two sessions coordinated well: the lab bound every plan to a card commit, rebound it when the card changed, asked Rob directly for approvals, and caught five real problems in the card before spending device time. Measurement held up under a real run.

The science did not get tested, and the reason is ours. Five design passes, four cold reads, a cross-family review and four reviews in the lab all checked the instrument the card described. None asked whether the model could do the task. A reviewer checks what the artifact puts in front of it.

Three other things went less well than they should have:

- **A wrong reading spread.** I read the case 002 result as "at chance in both templates" by averaging digits that behave differently. That reading went to the lab, into its outcome record, and into the brief for thirty sweep agents, who copied it into twenty-two card records. The redesign's attack caught it. All copies in this repository are now corrected; the lab has the corrections.
- **Search fan-out drained the budget twice.** About thirty agents searching at once used the shared allowance of about 200 calls per turn within minutes, so most search gates had to be rerun three agents at a time. A Personal OS inbox entry from 2026-09-04 already warned about this.
- **Cards grew instead of being rewritten.** Every review round appended. The first card reached about 15,000 words. Long cards are harder to run cold, which is what the cards are for.

One near-miss was handled well: an identity check's threshold sat below the card's own rounding bound, and a later check tripped on rounding. Both were fixed before any main run, but the second fix was accepted within a minute, without the lab's skeptic. That should have waited for review.

## Where it is headed

The programme's direction does not change: find results Rob could cite, test them on the Mac, carry them to larger models where the mechanism allows. What changes is the order of work.

1. Measure before designing further. Run the cheapest subject gates as a batch, and reorder the queue by which subjects exist.
2. Check the role-swap anomaly from case 002 (below) before trusting the redesigned card's fix for its second template.
3. Then resume with whichever top cards have a subject, one at a time, each through the lab.

## New avenues that appeared

- **The largest effect in case 002 went unseen.** In the second template, the arm that labels the history as another agent's stayed on format in 53 of 60 scenarios; every other arm managed 6 to 16. I checked it against the raw records. One hypothesis: the other arms carry a filler sentence cut mid-clause, and that broke the format. If so, the redesign's wording fix for that template is aimed at the wrong thing. A forward-only check costs about five GPU-minutes.
- **A slow path in one-row linear layers on MPS.** The lab's Q003 found that replacing a one-row `nn.Linear` with `addmm` on a pre-transposed weight cut a decode step by about two thirds, an MPS-specific effect. That fits this repository's own line of MPS slow-path findings, and could become an upstream fix. Candidate card, after a search.
- **Noise from prefix-cache reuse, the CPU matching the GPU on tiny decodes, and wrong answers that look like wrong lookups.** Each is now a line in [the musings](MUSINGS.md).
- **How agent-run research fails.** Every review layer missed the subject; every top card failed its first cold read; the cross-family reviewer caught what same-family passes missed; and the biggest effect went unremarked because reviewers read only planned contrasts. A table of defect against the layer that caught it is about an hour of reading. Two 2026 papers study this qualitatively. Worth it only if Rob wants a write-up outside the Mac work.

## What was cleaned up

- The frame marks its pre-amendment Done, Budget and Plan sections as superseded, records where the goal stands, updates the lab's status and ownership, and adds a third check on the subject, plus a rule that tolerances sit above the numeric format's own error.
- The tracker is rewritten around what waits on Rob, what is ours, and what the lab holds.
- The directions document, checkpoint, docs index, an upstream issue note and an older survey note are corrected.
- Agent memory has a new research-programme file; the grant and Modal files are corrected.
- "At chance" is corrected in every card record.

## What was captured, and where

- In this repository: [LESSONS.md](LESSONS.md) for card writers, and [a card template](designs/TEMPLATE.md) with a named subject gate, a fixed order inside "Step 1, exactly", a soft length cap, and the questions every attack and cold read must ask. The queue builder now prints the items waiting on Rob at the top of the queue and warns about cards without a runnable step or a subject gate.
- In Personal OS's inbox: three entries. The subject-question lesson for experiment reviews, with a candidate edit to dev-plan's proof mode. Cross-family review evidence from research designs, for the model-routing reference. And a recurrence of the search-budget drain, asking that the 2026-09-04 guardrail be promoted.
- To the lab, by message: the corrected reading for its three files, the role-swap finding, a shared name for the gate, a seconds discrepancy to reconcile, and the rule that a threshold changed after a failed check gets a skeptic first.

## Skills and processes

- **Build here, done:** the card template, the lessons file, and the queue-builder checks. The builder's first warnings show the work left: only 5 of 30 queued cards have a "Step 1, exactly" section a runner can read cold, and 10 name their subject gate in ways a script cannot find. Renaming the gates is a small cleanup; writing the runnable step is a cold-read loop per card, and is worth doing only for cards that reach the top after the gate batch.
- **Propose to Personal OS, not edit:** an experiment proof shape for dev-plan, with a pointer from dev-delivery-ops, whose work types cover only bug, polish and feature. Its three parts: the instrument is sound, the subject has the property, and the decision rule is fixed in advance. Change the model routing or add review passes? No: the miss came from a missing question, not too few reviewers.
- **A standing rule I broke:** the inbox holds Rob's rule that any gate able to block work gets a different-family review for needless slowness before it is adopted. The 27 new subject gates have not had that review. It should happen before the lab adopts them.
- **Keep as is:** commit binding between the sessions, the lab asking Rob directly, the lab's fit checks against real code, paced search, and the cross-family review at queue level.
