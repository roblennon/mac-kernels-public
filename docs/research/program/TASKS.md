# Research programme tracker

The running list for the programme framed in `FRAME.md`. One line per item. Decisions and their reasons go at the bottom so the next reader can see why, not only what. Lessons for card writers are in `LESSONS.md`.

## Now: gate batch 1 (2026-11-10)

Rob read the reorientation report and chose "Measure the gates first". The subject gates had a cross-family review (`reviews/2026-11-10-gate-review-sol.md`) and were revised. The first batch is in `GATE-BATCH.md`: the role-swap check, then three cheap training gates. It goes to the lab, which runs it once Rob sets device time.

## Waiting on Rob

- Whether the phase after the goal (handing cards to the lab, reading results) gets its own budget and stop, or stays attended only.
- copies-as-evidence: step 0's 1,200 s and step 1a's 2,400 s allowances; whether to download Qwen3-4B (about 8 GB) for the second ladder; his own yes before any Qwen3.5-4B ladder.

- semantic-latent-memorization: accept the DINOv3 licence.
- duplex-turn-temperature: record the stimuli its open half needs.
- Speech cards that end in a listening test need his ears.
- train-scaffolds step 2: days of unattended Mac time.
- Whether these research files go public. The repository is public once pushed, and the docs name his private projects.
- Kaggle or Colab login; the gaming PC's GPU model.
- Rotate the Modal token pasted in chat on 2026-11-09; delete the Modal Volume `mac-kernels-weight-error` (about 27 GB) when done with it.

## Open, ours

- [ ] copies-as-evidence role-swap check: R and RS differ only in the system prompt's last sentence (a 15-token fragment against a complete sentence), and T2 format went from 13 to 53 of 60. First item of gate batch 1, prediction written in the card.
- [ ] copies-as-evidence: the redesign's last cold read found blocking gaps that were fixed but not re-read. Re-read before it goes to the lab.
- [x] Subject gates reviewed by Sol for needless slowness and weakness; 141 changes applied across 23 cards, 20 points rejected with reasons
- [ ] Parked cards owe searches: self-training-37, nested-bitplanes (paired-metric-floor's park was confirmed by a second reader).
- [ ] Cards named but not written: card B for copies-as-evidence's loop-share question; a divergence-law card; an attention-hint card split from train-scaffolds.
- [ ] Candidate new card: a one-row `nn.Linear` slow path on MPS (from the lab's Q003), with a search gate first.
- [ ] Vendor branches not pushed to the public forks; replies owed on comfy-kitchen 249 and pytorch 200782. Local main is ahead of origin and not pushed.

## Lab state, as this session knows it

- Case 001 (character model, seconds to loss 1.70): Q001, Q003 and a composition ran; three training speed-ups confirmed. Q002 (train-scaffolds step 1a) is drafted against card 9966d2b, not opened; step 1a is unchanged since.
- Case 002 (copies-as-evidence step 1a): ran 2026-11-10, decided nothing; the redesign is held.

## Done

- [x] Three idea rounds, 314 ideas; nominations, merge, sceptical read; idea ledger, directions document, musings
- [x] 33 design cards, each searched, attacked, revised and search-gated; queue of 30; Sol's cross-family review; cold reads of the top five
- [x] Checkpoint and commit; goal met at c51d8a1; the lab's milestone arrived
- [x] Lab case 002 ran copies-as-evidence step 1a (2026-11-10): void, no greedy answer right (one template inaccurate, the other off-format), count probe failed
- [x] copies-as-evidence redesigned around a capability step; subject gates added to 27 of the 29 other queued cards
- [x] Reorientation sweep: stale docs, learnings, process, new avenues (`raw/reorient_*.md`)

## Decisions

- 2026-11-10: measure the gates first (Rob). A small first batch, not thirty harnesses; other cards read their gate from their own first run.

- 2026-11-10: every card has a subject gate before its hypothesis test. A planted-effect check tests the instrument, not the subject.
- 2026-11-10: copies-as-evidence step 2 is a fresh test, never pooled with step 1a.
- 2026-11-10: a reading of a lab result has one source, the card's result section; messages point to it.
- 2026-11-09 night: the queue follows most of Sol's cross-family review. Cards whose first slice decides something go above cards whose first slice only shows feasibility.
- 2026-11-09 night: web search for subagents is capped at about 200 calls per turn of the lead session, shared. Search-heavy workflows run three agents at a time.
- 2026-11-09 evening: Rob changed this pass to ideas and preparation only; idea rounds are no longer capped at two.
- 2026-11-09: no grant application yet. The Mozilla Fellowship letter of intent (2026-12-17) is the planning date.
