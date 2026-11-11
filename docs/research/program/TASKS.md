# Research programme tracker

The running list for the programme framed in `FRAME.md`. One line per item. Decisions and their reasons go at the bottom so the next reader can see why, not only what. Lessons for card writers are in `LESSONS.md`.

## Now: decode and small-batch speed on MPS (updated 2026-11-11)

Rob redirected the programme toward training, fine-tuning, inference, attention, context and gradients, and away from behaviour, and asked for the lab's decode-speed result to be explored hard. Where it stands is in `threads/mps-decode.md`, "Where it stands"; the numbers are in the findings log, entries from 2026-11-10 and 2026-11-11. In short:
- **Compile plus the multi-row kernel:** about 2x decode at batches 8 to 16.
- **Prompt-lookup speculation:** 2.9x to 3.3x on whole-file edits, with exact output. On the Kaggle T4 it shipped and gives +16% on real agent traffic.
- **The small-query attention kernel:** removes PyTorch's attention cliff above 8 query rows; whole-model verify is 2.7x faster at 4k context and 5.1x at 16k.
- **In progress:** kernel optimization (E2) and two upstream issue drafts. Behavioural cards are set aside in `QUEUE.md`.

## Waiting on Rob

- Whether the phase after the goal (handing cards to the lab, reading results) gets its own budget and stop, or stays attended only.
- semantic-latent-memorization: accept the DINOv3 licence.
- Thread experiment D needs a 7B model download (about 14 GB in fp16).
- train-scaffolds step 2: days of unattended Mac time.
- Whether these research files go public. The repository is public once pushed, and the docs name his private projects. Local main is 53+ commits ahead of origin.
- Before that push, the history must be rewritten. On 2026-11-11 one set-aside card and its mentions moved to the private mac-kernels repo (Rob's direction), but their text is still in the unpushed commits. The private repo's TASKS.md has the procedure. Don't push until it's done.
- The two upstream issue drafts (`docs/upstream/pytorch-mps-sdpa-small-query-cliff.md`, `docs/upstream/transformers-static-cache-compile-graph-break.md`): review, then post or not. Posting is public.
- Colab login; the gaming PC's GPU model. Kaggle works from the Studio through its API token, but the 30-hour weekly GPU quota is shared with `kaggle-qwen-serve` and model-research-lab, so check kaggle.com/settings before planning a run.
- Rotate the Modal token pasted in chat on 2026-11-09; delete the Modal Volume `mac-kernels-weight-error` (about 27 GB) when done with it.

## Open, ours

- [x] Decode thread experiments A to C (`threads/mps-decode.md`): fused norm and rope, compile with a static cache, the multi-row kernel, merged projections, prompt-lookup speculation; cross-family review and corrections (8a1ee6e)
- [x] Decode rerun under the protocol fixed in `threads/mps-decode.md` (2026-11-10): all checks pass; 4 of 5 predictions met, verify cost missed (1.52x, not 1.3x); findings log entry "Decode rerun under fixed rules"
- [x] Kaggle T4 server: n-gram drafting ahead of MTP shipped as serve v4 (2026-11-10): 2.8x on the edit benchmark, +16% decode on real OpenCode traffic (kaggle-qwen-serve EVIDENCE.md)
- [x] Kaggle T4 server: the Orca 27B fits both T4s with `-sm tensor` at 262144 context, 49-58 tok/s on edits and 22-25 on free writing (2026-11-10, kaggle-qwen-serve EVIDENCE.md, tune-27b v1); `serve.py` defaults to it, with an idle shutdown
- [ ] Kaggle T4 server, after the 2026-11-14 quota refresh: run `tune-27b-replay` (200K-class depth, tunnel cut, prefill batch, real traffic), then serve the 27B; owned by kaggle-qwen-serve's CHECKPOINTS.md
- [x] Experiment E (2026-11-11): small-query attention kernel (`bench/smallq_attention.*`, `bench/smallq_hf.py`), reviewed by workflow agents and Sol, fixes applied; findings log "A small-query attention kernel removes the 9-row cliff"
- [ ] E2: optimize the kernel against the targets in the thread (T = 1 to 2 at stock parity, 4 to 8 at 1.3x or better); running as a workflow on 2026-11-11
- [ ] Upstream drafts for PyTorch (the cliff) and transformers (the compiled graph break), with repro scripts; drafting on 2026-11-11, then Rob's review
- [ ] Scale folding (experiment C's other half) and the verified-rewrite design's next stage
- [x] Subject gates reviewed by Sol for needless slowness and weakness; 141 changes applied across 23 cards, 20 points rejected with reasons
- [ ] Parked cards owe searches: self-training-37, nested-bitplanes (paired-metric-floor's park was confirmed by a second reader).
- [ ] Cards named but not written: card B for copies-as-evidence's loop-share question; a divergence-law card; an attention-hint card split from train-scaffolds.
- [x] Candidate new card: a one-row `nn.Linear` slow path on MPS (from the lab's Q003). Superseded by the decode thread: torch 2.15's GEMV route, compile and the multi-row kernel cover it.
- [ ] Vendor branches not pushed to the public forks; replies owed on comfy-kitchen 249 and pytorch 200782. Local main is ahead of origin and not pushed.

## Lab state, as this session knows it (2026-11-11)

- **Earlier cases.** Case 001 ran (three training speed-ups confirmed). Case 002 (copies-as-evidence) decided nothing; it is set aside with the behavioural cards. Q002 (train-scaffolds step 1a) is parked inconclusive, median 0.948.
- **Verification line.** Q009 and Q009b measured verify cost on Qwen3.5-4B and found the attention cliff above 8 rows. Q009b's llama.cpp "no step" claim was withdrawn after review. The lab's next verification question adds `smallq_hf` as an arm, and it will check its earlier timings for the post-prefill penalty.
- **GPU sharing.** The lab runs unattended research on the Mac GPU every night, 22:00 to 05:00 (agreed with Rob on 2026-11-11). Every GPU run from this repo goes through `bench/gpu_runner.py` with a label, and nothing that saturates the machine starts without asking Rob.

## Done

- [x] Three idea rounds, 314 ideas; nominations, merge, sceptical read; idea ledger, directions document, musings
- [x] 33 design cards, each searched, attacked, revised and search-gated; queue of 30; Sol's cross-family review; cold reads of the top five
- [x] Checkpoint and commit; goal met at 28bb2a7; the lab's milestone arrived
- [x] Lab case 002 ran copies-as-evidence step 1a (2026-11-10): void, no greedy answer right (one template inaccurate, the other off-format), count probe failed
- [x] copies-as-evidence redesigned around a capability step; subject gates added to 27 of the 29 other queued cards
- [x] Reorientation sweep: stale docs, learnings, process, new avenues (`raw/reorient_*.md`)

## Decisions

- 2026-11-10: Rob: focus on training, fine-tuning, inference, attention, context and gradients, not behaviour. copies-as-evidence is set aside as model-dependent and not about speed or performance. The decode-speed thread leads.

- 2026-11-10: measure the gates first (Rob). A small first batch, not thirty harnesses; other cards read their gate from their own first run.

- 2026-11-10: every card has a subject gate before its hypothesis test. A planted-effect check tests the instrument, not the subject.
- 2026-11-10: copies-as-evidence step 2 is a fresh test, never pooled with step 1a.
- 2026-11-10: a reading of a lab result has one source, the card's result section; messages point to it.
- 2026-11-09 night: the queue follows most of Sol's cross-family review. Cards whose first slice decides something go above cards whose first slice only shows feasibility.
- 2026-11-09 night: web search for subagents is capped at about 200 calls per turn of the lead session, shared. Search-heavy workflows run three agents at a time.
- 2026-11-09 evening: Rob changed this pass to ideas and preparation only; idea rounds are no longer capped at two.
- 2026-11-09: no grant application yet. The Mozilla Fellowship letter of intent (2026-12-17) is the planning date.
- 2026-11-09: Inference.net (up to $10k compute, inference-performance focus) and Google TPU Research Cloud applications were drafted and kept in the private mac-kernels repo (`docs/grants/`); neither is submitted. Free-compute options are in `docs/research/2026-11-09-free-compute-options.md`.
