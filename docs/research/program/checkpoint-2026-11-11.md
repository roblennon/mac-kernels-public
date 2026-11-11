## Checkpoint: MPS decode-speed thread · 2026-11-11

Rob redirected the programme on 2026-11-10 toward speed, and this thread explored it.
- **Plain decoding.** Compile plus a multi-row Metal kernel gives about 2x decode on an M1 Max.
- **Speculative decoding.** Prompt-lookup speculation gives about 3x on whole-file edits with exact output. On the Kaggle T4 it now gives +16% on real agent traffic.
- **Attention.** A new small-query attention kernel removes PyTorch's attention cliff above 8 query rows: 2.7x faster verification at 4k context, 5.1x at 16k.
- **Finished later the same day.** Kernel optimization (E2) and the upstream drafts are done and folded in.

Rob needs to decide whether the research docs go public (local main is far ahead of origin, nothing pushed), and to review the PyTorch issue draft before anything is posted.

**Done**
- Cross-family review of the decode stack, with corrections; the rerun under fixed rules confirmed the gains at smaller sizes (findings log, 2026-11-10).
- The T4 A/B: n-gram drafting ahead of MTP shipped as serve v4. A 27B model was measured on both T4s and now serves by default (kaggle-qwen-serve EVIDENCE.md and CHECKPOINTS.md; thread section "Carrying this to the Kaggle T4 server").
- The attention cliff was confirmed in the whole model, and its cause was found in PyTorch's dispatch.
- Experiment E: the small-query kernel and its transformers wiring, reviewed by agents and Sol, with fixes applied (`bench/smallq_attention.*`, `bench/smallq_hf.py`; findings log, 2026-11-11).
- The post-prefill memory-clock penalty was found and the harness fixed (`bench_verify_cost.py --rewind`); the gate entry was corrected.
- The environment for these measurements was moved out of /tmp into `.venv-decode`, pinned in `bench/requirements-decode.txt`.
- E2: a row-vector kernel path. At T = 9 to 16, whole-model verify is 2.7x to 3.2x faster at 4k and 5.4x to 6.9x at 16k; T = 4 to 8 gains 1.05x to 1.5x (findin**Open**
- The kernel's small-T gap. Through the kernel, T = 1 is 4 to 9% slower in the whole model, probably a fixed custom-op cost of about 0.7 to 1 ms per step, so the threshold stays at 4. Two mutants survive the suite (vector-path slack, the pos upper clamp). All three are recorded in TASKS.md.
- Set aside: experiment D (the 17 to 32 row band on 7B shapes), because it needs a 7B download that is Rob's call. Also the behavioural cards, at Rob's direction.
- Deferred: scale folding and the verified-rewrite design's next stage.
- Note: during this thread another session, at Rob's request, rewrote the unpushed history to move a card to the private repository. This thread's commits survive under new hashes, with content unchanged apart from that card.

atic-cache-compile-graph-break.md`) and their repro scripts (`bench/repro_*.py`): drafted by the same workflow and under review there. After that they wait for Rob.
- Set aside: experiment D (the 17 to 32 row band on 7B shapes), because it needs a 7B download that is Rob's call. Also the behavioural cards, at Rob's direction.
- Deferred: scale folding and the verified-rewrite design's next stage. These wait behind E2.

**Next**
1. Rob reviews the PyTorch issue draft and decides on publication. The draft should link its repro and prototype from a public place, so the two decisions go together.
2. Long-draft speculation (k up to 15) at long context on MPS, measured on replayed real traffic, not the edit benchmark. Verification is now cheap, so this is where the kernel becomes user-visible speed.
3. Optionally move `.venv-decode` to transformers 5.20 or later, which removes the compiled graph break from stock numbers. It needs a re-baseline of the stock arms, so do it before the next measured experiment, not in the middle of one.

**Orphans**
- The rest of the scratchpad (sweep JSONs, probes, Sol prompts, 2.6 GB of misplaced GPU traces) is session-local. Everything that backs a finding has been copied to `docs/review/`. The rest can go with the session.

**Learned:** Three measurement lessons went to LESSONS.md:
- one process is a hint;
- time the steady state, and know what the step before did;
- review the harness before it spends the GPU.

Rob's working rules from this thread:
- he wants this session to lead execution decisions and run parallel tracks;
- GPU work needs no stopping during the day, but nothing may max out the machine without asking him;
- the lab owns the GPU from 22:00 to 05:00.

Two transferable technical facts:
- transformers 5.18's `_ignore_causal_mask_sdpa` breaks compiled StaticCache forwards with T > 1;
- llama.cpp b12228 always tries n-gram drafters before MTP, and its n-gram defaults silently disable short drafts.

**Safe to close:** yes. Everything is committed locally, and what is open has a home in TASKS.md and the thread.
