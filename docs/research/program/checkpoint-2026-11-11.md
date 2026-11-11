## Checkpoint: MPS decode-speed thread · 2026-11-11

Rob redirected the programme on 2026-11-10 toward speed, and this thread explored it.
- **Plain decoding.** Compile plus a multi-row Metal kernel gives about 2x decode on an M1 Max.
- **Speculative decoding.** Prompt-lookup speculation gives about 3x on whole-file edits with exact output. On the Kaggle T4 it now gives +16% on real agent traffic.
- **Attention.** A new small-query attention kernel removes PyTorch's attention cliff above 8 query rows: 2.7x faster verification at 4k context, 5.1x at 16k.
- **Still running.** Kernel optimization and two upstream issue drafts are running as a workflow when this record was written.

Rob needs to decide whether the research docs go public (local main is 53+ commits ahead, nothing pushed), and to review the two issue drafts before anything is posted.

**Done**
- Cross-family review of the decode stack, with corrections; the rerun under fixed rules confirmed the gains at smaller sizes (findings log, 2026-11-10).
- The T4 A/B: n-gram drafting ahead of MTP shipped as serve v4. A 27B model was measured on both T4s and now serves by default (kaggle-qwen-serve EVIDENCE.md and CHECKPOINTS.md; thread section "Carrying this to the Kaggle T4 server").
- The attention cliff was confirmed in the whole model, and its cause was found in PyTorch's dispatch.
- Experiment E: the small-query kernel and its transformers wiring, reviewed by agents and Sol, with fixes applied (`bench/smallq_attention.*`, `bench/smallq_hf.py`; findings log, 2026-11-11).
- The post-prefill memory-clock penalty was found and the harness fixed (`bench_verify_cost.py --rewind`); the gate entry was corrected.
- The environment for these measurements was moved out of /tmp into `.venv-decode`, pinned in `bench/requirements-decode.txt`.
- Coordination rules agreed with the lab and Rob: [[mac-gpu-sharing]] and [[kaggle-coordination]] in the harness memory; lab state in TASKS.md.

**Open**
- E2, kernel optimization against the targets written in the thread: running in workflow `smallq-upstream-and-optimize` (run wf_3f1faf1e-452). Its uncommitted edits to `bench/smallq_*`, `bench/bench_verify_cost.py`, `bench/check_smallq_model.py` and `tests/test_smallq_*` belong to it.
- The upstream issue drafts (`docs/upstream/pytorch-mps-sdpa-small-query-cliff.md`, `docs/upstream/transformers-static-cache-compile-graph-break.md`) and their repro scripts (`bench/repro_*.py`): drafted by the same workflow and under review there. After that they wait for Rob.
- Set aside: experiment D (the 17 to 32 row band on 7B shapes), because it needs a 7B download that is Rob's call. Also the behavioural cards, at Rob's direction.
- Deferred: scale folding and the verified-rewrite design's next stage. These wait behind E2.

**Next**
1. Fold in the workflow's results. Accept or reject E2 against its targets, commit what passed, and correct the findings. This comes first because uncommitted kernel edits are the only work at risk.
2. Rob reviews the two issue drafts and decides whether to post. They are the outward-facing result of E, and the transformers bug costs every compiled verify step.
3. Long-draft speculation (k up to 15) at long context on MPS, measured on real traffic, not the edit benchmark. Verification is now cheap, so this is where the kernel turns into user-visible speed.
4. Rob decides on publication, because the grant material and any upstream post will link to this repository.

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

**Safe to close:** not yet. The workflow's results (E2 and the issue drafts) must be folded in and committed first; after that, yes.
