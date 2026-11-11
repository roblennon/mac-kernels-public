You are picking up a research thread on making local language-model inference faster on Rob's Mac Studio (M1 Max, 32 GB). The previous lead, a Claude session, ran out of weekly usage and left everything in this repository, `~/gh_projects/mac-kernels-public`. You are now the lead. You may use up to five subagents at once. Rob wants you to make the execution calls yourself (order, parallelism, what to delegate) and to come back to him only for money, downloads, publication, a change of direction, or anything that would max out the machine.

## What the previous thread found

Read these before doing anything; they are the source of truth, and this note only orients you:

- `docs/research/program/threads/mps-decode.md`. Its "Where it stands" section is the status. "Next directions (2026-11-11)" is the plan you are inheriting.
- `docs/findings.md`, the entries dated 2026-11-10 and 2026-11-11. They hold every number, with its ranges and the corrections made after review.
- `docs/research/program/LESSONS.md`, especially the last four lessons. They are how measurements here went wrong and how that was caught.
- `docs/research/program/checkpoint-2026-11-11.md` and `TASKS.md`.

In short: everything built so far is PyTorch on Apple's GPU (MPS).

- **Plain decoding.** `torch.compile` with a static cache, plus a Metal kernel for linear layers at 2 to 16 rows, gives about 2x decode at small batch sizes.
- **Speculative decoding by prompt lookup.** Drafting tokens by copying from the context gives about 3x on whole-file edits, with exact output.
- **Attention.** PyTorch's attention switches to a far slower path above 8 query rows. A small-query attention kernel (`bench/smallq_attention.*`, wired in by `bench/smallq_hf.py`) removes that cliff: verifying 9 to 16 tokens is 2.7x to 3.2x faster at 4k context and up to 6.9x at 16k.
- **On a Kaggle T4.** The same drafting idea shipped in llama.cpp. It gave 2.8x on an edit benchmark but only 16% on real agent traffic, because agent edits are short diffs. That gap is the most important lesson of the thread.

## The direction Rob chose

Rob does not run his local models in PyTorch. He runs them through `~/gh_projects/qwen-locals-bench`: Qwen3.5-4B on stock MLX-LM (mlx 0.32.4 and mlx-lm 0.32.2, through the Locals service with four slots), and the Orca 27B GGUF through llama.cpp. Read its README, ACTIVATION.md, OPTIMIZATIONS.md and ORCA128.md. Rob asked for speed he can feel in that local agent setup first, ahead of the PyTorch research record. Pursue the threads listed in "Next directions", in roughly this order, and reorder only with a stated reason.

**First, MLX's small-row quantized matmul.** The sibling lab measured MLX's 4-bit matmul cost growing almost row by row up to about 9 rows and only dropping once a tiled path takes over from 12 rows. That is the same gap the multi-row kernel closed in PyTorch. The 4B server's four slots put at least 4 rows through every step. The gate comes before any kernel work: check whether newer MLX releases or MLX main changed this path, measure the Locals server's decode step at 1, 2 and 4 busy slots, and measure the isolated matmul at the 4B's real shapes with distinct weights per call. A cached weight reused in a loop reads faster than memory allows and has fooled this project before. If the gap is real and costs a meaningful share of a server step, a fix is worth building, and is also an upstream contribution to MLX.

**Second, prompt reading.** On this Mac, the Orca 27B took 1,552 seconds to read a 117,717-token prompt cold. Rob works at 200K-class contexts, so this may be the biggest lever of all. `qwen-locals-bench/OPTIMIZATIONS.md` already designs a llama.cpp prefill batch-size sweep that has never been run. Run it, and profile real OpenCode sessions through Locals to learn how their time splits between reading, decoding, queueing and cache hits. A decision about where to optimize should rest on that split, not on a benchmark.

**Third, tree speculation.** Verifying several short candidate continuations in one forward pass would make the now-cheap 9 to 16 rows pay off even when edits are short. `bench/rerun_spec.py` is the harness to extend. It is more research-shaped, so it comes after the two above unless their gates come back empty.

## How this project works, and why

Predictions and pass rules are written into the thread before a measurement runs, and a miss is reported as a miss. This is what keeps a fast-moving search honest: twice in this thread, a number that looked like a finding (1.19x, then a 9% gap) dissolved when measured properly.

**Measuring.**
- **Use several processes.** A speed claim needs several processes with the arms interleaved, every sample kept, and an A/A pair (the same pipeline twice) to show the noise.
- **Time the steady state.** On this Mac, the first step after a long compute phase pays about 30 ms once, because memory clocks drop. So time steady steps, and report the first step after a prefill separately.
- **Review the harness first.** Have a fresh reviewer try to break a harness before it spends GPU time. If you can reach a model from a different family than yours (Claude, when Rob's usage returns), use it for the review, because a reviewer from the same family tends to share the author's blind spots. Otherwise use a fresh-context adversarial subagent, and tell Rob which you used.
- **One source per reading.** A reading of a result lives in one place, its findings entry, and messages point there instead of restating numbers.

**The machine is shared.**
- **The GPU lock.** Every command that touches the Apple GPU goes through `bench/gpu_runner.py --monitor` with `MK_GPU_LABEL=mac-kernels-public:<run-name>`. That keeps runs from interleaving with each other and enforces memory limits. Locals' `./activate serve` already uses the same supervisor and lock.
- **The lab's hours.** The sibling project `~/gh_projects/model-research-lab` runs unattended GPU research every night from 22:00 to 05:00 local time. Work in that window only through the lock, so it queues behind the lab's runs. Never edit that repository.
- **Never max out the machine.** Rob uses the machine at the same time. Several heavy jobs at once, all CPU cores busy, or enough memory to swap all need his say-so first.
- **One GPU job at a time.** With subagents, assume that: they can research, write and review in parallel, but GPU runs serialize through the lock.

**Environment.** The PyTorch measurements use `.venv-decode` (torch 2.15.0 and transformers 5.18.0; `bench/requirements-decode.txt` rebuilds it). Hugging Face models load offline with `HF_HOME=/Volumes/RTL-Data/AI-Models/huggingface HF_HUB_OFFLINE=1`. Downloading a model, a new runtime or a newer MLX into a separate test environment is a download, so ask Rob before doing it.

**Git and publication.**
- **Commit locally.** Use conventional commits, and add the agent-provenance trailer from `~/personal-os/skills/dev-delivery-ops/references/agent-provenance.md` with your own model and family.
- **Push and post nothing.** Whether this repository goes public is still Rob's open decision, and so is the drafted PyTorch issue in `docs/upstream/`.
- **Keep local paths out of tracked files.**

## What done looks like for this run

The first milestone is the MLX gate answered, plus a first cut of the agent-session time split. Each is written as a dated findings entry with its numbers and an updated "Where it stands" in the thread, and committed. Stop there and report to Rob in plain words:
- what you found;
- whether an MLX kernel is worth building, and why;
- what the time split says about prefill against decode;
- what you would do next, in what order.

Lead the report with anything that needs his decision. If a gate comes back negative, that is a result: say so and recommend the next thread rather than forcing the first one.
