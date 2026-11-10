# Thread: decode and small-batch speed on MPS

Opened 2026-11-10 from the lab's Q003 result, at Rob's direction to explore it hard. Measurements so far are in [the findings log](../../../findings.md) under 2026-11-10; raw results in `docs/review/2026-11-10-experiments/`.

## Where it stands (updated 2026-11-10, after the cross-family review)

Qwen3-1.7B fp16, M1 Max, PyTorch 2.15.0. Details, and the corrections, in [the findings log](../../../findings.md).

- **Batch 1:** `torch.compile` with a static cache, 1.67x against eager with the same cache (about 1.5x against default eager).
- **Batches 4 to 16:** compile plus the multi-row kernel (final table), 1.64x, 2.02x and 2.13x at batches 4, 8 and 16 against eager with a static cache. The kernel alone gives 1.34x to 1.55x at batches 8 to 16. Error against float32 comparable to eager fp16.
- **Verification:** with the kernel, the forward pass that verifies 4 drafted tokens costs 1.19x a single step (one process; to be replicated).
- **Prompt-lookup speculation:** large gains on a copy-heavy code edit with output identical to plain decoding, but the measured 3.3x to 3.5x used a burdened baseline and will be rerun. Without the kernel it was already 2.3x to 2.6x; the kernel adds about 1.3x to 1.4x. Free writing gains nothing and adaptive drafting without the kernel lost 14%.
- **Prior work:** arXiv 2610.31479 already connects small-row kernels to speculative verification on MLX (0.99x to 1.13x in one setup). The open part here is PyTorch on MPS, M1, 9 to 16 rows, and the size of the gain.
- **Next:** the rerun below, harness written (`bench/rerun_all.sh`), rules fixed before running. Needs the GPU, which the lab holds now.

## Rerun protocol (fixed 2026-11-10, before any rerun result)

The harness: `bench/rerun_decode_step.py`, `bench/rerun_spec.py`, `bench/check_spec_rollback.py`, `bench/bench_verify_cost.py` and `bench/check_decode_equivalence.py`, driven by `bench/rerun_all.sh` and summarised by `bench/rerun_analyze.py`. Correctness checks run first. Then come four rounds, each with the arm order rotated, where every process runs every arm and keeps every sample. Ratios are formed inside a process, then summarised across processes as a median and a range. Four processes cannot support a 95% interval, so none is claimed. The arms are:

- default dynamic-cache eager and static-cache eager;
- two separately compiled copies of the compiled decoder, and two of compiled plus kernel;
- for speculation, an ordinary compiled decoder (two copies) against fixed and adaptive drafting, with and without the kernel.

Requests run to the end-of-sequence token, so whole-request time covers a finished answer. The prompts are a code edit, a prose edit and free writing, with texts frozen in `bench/rerun_texts/`.

**Rules.**

- A speed claim stands only if all four processes agree on its direction, and its median clears 1 by more than the largest A/A deviation at that setting.
- The rollback check passes only if three things hold: every verified row is finite; each row lies within twice the plain decoder's own largest difference from a freshly rebuilt cache (at least 0.1, at most 1.0); and its argmax matches wherever the rebuild's top-two margin exceeds 0.5. The check counts only if every accepted length from 0 to 4 occurred, and the decoder switched both ways between one-token and five-token steps. Otherwise it is inconclusive. The two planted faults must fail it. A failed check stops the run before any timing.
- The analysis refuses to summarise if a cell is missing a process, a request hit its length cap, or repeats of an arm gave different tokens.
- Equivalence passes if each faster arm's mean and 99.9th-percentile error against float32 is within 1.25 times eager fp16's, and its top-1 agreement is no more than 2 tokens below eager's.

**Predictions.** A miss is reported as a miss.

- **Batch 1:** compiled at least 1.4x static-cache eager and 1.3x default eager. The four compiled arms sit within the A/A spread of each other. If not, the 9% gap seen before between nominally identical pipelines is real and needs an explanation.
- **Batches 8 and 16:** compiled plus kernel at least 1.8x static-cache eager, and the kernel at least 1.3x over compiled alone.
- **Verify cost:** with the kernel, five tokens cost at most 1.3x one token.
- **Code edit:** speculation with the kernel at least 2.5x the ordinary compiled decoder in decode time, and 2x in whole-request time.
- **Free writing:** adaptive drafting with the kernel no worse than 0.95x.

## Carrying this to the Kaggle T4 server (assessed 2026-11-10)

Rob asked whether and when to upgrade the Kaggle T4 server (`kaggle-qwen-serve`) with any of this, for a bigger model. That server runs llama.cpp `b12228` on CUDA. It serves a 4-bit (Q4_0) Qwen3.5-9B, one copy per T4, with MTP drafting at depth 2, which is about 17% faster than plain decoding there.

**Not worth porting: the compile result and the multi-row kernel.** Both fix overhead that PyTorch adds on MPS. llama.cpp on CUDA already has the equivalents:

- CUDA graphs are on by default in its build. That removes per-kernel launch cost, which is what `torch.compile` removed here.
- Its quantized matrix-vector kernels take up to 8 rows (`MMVQ_MAX_BATCH_SIZE`). That is the small-row path the Metal kernel added here.

There is nothing to carry over.

**Worth testing now: prompt-lookup drafting.** This is the part that transfers, and llama.cpp already has it:

- `b12228` ships five n-gram drafters.
- `--spec-type` takes a list that it tries in order, n-gram before MTP. Asking for an n-gram drafter plus `draft-mtp` gives the adaptive policy built here: copy from the context when there is a match, otherwise use MTP.
- So the upgrade is a server flag and a tune run, with no code from this repository.

It pays on copy-heavy work, and the server feeds OpenCode, whose agent tasks rewrite and edit files. Free writing gains nothing, so the A/B needs both kinds of prompt.

**The wrinkle is rollback on a hybrid model.** Qwen3.5 has recurrent layers, and a recurrent state cannot be rewound. llama.cpp keeps recurrent snapshots for as many draft tokens as the draft-length setting, which is 2 on this server.

- When a longer draft is partly rejected, the server restores a full checkpoint and replays the accepted tokens. That costs an extra forward pass.
- The n-gram drafters default to drafts of 48 to 64 tokens, so on this model they will often take that path.
- The A/B must therefore try shorter n-gram drafts and a larger snapshot count. That setting also sets the MTP depth, so the two are coupled.

This is the T4 counterpart of the cache-rollback bug found here. It is also where the change could lose.

**A bigger model: decide after the 9B A/B.**

- **Speculation pays more on a bigger model.** Each step costs more, while n-gram drafting stays free. So a bigger model is where this pays most.
- **A bigger model won't fit one T4.** A 27B model at 4 bits is about 16 GB, which needs both cards. One copy would then be split across them over PCIe, so the server loses its second copy and runs well below the 9B's speed.
- **So it only makes sense when answer quality matters more than throughput.** Run it with n-gram plus MTP drafting from the start.

The 9B A/B costs about 15 minutes of the 30 free GPU hours a week. It does not wait on the Mac rerun, because it measures llama.cpp directly.

## Experiments, in order

Each states its prediction before the run. Each runs through `bench/gpu_runner.py`, checks outputs against the stock model, and reports milliseconds per step at batches 1, 4 and 16.

**A. Remove the small-kernel half of the batch-1 step.** Changes measured one at a time and together: PyTorch's fused normalisation (`F.rms_norm`, one kernel per layer, in place of Hugging Face's seven), a preallocated KV cache, and later a fused rotary step. Subject check first: the profile above already shows the kernels exist and cost GPU time, so this needs no separate gate. Prediction: the batch-1 step falls from 21.8 ms to 16 ms or less, about 1.35 times faster, with greedy tokens unchanged and logit differences inside the model's own fp16 run-to-run spread. A result under 1.1 times would mean the per-kernel cost is not where the time goes, and the profile was misread.

**B. A multi-row kernel for 2 to 16 rows.** A Metal kernel in which each thread group streams one weight row once and applies it to every input row, with an optional bias. It starts from the repository's small-row prototype. Subject check: the per-op sweep already shows the gap on real shapes, larger than the A/A spread. Prediction: the 197 linears at batch 16 fall from 26 ms to 14 ms or less, and the full batch-16 step from 42 ms to 30 ms or less, about 1.4 times the throughput, with outputs matching `F.linear` within fp16 accumulation error. If it cannot beat Apple's kernel by 1.3 times on the layer-level sweep, stop before wiring it into a model.

**B, the research question behind it.** A NeurIPS 2026 paper ("Evaluating the Applicability of LLM Inference Optimizations on Apple Silicon", https://neurips.cc/virtual/2026/170039, seen as a search summary, not yet read) reports that every speculative decoding method it evaluated reduced throughput on Apple Silicon. Speculative decoding verifies several guessed tokens per step, which is the 2 to 16 row regime measured here as running far below bandwidth. Hypothesis: speculative decoding fails on Apple GPUs mainly because verification falls off the one-row fast path, and a multi-row kernel makes it pay. Once the kernel exists, the test is the verification cost per guessed token, with and without it, against the acceptance rates a draft model gets. Read the paper first; it may already measure verification cost.

**Results so far, 2026-11-10.** A: fused normalisation plus a fused rotary kernel, 1.41x at batch 1 (findings log). `torch.compile` with a static cache does better with no custom kernel, 1.60x at batch 1, 1.29x at 4 and 1.11x at 16, and is closer to float32 than eager. Its shrinking gain with batch size is the multi-row problem again. C: merged projections are exact and give almost nothing (1.02x at batch 1, 1.07x at batch 16). Hugging Face has reported `torch.compile` decode gains on MPS with PyTorch 2.15 (unverified vendor figures, mind-verse.de, 2026-10), so the compile result confirms a known direction on an M1 rather than finding a new one.

**C. Exact rewrites of the model's computation.** Fold each normalisation layer's scale into the next linear layer's weights, and fuse the query, key and value projections, then the gate and up projections, into one matrix each. These compute the same function in exact arithmetic. Prediction: together they remove about 113 elementwise kernels and 112 linear calls per step, worth 1 to 2 ms at batch 1; outputs agree within the bound below.

**D. The 17 to 32 row band on 7B-sized layers.** On the Llama-7B up-projection shape, `F.linear` is 2.4 to 2.7 times slower than a pre-transposed multiply at 17 to 32 rows. That is the shape of short-prompt prefill and wider speculative verification. Needs a 7B model (a new download, Rob's call), so it waits.

## The verified-rewrite experiment, a first design

Rob asked about AI systems rewriting a model's math to make it faster. What is public (2026-11-10): OpenAI says it used its own models to write the inference kernels behind GPT-6 Astra Ultrafast, with no details published; Google DeepMind's AlphaEvolve edited the compiled intermediate code of a FlashAttention kernel for a 32% speedup, checked against the original on random inputs and confirmed by experts. The general technique is to find a cheaper program that computes the same function, and to prove it does. The proof is what separates real results from the retracted ones (Sakana's AI CUDA Engineer kernels gamed their evaluator). No published work has an agent discover new algebraic rewrites of a trained model and verify them formally. Sources: `docs/research/program/raw/pivot_rewrite-claim.md` (git-ignored).

A careful version, for Rob to react to before anything is built:

- **Subject.** Three models on this machine that differ in structure: Qwen3-1.7B (attention and MLP, no biases), Qwen3.5-4B (a hybrid with recurrent layers), and Whisper (biases throughout, an encoder and a decoder). Two backends, PyTorch MPS and MLX, so a gain is not one dispatcher's quirk.
- **The rewrite set is fixed before any measurement.** Levels: schedule (layout, fusion, dispatch; arithmetic unchanged), algebra (an identity over the reals that changes rounding, such as folding a scale or merging projections), and approximation (uses a property of the trained weights, such as low rank; needs a tolerance and is reported as a trade-off curve, not a pass).
- **Equivalence, in three tiers.** First, the identity itself, checked in exact arithmetic on small random instances. Second, numerics: against a float64 reference, the rewritten model's error must be no worse than the stock model's own error, with the allowance derived from the format's unit roundoff and the length of each sum. Bit-equality with the stock output is never the test, since reordering legitimately changes the last bits. Third, end to end: next-token divergence and greedy agreement on a fixed prompt set must stay inside the stock model's own run-to-run and backend-to-backend spread.
- **Speed is predicted, then measured.** For each rewrite, count the bytes and kernels it removes, predict the gain from the measured per-kernel cost and bandwidth, then measure. A rewrite counts only if the prediction holds within a stated margin, so the result is an explanation, not just a timing.
- **What would make it general.** Each rewrite is written as a pattern over the operation graph, not as a patch to one model. The set is fixed on one model and tested on the other two. The finding is the property that predicts the gain across models and backends, not "model X got faster".
- **The agent step comes second.** Only once the hand-written set and the verifier work does an agent get to propose new rewrites, with the evaluator outside its reach, fresh inputs every check, and comparison against the float64 reference rather than cached outputs.

Experiments A to C are the hand-written first stage of this design.
