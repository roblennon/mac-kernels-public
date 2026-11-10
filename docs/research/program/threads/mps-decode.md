# Thread: decode and small-batch speed on MPS

Opened 2026-11-10 from the lab's Q003 result, at Rob's direction to explore it hard. Measurements so far are in [the findings log](../../../findings.md) under 2026-11-10; raw results in `docs/review/2026-11-10-experiments/`.

## Where it stands (updated 2026-11-10, after the cross-family review)

Qwen3-1.7B fp16, M1 Max, PyTorch 2.15.0. Details, and the corrections, in [the findings log](../../../findings.md).

- **Batch 1:** `torch.compile` with a static cache, 1.67x against eager with the same cache (about 1.5x against default eager).
- **Batches 4 to 16:** compile plus the multi-row kernel (final table), 1.64x, 2.02x and 2.13x at batches 4, 8 and 16 against eager with a static cache. The kernel alone gives 1.34x to 1.55x at batches 8 to 16. Error against float32 comparable to eager fp16.
- **Verification:** with the kernel, the forward pass that verifies 4 drafted tokens costs 1.19x a single step (one process; to be replicated).
- **Prompt-lookup speculation:** large gains on a copy-heavy code edit with output identical to plain decoding, but the measured 3.3x to 3.5x used a burdened baseline and will be rerun. Without the kernel it was already 2.3x to 2.6x; the kernel adds about 1.3x to 1.4x. Free writing gains nothing and adaptive drafting without the kernel lost 14%.
- **Prior work:** arXiv 2610.31479 already connects small-row kernels to speculative verification on MLX (0.99x to 1.13x in one setup). The open part here is PyTorch on MPS, M1, 9 to 16 rows, and the size of the gain.
- **Next:** a rerun with every arm interleaved across processes, all samples kept, A/A arms, default dynamic-cache eager as a baseline, an ordinary compiled loop as the speculative baseline, and forced acceptance lengths checked against a rebuilt cache. Needs the GPU, which the lab holds now.

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
