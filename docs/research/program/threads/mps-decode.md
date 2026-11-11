# Thread: decode and small-batch speed on MPS

Opened 2026-11-10 from the lab's Q003 result, at Rob's direction to explore it hard. Measurements so far are in [the findings log](../../../findings.md) under 2026-11-10; raw results in `docs/review/2026-11-10-experiments/`.

## Where it stands (updated 2026-11-10, after the rerun under fixed rules)

Qwen3-1.7B fp16, M1 Max, PyTorch 2.15.0. These are medians of four interleaved rounds; the ranges and tables are in [the findings log](../../../findings.md), entry "Decode rerun under fixed rules".

- **Correctness.** Speculative rollback matches a freshly rebuilt cache at every acceptance length, and both planted faults are caught. Error against float32 is comparable to eager fp16 on real and random prompts.
- **Batch 1.** `torch.compile` with a static cache runs 1.61x faster than static-cache eager and 1.55x faster than default eager. With a 512-token context it is 1.72x. The kernel adds nothing at batch 1, as expected.
- **Batches 8 and 16.** Compile plus the multi-row kernel runs 1.9x to 2.1x faster than either eager baseline, and the kernel alone contributes 1.66x to 1.77x of that.
- **Verification.** Checking 4 drafted tokens costs 1.52x a single step with the kernel and 2.08x without it. That misses the 1.3x prediction; the earlier 1.19x was one favourable process.
- **Prompt-lookup speculation**, against an ordinary compiled decoder, requests run to completion:
  - **Edits.** The output is identical to plain decoding. Code edit: 2.9x in decode time and 2.8x for the whole request. Prose typo fix: 3.1x to 3.3x and 2.9x to 3.0x. Without the kernel it is 2.1x.
  - **Free writing.** Adaptive drafting with the kernel breaks even (1.02x). Fixed-length drafting loses (0.58x without the kernel, 0.84x with it).
- **Prior work.** arXiv 2610.31479 already connects small-row kernels to speculative verification on MLX, with 0.99x to 1.13x in one setup. The open part here is PyTorch on MPS, an M1, 9 to 16 rows, and the size of the gain.
- **Next.**
  - The T4 A/B is done and shipped (section below): 2.8x on the edit benchmark, 16% on real agent traffic.
  - Scale folding, the other half of experiment C.
  - Why T = 2 verifies more slowly than T = 3 with the kernel (1.52x against 1.45x); a configuration choice at two rows is the likely cause.
  - **E. The attention cliff above 8 query rows (from the lab's Q009b, 2026-11-11).** Torch MPS attention (`_scaled_dot_product_attention_math_for_mps`) switches to a much slower path above 8 query rows. Measured over eight layers at Qwen3.5-4B's shape:

    | Context | T = 8 | T = 9 |
    |---|---|---|
    | 1,024 | 2.8 ms | 11.4 ms |
    | 4,096 | 9.1 ms | 44.2 ms |
    | 16,384 | 72.6 ms | 172.2 ms |

    T = 16 costs the same as T = 9, so it is a change of path, not per-row growth. MLX steps too, more mildly; llama.cpp does not. Source: `model-research-lab/research/verification/outcomes/Q009b-2026-11-11.md`.
    - **Practical rule now:** on MPS, verify at most 8 tokens (k ≤ 7), and fewer at long context. The rerun's 256-token verify table hides this, because attention is cheap there.
    - **Gate prediction (written 2026-11-11 before the run):** in whole Qwen3-1.7B with the kernel, T = 9 costs at least 1.3x T = 8 at a 4,096-token context and at least 1.6x at 16,384, while T = 8 costs at most 1.15x T = 5 at 4,096. If T = 9 is within 1.1x of T = 8 at 4,096, the cliff does not show in this model and E stops there.
    - **Gate result (2026-11-11):** passed at 4,096 tokens. T = 9 costs 1.63x to 1.87x T = 8 in every arm, with no overlap between the two sets of samples. The 16,384-token run timed out because the harness re-prefills before each sample. Findings log entry: "The attention cliff above 8 query rows shows in the whole model".
    - **The experiment:** a Metal attention kernel for 2 to 16 query rows over a long cache, the attention counterpart of the multi-row linear kernel. Prediction: at 4,096 context the T = 9 to 16 attention falls to within 1.5x of T = 8. Speculative verify at OpenCode-like depths (16k to 20k) then stops paying the cliff. Gate first: confirm the step in the whole Qwen3-1.7B model at 4k and 16k, through the existing `bench_verify_cost.py` with a longer prompt.
  - **E, the build (prediction written 2026-11-11, before any kernel code).**
    - **Cause, from the PyTorch v2.15.0 source** (`aten/src/ATen/native/mps/operations/Attention.mm`). Up to 8 query rows, SDPA uses `sdpa_vector`, with a two-pass split over the cache from 1,024 keys. Above 8 it uses the prefill tile kernel, which processes 16- or 32-row query tiles and is mostly empty for T = 9 to 16. The vector path handles each query row and head separately, so it likely re-reads the cache once per query row and per head. That would explain why one-token decoding at 4k already costs about twice what it costs at 256.
    - **The kernel.** A grouped-query, split-over-the-cache attention kernel for T = 1 to 16, causal within the new block. Each threadgroup loads a block of keys and values once for every query row and every query head that shares it. Accumulation is in fp32, with fp16 input and output and head dim 128. It plugs in through transformers' attention interface as a custom op, so `torch.compile` works.
    - **A zero-kernel comparison arm.** Split T into chunks of at most 8 rows and call stock SDPA once per chunk, so every chunk stays on the vector path.
    - **Predictions.**
      - Attention alone, over 28 layers' caches (so the caches are not cache-resident): the kernel at 4,096 keys costs within 1.5x of its own T = 1 at every T up to 16, and is at least 3x faster than stock SDPA at T = 9 to 16.
      - Whole Qwen3-1.7B compiled with both kernels, at 4,096 context: one-token decoding at most 20 ms (stock: 27 to 45 ms), and T = 9 at most 1.3x T = 8.
      - Correctness: within fp32-accumulation error of a float64 reference, no worse than stock SDPA's own error. Rollback and equivalence checks still pass.
    - **Stop rule.** If the isolated kernel cannot beat stock SDPA at T = 8 and 4,096 keys by 1.5x, stop before wiring it into the model.
  - **E, after the build (2026-11-11).**
    - **The kernel and its wiring are built and reviewed.** The kernel files are `bench/smallq_attention.{metal,py}`; the transformers wiring is `bench/smallq_hf.py` (`enable(model)`). There are 392 tests, and rollback still passes with both planted faults caught.
    - **One prediction is void.** The "one-token decoding at most 20 ms" prediction rested on a measurement artifact. The step right after a 4,096-token prefill pays about 30 ms once, from memory clocks dropping during long compute (a profile in the findings log). Steady-state one-token decoding at 4k is already about 15 ms.
    - **The whole-model rerun, rule fixed before it runs.** `bench_verify_cost.py --rewind`, at 4,096 and 16,384 context, three rounds with the T order rotated. Arms: compiled plus multirow, with and without the small-query kernel (`compiled_kernel_smallq`). The kernel counts as a gain at a T where every round's median favours it and the median ratio clears the larger A/A deviation. Predictions: with the kernel, T = 9 costs at most 1.3x T = 8; and T = 9 to 16 is at least 1.8x faster than without it at 4,096 context.
  - **The MLX counterpart (also Q009b).** MLX's 4-bit matrix product runs a per-row path up to about 9 rows and switches to a tiled path only from 12 rows. That is the same small-row gap the multi-row kernel closed in PyTorch, so it is a candidate upstream issue for mlx.
  - Whether to write this up as one result: a PyTorch MPS path to about 3x on edit-style generation, exact output, with no draft model.

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

The A/B used about 36 minutes of GPU time (12 for the benchmark, 24 for the replay). Kaggle's settings page showed 27.5 of 30 weekly GPU hours used on 2026-11-10, refreshing 2026-11-14. The API's quota call misreports this.

**Result (2026-11-10).** Prompt lookup transferred and now ships in serve v4 as `--spec-type ngram-simple,draft-mtp`, with 8-token lookup and drafts and `--spec-draft-n-max 2`. The evidence is in `kaggle-qwen-serve`'s EVIDENCE.md (two dated entries) and its raw JSON.

- **On the frozen benchmark,** against MTP depth 2 alone: 2.76x on the code edit, 2.32x on the prose edit, and 1.10x on free writing, which is within noise. Edit outputs matched plain greedy exactly.
- **On real OpenCode traffic,** with 44 captured agent requests replayed: decode is 16% faster with OpenCode's sampling and 25% faster greedy, and a session is about 10 to 15% faster end to end.
  - **The gain is in tool-call arguments,** which run 50 to 80% faster. Reasoning gains 5 to 10% and answer text nothing.
  - **The benchmark overstated real use** by about ten times in the excess (2.8x against 1.16x). Agent edits are short diffs, and reasoning has nothing to copy.
- **The rollback wrinkle was cheap at n-max 2.** A long edit used about 100 checkpointed drafts and 3 to 5 restores. Raising n-max removes restores but deepens MTP, which cut free writing to 0.57x, so it was rejected.
- **Two b12228 defaults silently disable short drafts.** ngram-mod's `n-min` of 48 discards anything shorter. ngram-simple drafts only when `size-m` is at least `size-n`, whose default is 12.
- **On long agent contexts,** greedy output differs between MTP alone and n-gram plus MTP, from the batch-shape numerics in llama.cpp #25618. On short prompts it matched.

**Lesson for this thread.** A copy-heavy benchmark measures the mechanism, not the workload. The Mac's 2.9x to 3.3x on edits is real for whole-file rewrites. For an agent that emits short diffs, expect the T4's tens of percent. A real-traffic replay is the number to quote.

**The bigger model fits and is usable (2026-11-10, `kaggle-qwen-serve` EVIDENCE.md, tune-27b v1).** The test ran the Orca 27B (Qwen3.8-27B, mostly IQ4_XS, with its own MTP layer) on both T4s, with the cards splitting each layer's computation (`-sm tensor`) and the full 262,144-token context. It used 12 minutes of GPU time.

- **Plain decoding** runs at 18 tok/s, against the 9B's 31.
- **With n-gram drafting plus MTP,** the code edit runs at 49 to 58 tok/s and free writing at 22 to 25 tok/s. Against the 9B, that is about 0.45x on edits and 0.7x on free writing.
- **Prefill** is about 530 tok/s. A cold 19,552-token prompt gave its first token after 45 s.
- **Correctness.** Edits were exact against plain greedy, and every bar set before the run held.
- **How it compares with the earlier estimate.** Speculation pays proportionally more here: 2.4x over plain decoding against about 1.7x on the 9B. That matches the expectation. But MTP acceptance on free writing is only 53%, so deeper MTP slows it.
- **Plan.** The serve switch to the 27B waits for the quota refresh on 2026-11-14: a 20-minute tune run first, then serving. That is `kaggle-qwen-serve`'s call.

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
