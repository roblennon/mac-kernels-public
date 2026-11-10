# Decode-stack review by Sol, 2026-11-10

A cross-family review of the 2026-11-10 decode work (torch.compile, the multi-row kernel, verification cost, prompt-lookup speculation and the equivalence checks), by GPT-6.1 Sol via the Codex CLI, read-only, high reasoning effort. Its text follows unedited except that local absolute paths in links were made relative. What was done about it is recorded in the findings log entry "Corrections after the cross-family review".

---

## 1. Verdict on each headline claim

**(a) Holds with caveats.**  
The final JSON gives 38.439/19.026 = **2.020x** at batch 8 and 42.168/19.795 = **2.130x** at batch 16.  
Both arms use the same static-cache capacity, so this comparison is internally fair.  
The measurement is warmed decode latency on short random prompts, excluding prefill and compilation.  
The harness selects the **minimum of two pass medians**, runs arms in a fixed order, and saves neither individual samples nor both pass results ([timing code](../../../bench/bench_decode_compile_kernel.py:38)).  
The substantial improvement is credible, but the hundredths and repeatability are not established.  
These numbers do not establish the gain over default dynamic-cache eager decoding.

**(b) Holds with caveats.**  
The earlier batch-1 result records 24.921/14.949 = **1.667x**, against eager with a static cache.  
Earlier dynamic-cache eager measurements are faster, so “1.67x over default eager” would overstate the evidence.  
Batch 1 was not repeated in the final batch.  
The kernel-enabled arm takes 16.344 ms despite never invoking the kernel at one row, exposing an unexplained roughly 9% difference between nominally equivalent configurations.

**(c) Holds with caveats.**  
The final verification result gives 18.956/15.951 = **1.188x** for five input tokens, meaning the current token plus four drafts.  
This uses nine synchronized measurements after warm-up and resets/refills the cache before each measurement ([verification harness](../../../bench/bench_verify_cost.py:23)).  
It measures the forward pass, excluding drafting, acceptance decisions and rollback overhead.  
It therefore supports “verification forward costs 1.19x,” not “a speculative round costs 1.19x.”  
Only one final process measures these cells, so the exact 19% increment needs replication.

**(d) Holds with caveats for the saved code-edit experiment.**  
CPU comparison confirms that all four speculative code-edit outputs match every one of the plain run’s 160 saved tokens.  
The kernel ratios are **3.508x** and **3.260x**.  
Those ratios use an unnecessarily burdened plain baseline, described below.  
They measure warmed decoding of a 160-token prefix, excluding prefill, rather than completion of the edit or whole-request latency.  
Universal “exact” wording does not hold: adaptive kernel free writing differs in **121/160 tokens**, starting at zero-based position 34.  
A small margin in eager full-forward logits makes numerical divergence plausible, but cannot establish identity with compiled single-token greedy decoding.

**(e) Holds with caveats as observed numerical comparability.**  
The final kernel result improves mean absolute error from 0.004005 to 0.003044 and p99.9 error from 0.03180 to 0.02219.  
It worsens maximum error from 0.09962 to 0.10407 and top-one agreement from 511/512 to 510/512 ([final equivalence results](../../../docs/review/2026-11-10-experiments/final-kernel-rerun/equiv.json)).  
“Closer on every measure” is false.  
The decode comparison covers 16 random-token prompts and 32 subsequent positions at batch 16.  
Both equivalence scripts record metrics without enforcing their advertised acceptance rule.  
This supports comparable observed error on tested inputs, not a general no-worse fidelity guarantee.

## 2. The most serious flaw, and how to fix it

**The plain speculative-decoding baseline performs unnecessary rollback work after every token.**

The loop unconditionally fills every layer’s `cumulative_length`, including when `kk == 0` ([rollback code](../../../bench/bench_prompt_lookup.py:69)).  
Plain decoding therefore pays 28 redundant counter writes per token, while successful speculation amortizes those writes across several tokens.  
This systematically burdens the denominator of the reported speed-up.  
Its size cannot be recovered from the saved results.

Perform rollback only when drafts were rejected, meaning `n_ok < kk`.  
Benchmark a plain compiled loop without speculative bookkeeping.  
Rerun the comparison before presenting 3.26x–3.51x as performance against an ordinary compiled decoder.

The rollback’s position arithmetic itself looks sound for this Qwen configuration.  
After accepting `n_ok` drafts, `pos += 1 + n_ok` correctly identifies the first slot to overwrite.  
The installed Transformers code derives both rotary positions and causal-mask offsets from the live counter.  
Rejected slots remaining beyond each query’s causal boundary stay masked, and newly visible slots are overwritten before use.  
In-place `fill_` preserves the counter tensor’s identity, so there is no evident stale Python-value capture.  
Compiled execution still needs a targeted rejection-and-replay test.

## 3. Anything that would embarrass the author if published as is

- **The prior-work characterization is wrong.**  
  The October 22 paper explicitly connects small-row kernels to speculative verification in §2.3.2 and reports speculation improving from 0.99x to 1.13x versus plain decoding in Table 4.  
  Its §2.2.3 also reports positive speculative configurations, including n-gram decoding on M3 Max.  
  “It did not connect the two” and an unrestricted “every method loses on Apple Silicon” misrepresent the paper.  
  ([Paper](https://arxiv.org/pdf/2610.31479))

- **The kernel is not what first makes this code-edit speculation profitable.**  
  The final no-kernel speculative arms already achieve 2.28x–2.62x.  
  The kernel adds approximately 1.34x–1.43x over those arms.

- **The thread summary contradicts the final results.**  
  It retains provisional timings, categorical accuracy language and the assertion that adaptive drafting keeps free writing at parity.  
  The final no-kernel adaptive result is 0.86x.

- **A “tie flip” does not preserve exact output.**  
  The divergence checker uses eager fp16 full-forward execution as a third reference path.  
  Its preference cannot make two different generated continuations identical.

- **One kernel contract hole survives the extensive tests.**  
  `supports()` accepts zero-width inputs, but matrix variants divide by zero when calculating splits ([wrapper](../../../bench/multirow_linear.py:224)).  
  A CPU-only replay confirmed this for shapes `[5,0]` and `[3,0]`.  
  It does not affect the measured Qwen shapes.

- **PR 198721 attribution is broadly fair, but causality is overstated.**  
  The PR enables existing GEMV kernels for `F.linear`.  
  Comparing two whole PyTorch releases does not isolate how much of the observed 2.5 ms improvement that PR caused.  
  ([Merged PR](https://github.com/pytorch/pytorch/pull/198721))

## 4. What a careful follow-up run should add

- Compare eager dynamic-cache, eager static-cache, compiled-only and compiled-plus-kernel arms on identical workloads.

- Randomize or interleave warmed measurements across repeated independent processes, retain every sample, and report uncertainty instead of selecting the faster pass.

- Add A/A controls for the complete compiled pipelines, especially batch 1 and verification sizes.

- Separate prefill, compilation, decode and whole-request latency, and include completed code edits.

- Force every acceptance length from zero through four, including consecutive rejections, full acceptance and switches between one-token and five-token graphs.

- Compare rollback execution against a freshly rebuilt cache on the same history, using large finite sentinels in rejected slots to expose accidental visibility.

- Define the numerical acceptance rule before running, add real-text prompts and longer contexts, and compare candidate-token margins across execution paths.

- Carry the stronger relative-L2 kernel criterion into pytest, because existing mutation tests show that its max-error check alone misses meaningful arithmetic defects.

- Add compiled custom-op contract checks.  
  I found no apparent aliasing or fusion defect in the current registration: it declares no input mutation, returns fresh storage and supplies matching fake-output metadata.

The review remained read-only and used no GPU execution or installations.
