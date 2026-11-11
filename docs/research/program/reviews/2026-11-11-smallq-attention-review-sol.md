**The measured speedup holds for the tested workload, but the transformers wiring is not safe for every accepted call.** This is a read-only source/data review; the counterexamples below were not GPU-reproduced.

**1) Blocking defects**

- **P1 — RoPE positions are treated as cache slots.** [smallq_hf.py:143](../../../../bench/smallq_hf.py:143) uses `position_ids[:, 0]` after [line 111](../../../../bench/smallq_hf.py:111) discards the mask’s actual cache offset. Concrete input: StaticCache length 100, capacity 256, T=4, explicit `position_ids=[[50,51,52,53]]`. Transformers writes slots 100–103; the kernel masks at 50–53. With zero Q/K and `V[n]=n`, row zero returns **25 instead of 50**. These positions are within the kernel’s bounds, so clamping does not help. The fallback reconstructs the same wrong mask. Pass the actual cache offset through; documenting this assumption is insufficient for a generally enabled backend.

- **P1 — The noncausal fallback silently becomes causal.** [smallq_hf.py:146](../../../../bench/smallq_hf.py:146) rejects `is_causal=False`, then unconditionally builds a causal mask. Concrete input: B=Hq=Hkv=1, T=L=4, D=64, zero Q/K, `V[n]=n`, positions 0–3, `is_causal=False`. Expected output is **1.5 on every row**; this path returns **0, 0.5, 1, 1.5**. The fast-path predicate also overlooks `module.is_causal=False` when the keyword is absent.

- **P2 — An accepted finite scale can produce NaNs from finite inputs.** [smallq_attention.metal:159](../../../../bench/smallq_attention.metal:159) computes `scale*log2(e)` before multiplying scores. With Q=K=zeros `[1,1,1,64]`, V=ones, pos=0, `scale=3e38`, that intermediate overflows; [line 233](../../../../bench/smallq_attention.metal:233) evaluates `0*inf`. The mathematical answer is ones. Restrict the accepted scale domain or change the arithmetic.

**2) Nonblocking weaknesses**

- **One original prediction missed.** [mps-decode.md:39](../../../../docs/research/program/threads/mps-decode.md:39) predicts isolated T≤16 within 1.5× T1. T16/T1 exceeds that in **five of six** 4k runs: 1.61, 1.47, 2.17, 1.81, 2.04, 2.48. Disclose this alongside the later predictions that passed.
- **Several conclusions exceed their evidence.** [findings.md:1504](../../../../docs/findings.md:1504): low effective bandwidth alone does not establish an instruction-issue bottleneck; the byte estimate also omits repeated cache reads across row groups. [Line 1507](../../../../docs/findings.md:1507): “every” compiled StaticCache forward is too broad. Clock changes remain a hypothesis, although the post-prefill penalty itself is supported.
- **Validation is narrower than the wording.** Whole-model correctness covers T=1,4,8,9,16 at 4k, not every T4–16 or 16k. Profile drivers are missing, and timing JSONs omit useful run configuration.
- **API hardening remains.** [smallq_attention.py:205](../../../../bench/smallq_attention.py:205) synchronises if `scale` is an MPS scalar tensor. Empty dimensions, invalid configurations, and int32 narrowing lack guards. The custom op also needs an explicit inference/autograd contract before upstream use.

**3) Checked and found sound**

- Q/K/V indexing, GQA mapping, output permutation, per-batch position strides, and barriers protecting score/probability reuse are internally consistent.
- For valid positions, Lmax chunking handles empty chunks correctly; terminal loads avoid stale tail NaNs. Reduction takes the maximum across **all** chunks and consistently combines online-softmax states.
- Padding/custom/sliding masks are preserved in the normal mask-creation flow. The supported device-position path with ordinary Python scales has no tensor-value readback.
- **`--rewind` is valid for the tested default Qwen3 positions.** Each call overwrites its current verify slots; causal masking excludes earlier writes beyond them. It measures warmed, fixed-context verification.
- Raw results reproduce **2.65–2.75× at 4k**, **5.09–5.37× at 16k**, and kernel T9/T8≈**1.14**. The dispatch boundary matches [PyTorch v2.15.0](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/mps/operations/Attention.mm#L753).
- The post-prefill correction is supported: **14.52 ms steady, 44.47 ms first step, 15.94 ms second step**. Rollback coverage and both planted-fault failures are meaningful.
