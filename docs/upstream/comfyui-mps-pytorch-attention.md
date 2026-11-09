# ComfyUI PR (b): PyTorch attention by default on Apple GPUs

Branch `refresh/mps-pytorch-attention` (private remote), head `d092de0a`, four commits on `origin/master` `08ff3c11` (2026-10-09). It supersedes `upstream/mps-pytorch-attention` (`1121361c` on `65787d66`), which stays untouched. Independent of the other branches.

Commits:
- `3fb80e71` perf(mps): default to pytorch attention on Apple GPUs with torch >= 2.13 (original)
- `e1181459` fix(mps): preserve independent attention broadcasts (original)
- `f00f22ab` fix(mps): keep independent head broadcasts out of one native SDPA call (ported from `e11d9226` in the integrated runtime)
- `d092de0a` fix(mps): loop independent batch broadcasts and bound VAE attention (this refresh)

Verdict: **still needed on both Torch 2.14.1 and the 2.16 nightly.** The flash kernels are the reason to switch; the safeguards fix real native failures, two of which also exist on nightly.

## Title

Default to pytorch attention on Apple GPUs with torch >= 2.13

## Description

PyTorch 2.13 added flash-style SDPA kernels for MPS that do not materialize the attention matrix. ComfyUI still picks sub-quadratic attention on Apple GPUs, which now costs both speed and memory. This applies the rule NVIDIA and XPU already use: pytorch attention unless `--use-split-cross-attention` or `--use-quad-cross-attention` is given.

`attention_pytorch` gets MPS-only safeguards in one helper, `_sdpa_bounded`:
- **Split guard.** A call is split over batch, heads, then queries when its score tensor would exceed 2^31 elements (MPS 32-bit indexing, #14837; the approach is from #14838). Below the limit it is one call. The floor is one query row (`k_len` scores; `heads * k_len` with `enable_gqa`, since grouped heads are never split).
- **Independent broadcasts go through the loop.** One native MPS call with a singleton Q batch against K/V of batch b returns a (1, ...) output on torch 2.14.1, and a singleton K head against several Q heads aborts the process (`LLVM ERROR: Failed to infer result type(s)`) on 2.14.1 and the 2.16 nightly. When Q, K and V do not share batch and head extents the helper takes its loop even under the cap, so every native call sees equal extents through views of the singleton operand.
- **fp32 when asked.** The call is upcast when the model requests fp32 attention (`attn_precision`, e.g. SD2) or `--force-upcast-attention` is set, as the sub-quadratic path did.
- **VAE attention is bounded too.** The default switch turns on `pytorch_attention_enabled_vae()`, whose single head of width C is outside the flash kernels and materializes the score tensor; on MPS it now goes through the same helper. Off MPS it is one direct call as before.

The blanket macOS >= 14.5 fp16 upcast is not applied to SDPA; it exists for the matmul-softmax paths, and `attention_pytorch` has never applied it on any platform. Torch SDPA on MPS already accumulates in fp32: against an fp64 reference of the same fp16 inputs, at head dims 40, 64, 128 and 160 (flash and MPSGraph fallback), with no mask, a float mask and a bool mask, and with logits up to 1e7 (far past the fp16 maximum), no output was non-finite, and the fp16 result differed from the reference by at most 1.2e-4 at unit scale, less than one fp16 step. An fp32 upcast of the same inputs is more accurate only until the result is cast back to fp16. The upcast cost 1.74x at (1, 24, 9000, 128) on this machine (M1 Max, macOS 15.7, torch 2.14.1; 2.2x in the earlier Wan measurement).

Non-MPS behavior is unchanged: the MPS branch is a separate `if` in `attention_pytorch` and in the VAE `pytorch_attention`.

M1 Max 32 GB, torch 2.14.1, fp16 (2026-10-03, original branch; not re-measured in this refresh):

| | SDPA | sub-quadratic (today) |
|---|---|---|
| Wan 2.2 5B 704x416x33, per step | 3.33 s, 6.1 GB | 3.61 s, 8.3 GB |
| Wan 2.2 5B 1280x704x81, per step | 37.4 s, 9.1 GB | 67.4 s, 17.2 GB (swaps) |
| One 14B 832x480x81 self-attention (4.3e10 scores) | 3.0 s, 3.0 GB | 11.9 s, 11.0 GB |

## What the probes showed (2026-10-09, M1 Max 32 GB, macOS 15.7)

Torch 2.14.1 and 2.16.0.dev20261009 unless stated. Scripts live outside the branch (scratchpad `probes/`).

- **Flash path above 2^31.** b=h=1, head dim 64, fp16: one native call matches a query-chunked reference bit-for-bit at 2.15e9 and 4.36e9 scores (2x 2^31) with zero driver-memory growth, on both versions. The split guard is therefore not what protects the flash kernels; it bounds the MPSGraph fallback (head dims outside 32/64/72/80/96/128/256, or a float mask whose dtype differs from Q), which materializes about 10 bytes per score (head dim 48: +2.5 GiB at 2.7e8 scores). A 2^31-score fallback call would need about 20 GiB, so the corruption above 2^32 (pytorch#179352) could not be reproduced on this machine.
- **Head broadcast.** K with one head against Q with three: SIGABRT on both versions. V with one head: correct natively. Q with one head: rejected as invalid GQA.
- **Batch broadcast (review finding, reproduced).** Q batch 1 against K/V batch 2, under the cap: output shape (1,4,19,64) on 2.14.1, silently wrong; correct on nightly. K or V batch 1: correct on both.
- **fp16 overflow (review finding, not reproduced).** See the description.
- **Kernel dispatch** read from PyTorch 2.14.0 `Attention.mm`: the prefill (flash) path needs head dim in {32,64,72,80,96,128,256}, query length > 8, and a mask that is bool or Q's dtype; otherwise `sdpa_general_mps` materializes scores through MPSGraph and casts QK^T to fp32 before scaling.

## Test evidence (2026-10-09)

- `tests-unit/comfy_test/sdpa_bounded_test.py` and `mps_attention_default_test.py`: 122 passed on torch 2.14.1 and on 2.16.0.dev20261009 (MPS runs through the repo's GPU guard).
- Failing-first: with the review fixes reverted, 6 of the new tests fail on 2.14.1 (batch-broadcast fast-path skip x3, MPS singleton-Q parity x2, VAE bound); the pre-port `test_under_cap_broadcast_heads_use_matching_native_views` crashed the interpreter (the native abort).
- `tests-unit/comfy_test` in this worktree: 313 passed, 14 skipped (torch 2.14.1).
- CPU-only collection: importing `comfy.ldm.modules.attention` without `args.cpu` raises "Torch not compiled with CUDA enabled" on CPU-only torch (simulated by disabling MPS); the module now sets `args.cpu` the way neighbouring tests do.

## Independent review (read-only, 2026-10-09)

Three blocking, three should-fix, one nit. Reproduced and fixed: under-cap batch broadcast, unbounded VAE SDPA, CPU-only collection crash, the bound's floor (documented and asserted). Not reproduced: fp16 overflow on the fallback path (probe above). Not done: CPU-runnable selection-matrix tests for the default switch (the selection test still needs MPS), bf16 numerical evidence (see notes).

## Upstream overlap (checked 2026-10-09)

- #14838 (purposebuiltscott, open, last updated 2026-08-14): size-based chunking in every attention path plus `attn_precision` on SDPA. It does not switch the default, does not bound the VAE path, and does not handle the broadcast failures. If it lands first, this PR shrinks to the default switch, the VAE bound, the broadcast loop and the fp32 rule; the split arithmetic should then reuse its chunking.
- #15819 / issues #15818 and #15804: bf16 LTX renders black on MPS under the sub-quad default. #15804 traces it to `torch.baddbmm(torch.empty(...), beta=0)` in `sub_quadratic_attention.py` honouring the uninitialized input on MPS. SDPA never calls that path, so this PR removes that failure for the default configuration; worth saying in the PR.
- #15902: the same 5D `F.pad` corruption the conv3d branch avoids, in the H3 VAE.
- No newer MPS attention PR or issue in the last two months overlaps this change beyond the above.

## Dependencies and scope

None. The default switch and all guards apply only when the device is MPS. The maintainers may prefer the default-policy switch separate from the safeguards; the four commits already split that way (first commit versus the rest), and the safeguards stand on their own since `--use-pytorch-cross-attention` already reaches this code on MPS today.

## Notes (not part of the PR)

- Re-reviewed independently on October 9 with an independent probe: the no-upcast policy is correct. Native fp16/bf16 SDPA on MPS is accurate to the output rounding floor (fp16 rel 2e-4, bf16 1.7e-3, both equal to one output rounding) with no overflow at any tested head dim, mask or logit range; the upcast's extra accuracy is lost at the cast back. Upstream's pytorch path applies no upcast anywhere, so this is also the status quo for `--use-pytorch-cross-attention` on Macs today. The earlier "bit-identical" claim was wrong and is corrected in the branch comment.
- bf16 is not upcast by this branch (same as every other path today, since `FORCE_UPCAST_ATTENTION_DTYPE` maps fp16 only), but `--force-upcast-attention` does upcast bf16 here while the other paths ignore the flag for bf16. No bf16 numerical evidence was gathered; the fp16 overflow probe suggests the kernels accumulate in fp32, which should cover bf16 too, but that is inference.
- The Wan timing table is from 2026-10-03 and was not re-measured; the refresh ran unit tests and microprobes only. One early probe run was killed by the GPU guard for system memory pressure while another job was on the GPU; the rerun of the same sizes was clean, so it is attributed to concurrent load, unverified.
- V-only head broadcast works natively, yet the loop still runs one head per call for it (b*h calls). Harmless, no ComfyUI caller does this, but a reviewer may ask.
- The fp16 overflow result is from macOS 15.7; `Attention.mm` has a macOS 15 branch in the fallback, so older macOS could differ.
