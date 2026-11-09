# comfy-kitchen PR: mps backend (weight-only INT8 and W4A8/W6A8 on Apple GPUs)

Branch: `refresh/mps-backend` in `vendor/comfy-kitchen-upstream`, head `8782535`, eight commits on upstream `be003b7` (v0.2.37, current as of October 9, 2026). It supersedes `upstream/mps-backend`. Training, additional block formats and GGUF are [separate follow-up branches](comfy-kitchen-mps-followups.md).

| Commit | What |
|---|---|
| `cea0bc3` | Backend skeleton and weight-only `int8_linear` (INT8, ConvRot), transposed decode for wide inputs |
| `2a6e925` | Weight-only `w4a8_int8_linear` (W4A8 with or without codebook, W6A8); power-of-two row scaling so fp16 cannot overflow |
| `9a6898e` | Metal dequantize, and requantize registered for mps, for W4A8/W6A8 (LoRA merges load) |
| `5698363` | Range-safe fused prologue / decode / epilogue for both formats |
| `f331ad5` | Unmerged LoRA arguments (`lora_down`, `lora_up`, `lora_scale`) as opt-in parameters, registry check, README section |
| `19299c4` | Release previous chunk outputs and views before allocating the next decode |
| `575317b` | Empty contractions return before any reshape; `int8_linear` declares its output dtypes (from source fix `a138076`) |
| `8782535` | LoRA epilogue keeps tiny and cancelling terms; M = 1 activation copy; documented precision contracts (October 9 review) |

Cleanups made while splitting, with no change to code:
- `tests/test_mps_w4a8.py` lint fixed in each commit: dict literals, an explicit `zip(strict=False)`, and a `noqa` for the `F` import. These were the repo's only ruff findings.
- A README section for the backend, added in the last commit.

---

## PR title

Add an mps backend: weight-only INT8 and W4A8/W6A8 linears for Apple GPUs

## PR description

### What

A new `mps` backend for Apple GPUs that implements:
- `int8_linear` (INT8, optionally ConvRot).
- `w4a8_int8_linear` (the `AsymW4A8Int8Layout` formats: 4-bit with or without a Lloyd-Max codebook, 6-bit, symmetric or asymmetric).
- `dequantize_w4a8_int8_weight` and `quantize_w4a8_int8_weight`, so hosts can merge LoRAs into those weights.

`int8_linear` and `w4a8_int8_linear` also accept an unmerged LoRA (`lora_down` [R, K], `lora_up` [N, R], `lora_scale`). These are opt-in parameters: the registry skips a backend that does not declare them, so existing backends and callers are unchanged. The backend registers only when MPS is available and `torch.mps.compile_shader` exists.

### Why

Today these formats have no fast path on a Mac. On torch 2.14 and earlier MPS does not implement `torch._int_mm`, so ComfyUI disables the INT8 formats on Apple GPUs and dequantizes the weights in full on every call. PyTorch 2.15.0 (released 2026-10-28) adds a native MPS `_int_mm` ([pytorch#193153](https://github.com/pytorch/pytorch/pull/193153)); on M1 to M4 it is a generic tiled kernel, not a matrix-unit path, and this PR has not measured it. ComfyUI [#16451](https://github.com/Comfy-Org/ComfyUI/pull/16451) would re-enable INT8 compute on MPS for torch >= 2.15, which changes the stock baseline this PR argues against. W4A8/W6A8 cannot run there at all. On a 32 GB Mac that means:
- MiniMax H3 INT8 swaps (38 to 74 s per step).
- The 6-bit H3 file does not load.
- Krea 2 / Qwen-Image 2.1 INT8 pay for a full bf16 dequantization per layer per call.

### Design

- **Weight-only.** Apple GPUs before M5 have no fast integer matrix path. So integer weights stay resident, a Metal kernel decodes them to fp16 in one pass, and the stock MPS fp16 GEMM runs. Activations are not quantized. ConvRot checkpoints store the weight already rotated, so the activation is rotated online (group-wise, cheap) rather than un-rotating the weight each call.
- **Range safety by construction.**
  - A fused prologue divides each activation row by an exact power of two so its max after rotation is at most 1, and applies the rotation in the same pass.
  - The decode divides each weight row the same way.
  - Every dot product is then bounded by K, so the fp16 GEMM cannot overflow for K up to 65504. Wider layers compute in fp32.
  - A fused epilogue restores both exponents with integer exponent arithmetic. Apple GPUs flush fp32 subnormals in arithmetic, so this is done on the raw bits.
  - It adds bias and residual in eager's order for each format, and writes bf16 bit-exactly with torch's cast.
- **Unmerged LoRA.**
  - The down weight goes through the same prologue and rides as R extra rows of the first decoded weight chunk, so the main GEMM also yields the low-rank activation.
  - An epilogue applies the up weight in fp32 on the simdgroup matrix units.
  - The up weight is carried as mantissa plus exponent, and base and LoRA terms are summed in a common scaled domain. Extreme scales stay finite, and a LoRA that cancels the weight gives exactly 0.

### Measurements

Mac Studio M1 Max 32 GB, macOS 15.7.9, PyTorch 2.14.1, October 9, 2026. One run of `bench/bench_ck01_int8.py` (mac-kernels repo) against this branch; bf16 in and out, median ms per call, relative L2 error against an fp32 linear over the dequantized weight in brackets.

| Shape (M, K, N) | dense bf16 `F.linear` | dense fp16 `F.linear` | dequantize per call (today) | mps backend |
|---|---|---|---|---|
| INT8 4608, 6144, 6144 | 68.3 (2.3e-3) | 38.5 (1.7e-3) | 76.4 (2.3e-3) | 39.3 (1.7e-3) |
| INT8 4608, 6144, 16384 | 183.0 | 102.7 | 201.7 | 104.1 |
| INT8 4608, 16384, 6144 | 228.1 | 204.5 | 242.9 | 106.1 |
| W4A8 codebook 4608, 16384, 6144 | | | 285.7, fp32 scales only | 110.4 |
| W6A8 4608, 16384, 6144 | | | 299.4, fp32 scales only | 110.8 |

The backend runs 1.9–2.3× faster than dequantizing per call, at fp16-dense speed and accuracy. On the wide-input shape it is also about 2× faster than a dense fp16 `F.linear`, because dense `F.linear` hits a slow MPS path for transposed weights at large K (reported separately to PyTorch), which the transposed decode avoids. That ratio holds only against `F.linear`'s transposed path. Against a bare fp16 GEMM with a contiguous weight the kernel is about 4% slower at that shape (107 against 103 ms, October 3 hardening table in findings.md). Eager's W4A8/W6A8 dequantize cannot run on MPS with e4m3 scales (no fp8 arithmetic), so for those files there is no path today.

Unmerged LoRA, rank 64, fused overhead per call: +2.0 ms on 4608×6144×6144 (base 38.8 ms) and +3.2 ms on 4608×16384×6144 (base 105.5 ms).

End to end in ComfyUI, seconds per step, measured earlier with the integrated runtime (October 2–4). This needs the companion ComfyUI change that routes these formats to the backend:

| Model | Stock ComfyUI | With the mps backend |
|---|---|---|
| Krea 2 Turbo INT8 ConvRot, 512x512 | 8.7 | 3.7 |
| Qwen-Image 2.1 INT8 ConvRot, 1024x1024 | 15.1 | 8.8 |
| MiniMax H3 INT8, 480x288x22 frames | 38 to 74 (swapping) | 8.4 to 11.8 |
| MiniMax H3 W6A8, 480x288x22 frames | does not run | 6.5 |

### Relationship to #92, #107 and #145

This fixes [#92](https://github.com/Comfy-Org/comfy-kitchen/issues/92) on Apple GPUs. It is complementary to the two open fallback PRs and touches none of their files:
- [#145](https://github.com/Comfy-Org/comfy-kitchen/pull/145) is a portable eager float-GEMM fallback. It still matters wherever this backend cannot register (older torch without `compile_shader`) or declines a call, and it can land independently.
- [#107](https://github.com/Comfy-Org/comfy-kitchen/pull/107) adds an MPS float fallback and, in later commits, a Metal 4 kernel that quantizes activations to INT8 for M5-class GPUs. This backend is weight-only by design for M1–M4. An activation-quantized M5 path like #107's could live inside this backend later, selected by GPU family (see "What M5 would need" below). We have no M5 hardware to compare against.

### How tested

- `tests/test_mps_int8_linear.py` and `tests/test_mps_w4a8.py` compare against eager on the CPU:
  - every format and layout;
  - fp16, bf16 and fp32 inputs;
  - bias and residual order per format;
  - near-max, subnormal and zero rows;
  - exact residual cancellation;
  - K > 65504;
  - strided inputs and storage offsets;
  - empty and 1-D inputs;
  - shader index arithmetic past 2^32;
  - LoRA at rank tails and column tails, including cancelling and extreme-exponent LoRAs.
- `tests/test_constraints.py` covers the opt-in parameter check.
- `tests/test_mps_empty_linear.py` covers empty contractions and the output-dtype declaration; `tests/test_mps_linear_lifetime.py` covers chunk temporaries at M = 1 and M = 3.
- Full suite on the final head, M1 Max: 1060 passed, 3107 skipped (other backends). Ruff clean.

An October 9 independent re-review found a LoRA epilogue fast path that could flush a tiny base term or a near-cancelling sum, and an M = 1 copy that kept a chunk result alive. Both were reproduced on the M1 Max and fixed with tests that fail on the earlier code. Two further findings are deliberate differences from eager, now documented and pinned by tests: asymmetric W4A8 computes from the exact stored weight (eager rounds the weight to the input dtype first, so results can differ by that rounding), and dequantized values below the fp32 normal range flush to zero, as eager's own arithmetic does on Apple GPUs. Earlier, the code went through three rounds of independent review on this backend: the initial version, a re-review after hardening, and the LoRA arithmetic. A fourth round reviewed the ComfyUI side. The reviewer was a model from a different family than the author. Every finding was reproduced on the M1 Max and fixed with a test that fails on the earlier code, with one exception: a scale restore that overflows before a cancelling bias near 2e38. That case returns inf exactly as eager does and is pinned by a parity test.

### Limitations

- Weight-only: memory is saved and integer weights are decoded at about 3 to 5% over a resident fp16 GEMM (39.5 against 37.8, 104.7 against 101.4 and 107 against 103 ms after the October 3 hardening; the 4 to 9% figure in earlier drafts predates it), but the matmul itself is fp16. There is no speedup beyond fp16 dense; the 2× on the wide shape above is against `F.linear`'s transposed path, not against a bare GEMM.
- `quantize_w4a8_int8_weight` runs eager on the CPU (MPS has no fp8 arithmetic). Merging a LoRA into 4/6-bit weights is therefore slow (about 6.5 minutes for MiniMax H3) and lossy. The unmerged LoRA arguments exist so hosts can avoid that merge.
- Layers with K > 65504 compute in fp32 (slower, exact range).
- NVFP4, MXFP8, ConvRot W4A4 and GGUF are separate follow-up series, not part of this initial backend PR.
- Tested on an M1 Max only.
- No macOS CI runner upstream, so these tests skip in CI.

### Hardware tested

Mac Studio M1 Max (32-core GPU), 32 GB, macOS 15.7, PyTorch 2.14.1.

---

## Reviewer-facing notes

**Why weight-only.**
- Apple GPUs before M5 have no fast integer matrix path, and on torch 2.14 and earlier MPS has no `torch._int_mm`. Torch 2.15.0 adds one, as a generic tiled kernel on M1 to M4; it is unmeasured here (as of November 9, 2026).
- Quantizing activations to INT8 and multiplying in emulated integer arithmetic would be slower than the fp16 GEMM, and less accurate.
- Decoding the integer weight once per call costs about 3 to 5% over a resident fp16 weight (October 3 hardening table; 4 to 9% before hardening), at half (INT8) or under a third (W4A8) of the memory.

**Why fp16, not bf16.**
- On the M1 Max the bf16 GEMM is about 1.75x slower than fp16 (70.0 against 37.8 ms at 4608x6144x6144). The cause was not investigated.
- fp16 was also more accurate here: relative error 0.0004 against 0.0029.
- The cost of fp16 is its range, which the normalization below handles.

**Why power-of-two normalization.**
- fp16 overflows at 65504, and real activations get there. MiniMax H3's text conditioning reaches about 16,000, and the first version produced all-NaN latents.
- Dividing rows by an exact power of two is an exact scaling. It bounds every GEMM operand by 1 and every dot product by K, and the exponent comes back in integer arithmetic in the epilogue.
- Unlike a float scale it is exact, including for near-max values and zero rows. The fp16 cast after it keeps full precision only down to about rowmax*2^-14: smaller elements of a row become fp16 subnormals and lose bits, and the GEMM output carries fp16's 11 significand bits. For bf16 inputs the practical effect is negligible.

**What M5 would need.** M5-class Apple GPUs add matrix hardware with integer support, reachable through Metal's newer tensor APIs. An activation-quantized path there would need:
- per-row INT8 activation quantization (eager's `quantize_int8_rowwise` semantics) fused into the prologue;
- an int8 x int8 -> int32 GEMM through those APIs;
- dispatch by GPU family, keeping this weight-only path for M1 to M4.

None of this was attempted, because there was no M5 hardware to test on.

---

## CLA / DCO checklist

- [ ] Every commit carries a `Signed-off-by:` line for the PR author (DCO, required by CONTRIBUTING.md). True for all eight commits (the duplicated line on the lifetime commit was removed October 9); check with `git log --format='%h %(trailers:key=Signed-off-by)' be003b7..refresh/mps-backend`.
- [ ] Commit author and committer match the GitHub account opening the PR.
- [ ] After opening the PR, the CLA Assistant bot asks the PR author to sign. Reply on the PR with exactly: `I have read and agree to the Contributor License Agreement`.
- [ ] The commits carry `Agent-Provenance:` trailers naming the model that wrote the code. They are informational; keep them or drop them (the latter needs an interactive rebase, which is yours to run).
- [ ] The branch is rebased onto `be003b7`; fetch again before opening the PR in case `main` moved.
- [ ] The lifetime commit carries a `(cherry picked from commit 8e9ae45…)` line pointing at a private commit; drop it if you prefer.

## Suite per commit (October 4, historical)

`python -m pytest tests` in a worktree at each commit, M1 Max, under the shared GPU lock:

| Commit | Passed | Skipped | Wall time |
|---|---|---|---|
| `d285063` int8_linear backend | 754 | 3107 | 97 s |
| `7cc954a` W4A8/W6A8 linear | 795 | 3107 | 96 s |
| `79dd55c` dequantize / requantize | 841 | 3107 | 97 s |
| `788763a` fused prologue / epilogue | 918 | 3107 | 100 s |
| `74dc102` unmerged LoRA | 1028 | 3107 | 100 s |

The skips are other backends' tests (CUDA, HIP, Ascend, triton) that do not run on a Mac. The working branch passes 1049; the 21-test difference is the held-back autograd tests.
