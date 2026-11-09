# comfy-kitchen PR: mps backend

> Correction, 2026-11-09: the "4 to 9%" decode overhead below predates the October 3 hardening. The current measurement is about 3 to 5% (findings.md, hardened backend table), and the "about 2x faster than a dense fp16 F.linear" on the wide shape holds only against F.linear's transposed-weight path; against a bare fp16 GEMM the kernel is about 4% slower there. The text below is kept as posted.

Head: `roblennon/comfy-kitchen` branch `mps-backend` (`650dd6d`, eight commits rebased onto main `13d3468`, October 13, 2026). Base: `Comfy-Org/comfy-kitchen` `main`. Every commit carries a DCO sign-off. Training, additional block formats and GGUF stay as separate follow-up branches.

## Title

Add an mps backend: weight-only INT8 and W4A8/W6A8 linears for Apple GPUs

## Body

### What

A new `mps` backend for Apple GPUs that implements:

- `int8_linear` (INT8, optionally ConvRot).
- `w4a8_int8_linear` (the `AsymW4A8Int8Layout` formats: 4-bit with or without a Lloyd-Max codebook, 6-bit, symmetric or asymmetric).
- `dequantize_w4a8_int8_weight` and `quantize_w4a8_int8_weight`, so hosts can merge LoRAs into those weights.

`int8_linear` and `w4a8_int8_linear` also accept an unmerged LoRA (`lora_down` [R, K], `lora_up` [N, R], `lora_scale`). These are opt-in parameters: the registry skips a backend that does not declare them, so existing backends and callers are unchanged. The backend registers only when MPS is available and `torch.mps.compile_shader` exists.

### Why

These formats have no fast path on a Mac today. MPS does not implement `torch._int_mm`, so ComfyUI disables the INT8 formats on Apple GPUs and dequantizes the weights in full on every call, and W4A8/W6A8 cannot run there at all (#92). On a 32 GB Mac that means MiniMax H3 INT8 swaps at 38 to 74 s per step, the 6-bit H3 file does not load, and Krea 2 and Qwen-Image 2.1 INT8 pay for a full bf16 dequantization per layer per call.

### Design

- **Weight-only.** Apple GPUs before M5 have no fast integer matrix path. Integer weights stay resident, a Metal kernel decodes them to fp16 in one pass, and the stock MPS fp16 GEMM runs. Activations are not quantized. ConvRot checkpoints store the weight already rotated, so the activation is rotated online (group-wise, cheap) rather than un-rotating the weight each call.
- **Range safety by construction.** A fused prologue divides each activation row by an exact power of two so its max after rotation is at most 1, and applies the rotation in the same pass. The decode divides each weight row the same way. Every dot product is then bounded by K, so the fp16 GEMM cannot overflow for K up to 65504; wider layers compute in fp32. A fused epilogue restores both exponents with integer exponent arithmetic on the raw bits (Apple GPUs flush fp32 subnormals in arithmetic), adds bias and residual in eager's order for each format, and writes bf16 bit-exactly with torch's cast. Power-of-two scaling changes no mantissa, so it is exact at the dtype limits: near-max values, subnormals and zero rows.
- **fp16, not bf16.** On the M1 Max the bf16 GEMM is about 1.75x slower than fp16 (70.0 against 37.8 ms at 4608x6144x6144) because bf16 is emulated, and fp16 was more accurate here (relative error 0.0004 against 0.0029). The normalization above pays for fp16's range.
- **Unmerged LoRA.** The down weight goes through the same prologue and rides as R extra rows of the first decoded weight chunk, so the main GEMM also yields the low-rank activation. An epilogue applies the up weight in fp32 on the simdgroup matrix units. The up weight is carried as mantissa plus exponent, and base and LoRA terms are summed in a common scaled domain, so extreme scales stay finite and a LoRA that cancels the weight gives exactly 0.

### Measurements

Mac Studio M1 Max 32 GB, macOS 15.7.9, PyTorch 2.14.1, October 9, 2026. bf16 in and out, median ms per call, relative L2 error against an fp32 linear over the dequantized weight in brackets.

| Shape (M, K, N) | dense bf16 `F.linear` | dense fp16 `F.linear` | dequantize per call (today) | mps backend |
|---|---|---|---|---|
| INT8 4608, 6144, 6144 | 68.3 (2.3e-3) | 38.5 (1.7e-3) | 76.4 (2.3e-3) | 39.3 (1.7e-3) |
| INT8 4608, 6144, 16384 | 183.0 | 102.7 | 201.7 | 104.1 |
| INT8 4608, 16384, 6144 | 228.1 | 204.5 | 242.9 | 106.1 |
| W4A8 codebook 4608, 16384, 6144 | | | 285.7, fp32 scales only | 110.4 |
| W6A8 4608, 16384, 6144 | | | 299.4, fp32 scales only | 110.8 |

The backend runs 1.9 to 2.3x faster than dequantizing per call, at fp16-dense speed and accuracy. On the wide-input shape it is also about 2x faster than a dense fp16 `F.linear`, because dense `F.linear` hits a slow MPS path for transposed weights at large K (reported separately to PyTorch), which the transposed decode avoids. Eager's W4A8/W6A8 dequantize cannot run on MPS with e4m3 scales (no fp8 arithmetic), so for those files there is no path today.

Unmerged LoRA, rank 64, fused overhead per call: +2.0 ms on 4608x6144x6144 (base 38.8 ms) and +3.2 ms on 4608x16384x6144 (base 105.5 ms).

End to end in ComfyUI, seconds per step, measured with an integrated runtime that routes these formats to the backend (that ComfyUI change is a separate PR):

| Model | Stock ComfyUI | With the mps backend |
|---|---|---|
| Krea 2 Turbo INT8 ConvRot, 512x512 | 8.7 | 3.7 |
| Qwen-Image 2.1 INT8 ConvRot, 1024x1024 | 15.1 | 8.8 |
| MiniMax H3 INT8, 480x288x22 frames | 38 to 74 (swapping) | 8.4 to 11.8 |
| MiniMax H3 W6A8, 480x288x22 frames | does not run | 6.5 |

### Relationship to #92, #107 and #145

This fixes #92 on Apple GPUs. It is complementary to the two open fallback PRs and touches none of their files:

- #145 is a portable eager float-GEMM fallback. It still matters wherever this backend cannot register (older torch without `compile_shader`) or declines a call, and it can land independently.
- #107 adds an MPS float fallback and, in later commits, a Metal 4 kernel that quantizes activations to INT8 for M5-class GPUs. This backend is weight-only by design for M1 to M4. An activation-quantized M5 path like #107's could live inside this backend later, selected by GPU family: per-row INT8 activation quantization fused into the prologue, an int8 x int8 to int32 GEMM through Metal's tensor APIs, and dispatch by GPU family. None of that was attempted here, because there was no M5 hardware to test on.

### How tested

- `tests/test_mps_int8_linear.py` and `tests/test_mps_w4a8.py` compare against eager on the CPU: every format and layout; fp16, bf16 and fp32 inputs; bias and residual order per format; near-max, subnormal and zero rows; exact residual cancellation; K > 65504; strided inputs and storage offsets; empty and 1-D inputs; shader index arithmetic past 2^32; LoRA at rank tails and column tails, including cancelling and extreme-exponent LoRAs.
- `tests/test_constraints.py` covers the opt-in parameter check. `tests/test_mps_empty_linear.py` covers empty contractions and the output-dtype declaration; `tests/test_mps_linear_lifetime.py` covers chunk temporaries at M = 1 and M = 3.
- Full suite on the rebased head, M1 Max: 1082 passed, 3161 skipped (other backends), 101 s. Ruff clean. There is no macOS CI runner upstream, so these tests skip in CI.

Two deliberate differences from eager are documented and pinned by tests: asymmetric W4A8 computes from the exact stored weight (eager rounds the weight to the input dtype first, so results can differ by that rounding), and dequantized values below the fp32 normal range flush to zero, as eager's own arithmetic does on Apple GPUs. A scale restore that overflows before a cancelling bias near 2e38 returns inf exactly as eager does and is pinned by a parity test.

### Limitations

- Weight-only: memory is saved and integer weights are decoded at about 4 to 9% over a resident fp16 GEMM, but the matmul itself is fp16. There is no speedup beyond fp16 dense.
- `quantize_w4a8_int8_weight` runs eager on the CPU (MPS has no fp8 arithmetic). Merging a LoRA into 4/6-bit weights is therefore slow (about 6.5 minutes for MiniMax H3) and lossy. The unmerged LoRA arguments exist so hosts can avoid that merge.
- Layers with K > 65504 compute in fp32 (slower, exact range).
- NVFP4, MXFP8, ConvRot W4A4 and GGUF are separate follow-up series, not part of this PR.
- Tested on an M1 Max only.

### Hardware tested

Mac Studio M1 Max (32-core GPU), 32 GB, macOS 15.7, PyTorch 2.14.1.
