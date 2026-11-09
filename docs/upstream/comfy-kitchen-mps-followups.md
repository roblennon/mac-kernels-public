# Kitchen follow-up PR descriptions

**October 9, 2026 (evening):** each branch is refreshed as `refresh/mps-<name>` on the reviewed backend [`refresh/mps-backend`](comfy-kitchen-mps-backend.md) (`8782535`), with its share of source repair `a138076` folded in, and each has had an independent review with every finding reproduced on the M1 Max and fixed behind failing-first tests. Heads: training `bfca919`, block formats `9666b73`, GGUF `b34f36e`. `refresh/mps-stack` (`5e7f6a4`) composes block formats and GGUF on one backend: the only overlap was the registration file, resolved by keeping both insertions, and a new test pins that every family registers on the one backend and dispatches to its own module (1475 passed). The three fixes are also in the runtime Kitchen (`9825e98`, 1706 passed). Public PRs have not been opened.

Review outcomes. Training: one blocking bug, the ComfyUI attention override folded the key sequence into the head axis under grouped-query attention (K/V with fewer heads than Q), returning values 1.38 off SDPA; fixed, plus empty-query backward and sub-16-row budgets. Block formats: no blocking bug; the NVFP4/MXFP8 tensor handlers accepted inputs that were neither the logical nor the padded width on MPS (fixed and tested on the real `F.linear` path), eager `out_features` is bounds-checked, the offset test now builds its views on the device, CPU-only contracts run without a GPU, and the MPS path is documented as inference-only. GGUF: no blocking bug; a wide layer with K not divisible by four (K = 12289) was refused outright and now runs through the row decode via a `transposed` override on the shared driver; transposed-decode tests cover every format including IQ4 with partial column tiles and odd windows. Still unverified: real hardware behaviour of empty attention in a full training step, and whether maintainers want the stack as one PR or two.

The October 4 packaging below is historical; earlier checks are in [refresh-verification.md](refresh-verification.md).

## Training

Branch `refresh/mps-training`, head `bfca919`, seven commits after the backend: the original four, the `a138076` backward/mask repairs, Ruff fixes, and the October 9 review fixes (GQA heads, empty queries, sub-16-row budgets; packed-gradient tests over three rotation groups). Historical: `upstream/mps-training` `d402268`.

**Title:** Add bounded input-gradient backward paths for frozen MPS quantized weights and attention

Frozen INT8, W4A8 and W6A8 weights can now participate in LoRA training without saving a full decoded weight for backward. The wrappers preserve the existing quantized forward and compute input gradients from bounded decoded chunks. They retain the resident packed tensors, normalize the gradient and weight before contraction, accumulate chunks in fp32, undo ConvRot where needed, and restore exponents before the final cast.

The attention wrapper recomputes bounded score tiles during backward rather than retaining the full attention matrix. It supports independently broadcast inputs and masks, with zero gradients for fully masked rows. Quantized weight parameters stay frozen; the wrappers do not implement gradients for the codes, scales, correction or bias.

Tests compare gradients against dense CPU references, cover masked/broadcast attention, extreme scales, chunk tails, saved-tensor bounds, checkpointing and LoRA updates. The refresh CPU subset passes 48 tests with 57 MPS cases skipped. Source implementation commits: `caed5e2`, `54f6900`, `bbe6851`, `56e8b96`.

Limits: the packed backward uses one layer-wide weight exponent, so very large scale differences can round small channels away in the normalized compute dtype. This is a frozen-base training primitive, not an end-user training application. The trainer, cache and resume machinery in mac-kernels are outside this PR.

## Block formats

Branch `refresh/mps-block-formats`, head `9666b73`, five commits after the backend: the original three, the `a138076` MXFP8 fp32 compute repair (with the shared driver's `compute_dtype` override), and the October 9 review fixes (logical width, eager bounds, device-side offset test, CPU-only test module, inference-only docstrings). Historical: `upstream/mps-block-formats` `354d0f5`.

**Title:** Run NVFP4, MXFP8 and ConvRot W4A4 weights natively on MPS

The existing weight-only driver now accepts NVFP4, MXFP8 and ConvRot W4A4 weights. A Metal decode normalizes packed values, the shared GEMM contracts unquantized activations, and the epilogue restores scale. NVFP4 and MXFP8 dequantization also run directly on MPS. Optional LoRA factors use the same side path as the original formats.

The public float-activation operations are `nvfp4_linear` and `mxfp8_linear`; their tensor layouts dispatch to them. The activation-quantized `scaled_mm_nvfp4` and `scaled_mm_mxfp8` operations are not registered on MPS. The weight-only contract keeps the checkpoint's decoded weight and original float activation, rather than emulating activation quantization.

The 150-case format suite covers CPU-reference dequantization, output dtypes, layout details, extreme scales and LoRA. These are source-branch results until the separated branch runs its GPU suite; the refresh CPU pass collected and skipped all 150 while the shared lifetime regressions passed. Source commits: `c384a2e`, `0639e76`.

Depends on backend. ComfyUI needs its companion format/encoder dispatch branch. Integration with the GGUF follow-up requires resolving the shared MPS operation-registration list and rerunning dispatch tests.

## GGUF

Branch `refresh/mps-gguf`, head `b34f36e`, four commits after the backend: the original two, Ruff fixes, and the October 9 odd-K fix with wider transposed-decode tests. `refresh/mps-stack` carries these on top of block formats. Historical: `upstream/mps-gguf` `9d91f8f`.

**Title:** Add MPS weight-only linears for GGUF block-quantized weights

`gguf_linear` consumes the packed GGML block bytes directly and decodes bounded normalized chunks for the existing range-safe GEMM driver. `dequantize_gguf` exposes the same decode. Supported types are Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL and IQ4_XS.

The decode handles rows that start inside a quantization block and supports wide-input transposed decode. Bias and optional unmerged LoRA retain the shared driver's order. Tests compare decoded bytes and linear output against `gguf.quants`, including extreme scales, straddling rows, chunking, width tails and LoRA.

The source series reports 177 GPU tests. They require the `gguf` reference package; the root development environment has it, while the training environment does not. No GGUF GPU result is claimed by the CPU-only refresh. Source commits: `29ea044`, `da65b20`.

Depends on backend. This provides kernel operations; a GGUF host still has to call them with the correct type and packed shape. It does not make every third-party GGUF loader dispatch automatically. Integration with block-formats has the registration-list conflict described above.
