# GGUF diffusion models on Apple GPUs

2026-10-03. Mac Studio M1 Max 32 GB, torch 2.14.1, gguf 0.19.0.

**Question.** Can the published GGUF quantizations of Wan 2.2, Qwen-Image 2.1, Krea 2 and MiniMax H3 run on Apple GPUs with the same weight-only technique as the INT formats: packed weights resident, one Metal pass decoding them to normalized fp16, the stock fp16 GEMM?

**Answer.** Yes. The decode is bit-exact with the `gguf` package's reference for all twelve block types the kernel supports, including IQ4_NL and IQ4_XS (added after the first write-up; their 24 bit-exact tests passed on 2026-11-09). The codebook types IQ1, IQ2 and IQ3 are not covered. The ComfyUI change is a small patch to ComfyUI-GGUF. Measurements: see "Measured" below.

Code:
- Kernel: comfy-kitchen branch `mps-gguf` (`comfy_kitchen/backends/mps/gguf.py`, `tests/test_mps_gguf.py`).
- Node patch: `vendor/ComfyUI-GGUF-fork`, branch `mps-gguf-kernel` (`mps.py`, two hooks in `ops.py` and `loader.py`).
- Main repo: `bench/gguf_headers.py`, `bench/bench_gguf_linear.py`, `bench/gguf_node.py`, `tests/test_gguf_node.py`.

## 1. Bit layouts

A GGML tensor is a flat run of fixed-size blocks over its row-major elements. Structs are from `ggml/src/ggml-common.h` (ggml 89c4413, the stable-diffusion.cpp submodule at 3f8527a): `block_q4_0` :194, `block_q5_0` :229, `block_q8_0` :251, `block_q2_K` :298, `block_q3_K` :315, `block_q4_K` :327, `block_q5_K` :344, `block_q6_K` :362. The formulas below are the order `gguf.quants` computes them in fp32 (`gguf/quants.py`); "j" is the value's index in its block.

| Type | Values / bytes | Bits per weight | Fields in order | Value |
|---|---|---|---|---|
| Q4_0 | 32 / 18 | 4.5 | f16 d, u8 qs[16] | d · (nib − 8); nib = low nibble of qs[j] for j < 16, high nibble of qs[j−16] after |
| Q4_1 | 32 / 20 | 5.0 | f16 d, f16 m, qs[16] | d · nib + m |
| Q5_0 | 32 / 22 | 5.5 | f16 d, u32 qh, qs[16] | d · ((nib \| bit j of qh << 4) − 16) |
| Q5_1 | 32 / 24 | 6.0 | f16 d, f16 m, u32 qh, qs[16] | d · q + m |
| Q8_0 | 32 / 34 | 8.5 | f16 d, i8 qs[32] | qs[j] · d |
| Q2_K | 256 / 84 | 2.625 | u8 scales[16], qs[64], f16 d, f16 dmin | (d · (sc & 15)) · q − dmin · (sc >> 4), per 16 values; q = 2 bits |
| Q3_K | 256 / 110 | 3.4375 | hmask[32], qs[64], scales[12], f16 d | (d · s6) · (q2 − 4 · (1 − hbit)); s6 a signed 6-bit scale per 16 values |
| Q4_K | 256 / 144 | 4.5 | f16 d, f16 dmin, scales[12], qs[128] | (d · sc) · q − dmin · m per 32 values; sc, m 6-bit |
| Q5_K | 256 / 176 | 5.5 | f16 d, f16 dmin, scales[12], qh[32], qs[128] | as Q4_K with a fifth bit from qh |
| Q6_K | 256 / 210 | 6.5625 | ql[128], qh[64], i8 scales[16], f16 d | (d · sc) · (q − 32) per 16 values; q = 4 + 2 bits |

The details that are easy to get wrong:

- **Q4_K/Q5_K scale packing.** Twelve bytes hold eight (scale, min) pairs of 6 bits. Sub-blocks 0-3 take the low 6 bits of bytes 0-3 (scales) and 4-7 (mins). Sub-blocks 4-7 take a nibble of bytes 8-11 plus the top 2 bits of bytes 0-3 (scales) or 4-7 (mins). See `ggml-quants.c:915-922`. The 128 nibble bytes are four 32-byte chunks: chunk c holds sub-block 2c in its low nibbles and 2c+1 in its high nibbles.
- **Q5_K high bits.** Bit s of `qh[j % 32]` is the fifth bit of value j in sub-block s.
- **Q6_K.** Each 128-value half h reads `ql[64h + r % 64] >> 4·(r / 64)` and `qh[32h + r % 32] >> 2·(r / 32)`, where r is the index inside the half.
- **Q3_K.** The 16 signed scales are 4 low bits (bytes 0-7, low nibbles first) plus 2 high bits (bytes 8-11, 2 bits per sub-block group of four), minus 32. The high-mask bit is inverted: the offset of 4 applies when the bit is 0.

**Row alignment.** ggml itself requires each row to be a whole number of blocks. The GGUF loader rejects `ne[0] % blck_size != 0` (`gguf.cpp:731-737`), the quantizers assert it (`ggml-quants.c:1391`), and both stable-diffusion.cpp (`model_loader.cpp:1591`) and ComfyUI-GGUF's llama-quantize patch (`tools/lcpp.patch`, lines 227-253) keep such tensors unquantized. ComfyUI-GGUF's own converter is the exception: with `shape_fix` set it reshapes tensors whose last dim is not a multiple of 256 to `(-1, 256)` and records the original shape under `comfy.gguf.orig_shape.<name>` (`tools/convert.py`, `handle_tensors`). The torch row is then not the GGML row, and blocks straddle torch rows. The kernel indexes by flat element, `block = (n·K + k) / QK`, so it handles both cases. The one real file I found with `orig_shape` entries (Abiray's H3 Q4_K_M) uses them only for F32 tensors.

## 2. What the published files contain

`bench/gguf_headers.py` reads headers over HTTP range requests, a few MB per file. 22 files were surveyed; the full output is in the session notes. Tensors counted are the 2-D matrices.

| File | Producer (arch field) | Linear-weight types |
|---|---|---|
| QuantStack Wan2.2-TI2V-5B Q4_K_M | llama-quantize + lcpp.patch (`wan`) | Q4_K 210 (q, k, o, ffn.0), Q6_K 90 (self/cross `v`, ffn.2). F16 modulation, embeddings, head |
| QuantStack Wan2.2-T2V-A14B HighNoise Q4_K_M | same | Q4_K 280, Q6_K 120, same rule |
| Wan 5B Q8_0 / Q5_K_S / Q4_0 | same | Q8_0 only / Q5_K only / Q4_0 plus Q4_1 for ffn.2 |
| Wan 5B Q3_K_M / Q2_K | same | Q3_K + Q4_K (v, ffn.2) + 2 Q5_K / Q2_K + Q3_K (v) |
| unsloth Qwen-Image-2.1 Q4_K_M (the file run here) | stable-diffusion.cpp (no arch) | Q4_K 163, Q5_K 28, Q8_0 4, Q6_K 1; BF16 img_in/txt_in |
| unsloth Qwen-Image-2.1 Q4_K_S / Q3_K_XL / Q2_K / Q5_K_M / Q6_K_XL | same | Per-layer "dynamic" mixes from Q2_K to Q8_0: Q3_K/Q4_K/Q5_K/Q6_K, then Q2_K everywhere, then Q5_K/Q6_K/Q8_0 |
| leejet Qwen-Image-2.1 Q4_K | same | Q4_K 192, BF16 for everything else |
| abenzerps Qwen-Image-2.1-UC Q4_K_M | llama-quantize (`qwen_image21`) | Q4_K 165, Q6_K 64 (v, mlp.out). Its keys (`img_mlp.gate_layer/proj`) differ from ComfyUI's `gate_up` |
| vantagewithai Krea-2-Turbo Q4_K_M / Q4_K_S | llama-quantize (`qwen_image`, a mislabel) | Q4_K 201, Q5_K 63 (3 blocks and text fusion) |
| realrebelai Krea-2-Turbo Q4_K_M | llama-quantize (`krea2`) | Q4_K only (262) |
| gguf-org krea2_turbo iq4_xs | gguf-org (`pig`) | IQ4_XS: decode covered since 2026-11-09 (kernel and bit-exact tests); this file has not been run end to end |
| unsloth / leejet MiniMax-H3 pruned Q4_K(_M) | stable-diffusion.cpp (no arch) | Q4_K 208 (qkv, out, fc1, fc2); F16 adaln, BF16 small tensors |
| unsloth H3 UD-Q2_K_XL / Q5_0 | same | Q2_K 169 + Q5_K 40 / Q5_0 210 |
| Abiray H3 pruned Q4_K_M | llama-quantize (`wan`, a mislabel) | Q4_K 208; 57 `orig_shape` entries, all F32 |

What this means:
- **Q4_K_M, as published, mixes types.** Q4_K carries most of each model. llama-quantize with ComfyUI-GGUF's patch bumps attention V and FFN-down to Q6_K under Q4_K_M and Q5_K_M (`lcpp.patch`, lines 155-225). unsloth's dynamic files vary the type per layer. stable-diffusion.cpp has no K_M recipe: one target type plus optional regex rules (`src/convert.cpp:32-53`).
- **Every type the target files use is covered** (Q2_K, Q3_K, Q4_0, Q4_1, Q5_0, Q5_K, Q6_K, Q8_0, and since 2026-11-09 IQ4_XS). An IQ2/IQ3 file would not be, but none of the surveyed files uses those types.
- **Two producers disagree on metadata.** stable-diffusion.cpp writes no `general.architecture`. Stock ComfyUI-GGUF then tries its own key-based detection, which knows neither Qwen-Image nor H3, and raises "This model is not currently supported". It also rejects arch strings it does not list (`krea2`, `qwen_image21`). Its loader would refuse unsloth's and leejet's Qwen-Image 2.1 and H3 files, and realrebelai's Krea 2 file. The fork lets ComfyUI's own detection decide instead, and it recognizes the unsloth Qwen file (QwenImage21) and the QuantStack Wan file (WAN22).

## 3. How the two loaders dequantize

**ComfyUI-GGUF (city96, 6ea2651, last push 2026-01-12).**
- **Loading.** `loader.py` `gguf_sd_loader` memory-maps each tensor's bytes as a uint8 `GGMLTensor` carrying `tensor_type` and the logical `tensor_shape`. `GGUFModelPatcher` moves them to the GPU still packed.
- **Every call.** `GGMLOps.Linear.forward_ggml_cast_weights` calls `cast_bias_weight`, then `get_weight`, then `dequant.dequantize_tensor`. That dequantizes the whole layer with generic torch ops (`dequant.py`, `dequantize_blocks_*`), casts it to the input dtype and runs `F.linear` (`ops.py`).
- **Precision of the default path.** With `dequant_dtype=None`, the block scales stay fp16 (`d.view(torch.float16).to(None)`), so `d · sc · q − dmin · m` is computed and rounded in fp16. On real Qwen-Image 2.1 Q4_K/Q5_K tensors that differs from the reference in 71% of values, about 8e-4 relative L2 (Q8_0 2e-4, Q6_K 2.6e-4). The fp32 option is exact. The bf16 cast after it, about 4e-3, dominates anyway.
- **LoRAs.** These are never merged into the stored weight. They ride on the tensor as `patches` and are applied by `comfy.lora.calculate_weight` to the freshly dequantized full weight **on every call** (`get_weight`). On Apple GPUs that adds a full-size merge per layer per step.
- **Open Mac issues.**
  - #395 (2025-12): on an M4 Pro, every K-quant fails to produce an image while Q4_0 and Q8_0 work. Not reproducible here: there is no M4 on hand.
  - #320: a memory leak on an M2 Mac Studio with QuantStack Wan 2.2.
  - #236: Q4_0 is slower than Q8_0 on MPS, which fits a per-call dequantize that costs more for more complex unpacking.

**stable-diffusion.cpp (3f8527a) and ggml Metal (89c4413).**
- **Loading.** `ModelLoader::init_from_gguf_file` reads metadata only (`src/model_io/gguf_io.cpp:42-114`). With mmap, tensors whose type and size match are zero-copied into backend buffers (`model_loader.cpp:985-1007`). Weights stay quantized, except that Conv2d weights are forced to F16 and norms to F32 at load (`ggml_block.hpp:513, 819, 919`). Linear layers fall back to F32 when `in_features % blck_size != 0` (:160-164).
- **Matrix products.** Above 8 columns they go to `kernel_mul_mm` (`src/ggml-metal/kernels/mul_mm.metal`). On pre-M5 GPUs each thread dequantizes a 4x4 piece with `dequantize_q4_K` (`kernels/dequantize.h:431`) and the others into a threadgroup tile. Every instantiation has both tile types as **half** (:853-877), so the f32 activations are rounded to fp16 too (:280). The products accumulate in `simdgroup_float8x8`.
- **No range protection.** Plain `mul_mm` has no amax, no per-row scaling and no clamp. Only the MoE `mul_mm_id` has a power-of-two activation scale, and only under `GGML_PREC_F32` (:417-492). On these GPUs an activation above 65504, such as H3's text conditioning at about 16,000 times a large weight row, can reach inf. M5-class GPUs take a tensor-API path with float activations (`ggml-metal-device.m:1141-1148`).

## 4. The Metal path here

`comfy_kitchen.gguf_linear(x, weight, qtype, shape, bias, out_dtype, lora_down, lora_up, lora_scale)` and `dequantize_gguf`, on the INT8 driver (`linear.weight_only_linear`).

**Decode.** One Metal kernel per call (plus a tiny exponent pass) turns rows of blocks into fp16:
- **Exactness.** Each value is computed in fp32 exactly as `gguf.quants` computes it, with `#pragma METAL fp contract(off)` so no multiply-add fuses. It is then divided by 2^e on raw bits and rounded once.
- **Row exponent.** e is the exponent of an upper bound on |w| over every block the row touches, read from block scales and mins only (for example Q4_K: max over sub-blocks of |d·sc|·15 + |dmin·m|). That gives |w'| ≤ 1, so the fp16 GEMM cannot overflow for K ≤ 65504.
- **Wide layers.** For K ≥ 12288 the decode writes the weight transposed, as the INT8 path does (Qwen-Image 2.1's `img_mlp.out` has K = 12288).
- **Kernel layout.** Four consecutive values per thread. A run of four starting at a multiple of 4 always shares a block and a sub-block, whatever K is.

**Range safety and LoRA** are inherited unchanged:
- the activation prologue normalizes rows by powers of two;
- the epilogue restores exponents on raw bits;
- bias joins the fp32 sum before the one rounding, matching `F.linear` over the dequantized weight (ComfyUI-GGUF's semantics; pinned by an exact-GEMM test);
- the unmerged LoRA rides in the epilogue.

**Tests.** comfy-kitchen `tests/test_mps_gguf.py` has 177 tests against `gguf.quants` on CPU (153 when this was first written; the IQ4_NL and IQ4_XS cases added 24):
- `dequantize_gguf` is bit-exact for all twelve types. Q8_0, Q4_K and Q6_K are also checked with fp16 scales of 0, ±65504 and subnormals.
- The normalized decode equals ldexp(reference, −e) rounded to fp16 or fp32, bit for bit. That covers block-aligned rows, rows smaller than a block, rows starting mid-block, K % 4 ≠ 0, and transposed decode into a strided buffer.
- The linear matches a float64 oracle across x dtypes, with chunked and transposed decode, H3-scale activations (|x| ≈ 3e4) against near-max scales, and LoRA.

The main repo's `tests/test_gguf_node.py` (25 tests as of 2026-11-09, 17 at first writing) runs the fork's Linear with the kernel on and off:
- the two outputs agree;
- the kernel is actually called;
- whole-layer LoRAs pass fused (bf16, rank ≤ 128) or beside the layer;
- row-slice LoRAs run beside the slice;
- a LoRA whose fp16 factors would overflow but whose merged delta is finite matches stock exactly, whole-layer and sliced, in fp16 and bf16, and leaves the patch tensors untouched;
- DoRA falls back to the stock merge;
- factors follow patch changes, because they are rebuilt on each call;
- unquantized (F16) weights take the stock path.

## 5. ComfyUI integration: why a patch to ComfyUI-GGUF

The choice was between a new minimal node package (`comfy_nodes/mps_gguf/`) and a patch to ComfyUI-GGUF. The patch is smaller and serves the people who already use these files:
- **Weights are already packed and resident.** ComfyUI-GGUF's `GGMLTensor` already keeps the raw blocks on the GPU with their type and logical shape. The kernel needs exactly that, so the swap is one hook in `GGMLOps.Linear.forward_ggml_cast_weights`: call `mps.linear`, and if it returns None take the old path.
- **LoRAs are already unmerged.** Its patcher already keeps LoRAs as `patches` on the tensor instead of merging. `mps.py` turns plain LoRAs into low-rank factors in fp32 on each call and drops them when the call returns. An earlier version cached the factors per layer; that was removed because GGUF's patcher accounts only for its packed parameters and source patches, so cached GPU factors would have outlived a partial offload or an unpatch. The factors are fp32 because casting them, or the down projection, to fp16 can overflow even when the merged low-rank delta is finite. Anything that is not a plain LoRA keeps the stock per-call merge, so nothing is dropped.
- **Explicit precision settings keep the stock arithmetic.** When the loader sets `dequant_dtype`, or sets `patch_dtype` on a layer that has patches, `mps.linear` returns None and the stock path runs. The normalized kernel and its fp32 factors cannot promise the rounding those settings ask for.
- **A new package would duplicate a lot.** It would need its own GGUF reader, arch detection, sd.cpp key handling, mmap release and patcher, all of which ComfyUI-GGUF has and maintains.

The cost: the repo is dormant (247 open issues, last push 2026-01-12), so an upstream PR may sit. The fork works as a drop-in replacement in `custom_nodes`. The patch is 3 files: `mps.py` (new, 148 lines), 4 lines in `ops.py` and 7 in `loader.py`. Its tests live in this repo. Non-Linear quantized tensors (rare in these files: the survey found none in block layers) keep the stock path.

## 6. Measured

Both runs are from October 4, 2026 on the M1 Max, through the GGUF loader node in the fork, kernel on ("Metal") against kernel off ("stock", the per-call dequantize). Same captured conditioning and seed in both modes. One run each, one process each, so these are single measurements with no variance estimate. The dated record is in `docs/findings.md`.

| Model | File | Recipe | Sampling, Metal | Sampling, stock | Ratio | Output agreement |
|---|---|---|---|---|---|---|
| Qwen-Image 2.1 | Q4_K_M, with the Viggle LoRA (195 adapter-bearing layers) | 512², 6 steps | 39.56 s | 66.43 s | 1.68x | RGB PSNR 27.24 dB, luma SSIM 0.9499, mean abs. channel error 5.896/255 |
| Wan 2.2 5B | Q4_K_M | 704×416×33, 20 steps, CFG 5, Euler/simple | 145.93 s | 200.71 s | 1.38x | four-frame strips 24.10 dB PSNR / 0.9253 luma SSIM; final latents rel. L2 0.105 |

Both pairs produced coherent outputs, and both processes exited with flat swap. Qwen's driver allocation after sampling was 4.99 GiB Metal against 5.99 GiB stock (snapshots, not peaks). Wan made 6,000 kernel calls against 6,000 stock calls.

The Metal and stock images are close but not the same. I do not attribute the gap to any one cause. Three things differ between the two runs: the stock path computes the block scales in fp16 and then casts the whole dequantized weight to the activation dtype (section 3 measured that cast at about 4e-3 relative on Qwen tensors, larger than the fp16-scale error), the kernel path instead rounds the exactly computed, normalized weight to fp16 once (section 4), and the LoRA is applied differently (stock merges it into the dequantized weight on every call; the kernel path adds the low-rank factors in the epilogue or beside the layer). No run separates these. The agreement numbers above say the two paths produce close pictures, not that one is closer to the reference.

Not measured: text encoding, a full server graph, per-step timings over more than one process, and anything on another Apple chip.

## 7. Gaps and next steps

- **IQ types.** IQ4_NL and IQ4_XS (a 16-entry table) are in the kernel and the fork now. On 2026-11-09 I ran their 24 tests in `test_mps_gguf.py` under the GPU lock: bit-exact decode, row and straddling-block decode in both compute dtypes, and the linear against the float reference in three activation dtypes. All 24 passed. No IQ4 file has been run end to end; the only one surveyed is gguf-org's Krea 2 Turbo. IQ2/IQ3 and IQ1 use larger codebooks and are still not covered; no surveyed file needs them.
- **M4 K-quant failure (#395)**: not reproducible without an M4. The kernel path does not use the torch ops involved, so it may sidestep the failure. Unverified.
- **Text encoders in GGUF** (e.g. unsloth's Qwen3-VL GGUFs) go through ComfyUI-GGUF's CLIP loader. The same Linear hook covers them, but this was not tested.
