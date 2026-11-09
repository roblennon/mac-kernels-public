# Weight error by quantization format, per layer (2026-11-09)

How far does each storage format move the weights of a real diffusion transformer? This page
measures that directly, layer by layer, on two public checkpoints. It measures weights only.
It says nothing about image quality, and a format with a larger weight error can still
produce the better image.

Raw data: [modal/results/2026-11-09-weight-error.json](../../modal/results/2026-11-09-weight-error.json)
(506 per-layer rows plus the summary every number below is copied from).

## What was run

- Script: `modal/weight_error_app.py`, on Modal CPU containers (4 CPU, 8 GiB, torch 2.15.0),
  one container job per transformer block, 64 jobs, 2229.3 s of job time in total.
- Checkpoints, both public and not gated, pinned by revision in the script:
  - Qwen-Image 2.1 transformer, `Comfy-Org/Qwen-Image-2.1`, bf16 file. 200 linear weights,
    7,115,112,448 parameters. Chosen because it is this repo's second headline model and the
    same repository ships the publisher's INT8 ConvRot file, which gives an outside check.
    Its licence tag on the hub is "other".
  - Wan 2.1 T2V 1.3B transformer, `Wan-AI/Wan2.1-T2V-1.3B`, fp32 file, Apache-2.0. 306 linear
    weights, 1,417,641,984 parameters. Chosen as a small second architecture.
- Formats:
  - INT8 per row: one fp32 scale per output row, no rotation.
  - INT8 ConvRot: the same after the Hadamard rotation over groups of 256 input columns.
    This is the layout the repo ships.
  - W4A8: 4-bit codes, ConvRot 256, groups of 16, Lloyd-Max codebook, e4m3 group scales.
  - W6A8: 6-bit uniform codes, ConvRot 256, groups of 16, searched e4m3 group scales.
  - GGUF Q8_0 and Q4_K: quantized by ggml 0.25.3 (`ggml_quantize_chunk`, no importance
    matrix), decoded with gguf-py 0.19.0.
- Metrics: relative Frobenius error `||W_hat - W|| / ||W||` and max-abs error, both in fp32
  against the original weights, in the original basis. W4A8 and W6A8 are decoded through the
  int8 grid the kernels use.

## Results

Relative Frobenius error per layer, in percent. "Bits" is stored bits per weight, counting
codes, scales and codebook.

Qwen-Image 2.1, all linear weights:

| Format | Layers | Bits | Median | p95 | Worst | Worst layer |
|---|---:|---:|---:|---:|---:|---|
| INT8 per row | 200 | 8.007 | 1.066 | 1.489 | 1.805 | `time_text_embed.timestep_embedder.linear_2` |
| INT8 ConvRot | 199 | 8.007 | 0.869 | 0.956 | 1.006 | `transformer_blocks.25.attn.to_out.0` |
| W6A8 | 199 | 6.507 | 2.064 | 2.083 | 2.093 | `transformer_blocks.23.attn.to_out.0` |
| W4A8 | 199 | 4.507 | 7.307 | 7.311 | 7.312 | `transformer_blocks.0.img_mlp.out` |
| GGUF Q8_0 | 200 | 8.500 | 0.564 | 0.629 | 0.827 | `modulation.1` |
| GGUF Q4_K | 199 | 4.500 | 7.392 | 7.783 | 8.839 | `proj_out` |

Wan 2.1 T2V 1.3B, all linear weights:

| Format | Layers | Bits | Median | p95 | Worst | Worst layer |
|---|---:|---:|---:|---:|---:|---|
| INT8 per row | 306 | 8.016 | 0.852 | 1.128 | 1.950 | `time_embedding.2` |
| INT8 ConvRot | 306 | 8.016 | 0.811 | 0.910 | 0.914 | `blocks.1.ffn.2` |
| W6A8 | 306 | 6.516 | 2.054 | 2.073 | 2.094 | `blocks.19.cross_attn.o` |
| W4A8 | 306 | 4.516 | 7.300 | 7.310 | 7.328 | `blocks.19.cross_attn.o` |
| GGUF Q8_0 | 306 | 8.500 | 0.541 | 0.565 | 0.929 | `time_embedding.2` |
| GGUF Q4_K | 306 | 4.500 | 7.175 | 7.398 | 10.311 | `time_embedding.2` |

Max-abs error per layer, in weight units:

| Model | Format | Median | p95 | Worst | Worst layer |
|---|---|---:|---:|---:|---|
| Qwen | INT8 per row | 0.00336 | 0.00692 | 0.01359 | `txt_in.out_layer` |
| Qwen | INT8 ConvRot | 0.00389 | 0.01011 | 0.02196 | `txt_in.out_layer` |
| Qwen | W6A8 | 0.00923 | 0.02022 | 0.05992 | `transformer_blocks.0.attn.to_out.0` |
| Qwen | W4A8 | 0.03278 | 0.07635 | 0.18700 | `transformer_blocks.0.attn.to_out.0` |
| Qwen | GGUF Q8_0 | 0.00331 | 0.00673 | 0.01349 | `txt_in.out_layer` |
| Qwen | GGUF Q4_K | 0.03763 | 0.07829 | 0.13102 | `txt_in.out_layer` |
| Wan | INT8 per row | 0.00108 | 0.00262 | 0.00855 | `blocks.22.cross_attn.o` |
| Wan | INT8 ConvRot | 0.00181 | 0.00490 | 0.01874 | `blocks.22.cross_attn.o` |
| Wan | W6A8 | 0.00426 | 0.01208 | 0.06040 | `blocks.22.cross_attn.o` |
| Wan | W4A8 | 0.01516 | 0.04822 | 0.18432 | `blocks.19.cross_attn.o` |
| Wan | GGUF Q8_0 | 0.00106 | 0.00260 | 0.00871 | `blocks.22.cross_attn.o` |
| Wan | GGUF Q4_K | 0.01307 | 0.03664 | 0.14540 | `blocks.22.cross_attn.o` |

## What it means

- The rotation helps the Frobenius error and mostly helps the tail. ConvRot is lower than
  plain per-row INT8 in 198 of 199 Qwen layers and 199 of 306 Wan layers. The median moves
  from 1.066 % to 0.869 % on Qwen and from 0.852 % to 0.811 % on Wan; the p95 moves from
  1.489 % to 0.956 % and from 1.128 % to 0.910 %.
- The rotation costs max-abs error. ConvRot has the lower max-abs error in only 63 of 199 Qwen
  layers and 12 of 306 Wan layers. A likely reason, not tested here, is that rotating back
  spreads each code's rounding error over 256 columns, so single weights can move further
  even while the layer as a whole moves less.
- W4A8 and GGUF Q4_K are level at about 4.5 bits. Median error is 7.307 % against 7.392 % on
  Qwen and 7.300 % against 7.175 % on Wan. W4A8 is the lower of the two in 152 of 199 Qwen
  layers and 53 of 306 Wan layers. Neither format wins on weight error.
- W4A8 and W6A8 have almost no spread between layers. The W4A8 worst layer is 7.312 % on Qwen
  and 7.328 % on Wan against medians of 7.307 % and 7.300 %. Q4_K has a tail: 8.839 % on Qwen
  and 10.311 % on Wan.
- Every layer took the fixed Lloyd-Max codebook (199 of 199 on Qwen, 306 of 306 on Wan). The
  fitted-codebook path never fired on these two checkpoints.
- GGUF Q8_0 has a lower error than INT8 ConvRot (0.564 % against 0.869 % on Qwen, 0.541 %
  against 0.811 % on Wan) and stores 8.5 bits per weight against 8.007 and 8.016. It keeps one
  scale per 32 values where INT8 keeps one per row.
- W6A8 sits at 2.06 % for 6.5 bits, between the two.

## Checks

Raw: [validation](../../modal/results/2026-11-09-weight-error-validation.json),
[container vectors](../../modal/results/2026-11-09-weight-error-selftest-container.json).

- The quantizers in the script are a pure-torch port of the comfy-kitchen eager reference.
  `modal/weight_error_validate.py` compares the port with the vendor checkout on three seeded
  CPU tensors (fixed codebook, fitted codebook, chunked rows). Every stored tensor and every
  dequantized weight is bit-identical for all four formats. This was shown on macOS only.
  The vendor checkout is private and was not run in the container, so the container run that
  produced the numbers is covered by the local-against-container check below, not by this one.
- The publisher's own INT8 ConvRot file agrees with ours. Across the 192 layers it quantizes,
  828 of 6,979,321,856 codes differ, each by 1, and the largest scale difference is 4.66e-10.
- ggml Q8_0 output is byte-identical to the gguf-py numpy quantizer on the test tensor, in the
  container and locally. The gguf-py decode is bit-identical to the repo's GGUF dequant for
  Q8_0 and Q4_K.
- Local (macOS arm64, torch 2.14.1, ggml 0.26.0) against the container (Linux x86-64, torch
  2.15.0, ggml 0.25.3), same code, same three tensors, inputs bit-identical:
  - Per-row INT8 and Q8_0 are bit-identical in every case.
  - INT8 ConvRot: all codes identical. One fp32 row scale of 96 differs on the heavy-tailed
    tensor (1 byte).
  - W4A8: the fp32 channel scales differ in 54 of 96, 66 of 96 and 2689 of 4200 rows. On the
    4200-row tensor, which takes the chunked path that real layers above 4,194,304 weights
    take, the stored codes differ in 2 of 4200 rows and the group scales in 1 row. On the heavy-tailed
    tensor the fitted codebook differs in 10 of 64 bytes; the codes there are identical.
  - W6A8: codes and group scales identical in every case; channel scales differ in 32 of 96,
    31 of 96 and 1720 of 4200 rows.
  - Q4_K differs in 19 and 46 of 55,296 bytes on the two small tensors and in 137 of 4200
    rows on the large one, with a different ggml version on each side.
  - Effect on the measured error: the largest shift in relative Frobenius error is in Q4_K,
    7.05484 % locally against 7.05445 % in the container (heavy-tailed tensor). For the
    torch formats the largest is W4A8 on the 4200-row tensor, 7.2936486 % against
    7.2936466 %. The Q4_K shift is 0.0004 percentage points, so the last printed digit of
    a Q4_K figure in the tables could differ by one on another machine or ggml version.
  - The likely cause for the torch formats is that the rotation is a matmul and the two BLAS
    builds sum in a different order. That was not isolated.

## Limits

- Weights only. No activations, no images, no end-to-end error.
- Q4_K was run without an importance matrix. Whether an importance matrix would change its
  numbers was not measured.
- Errors are measured against fp32 decodes. The Apple GPU kernels decode to half precision.
- Every format was quantized from fp32 weights, so the Hadamard rotation ran in fp32. The
  vendor rotation runs in the dtype of the weight it is given, and the repo's on-load W4A8
  and W6A8 paths can be handed bf16 weights. That case was not measured on the checkpoints.
  On the three seeded test tensors, a bf16 input through the vendor quantizer moves W6A8
  from 2.055 % to 2.061 %, 2.787 % to 2.793 % and 2.049 % to 2.056 %, and W4A8 from 7.292 %
  to 7.289 %, 7.297 % to 7.297 % and 7.294 % to 7.296 % (`bf16_input_vendor` in the
  validation file). INT8 is not affected: the convert scripts cast to fp32 first, and the
  publisher-file match above confirms it.
- Not measured: Qwen `img_in` (4096 x 64) for ConvRot, W4A8, W6A8 and Q4_K, because 64 input
  columns is below the 256-column group; the Wan `patch_embedding`, which is a 5-D convolution
  weight.
- Two checkpoints. The Krea 2 transformer was not run.

## Reproduce

    modal run modal/weight_error_app.py
    .venv/bin/python modal/weight_error_validate.py --container modal/results/2026-11-09-weight-error-selftest-container.json

The second command needs the private vendor checkout and, for the GGUF part, a local
`libggml-base` passed with `--ggml-lib`.
