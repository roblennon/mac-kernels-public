# Results

Curated 2026-11-09 from the dated log in [findings](findings.md). This page keeps only the numbers that stand on their own: a clean stock-versus-ours pair, or a format stock cannot run at all. Every row names the findings line it comes from. The full log holds the rest, with its noise and caveats. [REPRODUCING.md](REPRODUCING.md) has the commands.

All numbers are from one M1 Max with 32 GB, torch 2.14.1, one process per run. "Stock" is ComfyUI as shipped, which dequantizes quantized weights on every call on Apple GPUs. "Ours" is the same ComfyUI with the quantized layers routed to the comfy-kitchen mps backend. Each number is a median or a steady-state step from one process, not a multi-process distribution. Memory is `driver_allocated_memory` after sampling unless the row says peak.

## End to end, stock versus ours

| Model and shape | Stock | Ours | Speedup | Memory | Source |
| --- | --- | --- | --- | --- | --- |
| Krea 2 Turbo INT8 ConvRot, 512x512, 8 steps | 8.7 s/step | 3.65 to 3.69 s/step | 2.4x | not recorded at 512 | findings.md:36 (stock), :83 (ours, after hardening) |
| Qwen-Image 2.1 INT8 ConvRot, 1024x1024, 25 steps | 15.1 s/step | 8.8 s/step | 1.7x | 9.1 GB both | findings.md:95-96 |
| Qwen-Image 2.1 Q4_K_M GGUF, 512x512, 6 steps, Viggle adapter, sampling phase | 66.43 s | 39.56 s | 1.68x | 5.99 vs 4.99 GiB | findings.md:546 |
| Wan 2.2 5B Q4_K_M GGUF, 704x416x33, 20 steps, sampling phase | 200.71 s | 145.93 s | 1.38x | not recorded | findings.md:560 |

The GGUF rows compare against ComfyUI-GGUF's own dequantize-per-call path, run with the same captured conditioning and seed. Outputs are close but not identical (Qwen: 27.24 dB PSNR, 0.9499 SSIM; Wan: 24.10 dB, 0.9253; findings.md:546, :560). The two paths round differently: ComfyUI-GGUF computes block scales in fp16 and then casts the weight to bf16 (research/2026-10-03-gguf-on-metal.md:71). For Qwen the adapter is also applied differently, merged on every call in stock and kept beside the layer in ours (research/2026-10-03-gguf-on-metal.md:72, :97).

## Packed 6-bit and 4-bit formats

| Model and shape | Ours | Memory | Source |
| --- | --- | --- | --- |
| MiniMax H3 pruned FL2VA, W6A8 (6-bit), 480x288x22, 4 steps, no LoRA | 6.5 s/step | 17.2 GB | findings.md:63 |
| MiniMax H3 pruned FL2VA, W4A8, 480x288x22, 4 steps, Turbo LoRA on the side | 6.6 s/step | 15.7 GB peak | findings.md:471 |

Stock ComfyUI cannot run the W6A8 format on Apple GPUs at all (findings.md:63). I did not record a stock attempt with the W4A8 file.

## Ours only, no clean stock pair

These are the largest recipes that complete on this machine. I either did not run stock at the same size or the stock run was too noisy to quote (the Krea 1024 stock run happened during a download and still needs a rerun, findings.md:36, :45).

| Model and shape | Ours | Memory | Source |
| --- | --- | --- | --- |
| Krea 2 Turbo INT8 ConvRot, 1024x1024, 8 steps, SDPA | 14.91 s/step | 15.0 GB | findings.md:331 |
| Wan 2.2 5B Turbo INT8 ConvRot, 1280x704x81, 4 steps | 37.4 s/step | 9.1 GB sampling peak | findings.md:152 |
| Wan 2.2 A14B INT8 ConvRot + lightx2v LoRAs, 832x480x81, 4 steps | 216 s/step | 21.2 GB sampling peak | findings.md:153 |

## Kernel level

INT8 ConvRot linear, bf16 in and out, median ms, after hardening (findings.md:76-80). "Bare fp16 GEMM" is a resident fp16 weight through the plain matmul, the ceiling the kernel cannot beat.

| Shape (M, K, N) | Bare fp16 GEMM | Ours | Overhead | Memory for the weight |
| --- | --- | --- | --- | --- |
| 4608, 6144, 6144 | 37.8 | 39.5 | 4.5% | half of fp16 |
| 4608, 6144, 16384 | 101.4 | 104.7 | 3.3% | half of fp16 |
| 4608, 16384, 6144 | 103 | 107 | 3.9% | half of fp16 |

Against dequantize per call, the path a host takes without a kernel, the same shapes measured 76.4, 201.7 and 242.9 ms versus 39.3, 104.1 and 106.1 ms for the kernel, so 1.9x to 2.3x (upstream/ready/kitchen-pr-mps-backend.md:38-40). The first, pre-hardening table had the kernel at 43.0, 107.2 and 223.3 ms against 76.7, 206.7 and 261.0 ms for dequantize per call (findings.md:17-19).

W4A8 at 4608, 16384, 6144: 111 ms after hardening, down from 218 (findings.md:81).

GGUF decode on Metal is bit-exact against the `gguf` package's reference for twelve block types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K to Q6_K, IQ4_NL and IQ4_XS (vendor/ck-gguf/tests/test_mps_gguf.py:28) (research/2026-10-03-gguf-on-metal.md:7, :99-100; the IQ4 tests are recorded there as passed on 2026-11-09).

## Training

Frozen-weight INT8 backward on real Krea 2 Raw layers at 1056 tokens, median ms (findings.md:383-388):

| Layer (K to N) | Dense fp16 backward | Ours | Dequantize whole layer, then GEMM |
| --- | --- | --- | --- |
| attn.wq 6144 to 6144 | 9.0 | 9.7 | 12.2 |
| mlp.up 6144 to 16384 | 23.9 | 25.2 | 32.4 |
| mlp.down 16384 to 6144 | 24.2 | 25.9 | 32.7 |

The small attn.wk layer (6144 to 1536) is the exception: 3.2 ms against 2.3 dense (findings.md:386). Gradient error sits at the fp16 floor, 3.6e-4 relative, for every layer.

A rank-16 LoRA trained on the frozen Krea 2 Raw INT8 base at 512px ran at 12.2 s/step for most steps with a 15.9 GB peak, and the file loads in ComfyUI on both Raw and Turbo (findings.md:404-405, :366).

LoRA beside a quantized layer instead of merged into it: on real H3 W6A8 layers against the exact fp32 merge, the side path is off by 0.04% and the requantized merge by 2.8% (findings.md:136).

## VAE decode on Apple GPUs

Both changes are in the ComfyUI branch and apply to stock models too.

| Change | Before | After | Source |
| --- | --- | --- | --- |
| fp16 instead of bf16 for the Wan 2.2 VAE, 704x416x33 decode | 972 s | 74 s | findings.md:100-101 |
| Stacked-conv2d path for causal conv3d, Wan 2.2 VAE, 704x416x33 | 72.8 s | 21.1 s (66.5 dB PSNR vs before) | findings.md:209 |
| Same, 1280x704x81 | 542 s | 155 s | findings.md:210 |

## Not in this table, and why

- Krea 2 Turbo 1024, stock 35 s/step versus 16.5 s/step. The stock number is one run during a download, and the ours number is older than the 14.91 s/step above (findings.md:36, :45, :331). It waits for a clean four-way rerun.
- MiniMax H3 INT8, stock 38 to 74 s/step versus 8.4 to 11.8 s/step. Stock was swapping, so the ratio is a memory story, not a kernel story (findings.md:112-117).
- The EXL3 prototype, 2.4 to 4.4x slower than the installed ggml Metal kernels. It is an LLM-weight experiment, not a diffusion result; see [the EXL3 note](research/2026-10-25-exl3-prototype.md) (:7, :72).
