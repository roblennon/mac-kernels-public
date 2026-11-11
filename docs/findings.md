# Findings

Machine: Mac Studio M1 Max 32 GB, macOS 15.7, torch 2.14.1.

This is a dated evidence log. Earlier “missing,” “not run” and branch-status statements describe their recording date. Headless stage times and calculated resident estimates do not establish repeated-server or template timing.

## 2026-10-03: Generation performance audit

The generation audit records twelve findings, dated research, bounded experiments, and independent probe review. Large-token INT8 remains within about 3% of resident fp16 GEMM. Direct loading was 2.66× faster for one warm 147 MiB weight, and a real W4A8 layer passed exact load/forward/offload/reload checks. Neither is a whole-model speedup claim. Memory-lifecycle defects and unproven repeated-server behavior now take priority. Today's full-model attempts stopped on pressure during loading; no training ran.

## 2026-10-02: INT8 ConvRot linear, microbenchmark

`bench/bench_int8_linear.py`. Median ms per call, relative error against a float32 CPU oracle.

| Shape (M, K, N) | bf16 dense | fp16 dense | ComfyUI-style dequant per call | Metal weight-only (ours) |
|---|---|---|---|---|
| 4608, 6144, 6144 | 70.0 | 39.4 | 76.7 | 43.0 |
| 4608, 6144, 16384 | 184.3 | 102.3 | 206.7 | 107.2 |
| 4608, 16384, 6144 | 234.0 | 207.5 | 261.0 | 223.3 |

- bf16 GEMM is about 1.75x slower than fp16 on this GPU. ComfyUI computes in bf16 on Apple GPUs.
- One-pass Metal decode of INT8 weights plus online activation rotation costs 4 to 9 percent over resident fp16 weights, at half the memory.
- fp16 compute is more accurate here than bf16 (relative error 0.0004 against 0.0029) and more accurate than true int8 activations.
- `torch._int_mm` is not implemented on MPS in torch 2.14.1 or earlier. PyTorch 2.15.0 (released 2026-10-28) added a generic tiled MPS `_int_mm` (PR #193153). That changes the baseline the weight-only kernel is compared against; see the 2026-11-09 entry.
- The wide-input case (K=16384) is anomalously slow in stock MPS matmul for both dtypes. Open question worth a kernel.
- fp16 GEMM runs near 9 TFLOPS at these shapes, close to the hardware limit. Beyond the dtype win, speed must come from fewer steps, caching, or attention.

Open risk: fp16 overflow inside real DiT activations. Unverified until end-to-end images are compared.

## 2026-10-02: Krea 2 Turbo INT8 ConvRot, end to end in ComfyUI

`bench/run_krea2.py`, 8 steps, euler, same seed. Seconds per step, steady state. Images match visually; no NaNs.

| Config | 512x512 | 1024x1024 | GPU memory after sampling (1024) |
|---|---|---|---|
| Stock ComfyUI (dequant per call, bf16) | 8.7 | about 35, noisy (29 to 50, measured during a download) | 21.7 GB |
| Metal weight-only INT8 | 4.65 | 24 | 19.4 GB |
| Metal weight-only INT8 + `--use-pytorch-cross-attention` | 4.6 | 16.5 | 15.0 GB |
| Metal + `--fp16-unet` | 4.7 | not run | |

- ComfyUI defaults to sub-quadratic attention on Apple GPUs. PyTorch SDPA is much faster at 1024 and uses less memory. This is a config default worth changing upstream.
- With both changes the step time sits at the fp16 GEMM estimate for the linear layers (about 16 s at 1024), so the model is now GEMM-bound on this GPU.
- fp16 compute inside the INT8 layers did not overflow on Krea 2 Turbo.
- Forcing the whole model to fp16 gave nothing extra at 512.
- Stock 1024 needs a clean rerun with no download in progress. Done 2026-11-09: 27.1 to 28.0 s/step over six clean runs; see the four-way sweep entry of 2026-11-09 at the end of this file.

## 2026-10-02: storage

- ASM external drive reads at 20 to 60 MB/s and takes seconds per small file. Do not import code from it.
- An external SSD reads at about 900 MB/s. Models live there. Code lives on the internal disk in `vendor/`.

## 2026-10-02: MiniMax H3 first clip on M1 Max 32 GB

`bench/run_h3.py`. Pruned FL2VA INT8 ConvRot (21 GB) + nvfp4 text encoder (15.7 GB, encoded first, then released) + 4-step Turbo LoRA + PyTorch attention, through the Metal backend.

- 480x288, 22 frames, 4 steps, with audio: correct clip. Steps 6.7 to 8.3 s each. GPU memory after sampling 21.6 GB.
- Text encoding 168 s (one time, cached to disk by prompt). Sampling wall time 105 to 270 s including model load. Video decode 55 to 85 s.
- First attempt produced all-NaN latents. Cause: H3 text conditioning reaches about 16,000 in magnitude and overflowed fp16 inside the weight-only kernel. Fix: divide each activation row by a power of two before the fp16 GEMM and multiply back in the output dtype. Exact scaling; the fp16 cast after it keeps full precision down to about rowmax*2^-14. Costs about 10 percent on bf16 inputs. Krea 2 never triggered it.
- Not yet run: stock ComfyUI baseline for H3, the 6-bit W6A8 file, larger sizes.

## 2026-10-02: H3 on 6-bit weights, LoRA merge, outside review

- W6A8 pruned H3 (16 GB) runs through the Metal kernel: 6.5 s/step at 480x288x22, 17.2 GB GPU memory without LoRA. Stock ComfyUI cannot run this format on Apple GPUs at all.
- With the 4-step Turbo LoRA the 6-bit file needed new dequantize and requantize functions for Apple GPUs. It now loads and renders correctly. Cost: the LoRA merge requantizes about 200 layers on CPU, about 6.5 minutes per load. Per-step time is unchanged.
- Merging a LoRA into 4-bit or 6-bit weights is lossy (about 3 percent relative error on random weights at 6-bit). A runtime low-rank side path would be exact and skip the merge time. Candidate next kernel feature.
- An independent review of the branch found seven findings, verdict ship-blocking. The main ones are range safety of the fp16 path (fp16 inputs skip scaling, dot products can still overflow, scale restore can overflow or underflow in fp16 output), bias added after rounding, and 32-bit index wrap in the grouped-format shader. Hardened in comfy-kitchen commit 9ce2bfd, see below.

## 2026-10-02: hardened backend (after review)

All seven review findings reproduced on the M1 Max and are fixed, each with a test that fails on the old code. 203 backend tests, full suite 906 passing.

Design: a fused Metal prologue normalizes each activation row by an exact power of two and applies the ConvRot rotation in one pass; the decode kernel normalizes each weight row the same way; the fp16 GEMM then cannot overflow for K up to 65504 (wider layers use fp32); a fused epilogue restores exponents, adds bias before rounding, applies the residual, and writes bf16 bit-exactly.

Per call, bf16 in and out, median ms:

| Shape (M, K, N) | Bare fp16 GEMM | Before hardening | After |
|---|---|---|---|
| INT8 4608, 6144, 6144 | 37.8 | 46.7 | 39.5 |
| INT8 4608, 6144, 16384 | 101.4 | 115.1 | 104.7 |
| INT8 4608, 16384, 6144 | 103 | 148.5 | 107 |
| W4A8 4608, 16384, 6144 | | 218 | 111 |

Krea 2 Turbo 512: 3.65 to 3.69 s/step (was 4.6), image statistics unchanged. Stock ComfyUI is 8.7. A clean rerun on 2026-11-09 measures 3.75 to 3.78 against 8.3 to 8.6; see the four-way sweep entry of 2026-11-09 at the end of this file.

Note: Apple GPUs flush fp32 subnormals to zero in shader arithmetic, so exponent handling is done on raw bits.

## 2026-10-02: Qwen-Image 2.1 first run

INT8 ConvRot DiT (7.3 GB) with the INT8 ConvRot text encoder (9.4 GB), Metal backend, 512x512: 2.4 s/step, 8.1 GB GPU memory, correct image. Text encoding 35 s. (Measured before hardening.)

## 2026-10-02: Qwen-Image 2.1 at 1024, Wan 2.2 5B first clip, VAE dtype

| Run | Seconds per step | GPU memory |
|---|---|---|
| Qwen-Image 2.1 INT8, 1024, stock ComfyUI | 15.1 | 9.1 GB |
| Qwen-Image 2.1 INT8, 1024, Metal backend | 8.8 | 9.1 GB |
| Wan 2.2 TI2V-5B Turbo INT8, 704x416, 33 frames, 4 steps, Metal | 3.3 | 6.1 GB |

- Wan 5B frames are clean with euler. No sign of the frame corruption reported with uni_pc on Apple GPUs.
- VAE decode dominated both: Wan decode 972 s, Qwen 1024 decode 75 s. Cause: ComfyUI picks bf16 for these VAEs and on this M1 Max bf16 convolutions run far slower than fp16 (not verified on other Apple GPUs). Measured conv3d at one Wan decoder shape: fp16 44 ms, fp32 235 ms, bf16 588 ms.
- With `--fp16-vae`: Wan decode 74 s (13x), Qwen decode 7.8 s (10x), identical frame statistics, no NaN. The ComfyUI branch now prefers fp16 for VAEs on Apple GPUs whenever the VAE lists fp16 as a working dtype.
- Whole Wan 5B clip (33 frames at 704x416) now takes about 2 minutes end to end on this machine.

## 2026-10-02: re-review fixes

comfy-kitchen commit c809b9a. Three of the four re-review findings fixed (asymmetric W4A8 bias precision, W4A8 channel-scale normalization order, bias rounding order now matches eager per format). The fourth, overflow during restore before a cancelling bias at values near 2e38, is matched to eager, which also returns inf there; a test pins that parity. 215 backend tests, suite 918 passing. Krea 512 unchanged at 3.66 to 3.69 s/step.

## 2026-10-02: H3 Metal vs stock, same settings

Pruned FL2VA INT8 ConvRot + 4-step Turbo LoRA, 480x288, 22 frames, 4 steps, quiet GPU.

| | Seconds per step | GPU memory after sampling |
|---|---|---|
| Stock ComfyUI | 38 to 74 | 24.2 GB |
| Metal backend | 8.4 to 11.8 | 21.6 GB |

About 5x. Stock's per-call dequantize of a 21 GB model pushes this machine into swapping, which is why its steps vary so much. H3 remains the one target that sits near this machine's memory limit; the 6-bit file (17 GB) is the comfortable choice here.

## 2026-10-02: Wan 2.2 T2V-A14B first clip

Both 14B experts as INT8 ConvRot (14.5 GB each), loaded one at a time, each with its 4-step lightx2v LoRA, euler, shift 5, cfg 1. 640x384, 33 frames, 4 steps (2 per expert): coherent clip, no NaN, 16.0 GB GPU memory after sampling. Steps about 34 s each; one step at 97 s absorbed the expert swap and LoRA merge. End to end about 5.6 minutes including a 38 s fp16 VAE decode.

All four target families now generate on this machine through the Metal backend: Krea 2, MiniMax H3, Qwen-Image 2.1, Wan 2.2 (5B and 14B).

## 2026-10-02: LoRA beside quantized layers instead of merged (approach A, ComfyUI patcher)

vendor/ComfyUI a7c36b2. On Apple GPUs ModelPatcher keeps a plain LoRA on an INT8/W4A8/W6A8 linear as low-rank factors and the layer adds `up(down(x))` to its output; the integer weight is never touched. `COMFY_MPS_LORA_SIDE_PATH=0` merges as before. Unit tests: tests/test_lora_sidepath.py (13, MPS).

H3 pruned FL2VA W6A8 + 4-step Turbo LoRA, 480x288, 22 frames, 4 steps, all 208 LoRA layers on the side:

| | Sample stage | Seconds per step | GPU memory after sampling | Peak footprint |
|---|---|---|---|---|
| Merge (rerun) | 496 s | 6.08 to 6.09 | 17.63 GB | 20.33 GB |
| Side path | 127 s | 6.63 to 6.64 | 18.14 GB | 20.17 GB |

The side term costs about 9% per step (qkv rank 384, roughly the FLOP ratio) and saves about 370 s of load. Outputs differ (sheet PSNR 17.8 dB, video latent rel-L2 0.38, against 1.02 for LoRA vs no LoRA). On real layers (blocks.10.mlp.fc2, blocks.30.attn.out_proj) against the exact fp32 merge, the side path is off by 0.04% of the output and the merge path by 2.8%: requantizing the merged 6-bit weight adds noise about 30 times larger than this LoRA's own contribution. The side path is the faithful one.

Krea 2 Turbo INT8 ConvRot + darkbrush LoRA, 512x512, 8 steps, SDPA: all 224 LoRA layers on the side, 3.78 s per step vs 3.70 merged (+2%); the INT8 merge is quick on this model, so load time is a wash and sat inside run-to-run disk noise (sample stage 96 and 125 s side, 84 s merge). Images match closely (PSNR 26.5 dB side vs merge, 9.2 dB vs no LoRA).

## 2026-10-02: Wan 2.2 on Apple GPUs: uni_pc fix, standard sizes, attention default

All runs M1 Max 32 GB, torch 2.14.1, macOS 15.7, seed 7, the subway-musician prompt, euler unless noted. Runs used a clean ComfyUI worktree (branch HEAD plus only these changes) so the LoRA work in progress did not leak in.

**uni_pc on MPS (ComfyUI #16573).** Reproduced. Wan 2.2 TI2V-5B fp16 (Comfy-Org repackage), 704x416x33, 20 steps, cfg 5, shift 8: unfixed uni_pc is smeared, blocky and green from frame 0 on, latent std 1.67 against 0.97 for euler; no NaN, so the frame statistics alone do not flag it. Cause: the phi_k recurrence in `multistep_uni_pc_bh_update` cancels catastrophically at the tiny early steps of a shift-8 schedule, and MPS `expm1` is less exact than CPU. Through the real sampler with a dummy model (`bench/unipc_coeffs.py`), the right-hand side `b` was up to 128x off on MPS and 1.1x off on CPU float32, giving corrector weights up to 21 where float64 gives 1.84.

Fix (ComfyUI 63e446e): compute `expm1`, B(h) and the recurrence in float64 on the CPU, move the few scalars back; the solve and tensor math stay on the device. After it, `b` matches float64 to 1e-7 and the weights to 2e-5 on both CPU and MPS, both bh1 and bh2. The fixed clip is clean and coherent, at the same 6.83 s per step. A unit test fails on all four device/variant cases without the fix. Upstream PR #16574 converts `b` only; ours also takes h_phi_1 and B(h) from the same float64 values.

**Standard sizes.**

| Run | s/step | Sampling peak | Decode | Decode peak | Fits 32 GB |
|---|---|---|---|---|---|
| 5B Turbo INT8, 1280x704x81, 4 steps | 37.4 | 9.1 GB | 542 s | 21.5 GB | yes; swap +1.8 GB during decode |
| 14B INT8 + lightx2v LoRAs, 832x480x81, 4 steps | 216 (281 with expert swap) | 21.2 GB | 142 s (Wan 2.1 VAE) | 21.2 GB | yes; about 19 min end to end |

Both clips clean. At 1280x704x81 the Wan 2.2 VAE decode is 78% of wall time and scales linearly with frames times pixels (73 s at 704x416x33), so it is compute-bound, not swapping. Tiled decode (VAEDecodeTiled, overlap 64, temporal overlap 8) of the same latent:

| Tiles | Decode | Peak | PSNR vs full |
|---|---|---|---|
| full | 542 s | 21.5 GB | n/a |
| 512, 64 frames | 723 s | 8.2 GB | 41.5 dB |
| 768, 64 frames | 624 s | 13.5 GB | 41.7 dB |
| 512, 32 frames | 813 s | 8.2 GB | 36.8 dB |

Tiling buys memory, not speed. Use it only when the full decode would swap (larger sizes or a busy machine); 768 is the better trade.

**Attention default.** ComfyUI picks sub-quadratic attention on MPS unless `--use-pytorch-cross-attention`.

| Wan 5B Turbo | SDPA s/step, peak | sub-quad s/step, peak |
|---|---|---|
| 704x416x33 | 3.33, 6.1 GB | 3.61, 8.3 GB |
| 1280x704x81 | 37.4, 9.1 GB | 67.4, 17.2 GB (swap +4.3 GB) |

Output at 704: not bit-equal (sub-quad upcasts to fp32), relative L2 0.059, cosine 0.998, frame stats equal to three decimals. Decode unchanged (VAE attention: 72.8 s SDPA, 74.4 s split). Single self-attention calls (`bench/sdpa_mps.py`, fp16, head dim 128):

| Shape (scores) | SDPA | sub-quad |
|---|---|---|
| 5B 1280x704x81 (8.2e9) | 0.57 s, 1.0 GB | 1.56 s, 6.4 GB |
| 14B 640x384x33 (3.0e9) | 0.21 s, 1.0 GB | 0.80 s, 10.3 GB |
| 14B 832x480x81 (4.3e10) | 2.97 s, 3.0 GB | 11.9 s, 11.0 GB |

SDPA matches a query-chunked SDPA reference exactly at every size, including far above 2^31 scores: torch 2.13+ flash-style MPS kernels never materialize the score matrix. fp16 SDPA stays within 3e-3 of float64 at head dims 40 to 160 with logits scaled up to 32x, no NaN.

Change (ComfyUI 89be84a): pytorch attention by default on MPS when torch >= 2.13, same rule as NVIDIA and XPU; `--use-split-cross-attention` or `--use-quad-cross-attention` opt out. Risks: `attention_pytorch` ignores the macOS >= 14.5 fp16 upcast (not reproduced at kernel level here; macOS 26 untested); head dims outside 32/64/72/80/96/128/256 or unusual masks fall back to an MPSGraph path that materializes scores and is known to corrupt above about 2^32 elements (pytorch #179352), which matters for SD1.5-style head dim 40 at very high resolution. The chunking guard in ComfyUI #14838 covers that case and pairs with this change.

**Other Wan paths.** Image-to-video encode (Wan22ImageToVideoLatent with a start image) 1.5 s, steps 3.38 s against 3.33 s for text-to-video. RoPE tables are built on the CPU each forward (MPS has no float64): 20 ms at 18.5k tokens, negligible. The 14B step cost at 832x480x81 is about half attention (40 layers at 3 s).

## 2026-10-02: SDPA chunk guard, fp32 safeguard, Wan VAE decode 3.5x

**Chunk guard (ComfyUI cf1c429).** `attention_pytorch` on MPS splits a call over batch, heads, then queries whenever the score tensor would exceed 2^31 elements (idea from #14838). Cost: none measurable. One layer at 5B 1280x704x81, 4 head groups: 0.571 vs 0.568 s. At 14B 832x480x81, 20 groups: 2.98 vs 2.95 s. Outputs bit-identical. Wan 5B 704x416x33 (no split): 3.32/3.32 s/step before, 3.32/3.33 after. Tests force the split with tiny thresholds across mask shapes and grouped-query heads.

**fp32 safeguard.** On MPS, SDPA now upcasts when the model asks for fp32 attention (`attn_precision`, e.g. SD2) or `--force-upcast-attention` is set, as sub-quad did. It skips the blanket macOS >= 14.5 fp16 upcast, for two reasons:
- **No precision gain.** fp16 SDPA output was bit-identical to fp32-upcast output on 84 of 84 probed real Wan 5B attention calls: self- and cross-attention, at 704 and at 1280, including the split path. On random data it was identical at every head dim from 16 to 256 except 160 without a mask, where both versions differ from float64 by similar amounts (2e-3).
- **Double the cost.** The upcast takes 2.2x the attention time: 0.57 vs 1.28 s per 1280 layer.

**Decode profile, Wan 2.2 VAE, 1280x704, fp16.** Conv3d is 86% of decoder time, and MPS runs it at about 2 TFLOPS against 6 for conv2d at similar sizes. RMS_norm comes next at 6%; upsampling and DupUp3D are small. Single conv at decoder shapes (`bench/conv3d_mps.py`):

| Variant | 256ch, 6x352x640 | 1024->512ch, 6x176x320 |
|---|---|---|
| torch conv3d | 1.35 s | 2.74 s |
| channels_last_3d | 1.36 s | 2.72 s |
| per-tap conv2d, summed | 1.82 s | 3.81 s |
| im2col + fp16 GEMM, best chunk | 0.46 s | 0.80 s |
| taps stacked on channels, one conv2d | **0.30 s** | **0.61 s** |

ComfyUI f656f4d makes the stacked conv2d the MPS path of `CausalConv3d`, one output frame at a time. It serves both Wan VAEs.

| Decode | Before | After | PSNR vs before | Peak |
|---|---|---|---|---|
| 5B 704x416x33 (Wan 2.2 VAE) | 72.8 s | 21.1 s | 66.5 dB | 7.2 -> 6.2 GB |
| 5B 1280x704x81 | 542 s | 155 s | n/a | 21.5 -> 18.5-19.5 GB |
| 5B 1280x704x81, tiled 768 | 624 s | 173 s | 41.6 dB vs full | 12.3 GB |
| 5B 1280x704x81, tiled 512 | 723 s | 204 s | 41.3 dB vs full | 7.1 GB |
| 14B 832x480x81 (Wan 2.1 VAE) | 139 s | 43.8 s | 67.6 dB | 6.5 -> 7.5 GB |

The 1280 clip end to end is now about 5.5 minutes (2.5 sampling, 2.6 decode) instead of 11.5. Several latent frames per decoder call (the causal cache allows it, output identical) made it worse: 2 frames per call took 217 s against 47 s for 7 latent frames, at 26 GB, swapping. RMS_norm rewrites gained at most 6% and were not kept.

**Tiny VAE draft option.** `lighttaew2_2` and `taew2_2` from lightx2v/Autoencoders sit in the wan vae folder; ComfyUI loads them from `vae_approx` by name.

| Tiny decode | 704x416x33 | 1280x704x81 | PSNR vs full VAE |
|---|---|---|---|
| lighttaew2_2 | 0.9 s, 1.1 GB | 2.2 s, 1.1 GB | 25.0 dB at 704, 28.6 dB at 1280 |
| taew2_2 | 0.8 s, 1.1 GB | n/a | 19.5 dB at 704 |

lighttaew2_2 keeps color and composition; it is softer and smears fine detail in later frames. Good for previews and drafts. taew2_2 is visibly blotchy.

## 2026-10-02: conv3d path generalized; H3 decode; tiny VAE discoverability

**General conv3d path (ComfyUI 75088b9).** The stacked-tap conv2d path moved from the Wan `CausalConv3d` into `comfy.ops` `Conv3d._conv_forward`. The `ops.py` edit is confined to the Conv3d class. The path now covers every `ops.Conv3d` on MPS with any kernel, stride and numeric zero padding; a kernel one frame deep runs as one batched conv2d. groups, dilation, string padding and non-zero padding modes fall back to torch.

Configurations the models use, each tested against CPU fp32 torch conv3d:
- **Wan VAEs:** 3x3x3 after causal padding, 1x1x1, 3x1x1 with time stride 1 and 2.
- **MiniMax H3 VAE:** 3x3x3 pre-padded, 3x3x3 with space stride 2 and with time and space stride 2.
- **Wan DiT patch embedding:** 1x2x2 with stride 1x2x2.
- **Single-frame `causal_zero`:** both VAEs.

Errors: at most 2.4e-6 in fp32 (tolerance 1e-5) and 6.1e-4 of output scale in fp16 (tolerance 2e-3). Wan 2.2 704x416x33 decode is unchanged from the Wan-only path: 21.2 s, 66.5 dB vs conv3d.

**MiniMax H3 (480x288x22 latent, decode alone, quiet GPU).**

| | Before (620bacf) | After | Note |
|---|---|---|---|
| Video decode | 7.33 s, 6.1 GB | 7.35 s, 6.1 GB | bit-identical |
| Encode of the 22-frame clip | 49.4 s | 16.8 s | conv3d path; peak 36 GB |

The H3 decoder is a 36-layer ViT with no conv3d: linear layers are 7.1 s of a 9.1 s hooked decode, attention 3.2 s. The 38-142 s decode times in the earlier H3 logs were the post-sampling memory state (22 GB DiT plus LoRA, swap), not decoder compute: the same decode alone takes 7.3 s. The H3 encoder is conv3d and gains 2.9x. It matters for first/last-frame conditioning, which encodes single frames. The 36 GB encode peak on a full clip is an open issue (ComfyUI encodes the clip in one batch).

**DiT conv3d.** The Wan 2.1/2.2 DiTs patch-embed with one Conv3d (1x2x2): 57.6 -> 13.6 ms (5B, 1280x704x81) and 86.5 -> 17.9 ms (14B, 832x480x81), bit-identical. That is once per forward, about 0.1% of a step, so it does not matter. The H3 DiT patchifies with a reshape and has no conv3d.

**Tiny VAEs.** The runners register the wan vae folder as `vae_approx`, and `VAE_NAME=lighttaew2_2.safetensors` decodes through the normal VAELoader: 704x416x33 in 1.37 s inside the runner. In the ComfyUI UI, put the file in `models/vae_approx` under its original name (the loader recognizes the `lighttaew2_2` / `lighttaew2_1` / `taehv` prefixes) and pick it in the standard **Load VAE** node; it feeds VAEDecode like the full VAE.

## 2026-10-02: LoRA side path, A and B combined (fused kernel where it pays)

vendor/ComfyUI 7a67766 on comfy-kitchen aa3b8d5. ComfyUI's side path now passes a layer's whole-layer LoRAs, stacked along the rank, to `int8_linear` / `w4a8_int8_linear` as `lora_down`/`lora_up`, from both the Linear weight-only path and `linear_input_act`. It fuses only with bf16 activations (both models' compute dtype here), summed rank up to 128, and no slice offsets. Otherwise, or when the kernel declines, up(down(x)) runs beside the layer.

Policy from bench/bench_lora_fused_rank.py (extra ms over the base layer, M1 Max):
- bf16, H3 and Krea shapes: fused and separate within about 1 ms up to rank 128; fused behind from 192 (H3 qkv at 384: +18.1 vs +15.7 INT8, +17.1 vs +14.5 W6A8).
- bf16, 4608x6144x6144 rank 64: fused +1.4 vs +2.6 (INT8), +1.7 vs +2.4 (W6A8), matching B's numbers.
- fp16: separate wins at most shapes, by 0.5 to 2 ms.
- q/k/v row slices, 3 x rank 64, bf16: per-slice adds +6.1, zero-padded fused stack +9.4 (INT8); +2.9 vs +6.2 (W6A8).

End to end:

| | Layers fused / beside | Seconds per step | Sample stage |
|---|---|---|---|
| H3 W6A8 + Turbo, A (all beside) | 0 / 208 | 6.63 to 6.64 | 127 s |
| H3 W6A8 + Turbo, combined | 156 / 52 (qkv rank 384 beside) | 6.58 to 6.60 | 136 s |
| H3 W6A8 + Turbo, B (all fused) | 208 / 0 | 7.3 to 13.2 | 115 s |
| Krea 2 + darkbrush 512, A | 0 / 224 | 3.78 | 96 and 125 s |
| Krea 2 + darkbrush 512, combined | 224 / 0 | 3.76 to 3.79 | 114 s |
| Krea 2 + darkbrush 512, merged | n/a | 3.70 | 84 s |

The fused epilogue is a wash at these shapes and ranks: under 1% per step on H3, nothing measurable on Krea. Sample-stage totals swing about 30 s run to run on load from the external disk. On a real fused layer (H3 blocks.10.mlp.fc2, rank 128, bf16) fused and beside are both within 0.3% of the exact fp32 merge, which is bf16 output rounding. The H3 sheet differs from A's by 21 dB PSNR, from 4-step amplification of rounding; it is 18.7 dB from the merged sheet (A: 17.8). Krea combined vs merged: 26.6 dB (A: 26.5).

## 2026-10-02: attention efficiency study (measurement only, nothing landed)

Real SDPA calls of one sampling evaluation per model, captured by `bench/capture_attn.py`. Every model uses head_dim 128; the only mask is Qwen's 45-token text self-attention. Each call was replayed with its captured strides by `bench/attn_study.py`. Effective TFLOPS counts both matmuls: 4·B·H·Nq·Nk·d per second. The plain fp16 GEMM ceiling measured in the same session is 9.29 TFLOPS (8192³).

| Call (count per evaluation) | dtype | Time | SDPA TFLOPS | Contiguous | Other dtype | Manual fp16 |
|---|---|---|---|---|---|---|
| Wan 5B 1280x704x81 self, 6 heads x 18480 (120) | fp16 | 146 ms | 7.19 | 7.27 | bf16 5.27 | 4.62 |
| Wan 5B cross, 24 heads x 18480 x 512 (30) | fp16 | 16 ms | 7.23 | 7.23 | bf16 5.25 | 4.55 |
| Wan 14B 832x480x81 self, 2 heads x 32760 (800) | fp16 | 152 ms | 7.23 | 7.24 | bf16 5.25 | 3.91 |
| Wan 14B cross, 40 heads x 32760 x 512 (40) | fp16 | 52 ms | 6.59 | 7.26 | bf16 5.28 | 4.55 |
| Krea 2 1024, 48 heads x 4142 (28) | bf16 | 80 ms | 5.28 | 5.26 | fp16 7.28 | 5.14 |
| Qwen-Image 2.1 1024, 32 heads x 4096 x 4141 (32) | bf16 | 56 ms | 4.94 | 5.21 | fp16 7.27 | 5.01 |
| Qwen text self-attention, 32 heads x 45, bool mask (32) | bf16 | 0.19 ms | 0.17 (0.46 without the mask) | | | |
| H3 480x288x22, 56 heads x 1064 (50) | bf16 | 7 ms | 4.67 | 4.92 | fp16 6.96 | 4.66 |

Wan self-attention appears as 6-head and 2-head calls because the 2^31 guard splits the heads; unsplit calls run at the same rate.

**Diagnosis.** SDPA in fp16 at head_dim 128 runs at 7.2 TFLOPS, 78% of the GEMM ceiling. It scales purely quadratically: 7.05-7.34 TFLOPS from 1k to 32k tokens. Head dims 64, 80 and 96 run equally fast; 160 (2.55) and 256 (3.09) are slow, but no model uses them. A bare QK^T matmul at K=128 reaches only 6.7-8.0 TFLOPS, so the fused kernel is already at matmul speed for this inner dimension. Torch's fused kernel is selected: a manual fp16 attention (QK^T, softmax, PV, chunked over queries) is 1.6-1.8x slower, with outputs matching at cosine > 0.99999. Strides as ComfyUI passes them cost nothing, except about 10% on Wan 14B cross-attention, which is under 2% of attention time. The small mask matters only on a 0.2 ms call.

**Wan 14B: nothing to gain.** Attention is 124 s of a 216 s step: 800 self-attention calls x 152 ms plus 40 cross-attention calls x 52 ms. A kernel at the full 9.29 TFLOPS ceiling would save at most 27 s (12%). Anything built from MPS GEMMs plus a softmax pass is bounded by the 6.7-8.0 TFLOPS matmul rate at K=128 and lands below SDPA, as the manual version shows. No prototype was built.

**The one real gap is bf16.** Krea 2, Qwen-Image 2.1 and H3 run attention in bf16 at 4.7-5.3 TFLOPS; the same calls in fp16 run at 7.0-7.3. A test wrapper (`bench/attn_fp16_wrap.py`) computed large bf16 calls in fp16 with an amax guard against overflow. Krea 2 1024: 15.02-15.06 s/step in bf16 vs 14.49-14.68 s/step in fp16 (two runs each, 2.5-3.5%), final latent cosine 0.996. Attention is only 2.2 s of the 15 s step, and the casts plus the guard's sync eat part of the 1.4x. Not landed: the gain is small and the output is not identical.

## 2026-10-02: step-reduction LoRAs end to end; best times per model

All on M1 Max 32 GB, ComfyUI fork 7a67766 (LoRA side path, SDPA default, conv3d path), torch 2.14.1, one process per image. "Cold" includes text encoding and the first-step weight load from the external disk. That load is about 45 s for Qwen and about 97 s for Krea, and a running ComfyUI server pays it once.

**LoRA side path.** Every LoRA layer on a quantized linear stayed beside it; nothing fell back to merging. Counts come from `bench/lora_audit.py`, checked after sampling, and the fork's log line:

| LoRA | Side path (fused-kernel stacked) | Merged into a quantized layer | Merged into a non-quantized weight |
|---|---|---|---|
| Qwen Fun-Acc 4-step | 192 (160) | 0 | 15 (img_in, txt_in, modulation, norm/head diffs) |
| Qwen Viggle v0.3 6-step r128 | 192 (160) | 0 | 3 |
| Krea lvladikov 4-step r64 | 224 (224) | 0 | 4 |

**Qwen-Image 2.1 INT8 ConvRot, 1024², euler, cfg 1, seed 42, portrait prompt.**

| Setup | Steps | s/step | Sampling, cold | Total, cold | GPU memory | vs 25-step: PSNR / SSIM |
|---|---|---|---|---|---|---|
| Base | 25 | 8.78 | 257 s | 306 s | 9.1 GB | n/a |
| Fun-Acc 4-step (eastmoe ComfyUI conversion), PDD sigmas | 4 | 9.11 | 75 s | 118 s | 9.2 GB | 15.0 dB / 0.743 |
| Viggle v0.3 6-step LoRA r128, Viggle sigmas | 6 | 9.41 | 101 s | 147 s | 9.4 GB | 25.8 dB / 0.903 |
| **Qwen-Image-2.1-Turbo** (official 8-step checkpoint, our INT8 ConvRot conversion), saved sigmas | 8 | 8.80 | 104 s | 156 s | 9.1 GB | 14.2 dB / 0.702 (different checkpoint; not a fidelity measure). INT8 vs its own bf16: 41.3 dB / 0.997. |

- **Viggle:** keeps the 25-step composition and look, clean and slightly smoother.
- **Fun-Acc:** coherent but a different composition, visibly grainy, with harsher skin.

Fun-Acc caveats:
- **Not a plain LoRA.** The official release (alibaba-pai, `models/Qwen-Image-2.1-Fun-Acc-4Step.safetensors`) is a PDD bundle: four per-step output heads, replaced q/k norm weights, and a fixed 2048-trained sigma grid. Core ComfyUI cannot express per-step heads.
- **What ran:** eastmoe's `Qwen-Image-2.1-Fun-Acc-4Step-comfyui.safetensors`, verified against the official file. Its LoRA pairs are bit-identical, its norm diffs are exact, and `proj_out.diff` is exactly the mean of the four heads minus base; the heads differ from that mean by 0.6-0.8%.
- **Exact version:** the per-step heads need T8star's custom sampler node.
- **Viggle settings:** the sigmas are Viggle's raw 6-step nodes through their resolution shift: 1.0, 0.9678, 0.9334, 0.8572, 0.6668, 0.4001, 0.

**Krea 2 Turbo INT8 ConvRot, 1024², euler simple, cfg 1, seed 42, soccer prompt.**

| Setup | Steps | s/step | Sampling, cold | Total, cold | GPU memory | vs 8-step: PSNR / SSIM |
|---|---|---|---|---|---|---|
| Base | 8 | 14.91 | 216 s | 256 s | 15.0 GB | n/a |
| lvladikov Krea2-Turbo-Distill 4-step r64 (`krea2_turbo_4step_rank_64_lora_comfyui.safetensors`) | 4 | 15.29 | 155 s | 197 s | 15.1 GB | 14.9 dB / 0.758 |

Sharp and well lit, with a different pose; one small malformed hand behind the leg. The side path costs 2.6% per step.

**Wan 2.2 TI2V-5B, 704x416x33.**

| Setup | Steps | s/step | Load | Sampling | Decode | Peak |
|---|---|---|---|---|---|---|
| Turbo INT8 ConvRot, euler, shift 8 | 4 | 3.27 | 0.6 s | 21 s | 20.6 s | 6.1 GB |
| FastWan2.2 FullAttn bf16 (Kijai/WanVideo_comfy, SHA-256 5f464f04…), lcm, sigmas 1.0, 0.93968, 0.84521, 0 | 3 | 3.4-3.5 | 107 s | 19 s | 32.5 s (swap rose) | 10.1 GB |

The FastWan sigmas are FastVideo's DMD steps 1000/757/522 warped through the shift-5 flow schedule, which is FastVideo's own default. The clip is coherent and more photographic, but softer, and the face smears in late frames. It saves about 3 s of sampling against a 10 GB bf16 load and 4 GB more memory, so the Turbo INT8 remains the better choice. The official diffusers-format file does not load in ComfyUI core (no diffusers-to-native key conversion for Wan); Kijai's repackage has keys and shapes identical to the Comfy-Org 5B.

**Best end-to-end time per model on this machine now.**

| Model | Exact settings | Steps × s/step | Decode | End to end |
|---|---|---|---|---|
| Krea 2, 1024² | Turbo INT8 ConvRot + lvladikov 4-step LoRA r64 (side path), euler simple, cfg 1, SDPA | 4 × 15.3 | 2.5 s | 197 s cold; about 64 s sampling+decode estimate, not a measured warm-server request |
| Qwen-Image 2.1, 1024² | INT8 ConvRot + Viggle v0.3 6-step LoRA r128 (side path), euler, Viggle sigmas, cfg 1 | 6 × 9.4 | 3.0 s | 147 s cold; about 60 s sampling+decode estimate, not a measured warm-server request. Fun-Acc 4-step: 118 s, lower quality |
| Wan 2.2 5B, 704x416x33 | Turbo INT8 ConvRot, euler simple, shift 8, cfg 1, fp16 VAE | 4 × 3.27 | 20.6 s | 42 s (cached text conditioning) |
| Wan 2.2 5B, 1280x704x81 | same | 4 × 37.4 | 155 s | about 5.1 min |
| Wan 2.2 14B, 832x480x81 | T2V-A14B INT8 ConvRot high/low + lightx2v 4-step LoRAs, euler, shift 5, cfg 1, 2+2 steps | 4 × 216 | 44 s | about 17 min |
| MiniMax H3, 480x288x22 | pruned FL2VA W6A8 + 4-step Turbo LoRA (side path), res_multistep | 4 × 6.6 | 7.3 s alone | sample stage 127 s |

Reproduce:
- **Krea 2:** `MODE=metal ATTN=pytorch LORA=1 LORA_NAME=krea2_turbo_4step_rank_64_lora_comfyui.safetensors STEPS=4 bench/run_krea2.py`
- **Qwen-Image 2.1:** `STEPS=6 LORA=Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors SIGMAS=viggle bench/run_qwen21.py`
- **Wan 2.2 5B:** `bench/run_wan5b.py`, with `W=1280 H=704 LENGTH=81` for the large size
- **Wan 2.2 14B:** `W=832 H=480 LENGTH=81 bench/run_wan14b.py`

Run each through `bench/with_gpu.sh`.

## 2026-10-03: Training, milestone 1 (Krea 2 Raw LoRA on the frozen INT8 ConvRot base)

A rank-16 LoRA trained on Comfy-Org's `krea2_raw_int8_convrot` on the M1 Max 32 GB. The INT8 weights stay frozen and resident and are never dequantized as a whole. It learns the target style, and the file loads in ComfyUI on both Raw and Turbo.

**Approach: a minimal trainer on ComfyUI (B), not ai-toolkit (A).** I inspected ai-toolkit at ecee894 (2026-09-27) and stopped after about 20 minutes, for structural reasons. Its Krea 2 loader reads the bf16 checkpoint (about 26 GB, `krea2.py:_load_transformer`) and quantizes it at load time with its own convrot8. It has no path for Comfy-Org's pre-quantized file. Its MPS training branch also fake-quantizes activations to int8 (a W8A8 straight-through estimator), which matches its CUDA deployment rather than our weight-only one. It would also need the Qwen3-VL text encoder and the VAE in Hugging Face format, plus a second pinned stack (transformers 5.5.3, a diffusers commit, torchao 0.10). ComfyUI already loads all three of our files and routes INT8 layers to our kernel. So `train/` reuses it.

The upstream hook for ai-toolkit stays small: in `ConvRotInt8Quantizer.forward` (`toolkit/util/convrot_quant.py`, the `x.requires_grad` branch without int8 hardware), call `int8_linear_trainable(x, cr8_qdata, cr8_scales.view(float32), bias, convrot=True, convrot_groupsize=rot)` on the *unrotated* x. Its R4 Kronecker rotation and per-row fp32 scales match ours. It was untested at this milestone; the October 4 boundary trial now passes. It remains an explicit weight-only experiment because it drops their activation fake-quant.

**Backward kernel (comfy-kitchen caed5e2, `backends/mps/autograd.py`).** `int8_linear_trainable` wraps the fused forward in an autograd Function. Autograd saves only the int8 codes and scales. The backward computes `dx = dy @ W` and nothing for W:
- Each gradient row is normalized by a power of two.
- The weight is decoded in row chunks as `q * s / 2^ew`, with one exponent for the whole layer.
- The product runs as an fp16 GEMM over N, with fp32 accumulation across chunks.
- ConvRot results are re-rotated with the forward's butterfly kernel.
- Exponents are restored on raw bits.

21 tests against a CPU fp32 oracle pass (`tests/test_mps_autograd.py`). They cover three dtypes, ConvRot on and off, per-channel and scalar scales, gradients from 2e-5 to 1e6, chunking, K not divisible by 4, no saved dequantized weight, non-reentrant checkpointing and frozen-bias rejection. The "fold the scale into the fp16 gradient" trap reproduces in the test (more than 1% error), and the kernel avoids it.

Real Krea 2 Raw layers, 1056 tokens (`bench/bench_int8_backward.py`):

| Layer (K to N) | Forward | Backward | Dense fp16 backward | Dequantize whole layer, then GEMM | Error, fp16 grad | Error, 2e-5 bf16 grad |
|---|---|---|---|---|---|---|
| attn.wq 6144 to 6144 | 9.4 ms | 9.7 ms | 9.0 ms | 12.2 ms | 3.6e-4 | 1.7e-3 |
| attn.wk 6144 to 1536 | 3.4 | 3.2 | 2.3 | 3.2 | 3.6e-4 | 1.7e-3 |
| mlp.up 6144 to 16384 | 24.9 | 25.2 | 23.9 | 32.4 | 3.6e-4 | 1.7e-3 |
| mlp.down 16384 to 6144 | 26.6 | 25.9 | 24.2 | 32.7 | 3.6e-4 | 1.7e-3 |

The backward runs within 5 to 8% of a dense fp16 GEMM with resident weights on the three large layers (attn.wq, mlp.up, mlp.down) and about 25% faster than ai-toolkit's per-call whole-layer dequantize there. The narrow attn.wk layer is the exception: 3.2 ms against 2.3 ms dense, 39% slower, and no faster than the whole-layer dequantize. Errors sit at the fp16 floor (fp16 gradient) and at bf16 output rounding (bf16 gradient).

**Trainer (`train/`).** Its own uv venv (`train/setup.sh`, the inference venv's pins, comfy-kitchen path-linked).
- `make_dataset.py`: 20 Krea 2 Turbo INT8 images, 512px, 8 steps. Each is rendered from "<subject>, flat vector art in only red, white and black, ..." and captioned "rwbflat style, <subject>". The captions deliberately leave out the style words, so the LoRA has to learn the style from the pixels and tie it to the trigger.
- `train_lora.py`: text embeddings and latents are cached, and the encoders stay off the GPU. All 224 INT8 block linears run through the trainable kernel, and the 224 attention and MLP linears carry rank-16 LoRAs (53.7M parameters).
  - Init: A ~ N(0, 1/r), B = 0, alpha = rank.
  - Loss: rectified flow (target `e - x0`, fp32 MSE).
  - Noise level: logit-normal t mapped through the model's own shift (mu = 1.15).
  - Non-reentrant checkpoint per block, fused AdamW (lr 2e-4, weight decay 0.01, grad clip 1.0), fp16 compute with GradScaler.
  - It sets ComfyUI's `in_training` flag, which swaps comfy_kitchen's RoPE op (no autograd formula) for the torch one, as ComfyUI's TrainLoraNode does.
- Saves use Krea's diffusers key layout (`transformer.transformer_blocks.N.attn.to_q.lora_A/B`, the same as `krea2_darkbrush`).

| Run: Raw INT8, 512px, rank 16, batch 1, 500 steps | |
|---|---|
| Seconds per step | 12.2 for most steps, 13.3 mean; steps 400 to 500 averaged 15.2 (cause not identified; the run held the GPU lock throughout) |
| Peak GPU driver memory | 15.9 GB (model 13.0 GB after load) |
| Step 0 against ComfyUI's own forward | 4.4e-4 relative (B = 0; fp16 rounding) |
| Fixed training-set loss (training images 0 and 7 x 3 sigmas, same noise; not held-out data) | 0.0473 at step 0, 0.0318 at 50, 0.0294 at 150, 0.0280 at 400, 0.0281 at 500 |
| GradScaler | scale held at 65536, no skipped steps, no non-finite loss |

The per-step training loss is dominated by the random noise level (10-step means range from 0.013 to 0.07), so the fixed training-set loss is the trend to read. It falls by 41%, most of it in the first 50 steps. Curve: `out/train-rwbflat-raw-loss.png`. The research estimate was 11 to 14 s/step and about 16 GB; both hold.

**Generation proof (`train/eval_lora.py`, the LoraLoaderModelOnly path).** Seed 7, 512px, held-out subjects. Palette share is the fraction of pixels that are near black, near white, or saturated red.

| Model, LoRA | Prompt | PSNR LoRA vs none | Palette, no LoRA | Palette, LoRA |
|---|---|---|---|---|
| Turbo 8 steps, s500 | rwbflat style, a horse galloping on a beach | 6.4 dB | 0.26 | 0.99 |
| | rwbflat style, a teapot on a wooden table | 6.2 | 0.78 | 0.88 |
| | rwbflat style, a hot air balloon over a valley | 4.8 | 0.24 | 0.99 |
| | a portrait photo of an old fisherman (no trigger) | 14.7 | 0.50 | 0.64 |
| Turbo 8 steps, s250 | horse / teapot / balloon / portrait | 4.6 / 5.2 / 4.9 / 14.2 | | 0.99 / 0.99 / 0.99 / 0.68 |
| Raw 30 steps, cfg 3.5, empty-prompt negative, s500 | a portrait photo of an old fisherman (no trigger) | 15.1 | 0.34 | 0.45 |
| | rwbflat style, a teapot on a wooden table | 6.7 | 0.54 | 0.98 |
| Raw, same, zeroed negative (grainy, see below) | horse / teapot / portrait | 7.4 / 9.7 / 16.2 | 0.85 / 0.42 / 0.38 | 0.97 / 0.97 / 0.42 |

On Turbo, the trigger turns every held-out subject into flat red, white and black poster art with thick black outlines, the dataset's look. Without the LoRA, "rwbflat style" gives a colorful flat illustration (blue sky, brown horse, green trees). The no-trigger portrait stays a photograph with the same man and composition; only the background shifts. Step 250 is already as strong as step 500 here. All 224 layers map in ComfyUI. Applied through the side path, Turbo runs at 3.74 s/step against 3.7 without a LoRA. The standard runner loads the file too: `MODE=metal LORA=1 LORA_NAME=krea2_rwbflat_raw_s500.safetensors bench/run_krea2.py` renders the styled horse at 3.75 to 3.79 s/step, 14.1 GB (`out/krea2-metal-lora-side-512x512-rwbflat-horse.png`). Sheets: `out/train-krea2_rwbflat_raw_s500-turbo-sheet.png`, `..._s250-turbo-sheet.png`, `..._s500-raw-negempty-sheet.png`.

**Raw sampling trap: the zeroed negative.** My first Raw renders were grainy with and without the LoRA (`..._s500-raw-sheet.png`). The cause was the negative conditioning. `bench/run_krea2.py` uses `ConditioningZeroOut`, which is harmless on Turbo at CFG 1, where the negative is never used. Under CFG 3.5 the all-zeros text context is out of distribution for Raw, and CFG amplifies its prediction. The resolution-dependent shift (mu 0.63 at 512px, `--dynamic-shift`) changed nothing. An encoded empty prompt gives clean Raw images on the INT8 backend at 512px and 30 steps. The runner now encodes an empty negative whenever CFG > 1.

**Does a LoRA trained on Raw INT8 behave the same on Turbo INT8?** Mostly yes. The file applies unchanged, because both checkpoints have the same 224 INT8 linears. On both models the trigger imposes the trained palette and the flat, outlined look on held-out subjects, with palette shares of 0.88 to 0.99 on Turbo and 0.97 to 0.98 on Raw. On both, the no-trigger photo stays a photo (PSNR 14.7 dB vs base on Turbo, 15.1 on Raw).

The differences are small:
- Raw drifts a little more on the untriggered prompt: the man's hat and background change.
- Turbo's styled images come out flatter and cleaner, which fits the dataset having been rendered by Turbo.

This is the same practice as Krea's own LoRAs, whose cards show them on Turbo. Not measured: transfer to a bf16 base.

Training noise levels used the model's fixed shift (mu 1.15, the 1280px value) at 512px. ai-toolkit's resolution-dependent rule gives about 0.6 there. For style that weights high noise levels more; it worked here, but the resolution-dependent value is the better default.

**Milestone-1 gap list, recorded before the later trainer work.** Resume, chunked attention, previews, buckets/config, packed backward and the ai-toolkit trial below are now implemented/proved at their stated boundaries; see the October 4 results. The remaining optional performance/compute comparisons retain their original status.
- Sample previews during training.
- Resolution buckets, and a config file instead of flags.
- Resume (the optimizer state is not saved).
- W4A8/W6A8 backward. It needs grouped scales and is not written yet.
- Query-chunked attention for 1024px, which is over budget per the research note.
- A Metal backward kernel that contracts over N without the decode buffer.
- An INT8 versus bf16 base A/B, and a bf16 versus fp16 compute comparison (not run).
- A trial of the ai-toolkit hook described above.
- Evaluation takes about 100 s per Turbo image against 30 s of sampling. My unverified guess is that ComfyUI re-patches the model when switching between the base and LoRA patchers; rendering all base images first, then all LoRA images, should help.

## 2026-10-03: review fixes on the ComfyUI branch

All seven findings of an independent review of the ComfyUI branch are fixed in vendor/ComfyUI 145aece..f42b777. The fixes:
- uni_pc's float64 path now runs on MPS only.
- Split attention keeps K/V that broadcast over Q's batch.
- The VAE fp16 preference and the Metal formats follow the actual device, the backend's state, and FORCE_FP32.
- Saving writes the merged LoRA weight. This also fixes the lazy saver failing on integer weights.
- Weight wrappers and hooks force a merge.
- Resident LoRA factors are counted in memory.

Tests: comfy_test 298 passed, 14 skipped. tests/test_lora_sidepath.py: 33 passed. Each new test for a finding fails on the previous code.

Krea 2 + darkbrush, 512, after the fixes: all 224 layers fused, 3.76 to 3.78 s per step, fp16 VAE on MPS. The image is pixel-identical to the combined run before the fixes.

## 2026-10-03: MiniMax H3 on M1 Max 32 GB, how far it goes

Pruned FL2VA with the 4-step Turbo LoRA kept beside the quantized layers (208 layers for W6A8, 200 for W4A8, none merged), res_multistep, fp16 VAE, SDPA, beach prompt, fork 7a67766 plus later fixes. W4A8 is the AsymW4A8 pruned FL2VA, file `minimaxH3INT8INT6INT4_flREF2VAPruned_3089623`, 12.5 GB.

| Size (frames) | Format | s/step | Sampling wall | Of which load | Decode | Peak GPU | Swap growth | Frames |
|---|---|---|---|---|---|---|---|---|
| 480x288 (22) | W6A8 | 6.55 | 582-740 s | 556-720 s | 36-39 s | 19.1 GB | +23-26 GB | clean |
| 480x288 (22) | W4A8 | 6.6 | 329 s | ~310 s | 41 s | 15.7 GB | +14 GB | clean |
| 640x352 (56) | W4A8 | 25.5 | 382 s | ~280 s | 57 s | 16.7 GB | +15 GB | clean |
| 640x352 (56) | W6A8 | 25.4 | 1355 s | ~1250 s | 52 s | 20.1 GB | +24 GB | clean |
| 832x480 (56) | W4A8 | 48.6 | 465 s | ~270 s | 78 s | 17.7 GB | +9 GB | clean |
| 832x480 (56) | W6A8 | 48.6 | 1014 s | ~820 s | 75 s | 21.0 GB | +25 GB | clean |
| 1344x768 (56) | W4A8 | 172.6 | 835 s | ~316 s | 129 s | 22.0 GB | +10 GB | clean |
| 1344x768 (56) | W6A8 | 182-210 | 2867 s | ~2000 s | 139 s | 25.3 GB | +25 GB | clean, swapping |
| 1344x768 (124, native ~5 s) | W4A8 | 1420-1940 | 6890 s | ~300 s | 271 s | 30.1 GB | +21 GB | clean, memory-bound |

640x352 and 832x480 replace the requested 640x368 and 848x480, which are not on the node's 32-pixel grid. Swap started near 10 GB on every run because of other resident work.

**Text encoding.** Loading the encoder takes 21 s and encoding takes 123 s, once per prompt. For text-to-video the conditioning does not depend on width, height or length; only the empty latent does. A fresh encode at 640x352x56 was bit-identical to the cached one. `bench/run_h3.py` now caches conditioning per prompt and builds each size's latent with the node's own `_empty_av_latent`.

**Load is the hidden cost.** Stack samples during the slow first step (`logs/h3-load-stacks.txt`) show the main thread in the module-to-GPU copy of the quantized weights (`ModelPatcher.load` into comfy-kitchen `_handle_to`). The disk reads at 677 MB/s, so 16 GB should take about 25 s. Under memory pressure the copy runs at about 22 MB/s while the system swaps out other work. The W4A8 file is 3.5 GB smaller and loads 2-4x faster (about 5 min against 9-33 min for W6A8). It also runs steps at the same speed, using 3-4 GB less GPU memory. The earlier 127 s W6A8 sample stage was measured with less resident on the machine. A ComfyUI server pays the load once and keeps the model.

**Where the limit is.** At 1344x768 with 56 frames (17.3k tokens) compute is the limit. Attention is about 82 s of the 172.6 s step (47%), at the measured bf16 SDPA rate of about 5.25 TFLOPS; the rest is linears. At the native 124 frames (37.6k tokens) compute alone predicts about 580 s per step: 386 s attention plus 197 s linears. The measured 1420-1940 s, with 30 GB on the GPU and swap rising, means memory is the limit there. The H3 VAE already tiles spatially and chunks temporally (`decode_tiled` is the same path), and decode is never the problem (271 s at native size, 7.2 GB). The text encoder is already released before the DiT loads. `COMFY_KITCHEN_MPS_DECODE_BYTES` only sizes the kernels' 512 MB weight-dequantize scratch; its effect on a 30 GB peak is small, and it was not tested here because the GPU lock was busy with other work.

**H3 on M1 Max 32 GB, recommended.**

| Use | Settings | Time (warm model, cached prompt) | Time (cold) |
|---|---|---|---|
| Quick preview | W4A8 + Turbo LoRA, 832x480, 56 frames (2.3 s), 4 steps | 4 × 48.6 s + 78 s decode, about 4.6 min | about 7.8 min, plus 2.4 min for a new prompt |
| Best that fits | W4A8 + Turbo LoRA, 1344x768, 56 frames, 4 steps | 4 × 172.6 s + 129 s decode, about 13.7 min | about 16 min |
| Native 5 s clip | W4A8 + Turbo LoRA, 1344x768, 124 frames | about 2 h (swapping) | about 2 h 2 min |

Prefer W4A8 to W6A8 on this machine: same speed, 3-4 GB less memory, much faster load. The largest clean clip is the native 1344x768x124: it completes in 2 h 2 min with 30 GB on the GPU. The largest clip that runs without heavy swapping is 1344x768x56 in about 16 minutes cold.


## 2026-10-03: reviewed generation and recovery implementation (evening)

GPU ownership, attention broadcasting, LoRA accounting and tensor lifetimes are repaired. Explicit MPS pread now passes real Wan construction/offload/reload and exact generated-output parity; default loaders remain unchanged. Isolated registered-model loading measured 6.52s direct versus 12.29s CPU staging, but real load-plus-sampling was 6.93s versus 7.23s, so the transfer ratio is not an end-to-end claim. The known-good 704×416×33 Wan recipe still produces a recognizable musician at 3.3s/step; 512×288×17 is unsuitable as a visual-quality gate.

The initial persistent server stopped under memory pressure while copying UMT5 weights. The opt-in CPU ownership-transfer experiment loaded the model without that copy and encoded the canonical prompt in a 62.61s process, with exact cached-conditioning parity and no swap growth.

Training recovery now writes atomic periodic optimizer/RNG state and rejects corrupt caches/non-finite updates. The real Qwen1024 seed1 save/resume pilot completed steps 0–4 and 4–8 in separate processes at ~35s/step, 11.48GiB driver peak, flat swap, and validation 0.20313→0.18720→0.18595. State and both adapter exports are finite and reloadable. At that stage a reviewed four-hour guarded continuation was prepared. The later padding diagnosis below changes the input targets: preserve this old pilot and use distinct corrected runs instead.


## 2026-10-04: exclusive MPS wide-linear reproduction

The pending issue's clean run completed under the monitored GPU lock in 176.03 seconds, with normal system pressure and no swap growth. PyTorch 2.14.1 on M1 Max/macOS 15.7.9; M=4608, N=6144; K=4096/8192/16384/32768; fp16 and fp32; two interleaved rounds of three timed iterations after warm-up. All eight implementation/layout variants matched exactly at every shape (max absolute difference zero).

At K=32768, fp16 F.linear took 697.9 ms against 198.2 ms for a contiguous pre-transposed weight (3.52x); fp32 took 2117.6 ms against 239.7 ms (8.83x). A transposed left operand with contiguous right operand stayed near the fast path. This supports the existing wide-weight decode layout decision, not a new end-to-end speed claim. The [issue draft](upstream/pytorch-mps-wide-linear-issue.md) now contains the clean table. No issue filed and no nightly build tested. Raw evidence: `logs/repro-mps-wide-linear.log`.


## 2026-10-04: Training milestone 2: completed experiments and remaining evaluation

Qwen seed-0 1024px training is already complete at 300 steps with query-chunked attention: approximately 36 s/step, 11.49 GiB driver peak, final fixed training-set loss 0.17037 (logged as `val_loss`; computed on two training images, not held-out data). The 512px comparison also completed all three 300-step arms:

| Training base | Training seed | Mean seconds/update | Driver peak | Final fixed training-set loss |
|---|---:|---:|---:|---:|
| INT8 ConvRot | 0 | 8.389 | 9.46 GiB | 0.16436 |
| bf16 | 0 | 12.350 | 15.57 GiB | 0.16506 |
| INT8 ConvRot | 1 | 8.591 | 9.46 GiB | 0.16337 |

INT8 used 6.11 GiB less driver memory and 32% less update time than the matched bf16-base run. The time gap is confounded by dtype: bf16 GEMM runs about 1.75x slower than fp16 on this M1 Max (first entry of this file), and no fp16 dense arm was run, so the 32% is not the kernel's contribution alone. The same-seed fixed training-set loss gap (0.00070) is smaller than the two INT8 seeds' gap (0.00099). On four prompts rendered on the same INT8 inference base with Viggle, mean image PSNR between the same-seed training-base pair was 20.76/24.80 dB at 512/1024px, versus 14.28/16.99 dB between the two INT8 training seeds. In this sample, training-base changes affected rendered images less than changing seed. This is not a general quality bound or proof of transfer to bf16 inference. Logs: `train-qwen-512-*.log` and `train-eval-qwen-ab{512,1024}.log`.

Remaining T1 acceptance is the 1024-trained adapter's held-out trigger/no-trigger evaluation on plain and Viggle INT8 inference, plus bf16 inference-base portability. The reviewed single-column runner prepares twelve individually bounded images without repeating training. T2 initially targeted continuation from the finite step-8 seed-1 pilot. The later confirmed padding defect below requires distinct corrected cache/run identities instead; the old pilot remains historical save/resume evidence, not the starting point for corrected training. Matched final visual comparison remains required.

## 2026-10-04: Usable trainer features and packed-weight proof

The trainer now has JSON configuration with CLI overrides, deterministic aspect-ratio buckets and distinct cache identity, atomic recovery with immutable configuration/content fingerprints, and opt-in scheduled previews. A real three-image run used 384×512, 512×320 and 448×448 buckets over four updates in two processes. Identity-validated resume completed; fixed training-set loss (logged as val_loss) changed 0.21995→0.21594→0.21410, updates took 4.84–6.06 s, driver peak was 9.44 GiB, and both supervisor exits were normal without swap growth. The resumed run produced a coherent inspected blue-fox preview through real Qwen and CPU VAE. This closes T3/T4/T6 at the measured pilot scope, not at arbitrary preview resolutions.

Packed W4 and W6 input backward passed 43 MPS tests and separate two-update real Qwen proofs. Each run converted all six projections in transformer block 1, leaving the remaining base INT8. Every converted layer returned finite nonzero input gradients twice; LoRA weights updated, packed weights stayed unchanged, and initial output matched the ordinary route exactly. Both runs peaked at 9.45 GiB driver memory and completed without swap increase. The quantization preparation ran in a separate CPU process. This closes T5's representative mixed-base acceptance; it does not claim complete W4/W6 checkpoint training or H3 training.

The real ai-toolkit `ConvRotInt8Quantizer`/`OstrisLinear` trial passed output/input-gradient, checkpoint recomputation, resident-buffer and two-step optimizer checks. Loss scaling resolved the wide fp16 subnormal-output floor without relaxing tolerances. The candidate removes activation fake quantization, so it is an explicit weight-only mode, not a transparent replacement for toolkit W8A8 training. T7 is complete as an integration trial; the standalone trainer remains the supported end-to-end path.

## 2026-10-04: Generation acceptance and memory limits

The initial October 4 reconciliation found G1–G3 and F1–F3 still open. Later evidence now closes the bounded Wan three-request server gate and representative LoRA/LoHa/LoKr application. Krea and bounded Wan 5B API templates were accepted at that stage. The later Qwen API request below now also passes; Wan 14B and H3 remain pending. Current Qwen and Krea attempts can stop on system pressure before a completed step. CPU encoder weakrefs and mappings release; private footprint after encoding is close to baseline while system file-backed residency remains elevated. Ordinary dense LoRA backups can preserve a native whole-file mapping even with mapped offload disabled, but that mechanism is not yet a demonstrated pressure cause. The fresh-conditioning/direct-loading controls are diagnostic and do not establish default UI behavior.


## 2026-10-04: Actual H3 NVFP4 layers and Qwen GGUF outputs

Two real layers from H3's 32B NVFP4 text-encoder checkpoint pass native MPS execution: plain `k_proj` (1024×5120) and AWQ `o_proj` (5120×8192). Each completed eight BF16/FP32 forwards across initial load and CPU offload/reload, against the existing independent Float64 oracle. Reload outputs were exact; packed/scaling bytes and the original full-precision configuration survived; no activation quantization occurred. Only 2.81/22.52 MiB of selected payload were read. Active GPU allocation returned to 1,024 bytes after offload, and both runs completed with flat swap. Krea NVFP4 with temporary direct loading subsequently reached one unsynchronized forward return before a pressure stop at 79.47 s, without a finished image or swap growth. A later isolated CPU-conditioning capture and fresh-process replay completed the real Krea NVFP4 soccer image: 39.875 s capture, 44.878 s replay including 29.45 s sampling, 2,048 native NVFP4 calls, normal pressure and flat swap. This explicit direct-load route passes bounded model proof; F1's full-encoder, other formats and ordinary default/server boundaries remain open.

Qwen's actual Q4_K_M checkpoint completed six-step 512² generation through the GGUF loader node in Metal and stock modes with the same captured conditioning, seed and Viggle adapter. Metal's sampling phase took 39.56 s versus 66.43 s, with 4.99 versus 5.99 GiB driver allocation after sampling (snapshots, not peaks). Both portraits are coherent and compositionally close, but not exact: RGB PSNR 27.24 dB, luma SSIM 0.9499 and mean absolute channel error 5.896/255. All 195 adapter-bearing layers were reached. Both processes exited normally with flat swap. This proves the bounded Qwen generation comparison; it excludes text encoding, a submitted server graph, explicit full-model offload/reload and Wan acceptance. F2 was open at that stage; the subsequent Wan and Qwen lifecycle evidence below closes its representative node/model scope.

## 2026-10-04: Qwen1024 training-input quality hold

The first matched seed-0 1024 adapter evaluation loses cyanoprint colors; the plain-recipe adapter image also has the beige etched appearance. A fresh FP16 MPS encoding of dataset image 00 exactly reproduces a finite cached latent whose reconstruction is only 7.98 dB. Fresh 512 encoding was visually plausible (26.55 dB), but the later CPU reference below reveals a smaller defect there too. FP32 MPS encoding and contiguous strip inputs do not repair the 1024 example; CPU FP32 encoding through the same ordinary MPS FP16 decoder restores the blue source at 38.38 dB. This initially localized the demonstrated failure to MPS encoding; the operation and repair are now established below. It does not establish all 24 inputs are affected or bind the historical seed-0 run to current cache bytes. T1/T2 remain open; hold long training for repaired, inspected inputs and distinct corrected cache/run identities.

## 2026-10-04: Wan small-recipe constraint

Completing the missing corners did not recover the tiny Wan quality gate. At 512×288×33 the guitarist/guitar become recognizable but distorted; at 704×416×17 the scene remains heavily corrupted. Both runs were finite with normal pressure and flat swap (56.866 / 29.419 s). Together with abstract 512×288×17 outputs and the coherent 704×416×33 control, this supports retaining exactly the latter bounded recipe. Earlier direct/CPU loader outputs were exact, and UniPC/shift5 did not repair tiny output. G3 closes through its explicit recipe-restriction acceptance route, while the operation-level cause remains unresolved; no pure temporal cause or universal minimum is established.

## 2026-10-04: F2 representative GGUF acceptance and Qwen API template

Actual Qwen/Wan Q4_K_M loader-node runs now pass useful-output comparison against stock. Qwen also passed two identical-input sampler passes separated by explicit full-model offload: all 199 packed parameters retained equal bytes/metadata, active MPS allocation returned to zero, and repeated sampled latents were bit-exact. Driver caches remained nonzero, and this is not an OS mapping-release claim.

Wan's matched 704×416×33, twenty-step CFG-5 Euler/simple pair produced coherent cropped guitarist videos in both modes, with independent verification of all 33 MP4 frames. Native sampling took 145.93 s versus 200.71 s stock, with 6,000 native versus 6,000 stock calls and no swap growth. Original four-frame strips compare at 24.10 dB PSNR/0.9253 luma SSIM; final latents differ (relative L2 0.104596). F2 closes at representative headless node/model scope, with Qwen representing explicit lifecycle; fresh encoding and GGUF server templates are separate.

Separately, Qwen's actual API template passed a fresh native-server request: 512×512 latent, native direct INT8 loader enabled, ordinary fresh CPU encoding, 25 Euler/simple steps, CFG 1, no LoRA, full FP16 VAE, enhancer off. The Qwen prefix-cache node remained auto/default and encoder resolution remained 1024. Client time was 103.524 s; the complete monitored process used 9.608 GiB peak group RSS, normal pressure and flat swap. The saved lime/gray fashion portrait is coherent. G2 remains open for Wan 14B/H3; repeated Qwen requests and its larger defaults remain unproved.

## 2026-10-04: VAE temporal-padding root cause repaired

Exact-input differential probes isolate rank-five `F.pad` in the MPS `AvgDown3D` shortcut: padding a 1×96×1×512×512 tensor by one temporal frame returns all zeros. The grouped mean itself is bit-exact on correctly prepared input. A scoped zero-prefix concatenation repair leaves CPU/zero-pad paths unchanged and passes eighteen regressions; both large FP16/FP32 MPS tests failed before it. The integrated ComfyUI change is `9d9b4232`, independently reviewed. [Upstream preparation](upstream/comfyui-mps-vae-temporal-padding.md).

Unhooked native FP16 encoding now restores image 00 at 1024 to 38.3733 dB versus CPU FP32's 38.3755 dB, normal pressure and flat swap. At 512, native/CPU reconstructions agree at 28.6220/28.6212 dB; old cached output was 26.55 dB. Same-dtype native 512 FP32 latent relative L2 is 1.88e-6 versus CPU, while FP16 comparison is 0.002389 with maximum absolute difference 0.69979. Earlier 512 A/B results remain observations under shared historical inputs, not proof on unaffected targets. The existing PyTorch rank-five padding fix landed August 31 but is absent from the installed build's source. No duplicate issue or upgrade was made. The subsequent all-24 corrected-cache validation below passes; corrected training and final adapter evaluation remain open.

## 2026-10-04: All corrected Qwen1024 inputs validated

All 24 native FP16 MPS encodes pass full finite RGBA reconstruction through the ordinary FP16 MPS decoder. RGB PSNR is 32.846–42.719 dB (mean 39.918), and both independently inspected source/reconstruction sheets preserve coherent cyan/blue subjects and paper texture. Successful build/decode take 24.168/76.545 seconds with 2.397/2.462 GiB sampled peak group RSS, normal pressure and flat swap. Old conditioning remains bit-exact; every latent changes. New cache SHA256 starts `2ab5bad8`; the full identity, preserved RGBA/RGB outputs and distinct prepared seed-0/seed-1 configurations are retained. Two helper API/shape mistakes were corrected before acceptance. Bounded save/resume, corrected training and final visual evaluation remain required.


## 2026-10-04: Corrected Qwen seed-0 save/resume and cache preservation

Corrected seed 0 passes four fresh and four identity-checked resumed updates in separate bounded processes (290.097/222.100 s). Both exit normally with unchanged swap, 1.098/1.177 GiB sampled peak group RSS and 11.48 GiB driver peak. Fixed training-set loss declines 0.230067→0.224644→0.223674; this is not visual adapter acceptance. Independent CPU inspection finds all 1,537 step-8 recovery tensors finite, verifies the exact corrected input identity, and matches every step-8 exported tensor to the existing trainer's saved-state export mapping. Both step-4/8 exports have 672 finite tensors. Full corrected training and evaluation remain open.

Both old default caches are retained under `.pre-avgdown-fix.pt` names with verified hashes. The default 1024 path now contains an independent corrected copy; default 512 is absent for fresh rebuilding. Explicit corrected run configurations remain bound to their original validated cache. Unfingerprinted legacy Qwen resumes are refused before model loading (`b888630`), preventing a filename match from silently accepting changed targets. Historical checkpoint preservation is not independently established by these cache hash checks.


## 2026-10-04: Krea NVFP4 matched stock reference passes

The actual stock BF16 dense route completes the same captured-conditioning, direct-load Krea NVFP4 recipe in 70.417 s total and 50.86 s sampling, versus native 44.878/29.45 s: a 1.727× sampling ratio for this single pair. Stock executes 2,048 native NVFP4 dequantizations and ordinary dense linears, with zero native NVFP4 matrix multiplies; all 256 layers remain packed and decoded Python references release. Its initial helper-only dtype assertion failure remains recorded; requested `FP16=1` does not override actual BF16 quantized compute.

Both soccer images independently pass useful-output inspection and compare at 32.1203 dB RGB PSNR, mean absolute channel error 1.6485/255, maximum 186/255. Stock peaks at 7.812 GiB sampled group RSS; normal pressure and swap growth of approximately 123 KiB stay within the unchanged guards. This accepts the diagnostic NVFP4 model comparison; other formats, complete quantized encoders and default/server paths remain distinct.


## 2026-10-04: Partial-residency limits remain incomplete-model boundaries

Compatible MXFP8 and mixed W4A4/INT8 Krea payloads are local and their complete hashes match pinned publisher values. Four tiny synthetic MPS loader/residency cases pass; separate MXFP8 CPU conditioning capture also passes. The full MXFP8 replay then stops for system pressure during model admission at 56.564 s, before any forward. Its CPU construction retains 12,498,501,632 bytes of mapped packed weights and 390,578,176 bytes of owned scales; these logical sizes are not physical residency measurements.

Wan14's 4 GiB limiter succeeds, but the API run stops at the first high-noise sampler after 107.620 s. The Qwen bf16 mapped-assignment retry similarly stops at its first sampler after 176.079 s; neither completes a step. Wan14 has four times Wan5's transformer-token count at the same output size, so the accepted Wan5 recipe does not establish matching working-memory needs. Encoder ownership, transfer/source-page residency and first-forward scratch remain competing hypotheses; MXFP8's earlier admission stop cannot be attributed to attention. H3 and complete W4A4 proof stay pending. These failures leave the accepted NVFP4 comparison and corrected step-8 training recovery intact.


## 2026-10-04: Corrected seed-0 step-50 continuation gate

The identity-checked step-8-to-50 segment completed in 1,558.269 seconds, at 11.48 GiB driver peak and 1.229 GiB sampled group RSS, with flat swap and no pressure stop. Fixed training-set loss declined to 0.22025 from 0.230067 initially. Independent CPU verification reproduces all 672 exported tensors from the actual trainer mapping; the staged export is byte-identical. A temporary staging-reader suffix mismatch was repaired from down/up to A/B without changing training or exported weights.

The 1024px, seed-7, six-step Viggle-stacked INT8 held-out horse image completed in 127.264 seconds with flat swap and 10.258 GiB sampled group RSS. Independent inspection finds coherent blue horse/beach imagery and stronger grain than the preserved base, with no beige collapse. The base was already blue and both palette measurements are 1.0, so this is an early continuation gate, not proof of learned style or final T1/T2 acceptance. Corrected full training, the second seed, trigger/no-trigger comparisons and bf16 portability remain open.


## 2026-10-04: Bounded Qwen repeated-server acceptance

A fresh production-launcher server completed the exact accepted 512×512 direct-INT8 Qwen graph three times: initial seed 7, changed seed 8, then seed 8 with lime→cobalt prompt change. Each uses 25 Euler/simple steps, CFG 1, ordinary fresh CPU encoding, retained Qwen prefix cache and full VAE. Client wall times are 114.675/102.286/107.578 seconds; all graph cache lists are empty. Three decoded RGB hashes differ, and independent inspection confirms coherent fashion portraits, changed composition and visible blue background.

The 332.100-second monitored session peaks at 9.620 GiB group RSS, with no pressure stop and flat 1.739863 GiB swap. Watermarks 0.5/0.4 and cache-ram 18 remain unchanged; no warm resident speedup is proved. This closes the bounded Qwen repeated-request gap while G2 Wan14/H3, larger Qwen defaults, prompt enhancer and ordinary diffusion loader remain open.

## 2026-10-04: Corrected second-seed recovery and narrower admission controls

*The entries from here through the 2026-10-05 H3 task-VM observation entry are process receipts (per-run tool counts, pressure stops, review outcomes) kept for provenance. docs/RESULTS.md is the curated view of what the kernels measure.*

Corrected seed 1 now passes four fresh updates and four identity-checked resumed updates in separate processes: 289.099 / 225.406 seconds, 1.170 / 1.032 GiB sampled group RSS, 11.48 GiB driver peak and flat swap. Validation decreases 0.23007 → 0.22448 → 0.22342. Saved CPU verification records confirm 1,537 finite state tensors and 192 LoRA pairs at both checkpoints with the corrected cache and seed-1 identity. Final training, exact seed-1 export comparison and matched image evaluation are not inferred.

H3’s ordinary headless CPU encoder constructs but hits the 16 GiB RSS guard during encoding, before conditioning returns. Qwen bf16 4/1 GiB first-forward probes reach blocks 10/22 then stop on an additional psutil-available threshold. The VM-observer attempt stops inside its post-admission observation; it supplies no physical admission snapshot or failing model forward. Earlier completed CPU-phase snapshots show released encoder mappings and system file-backed growth, without proving a universal retention cause. The first native-pressure successor fails closed because Comfy changes the directory before its lazy reader import. Its repaired retry records normal native pressure at block 23, then pressure level 2 at block 29 and stops before a full forward. The 133.439-second process exits cleanly with 7.604 GiB peak group RSS and flat swap; cleanup success does not pass the model gate. These failures leave G2/F1/bf16 acceptance open and do not invalidate corrected training recovery.


## 2026-10-04: Krea MXFP8 and mixed W4A4/INT8 matched diagnostics pass

Both remaining Krea block-format pairs complete with separate CPU conditioning and a temporary 1 GiB persistent-residency cap after ordinary mapped CPU construction. W4A4/INT8 native/stock sampling is 75.95/122.78 s, with 34.3365 dB RGB PSNR; MXFP8 is 103.78/112.94 s, with 36.8756 dB. All four outputs were independently inspected and are coherent matching soccer scenes. W4A4/INT8 dispatch is exactly 1,024 W4A4 plus 768 INT8 native calls versus 1,792 stock dense/decode calls; MXFP8 is 2,048 native versus 2,048 stock dense/decode calls. All four supervisors exit normally with flat swap. Ratios describe individual pairs, not a general or cross-format speed ranking.

Together with the earlier NVFP4 pair, this closes Krea's advertised block-format diagnostic model comparisons. The historical 4 GiB MXFP8 failure is preserved; the new result qualifies only the measured 1 GiB separate-process route and does not isolate a universal memory cause. F1 still requires complete H3 quantized-encoder acceptance, and G2 still requires Wan14/H3 templates. Ordinary default/server behavior remains distinct.


## 2026-10-04: H3 FP32 packed-dispatch repair passes; full acceptance remains open

CLIP's real FP32 compute override had bypassed native NVFP4 despite the weight-only text context. Reviewed fix `6ad852be` preserves packed dispatch at that narrow boundary and passes all 28 integrated CPU/MPS regressions. Its unchanged prepared counterpart `ca51eaed` passes 29 CPU checks and a separate parent-owned 16/16 MPS run, zero skips, against Kitchen block-formats `354d0f5`.

One full encode executed all 350 NVFP4 layers plus the INT8 embedding and reached finite conditioning, then failed an incorrect helper assertion that required positive persistent residency. The corrected validator accepts fully offloaded execution; its final retry instead stopped for system memory pressure after 122.875 s, with 337 native returns, 3.560 GiB sampled peak group RSS and flat swap. A retained ENOSPC attempt also executed 33 native operations before supervisor disk errors; it was not merely prelaunch despite its filename. None supplies a successful final guarded full-encoder report, so F1 H3 encoder and G2 H3 template acceptance remain open.


## 2026-10-04: Guarded full H3 encoder and BF16 demand-read first return

A quiet-host retry completes the entire H3 quantized encoder under the unchanged frozen recipe and guards: 80.073 seconds, 3.546 GiB sampled peak group RSS and flat 0.569521 GiB swap. All 350 NVFP4 layers plus the INT8 embedding execute once, yielding finite CPU conditioning `[1,388,5120]` and checked video/audio latent geometry, with zero persistent MPS weights inside the 0–1 GiB cap. Actual initial free memory is 83%. This closes F1 at bounded diagnostic scope alongside the three Krea format pairs; independent numerical reference evidence remains at selected-layer scope. No diffusion/VAE or H3 API/video acceptance follows. Earlier helper, pressure and disk failures above remain preserved.

The separate Qwen BF16 demand-read diagnostic passes its tiny numerical/lifetime prerequisite and conditioning capture, then returns from the first real Python model forward under normal pressure. It covers 194 eligible parameters with 383 reads and 1,427 owner checks, closes with zero Python read-tensor references, and exits normally in 82.236 seconds at 3.178 GiB peak group RSS with flat swap. The stop is explicitly unsynchronized; no completed sampler step or image is established. T1 full BF16 inference portability remains open.

## 2026-10-04: Corrected seed-1 step-50 state and actual exporter verified

Corrected seed 1 completes its guarded 8→50 segment in 1,552.587 seconds, with validation 0.21996, 34.884 seconds/update, reported 11.49 GB driver peak, 1.455 GiB sampled peak group RSS and flat swap. Its step-50 recovery state is preserved separately. CPU-only verification against the source-bound actual trainer exporter confirms 1,537 finite checkpoint tensors, 192 state layers, and exact equality for all 672 exported tensors / 224 split factor pairs. The verified adapter is staged. This supersedes the earlier pending exact-export check without claiming seed-1 image quality.

Corrected seed 0's continuation toward 300 was launched and its log confirms recovery at step 50; final completion is not inferred from that launch. Both final states/exports and matched INT8/BF16 images remain required.

## 2026-10-05: final corrected Qwen and BF16 hardware qualification

Both corrected 1024 seeds finish step 300 with finite preserved checkpoints and exact actual-exporter agreement for all 672 adapter tensors. Seed 1 preserves its pressure stop at step 210 and completes the guarded resume with flat swap; validation is 0.21592 versus seed 0's 0.21524. Matched evaluation is still underway. Final triggered images retain cyan, but the base prompt already induces cyan; palette alone cannot attribute learned style. Without the trigger, both adapters change a photographic fisherman into a coherent graphite portrait. Seed 1's triggered horse face is weaker than seed 0's. The plain-schedule seed-0 horse remains coherent and cyan.

The source-bound tiny stacked BF16/F32 LoRA MPS gate passes in 5.549 seconds at 0.573 GiB peak group RSS with flat swap. Twelve forwards, including six omission controls, match the independent scalar oracle and ordinary MPS output exactly; fourteen fresh demand reads preserve mapped ownership/version/bytes. Four queued groups each settle after exactly one cleanup at 8,781,824 driver bytes, below the warmed 8,798,208-byte baseline; final release is 393,216 bytes. The unchanged 16 MiB ceiling holds. This qualifies the tiny stacked/offset and queued-read boundary, not real-model image portability or its six-case matrix.

## 2026-10-05: final encoder captures and an INT8 admission boundary

Both exact final BF16 conditioning recipes pass encoder-only capture and independent saved-byte/source/model/recipe verification. Trigger takes 12.544 s at 6.253 GiB peak group RSS; untriggered takes 11.898 s at 6.866 GiB, both with normal native pressure and unchanged 0.4992 GiB swap. UNET and VAE loading are forbidden. Six parent gate tests and eight unchanged serializer/runtime tests pass; completed independent review has no P0/P1. This does not accept full BF16 rendering.

The original source-bound plain untriggered INT8 no-adapter attempt fails before sampling: the direct loader requires 12.11 GiB and observes 11.90 GiB system headroom (21.33 GiB driver headroom). The outer monitor records normal initial pressure, 40.254 s, 7.584 GiB peak RSS and flat swap; exit 1 is preserved. No retry or weaker admission is accepted. Investigating a separate encoder process must retain the actual encoded empty negative used by the plain recipe.

## 2026-10-05: H3 deferred handles reach actual encoder entry

The completed independent review and 56 parent CPU checks qualify one controlled observation of the original H3 graph. Actual entry shows two distinct live video/audio VAE handles at their exact frozen paths, one handle each, zero original-factory/private-decoder calls, and no active VAE patcher or model. Both remain absent from the eager owner census. However the outer monitor stops for system pressure at 83.419 s with peak group RSS 3.521 GiB and flat swap. Only initialization, encoder loader return and encoder entry are observed; no selected native return, encoder completion, diffusion or decode is accepted. This disproves deferral alone as a sufficient remedy on this host. The attempt and cleanup failure are preserved; no quiet retry.

## 2026-10-05: Wan14 casts and first attention, still below acceptance

The reviewed concrete observer uses the original graph, weights, scheduler, cast arithmetic and guards. Both fresh CPU encoder returns precede high sampling. It observes the first attention input [1, 10296, 5120] with 40 heads and unsynchronized attention return, plus paired CPU-packed INT8 ConvRot q/ffn casts returning MPS FP16. It then stops for system pressure at 139.162 s, 10.303 GiB peak group RSS, flat swap, before the first high Python forward returns. No sampler step, settled GPU or generation is claimed. An initial tqdm prefix makes the original anchored log parser omit the attention-entry row; raw bytes retain it and a separate strict CPU extraction successor now passes ten parent CPU checks and independent review. This metadata correction cannot change the authoritative failed pressure outcome or permit a retry.

## 2026-10-05: final INT8 matrix complete with separate plain conditioning

The preserved plain untriggered base refusal is resolved by a source-bound separate encoder/diffusion evaluation route. Two captures retain exact positive and encoded-empty-negative tensors; both have the same actual negative bytes. Thirteen parent CPU checks and completed independent review find no blockers. Canonical source/export/model pins, actual outer supervisor/process group, single-attempt directories, normal-monitor receipts and exited encoder identities are checked before each render. Original sampler, VAE, 224-layer adapter mapping, INT8 direct loader and 0.65 allocator fraction are preserved; no BF16 demand-read policy runs.

All four remaining plain20/CFG4 1024px cases complete normally: untriggered base 367.899 s / 9.495 GiB peak group RSS; seed-0 untriggered adapter 383.404 s / 8.843 GiB; seed-1 triggered adapter 379.800 s / 6.860 GiB; seed-1 untriggered adapter 383.119 s / 9.673 GiB. All have flat 0.4992 GiB swap, initial normal pressure and 76–77% native free. Each actual output and receipt is inspected and hash-verified. This completes execution and visual comparison for the eleven original INT8 cases plus one fresh trigger base control; it does not qualify ordinary server residency.

Quality depends on recipe. Seed 0’s Viggle horse is clearer; seed 1’s Viggle face/anatomy is degraded. Both plain horses are coherent, with different composition/texture and no general winner. Both plain untriggered portraits are photographic with little cyan; the graphite change is strong under Viggle. The recipes differ in stacking, guidance and steps together, so this does not isolate a cause. Triggered base controls already generate cyan, and the different horse/fisherman subjects cannot establish trigger selectivity. Parent and independent visual inspection agree on these limits. BF16 image portability remains open, as do H3/Wan14 API/video gates.

## 2026-10-05: native128 BF16 control passes

Raising only the native observation count from 64 to 128 allows the same original BF16 base control to finish: 79 observations, six forwards covering all 194 target weights each, finite synchronized latent [1,64,64,64] and RGBA [1,1024,1024,4], demand reads closed before VAE, normal 314.163-second outer completion, 3.188 GiB peak group RSS and flat 0.4836 GiB swap. Time, allocator, residency, RSS, driver, native pressure/free and freshness limits stay unchanged. Parent source/output verification and image inspection pass. The coherent cyan horse is a base control; this does not close trained-adapter portability. The six-case corrected-final successor is undergoing loader path/byte binding and independent review before dispatch.

## 2026-10-05: corrected-final BF16 portability accepted

All six corrected-final BF16 comparisons now complete six forwards and finite synchronized 1024px RGBA images, with exact final-adapter/stack byte binding and parent inspection. Each has normal monitored completion (256.273–517.711 seconds, 2.567–3.269 GiB peak group RSS); one swap increase is about 0.31 MiB within the existing guard. All six INT8/BF16 matched pairs have close composition/style. Seed 1’s Viggle horse remains degraded in both routes; the BF16 result establishes operational portability, not improved quality. Scope is the frozen six-step Viggle, CFG 1, seed 7, captured-conditioning/demand-read route with 1 GiB residency; plain BF16 and ordinary server precision equivalence are untested.

This closes T1/T2 at their documented final-training, held-out INT8 and matched BF16 diagnostic scope. Plain20 INT8 results remain separately qualified. Unaligned cross-precision RGB PSNR is 26.070–37.532 dB; these measure image difference, not fidelity or quality. Prior budget, RGBA and admission failures remain preserved. H3/Wan14 API/video acceptance remains G2.

## 2026-10-05: separate H3 capture and WAN14 transport boundaries

H3’s reviewed encoder-only capture stops at 75.030 seconds for native pressure 2, free 42%, unchanged 0.3235 GiB swap and 3.557 GiB peak group RSS. It records 256 NVFP4 returns and one INT8 embedding return, then enters layer 36’s gate projection without a return. The 1,029 module-related memory samples have peak driver 839,516,160 bytes and a flat final 100 driver readings; this does not establish a native allocation-growth cause or invalidate the earlier complete standalone encoder proof. No capture artifact or API replay is accepted.

WAN14’s reviewed full-graph SDPA candidate exits after 60.082 seconds on client HTTP timeout, with a second timeout during cleanup. Outer guard reason is null; pressure stays 1, free 70%, swap unchanged and peak group RSS 10.309 GiB. Both encoder calls return, and 40 high-expert self-attention wrappers install at the two-head cap, but sampling enters without any recorded expert forward or SDPA call. Tiny parity and CPU review do not substitute for this missing execution proof. The actual timeout request path was not logged, so event-loop blocking is not established. Both original attempts remain consumed; new diagnostic policy and transport candidates preserve their evidence.


## 2026-10-05: WAN14 v4 transport survives timeouts, then pressure stops the graph

Both CPU encodings return and the high-expert attention profile installs before sampling enters. Four owned history GET socket timeouts last about five seconds each; cancellation and a cleanup connection reset follow the outer system-pressure stop at 84.701 seconds. The configured 480-second client deadline does not expire. Pressure reaches 2, free memory 44%, peak group RSS 10.304 GiB and swap remains 0.323486 GiB. No expert-forward event or full-model SDPA execution is established, and no video is accepted.

The guard retains a child MPS periodic sample from 22.8507 seconds earlier. Its zero current allocation and 393,216 driver bytes cannot establish allocation at the stop boundary. The native pressure sample is current to the guard query; these observations must stay separate. All 88 source and 22 evidence pins, the completed 372.553-second independent review, parent CPU evidence and actual outcome/log identities are verified. No stack capture was obtained.

## 2026-10-05: H3 bounded-warning capture reaches layer 48, then swap growth stops it

The capture records 344 module entries and 343 complete packed-cast/native/module returns: 342 NVFP4 operations plus the INT8 embedding. Its last event enters `layers.48.mlp.down_proj` at capture elapsed 112.2389 seconds. The guard stops at 118.760 seconds for measured system swap growth from 0.323486 to 1.775391 GiB, an increase of 1.451904 GiB against the 0.0625 GiB limit. Pressure is 2, free memory 35%, peak group RSS 3.563 GiB and final group RSS 0.650 GiB. Cumulative observed warning exposure is 10.3397 of 20 seconds; neither warning-budget exhaustion nor the 120-second deadline causes the stop. No encoder return, conditioning artifact, receipt or API attempt is present.

The last module event reports 64,938,752 MPS current bytes and 839,516,160 driver bytes; the observed driver peak is 839,843,840 bytes. Common monotonic timestamps place the terminal primary memory query 0.5688–0.5747 seconds after that event. Raw VM statistics support temporal comparison, but cannot attribute system-wide swap growth to a particular MPS allocation. There are 59 raw guard samples including admission and six pressure-2 rows; the policy counts 58 healthy observations and five warning observations because the terminal swap breach returns before incrementing those counters. All 165 source and 14 evidence pins, the completed 476.041-second review and parent 26-test/33-subtest evidence are verified.

At this earlier boundary, all failed attempts remained consumed and admission/transfer successors were under preparation. Raising the swap limit does not remedy the measured growth. The later results below preserve the earlier standalone H3 encoder acceptance and leave G2 API/video acceptance open.


## 2026-10-05: tiny H3 qdata-transfer prerequisite accepted

The reviewed synthetic mapped/demand differential completes normally in 11.976 seconds, with 2.156 GiB peak group RSS and flat 1.812871 GiB swap. Twelve ABBA cases produce exact outputs for plain NVFP4, AWQ NVFP4 and INT8 embedding fixtures. The complete test makes 12 native NVFP4 calls and six INT8 embedding calls through eager PyTorch row gathering on MPS, and reads 4,361,420,800 packed-data bytes. Original source ownership, versions and bytes remain unchanged. The reader closes before explicit final synchronization, and driver allocation settles exactly to its 44,433,408-byte baseline.

This is a transfer-ownership prerequisite on repeated rows and uniform scales. Original scale transfers may implicitly synchronize, so it does not prove all calls were concurrently pending. It does not establish a whole-encoder memory remedy, physical attribution, captured conditioning or API/video acceptance. The full H3 capture’s earlier swap-growth failure remains unchanged.

## 2026-10-05: WAN admission reaches the pre-body stop, but collection fails

The actual attempt returns both encodings and records 14 admission entry/return rows, ending in model-load return with 739,943,248 loaded bytes. Sequence 15 contains the intended exception before the first high expert body. The monitored run ends at 64.717 seconds with normal pressure, flat 1.812871 GiB swap and 9.703 GiB peak group RSS. Its original `observation_ok=false`, `generation_accepted=false` and consumed-attempt status remain authoritative.

Two independent collection assumptions cause rejection. The history error’s `current_outputs` lists all 31 graph IDs, so treating it as completed output falsely implies downstream execution. A progress-bar prefix hides the valid sequence-15 JSON from the anchored collector, while carriage-return fragments supply the two malformed-row counts. CPU replay reproduces both original reasons; correcting either assumption alone leaves one rejection, and correcting both clears only the forensic validator. It does not retroactively qualify the MPS attempt. Client failure triggers owned teardown during native sampling, leaving no completed stack artifact. This normal-pressure admission interval does not explain the earlier v4 pressure stop.

These earlier failures remain unchanged. The subsequently activated dedicated-journal WAN and full H3 qdata-capture attempts also fail, as recorded below.


## 2026-10-05: H3 qdata capture finishes layer returns, then fails metadata validation

The full capture records 351 qdata reads totaling 12,967,608,320 bytes, with 350 native NVFP4 returns and one eager PyTorch MPS INT8 embedding return. It exits unsuccessfully after 98.232 seconds with no guard reason, pressure 1 throughout the observations, zero warning exposure, flat 1.781621 GiB swap and 3.893 GiB peak group RSS. `Archive.validate` rejects the scheduled CLIP metadata before serialization: its expected key set excludes `pooled_output`, which the actual CLIP wrapper adds. The exact returned metadata contract is under separate CPU investigation.

There is no conditioning artifact or capture receipt, and the operation returns do not replace finite serialized conditioning or final source/admission validation. No full capture, memory remedy or API/video acceptance follows. The original failed outcome remains consumed; its group and workers are reaped. The earlier tiny transfer prerequisite remains accepted at its synthetic diagnostic scope.

## 2026-10-05: WAN admission v2 captures a native stack, then stops for pressure

The strict dedicated journal contains ten valid rows, ending at `_load_list` return at server elapsed 47.692293 seconds. It records no model-load return or intentional pre-body stop. The pressure guard ends the attempt at 74.559 seconds: pressure 2, free memory 40%, flat 1.781621 GiB swap and 9.709 GiB peak group RSS. The retained child MPS sample is 21.662 seconds stale at that boundary and cannot describe failure-time allocation.

The owned native sampler completes before teardown, preserving a 139,421-byte snapshot of Python fork and MPS copy waits. An interaction with telemetry is plausible; the exact lock cycle and its relation to pressure remain unproved. The snapshot does not establish expert-body execution. The failed outcome remains consumed, all workers and their group are reaped, and the original v1 collector failure is unchanged.

At this earlier boundary, metadata-contract and fork-free telemetry successors were CPU investigations only. The H3 metadata successor subsequently completes the accepted standalone capture below; the preceding failed attempts retain their original outcomes.


## 2026-10-05: H3 qdata capture produces an accepted encoder artifact

The v2 standalone capture completes normally in 74.721 seconds at 4.186218 GiB peak group RSS. All 38 observed pressure samples are 1, warning exposure is zero and swap stays at 1.781621 GiB. Each of 351 qdata reads executes once, totaling 12,967,608,320 bytes; all 350 native NVFP4 operations and the eager PyTorch MPS INT8 embedding return once. Source owners, versions and persistent storage remain unchanged.

A fresh parent CPU reload verifies finite FP32 conditioning `[1,388,5120]`, I64 modality tags `[388]` containing only ones, and the exact metadata keys `minimax_token_tags` and `pooled_output`, with the latter `None`. Tensor records match the saved capture report exactly. Producer 9310 is reaped and group 9307 is empty before the artifact is handed off.

This accepts a saved encoder artifact as a prerequisite for a separate-process workflow. No diffusion or VAE payload was loaded, no API/video result is accepted, and numerical reference parity remains unknown because there is no matching saved reference tensor. The v1 metadata failure, WAN v2 pressure failure and all earlier receipts remain unchanged. The H3 API successor is under CPU preparation, with no API activation. G2 H3/WAN14 API/video acceptance stays open.


## 2026-10-05: WAN fork-free admission passes with a deliberate pre-body stop

The v3 observation completes normally in 73.066 seconds at 9.933945 GiB peak group RSS, with pressure 1 and swap unchanged at 1.781621 GiB. Both encodings return. All 15 strict journal rows are present: seven admission entries, seven returns and the unconditional pre-body stop. High-model loading reaches 739,943,248 bytes within the 1 GiB budget; `_load_list` return to `load` return takes 1.356296 seconds. The exact stop propagates without calling the first high expert body. Parent admission and stack validators pass against the actual journal and reports, with no history timeout.

The owned sampler preserves a 127,926-byte native snapshot using one second of sampling at 10 ms intervals. Capture crosses the saved pre-body stop and shows idle event-loop/wait stacks, without the earlier fork/MPS-copy wait signature. This snapshot cannot prove the specific earlier lock cycle or its causal relation to pressure. Group 19604 is empty; all four workers and the native sampler are dead. Acceptance covers both encoders and bounded high loading only: no expert body, attention, sampling, low expert, decode or video is accepted.

The H3 saved encoder artifact remains accepted at its separate-process prerequisite scope. The WAN v1 collection failure, WAN v2 pressure failure and all earlier failed receipts remain unchanged. H3 API preparation has passed CPU checks and is under independent review; WAN full-graph work is CPU planning only. Neither successor is activated. G2 H3/WAN14 API/video acceptance remains open.


## 2026-10-05: H3 API replays conditioning, then stops for pressure

The first H3 API attempt using the accepted saved artifact replays finite CPU conditioning without constructing or invoking a real encoder. Diffusion admission loads 162,660,112 bytes within the 1 GiB budget. The outer guard then stops for system pressure at 66.675 seconds, with pressure 2, 3.303421 GiB peak group RSS and swap unchanged at 1.781621 GiB; observed free memory falls from 73% to 39%. No completed forward, sampling callback, decode or video is recorded.

The last of eight phase events is admission at server elapsed 28.970720 seconds. Its MPS allocation sample is at least 37.7 seconds old at failure and cannot establish current allocation or the pressure cause. Both VAE factory/decode counters are zero at that last phase; there is no inside-forward telemetry. All three workers are dead and process group 33995 is empty. The attempt remains failed and consumed.

The accepted H3 encoder capture, accepted WAN admission observation and all preceding failure receipts remain unchanged. At this earlier boundary, H3 CPU investigation had no proved remedy and WAN v5 was frozen under independent review. The subsequent WAN attempt is recorded below. G2 H3/WAN14 API/video acceptance remains open.


## 2026-10-05: WAN14 fullgraph v5 stops for pressure and rejects console qualification

The reviewed fork-free full-graph attempt stops for system pressure at 163.452 seconds, with 9.761597 GiB peak group RSS and swap unchanged at 1.781621 GiB. Native pressure reaches 2 and observed free memory falls from 72% to 40%. Both encodings return and high-model admission loads 739,943,248 bytes within the 1 GiB budget. The attempt remains failed and consumed, with no accepted full-graph observation or generation.

The strict console validator rejects the first high-expert entry record because a progress-bar prefix precedes its JSON on line 129. Raw extraction for investigation finds that entry at server elapsed 62.0788 seconds, followed by first-block query and feed-forward casts; the last nonperiodic event is the feed-forward cast return at 65.184 seconds. It finds 90 phase records: 77 periodic memory samples and 13 other events. Later casts of those kinds are deliberately unreported, so these representative markers do not identify the last block reached. This corrects an initial interpretation that omitted the embedded entry; it does not repair the qualification failure or turn raw console evidence into acceptance. No expert-forward return, sampler return, low-expert execution, decode or video is recorded.

The guard retains a fresh child sample with a recorded age of 0.554653 seconds at the native boundary: 1,186,203,136 MPS current bytes, 3,368,173,568 driver bytes and 1,179,156,480 process RSS bytes. This sample is unsynchronized and establishes neither a peak nor the allocation responsible for pressure. The exact pressure cause remains unknown; console contamination is a separate evidence-collection defect. All three workers are dead and process group 43362 is empty.

The accepted H3 encoder capture, accepted WAN admission observation, H3 API pressure failure and all preceding failure receipts remain unchanged. At this earlier boundary, H3 phase-only CPU implementation was prepared and a new WAN CPU investigation was at the planning stage. The subsequent H3 phase attempt is recorded below. G2 H3/WAN14 API/video acceptance remains open.


## 2026-10-05: H3 phase attempt reaches block 20, then stops for pressure

The phase-instrumented H3 API attempt stops for system pressure at 73.017 seconds, with 3.303177 GiB peak group RSS. Native pressure rises from 1 to 2 and observed free memory falls from 75% to 36%. Swap peaks at its initial 1.773809 GiB and ends lower at 1.765996 GiB. The monitor process exits 125, the server exits −15 and the client exits 1. The client reports deadline or cancellation, but the authoritative monitor reason is system memory pressure. The attempt remains failed and consumed.

The 132 phase rows record Python returns for diffusion blocks `dit.0` through `dit.19`; the last row is `dit.20` entry at server elapsed 70.703108 seconds. The first full diffusion forward never returns, and no video is produced. All phase memory observations are unsynchronized. These Python boundaries show progress through the instrumented code, not settled GPU completion, peak allocation or the allocation responsible for pressure. The pressure cause remains unknown.

No selected native call is observed in the instrumented first block. Source inspection supports the valid low-VRAM LoRA path through dequantization and `F.linear`; that route is inferred from source and does not qualify native-kernel execution. The phase validator rejects missing administrative boundaries. Its mandatory-native assumption and literal `mps` check against the observed `mps:0` are additional latent validation defects, neither of which caused the observed guard stop.

The completed independent review reports no findings. The parent accepts its correctness verdict while explicitly preserving two scope deviations: execution of a copied standard-library-only contract module, and reads of byte-identical helper source outside the declared review directory. No model or GPU execution occurred in that review; procedural compliance is not claimed. Two earlier `NO_SIGNAL` review attempts remain preserved and grant no authority. The actual runtime failure remains unchanged by the review verdict. At parent closeout, the process group is empty, server and runner are absent, and the permanent GPU lock is idle.

The accepted H3 encoder capture, accepted WAN admission observation, prior H3 API pressure failure, WAN v5 pressure and console failure and all preceding receipts remain unchanged. At this earlier boundary, WAN v7 CPU packaging corrections were under independent review. Its subsequent attempt is recorded below. G2 H3/WAN14 API/video acceptance remains open.


## 2026-10-05: WAN14 v7 phase attempt reaches block 23, then stops for pressure

The one reviewed v7 attempt stops for system memory pressure at 119.631 seconds. Native pressure rises from 1 to 2, observed free memory falls from 73% to 31%, and swap grows from 1.765996 to 2.004023 GiB, an increase of 243.74 MiB. Peak group RSS is 10.309509 GiB; the last observed group RSS is 1.038605 GiB. The attempt remains failed and consumed, with no retry authorized by its receipt.

First high-expert entry is recorded at server elapsed 47.8291 seconds. The stage stream contains exactly 281 rows and 107,170 bytes: Python blocks 0–22 return, then block 23 self-attention returns at 112.512966 seconds and cross-attention enters at 112.513382 seconds. The first full high-expert forward does not return. All stage rows are diagnostic and unsynchronized; no settled GPU completion, native-kernel qualification or video acceptance follows. Cross-attention entry is the last recorded boundary, not an established cause of pressure.

The coarse stream contains 70 rows and 54,753 bytes, including 68 phase records and two partial-loading records. Parent validation binds both streams to their owner, attempt, plan and final byte/row metadata after teardown. Both states retain `closed=false` because the guard interrupted execution; valid captured records do not establish complete qualification. The outcome retains `observation_ok=false` and `generation_accepted=false`. There is no completed history video output.

The guard’s child memory sample is marked fresh with a recorded age of 1.387595 seconds, but is unsynchronized. It records 1,141,079,808 MPS current bytes, 3,297,411,072 driver bytes and 1,067,089,920 process RSS bytes. These samples establish neither peak allocation nor the allocation responsible for system pressure. Source-level causality remains unknown. All owned workers and their group are gone, and the permanent GPU lock is idle at parent closeout.

The completed 681.512-second independent review has no blocking findings. The parent accepts that bounded source-review verdict while narrowing claims about complete source coverage and rejecting the claim that a generic restoration error necessarily latches a writer failure. The retained fail-closed live-snapshot limit is not relaxed. Review and CPU checks do not qualify this failed runtime attempt.

The accepted H3 encoder capture, accepted WAN admission observation, H3 phase pressure failure, WAN v5 pressure and console failure and all earlier receipts remain unchanged. At this earlier boundary, a smaller 512×288×22 H3 candidate was in CPU preparation. Its subsequent failed attempt is recorded below. G2 H3/WAN14 API/video acceptance is the only remaining project gate.


## 2026-10-05: smaller H3 still stops for pressure; WAN v8 review is invalid

The reviewed 512×288×22 H3 attempt stops for system pressure at 80.882 seconds, with 3.327911 GiB peak group RSS and swap unchanged at 1.933711 GiB. Native pressure rises from 1 to 2 and observed free memory falls from 75% to 38%. Its 134 phase rows record exactly 21 whole-block Python returns, `dit.0` through `dit.20`, then `dit.21` entry at server elapsed 79.644592 seconds. The first full diffusion forward does not return; there is no terminal journal, sampler/decode/history result or useful video. The attempt remains failed and consumed.

The canonical v2 parent receipt corrects the earlier summary’s whole-block filter, which had included six nested first-block operations. It preserves the original receipt hash and leaves every raw log, monitor result and artifact binding unchanged. Four selected first-block casts return ordinary dense BF16 MPS weights. Six SDPA rows form three Python entry/return pairs for the two refiners and the first diffusion block; the latter has sequence length 1,470. No native-helper row is observed. These unsynchronized records do not establish GPU settlement, native-helper qualification or full-forward correctness. Smaller counters do not explain physical pressure or identify the allocation that triggered it.

Eight coarse events retain conditioning replay with zero real encoder constructions and zero original audio/video VAE factory calls; diffusion admission remains 162,660,112 bytes. All owned processes and their group are absent, and the permanent GPU lock is idle at parent closeout. The allocation cause remains unknown.

An independent source review preceded this attempt and found nothing blocking. Its process record is in [review/2026-10-05-review-process-receipts.md](review/2026-10-05-review-process-receipts.md#h3-smaller-review). It does not change the failed GPU outcome.

The separate WAN v8 512×288×33 candidate passes 56 parent CPU tests in 4.571 seconds with its 2^26 attention cap. Its first independent review was stopped without a verdict and is invalid; no GPU execution or activation occurred. Process record: [review/2026-10-05-review-process-receipts.md](review/2026-10-05-review-process-receipts.md#wan-v8-first-review).

At this earlier boundary, H3 pressure and job-local read-only review permissions were under CPU investigation. The subsequent separate replacement review and bounded WAN attempt are recorded below. The accepted H3 encoder capture, accepted WAN admission observation, earlier H3 phase failure, WAN v7 phase failure and all preceding receipts remain unchanged. Neither useful video is accepted; G2 is the only remaining project gate.


## 2026-10-05: bounded WAN v8 still stops for pressure after a compliant replacement review

The one 512×288×33 WAN v8 attempt stops for system pressure at 80.197 seconds, with 10.108887 GiB peak group RSS and flat 1.925898 GiB swap. Native pressure rises from 1 to 2 and observed free memory falls from 74% to 34%. Last observed group RSS is 1.543320 GiB. The outcome retains `observation_ok=false`, `generation_accepted=false` and `retry_allowed=false`. Smaller geometry does not resolve the pressure gate.

The negative and positive CPU encodings return at server elapsed 41.1238 and 46.2136 seconds. Their observed process RSS is about 10.77 and 10.80 billion bytes. Encoder owners are no longer live before high sampling. At 46.5803 seconds the actual high-expert profile installs the 67,108,864-element cap on 40 self-attention modules, while the global default remains 2^31. The active cap is 2^26; inherited baseline route metadata is not its runtime source. High loading reaches 739,943,248 bytes within the 1 GiB budget. Two selected first-block casts return ordinary FP16 MPS weights with no separate LoRA side entries; this does not qualify the native INT8 helper.

The coarse stream has 52 rows, comprising 50 phase records and two target/loading records. The stage stream has 195 rows: whole Python blocks 0–15 return, then block 16 self-attention enters at 74.432472 seconds. Both owned streams independently match the final outcome and metadata, but remain unclosed and incomplete. These unsynchronized Python boundaries do not establish settled GPU completion. There is no full first high-expert return, low-expert completion, useful history/video output or full-shape native SDPA qualification.

The guard retains a fresh child sample with a recorded age of 0.329490 seconds: 1,153,711,616 MPS current bytes, 3,368,173,568 driver bytes, 1,591,672,832 process RSS bytes and 5,853,839,360 system-available bytes. The child reports pressure 2 and the same flat swap; its counters are unsynchronized. These measurements do not identify task footprint, compression or the allocation causing system pressure. Different host intervals and changed observation/geometry prevent a controlled claim of physical memory savings. All owned processes and group 36434 are absent, and the permanent GPU lock is idle at parent closeout.

A replacement source review completed with no blocking findings; it does not qualify the failed GPU result. Process record: [review/2026-10-05-review-process-receipts.md](review/2026-10-05-review-process-receipts.md#wan-v8-replacement-review).

At this earlier boundary, H3’s read-only comparison had found no proved remedy and a bounded footprint, compression and global-VM observer was in CPU preparation. Its subsequent attempt is recorded below. The accepted H3 encoder capture, accepted WAN admission observation, smaller H3 failure, first invalid WAN v8 review and all preceding receipts remain unchanged. Neither useful video is accepted; G2 is the only remaining project gate.


## 2026-10-05: H3 task-VM observation stops for pressure without a completed forward

The task-VM observation uses the same 512×288×22 H3 workload and unchanged guards. It stops for system pressure at 53.369 seconds, with 3.296631 GiB peak group RSS and flat 1.628906 GiB swap. Native pressure rises from 1 to 2 and observed free memory falls from 69% to 37%; last group RSS is 0.565262 GiB. All 122 phase rows retain the attempt and plan identity. Python blocks 0–14 return, followed by block 15 entry at server elapsed 51.987326 seconds. No first full forward, sampler return, decode, useful video or native qualification is established.

The 122 task queries cost 10.827711 ms in total, with a maximum of 1.033875 ms. Their PID identity and monotonic query boundaries verify. The monitor records 28 global observations and 27 VM queries costing 85.408588 ms. These measurements remain within the frozen observation budgets; they are diagnostic measurements, not generation acceptance.

Task footprint grows from 472,370,496 bytes at configuration to 5,235,008,000 bytes at the first whole-block return; task compressed bytes grow from zero to 1,716,273,152. During the later interval from block 0 return to block 14 return, footprint rises by only 28,983,488 bytes to 5,263,991,488, compressed bytes fall to 1,599,045,632, and driver allocation stays at 3,322,544,128. Process resident bytes fall from 1,733,754,880 to 614,055,936. The earlier growth and later interval must remain separate: these counters do not prove that H3 made no contribution to pressure, that no task allocation occurred, or that another process caused the stop. The completed offline comparison verifies the exact source bytes, task/global deltas and monotonic sample joins; it does not establish exclusive ownership of the global changes.

The matched late global window, records 14–26, shows compressor physical occupancy increasing by 67,095 pages (1.0238 GiB), while wired pages decrease by 32,431 (0.4949 GiB). This physical compressor gauge is distinct from stored-page representation and task compressed accounting. The surrounding task samples give the same qualitative result on both sides of the global intervals: rows 93–117 add 28,950,720 footprint bytes while compressed bytes fall by 179,929,088; rows 94–118 add 21,037,184 footprint bytes while compressed bytes fall by 165,789,696. Driver bytes are unchanged in both comparisons. These are observed brackets, not simultaneous measurements or bounds on unobserved peaks. The last global VM query ends 2.089128 seconds before the unsafe pressure reading; the final phase sample ends 0.117637 seconds before it. The unfinished block and the final global gap remain unobserved.

An independent source review found nothing blocking and left two nonblocking limitations: the cumulative telemetry-cost budget can stop a sufficiently slow observation, and an oversized supplemental diagnostic may be omitted after the workload is already stopping. Neither weakens the primary guard or accepts an incomplete run. All owned processes and their group are gone, and the GPU lock is idle at parent closeout. Process record: [review/2026-10-05-review-process-receipts.md](review/2026-10-05-review-process-receipts.md#h3-task-vm-review).

The completed comparison meets the predeclared stop rule: stop the local H3 diagnostic work and current local G2 runtime work under these conditions. No evidence-backed local retry or causal source fix is identified. Meaningful runtime qualification needs changed execution conditions or concrete evidence of an avoidable allocation; further resize or guard cycles are unsupported. This is an evidence boundary, not a permission gate, proof that the hardware can never fit the models, or attribution to another process. The bounded WAN v8 failure, accepted prerequisites and all prior receipts remain unchanged. G2 remains the only open project gate, with neither useful video accepted.


## 2026-10-05: range/API fixes and initial spectral contract

The improvement pass closes the reproduced GGUF factor overflow, MXFP8 mixed-scale loss, INT8 gradient bound overflow, empty contractions/dtype admission, vector masks, Conv3d temporal padding, independent-head native SDPA crash and input-axis DoRA column overflow. Each implementation has an independent review; combined native suites pass 354 Kitchen and 267 ComfyUI tests with normal pressure and flat swap.

MXFP8's corrected FP32 route costs 50–60% at the measured 4096-token layers, despite only 1–6% at the smaller fixtures. It also doubles the measured activation/GEMM/decoded buffers while respecting the existing byte ceiling. This is a correctness trade-off, not a performance improvement. A source-bound range admission flag needs separate load/mutation/clone/offload proof; per-call device readback is not an accepted shortcut.

The opt-in spectral adapter reuses the pinned MIT [LC123 implementation](https://github.com/lonecatone23/ComfyUI_LC123_nodes/blob/e73bdd3aca64c207c821d188ff17356f0963f205/lc_speed_boost.py), with actual Krea/CONST/Euler admission. Tiny CPU/MPS checks establish eight model calls, one resize, single outer noise scaling, independent DCT parity and unchanged global RNG. They do not establish learned-model quality or request speed. Real-image comparison remains the next evidence boundary; no spectral model acceptance or public submission is claimed.


## 2026-10-05: source-aware Wan conditioning and bounded Krea spectral experiment

Wan5B/14B headless cache keys now separate positive/negative text and bind local encoder stat identity, sources, runtime and policy. Nine CPU checks exercise collision/invalidation, serialization and both actual runners' hit/miss/off branches; independent review has no actionable findings. Old prompt-only caches are not reused.

The optional Krea spectral adapter passes 19 CPU/MPS checks, including actual tiny network forwards after real node preparation, but full checkpoint baselines at 1024px and changed 512px both hit the memory-pressure guard before completion. No boosted checkpoint render, speed or image-quality acceptance exists. Full-model retries are paused.

## 2026-10-06: ordinary Krea Turbo generation with bounded readonly residency

Eight-step native INT8 Turbo generation now completes through the ordinary headless entry, including fresh CPU encoding and VAE decode, with the shared readonly loader and a 1 GiB resident-weight cap. Recorded 512px repeats are exact in latent and decoded float image; the current default-cap 1024px cold run completes in 294.623 seconds (254.42 sampling), at 7.272 GiB peak process RSS with flat swap and an inspected useful image. All complete ordinary renders execute eight denoiser calls and 1,792 native BF16 activation calls. Larger caps and repeated API-server behavior remain separate acceptance boundaries.

Optional Krea conditioning reuse binds stronger encoder stat identity, tokenizer/source bytes and runtime/policy, and verifies exact loaded/published bytes. Completed cache controls reproduce exact images, but deadline stops and a slow 2 GiB replay prevent a reliable latency claim. Spectral's two 512px outputs repeat exactly and pass inspection with changed identity/pose details; neither earns a fresh-run speed benefit. The ordinary writable baseline pressure-stops before sampling, so four-step LoRA is deferred without bypassing readonly refusal. Final independent review finds no retained cache/residency defects.


## 2026-10-06: internal-checkpoint 1024px repeats and completed warm spectral sequence

After the storage controls, the unchanged ordinary Krea entry completes 1024px fresh encoding/off in 189.190s and a source-bound conditioning hit in 140.040s. Sampling is 135.39s/132.43s, decode 1.19s/1.17s, peak group RSS 7.393/1.323 GiB. Both have normal pressure, flat swap, eight denoiser calls, 1,792 native calls and 771,686,304-byte resident weights. Latent and decoded float image match each other and the earlier accepted 1024px output bit-exactly; parent inspection passes with the same cropped hands/foot and blurred rear hand. These natural-file-cache acceptance runs do not prove a universal speedup.

The previously unfinished headless same-process 512px baseline/baseline/spectral/spectral/baseline sequence now completes in 187.419s, peak RSS 1.357 GiB, normal pressure and flat swap. All 40 forwards and 8,960 native calls complete; three baselines and two spectral outputs each match their earlier respective tensor references exactly. Warm baseline mean sampling is 39.558s versus spectral 30.162s: about 24% lower sampling time in this one session. Spectral preserves a coherent high kick and scene but changes face identity/beard and framing. This earns a conditional warm-session trade-off, not image equivalence, fresh-process or API-server speed. Writable four-step LoRA remains deferred after its separate pressure stop; static readonly residency refusal is preserved.


## 2026-10-06: 1024px spectral bracket and second-subject quality

Four fresh ordinary-entry processes complete baseline/spectral/spectral/baseline at 1024px, with exact latent/image repeats within each route, all 32 denoiser forwards and 7,168 native INT8 calls. Mean sampling is 170.28s baseline versus 128.685s spectral, about a 24% reduction in this natural-cache bracket. Parent inspection passes the changed soccer composition. A seed 7 snowy-fox pair at 512px completes sixteen more forwards and 3,584 native calls with coherent anatomy/fur in both images. First fox request encodes while the second hits conditioning, so whole-request times are not compared. No runtime source changes.

The separate writable baseline adds the existing generic 1 GiB cap and 0.5 allocator fraction but pressure-stops at 17.513s before completing the first forward. Euler's 0/8 progress bar indicates sampling preparation returned; admission alone is not established as the failure phase. No adapter was loaded. Private mapped-weight streaming remains a hypothesis requiring persisted cap/prediction/footprint observations; unchanged retries and readonly-refusal bypass remain unjustified.

## 2026-10-06: immutable Krea four-step adapter and repeated readonly API

The explicitly admitted released rank64/four-step adapter now renders the 512px soccer/seed42 recipe with 896 native/fused-side calls, all 228 patched layers each step, finite inspected decode, normal pressure and flat swap. Shared ACCESS_READ construction mitigates the earlier writable pressure boundary without closing its exact OS cause. The first immutable successor completes but fails native qualification because LowVramPatch merges offloaded quantized weights into dense temporaries. The retained bench-only successor uses Comfy's existing temporary side representation across the whole forward/direct activation entry, restoring all state in finally; no vendor kernel or sampler change. Sampling 38.42s versus 43.82s baseline is one scoped comparison (about 12% lower), not 2× or a general default.

The dedicated localhost base 512px API profile passes initial/changed-seed/changed-prompt requests in 50.581/37.976/73.778s, 24 forwards/5,376 native calls, one retained model/load identity, expected graph-cache invalidation and finite inspected decode. Both fresh encoder calls release their objects before sampling; persistent conditioning reuse is disabled to prove this path. Aggregate 167.862s, peak RSS 7.431 GiB, normal pressure and flat swap; owned server/group stopped. Initial saved pixels match accepted headless baseline exactly. An earlier metadata guard failure is preserved; the correct EmptyLatentImage `downscale_ratio_spacial:8` is retained and a real CPU node-boundary test prevents recurrence.

Independent reviews resolve failed-load publication, skipped encoder proof and dead weakref accumulation, then report complete coverage/no actionable findings for final runtime/test deltas. Root 422 pass/3 skip/14 subtests and 25 GGUF-node tests pass. Static readonly continues to refuse LoRA. Adapter 1024/broader quality/repeated/API acceptance, mixed/default-server behavior and exact OS pressure cause remain separate.

## 2026-10-06: Spectral combines with the Krea Workbench

The 17-request GUI proof qualifies native INT8 with Spectral, three/five adapter files, Style Reference and one/two-image Identity Edit. Matched bracket sampling means are 94.46→79.06s for three LoRAs (16.3% lower), 129.99→95.68s for two-image Identity Edit (26.4% lower), and 99.27→103.45s for Style Reference (4.2% slower). Whole-request savings differ; these are conditional session observations. Exact repeats/base restoration pass; maximum observed RSS is 8.809 GiB with normal pressure/flat swap. Faces/framing and fine likeness change. Ordinary stays default; Style Spectral experimental and four-step integration parked.

## 2026-10-08: project reconciliation after Workbench and memory acceptance

The project sweep closes stale current-doc claims about Krea restoration/warm export and WAN14/H3 pressure blocks at their accepted scopes. WAN14 and H3 use specific bounded loaders; original untouched official templates remain unqualified. T1–T7 are complete at their stated training/portability/pilot scopes. The old same-subject learned-style ablation and the requested Identity Edit training exploration were restored as open work. No inference, training or new benchmark was run for this audit. All dated failures and original immutable receipts remain unchanged.

## 2026-11-09: pre-grant review

A review of this repo for grant use is at [docs/review/2026-11-09-pre-grant-review.md](review/2026-11-09-pre-grant-review.md). It changed the headline numbers as follows.

- The Krea 2 Turbo headline is now 512²: 8.7 s/step stock against 3.65 to 3.69 s/step on the hardened Metal backend (2.36 to 2.38x; the README rounds this to 3.7 s and 2.35x). The 1024 row stays caveated: the stock 35 s was one run during a download and the 16.5 s is older than the 14.91 s measured later. Superseded on 2026-11-09 by clean numbers at four sizes; see the four-way sweep entry of 2026-11-09 at the end of this file.
- Qwen-Image 2.1 1024² is the second headline: 15.1 to 8.8 s/step at 9.1 GB.
- The INT8 overhead over a resident fp16 GEMM is quoted from the hardened table (3.3 to 4.5 percent here, 1.4 to 2.1 percent in the comfy-kitchen PR writeup), not the 4 to 9 percent measured before hardening.
- Verification wording: kernel outputs are checked against a float32 or float64 reference; end-to-end runs are checked for NaNs and compared to stock by PSNR/SSIM where noted; timings are single-machine measurements.
- The wide-linear PyTorch figure is given per dtype: 3.52x fp16 and 8.83x fp32 at K=32768 on torch 2.14.1. The `_int_mm` statement above is scoped to torch 2.14.1 and earlier, since 2.15.0 ships one on MPS.
- The EXL3 27B ratio (2.4 to 4.4x slower than the ggml Metal kernels) now has its evidence in this repo: [docs/research/2026-10-25-exl3-prototype.md](research/2026-10-25-exl3-prototype.md) and its JSON, ported from the private checkout.
- Memory figures labeled "after sampling" are driver snapshots, not peaks.

## 2026-11-09: torch 2.15.0 `_int_mm` on MPS, measured

Throwaway venv with torch 2.15.0 (repo `.venv` untouched), macOS 27.0.1, same M1 Max. Five rounds, variants interleaved, three calls per round, median over 15 samples, `torch.mps.synchronize` after each call, GPU-serialized. Raw numbers: [review/2026-11-09-experiments/int-mm-2.15.json](review/2026-11-09-experiments/int-mm-2.15.json).

| Shape (M, K, N) | fp16 `F.linear` | bf16 `F.linear` | int8 dequant per call (fp16) + `F.linear` | `torch._int_mm` (int8 in, int32 out) |
|---|---|---|---|---|
| 4608, 6144, 6144 | 39.4 | 70.6 | 43.1 | 42.1 |
| 4608, 6144, 16384 | 104.5 | 186.0 | 111.4 | 113.4 |
| 4608, 16384, 6144 | 226.3 | 245.8 | 218.2 | 112.6 |

- `torch._int_mm` exists on MPS in 2.15.0 and its int32 output is bit-exact against an int64 CPU reference. The 2026-10-02 statement that it is missing holds for torch 2.14.1 and earlier only.
- It is not faster than fp16: 1.07x and 1.08x slower at the two K=6144 shapes, and within noise of this repo's weight-only kernel (39.5 and 104.7 ms, hardened table above) while still omitting the scale restore a real INT8 linear needs. On a GPU without integer matrix units, integer GEMM buys nothing over fp16, which is the premise of the weight-only design.
- At K=16384 `_int_mm` is 2.0x faster than fp16 `F.linear`, but that is the wide-K transposed-weight anomaly in `F.linear` (next entry), not integer compute. Against the bare fp16 GEMM (103 ms) and the kernel (107 ms) it is 5 to 9 percent slower.
- Dequantize-per-call in fp16 costs about 9 percent here. The 1.2 to 1.9x stock-ComfyUI penalty in the 2026-10-02 table is therefore mostly bf16 compute (1.8x slower than fp16 on this GPU) plus the dequantize, not the dequantize alone. The kernel removes both.

## 2026-11-09: wide-K transposed-weight slowdown, rechecked

`bench/repro_mps_wide_linear.py` on torch 2.14.1, GPU-serialized, host load average 14 to 22 during the run. Raw: [review/2026-11-09-experiments/wide-k-32769.json](review/2026-11-09-experiments/wide-k-32769.json).

- Still reproduces: at K=32768, `F.linear` is 3.23x slower than the contiguous pre-transposed matmul in fp16 and 7.89x in fp32, identical outputs. Slightly under the October 4 ratios (3.52x, 8.83x) because the fast path measured slower under today's host load; the slow-path times are unchanged (684 against 715 ms fp16).
- The slowdown grows smoothly with K and does not depend on K being a power of two: fp16 1.17x at 8192, 1.26x at 10240, 1.33x at 12288, 1.90x at 16384, 2.16x at 24576, 3.23x at 32768.
- K=32769 is a different regime: every variant, including the contiguous one, jumps to about 2.3 s (10.8x cliff over K=32768). That is the K > 32768 reroute on M1/M2 the script docstring warns about, not evidence about the transposed-weight bug. A K=32769 row in the PyTorch issue should be labeled as such.

## 2026-11-09: CUDA anchor for the GEMM table (Modal, NVIDIA)

Same three shapes on NVIDIA T4, A10G and A100-80GB through Modal, torch 2.15.0+cu132, median of 20 calls after 5 warmups. Full table and discussion: [modal/README.md](../modal/README.md); raw: [modal/results/2026-11-09-cuda-gemm-anchor.json](../modal/results/2026-11-09-cuda-gemm-anchor.json). This is cuBLAS against cuBLAS, a cross-platform column for the stock paths, not a Metal-versus-CUDA kernel comparison.

| Shape (M, K, N) | GPU | fp16 | bf16 | fp16, transposed weight | int8 dequant per call + fp16 | `_int_mm` |
|---|---|---|---|---|---|---|
| 4608, 6144, 6144 | T4 | 16.9 | 151.7 | 19.7 | 18.9 | 9.9 |
| 4608, 16384, 6144 | T4 | 52.6 | 497.4 | 80.7 | 60.5 | 30.6 |
| 4608, 6144, 6144 | A10G | 5.28 | 5.18 | 5.35 | 5.85 | 2.75 |
| 4608, 16384, 6144 | A10G | 13.7 | 13.6 | 15.4 | 15.2 | 6.96 |
| 4608, 6144, 6144 | A100 | 1.51 | 1.52 | 1.48 | 1.57 | 0.84 |
| 4608, 16384, 6144 | A100 | 3.55 | 3.61 | 3.59 | 4.28 | 2.53 |

- The transposed-weight penalty at K=16384 is 1.53x on T4 (Turing), 1.12x on A10G (Ampere) and absent on A100. That column times `F.linear(x, wt.t())` with the weight stored `[K, N]`, which is the layout MPS runs fast, so it is a slowdown in the opposite direction from the MPS one and is not a control for it. The control is the wide-K entry below.
- bf16 is 9x slower than fp16 on T4 and level with fp16 on A10G and A100. The M1 Max shows a smaller penalty of the same kind (1.75x). The cause was not investigated on either part. bf16-by-default is a datacenter assumption.
- True INT8 GEMM is 1.7 to 2.0x faster than fp16 wherever integer tensor units exist. That is the ceiling Apple M5 could bring and the M1 Max `_int_mm` above does not reach.
- fp8 `_scaled_mm` is unsupported on all three parts (needs sm89 or newer).

## 2026-11-09: Krea 2 Turbo four-way sweep, refused at admission and then stopped on swap

Attempted the clean four-way sweep the pre-grant review asked for (stock and Metal, each with sub-quadratic and SDPA attention, 512 and 1024, fresh process per run, shuffled order): `bench/krea_four_way.py`, new today, with `ATTN=quad` added to `bench/run_krea2.py` because the fork now defaults MPS to SDPA and the upstream sub-quadratic default has to be requested explicitly. Record: [review/2026-11-09-experiments/e1-krea-four-way-refused.jsonl](review/2026-11-09-experiments/e1-krea-four-way-refused.jsonl).

- First attempt, conditioning cache off: the memory monitor stopped the stock 512 run after 78 s during text-encoder load. Swap grew from 0.6 to 4.6 GB while the process group held 0.27 GiB; the internal disk went from 9.9 to 5.9 GiB free. Stop reason: swap growth over 4 GiB.
- Second attempt, conditioning captured once per cell (39 s, then cache hits) and a 2 GiB swap-growth cap: the first timed run (stock with sub-quadratic attention) was admitted with swap at 4.74 GiB and stopped after 60 s on swap growth, peaking at 8.30 GiB; the other eleven 512 runs were refused at admission with "swap-volume disk headroom". Swap stood at 8.3 GB and the internal disk at 3.5 GiB free.
- Cause: the system volume is nearly full, so macOS swap cannot grow, and the machine was carrying several other resident applications. The October runs had the same models and limits on a quieter machine. No timing was recorded, and none should be inferred from this entry. The Krea 1024 row keeps its caveat.
- Third attempt, after the disk was freed (27 GiB free) and with the conditioning cache: both timed runs that started were stopped by the monitor on swap growth, and the driver halted after the second. Stock with sub-quadratic attention at 1024 stopped at 113 s with swap up from 6.5 to 11.1 GiB. Metal with SDPA at 512 stopped at 125 s with swap up from 6.6 to 15.3 GiB. Record: [review/2026-11-09-experiments/e1-krea-four-way-stopped-swap.jsonl](review/2026-11-09-experiments/e1-krea-four-way-stopped-swap.jsonl).
- Cause of the third stop: a Colima Docker virtual machine with about twenty running containers held about 17 GiB (6 resident, 11 compressed), so even the 512 Metal cell pushed other memory to swap. Raising the swap cap would time the cells while the machine pages, which is the noise this sweep exists to remove. Still no timing recorded.
- To run: stop the Colima VM and other large resident applications, confirm swap is near zero, then from the repo root: `MK_ROOT=<models> .venv/bin/python bench/krea_four_way.py --out docs/review/2026-11-09-experiments/e1-krea-four-way.jsonl --sizes 512,1024 --repeats 3 --cond-cache auto --capture`. The driver appends as it goes, stops after two consecutive refusals or stops, and summarizes with `--summarize`. The sweep completed later the same day with an 8 GiB cap; see the next four-way entry at the end of this file.

## 2026-11-09: review should-fix pass

A documentation pass over section 3 and Appendix B of the [pre-grant review](review/2026-11-09-pre-grant-review.md). Nothing was measured; every figure below was already in this file.

- fp16 precision wording. The power-of-two row scaling is exact. The fp16 cast after it keeps full precision only down to about rowmax*2^-14, and the GEMM output carries fp16's 11 significand bits. The two comfy-kitchen PR write-ups under `upstream/` now say this in place of "changes no mantissa ... exact at subnormals".
- "Validation loss" is renamed "fixed training-set loss" in this file, in the trainer's console output and in the loss plots. It is computed on training images 0 and 7 at three noise levels, not on held-out data. The log key stays `val_loss`.
- The packed backward is within 5 to 8% of a dense fp16 GEMM on the three large Krea 2 Raw layers. The narrow attn.wk layer is 39% slower (3.2 against 2.3 ms). The earlier "2 to 8%" sentence is corrected in place.
- The 32% training-time advantage of the INT8 base over bf16 at 512px is confounded by dtype: bf16 GEMM is about 1.75x slower than fp16 on this M1 Max and no fp16 dense arm was run. The write-ups that gave "because bf16 is emulated" as the cause of that gap now state only the measured slowdown; the cause was not investigated.
- Spectral speedups are now quoted as about 24%, about 24% and about 12%. The run counts are small: two warm baselines against two spectral runs in one session at 512px, two against two across four fresh processes at 1024px, and one against one for the adapter case. The seconds they come from are unchanged.
- Four paragraphs describing how the October 5 source reviews were run (durations, tool-call counts, restriction violations) moved to [review/2026-10-05-review-process-receipts.md](review/2026-10-05-review-process-receipts.md), with pointers left in the entries.
- The Milestone 2 launchers moved to `train/historical/`. CONTRIBUTING.md now explains the thirteen test cases that skip without the pinned Krea2Edit runtime, how to set it up, and that the two Spectral cases among them also need the pinned LC helper.
- Still open and needing measurement: repeat-run variance with raw per-iteration logs, host thermal and load state in result lines, external baselines, LPIPS/CLIP on a prompt set, the one-block gradient check against fp32 on real weights, and an fp16 dense training arm. Two items on the review's list were measured the same day and have their own entries below: the K=16384 fp16 error and the packed W4/W6 backward at real Qwen widths.

## 2026-11-09: Web research on the pre-grant review's open questions

Nothing in this entry was measured. It is web research only, read on 2026-11-09.

- The write-up is [review/2026-11-09-gap-research.md](review/2026-11-09-gap-research.md). The source ledger, with each URL, source date and figure, is [review/2026-11-09-experiments/gap-research-sources.json](review/2026-11-09-experiments/gap-research-sources.json).
- Mozilla's own Human-Scale AI pages could not be read (HTTP 403). The grant facts come from two third-party summaries. Whether an individual applicant needs a US 501(c)(3) fiscal sponsor is unconfirmed. One summary says a selected team "must be able to receive funding from a U.S.-based 501(c)(3)", which Mozilla Foundation itself is. The other lists "individual" and fiscal sponsorship as separate applicant types.
- Siliconed, read from source, widens weights to fp32 on the CPU on pre-M5 chips. Its integer paths use Metal 4 `matmul2d`. Its own source comment says `matmul2d` is slower than MPS on M1 Pro (2.36 against 3.46 TFLOP/s, formats not named). On M5 the comment gives 6.0 TFLOP/s for bf16 `matmul2d` against 2.9 for MPS fp32.
- Siliconed's published M1 Pro times for Krea 2 Turbo at 1024 square (app 0.14, 2026-10-23, Standard) are 135.4 to 137.4 s per render with its Accelerators switch on and 144.6 to 144.9 s with it off. On M1 Pro that switch mainly means Neural Engine fp16 products. Those runs had Faster Rendering on, which runs 4 of the 8 evaluations on a half-size grid. They are not full-size renders and must not be compared per step with this repo's numbers.
- The review's "about 18 s/step in 2.2 GB on M1 Pro" Siliconed figure was not found in Siliconed's current docs and needs re-sourcing.
- What Radiant Canvas's "INT4 ConvRot" does internally is unconfirmed. The app is closed source and its published numbers are M5 Max only.
- No published Draw Things, stable-diffusion.cpp or Wan per-step number on M1 or M2 was found. The independent M1 Max numbers found are bf16 on a 64 GB machine.
- torch 2.15.0 ships a native MPS `_int_mm`. The only published timing is on M4 Pro and has no fp16 baseline, so our own M1 Max measurement ([review/2026-11-09-experiments/int-mm-2.15.json](review/2026-11-09-experiments/int-mm-2.15.json)) remains the only evidence for that verdict on M1.
- Modal does not publish a numeric CPU or memory maximum per container. Its per-container disk write quota defaults to 512 GiB, and `ephemeral_disk` can go up to 3.0 TiB. No open Prime Intellect compute grant was found.

## 2026-11-09: Per-layer weight error by format on two real checkpoints

Every 2-D linear weight of two public, ungated checkpoints was quantized in six formats on Modal CPU containers (4 CPU, 8 GiB, torch 2.15.0, 64 block jobs, 2229.3 s of job time) and compared with the original weights in fp32: Qwen-Image 2.1 transformer (bf16, `Comfy-Org/Qwen-Image-2.1`, 200 layers, 7,115,112,448 parameters) and Wan 2.1 T2V 1.3B (fp32, Apache-2.0, 306 layers, 1,417,641,984 parameters). INT8, W4A8 and W6A8 use a pure-torch port of the comfy-kitchen eager quantizers, fed fp32 weights; Q8_0 and Q4_K use ggml 0.25.3 without an importance matrix. This measures weights only, not image quality.

Relative Frobenius error per layer, percent (median / p95 / worst), and stored bits per weight (Qwen / Wan):

| Format | Bits (Qwen / Wan) | Qwen-Image 2.1 | Wan 2.1 1.3B |
|---|---:|---|---|
| INT8 per row, no rotation | 8.007 / 8.016 | 1.066 / 1.489 / 1.805 | 0.852 / 1.128 / 1.950 |
| INT8 ConvRot (shipped layout) | 8.007 / 8.016 | 0.869 / 0.956 / 1.006 | 0.811 / 0.910 / 0.914 |
| W6A8 | 6.507 / 6.516 | 2.064 / 2.083 / 2.093 | 2.054 / 2.073 / 2.094 |
| W4A8 | 4.507 / 4.516 | 7.307 / 7.311 / 7.312 | 7.300 / 7.310 / 7.328 |
| GGUF Q8_0 | 8.500 / 8.500 | 0.564 / 0.629 / 0.827 | 0.541 / 0.565 / 0.929 |
| GGUF Q4_K | 4.500 / 4.500 | 7.392 / 7.783 / 8.839 | 7.175 / 7.398 / 10.311 |

- The rotation lowers the Frobenius error in 198 of 199 Qwen layers and 199 of 306 Wan layers, and mostly trims the tail (p95 1.489 % to 0.956 % on Qwen). It raises the max-abs error in most layers: ConvRot has the lower max-abs in only 63 of 199 Qwen layers and 12 of 306 Wan layers. Why was not tested.
- W4A8 and Q4_K are level at about 4.5 bits. W4A8 is lower in 152 of 199 Qwen layers and 53 of 306 Wan layers. W4A8 has almost no spread between layers; Q4_K has a tail (worst 8.839 % and 10.311 %).
- Worst max-abs error: W4A8 0.187 (Qwen `transformer_blocks.0.attn.to_out.0`) and 0.184 (Wan `blocks.19.cross_attn.o`); Q4_K 0.131 and 0.145; INT8 ConvRot 0.022 and 0.019.
- Q8_0 has a lower weight error than INT8 ConvRot at 8.5 bits against about 8.01; it keeps one scale per 32 values where INT8 keeps one per row.
- Every layer took the fixed Lloyd-Max codebook (199 of 199, 306 of 306); the fitted path never fired.
- Check against the vendor quantizers: the port is bit-identical on three seeded CPU tensors (fixed codebook, fitted codebook, chunked rows). This was shown on macOS only; the private vendor code was not run in the container.
- Check against the publisher: the publisher's own Qwen INT8 ConvRot file agrees with ours in all but 828 of 6,979,321,856 codes (each off by 1; largest scale difference 4.66e-10).
- Local macOS (torch 2.14.1, ggml 0.26.0) against the Linux container (torch 2.15.0, ggml 0.25.3), same code and bit-identical inputs: per-row INT8 and Q8_0 are bit-identical. INT8 ConvRot codes are identical; one row scale of 96 differs on the heavy-tailed tensor. W4A8 fp32 channel scales differ in 54 of 96, 66 of 96 and 2689 of 4200 rows; on the 4200-row tensor (the chunked path that large real layers take) the stored W4A8 codes differ in 2 of 4200 rows and the group scales in 1 row; on the heavy-tailed tensor the fitted codebook differs in 10 of 64 bytes. W6A8 codes and group scales are identical; channel scales differ in 32 of 96, 31 of 96 and 1720 of 4200 rows. Q4_K differs in 19 and 46 of 55,296 bytes and in 137 of 4200 rows. The effect on the measured error is small: W4A8 7.2936486 % locally against 7.2936466 % in the container; Q4_K 7.05484 % against 7.05445 %, so the last printed digit of a Q4_K figure could differ by one elsewhere. The likely cause for the torch formats is BLAS summation order in the rotation; that was not isolated.
- Limit on W4A8 and W6A8: the rotation ran in fp32. The vendor rotation runs in the dtype it is given, and the repo's on-load paths can be handed bf16 weights. That was not measured on the checkpoints. On the three seeded tensors a bf16 input moves W6A8 from 2.055 % to 2.061 %, 2.787 % to 2.793 % and 2.049 % to 2.056 %, and W4A8 from 7.292 % to 7.289 %, 7.297 % to 7.297 % and 7.294 % to 7.296 %.
- Not measured: Qwen `img_in` (64 input columns, below the 256 group) for ConvRot, W4A8, W6A8 and Q4_K; the Wan 5-D `patch_embedding`; Krea 2.

Raw: [modal/results/2026-11-09-weight-error.json](../modal/results/2026-11-09-weight-error.json), [validation](../modal/results/2026-11-09-weight-error-validation.json), [container vectors](../modal/results/2026-11-09-weight-error-selftest-container.json). Write-up: [research/2026-11-09-weight-error-by-format.md](research/2026-11-09-weight-error-by-format.md).

## 2026-11-09: MPS fp16 GEMM accumulates in fp32, and the K=16384 fp16 error bound

**What was run.** `bench/probe_fp16_accumulation.py` on an Apple M1 Max, macOS 27.0.1, torch 2.14.1, through `bench/with_gpu.sh` (24.4 s, about 1.04 GiB of GPU memory). Only float16 and float32 tensors are used; bf16-valued activations are emulated as 8-bit significands. Every MPS result is compared with an fp64 CPU oracle and with numpy simulations that round every addition to fp16 or to fp32 (sequential, pairwise, 2 to 16 lanes, blocks of 64 and 1024). Eleven shapes from 1 x 16 x 1 to 512 x 16384 x 512 and 8 x 32768 x 8, and three call forms: `F.linear(x, w)`, `x @ wt` with a contiguous transposed weight, and `x @ w.t()`. Raw output: [fp16-accumulation.json](review/2026-11-09-experiments/fp16-accumulation.json).

**Question 1: does the fp16 GEMM accumulate in fp16?** No.

- A case separates the two hypotheses when every fp16-accumulation simulation gives a wrong answer and every fp32 simulation gives the right one. That holds for 33 shape-pattern pairs, 99 MPS results. 99 of 99 equal the exactly rounded value. 0 match any fp16 simulation.
- Which patterns separate, out of 11 shapes each: a single product above 65504 that is cancelled later (11), a product whose low bits do not fit in fp16 (11), small terms absorbed after 1.0 (6, all with K up to 4096), alternating cancellation of 2048 against 0.25 (5, all with K of 16384 or more), a partial sum above 65504 that cancels (0).
- Separating examples at 512 x 16384 x 512: the single product above 65504 returns 3.0 with no inf or nan; the low-bits product returns 2^-20; alternating cancellation returns 2048.0, exact.
- Two patterns do not separate at that shape and are not evidence there. Small terms after 1.0 return 5.0, which fp16 pairwise order also gives. The partial sum above 65504 returns 0.0, which fp16 accumulation in 4, 8 or 16 lanes also gives.
- Structured inputs built to defeat fp32 accumulation: 22 pairs, 66 MPS results, only 4 exact. At 512 x 16384 x 512 one pattern returns 4.0 against an exact 7.999 (8.0 in fp16). So partial sums are held in fp32, not wider, at some stage of the reduction. This does not rule out a wider accumulator inside a block whose block sums are then combined in fp32.
- On those patterns the results change with shape and call form. Of the 66 results, 47 match a simulated fp32 order: 19 match sequential order (64 x 1024 x 64, 512 x 512 x 512, 512 x 4096 x 512, and one call form at 3 x 1023 x 5), 28 match only a lane, pairwise or block order. 19 match no simulated order. The summation order is not identified. These patterns do not separate fp16 from fp32 (the sequential-order matches also match fp16 simulations).
- Random data with mild cancellation, share of MPS outputs equal to the correctly rounded fp64 result: at least 0.9918 for Gaussian inputs, 0.9991 for positive uniform inputs and 0.984 for inputs in the kernel's normalized range, on every shape. At 512 x 16384 x 512 the fp16 sequential simulation is correct on 0.0117 of Gaussian outputs with a maximum error of 3273 ulp, and on none of the positive uniform outputs (maximum 904 ulp); MPS has a maximum of 0.503 ulp on positive uniform. fp16 pairwise order is 72 percent correct there (maximum 1.26 ulp), so this test rules out sequential and lane orders, and the structured cases rule out pairwise.
- Random data with heavy cancellation (partial sums near 2^16, results of order 10): MPS is not correctly rounded here. At 512 x 16384 x 512 only 12 percent of MPS outputs equal the rounded fp64 result; the fp32 simulations range from 10 to 62 percent on the simulated corner, and the fp16 simulations reach none. That is what fp32 partial sums predict. The largest MPS error is 1.7e-7 of the sum of absolute products over all shapes. On the simulated subset at K >= 4096, MPS and the fp32 simulations have the same maximum, 8.6e-8, and the best fp16 simulation is 1.1e-6.

Reading: on this stack the fp16 GEMM computes products and partial sums in fp32 and rounds once to fp16 at the end. "Rounds once" does not mean correctly rounded: the fp32 sum carries its own error, visible under heavy cancellation. The limit `_FP16_MAX_K = 65504` and the row normalization protect the final fp16 output, not the partial sums.

**Question 2: the K=16384 fp16 error bound.** Shape 512 x 16384 x 512, int8 weights with per-row scales, activations normalized per row by a power of two as the kernel does. The three call forms give identical statistics.

What 11 bits imply: a correctly rounded fp16 output has a relative error of at most 2^-11 = 4.88e-4 per element against the exact product of the fp16 operands, for outputs in the fp16 normal range (magnitude at least 2^-14 after normalization). Subnormal outputs are worse: the ideal rounding has a per-element maximum of 0.016 on Gaussian inputs. The measured ideal value is 4.7e-4 at p99.9, with a relative L2 of 2.07e-4.

Measured, MPS GEMM alone against the exact product of the same fp16 operands (Gaussian activations): 99.16 percent of outputs are correctly rounded, 99.92 percent are within 1 ulp, the maximum is 85 ulp. Relative error median 1.7e-4, p99.9 8.0e-4, relative L2 2.07e-4.

Measured, end to end against fp64 on the unrounded operands:

| Arm | Activations | Relative L2 | Max error / output RMS | p99.9 error / output RMS | Per-element relative, p99.9 | Per-element relative, max |
|---|---|---|---|---|---|---|
| emulated | Gaussian, fp16 values | 2.94e-4 | 1.97e-3 | 1.16e-3 | 0.124 | 37.5 |
| emulated | Gaussian, bf16 values | 2.92e-4 | 2.06e-3 | 1.14e-3 | 0.135 | 41.7 |
| emulated | Laplace, bf16 values | 2.94e-4 | 1.83e-3 | 1.21e-3 | 0.125 | 43.8 |
| emulated | one outlier of 16000 per row | 2.92e-4 | 2.97e-3 | 1.43e-3 | 1.90e-3 | 0.707 |
| emulated | one outlier of 2^20 per row | 2.30e-4 | 1.79e-3 | 9.7e-4 | 0.285 | 105 |
| kernel, no ConvRot | Gaussian | 2.94e-4 | 2.16e-3 | 1.15e-3 | 0.123 | 15.3 |
| kernel, no ConvRot | one outlier of 16000 per row | 2.94e-4 | 2.88e-3 | 1.43e-3 | 1.96e-3 | 0.358 |
| kernel, ConvRot 256 | Gaussian | 3.59e-4 | 2.14e-3 | 1.30e-3 | 0.171 | 66.5 |
| kernel, ConvRot 256 | one outlier of 16000 per row | 3.30e-4 | 2.13e-3 | 1.24e-3 | 0.165 | 78.0 |

The kernel rows call `int8_linear` from the vendored comfy-kitchen (commit b5dd528, clean tree). fp32 and fp16 output give identical statistics. No output is non-finite.

Each inexact step alone, computed on the CPU in fp64 (emulated arm, error RMS as a share of the stated reference):

| Activations | Reference | Scaling only | Activation cast only | Weight cast only | Output rounding only | All together (MPS) |
|---|---|---|---|---|---|---|
| Gaussian, fp16 values | whole output | 0 | 2.6e-9 | 2.07e-4 | 2.08e-4 | 2.94e-4 |
| one outlier of 16000 per row | whole output | 0 | 8.1e-7 | 2.07e-4 | 2.07e-4 | 2.92e-4 |
| one outlier of 16000 per row | output of the rest of the row | 0 | 1.0e-4 | 2.58e-2 | 2.59e-2 | 3.65e-2 |
| one outlier of 2^20 per row | whole output | 0 | 4.4e-6 | 2.07e-4 | 2.08e-4 | 2.30e-4 |
| one outlier of 2^20 per row | output of the rest of the row | 0 | 3.6e-2 | 1.70 | 1.70 | 1.88 |

What it means:

- The per-element relative error has no useful maximum. Outputs that cancel to near zero carry an absolute error of about 3e-4 of output RMS, so their relative error is large. The usable bound is absolute: no element is off by more than about 3e-3 of output RMS, and 99.9 percent are within about 1.4e-3.
- The error has two equal parts: the fp16 cast of the decoded weights (it changes 98.7 percent of weight elements; relative L2 2.07e-4) and the 11-bit output rounding (2.07e-4). Together about 2.9e-4. The activation cast is not a part that matters for ordinary inputs (2.6e-9), and fp16 accumulation is not a part at all, because there is none.
- The review caveat, with numbers. The power-of-two scaling is exact: with exact operands and an unrounded output, the error is 0 in every row. The fp16 cast of the activations after it is not exact, but it is small. With one activation of 16000 in a row of unit-scale values, 68 percent of the scaled activations fall below 2^-14 and 6.6 percent change in the cast, yet that cast costs only 1.0e-4 of the output that the rest of the row carries. At an outlier of 2^20, 96 percent of activations change, 5.0 percent flush to zero, and the cast costs 3.6e-2 of it.
- A dominant outlier still hides the rest of the row, for a different reason. The outlier's own product dominates the output, and it is carried at fp16 precision twice: in the fp16 weight and in the 11-bit output. Each costs about 2.07e-4 of the whole output. The rest of the row is 8.0e-3 of the output at an outlier of 16000 and 1.2e-4 at 2^20, so the same absolute error is 3.65 percent of the rest-of-row signal at 16000 (3.68 percent through the kernel, 4.12 percent with ConvRot 256) and 188 percent at 2^20. At 2^20 the rest of the row is lost, while the error against the full output still reads 2.3e-4. Whole-output metrics do not show this.

**Not covered.** Synthetic activations only; no checkpoint was loaded, so how often real rows have such an outlier is not measured. torch 2.14.1 and one chip (M1 Max) only. The bf16 tensor input path of the kernel was not exercised. Whether the kernel's fp32 compute path avoids the outlier loss was not measured.

## 2026-11-09: Packed W4A8/W6A8 backward checked against fp64 at real Qwen widths

The pre-grant review noted that the packed backward had only been checked numerically at n=67 and K up to 768. This closes that gap for single layers and for one MLP pair.

**What was run.** `bench/probe_packed_backward_real_k.py`, through `bench/with_gpu.sh`, on an M1 Max (32 GB, torch 2.14.1, comfy-kitchen b5dd528). For each layer it draws a synthetic weight (normal times a log-normal mixture, log-normal row and column scales, std about 0.019, kurtosis 16 to 24), quantizes it with the repo's own eager quantizers, and calls the MPS input-gradient kernel (`int8_linear_grad_input`, `w4a8_linear_grad_input`). The reference is fp64 on CPU: the same gradient times the weight decoded in fp64 from the same packed tensors. So the numbers below are kernel arithmetic error only. Quantization error is separate and is not what this measures.

- Formats: INT8 ConvRot, W6A8 (group 32), W4A8 (group 16, codebook), W4A8 asymmetric.
- Shapes (K to N): 3072 to 3072, 3072 to 12288, 12288 to 3072, 12288 to 12288 (the review's grid), plus the three shapes in the real Qwen-Image 2.1 INT8 header: 4096 to 4096, 4096 to 24576 (`img_mlp` gate_up), 12288 to 4096 (`img_mlp` out). Only the header was read; no checkpoint weights were loaded.
- M = 512 tokens. Gradient dtypes bf16 (scale 2e-5), fp16 (scale 1.0), fp32 (scale 2e-5). The 256 MB decode bound split the backward into 1 to 3 chunks depending on shape.
- Largest sampled MPS driver memory 2.055 GiB (sampled at sync points). Wall time 2 minutes. No non-finite values in any of the 84 cases.

**Single-layer input gradient, range over all 28 layer cases per dtype (relative L2 against fp64):**

| gradient dtype | kernel | fp64 rounded to that dtype (floor) | dense matmul on MPS, same dtype | kernel with fp32 output |
|---|---|---|---|---|
| bf16 | 1.676e-3 to 1.691e-3 | 1.650e-3 to 1.665e-3 | 2.334e-3 to 2.362e-3 | 2.914e-4 to 2.947e-4 |
| fp16 | 3.568e-4 to 3.600e-4 | 2.063e-4 to 2.092e-4 | 2.920e-4 to 2.948e-4 | 2.909e-4 to 2.941e-4 |
| fp32 | 9.654e-7 to 1.625e-6 | 2.523e-8 to 2.555e-8 | not run | same as kernel |

Cosine similarity rounds to 1 in every case; 1 minus cosine is at most 1.43e-6 (bf16). The worst single token row is 1.80e-3 relative L2 in bf16. The largest absolute error is 0.128 on an fp16 gradient whose reference peak is 306.9; in bf16 at training scale it is at most 1.5e-5. The autograd path returned the same tensor as the direct kernel call in every bf16 case.

**MLP pair with LoRA (4096 to 24576, SwiGLU, 12288 to 4096, rank 16, M = 256).** Relative L2 of the LoRA A and B gradients of both layers against a hand-written fp64 chain:

| activations | kernel path | dense MPS path, same dtype |
|---|---|---|
| bf16 | 5.08e-3 to 5.31e-3 | 5.62e-3 to 5.84e-3 |
| fp32 | 1.72e-6 to 1.78e-6 | 1.73e-6 to 1.78e-6 |

The input gradient of the pair is 4.81e-3 to 4.83e-3 (bf16 kernel) against 5.67e-3 to 5.70e-3 (bf16 dense).

**What it means.**

- The packed backward is correct at Qwen's real widths. Within this probe (K from 3072 to 12288, N up to 24576, 1 to 3 decode chunks) the error does not depend on width or chunk count in bf16 or fp16; with fp32 gradients it grows from 9.7e-7 to 1.6e-6 with N. Widths below 3072 were not rerun here. All four formats give the same error to within 1 percent of each other, so the 4-bit and 6-bit unpack adds no error of its own.
- In bf16 the kernel sits about 1.5 percent above the best any bf16 output can do. It reads lower than the dense bf16 matmul on MPS, but that comparator stores the dequantized weight in bf16 (8 significand bits) while the kernel decodes to fp16 (11 bits), so the gap mostly reflects weight storage, not better arithmetic. The error a trainer sees is bf16 rounding of the result, not the kernel.
- In fp16 the kernel is not at the floor: 3.57e-4 to 3.60e-4 against 2.06e-4 to 2.09e-4, and about 22 percent above a dense fp16 matmul on MPS.
- The kernel's own contribution is the fp16 GEMM, about 2.9e-4, visible when the output is fp32. With fp32 gradients it drops to about 1e-6.
- For scale: the quantization error of these synthetic weights (relative L2 of dequantized against source) is 0.0093 to 0.0100 for INT8, 0.0231 to 0.0233 for W6A8, and 0.071 to 0.073 for W4A8. The bf16 backward arithmetic error is 5.5 to 43 times smaller than that.

**Limits.** The weights are synthetic, not Qwen's. The review's "3072" does not appear among the quantized layers in the Qwen-Image 2.1 INT8 header, whose widths are 4096 and 12288; unquantized layers were not checked. Both sets were run. This is one layer and one MLP pair. It says nothing about error accumulated through the full depth of the model, which is a separate open item. No timings were recorded because the host was under heavy load (load average 21 to 37).

Raw: [review/2026-11-09-experiments/packed-backward-real-k.json](review/2026-11-09-experiments/packed-backward-real-k.json). The `command` field in that file is a fixed string, not a captured command line.

## 2026-11-09: CUDA control for the wide-K slowdown (partial: T4 at two widths only)

`bench/repro_mps_wide_linear.py` now takes `--device` (default `mps`), and `modal/modal_app.py` runs the same script on CUDA (`cuda_wide_k_t4`, `_a10g`, `_a100`): same eight variants, interleaved rounds, minimum over 3 rounds x 5 calls, M=4608, N=6144. Raw: [../modal/results/2026-11-09-cuda-wide-k.json](../modal/results/2026-11-09-cuda-wide-k.json). The T4 object in that file is the script's output; the underscore-prefixed keys and the not-run entries were added by hand.

Only a two-width check on a Tesla T4 (torch 2.15.0+cu132) was measured. The full sweep (K up to 32768 plus 32769, on T4, A10G and A100) was refused by the session's permission layer before launch, twice, and has not run. A10G and A100 have no measurement.

Milliseconds per call; ratio is `F.linear(x, w)` over `x @ wt` (contiguous). M1 Max figures are from [review/2026-11-09-experiments/wide-k-32769.json](review/2026-11-09-experiments/wide-k-32769.json), torch 2.14.1.

| dtype | K | T4 F.linear | T4 x @ wt | T4 ratio | M1 Max F.linear | M1 Max x @ wt | M1 Max ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 | 4096 | 9.1 | 9.8 | 0.93 | 25.3 | 26.1 | 0.97 |
| fp16 | 8192 | 17.7 | 20.3 | 0.87 | 60.2 | 51.4 | 1.17 |
| fp32 | 4096 | 49.4 | 51.2 | 0.96 | 44.0 | 27.7 | 1.59 |
| fp32 | 8192 | 102.9 | 116.3 | 0.89 | 141.2 | 56.1 | 2.52 |

What it means. At K=4096 and 8192 the T4 does not show the MPS effect: `F.linear` with the standard `[N, K]` weight is 4 to 13 percent faster than the pre-transposed contiguous matmul, where the M1 Max is already 1.17x slower in fp16 at K=8192 and 1.59x and 2.52x slower in fp32. On T4 the variants differ by up to 0.0039 in fp16 (0.0 at K=4096) and 2.3e-05 in fp32; on MPS they are bit-identical. The widths where the MPS gap is large (16384, 32768; 3.23x in fp16 and 7.89x in fp32 at K=32768 in the same day's M1 Max rerun) and the reroute width (32769) are not measured on CUDA, so "CUDA is about 1.0x at K=32768" remains untested.

Only the ratios are meant to be compared. Absolute milliseconds vary between T4 containers (the GEMM anchor measured 16.8 ms median for fp16 `F.linear` at K=6144; this run measured 17.7 ms minimum at K=8192 with the same M and N) and are not a speed comparison against the M1 Max. The T4 fp32 rows do not record the TF32 setting (T4 has no TF32; torch's default is off); the function now records it for the full run.

A wording point for the earlier anchor: its "fp16, transposed weight" column times `F.linear(x, wt.t())` with the weight stored `[K, N]`, the layout MPS runs fast. Its 1.53x on T4 at K=16384 ([../modal/results/2026-11-09-cuda-gemm-anchor.json](../modal/results/2026-11-09-cuda-gemm-anchor.json)) is a slowdown in the opposite direction from the MPS one and is not a control for it. The earlier entry is corrected in place.

To finish: `modal run modal/modal_app.py --wide-k --gpus T4,A10G,A100 --out modal/results/2026-11-09-cuda-wide-k.json`.

## 2026-11-09: Krea 2 Turbo four-way sweep, completed at four sizes

The sweep the pre-grant review asked for (E1) ran to completion after the Colima VM and other large applications were stopped. `bench/krea_four_way.py` runs every cell in a fresh process through `bench/gpu_runner.py --monitor`, in a shuffled order, with host state recorded before each run. Krea 2 Turbo INT8 ConvRot, 8 steps, cfg 1, seed 42, text conditioning captured once per cell and reused. M1 Max 32 GB, torch 2.14.1, macOS 27.0.1.

What the four cells are:

- "stock" is this repo's ComfyUI fork with the Metal formats switched off (`MODE=stock`): weights are dequantized on every call and computed in bf16. It is not a separate upstream checkout.
- "kernel" is the same fork with the comfy-kitchen mps backend (`MODE=metal`).
- Each runs with ComfyUI's sub-quadratic attention (`ATTN=quad`) and with PyTorch SDPA (`ATTN=pytorch`). Stock with sub-quadratic attention is the closest cell to upstream's default on Apple GPUs.

How a step is counted: `bench/run_krea2.py` reports the intervals between sampler callbacks, so 8 steps give 7 intervals and the first step, which includes the weight load, is never in them. The figure per run is the median of the intervals after dropping the first.

Files, all under `docs/review/2026-11-09-experiments/`:

- `e1-krea-four-way-pass2.jsonl`: 512 and 1024, three repeats, 24 of 24 runs completed. `--sizes 512,1024 --repeats 3 --cond-cache auto --capture --swap-growth-gib 8`.
- `e1-krea-768-check.jsonl`: 768, two repeats, 8 of 8. Same flags with `--sizes 768 --repeats 2`.
- `e1-krea-256-check.jsonl`: 256, two repeats, 8 of 8. Same flags with `--sizes 256 --repeats 2`.
- `e1-krea-four-way.jsonl`: pass 1 with the default 4 GiB swap-growth cap, stopped by hand after 12 of 24 runs had started. Six completed and six were stopped by the monitor (below).
- `e1-krea-predictions.md`: the 768 and 256 predictions, written before those runs.

Seconds per step, lowest to highest run in each cell. GB is `driver_allocated_memory` read once after sampling and before VAE decode; it is not a peak and it excludes the text encoder.

| Size | Stock, sub-quadratic | Stock, SDPA | Kernel, sub-quadratic | Kernel, SDPA | Speedup |
| --- | --- | --- | --- | --- | --- |
| 256 (n=2) | 4.27 to 4.29, 15.1 GB | 4.25 to 4.29, 15.1 GB | 1.23, 14.1 GB | 1.22, 13.1 GB | 3.5x |
| 512 (n=3) | 8.32 to 8.57, 15.1 GB | 8.35 to 8.56, 15.1 GB | 3.75 to 3.76, 14.1 GB | 3.76 to 3.78, 14.1 GB | 2.2x to 2.3x |
| 768 (n=2) | 15.96 to 16.17, 16.2 GB | 15.89 to 15.90, 15.2 GB | 8.41 to 8.48, 15.2 GB | 8.54 to 8.57, 14.2 GB | about 1.9x |
| 1024 (n=3) | 27.89 to 27.92, 19.3 to 20.9 GB | 27.13 to 28.01, 16.3 GB | 16.08 to 16.12, 18.3 to 19.8 GB | 15.87 to 16.08, 15.2 GB | about 1.7x |

What this shows:

- **The speedup shrinks as the image grows.** The gap between stock and kernel is about 3.05 s/step at 256, 4.6 to 4.8 at 512, 7.3 to 7.8 at 768 and 11.3 to 11.8 at 1024 (the low ends pair the faster stock runs after 14:00). With M = (side/16)^2 + 46 tokens (302, 1070, 2350, 4142) that fits a fixed cost of about 2.3 to 2.4 s/step plus about 2.2 ms per token. The 2.26 s plus 2.17 ms figure in `e1-krea-predictions.md` is fitted to the after-14:00 SDPA runs at 512 and 1024 only. A fixed cost is consistent with stock dequantizing and un-rotating every weight on every call, which does not depend on image size. No profile was taken, so this is a fit, not a measured breakdown.
- **Predictions written before the two extra sizes mostly held.** They are in `e1-krea-predictions.md`. The 768 prediction used an earlier fit to the 512 and 1024 runs (stock as a fixed 2.10 s plus a per-pixel term, text tokens ignored) and both halves held (stock 16.6 predicted, band 15.8 to 17.4; kernel 8.70, band 8.3 to 9.1). The 256 prediction used the token fit above, made after 768: stock held (3.97 predicted, band 3.6 to 4.4), the kernel missed narrowly (1.05 predicted, band 0.9 to 1.2, measured 1.22 to 1.23), and the gap came in at 3.05 s against 2.91 predicted. The alternative with no fixed cost predicted stock at 1.7 to 2.4 s/step at 256 and is ruled out.
- **Attention mode does not change speed on Krea today.** Within each of stock and kernel the two attention modes are within about 2 percent of each other at every size, with no consistent direction. October's "24 s/step with the kernel alone, 16.5 with SDPA" at 1024 does not reproduce: kernel with sub-quadratic attention measures 16.1. The October run was probably under memory pressure; that is not proven. One confound in today's 1024 data: all six sub-quadratic runs started before 13:50 and five of the six SDPA runs started at 13:54 or later, so attention mode and time of day are partly tied together there. Stock runs late in the sweep were about 3 percent faster than early ones (512: 8.32 to 8.35 against 8.54 to 8.57; 1024 SDPA: 27.13 against 28.01); the cause is unknown.
- **Memory.** The kernel holds about 1 GB less than stock after sampling in the like-for-like SDPA cells at 512, 768 and 1024, and 2 GB less at 256. Readings in the SDPA cells and at 256 to 768 move in steps of about 1 GB; the 1024 sub-quadratic readings move by about 0.5 and 1.5 GB between runs. Sub-quadratic attention holds 3 to 5 GB more than SDPA at 1024 on either path, and that reading is not repeatable: kernel 18.30, 19.83, 18.30 in pass 2 and 19.83, 20.37 in pass 1; stock 20.85, 20.86, 19.32.
- **The stock 1024 figure is now clean.** "About 35 s, one run during a download" (line 36, flagged at line 45) is replaced by 27.1 to 28.0 over six runs.
- **The kernel is slower than its October best.** Kernel with SDPA at 1024 was 14.91 s/step (line 331) and 15.02 to 15.06 (line 295) in October and is 15.87 to 16.08 today, 5 to 8 percent slower. At 512 it was 3.65 to 3.69 (line 83) and is 3.75 to 3.78 in pass 2, 2 to 4 percent slower; one pass 1 run today measured 3.69. The cause is not established. macOS went from 15.7 to 27.0.1 and the vendored ComfyUI and comfy-kitchen have commits since. The linear-layer microbenchmark has not slowed (below), so the single-layer decode and GEMM at Krea shapes are not the cause.

Repeats: three per cell at 512 and 1024 and two at 256 and 768, not the five the review asked for. The ranges inside a cell are up to 1.3 percent for the kernel (1024 SDPA; under 1 percent in the other seven kernel cells) and up to about 3 percent for stock, so two more repeats would not change a headline.

### The swap guard, pass 1 against pass 2

Pass 1 used the monitor's default cap of 4 GiB of system swap growth and stopped six of twelve runs (return code 125, "swap growth"): 1024 stock sub-quadratic twice, 1024 stock SDPA once, 512 stock sub-quadratic twice and 512 kernel sub-quadratic once. Five of the six stops were stock runs and one was a kernel run.

- Every run that started with system swap at about 2.8 to 3.9 GiB was stopped, with swap rising to about 7.5 to 8.2 GiB, except one (1024 kernel sub-quadratic at 12:21, 3.67 GiB). That is all five stock runs and one of two kernel runs.
- Every run that started right after a stop showed 7.5 to 8.2 GiB of swap in use in the driver's reading just before launch, and completed. The monitor's own baseline is not recorded for completed runs. `gpu_runner` reads it once at start, and the previous job's swap had not been released yet, so those runs were most likely measured against a baseline that was too high. That is consistent with the stop, complete, stop alternation.
- Host state also differed: the 1-minute load average at pass 1 starts was 5 to 21. The five completed runs that started right after a stop, with swap near 8 GiB, had far longer wall times (512 kernel SDPA 287 and 305 s against 73 to 102 s in pass 2). The one completed run that started at a normal swap level (1024 kernel sub-quadratic, 179 s at load average 19) was not slower than pass 2 (210 to 234 s). Pass 2 starts show a load average of 0.9 to 3.6 for every run but the first (12.4 at 13:06, still decaying from pass 1) and swap of 2.8 to 3.9 GiB before each run.
- What is not known: how much swap pass 2's runs grew, because growth is recorded only for stopped runs, and so whether today's sweep would pass the 4 GiB cap on a quiet host.

Changes, in effect from the 256 check onward: the driver waits for swap to stop falling before each run (`settle`, up to 90 s) and records the wait in `host_before.settle`, and it writes the swap cap, RSS cap, deadline, cond-cache mode, sizes and repeats into the header. Pass 2 and the 768 check predate this: they ran without the settle wait, and their headers do not record the 8 GiB cap, which is known only from the commands above. `gpu_runner`'s default cap is unchanged at 4 GiB. The six completed pass 1 runs are within 2 percent of pass 2 and are not in the table above: 512 kernel SDPA 3.77 to 3.78 and 512 stock SDPA 8.54 fall inside the pass 2 ranges, while kernel sub-quadratic was 1 to 2 percent faster than in pass 2 (512: 3.69 against 3.75 to 3.76; 1024: 15.93 to 15.95 against 16.08 to 16.12).

### Checks run alongside

- `bench/bench_int8_linear.py`, two runs, saved as `docs/review/2026-11-09-experiments/int8-linear-microbench-rerun.txt`. Milliseconds per call for bf16, fp16, ComfyUI-style dequantize per call, and the prototype kernel: K=6144, N=6144: 68.6, 37.8 to 38.6, 76.4 to 76.5, 41.2 to 42.8. K=6144, N=16384: 181.8, 101.2 to 101.7, 204.2 to 205.1, 106.5 to 107.0. K=16384, N=6144: 225.4, 200.7 to 222.6, 238.7 to 249.5, 208.9 to 213.8. These are at or below the October 2 table from the same script (lines 17 to 19): within 4 percent at K=6144 and 4 to 9 percent faster at K=16384 in three of four columns, where fp16 also varied by 11 percent between today's two runs. The 103 and 107 ms at K=16384 in the hardened table (lines 76 to 81) come from the comfy-kitchen backend path, which avoids the wide-K `F.linear` slowdown; they are a different measurement, not a regression.
- CPU test suite: `pytest tests --ignore=tests/test_gguf_node.py`, 685 passed, 16 skipped.

## 2026-11-10: Where decode time goes on MPS, and the one-row linear "slow path"

Prompted by the model-research-lab's Q003 (2026-11-10): on a 3.2M-parameter character model, swapping one-row `nn.Linear` calls for `addmm` on a pre-transposed weight cut the cached decode step to about a third. The question here: does that carry to real models? Results are in `docs/review/2026-11-10-experiments/`. Scripts: `bench/bench_onerow_linear.py`, `bench/bench_smallm_layout.py`, `bench/bench_linear_rank.py`, `bench/bench_decode_layout.py`, `bench/bench_decode_breakdown.py`, `bench/profile_decode_step.py`. All runs went through `bench/gpu_runner.py` on the M1 Max, torch 2.14.1 unless marked 2.15.0 (a separate environment; the repo venv is unchanged).

**The tiny model's gain does not carry to a 1.7B model in its simple form.** Swapping all 197 linear layers of Qwen3-1.7B (fp16) to `mm` on pre-transposed weights made decode slower at every batch size: 0.94x at batch 1, 0.76x to 0.89x at batches 4 to 24, with identical greedy tokens. Most of the tiny model's gain was a fixed cost of `F.linear` with a bias (20 to 30 µs per call at every shape) plus one-row dispatch overhead on tiny matrices, both of which vanish against a large layer's memory traffic.

**A microbenchmark pitfall.** Looping one weight that fits in the chip's on-die cache measures cache bandwidth, not decode: a 2048x6144 fp16 one-row `mm` timed at 44 µs, about 570 GB/s, above the M1 Max's roughly 400 GB/s memory bandwidth. In model order, with each layer's weight streaming from memory, the pre-transposed layout is the slower one (14.6 against 12.2 ms for all 197 linears at batch 1). Layout conclusions need weights larger than the cache or rotated through it.

**Where a decode step goes (Qwen3-1.7B fp16, 32-token prompt).**

| | torch 2.14.1 | torch 2.15.0 |
|---|---|---|
| Full step, batch 1 | 24.3 ms | 21.8 ms |
| The 197 linears alone, in model order, batch 1 | 12.2 ms (281 GB/s) | 10.8 ms (318 GB/s) |
| Full step, batch 16 | 41.6 ms | 42.3 ms |
| The 197 linears alone, batch 16 | 26.2 ms (131 GB/s) | 26.2 ms (131 GB/s) |

- At batch 1 the linears run near the memory limit, and about half the step (11 ms) is everything else. Torch 2.15.0's one-row GEMV routing for `F.linear` (PR 198721) accounts for the 2.5 ms gained.
- The rest is GPU time, not host overhead: enqueuing a step takes about 8 ms of CPU against a 22 ms step. A profile of one step shows about 6,000 aten calls: 113 normalisation layers each run as about seven elementwise kernels (pow, mean, add, rsqrt, mul, mul, dtype copies), 113 `cat` calls (about 56 from the rotary positional encoding's rotate-half and 56 growing the KV cache), and the elementwise steps around them. Hundreds of near-empty kernels make up the second half of the step. Correction, same day: an earlier version of this entry said attention ran on PyTorch's unfused fallback. That came from an operation name the profiling script cut at 40 characters; the full name is `_scaled_dot_product_attention_math_for_mps`, the fused Metal path, and a direct capture of Qwen3's attention arguments confirmed it.
- At batch 16 the linears read the same weights as at batch 1 but take 2.4 times as long. Between 2 and about 16 rows, PyTorch leaves its own GEMV path for Apple's general matrix kernels, which pay encode overhead per call and, with the weight in `nn.Linear`'s [out, in] layout, appear to read it more than once. Unchanged in 2.15.0 and on main as of 2026-11-10 (source reading, `docs/research/program/raw/pivot_mps-source.md`, git-ignored).

**The row-count bands.** With interleaved arms and an A/A control (median ratio 1.00, range 0.70 to 1.18 on small cells), `F.linear` against `mm` on a pre-transposed weight changes in steps at row counts 1, 2 to 7, 8 to 9, 10 to 16, 17 to 24 and 25 to 32. The largest effect is on the Llama-7B up-projection shape (4096 to 11008, fp16, 90 MB, larger than the on-die cache): at 17 to 32 rows `F.linear` takes about 980 to 1,220 µs against 360 to 470 µs, 2.4 to 2.7 times slower, with non-overlapping repeats, on both torch versions. On other shapes and in bf16 the direction can reverse (Qwen3-4B up-projection, bf16, 17 to 32 rows: pre-transposed 1.5 times slower).

What this does not show: no end-to-end model gain yet. The 2 to 16 row and 17 to 32 row slowdowns are per-op measurements; their effect on speculative-decoding verification, small-batch serving and short prefills is a prediction until measured. Single machine, one model family end to end.

Open levers, in order of expected payoff at batch 1 and batch 16: fusing normalisation (`F.rms_norm` is one fused kernel on MPS; Hugging Face's Qwen3 does not use it), a preallocated KV cache and a fused rotary step (the 11 ms); a multi-row kernel for 2 to 16 rows that streams each weight row once (bringing batch-16 linears from 26 ms toward the batch-1 figure); folding normalisation scales into the next weight, an exact rewrite that removes kernels.

## 2026-11-10: Fused normalisation and a fused rotary kernel, 1.41x faster Qwen3-1.7B decode on MPS

Thread experiment A ([the decode thread](research/program/threads/mps-decode.md)). Qwen3-1.7B, fp16, torch 2.15.0, 32-token prompt, 32 greedy tokens, arms interleaved in one process, median of 3. Scripts: `bench/bench_decode_fusions.py`, `bench/fused_rope.py` (the kernel and its unit test), `bench/check_norm_equivalence.py`, `bench/check_divergence_margin.py`. Results in `docs/review/2026-11-10-experiments/decode-fusions-*.json` and `*-equivalence-*.json`.

| ms per decode step | Stock | Fused normalisation | Fused rotary | Both |
|---|---|---|---|---|
| Batch 1 | 23.5 | 18.1 (1.30x) | 22.7 (1.04x) | 16.8 (1.41x) |
| Batch 4 | 28.4 | 22.9 (1.24x) | 26.6 (1.07x) | 21.7 (1.31x) |
| Batch 16 | 44.2 | 38.3 (1.15x) | 42.8 (1.03x) | 35.8 (1.23x) |

- **Fused normalisation** is a one-line change: `F.rms_norm` (one fused kernel on MPS) in place of Hugging Face's `Qwen3RMSNorm`, which runs about seven elementwise kernels per layer and 113 layers per step. Repeated in three separate runs at 1.26x to 1.30x at batch 1.
- **A variant that keeps Hugging Face's float32 arithmetic exactly** (upcast, fused norm, cast back, then the weight) reproduced stock's logits exactly and gained nothing (22.9 against 23.2 ms): the dtype conversions are kernels too. The gain comes from doing the normalisation in fp16 inside one kernel.
- **The fused rotary kernel** replaces the rotate-half construction (a negate, a concatenate, two multiplies and an add for queries and again for keys) with one Metal kernel per tensor that computes in float32 and rounds once. Against a float64 reference it is about twice as accurate as Hugging Face's fp16 formula in every unit-test case (about 1e-3 against 1.5e-3 to 3e-3). A first version was wrong inside the model: Hugging Face passes cosine and sine tables of batch 1, broadcast across the batch, and the kernel indexed past them for later sequences (NaN at batches 4 and 16). The unit test had used one table per sequence. Both shapes are now tested.
- **The preallocated `StaticCache` gained nothing** (0.96x to 1.01x). Concatenating the KV cache each step is not where the time goes at this length.
- **Equivalence.** Teacher-forced logits on 16 sequences of 96 tokens against a float32 run of the same model: stock fp16 max error 0.262, mean 0.0038, top-1 agreement with float32 99.22%; fused normalisation 0.230, 0.0036, 99.35%; both fused 0.230, 0.0036, 99.02% (3 positions of 1,536 fewer than stock). Greedy decoding with both fused diverged from stock once at batch 1; at that step stock's top two tokens were 0.016 logits apart, about one fp16 step, and the two models' logits never differed by more than 0.055 over 32 steps. That is a tie flip, not a loss. Random-token prompts make near-ties common; a real-text set would give a sharper check.

Prediction written before the run (thread document): batch 1 at 16 ms or less. Measured 16.8 ms, so the prediction missed narrowly. After both changes about 3,700 aten calls remain per step; the next candidates are merged projections (query, key and value in one matrix; gate and up in one) and a fused gated activation.

## 2026-11-10: torch.compile with a static cache, 1.6x Qwen3-1.7B decode on MPS and more accurate than eager

Same model and settings as the entry above. `bench/bench_decode_compile.py`: the decode step wrapped in `torch.compile(dynamic=False)` with a `StaticCache` sized to prompt plus new tokens, against the same step eager with the same cache. Results in `docs/review/2026-11-10-experiments/decode-compile-*.json` and `equivalence-fp32-with-compile-2.15.0.json`.

| ms per decode step | Eager, static cache | Compiled | Speed-up | Greedy agreement |
|---|---|---|---|---|
| Batch 1 | 25.2 | 15.8 | 1.60x | 100% |
| Batch 4 | 29.5 | 22.9 | 1.29x | 91% |
| Batch 16 | 42.5 | 38.2 | 1.11x | 95% |

- First compiled pass about 15 to 17 s, once per shape. With the default dynamic cache, eager is a little faster (23.5 ms at batch 1 in the entry above), so against the default eager setup the batch-1 gain is about 1.5x.
- **The compiled model is closer to float32 than eager.** Teacher-forced logits, 16 sequences of 96 tokens, against float32: compiled max error 0.097, mean 0.0027, 99.9th percentile 0.023, top-1 agreement 99.28%; eager fp16 0.262, 0.0038, 0.054, 99.22%. Inductor computes the fused elementwise chains in float32. The greedy disagreements at batches 4 and 16 are tie flips on the more accurate side.
- Compile beats the hand-written fusions at batch 1 (15.8 against 16.8 ms) and needs no custom kernel. Its gain shrinks with batch because the time moves into the 2 to 16 row matrix kernels, which compile does not replace. The multi-row kernel (thread experiment B) targets exactly that.
- **Merged projections** (query, key and value in one matrix, gate and up in one; `decode-fusions-merged-projections-2.15.0.json`) gave identical tokens and logits to stock and almost no speed: 1.02x at batch 1, 0.97x at batch 4, 1.07x at batch 16. At batch 1 the linears are bandwidth-bound and fewer calls do not help.

Not yet checked: whether this recipe is documented for MPS (it is the standard recipe on CUDA), other models, long contexts, and memory overhead of the static cache.

## 2026-11-10: A multi-row Metal linear kernel, 1.29x Qwen3-1.7B decode at batch 8 and 16 on MPS

Thread experiment B. A Metal kernel for linear layers with 2 to 32 input rows that streams each weight row once and uses the GPU's matrix units, built by a subagent and reviewed by a second (`docs/review/2026-11-10-experiments/multirow/`: `README.md`, `REVIEW.md`, raw JSON; code in `bench/multirow_linear.metal`, `bench/multirow_linear.py`, `bench/bench_multirow.py`, `tests/test_multirow_linear.py`).

- **Decode, Qwen3-1.7B fp16, torch 2.15.0,** stock against every linear routed to the kernel at 2 to 16 rows: batch 8, 36.2 → 28.0 ms (1.29x); batch 16, 41.6 → 32.0 ms (1.30x); reproduced by the reviewer in a fresh process. No clear change at batch 1 or 4.
- **Correct:** 506 cases and 22,351 runs against float64, 0 failures, error matching `F.linear`'s; four gross and three subtle planted bugs caught. Greedy divergences are tie flips (top-two gaps of 0.0 and one fp16 step), and stock's own logit difference between batch sizes is larger than the kernel's difference from stock.
- **Where the gain comes from.** At 8 rows the per-layer linears gain 1.68x. At 16 rows they gain 1.27x, and the 151,936-row output head, which `F.linear` handles badly at 12 to 32 rows (7.2 to 10.5 ms per call against 2.1 to 4.2), supplies 55% of the saving.
- **Prediction missed:** the thread predicted the batch-16 linears at 14 ms or less; measured about 18 ms. Not a hardware limit: the kernel reaches about 3.1 TFLOPS at 16 rows where Apple's kernel does 5.4 at 32 rows on the same shape.

## 2026-11-10: Both levers together, 1.5x to 1.8x Qwen3-1.7B decode on MPS at batch 1 to 16, closer to float32 than eager

`torch.compile` with a static cache, plus the multi-row kernel (round 1) registered as a custom op (`mk::multirow_linear`) so Inductor treats it as opaque and routes linears with 2 to 16 rows to it. `bench/bench_decode_compile_kernel.py`, one batch size per process, best of two passes after compiling; equivalence by `bench/check_decode_equivalence.py`. Results in `docs/review/2026-11-10-experiments/decode-compile-kernel-b*.json` and `decode-equivalence-fp32-b16-2.15.0.json`.

| ms per decode step | Eager, static cache | Compiled | Compiled + kernel | Best speed-up | Tokens per second, best |
|---|---|---|---|---|---|
| Batch 1 | 24.9 | 14.9 | 16.3 (kernel not used at 1 row) | 1.67x (compiled) | 67 |
| Batch 4 | 32.5 | 29.4 | 21.5 | 1.51x | 186 |
| Batch 8 | 37.1 | 31.5 | 20.7 | 1.79x | 386 |
| Batch 16 | 43.2 | 38.3 | 25.5 | 1.69x | 627 |

- The two levers compound. Compile removes the small-kernel overhead; what remains at batch 4 to 16 is the 2-to-16-row matrix path, which the kernel fixes. Neither alone gets past 1.3x at batch 8 or 16.
- At batch 1 the compiled-plus-kernel arm read 16.3 ms against 14.9 for compiled alone, although the kernel does not run at 1 row. The two compiles produce different graphs (the custom-op branch is traced); not investigated. Use compile alone at batch 1.
- **Equivalence, decode path at batch 16** (teacher-forced one token at a time on a float32 model's greedy tokens, 512 positions): eager fp16 max error 0.100, mean 0.0040, top-1 agreement 99.80%; compiled 0.064, 0.0031, 99.80%; compiled plus kernel 0.056, 0.0030, 99.61% (one position fewer). The combined arm is closest to float32 on every error measure.

Kernel round 2, aimed at the 16-row efficiency gap, is running; these numbers use round 1.

## 2026-11-10: What verifying drafted tokens costs on MPS, and why speculative decoding loses on Apple stacks

Speculative decoding verifies k drafted tokens with one forward of k + 1 tokens, which puts the model's linears in the 2-to-16-row regime. `bench/bench_verify_cost.py`: one forward of T new tokens against a static cache holding a 256-token prompt, batch 1, Qwen3-1.7B fp16, torch 2.15.0, the cache reset and the prompt re-filled before every timed call, median of 9. Results in `docs/review/2026-11-10-experiments/verify-cost/`.

| ms per forward of T tokens | T = 1 | 2 | 3 | 5 | 9 | 13 | 17 |
|---|---|---|---|---|---|---|---|
| Eager | 25.2 | 29.5 | 31.0 | 33.8 | 46.2 | 51.5 | 56.1 |
| Compiled | 14.9 | 21.5 | 22.2 | 26.1 | 38.2 | 47.5 | 46.2 |
| Compiled + multi-row kernel | 13.9 | 18.6 | 18.5 | 20.3 | 30.5 | 27.3 | 46.3 |

- **Compile alone raises the relative cost of verifying.** It speeds the one-token step far more than the few-token step: verifying 2 tokens costs 1.45x a single step compiled, against 1.17x eager. A faster plain decode leaves less room for speculation to pay.
- **The kernel brings it back.** Compiled plus kernel, verifying 1 to 4 drafts costs 1.33x to 1.46x a single step, and 12 drafts about 2x. At 17 rows the kernel is off and the cost returns to the compiled figure. The T = 9 and T = 13 cells reflect the kernel's tuned bands (13 rows is cheaper than 9).
- **Context.** A 2026 study of speculative decoding on Apple Silicon (arXiv 2610.31479, MLX only, M2 Pro and M3 Ultra) found every method it tested reduced throughput, while its own kernel section measured MLX's 4-bit matmul at about 5x the one-row time at 8 rows and showed that a small-row kernel turned one speculative setup from break-even to 1.13x. It did not connect the two, and did not test PyTorch, M1 or 9 to 16 rows.
- **Projection, not a measurement.** At that paper's n-gram acceptance rates (about 0.6 to 1.6 accepted drafts per round with 2 drafted), prompt-lookup speculation on the compiled-plus-kernel stack would run about 1.3x to 2x faster than the best plain decode (compiled, 14.9 ms), before drafting overhead. The end-to-end test on real text is next.

**Correction, same day: the first prompt-lookup speculative results were invalid.** An end-to-end run (`bench/bench_prompt_lookup.py`) reported 2.03x on a code edit and 1.86x on free writing, but its output differed from plain greedy decoding at tokens 73 and 7, and `bench/check_spec_divergence.py` showed those were not near-tie flips: at the code edit's first difference the model's own top choice led by 20 logits, and 29 tokens of the story were not the model's argmax. Cause: transformers 5.18's `StaticLayer.update` writes each new entry at the cache's own running count (`cumulative_length`) and ignores `cache_position`. After a round with rejected drafts that count runs ahead of the true position, so later tokens were written to the wrong slots. The same count is why repeated timing calls overflowed the cache in the verify-cost run, which then reset the cache before every call; those timings are unaffected. The loop now resets the count to the true position after every round. The corrected run is pending.

## 2026-11-10: Prompt-lookup speculative decoding on MPS, exact, 3.0x to 3.7x on a copy-heavy task with the multi-row kernel

`bench/bench_prompt_lookup.py`, with the cache-position fix above. Qwen3-1.7B fp16, batch 1, greedy, 160 new tokens, thinking off, `torch.compile` throughout; drafts of 4 tokens taken from the context itself (the latest earlier occurrence of the last 3, 2 or 1 tokens). Two tasks: rewriting a 1,600-token Python file with one rename (output mostly repeats input) and a 300-word short story (little repeats). Arms in separate processes. Results in `docs/review/2026-11-10-experiments/prompt-lookup/`.

| ms per token | Plain | Speculation | Speculation + kernel | Drafts accepted per round |
|---|---|---|---|---|
| Code edit, always draft | 17.8 | 7.7 (2.31x) | 4.75 (3.74x) | 3.42 |
| Code edit, adaptive | 15.2 | 7.1 (2.14x) | 5.0 (3.03x) | 3.30 |
| Free writing, always draft | 17.3 | 26.3 (0.66x) | 19.3 (0.89x) | 0.02 |
| Free writing, adaptive | 14.6 | 15.0 (0.97x) | 14.0 (1.04x) | 0.01 |

- **Output identical to plain greedy decoding in every run** (all 160 tokens, both tasks, all arms).
- "Adaptive" drafts only when the last two or more tokens match earlier context, and otherwise takes a plain one-token step, so text that does not repeat loses nothing.
- **The kernel accounts for much of the gain:** on the code edit, 2.1x to 2.3x without it and 3.0x to 3.7x with it, the same drafts accepted. It makes the 5-token verification cheap (see the verify-cost entry).
- **Noise:** the plain baseline differed between the two series (17.8 against 15.2 ms on the code edit), because arms ran in separate processes. Read each speed-up within its own row, and the code-edit gain as 3.0x to 3.7x.
- **What this does not show:** other models, other draft lengths, long contexts, batches above 1, or real-world task mixes. Copy-heavy work (editing code, rewriting, retrieval-augmented answers) is where prompt lookup is known to help. The new part is that on an Apple GPU it pays mainly once the few-row verification path is fixed; arXiv 2610.31479 found n-gram speculation near break-even on stock MLX.

## 2026-11-10: Multi-row kernel round 2, and which numbers above are provisional

Round 2 of the multi-row kernel (`docs/review/2026-11-10-experiments/multirow/round2/`, `README.md` and `REVIEW.md`): a fragment order that makes each weight fragment one word a GPU lane already loads removed most of the non-multiply overhead at 9 to 16 rows. The kernel alone takes Qwen3-1.7B fp16 decode at batch 16 from about 41 to 28 ms per step (1.47x to 1.49x, three processes, both noise arms within 1%), batch 12 1.50x, batch 8 1.33x to 1.37x. The 197 linears at batch 16 came in at 15 ms or less in 7 of 10 samples, and the kernel is now limited by streaming weights at about 280 GB/s. Correct in about 70,000 more runs with 18 planted bugs caught. The review found the tuning could not be rerun from the code and that several per-cell choices were noise; those are being fixed.

**Provisional:** the verification-cost entry and the two prompt-lookup series above (and the compile-plus-kernel table) were measured while the round-2 builder was editing the kernel and its tuning table in the same working tree, so which kernel version they used is not known. The reviewer's rerun of verification cost with the round-2 table gave higher absolute times in both arms (for example 23.1 against 18.6 ms at T = 2 with the kernel) while the kernel still helped. All of these will be rerun on the final table, and this entry will say so.

## 2026-11-10: Final numbers on the final kernel table (replaces the provisional entries above)

Every benchmark that uses the multi-row kernel, rerun in one labelled batch on the final tuning table (commit 5e8c9a5), Qwen3-1.7B fp16, M1 Max, torch 2.15.0, with the machine otherwise quiet (the lab held its GPU and CPU work for the hour). Raw results and the batch script in `docs/review/2026-11-10-experiments/final-kernel-rerun/`. These supersede the compile-plus-kernel, verify-cost and prompt-lookup numbers recorded earlier today, which were measured while the kernel was being edited.

**Decode, compile plus the multi-row kernel:**

| ms per step | Eager, static cache | Compiled | Compiled + kernel | Speed-up |
|---|---|---|---|---|
| Batch 4 | 28.3 | 22.2 | 17.3 | 1.64x |
| Batch 8 | 38.4 | 31.3 | 19.0 | 2.02x |
| Batch 16 | 42.2 | 36.9 | 19.8 | 2.13x (about 810 tokens per second) |

Batch 1 is unchanged: compile alone, 1.67x (entry above).

**Verifying drafted tokens** (one forward of T tokens on a 256-token context, ms):

| T | 1 | 2 | 3 | 5 | 9 | 13 | 17 |
|---|---|---|---|---|---|---|---|
| Eager | 23.1 | 27.4 | 27.9 | 31.4 | 43.9 | 48.2 | 52.8 |
| Compiled | 16.1 | 30.7 | 25.0 | 29.4 | 40.7 | 43.9 | 50.0 |
| Compiled + kernel | 16.0 | 19.4 | 18.8 | 19.0 | 23.0 | 24.3 | 52.1 |

With the kernel, verifying 4 drafted tokens costs 1.19x a single step and 12 drafts 1.52x. Compiled without it, 1 draft costs 1.9x. At 17 rows the kernel is off.

**Prompt-lookup speculative decoding** (160 tokens, greedy, ms per token against plain compiled decoding):

| | Plain | Speculation | Speculation + kernel |
|---|---|---|---|
| Code edit, always draft | 17.6 | 6.70 (2.62x) | 5.01 (3.51x) |
| Code edit, adaptive | 17.6 | 7.70 (2.28x) | 5.39 (3.26x) |
| Free writing, always draft | 14.1 | 29.2 (0.48x) | 20.0 (0.70x) |
| Free writing, adaptive | 14.1 | 16.4 (0.86x) | 14.4 (0.98x) |

- Output identical to plain greedy in every run but one: adaptive with the kernel on free writing differs at token 34, where the top two candidates are 0.0156 logits apart (one fp16 step) and it is the plain run that departs from the model's own full-forward choice. Tie flip, checked with `bench/check_spec_divergence.py`.
- Adaptive drafting no longer fully protects free writing without the kernel (0.86x, against 0.97x earlier). Likely the per-step scan of the whole context for matches, in Python; not measured.
- **Equivalence at batch 16, decode path, against float32:** compiled plus kernel mean error 0.0030 and 99.9th percentile 0.022, against eager's 0.0040 and 0.032; max 0.104 against eager's 0.100; top-1 agreement 99.61% against 99.80% (one position of 512). Comparable to eager, better on mean and tail, a hair worse on max.

## 2026-11-10: Corrections after the cross-family review

GPT-6.1 Sol reviewed today's decode work read-only ([its review](research/program/reviews/2026-11-10-decode-stack-review-sol.md)). The direction of every result holds; these statements above were wrong or overstated and are corrected here.

- **The prompt-lookup speed-ups (3.26x to 3.51x on the code edit) used a burdened baseline.** The loop reset every layer's cache counter after every round, including plain one-token steps, so the plain arm did 28 extra writes per token. The speed-up's denominator was inflated by an amount the saved data cannot recover. Fixed in `bench/bench_prompt_lookup.py`: the counter is reset only after rejected drafts. The comparison will be rerun against an ordinary compiled decoding loop; until then, read the code-edit gain as unconfirmed in size.
- **The kernel does not make this speculation pay by itself.** Without the kernel, speculation on the code edit already gave 2.28x to 2.62x; the kernel adds about 1.34x to 1.43x on top.
- **arXiv 2610.31479 was described unfairly.** Its kernel section (§2.3.2) does connect small-row kernels to speculative verification, and reports one setup going from 0.99x to 1.13x against plain decoding with its kernel; §2.2.3 reports positive speculative setups, including n-gram decoding on an M3 Max. The open part here is narrower: PyTorch on MPS, an M1, 9 to 16 rows, and the size of the gain.
- **"Identical to greedy" does not hold for every run.** The adaptive kernel run on free writing differs from plain decoding in 121 of 160 tokens, starting at a near-tie. A tie explains the divergence; it does not make the outputs the same. The code-edit runs are identical to the plain run.
- **"Closer to float32 on every measure" does not hold for the final run.** Compiled plus kernel improves mean and 99.9th-percentile error but has a slightly larger maximum (0.104 against 0.100) and one more top-1 disagreement in 512. "Comparable to eager fp16 on these inputs" is what the evidence supports.
- **The batch-1 1.67x is against eager with a static cache.** Default dynamic-cache eager is faster, so against default eager the gain is about 1.5x. The two compiled arms at batch 1 also differed by about 9% with the kernel never invoked, unexplained.
- **PyTorch PR 198721 was credited with the 2.5 ms batch-1 gain from 2.14.1 to 2.15.0.** Comparing two releases does not isolate one change.
- **A kernel contract hole:** `supports()` accepted zero-width inputs, which divide by zero in the split calculation. Fixed and tested; no measured shape was affected.
- **Method:** the decode harness takes the faster of two pass medians, runs arms in a fixed order, and keeps no per-sample data. The large gains are credible; the second digit is not established. The rerun will interleave arms across repeated processes, keep every sample, add A/A arms, include default dynamic-cache eager, and force every acceptance length in the speculative loop against a freshly rebuilt cache.

## 2026-11-10: Decode rerun under fixed rules

The rerun Sol's review asked for, under the protocol and predictions in [the decode thread](research/program/threads/mps-decode.md) written before any result. Qwen3-1.7B fp16, M1 Max, torch 2.15.0, transformers 5.18.0.

**How it was run.**

- Four rounds. Every process runs every arm, in a rotated order, and keeps every sample.
- Ratios are formed inside each process and reported as the median across the four rounds, with the range.
- A/A arms pair two separately compiled copies of the same pipeline, so they measure the noise.
- Raw data and `summary.md` are in [rerun/](review/2026-11-10-experiments/rerun/).
- In round 2, a local CPU-only llama.cpp dry run from another session overlapped the code-edit and free-writing speculative processes. Their A/A pairs read 1.25 and 1.11, where the other rounds read 0.97 to 1.04. Medians absorb that one round; ranges that include round 2 are marked.

**Correctness, checked before any timing.**

- **Speculative rollback is exact.** This holds in eager, compiled, and compiled with the kernel:
  - every acceptance length from 0 to 4 was forced, along with switches between one-token and five-token steps;
  - rejected cache slots were filled with sentinels;
  - every verified row was compared with a freshly rebuilt cache.
  - Both planted faults failed. Skipping the counter reset was off by 25 to 45 logits against a tolerance of 0.15.
- **Equivalence against float32 passes the rule fixed in advance,** on both real-text and random prompts. On real text, compiled with the kernel against eager fp16:
  - mean error 0.0043 against 0.0051;
  - 99.9th percentile 0.029 against 0.034;
  - maximum 0.073 against 0.090;
  - top-1 agreement 511 of 512 for both.

**Decode step against static-cache eager, and against default dynamic-cache eager, median (range):**

| Setting | Compiled | Compiled + kernel | Kernel over compiled |
|---|---|---|---|
| Batch 1, context 32 | 1.61x (1.59–1.80); 1.55x default | 1.64x; 1.55x default | 1.01x (0.91–1.05) |
| Batch 1, context 512 | 1.72x; 1.72x default | 1.78x; 1.74x default | 1.02x |
| Batch 4 | 1.22x | 1.65x (1.53–1.77); 1.63x default | 1.42x |
| Batch 8 | 1.19x | 1.93x (1.91–2.01); 1.92x default | 1.66x (1.59–1.70) |
| Batch 16, context 32 | 1.16x | 2.04x (1.99–2.30); 2.07x default | 1.77x (1.69–2.02) |
| Batch 16, context 512 | 1.12x | 1.89x (1.76–1.98); 2.05x default | 1.69x |

- **Batch-1 A/A pairs vary by up to 9% in a single process.** The 9% gap between nominally identical pipelines reported earlier is that noise. The kernel arms at batch 1 sit inside it.
- **Default dynamic-cache eager runs 3 to 6% faster than static-cache eager at batch 1, and up to 8% slower at batch 16 with a 512-token context.**

**Verify cost misses its prediction.** One forward of T tokens on a 256-token context, as a ratio to T = 1:

| T | 2 | 3 | 5 | 9 | 13 | 17 |
|---|---|---|---|---|---|---|
| Compiled + kernel | 1.52 | 1.45 | 1.52 (1.42–1.64) | 1.79 | 1.65 | 3.24 |
| Compiled | 1.77 | 1.57 | 2.08 | 2.72 | 3.09 | 3.30 |

The prediction was that five tokens would cost at most 1.3x one token. They cost 1.52x. The earlier 1.19x came from a single favourable process. The kernel still roughly halves the cost of verification against compiled alone, up to 13 tokens. At 17 rows it switches off.

**Prompt-lookup speculation against an ordinary compiled decoder** (batch 1, greedy, every request to end-of-sequence; median over four rounds):

| Task | Speculation | Adaptive | Speculation + kernel | Adaptive + kernel |
|---|---|---|---|---|
| Code edit (853 tokens), decode | 2.08x | 2.07x | 2.92x (2.81–3.53 with round 2) | 2.97x |
| Code edit, whole request | | | 2.76x | 2.80x |
| Prose typo fix (244 tokens), decode | 2.08x | 2.22x | 3.14x | 3.31x (3.26–3.42) |
| Prose typo fix, whole request | | | 2.86x | 3.01x |
| Free writing, per token | 0.58x | 0.92x | 0.84x | 1.02x (1.00–1.09) |

- **Every edit-task output is token-for-token identical to plain decoding.** That covers every arm and every round.
- **Free writing diverges after a near-tie.** It happens at token 34 with adaptive drafting plus the kernel, and at token 296 with fixed drafting plus the kernel. Free writing is therefore compared per token.

**Against the fixed predictions.**

- Batch 1, at least 1.4x static and 1.3x default: met (1.61x and 1.55x).
- Batches 8 and 16, at least 1.8x, with the kernel adding at least 1.3x: met.
- Verify at most 1.3x: missed (1.52x).
- Code edit, at least 2.5x decode and 2x whole request: met (2.92x and 2.76x).
- Free writing, adaptive with the kernel, at least 0.95x: met (1.02x).

Every claimed gain clears its setting's largest A/A deviation in all four rounds. Two arms do not clear it, so neither is a gain: free-writing parity, and the kernel at batch 1.

**What changes from the earlier entries.**

- The batch-16 result is about 2.0x, not 2.13x.
- Batch 1 is 1.6x against static-cache eager and 1.55x against default eager, not 1.67x.
- Verify costs 1.5x, not 1.19x.
- The code-edit speculation is 2.9x against an honest baseline, not 3.3 to 3.5x. A prose edit does slightly better, at 3.1 to 3.3x.
- Without the kernel, speculation gives 2.1x on both edits.
- One reporting change was made after round 0, and is disclosed here. Free writing is compared per token, because diverging answers differ in length.

## 2026-11-11: The attention cliff above 8 query rows shows in the whole model

The lab's Q009b found that torch MPS attention switches to a much slower path above 8 query rows (`model-research-lab/research/verification/outcomes/Q009b-2026-11-11.md`). This is the gate for decode experiment E. It measures one verify forward of T tokens in whole Qwen3-1.7B fp16 at a 4,096-token context, with seven samples per T and one process per arm (torch 2.15.0, M1 Max). Raw data is in [2026-11-11-gateE/](review/2026-11-11-gateE/).

Median ms per forward:

| T | 1 | 5 | 8 | 9 | 12 | 16 |
|---|---|---|---|---|---|---|
| Eager | 56.0 | 68.7 | 76.0 | **124.0** | 125.3 | 130.5 |
| Compiled | 40.8 | 60.3 | 70.5 | **122.1** | 114.8 | 124.4 |
| Compiled + kernel | 42.8 | 37.0 | 57.8 | **108.3** | 103.8 | 105.0 |

- **The cliff is present in every arm.** T = 9 costs 1.63x to 1.87x as much as T = 8. Then cost flattens: T = 16 is within 5% of T = 9.
- **It is robust to the noise.** The samples are noisy at this context (T = 1 ranges from 27 to 47 ms within one arm), but in every arm the slowest T = 8 sample is faster than the fastest T = 9 sample.
- **Against the gate prediction written before the run:**
  - T = 9 at least 1.3x T = 8: **met** in every arm.
  - T = 8 at most 1.15x T = 5: met for eager (1.11x), about met for compiled (1.17x), missed for the kernel arm (1.56x, where T = 5 read low).
  - The 16,384-token cells **did not run**. The harness re-reads the whole prompt before every timed sample, and each process hit its 400-second limit. A 16k measurement needs a cache snapshot rather than a re-prefill.
- **What it means for speculation on MPS.** At a 4k context, verifying 8 drafts (T = 9) costs about 2.5x to 4x a single step, against about 1.4x to 2x for 7 drafts (T = 8). So drafts should be capped at 7 on MPS until attention is fixed. A Metal attention kernel for 2 to 16 query rows (experiment E) is justified.
- **Decoding one token is also much slower at 4k than at 256 context:** 27 to 45 ms against about 15 ms. Attention over the cache already dominates at T = 1 on this path.

## 2026-11-11: A small-query attention kernel removes the 9-row cliff (experiment E)

Qwen3-1.7B fp16, M1 Max, torch 2.15.0, transformers 5.18.0. The plan and predictions were written in [the decode thread](research/program/threads/mps-decode.md) before any kernel code. Raw data is in [2026-11-11-E/](review/2026-11-11-E/). Workflow agents built the kernel and two adversarial reviewers checked it; I checked their results.

**The cause.** In PyTorch v2.15.0 (`aten/src/ATen/native/mps/operations/Attention.mm`), attention with at most 8 query rows uses a vector kernel. Above 8 rows it uses a prefill kernel built for long prompts, whose 16- or 32-row tiles are mostly empty here.

**The kernel** (`bench/smallq_attention.{metal,py}`, wired in by `bench/smallq_hf.py`).
- **Design.** It is grouped-query and split over the cache: two passes, and each key/value block is loaded once for every new query row and every head that shares it. Accumulation is in fp32, and masking is NaN-safe.
- **Device position.** One entry point reads the position from GPU memory, so a compiled model never synchronises with the host.
- **Wiring.** A custom op plus a transformers attention and mask function (`mk_smallq`). It takes T = 4 to 16; stock SDPA handles everything else, and any padding mask.
- **Correctness.** 392 tests against a float64 reference, under a rule fixed in advance (max error at most 1.5x stock SDPA's, mean at most 1.25x). Every planted fault is caught.
- **In the whole model,** at 4,096 context, logits stay within 0.02 of stock compiled at T = 4 to 16 (stock's own eager-to-compiled spread is 0.02 to 0.04), and argmax agrees everywhere.
- **Rollback** passes with the new arm, and both planted faults fail the check.

**Whole model, verify forward against compiled plus multirow without the kernel.** The run uses `bench_verify_cost.py --rewind`: three processes per arm, T order rotated, ms are medians of 9 samples, and the ratio is stock over kernel per round.

| T | 4,096 stock | 4,096 kernel | ratio (3 rounds) | 16,384 stock | 16,384 kernel | ratio |
|---|---|---|---|---|---|---|
| 1 | 14.3 | 14.4 | 0.96–1.00 (stock path) | 18.4 | 18.3 | 0.97–1.02 (stock path) |
| 4 | 21.9 | 22.6 | 0.95–0.99 | 33.2 | 30.9 | 1.06–1.09 |
| 5 | 23.5 | 24.4 | 0.95–0.98 | 36.9 | 36.0 | 1.01–1.03 |
| 8 | 26.8 | 24.7 | 1.07–1.10 | 48.3 | 36.6 | 1.31–1.33 |
| 9 | 76.7 | 28.2 | **2.65–2.73** | 250.3 | 48.7 | **5.09–5.37** |
| 12 | 77.1 | 28.1 | **2.72–2.75** | 250.0 | 48.8 | **5.09–5.35** |
| 16 | 77.0 | 28.4 | **2.70–2.73** | 250.8 | 49.0 | **5.12–5.27** |

- **Noise.** A/A (the first T measured again at the end of each process) stayed within 6% at 4,096 and within 2.3% at 16,384.
- **Predictions met.** With the kernel, T = 9 costs 1.14x T = 8 at 4,096 (predicted at most 1.3x). T = 9 to 16 is 2.7x faster (predicted at least 1.8x).
- **Where it does not help.** Below 8 rows at 4,096 the kernel brings no gain: 0.95 to 0.99x, inside the A/A spread. So the threshold should be T >= 8 at short contexts; at 16,384, T >= 4 already gains 7%.
- **What it means for speculation.** At 4,096 context, verifying 15 drafts now costs about 2x a one-token step (28.4 against 14.4 ms). Before, it cost 5.4x. Drafters that propose long runs, such as n-gram lookup, become usable on MPS at long context.
- **Isolated attention** (28 separate caches, paired per-round ratios): 5.6x over stock at T = 9 to 16 with 4,096 keys, 7x at 16,384, and 1.65x at T = 8. At T = 1 to 2 it is slower than stock (0.5 to 0.75x). The kernel reaches 40 to 125 GB/s by a byte count that omits repeated reads across row groups. That is well below the bandwidth limit, so it has room to improve; what limits it is not established.

**Two side findings.**
- **In transformers 5.18, a compiled forward with T > 1 and a StaticCache that reaches sdpa_mask's causal-skip check breaks into 6 graphs and reads the GPU from the host.** That was every such forward measured here. The cause is `q_offset == 0` in `_ignore_causal_mask_sdpa`, and it costs about 1.5 to 2 ms per verify. The `mk_smallq` path compiles to 1 graph. A candidate upstream issue.
- **The step right after a long prefill pays a one-off penalty of about 30 ms.** It reaches any memory-bound work: a bare pass over the weights runs at about 72 GB/s instead of 265. One bandwidth-bound pass of about 10 ms clears it. The pattern fits memory or fabric clocks dropping during long compute-bound phases (an inference; confirming it needs powermetrics). A real request pays it once, after its prefill.

**Correction to the entry "The attention cliff above 8 query rows shows in the whole model".** That run re-prefilled before every sample, so each sample was a first step after a 4,096-token prefill and carried the penalty above.
- Its one-token figure of "27 to 45 ms at 4k" was mostly that penalty. A steady step at 4,096 is 14.3 ms.
- Its cliff ratios (1.63x to 1.87x) were diluted by the same penalty. In steady state, stock T = 9 costs 2.87x T = 8 at 4,096 and 5.2x at 16,384.

**After the cross-family review (2026-11-11).** GPT-6.1 Sol reviewed experiment E read-only ([its review](research/program/reviews/2026-11-11-smallq-attention-review-sol.md)). It reproduced the timing ratios from the raw data and found the kernel's indexing, barriers and reduce sound. It found three defects, all fixed:

- **The kernel path masked by `position_ids`, not by the cache slot.** A caller passing rotary positions that differ from the cache write offset would have got a wrong answer. The mask function now passes transformers' own cache offset (`q_offset`) to the attention function, as a copied int64 tensor, and the kernel masks by that.
  - A first version returned a view of the cache's counter, which each layer advances in place before its attention runs. A repeat-run test caught it, and a direct test now guards it.
- **A call with no mask and `is_causal=False` was turned causal.** Calls without the mask function's offset now go to transformers' `sdpa_attention_forward` unchanged.
- **A finite but huge `scale` (3e38) could give NaN** through an fp32 overflow in `scale * log2(e)`. Scales above 1e4 in magnitude, non-finite scales and tensor scales are now refused, and the wiring falls back to stock.

All 512 kernel tests pass (Sol's two counterexamples are now tests). The whole-model check against stock passes again at T = 1, 4, 8, 9 and 16 with 4,096 context: max logit gap 0.02, argmax agreeing everywhere. Rollback passes, and both planted faults fail. The timing table above was measured before these fixes. It used the model's default positions, where `position_ids` equal the cache slots, and the kernel itself did not change, so it stands.

**A prediction I had not reported.** The isolated prediction "within 1.5x of its own T = 1 at every T up to 16" failed: T = 16 over T = 1 was 1.47 to 2.48 across six 4,096-key runs, above 1.5 in five. The whole-model predictions held.

**Scope.** Whole-model correctness covers Qwen3-1.7B at T = 1, 4, 8, 9 and 16 with 4,096 context, not every T from 4 to 16 and not 16k. The post-prefill penalty is measured. That memory clocks cause it is a hypothesis.
