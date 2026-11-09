# Inference acceleration survey for M1-M4 (2026-10-02)

Tags: [V] primary source read (repo, release notes, model card, arXiv abstract); [S] secondary; [I] my inference. Web fetches were summarized by a small model, so numbers from them are leads to re-check before a decision.

## Bottom line

Per-step linear layers sit at the fp16 ALU ceiling, so nothing below beats them by more than a few percent. The remaining wins are fewer forward passes, cheaper decode and load, and less attention work. Your own findings show where wall time goes: Wan 5B sampling is 4 x 3.3 s but VAE decode is 74 s; H3 sampling is about 32 s but text encode (168 s, cached) and decode (55-85 s) dwarf it; the 6-bit H3 LoRA merge costs 6.5 min per load. Step counts are already 4 to 8, which kills most caching work. Your torch 2.14.1 already contains the new fused Metal attention, so attention is mostly done.

## 1. Attention

- **PyTorch >= 2.13 fused prefill SDPA.** Simdgroup-MMA flash kernel, head dims 32-256, fp16/bf16/fp32, bool or same-dtype masks [V: `PrefillAttention.h`, `Attention.mm` at v2.14.0]. PR #181575 reports 3-4x over the MPSGraph path at 1-4k tokens (bf16; GPU not stated) [V]. Your 16.5 s/step run likely already uses it. Check: A/B with `PYTORCH_MPS_DISABLE_PREFILL_ATTENTION=1` (exists in the source). Effort: hours.
- **Metal FlashAttention (Draw Things, mfa v2).** ~2% gain on M1/M2, up to 20% on M3/M4 for FLUX [V: engineering.drawthings.ai, Jan 2025]; the 4.6x claim is M5 Neural Accelerators only [V]. Not worth porting.
- **MLX fast SDPA.** Same tiling family; issue #3837 is about decode, not prefill [V]. No gain expected.
- **FlexAttention on MPS (2.13).** Fast only when very sparse: 4.15x at 0.8% density, 1.07x at 12.5%, 0.34x at 50% (S=8192) [V: PR #182552]. Video masks at 8.6k tokens are denser. Skip.
- **Radial/STA/SVG2/VSA.** Radial: up to 1.9x on Wan 2.1 14B at long lengths, needs LoRA tuning [V arXiv 2506.19852]; VSA needs training [S]. My estimate: at 640x384x33 (8,640 tokens) attention is ~23% of Wan 14B FLOPs and ~12% for the 5B [I], so 50% sparsity gives at most ~10-15% on the 14B and ~0 on the 5B. Exception: H3 ships VSA routing; FastVideo has an Apache-2.0 Metal simdgroup block-sparse kernel (tile 64, D=128) [V: `minimax_h3_vsa_simd.py`]. Whether your pruned checkpoint has the trained gate is unknown. Effort: 2-3 weeks.
- **SageAttention/INT8 attention.** No int matrix units before M5. Skip.

## 2. Caching and step skipping

TeaCache, MagCache (2.1-2.7x [V]), EasyCache (2.1-3.3x [V]), Spectrum (4.67x on Wan 14B, 10 evaluations of 50 [V], CVPR 2026) and cache-dit all report gains against 50-step baselines. At 4-8 steps there is nothing to skip. ComfyUI core already has `EasyCache` and `LazyCache` nodes in pure PyTorch [V: `comfy_extras/nodes_easycache.py`]; a 20-minute test on Krea 2 Turbo (8 steps) is the only cheap check. Prefix KV cache: Qwen-Image 2.1 has it built in and on by default (`QwenImage21Cache`, [V docs.comfy.org]); confirm your backend does not disable it. Token merging: the only Apple study I found (arXiv 2605.16259, M3 Ultra, SDXS) says quantization and parallelism gave no benefit and does not isolate ToMe [S].

## 3. Step distillation (largest lever)

- **Krea 2 Turbo 4-step LoRA** (lvladikov, 2026-08-20): rank 64, plain LoRA, guidance 0; 54.5 s vs 88.7 s at 1024 on Apple MLX bf16; judge preferred teacher in 6 of 45 [V model card]. A 2-step draft LoRA exists (2026-09-17, 4.2x denoise) [S comfyui-wiki]. Community, Krea 2 Community License.
- **Qwen-Image 2.1 Fun-Acc 4-step LoRA** (Alibaba PAI, PDD, 40 -> 4 NFE, rank 64, Qwen Research license) [V]. Needs its custom sigma schedule (1.0, 0.917, 0.786, 0.549, 0.0); no official ComfyUI support; small dense text degrades [V].
- **Wan 2.2 5B.** FastMetal uses 3-step DMD2 (4 -> 3 is only 25%) [V Hao AI Lab blog, 2026-08-10]. rCM/TurboDiffusion gives 4 steps on Wan 14B, quality drops below that [V arXiv 2512.16093].
- Integration: LoRA loader plus sampler sigma list. Effort: days. Blocker: merging into INT8/6-bit requantizes (see 4).

## 4. Quantization beyond weights

Activation quantization is dead on M1-M4 (no int8 rate advantage; `_int_mm` absent). The useful idea is SVDQuant/Nunchaku's fused low-rank branch [V arXiv 2411.05007]: x@A fused into the prologue that already reads x, (x@A)@B added in the epilogue. FLOP overhead at rank 64, K=N=6144 is about 2% [I]. Payoff for you is not step time but exactness and load time: no 6.5-minute requantizing merge, runtime LoRA strength, and every distill LoRA in section 3 applies cleanly. Effort: 1-2 weeks inside comfy-kitchen.

## 5. GEMM level

Your bare fp16 GEMM is ~9 TFLOPS of 10.4 [V findings]; MFA reports 83% ALU utilization on M1 Max [V]. No custom GEMM will beat it meaningfully. Epilogue fusion of SwiGLU/gates saves one memory pass per layer, ~1-2% of a step [I]. Ready code for fused RMSNorm+modulation+residual and fused RoPE via `compile_shader` runs on any MPS (MIT) [V: ComfyUI-AppleSilicon-FP8 README, 2026-09-23]; that repo's 2.7x conv3d is M5-only. Upstream: I found no PyTorch issue for the wide-K (K=16384) slow `mm`; nearest is #189847 (half-precision 3D `F.linear`, closed 2026-07-14) [V]. File one with your repro. Effort: days.

## 6. Memory and decode

- **Tiny VAEs.** ComfyUI core already loads `lighttaew2_2`, `lighttaew2_1`, `taeh3` through VAELoader [V: `nodes.py`, `comfy/sd.py`]. lightx2v table (H100): Wan 2.2 VAE decode 3.13 s vs 0.09 s (taew2_2); Wan 2.1 `lightvaew2_1` 2-3x faster at half memory [V]. Quality is softer. Use for drafts; full VAE for finals. Effort: a day plus quality check.
- **Full-VAE speed.** 74 s for 33 frames is within ~2x of what the FLOPs imply at 9 TFLOPS [I]; gains need conv3d as im2col + our GEMM (unmeasured on M1). Effort: 1-2 weeks.
- **Expert swap.** Wan 14B swaps once per clip (97 s step); keep one expert resident and prefetch the other. SSD streaming (h3.c) costs 26-84% speed and is unneeded at 17 GB [V].

## 7. Adjacent ideas

- **h3.c compute knobs** [V README]: reuse of whole denoiser outputs, 45 of 50 blocks, middle-block token pair-pooling, 384x384 internal canvas for 512 output; all change composition, tune per model. FastVideo adds spatial fast mode (denoise small, upscale decoded pixels) and two-pass refine [V]. Effort: 1-2 weeks.
- **FastVideo MLX runtime** (Apache-2.0, 14B at 21.7 GB peak on M4 Max) [V]: reference, not a drop-in for ComfyUI.
- Speculative decoding and parallel sampling: no mature DiT analog found; parallel sampling trades compute for latency and you are compute-bound.

## Top 5 (speedup x feasibility)

1. **Step-reduction LoRAs** (Krea 4-step, Qwen Fun-Acc, Wan 5B 3-step): 1.6-2x on Krea per image, up to 10x on Qwen if you run 40 steps. Days.
2. **Tiny VAE draft path** (lighttaew2_2, lighttaew2_1, taeh3): decode 74 s -> seconds, ~60% of Wan 5B wall time. A day.
3. **Fused low-rank LoRA side path**: exact, kills the merge, enables #1 on quantized weights. 1-2 weeks.
4. **Faster full-quality VAE** (conv3d as im2col + fp16 GEMM) and fused adaLN/RoPE elementwise kernels. 1-2 weeks.
5. **H3 compute reduction** (VSA kernel if the gate is present, block skipping, token pooling). 2 weeks, quality risk.

## Do not bother

- INT8 attention, W8A8: no integer units before M5.
- Custom fp16 GEMM, MLX or MFA ports: at 87% of peak; MFA gains ~2% on M1/M2.
- TeaCache family, Spectrum, cache-dit at 4-8 steps; useful only on 20-50-step base models.
- FlexAttention for video masks below ~10k tokens.
- SSD or layer streaming: only for models over 32 GB.
- Parallel sampling, LLM speculative decoding, `torch.compile` fusion (no MPS evidence found).
