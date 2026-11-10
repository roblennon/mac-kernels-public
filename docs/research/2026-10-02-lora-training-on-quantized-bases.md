# LoRA training on our INT8 ConvRot base (M1 Max 32 GB)

2026-10-02. Tags: [V] read in source or measured here today, [S] secondary, [I] inference. Repos cloned today at latest commits (ai-toolkit 2026-09-27, mflux and diffusers 2026-10-02, bitsandbytes/torchao/quanto Sept 2026).

*Dated note, 2026-11-10: this survey predates mflux's Qwen-Image trainer, which mflux added on 2026-11-06 (see [the funder research](../review/2026-11-09-funder-research.md)). Read any statement here about what mflux can or cannot train as true on 2026-10-02 only.*

## Decision first

Two findings change the plan. [V] **ai-toolkit already trains on MPS over a ConvRot INT8 base**: qfloat8 is remapped to `convrot8` on MPS (`toolkit/config_modules.py:727`), with an autograd STE (`toolkit/util/convrot_quant.py:992-1040`) and Krea 2, Qwen-Image 2.1 and Wan 2.2 5B arches. Its MPS training branch (`convrot_quant.py:1442-1480`) dequantizes the whole layer weight per call. Our kernel replaces exactly that. [V] **mflux already trains Krea 2 Raw on MLX** (QLoRA q8, 512px, PR #462, merged 2026-08-13) but has no Qwen-Image or Wan trainer (`src/mflux/models/*/training_adapter`: flux2, krea2, z_image, ernie).

Recommendation: write a `torch.autograd.Function` around our kernel and plug it into ai-toolkit's quantizer interface. First milestone: Krea 2 Raw, 512px, about 1.5 h of training.

## 1. Backward for a frozen quantized weight

Forward `y = x Wᵀ`; backward needs only `dx = dy · W`, never `dW`. With ConvRot, `y = (xR)(WR)ᵀ`, so `dx = (dy · W′) R`, where W′ is the stored rotated weight and R the block Hadamard (symmetric, orthogonal, self-inverse) [V].

How others do it [V]:
- bitsandbytes `MatMul4Bit.backward`: `grad_output @ dequantize_4bit(B, state)`; the `grad_B` branch is "not supported by PyTorch" (`autograd/_functions.py:300-380`). It has an MPS 4-bit backend; 8-bit optimizers on MPS are marked in progress (README.md:179-190).
- torchao `LinearNF4`: `grad_output @ weight.to(grad_output.dtype)` (`nf4_tensor.py:1053-1064`).
- quanto `QuantizedLinearFunction`: explicit backward so ops without autograd still work (`optimum/quanto/tensor/function.py:21-70`).
- **Nunchaku has no training mode**: no backward code, quantized tensors `requires_grad=False` (`nunchaku/models/linear.py:97-342`). The first W4A4 trainer is FourTune (arXiv 2607.05711, 2026-07-07, same MIT group) [S].
- ai-toolkit: custom op with `register_autograd`; saves only `qdata` and scales, re-dequantizes in backward (`convrot_quant.py:992-1040`); bit-packed variants save packed codes (`:2138-2200`).
- MLX `QuantizedMatmul::vjp`: gradient for x only (`mlx/primitives.cpp`).

Wrapper design [I]:
1. `forward` calls our fused kernel; `ctx.save_for_backward(qdata, scales)` only. Never save a dequantized weight: `F.linear(x, dequant(W))` makes autograd keep it, 26 GB across Krea 2 without checkpointing.
2. `backward` returns `dx` only (gate on `needs_input_grad[0]`), computed as chunked decode-GEMM over 1024 output columns (33 MB transient), then R. The widest full layer is only 201 MB, so chunking is a nicety; the real win is holding no weights across layers. Best: a Metal kernel contracting over N.
3. Verified here (torch 2.14.1): custom Function plus non-reentrant `torch.utils.checkpoint` works on MPS.

**Trap, measured [V]:** mean-MSE upstream gradients sit near 2e-5, in fp16's subnormal range. Folding the per-row weight scale (~5e-3) into `dy` before the GEMM gave 2.8% error. Decoding `s·q` per chunk, or normalizing `dy` rows by a power of two as our forward does, gave 3.5e-4, the fp16 floor.

## 2. Memory and step time

Architectures [V]: Krea 2 has 28 blocks, d=6144, 48 heads, MLP 16384, 16 px per token (`mflux/.../krea2_transformer/`). Qwen 2.1 has 32 blocks, d=4096, 32 heads, MLP 12288 (`qwen21_transformer.py:26-37`; 16 px per token inferred). Wan 5B has 30 blocks, d=3072, FFN 14336, 24 heads, 32 px × 4 frames per token (HF config).

Budget: MPS allows **21.3 GiB** here [V, `recommended_max_memory()`]. Embeddings and latents are cached, so encoders are off-GPU. Adam fp32 is 16 B per parameter; rank-32 LoRA on all attention and MLP linears is 107M parameters on Krea 2 (1.7 GB), 84M on Qwen, 81M on Wan. Optimizer state is small, so 8-bit optimizers do not matter; fused AdamW is fine.

The unknown is attention backward: **MPS SDPA has no dedicated backward**; it uses the composite math path (pytorch#179294, 2026-04-03 [S]), so each N×N tensor costs heads × N² × 2 B, with 3 to 4 live. My own SDPA measurement was still queued behind the GPU lock at writing, so attention figures are [I].

| Model (INT8) | Tokens | Weights | LoRA+Adam | Ckpts + block | Attention | Total GB | Fits 21.3? |
|---|---|---|---|---|---|---|---|
| Krea 2 512px | ~1300 | 13.0 | 1.7 | 1.0 | 0.5 | ~16 | yes |
| Krea 2 1024px | ~4350 | 13.0 | 1.7 | 3.0 | 6 to 7 | 24 to 25 | no; ~18 with chunked attention |
| Qwen 2.1 512px | ~1300 | 7.3 | 1.3 | 0.8 | 0.4 | ~10 | yes |
| Qwen 2.1 1024px | ~4400 | 7.3 | 1.3 | 2.2 | 4 to 5 | 15 to 16 | yes |
| Wan 5B 704x416x33 | 2574 | 5.1 | 1.3 | 0.9 | 1.3 | ~9 | yes |
| Wan 5B 1280x704x81 | 18480 | 5.1 | 1.3 | 4 | 16 per tensor | n/a | needs flash backward |

Step time [I]: forward is measured (`docs/findings.md`: Krea 2 3.65 s at 512, 16.5 s at 1024; Qwen 2.4 s and 8.8 s; Wan 5B 3.3 s). Checkpointed training is about 3× forward. Estimates: **Krea 2 512 11 to 14 s/step, 1024 50 to 60 s; Qwen 512 7 to 9 s, 1024 27 to 32 s; Wan 5B 10 to 12 s.** (2026-11-09 rerun: Krea 2 forward is 3.75 to 3.78 s at 512 and 15.9 to 16.1 s at 1024; the estimates were not recomputed.)

## Loss recipes to copy [V]

All use flow matching: `x_t = (1-σ)x0 + σε`, target `ε - x0`, unweighted MSE.
- **diffusers** `examples/dreambooth/train_dreambooth_lora_krea2.py`: uniform `u` by default (`logit_normal` optional), weighting 1; Krea authors' rank 32, alpha = rank, lr 3e-4 to 7e-4, PEFT gaussian init (B = 0); target list at lines 1354-1380. A `_qwenimage21.py` sibling exists (not read).
- **ai-toolkit** `BaseSDTrainProcess.py:1170-1300`: default `timestep_type: sigmoid` (`config_modules.py:557`). Krea 2 dynamic exponential shift 0.5 to 1.15 over 256 to 6400 tokens (`krea2/krea2.py:81-96`); Qwen 2.1 shift 0.5 to 0.9 (`qwen_image_2.py:61-73`); Wan 5B static shift 5.0 (`wan22_5b_model.py`). Loss in fp32 (`SDTrainer.py:960`).
- **musubi-tuner** `docs/krea2.md:125-137`: shift 2.5 at 1024px or `krea2_shift` (1.6 at 256px to 3.2 at 1280px), rank 32, lr 1e-4. `docs/wan.md:177-197`: shift 3.0 start; official Wan 2.2 inference shift 12 T2V, 5 I2V.
- **mflux** `common/training/trainer.py`: uniform step index, rank 16, lr 2e-4. SimpleTuner's Krea 2 doc: Apple GPUs "not currently a recommended target".

## 3. MPS traps

- **bf16 is slow on this M1 Max.** Measured: GEMM 1.75× slower than fp16, conv3d 13×; whether that is emulation was not established (`findings.md`). Diffusers' Krea 2 script raises on bf16 under MPS (`:1160, 1318`); SimpleTuner force-overrides it (`trainer.py:1262`). ai-toolkit #925 [S]: bf16 + convrot8 + automagic on MPS gave NaN loss and all-zero `lora_B`; fp32 + AdamW worked. Plan: fp16 compute, fp32 LoRA masters.
- **fp16 range.** Forward is hardened; backward needs the same row scaling (§1) plus dynamic loss scaling. [V] On torch 2.14.1 `GradScaler("mps")`, `autocast("mps")` in fp16 and bf16, fused AdamW and `torch.optim.Adafactor` all run.
- **mflux lessons** [V, #739, #740]: zero-init B (their step 0 differed from base by ~3% and showed an 8 px grid); cast LoRA factors to activation dtype inside forward, or the net silently runs fp32 (74 s vs 9.5 s per step).
- **Checkpointing** works in non-reentrant mode. **torch.compile**: a trivial inductor elementwise compile ran on MPS; ai-toolkit wraps its MPS fusion in a fallback (`convrot_quant.py:1236-1260`). Do not depend on it.
- **Missing ops:** I found no reported backward gap for the three DiTs, but ai-toolkit has MPS failures (`histc` #853; FLUX.2 non-convergence #871, closed stale), and nobody has published a Krea 2 or Qwen 2.1 MPS training run on 32 GB.

## 4. Training on INT8, deploying elsewhere

[V] ConvRot INT8 weight error is about 1% (ai-toolkit docstring); our fp16 path measured 4e-4 relative output error. The LoRA learns against `W_q`; on bf16 `W` the unseen term is `W_q - W`, small beside typical LoRA deltas at INT8, larger at W4A8. I found no paper testing cross-precision transfer of diffusion LoRAs. FourTune trains on a frozen 4-bit base and compares with BF16 and NF4 LoRA, but I saw no train-4-bit, apply-to-BF16 test [V, read the PDF]. Portability is practice, not measured fact [I].

LoftQ (2310.08659) and PiSSA/QPiSSA (2404.02948) shrink the starting error, mainly at 2 to 4 bits [V abstracts]. **Avoid them for a portable LoRA**: LoftQ bakes a `W_q` correction into A·B and PiSSA moves principal components out of the base, so both break on a different base. QA-LoRA (2309.14717) solves merging into a quantized base; we apply adapters at runtime, so it is moot. Use B = 0 at INT8 and A/B-test: Qwen 2.1 bf16 (14 GB) fits on this GPU, so train on INT8 and render the same seeds on both bases.

## 5. MLX vs MPS

mflux q8 at ~1300 tokens ran 27 to 37% slower than dense on M2 Ultra (#740), and defaults to bf16 (`model_config.py:9`), which MLX does not accelerate on M1 [S]. ConvRot to MLX affine is exact [I]: q+128 with bias −128·scale, plus a separate rotation op. No published MLX-vs-MPS diffusion training comparison exists; microbenchmarks are 2023-24 [S]. Scaling mflux's M2 Ultra number (9.5 s dense, 13.0 q8 at 512; 60 or 76 cores) to 32 cores gives roughly 25 to 35 s/step against my 11 to 14 [I, wide error]. Settle it with 10 steps of `mflux-train` on `fxd0h88/krea-2-raw-mlx-q8`, about 30 minutes after download.

## 6. Plan

**Milestone 1.** Krea 2 Raw INT8 ConvRot, 512px, 8 to 12 images, rank 16 attention + MLP, 400 steps, batch 1, lr 1e-4, `krea2_shift`, fp16. Pass when: (1) one-block gradient matches an fp32 dequantized reference under 1e-3; (2) step-0 output equals base; (3) one image overfits in 100 steps; (4) the LoRA on Turbo through our ComfyUI backend imprints the style; (5) step time and memory beat the mflux baseline. Cost: a few days to build, about 1.5 h to train.

**Milestone 2.** Qwen 2.1 at 1024px, plus the INT8 versus bf16 transfer A/B.

Build order: (1) autograd Function with chunked decode backward, then W4A8/W6A8 with grouped scales (no scale folding); (2) plug into ai-toolkit's `OstrisQuantizer.forward`, inheriting caching, sampling, previews and ComfyUI-format saves, upstreamable; (3) runtime LoRA side path shared with the inference kernel already in progress (`bench/bench_lora_sidepath_kernel.py`); (4) query-chunked attention with per-chunk checkpoint for 1024px, later a Metal flash backward.

Risks: attention backward memory (largest); fp16 backward range on Raw weights is untested; ai-toolkit MPS bugs (#925, #871); Qwen 2.1 license is research-only; backward rotation must match the forward kernel exactly.
