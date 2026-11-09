# ComfyUI PR: fp16 VAEs on Apple GPUs

Head: `roblennon/ComfyUI` branch `mps-fp16-vae` (`751a75c0`, one commit rebased onto master `8010b48d`, October 13, 2026). Base: `Comfy-Org/ComfyUI` `master`.

## Title

Prefer fp16 over bf16 for VAEs on Apple GPUs

## Body

On Apple GPUs, ComfyUI decodes most video and image VAEs in bf16 on macOS 14 and later, because those VAEs list bf16 first. On an M1 Max, bf16 decodes are 5 to 10x slower than fp16 and land further from an fp32 reference:

| Same latent, M1 Max 32 GB, macOS 15.7, Torch 2.14.1 | fp32 | fp16 | bf16 (today) | PSNR vs fp32: fp16 / bf16 |
|---|---|---|---|---|
| Wan 2.2 VAE, 704x416x5 frames | 48.9 s | 10.5 s | 108.7 s | 76.1 / 57.4 dB |
| Wan 2.1 VAE, 832x480x5 frames | 36.7 s | 8.0 s | 82.0 s | 79.0 / 60.8 dB |
| Qwen-Image 2.1 VAE, 768x768 | 17.8 s | 4.2 s | 39.7 s | 75.0 / 56.8 dB |

None of the decodes produced non-finite values. With fp16, at most 0.002% of output values differ from fp32 by more than 1/255; with bf16, 0.5 to 2.1% do. Timings include moving weights to the GPU on the first call.

Encoding changes too. Against an fp32 encode of the same 512px image, fp16 latents differ by 0.07 to 0.33% relative L2 and bf16 latents by 0.4 to 1.2%, and fp16 encodes run 4 to 6x faster than bf16.

`vae_dtype` now returns fp16 when the VAE runs on MPS and lists fp16 among its working dtypes. These still win:

- the explicit `--fp16-vae`, `--bf16-vae` and `--fp32-vae` flags;
- `--force-fp32`;
- `--cpu-vae`, which puts the VAE on the CPU and keeps today's choice.

VAEs that do not list fp16 as a working dtype are unaffected. Other devices keep their current choice. The change is six lines in `comfy/model_management.py`.

## Tests

`tests-unit/comfy_test/mps_vae_dtype_test.py`, seven policy tests: fp16 on MPS; the usual choice when fp16 is not a working dtype; a CPU VAE on an MPS host; `--force-fp32`; each explicit flag. The fp16 case fails on unpatched master. Full `tests-unit/comfy_test` on the rebased branch: 201 passed, 14 skipped. Ruff clean.

Measured on one M1 Max only. M2 and later GPUs may handle bf16 differently, so numbers from other Apple chips would help.
