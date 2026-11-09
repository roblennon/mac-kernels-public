# ComfyUI PR (a): prefer fp16 VAEs on Apple GPUs

Branch `refresh/mps-fp16-vae` in `vendor/ComfyUI-upstream`: head `0908e9d7`, one commit on upstream `08ff3c11` (October 9, 2026). It supersedes `upstream/mps-fp16-vae` (`d3bdb46`) and is independent of every other PR here.

## Title

Prefer fp16 over bf16 for VAEs on Apple GPUs

## Description

On Apple GPUs, ComfyUI decodes most video and image VAEs in bf16 on macOS 14 and later, because those VAEs list bf16 first. On an M1 Max, bf16 decodes are 5–10× slower than fp16, and they land further from an fp32 reference:

| Same latent, M1 Max 32 GB, macOS 15.7, Torch 2.14.1 | fp32 | fp16 | bf16 (today) | PSNR vs fp32: fp16 / bf16 |
|---|---|---|---|---|
| Wan 2.2 VAE, 704×416×5 frames | 48.9 s | 10.5 s | 108.7 s | 76.1 / 57.4 dB |
| Wan 2.1 VAE, 832×480×5 frames | 36.7 s | 8.0 s | 82.0 s | 79.0 / 60.8 dB |
| Qwen-Image 2.1 VAE, 768×768 | 17.8 s | 4.2 s | 39.7 s | 75.0 / 56.8 dB |

None of the decodes produced non-finite values. With fp16, at most 0.002% of output values differ from fp32 by more than 1/255; with bf16, 0.5–2.1% do. Timings include moving weights to the GPU on the first call.

Encoding changes too. Against an fp32 encode of the same 512px image, fp16 latents differ by 0.07–0.33% relative L2 and bf16 latents by 0.4–1.2%, and fp16 encodes run 4–6× faster than bf16.

`vae_dtype` now returns fp16 when the VAE runs on MPS and lists fp16 among its working dtypes. These still win:
- the explicit `--fp16-vae`, `--bf16-vae` and `--fp32-vae` flags;
- `--force-fp32`;
- `--cpu-vae`, which puts the VAE on the CPU and keeps today's choice.

VAEs that do not list fp16 as a working dtype are unaffected. Other devices keep their current choice. The change is six lines in `comfy/model_management.py`.

Measured on one M1 Max only. M2 and later GPUs may handle bf16 differently, so numbers from other Apple chips would help.

## Test evidence

- `tests-unit/comfy_test/mps_vae_dtype_test.py`, 7 policy tests: fp16 on MPS; the usual choice when fp16 is not a working dtype; a CPU VAE on an MPS host; `--force-fp32`; each explicit flag. The fp16 case fails on unpatched upstream.
- Output parity and timing: `bench/vae_dtype_parity.py` in the mac-kernels repo decodes one latent per dtype under the GPU guard. Wan latents are saved generation latents (first two latent frames); the Qwen-Image latent is a CPU fp32 encode of a real image.

## Notes (not part of the PR)

- Evidence runs October 9, 2026, under `bench/gpu_runner.py --monitor`, against the refreshed branch. The earlier draft's "frame statistics identical" claim was replaced with the parity table above, and its "Apple GPUs emulate bf16" claim was dropped as unproven.
- Wan latents are saved generation latents. The Qwen latent is a CPU fp32 encode, because on Torch 2.14.1 the 5-D temporal `F.pad` in `AvgDown3D` zeroes its data on MPS in every dtype ([CU-01](comfyui-mps-vae-temporal-padding.md)). Wan 2.2 and Qwen encode references were CPU fp32; the Wan 2.1 VAE has no `AvgDown3D` and used MPS fp32. The Qwen encode figure above was measured with CU-01 applied; without it, Qwen 2.1 encodes on MPS are about 13% off in fp16, bf16 and fp32 alike, and sometimes all-NaN in half precision.
- An independent review of the commit found nothing blocking. Its two test findings (host-OS dependence, CPU-only import order) are fixed. Full `tests-unit/comfy_test`: 198 passed, 14 skipped.
