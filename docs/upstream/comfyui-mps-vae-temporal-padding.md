# ComfyUI PR: preserve temporal VAE padding on MPS

Branch `upstream/mps-vae-temporal-padding`, head `0252e040`, is an independent cherry-pick of reviewed source `9d9b4232` onto `65787d66`. The branch and integrated source are verified at their private remote refs. No public PR is open.

## Title

Fix corrupted Wan VAE temporal padding on MPS

## Description

Large five-dimensional temporal padding on MPS can silently discard the image data. With PyTorch 2.14.1, the first temporal shortcut in a 1024px Qwen Image 2.1 encode received an all-zero padded tensor. The latent remained finite but decoded into a flat, incorrect image. The 512px path was also affected less severely.

For nonzero temporal padding in `AvgDown3D` on MPS, prepend zero frames using `torch.cat`. Allocation stays on the input device and in its dtype. Other devices retain native `F.pad`; grouping, averaging, output shapes and gradients are unchanged. The implementation changes seven lines in `comfy/ldm/wan/vae2_2.py`.

This matches [PyTorch issue 194922](https://github.com/pytorch/pytorch/issues/194922), opened August 26, 2026. The [upstream padding fix](https://github.com/pytorch/pytorch/commit/72bca5e6d542341e6bcd795aaa293968843c29b2) landed August 31, but the installed 2.14.1 build lacks that change. No new PyTorch issue or dependency upgrade is needed for this patch.

## Verification

Measured October 4, 2026 UTC on an M1 Max with 32 GiB unified memory, macOS 15.7.9 and PyTorch 2.14.1.

The exact-size FP16 and FP32 MPS regressions both fail before the change and pass afterward. The integrated suite passes all 18 cases: temporal alignment for odd/even frame counts, a no-temporal-padding case, gradients, dtype/device preservation, and the large-image failure shape. The independent branch first passed its eight CPU cases, then all 18 CPU/MPS cases in a separate parent-owned run with the imported checkout paths asserted. These repeat the same contracts rather than adding independent scenarios.

An unhooked production encode followed by an ordinary FP16 decode restores the same source image:

| Resolution | Previous MPS FP16 PSNR | Corrected MPS FP16 PSNR | CPU FP32 reference PSNR |
|---|---:|---:|---:|
| 512 | 26.55 dB | 28.6220 dB | 28.6212 dB |
| 1024 | 7.98 dB | 38.3733 dB | 38.3755 dB |

The unhooked 1024px latent is byte-for-byte identical to the verified temporary candidate. At 512px, native MPS FP32 and CPU FP32 latents differ by relative L2 `1.88e-6`; the corrected FP16 result measures `0.002389` relative L2 in the separate FP16-versus-FP32 comparison. The temporary 1024px FP32 candidate agrees with CPU FP32 within `1.22e-6` relative L2. These are encoder roundtrip checks on one image, not a full-model training or generation acceptance claim.

The native 1024px run completes in 9.906 seconds with 2.383 GiB peak process RSS, normal memory pressure and no swap growth. An independent review covered the exact implementation and tests with complete coverage and no actionable findings (`20261004T041706106674Z-0a145e37301c`).

## Dependencies and scope

Independent of the other ComfyUI and Kitchen branches. The stable patch ID matches the source exactly; whitespace, Python compilation, contributor-trailer and merge checks onto fetched upstream `f1072eb0` pass. Human authorship, source reference, agent provenance and DCO sign-off are preserved.

This fixes the demonstrated `AvgDown3D` caller. It does not change unrelated padding operations or rewrite existing latent caches; affected caches are rebuilt separately.
