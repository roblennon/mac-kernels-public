# ComfyUI PR (e): run INT8 and W4A8/W6A8 natively with comfy-kitchen's Metal backend

Branch `upstream/mps-kitchen-formats`, one commit (`f9d8926`) on `65787d6`. It merges cleanly onto the current `origin/master`.

## Title

Run INT8 and W4A8/W6A8 weights natively on Apple GPUs with comfy-kitchen's mps backend

## Description

`get_disabled_quant_formats` disables the INT8 formats when the device lacks `torch._int_mm`, which MPS does not implement. On Apple GPUs, INT8 (ConvRot) checkpoints are therefore dequantized in full on every call, and W4A8/W6A8 do not run. comfy-kitchen's new `mps` backend runs these formats weight-only and needs no integer matmul.

`mps_quant_formats(device)` maps the backend's capabilities to formats:
- `int8_linear` -> `int8_tensorwise`;
- `w4a8_int8_linear` -> `asym_w4a8_int8` and `w6a8_int8`.

`get_disabled_quant_formats` keeps those enabled. It returns nothing when:
- the backend is missing, disabled or unavailable (comfy-kitchen keeps listing a disabled backend's capabilities, so this is checked explicitly);
- the device is explicitly not MPS.

25 lines in `model_management.py` and `ops.py`.

M1 Max 32 GB, seconds per step, stock against this change:

| Model | Stock | With the mps backend |
|---|---|---|
| Krea 2 Turbo INT8 ConvRot, 512x512 | 8.7 | 3.7 |
| Krea 2 Turbo INT8 ConvRot, 1024x1024 | about 35 (noisy run) | 24 |
| Qwen-Image 2.1 INT8 ConvRot, 1024x1024 | 15.1 | 8.8 |
| MiniMax H3 INT8, 480x288x22 | 38 to 74 (swapping) | 8.4 to 11.8 |
| MiniMax H3 W6A8, 480x288x22 | does not run | 6.5 |

## Test evidence

- `tests-unit/comfy_test/mps_quant_formats_test.py` (4 tests): formats on MPS; none for a disabled or unavailable backend; none for an explicit CPU device on an MPS host.
- `tests-unit/comfy_test`: 194 passed, 14 skipped.

## Dependencies

- **Needs the comfy-kitchen mps backend PR** (and a comfy-kitchen release containing it) to have any effect. Without it this is a no-op, so it can merge before or after.
- Bumping `comfy-kitchen` in `requirements.txt` to the release with the backend belongs here or in a follow-up, once that release exists.
- Apple-specific.
