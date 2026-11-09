# ComfyUI PR (d): conv3d as stacked-tap conv2d on Apple GPUs

Branch `refresh/mps-conv3d` (private remote), head `66d9e336`, two commits on `origin/master` `08ff3c11` (2026-10-09). It supersedes `upstream/mps-conv3d` (`cc9fc6c7` on `65787d66`), which stays untouched. Independent of the other branches.

Commits:
- `e424828a` perf(mps): run conv3d as stacked-tap conv2d on Apple GPUs (original)
- `66d9e336` fix(mps): read temporal padding from a shared zero frame (this refresh; carries the padding, stride and gradient tests from `e11d9226` in the integrated runtime)

Verdict: **still needed on both Torch 2.14.1 and the 2.16 nightly.** Native conv3d is 3-4x slower in fp16 and 11-18x slower in fp32 at a Wan decoder shape on both versions. The F.pad corruption the second commit avoids is fixed on nightly, but the refreshed helper no longer pads at all, so nothing in the branch is version-specific.

## Title

Run conv3d as stacked-tap conv2d on Apple GPUs

## Description

MPS runs conv3d much slower than conv2d. On MPS, `comfy.ops` `Conv3d` now computes each output frame as one conv2d over the time taps stacked on the channel axis; a kernel one frame deep without temporal padding runs as a single batched conv2d. Temporal zero padding is never materialized: a tap that falls outside the input reads one shared zero frame. It handles any kernel, stride and numeric zero padding. groups != 1, dilation != 1, string padding, non-zero padding modes, inputs shorter than the kernel and an empty time axis keep torch conv3d (so errors are torch's own). Non-MPS devices are untouched: the dispatch is one `input.device.type == "mps"` test in `_conv_forward`.

Single conv, (1,384,5,52,88) 384->384 k3 pad 1, M1 Max 32 GB, 2026-10-09:

| | torch 2.14.1 | torch 2.16.0.dev20261009 |
|---|---|---|
| fp16: native conv3d / stacked | 76.6 ms / 19.0 ms (4.0x); a later run under concurrent load 122 ms / 40 ms (3.1x) | 76.6 ms / 18.5 ms (4.1x) |
| fp32: native conv3d / stacked | 399 ms / 22.0 ms (18x); later run 430 ms / 37.5 ms (11x) | 401 ms / 21.1 ms (19x) |
| max error vs CPU fp32 (output scale 10.9) | native 5.3e-3, stacked 5.3e-3 (fp16); 5.0e-5 both (fp32) | same |

Memory, (1,256,6,352,640) 256->256 k3 pad 1, fp16, torch 2.14.1: native conv3d +1.97 GiB driver memory and 2.9-4.2 s; stacked +2.31 GiB and 0.85-0.91 s. The padded copy of the first draft cost +3.31 GiB.

End to end, fp16 VAE decode on an M1 Max 32 GB (2026-10-03, original branch; not re-measured in this refresh):

| | Before | After | PSNR vs before |
|---|---|---|---|
| Wan 2.2 VAE, 704x416x33 | 72.8 s | 21.1 s | 66.5 dB |
| Wan 2.2 VAE, 1280x704x81 | 542 s | 155 s | |
| Wan 2.1 VAE, 832x480x81 | 139 s | 43.8 s | 67.6 dB |
| MiniMax H3 VAE encode, 22 frames 480x288 | 49.4 s | 16.8 s | |

## What the probes showed (2026-10-09, M1 Max 32 GB, macOS 15.7)

- **5D F.pad on MPS, torch 2.14.1**: the interior after a temporal pad differs from the input at (1,1,1,512,512) and at the Wan decoder shape (1,256,6,352,640), fp16 and fp32; (1,384,5,52,88) is fine (4,576 elements per frame, under the ~2^16 threshold PR #15902 describes). **Nightly 2.16.0.dev20261009**: all shapes intact. The ported regression test fails before the fix on 2.14.1 (2 failed) and passes after.
- Stacked-tap parity equals native parity against CPU fp32 at the decoder shape on both versions (table above).

## Test evidence (2026-10-09)

- `tests-unit/comfy_test/conv3d_mps_test.py`: 59 passed on torch 2.14.1 and on 2.16.0.dev20261009 (MPS through the repo's GPU guard). Against CPU fp32 torch conv3d: the Wan 2.1/2.2, H3 and Wan DiT configurations; temporal padding wider than the kernel reach with stride 2; a one-deep kernel with temporal padding and stride; an even kernel; asymmetric padding; the helper on CPU; fp32, fp16, bf16; no bias; noncontiguous input with stride and gradients for input, weight and bias; manual_cast weights; the causal_zero single-frame path; the fallbacks; error parity with conv3d on too-small and empty inputs; the large-spatial padding regression.
- Failing-first: the empty-time-axis parity case fails on the previous helper (it returned bias frames) and passes now; the padding regression fails with F.pad on 2.14.1.
- `tests-unit/comfy_test` in this worktree: 250 passed, 14 skipped (torch 2.14.1).
- CPU-only collection: importing `comfy.ops` without `args.cpu` raises "Torch not compiled with CUDA enabled" on CPU-only torch (simulated); the module now sets `args.cpu` the way neighbouring tests do.

## Independent review (read-only, 2026-10-09)

One blocking (CPU-only collection crash: reproduced, fixed), two should-fix (activation copies without memory evidence: measured, padded copy removed; coverage gaps: added the listed configurations and error parity), two nits (empty time axis: reproduced, fixed; "exact on every version" claim: dropped). The reviewer's own CPU differential pass: 648 helper cases and 36 module cases bitwise-equal to conv3d, gradcheck on both branches.

## Upstream overlap (checked 2026-10-09)

- Issue #2044 "Conv3D is not supported on MPS" (2023, open) is the historical context; conv3d runs on MPS today, slowly.
- #15902 (open): replaces the H3 VAE's temporal `F.pad` with `torch.cat` for the same pytorch#194922 corruption. Independent of this branch (this helper never pads); the two together cover the H3 encoder.
- No conv3d or MPS convolution PR in the last two months overlaps this change.

## Dependencies and scope

None. Apple-specific: on CUDA, cuDNN conv3d is fast and nothing changes.

## Notes (not part of the PR)

- Re-reviewed independently on October 9: replacing the ported `torch.cat` zero-frame padding with the no-pad helper is correct and better. Read line by line, a tap outside the input reads one shared zero frame, the reshaped weight maps tap d to channel block d, bias is added once per frame, and the output length is the standard formula; only groups = 1, dilation 1 and numeric zero padding are routed, which the guard enforces. An independent probe over 132 configurations (kernel depth 1–4, time stride 1–2, time pad 0–2, three spatial stride/pad pairs, fp32 and fp16) matched CPU conv3d to 6e-7 (fp32) and 5.6e-4 (fp16), and the input gradient matched to 1.7e-7. Temporary memory is kt frames instead of the whole padded input, which is why it beats the port.

- The stacked path still uses about 0.34 GiB more driver memory than native at the (1,256,6,352,640) fp16 shape (the per-frame 3-tap stack, 330 MiB, plus the output buffer). A decode that sits within 0.3 GiB of the MPS limit could tip; not measured end to end. SeedVR's `_conv_without_padding_copy` optimisation is bypassed on MPS (the review noted it); not measured.
- The end-to-end decode table is from 2026-10-03 and was not re-measured; the refresh ran unit tests and microprobes only.
- Per-conv timings varied about 1.6x between runs (another job was on the GPU; the guard serializes only guarded runs). Ratios held at 3-4x fp16, 11-19x fp32.
