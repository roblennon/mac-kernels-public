# ComfyUI UniPC precision: support #16574 instead of a separate PR

**October 9, 2026 disposition:** withdraw `upstream/mps-unipc-float64` (`d9602ea`). Upstream [#16574](https://github.com/Comfy-Org/ComfyUI/pull/16574) fixes the same issue ([#16573](https://github.com/Comfy-Org/ComfyUI/issues/16573)) on every device. Our contribution is MPS confirmation on that PR, plus an optional regression test.

## Why the branch is withdrawn

- #16574 computes the B(h) recurrence in float64 on the CPU for all devices. That recurrence is the actual fix.
- Our extra float64 `h_phi_1` and `B_h` change nothing measurable. The `expm1` error in `h_phi_1` is about one machine epsilon and is not amplified; only the 1/h^k recurrence amplifies error.
- Our MPS-only gate kept a real CPU/CUDA bug. On the shift-8, 20-step schedule, float32 master on CPU gets `b` up to 109% wrong. The old commit message's "about 1e-7" holds only for final samples on a toy model, not for the coefficients.
- The branch's whole-sample tests do not separate broken from fixed: they pass on unpatched master on MPS. Its CPU test checked how the code is built, not what it does.

## Evidence on the PR head

October 9, M1 Max, macOS 15.7.9, Torch 2.14.1, under the GPU guard. The corrector-weight test from our branch, made device-agnostic, compares every `b` vector and corrector weight from `sample_unipc` on a shift-8 schedule against a float64 CPU run.

| Code | CPU bh1/bh2 | MPS bh1/bh2 |
| --- | --- | --- |
| master `08ff3c11` | fail | fail |
| #16574 head `045def13` | pass | pass |

An end-to-end Wan 2.2 render on the PR head was not rerun; earlier clip evidence was on our branch.

## Prepared comment for #16574

> Confirmed on Apple Silicon (M1 Max, macOS 15.7, Torch 2.14.1), since the description mentions no MPS hardware was available. On a shift-8 flow schedule (20 steps, simple scheduler), every `b` vector and corrector weight from `sample_unipc` in float32 now matches a float64 CPU run, for bh1 and bh2, on both MPS and CPU. On current master the same check fails on both devices; on MPS the corrector weights reach about 21 where float64 gives 1.84.
>
> The check runs through the real `sample_unipc` path with a spy on `torch.linalg.solve`. If useful, I can contribute it as an extra test: [branch link]. It runs on CPU in CI and also on MPS when available.

Before posting, push `refresh/unipc-pr16574-mps-evidence` (one test commit on the PR head) somewhere public, or attach the test inline.

## Branches

- `refresh/unipc-pr16574-mps-evidence` in `vendor/ComfyUI-upstream`: the test on top of #16574's head.
- `upstream/mps-unipc-float64`: historical, not for submission.
