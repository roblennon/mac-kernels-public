# Comment for Comfy-Org/ComfyUI#16574 (UniPC float64 recurrence)

Our own `upstream/mps-unipc-float64` branch is withdrawn; this comment is the contribution. The test commit is public at `roblennon/ComfyUI` branch `unipc-pr16574-mps-evidence` (`fec872c0`, one commit on the PR head `045def13`).

## Comment

Confirmed on Apple Silicon (M1 Max, macOS 15.7, Torch 2.14.1), since the description mentions no MPS hardware was available. On a shift-8 flow schedule (20 steps, simple scheduler), every `b` vector and corrector weight from `sample_unipc` in float32 now matches a float64 CPU run, for bh1 and bh2, on both MPS and CPU. On current master the same check fails on both devices; on MPS the corrector weights reach about 21 where float64 gives 1.84.

The check runs through the real `sample_unipc` path with a spy on `torch.linalg.solve`. If useful, I can contribute it as an extra test: https://github.com/roblennon/ComfyUI/commit/fec872c0. It runs on CPU in CI and also on MPS when available.
