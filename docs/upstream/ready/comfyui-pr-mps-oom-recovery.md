# ComfyUI PR: MPS OOM recognition

Head: `roblennon/ComfyUI` branch `mps-oom-recovery` (`861b6169`, one commit rebased onto master `8010b48d`, October 13, 2026). Base: `Comfy-Org/ComfyUI` `master`.

## Title

Recognize MPS allocator out-of-memory errors in `is_oom`

## Body

On Torch versions that have `torch.AcceleratorError`, `is_oom` checks the "out of memory" text only on that type. The MPS allocator still raises a plain `RuntimeError` ("MPS backend out of memory ...", confirmed on Torch 2.14.1). So on Apple GPUs every OOM fallback treats it as an ordinary error: tiled VAE decode and encode, attention and upscale tiling, SeedVR, mesh post-processing, and the executor's unload-all-models response. On older Torch, where `ACCELERATOR_ERROR` falls back to `RuntimeError`, the substring check already matched, so this restores the earlier behavior.

`is_oom` now also accepts a `RuntimeError` whose message starts with the MPS allocator prefix. Typed CUDA and PyTorch OOM handling is unchanged. A message that merely contains the prefix, or another exception type with the same text, still propagates.

## Tests

Ten CPU tests drive the real VAE decode path with a failing backend boundary. An MPS OOM on the whole 320-wide latent recovers through four smaller 3D tiles with exact output; a failure during the tiled retry propagates without a loop; unrelated errors propagate untouched. Three of them fail on unpatched master.

The test builds a `VAE` without a checkpoint, so it is coupled to `VAE` attributes. If you would rather keep only the `is_oom` cases, I can trim it.

Full `tests-unit/comfy_test` on the rebased branch: 202 passed, 14 skipped. Ruff clean. Real hardware OOM recovery inside a full workflow was not rerun for this PR.
