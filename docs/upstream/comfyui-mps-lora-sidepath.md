# ComfyUI PR: keep LoRAs beside weight-only quantized layers on Apple GPUs

Branch `upstream/mps-lora-sidepath`, one commit (`fedb008`) on top of `upstream/mps-kitchen-formats`. It merges cleanly onto the current `origin/master`.

## Title

Keep LoRAs beside weight-only quantized linears on Apple GPUs instead of merging them

## Description

Loading a LoRA onto an INT8/W4A8/W6A8 model merges it into every patched weight: dequantize, add in fp32, requantize. On Apple GPUs this has two problems:
- **Slow.** The 4/6-bit requantize runs on the CPU. MiniMax H3 W6A8 with its 4-step Turbo LoRA spends about 390 s of a 496 s sample stage merging.
- **Lossy.** Requantizing every layer adds about 2.8% error. On H3's layers that is 30 times the LoRA's own contribution.

On MPS, `ModelPatcher.load` now keeps a plain LoRA on such a layer as low-rank factors, with strength * alpha / rank folded into the up weight. The layer adds the low-rank term to its output, and the integer weight is never touched.

How the term is computed:
- **Fused.** Whole-layer LoRAs with bf16 activations and a summed rank up to 128 run inside comfy-kitchen's quantized kernel (`lora_down` / `lora_up`), stacked along the rank.
- **Beside the layer.** In every other case `up(down(x))` runs next to the layer: higher ranks, fp16 activations, LoRAs on a row or column slice of a fused layer (qkv), or a kernel that declines the call.

The 128-rank and bf16 thresholds come from a fused-vs-separate sweep. Above them, or in fp16, MPS's own GEMMs were as fast or faster.

`linear_input_act`'s direct INT8 kernel call takes the same path.

Fallbacks to the merge:
- **Anything that needs the effective weight merges as before:** DoRA, diff/set patches, `strength_model`, patch functions, LoCon mids, weight wrappers (`f(W + delta)` is not `f(W) + delta`) and weight hooks.
- **Saving writes the merged weight** through the lazy saver path. That path's casters now pass `requires_grad=False`, which integer weights need: saving any INT8 layer through it failed before.
- **The resident factors count** in `module_size`, `model_size` and the loaded memory.
- **Lifecycle.** `unpatch_model`, `partially_unload` and the lowvram branch drop the factors; an offloaded layer gets its LoRA from `LowVramPatch` instead.
- **Opt-out.** `COMFY_MPS_LORA_SIDE_PATH=0` restores the merge.

M1 Max 32 GB:

| | Merged (today) | Side path |
|---|---|---|
| H3 W6A8 + Turbo LoRA, 480x288x22, 4 steps: sample stage | 491-496 s | 127-136 s |
| same, per step | 6.1 s | 6.6 s |
| error on real H3 layers vs an exact fp32 merge | 2.8% | 0.04% (fp16) / 0.3% (bf16 rounding) |
| Krea 2 INT8 + rank-32 LoRA, 512, per step | 3.70 s | 3.77 s |

## Test evidence

- `tests-unit/comfy_test/lora_sidepath_mps_test.py` (33 tests, on MPS with comfy-kitchen's mps backend; skipped otherwise). All against the fp32 merged-LoRA oracle:
  - INT8 ConvRot, W6A8 and W4A8: the base weight's bits are untouched, and each layer makes one fused kernel call.
  - Stacked LoRAs; q/k/v row slices; column slices.
  - Rank over the limit, fp16 activations, a kernel that declines, a dequantized weight.
  - `linear_input_act` with activation and residual.
  - Clone switching; unpatch; partial unload then reload.
  - Save and reload (INT8, W6A8); `f(W) = 2W`; a weight hook.
  - Memory accounting; every merge fallback type; the opt-out.
- `tests-unit/comfy_test`: 227 passed, 14 skipped.
- End to end: H3 and Krea as above. Qwen-Image 2.1 (192 layers) and Krea 2 with a 4-step distill LoRA (224 layers) all stayed on the side path.

## Dependencies

- **Format-dispatch PR (`upstream/mps-kitchen-formats`)**: this branch builds on it.
- **comfy-kitchen mps backend PR**, including its unmerged-LoRA kernel arguments (commit 5 of that series). This is a hard dependency. The side path is enabled only for formats the mps backend lists, so a comfy-kitchen without the backend leaves it off. But a comfy-kitchen that has the backend without commit 5 would raise `TypeError` on the `lora_down` keyword in the fused path: only `BackendError` is caught.
  - If the maintainers take the backend but hold back the LoRA commit, this PR needs a small guard: check the signature, or catch `TypeError` in `_fused_lora_linear`.

## Scope and review risks

**Valid on other platforms in principle.** Merging into 4/6-bit weights is lossy everywhere. The side path could serve CUDA too, once a backend accepts the LoRA arguments there. It is gated to MPS formats (`mps_lora_side_path_formats`) because only the mps kernels take them today.

ComfyUI's `AGENTS.md` asks for hand-written-looking code, no capability probing of child objects through `getattr`, and module-scope imports. The branch has a few spots a maintainer may push on:
- `supports_lora_side_path` / `lora_side_patches` are read with `getattr` in `model_patcher.py` and `model_management.lora_side_size`.
- `ops._fused_lora_linear` imports `comfy_kitchen.tensor.w4a8_int8` inline.
- The opt-out is an environment variable, where ComfyUI usually uses CLI flags.

None of these are bugs. Expect review asks, or make these changes before opening.

## Follow-ups

The original sidepath PR stays focused on plain LoRA. The prepared `upstream/mps-lora-accounting` branch fixes clone accounting first. `upstream/mps-lora-types` follows it with LoKr, LoHa, DoRA, LoCon mids, small diffs and `strength_model`, plus rejection of scale folds that cannot be represented. See [the follow-up descriptions](comfyui-mps-followups.md). Those expanded types have their own exact-merge equivalence tests; the initial PR's fallback list above describes only its own scope.
