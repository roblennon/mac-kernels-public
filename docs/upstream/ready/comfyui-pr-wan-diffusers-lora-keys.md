# ComfyUI PR: Diffusers-format WAN LoRAs

Head: `roblennon/ComfyUI` branch `wan-diffusers-lora-keys` (`f647d9ed`, three commits rebased onto master `8010b48d`, October 13, 2026). Base: `Comfy-Org/ComfyUI` `master`.

## Title

Support Diffusers-format WAN LoRAs

## Body

WAN LoRAs saved by Diffusers or PEFT name their layers the Diffusers way: `transformer.blocks.0.attn1.to_q`, `blocks.0.ffn.net.0.proj`, `condition_embedder.time_proj`, `proj_out`. `model_lora_keys_unet` has no WAN entry for these names, so ComfyUI logs every tensor as "lora key not loaded" and the LoRA has no effect.

This adds a `WAN21` branch (it covers WAN 2.1, 2.2 and the WAN-derived model classes) that aliases those names to the native WAN weights, with and without the `transformer.` prefix. The name table follows `_convert_non_diffusers_wan_lora_to_diffusers` in diffusers' `lora_conversion_utils.py`:

- per block: `attn1`/`attn2` `to_q`, `to_k`, `to_v`, `to_out.0`; `attn2.add_k_proj`/`add_v_proj` (the I2V image keys `k_img`/`v_img`); `ffn.net.0.proj`, `ffn.net.2`
- top level: `condition_embedder.text_embedder.linear_1/2`, `time_embedder.linear_1/2`, `time_proj`, `image_embedder.ff.net.0.proj`/`ff.net.2`, `proj_out`

An alias is only added for a weight that exists in the model's state dict. Alpha and shape handling are the existing `load_lora` and `calculate_weight` behavior.

Diffusers' WAN conversion also emits bias updates as `<layer>.lora_B.bias` (for example CausVid and the output head). The WAN branch lists each aliased layer's `lora_B.bias` against the native bias, and `load_lora` applies a listed `lora_B.bias` as a bias diff. It is scaled by `alpha / rank` when an alpha is present, which is how PEFT merges a `lora_bias` (`lora_B.bias * scaling`). `load_lora` only does this for keys the model's key map lists, so other model families keep their current behavior. If you would rather make `lora_B.bias` generic for every family, that is a behavior change for other families, so it is left out here.

Example: [Markus-Pobitzer/wlp-Wan2.2-TI2V-5B-lora](https://huggingface.co/Markus-Pobitzer/wlp-Wan2.2-TI2V-5B-lora) (`base.safetensors`) has 600 tensors over 300 targets in this format. Current master loads none of them; with this change all 300 map onto Wan2.2 TI2V 5B weights with matching shapes.

Not covered: VACE `vace_blocks.*` are not aliased, and LoRA bias on layers outside the alias table (norms, `patch_embedding`) is not loaded, as in Diffusers.

## Tests

New `tests-unit/comfy_test/wan_diffusers_lora_test.py`, CPU only, with tiny WAN 2.1 T2V, WAN 2.1 I2V and Wan 2.2 TI2V models:

- Every Diffusers name, with and without `transformer.`, maps to its exact native weight and its `lora_B.bias` to the native bias. Image-only names are absent on models without image weights.
- Distinct random rank-2 factors and biases for every target, applied through `calculate_weight` at strength 0.7, with and without alpha, checked numerically on square and rectangular targets, weight and bias.
- A non-WAN model with the same layer names gets none of these keys.
- An fp16 `lora_B.bias` with alpha is scaled in fp32, so it no longer overflows.

With master's `comfy/lora.py`, 7 of 8 fail. Swapping the `q`/`k` aliases fails 7 tests; scaling the bias wrongly fails both alpha cases. Full `tests-unit/comfy_test` on the rebased branch: 199 passed, 14 skipped. Ruff clean.

The real-file check above reads only safetensors headers and runs the key mapping on the meta device; no render was run with this branch.
