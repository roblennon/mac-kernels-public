# Draft PR: Map Diffusers-format WAN LoRA keys

Branch `refresh/wan-diffusers-lora-keys` in `vendor/ComfyUI-upstream`, three commits, head `75728d8d`, based on upstream `master` at `08ff3c11` (October 9, 2026). Pushed to the private fork only. Source of the idea is CU-16 in [the contribution review](contribution-review-2026-10-08.md).

---

**Title:** Support Diffusers-format WAN LoRAs

WAN LoRAs saved by Diffusers or PEFT name their layers the Diffusers way: `transformer.blocks.0.attn1.to_q`, `blocks.0.ffn.net.0.proj`, `condition_embedder.time_proj`, `proj_out`. `model_lora_keys_unet` has no WAN entry for these names, so ComfyUI logs every tensor as "lora key not loaded" and the LoRA has no effect.

This adds a `WAN21` branch (it covers WAN 2.1, 2.2 and the WAN-derived model classes) that aliases those names to the native WAN weights, with and without the `transformer.` prefix. The name table follows `_convert_non_diffusers_wan_lora_to_diffusers` in diffusers' `lora_conversion_utils.py`:

- per block: `attn1`/`attn2` `to_q`, `to_k`, `to_v`, `to_out.0`; `attn2.add_k_proj`/`add_v_proj` (the I2V image keys `k_img`/`v_img`); `ffn.net.0.proj`, `ffn.net.2`
- top level: `condition_embedder.text_embedder.linear_1/2`, `time_embedder.linear_1/2`, `time_proj`, `image_embedder.ff.net.0.proj`/`ff.net.2`, `proj_out`

An alias is only added for a weight that exists in the model's state dict. Alpha and shape handling are the existing `load_lora` and `calculate_weight` behavior.

Diffusers' WAN conversion also emits bias updates as `<layer>.lora_B.bias` (for example CausVid and the output head). The WAN branch lists each aliased layer's `lora_B.bias` against the native bias, and `load_lora` applies a listed `lora_B.bias` as a bias diff. It is scaled by `alpha / rank` when an alpha is present, which is how PEFT merges a `lora_bias` (`lora_B.bias * scaling`). `load_lora` only does this for keys the model's key map lists, so other model families keep their current behavior.

Example: [Markus-Pobitzer/wlp-Wan2.2-TI2V-5B-lora](https://huggingface.co/Markus-Pobitzer/wlp-Wan2.2-TI2V-5B-lora) (`base.safetensors`) has 600 tensors over 300 targets in this format. Current master loads none of them; with this change all 300 map onto Wan2.2 TI2V 5B weights with matching shapes.

## Tests

New `tests-unit/comfy_test/wan_diffusers_lora_test.py`, CPU only, with tiny WAN 2.1 T2V, WAN 2.1 I2V and Wan 2.2 TI2V models:

- Every Diffusers name, with and without `transformer.`, maps to its exact native weight and its `lora_B.bias` to the native bias. Image-only names (`add_k_proj`, `add_v_proj`, `image_embedder`) are absent on models without image weights.
- Distinct random rank-2 factors and biases for every target, applied through `calculate_weight` at strength 0.7, with and without alpha. Checked numerically on square targets that share a shape (`q`, `k`, `v`, `k_img`) and rectangular ones (`ffn`, `time_embedding.0`, `img_emb.proj.3`, `head.head`), weight and bias.
- A non-WAN model with the same layer names gets none of these keys.

Results:

- With master's `comfy/lora.py`: 7 of 8 fail. With only the alias commit: 7 of 8 fail (the bias keys are missing). With this branch: 8 pass.
- Mutation checks: swapping the `q`/`k` aliases fails 7 tests; scaling the bias by `alpha` or not scaling it fails both alpha cases.
- An fp16 `lora_B.bias` with alpha is scaled in fp32 (40000 × 2 no longer overflows); its test fails without the fix.
- `pytest tests-unit/comfy_test`: 200 passed, 14 skipped.
- `ruff check` on both files: clean.

---

## Notes (not part of the PR)

**Evidence gathered October 9, 2026, CPU only.**

- Duplicate check: `gh` searches of Comfy-Org/ComfyUI issues and PRs (open, closed and merged) found no WAN Diffusers LoRA work. Nearby open PRs touch other families only: #16863 (Qwen Image 2.1 aliases), #16446 (Qwen PEFT), #14182 (Anima Diffusers LoRAs, same pattern as this one), #16857 (fused `to_qkv` split). `git log` of `comfy/lora.py` and `comfy/lora_convert.py` since April has no WAN entry. Issue #7242 (March 2025) asked about diffusion-pipe WAN LoRAs, a different format.
- Real-file proof: a probe script reads only safetensors headers, detects the model from the real `wan2.2_ti2v_5B_fp16.safetensors` key list on the meta device and runs ComfyUI's own `convert_lora`, `model_lora_keys_unet` and `load_lora`. Stock master: 0 of 300 targets. Branch: 300 of 300, all shape-valid. No weights were loaded and no render was run on the branch; the I2V render in the acceptance review used our bench helper, not this code.
- Differences from our helper (`2eff11c`, `bench/video_loras.py`). The helper misses `attn2.add_k_proj`/`add_v_proj` (WAN 2.1 I2V/FLF image cross-attention) and all top-level `condition_embedder.*`/`proj_out` names. A Diffusers LoRA touching those is refused whole by our bench (fail-closed, so no wrong output, but a false incompatibility). Worth backporting into the bench helper or replacing it with this upstream mapping. Everything the helper does map is correct and family-gated the same way.
- The bench helper's strict "all targets or refuse" admission and shape pre-check are our policy, not upstream's. Upstream keeps its own behavior: unmatched keys are logged and shape mismatches are skipped at patch time. The PR does not change that.
- Bias design choice: the opt-in runs through the key map because some callers (`comfy/hooks.py`) skip `convert_lora`, and a filename-based rename in `convert_lora` could not tell WAN from HunyuanDiT, which also uses `blocks.N.attn1/attn2`. A reviewer may prefer making `lora_B.bias` generic for every family. That is a behavior change for other families, so it is left out here.
- Bias with alpha: PEFT scales the bias by `alpha / r` (checked in `peft/tuners/lora/layer.py` on October 9). Diffusers' own WAN conversion bakes alpha into A/B and drops it, so real Diffusers files load the bias unscaled. The scaled path only matters for a non-standard file that carries both `.alpha` and `lora_B.bias`.
- Not covered: VACE `vace_blocks.*` are not aliased. LoRA bias on layers outside the alias table (norms, `patch_embedding`) is not loaded, as in Diffusers.
- No real file with `lora_B.bias` was tested; only the tiny models exercise it.
- Not verified: a WAN 2.1 I2V/14B Diffusers LoRA from a real file (only the tiny test model exercises `add_k_proj`, `add_v_proj` and the top-level names). The 14B int8 convrot checkpoint could not be built on the meta device with stock master, so it was not used as evidence.
- Commit trailer is `Agent-Provenance: claude-code; model=claude-opus-5-5; family=claude`. An independent review of the first commit on October 9 found nothing blocking; its should-fix items (unloaded `lora_B.bias`, tests that could not catch swapped destinations or scaling, narrow fixtures) are addressed in the second commit. The review of that commit found nothing blocking; its fp16 bias-overflow finding is fixed in the third.
