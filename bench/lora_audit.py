"""Where a model's LoRA patches went: kept beside quantized layers (side path), merged into a quantized layer (fallback),
or merged into a non-quantized weight (by design). Call after sampling, while the model is still loaded."""


def audit(patcher):
    from comfy.quant_ops import QuantizedTensor
    side, merged_quant, merged_plain, other_keys, examples = 0, 0, 0, 0, []
    for key in patcher.patches:
        if not key.endswith(".weight"):
            other_keys += 1
            continue
        try:
            mod = patcher.model.get_submodule(key[:-len(".weight")])
        except AttributeError:
            other_keys += 1
            continue
        if getattr(mod, "lora_side_patches", None):
            side += 1
        elif isinstance(getattr(mod, "weight", None), QuantizedTensor):
            merged_quant += 1
            if len(examples) < 5:
                examples.append(key)
        else:
            merged_plain += 1
    return dict(patched_keys=len(patcher.patches), side_path=side, merged_into_quantized=merged_quant,
                merged_non_quantized=merged_plain, non_weight_keys=other_keys, fallback_examples=examples)
