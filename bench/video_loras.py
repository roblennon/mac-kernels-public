"""Optional model-only video adapters, using Comfy's own conversion and patching."""
import logging
import math
import re


def model_lora_keys(model):
    """Extend stock keys with WAN Diffusers aliases, without rewriting tensors.

    Names follow Diffusers' WAN conversion (block attention/FFN, I2V image
    keys, and the top-level embedders and head); only actual WAN model
    weights receive aliases. Existing admission still checks every tensor and
    shape. Diffusers ``lora_B.bias`` is not applied by this runtime's loader,
    so a file carrying it is refused rather than half-applied.
    """
    import comfy.lora
    import comfy.model_base
    mapping = comfy.lora.model_lora_keys_unet(model, {})
    if not isinstance(model, comfy.model_base.WAN21):
        return mapping
    parts = {f'{native}.{proj}': f'{diffusers}.{converted}'
             for native, diffusers in (('self_attn', 'attn1'), ('cross_attn', 'attn2'))
             for proj, converted in (('q', 'to_q'), ('k', 'to_k'), ('v', 'to_v'), ('o', 'to_out.0'))}
    parts.update({'ffn.0': 'ffn.net.0.proj', 'ffn.2': 'ffn.net.2',
                  'cross_attn.k_img': 'attn2.add_k_proj', 'cross_attn.v_img': 'attn2.add_v_proj'})
    top = {'text_embedding.0': 'condition_embedder.text_embedder.linear_1',
           'text_embedding.2': 'condition_embedder.text_embedder.linear_2',
           'time_embedding.0': 'condition_embedder.time_embedder.linear_1',
           'time_embedding.2': 'condition_embedder.time_embedder.linear_2',
           'time_projection.1': 'condition_embedder.time_proj',
           'img_emb.proj.1': 'condition_embedder.image_embedder.ff.net.0.proj',
           'img_emb.proj.3': 'condition_embedder.image_embedder.ff.net.2',
           'head.head': 'proj_out'}
    for key, target in list(mapping.items()):
        if target != key + '.weight':
            continue
        match = re.fullmatch(r'diffusion_model\.blocks\.(\d+)\.(.+)', key)
        if match and match[2] in parts:
            alias = f'blocks.{match[1]}.{parts[match[2]]}'
        elif key.startswith('diffusion_model.') and key[len('diffusion_model.'):] in top:
            alias = top[key[len('diffusion_model.'):]]
        else:
            continue
        mapping[alias] = target
        mapping['transformer.' + alias] = target
    return mapping


def validate_patch_shape(model, target, patch):
    """Admit exact plain A/B or difference geometry before native patching.

    Comfy can reshape incompatible products or log a failure and return an
    unchanged weight. Explicit checks avoid both, without model allocations.
    """
    from comfy.model_patcher import get_key_weight
    from comfy.weight_adapter.lora import LoRAAdapter
    key, offset = (target[0], target[1]) if isinstance(target, tuple) else (target, None)
    try:
        weight = get_key_weight(model, key)[0]
    except AttributeError as error:
        raise ValueError(f'Incompatible video adapter target: {target}') from error
    if weight is None:
        raise ValueError(f'Incompatible video adapter target: {target}')
    shape = list(weight.shape)
    if offset is not None:
        shape[offset[0]] = offset[2]
    if isinstance(patch, tuple) and patch[0] in ('diff', 'set'):
        valid = tuple(patch[1][0].shape) == tuple(shape)
    elif isinstance(patch, LoRAAdapter):
        up, down, _, mid, dora, reshape = patch.weights
        if reshape is not None:
            raise ValueError(f'Video stack does not support reshape adapters: {target}')
        if mid is not None or dora is not None:
            raise ValueError(f'Video stack supports plain A/B adapters only: {target}')
        valid = (up.ndim >= 2 and down.ndim >= 2 and len(shape) >= 2
                 and up.shape[0] == shape[0]
                 and math.prod(up.shape[1:]) == down.shape[0]
                 and math.prod(down.shape[1:]) == math.prod(shape[1:]))
    else:
        raise ValueError(f'Video stack supports plain A/B adapters only: {target}')
    if not valid:
        raise ValueError(f'Incompatible video adapter geometry: {target}')


def _enabled_slots(slots):
    for index in range(1, 6):
        name = slots.get(f'lora_{index}', 'none')
        strength = slots.get(f'strength_{index}', 1.0)
        if type(strength) not in (int, float) or not math.isfinite(strength) or not -2 <= strength <= 2:
            raise ValueError(f'LoRA slot {index}: strength must be finite and between -2 and 2')
        if name == 'none' or strength == 0:
            continue
        if not isinstance(name, str) or not name:
            raise ValueError(f'LoRA slot {index}: select a filename or none')
        yield index, name, strength


class BenchVideoLoRAStack:
    CATEGORY = 'bench/video'
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    DESCRIPTION = 'Five optional model-only LoRAs. Use adapters trained for the selected model/expert; none or strength zero disables a slot.'

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        names = ['none'] + folder_paths.get_filename_list('loras')
        required = {'model': ('MODEL',)}
        for index in range(1, 6):
            required[f'lora_{index}'] = (names,)
            required[f'strength_{index}'] = ('FLOAT', {'default': 1.0, 'min': -2.0, 'max': 2.0, 'step': .05})
        return {'required': required}

    @classmethod
    def IS_CHANGED(cls, **slots):
        import folder_paths
        from pathlib import Path
        # Metadata only: do not read multi-GB adapter payloads on every queue.
        identities = []
        for index, name, strength in _enabled_slots(slots):
            path = Path(folder_paths.get_full_path_or_raise('loras', name)).resolve(strict=True)
            stat = path.stat()
            identities.append((index, str(path), stat.st_dev, stat.st_ino,
                               stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        return tuple(identities)

    def apply(self, model, **slots):
        import folder_paths
        import comfy.lora
        import comfy.lora_convert
        import comfy.utils
        current = model
        for index, name, strength in _enabled_slots(slots):
            path = folder_paths.get_full_path_or_raise('loras', name)
            state = comfy.utils.load_torch_file(path, safe_load=True)
            converted = comfy.lora_convert.convert_lora(state)
            mapping = model_lora_keys(current.model)
            patches = comfy.lora.load_lora(converted, mapping, log_missing=False)
            if not patches:
                raise ValueError(f'LoRA slot {index}: {name} has no compatible model patches')
            # Comfy adapters expose the exact keys they consumed. Stock loaders
            # log unmatched tensors and proceed; an explicit stack must fail instead
            # (especially for unpruned H3 adapters against a pruned checkpoint).
            consumed = set()
            import torch
            sources = {id(value): key for key, value in converted.items() if isinstance(value, torch.Tensor)}
            def tensors(value):
                if isinstance(value, torch.Tensor):
                    yield value
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        yield from tensors(item)
            prefixes = {target: [] for target in patches}
            for prefix, target in mapping.items():
                if target in prefixes:
                    prefixes[target].append(prefix)
                if isinstance(target, str) and target.endswith('.weight'):
                    bias = target[:-len('.weight')] + '.bias'
                    if bias in prefixes:
                        prefixes[bias].append(prefix)
            for target, adapter in patches.items():
                if not hasattr(adapter, 'loaded_keys'):
                    if not isinstance(adapter, tuple) or adapter[0] not in ('diff', 'set'):
                        raise ValueError(f'LoRA slot {index}: unsupported Comfy patch type')
                    # Stock diff/bias/norm patches retain the original tensor.
                    # Count only the exact source actually used by this patch;
                    # overwritten or unmatched A/B matrices still fail below.
                    suffixes = ('.set_weight',) if adapter[0] == 'set' else ('.diff', '.diff_b', '.w_norm', '.b_norm')
                    for prefix in prefixes[target]:
                        consumed.update(prefix + suffix for suffix in suffixes
                                        if converted.get(prefix + suffix) is adapter[1][0])
                    continue
                # loaded_keys is shared by Comfy's parser across all targets.
                # Retained tensor references identify this surviving adapter;
                # global loaded_keys alone would count overwritten patches.
                retained = {sources[id(value)] for value in tensors(adapter.weights) if id(value) in sources}
                consumed.update(retained)
                for prefix in prefixes[target]:
                    if any(key.startswith((prefix + '.', prefix + '_lora.')) for key in retained):
                        consumed.update(key for key in (prefix + '.alpha', prefix + '.reshape_weight')
                                        if key in converted and key in adapter.loaded_keys)
            unmatched = set(converted) - consumed
            if unmatched:
                sample = ', '.join(sorted(unmatched)[:3])
                raise ValueError(f'LoRA slot {index}: {name} has {len(unmatched)} unmatched tensors ({sample}); use a compatible model-only adapter')
            for target, patch in patches.items():
                validate_patch_shape(current.model, target, patch)
            candidate = current.clone()
            accepted = candidate.add_patches(patches, strength)
            if set(accepted) != set(patches):
                raise ValueError(f'LoRA slot {index}: {name} contains unsupported model targets')
            logging.info("Video LoRA slot %s: %s (%s targets, strength %s)", index, name, len(accepted), strength)
            current = candidate
        return (current,)


NODE_CLASS_MAPPINGS = {'BenchVideoLoRAStack': BenchVideoLoRAStack}
NODE_DISPLAY_NAME_MAPPINGS = {'BenchVideoLoRAStack': 'Video LoRA Stack'}
