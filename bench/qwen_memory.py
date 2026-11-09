"""Reuse mapped construction and a 4 GiB sampling cap for installed Qwen INT8."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import weakref
from . import krea_readonly as readonly

NAME = 'qwen_image_2.1_int8_convrot.safetensors'
TURBO_NAME = 'qwen_image_2.1_turbo_int8_convrot.safetensors'
# Admitted checkpoints by safetensors header SHA-256. Turbo is our conversion of Qwen's official
# 8-step release (bench/qwen_int8_convert.py), same keys, dtypes and shapes as the base file.
ADMITTED = {
    NAME: '64ac1c737e46bf4f181d302e4836591306e26e4f62d547b2bef08270c9c97567',
    TURBO_NAME: '8b476a9ddf8f6dd178e8f5a1c05e72688d90e95a1fa7a156cfbe6a48435c0b4e',
}
RESIDENT_BYTES = 4 * 2**30


def validate_patcher(patcher, target):
    import torch
    from comfy.quant_ops import QuantizedTensor
    from comfy.weight_adapter import LoRAAdapter
    import comfy.ops as ops
    if (patcher.model is not target or patcher.load_device.type != 'mps'
            or patcher.offload_device.type != 'cpu' or patcher.is_dynamic() or patcher.force_cast_weights
            or patcher.weight_inplace_update or patcher.hook_patches or patcher.current_hooks
            or patcher.forced_hooks or patcher.weight_wrapper_patches
            or set(patcher.object_patches) - {'model_sampling'}):
        raise ValueError('Bounded Qwen requires static immutable inference without hooks or weight wrappers')
    for key, entries in patcher.patches.items():
        module = target.get_submodule(key[:-7]) if key.endswith('.weight') else None
        weight = getattr(module, 'weight', None)
        if (not key.startswith('diffusion_model.') or not isinstance(module, ops.MixedPrecisionOp)
                or type(module).__name__ != 'Linear' or weight.ndim != 2
                or getattr(module, 'pre_quant_scale', None) is not None or module.bias_function):
            raise ValueError('Bounded Qwen admits plain Linear-weight LoRAs only')
        if isinstance(weight, QuantizedTensor):
            if (not key.startswith('diffusion_model.transformer_blocks.')
                    or weight._layout_cls != 'TensorWiseINT8Layout' or weight._params.orig_dtype not in (torch.float16, torch.bfloat16)
                    or not weight._params.convrot or weight._params.convrot_groupsize != 256
                    or module.quant_format != 'int8_tensorwise' or module._full_precision_mm
                    or getattr(module, 'comfy_force_cast_weights', False)):
                raise ValueError('Bounded Qwen requires packed INT8 ConvRot block adapters')
        elif weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError('Unexpected Qwen global adapter dtype')
        for strength, adapter, model_strength, offset, function in entries:
            # A row slice is how Comfy addresses one half of a weight it fused at load (Qwen 2.1's
            # img_mlp gate_up); the side path applies it to exactly those output rows.
            rows = weight.shape[0] if offset is None else offset[2]
            if (type(adapter) is not LoRAAdapter or function is not None
                    or model_strength != 1.0 or not math.isfinite(strength)
                    or (offset is not None and (len(offset) != 3
                                                or any(type(x) is not int for x in offset)
                                                or offset[0] != 0 or offset[1] < 0 or offset[2] <= 0
                                                or offset[1] + offset[2] > weight.shape[0]))):
                raise ValueError('Bounded Qwen admits untransformed plain A/B LoRAs only')
            up, down, alpha, mid, dora, reshape = adapter.weights
            if (mid is not None or dora is not None or reshape is not None
                    or up.ndim != 2 or down.ndim != 2 or up.shape[1] <= 0
                    or up.shape[1] != down.shape[0] or (up.shape[0], down.shape[1]) != (rows, weight.shape[1])
                    or (alpha is not None and not math.isfinite(float(alpha)))):
                raise ValueError('Incompatible Qwen adapter geometry')


def adapter_factor_bytes(patcher):
    """LoRA factor bytes Comfy moves with loaded modules after its residency fit check.

    Counted at no less than two bytes per value: the side path widens 8-bit factors.
    """
    return sum(t.numel() * max(t.element_size(), 2) for entries in patcher.patches.values()
               for entry in entries for t in entry[1].weights if hasattr(t, 'element_size'))


@contextmanager
def mapped_model(utils, path, mm):
    path = Path(path).resolve(strict=True)
    if (path.name not in ADMITTED or mm.get_torch_device().type != 'mps'
            or not mm.torch.is_inference_mode_enabled() or mm.in_training
            or utils.DISABLE_MMAP or utils.comfy.memory_management.aimdo_enabled):
        raise ValueError('Mapped Qwen requires an admitted INT8 checkpoint and ordinary MPS inference')
    identity = readonly._stat(path.stat())
    header = utils.safetensors_header(str(path))
    if header is None or hashlib.sha256(header).hexdigest() != ADMITTED[path.name]:
        raise ValueError('Qwen checkpoint header differs from the admitted release')
    original = utils.load_torch_file
    def load(ckpt, safe_load=False, device=None, return_metadata=False):
        if Path(ckpt).resolve() != path:
            return original(ckpt, safe_load=safe_load, device=device, return_metadata=return_metadata)
        if device is not None and mm.torch.device(device).type != 'cpu':
            raise ValueError('Mapped Qwen must be constructed on CPU')
        state, metadata = readonly._load(utils, path, identity, header,
            allowed_dtypes=frozenset({'F16', 'BF16', 'F32', 'I8', 'U8'}))
        return (state, metadata) if return_metadata else state
    utils.load_torch_file = load
    try:
        yield
    finally:
        utils.load_torch_file = original


def load_bounded(unet_name):
    import folder_paths
    import nodes
    import comfy.utils as utils
    import comfy.model_management as mm
    from comfy.model_base import QwenImage21
    from comfy.patcher_extension import WrappersMP
    from .krea_adapter import streamed_adapter_sides
    try:
        owner = json.loads(Path(os.environ.get('MK_GPU_LOCK', '/tmp/mac-kernels-gpu.lock')).read_text())
        if owner.get('group') != os.getpgrp():
            raise ValueError('Start Qwen Workbench to use the memory loader')
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError('Start Qwen Workbench to use the memory loader') from error
    path = folder_paths.get_full_path_or_raise('diffusion_models', unet_name)
    with mapped_model(utils, path, mm):
        patcher = nodes.UNETLoader().load_unet(unet_name, 'default', mps_direct_load=False)[0]
    if type(patcher.model) is not QwenImage21 or patcher.loaded_size() != 0:
        raise ValueError('Unexpected Qwen model or initial residency')
    validate_patcher(patcher, patcher.model)
    target_ref = weakref.ref(patcher.model)
    def sample(executor, *args, **kwargs):
        current = executor.class_obj.model_patcher
        target = target_ref()
        validate_patcher(current, target)
        if not mm.torch.is_inference_mode_enabled() or mm.in_training:
            raise ValueError('Bounded Qwen sampling requires inference mode')
        with readonly._bounded_krea_residency(mm, current, RESIDENT_BYTES,
                lambda candidate, model: validate_patcher(candidate, target),
                reserve_bytes=adapter_factor_bytes(current)), \
                streamed_adapter_sides(current, block_marker='.transformer_blocks.'):
            return executor(*args, **kwargs)
    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, 'qwen_memory', sample)
    return patcher
