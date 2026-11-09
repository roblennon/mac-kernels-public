"""Scoped mapped H3 construction and bounded packed sampling on MPS.

No public model admission or sampler hook is installed by importing this module.
"""
from contextlib import contextmanager
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import weakref

from .krea_readonly import OWNER_ATTRIBUTE
from . import krea_readonly as readonly

NAMES = ('minimax_h3_fl2va_pruned_w4a8.safetensors',)
HEADER_SHA256 = '59e9c3b4ab90099dfcb6f26c68e93df52c599cea6f8733bc2732e2e7f0ce08ba'
DTYPES = frozenset({'F16', 'BF16', 'F32', 'I8', 'U8', 'F8_E4M3'})
RESIDENT_BYTES = 2**30
REFINER_KEYS = frozenset(
    f'diffusion_model.token_refiner.blocks.{block}.{layer}.weight'
    for block in (0, 1)
    for layer in ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2'))


@contextmanager
def readonly_grouped_detach():
    """Avoid cloning immutable mapped W4A8 metadata during Parameter wrapping.

    Scoped to inference construction/offload and serialized by the GPU owner.
    Explicit clone and writable/training tensors retain ordinary vendor behavior.
    """
    import torch
    from comfy_kitchen.tensor import base
    op = torch.ops.aten.detach.default
    original = base._DISPATCH_TABLE[op]

    def mapped(value):
        owner = getattr(value.untyped_storage(), OWNER_ATTRIBUTE, None)
        return (value.device.type == 'cpu' and value.is_inference()
                and owner is not None and owner[1].readonly)

    def detach(weight, args, kwargs):
        values = [weight._qdata] + [getattr(weight._params, field)
                                  for field in weight._params._tensor_fields()]
        if (weight._layout_cls != 'AsymW4A8Int8Layout'
                or not torch.is_inference_mode_enabled() or not all(map(mapped, values))):
            return original(weight, args, kwargs)
        params = dataclasses.replace(weight._params, **{
            field: getattr(weight._params, field).detach()
            for field in weight._params._tensor_fields()})
        return weight._copy_with(qdata=weight._qdata.detach(), params=params, clone_params=False)

    base._DISPATCH_TABLE[op] = detach
    try:
        yield
    finally:
        base._DISPATCH_TABLE[op] = original


def _packed_module(module):
    import torch
    from comfy.quant_ops import QuantizedTensor
    weight = getattr(module, 'weight', None)
    if (not isinstance(weight, QuantizedTensor) or weight._layout_cls != 'AsymW4A8Int8Layout'
            or weight._params.orig_dtype != torch.bfloat16
            or weight._params.group_size != 16 or weight._params.convrot_groupsize != 256
            or module.quant_format != 'asym_w4a8_int8' or module._full_precision_mm
            or getattr(module, 'comfy_force_cast_weights', False)
            or getattr(module, 'pre_quant_scale', None) is not None or module.bias_function):
        raise ValueError('Bounded H3 requires native packed BF16 W4A8 without transforms')


def validate_patcher(patcher, target):
    """Admit plain block factors and the release's ordinary BF16 refiner targets."""
    import torch
    from comfy.quant_ops import QuantizedTensor
    from comfy.weight_adapter import LoRAAdapter
    import comfy.ops as ops
    if (patcher.model is not target or patcher.load_device.type != 'mps'
            or patcher.offload_device.type != 'cpu' or patcher.is_dynamic() or patcher.force_cast_weights
            or patcher.weight_inplace_update or patcher.hook_patches or patcher.current_hooks
            or patcher.forced_hooks or patcher.weight_wrapper_patches
            or set(patcher.object_patches) - {'model_sampling'}):
        raise ValueError('Bounded H3 requires a static immutable model without hooks or weight wrappers')
    for module in target.modules():
        if isinstance(getattr(module, 'weight', None), QuantizedTensor):
            _packed_module(module)
    for key, entries in patcher.patches.items():
        try:
            module = target.get_submodule(key[:-7]) if key.endswith('.weight') else None
        except AttributeError as error:
            raise ValueError('Unknown H3 adapter target') from error
        weight = getattr(module, 'weight', None)
        if key.startswith('diffusion_model.blocks.'):
            _packed_module(module)
        elif (key not in REFINER_KEYS or isinstance(weight, QuantizedTensor)
                or not isinstance(module, ops.MixedPrecisionOp) or type(module).__name__ != 'Linear'
                or weight.dtype != torch.bfloat16 or weight.ndim != 2
                or getattr(module, 'pre_quant_scale', None) is not None or module.bias_function):
            raise ValueError('Bounded H3 adapter target is not a packed block or BF16 token refiner')
        for strength, adapter, model_strength, offset, function in entries:
            if (type(adapter) is not LoRAAdapter or offset is not None or function is not None
                    or model_strength != 1.0 or not math.isfinite(strength)):
                raise ValueError('Bounded H3 supports untransformed plain A/B LoRAs only')
            up, down, alpha, mid, dora, reshape = adapter.weights
            if (mid is not None or dora is not None or reshape is not None
                    or up.ndim != 2 or down.ndim != 2 or up.shape[1] <= 0
                    or up.shape[1] != down.shape[0] or tuple(weight.shape) != (up.shape[0], down.shape[1])
                    or (alpha is not None and not math.isfinite(float(alpha)))):
                raise ValueError('Bounded H3 adapter geometry is incompatible')


def attach_memory_wrapper(patcher):
    from comfy.patcher_extension import WrappersMP
    target_ref = weakref.ref(patcher.model)

    def sample(executor, *args, **kwargs):
        import comfy.model_management as mm
        from .krea_adapter import streamed_adapter_sides
        current = executor.class_obj.model_patcher
        target = target_ref()
        validate_patcher(current, target)
        if not mm.torch.is_inference_mode_enabled() or mm.in_training:
            raise ValueError('Bounded H3 sampling requires inference mode')
        with readonly_grouped_detach(), readonly._bounded_krea_residency(
                mm, current, RESIDENT_BYTES, lambda candidate, model: validate_patcher(candidate, target)), \
                streamed_adapter_sides(current):
            return executor(*args, **kwargs)

    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, 'h3_memory', sample)
    return patcher


@contextmanager
def mapped_model(utils, path, mm):
    """Map only the installed pruned W4A8 release, retaining stat/header checks."""
    path = Path(path).resolve(strict=True)
    if path.name not in NAMES:
        raise ValueError('Select the installed pruned H3 W4A8 checkpoint')
    if (mm.get_torch_device().type != 'mps' or not mm.torch.is_inference_mode_enabled()
            or mm.in_training or utils.DISABLE_MMAP or utils.comfy.memory_management.aimdo_enabled):
        raise ValueError('Mapped H3 requires ordinary immutable MPS inference')
    identity = readonly._stat(path.stat())
    header = utils.safetensors_header(str(path))
    if header is None or hashlib.sha256(header).hexdigest() != HEADER_SHA256:
        raise ValueError('H3 checkpoint header differs from the qualified release')
    original = utils.load_torch_file

    def load(ckpt, safe_load=False, device=None, return_metadata=False):
        if Path(ckpt).resolve() != path:
            return original(ckpt, safe_load=safe_load, device=device, return_metadata=return_metadata)
        if not mm.torch.is_inference_mode_enabled() or mm.in_training:
            raise ValueError('Mapped H3 construction requires inference mode')
        if device is not None and mm.torch.device(device).type != 'cpu':
            raise ValueError('Mapped H3 weights must be constructed on CPU')
        state, metadata = readonly._load(utils, path, identity, header, allowed_dtypes=DTYPES)
        return (state, metadata) if return_metadata else state

    utils.load_torch_file = load
    try:
        with readonly_grouped_detach():
            yield
    finally:
        utils.load_torch_file = original


class BenchH3ModelLoader:
    CATEGORY = 'bench/video'
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'load'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'unet_name': (NAMES,)}}

    @classmethod
    def IS_CHANGED(cls, unet_name):
        import folder_paths
        path = Path(folder_paths.get_full_path_or_raise('diffusion_models', unet_name)).resolve(strict=True)
        return (str(path), readonly._stat(path.stat()))

    def load(self, unet_name):
        import folder_paths
        import nodes
        import comfy.utils as utils
        import comfy.model_management as mm
        from comfy.model_base import MiniMaxH3
        from comfy.quant_ops import QuantizedTensor
        try:
            owner = json.loads(Path(os.environ.get('MK_GPU_LOCK', '/tmp/mac-kernels-gpu.lock')).read_text())
            if owner.get('group') != os.getpgrp():
                raise ValueError('Start Video Workbench to use the H3 memory loader')
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError('Start Video Workbench to use the H3 memory loader') from error
        path = folder_paths.get_full_path_or_raise('diffusion_models', unet_name)
        with mapped_model(utils, path, mm):
            patcher = nodes.UNETLoader().load_unet(unet_name, 'default', mps_direct_load=False)[0]
        if type(patcher.model) is not MiniMaxH3 or patcher.loaded_size() != 0:
            raise ValueError('Unexpected H3 model or initial residency')
        validate_patcher(patcher, patcher.model)
        weights = [m.weight for m in patcher.model.modules()
                   if isinstance(getattr(m, 'weight', None), QuantizedTensor)]
        if len(weights) != 200:
            raise ValueError('Unexpected H3 packed layer count')
        for weight in weights:
            values = [weight._qdata] + [getattr(weight._params, field) for field in weight._params._tensor_fields()]
            for value in values:
                owner = getattr(value.untyped_storage(), OWNER_ATTRIBUTE, None)
                if value.device.type != 'cpu' or owner is None or not owner[1].readonly:
                    raise ValueError('H3 construction lost immutable packed storage or metadata')
        return (attach_memory_wrapper(patcher),)


NODE_CLASS_MAPPINGS = {'BenchH3ModelLoader': BenchH3ModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {'BenchH3ModelLoader': 'H3 Memory Loader'}
