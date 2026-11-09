"""Scoped read-only WAN14 construction and bounded packed sampling on MPS."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import weakref

from . import krea_readonly as readonly

HEADER_SHA256 = '9c0d0f5aec1c7cf2756bacbd7e3ff5b00d07680e8e5cb4eff06af8b8087bed60'
# Qualified experts and their exact safetensors header hashes. The I2V pair is our own INT8 ConvRot
# conversion (bench/wan_int8_convert.py); a name whose hash is None is listed so presets can reference
# it, but the loader refuses it until the converted file's header hash is recorded here.
HEADER_SHA256S = {f'wan2.2_t2v_{role}_noise_14B_int8_convrot.safetensors': HEADER_SHA256 for role in ('high', 'low')}
# I2V experts converted October 13 by bench/wan_int8_convert.py from the Comfy-Org fp16 repack (ee6f4a40); both share one header.
HEADER_SHA256S.update({f'wan2.2_i2v_{role}_noise_14B_int8_convrot.safetensors': '72e5d0a8885ecc3ff82c56ce4ef9ac4cc38c678ae3dbf30eaacf702cacda3da2' for role in ('high', 'low')})
NAMES = tuple(HEADER_SHA256S)
DTYPES = frozenset({'F16', 'F32', 'I8', 'U8'})
RESIDENT_BYTES = 2**30


def validate_patcher(patcher, target):
    """Only plain packed block adapters can use the streamed side path."""
    from comfy.quant_ops import QuantizedTensor
    from comfy.weight_adapter import LoRAAdapter
    import torch
    if (patcher.model is not target or patcher.load_device.type != 'mps'
            or patcher.offload_device.type != 'cpu' or patcher.is_dynamic() or patcher.force_cast_weights
            or patcher.weight_inplace_update or patcher.hook_patches or patcher.current_hooks
            or patcher.forced_hooks or patcher.weight_wrapper_patches
            or set(patcher.object_patches) - {'model_sampling'}):
        raise ValueError('Bounded WAN14 requires a static immutable model without hooks or weight wrappers')
    for key, entries in patcher.patches.items():
        module = target.get_submodule(key[:-7]) if key.endswith('.weight') else None
        weight = getattr(module, 'weight', None)
        if (not key.startswith('diffusion_model.blocks.') or not isinstance(weight, QuantizedTensor)
                or weight._layout_cls != 'TensorWiseINT8Layout' or weight._params.orig_dtype != torch.float16
                or not weight._params.convrot or weight._params.convrot_groupsize != 256):
            raise ValueError('Bounded WAN14 supports plain LoRAs on INT8 block weights only')
        if (module.quant_format != 'int8_tensorwise' or module._full_precision_mm
                or getattr(module, 'comfy_force_cast_weights', False)
                or getattr(module, 'pre_quant_scale', None) is not None or module.bias_function):
            raise ValueError('Bounded WAN14 requires native packed FP16 compute without transforms')
        for strength, adapter, model_strength, offset, function in entries:
            if type(adapter) is not LoRAAdapter or offset is not None or function is not None:
                raise ValueError('Bounded WAN14 supports untransformed plain A/B LoRAs only')
            up, down, alpha, mid, dora, reshape = adapter.weights
            if (mid is not None or dora is not None or reshape is not None
                    or up.ndim != 2 or down.ndim != 2 or up.shape[1] != down.shape[0]
                    or tuple(weight.shape) != (up.shape[0], down.shape[1])):
                raise ValueError('Bounded WAN14 adapter geometry is incompatible')


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
            raise ValueError('Bounded WAN14 sampling requires inference mode')
        with readonly._bounded_krea_residency(
                mm, current, RESIDENT_BYTES, lambda candidate, model: validate_patcher(candidate, target)), \
                streamed_adapter_sides(current):
            return executor(*args, **kwargs)

    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, 'wan14_memory', sample)
    return patcher


@contextmanager
def mapped_expert(utils, path, mm):
    """Exact qualified release header; other files retain ordinary loading."""
    path = Path(path).resolve(strict=True)
    if path.name not in NAMES:
        raise ValueError('Select an installed WAN2.2 14B INT8 ConvRot expert')
    if HEADER_SHA256S[path.name] is None:
        raise ValueError(f'{path.name} is not qualified yet: convert it and record its header hash in wan_memory.py')
    if (mm.get_torch_device().type != 'mps' or not mm.torch.is_inference_mode_enabled()
            or mm.in_training or utils.DISABLE_MMAP or utils.comfy.memory_management.aimdo_enabled):
        raise ValueError('Mapped WAN14 requires ordinary immutable MPS inference')
    identity = readonly._stat(path.stat())
    header = utils.safetensors_header(str(path))
    if header is None or hashlib.sha256(header).hexdigest() != HEADER_SHA256S[path.name]:
        raise ValueError('WAN14 checkpoint header differs from the qualified release')
    original = utils.load_torch_file

    def load(ckpt, safe_load=False, device=None, return_metadata=False):
        if Path(ckpt).resolve() != path:
            return original(ckpt, safe_load=safe_load, device=device, return_metadata=return_metadata)
        if not mm.torch.is_inference_mode_enabled() or mm.in_training:
            raise ValueError('Mapped WAN14 construction requires inference mode')
        if device is not None and mm.torch.device(device).type != 'cpu':
            raise ValueError('Mapped WAN14 weights must be constructed on CPU')
        state, metadata = readonly._load(utils, path, identity, header, allowed_dtypes=DTYPES)
        return (state, metadata) if return_metadata else state

    utils.load_torch_file = load
    try:
        yield
    finally:
        utils.load_torch_file = original


class BenchWan14ModelLoader:
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
        from comfy.model_base import WAN21
        try:
            owner = json.loads(Path(os.environ.get('MK_GPU_LOCK', '/tmp/mac-kernels-gpu.lock')).read_text())
            if owner.get('group') != os.getpgrp():
                raise ValueError('Start Video Workbench to use the WAN14 memory loader')
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError('Start Video Workbench to use the WAN14 memory loader') from error
        path = folder_paths.get_full_path_or_raise('diffusion_models', unet_name)
        with mapped_expert(utils, path, mm):
            patcher = nodes.UNETLoader().load_unet(unet_name, 'default', mps_direct_load=False)[0]
        if type(patcher.model) is not WAN21 or patcher.loaded_size() != 0:
            raise ValueError('Unexpected WAN14 model or initial residency')
        validate_patcher(patcher, patcher.model)
        return (attach_memory_wrapper(patcher),)


NODE_CLASS_MAPPINGS = {'BenchWan14ModelLoader': BenchWan14ModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {'BenchWan14ModelLoader': 'WAN14 Memory Loader'}
