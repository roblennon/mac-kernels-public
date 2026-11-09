"""Explicit immutable-base inference for released acceleration and plain style LoRAs.

The static readonly route still refuses patches. This separate admission uses
Comfy's existing side and temporary-copy patch machinery; it never merges into
mapped base storage. Nothing here changes quantized kernels or sampler math.
"""
from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import sys

if __package__:
    from . import krea_readonly as readonly
else:
    import krea_readonly as readonly

ADAPTER_NAME = 'krea2_turbo_4step_rank_64_lora_comfyui.safetensors'
ADAPTER_SHA256 = '7590b2d2772f45d7b1252124c63918b1b807e93fbf9eab20546fa7233b4dc482'
ADAPTER_BYTES = 438140968
BLOCK_LAYERS = ('attn.gate', 'attn.wk', 'attn.wo', 'attn.wq', 'attn.wv',
                'mlp.down', 'mlp.gate', 'mlp.up')
PATCH_KEYS = frozenset(
    [f'diffusion_model.blocks.{i}.{layer}.weight' for i in range(28) for layer in BLOCK_LAYERS]
    + [f'diffusion_model.{layer}.weight' for layer in ('last.linear', 'tmlp.0', 'tmlp.2', 'tproj.1')])


def verify_adapter(path, *, name, strength, steps, cfg, spectral=False):
    if (name != ADAPTER_NAME or strength != 1.0 or steps != 4 or cfg != 1.0 or spectral):
        raise ValueError('Immutable Krea adapter requires the released rank-64 adapter, strength 1, four steps, CFG 1 and no spectral')
    path = Path(path).expanduser().resolve(strict=True)
    before = readonly._stat(path.stat())
    if not path.is_file() or path.name != ADAPTER_NAME or before[2] != ADAPTER_BYTES:
        raise ValueError('Unexpected Krea adapter asset')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        if readonly._stat(os.fstat(stream.fileno())) != before:
            raise RuntimeError('Krea adapter changed before verification')
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(chunk)
        if readonly._stat(os.fstat(stream.fileno())) != before:
            raise RuntimeError('Krea adapter changed during verification')
    if readonly._stat(path.stat()) != before or h.hexdigest() != ADAPTER_SHA256:
        raise RuntimeError('Krea adapter identity/digest differs from the admitted release')
    return {'path': str(path), 'file_identity': list(before), 'sha256': h.hexdigest()}


def verify_plain_adapter(path, *, strength, steps, cfg, spectral=False):
    """Record an ordinary style asset; tensor admission follows base loading."""
    if not math.isfinite(strength) or strength == 0 or steps != 8 or cfg != 1.0 or spectral:
        raise ValueError('Plain immutable Krea LoRA requires finite nonzero strength, eight steps, CFG 1 and no spectral')
    path = Path(path).expanduser().resolve(strict=True)
    before = readonly._stat(path.stat())
    if not path.is_file() or path.suffix != '.safetensors':
        raise ValueError('Plain Krea LoRA requires a safetensors file')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        if readonly._stat(os.fstat(stream.fileno())) != before:
            raise RuntimeError('Krea adapter changed before verification')
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(chunk)
        if readonly._stat(os.fstat(stream.fileno())) != before:
            raise RuntimeError('Krea adapter changed during verification')
    receipt = {'path': str(path), 'file_identity': list(before), 'sha256': h.hexdigest(),
               'policy': 'plain', 'strength': strength}
    _asset_unchanged(receipt)
    return receipt


def load_plain_adapter(patcher, utils, receipt, strength):
    """Reuse Comfy's parser but fail closed on partial or special adapters."""
    import comfy.lora as lora
    import comfy.lora_convert as convert
    from comfy.weight_adapter import LoRAAdapter
    if patcher.patches or receipt.get('policy') != 'plain' or strength != receipt['strength']:
        raise ValueError('Plain adapter requires a fresh base and verified strength')
    _asset_unchanged(receipt)
    tensors = convert.convert_lora(utils.load_torch_file(receipt['path'], safe_load=True))
    try:
        loaded = lora.load_lora(tensors, lora.model_lora_keys_unet(patcher.model, {}), log_missing=False)
    except (KeyError, TypeError, IndexError, RuntimeError) as error:
        raise ValueError('Malformed ordinary LoRA payload') from error
    if not loaded or any(type(value) is not LoRAAdapter for value in loaded.values()):
        raise ValueError('Only ordinary model LoRA factors are supported')
    consumed = set().union(*(value.loaded_keys for value in loaded.values()))
    if set(tensors) != consumed:
        raise ValueError('Unconsumed LoRA tensors: ' + ', '.join(sorted(set(tensors) - consumed)))
    # The parser records alpha even without a pair. Require every payload factor
    # to participate, and every alpha to have its own parsed factor prefix.
    factor_ids = {id(t) for value in loaded.values() for t in value.weights[:2]}
    factor_suffixes = ('.lora_up.weight', '.lora_down.weight', '_lora.up.weight',
                       '_lora.down.weight', '.lora_B.weight', '.lora_A.weight',
                       '.lora.up.weight', '.lora.down.weight', '.lora_B', '.lora_A',
                       '.lora_linear_layer.up.weight', '.lora_linear_layer.down.weight',
                       '.lora_B.default.weight', '.lora_A.default.weight')
    parsed_prefixes = {key[:-len(suffix)] for key, tensor in tensors.items()
                       if id(tensor) in factor_ids for suffix in factor_suffixes
                       if key.endswith(suffix)}
    for key, tensor in tensors.items():
        if key.endswith('.alpha'):
            if key[:-6] not in parsed_prefixes:
                raise ValueError('Orphan LoRA alpha: ' + key)
        elif id(tensor) not in factor_ids:
            raise ValueError('Unsupported or unused LoRA tensor: ' + key)
    result = patcher.clone()
    if set(result.add_patches(loaded, strength)) != set(loaded):
        raise ValueError('LoRA contains unapplied model targets')
    _patches(result, plain=True)
    torch = sys.modules['torch']
    if any(not t.is_floating_point() or not torch.isfinite(t).all().item()
           for value in loaded.values() for t in value.weights[:2]):
        raise ValueError('LoRA factors must be finite floating point tensors')
    _asset_unchanged(receipt)
    return result


MAX_PLAIN_STACK_ADAPTERS = 5
MAX_PLAIN_STACK_BYTES = 2 * 2**30


def _stack_receipts(receipt):
    adapters = receipt.get('adapters')
    if (receipt.get('policy') != 'plain-stack' or not isinstance(adapters, list)
            or not 1 <= len(adapters) <= MAX_PLAIN_STACK_ADAPTERS):
        raise ValueError('Ordinary Krea stack requires one to five adapters')
    hashes = []
    total = 0
    for item in adapters:
        if (item.get('policy') != 'plain' or type(item.get('strength')) not in (int, float)
                or not math.isfinite(item['strength']) or item['strength'] == 0):
            raise ValueError('Stack entries require finite nonzero ordinary strengths')
        hashes.append(item['sha256'])
        total += item['file_identity'][2]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate LoRA content in ordinary Krea stack')
    if total > MAX_PLAIN_STACK_BYTES:
        raise ValueError('Ordinary Krea stack assets exceed the 2 GiB resource limit')
    if receipt.get('selection_order') != hashes or receipt.get('asset_bytes') != total:
        raise ValueError('Ordinary Krea stack receipt differs from its ordered assets')
    return adapters


def verify_plain_stack(selections):
    """Verify an ordered, bounded eight-step stack without loading its payloads."""
    if not isinstance(selections, list) or not 1 <= len(selections) <= MAX_PLAIN_STACK_ADAPTERS:
        raise ValueError('Ordinary Krea stack requires one to five selections')
    adapters = []
    total = 0
    for selection in selections:
        if (not isinstance(selection, dict) or set(selection) != {'path', 'strength'}
                or type(selection['strength']) not in (int, float)
                or not math.isfinite(selection['strength']) or selection['strength'] == 0):
            raise ValueError('Stack selections require a path and finite nonzero strength')
        path = Path(selection['path']).expanduser().resolve(strict=True)
        total += path.stat().st_size
        if total > MAX_PLAIN_STACK_BYTES:
            raise ValueError('Ordinary Krea stack assets exceed the 2 GiB resource limit')
        adapters.append(verify_plain_adapter(path, strength=selection['strength'], steps=8, cfg=1.))
    receipt = {'policy': 'plain-stack', 'adapters': adapters,
               'selection_order': [item['sha256'] for item in adapters],
               'asset_bytes': sum(item['file_identity'][2] for item in adapters)}
    _stack_receipts(receipt)
    _asset_unchanged(receipt)
    return receipt


def load_plain_stack(patcher, utils, receipt):
    """Admit every asset against the untouched base before publishing one clone."""
    if patcher.patches or patcher.loaded_size() or patcher.backup:
        raise ValueError('Ordinary Krea stack requires a fresh unloaded base')
    adapters = _stack_receipts(receipt)
    _asset_unchanged(receipt)
    admitted = [load_plain_adapter(patcher, utils, item, item['strength']) for item in adapters]
    result = patcher.clone()
    for candidate in admitted:
        for key, entries in candidate.patches.items():
            strength, value, _, _, _ = entries[0]
            if set(result.add_patches({key: value}, strength)) != {key}:
                raise ValueError('Stack contains unapplied model targets')
    _patches(result, stack=True)
    _asset_unchanged(receipt)
    return result


def _runtime(mm, utils):
    if (sys.platform != 'darwin' or mm.get_torch_device().type != 'mps'
            or 'int8_tensorwise' not in mm.mps_quant_formats(mm.get_torch_device())
            or 'int8_tensorwise' not in mm.mps_lora_side_path_formats(mm.get_torch_device())):
        raise ValueError('Immutable Krea adapter requires native INT8 MPS and the enabled side path')
    if (mm.in_training or not utils.torch.is_inference_mode_enabled()
            or utils.DISABLE_MMAP or utils.comfy.memory_management.aimdo_enabled):
        raise ValueError('Immutable Krea adapter requires ordinary mmap inference without training/AIMDO')


def _asset_unchanged(receipt):
    if receipt.get('policy') == 'plain-stack':
        for item in _stack_receipts(receipt):
            _asset_unchanged(item)
        return
    if list(readonly._stat(Path(receipt['path']).stat())) != receipt['file_identity']:
        raise RuntimeError('Admitted Krea adapter changed')


@contextmanager
def readonly_adapter_load(utils, checkpoint, *, mm, adapter):
    """Load the immutable base under the separately verified adapter policy."""
    def policy():
        _runtime(mm, utils)
        _asset_unchanged(adapter)
    with readonly._mapped_krea_load(utils, checkpoint, policy) as audit:
        yield audit


def _mapped_storage(weight):
    value = weight._qdata if hasattr(weight, '_qdata') else weight
    storage = value.untyped_storage()
    owner = getattr(storage, readonly.OWNER_ATTRIBUTE, None)
    if weight.device.type != 'cpu' or owner is None or not owner[1].readonly:
        raise ValueError('Adapter base/backup must retain readonly CPU checkpoint storage')
    return storage.data_ptr()


def _patches(patcher, plain=False, stack=False):
    plain = plain or stack
    adapter_type = sys.modules['comfy.weight_adapter'].LoRAAdapter
    quantized_type = sys.modules['comfy.quant_ops'].QuantizedTensor
    if not plain and set(patcher.patches) != PATCH_KEYS:
        raise ValueError('Expected exactly the released 228 Krea adapter patches')
    signature = []
    if not patcher.patches:
        raise ValueError("Empty Krea adapter")
    for key in sorted(patcher.patches):
        if not key.startswith("diffusion_model.") or not key.endswith(".weight"):
            raise ValueError("Only diffusion model Linear weights are supported")
        entries = patcher.patches[key]
        if not stack and len(entries) != 1:
            raise ValueError('Stacked Krea adapters are unsupported')
        if stack and not 1 <= len(entries) <= MAX_PLAIN_STACK_ADAPTERS:
            raise ValueError('Unsupported Krea adapter entry count')
        for entry in entries:
            strength, value, model_strength, offset, function = entry
            if ((not math.isfinite(strength) or strength == 0 or (not plain and strength != 1.0)) or model_strength != 1.0 or offset is not None or function is not None
                    or type(value) is not adapter_type):
                raise ValueError('Only ordinary whole-weight LoRA patches are admitted')
            up, down, alpha, mid, dora, reshape = value.weights
            module = patcher.model.get_submodule(key[:-len('.weight')])
            if (up.ndim != 2 or down.ndim != 2 or up.shape[1] <= 0 or down.shape[0] != up.shape[1]
                    or (not plain and up.shape[1] != 64)
                    or tuple(module.weight.shape) != (up.shape[0], down.shape[1])
                    or (alpha is not None and (not math.isfinite(float(alpha)) or (not plain and float(alpha) != 64.0)))
                    or any(x is not None for x in (mid, dora, reshape))):
                raise ValueError(f'Unsupported Krea adapter shape/transform: {key}')
            weight = module.weight
            if plain and (not (isinstance(module, sys.modules["torch"].nn.Linear)
                               or isinstance(module, sys.modules["comfy.ops"].MixedPrecisionOp)
                               and type(module).__name__ == "Linear") or weight.ndim != 2):
                raise ValueError(f"Expected two-dimensional Linear target: {key}")
            if '.blocks.' in key:
                if (not isinstance(weight, quantized_type) or module.quant_format != 'int8_tensorwise'
                        or not weight._params.convrot or weight._params.convrot_groupsize != 256):
                    raise ValueError(f'Expected INT8 ConvRot-256 base layer: {key}')
            elif isinstance(weight, quantized_type) or weight.dtype != sys.modules['torch'].bfloat16:
                raise ValueError(f'Expected BF16 global base layer: {key}')
            signature.append((key, id(value), id(up), id(down), alpha, strength)
                             + ((id(entry),) if stack else ()))
    return tuple(signature)


@contextmanager
def streamed_adapter_sides(patcher, *, block_marker='.blocks.'):
    """Keep admitted offloaded quantized layers packed for the whole forward.

    LowVramPatch otherwise expands and merges a dense temporary weight. Use
    Comfy's existing side representation while this layer executes, including
    its unfused fallbacks. Both entry points need coverage: linear_input_act
    can bypass Module.forward. Factors live only for the current layer.
    """
    import comfy.lora as lora
    import comfy.ops as ops
    from comfy.model_patcher import LowVramPatch
    from comfy.quant_ops import QuantizedTensor
    modules = {}
    for key in patcher.patches:
        if block_marker not in key:
            continue
        module = patcher.model.get_submodule(key[:-7])
        # Some video models have ordinary BF16 token-refiner blocks too.
        # Their LowVramPatch merge remains Comfy's ordinary dense path.
        if isinstance(module.weight, QuantizedTensor):
            modules[id(module)] = (key, module)
    active = set()
    audit = {'streamed_side_entries': 0}

    @contextmanager
    def layer(module, input):
        identity = id(module)
        if identity not in modules or identity in active or not module.weight_function:
            yield
            return
        key = modules[identity][0]
        functions = module.weight_function
        if (len(functions) != 1 or type(functions[0]) is not LowVramPatch
                or functions[0].key != key
                or len(functions[0].patches[key]) != len(patcher.patches[key])
                or any(actual is not expected for actual, expected in
                       zip(functions[0].patches[key], patcher.patches[key]))
                or module.lora_side_patches or module.bias_function
                or getattr(module, 'pre_quant_scale', None) is not None
                or module.weight.device.type != 'cpu'):
            raise RuntimeError(f'Unexpected streamed adapter state: {key}')
        previous_side = module.lora_side_patches
        side = lora.lora_side_patches(patcher.patches[key], module.out_features,
                                     module.in_features, input.device,
                                     module.weight._params.orig_dtype, weight=module.weight)
        if side is None or not side or side.row_scale is not None or side.col_scale is not None:
            raise RuntimeError(f'Cannot express admitted streamed adapter: {key}')
        active.add(identity)
        module.weight_function = []
        module.lora_side_patches = side
        try:
            yield
            audit['streamed_side_entries'] += 1
        finally:
            module.weight_function = functions
            module.lora_side_patches = previous_side
            active.remove(identity)

    original_input_act = ops.linear_input_act
    forwards = []
    def input_act(module, input, *args, **kwargs):
        with layer(module, input):
            return original_input_act(module, input, *args, **kwargs)
    try:
        for _, module in modules.values():
            previous = module.__dict__.get('forward')
            had_instance_forward = 'forward' in module.__dict__
            original = module.forward
            def forward(input, *args, _module=module, _original=original, **kwargs):
                with layer(_module, input):
                    return _original(input, *args, **kwargs)
            forwards.append((module, had_instance_forward, previous))
            module.forward = forward
        ops.linear_input_act = input_act
        yield audit
    finally:
        ops.linear_input_act = original_input_act
        for module, had_instance_forward, previous in reversed(forwards):
            if had_instance_forward:
                module.forward = previous
            else:
                del module.forward


class AdapterContract:
    """Bind one fresh base/adapter pair; validate identity around cap requests."""
    def __init__(self, mm, utils, patcher, adapter, *, backup_validator=None, model_options_validator=None):
        self.mm, self.utils, self.target, self.adapter = mm, utils, patcher.model, adapter
        self.backup_validator = backup_validator
        self.model_options_validator = model_options_validator
        self.validate(patcher, self.target, initial=True)
        if patcher.loaded_size() or patcher.backup:
            raise ValueError('Adapter admission requires a fresh unloaded base')
        self.plain = adapter.get("policy") == "plain"
        self.stack = adapter.get("policy") == "plain-stack"
        self.signature = _patches(patcher, plain=self.plain, stack=self.stack)
        self.sources = {key: _mapped_storage(self.target.get_submodule(key[:-7]).weight)
                        for key in patcher.patches}
        # Comfy's existing mmap offload restores the original quantized mapping.
        patcher.mmap_offload = True

    def validate(self, patcher, target, initial=False):
        _runtime(self.mm, self.utils)
        _asset_unchanged(self.adapter)
        krea_type = getattr(sys.modules.get('comfy.model_base'), 'Krea2', None)
        if krea_type is None or type(target) is not krea_type or patcher.model is not self.target:
            raise ValueError('Adapter residency requires the admitted Krea2 identity')
        if (patcher.is_dynamic() or patcher.load_device.type != 'mps' or patcher.offload_device.type != 'cpu'
                or patcher.weight_inplace_update or patcher.hook_patches or patcher.current_hooks
                or patcher.forced_hooks or patcher.object_patches or patcher.weight_wrapper_patches
                or patcher.additional_models or patcher.injections or patcher.is_injected
                or any(v for group in patcher.wrappers.values() for v in group.values())
                or any(v for group in patcher.callbacks.values() for v in group.values())
                or (self.model_options_validator is None
                    and patcher.model_options != {"transformer_options": {}})):
            raise ValueError('Adapter residency requires static inference without hooks/wrappers/in-place updates')
        if self.model_options_validator is not None:
            self.model_options_validator(patcher)
        if not initial:
            if not patcher.mmap_offload or _patches(patcher, plain=self.plain, stack=self.stack) != self.signature:
                raise RuntimeError('Admitted Krea adapter policy or patches changed')
            if self.backup_validator is not None:
                self.backup_validator(patcher)
            for key, backup in patcher.backup.items():
                if self.backup_validator is not None and key not in self.sources:
                    continue
                pointer = _mapped_storage(backup.weight)
                if backup.inplace_update or pointer != self.sources.get(key, pointer):
                    raise RuntimeError('Adapter backup lost immutable base storage')

    @contextmanager
    def residency(self, patcher, budget_bytes):
        import comfy.samplers as samplers
        original_predict = samplers.CFGGuider.predict_noise
        predictions = 0
        def predict(guider, *args, **kwargs):
            nonlocal predictions
            candidate = guider.model_patcher
            if candidate.model is not self.target:
                return original_predict(guider, *args, **kwargs)
            self.validate(candidate, self.target)
            readonly._resident_size(candidate, budget_bytes)
            result = original_predict(guider, *args, **kwargs)
            self.validate(candidate, self.target)
            readonly._resident_size(candidate, budget_bytes)
            predictions += 1
            return result
        try:
            with readonly._bounded_krea_residency(self.mm, patcher, budget_bytes, self.validate) as audit:
                with streamed_adapter_sides(patcher) as streamed:
                    samplers.CFGGuider.predict_noise = predict
                    yield audit
                audit.update(streamed)
                self.validate(patcher, self.target)
                audit['completed_predictions'] = predictions
        finally:
            samplers.CFGGuider.predict_noise = original_predict
