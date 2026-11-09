"""Asset and real adapter membership contracts for the explicit successor."""
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from bench import krea_adapter as adapter

ROOT = Path(__file__).resolve().parents[1]


def test_adapter_asset_digest_and_recipe_are_both_required(tmp_path, monkeypatch):
    path = tmp_path / adapter.ADAPTER_NAME
    path.write_bytes(b'fixture adapter bytes')
    monkeypatch.setattr(adapter, 'ADAPTER_BYTES', path.stat().st_size)
    monkeypatch.setattr(adapter, 'ADAPTER_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    recipe = dict(name=adapter.ADAPTER_NAME, strength=1.0, steps=4, cfg=1.0)
    receipt = adapter.verify_adapter(path, **recipe)
    assert receipt['file_identity'][2] == path.stat().st_size
    path.write_bytes(b'changed adapter bytes')
    with pytest.raises(RuntimeError, match='digest'):
        adapter.verify_adapter(path, **recipe)


@pytest.mark.parametrize('override', [dict(strength=.5), dict(steps=8), dict(cfg=2),
                                      dict(spectral=True), dict(name='other.safetensors')])
def test_unsupported_recipe_is_refused_before_asset_io(tmp_path, override):
    recipe = dict(name=adapter.ADAPTER_NAME, strength=1.0, steps=4, cfg=1.0)
    recipe.update(override)
    with pytest.raises(ValueError, match='requires'):
        adapter.verify_adapter(tmp_path / 'absent', **recipe)


def test_real_patcher_membership_and_factor_transforms():
    script = r'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import comfy.weight_adapter
import comfy.model_patcher
import comfy.ops
from comfy.quant_ops import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from krea_adapter import _patches, PATCH_KEYS
model = torch.nn.Module()
for key in sorted(PATCH_KEYS):
    parent = model
    pieces = key[:-len('.weight')].split('.')
    for piece in pieces[:-1]:
        if piece not in parent._modules:
            parent.add_module(piece, torch.nn.Module())
        parent = parent._modules[piece]
    module = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(256,64,bias=False,device='cpu')
    if '.blocks.' in key:
        params = TensorWiseINT8Layout.Params(scale=torch.ones(64,1), orig_dtype=torch.bfloat16,
            orig_shape=(64,256), convrot=True, convrot_groupsize=256)
        module.weight = torch.nn.Parameter(QuantizedTensor(torch.ones(64,256,dtype=torch.int8),
            'TensorWiseINT8Layout', params), requires_grad=False)
        module.quant_format = 'int8_tensorwise'
    else:
        module.weight = torch.nn.Parameter(torch.ones(64,256,dtype=torch.bfloat16), requires_grad=False)
    parent.add_module(pieces[-1], module)
patcher = comfy.model_patcher.ModelPatcher(model, torch.device('cpu'), torch.device('cpu'))
values = {key: comfy.weight_adapter.LoRAAdapter(set(),
    (torch.ones(64,64), torch.ones(64,256), None, None, None, None)) for key in PATCH_KEYS}
assert set(patcher.add_patches(values)) == PATCH_KEYS
signature = _patches(patcher)
assert len(signature) == 228
# A normal Comfy clone shares adapter factors; list container identity is irrelevant.
assert _patches(patcher.clone()) == signature
key = sorted(PATCH_KEYS)[0]
entry = patcher.patches[key][0]
def refuses(change, text):
    patcher.patches[key] = [change]
    try:
        _patches(patcher)
    except ValueError as exc:
        assert text in str(exc), exc
    else:
        raise AssertionError('Unsupported patch accepted')
    finally:
        patcher.patches[key] = [entry]
refuses((.5, entry[1], 1., None, None), 'whole-weight')
refuses((1., entry[1], 1., (0,0,64), None), 'whole-weight')
refuses((1., entry[1], 1., None, lambda value: value), 'whole-weight')
wrong_rank = comfy.weight_adapter.LoRAAdapter(set(),
    (torch.ones(64,32), torch.ones(32,256), None, None, None, None))
refuses((1., wrong_rank, 1., None, None), 'shape')
wrong_shape = comfy.weight_adapter.LoRAAdapter(set(),
    (torch.ones(32,64), torch.ones(64,256), None, None, None, None))
refuses((1., wrong_shape, 1., None, None), 'shape')
dora = comfy.weight_adapter.LoRAAdapter(set(),
    (torch.ones(64,64), torch.ones(64,256), None, None, torch.ones(64), None))
refuses((1., dora, 1., None, None), 'shape')
removed = patcher.patches.pop(key)
try:
    _patches(patcher)
except ValueError as exc:
    assert '228' in str(exc)
else:
    raise AssertionError('Missing adapter key accepted')
patcher.patches[key] = removed
module = model.get_submodule(key[:-7])
original = module.weight
module.weight = torch.nn.Parameter(torch.ones(64,256,dtype=torch.bfloat16), requires_grad=False)
try:
    _patches(patcher)
except ValueError as exc:
    assert 'ConvRot' in str(exc)
else:
    raise AssertionError('Dense replacement for quantized base accepted')
module.weight = original
patcher.patches[key].append(entry)
try:
    _patches(patcher)
except ValueError as exc:
    assert 'Stacked' in str(exc)
else:
    raise AssertionError('Stacked adapter accepted')
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('block_marker', ['.blocks.', '.transformer_blocks.'])
def test_streamed_side_entries_preserve_real_adapter_math_and_restore_state(block_marker):
    script = r'''
import sys
import tempfile
import mmap
import warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import torch._dynamo
# Keep this fixture CPU-only, including optional kitchen backend registration.
torch.backends.mps.is_available = lambda: False
import comfy.lora as lora
import comfy.ops as ops
import comfy.model_patcher as mp
from comfy.quant_ops import QuantizedTensor
from comfy.weight_adapter import LoRAAdapter
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
import krea_adapter as adapter

CPU = torch.device('cpu')
key = 'diffusion_model.blocks.0.attn.wq.weight'
adapter.PATCH_KEYS = frozenset({key})
N, K, R = 64, 256, 64
operations = ops.mixed_precision_ops(compute_dtype=torch.float32)

class CPULinear(operations.Linear):
    calls = 0
    fail = False
    def forward(self, x):
        self.calls += 1
        assert self.weight_function == []
        assert len(self.lora_side_patches) == 1
        if self.fail:
            raise ArithmeticError('fixture forward failure')
        base = torch.nn.functional.linear(x, self.weight.dequantize().float())
        return ops._apply_lora_side_patches(self, base, x)

with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
    path = Path(directory) / 'packed.bin'
    path.write_bytes(torch.arange(N*K, dtype=torch.int32).remainder(127).to(torch.int8).numpy().tobytes())
    stream = path.open('rb')
    mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    original_bytes = bytes(mapping)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The given buffer is not writable')
        packed = torch.frombuffer(mapping, dtype=torch.int8).reshape(N, K)
    params = TensorWiseINT8Layout.Params(scale=torch.full((N,1), .001),
        orig_dtype=torch.float32, orig_shape=(N,K), convrot=True, convrot_groupsize=256)
    layer = CPULinear(K,N,bias=False,device=CPU)
    layer.weight = torch.nn.Parameter(QuantizedTensor(packed, 'TensorWiseINT8Layout', params), requires_grad=False)
    layer.quant_format = 'int8_tensorwise'
    model = torch.nn.Module()
    module = model
    pieces = key[:-7].split('.')
    for piece in pieces[:-1]:
        module.add_module(piece, torch.nn.Module())
        module = module._modules[piece]
    module.add_module(pieces[-1], layer)
    patcher = mp.ModelPatcher(model, CPU, CPU)
    gen = torch.Generator().manual_seed(17)
    value = LoRAAdapter(set(), (torch.randn(N,R,generator=gen)*.01,
        torch.randn(R,K,generator=gen)*.01, None, None, None, None))
    patcher.add_patches({key:value})
    lazy = mp.LowVramPatch(key, patcher.patches)
    functions = [lazy]
    layer.weight_function = functions
    layer.bias_function = []
    layer.lora_side_patches = ()
    source_pointer = packed.data_ptr()
    x = torch.randn(2,K,generator=gen)
    wide_x = torch.randn(2,K*2,generator=gen)
    activated = torch.nn.functional.silu(wide_x[:,:K]) * wide_x[:,K:]
    merged = lora.calculate_weight(patcher.patches[key], layer.weight.dequantize().float().clone(), key)
    reference = torch.nn.functional.linear(x, merged)
    original_act = ops.linear_input_act
    original_forward = layer.forward.__func__
    original_builder = lora.lora_side_patches
    built = [0]
    def counted_builder(*args, **kwargs):
        built[0] += 1
        return original_builder(*args, **kwargs)
    lora.lora_side_patches = counted_builder

    def pristine():
        assert layer.weight_function is functions
        assert layer.lora_side_patches == ()
        assert layer.weight._qdata.data_ptr() == source_pointer
        assert bytes(mapping) == original_bytes and path.read_bytes() == original_bytes
        assert not patcher.backup

    try:
        with adapter.streamed_adapter_sides(patcher) as audit:
            got = layer(x)
            torch.testing.assert_close(got, reference, rtol=2e-5, atol=2e-5)
            pristine()
            assert built[0] == 1
            # Original direct entry executes its own cast/activation path, bypassing Module.forward.
            layer._full_precision_mm = False
            calls = layer.calls
            residual = torch.randn(2,N,generator=gen)
            scale = torch.full((N,), .3)
            got = ops.linear_input_act(layer, wide_x, 'swiglu', residual=residual, residual_scale=scale)
            # CPU eager INT8 quantizes activations; compare to the real resident-side path,
            # rather than to a dense GEMM with a different activation algorithm.
            layer.weight_function = []
            layer.lora_side_patches = original_builder(patcher.patches[key], N,K,CPU,torch.float32)
            try:
                resident = original_act(layer, wide_x, 'swiglu', residual=residual, residual_scale=scale)
            finally:
                layer.weight_function = functions
                layer.lora_side_patches = ()
            torch.testing.assert_close(got, resident, rtol=0, atol=0)
            assert layer.calls == calls
            assert built[0] == 2
            pristine()
            # This fallback enters the wrapped module.forward inside wrapped linear_input_act.
            layer._full_precision_mm = True
            got = ops.linear_input_act(layer, wide_x, 'swiglu')
            expected = torch.nn.functional.linear(activated, merged)
            torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-5)
            assert layer.calls == calls + 1 and built[0] == 3
            pristine()
            assert audit['streamed_side_entries'] == 3
            layer.fail = True
            try:
                ops.linear_input_act(layer, wide_x, 'swiglu')
            except ArithmeticError:
                pass
            else:
                raise AssertionError('Forward exception was swallowed')
            pristine()
            # An exception must clear the active marker; the next call builds and applies once.
            layer.fail = False
            got = layer(x)
            torch.testing.assert_close(got, reference, rtol=2e-5, atol=2e-5)
            assert built[0] == 5
            pristine()
        assert ops.linear_input_act is original_act
        assert 'forward' not in layer.__dict__ and layer.forward.__func__ is original_forward
        pristine()

        # Propagation through the whole context also restores global and inherited methods.
        layer.fail = True
        try:
            with adapter.streamed_adapter_sides(patcher):
                layer(x)
        except ArithmeticError:
            pass
        else:
            raise AssertionError('Context swallowed the forward exception')
        layer.fail = False
        assert ops.linear_input_act is original_act and 'forward' not in layer.__dict__
        pristine()
        instance_forward = lambda input: CPULinear.forward(layer, input)
        layer.forward = instance_forward
        with adapter.streamed_adapter_sides(patcher):
            torch.testing.assert_close(layer(x), reference, rtol=2e-5, atol=2e-5)
        assert layer.forward is instance_forward
        del layer.forward
        pristine()

        states = [
            ('extra function', lambda: setattr(layer, 'weight_function', [lazy, lambda w:w])),
            ('wrong key', lambda: setattr(layer, 'weight_function', [mp.LowVramPatch('other', patcher.patches)])),
            ('bias function', lambda: setattr(layer, 'bias_function', [lambda b:b])),
            ('prequant scale', lambda: setattr(layer, 'pre_quant_scale', torch.ones(K))),
            ('existing side', lambda: setattr(layer, 'lora_side_patches', original_builder(patcher.patches[key], N,K,CPU,torch.float32))),
        ]
        for label, setup in states:
            setup()
            before_functions, before_side = layer.weight_function, layer.lora_side_patches
            try:
                with adapter.streamed_adapter_sides(patcher):
                    layer(x)
            except RuntimeError as exc:
                assert 'Unexpected streamed adapter state' in str(exc), (label,exc)
            else:
                raise AssertionError('Unsupported state accepted: '+label)
            assert layer.weight_function is before_functions and layer.lora_side_patches is before_side
            assert ops.linear_input_act is original_act and 'forward' not in layer.__dict__
            layer.weight_function = functions
            layer.bias_function = []
            layer.lora_side_patches = ()
            if 'pre_quant_scale' in layer.__dict__:
                del layer.pre_quant_scale
            pristine()

        # Side construction genuinely refuses a patch function; the mapped base is still untouched.
        entry = patcher.patches[key][0]
        patcher.patches[key][0] = (*entry[:4], lambda w:w)
        try:
            with adapter.streamed_adapter_sides(patcher):
                layer(x)
        except RuntimeError as exc:
            assert 'Cannot express admitted streamed adapter' in str(exc)
        else:
            raise AssertionError('Inexpressible side patch accepted')
        patcher.patches[key][0] = entry
        pristine()
        assert ops.linear_input_act is original_act and 'forward' not in layer.__dict__
    finally:
        lora.lora_side_patches = original_builder
'''
    if block_marker != '.blocks.':
        script = script.replace('diffusion_model.blocks.', 'diffusion_model.transformer_blocks.')
        script = script.replace('streamed_adapter_sides(patcher)',
                                f'streamed_adapter_sides(patcher, block_marker={block_marker!r})')
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
