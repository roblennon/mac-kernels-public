"""Bounded stack admission and real Comfy additive arithmetic on CPU."""
from pathlib import Path
import subprocess
import sys

import pytest

from bench import krea_adapter as adapter

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('selections', [[], [{}] * 6, [{'path': 'missing', 'strength': True}],
                                      [{'path': 'missing', 'strength': 0}],
                                      [{'path': 'missing', 'strength': float('nan')}],
                                      [{'path': 'missing', 'strength': float('inf')}],
                                      [{'path': 'missing', 'strength': 1, 'extra': 1}]])
def test_stack_invalid_selection_refused_before_io(selections):
    with pytest.raises(ValueError):
        adapter.verify_plain_stack(selections)


def test_stack_asset_bytes_refused_before_payload_io(tmp_path):
    path = tmp_path / 'too-large.safetensors'
    with path.open('wb') as stream:
        stream.truncate(adapter.MAX_PLAIN_STACK_BYTES + 1)
    with pytest.raises(ValueError, match='2 GiB'):
        adapter.verify_plain_stack([{'path': path, 'strength': 1}])


def test_stack_cumulative_asset_limit_checked_before_next_payload(tmp_path, monkeypatch):
    selections = []
    for i in range(3):
        path = tmp_path / f'large{i}.safetensors'
        with path.open('wb') as stream:
            stream.truncate(800 * 2**20)
        selections.append({'path': path, 'strength': .5})
    verified = []
    def verify(path, **recipe):
        verified.append(path)
        return {'path': str(path), 'policy': 'plain', 'strength': recipe['strength'],
                'sha256': str(len(verified)), 'file_identity': list(adapter.readonly._stat(path.stat()))}
    monkeypatch.setattr(adapter, 'verify_plain_adapter', verify)
    with pytest.raises(ValueError, match='2 GiB'):
        adapter.verify_plain_stack(selections)
    assert verified == [item['path'] for item in selections[:2]]


def test_real_stack_parser_additive_delta_order_and_atomic_failure():
    script = r'''
import sys, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import torch._dynamo
torch.backends.mps.is_available = lambda: False
import comfy.lora as lora
import comfy.utils as utils
import comfy.model_patcher as mp
import comfy.ops as ops
from comfy.quant_ops import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from safetensors.torch import save_file
from krea_adapter import verify_plain_stack, load_plain_stack, _patches, _asset_unchanged
CPU = torch.device('cpu')
key = 'diffusion_model.blocks.0.attn.wq.weight'
global_key = 'diffusion_model.projector.weight'
model = torch.nn.Module()
operations = ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
def layer(key, n, k, quant):
    parent = model
    pieces = key[:-7].split('.')
    for piece in pieces[:-1]:
        if piece not in parent._modules:
            parent.add_module(piece, torch.nn.Module())
        parent = parent._modules[piece]
    value = operations.Linear(k,n,bias=False,device=CPU)
    if quant:
        params = TensorWiseINT8Layout.Params(scale=torch.ones(n,1),orig_dtype=torch.bfloat16,
            orig_shape=(n,k),convrot=True,convrot_groupsize=256)
        value.weight = torch.nn.Parameter(QuantizedTensor(torch.ones(n,k,dtype=torch.int8),
            'TensorWiseINT8Layout',params),requires_grad=False)
        value.quant_format = 'int8_tensorwise'
    else:
        value.weight = torch.nn.Parameter(torch.ones(n,k,dtype=torch.bfloat16),requires_grad=False)
    parent.add_module(pieces[-1],value)
    return value
quant = layer(key,64,256,True)
global_layer = layer(global_key,1,12,False)
base = mp.ModelPatcher(model,CPU,CPU)
original_mapping = lora.model_lora_keys_unet
lora.model_lora_keys_unet = lambda actual, mapping: {**mapping,key[:-7]:key,global_key[:-7]:global_key}
gen = torch.Generator().manual_seed(615)
def factors(key,n,k,rank,alpha):
    prefix = key[:-7]
    return {prefix+'.lora_up.weight':torch.randn(n,rank,generator=gen)*.02,
            prefix+'.lora_down.weight':torch.randn(rank,k,generator=gen)*.02,
            prefix+'.alpha':torch.tensor(alpha)}
qptr = quant.weight._qdata.data_ptr()
gptr = global_layer.weight.data_ptr()
qbefore = quant.weight._qdata.clone()
gbefore = global_layer.weight.clone()
with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
    directory = Path(directory)
    selections = []
    expected_q = torch.zeros(64,256)
    expected_g = torch.zeros(1,12)
    strengths = [.6,-.3,.4,.7,.2]
    ranks = [16,32,48,32,32]  # summed rank160 selects eager sides
    for i,(rank,strength) in enumerate(zip(ranks,strengths)):
        state = factors(key,64,256,rank,float(rank)/2)
        if i % 2 == 0:
            state.update(factors(global_key,1,12,rank,float(rank)*1.5))
            expected_g += strength*1.5*(state[global_key[:-7]+'.lora_up.weight'] @ state[global_key[:-7]+'.lora_down.weight'])
        expected_q += strength*.5*(state[key[:-7]+'.lora_up.weight'] @ state[key[:-7]+'.lora_down.weight'])
        path = directory / f'style{i}.safetensors'
        save_file(state,str(path))
        selections.append({'path':path,'strength':strength})
    receipt = verify_plain_stack(selections)
    assert receipt['policy'] == 'plain-stack' and len(receipt['adapters']) == 5
    assert receipt['asset_bytes'] == sum(item['path'].stat().st_size for item in selections)
    patched = load_plain_stack(base,utils,receipt)
    assert patched.model is model and patched is not base
    assert len(patched.patches[key]) == 5 and len(patched.patches[global_key]) == 3
    assert [entry[0] for entry in patched.patches[key]] == strengths
    signature = _patches(patched,stack=True)
    assert _patches(patched.clone(),stack=True) == signature
    torch.testing.assert_close(lora.calculate_weight(patched.patches[key],torch.zeros(64,256),key),expected_q)
    torch.testing.assert_close(lora.calculate_weight(patched.patches[global_key],torch.zeros(1,12),global_key),expected_g)
    side = lora.lora_side_patches(patched.patches[key],64,256,CPU,torch.float32,weight=quant.weight)
    assert len(side) == 5 and not any(part.fused for part in side)
    x = torch.randn(7,256,generator=gen)
    out = torch.zeros(7,64)
    for part in side:
        out = part.apply(out,x)
    torch.testing.assert_close(out,torch.nn.functional.linear(x,expected_q),rtol=1e-5,atol=1e-6)
    fused = lora.lora_side_patches(patched.patches[key][:3],64,256,CPU,torch.float32,weight=quant.weight)
    assert len(fused) == 1 and fused[0].fused and fused[0].down.shape[0] == 96
    patched.patches[key].reverse()
    assert _patches(patched,stack=True) != signature
    patched.patches[key].reverse()
    try:
        _patches(patched,plain=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Legacy single route admitted stack')
    duplicate = directory/'alias.safetensors'
    shutil.copyfile(selections[0]['path'],duplicate)
    try:
        verify_plain_stack([selections[0],{'path':duplicate,'strength':.1}])
    except ValueError as error:
        assert 'Duplicate' in str(error)
    else:
        raise AssertionError('Duplicate content admitted')
    # Unsupported fifth payload cannot publish or contaminate either live clone.
    bad = directory/'bad.safetensors'
    save_file({'unmapped.weight':torch.ones(2,2)},str(bad))
    bad_receipt = verify_plain_stack(selections[:4]+[{'path':bad,'strength':.8}])
    try:
        load_plain_stack(base,utils,bad_receipt)
    except ValueError:
        pass
    else:
        raise AssertionError('Unsupported fifth asset admitted')
    assert not base.patches and not base.backup
    assert _patches(patched,stack=True) == signature
    assert quant.weight._qdata.data_ptr() == qptr and global_layer.weight.data_ptr() == gptr
    assert torch.equal(quant.weight._qdata,qbefore) and torch.equal(global_layer.weight,gbefore)
    # A vetted model-options callback replaces only that options check. Other
    # hooks/wrappers remain refused before the callback can admit anything.
    import krea_adapter as helper
    import comfy.model_base
    helper._runtime = lambda mm, utils: None  # CPU proof isolates options admission.
    comfy.model_base.Krea2 = type(model)
    base.load_device = torch.device('mps')
    contract = helper.AdapterContract.__new__(helper.AdapterContract)
    contract.mm, contract.utils, contract.target, contract.adapter = None,utils,model,receipt
    contract.model_options_validator = None
    allowed = {'transformer_options': {'controlled_reference': object()}}
    base.model_options = allowed
    try:
        contract.validate(base,model,initial=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Default policy admitted nonstandard model options')
    calls = []
    def validate_options(candidate):
        calls.append(candidate)
        if candidate.model_options is not allowed:
            raise ValueError('Reference options differ from admitted identity')
    contract.model_options_validator = validate_options
    contract.validate(base,model,initial=True)
    assert calls == [base]
    base.model_options = {'transformer_options': {}}
    try:
        contract.validate(base,model,initial=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Controlled options change escaped callback')
    base.model_options = allowed
    base.hook_patches = {'unexpected': {}}
    previous_calls = len(calls)
    try:
        contract.validate(base,model,initial=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Options callback bypassed hook refusal')
    assert len(calls) == previous_calls
    base.hook_patches = {}
    selections[-1]['path'].write_bytes(selections[-1]['path'].read_bytes()+b'changed')
    try:
        _asset_unchanged(receipt)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Changed last asset escaped stack validation')
lora.model_lora_keys_unet = original_mapping
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
