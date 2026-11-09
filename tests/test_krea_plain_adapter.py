"""Ordinary adapter admission uses real Comfy parsing and CPU patch math."""
from pathlib import Path
import subprocess
import sys

import pytest

from bench import krea_adapter as adapter

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('override', [dict(strength=0), dict(strength=float('nan')),
                                     dict(strength=float('inf')), dict(steps=4),
                                     dict(cfg=2), dict(spectral=True)])
def test_plain_recipe_refuses_invalid_values_before_asset_io(tmp_path, override):
    recipe = dict(strength=.8, steps=8, cfg=1.)
    recipe.update(override)
    with pytest.raises(ValueError):
        adapter.verify_plain_adapter(tmp_path / 'missing.safetensors', **recipe)


def test_real_plain_parser_subset_overcomplete_rank_and_failure_atomicity():
    script = r'''
import sys
import tempfile
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
from comfy.weight_adapter import LoRAAdapter
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from safetensors.torch import save_file
from krea_adapter import verify_plain_adapter, load_plain_adapter, _patches

CPU = torch.device('cpu')
quant_key = 'diffusion_model.blocks.0.attn.wq.weight'
global_key = 'diffusion_model.projector.weight'
model = torch.nn.Module()
operations = ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
def add_linear(key, n, k, quantized):
    parent = model
    pieces = key[:-7].split('.')
    for piece in pieces[:-1]:
        if piece not in parent._modules:
            parent.add_module(piece, torch.nn.Module())
        parent = parent._modules[piece]
    layer = operations.Linear(k, n, bias=False, device=CPU)
    if quantized:
        params = TensorWiseINT8Layout.Params(scale=torch.ones(n, 1),
            orig_dtype=torch.bfloat16, orig_shape=(n,k), convrot=True, convrot_groupsize=256)
        layer.weight = torch.nn.Parameter(QuantizedTensor(torch.ones(n,k,dtype=torch.int8),
            'TensorWiseINT8Layout', params), requires_grad=False)
        layer.quant_format = 'int8_tensorwise'
    else:
        layer.weight = torch.nn.Parameter(torch.ones(n,k,dtype=torch.bfloat16), requires_grad=False)
    parent.add_module(pieces[-1], layer)
    return layer
quant = add_linear(quant_key, 64, 256, True)
projector = add_linear(global_key, 1, 12, False)
base = mp.ModelPatcher(model, CPU, CPU)
# The mapping boundary is reduced to exactly the fixture's real model keys.
# Converter, parser, adapter class, patcher, safetensors I/O and arithmetic remain real.
original_mapping = lora.model_lora_keys_unet
aliases = {key[:-7]: key for key in (quant_key, global_key)}
lora.model_lora_keys_unet = lambda actual_model, mapping: {**mapping, **aliases}
gen = torch.Generator().manual_seed(71)
def factors(key, n, k, rank, alpha=None):
    prefix = key[:-7]
    state = {prefix+'.lora_up.weight': torch.randn(n,rank,generator=gen)*.01,
             prefix+'.lora_down.weight': torch.randn(rank,k,generator=gen)*.01}
    if alpha is not None:
        state[prefix+'.alpha'] = torch.tensor(alpha)
    return state
quant_state = factors(quant_key,64,256,16,8.)
global_state = factors(global_key,1,12,32,48.)
quant_pointer = quant.weight._qdata.data_ptr()
global_pointer = projector.weight.data_ptr()
quant_before = quant.weight._qdata.clone()
global_before = projector.weight.clone()

with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
    path = Path(directory) / 'ordinary.safetensors'
    def load(state, strength=.8):
        save_file(state, str(path))
        receipt = verify_plain_adapter(path, strength=strength, steps=8, cfg=1.)
        assert receipt['policy'] == 'plain'
        assert receipt['file_identity'][2] == path.stat().st_size
        assert len(receipt['sha256']) == 64
        return load_plain_adapter(base, utils, receipt, strength)
    patched = load({**quant_state, **global_state})
    assert patched is not base and patched.model is model
    assert set(patched.patches) == {quant_key,global_key}
    signature = _patches(patched, plain=True)
    assert len(signature) == 2 and _patches(patched.clone(),plain=True) == signature
    receipt = verify_plain_adapter(path,strength=.8,steps=8,cfg=1.)
    try:
        load_plain_adapter(base,utils,receipt,.5)
    except ValueError:
        pass
    else:
        raise AssertionError('Strength bypassed its verified recipe')
    path.write_bytes(path.read_bytes() + b'changed')
    try:
        load_plain_adapter(base,utils,receipt,.8)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Changed asset accepted after verification')
    # Non-unit alpha and strength must affect the actual patch delta, including
    # the rank-32 projector whose output dimension is only one.
    for key, state, alpha, rank in ((quant_key,quant_state,8.,16),
                                   (global_key,global_state,48.,32)):
        prefix = key[:-7]
        n,k = model.get_submodule(prefix).weight.shape
        expected = .8 * alpha/rank * (state[prefix+'.lora_up.weight'] @ state[prefix+'.lora_down.weight'])
        actual = lora.calculate_weight(patched.patches[key], torch.zeros(n,k), key)
        torch.testing.assert_close(actual, expected)
    subset = load(global_state, -.5)
    assert set(subset.patches) == {global_key}
    actual = lora.calculate_weight(subset.patches[global_key], torch.zeros(1,12), global_key)
    expected = -.5*48./32*(global_state[global_key[:-7]+'.lora_up.weight'] @ global_state[global_key[:-7]+'.lora_down.weight'])
    torch.testing.assert_close(actual, expected)

    def refuses(state):
        try:
            load(state)
        except (ValueError,RuntimeError):
            pass
        else:
            raise AssertionError('Unsupported or partial adapter accepted')
        assert not base.patches and not base.backup
        assert quant.weight._qdata.data_ptr() == quant_pointer
        assert projector.weight.data_ptr() == global_pointer
        assert torch.equal(quant.weight._qdata,quant_before)
        assert torch.equal(projector.weight,global_before)
    refuses({**quant_state, 'text_encoders.unrelated.lora_up.weight': torch.ones(1,1)})
    refuses({**quant_state, 'unmapped.alpha': torch.tensor(1.)})
    # Comfy records a mapped alpha as consumed even when that target has no pair.
    refuses({**quant_state, global_key[:-7]+'.alpha': torch.tensor(32.)})
    prefix = quant_key[:-7]
    refuses({prefix+'.lora_up.weight': quant_state[prefix+'.lora_up.weight']})
    refuses({**quant_state, prefix+'.lora_down.weight': torch.ones(15,256)})
    refuses({**quant_state, prefix+'.dora_scale': torch.ones(64)})
    refuses({**quant_state, prefix+'.lora_mid.weight': torch.ones(16,16,1,1)})
    refuses({**quant_state, prefix+'.reshape_weight': torch.tensor([64,256])})
    refuses({**quant_state, prefix+'.alpha': torch.tensor(float('nan'))})
    refuses({**quant_state, prefix+'.lora_up.weight': torch.full((64,16),float('inf'))})
    # A genuinely mapped-but-absent target must not become a silently ignored patch.
    absent = 'diffusion_model.missing.weight'
    aliases[absent[:-7]] = absent
    refuses({**quant_state, **factors(absent,2,12,3)})
    # Two dimensions alone do not make an embedding an admitted Linear target.
    embedding_key = 'diffusion_model.embedding.weight'
    model.diffusion_model.add_module('embedding', torch.nn.Embedding(4,12,dtype=torch.bfloat16))
    aliases[embedding_key[:-7]] = embedding_key
    refuses({**quant_state, **factors(embedding_key,4,12,3)})

    original_quant = quant.weight
    quant.weight = torch.nn.Parameter(torch.ones(64,256,dtype=torch.bfloat16),requires_grad=False)
    try:
        _patches(patched,plain=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Dense replacement bypassed the native INT8 base contract')
    finally:
        quant.weight = original_quant

    # Krea maps both attn.to_out and attn.to_out.0 onto the same output weight.
    # An alpha-only shorter alias must not borrow the longer alias's pair:
    # Comfy marks the alpha consumed but assigns alpha=None to that pair.
    alias = 'transformer_blocks.0.attn.to_out'
    aliases[alias] = quant_key
    aliases[alias + '.0'] = quant_key
    refuses({alias+'.0.lora_up.weight': quant_state[quant_key[:-7]+'.lora_up.weight'],
             alias+'.0.lora_down.weight': quant_state[quant_key[:-7]+'.lora_down.weight'],
             alias+'.alpha': torch.tensor(8.)})

    entry = patched.patches[quant_key][0]
    def bad_entry(value):
        patched.patches[quant_key] = [value]
        try:
            _patches(patched,plain=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported whole-weight patch transform accepted')
        finally:
            patched.patches[quant_key] = [entry]
    bad_entry((entry[0],entry[1],.5,None,None))
    bad_entry((entry[0],entry[1],1.,(0,0,64),None))
    bad_entry((entry[0],entry[1],1.,None,lambda x:x))
    patched.patches[quant_key] = [entry,entry]
    try:
        _patches(patched,plain=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Stacking accepted')
    patched.patches[quant_key] = [entry]
    # General subset support cannot weaken the separately pinned release contract.
    try:
        _patches(patched)
    except ValueError:
        pass
    else:
        raise AssertionError('Strict released adapter accepted a partial style')
lora.model_lora_keys_unet = original_mapping
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
