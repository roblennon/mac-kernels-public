"""Explicit MPS safetensors loading preserves bytes and CPU quantization metadata."""
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor/ComfyUI'))
import comfy.utils as utils

pytestmark = pytest.mark.skipif(
    os.environ.get('RUN_MPS_TESTS') != '1' or not torch.backends.mps.is_available(),
    reason='Opt-in MPS loader integration',
)


@pytest.mark.parametrize('disable_mmap', [False, True])
def test_direct_load_preserves_quant_metadata_and_weight_bytes(tmp_path, monkeypatch, disable_mmap):
    # The metadata tensor must remain CPU-readable by the actual quant loader.
    config = json.dumps({'format':'int8_tensorwise','convrot':False}).encode()
    state = {
        'layer.weight': torch.arange(1024, dtype=torch.int32).remainder(255).sub(127).to(torch.int8).reshape(4,256),
        'layer.weight_scale': torch.tensor([.001,.002,.003,.004]),
        'layer.comfy_quant': torch.tensor(list(config), dtype=torch.uint8),
    }
    path = tmp_path/'quant.safetensors'
    save_file(state, path, metadata={'description':'direct-load contract'})
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(utils, 'DISABLE_MMAP', disable_mmap)
    monkeypatch.setattr(utils.comfy.memory_management, 'aimdo_enabled', False)
    loaded, metadata = utils.load_torch_file(str(path), device=torch.device('mps'), return_metadata=True)
    assert metadata == {'description':'direct-load contract'}
    assert loaded['layer.weight'].device.type == 'mps'
    assert loaded['layer.weight_scale'].device.type == 'mps'
    assert json.loads(loaded['layer.comfy_quant'].numpy().tobytes()) == json.loads(config)
    for key, value in state.items():
        assert torch.equal(loaded[key].cpu(), value)
    from comfy.ops import mixed_precision_ops
    layer = mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(256,4,bias=False,device=None)
    pointer = loaded['layer.weight'].data_ptr()
    layer.load_state_dict({k.removeprefix('layer.'):v for k,v in loaded.items()}, strict=True)
    assert layer.weight._qdata.data_ptr() == pointer
    assert layer.weight._qdata.device.type == 'mps'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    del layer, loaded
    torch.mps.synchronize()
    torch.mps.empty_cache()
