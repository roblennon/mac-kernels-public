"""Run the actual evaluation entrypoint with only Comfy node/service boundaries isolated."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    calls = []
    (tmp_path / 'out').mkdir()
    model = SimpleNamespace(model=torch.nn.Linear(4, 4), adapter=False)
    clip = SimpleNamespace(tokenize=lambda text, **kw: text,
                           encode_from_tokens_scheduled=lambda tokens: [[torch.zeros(1, 2, 4), {}]])
    def sample(current, *_args):
        calls.append(current.adapter)
        return ({'samples': torch.full((1, 32, 32, 3), .7 if current.adapter else .3)},)
    node_types = {
        'CLIPLoader': lambda: SimpleNamespace(load_clip=lambda *_: (clip,)),
        'UNETLoader': lambda: SimpleNamespace(load_unet=lambda *_: (model,)),
        'LoraLoaderModelOnly': lambda: SimpleNamespace(load_lora_model_only=lambda base, *_:
                                                       (SimpleNamespace(model=base.model, adapter=True),)),
        'VAELoader': lambda: SimpleNamespace(load_vae=lambda *_: (object(),)),
        'ConditioningZeroOut': lambda: SimpleNamespace(zero_out=lambda pos: (pos,)),
        'KSampler': lambda: SimpleNamespace(sample=sample),
        'VAEDecode': lambda: SimpleNamespace(decode=lambda _vae, latent: (latent['samples'],)),
    }
    cb = SimpleNamespace(ROOT=str(tmp_path), N=node_types, call=None,
                         mm=SimpleNamespace(unload_all_models=lambda: None, soft_empty_cache=lambda: None),
                         folder_paths=SimpleNamespace(get_full_path=lambda _kind, name: name),
                         comfy=SimpleNamespace(
                             utils=SimpleNamespace(load_torch_file=lambda _: {'layer.lora_A.weight': torch.ones(1, 4)}),
                             lora=SimpleNamespace(model_lora_keys_unet=lambda *_: {}, load_lora=lambda *_args, **_kw: {'weight': object()})))
    monkeypatch.setitem(sys.modules, 'comfy_boot', cb)
    path = Path(__file__).resolve().parents[1] / 'train/eval_lora.py'
    spec = importlib.util.spec_from_file_location('tested_eval_lora_columns', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, cb, calls, tmp_path


@pytest.mark.parametrize('column,expected', [('all', [False, True]), ('base', [False]), ('adapter', [True])])
def test_single_column_renders_one_image_and_default_keeps_comparison(evaluation, monkeypatch, column, expected):
    module, cb, calls, root = evaluation
    monkeypatch.setattr(sys, 'argv', ['eval', '--arch', 'qwen21', '--unet', 'base.safetensors',
                                    '--lora', 'trained.safetensors', '--prompts', 'held-out subject',
                                    '--res', '32', '--palette', 'cyan', '--tag', column, '--column', column])
    module.main()
    assert calls == expected
    result = json.loads((root / 'out/results.jsonl').read_text())
    assert result['column'] == column and Path(result['sheet']).is_file()
    assert result['mps_direct_load'] is False
    assert len(list((root / 'out').glob('*.png'))) == len(expected) + 1
    assert ('palette_none' in result['stats'][0]) == (column != 'adapter')
    assert ('psnr_vs_none[trained.safetensors]' in result['stats'][0]) == (column == 'all')


def test_nonfinite_decoded_image_is_rejected_before_artifact_export(evaluation, monkeypatch):
    module, cb, calls, root = evaluation
    cb.N['VAEDecode'] = lambda: SimpleNamespace(decode=lambda *_: (torch.full((1, 32, 32, 3), float('nan')),))
    monkeypatch.setattr(sys, 'argv', ['eval', '--arch', 'qwen21', '--unet', 'base.safetensors',
                                    '--lora', 'trained.safetensors', '--prompts', 'held-out subject',
                                    '--res', '32', '--tag', 'bad', '--column', 'adapter'])
    with pytest.raises(FloatingPointError, match='Non-finite decoded image'):
        module.main()
    assert calls == [True] and not list((root / 'out').iterdir())


def test_single_column_refuses_multiple_prompts_before_loading(evaluation, monkeypatch):
    module, cb, calls, root = evaluation
    cb.N['CLIPLoader'] = lambda: pytest.fail('Model loading must not occur')
    monkeypatch.setattr(sys, 'argv', ['eval', '--arch', 'qwen21', '--unet', 'base.safetensors',
                                    '--lora', 'trained.safetensors', '--tag', 'invalid', '--column', 'adapter'])
    with pytest.raises(SystemExit, match='2'):
        module.main()
    assert calls == []


@pytest.mark.parametrize('direct', [False, True])
def test_direct_load_is_explicit_node_option_and_recorded(evaluation, monkeypatch, direct):
    module, cb, calls, root = evaluation
    loaded = []
    model = SimpleNamespace(model=torch.nn.Linear(4, 4), adapter=False)
    if direct:
        def load(unet, weight_dtype, *, mps_direct_load):
            loaded.append((unet, weight_dtype, mps_direct_load))
            return (model,)
    else:
        # An old two-argument node remains callable when the option is omitted.
        def load(unet, weight_dtype):
            loaded.append((unet, weight_dtype))
            return (model,)
    cb.N['UNETLoader'] = lambda: SimpleNamespace(load_unet=load)
    argv = ['eval', '--arch', 'qwen21', '--unet', 'base.safetensors', '--lora', 'trained.safetensors',
            '--prompts', 'held-out subject', '--res', '32', '--tag', 'direct', '--column', 'base']
    monkeypatch.setattr(sys, 'argv', argv + (['--mps-direct-load'] if direct else []))
    module.main()
    assert loaded == [('base.safetensors', 'default', True) if direct else ('base.safetensors', 'default')]
    result = json.loads((root / 'out/results.jsonl').read_text())
    assert result['mps_direct_load'] is direct and calls == [False]
