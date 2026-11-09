"""Real CPU safetensors/T5 lifecycle proof, isolated from pytest's Comfy globals."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_real_t5_storage_and_patch_lifecycle(tmp_path):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ENCODER_FIXTURE_OK' in result.stdout
    assert 'ENCODER_MAIN_PATH_OK' in result.stdout


def test_probe_requires_supervised_group_before_loading(tmp_path):
    # A bounded header is enough for dry inspection; no tensor payload is read.
    import struct
    header = json.dumps({'shared.weight': {'dtype': 'F16', 'shape': [256384, 4096],
                                          'data_offsets': [0, 2100297728]}}).encode()
    checkpoint = tmp_path / 'header-only.safetensors'
    checkpoint.write_bytes(struct.pack('<Q', len(header)) + header)
    env = dict(os.environ, MK_GPU_LOCK=str(tmp_path / 'absent-lock'))
    command = [sys.executable, str(ROOT / 'bench/probe_text_encoder_load.py'), str(checkpoint)]
    dry = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)['run'] is False
    run = subprocess.run(command + ['--run'], env=env, capture_output=True, text=True, timeout=10)
    assert run.returncode != 0
    assert 'gpu_runner.py --monitor' in run.stderr
    assert 'state_read' not in run.stdout


def run_fixture(directory):
    import gc
    import hashlib
    sys.path.insert(0, str(ROOT))
    from bench.probe_text_encoder_load import bootstrap_cpu, make_model_class, load_validated_state
    bootstrap_cpu()
    import torch
    from safetensors.torch import save_file
    import comfy.sd
    import comfy.utils
    from comfy.text_encoders.wan import UMT5XXlModel, WanT5Model
    from types import SimpleNamespace
    config = json.loads((ROOT / 'vendor/ComfyUI/comfy/text_encoders/umt5_config_xxl.json').read_text())
    config.update(d_model=32, d_ff=64, d_kv=8, num_heads=4, num_layers=1,
                  num_decoder_layers=1, vocab_size=128)
    class TokenizerBoundary:
        def __init__(self, **kwargs):
            pass
    target = SimpleNamespace(params={}, clip=WanT5Model, tokenizer=TokenizerBoundary)
    def build(state, assign):
        events = []
        cls = make_model_class(UMT5XXlModel, assign=assign,
                               emit=lambda event, **values: events.append((event, values)))
        clip = comfy.sd.CLIP(target=target, state_dict=[state], model_options={
            'umt5xxl_class': cls, 'umt5xxl_model_config': config.copy(),
            'dtype': torch.float16, 'load_device': torch.device('cpu'),
            'offload_device': torch.device('cpu'), 'initial_device': torch.device('cpu')})
        return clip, events
    torch.manual_seed(7)
    seed = UMT5XXlModel(device='cpu', dtype=torch.float16,
        model_options={'model_name': 'umt5xxl', 'umt5xxl_model_config': config.copy()})
    original = {k: (torch.ones_like(v) if 'layer_norm' in k else torch.randn_like(v)*0.1)
                for k, v in seed.transformer.state_dict().items()}
    del seed
    path = Path(directory) / 'tiny.safetensors'
    save_file(original, str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ids = torch.tensor([[3, 5, 7, 11]])
    def output(clip):
        with torch.no_grad():
            return clip.cond_stage_model.umt5xxl.transformer(ids, None)[0].clone()
    arm_outputs = []
    for assign in (False, True):
        source = comfy.utils.load_torch_file(str(path), safe_load=True)
        clip, events = build(source, assign)
        assert [e[0] for e in events] == ['construct_start', 'constructed', 'load_start', 'state_applied']
        assert events[-1][1]['shared_storage_tensors'] == (len(original) if assign else 0)
        baseline = output(clip)
        assert torch.isfinite(baseline).all()
        arm_outputs.append(baseline)
        del source
        gc.collect()
        assert torch.equal(output(clip), baseline)
        key = 'umt5xxl.transformer.shared.weight'
        weight = clip.cond_stage_model.umt5xxl.transformer.shared.weight.detach().clone()
        clone = clip.clone()
        clone.add_patches({key: ('diff', (torch.full_like(weight, 0.125),))})
        for inplace in (False, True):
            clone.patcher.patch_weight_to_device(key, device_to=torch.device('cpu'), inplace_update=inplace)
            assert torch.equal(clone.cond_stage_model.umt5xxl.transformer.shared.weight, weight + 0.125)
            assert not torch.equal(output(clone), baseline)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            reread = comfy.utils.load_torch_file(str(path), safe_load=True)
            assert torch.equal(reread['shared.weight'], weight)
            del reread
            clone.patcher.unpatch_model(device_to=torch.device('cpu'))
            assert torch.equal(output(clip), baseline)
        clip.cond_stage_model.to(dtype=torch.float32)
        clip.cond_stage_model.to(dtype=torch.float16)
        assert torch.equal(output(clip), baseline)
        del clip, clone
        comfy.model_management.unload_all_models()
    assert torch.equal(*arm_outputs)
    # Disconfirming cases: validate the full contract before mutating any parameter.
    source = comfy.utils.load_torch_file(str(path), safe_load=True)
    clip, _ = build(source, True)
    inner = clip.cond_stage_model.umt5xxl
    before = output(clip)
    cases = []
    bad = dict(source); bad.pop('shared.weight'); cases.append(bad)
    bad = dict(source); bad['unexpected.weight'] = torch.ones(1, dtype=torch.float16); cases.append(bad)
    bad = dict(source); bad['shared.weight'] = source['shared.weight'].float(); cases.append(bad)
    bad = dict(source); bad['shared.weight'] = source['shared.weight'][:1]; cases.append(bad)
    bad = dict(source); bad['shared.weight'] = source['shared.weight'].to('meta'); cases.append(bad)
    bad = dict(source); bad['spiece_model'] = torch.ones(2, dtype=torch.float16); cases.append(bad)
    for bad in cases:
        try:
            load_validated_state(inner, bad, assign=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported checkpoint was accepted')
        assert torch.equal(output(clip), before)
    with_metadata = dict(source, spiece_model=torch.tensor([1, 2, 3], dtype=torch.uint8))
    load_validated_state(inner, with_metadata, assign=True)
    assert torch.equal(output(clip), before)
    # Explicit ownership difference: assigned parameters alias caller-held tensors.
    with torch.no_grad():
        source['shared.weight'].add_(0.125)
    assert not torch.equal(output(clip), before)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    print('ENCODER_FIXTURE_OK')
    run_main_fixture(directory)


def run_main_fixture(directory):
    """Exercise actual main, detection, Wan tokenizer/te(), inference, save, unload.

    Only external config/header dimensions and supervisor membership are replaced.
    24 thin layers retain the real XXL detector key/shape, using about 6 MiB.
    """
    import builtins
    import io
    from unittest.mock import patch
    import torch
    import sentencepiece
    from safetensors.torch import save_file
    import comfy.model_management as mm
    from comfy.text_encoders.wan import UMT5XXlModel
    import bench.probe_text_encoder_load as probe
    config_path = ROOT / 'vendor/ComfyUI/comfy/text_encoders/umt5_config_xxl.json'
    config = json.loads(config_path.read_text())
    config.update(d_model=4, d_kv=4, num_heads=1, vocab_size=128)
    # Real SentencePiece bytes make the production Wan tokenizer path executable.
    writer = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(['a musician plays guitar in the subway'] * 8),
        model_writer=writer, vocab_size=32, hard_vocab_limit=False,
        bos_id=-1, eos_id=1, unk_id=2, pad_id=0, minloglevel=2)
    torch.manual_seed(11)
    seed = UMT5XXlModel(device='cpu', dtype=torch.float16,
        model_options={'model_name': 'umt5xxl', 'umt5xxl_model_config': config.copy()})
    state = {k: (torch.ones_like(v) if 'layer_norm' in k else torch.randn_like(v)*0.01)
             for k, v in seed.transformer.state_dict().items()}
    state['spiece_model'] = torch.tensor(list(writer.getvalue()), dtype=torch.uint8)
    path = Path(directory) / 'thin-xxl.safetensors'
    save_file(state, str(path))
    del state, seed
    real_open = builtins.open
    tiny_config = Path(directory) / 'thin-config.json'
    tiny_config.write_text(json.dumps(config))
    def config_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file).resolve() == config_path.resolve():
            file = tiny_config
        return real_open(file, *args, **kwargs)
    payloads = []
    for assign in (False, True):
        output_path = Path(directory) / f'conditioning-{assign}.pt'
        argv = ['probe', str(path), '--run', '--seconds', '45', '--prompt',
                'a musician plays guitar', '--output', str(output_path)]
        if assign:
            argv.append('--assign')
        with patch.object(sys, 'argv', argv), patch.object(probe, 'require_supervisor'), \
                patch.object(probe, 'inspect_checkpoint', return_value=path.stat().st_size), \
                patch('builtins.open', side_effect=config_open):
            probe.main()
        payload = torch.load(output_path, weights_only=True)
        assert payload['assign'] is assign
        assert payload['prompt'] == 'a musician plays guitar'
        cond = payload['conditioning']['cond']
        assert list(cond.shape) == [1, 512, 4]
        assert cond.device.type == 'cpu' and torch.isfinite(cond).all()
        assert not mm.current_loaded_models
        payloads.append(payload)
    assert torch.equal(payloads[0]['conditioning']['cond'], payloads[1]['conditioning']['cond'])
    print('ENCODER_MAIN_PATH_OK')


if __name__ == '__main__':
    run_fixture(sys.argv[1])
