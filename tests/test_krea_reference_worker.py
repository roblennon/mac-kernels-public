"""Real CPU serialization and owned child exit at the VAE/encoder boundary."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from bench import krea_reference as reference


def fixture_result(tmp_path):
    images = (torch.linspace(0, 1, 16*16*3).reshape(1,16,16,3),)
    latent = torch.arange(64, dtype=torch.float32).reshape(1,16,1,2,2)
    digest = reference._digest(latent)
    metadata = {'mode': 'style_reference', 'spectral': False, 'reference_count': 1, 'output_size': [512,512],
                'images': [{'sha256': reference._digest(x), 'shape': list(x.shape)} for x in images],
                'latent_shapes': [list(latent.shape)], 'latent_sha256': [digest],
                'order': 'input-order', 'author_revision': None,
                'reference_method': 'index_timestep_zero', 'ref_boost': None, 'grounding_size': 384,
                'encoded': [{'index': 0, 'target': [64,64], 'pixels': reference._digest(images[0]),
                             'shape': list(latent.shape), 'sha256': digest}]}
    # Same latent referenced by the ordered list and prepared-pixel lookup must
    # have one serialized payload, not duplicate shared safetensors storage.
    prepared = reference.PreparedReference('style_reference', images, (latent,),
                                          {reference._digest(images[0]): latent}, metadata)
    bindings = {'vae_identity': [1,2,3,4,5], 'sources': {'worker': 'fixture'},
                'comfy_root': str(tmp_path), 'comfy_sources': {'sd.py': 'fixture', 'utils.py': 'fixture'}}
    reference._write_reference_result(prepared, tmp_path, torch=torch, bindings=bindings, pid=os.getpid())
    kwargs = dict(images=images, mode='style_reference', torch=torch, bindings=bindings,
                  pid=os.getpid(), runtime=None, ref_boost=4., grounding_size=512)
    return prepared, kwargs


def test_reference_result_deduplicates_and_restores_exact_cpu_payload(tmp_path):
    prepared, kwargs = fixture_result(tmp_path)
    result = reference._read_reference_result(tmp_path, **kwargs)
    torch.testing.assert_close(result.latents[0], prepared.latents[0], rtol=0, atol=0)
    assert result.latents[0].device.type == 'cpu' and result.latents[0].dtype == torch.float32
    assert next(iter(result.encoded.values())) is result.latents[0]
    assert result.author is None
    assert result.metadata['vae_preparation']['worker_exited'] is True
    assert len(json.loads((tmp_path/'result.json').read_text())['tensors']) == 1


@pytest.mark.parametrize('corrupt', ['latent_hash', 'images', 'order', 'bindings', 'pid', 'membership', 'spectral', 'output_size', 'encoded_hash', 'extra_encoded', 'missing_encoded'])
def test_reference_result_refuses_changed_boundary_receipt(tmp_path, corrupt):
    _, kwargs = fixture_result(tmp_path)
    path = tmp_path/'result.json'
    result = json.loads(path.read_text())
    if corrupt == 'latent_hash':
        result['metadata']['latent_sha256'] = ['0'*64]
    elif corrupt == 'images':
        result['metadata']['images'][0]['sha256'] = '0'*64
    elif corrupt == 'order':
        result['metadata']['order'] = 'scene,subject'
    elif corrupt == 'bindings':
        result['bindings']['vae_identity'][0] = 99
    elif corrupt == 'pid':
        result['pid'] += 1
    elif corrupt == 'spectral':
        result['metadata']['spectral'] = True
    elif corrupt == 'output_size':
        result['metadata']['output_size'] = [512,768]
    elif corrupt == 'encoded_hash':
        result['metadata']['encoded'][0]['sha256'] = '0'*64
    elif corrupt == 'extra_encoded':
        result['encoded']['0'*64] = 'latent_0'
    elif corrupt == 'missing_encoded':
        result['metadata']['encoded'] = []
    else:
        result['latent_keys'] = ['latent_0', 'latent_0']
    path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError):
        reference._read_reference_result(tmp_path, **kwargs)


def test_owned_worker_waits_for_exit_and_reaps_timeout(tmp_path):
    with (tmp_path/'worker.log').open('wb') as log:
        pid = reference._run_reference_worker([sys.executable, '-c', 'pass'], log, timeout=5)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    pid_path = tmp_path/'pid'
    script = 'import os,sys,time; open(sys.argv[1], "w").write(str(os.getpid())); time.sleep(20)'
    with (tmp_path/'timeout.log').open('wb') as log, pytest.raises(subprocess.TimeoutExpired):
        reference._run_reference_worker([sys.executable, '-c', script, str(pid_path)], log, timeout=.3)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)


def test_owned_worker_refuses_failed_child_after_exit(tmp_path):
    with (tmp_path/'worker.log').open('wb') as log, pytest.raises(RuntimeError, match='exit 7'):
        reference._run_reference_worker([sys.executable, '-c', 'raise SystemExit(7)'], log, timeout=5)


def test_worker_comfy_binding_uses_selected_root_and_rejects_foreign_modules(tmp_path, monkeypatch):
    from types import SimpleNamespace
    selected = tmp_path/'selected'
    foreign = tmp_path/'foreign'
    for root in (selected, foreign):
        (root/'comfy').mkdir(parents=True)
        for name in ('sd.py', 'utils.py'):
            (root/'comfy'/name).write_text('# ' + root.name + name)
    monkeypatch.setenv('COMFY_DIR', str(foreign))
    binding = reference._worker_comfy_binding(selected)
    assert binding['comfy_root'] == str(selected.resolve())
    assert set(binding['comfy_sources']) == {'sd.py', 'utils.py'}
    sd = SimpleNamespace(__file__=str(selected/'comfy/sd.py'))
    utils = SimpleNamespace(__file__=str(selected/'comfy/utils.py'))
    reference._validate_worker_comfy_modules(selected, sd, utils)
    with pytest.raises(RuntimeError, match='foreign Comfy'):
        reference._validate_worker_comfy_modules(selected,
            SimpleNamespace(__file__=str(foreign/'comfy/sd.py')), utils)
    with pytest.raises(RuntimeError, match='foreign Comfy'):
        reference._validate_worker_comfy_modules(selected, sd,
            SimpleNamespace(__file__=str(foreign/'comfy/utils.py')))
    (selected/'comfy/sd.py').unlink()
    (selected/'comfy/sd.py').symlink_to(foreign/'comfy/sd.py')
    with pytest.raises(ValueError, match='outside'):
        reference._worker_comfy_binding(selected)


def test_owned_request_binds_rectangular_output_into_worker_and_reader(tmp_path, monkeypatch):
    from safetensors.torch import load_file
    from bench import probe_text_encoder_load
    monkeypatch.setattr(probe_text_encoder_load, 'require_supervisor', lambda: None)
    (tmp_path/'out').mkdir()
    vae_path = tmp_path/'vae.safetensors'
    vae_path.write_bytes(b'external VAE boundary fixture')
    comfy_root = tmp_path/'comfy-root'
    (comfy_root/'comfy').mkdir(parents=True)
    for name in ('sd.py', 'utils.py'):
        (comfy_root/'comfy'/name).write_text('# selected source boundary')
    seen = []
    class VAE:
        def encode(self, pixels):
            h,w = pixels.shape[1:3]
            return torch.zeros(1,16,1,h//8,w//8)
    def child_boundary(command, log, *, timeout=120):
        directory = Path(command[2])
        config = json.loads((directory/'config.json').read_text())
        seen.append(config['output_size'])
        image = load_file(str(directory/'images.safetensors'))['image_0']
        prepared = reference.prepare_reference(config['mode'], image, vae=VAE(), torch=torch,
            spectral=config['spectral'], output_size=config['output_size'])
        reference._write_reference_result(prepared, directory, torch=torch,
            bindings=config['bindings'], pid=os.getpid())
        return os.getpid()
    monkeypatch.setattr(reference, '_run_reference_worker', child_boundary)
    prepared = reference.prepare_reference_owned('style_reference', torch.zeros(1,16,32,3),
        vae_path=vae_path, root=tmp_path, torch=torch, comfy_root=comfy_root,
        spectral=True, output_size=(832,1248))
    assert seen == [[832,1248]]
    assert prepared.output_size == (832,1248)
    assert prepared.metadata['output_size'] == [832,1248]
    assert prepared.metadata['encoded'][0]['target'] == [156,104]
    assert prepared.latents[0].shape == (1,16,1,2,4)
    prepared.metadata['output_size'] = [1248,832]
    with pytest.raises(ValueError, match='changed after preparation'):
        reference._validate_prepared_tensors(prepared)
