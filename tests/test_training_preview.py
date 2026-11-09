"""CPU contracts for previews: RNG continuity, state cleanup, admission and Comfy sampling."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
import training_preview as preview


def arguments(**kwargs):
    p = argparse.ArgumentParser()
    preview.add_preview_arguments(p)
    a = p.parse_args([])
    a.profile_steps, a.steps = 0, 11
    for key, value in kwargs.items():
        setattr(a, key, value)
    return a


def test_preview_cadence_is_absolute_across_resume():
    args = arguments(preview_every=4)
    assert [n for n in range(12) if preview.preview_due(args, n)] == [0, 4, 8, 11]
    assert [n for n in range(7, 12) if preview.preview_due(args, n)] == [8, 11]
    assert not any(preview.preview_due(arguments(), n) for n in range(12))
    args.profile_steps = 2
    assert not preview.preview_due(args, 8)


@pytest.mark.parametrize('fail', [False, True])
def test_preview_preserves_next_optimizer_update_and_all_rng_on_failure(fail):
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.2))
    model[0].eval()  # preserve mixed modes, not just the root's flag
    lora = torch.nn.Linear(3, 3)
    opt = torch.optim.AdamW([*model.parameters(), *lora.parameters()])
    modes = [m.training for m in model.modules()]
    before = [p.detach().clone() for p in [*model.parameters(), *lora.parameters()]]
    rng = torch.get_rng_state()
    py, npstate = random.getstate(), np.random.get_state()
    expected = torch.randn(3)
    torch.set_rng_state(rng)
    try:
        with preview.preview_state(model, {'lora': lora}, torch.device('cpu')):
            assert not any(m.training for m in model.modules())
            assert not lora.training and not torch.is_grad_enabled()
            model(torch.randn(4, 3)) + lora(torch.randn(4, 3))
            random.random()
            np.random.rand()
            if fail:
                raise RuntimeError('decoder failed')
    except RuntimeError:
        assert fail
    torch.testing.assert_close(torch.randn(3), expected, rtol=0, atol=0)
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], npstate[1])
    assert [m.training for m in model.modules()] == modes and lora.training
    assert not opt.state
    for param, old in zip([*model.parameters(), *lora.parameters()], before):
        assert param.grad is None
        torch.testing.assert_close(param, old, rtol=0, atol=0)
    # Training remains differentiable after the preview, including its external LoRA.
    (model(torch.ones(4, 3)) + lora(torch.ones(4, 3))).square().mean().backward()
    opt.step()
    assert len(opt.state) == 4


def test_cpu_decode_admission_precedes_loading(tmp_path):
    path = tmp_path / 'vae.safetensors'
    save_file({'weight': torch.ones(3, 4, dtype=torch.float16)}, path)
    latent = torch.zeros(1, 64, 32, 32)
    need = preview.admit_cpu_decode(path, 'qwen21', latent, available=100 * 2**30)
    assert need == 2 * 12 * 4 + 900 * 512 * 512 * 4 + 2 * 2**30
    with pytest.raises(ValueError, match='exactly one image latent'):
        preview.admit_cpu_decode(path, 'qwen21', torch.zeros(1, 64, 5, 32, 32), available=100 * 2**30)
    with pytest.raises(MemoryError, match='CPU preview decode needs'):
        preview.admit_cpu_decode(path, 'qwen21', latent, available=need - 1)


def test_actual_comfy_sampler_matches_flow_and_restores_sampling_object(tmp_path):
    # A subprocess selects CPU before importing ComfyUI; it never touches the MPS queue.
    script = r'''
import sys
sys.path[:0] = [sys.argv[1] + '/vendor/ComfyUI', sys.argv[1] + '/train']
sys.argv = [sys.argv[0], '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch
from comfy.model_sampling import ModelSamplingFlux, CONST
from training_preview import sample_preview, preview_state
class Sampling(ModelSamplingFlux, CONST):
    pass
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_sampling = Sampling()
        self.fail = False
    def apply_model(self, x, sigma, c_crossattn, transformer_options):
        assert not torch.is_grad_enabled()
        if self.fail:
            raise RuntimeError('intentional forward failure')
        sampling = self.model_sampling
        velocity = torch.full_like(sampling.calculate_input(sigma, x), 2.)
        return sampling.calculate_denoised(sigma, velocity, x)
    def process_latent_out(self, value):
        return value * 3
m = Model()
original = m.model_sampling
latent = torch.zeros(1, 4, 3, 5)
kwargs = dict(steps=7, seed=71, mu=.54, transformer_options={})
rng = torch.get_rng_state()
with preview_state(m, {}, latent.device):
    result, sigmas = sample_preview(m, latent, None, **kwargs)
noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(71))
torch.testing.assert_close(result, (noise - 2) * 3, atol=2e-6, rtol=2e-6)
assert len(sigmas) == 8 and sigmas[0] == 1 and sigmas[-1] == 0
assert m.model_sampling is original
assert torch.equal(torch.get_rng_state(), rng)
m.fail = True
try:
    with preview_state(m, {}, latent.device):
        sample_preview(m, latent, None, **kwargs)
except RuntimeError as e:
    assert 'intentional' in str(e)
else:
    raise AssertionError('missing failure')
assert m.model_sampling is original and m.training
assert torch.equal(torch.get_rng_state(), rng)
'''
    completed = subprocess.run([sys.executable, '-c', script, str(ROOT)], capture_output=True, text=True, timeout=90,
                               env={**os.environ, 'COMFY_MPS_INT8': '0'})
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_decode_failure_is_recorded_and_modes_restored(tmp_path, monkeypatch):
    args = arguments(out=str(tmp_path), arch='krea2', unet='base.safetensors')
    model = torch.nn.Linear(3, 3)
    monkeypatch.setattr(preview, 'sample_preview', lambda *a, **kw: (torch.zeros(1, 4, 2, 2), [1., 0.]))

    def failed_decode(latent):
        raise MemoryError('not admitted')

    with pytest.raises(MemoryError, match='not admitted'):
        preview.write_preview(args, 4, model, {}, [torch.zeros(1, 4, 2, 2)], [None], ['fixed caption'], [.54], {}, failed_decode)
    record = json.loads((tmp_path / 'previews/step-000004.json').read_text())
    assert record['status'] == 'failed' and record['error'] == 'MemoryError: not admitted'
    assert record['caption'] == 'fixed caption' and model.training
    assert not list(tmp_path.rglob('*.png'))


@pytest.mark.parametrize('fail', [False, True])
def test_cpu_decoder_is_admitted_then_released_even_on_failure(monkeypatch, fail):
    import types
    import weakref

    events, refs = [], []
    comfy = types.ModuleType('comfy')
    comfy.sd = types.ModuleType('comfy.sd')

    class VAE:
        def __init__(self, sd, device, dtype):
            assert events == ['admit', 'weights']
            assert device == torch.device('cpu') and dtype == torch.float32
            self.cycle = self  # exercise collection, as real model/patcher references can cycle
            refs.append(weakref.ref(self))

    comfy.sd.VAE = VAE
    monkeypatch.setitem(sys.modules, 'comfy', comfy)
    monkeypatch.setitem(sys.modules, 'comfy.sd', comfy.sd)
    monkeypatch.setattr(preview, 'admit_cpu_decode', lambda *a: events.append('admit'))

    def load(path):
        events.append('weights')
        return {'weight': torch.ones(1)}

    def decode(vae, latent):
        assert vae.output_device == torch.device('cpu')
        if fail:
            raise RuntimeError('decoder failed')
        return (torch.full((1, 4, 6, 4), .5),)

    boot = SimpleNamespace(
        folder_paths=SimpleNamespace(get_full_path_or_raise=lambda *a: 'unused.safetensors'),
        comfy=SimpleNamespace(utils=SimpleNamespace(load_torch_file=load)),
        N={'VAEDecode': lambda: SimpleNamespace(decode=decode)},
        mm=SimpleNamespace(cleanup_models=lambda: events.append('cleanup')),
    )
    if fail:
        with pytest.raises(RuntimeError, match='decoder failed'):
            preview.decode_preview(boot, {'vae': 'vae'}, 'qwen21', torch.zeros(1, 64, 2, 2))
    else:
        pixels = preview.decode_preview(boot, {'vae': 'vae'}, 'qwen21', torch.zeros(1, 64, 2, 2))
        assert pixels.shape == (4, 6, 3) and np.all(pixels == 128)
    assert events == ['admit', 'weights', 'cleanup']
    # On an exception, pytest's temporary traceback is now out of scope too.
    import gc
    gc.collect()
    assert refs[0]() is None


from test_training_recovery import trainer  # noqa: E402,F401


def test_trainer_saves_recovery_before_preview_and_uses_absolute_resume_cadence(trainer, tmp_path, monkeypatch):
    dataset, out = tmp_path / 'data', tmp_path / 'run'
    dataset.mkdir()
    torch.save({'latents': [torch.ones(1, 3)], 'conds': [torch.ones(1, 3)], 'captions': ['fixed caption']},
               dataset / 'cache_qwen21_512.pt')
    (dataset / 'dataset.json').write_text('{"images": []}')
    base = tmp_path / 'base.safetensors'
    base.write_bytes(b'frozen model identity fixture')
    active = {}

    class CPUDeviceBoundary:
        mps = SimpleNamespace(driver_allocated_memory=lambda: 0, synchronize=lambda: None)

        def device(self, value):
            return torch.device('cpu' if value == 'mps' else value)

        def __getattr__(self, name):
            return getattr(torch, name)

    class Diffusion(torch.nn.Module):
        transformer_blocks = []

        def forward(self, x, *args, **kwargs):
            return active['layer'](x)

    def load_model(*args):
        active['layer'] = trainer.LoRA(3, 3, 2, 2, torch.Generator().manual_seed(3))
        model = SimpleNamespace(diffusion_model=Diffusion(), process_latent_in=lambda x: x,
                                get_dtype_inference=lambda: torch.float32)
        return (SimpleNamespace(model=model, set_model_compute_dtype=lambda dtype: None),)

    trainer.cb.N = {'UNETLoader': lambda: SimpleNamespace(load_unet=load_model)}
    trainer.cb.mm = SimpleNamespace(load_models_gpu=lambda *a, **kw: None)
    trainer.cb.folder_paths = SimpleNamespace(get_full_path_or_raise=lambda *a: str(base))
    monkeypatch.setattr(trainer, 'torch', CPUDeviceBoundary())
    monkeypatch.setattr(trainer, 'patch_arch_for_training', lambda *a: None)
    monkeypatch.setattr(trainer, 'patch_model', lambda *a: (active.copy(), 0))
    monkeypatch.setattr(trainer, 'threading', SimpleNamespace(Thread=lambda **kw: SimpleNamespace(start=lambda: None)))
    previews = []

    def render(args, step, model, loras, latents, conds, captions, mus, transformer_options, decode, **kwargs):
        state = torch.load(out / 'state.pt', map_location='cpu')
        assert state['step'] == step, 'preview must follow a durable checkpoint for this update'
        torch.testing.assert_close(state['lora']['layer'][0], loras['layer'].down, rtol=0, atol=0)
        torch.testing.assert_close(state['lora']['layer'][1], loras['layer'].up, rtol=0, atol=0)
        assert captions == ['fixed caption']
        previews.append(step)
        return {'status': 'ok', 'step': step}

    monkeypatch.setattr(trainer, 'write_preview', render)
    argv = ['train_lora.py', '--arch', 'qwen21', '--dataset', str(dataset), '--out', str(out), '--name', 'preview',
            '--unet', base.name, '--rank', '2', '--steps', '4', '--stop-after', '3', '--preview-every', '2',
            '--state-every', '100', '--save-every', '100', '--no-checkpoint']
    monkeypatch.setattr(sys, 'argv', argv)
    trainer.main()
    assert previews == [0, 2]
    assert torch.load(out / 'state.pt', map_location='cpu')['step'] == 3
    monkeypatch.setattr(sys, 'argv', argv + ['--resume'])
    trainer.main()
    assert previews == [0, 2, 4]
    assert torch.load(out / 'state.pt', map_location='cpu')['step'] == 4


def test_only_admission_refusal_is_skipped(tmp_path, monkeypatch):
    args = arguments(out=str(tmp_path), arch='qwen21', unet='base.safetensors')
    model = torch.nn.Linear(3, 3)
    monkeypatch.setattr(preview, 'sample_preview', lambda *a, **kw: (torch.zeros(1, 4, 2, 2), [1., 0.]))

    def refused(latent):
        raise preview.PreviewNotAdmitted('not enough free RAM')

    record = preview.write_preview(args, 0, model, {}, [torch.zeros(1, 4, 2, 2)], [None], ['caption'], [.54], {}, refused)
    assert record['status'] == 'skipped' and 'not enough free RAM' in record['error']
    assert model.training
    assert json.loads((tmp_path / 'previews/step-000000.json').read_text())['status'] == 'skipped'
    # Ordinary MemoryError remains a failed run, covered by decode_failure above.


def test_preview_reuse_requires_exact_weights_model_and_fixed_config(tmp_path, monkeypatch):
    args = arguments(out=str(tmp_path), arch='qwen21', unet='base.safetensors')
    model = torch.nn.Linear(3, 3)
    lora = torch.nn.Linear(3, 3)
    calls = []

    def sample(*a, **kw):
        calls.append(kw['seed'])
        return torch.zeros(1, 4, 2, 2), [1., 0.]

    monkeypatch.setattr(preview, 'sample_preview', sample)
    model_identity = {'base': 'actual base bytes hash', 'cache': 'actual cache bytes hash'}

    def render(step=2, **kwargs):
        return preview.write_preview(args, step, model, {'layer': lora}, [torch.zeros(1, 4, 2, 2)], [None],
                                     ['caption'], [.54], {}, lambda _: np.zeros((16, 16, 3), dtype='uint8'),
                                     model_identity=kwargs.get('identity', model_identity))

    assert render()['status'] == 'ok'
    assert render()['reused'] is True and len(calls) == 1
    with torch.no_grad():
        lora.weight[0, 0] += .1
    assert 'reused' not in render() and len(calls) == 2
    args.preview_seed += 1
    assert 'reused' not in render() and len(calls) == 3
    assert 'reused' not in render(identity={'base': 'different'}) and len(calls) == 4
    assert 'reused' not in render(step=3) and len(calls) == 5


def test_disabling_preview_does_not_change_resume_identity(trainer, tmp_path, monkeypatch):
    dataset = tmp_path / 'data'
    dataset.mkdir()
    (dataset / 'dataset.json').write_text('{"images": []}')
    base, cache = tmp_path / 'base.safetensors', dataset / 'cache.pt'
    base.write_bytes(b'base identity')
    cache.write_bytes(b'cache identity')
    trainer.cb.folder_paths = SimpleNamespace(get_full_path_or_raise=lambda *a: str(base))
    argv = ['train_lora.py', '--dataset', str(dataset), '--out', str(tmp_path / 'run'), '--preview-every', '2']
    monkeypatch.setattr(sys, 'argv', argv)
    enabled = trainer.parse()
    original = trainer.training_identity(enabled, trainer.ARCHS[enabled.arch], cache)
    monkeypatch.setattr(sys, 'argv', argv + ['--preview-every', '0'])
    disabled = trainer.parse()
    identity = trainer.training_identity(disabled, trainer.ARCHS[disabled.arch], cache)
    assert identity == original
    assert trainer.validate_resume_identity({'identity': original}, identity, disabled) is False
