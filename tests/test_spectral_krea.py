"""Real Euler/KSAMPLER contracts with a controlled pretrained-network boundary."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.fft import dctn, idctn
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/ComfyUI'))
from comfy import model_base, model_patcher, samplers, supported_models
from bench import spectral_krea as spectral

pytestmark = pytest.mark.skipif(not spectral.SOURCE.is_file(), reason='Pinned LC helper must be prepared')


class NetworkBoundary(samplers.CFGGuider):
    """Use real guider state and scaling; replace only the trained-network prediction."""
    def __init__(self, patcher):
        super().__init__(patcher)
        self.inner_model = patcher.model
        self.calls = []

    def predict_noise(self, x, sigma, model_options={}, seed=None):
        self.calls.append((x.detach().cpu().clone(), float(sigma[0])))
        return torch.zeros_like(x)


def context():
    config = supported_models.Krea2({'image_model': 'krea2', 'disable_unet_model_creation': True})
    # The framework's explicit no-network option avoids creating pretrained weights.
    base = model_base.Krea2(config, device=torch.device('cpu'))
    patcher = model_patcher.ModelPatcher(base, torch.device('cpu'), torch.device('cpu'))
    return patcher, NetworkBoundary(patcher)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_one_size_transition_eight_calls_and_single_noise_scaling(device, monkeypatch):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Apple GPU required')
    dtype = torch.float32
    patcher, model = context()
    inner = samplers.sampler_object('euler')
    events, callbacks = [], []
    wrapped = spectral.wrap_euler(inner, patcher, [], [], enabled=True, report=events.append)
    noise = torch.randn(1, 16, 16, 24, generator=torch.Generator().manual_seed(19)).to(device, dtype)
    before = noise.clone()
    # CFGGuider.outer_sample explicitly prepares FP32 noise and device-local sigmas.
    sigmas = torch.linspace(0.95, 0, 9, device=device)
    scaling = patcher.model.model_sampling
    counts = {'initial': 0, 'inverse': 0}
    for name, label in [('noise_scaling', 'initial'), ('inverse_noise_scaling', 'inverse')]:
        original = getattr(scaling, name)
        def counted(*args, _original=original, _label=label, **kwargs):
            counts[_label] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(scaling, name, counted)
    rng_before = torch.get_rng_state().clone()

    output = wrapped.sample(model, sigmas, {'seed': 7}, lambda i, x0, x, total: callbacks.append((i, total)),
                            noise, torch.zeros_like(noise), disable_pbar=True)

    assert len(model.calls) == 8 and callbacks == [(i, 8) for i in range(8)]
    assert counts == {'initial': 1, 'inverse': 1}
    assert output.shape == noise.shape and output.dtype == dtype and torch.isfinite(output).all()
    assert torch.equal(noise, before) and torch.equal(torch.get_rng_state(), rng_before)
    assert [tuple(x.shape[-2:]) for x, _ in model.calls] == [(8, 12)] * 4 + [(16, 24)] * 4
    event, = events
    assert event['expansion_seed'] == 14 and event['sigmas'] == sigmas.tolist()
    assert [t for _, t in model.calls] == pytest.approx([*sigmas[:4].tolist(), *event['remaining_sigmas'][:-1]], abs=1e-7)

    # Independent SciPy DCT oracle: with x0=0, Euler preserves x(t)=t*noise.
    spectrum = dctn(noise.float().cpu().numpy(), axes=(-2, -1), norm='ortho')
    coarse = idctn(spectrum[..., :8, :12], axes=(-2, -1), norm='ortho') * float(sigmas[0])
    fresh = torch.randn(noise.shape, generator=torch.Generator().manual_seed(14)).numpy()
    fresh[..., :8, :12] = spectrum[..., :8, :12]
    fine = idctn(fresh, axes=(-2, -1), norm='ortho') * event['resumed_sigma']
    tolerance = 2e-5
    np.testing.assert_allclose(model.calls[0][0].float().numpy(), coarse, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(model.calls[4][0].float().numpy(), fine, rtol=tolerance, atol=tolerance)


def test_disabled_route_is_the_original_sampler_and_output():
    patcher, model = context()
    sampler = samplers.sampler_object('euler')
    disabled = spectral.wrap_euler(sampler, patcher, [], [], enabled=False)
    assert disabled is sampler
    noise = torch.randn(1, 16, 16, 24, generator=torch.Generator().manual_seed(20))
    args = (model, torch.linspace(1, 0, 9), {'seed': 7}, None, noise, torch.zeros_like(noise))
    actual = disabled.sample(*args, disable_pbar=True)
    expected = sampler.sample(*args, disable_pbar=True)
    assert torch.equal(actual, expected)


def test_shrink_preserves_white_noise_covariance_without_rescaling():
    basis = torch.eye(16 * 24).reshape(16 * 24, 1, 16, 24)
    shrunk = spectral.helper().shrink(basis, 8, 12).reshape(16 * 24, 8 * 12)
    torch.testing.assert_close(shrunk.T @ shrunk, torch.eye(8 * 12), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('kind', ['mask', 'schedule', 'nonzero_latent', 'cfg', 'half_latent', 'video',
                                 'clone', 'execution_wrapper', 'execution_schedule'])
def test_unsupported_run_refused_before_model_calls(kind):
    patcher, model = context()
    sampler = spectral.wrap_euler(samplers.sampler_object('euler'), patcher, [], [], enabled=True)
    noise = torch.zeros(1, 16, 16, 24, dtype=torch.float16 if kind == 'half_latent' else torch.float32)
    if kind == 'video':
        noise = noise.unsqueeze(2).expand(-1, -1, 2, -1, -1).clone()
    latent = torch.ones_like(noise) if kind == 'nonzero_latent' else noise.clone()
    sigmas = torch.linspace(1, 0, 8 if kind == 'schedule' else 9)
    if kind == 'cfg':
        model.set_cfg(2)
    if kind == 'clone':
        model.model_patcher = patcher.clone()
    extra = {'seed': 7}
    if kind in ('execution_wrapper', 'execution_schedule'):
        extra['model_options'] = {'transformer_options': {'sample_sigmas': sigmas}}
        if kind == 'execution_wrapper':
            extra['model_options']['transformer_options']['wrappers'] = {'foreign': [lambda *a: None]}
        else:
            extra['model_options']['transformer_options']['sample_sigmas'] = sigmas.clone()
    with pytest.raises(ValueError):
        sampler.sample(model, sigmas, extra, None, noise, latent,
                       denoise_mask=torch.ones_like(noise) if kind == 'mask' else None, disable_pbar=True)
    assert model.calls == []


def test_control_conditioning_refused_at_construction():
    patcher, _ = context()
    with pytest.raises(ValueError, match='control'):
        spectral.wrap_euler(samplers.sampler_object('euler'), patcher,
                            [[torch.zeros(1, 2, 4), {'control': object()}]], [], enabled=True)


@pytest.mark.parametrize('key', ['reference_latents', 'reference_latents_method', 'hooks', 'area'])
def test_unowned_conditioning_remains_refused(key):
    patcher, _ = context()
    with pytest.raises(ValueError, match='excludes'):
        spectral.wrap_euler(samplers.sampler_object('euler'), patcher,
                            [[torch.zeros(1, 2, 4), {key: object()}]], [], enabled=True)


def test_plain_spectral_does_not_admit_arbitrary_model_wrapper():
    patcher, _ = context()
    patcher.model_options['transformer_options']['wrappers'] = {'foreign': [lambda *a: None]}
    with pytest.raises(ValueError, match='unowned model options'):
        spectral.wrap_euler(samplers.sampler_object('euler'), patcher, [], [], enabled=True)


def test_conditioning_drift_refused_before_prediction():
    patcher, model = context()
    conditioning = [[torch.zeros(1,2,4), {}]]
    wrapped = spectral.wrap_euler(samplers.sampler_object('euler'), patcher,
                                  conditioning, [], enabled=True)
    conditioning[0][1]['mask'] = torch.ones(1,16,24)
    noise = torch.zeros(1,16,16,24)
    with pytest.raises(ValueError, match='excludes'):
        wrapped.sample(model,torch.linspace(1,0,9),{'seed':7},None,noise,noise,
                       disable_pbar=True)
    assert model.calls == []


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('network', ['oracle', 'real'])
def test_real_node_preparation_admits_expanded_latent_and_owned_patcher(device, network, monkeypatch):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Apple GPU required')
    import nodes
    import comfy.model_management as mm
    config = supported_models.Krea2({
        'image_model': 'krea2', 'features': 64, 'tdim': 16, 'txtdim': 64,
        'heads': 4, 'kvheads': 4, 'layers': 1, 'txtlayers': 1,
        'txtheads': 4, 'txtkvheads': 4, 'multiplier': 1, 'dtype': torch.float32,
    })
    base = model_base.Krea2(config, device=torch.device('cpu'))
    if network == 'real':
        generator = torch.Generator().manual_seed(991)
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * .01)
    patcher = model_patcher.ModelPatcher(base, torch.device(device), torch.device('cpu'))
    conditioning = [[torch.zeros(1, 2, 64), {}]]
    sampler = spectral.wrap_euler(samplers.sampler_object('euler'), patcher,
                                  conditioning, conditioning, enabled=True)
    calls = []
    original_predict = samplers.CFGGuider.predict_noise
    def network_boundary(self, x, sigma, model_options={}, seed=None):
        calls.append((self.model_patcher.model is base, x.shape, x.dtype, x.device.type))
        return (original_predict(self, x, sigma, model_options=model_options, seed=seed)
                if network == 'real' else torch.zeros_like(x))
    monkeypatch.setattr(samplers.CFGGuider, 'predict_noise', network_boundary)
    monkeypatch.setattr(samplers, 'sampler_object', lambda name: sampler)
    try:
        # The real node expands EmptyLatentImage's four channels to Krea's sixteen,
        # then real guider preparation loads the patcher and moves FP32 state/sigmas.
        result = nodes.common_ksampler(patcher, 7, 8, 1, 'euler', 'simple',
                                       conditioning, conditioning,
                                       {'samples': torch.zeros(1, 4, 16, 24)})[0]['samples']
    finally:
        mm.unload_all_models()
    assert result.shape == (1, 16, 1, 16, 24) and torch.isfinite(result).all()
    assert len(calls) == 8 and all(same and dtype == torch.float32 and dev == device for same, _, dtype, dev in calls)
    assert [shape[-2:] for _, shape, _, _ in calls] == [(8, 12)] * 4 + [(16, 24)] * 4


@pytest.mark.parametrize('license_text', [None, b'changed'])
def test_existing_helper_requires_pinned_license(tmp_path, monkeypatch, license_text):
    source = tmp_path / 'lc_speed_boost.py'
    source.write_bytes(spectral.SOURCE.read_bytes())
    if license_text is not None:
        source.with_name('LICENSE').write_bytes(license_text)
    monkeypatch.setattr(spectral, 'SOURCE', source)
    monkeypatch.setattr(spectral, '_HELPER', None)
    with pytest.raises(RuntimeError, match='license'):
        spectral.prepare()
    with pytest.raises(RuntimeError, match='license'):
        spectral.helper()


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_grow_rebalances_nonzero_picture_and_white_noise(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Apple GPU required')
    patcher, model = context()
    # A band-limited nonzero image has twice the DCT amplitude at full size.
    # Construct its small picture independently of the helper's grow algorithm.
    row = np.arange(16)[:, None]
    col = np.arange(24)[None, :]
    picture = (0.4 + 0.2 * np.cos(np.pi * (row + .5) / 16)
               + 0.1 * np.cos(np.pi * (col + .5) / 24))
    picture = np.broadcast_to(picture, (1, 16, 16, 24)).copy()
    coarse_picture = idctn(dctn(picture, axes=(-2, -1), norm='ortho')[..., :8, :12],
                          axes=(-2, -1), norm='ortho') / 2
    predictions = {(16, 24): torch.tensor(picture, dtype=torch.float32, device=device),
                   (8, 12): torch.tensor(coarse_picture, dtype=torch.float32, device=device)}
    def fixed_picture(x, sigma, model_options={}, seed=None):
        model.calls.append((x.detach().cpu().clone(), float(sigma[0])))
        return predictions[tuple(x.shape[-2:])]
    model.predict_noise = fixed_picture
    noise = torch.randn(1, 16, 16, 24, generator=torch.Generator().manual_seed(51)).to(device)
    wrapped = spectral.wrap_euler(samplers.sampler_object('euler'), patcher, [], [], enabled=True)
    output = wrapped.sample(model, torch.linspace(1, 0, 9, device=device), {'seed': 7}, None,
                            noise, torch.zeros_like(noise), disable_pbar=True)

    # Unit-start Euler with fixed x0 follows x(t)=(1-t)*picture+t*white_noise.
    # The full-size grow state must obey that mixture at its resumed sigma.
    fresh = torch.randn(noise.shape, generator=torch.Generator().manual_seed(14)).numpy()
    fresh[..., :8, :12] = dctn(noise.cpu().numpy(), axes=(-2, -1), norm='ortho')[..., :8, :12]
    white_noise = idctn(fresh, axes=(-2, -1), norm='ortho')
    resumed_sigma = model.calls[4][1]
    assert resumed_sigma == pytest.approx(2 / 3, abs=1e-7)
    expected = (1 - resumed_sigma) * picture + resumed_sigma * white_noise
    np.testing.assert_allclose(model.calls[4][0].numpy(), expected, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(output.cpu().numpy(), picture, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize('mode,count', [('style_reference', 1), ('identity_edit', 1), ('identity_edit', 2)])
def test_owned_references_cross_both_sizes_through_real_krea(mode, count):
    """Real node/guider/network and pinned author wrapper; only VAE is isolated."""
    import os
    import subprocess
    if mode == 'identity_edit' and not os.environ.get('KREA_EDIT_RUNTIME'):
        pytest.skip('Pinned external Krea2Edit runtime is not configured')
    script = r'''
import sys
from pathlib import Path
mode, count = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, str(Path.cwd() / 'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import torch._dynamo
import nodes
import comfy.model_management as mm
from comfy import model_base, model_patcher, supported_models, samplers
from krea_reference import prepare_reference, apply_reference
from spectral_krea import wrap_euler
torch.set_num_threads(2)
class VAE:
    def __init__(self): self.calls = 0
    def encode(self, pixels):
        self.calls += 1
        channels = torch.nn.functional.avg_pool2d(pixels.movedim(-1,1), 8)
        return channels.repeat(1,6,1,1)[:,:16].unsqueeze(2)
vae = VAE()
g = torch.Generator().manual_seed(413)
images = [torch.rand(1,512,256,3,generator=g), torch.rand(1,256,512,3,generator=g)]
prepared = prepare_reference(mode, images[0], images[1] if count == 2 else None,
    vae=vae, torch=torch, spectral=True)
vae_calls = vae.calls
config = supported_models.Krea2({'image_model':'krea2', 'features':64, 'tdim':16,
    'txtdim':64,'heads':4,'kvheads':4,'layers':1,'txtlayers':1,'txtheads':4,
    'txtkvheads':4,'multiplier':1,'dtype':torch.float32})
base = model_base.Krea2(config,device=torch.device('cpu'))
with torch.no_grad():
    for parameter in base.parameters():
        parameter.copy_(torch.randn(parameter.shape,generator=g)*.01)
patcher = model_patcher.ModelPatcher(base,torch.device('cpu'),torch.device('cpu'))
conditioning = [[torch.randn(1,2,64,generator=g),{}]]
candidate,pos,neg = apply_reference(patcher,conditioning,conditioning,prepared)
positions = []
handle = base.diffusion_model.pe_embedder.register_forward_pre_hook(
    lambda module,args: positions.append(args[0].detach().clone()))
wrapped = wrap_euler(samplers.sampler_object('euler'),candidate,pos,neg,
    enabled=True,prepared_reference=prepared)
original = samplers.sampler_object
samplers.sampler_object = lambda name: wrapped
try:
    output = nodes.common_ksampler(candidate,7,8,1,'euler','simple',pos,neg,
        {'samples':torch.zeros(1,4,64,64)})[0]['samples']
finally:
    samplers.sampler_object = original
    handle.remove()
    mm.unload_all_models()
assert output.shape == (1,16,1,64,64) and torch.isfinite(output).all()
assert len(positions) == 8 and vae.calls == vae_calls
for i,position in enumerate(positions):
    frame = position[0,:,0]
    target_tokens = 256 if i < 4 else 1024
    assert int((frame == 0).sum()) == target_tokens + 2
    ref_tokens = 512 if mode == 'style_reference' or i >= 4 else 128
    assert int((frame == 1).sum()) == ref_tokens
    if count == 2:
        assert int((frame == 2).sum()) == ref_tokens
    if mode == 'identity_edit':
        assert torch.all(frame[2:2+ref_tokens] == 1)
        assert torch.all(frame[-target_tokens:] == 0)
    else:
        assert torch.all(frame[2:2+target_tokens] == 0)
        assert torch.all(frame[-ref_tokens:] == 1)
# No VAE object retained by the prepared replay; ordinary model remains untouched.
assert patcher.model_options == {'transformer_options':{}}
# Replacing the copied execution wrapper cannot bypass the patcher's admission.
if mode == 'identity_edit':
    from comfy.patcher_extension import WrappersMP
    guider = samplers.CFGGuider(candidate)
    guider.inner_model = base
    sigmas = torch.linspace(1,0,9)
    execution_options = model_patcher.create_model_options_clone(candidate.model_options)
    execution_options['transformer_options']['sample_sigmas'] = sigmas
    execution_options['transformer_options']['wrappers'][WrappersMP.DIFFUSION_MODEL]['krea2_edit'][0] = lambda *a: None
    noise = torch.zeros(1,16,1,64,64)
    try:
        wrapped.sample(guider,sigmas,{'seed':7,'model_options':execution_options},None,
                       noise,noise,disable_pbar=True)
    except ValueError as error:
        assert 'execution options' in str(error)
    else:
        raise AssertionError('execution-only wrapper replacement accepted')
print('Owned reference real Krea stage geometry passed')
'''
    result = subprocess.run([sys.executable, '-c', script, mode, str(count)], cwd=ROOT,
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
