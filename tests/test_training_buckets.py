"""Bucket preprocessing and per-sample shift through real trainer entry points on CPU."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from test_training_recovery import trainer  # noqa: F401


def test_build_cache_records_actual_image_and_latent_geometry(trainer, tmp_path, monkeypatch):
    for name, size, color in [('portrait.png', (40, 90), 'red'), ('landscape.png', (100, 40), 'blue')]:
        Image.new('RGB', size, color).save(tmp_path / name)
    (tmp_path / 'dataset.json').write_text(json.dumps({'images': [
        {'file': 'portrait.png', 'caption': 'portrait'}, {'file': 'landscape.png', 'caption': 'landscape'},
    ]}))
    encoded = []

    def encode(vae, pixels):
        encoded.append(tuple(pixels.shape))
        return ({'samples': torch.ones(1, 4, pixels.shape[1] // 8, pixels.shape[2] // 8)},)

    trainer.cb.N = {'VAELoader': lambda: SimpleNamespace(load_vae=lambda _: (object(),)),
                    'VAEEncode': lambda: SimpleNamespace(encode=encode),
                    'CLIPLoader': lambda: SimpleNamespace(load_clip=lambda *a: (object(),))}
    trainer.cb.mm = SimpleNamespace(unload_all_models=lambda: None, soft_empty_cache=lambda: None)
    monkeypatch.setattr(trainer, 'encode_text', lambda *a: torch.ones(1, 3, 16))
    args = SimpleNamespace(arch='qwen21', dataset=str(tmp_path), res=512, buckets=[[32, 64], [64, 32]])
    path = trainer.cache_filename(tmp_path, args.arch, args.res, args.buckets)
    got = trainer.build_cache(args, trainer.ARCHS[args.arch], path)
    assert encoded == [(1, 64, 32, 3), (1, 32, 64, 3)]
    assert [sample['target_size'] for sample in got['samples']] == [[32, 64], [64, 32]]
    assert [sample['file'] for sample in got['samples']] == ['portrait.png', 'landscape.png']
    saved = torch.load(path, map_location='cpu')
    trainer.validate_bucket_cache(saved, trainer.shape_policy(args.res, args.buckets))
    assert saved['samples'] == got['samples']


@pytest.mark.parametrize('bucket_mode', [False, True])
def test_main_samples_mu_for_selected_latent_without_changing_square_recipe(trainer, tmp_path, monkeypatch, bucket_mode):
    buckets = [[64, 32], [64, 64]] if bucket_mode else None
    cache = {'latents': [torch.ones(1, 4, 4, 8), torch.ones(1, 4, 8, 8)],
             'conds': [torch.ones(1, 3, 16), torch.ones(1, 5, 16)]}
    if bucket_mode:
        cache.update(shape_policy=trainer.shape_policy(512, buckets), samples=[
            {'target_size': [64, 32], 'latent_shape': [1, 4, 4, 8]},
            {'target_size': [64, 64], 'latent_shape': [1, 4, 8, 8]},
        ])
    torch.save(cache, trainer.cache_filename(tmp_path, 'qwen21', 512, buckets))
    predictions, sampled_mus = [], []

    class CPUDeviceBoundary:
        mps = SimpleNamespace(driver_allocated_memory=lambda: 0, synchronize=lambda: None)

        def device(self, value):
            return torch.device('cpu' if value == 'mps' else value)

        def __getattr__(self, name):
            return getattr(torch, name)

    class Diffusion(torch.nn.Module):
        transformer_blocks = []

        def forward(self, x, sigma, context, **kwargs):
            predictions.append((tuple(x.shape), tuple(context.shape)))
            return torch.zeros_like(x)

    model = SimpleNamespace(diffusion_model=Diffusion(), process_latent_in=lambda x: x,
                            get_dtype_inference=lambda: torch.float32)
    patcher = SimpleNamespace(model=model, set_model_compute_dtype=lambda dtype: None)
    trainer.cb.N = {'UNETLoader': lambda: SimpleNamespace(load_unet=lambda *a: (patcher,))}
    trainer.cb.mm = SimpleNamespace(load_models_gpu=lambda *a, **kw: None)
    monkeypatch.setattr(trainer, 'torch', CPUDeviceBoundary())
    monkeypatch.setattr(trainer, 'patch_arch_for_training', lambda arch: None)
    lora = trainer.LoRA(4, 4, 2, 2, torch.Generator().manual_seed(8))
    monkeypatch.setattr(trainer, 'patch_model', lambda *a: ({'layer': lora}, 0))
    monkeypatch.setattr(trainer, 'threading', SimpleNamespace(Thread=lambda **kw: SimpleNamespace(start=lambda: None)))
    monkeypatch.setattr(trainer, 'optimizer_step', lambda loss, *a: (loss.item(), 0.))
    sample = trainer.sample_sigma

    def sample_sigma(n, mu, gen):
        sampled_mus.append(mu)
        return sample(n, mu, gen)

    monkeypatch.setattr(trainer, 'sample_sigma', sample_sigma)
    argv = ['train_lora.py', '--arch', 'qwen21', '--dataset', str(tmp_path), '--out', str(tmp_path / 'run'),
            '--profile-steps', '8', '--no-checkpoint', '--seed', '4']
    if buckets:
        argv += ['--buckets', '64x32,64x64']
    monkeypatch.setattr(sys, 'argv', argv)
    trainer.main()
    updates = predictions[-8:]
    assert len({shape for shape, _ in updates}) == 2
    for (shape, conditioning), actual_mu in zip(updates, sampled_mus, strict=True):
        latent = torch.empty(shape) if bucket_mode else cache['latents'][0]
        assert actual_mu == trainer.shift_mu('qwen21', model, latent)
        assert conditioning[1] == (3 if shape[-2] == 4 else 5)


def test_real_cli_can_override_all_file_boolean_and_bucket_defaults(trainer, tmp_path, monkeypatch):
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps({'buckets': [[64, 32]], 'no_checkpoint': True,
                               'chunked_attn': True, 'resume': True, 'out': 'new-run'}))
    monkeypatch.setattr(sys, 'argv', ['train_lora.py', '--config', str(path), '--no-buckets',
                                     '--checkpoint', '--no-chunked-attn', '--no-resume'])
    args = trainer.parse()
    assert args.buckets is None and not args.no_checkpoint and not args.chunked_attn and not args.resume
    assert args.out == str(tmp_path / 'new-run')


@pytest.mark.parametrize('buckets', [None, [[64, 32], [64, 64]]])
def test_adapter_export_records_actual_shape_policy(trainer, tmp_path, buckets):
    from safetensors import safe_open
    lora = trainer.LoRA(4, 4, 2, 2, torch.Generator().manual_seed(8))
    args = SimpleNamespace(profile_steps=0, save_every=2, steps=2, out=str(tmp_path), name='adapter',
                           arch='qwen21', unet='base', rank=2, res=512, lr=.0002, buckets=buckets)
    trainer.export_lora_if_due(args, 2, {'layer': lora}, 0, torch.bfloat16)
    with safe_open(tmp_path / 'adapter_s2.safetensors', framework='pt') as artifact:
        metadata = artifact.metadata()
    if buckets is None:
        assert metadata['res'] == '512' and 'shape_policy' not in metadata
    else:
        assert 'res' not in metadata
        assert json.loads(metadata['shape_policy']) == trainer.shape_policy(512, buckets)
