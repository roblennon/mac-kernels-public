"""Opt-in single-image previews using the live trainer and ComfyUI's Euler sampler."""
from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
from PIL import Image
import torch


class PreviewNotAdmitted(MemoryError):
    """Optional CPU decode was refused before allocating its VAE."""


def add_preview_arguments(parser):
    parser.add_argument('--preview-every', type=int, default=0, help='preview interval; 0 disables previews')
    parser.add_argument('--preview-steps', type=int, default=28, help='Euler steps for the raw training base')
    parser.add_argument('--preview-seed', type=int, default=7)
    parser.add_argument('--preview-index', type=int, default=0, help='fixed cached dataset caption and shape')


def validate_preview_arguments(args, parser):
    if args.preview_every < 0 or args.preview_steps <= 0 or args.preview_index < 0:
        parser.error('preview interval/index must be nonnegative and preview steps must be positive')


def preview_due(args, step):
    return (not args.profile_steps and args.preview_every > 0 and
            (step % args.preview_every == 0 or step == args.steps))


@contextmanager
def preview_state(model, loras, device):
    """Restore RNG and mixed train/eval flags even if sampling or decoding fails."""
    modules = list(dict.fromkeys(module for root in [model, *loras.values()] for module in root.modules()))
    modes = [(module, module.training) for module in modules]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    mps_rng = torch.mps.get_rng_state() if device.type == 'mps' else None
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            for module in modules:
                module.training = False
            yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)


def sample_preview(model, latent, cond, *, steps, seed, mu, transformer_options):
    """Use the normal inference schedule, input conversion and denoised prediction."""
    from comfy.k_diffusion.sampling import sample_euler
    from comfy.samplers import calculate_sigmas

    original_sampling = model.model_sampling
    sampling = copy.deepcopy(original_sampling)
    sampling.set_parameters(shift=mu)
    sigmas = calculate_sigmas(sampling, 'simple', steps).to(latent.device)
    noise = torch.randn(latent.shape, generator=torch.Generator(device='cpu').manual_seed(seed))
    noise = noise.to(device=latent.device, dtype=torch.float32)
    x = sampling.noise_scaling(sigmas[0], noise, torch.zeros_like(noise), max_denoise=True)
    try:
        model.model_sampling = sampling

        def denoise(x, sigma):
            return model.apply_model(x, sigma, c_crossattn=cond, transformer_options=dict(transformer_options))

        result = sample_euler(denoise, x, sigmas, disable=True)
        if not torch.isfinite(result).all():
            raise FloatingPointError('preview sampler returned nonfinite latents')
        return model.process_latent_out(result).detach().float().cpu(), sigmas.cpu().tolist()
    finally:
        model.model_sampling = original_sampling


def admit_cpu_decode(path, arch, latent, *, available=None):
    """Reserve weights, a conservative full decode estimate, and 2 GiB headroom."""
    import psutil
    from safetensors import safe_open

    if latent.ndim not in (4, 5) or latent.shape[0] != 1 or (latent.ndim == 5 and latent.shape[2] != 1):
        raise ValueError('training preview decode requires exactly one image latent')
    with safe_open(str(path), framework='pt', device='cpu') as source:
        fp32_bytes = sum(4 * np.prod(source.get_slice(key).get_shape(), dtype=np.int64) for key in source.keys())
    compression, factor = (16, 900) if arch == 'qwen21' else (8, 2200)
    pixels = latent.shape[-2] * latent.shape[-1] * compression ** 2
    # Two copies allow state-dict conversion while the fp32 decoder is constructed.
    required = int(2 * fp32_bytes + factor * pixels * 4 + 2 * 2**30)
    available = psutil.virtual_memory().available if available is None else available
    if required > available:
        raise PreviewNotAdmitted(f'CPU preview decode needs {required / 2**30:.2f} GiB free including reserve; '
                          f'{available / 2**30:.2f} GiB available')
    return required


def decode_preview(cb, spec, arch, latent):
    """A CPU-only VAE cannot evict the MPS trainer through model admission."""
    import gc
    import comfy.sd

    path = cb.folder_paths.get_full_path_or_raise('vae', spec['vae'])
    admit_cpu_decode(path, arch, latent)
    vae = None
    try:
        state = cb.comfy.utils.load_torch_file(path)
        vae = comfy.sd.VAE(sd=state, device=torch.device('cpu'), dtype=torch.float32)
        del state
        vae.output_device = torch.device('cpu')
        pixels = cb.N['VAEDecode']().decode(vae, {'samples': latent})[0]
        if not torch.isfinite(pixels).all():
            raise FloatingPointError('preview decoder returned nonfinite pixels')
        if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[-1] not in (3, 4):
            raise ValueError(f'expected one decoded image, received {tuple(pixels.shape)}')
        return (pixels[0, ..., :3].float().numpy().clip(0, 1) * 255).round().astype('uint8')
    finally:
        del vae
        gc.collect()
        cb.mm.cleanup_models()


def _atomic_output(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def lora_fingerprint(loras):
    """Image identity uses exact live adapter bytes, including a resumed adapter."""
    digest = hashlib.sha256()
    for name, lora in sorted(loras.items()):
        digest.update(json.dumps([name, getattr(lora, 'scale', None)]).encode())
        for key, parameter in sorted(lora.named_parameters()):
            value = parameter.detach().cpu().contiguous()
            digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_preview(args, step, model, loras, latents, conds, captions, mus, transformer_options, decode,
                  *, model_identity=None):
    """One fixed caption, shape and seed. A JSON record distinguishes failures from images."""
    index = args.preview_index
    if index >= len(latents):
        raise ValueError(f'preview index {index} is outside the {len(latents)} cached samples')
    stem = Path(args.out) / 'previews' / f'step-{step:06d}'
    record = {'step': step, 'index': index, 'caption': captions[index], 'seed': args.preview_seed,
              'steps': args.preview_steps, 'cfg': 1.0, 'sampler': 'euler', 'scheduler': 'simple',
              'shift_mu': mus[index], 'shape': list(latents[index].shape), 'arch': args.arch, 'unet': args.unet,
              'model_identity': model_identity, 'lora_sha256': lora_fingerprint(loras)}
    # Refuse to infer a model identity for callers that have not established one.
    if model_identity is not None and stem.with_suffix('.png').is_file():
        try:
            previous = json.loads(stem.with_suffix('.json').read_text())
            if previous.get('status') == 'ok' and all(previous.get(key) == value for key, value in record.items()):
                return dict(previous, reused=True)
        except (OSError, ValueError):
            pass
    try:
        with preview_state(model, loras, latents[index].device):
            latent, sigmas = sample_preview(model, latents[index], conds[index], steps=args.preview_steps,
                                           seed=args.preview_seed, mu=mus[index], transformer_options=transformer_options)
            record['sigmas'] = sigmas
            pixels = decode(latent)
            _atomic_output(stem.with_suffix('.png'), lambda path: Image.fromarray(pixels).save(path, format='PNG'))
        record.update(status='ok', image=str(stem.with_suffix('.png')))
    except PreviewNotAdmitted as error:
        record.update(status='skipped', error=f'{type(error).__name__}: {error}')
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        _atomic_output(stem.with_suffix('.json'), lambda path: Path(path).write_text(json.dumps(record, indent=2) + '\n'))
    return record
