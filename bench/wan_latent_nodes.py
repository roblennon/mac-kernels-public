"""Saved Wan samples and draft/final decode, delegating to stock Comfy nodes."""
import hashlib
import os
from pathlib import Path

from .wan_latents import load_saved_latent

EMPTY = '(no saved WAN latents)'
QUALITY = ('Draft (tiny)', 'Final (full)')
DECODERS = {48: ('lighttaew2_2.safetensors', 'wan2.2_vae.safetensors'),
            16: ('lighttaew2_1.safetensors', 'wan_2.1_vae.safetensors')}


def saved_files():
    import folder_paths
    roots = {'Comfy': Path(folder_paths.get_output_directory()) / 'latents' / 'wan',
             'Headless': Path(os.environ.get('MK_ROOT', os.path.expanduser('~/mac-kernels-data'))) / 'out' / 'latents'}
    result = {}
    for label, root in roots.items():
        for path in root.rglob('*.latent' if label == 'Comfy' else 'wan*.pt'):
            if path.is_file() and path.resolve().is_relative_to(root.resolve()):
                result[f'{label}/{path.relative_to(root).as_posix()}'] = path
    return result


def saved_path(name):
    try:
        return saved_files()[name]
    except KeyError as error:
        raise ValueError('Select a saved WAN latent; refresh the model lists after saving a draft') from error


class BenchWanSavedLatent:
    CATEGORY = 'bench/video'
    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'load'
    DESCRIPTION = 'Open a saved WAN sample. Draft/final decoding does not run the sampler.'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'latent': (sorted(saved_files()) or [EMPTY],)}}

    @classmethod
    def IS_CHANGED(cls, latent):
        with saved_path(latent).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    def load(self, latent):
        samples, _, _ = load_saved_latent(saved_path(latent))
        return (samples,)


class BenchWanDecode:
    CATEGORY = 'bench/video'
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'decode'
    DESCRIPTION = 'Decode the same WAN sample as a quick tiny-VAE draft or full-quality final. Only the selected decoder loads.'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'samples': ('LATENT',), 'quality': (QUALITY,)},
                'optional': {'vae': ('VAE',)}}

    def decode(self, samples, quality, vae=None):
        import torch
        import nodes
        from comfy.model_management import throw_exception_if_processing_interrupted
        tensor = samples.get('samples')
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 5
                or not tensor.is_floating_point() or min(tensor.shape) < 1
                or tensor.shape[1] not in DECODERS or not bool(torch.isfinite(tensor).all())):
            raise ValueError('Expected a floating WAN video latent with 16 or 48 channels')
        if quality not in QUALITY:
            raise ValueError('Select Draft (tiny) or Final (full)')
        name = DECODERS[tensor.shape[1]][QUALITY.index(quality)]
        decoder = vae if quality == QUALITY[1] else None
        if decoder is None:
            try:
                decoder = nodes.VAELoader().load_vae(name)[0]
            except FileNotFoundError as error:
                raise ValueError(f'WAN decoder {name} is not installed; use Final if the matching tiny decoder is unavailable') from error
        if decoder.latent_channels != tensor.shape[1]:
            raise ValueError('Selected decoder does not match the saved WAN latent channels')
        throw_exception_if_processing_interrupted()
        images = nodes.VAEDecode().decode(decoder, samples)[0]
        scale = 16 if tensor.shape[1] == 48 else 8
        expected = (tensor.shape[0] * (4 * tensor.shape[2] - 3), tensor.shape[3] * scale, tensor.shape[4] * scale, 3)
        if tuple(images.shape) != expected or not bool(torch.isfinite(images).all()):
            raise RuntimeError('WAN decode did not preserve finite frames and dimensions')
        return (images,)


NODE_CLASS_MAPPINGS = {'BenchWanSavedLatent': BenchWanSavedLatent, 'BenchWanDecode': BenchWanDecode}
NODE_DISPLAY_NAME_MAPPINGS = {'BenchWanSavedLatent': 'Saved WAN Latent', 'BenchWanDecode': 'WAN Draft / Final'}
