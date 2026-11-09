"""Shared CPU loading and identity checks for saved Wan video samples."""
import hashlib
from pathlib import Path


def latent_identity(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def load_saved_latent(path):
    """Read the same finite CPU Wan sample for draft and final decoding."""
    import torch
    path = Path(path).resolve(strict=True)
    identity = latent_identity(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(chunk)
    if path.suffix == '.latent':
        from safetensors.torch import load_file
        native = load_file(path, device='cpu')
        if 'latent_format_version_0' not in native:
            raise ValueError('Expected an unscaled native Comfy latent')
        samples = native.get('latent_tensor')
    else:
        samples = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(samples, torch.Tensor) or samples.device.type != 'cpu'
            or samples.ndim != 5 or not samples.is_floating_point()
            or min(samples.shape) < 1 or samples.shape[1] not in (16, 48)
            or not bool(torch.isfinite(samples).all())):
        raise ValueError('Expected a finite CPU floating Wan video latent with 16 or 48 channels')
    if latent_identity(path) != identity:
        raise RuntimeError('Saved latent changed while loading')
    batch, channels, temporal, height, width = samples.shape
    spatial_scale = 16 if channels == 48 else 8
    geometry = (batch * (4 * temporal - 3), height * spatial_scale, width * spatial_scale, 3)
    receipt = {'path': str(path), 'file_identity': identity, 'sha256': digest.hexdigest(),
               'shape': list(samples.shape), 'dtype': str(samples.dtype), 'finite': True}
    return {'samples': samples}, receipt, geometry
