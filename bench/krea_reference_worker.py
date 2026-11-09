#!/usr/bin/env python3
"""Owned CPU-only VAE preparation worker; never launch outside gpu_runner."""
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise ValueError('Krea reference worker requires one owned request directory')
    directory = Path(args[0]).resolve(strict=True)
    config = json.loads((directory / 'config.json').read_text())
    from probe_text_encoder_load import require_supervisor, bootstrap_cpu
    require_supervisor()
    from krea_reference import _worker_comfy_binding, _validate_worker_comfy_modules, validate_output_size
    output_size = validate_output_size(config['output_size'])
    bindings = config['bindings']
    comfy_binding = _worker_comfy_binding(bindings['comfy_root'])
    if any(bindings.get(key) != value for key, value in comfy_binding.items()):
        raise RuntimeError('Krea reference worker selected Comfy sources changed')
    os.environ['COMFY_DIR'] = comfy_binding['comfy_root']
    bootstrap_cpu()
    import torch
    import comfy.utils
    import comfy.sd
    _validate_worker_comfy_modules(comfy_binding['comfy_root'], comfy.sd, comfy.utils)
    from safetensors.torch import load_file
    from krea_reference import _file_identity, _worker_sources, _digest, prepare_reference, _write_reference_result
    path = Path(bindings['vae_path']).resolve(strict=True)
    if (_file_identity(path) != bindings['vae_identity'] or _worker_sources() != bindings['sources']
            or _worker_comfy_binding(comfy_binding['comfy_root']) != comfy_binding):
        raise RuntimeError('Krea reference worker admission sources changed')
    images = load_file(str(directory / 'images.safetensors'), device='cpu')
    if set(images) != {f'image_{i}' for i in range(len(config['images']))} or not 1 <= len(images) <= 2:
        raise ValueError('Invalid Krea reference worker image membership')
    ordered = [images[f'image_{i}'] for i in range(len(images))]
    if config['images'] != [{'sha256': _digest(x), 'shape': list(x.shape)} for x in ordered]:
        raise RuntimeError('Krea reference worker input images changed')
    with torch.inference_mode():
        state, metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
        vae = comfy.sd.VAE(sd=state, metadata=metadata, device=torch.device('cpu'), dtype=torch.float32)
        state = None
        prepared = prepare_reference(config['mode'], ordered[0], ordered[1] if len(ordered) == 2 else None,
            vae=vae, torch=torch, runtime=config['runtime'], ref_boost=config['ref_boost'],
            grounding_size=config['grounding_size'], spectral=config['spectral'], output_size=output_size)
        if (_file_identity(path) != bindings['vae_identity'] or _worker_sources() != bindings['sources']
                or _worker_comfy_binding(comfy_binding['comfy_root']) != comfy_binding):
            raise RuntimeError('Krea reference worker sources changed during encoding')
        _write_reference_result(prepared, directory, torch=torch, bindings=bindings, pid=os.getpid())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
