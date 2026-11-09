"""Source-aware CPU conditioning cache for headless generation runners."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile


def file_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def context(prompt, negative, encoder, source_roots, runtime):
    sources = {}
    for root in source_roots:
        root = Path(root).resolve(strict=True)
        paths = [root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in {'.py', '.json', '.metal', '.mm', '.h', '.cpp', '.so', '.dylib'})
        for path in paths:
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'version': 1, 'prompt': prompt, 'negative': negative, 'encoder': file_identity(encoder),
            'sources': sources, 'runtime': runtime}


def path_for(directory, identity, policy='auto', *, namespace='wan'):
    if policy not in {'auto', 'off'}:
        raise ValueError('COND_CACHE must be auto or off')
    if policy == 'off':
        return None
    if not isinstance(namespace, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', namespace):
        raise ValueError('Conditioning cache namespace must be a filename component')
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    return Path(directory) / f'{namespace}-cond-v1-{digest}.pt'


def load(path, identity, *, with_provenance=False, with_digest=False):
    if path is None or not path.is_file():
        return None
    import torch
    if with_digest:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        payload = torch.load(io.BytesIO(raw), weights_only=True, map_location='cpu')
    else:
        payload = torch.load(path, weights_only=True, map_location='cpu')
    if not isinstance(payload, dict) or payload.get('context') != identity:
        return None
    if not {'positive', 'negative'} <= payload.keys():
        raise ValueError('Malformed conditioning cache')
    result = (payload['positive'], payload['negative'])
    if with_provenance:
        result = (*result, payload.get('provenance'))
    return (*result, digest) if with_digest else result


def _cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f'Unsupported conditioning cache value: {type(value).__name__}')


def save(path, identity, positive, negative, *, provenance=None, with_digest=False):
    if path is None:
        return
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'context': identity, 'positive': _cpu(positive), 'negative': _cpu(negative)}
    if provenance is not None:
        payload['provenance'] = _cpu(provenance)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest() if with_digest else None
        os.replace(temporary, path)
        return digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def wan_context(prompt, negative, code, folder_paths, torch, mm, mode, argv):
    """Bind local checkpoint stat identity, encoder sources and execution policy."""
    from importlib.metadata import version
    import platform
    import comfy_kitchen
    roots = [Path(code) / 'comfy', Path(code) / 'nodes.py', Path(code) / 'folder_paths.py',
             Path(comfy_kitchen.__file__).parent, Path(__file__)]
    embeddings = [file_identity(p) for root in folder_paths.get_folder_paths('embeddings')
                  for p in sorted(Path(root).rglob('*')) if p.is_file()]
    runtime = {'torch': str(torch.__version__), 'python': platform.python_version(),
               'platform': platform.platform(), 'device': str(mm.text_encoder_device()), 'mode': mode,
               'argv': list(argv), 'embeddings': embeddings,
               'packages': {name: version(name) for name in ('transformers', 'tokenizers', 'sentencepiece', 'safetensors')},
               'environment': {key: value for key, value in sorted(os.environ.items())
                               if key.startswith(('COMFY_', 'CK_', 'PYTORCH_'))}}
    return context(prompt, negative, folder_paths.get_full_path_or_raise('text_encoders', 'umt5_xxl_fp16.safetensors'), roots, runtime)
