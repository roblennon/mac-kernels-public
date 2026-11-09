"""Optional source-bound CPU conditioning reuse for text-only Krea Turbo runs."""
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform

if __package__:
    from . import conditioning_cache as cache
else:
    import conditioning_cache as cache


ENCODER = 'qwen3vl_4b_bf16.safetensors'
WIDTH = 12 * 2560


def negative_policy(cfg):
    if not math.isfinite(cfg):
        raise ValueError('Krea CFG must be finite')
    return 'encoded-empty' if cfg > 1 else 'zero-positive'


def lookup_context(*, code, folder_paths, torch, mm, mode, argv, te, prompt, cfg):
    """Bind requested policy; actual checkpoint-selected execution is separate."""
    import comfy_kitchen
    device = mm.text_encoder_device()
    if te != ENCODER or device.type != 'cpu':
        raise ValueError('Krea caching currently requires the BF16 Qwen3-VL 4B CPU encoder')
    code = Path(code)
    roots = [code / 'comfy', code / 'nodes.py', code / 'folder_paths.py',
             Path(comfy_kitchen.__file__).parent, Path(__file__), Path(cache.__file__),
             Path(__file__).with_name('run_krea2.py')]
    # context() filters extensions for source directories. Individual assets
    # also bind merges.txt and any non-JSON tokenizer data actually on disk.
    roots += sorted(p for p in (code / 'comfy/text_encoders/qwen25_tokenizer').rglob('*') if p.is_file())
    embeddings = [cache.file_identity(p) for root in folder_paths.get_folder_paths('embeddings')
                  for p in sorted(Path(root).rglob('*')) if p.is_file()]
    runtime = {'torch': str(torch.__version__), 'python': platform.python_version(),
               'platform': platform.platform(), 'requested_encoder_device': str(device),
               'requested_encoder_dtype': str(mm.text_encoder_dtype(device)),
               'mode': mode, 'argv': list(argv), 'embeddings': embeddings,
               'packages': {name: version(name) for name in ('transformers', 'tokenizers', 'sentencepiece', 'safetensors')},
               'environment': {key: value for key, value in sorted(os.environ.items())
                               if key.startswith(('COMFY_', 'CK_', 'PYTORCH_'))}}
    checkpoint = Path(folder_paths.get_full_path_or_raise('text_encoders', te)).resolve(strict=True)
    identity = cache.context(prompt, {'policy': negative_policy(cfg), 'text': '' if cfg > 1 else None},
                             checkpoint, roots, runtime)
    # Preserve Wan's existing identity format; Krea also rejects replacements
    # and same-size edits whose original modification time was restored.
    stat = checkpoint.stat()
    identity['encoder'] = {'path': str(checkpoint), 'size': stat.st_size,
                           'mtime_ns': stat.st_mtime_ns, 'device': stat.st_dev,
                           'inode': stat.st_ino, 'ctime_ns': stat.st_ctime_ns}
    return identity


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def tensor_records(positive, negative, *, torch, cfg):
    """Validate the real flattened Krea node shape and bind all payload values."""
    policy = negative_policy(cfg)
    for label, conditioning in (('positive', positive), ('negative', negative)):
        if not isinstance(conditioning, (list, tuple)) or not conditioning:
            raise ValueError(f'Invalid Krea {label} conditioning')
        for item in conditioning:
            if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[1], dict):
                raise ValueError(f'Invalid Krea {label} conditioning entry')
            value = item[0]
            if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                    or not value.is_floating_point() or value.ndim != 3
                    or value.shape[0] != 1 or value.shape[1] <= 0 or value.shape[2] != WIDTH):
                raise ValueError('Expected CPU Krea conditioning shape [1, tokens, 30720]')
    if policy == 'zero-positive':
        if len(positive) != len(negative) or any(p[0].shape != n[0].shape for p, n in zip(positive, negative)):
            raise ValueError('Krea zero-negative shape differs from positive')
        if any(bool(torch.count_nonzero(item[0]).item()) for item in negative):
            raise ValueError('Krea zero-negative conditioning contains nonzero values')
    records = []

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            if value.device.type != 'cpu' or not bool(torch.isfinite(value).all().item()):
                raise ValueError('Krea conditioning tensors must be finite CPU values')
            raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            records.append({'path': path, 'kind': 'tensor', 'shape': list(value.shape),
                            'dtype': str(value.dtype), 'sha256': hashlib.sha256(raw).hexdigest()})
        elif isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise ValueError('Krea conditioning metadata keys must be strings')
            records.append({'path': path, 'kind': 'dict', 'keys': sorted(value)})
            for key in sorted(value):
                visit(value[key], path + '.' + key)
        elif isinstance(value, (list, tuple)):
            records.append({'path': path, 'kind': type(value).__name__, 'length': len(value)})
            for index, item in enumerate(value):
                visit(item, path + f'[{index}]')
        elif value is None or type(value) in (str, int, float, bool):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError('Nonfinite Krea conditioning metadata')
            records.append({'path': path, 'kind': type(value).__name__, 'value': value})
        else:
            raise ValueError(f'Unsupported Krea conditioning metadata: {type(value).__name__}')

    visit({'positive': positive, 'negative': negative}, 'root')
    return records


def _observed_encoder(clip):
    parameters = list(clip.cond_stage_model.parameters())
    return {'load_device': str(clip.patcher.load_device),
            'parameter_devices': sorted({str(value.device) for value in parameters}),
            'parameter_dtypes': sorted({str(value.dtype) for value in parameters}),
            'declared_dtypes': sorted(str(dtype) for dtype in clip.cond_stage_model.dtypes)}


def _validate_observed(observed):
    if (not isinstance(observed, dict)
            or set(observed) != {'load_device', 'parameter_devices', 'parameter_dtypes', 'declared_dtypes'}
            or observed['load_device'] != 'cpu'
            or observed['parameter_devices'] != ['cpu']
            or observed['declared_dtypes'] != ['torch.bfloat16']
            or not isinstance(observed['parameter_dtypes'], list)
            or not all(isinstance(dtype, str) for dtype in observed['parameter_dtypes'])
            or 'torch.bfloat16' not in observed['parameter_dtypes']
            or not set(observed['parameter_dtypes']) <= {'torch.bfloat16', 'torch.float32'}):
        raise ValueError('Krea cache encoder provenance requires observed CPU BF16 parameters')


def encode_conditioning(N, *, root, code, folder_paths, torch, mm, mode, argv,
                        te, prompt, cfg, policy='off', stage='generate', lap):
    """Reuse owned cache serialization; the caller handles a capture-stage exit."""
    negative = negative_policy(cfg)
    if policy not in {'auto', 'off'} or stage not in {'generate', 'capture'}:
        raise ValueError('COND_CACHE must be auto/off and COND_STAGE must be generate/capture')
    if stage == 'capture' and policy != 'auto':
        raise ValueError('Krea conditioning capture requires COND_CACHE=auto')
    identity_args = dict(code=code, folder_paths=folder_paths, torch=torch, mm=mm,
                         mode=mode, argv=argv, te=te, prompt=prompt, cfg=cfg)
    identity = lookup_context(**identity_args) if policy == 'auto' else None
    path = cache.path_for(Path(root) / 'out/cache', identity, policy, namespace='krea')
    cached = cache.load(path, identity, with_provenance=True, with_digest=True)
    if cached is not None:
        pos, neg, provenance, cache_digest = cached
        records = tensor_records(pos, neg, torch=torch, cfg=cfg)
        if (not isinstance(provenance, dict) or provenance.get('version') != 1
                or provenance.get('negative_policy') != negative
                or provenance.get('tensor_records') != records):
            raise ValueError('Krea conditioning cache provenance or tensor bytes disagree')
        _validate_observed(provenance.get('encoder_observed'))
        if lookup_context(**identity_args) != identity:
            raise RuntimeError('Encoder identity changed while reading Krea conditioning')
        origin = 'cache-hit'
        lap('load_cached_cond')
    else:
        clip = None
        try:
            clip = N['CLIPLoader']().load_clip(te, 'krea2', 'default')[0]
            lap('load_te')
            pos = N['CLIPTextEncode']().encode(clip, prompt)[0]
            neg = (N['CLIPTextEncode']().encode(clip, '')[0] if cfg > 1
                   else N['ConditioningZeroOut']().zero_out(pos)[0])
            lap('encode')
            provenance = None
            if policy == 'auto':
                records = tensor_records(pos, neg, torch=torch, cfg=cfg)
                observed = _observed_encoder(clip)
                _validate_observed(observed)
                provenance = {'version': 1, 'negative_policy': negative,
                              'tensor_records': records, 'encoder_observed': observed}
            if path is not None and lookup_context(**identity_args) != identity:
                raise RuntimeError('Encoder identity changed while encoding Krea conditioning')
            cache_digest = cache.save(path, identity, pos, neg, provenance=provenance, with_digest=True)
        finally:
            clip = None
            mm.unload_all_models()
            mm.soft_empty_cache()
        origin = 'fresh-encoding'
    audit = {'cache_policy': policy, 'stage': stage, 'origin': origin,
             'cache_path': str(path) if path is not None else None,
             'cache_sha256': cache_digest,
             'context_sha256': _digest(identity) if identity is not None else None,
             'provenance': provenance, 'scope': 'text-only Krea conditioning; encoder observation belongs to the cache miss'}
    return pos, neg, audit
