"""CPU-only, bounded-row Krea Turbo merge/export into the qualified INT8 layout.

Turbo lineage is an explicit declaration: RAW and Turbo have identical shapes.
This exporter does not distill RAW, convert diffusers directories, or bake edit
conditioning. The installed stock checkpoint supplies the storage schema.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import sys
import tempfile

METADATA_KEY = 'mac_kernels_krea_export'
QUANT_CONFIG = {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 256}
DEFAULT_TEMPLATE = Path(__file__).resolve().parents[1] / 'out/source-device/root/models/diffusion_models/krea2_turbo_int8_convrot.safetensors'
TEMPLATE_HEADER_SHA256 = '59ba84d578d8e540a27183c54a798b553670f0eb2afd2ad827821b9aa156cda6'
SPECIAL_NAMES = {'krea2_identity_edit_v1_2_r64.safetensors', 'krea2_style_reference.safetensors',
                 'krea2_turbo_4step_rank_64_lora_comfyui.safetensors'}
SPECIAL_HASHES = {'7590b2d2772f45d7b1252124c63918b1b807e93fbf9eab20546fa7233b4dc482',
                  'f794b47142555c929cf536a2f1e4f335174b9aedbb08572b07d45814d4242423',
                  'f50df5a9e62e4be8aa926a63dd5bb1a64770c4004f763c1208007ae13daa82b8'}
DTYPE_BYTES = {'F32': 4, 'F16': 2, 'BF16': 2, 'I8': 1, 'U8': 1}
FACTOR_SUFFIXES = (
    ('.lora_up.weight', '.lora_down.weight'), ('.lora_B.weight', '.lora_A.weight'),
    ('_lora.up.weight', '_lora.down.weight'), ('.lora.up.weight', '.lora.down.weight'),
    ('.lora_B', '.lora_A'), ('.lora_linear_layer.up.weight', '.lora_linear_layer.down.weight'),
    ('.lora_B.default.weight', '.lora_A.default.weight'))


def _identity(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def inspect_header(path):
    """Read validated metadata without reading tensor payloads or querying a GPU."""
    from safetensors import safe_open
    path = Path(path).expanduser().resolve(strict=True)
    before = _identity(path)
    with path.open('rb') as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError('Incomplete safetensors header')
        n = struct.unpack('<Q', prefix)[0]
        if n > 16 * 2**20:
            raise ValueError('Safetensors header exceeds exporter limit')
        raw = f.read(n)
    header = json.loads(raw)
    with safe_open(str(path), framework='numpy', device='cpu') as f:
        if set(f.keys()) != set(header) - {'__metadata__'}:
            raise ValueError('Safetensors keys disagree')
        for key in f.keys():
            info = f.get_slice(key)
            if header[key]['shape'] != info.get_shape() or header[key]['dtype'] != info.get_dtype():
                raise ValueError('Safetensors tensor metadata disagrees')
    if before != _identity(path):
        raise RuntimeError('Asset changed during admission')
    return {'path': str(path), 'file_identity': before, 'header_sha256': hashlib.sha256(raw).hexdigest(),
            'header': header, 'data_start': 8 + n}


def _unchanged(asset):
    if _identity(asset['path']) != asset['file_identity']:
        raise RuntimeError('Export input changed: ' + asset['path'])


def _hash(asset):
    digest = hashlib.sha256()
    with open(asset['path'], 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b''):
            digest.update(chunk)
    _unchanged(asset)
    return digest.hexdigest()


def _quant_keys(header):
    return {k[:-len('.comfy_quant')] + '.weight' for k in header if k.endswith('.comfy_quant')}


def validate_source(source, reference=DEFAULT_TEMPLATE):
    """Require the exact admitted Comfy Krea schema, with floating or INT8 blocks."""
    src = inspect_header(source)
    template = inspect_header(reference)
    if template['header_sha256'] != TEMPLATE_HEADER_SHA256:
        raise ValueError('Template header differs from the pinned Krea INT8 release')
    expected = template['header']
    quant = _quant_keys(expected)
    if len(quant) != 224 or any(not k.startswith('blocks.') for k in quant):
        raise ValueError('Template must contain 224 Krea block linears')
    actual = src['header']
    quantized = any(v.get('dtype') == 'I8' for k, v in actual.items() if k != '__metadata__')
    expected_keys = set(expected) - {'__metadata__'}
    if not quantized:
        expected_keys = {k for k in expected_keys if not k.endswith(('.weight_scale', '.comfy_quant'))}
    if set(actual) - {'__metadata__'} != expected_keys:
        raise ValueError('Checkpoint keys do not match the qualified Krea architecture')
    for key in expected_keys:
        entry, target = actual[key], expected[key]
        if key.endswith('.comfy_quant'):
            if entry['dtype'] != 'U8' or len(entry['shape']) != 1 or entry['shape'][0] > 512:
                raise ValueError('Invalid ConvRot metadata tensor')
            with open(src['path'], 'rb') as f:
                a, b = entry['data_offsets']
                f.seek(src['data_start'] + a)
                if json.loads(f.read(b - a)) != QUANT_CONFIG:
                    raise ValueError('Only INT8 ConvRot group256 is supported')
        elif entry['shape'] != target['shape']:
            raise ValueError('Krea tensor shape differs: ' + key)
        elif quantized:
            if entry['dtype'] != target['dtype']:
                raise ValueError('Quantized source storage differs: ' + key)
        elif entry['dtype'] not in {'F16', 'BF16', 'F32'}:
            raise ValueError('Full precision source requires floating tensors: ' + key)
    _unchanged(src)
    src['quantized'] = quantized
    return src, template


def _runtime():
    """Import the pinned CPU math and naming conversion; never construct a model."""
    if __package__:
        from .probe_text_encoder_load import bootstrap_cpu
    else:
        from probe_text_encoder_load import bootstrap_cpu
    # Preserve caller arguments: Comfy's bootstrap deliberately substitutes its own.
    arguments = sys.argv[:]
    try:
        bootstrap_cpu()
        import torch
        import comfy.utils as utils
        from comfy_kitchen.backends.eager.quantization import (
            quantize_int8_convrot_weight, dequantize_int8_convrot_weight)
    finally:
        sys.argv = arguments
    return torch, utils, quantize_int8_convrot_weight, dequantize_int8_convrot_weight


def _aliases(header, utils):
    result = {}
    for key, entry in header.items():
        if key.endswith('.weight') and len(entry['shape']) == 2:
            prefix = key[:-7]
            for alias in (prefix, 'diffusion_model.' + prefix, 'lora_unet_' + prefix.replace('.', '_')):
                result[alias] = key
    for alias, target in utils.krea2_to_diffusers({'layers': 28}).items():
        if target in header:
            alias = alias[:-7]
            for prefix in ('', 'transformer.', 'diffusion_model.'):
                result[prefix + alias] = target
            result['lycoris_' + alias.replace('.', '_')] = target
    return result


def _admit_loras(selections, header, utils):
    if not isinstance(selections, list) or len(selections) > 5:
        raise ValueError('At most five ordinary LoRAs can be baked')
    aliases = _aliases(header, utils)
    admitted = []
    total = 0
    for selection in selections:
        if (not isinstance(selection, dict) or set(selection) != {'path', 'strength'}
                or type(selection['strength']) not in (int, float)
                or not math.isfinite(selection['strength']) or not 0 < abs(selection['strength']) <= 2):
            raise ValueError('LoRAs require a path and finite nonzero strength in [-2, 2]')
        asset = inspect_header(selection['path'])
        total += asset['file_identity'][2]
        if total > 2 * 2**30:
            raise ValueError('LoRA assets exceed the 2 GiB resource limit')
        sha = _hash(asset)
        if Path(asset['path']).name in SPECIAL_NAMES or sha in SPECIAL_HASHES:
            raise ValueError('Special reference or acceleration adapters cannot be baked')
        if any(x['sha256'] == sha for x in admitted):
            raise ValueError('Duplicate LoRA content')
        keys = set(asset['header']) - {'__metadata__'}
        consumed = set()
        targets = {}
        for alias, target in aliases.items():
            for up_suffix, down_suffix in FACTOR_SUFFIXES:
                up, down = alias + up_suffix, alias + down_suffix
                if up not in keys or down not in keys:
                    continue
                if target in targets:
                    raise ValueError('Multiple factor pairs for one LoRA target')
                u, d = asset['header'][up], asset['header'][down]
                shape = header[target]['shape']
                if (u['dtype'] not in {'F16', 'BF16', 'F32'} or d['dtype'] not in {'F16', 'BF16', 'F32'}
                        or len(u['shape']) != 2 or len(d['shape']) != 2 or u['shape'][1] <= 0
                        or d['shape'][0] != u['shape'][1] or shape != [u['shape'][0], d['shape'][1]]):
                    raise ValueError('Invalid ordinary LoRA factors: ' + target)
                alpha = alias + '.alpha'
                if alpha in keys:
                    a = asset['header'][alpha]
                    if a['dtype'] not in {'F16', 'BF16', 'F32'} or math.prod(a['shape']) != 1:
                        raise ValueError('Invalid LoRA alpha')
                    consumed.add(alpha)
                else:
                    alpha = None
                targets[target] = (up, down, alpha)
                consumed.update((up, down))
        if not targets or consumed != keys:
            raise ValueError('Unsupported or unconsumed LoRA keys: ' + ', '.join(sorted(keys - consumed)))
        asset.update(strength=selection['strength'], sha256=sha, name=Path(asset['path']).name, targets=targets)
        admitted.append(asset)
    return admitted


def _receipt(asset):
    return {k: asset[k] for k in ('path', 'file_identity', 'header_sha256', 'quantized') if k in asset}


def _raw_copy(source, key, destination):
    start, end = source['header'][key]['data_offsets']
    with open(source['path'], 'rb') as stream:
        stream.seek(source['data_start'] + start)
        remaining = end - start
        while remaining:
            chunk = stream.read(min(remaining, 8 * 2**20))
            if not chunk:
                raise RuntimeError('Source truncated during export')
            destination.write(chunk)
            remaining -= len(chunk)


def _tensor_bytes(tensor):
    return tensor.contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes()


def _read_tensor(asset, stream, key, torch, *, start=None, end=None):
    """Read only requested rows into owned bytes, with no persistent payload mmap.

    A CPU safetensors get_slice retains its entire file mapping. Sequentially
    touching every matrix then grows process RSS even after each row is freed.
    Ordinary buffered reads keep that residency outside this process's mapping.
    The official safetensors validator has already checked offsets and shapes.
    """
    info = asset['header'][key]
    shape = info['shape']
    first, last = info['data_offsets']
    if start is not None:
        if not shape or not 0 <= start <= end <= shape[0]:
            raise ValueError('Invalid tensor row range')
        row_bytes = math.prod(shape[1:]) * DTYPE_BYTES[info['dtype']]
        first += start * row_bytes
        last = first + (end - start) * row_bytes
        shape = [end - start, *shape[1:]]
    stream.seek(asset['data_start'] + first)
    payload = bytearray(last - first)
    view = memoryview(payload)
    received = 0
    while received < len(payload):
        count = stream.readinto(view[received:])
        if not count:
            raise RuntimeError('Source truncated during tensor reading')
        received += count
    dtype = {'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16,
             'I8': torch.int8, 'U8': torch.uint8}[info['dtype']]
    if not payload:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(payload, dtype=dtype).reshape(shape)


def export_checkpoint(source, output, *, source_variant, loras=None,
                      reference=DEFAULT_TEMPLATE, row_chunk=256, progress=None):
    """Publish one complete no-overwrite safetensors file and return provenance.

    Matrix work is at most ``row_chunk`` output rows plus one layer's factors.
    Untouched quantized tensors are copied byte-for-byte. Source and adapter
    identities are checked again before publication, including ctime.
    """
    if source_variant != 'turbo':
        raise ValueError('Explicit source_variant=turbo is required; no RAW distillation is performed')
    if type(row_chunk) is not int or not 1 <= row_chunk <= 1024:
        raise ValueError('row_chunk must be between 1 and 1024')
    output = Path(output).expanduser().absolute()
    if output.suffix != '.safetensors' or output.exists():
        raise ValueError('Export requires a new .safetensors output path')
    src, template = validate_source(source, reference)
    torch, utils, quantize, dequantize = _runtime()
    admitted = _admit_loras(loras or [], template['header'], utils)
    existing = src['header'].get('__metadata__', {}).get(METADATA_KEY)
    previous = []
    if existing:
        old = json.loads(existing)
        if (old.get('schema') != 1 or old.get('architecture') != 'krea2'
                or old.get('base_variant') != 'turbo' or old.get('quantization') != 'int8-convrot-group256'):
            raise ValueError('Unsupported existing export provenance')
        previous = old.get('baked_loras')
        if not isinstance(previous, list) or len(previous) > 5:
            raise ValueError('Invalid existing baked LoRA provenance')
        hashes = []
        for item in previous:
            if (not isinstance(item, dict) or not isinstance(item.get('name'), str)
                    or not isinstance(item.get('path'), str) or not isinstance(item.get('file_identity'), list)
                    or len(item['file_identity']) != 5 or not all(type(x) is int for x in item['file_identity'])
                    or not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64
                    or any(x not in '0123456789abcdef' for x in item['sha256'])
                    or type(item.get('strength')) not in (int, float)
                    or not math.isfinite(item['strength']) or not 0 < abs(item['strength']) <= 2):
                raise ValueError('Invalid existing baked LoRA provenance')
            if item['name'] in SPECIAL_NAMES or item['sha256'] in SPECIAL_HASHES:
                raise ValueError('Special reference or acceleration adapters cannot be baked')
            hashes.append(item['sha256'])
        if len(set(hashes)) != len(hashes):
            raise ValueError('Duplicate existing baked LoRA content')
    if len(previous) + len(admitted) > 5:
        raise ValueError('At most five total baked LoRAs are supported')
    if {x['sha256'] for x in previous} & {x['sha256'] for x in admitted}:
        raise ValueError('Duplicate previously baked LoRA content')
    provenance = {'schema': 1, 'exporter_version': 1, 'architecture': 'krea2', 'base_variant': 'turbo',
                  'lineage': 'user-declared', 'quantization': 'int8-convrot-group256',
                  'source': _receipt(src), 'template': _receipt(template),
                  'baked_loras': previous + [{k: a[k] for k in ('name', 'path', 'file_identity', 'sha256', 'strength')} for a in admitted],
                  'created_at': datetime.now(timezone.utc).isoformat(), 'row_chunk': row_chunk}
    output.parent.mkdir(parents=True, exist_ok=True)
    quant_keys = _quant_keys(template['header'])
    transformed = {k for a in admitted for k in a['targets']}
    if not src['quantized']:
        transformed |= quant_keys
        transformed |= {k for k, v in src['header'].items() if k != '__metadata__' and v['dtype'] != template['header'][k]['dtype']}
    with tempfile.TemporaryDirectory(prefix='.krea-export-', dir=output.parent) as temporary, ExitStack() as stack, torch.inference_mode():
        reader = stack.enter_context(open(src['path'], 'rb'))
        readers = [stack.enter_context(open(a['path'], 'rb')) for a in admitted]
        # Even untouched float payloads must be finite. Scan bounded rows rather
        # than loading large global matrices; scale/norm vectors are small.
        for key, info in src['header'].items():
            if key == '__metadata__' or info['dtype'] not in {'F16', 'BF16', 'F32'}:
                continue
            shape = info['shape']
            count = shape[0] if shape else 1
            for start in range(0, count, row_chunk):
                value = (_read_tensor(src, reader, key, torch, start=start, end=min(count, start + row_chunk))
                         if shape else _read_tensor(src, reader, key, torch))
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite checkpoint tensor: ' + key)
                if key.endswith('.weight_scale') and not (value > 0).all():
                    raise ValueError('INT8 scales must be positive: ' + key)
        parts = {}
        # Changed matrix payloads are spooled separately, never retained together.
        for completed, key in enumerate(sorted(transformed), 1):
            if progress:
                progress({'completed': completed - 1, 'total': len(transformed), 'tensor': key})
            shape = template['header'][key]['shape']
            is_quant = key in quant_keys
            dtype = {'BF16': torch.bfloat16, 'F32': torch.float32}[template['header'][key]['dtype']] if not is_quant else torch.int8
            factors = []
            for asset, lora_reader in zip(admitted, readers):
                if key not in asset['targets']:
                    continue
                up, down, alpha = asset['targets'][key]
                u = _read_tensor(asset, lora_reader, up, torch).float()
                d = _read_tensor(asset, lora_reader, down, torch).float()
                value = float(_read_tensor(asset, lora_reader, alpha, torch).item()) if alpha else u.shape[1]
                if not math.isfinite(value) or not torch.isfinite(u).all() or not torch.isfinite(d).all():
                    raise ValueError('LoRA factors and alpha must be finite')
                factors.append((u, d, asset['strength'] * value / u.shape[1]))
            # Nonmatrix float normalizations are normalized without row quantization.
            if len(shape) != 2:
                tensor = _read_tensor(src, reader, key, torch).to(dtype)
                if not torch.isfinite(tensor).all():
                    raise ValueError('Nonfinite checkpoint tensor: ' + key)
                part = Path(temporary) / str(len(parts))
                part.write_bytes(_tensor_bytes(tensor))
                parts[key] = part
                continue
            part = Path(temporary) / str(len(parts))
            scales = Path(temporary) / (str(len(parts)) + '-scales') if is_quant else None
            with part.open('wb') as weights, (scales.open('wb') if scales else open(os.devnull, 'wb')) as scale_stream:
                for start in range(0, shape[0], row_chunk):
                    end = min(shape[0], start + row_chunk)
                    weight = _read_tensor(src, reader, key, torch, start=start, end=end).float()
                    if is_quant and src['quantized']:
                        scale = _read_tensor(src, reader, key[:-7] + '.weight_scale', torch, start=start, end=end).float()
                        weight = dequantize(weight.to(torch.int8), scale, 256)
                    for up, down, multiplier in factors:
                        weight.addmm_(up[start:end], down, beta=1, alpha=multiplier)
                    if not torch.isfinite(weight).all():
                        raise ValueError('Nonfinite merged checkpoint tensor: ' + key)
                    if is_quant:
                        q, scale = quantize(weight, 256)
                        weights.write(_tensor_bytes(q))
                        scale_stream.write(_tensor_bytes(scale.float()))
                    else:
                        cast = weight.to(dtype)
                        if not torch.isfinite(cast).all():
                            raise ValueError('Merged tensor overflows output dtype: ' + key)
                        weights.write(_tensor_bytes(cast))
            parts[key] = part
            if scales:
                parts[key[:-7] + '.weight_scale'] = scales
                config = Path(temporary) / (str(len(parts)) + '-config')
                config.write_bytes(json.dumps(QUANT_CONFIG).encode())
                parts[key[:-7] + '.comfy_quant'] = config
            del factors
        header = {'__metadata__': {METADATA_KEY: json.dumps(provenance, sort_keys=True)}}
        offset = 0
        for key in sorted(set(template['header']) - {'__metadata__'}):
            spec = template['header'][key]
            size = parts[key].stat().st_size if key in parts else src['header'][key]['data_offsets'][1] - src['header'][key]['data_offsets'][0]
            shape = [size] if key.endswith('.comfy_quant') else spec['shape']
            header[key] = {'dtype': spec['dtype'], 'shape': shape, 'data_offsets': [offset, offset + size]}
            offset += size
        raw = json.dumps(header, separators=(',', ':')).encode()
        raw += b' ' * (-len(raw) % 8)
        candidate = Path(temporary) / 'complete.safetensors'
        with candidate.open('wb') as destination:
            destination.write(struct.pack('<Q', len(raw)))
            destination.write(raw)
            for key in sorted(set(header) - {'__metadata__'}):
                if key in parts:
                    with parts[key].open('rb') as part:
                        shutil.copyfileobj(part, destination, 8 * 2**20)
                else:
                    _raw_copy(src, key, destination)
            destination.flush()
            os.fsync(destination.fileno())
        result = inspect_header(candidate)
        sha256 = _hash(result)
        for asset in [src, template, *admitted]:
            _unchanged(asset)
        os.link(candidate, output)  # Atomic publication; refuses a concurrent destination.
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    result = inspect_header(output)
    return {'output': {**_receipt(result), 'sha256': sha256}, 'provenance': provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-variant', choices=['turbo'], required=True,
                        help='Explicitly attest Turbo lineage; this exporter cannot distill RAW')
    parser.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument('--lora', nargs=2, action='append', default=[], metavar=('PATH', 'STRENGTH'))
    parser.add_argument('--row-chunk', type=int, default=256)
    parser.add_argument('--cpu-threads', type=int)
    args = parser.parse_args()
    if args.cpu_threads is not None:
        if not 1 <= args.cpu_threads <= 32:
            parser.error('--cpu-threads must be between 1 and 32')
        import torch
        torch.set_num_threads(args.cpu_threads)
    def progress(value):
        if value['completed'] % 8 == 0:
            print(f"Exporting {value['completed'] + 1}/{value['total']}: {value['tensor']}", file=sys.stderr, flush=True)
    def interrupted(_number, _frame):
        raise KeyboardInterrupt
    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        result = export_checkpoint(args.source, args.output, source_variant=args.source_variant,
                                   loras=[{'path': path, 'strength': float(strength)} for path, strength in args.lora],
                                   reference=args.template, row_chunk=args.row_chunk, progress=progress)
    except KeyboardInterrupt:
        print('Export interrupted; unpublished temporary files removed.', file=sys.stderr)
        raise SystemExit(130) from None
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
