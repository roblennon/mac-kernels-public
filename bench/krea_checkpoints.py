"""Admission of exported Turbo INT8 files against the installed native topology."""
import hashlib
import json
import math
from pathlib import Path
import struct

DEFAULT_CHECKPOINT = 'krea2_turbo_int8_convrot.safetensors'
EXPORT_METADATA = 'mac_kernels_krea_export'
QUANTIZATION = {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 256}


def file_identity(path):
    st = Path(path).stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def safe_name(name):
    return (isinstance(name, str) and name.endswith('.safetensors')
            and name not in ('', '.', '..') and '..' not in name
            and not any(c in name for c in '/\\\x00'))


def read_header(path):
    with Path(path).open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Incomplete checkpoint header')
        size = struct.unpack('<Q', prefix)[0]
        if not 2 <= size <= 2**20:
            raise ValueError('Checkpoint header exceeds the supported limit')
        raw = stream.read(size)
    if len(raw) != size:
        raise ValueError('Incomplete checkpoint header')
    try:
        header = json.loads(raw)
    except RecursionError as error:
        raise ValueError('Checkpoint header nesting exceeds the supported limit') from error
    if not isinstance(header, dict):
        raise ValueError('Invalid checkpoint header')
    return header, raw


def validate_custom_checkpoint(path, reference):
    """Metadata-only admission; official safetensors validates all tensor offsets."""
    from safetensors import safe_open, SafetensorError
    path = Path(path).resolve(strict=True)
    before = file_identity(path)
    header, raw = read_header(path)
    template, _ = read_header(reference)
    try:
        metadata = header.get('__metadata__', {})
        if not isinstance(metadata, dict):
            raise ValueError('Checkpoint metadata must be an object')
        provenance = json.loads(metadata.get(EXPORT_METADATA, 'null'))
        if (not isinstance(provenance, dict) or provenance.get('schema') != 1
                or provenance.get('architecture') != 'krea2' or provenance.get('base_variant') != 'turbo'
                or provenance.get('quantization') != 'int8-convrot-group256'):
            raise ValueError('Select a Turbo INT8 checkpoint produced by the custom exporter')
        baked = provenance['baked_loras']
        if not isinstance(baked, list) or len(baked) > 5:
            raise ValueError('Invalid baked LoRA receipt')
        hashes = set()
        for adapter in baked:
            digest, strength = adapter['sha256'], adapter['strength']
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest) or digest in hashes
                    or type(strength) not in (int, float) or not 0 < abs(strength) <= 2
                    or not math.isfinite(strength)):
                raise ValueError('Invalid baked LoRA receipt')
            hashes.add(digest)
        names = set(header) - {'__metadata__'}
        if names != set(template) - {'__metadata__'}:
            raise ValueError('Checkpoint tensor names differ from native Krea Turbo')
        with safe_open(str(path), framework='numpy', device='cpu') as reader:
            if set(reader.keys()) != names:
                raise ValueError('Checkpoint safetensors names differ')
            with path.open('rb') as stream:
                for key in names:
                    info, expected = header[key], template[key]
                    quant = key.endswith('.comfy_quant')
                    if info['dtype'] != expected['dtype'] or (not quant and info['shape'] != expected['shape']):
                        raise ValueError(f'Checkpoint tensor geometry or dtype differs: {key}')
                    if quant:
                        if len(info['shape']) != 1 or not 1 <= info['shape'][0] <= 512:
                            raise ValueError('Invalid ConvRot metadata tensor')
                        start, end = info['data_offsets']
                        stream.seek(8 + len(raw) + start)
                        if json.loads(stream.read(end - start)) != QUANTIZATION:
                            raise ValueError('Checkpoint requires native INT8 ConvRot group256')
        if file_identity(path) != before:
            raise RuntimeError('Checkpoint changed during admission')
    except (KeyError, TypeError, json.JSONDecodeError, RecursionError, SafetensorError) as error:
        raise ValueError('Malformed custom checkpoint receipt or tensor header') from error
    return {'path': str(path), 'file_identity': before,
            'header_sha256': hashlib.sha256(raw).hexdigest(), 'provenance': provenance}


def reject_baked_replay(checkpoint, stack):
    baked = checkpoint.get('provenance') or {}
    baked_hashes = {x['sha256'] for x in baked.get('baked_loras', [])}
    if any(x['sha256'] in baked_hashes for x in stack['adapters']):
        raise ValueError('This LoRA is already baked into the selected checkpoint; remove it from the stack')


class CheckpointCatalog:
    """Snapshot installed exports; file replacement requires a server restart."""
    def __init__(self, root, folder_paths):
        self.root = Path(root)
        self.folder_paths = folder_paths
        self.directory = self.root / 'models/diffusion_models'
        self.reference = self.directory / DEFAULT_CHECKPOINT
        self.entries = {DEFAULT_CHECKPOINT: {'path': str(self.reference.resolve(strict=True)),
            'file_identity': file_identity(self.reference), 'provenance': None}}
        self.rejected = {}
        for path in sorted(self.directory.glob('*.safetensors')):
            if path.name == DEFAULT_CHECKPOINT or not safe_name(path.name):
                continue
            try:
                self.entries[path.name] = validate_custom_checkpoint(path, self.reference)
            except (OSError, ValueError, RuntimeError) as error:
                self.rejected[path.name] = str(error)

    def names(self):
        return list(self.entries)

    def resolve(self, name):
        if not safe_name(name) or name not in self.entries:
            raise ValueError('Checkpoint is not an admitted Turbo INT8 export; restart after installing it')
        entry = self.entries[name]
        selected = Path(self.folder_paths.get_full_path_or_raise('diffusion_models', name)).resolve(strict=True)
        expected = (self.directory / name).resolve(strict=True)
        if selected != expected or str(selected) != entry['path']:
            raise RuntimeError('Checkpoint selection differs from the profile root')
        if file_identity(selected) != entry['file_identity']:
            raise RuntimeError('Admitted checkpoint changed; restart the Workbench')
        return entry
