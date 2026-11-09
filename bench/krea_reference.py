"""Bounded Krea reference preparation; diffusion stays in the existing Comfy runtime."""
from dataclasses import dataclass
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import weakref

EDIT_REVISION = '86f886dac23013d88996e3a2e99093ba44d322fb'
EDIT_SOURCE_SHA256 = '43c5f9aa2292bd4174dfe9aca0baa10d82713dafe2e04dd1ba2fde2ce89c3d39'  # Set below to the independently pinned upstream source.
MODES = {'identity_edit', 'style_reference'}


@dataclass
class PreparedReference:
    mode: str
    images: tuple
    latents: tuple
    encoded: dict
    metadata: dict
    author: object = None
    options_signature: object = None
    output_size: tuple = (512, 512)


def validate_output_size(output_size):
    """Bound reference geometry; the Workbench owns its named preset selection."""
    if (not isinstance(output_size, (tuple, list)) or len(output_size) != 2
            or any(type(side) is not int or side < 32 or side > 1568 or side % 32
                   for side in output_size)
            or output_size[0] * output_size[1] > 928 * 1152):
        raise ValueError('Krea output size requires bounded integer dimensions divisible by 32')
    return tuple(output_size)


def _digest(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def load_edit_runtime(runtime=None):
    root = Path(runtime or os.environ.get('KREA_EDIT_RUNTIME', ''))
    if not str(runtime or os.environ.get('KREA_EDIT_RUNTIME', '')):
        raise ValueError('KREA_EDIT_RUNTIME must name the pinned external Krea2Edit runtime')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('revision') != EDIT_REVISION:
        raise ValueError('Krea2Edit runtime revision differs from the qualified author revision')
    source = root / '__init__.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EDIT_SOURCE_SHA256 or manifest.get('files', {}).get('__init__.py') != digest:
        raise ValueError('Krea2Edit runtime source digest changed')
    spec = importlib.util.spec_from_file_location('_mk_pinned_krea2edit', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_image(image, torch):
    if (not isinstance(image, torch.Tensor) or image.device.type != 'cpu'
            or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3
            or min(image.shape[1:3]) < 16 or max(image.shape[1:3]) > 512
            or not image.is_floating_point() or not bool(torch.isfinite(image).all())
            or bool((image < 0).any()) or bool((image > 1).any())):
        raise ValueError('Krea references require one finite CPU RGB image, 16–512px, values 0–1')
    return image.detach().to(dtype=torch.float32).contiguous().clone()


class _RecordingVAE:
    def __init__(self, vae, encoded):
        self.vae, self.encoded = vae, encoded

    def encode(self, pixels):
        latent = self.vae.encode(pixels)
        self.last_pixels = _digest(pixels)
        self.encoded[self.last_pixels] = latent
        return latent


class _PreparedVAE:
    """Author pixel preparation can only replay an already encoded exact image."""
    def __init__(self, encoded):
        self.encoded = encoded

    def encode(self, pixels):
        digest = _digest(pixels)
        if digest not in self.encoded:
            raise RuntimeError('Identity reference attempted an unprepared VAE encode')
        return self.encoded[digest]


def prepare_reference(mode, source_image, ref_image=None, *, vae, torch, runtime=None,
                      ref_boost=4., grounding_size=512, spectral=False, output_size=(512, 512)):
    """Run only before fresh text encoding and diffusion residency."""
    output_size = validate_output_size(output_size)
    target_h, target_w = output_size[1] // 8, output_size[0] // 8
    if type(spectral) is not bool:
        raise ValueError('Krea reference spectral option must be boolean')
    if not math.isfinite(ref_boost) or not .1 <= ref_boost <= 10:
        raise ValueError('Krea reference boost must be finite and within .1–10')
    if grounding_size not in (384, 512, 640, 768):
        raise ValueError('Identity grounding must use a bounded trained resolution')
    if mode not in MODES or source_image is None:
        raise ValueError('Krea reference mode requires a source image')
    images = tuple(validate_image(x, torch) for x in (source_image, ref_image) if x is not None)
    encoded, latents, stages = {}, [], []
    author = load_edit_runtime(runtime) if mode == 'identity_edit' else None
    recording = _RecordingVAE(vae, encoded)
    for index, image in enumerate(images):
        if mode == 'identity_edit':
            latent = author._fit_encode_image(image, recording, target_h, target_w, {},
                (index, target_h, target_w), 'fit')
        else:
            # Bound the reference token budget without enlarging small sources.
            h, w = image.shape[1:3]
            nh, nw = max(16, h // 16 * 16), max(16, w // 16 * 16)
            pixels = torch.nn.functional.interpolate(image.movedim(-1, 1), size=(nh, nw),
                mode='area').movedim(1, -1).contiguous()
            latent = recording.encode(pixels)
        if (not isinstance(latent, torch.Tensor) or latent.ndim != 5
                or tuple(latent.shape[:3]) != (1, 16, 1)
                or min(latent.shape[-2:]) < 2 or max(latent.shape[-2:]) > 196
                or not bool(torch.isfinite(latent).all())):
            raise RuntimeError('Unexpected Krea reference VAE latent')
        latent = latent.detach().cpu()
        latents.append(latent)
        stages.append({'index': index, 'target': [target_h, target_w], 'pixels': recording.last_pixels})
        if spectral and author:
            half_h, half_w = target_h // 2, target_w // 2
            author._fit_encode_image(image, recording, half_h, half_w, {}, (index, half_h, half_w), 'fit')
            stages.append({'index': index, 'target': [half_h, half_w], 'pixels': recording.last_pixels})
        # Replay uses CPU latents; never retain a GPU allocation from VAE preparation.
        for key, value in list(encoded.items()):
            if value is not latent:
                encoded[key] = value.detach().cpu()
    metadata = {'mode': mode, 'spectral': spectral, 'reference_count': len(images), 'output_size': list(output_size),
        'images': [{'sha256': _digest(image), 'shape': list(image.shape)} for image in images],
        'latent_shapes': [list(latent.shape) for latent in latents],
        'latent_sha256': [_digest(latent) for latent in latents],
        'order': 'scene,subject' if mode == 'identity_edit' and len(images) == 2 else 'input-order',
        'author_revision': EDIT_REVISION if author else None,
        'reference_method': 'author-fit' if author else 'index_timestep_zero',
        'ref_boost': ref_boost if author else None,
        'grounding_size': grounding_size if author else 384}
    metadata['encoded'] = [{**stage, 'shape': list(encoded[stage['pixels']].shape),
                            'sha256': _digest(encoded[stage['pixels']])} for stage in stages]
    prepared = PreparedReference(mode, images, tuple(latents), encoded, metadata, author, output_size=output_size)
    _validate_prepared_tensors(prepared)
    return prepared


def _encode_grounded(clip, prompt, prepared, *, grounding_px=512):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Krea reference instruction must be nonempty')
    if grounding_px not in (384, 512, 640, 768):
        raise ValueError('Identity grounding must use a bounded trained resolution')
    if prepared.mode == 'identity_edit':
        return prepared.author.Krea2EditGroundedEncode().encode(clip, prompt,
            image=prepared.images[0], image_b=prepared.images[1] if len(prepared.images) == 2 else None,
            grounding_px=grounding_px)[0]
    # Reuse the core grounded template and image preparation, but VAE work was
    # completed separately at a bounded size before loading the text encoder.
    from comfy_extras.nodes_qwen import TextEncodeQwenImageEditPlus
    return TextEncodeQwenImageEditPlus.execute(clip, prompt, vae=None,
        image1=prepared.images[0], image2=prepared.images[1] if len(prepared.images) == 2 else None)[0]


def encode_grounded(nodes, *, mm, te, prompt, prepared, torch=None, lap=lambda name: None):
    """Fresh CPU encoder, uncached grounded embeddings, no retained encoder object."""
    from krea_conditioning import _observed_encoder, _validate_observed
    clip = None
    refs = []
    try:
        clip = nodes['CLIPLoader']().load_clip(te, 'krea2', 'default')[0]
        refs = [weakref.ref(x) for x in (clip, clip.cond_stage_model, clip.patcher)]
        lap('load_te')
        observed = _observed_encoder(clip)
        _validate_observed(observed)
        positive = _encode_grounded(clip, prompt, prepared,
            grounding_px=prepared.metadata['grounding_size'])
        negative = nodes['ConditioningZeroOut']().zero_out(positive)[0]
        lap('encode')
    finally:
        clip = None
        mm.unload_all_models()
        mm.soft_empty_cache()
    gc.collect()
    mm.cleanup_models_gc()
    if any(ref() is not None for ref in refs):
        raise RuntimeError('Krea grounded encoding retained its encoder or patcher')
    return positive, negative, {'origin': 'fresh-encoding', 'cache_policy': 'off',
        'scope': 'image-grounded Krea conditioning', 'encoder_refs_dead': True,
        'encoder_observed': observed, 'reference': prepared.metadata}


def apply_reference(patcher, positive, negative, prepared, *, torch=None, ref_boost=None):
    """Return candidate model/conditioning; no VAE or encoder may load here."""
    if torch is None:
        import torch
    _validate_prepared_tensors(prepared)
    if prepared.mode == 'identity_edit':
        if ref_boost is None:
            ref_boost = prepared.metadata['ref_boost']
        if not math.isfinite(ref_boost) or not .1 <= ref_boost <= 10:
            raise ValueError('Krea reference boost must be finite and within .1–10')
        candidate = prepared.author.Krea2EditModelPatch().patch(patcher,
            {'samples': prepared.latents[0]},
            source_latent_b={'samples': prepared.latents[1]} if len(prepared.latents) == 2 else None,
            vae=_PreparedVAE(prepared.encoded), source_image=prepared.images[0],
            source_image_b=prepared.images[1] if len(prepared.images) == 2 else None,
            target_latent={'samples': torch.zeros(1, 4, prepared.output_size[1] // 8, prepared.output_size[0] // 8)},
            fit_mode='fit', ref_boost=ref_boost)[0]
        prepared.options_signature = _reference_option_signature(candidate, prepared.mode)
        return candidate, positive, negative
    def attach(conditioning):
        result = []
        for tensor, values in conditioning:
            if 'reference_latents' in values or 'reference_latents_method' in values:
                raise ValueError('Krea conditioning already contains reference latents')
            result.append([tensor, {**values, 'reference_latents': list(prepared.latents),
                'reference_latents_method': 'index_timestep_zero'}])
        return result
    prepared.options_signature = _reference_option_signature(patcher, prepared.mode)
    return patcher, attach(positive), attach(negative)


def _reference_option_signature(patcher, mode):
    options = patcher.model_options
    if set(options) != {'transformer_options'}:
        raise ValueError('Krea reference model has unexpected model options')
    transformer = options['transformer_options']
    if mode == 'style_reference':
        if transformer:
            raise ValueError('Native style references require empty transformer options')
        return ('style_reference',)
    from comfy.patcher_extension import WrappersMP
    if set(transformer) != {'wrappers'}:
        raise ValueError('Identity edit only permits the pinned diffusion wrapper')
    wrappers = transformer['wrappers']
    if set(wrappers) != {WrappersMP.DIFFUSION_MODEL}:
        raise ValueError('Unexpected Identity Edit wrapper target')
    diffusion = wrappers[WrappersMP.DIFFUSION_MODEL]
    if set(diffusion) != {'krea2_edit'} or len(diffusion['krea2_edit']) != 1:
        raise ValueError('Unexpected Identity Edit wrapper count or key')
    wrapper = diffusion['krea2_edit'][0]
    if (not callable(wrapper) or wrapper.__module__ != '_mk_pinned_krea2edit'
            or wrapper.__qualname__ != 'Krea2EditModelPatch.patch.<locals>.wrapper'):
        raise ValueError('Invalid Identity Edit wrapper')
    return ('identity_edit', wrapper)


def _expected_encoded(prepared):
    """Replay pinned pixel geometry only; never construct or call a real VAE."""
    import torch
    rows = []
    class Geometry:
        def encode(self, pixels):
            self.pixels = _digest(pixels)
            self.shape = [1, 16, 1, pixels.shape[1] // 8, pixels.shape[2] // 8]
            return torch.zeros(self.shape)
    geometry = Geometry()
    target_h, target_w = prepared.output_size[1] // 8, prepared.output_size[0] // 8
    for index, image in enumerate(prepared.images):
        targets = ((target_h, target_w), (target_h // 2, target_w // 2)) if prepared.mode == 'identity_edit' and prepared.metadata['spectral'] else ((target_h, target_w),)
        for height, width in targets:
            if prepared.mode == 'identity_edit':
                prepared.author._fit_encode_image(image, geometry, height, width, {}, (index, height, width), 'fit')
            else:
                h, w = image.shape[1:3]
                pixels = torch.nn.functional.interpolate(image.movedim(-1, 1),
                    size=(max(16, h // 16 * 16), max(16, w // 16 * 16)), mode='area').movedim(1, -1).contiguous()
                geometry.encode(pixels)
            rows.append({'index': index, 'target': [height, width], 'pixels': geometry.pixels, 'shape': geometry.shape})
    return rows


def _validate_prepared_tensors(prepared):
    import torch
    metadata = prepared.metadata
    output_size = validate_output_size(prepared.output_size)
    if (metadata.get('output_size') != list(output_size)
            or type(metadata.get('spectral')) is not bool or metadata['mode'] != prepared.mode
            or len(prepared.images) not in (1, 2) or len(prepared.latents) != len(prepared.images)
            or [{'sha256': _digest(image), 'shape': list(image.shape)} for image in prepared.images] != metadata['images']
            or [list(x.shape) for x in prepared.latents] != metadata['latent_shapes']
            or [_digest(x) for x in prepared.latents] != metadata['latent_sha256']):
        raise ValueError('Krea prepared reference tensors changed after preparation')
    expected = _expected_encoded(prepared)
    if (len(metadata.get('encoded', [])) != len(expected)
            or set(prepared.encoded) != {row['pixels'] for row in expected}):
        raise ValueError('Krea prepared encoded reference membership changed')
    for row, bound in zip(expected, metadata['encoded']):
        value = prepared.encoded[row['pixels']]
        if (bound != {**row, 'sha256': _digest(value)} or list(value.shape) != row['shape']
                or value.device.type != 'cpu' or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Krea prepared encoded reference geometry or digest changed')
        if row['target'] == [output_size[1] // 8, output_size[0] // 8] and _digest(value) != metadata['latent_sha256'][row['index']]:
            raise ValueError('Krea full prepared reference differs from its ordered latent')


def validate_reference_options(patcher, prepared):
    _validate_prepared_tensors(prepared)
    if prepared.options_signature is None:
        raise ValueError('Reference options were not bound during preparation')
    actual = _reference_option_signature(patcher, prepared.mode)
    if len(actual) != len(prepared.options_signature) or any(
            a != b if isinstance(a, str) else a is not b
            for a, b in zip(actual, prepared.options_signature)):
        raise ValueError('Krea reference model options changed after binding')


def validate_spectral_reference(patcher, positive, negative, prepared):
    """Admit only bound owned reference semantics, without masks or other controls."""
    validate_reference_options(patcher, prepared)
    if prepared.metadata['spectral'] is not True:
        raise ValueError('Spectral references require spectral preparation')
    excluded = {'control', 'gligen', 'concat_latent_image', 'mask', 'area', 'hooks'}
    for conditioning in (positive, negative):
        if not conditioning:
            raise ValueError('Spectral references require both conditioning branches')
        for _, values in conditioning:
            if any(values.get(key) is not None for key in excluded):
                raise ValueError('Spectral references exclude control, masked and area conditioning')
            if prepared.mode == 'identity_edit':
                if 'reference_latents' in values or 'reference_latents_method' in values:
                    raise ValueError('Identity Edit cannot combine native reference conditioning')
            else:
                actual = values.get('reference_latents')
                if (not isinstance(actual, list) or len(actual) != len(prepared.latents)
                        or any(a is not b for a, b in zip(actual, prepared.latents))
                        or values.get('reference_latents_method') != 'index_timestep_zero'):
                    raise ValueError('Spectral style references differ from the owned ordered tensors')


encode_reference_conditioning = encode_grounded


def _file_identity(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _worker_sources():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('krea_reference.py', 'krea_reference_worker.py', 'probe_text_encoder_load.py')}


def _worker_comfy_binding(comfy_root):
    """Bind the caller's selected Comfy checkout, never an inherited override."""
    root = Path(comfy_root).expanduser().resolve(strict=True)
    sources = {}
    for name in ('sd.py', 'utils.py'):
        path = (root / 'comfy' / name).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError('Krea reference worker Comfy source is outside its selected root')
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'comfy_root': str(root), 'comfy_sources': sources}


def _validate_worker_comfy_modules(comfy_root, sd, utils):
    root = Path(comfy_root).resolve(strict=True)
    for module, name in ((sd, 'sd.py'), (utils, 'utils.py')):
        path = Path(module.__file__).resolve(strict=True)
        expected = (root / 'comfy' / name).resolve(strict=True)
        if not path.is_relative_to(root) or path != expected:
            raise RuntimeError('Krea reference worker imported a foreign Comfy module: ' + name)


def _run_reference_worker(command, log, *, timeout=120):
    """Wait for the owned same-group child to exit before returning any latents."""
    import subprocess
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        child.wait(timeout=timeout)
        if child.returncode:
            raise RuntimeError(f'Krea CPU reference worker failed with exit {child.returncode}')
        return child.pid
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)


def _write_reference_result(prepared, directory, *, torch, bindings, pid):
    from safetensors.torch import save_file
    tensors, names, encoded = {}, {}, {}
    def name(value):
        value = value.detach().cpu().contiguous()
        signature = (str(value.dtype), tuple(value.shape), _digest(value))
        if signature not in names:
            key = f'latent_{len(names)}'
            names[signature] = key
            tensors[key] = value.clone()
        return names[signature]
    for pixels, latent in prepared.encoded.items():
        encoded[pixels] = name(latent)
    latent_keys = [name(latent) for latent in prepared.latents]
    save_file(tensors, str(Path(directory) / 'latents.safetensors'))
    result = {'metadata': prepared.metadata, 'encoded': encoded, 'latent_keys': latent_keys,
              'tensors': {key: {'sha256': _digest(value), 'shape': list(value.shape),
                               'dtype': str(value.dtype)} for key, value in tensors.items()},
              'bindings': bindings, 'pid': pid, 'group': os.getpgrp()}
    (Path(directory) / 'result.json').write_text(json.dumps(result) + '\n')


def _read_reference_result(directory, *, images, mode, torch, bindings, pid,
                           runtime, ref_boost, grounding_size, spectral=False, output_size=(512, 512)):
    from safetensors.torch import load_file
    output_size = validate_output_size(output_size)
    result = json.loads((Path(directory) / 'result.json').read_text())
    metadata = result['metadata']
    expected_order = 'scene,subject' if mode == 'identity_edit' and len(images) == 2 else 'input-order'
    if (result['bindings'] != bindings or result['pid'] != pid or result['group'] != os.getpgrp()
            or metadata.get('spectral') is not spectral or metadata['mode'] != mode or metadata['reference_count'] != len(images)
            or metadata['images'] != [{'sha256': _digest(x), 'shape': list(x.shape)} for x in images]
            or metadata['order'] != expected_order or metadata['output_size'] != list(output_size)
            or metadata['author_revision'] != (EDIT_REVISION if mode == 'identity_edit' else None)
            or metadata['reference_method'] != ('author-fit' if mode == 'identity_edit' else 'index_timestep_zero')
            or metadata['ref_boost'] != (ref_boost if mode == 'identity_edit' else None)
            or metadata['grounding_size'] != (grounding_size if mode == 'identity_edit' else 384)):
        raise RuntimeError('Krea CPU reference worker result differs from its admitted request')
    tensors = load_file(str(Path(directory) / 'latents.safetensors'), device='cpu')
    if (not 1 <= len(tensors) <= len(images) * (2 if spectral and mode == 'identity_edit' else 1) or set(tensors) != set(result['tensors'])
            or len(result['latent_keys']) != len(images)
            or not 1 <= len(result['encoded']) <= len(images) * (2 if spectral and mode == 'identity_edit' else 1)
            or set(result['latent_keys']) | set(result['encoded'].values()) != set(tensors)):
        raise RuntimeError('Krea CPU reference worker has unexpected latent membership')
    for key, value in tensors.items():
        if (value.dtype != torch.float32 or value.ndim != 5 or tuple(value.shape[:3]) != (1, 16, 1)
                or min(value.shape[-2:]) < 2 or max(value.shape[-2:]) > 196
                or not bool(torch.isfinite(value).all())
                or result['tensors'][key] != {'sha256': _digest(value), 'shape': list(value.shape),
                                             'dtype': str(value.dtype)}):
            raise RuntimeError('Krea CPU reference worker produced invalid latent tensors')
    latents = tuple(tensors[key] for key in result['latent_keys'])
    if (metadata['latent_shapes'] != [list(x.shape) for x in latents]
            or metadata['latent_sha256'] != [_digest(x) for x in latents]):
        raise RuntimeError('Krea CPU reference worker latent ordering or hashes differ')
    encoded = {key: tensors[value] for key, value in result['encoded'].items()}
    if any(not isinstance(key, str) or len(key) != 64 or any(x not in '0123456789abcdef' for x in key)
           for key in encoded):
        raise RuntimeError('Krea CPU reference worker has invalid prepared pixel hashes')
    author = load_edit_runtime(runtime) if mode == 'identity_edit' else None
    metadata = {**metadata, 'vae_preparation': {'mechanism': 'owned-cpu-process',
                'worker_pid': pid, 'worker_exited': True, 'vae_identity': bindings['vae_identity'],
                'sources': bindings['sources'], 'comfy_root': bindings['comfy_root'],
                'comfy_sources': bindings['comfy_sources']}}
    prepared = PreparedReference(mode, images, latents, encoded, metadata, author, output_size=output_size)
    try:
        _validate_prepared_tensors(prepared)
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError("Krea CPU reference worker encoded receipt differs") from error
    return prepared


def prepare_reference_owned(mode, source_image, ref_image=None, *, vae_path, root, torch, comfy_root,
                            runtime=None, ref_boost=4., grounding_size=512, spectral=False, output_size=(512, 512)):
    """Prepare CPU VAE latents in a child that exits before encoder construction."""
    import subprocess
    import sys
    import tempfile
    from safetensors.torch import save_file
    if __package__:
        from .probe_text_encoder_load import require_supervisor
    else:
        from probe_text_encoder_load import require_supervisor
    require_supervisor()
    output_size = validate_output_size(output_size)
    if (type(spectral) is not bool or mode not in MODES or source_image is None or not math.isfinite(ref_boost)
            or not .1 <= ref_boost <= 10 or grounding_size not in (384, 512, 640, 768)):
        raise ValueError('Invalid owned Krea reference preparation request')
    images = tuple(validate_image(value, torch) for value in (source_image, ref_image) if value is not None)
    path = Path(vae_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError('Krea CPU reference worker requires an installed VAE asset')
    runtime = str(Path(runtime or os.environ.get('KREA_EDIT_RUNTIME', '')).expanduser().resolve(strict=True)) if mode == 'identity_edit' else None
    if mode == 'identity_edit':
        load_edit_runtime(runtime)  # Pin author bytes before child admission.
    comfy_binding = _worker_comfy_binding(comfy_root)
    bindings = {'vae_path': str(path), 'vae_identity': _file_identity(path),
                'sources': _worker_sources(), **comfy_binding}
    output = (Path(root) / 'out').resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='krea-reference-', dir=output) as temporary:
        directory = Path(temporary)
        save_file({f'image_{index}': value for index, value in enumerate(images)}, str(directory / 'images.safetensors'))
        config = {'mode': mode, 'spectral': spectral, 'runtime': runtime, 'ref_boost': ref_boost,
                  'output_size': list(output_size),
                  'grounding_size': grounding_size, 'bindings': bindings,
                  'images': [{'sha256': _digest(value), 'shape': list(value.shape)} for value in images]}
        (directory / 'config.json').write_text(json.dumps(config) + '\n')
        try:
            with (directory / 'worker.log').open('wb') as log:
                pid = _run_reference_worker([sys.executable, str(Path(__file__).with_name('krea_reference_worker.py')),
                                             str(directory)], log)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            detail = (directory / 'worker.log').read_text(errors='replace')[-4000:]
            raise RuntimeError(f'Owned Krea reference preparation failed: {error}\n{detail}') from error
        if (_file_identity(path) != bindings['vae_identity'] or _worker_sources() != bindings['sources']
                or _worker_comfy_binding(comfy_root) != comfy_binding):
            raise RuntimeError('Krea reference worker source or VAE asset changed during preparation')
        return _read_reference_result(directory, images=images, mode=mode, torch=torch,
            bindings=bindings, pid=pid, runtime=runtime, ref_boost=ref_boost, grounding_size=grounding_size,
            spectral=spectral, output_size=output_size)
