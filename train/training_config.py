"""Saved CLI defaults and deterministic image shapes for the batch-one trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageOps


def normalize_buckets(value):
    """Canonical width/height pairs, accepted as JSON pairs or comma-separated WxH."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = [[int(v) for v in pair.lower().split('x')] for pair in value.split(',')]
        except ValueError:
            raise ValueError('buckets must be comma-separated WIDTHxHEIGHT pairs') from None
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError('buckets must be a nonempty list of [width, height] pairs')
    pairs = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError('each bucket must contain width and height')
        if any(type(v) is not int or v <= 0 or v % 32 for v in pair):
            raise ValueError('bucket dimensions must be positive integer multiples of 32')
        pairs.append(tuple(pair))
    if len(set(pairs)) != len(pairs):
        raise ValueError('duplicate buckets are not allowed')
    return [list(pair) for pair in sorted(pairs)]


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f'duplicate configuration key: {key}')
        result[key] = value
    return result


def _config_value(action, value):
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
        if type(value) is not bool:
            raise ValueError(f'{action.dest} must be a JSON boolean')
    elif action.type is normalize_buckets:
        value = normalize_buckets(value)
    elif action.type is int:
        if type(value) is not int:
            raise ValueError(f'{action.dest} must be a JSON integer')
    elif action.type is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{action.dest} must be a finite JSON number')
        value = float(value)
    elif not (value is None and action.default is None) and not isinstance(value, str):
        raise ValueError(f'{action.dest} must be a JSON string')
    if action.choices is not None and value is not None and value not in action.choices:
        raise ValueError(f'{action.dest} must be one of {list(action.choices)}')
    return value


def parse_config_args(parser, argv=None, *, base_dir=None):
    """Parse JSON defaults through the existing CLI schema; explicit CLI values win.

    The caller defines every ordinary option, including --buckets. This function adds
    --config. File-supplied dataset/out/cache_file paths are relative to the JSON file.
    """
    base_dir = Path(base_dir or Path.cwd()).resolve()
    parser.add_argument('--config', help='JSON argument defaults; explicit CLI options override them')
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config')
    config_arg, _ = pre.parse_known_args(argv)
    if config_arg.config:
        path = Path(config_arg.config).expanduser()
        path = (path if path.is_absolute() else base_dir / path).resolve()
        try:
            values = json.loads(path.read_text(), object_pairs_hook=_unique_object)
            if not isinstance(values, dict):
                raise ValueError('configuration must be a JSON object')
            actions = {}
            for action in parser._actions:
                if action.dest not in ('help', 'config'):
                    actions.setdefault(action.dest, action)
            unknown = values.keys() - actions.keys()
            if unknown:
                raise ValueError(f'unknown configuration keys: {", ".join(sorted(unknown))}')
            values = {key: _config_value(actions[key], value) for key, value in values.items()}
            for key in ('dataset', 'out', 'cache_file'):
                if key in values and values[key] is not None:
                    candidate = Path(values[key]).expanduser()
                    values[key] = str(candidate if candidate.is_absolute() else path.parent / candidate)
            parser.set_defaults(**values)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(f'{path}: {exc}')
    args = parser.parse_args(argv)
    if config_arg.config:
        args.config = str(path)
    for key in ('dataset', 'out', 'cache_file'):
        if getattr(args, key, None) is not None:
            value = Path(getattr(args, key)).expanduser()
            setattr(args, key, str(value if value.is_absolute() else base_dir / value))
    for key, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            parser.error(f'{key} must be finite')
    return args


def shape_policy(resolution, buckets=None):
    buckets = normalize_buckets(buckets)
    if buckets is None:
        return {'version': 1, 'mode': 'square', 'resolution': resolution, 'resize': 'stretch-lanczos'}
    return {'version': 1, 'mode': 'buckets', 'buckets': buckets, 'orientation': 'exif',
            'assignment': 'nearest-log-aspect-smallest-area', 'resize': 'cover-lanczos-ceil',
            'crop': 'center-floor'}


def cache_filename(dataset, arch, resolution, buckets=None):
    dataset = Path(dataset)
    policy = shape_policy(resolution, buckets)
    if policy['mode'] == 'square':
        legacy = dataset / 'cache_krea2.pt'
        if arch == 'krea2' and resolution == 512 and legacy.exists():
            return str(legacy)
        return str(dataset / f'cache_{arch}_{resolution}.pt')
    encoded = json.dumps(policy, sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return str(dataset / f'cache_{arch}_buckets_{digest}.pt')


def choose_bucket(size, buckets):
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError('source image dimensions must be positive')
    pairs = normalize_buckets(buckets)
    if pairs is None:
        raise ValueError('bucket selection requires explicit buckets')
    ratio = math.log(width / height)
    return tuple(min(pairs, key=lambda pair: (
        abs(math.log(pair[0] / pair[1]) - ratio), pair[0] * pair[1], pair[0], pair[1]
    )))


def prepare_image(image, resolution, buckets=None):
    """Return RGB pixels and serializable geometry; no randomness enters preprocessing."""
    source = image.size
    policy = shape_policy(resolution, buckets)
    if policy['mode'] == 'square':
        image = image.convert('RGB')
        target = (resolution, resolution)
        if image.size != target:
            image = image.resize(target, Image.Resampling.LANCZOS)
        return image, {'source_size': list(source), 'oriented_size': list(source),
                       'target_size': list(target), 'resized_size': list(target),
                       'crop_box': [0, 0, *target]}
    image = ImageOps.exif_transpose(image).convert('RGB')
    width, height = oriented = image.size
    tw, th = choose_bucket(oriented, policy['buckets'])
    # Integer ceil preserves the exact cover bound even for awkward source dimensions.
    if tw * height >= th * width:
        resized = (tw, (height * tw + width - 1) // width)
    else:
        resized = ((width * th + height - 1) // height, th)
    left, top = (resized[0] - tw) // 2, (resized[1] - th) // 2
    box = (left, top, left + tw, top + th)
    image = image.resize(resized, Image.Resampling.LANCZOS).crop(box)
    return image, {'source_size': list(source), 'oriented_size': list(oriented),
                   'target_size': [tw, th], 'resized_size': list(resized), 'crop_box': list(box)}


def validate_bucket_cache(cache, policy):
    """Check new bucket metadata before model loading; legacy square caches stay valid."""
    if policy['mode'] == 'square':
        if cache.get('shape_policy', policy) != policy:
            raise ValueError('cache shape policy does not match square training')
        return
    if cache.get('shape_policy') != policy:
        raise ValueError('cache shape policy does not match configured buckets')
    latents, conds, samples = cache.get('latents', []), cache.get('conds', []), cache.get('samples', [])
    if not latents or len(latents) != len(conds) or len(samples) != len(latents):
        raise ValueError('bucket cache must have one shape record and condition per latent')
    for i, (latent, cond, sample) in enumerate(zip(latents, conds, samples, strict=True)):
        target = sample.get('target_size')
        if target not in policy['buckets'] or sample.get('latent_shape') != list(latent.shape):
            raise ValueError(f'bucket cache sample {i} has inconsistent shape metadata')
        if latent.ndim not in (4, 5) or latent.shape[0] != 1 or latent.shape[:-2] != latents[0].shape[:-2]:
            raise ValueError(f'bucket cache sample {i} has inconsistent latent dimensions')
        width, height = target
        lh, lw = latent.shape[-2:]
        if lh <= 0 or lw <= 0 or width % lw or height % lh or width // lw != height // lh:
            raise ValueError(f'bucket cache sample {i} latent aspect ratio does not match its target')
        if i and width // lw != samples[0]['target_size'][0] // latents[0].shape[-1]:
            raise ValueError(f'bucket cache sample {i} has inconsistent VAE compression')
        if cond.ndim < 2 or cond.shape[0] != 1 or cond.shape[-1] != conds[0].shape[-1]:
            raise ValueError(f'bucket cache sample {i} has inconsistent conditioning dimensions')
