"""Saved arguments and real image preprocessing, without ComfyUI or model loading."""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageChops

_spec = importlib.util.spec_from_file_location(
    'training_config_subject', Path(__file__).resolve().parents[1] / 'train' / 'training_config.py'
)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['qwen21', 'krea2'], default='krea2')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--lr', type=float, default=.0002)
    p.add_argument('--dtype', choices=['fp16', 'bf16'], default=None)
    p.add_argument('--dataset', default='default-data')
    p.add_argument('--cache-file', default=None)
    p.add_argument('--out', default='default-out')
    p.add_argument('--chunked-attn', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--buckets', type=config.normalize_buckets, default=None)
    return p


def test_json_defaults_explicit_overrides_and_relative_paths(tmp_path):
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps({'arch': 'qwen21', 'steps': 30, 'chunked_attn': True,
                               'dataset': 'images', 'out': 'runs/one', 'buckets': [[64, 32], [32, 64]]}))
    args = config.parse_config_args(parser(), ['--config', str(path), '--steps', '7', '--no-chunked-attn',
                                              '--out', 'cli-output'])
    assert args.arch == 'qwen21' and args.steps == 7 and args.chunked_attn is False
    assert args.dataset == str(tmp_path / 'images') and args.out == str(Path.cwd() / 'cli-output')
    assert args.buckets == [[32, 64], [64, 32]]


def test_boolean_override_can_enable_file_default(tmp_path):
    path = tmp_path / 'recipe.json'
    path.write_text('{"chunked_attn": false}')
    assert config.parse_config_args(parser(), ['--config', str(path), '--chunked-attn']).chunked_attn


def test_explicit_cache_paths_follow_config_then_cli_invocation(tmp_path):
    recipes = tmp_path / 'recipes'
    recipes.mkdir()
    path = recipes / 'recipe.json'
    path.write_text('{"cache_file": "../corrected.pt"}')
    saved = config.parse_config_args(parser(), ['--config', str(path)], base_dir=tmp_path)
    overridden = config.parse_config_args(parser(), ['--config', str(path), '--cache-file', 'override.pt'],
                                         base_dir=tmp_path)
    assert Path(saved.cache_file).resolve() == tmp_path / 'corrected.pt'
    assert overridden.cache_file == str(tmp_path / 'override.pt')
    assert config.parse_config_args(parser(), [], base_dir=tmp_path).cache_file is None
    path.write_text('{"cache_file": null}')
    assert config.parse_config_args(parser(), ['--config', str(path)], base_dir=tmp_path).cache_file is None


@pytest.mark.parametrize('value', [
    '{"unknown": 1}', '{"steps": "30"}', '{"steps": true}', '{"lr": NaN}',
    '{"chunked_attn": "false"}', '{"arch": "wrong"}', '{"config": "recursive.json"}',
    '{"steps": 20, "steps": 40}', '[]', '{"buckets": [[64, 31]]}', '{"dataset": null}',
])
def test_invalid_file_fails_early(tmp_path, value):
    path = tmp_path / 'bad.json'
    path.write_text(value)
    with pytest.raises(SystemExit):
        config.parse_config_args(parser(), ['--config', str(path)])


def test_nonfinite_cli_value_fails():
    with pytest.raises(SystemExit):
        config.parse_config_args(parser(), ['--lr', 'nan'])


@pytest.mark.parametrize('value', ['', [], [[64, 64], [64, 64]], [[True, 64]], [[0, 64]], [[64.0, 64]], '64x33'])
def test_invalid_buckets(value):
    with pytest.raises(ValueError):
        config.normalize_buckets(value)


def test_bucket_names_are_order_independent_and_separate_from_square(tmp_path):
    a = config.cache_filename(tmp_path, 'qwen21', 512, [[512, 768], [768, 512]])
    b = config.cache_filename(tmp_path, 'qwen21', 1024, '768x512,512x768')
    assert a == b  # res is unused when explicit buckets define the shape policy.
    assert a != config.cache_filename(tmp_path, 'qwen21', 512)
    assert a != config.cache_filename(tmp_path, 'qwen21', 512, [[512, 512]])


def test_square_cache_legacy_name_preserved(tmp_path):
    assert config.cache_filename(tmp_path, 'krea2', 512).endswith('cache_krea2_512.pt')
    (tmp_path / 'cache_krea2.pt').touch()
    assert config.cache_filename(tmp_path, 'krea2', 512).endswith('cache_krea2.pt')
    assert config.cache_filename(tmp_path, 'krea2', 1024).endswith('cache_krea2_1024.pt')


def test_bucket_selection_portrait_landscape_and_ties():
    buckets = [[512, 1024], [1024, 512]]
    assert config.choose_bucket((200, 400), buckets) == (512, 1024)
    assert config.choose_bucket((400, 200), buckets) == (1024, 512)
    assert config.choose_bucket((400, 400), buckets) == (512, 1024)
    assert config.choose_bucket((400, 200), [[1024, 512], [512, 256]]) == (512, 256)


def test_real_image_center_crop_is_deterministic():
    image = Image.new('RGB', (96, 32), 'green')
    image.paste('red', (0, 0, 32, 32))
    image.paste('blue', (64, 0, 96, 32))
    first, geometry = config.prepare_image(image, 512, [[64, 64]])
    second, again = config.prepare_image(image, 512, [[64, 64]])
    assert first.size == (64, 64) and first.getpixel((32, 32)) == (0, 128, 0)
    assert geometry['resized_size'] == [192, 64] and geometry['crop_box'] == [64, 0, 128, 64]
    assert ImageChops.difference(first, second).getbbox() is None and geometry == again


def test_exif_orientation_precedes_bucket_selection():
    image = Image.new('RGB', (96, 32), 'red')
    image.getexif()[274] = 6  # Rotate 90 degrees: portrait after orientation.
    got, geometry = config.prepare_image(image, 512, [[32, 96], [96, 32]])
    assert got.size == (32, 96) and geometry['oriented_size'] == [32, 96]
    assert geometry['source_size'] == [96, 32]


def test_legacy_square_pixels_are_unchanged():
    image = Image.new('RGBA', (95, 37), (120, 40, 220, 80))
    image.paste((0, 240, 0, 255), (10, 3, 39, 22))
    image.getexif()[274] = 6  # Old square mode does not transpose EXIF.
    got, _ = config.prepare_image(image, 64)
    want = image.convert('RGB').resize((64, 64), Image.Resampling.LANCZOS)
    assert ImageChops.difference(got, want).getbbox() is None


def mixed_cache():
    policy = config.shape_policy(512, [[64, 32], [32, 64]])
    cache = {'shape_policy': policy, 'latents': [torch.zeros(1, 4, 4, 8), torch.zeros(1, 4, 8, 4)],
             'conds': [torch.zeros(1, 3, 16), torch.zeros(1, 5, 16)],
             'samples': [{'target_size': [64, 32], 'latent_shape': [1, 4, 4, 8]},
                         {'target_size': [32, 64], 'latent_shape': [1, 4, 8, 4]}]}
    return cache, policy


def test_mixed_cache_accepts_variable_shapes_and_text_lengths():
    cache, policy = mixed_cache()
    config.validate_bucket_cache(cache, policy)
    config.validate_bucket_cache({'latents': [torch.zeros(1)]}, config.shape_policy(512))


@pytest.mark.parametrize('defect', ['policy', 'missing_sample', 'latent_shape', 'target', 'ratio', 'condition', 'compression'])
def test_cache_rejects_stale_or_inconsistent_geometry(defect):
    cache, policy = mixed_cache()
    if defect == 'policy':
        cache['shape_policy'] = config.shape_policy(512, [[32, 32]])
    elif defect == 'missing_sample':
        cache['samples'].pop()
    elif defect == 'latent_shape':
        cache['samples'][0]['latent_shape'] = [1, 4, 8, 4]
    elif defect == 'target':
        cache['samples'][0]['target_size'] = [32, 32]
    elif defect == 'ratio':
        cache['latents'][0] = torch.zeros(1, 4, 4, 4)
        cache['samples'][0]['latent_shape'] = [1, 4, 4, 4]
    elif defect == 'compression':
        cache['latents'][1] = torch.zeros(1, 4, 4, 2)
        cache['samples'][1]['latent_shape'] = [1, 4, 4, 2]
    elif defect == 'condition':
        cache['conds'][1] = torch.zeros(1, 5, 17)
    with pytest.raises(ValueError):
        config.validate_bucket_cache(cache, policy)


def test_relative_config_uses_invocation_directory_after_boot_changes_cwd(tmp_path, monkeypatch):
    invocation = tmp_path / 'caller'
    invocation.mkdir()
    (invocation / 'recipe.json').write_text('{"steps": 9, "dataset": "images"}')
    monkeypatch.chdir(tmp_path)
    args = config.parse_config_args(parser(), ['--config', 'recipe.json', '--out', 'new-run'], base_dir=invocation)
    assert args.steps == 9
    assert args.dataset == str(invocation / 'images') and args.out == str(invocation / 'new-run')
