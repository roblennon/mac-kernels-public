"""Real file and CPU-image admission at the user-facing workbench boundary."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from krea_workbench import SPECIAL, SPECIAL_SHA256, selections, validate_input_files, fit_input_image, validate_mode_adapter


@pytest.mark.parametrize('mode', tuple(SPECIAL))
def test_reference_adapter_is_bound_to_its_trained_release(mode):
    validate_mode_adapter(mode, {'adapters': [{'sha256': SPECIAL_SHA256[mode]}]})
    with pytest.raises(ValueError, match='qualified workflow release'):
        validate_mode_adapter(mode, {'adapters': [{'sha256': 'a-renamed-old-release'}]})
    with pytest.raises(ValueError, match='qualified workflow release'):
        validate_mode_adapter(mode, {'adapters': []})


@pytest.fixture
def local_files(tmp_path):
    root = tmp_path / 'profile'
    loras = root / 'models/loras'
    inputs = root / 'input'
    loras.mkdir(parents=True)
    inputs.mkdir()
    for name in (*SPECIAL.values(), 'style_a.safetensors', 'style_b.safetensors'):
        save_file({'local_fixture': torch.ones(1)}, str(loras / name))

    def annotated(name):
        if name.endswith(' [input]'):
            name = name[:-8]
        return str(inputs / name)

    folders = SimpleNamespace(
        get_full_path_or_raise=lambda kind, name: str(loras / name),
        get_input_directory=lambda: str(inputs), get_annotated_filepath=annotated)
    return root, loras, inputs, folders


def test_selection_preserves_order_individual_strengths_and_automatic_adapter(local_files):
    root, loras, _, folders = local_files
    values = {'lora_1': 'style_b.safetensors', 'strength_1': -.25,
              'lora_3': 'style_a.safetensors', 'strength_3': .8}
    text = selections(root, folders, 'text', values)
    assert text == [{'name': 'style_b.safetensors', 'strength': -.25,
                     'path': str(loras / 'style_b.safetensors')},
                    {'name': 'style_a.safetensors', 'strength': .8,
                     'path': str(loras / 'style_a.safetensors')}]
    edit = selections(root, folders, 'identity_edit', values)
    assert edit[0] == {'name': SPECIAL['identity_edit'], 'strength': 1.,
                       'path': str(loras / SPECIAL['identity_edit'])}
    assert edit[1:] == text
    assert selections(root, folders, 'text', {}) == []


def test_mode_adapter_counts_toward_five_and_wrong_folder_resolution_fails(local_files, tmp_path):
    root, _, _, folders = local_files
    values = {f'lora_{index}': 'style_a.safetensors' for index in range(1, 6)}
    assert len(selections(root, folders, 'text', values)) == 5
    with pytest.raises(ValueError, match='At most five'):
        selections(root, folders, 'style_reference', values)
    other = tmp_path / 'style_a.safetensors'
    save_file({'local_fixture': torch.ones(1)}, str(other))
    mismatched = SimpleNamespace(get_full_path_or_raise=lambda kind, name: str(other))
    with pytest.raises(RuntimeError, match='differs from the profile root'):
        selections(root, mismatched, 'text', {'lora_1': 'style_a.safetensors'})


@pytest.mark.parametrize('values,error', [
    ({'lora_1': 'missing.safetensors'}, FileNotFoundError),
    ({'lora_1': '../style_a.safetensors'}, ValueError),
    ({'lora_1': 'style_a.safetensors', 'strength_1': float('nan')}, ValueError),
])
def test_file_selection_refuses_missing_unsafe_and_invalid_strength(local_files, values, error):
    root, _, _, folders = local_files
    with pytest.raises(error):
        selections(root, folders, 'text', values)


def image_graph(name):
    return {'5': {'class_type': 'LoadImage', 'inputs': {'image': name}}}


def test_uploaded_image_validation_uses_actual_file_contents_and_symlink_owner(local_files, tmp_path):
    _, _, inputs, folders = local_files
    Image.new('RGB', (32, 16), 'blue').save(inputs / 'source.png')
    validate_input_files(image_graph('source.png [input]'), folders)
    outside = tmp_path / 'outside.png'
    Image.new('RGB', (32, 16), 'blue').save(outside)
    (inputs / 'outside-link.png').symlink_to(outside)
    with pytest.raises(ValueError, match='uploaded image'):
        validate_input_files(image_graph('outside-link.png'), folders)
    with pytest.raises(ValueError, match='uploaded image'):
        validate_input_files(image_graph('../outside.png'), folders)
    with pytest.raises(ValueError, match='existing readable'):
        validate_input_files(image_graph('missing.png'), folders)
    (inputs / 'corrupt.png').write_bytes(b'this is not an image')
    with pytest.raises(ValueError, match='existing readable'):
        validate_input_files(image_graph('corrupt.png'), folders)


def test_upload_pixel_and_byte_limits_are_checked_before_reference_encoding(local_files):
    _, _, inputs, folders = local_files
    for name, size in (('too-small.png', (15, 32)), ('too-many-pixels.png', (3000, 3000))):
        Image.new('RGB', size).save(inputs / name)
        with pytest.raises(ValueError, match='minimum sides|eight megapixels'):
            validate_input_files(image_graph(name), folders)
    oversized = inputs / 'oversized.png'
    with oversized.open('wb') as stream:
        stream.truncate(16 * 2**20 + 1)
    with pytest.raises(ValueError, match='16 MiB'):
        validate_input_files(image_graph(oversized.name), folders)


def test_reference_fit_preserves_small_tensor_and_downsamples_aspect_without_value_drift():
    assert fit_input_image(None, torch) is None
    small = torch.full((1, 64, 32, 3), .4)
    assert fit_input_image(small, torch) is small
    large = torch.full((1, 384, 768, 3), .4)
    before = large.clone()
    fitted = fit_input_image(large, torch)
    assert tuple(fitted.shape) == (1, 256, 512, 3)
    assert fitted.is_contiguous() and fitted.device.type == 'cpu'
    torch.testing.assert_close(fitted, torch.full_like(fitted, .4))
    torch.testing.assert_close(large, before)
    narrow = fit_input_image(torch.ones(1, 16, 1024, 3), torch)
    assert tuple(narrow.shape) == (1, 16, 512, 3)


@pytest.mark.parametrize('image', [
    torch.zeros(2, 16, 16, 3), torch.zeros(1, 16, 16, 4),
    torch.zeros(1, 15, 16, 3), torch.zeros(1, 16, 16, 3, dtype=torch.uint8),
    torch.full((1, 16, 16, 3), float('nan')), torch.full((1, 16, 16, 3), 1.01),
    torch.full((1, 16, 16, 3), -.01), torch.empty(1, 16, 16, 3, device='meta'),
])
def test_reference_fit_rejects_invalid_real_tensor_boundaries(image):
    with pytest.raises(ValueError, match='finite CPU RGB'):
        fit_input_image(image, torch)
