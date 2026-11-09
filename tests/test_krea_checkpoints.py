"""Real file admission and checkpoint lifecycle without GPU allocation."""
import json
import struct
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from krea_checkpoints import CheckpointCatalog, DEFAULT_CHECKPOINT, EXPORT_METADATA, QUANTIZATION, reject_baked_replay
from krea_workbench import WorkbenchProfile
from krea_server import resolve_profile_assets, identity


@pytest.fixture
def profile_files(tmp_path):
    root = tmp_path / 'profile'
    models = root / 'models/diffusion_models'
    models.mkdir(parents=True)
    state = {'blocks.0.attn.wq.weight': torch.ones((4, 256), dtype=torch.int8),
        'blocks.0.attn.wq.weight_scale': torch.ones((4, 1)),
        'blocks.0.attn.wq.comfy_quant': torch.tensor(list(json.dumps(QUANTIZATION).encode()), dtype=torch.uint8),
        'first.weight': torch.ones((4, 16), dtype=torch.bfloat16)}
    save_file(state, models / DEFAULT_CHECKPOINT)
    provenance = {'schema': 1, 'architecture': 'krea2', 'base_variant': 'turbo',
        'quantization': 'int8-convrot-group256', 'baked_loras': []}
    custom = models / 'my_turbo.safetensors'
    save_file(state, custom, metadata={EXPORT_METADATA: json.dumps(provenance)})
    folders = SimpleNamespace(get_full_path_or_raise=lambda kind, name: str(root / 'models' / kind / name))
    return root, folders, custom, state, provenance


def test_catalog_admits_export_and_rejects_quantization_shape_or_receipt_drift(profile_files):
    root, folders, custom, state, provenance = profile_files
    catalog = CheckpointCatalog(root, folders)
    assert catalog.names() == [DEFAULT_CHECKPOINT, custom.name]
    assert catalog.resolve(custom.name)['provenance'] == provenance
    assert catalog.resolve(DEFAULT_CHECKPOINT)['provenance'] is None
    for name in ('../my_turbo.safetensors', str(custom), 'missing.safetensors'):
        with pytest.raises(ValueError, match='admitted'):
            catalog.resolve(name)
    for mutate in (
        lambda s, p: p.update(base_variant='raw'),
        lambda s, p: s.update({'first.weight': torch.ones((4, 32), dtype=torch.bfloat16)}),
        lambda s, p: s.update({'blocks.0.attn.wq.comfy_quant': torch.tensor(list(b'{"format":"int8_tensorwise"}'), dtype=torch.uint8)}),
        lambda s, p: p.update(baked_loras=[{'sha256': 'a'*64, 'strength': float('inf')}]),
    ):
        changed, metadata = dict(state), dict(provenance)
        mutate(changed, metadata)
        save_file(changed, custom, metadata={EXPORT_METADATA: json.dumps(metadata)})
        failed = CheckpointCatalog(root, folders)
        assert failed.names() == [DEFAULT_CHECKPOINT]
        assert custom.name in failed.rejected


def test_admitted_file_replacement_and_foreign_folder_resolution_reject(profile_files, tmp_path):
    root, folders, custom, state, provenance = profile_files
    catalog = CheckpointCatalog(root, folders)
    save_file(state, custom, metadata={EXPORT_METADATA: json.dumps(provenance)})
    with pytest.raises(RuntimeError, match='changed'):
        catalog.resolve(custom.name)
    catalog = CheckpointCatalog(root, folders)
    foreign = tmp_path / custom.name
    save_file(state, foreign)
    catalog.folder_paths = SimpleNamespace(get_full_path_or_raise=lambda kind, name: str(foreign))
    with pytest.raises(RuntimeError, match='profile root'):
        catalog.resolve(custom.name)


def test_baked_replay_uses_content_digest_even_after_adapter_rename():
    entry = {'provenance': {'baked_loras': [{'name': 'original.safetensors', 'sha256': 'a'*64, 'strength': .5}]}}
    with pytest.raises(ValueError, match='already baked'):
        reject_baked_replay(entry, {'adapters': [{'name': 'renamed.safetensors', 'sha256': 'a'*64}]})
    reject_baked_replay(entry, {'adapters': [{'sha256': 'b'*64}]})
    reject_baked_replay({'provenance': None}, {'adapters': [{'sha256': 'a'*64}]})


def test_unrelated_malformed_headers_are_rejected_without_blocking_catalog(profile_files):
    root, folders, custom, state, provenance = profile_files
    for name, raw in (('wrong_metadata', b'{"__metadata__":[]}'),
                      ('deep_header', b'['*1500 + b'0' + b']'*1500)):
        path = custom.parent / (name + '.safetensors')
        path.write_bytes(struct.pack('<Q', len(raw)) + raw)
    malformed = dict(provenance, baked_loras=[{'sha256': 'a'*64, 'strength': 10**400}])
    save_file(state, custom.parent / 'huge_strength.safetensors',
              metadata={EXPORT_METADATA: json.dumps(malformed)})
    catalog = CheckpointCatalog(root, folders)
    assert catalog.names() == [DEFAULT_CHECKPOINT, custom.name]
    assert set(catalog.rejected) == {'wrong_metadata.safetensors', 'deep_header.safetensors',
                                     'huge_strength.safetensors'}


def test_checkpoint_switch_releases_model_then_binds_new_asset_and_restores_default(profile_files):
    root, folders, custom, _, _ = profile_files
    for kind, name in (('text_encoders', 'qwen3vl_4b_bf16.safetensors'), ('vae', 'qwen_image_vae.safetensors')):
        path = root / 'models' / kind / name
        path.parent.mkdir()
        path.write_bytes(b'fixture')
    profile = WorkbenchProfile.__new__(WorkbenchProfile)
    events = []
    profile.root, profile.folder_paths = root, folders
    profile.checkpoints = CheckpointCatalog(root, folders)
    profile.assets = resolve_profile_assets(root, folders)
    profile.asset_identities = {kind: identity(path) for kind, path in profile.assets.items()}
    profile.checkpoint_name = DEFAULT_CHECKPOINT
    profile.base_model = None
    profile.model, profile.loader, profile.original_parameters, profile.conditioning = object(), object(), object(), object()
    profile.mm = SimpleNamespace(cleanup_models_gc=lambda: events.append('gc'), soft_empty_cache=lambda: events.append('empty'))
    profile.emit = lambda event, payload: events.append(event)
    profile.select_checkpoint(custom.name)
    assert profile.checkpoint == custom.resolve()
    assert profile.model is None and profile.loader is None and profile.original_parameters is None
    assert profile.conditioning is None
    assert events == ['gc', 'empty', 'checkpoint_selected']
    profile.select_checkpoint(custom.name)
    assert len(events) == 3
    profile.select_checkpoint(DEFAULT_CHECKPOINT)
    assert profile.checkpoint == (custom.parent / DEFAULT_CHECKPOINT).resolve()
    assert len(events) == 6
    original_identities = dict(profile.asset_identities)
    Path(profile.assets['vae']).write_bytes(b'replaced')
    with pytest.raises(RuntimeError, match='encoder or VAE'):
        profile.select_checkpoint(custom.name)
    assert profile.checkpoint_name == DEFAULT_CHECKPOINT
    assert profile.asset_identities == original_identities


def test_checkpoint_switch_refuses_live_previous_model_and_retains_failure_evidence(profile_files):
    root, folders, custom, _, _ = profile_files
    profile = WorkbenchProfile.__new__(WorkbenchProfile)
    profile.root, profile.folder_paths = root, folders
    profile.checkpoints = CheckpointCatalog(root, folders)
    profile.checkpoint_name = DEFAULT_CHECKPOINT
    previous = SimpleNamespace(patches={'pending': 1}, backup={}, loaded_size=lambda: 0)
    profile.base_model = previous
    profile.restore_failure = None
    def check():
        raise RuntimeError(profile.restore_failure)
    profile.check = check
    with pytest.raises(RuntimeError, match='restored and unloaded'):
        profile.select_checkpoint(custom.name)
    assert profile.base_model is previous and profile.checkpoint_name == DEFAULT_CHECKPOINT
    assert profile.restore_failure is not None
