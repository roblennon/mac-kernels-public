"""CPU conditioning reuse contracts; no Comfy runtime or model initialization."""
import ast
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

from bench import conditioning_cache as cache
from bench import krea_conditioning as krea


ROOT = Path(__file__).resolve().parents[1]
TE = 'qwen3vl_4b_bf16.safetensors'
LOOKUP_CONTEXT = krea.lookup_context


def real_zero_out():
    # Read the actual pure node implementation without importing Comfy's devices.
    tree = ast.parse((ROOT / 'vendor/ComfyUI/nodes.py').read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == 'ConditioningZeroOut')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'nodes.py', 'exec'), namespace)
    return namespace['ConditioningZeroOut']


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    calls, laps, refs = [], [], []
    encoder = tmp_path / TE
    encoder.write_bytes(b'checkpoint identity only')
    source = tmp_path / 'encoder.py'
    source.write_text('encoder_source = 1\n')
    identity = cache.context('a soccer player', 'zero-positive', encoder, [source], {'device': 'cpu'})
    monkeypatch.setattr(krea, 'lookup_context', lambda **kwargs: identity)

    class Clip:
        def __init__(self):
            self.cond_stage_model = SimpleNamespace(
                parameters=lambda: iter([torch.ones(2, dtype=torch.bfloat16)]),
                dtypes={torch.bfloat16})
            self.patcher = SimpleNamespace(load_device=torch.device('cpu'))

    class Loader:
        def load_clip(self, *args):
            calls.append(('load', args))
            clip = Clip()
            refs.append(weakref.ref(clip))
            return (clip,)

    class Encode:
        def encode(self, clip, text):
            calls.append(('encode', text))
            count = 1 if text == '' else 2
            return ([[torch.full((1, count, 30720), 0.5 if text == '' else 2., dtype=torch.bfloat16),
                      {'pooled_output': torch.ones(1, 4, dtype=torch.bfloat16),
                       'attention_mask': torch.ones(1, count, dtype=torch.int64),
                       'strength': 1.0, 'label': 'cpu fixture'}]],)

    mm = SimpleNamespace(unload_all_models=lambda: calls.append(('unload',)),
                         soft_empty_cache=lambda: calls.append(('empty',)))
    arguments = dict(root=tmp_path, code=tmp_path, folder_paths=None, torch=torch, mm=mm,
                     mode='native', argv=[], te=TE, prompt='a soccer player', cfg=1., lap=laps.append)
    nodes = {'CLIPLoader': Loader, 'CLIPTextEncode': Encode, 'ConditioningZeroOut': real_zero_out()}
    return SimpleNamespace(nodes=nodes, kwargs=arguments, calls=calls, laps=laps, refs=refs,
                           identity=identity, root=tmp_path)


def encode(runtime, **overrides):
    arguments = {**runtime.kwargs, **overrides}
    return krea.encode_conditioning(runtime.nodes, **arguments)


def cache_file(runtime):
    paths = list(runtime.root.rglob('krea-cond-v1-*.pt'))
    assert len(paths) == 1
    return paths[0]


def test_auto_miss_then_hit_reuses_real_cpu_tensors_and_releases_encoder(runtime):
    positive, negative, _ = encode(runtime, policy='auto')
    assert runtime.calls == [('load', (TE, 'krea2', 'default')), ('encode', 'a soccer player'),
                             ('unload',), ('empty',)]
    gc.collect()
    assert runtime.refs[0]() is None
    assert positive[0][0].device.type == 'cpu'
    assert positive[0][0].dtype == torch.bfloat16
    assert torch.count_nonzero(negative[0][0]) == 0
    assert torch.count_nonzero(negative[0][1]['pooled_output']) == 0
    runtime.calls.clear()
    cached_positive, cached_negative, _ = encode(runtime, policy='auto')
    assert runtime.calls == []
    assert torch.equal(cached_positive[0][0], positive[0][0])
    assert torch.equal(cached_negative[0][0], negative[0][0])
    assert torch.equal(cached_positive[0][1]['attention_mask'], positive[0][1]['attention_mask'])
    assert not list(runtime.root.rglob('*.tmp'))


def test_default_off_encodes_each_time_and_writes_no_cache(runtime):
    for _ in range(2):
        encode(runtime)
    assert [call for call in runtime.calls if call[0] == 'encode'] == [('encode', 'a soccer player')] * 2
    assert len([call for call in runtime.calls if call[0] == 'unload']) == 2
    assert not list(runtime.root.rglob('*-cond-v1-*.pt'))


def test_guided_cfg_encodes_empty_negative_with_its_own_sequence_length(runtime):
    positive, negative, _ = encode(runtime, policy='auto', cfg=2.)
    assert [call for call in runtime.calls if call[0] == 'encode'] == [
        ('encode', 'a soccer player'), ('encode', '')]
    assert positive[0][0].shape == (1, 2, 30720)
    assert negative[0][0].shape == (1, 1, 30720)
    assert torch.count_nonzero(negative[0][0]) > 0
    runtime.calls.clear()
    _, cached_negative, _ = encode(runtime, policy='auto', cfg=2.)
    assert runtime.calls == [] and torch.equal(cached_negative[0][0], negative[0][0])


@pytest.mark.parametrize('options', [dict(policy='invalid'), dict(stage='invalid'),
                                    dict(stage='capture', policy='off'), dict(cfg=float('nan')),
                                    dict(cfg=float('inf'))])
def test_invalid_options_fail_before_encoder_initialization(runtime, options):
    with pytest.raises(ValueError):
        encode(runtime, **options)
    assert runtime.calls == []
    assert not list(runtime.root.rglob('*-cond-v1-*.pt'))


def test_capture_returns_cache_and_cleanup_before_runner_exit(runtime):
    positive, negative, audit = encode(runtime, stage='capture', policy='auto')
    assert positive and negative and audit
    assert runtime.calls[-2:] == [('unload',), ('empty',)]
    assert cache_file(runtime).is_file()


@pytest.mark.parametrize('mutation', ['tensor_bytes', 'width', 'nonfinite', 'metadata', 'negative', 'provenance'])
def test_corrupt_cache_is_refused_without_encoder_fallback(runtime, mutation):
    encode(runtime, policy='auto')
    path = cache_file(runtime)
    payload = torch.load(path, weights_only=True, map_location='cpu')
    if mutation == 'tensor_bytes':
        payload['positive'][0][0][0, 0, 0] += 1
    elif mutation == 'width':
        payload['positive'][0][0] = payload['positive'][0][0][:, :, :4]
    elif mutation == 'nonfinite':
        payload['positive'][0][0][0, 0, 0] = float('nan')
    elif mutation == 'metadata':
        payload['positive'][0][1]['strength'] = 0.25
    elif mutation == 'negative':
        payload['negative'][0][0][0, 0, 0] = 1
    else:
        payload['provenance'] = {}
    torch.save(payload, path)
    runtime.calls.clear()
    with pytest.raises((ValueError, RuntimeError)):
        encode(runtime, policy='auto')
    assert runtime.calls == []


def test_unreadable_cache_is_refused_without_encoder_fallback(runtime):
    encode(runtime, policy='auto')
    cache_file(runtime).write_bytes(b'not a torch cache')
    runtime.calls.clear()
    with pytest.raises(Exception):
        encode(runtime, policy='auto')
    assert runtime.calls == []


def test_encoder_failure_still_unloads_and_does_not_publish_cache(runtime):
    class Failure:
        def encode(self, *args):
            raise LookupError('external encoder failed')
    runtime.nodes['CLIPTextEncode'] = Failure
    with pytest.raises(LookupError, match='external encoder failed'):
        encode(runtime, policy='auto')
    assert runtime.calls[-2:] == [('unload',), ('empty',)]
    gc.collect()
    assert runtime.refs[0]() is None
    assert not list(runtime.root.rglob('*-cond-v1-*.pt'))


def test_krea_cache_namespace_cannot_collide_with_wan(runtime):
    assert cache.path_for(runtime.root, runtime.identity, namespace='krea') != cache.path_for(
        runtime.root, runtime.identity)


@pytest.fixture
def identity_runtime(tmp_path, monkeypatch):
    # Fake only external installation/package discovery, keeping source hashing,
    # file identity, requested policy and cache selection real.
    import sys
    code = tmp_path / 'ComfyUI'
    tokenizer = code / 'comfy/text_encoders/qwen25_tokenizer'
    tokenizer.mkdir(parents=True)
    (tokenizer / 'tokenizer.json').write_text('{"version": 1}')
    merges = tokenizer / 'merges.txt'
    merges.write_text('a b\n')
    source = code / 'comfy/text_encoders/qwen.py'
    source.write_text('encoder_source = 1\n')
    (code / 'nodes.py').write_text('nodes_source = 1\n')
    (code / 'folder_paths.py').write_text('paths_source = 1\n')
    kitchen = tmp_path / 'kitchen'
    kitchen.mkdir()
    (kitchen / '__init__.py').write_text('kitchen_source = 1\n')
    monkeypatch.setitem(sys.modules, 'comfy_kitchen', SimpleNamespace(__file__=str(kitchen / '__init__.py')))
    monkeypatch.setattr(krea, 'version', lambda name: 'fixture-1')
    encoder = tmp_path / TE
    encoder.write_bytes(b'checkpoint')
    folder_paths = SimpleNamespace(get_folder_paths=lambda name: [],
                                  get_full_path_or_raise=lambda kind, name: str(encoder))
    mm = SimpleNamespace(text_encoder_device=lambda: torch.device('cpu'),
                         text_encoder_dtype=lambda device: torch.float32)
    kwargs = dict(code=code, folder_paths=folder_paths, torch=torch, mm=mm, mode='native',
                  argv=['--cpu'], te=TE, prompt='a soccer player', cfg=1.)
    return SimpleNamespace(kwargs=kwargs, source=source, merges=merges, encoder=encoder)


@pytest.mark.parametrize('mutation', ['tokenizer_txt', 'source', 'checkpoint', 'negative_policy', 'runtime'])
def test_real_lookup_identity_changes_prevent_stale_cache_hits(identity_runtime, tmp_path, mutation):
    runtime = identity_runtime
    before = LOOKUP_CONTEXT(**runtime.kwargs)
    path = cache.path_for(tmp_path / 'cache', before, namespace='krea')
    value = [[torch.ones(1, 2, 30720, dtype=torch.bfloat16), {}]]
    cache.save(path, before, value, value)
    if mutation == 'tokenizer_txt':
        # Preserve byte length; a filename/stat-only binding is insufficient.
        runtime.merges.write_text('c d\n')
    elif mutation == 'source':
        runtime.source.write_text('encoder_source = 2\n')
    elif mutation == 'checkpoint':
        runtime.encoder.write_bytes(b'new checkpoint bytes')
    elif mutation == 'negative_policy':
        runtime.kwargs['cfg'] = 2.
    else:
        runtime.kwargs['argv'] = ['--cpu', '--disable-smart-memory']
    after = LOOKUP_CONTEXT(**runtime.kwargs)
    assert before != after
    assert cache.path_for(tmp_path / 'cache', after, namespace='krea') != path
    assert cache.load(path, after) is None


def test_requested_dtype_is_separate_from_checkpoint_observed_dtype(runtime, identity_runtime, monkeypatch):
    identity = LOOKUP_CONTEXT(**identity_runtime.kwargs)
    monkeypatch.setattr(krea, 'lookup_context', lambda **kwargs: identity)
    _, _, audit = encode(runtime, policy='auto')
    assert identity['runtime']['requested_encoder_dtype'] == 'torch.float32'
    observed = audit['provenance']['encoder_observed']
    assert observed == {'load_device': 'cpu', 'parameter_devices': ['cpu'],
                        'parameter_dtypes': ['torch.bfloat16'], 'declared_dtypes': ['torch.bfloat16']}
    _, _, hit_audit = encode(runtime, policy='auto')
    assert hit_audit['origin'] == 'cache-hit'
    assert hit_audit['provenance']['encoder_observed'] == observed


@pytest.mark.parametrize('field,value', [('parameter_devices', ['mps:0']),
                                        ('parameter_dtypes', ['invalid-dtype'])])
def test_corrupt_observed_encoder_provenance_is_refused_on_hit(runtime, field, value):
    encode(runtime, policy='auto')
    path = cache_file(runtime)
    payload = torch.load(path, weights_only=True, map_location='cpu')
    payload['provenance']['encoder_observed'][field] = value
    torch.save(payload, path)
    runtime.calls.clear()
    with pytest.raises(ValueError):
        encode(runtime, policy='auto')
    assert runtime.calls == []


def test_off_preserves_encoder_outputs_without_cache_admission(runtime):
    # Historical encoding can use a clip without introspection attributes, a
    # different tensor shape and metadata which cache serialization cannot own.
    marker = object()
    positive = [[torch.ones(1, 2, 4), {'external_metadata': marker}]]
    negative = [[torch.ones(1, 1, 4), {'external_metadata': marker}]]

    class Loader:
        def load_clip(self, *args):
            return (object(),)

    class Encode:
        def encode(self, clip, text):
            return (negative if text == '' else positive,)

    runtime.nodes.update(CLIPLoader=Loader, CLIPTextEncode=Encode)
    actual_positive, actual_negative, audit = encode(runtime, policy='off', cfg=2.)
    assert actual_positive is positive and actual_negative is negative
    assert actual_positive[0][1]['external_metadata'] is marker
    assert audit['provenance'] is None and audit['cache_sha256'] is None
    assert runtime.calls == [('unload',), ('empty',)]
    assert not list(runtime.root.rglob('*-cond-v1-*.pt'))


@pytest.mark.parametrize('mutation', ['in_place', 'atomic_replace'])
def test_same_size_encoder_change_with_restored_mtime_invalidates_cache(identity_runtime, tmp_path, mutation):
    import os
    runtime = identity_runtime
    before = LOOKUP_CONTEXT(**runtime.kwargs)
    path = cache.path_for(tmp_path / 'cache', before, namespace='krea')
    positive = [[torch.ones(1, 2, 30720, dtype=torch.bfloat16), {}]]
    cache.save(path, before, positive, [])
    original = runtime.encoder.stat()
    previous_bytes = runtime.encoder.read_bytes()
    changed_bytes = previous_bytes.upper()
    assert len(changed_bytes) == len(previous_bytes) and changed_bytes != previous_bytes
    if mutation == 'in_place':
        runtime.encoder.write_bytes(changed_bytes)
        os.utime(runtime.encoder, ns=(original.st_atime_ns, original.st_mtime_ns))
    else:
        replacement = tmp_path / 'replacement.safetensors'
        replacement.write_bytes(changed_bytes)
        os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
        replacement.replace(runtime.encoder)
    after = LOOKUP_CONTEXT(**runtime.kwargs)
    # The former three-field identity falsely treated both edits as unchanged.
    # Keep that failure proof explicit without modifying the active helper.
    legacy_before = {key: before['encoder'][key] for key in ('path', 'size', 'mtime_ns')}
    legacy_after = {key: after['encoder'][key] for key in ('path', 'size', 'mtime_ns')}
    assert legacy_before == legacy_after
    assert before != after
    assert before['encoder']['ctime_ns'] != after['encoder']['ctime_ns']
    if mutation == 'atomic_replace':
        assert before['encoder']['inode'] != after['encoder']['inode']
    assert cache.path_for(tmp_path / 'cache', after, namespace='krea') != path
    assert cache.load(path, after) is None
