"""Conditioning reuse follows encoder identity and the real runner branches."""
import ast
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from bench import conditioning_cache as cache


def identity(tmp_path, prompt='ab', negative='c', runtime=None):
    encoder = tmp_path / 'encoder.safetensors'
    if not encoder.exists(): encoder.touch()
    source = tmp_path / 'wan.py'
    if not source.exists(): source.write_text('encoder_source = 1\n')
    return cache.context(prompt, negative, encoder, [source], runtime or {'device': 'cpu', 'torch': 'test'})


def test_prompt_boundaries_no_longer_collide(tmp_path):
    first = identity(tmp_path)
    second = identity(tmp_path, 'a', 'bc')
    assert hashlib.sha1((first['prompt'] + first['negative']).encode()).digest() == hashlib.sha1((second['prompt'] + second['negative']).encode()).digest()
    assert cache.path_for(tmp_path, first) != cache.path_for(tmp_path, second)


@pytest.mark.parametrize('changed', ['source', 'encoder', 'runtime'])
def test_identity_change_never_loads_old_conditioning(tmp_path, changed):
    before = identity(tmp_path)
    path = cache.path_for(tmp_path / 'cache', before)
    cache.save(path, before, [[torch.arange(12).reshape(1, 3, 4), {'mask': torch.ones(1, 3, dtype=torch.bool)}]], [])
    if changed == 'source': (tmp_path / 'wan.py').write_text('encoder_source = 2\n')
    if changed == 'encoder': (tmp_path / 'encoder.safetensors').write_bytes(b'new-checkpoint')
    after = identity(tmp_path, runtime={'device': 'mps', 'torch': 'test'} if changed == 'runtime' else None)
    assert after != before
    assert cache.path_for(tmp_path / 'cache', after) != path
    assert cache.load(path, after) is None


def test_tensor_and_metadata_round_trip_cpu_safe_load(tmp_path):
    context = identity(tmp_path)
    path = cache.path_for(tmp_path, context)
    embedding = torch.randn(1, 512, 4096, generator=torch.Generator().manual_seed(9), dtype=torch.float16)
    positive = [[embedding, {'pooled_output': None, 'attention_mask': torch.ones(1, 512)}]]
    negative = [[embedding[:, :4], {}]]
    cache.save(path, context, positive, negative)
    actual, neg = cache.load(path, context)
    assert torch.equal(actual[0][0], embedding) and actual[0][0].device.type == 'cpu'
    assert actual[0][1]['pooled_output'] is None and torch.equal(neg[0][0], negative[0][0])
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('name', ['run_wan5b.py', 'run_wan14b.py'])
@pytest.mark.parametrize('policy', ['auto', 'off'])
def test_real_runner_encoding_and_reuse_branches(tmp_path, monkeypatch, name, policy):
    # Execute the real pre-VAE runner statements; isolate only filesystem identity,
    # encoder node and unload boundaries. No diffusion/VAE or GPU initialization.
    source = Path(__file__).resolve().parents[1] / 'bench' / name
    tree = ast.parse(source.read_text())
    scope = next(node for node in tree.body if isinstance(node, ast.With))
    cut = next(i for i, node in enumerate(scope.body) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'vae' for target in node.targets))
    program = compile(ast.fix_missing_locations(ast.Module(body=scope.body[:cut], type_ignores=[])), str(source), 'exec')
    context = identity(tmp_path)
    monkeypatch.setitem(sys.modules, 'conditioning_cache', cache)
    monkeypatch.setattr(cache, 'wan_context', lambda *args: context)
    monkeypatch.setenv('COND_CACHE', policy)
    calls, laps = [], []
    class Loader:
        def load_clip(self, *args):
            calls.append(args)
            return (object(),)
    class Encode:
        def encode(self, clip, prompt):
            return ([[torch.full((1, 2, 4), float(len(prompt))), {}]],)
    namespace = dict(ROOT=str(tmp_path), PROMPT='ab', NEG='c', CODE='unused', folder_paths=None,
                     torch=torch, mm=SimpleNamespace(unload_all_models=lambda: None, soft_empty_cache=lambda: None),
                     MODE='metal', sys=sys, os=__import__('os'), gc=__import__('gc'),
                     N={'CLIPLoader': Loader, 'CLIPTextEncode': Encode}, lap=laps.append)
    for _ in range(2): exec(program, namespace)
    assert len(calls) == (1 if policy == 'auto' else 2)
    assert ('load_cached_cond' in laps) == (policy == 'auto')
    assert torch.equal(namespace['pos'][0][0], torch.full((1, 2, 4), 2.))
    assert (tmp_path / 'out/cache').exists() == (policy == 'auto')


def test_load_digest_and_tensors_share_snapshot_when_path_is_replaced(tmp_path, monkeypatch):
    context = identity(tmp_path)
    path = cache.path_for(tmp_path, context)
    first = [[torch.ones(1, 2, 4), {}]]
    second = [[torch.full((1, 2, 4), 9.), {}]]
    cache.save(path, context, first, [])
    original_bytes = path.read_bytes()
    replacement = tmp_path / 'replacement.pt'
    cache.save(replacement, context, second, [])
    real_load = torch.load

    def replaced_during_load(stream, *args, **kwargs):
        replacement.replace(path)
        return real_load(stream, *args, **kwargs)

    monkeypatch.setattr(torch, 'load', replaced_during_load)
    positive, negative, digest = cache.load(path, context, with_digest=True)
    assert torch.equal(positive[0][0], first[0][0]) and negative == []
    assert digest == hashlib.sha256(original_bytes).hexdigest()
    assert digest != hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(torch, 'load', real_load)
    current_positive, _ = cache.load(path, context)
    assert torch.equal(current_positive[0][0], second[0][0])


def test_save_digest_binds_published_bytes_even_if_path_changes_after_publish(tmp_path, monkeypatch):
    context = identity(tmp_path)
    path = cache.path_for(tmp_path, context)
    positive = [[torch.arange(8).reshape(1, 2, 4), {}]]
    real_replace = cache.os.replace
    published = []

    def changed_after_publish(source, destination):
        published.append(Path(source).read_bytes())
        real_replace(source, destination)
        replacement = tmp_path / 'external-replacement.pt'
        replacement.write_bytes(b'another writer replaced this pathname')
        real_replace(replacement, destination)

    monkeypatch.setattr(cache.os, 'replace', changed_after_publish)
    digest = cache.save(path, context, positive, [], with_digest=True)
    assert len(published) == 1
    assert digest == hashlib.sha256(published[0]).hexdigest()
    assert digest != hashlib.sha256(path.read_bytes()).hexdigest()
    import io
    payload = torch.load(io.BytesIO(published[0]), weights_only=True, map_location='cpu')
    assert payload['context'] == context and torch.equal(payload['positive'][0][0], positive[0][0])
    assert not list(tmp_path.glob('*.tmp'))
