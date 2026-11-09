"""Dedicated API contracts and encoder lifetime checks without a GPU/server."""
from contextlib import contextmanager
import copy
from pathlib import Path
import sys
import subprocess
import os
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from krea_server import Profile, validate_recipe, resolve_profile_assets, identity, UNET, ENCODER, VAE
from probe_krea_server import graphs, summarize


class KreaAPIContracts(unittest.TestCase):
    def test_three_graphs_change_seed_then_prompt_only(self):
        jobs = graphs('cpu')
        self.assertEqual([g['3']['inputs']['seed'] for _, g in jobs], [42, 43, 43])
        self.assertEqual(jobs[0][1]['1'], jobs[1][1]['1'])
        self.assertNotEqual(jobs[1][1]['1'], jobs[2][1]['1'])
        self.assertEqual(jobs[0][1]['4'], jobs[2][1]['4'])
        self.assertEqual(len(jobs[0][1]), 6)
        self.assertFalse(any(n['class_type'] == 'CLIPLoader' for n in jobs[0][1].values()))

    def test_profile_rejects_duplicate_name_selected_from_another_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            for kind, name in (('diffusion_models', UNET), ('text_encoders', ENCODER), ('vae', VAE)):
                asset = Path(temporary) / 'models' / kind / name
                asset.parent.mkdir(parents=True)
                asset.write_bytes(b'fixture')
            selected = lambda kind, name: str(Path(temporary) / 'models' / kind / name)
            paths = SimpleNamespace(get_full_path_or_raise=selected)
            self.assertEqual(len(resolve_profile_assets(temporary, paths)), 3)
            duplicate = Path(temporary) / UNET
            duplicate.write_bytes(b'duplicate')
            paths.get_full_path_or_raise = lambda kind, name: str(duplicate) if kind == 'diffusion_models' else selected(kind, name)
            with self.assertRaisesRegex(RuntimeError, 'selection differs'):
                resolve_profile_assets(temporary, paths)

    def test_sampler_refuses_changed_recipe_or_latent(self):
        args = [42, 8, 1, 'euler', 'simple', 1, {'samples': SimpleNamespace(shape=(1, 4, 64, 64)), 'downscale_ratio_spacial': 8}]
        validate_recipe(*args)
        for index, value in [(0, True), (1, 4), (2, 2), (3, 'heun'), (4, 'normal'), (5, .9),
                             (6, {'samples': SimpleNamespace(shape=(2, 4, 64, 64)), 'downscale_ratio_spacial': 8})]:
            changed = args.copy()
            changed[index] = value
            with self.assertRaises(ValueError):
                validate_recipe(*changed)
        for ratio in (None, 4, True, 8.0):
            changed = args.copy()
            changed[6] = dict(args[6], downscale_ratio_spacial=ratio)
            with self.assertRaises(ValueError):
                validate_recipe(*changed)
        missing = args.copy()
        missing[6] = {'samples': args[6]['samples']}
        with self.assertRaises(ValueError):
            validate_recipe(*missing)
        masked = args.copy()
        masked[6] = dict(args[6], noise_mask=object())
        with self.assertRaises(ValueError):
            validate_recipe(*masked)

    def test_rectangular_recipe_validates_width_height_without_changing_base_api(self):
        for width, height in ((512, 768), (768, 512), (1024, 576), (1024, 1024)):
            latent = {'samples': SimpleNamespace(shape=(1, 4, height // 8, width // 8)),
                      'downscale_ratio_spacial': 8}
            validate_recipe(42, 8, 1, 'euler', 'simple', 1, latent, output_size=(width, height))
            with self.assertRaises(ValueError):
                validate_recipe(42, 8, 1, 'euler', 'simple', 1, latent)
            if width != height:
                with self.assertRaises(ValueError):
                    validate_recipe(42, 8, 1, 'euler', 'simple', 1, latent, output_size=(height, width))

    def test_real_cpu_empty_latent_node_preserves_required_spatial_metadata(self):
        repo = Path(__file__).resolve().parents[1]
        script = """
import sys
from pathlib import Path
repo = Path(sys.argv[1])
sys.path[:0] = [str(repo), str(repo / 'vendor/ComfyUI')]
sys.argv = [sys.argv[0], '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import nodes
from bench.krea_server import validate_recipe
latent = nodes.EmptyLatentImage().generate(512, 512, 1)[0]
assert latent['samples'].device.type == 'cpu'
assert set(latent) == {'samples', 'downscale_ratio_spacial'}
assert latent['downscale_ratio_spacial'] == 8
samples = latent['samples']
validate_recipe(42, 8, 1, 'euler', 'simple', 1, latent)
assert latent['samples'] is samples
assert latent['downscale_ratio_spacial'] == 8
"""
        env = dict(os.environ, PYTHONPATH=str(repo / 'vendor/comfy-kitchen'))
        result = subprocess.run([sys.executable, '-c', script, str(repo)], cwd=repo,
                                env=env, capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_history_requires_real_route_and_cache_transition(self):
        history = {'status': {'status_str': 'success', 'completed': True, 'messages':
            [['execution_cached', {'nodes': ['1']}]]}, 'outputs': {
                '3': {'krea_audit': [{'sample_index': 2, 'native_calls': 1792, 'forward_s': [1] * 8,
                    'finite_latent': True, 'loader_count': 1, 'seed': 43, 'server_pid': 99,
                    'model_identity': 10, 'checkpoint_identity': [1], 'asset_identities': {'vae': [2]},
                    'sources': {'server': 'abc'}, 'conditioning': {'encoder_created': 1,
                    'encoder_refs_dead': True, 'conditioning': {'origin': 'fresh-encoding', 'cache_policy': 'off'}}}]},
                '5': {'krea_decode': [{'sample_index': 2, 'finite_image': True}]},
                '6': {'images': [{'filename': 'image.png'}]}}}
        receipt = summarize(history, 2, 43)['receipt']
        self.assertEqual(summarize(history, 2, 43, receipt)['cached_nodes'], ['1'])
        with self.assertRaisesRegex(RuntimeError, 'seed differs'):
            summarize(history, 2, 42)
        for key in ('server_pid', 'model_identity', 'checkpoint_identity', 'asset_identities', 'sources'):
            changed_receipt = dict(receipt)
            changed_receipt[key] = 'changed'
            with self.assertRaisesRegex(RuntimeError, 'changed across requests'):
                summarize(history, 2, 43, changed_receipt)
        malformed = copy.deepcopy(history)
        del malformed['outputs']['3']['krea_audit'][0]['sources']
        with self.assertRaisesRegex(RuntimeError, 'Malformed'):
            summarize(malformed, 2)
        mutations = [lambda h: h['outputs']['3']['krea_audit'][0]['conditioning']['conditioning'].update(origin='cache-hit'),
                     lambda h: h['outputs']['3']['krea_audit'][0]['conditioning'].update(encoder_created=0),
                     lambda h: h['status']['messages'][0][1].update(nodes=['1', '3']),
                     lambda h: h['outputs']['3']['krea_audit'][0].update(native_calls=0),
                     lambda h: h['outputs']['5']['krea_decode'][0].update(finite_image=False),
                     lambda h: h['outputs']['3']['krea_audit'][0].update(loader_count=2),
                     lambda h: h['status']['messages'][0][1].update(nodes=[])]
        for mutate in mutations:
            bad = copy.deepcopy(history)
            mutate(bad)
            with self.assertRaises(RuntimeError):
                summarize(bad, 2)

    def test_local_encoder_dies_and_model_is_loaded_once_across_prompts(self):
        created = []
        calls = []
        retained = []
        class EncoderObject:
            pass
        class Clip:
            def __init__(self):
                self.cond_stage_model = EncoderObject()
                self.patcher = EncoderObject()
        class Loader:
            def load_clip(self, *args):
                value = Clip()
                if retained:
                    retained[0].append(value.cond_stage_model)
                return (value,)
        original_clip_loader = Loader.load_clip
        model = SimpleNamespace(model=object())
        class Unet:
            def load_unet(self, *args):
                calls.append('unet')
                return (model,)
        encoding = ModuleType('krea_conditioning')
        def encode(mapping, **kwargs):
            clip = mapping['CLIPLoader']().load_clip('fixed')[0]
            self.assertEqual(kwargs['policy'], 'off')
            created.append(kwargs['prompt'])
            return [], [], {'origin': 'fresh-encoding'}
        encoding.encode_conditioning = encode
        readonly = ModuleType('krea_readonly')
        receipts = [0, 1]
        @contextmanager
        def load(*args, **kwargs):
            yield {'loads': receipts.pop(0) if receipts else 1}
        readonly.readonly_krea_load = load
        with tempfile.TemporaryDirectory() as temporary:
            for kind, name in (('diffusion_models', UNET), ('text_encoders', ENCODER), ('vae', VAE)):
                asset = Path(temporary) / 'models' / kind / name
                asset.parent.mkdir(parents=True)
                asset.write_bytes(b'fixture')
            checkpoint = Path(temporary) / 'models/diffusion_models' / UNET
            paths = SimpleNamespace(get_full_path_or_raise=lambda kind, name: str(Path(temporary) / 'models' / kind / name))
            mm = SimpleNamespace(unload_all_models=lambda: calls.append('unload'),
                soft_empty_cache=lambda: None, cleanup_models_gc=lambda: None,
                get_torch_device=lambda: 'fake-mps')
            torch = SimpleNamespace(mps=SimpleNamespace(set_per_process_memory_fraction=lambda n: calls.append(n)))
            nodes = SimpleNamespace(NODE_CLASS_MAPPINGS={'CLIPLoader': Loader, 'UNETLoader': Unet})
            profile = Profile(root=temporary, code=temporary, torch=torch, folder_paths=paths, mm=mm,
                              utils=object(), nodes=nodes, argv=[])
            profile.emit = lambda *_: None
            with patch.dict(sys.modules, {'krea_conditioning': encoding, 'krea_readonly': readonly}):
                with self.assertRaisesRegex(RuntimeError, 'exactly one readonly'):
                    profile.prepare('failed-admission')
                self.assertIsNone(profile.model)
                self.assertIsNone(profile.loader)
                first = profile.prepare('first')
                second = profile.prepare('second')
                self.assertIs(first[0], second[0])
                self.assertEqual(calls.count('unet'), 2)
                self.assertEqual(created, ['failed-admission', 'first', 'second'])
                self.assertFalse(hasattr(profile, 'encoder_refs'))
                vae = Path(temporary) / 'models/vae' / VAE
                vae.write_bytes(b'changed-vae')
                with self.assertRaisesRegex(RuntimeError, 'encoder or VAE asset identity changed'):
                    profile.check()
                profile.asset_identities['vae'] = identity(vae)
                retained.append([])
                with self.assertRaisesRegex(RuntimeError, 'retained its CLIP'):
                    profile.prepare('third')
                self.assertIs(Loader.load_clip, original_clip_loader)
                source = Path(temporary) / 'source.py'
                source.write_bytes(b'original')
                profile.source_paths = [source]
                import hashlib
                profile.sources = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
                source.write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'source bytes changed'):
                    profile.check()
                profile.sources = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
                checkpoint.write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                    profile.prepare('fourth')


if __name__ == '__main__':
    unittest.main()
