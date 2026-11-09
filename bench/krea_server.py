"""Dedicated, opt-in localhost Krea API qualification; never production defaults.

The caller owns this process under gpu_runner.py --monitor. Only the fixed
512px eight-step immutable profile is supported. Ordinary model LoRAs require
--plain-loras; the workbench optionally admits spectral sampling. Text encoding releases the
local CLIP before diffusion.
"""
import argparse
from collections import namedtuple
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import weakref

UNET = 'krea2_turbo_int8_convrot.safetensors'
ENCODER = 'qwen3vl_4b_bf16.safetensors'
VAE = 'qwen_image_vae.safetensors'
BUDGET = 2**30
MappedBackup = namedtuple('Dimension', ('weight', 'inplace_update'))


def identity(path):
    st = Path(path).stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def source_pointer(weight, key):
    """Mapped payloads stay strict; Comfy's small converted auxiliaries stay CPU-owned."""
    from krea_adapter import _mapped_storage
    try:
        return _mapped_storage(weight)
    except ValueError:
        import torch
        allowed = key.endswith(('.bias', '.scale', '.mod.lin', '.modulation.lin'))
        if (not allowed or type(weight) not in (torch.Tensor, torch.nn.Parameter)
                or weight.device.type != 'cpu' or not weight.is_floating_point()
                or weight.untyped_storage().nbytes() > 2**20
                or not torch.isfinite(weight).all()):
            raise ValueError('Unsupported unmapped Krea immutable base storage: ' + key)
        return weight.untyped_storage().data_ptr()


def capture_mapped_parameters(patcher):
    """Retain mapped weights and bounded converted auxiliary parameters independently."""
    from krea_adapter import _mapped_storage
    if patcher.patches or patcher.backup or patcher.loaded_size():
        raise ValueError('Mapped source capture requires a fresh unloaded base')
    originals = {}
    owned_bytes = 0
    for key, weight in patcher.model.named_parameters():
        pointer = source_pointer(weight, key)
        # Module.to can change Parameter.data; detach retains original CPU storage.
        snapshot = weight.detach()
        if source_pointer(snapshot, key) != pointer:
            raise RuntimeError('Detached Krea source lost its original CPU storage')
        try:
            _mapped_storage(snapshot)
        except ValueError:
            owned_bytes += snapshot.untyped_storage().nbytes()
            if owned_bytes > 8 * 2**20:
                raise ValueError('Converted Krea auxiliaries exceed the 8 MiB bound')
            # Unlike readonly mappings, anonymous auxiliary storage needs a value guard.
            snapshot._krea_owned_expected = snapshot.clone()
        originals[key] = (snapshot, pointer)
    if not originals:
        raise ValueError('Mapped Krea base has no parameters')
    return originals


def check_mapped_parameters(patcher, originals):
    """Verify complete restoration, including converted biases and normalization."""
    import torch
    from krea_adapter import _mapped_storage
    parameters = dict(patcher.model.named_parameters())
    if set(parameters) != set(originals):
        raise RuntimeError('Krea immutable base parameter membership changed')
    for key, (snapshot, pointer) in originals.items():
        module_name = key.rsplit('.', 1)[0] if '.' in key else ''
        module = patcher.model.get_submodule(module_name)
        expected = getattr(snapshot, '_krea_owned_expected', None)
        source = _mapped_storage if expected is None else lambda weight: source_pointer(weight, key)
        if (source(snapshot) != pointer or source(parameters[key]) != pointer
                or (expected is not None and (not torch.equal(snapshot, expected)
                                             or not torch.equal(parameters[key], expected)))
                or getattr(module, 'weight_function', ()) or getattr(module, 'bias_function', ())
                or getattr(module, 'lora_side_patches', ())):
            raise RuntimeError('Krea selection switch lost restored immutable base storage: ' + key)


def check_source_backups(patcher, originals):
    """Allow only our exact retained originals as auxiliary Comfy backups."""
    import torch
    from krea_adapter import _mapped_storage
    for key, entry in patcher.backup.items():
        if key not in originals:
            raise RuntimeError('Unknown Krea source backup: ' + key)
        snapshot, pointer = originals[key]
        expected = getattr(snapshot, '_krea_owned_expected', None)
        source = _mapped_storage if expected is None else lambda weight: source_pointer(weight, key)
        if (entry.weight is not snapshot or entry.inplace_update
                or source(snapshot) != pointer
                or (expected is not None and not torch.equal(snapshot, expected))):
            raise RuntimeError('Krea source backup lost immutable storage: ' + key)


def seed_mapped_backups(patcher, originals):
    """Use Comfy's ordinary unpatch backups for all base parameters before device load."""
    if patcher.backup or patcher.loaded_size():
        raise RuntimeError('Krea mapped backups require an unloaded candidate without old backups')
    check_mapped_parameters(patcher, originals)
    seeded = {key: MappedBackup(snapshot, False) for key, (snapshot, _) in originals.items()}
    patcher.backup.update(seeded)
    return seeded


def rollback_untouched_backups(patcher, originals, seeded):
    """Undo only our unchanged seed after a refusal before any weight migration."""
    if patcher.loaded_size() != 0 or set(patcher.backup) != set(seeded):
        return False
    if any(patcher.backup[key] is not entry or entry.weight is not originals[key][0]
           or entry.inplace_update for key, entry in seeded.items()):
        return False
    try:
        check_mapped_parameters(patcher, originals)
    except Exception:
        # Retain evidence for normal unload and the sticky restoration guard.
        return False
    patcher.backup.clear()
    return True


def resolve_profile_assets(root, folder_paths, checkpoint_name=UNET):
    assets = {}
    for kind, name in (('diffusion_models', checkpoint_name), ('text_encoders', ENCODER), ('vae', VAE)):
        expected = (Path(root) / 'models' / kind / name).resolve(strict=True)
        selected = Path(folder_paths.get_full_path_or_raise(kind, name)).resolve(strict=True)
        if selected != expected:
            raise RuntimeError(f'Dedicated Krea {kind} selection differs from the profile root')
        assets[kind] = str(selected)
    return assets


def resolve_plain_lora(root, folder_paths, name, strength):
    """Resolve a single profile-owned filename, never an arbitrary path."""
    if (not isinstance(name, str) or type(strength) not in (int, float)
            or not math.isfinite(strength) or strength == 0):
        raise ValueError('Plain Krea selection requires a filename and finite nonzero strength')
    if not name:
        return None
    if (Path(name).name != name or '/' in name or '\\' in name or '..' in name
            or not name.endswith('.safetensors')):
        raise ValueError('Plain Krea LoRA must be a single safetensors filename')
    expected = (Path(root) / 'models' / 'loras' / name).resolve(strict=True)
    selected = Path(folder_paths.get_full_path_or_raise('loras', name)).resolve(strict=True)
    if selected != expected:
        raise RuntimeError('Dedicated Krea LoRA selection differs from the profile root')
    return selected


def validate_graph(current, *, plain_loras=False):
    """Admit only the fixed graph and the explicit selection inputs."""
    from probe_krea_server import graph
    seed = current['3']['inputs']['seed']
    inputs = current['1']['inputs']
    prompt = inputs['prompt']
    prefix = current['6']['inputs']['filename_prefix']
    expected = graph(seed, prompt, prefix)
    if plain_loras:
        name, strength = inputs['lora_name'], inputs['lora_strength']
        # Path/asset admission also occurs against folder_paths in prepare.
        if (not isinstance(name, str) or type(strength) not in (int, float)
                or not math.isfinite(strength) or strength == 0
                or (name and (Path(name).name != name or '/' in name or '\\' in name
                              or '..' in name or not name.endswith('.safetensors')))):
            raise ValueError('Invalid plain Krea LoRA selection')
        expected['1']['inputs'].update(lora_name=name, lora_strength=strength)
    if (type(seed) is not int or not 0 <= seed < 2**64 or not isinstance(prompt, str)
            or not isinstance(prefix, str) or not prefix.startswith('review/krea-api-')
            or '..' in prefix or '\\' in prefix or current != expected):
        raise ValueError('Graph differs from the dedicated fixed Krea profile')


def validate_recipe(seed, steps, cfg, sampler_name, scheduler, denoise, latent_image, *, output_size=(512, 512)):
    if (type(seed) is not int or not 0 <= seed < 2**64 or steps != 8 or cfg != 1
            or sampler_name != 'euler' or scheduler != 'simple' or denoise != 1):
        raise ValueError('Dedicated Krea API requires eight-step CFG1 Euler/simple base recipe')
    width, height = output_size
    if tuple(latent_image['samples'].shape) != (1, 4, height // 8, width // 8):
        raise ValueError('Dedicated Krea API requires an empty latent matching its admitted output size')
    if (set(latent_image) != {'samples', 'downscale_ratio_spacial'}
            or type(latent_image['downscale_ratio_spacial']) is not int
            or latent_image['downscale_ratio_spacial'] != 8):
        raise ValueError('Dedicated Krea API requires the exact EmptyLatentImage spatial metadata without masks or extra inputs')


class Profile:
    def __init__(self, *, root, code, torch, folder_paths, mm, utils, nodes, argv, plain_loras=False):
        self.root, self.code, self.torch = Path(root), Path(code), torch
        self.folder_paths, self.mm, self.utils, self.nodes, self.argv = folder_paths, mm, utils, nodes, argv
        self.assets = resolve_profile_assets(root, folder_paths)
        self.checkpoint_name = UNET
        self.checkpoint = Path(self.assets['diffusion_models'])
        self.asset_identities = {kind: identity(path) for kind, path in self.assets.items()}
        self.checkpoint_identity = self.asset_identities['diffusion_models']
        self.model = None
        self.plain_loras = plain_loras
        self.base_model = None
        self.active_contract = None
        self.adapter_receipt = None
        self.lora = None
        self.restore_failure = None
        self.original_parameters = None
        self.loader = None
        self.conditioning = None
        self.mps_cap = False
        self.sample_count = 0
        self.source_paths = [Path(__file__), Path(__file__).with_name('krea_readonly.py'),
                         Path(__file__).with_name('krea_conditioning.py'),
                         Path(__file__).with_name('conditioning_cache.py'),
                         Path(__file__).with_name('probe_krea_server.py')]
        if plain_loras:
            self.source_paths.extend(Path(__file__).with_name(name) for name in
                                     ('krea_adapter.py', 'probe_krea_server_lora.py'))
        self.sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source_paths}

    def check(self):
        if self.restore_failure is not None:
            raise RuntimeError('Krea immutable base restoration failed; restart the dedicated server: '
                               + self.restore_failure)
        if identity(self.checkpoint) != self.checkpoint_identity:
            raise RuntimeError('Dedicated Krea checkpoint identity changed')
        resolve_profile_assets(self.root, self.folder_paths, self.checkpoint_name)
        if {kind: identity(path) for kind, path in self.assets.items()} != self.asset_identities:
            raise RuntimeError('Dedicated Krea encoder or VAE asset identity changed')
        if {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source_paths} != self.sources:
            raise RuntimeError('Dedicated Krea profile source bytes changed')
        if self.adapter_receipt is not None:
            from krea_adapter import _asset_unchanged
            _asset_unchanged(self.adapter_receipt)
            selections = (list(zip(self.lora['adapters'], self.adapter_receipt['adapters']))
                          if self.adapter_receipt.get('policy') == 'plain-stack'
                          else [(self.lora, self.adapter_receipt)])
            for selection, receipt in selections:
                selected = resolve_plain_lora(self.root, self.folder_paths,
                                              selection['name'], selection['strength'])
                if str(selected) != receipt['path']:
                    raise RuntimeError('Admitted API LoRA path changed')

    def emit(self, event, data):
        print('KREA_API_AUDIT ' + json.dumps({'event': event, **data}), flush=True)

    def encode(self, prompt, lap):
        from krea_conditioning import encode_conditioning
        return encode_conditioning(self.nodes.NODE_CLASS_MAPPINGS,
            root=self.root, code=self.code, folder_paths=self.folder_paths, torch=self.torch,
            mm=self.mm, mode='metal', argv=self.argv, te=ENCODER, prompt=prompt,
            cfg=1.0, policy='off', stage='generate', lap=lap)

    def prepare(self, prompt, lora_name='', lora_strength=1.0):
        from krea_readonly import readonly_krea_load
        self.check()
        selected = None
        if self.plain_loras:
            selected = resolve_plain_lora(self.root, self.folder_paths, lora_name, lora_strength)
        elif lora_name or lora_strength != 1.0:
            raise ValueError('Plain Krea API LoRAs were not enabled')
        # An encoder miss may need the full CPU BF16 encoder. On this dedicated
        # server, unload GPU residency first; node-owned model objects survive.
        self.mm.unload_all_models()
        self.mm.soft_empty_cache()
        if self.plain_loras:
            previous = self.model
            # Preserve the source evidence until restoration is proven. A failed
            # restore poisons this shared model permanently; a later base request
            # must not lose the old contract and bypass the pointer check.
            try:
                if self.base_model is not None:
                    if (self.base_model.patches or self.base_model.backup or self.base_model.loaded_size()
                            or (previous is not None and (previous.backup or previous.loaded_size()))):
                        raise RuntimeError('Krea selection switch did not restore an unloaded unpatched base')
                    check_mapped_parameters(self.base_model, self.original_parameters)
            except Exception as error:
                self.restore_failure = str(error)
                self.check()
            self.model, self.active_contract, self.adapter_receipt, self.lora = None, None, None, None
            self.conditioning = None
        # Restore the full allocator allowance for CPU text encoding after a
        # previous request; sampling restores the qualified 0.5 fraction.
        if self.mps_cap:
            self.torch.mps.set_per_process_memory_fraction(1.0)
        loader_type = self.nodes.NODE_CLASS_MAPPINGS['CLIPLoader']
        original = loader_type.load_clip
        refs = []
        def observe(node, *args, **kwargs):
            result = original(node, *args, **kwargs)
            refs.extend(weakref.ref(value) for value in
                        (result[0], result[0].cond_stage_model, result[0].patcher))
            return result
        loader_type.load_clip = observe
        laps = {}
        started = time.perf_counter()
        def lap(name):
            nonlocal started
            now = time.perf_counter()
            laps[name] = now - started
            started = now
        try:
            positive, negative, audit = self.encode(prompt, lap)
        finally:
            loader_type.load_clip = original
        gc.collect()
        self.mm.cleanup_models_gc()
        if any(ref() is not None for ref in refs):
            raise RuntimeError('Dedicated Krea conditioning retained its CLIP, encoder, or patcher object')
        self.torch.mps.set_per_process_memory_fraction(0.5)
        self.mps_cap = True
        if (self.base_model if self.plain_loras else self.model) is None:
            with readonly_krea_load(self.utils, self.checkpoint,
                    execution_device=self.mm.get_torch_device(), native_int8=True,
                    **({'reference_checkpoint': self.root / 'models/diffusion_models' / UNET}
                       if self.checkpoint_name != UNET else {})) as receipt:
                candidate = self.nodes.NODE_CLASS_MAPPINGS['UNETLoader']().load_unet(self.checkpoint_name, 'default')[0]
            if receipt['loads'] != 1:
                raise RuntimeError('Expected exactly one readonly UNET load')
            self.check()
            if self.plain_loras:
                originals = capture_mapped_parameters(candidate)
                self.base_model, self.loader = candidate, receipt
                self.original_parameters = originals
            else:
                self.model, self.loader = candidate, receipt
        if self.plain_loras:
            candidate = self.base_model
            contract, adapter_receipt, selection = None, None, None
            if selected is not None:
                from krea_adapter import verify_plain_adapter, load_plain_adapter, AdapterContract
                adapter_receipt = verify_plain_adapter(selected, strength=lora_strength, steps=8, cfg=1.)
                candidate = load_plain_adapter(candidate, self.utils, adapter_receipt, lora_strength)
                contract = AdapterContract(self.mm, self.utils, candidate, adapter_receipt,
                    backup_validator=lambda patcher: check_source_backups(patcher, self.original_parameters))
                selection = {'name': lora_name, 'strength': lora_strength,
                             'asset': adapter_receipt, 'patch_keys': sorted(candidate.patches)}
            self.check()
            self.model, self.active_contract = candidate, contract
            self.adapter_receipt, self.lora = adapter_receipt, selection
        self.check()
        self.conditioning = {'conditioning': audit, 'times': laps, 'encoder_created': len(refs) // 3,
            'encoder_refs_dead': True, 'model_identity': id(self.model.model), 'loader': self.loader,
            'sources': self.sources, 'checkpoint': self.checkpoint_name,
            'lifecycle': 'dedicated GPU residency unloaded before encoding; model object retained'}
        if self.plain_loras:
            self.conditioning['lora'] = self.lora
        self.emit('conditioning', self.conditioning)
        return self.model, positive, negative

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise):
        self.check()
        import comfy.samplers
        import comfy_kitchen.backends.mps as mps
        from krea_readonly import bounded_krea_residency
        dimensions = getattr(self, 'workbench_output_size', (512, 512))
        validate_recipe(seed, steps, cfg, sampler_name, scheduler, denoise, latent_image, output_size=dimensions)
        self.check()
        if model is not self.model:
            raise ValueError('Sampler model differs from the dedicated admitted model')
        counts = Counter()
        forwards = []
        forward_shapes = []
        spectral_plan = []
        native = mps.int8_linear
        predict = comfy.samplers.CFGGuider.predict_noise
        original_sampler = comfy.samplers.sampler_object
        spectral = getattr(self, 'workbench_spectral', False)
        def counted(*args, **kwargs):
            x = args[0] if args else kwargs['x']
            counts[str((str(x.dtype), str(x.device), tuple(x.shape)))] += 1
            return native(*args, **kwargs)
        def timed(guider, *args, **kwargs):
            self.torch.mps.synchronize()
            start = time.perf_counter()
            if hasattr(self, 'check_request'):
                self.check_request()
            result = predict(guider, *args, **kwargs)
            if hasattr(self, 'check_request'):
                self.check_request()
            self.torch.mps.synchronize()
            forwards.append(time.perf_counter() - start)
            forward_shapes.append(list(args[0].shape if args else kwargs['x'].shape))
            return result
        def selected_sampler(name):
            from spectral_krea import wrap_euler
            if name != 'euler':
                raise ValueError('Spectral workbench requires deterministic Euler')
            return wrap_euler(original_sampler(name), model, positive, negative, enabled=True,
                prepared_reference=getattr(self, 'prepared_reference', None), report=spectral_plan.append)
        seeded = None
        try:
            if self.plain_loras:
                seeded = seed_mapped_backups(model, self.original_parameters)
            mps.int8_linear = counted
            comfy.samplers.CFGGuider.predict_noise = timed
            if spectral:
                comfy.samplers.sampler_object = selected_sampler
            start = time.perf_counter()
            admission = (self.active_contract.residency(model, BUDGET) if self.active_contract is not None
                         else bounded_krea_residency(self.mm, model, BUDGET))
            with admission as residency:
                result = self.nodes.NODE_CLASS_MAPPINGS['KSampler']().sample(model, seed, steps, cfg,
                    sampler_name, scheduler, positive, negative, latent_image, denoise)[0]
            self.torch.mps.synchronize()
        except Exception:
            if seeded is not None:
                rollback_untouched_backups(model, self.original_parameters, seeded)
            raise
        finally:
            mps.int8_linear = native
            comfy.samplers.CFGGuider.predict_noise = predict
            comfy.samplers.sampler_object = original_sampler
        self.check()
        if len(forwards) != 8 or sum(counts.values()) != 1792:
            raise RuntimeError('Dedicated Krea sampler lost the eight-forward native INT8 route')
        if not bool(self.torch.isfinite(result['samples']).all().item()):
            raise RuntimeError('Nonfinite Krea API latent')
        width, height = dimensions
        full = [height // 8, width // 8]
        half = [height // 16, width // 16]
        expected_spatial = [half] * 4 + [full] * 4 if spectral else [full] * 8
        if [shape[-2:] for shape in forward_shapes] != expected_spatial or len(spectral_plan) != int(spectral):
            raise RuntimeError('Krea sampler lost its admitted spatial schedule')
        self.sample_count += 1
        audit = {'sample_index': self.sample_count, 'seed': seed, 'seconds': time.perf_counter() - start,
            'native_calls': sum(counts.values()), 'native_routes': dict(counts), 'forward_s': forwards,
            'forward_shapes': forward_shapes, 'output_size': list(dimensions), 'spectral': spectral, 'spectral_plan': spectral_plan,
            'finite_latent': True, 'latent_shape': list(result['samples'].shape), 'residency': residency,
            'model_identity': id(model.model), 'checkpoint_identity': self.checkpoint_identity,
            'loader_count': self.loader['loads'], 'sources': self.sources, 'conditioning': self.conditioning,
            'server_pid': os.getpid(), 'asset_identities': self.asset_identities}
        audit['checkpoint'] = self.checkpoint_name
        if self.plain_loras:
            audit['lora'] = self.lora
        self.emit('sample', audit)
        return {'ui': {'krea_audit': [audit]}, 'result': (result,)}


def node_classes(profile):
    class ConditioningModel:
        @classmethod
        def INPUT_TYPES(cls):
            required = {'prompt': ('STRING', {'multiline': True})}
            if profile.plain_loras:
                required.update(lora_name=('STRING', {'default': ''}),
                                lora_strength=('FLOAT', {'default': 1.0}))
            return {'required': required}
        RETURN_TYPES = ('MODEL', 'CONDITIONING', 'CONDITIONING')
        FUNCTION = 'prepare'
        CATEGORY = 'bench/krea'
        def prepare(self, prompt, **kwargs):
            return profile.prepare(prompt, **kwargs)

    if profile.plain_loras:
        # A cached clone from an earlier selection must never bypass prepare's
        # unload/restore/admit sequence, even when switching back to old inputs.
        ConditioningModel.IS_CHANGED = classmethod(lambda cls, **kwargs: float('nan'))

    class Sampler:
        @classmethod
        def INPUT_TYPES(cls):
            return profile.nodes.NODE_CLASS_MAPPINGS['KSampler'].INPUT_TYPES()
        RETURN_TYPES = ('LATENT',)
        FUNCTION = 'sample'
        CATEGORY = 'bench/krea'
        def sample(self, *args, **kwargs):
            return profile.sample(*args, **kwargs)

    class Decode:
        @classmethod
        def INPUT_TYPES(cls):
            return profile.nodes.NODE_CLASS_MAPPINGS['VAEDecode'].INPUT_TYPES()
        RETURN_TYPES = ('IMAGE',)
        FUNCTION = 'decode'
        CATEGORY = 'bench/krea'
        def decode(self, vae, samples):
            image = profile.nodes.NODE_CLASS_MAPPINGS['VAEDecode']().decode(vae, samples)[0]
            width, height = getattr(profile, 'workbench_output_size', (512, 512))
            if tuple(image.shape) != (1, height, width, 3):
                raise RuntimeError('Dedicated Krea API decoded an unexpected image shape')
            if not bool(profile.torch.isfinite(image).all().item()):
                raise RuntimeError('Nonfinite Krea API image')
            if hasattr(profile, 'check_request'):
                profile.check_request()
            audit = {'finite_image': True, 'shape': list(image.shape), 'sample_index': profile.sample_count}
            profile.emit('decode', audit)
            return {'ui': {'krea_decode': [audit]}, 'result': (image,)}
    return {'BenchKreaConditioningModel': ConditioningModel, 'BenchKreaSampler': Sampler, 'BenchKreaDecode': Decode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8191)
    parser.add_argument('--plain-loras', action='store_true')
    parser.add_argument('--workbench', action='store_true')
    parser.add_argument('--comfy-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'vendor/ComfyUI')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Use an unprivileged dedicated localhost port')
    code = args.comfy_dir.resolve(strict=True)
    root = args.root.resolve(strict=True)
    sys.path.insert(0, str(code))
    os.chdir(code)
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
    sys.argv = [str(code / 'main.py'), '--listen', '127.0.0.1', '--port', str(args.port),
        '--fp16-unet', '--use-pytorch-cross-attention', '--cache-classic', '--disable-all-custom-nodes',
        '--disable-partner-nodes', '--disable-auto-launch', '--output-directory', str(root / 'out/comfyui')]
    import main as server_main
    import torch, folder_paths, nodes, comfy.utils
    import comfy.model_management as mm
    if sys.platform != 'darwin' or not torch.backends.mps.is_available():
        raise RuntimeError('Dedicated Krea API requires macOS MPS')
    if 'int8_tensorwise' not in mm.mps_quant_formats(mm.get_torch_device()):
        raise RuntimeError('Dedicated Krea API requires native INT8')
    for kind in ('diffusion_models', 'text_encoders', 'vae') + (('loras',) if args.plain_loras or args.workbench else ()):
        folder_paths.add_model_folder_path(kind, str(root / 'models' / kind), is_default=True)
    if args.workbench:
        nodes.EXTENSION_WEB_DIRS["krea_workbench"] = str(Path(__file__).with_name("web"))
    loop, server, start = server_main.start_comfyui()
    profile_type = Profile
    if args.workbench:
        from krea_workbench import WorkbenchProfile
        profile_type = WorkbenchProfile
    profile = profile_type(root=root, code=code, torch=torch, folder_paths=folder_paths, mm=mm,
                      utils=comfy.utils, nodes=nodes, argv=sys.argv[1:], plain_loras=args.plain_loras or args.workbench)
    if args.workbench:
        folder_paths.set_input_directory(str(root / 'input'))
        (root / 'input').mkdir(parents=True, exist_ok=True)
    nodes.NODE_CLASS_MAPPINGS.update(node_classes(profile))
    if args.workbench:
        from krea_workbench import workbench_nodes
        nodes.NODE_CLASS_MAPPINGS.update(workbench_nodes(profile))
    from aiohttp import web
    if args.workbench:
        from krea_export_jobs import ExportJobs
        exports = ExportJobs(profile, server.prompt_queue)
        server.app.on_cleanup.append(exports.close)

        @web.middleware
        async def export_admission(request, handler):
            if request.path.startswith('/krea-workbench/export'):
                return await exports.handle(request)
            if request.method == 'POST' and request.path in ('/prompt', '/api/prompt'):
                async with exports.lock:
                    if exports.active:
                        return web.json_response({'error': 'Wait for the checkpoint export to finish'}, status=409)
                    return await handler(request)
            return await handler(request)
        server.app.middlewares.append(export_admission)

    @web.middleware
    async def fixed_profile(request, handler):
        if args.workbench and request.method == 'GET' and request.path == '/krea-workbench/default-workflow':
            from krea_workflows import ui_workflow
            return web.json_response(ui_workflow('text'))
        if request.method == 'POST' and request.path in ('/prompt', '/api/prompt'):
            body = await request.json()
            current = body.get('prompt', {})
            try:
                if args.workbench:
                    from krea_workflows import validate_workbench_graph
                    canonical = validate_workbench_graph(current)
                    from krea_workbench import validate_input_files
                    validate_input_files(canonical, folder_paths)
                    profile.checkpoints.resolve(canonical['1']['inputs']['checkpoint'])
                else:
                    validate_graph(current, plain_loras=profile.plain_loras)
            except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
                return web.json_response({'error': str(error)}, status=400)
        return await handler(request)
    server.app.middlewares.append(fixed_profile)
    profile.emit('startup', {'pid': os.getpid(), 'sources': profile.sources,
                            'checkpoint_identity': profile.checkpoint_identity, 'assets': profile.assets, 'asset_identities': profile.asset_identities, 'port': args.port})
    try:
        loop.run_until_complete(start())
    except KeyboardInterrupt:
        pass
    finally:
        server.asset_manager.shutdown()


if __name__ == '__main__':
    main()
