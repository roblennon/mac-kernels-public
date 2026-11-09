"""Bounded ComfyUI controls over the qualified immutable Krea lifecycle."""
import gc
import hashlib
import math
import time
from pathlib import Path

from krea_sizes import LEGACY_SIZE

from krea_server import Profile, ENCODER, VAE, resolve_plain_lora, check_mapped_parameters, check_source_backups

SPECIAL = {'identity_edit': 'krea2_identity_edit_v1_2_r64.safetensors',
           'style_reference': 'krea2_style_reference.safetensors'}
SPECIAL_SHA256 = {
    'identity_edit': 'f794b47142555c929cf536a2f1e4f335174b9aedbb08572b07d45814d4242423',
    'style_reference': 'f50df5a9e62e4be8aa926a63dd5bb1a64770c4004f763c1208007ae13daa82b8',
}


def validate_mode_adapter(mode, receipt):
    if mode in SPECIAL and (not receipt.get('adapters') or
            receipt['adapters'][0]['sha256'] != SPECIAL_SHA256[mode]):
        raise ValueError('Reference adapter differs from the qualified workflow release')


def selections(root, folder_paths, mode, values):
    if mode not in ('text', *SPECIAL):
        raise ValueError('Unknown Krea workbench mode')
    result = []
    if mode in SPECIAL:
        result.append({'name': SPECIAL[mode], 'strength': 1.0})
    for index in range(1, 6):
        name, strength = values.get(f'lora_{index}', ''), values.get(f'strength_{index}', 1.0)
        path = resolve_plain_lora(root, folder_paths, name, strength)
        if path is not None:
            result.append({'name': name, 'strength': strength})
    if len(result) > 5:
        raise ValueError('At most five LoRAs including the reference adapter')
    for item in result:
        item['path'] = str(resolve_plain_lora(root, folder_paths, item['name'], item['strength']))
    return result


def validate_input_files(graph, folder_paths):
    from PIL import Image
    try:
        root = Path(folder_paths.get_input_directory()).resolve(strict=True)
        for node in graph.values():
            if node['class_type'] != 'LoadImage':
                continue
            path = Path(folder_paths.get_annotated_filepath(node['inputs']['image'])).resolve(strict=True)
            if not path.is_relative_to(root) or path.stat().st_size > 16 * 2**20:
                raise ValueError('Reference input must be an uploaded image of at most 16 MiB')
            with Image.open(path) as image:
                if image.width * image.height > 8_000_000 or min(image.size) < 16:
                    raise ValueError('Reference upload must have 16px minimum sides and at most eight megapixels')
    except OSError as error:
        raise ValueError("Reference input must be an existing readable uploaded image") from error


def fit_input_image(image, torch):
    if image is None:
        return None
    if (not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1
            or image.shape[-1] != 3 or image.device.type != 'cpu'
            or not image.is_floating_point() or not torch.isfinite(image).all()
            or min(image.shape[1:3]) < 16 or image.shape[1] * image.shape[2] > 8_000_000
            or (image < 0).any() or (image > 1).any()):
        raise ValueError('Expected one finite CPU RGB reference image')
    h, w = image.shape[1:3]
    if max(h, w) <= 512:
        return image
    factor = 512 / max(h, w)
    size = (max(16, int(h * factor)), max(16, int(w * factor)))
    return torch.nn.functional.interpolate(image.movedim(-1, 1), size=size,
        mode='area').movedim(1, -1).contiguous()


class WorkbenchProfile(Profile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from krea_checkpoints import CheckpointCatalog
        self.checkpoints = CheckpointCatalog(self.root, self.folder_paths)
        self.checkpoint_receipt = self.checkpoints.resolve(self.checkpoint_name)
        self.emit('checkpoint_catalog', {'names': self.checkpoints.names(), 'rejected': self.checkpoints.rejected})
        self.request_started = None
        self.prepared_reference = None
        self.workbench_mode = 'text'
        self.workbench_ref_boost = 4.0
        self.workbench_grounding_size = 512
        self.workbench_vae = None
        self.workbench_spectral = False
        self.workbench_output_size = (512, 512)
        self.source_paths.extend(Path(__file__).with_name(name) for name in
            ('krea_workbench.py', 'krea_workflows.py', 'krea_reference.py',
             'krea_reference_worker.py', 'probe_text_encoder_load.py', 'spectral_krea.py', 'krea_sizes.py',
             'krea_checkpoints.py', 'krea_export_jobs.py', 'krea_export.py'))
        self.source_paths.extend(Path(__file__).with_name('web') / name
                                 for name in ('krea_default.js', 'default_workflow.mjs',
                                              'krea_export.js', 'export_helpers.mjs'))
        from spectral_krea import SOURCE, verify_install
        verify_install()
        self.source_paths.extend((SOURCE, SOURCE.with_name('LICENSE')))
        self.sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source_paths}

    def check_request(self):
        if self.request_started is not None and time.monotonic() - self.request_started > 360:
            raise RuntimeError('Krea workbench request exceeded its 360-second deadline')

    def select_checkpoint(self, name):
        """Called only after prior patches/residency have been restored and checked."""
        entry = self.checkpoints.resolve(name)
        if name == self.checkpoint_name:
            return
        if self.base_model is not None:
            try:
                if self.base_model.patches or self.base_model.backup or self.base_model.loaded_size():
                    raise RuntimeError('Previous checkpoint has not been fully restored and unloaded')
                check_mapped_parameters(self.base_model, self.original_parameters)
            except Exception as error:
                self.restore_failure = str(error)
                self.check()
        from krea_server import resolve_profile_assets, identity
        assets = resolve_profile_assets(self.root, self.folder_paths, name)
        # Only the denoiser may change. Retain startup identity for shared assets.
        for kind in ('text_encoders', 'vae'):
            if assets[kind] != self.assets[kind] or identity(assets[kind]) != self.asset_identities[kind]:
                raise RuntimeError('Dedicated Krea encoder or VAE asset identity changed')
        asset_identities = dict(self.asset_identities)
        asset_identities['diffusion_models'] = entry['file_identity']
        self.base_model = None
        self.model = None
        self.loader = None
        self.original_parameters = None
        self.conditioning = None
        gc.collect()
        self.mm.cleanup_models_gc()
        self.mm.soft_empty_cache()
        self.assets = assets
        self.asset_identities = asset_identities
        self.checkpoint_name = name
        self.checkpoint = Path(assets['diffusion_models'])
        self.checkpoint_identity = entry['file_identity']
        self.checkpoint_receipt = entry
        self.emit('checkpoint_selected', {'name': name, 'receipt': entry})

    def encode(self, prompt, lap):
        if self.workbench_mode == 'text':
            return super().encode(prompt, lap)
        from krea_reference import encode_grounded
        return encode_grounded(self.nodes.NODE_CLASS_MAPPINGS, te=ENCODER, prompt=prompt,
            prepared=self.prepared_reference, lap=lap, mm=self.mm, torch=self.torch)

    def prepare_workbench(self, prompt, mode, ref_boost=4.0, grounding_size=512,
                          source_image=None, ref_image=None, spectral=False, size=LEGACY_SIZE,
                          checkpoint='krea2_turbo_int8_convrot.safetensors', **values):
        self.request_started = time.monotonic()
        self.check()
        if (type(spectral) is not bool or not isinstance(prompt, str) or type(ref_boost) not in (float, int)
                or not math.isfinite(ref_boost) or not .25 <= ref_boost <= 8
                or type(grounding_size) is not int or grounding_size not in (384, 512)):
            raise ValueError('Invalid Krea workbench reference controls')
        if ref_image is not None and grounding_size != 384:
            raise ValueError('Two-reference Identity Edit requires 384px grounding in this memory profile')
        from krea_sizes import output_size
        dimensions = output_size(size)
        # Validate the new selection before releasing any prior model objects.
        checkpoint_receipt = self.checkpoints.resolve(checkpoint)
        chosen = selections(self.root, self.folder_paths, mode, values)
        receipt = None
        if chosen:
            from krea_adapter import verify_plain_stack, load_plain_stack, AdapterContract
            receipt = verify_plain_stack([{'path': x['path'], 'strength': x['strength']} for x in chosen])
            validate_mode_adapter(mode, receipt)
            from krea_checkpoints import reject_baked_replay
            reject_baked_replay(checkpoint_receipt, receipt)
        source_image = fit_input_image(source_image, self.torch)
        ref_image = fit_input_image(ref_image, self.torch)
        # Restore before VAE/vision work; never overlap prior GPU residency with encoding.
        self.mm.unload_all_models()
        self.mm.soft_empty_cache()
        if self.base_model is not None:
            try:
                if (self.base_model.patches or self.base_model.backup or self.base_model.loaded_size()
                        or (self.model is not None and (self.model.backup or self.model.loaded_size()))):
                    raise RuntimeError('Workbench base was not fully unloaded')
                check_mapped_parameters(self.base_model, self.original_parameters)
            except Exception as error:
                self.restore_failure = str(error)
                self.check()
        self.model, self.active_contract, self.adapter_receipt, self.lora = None, None, None, None
        self.conditioning = None
        self.workbench_vae = None
        self.prepared_reference = None
        self.select_checkpoint(checkpoint)
        self.workbench_mode = mode
        self.workbench_spectral = spectral
        self.workbench_output_size = dimensions
        gc.collect()
        if mode != 'text':
            from krea_reference import prepare_reference_owned
            self.prepared_reference = prepare_reference_owned(mode, source_image, ref_image,
                vae_path=self.assets['vae'], root=self.root, torch=self.torch, comfy_root=self.code, ref_boost=ref_boost,
                grounding_size=grounding_size, spectral=spectral, output_size=dimensions)
        elif source_image is not None or ref_image is not None:
            raise ValueError('Text mode does not accept reference images')
        candidate, positive, negative = super().prepare(prompt)
        if chosen:
            candidate = load_plain_stack(self.base_model, self.utils, receipt)
        validator = None
        if self.prepared_reference is not None:
            from krea_reference import apply_reference, validate_reference_options
            candidate, positive, negative = apply_reference(candidate, positive, negative,
                                                            self.prepared_reference)
            validator = lambda patcher: validate_reference_options(patcher, self.prepared_reference)
        contract = None
        if receipt is not None:
            contract = AdapterContract(self.mm, self.utils, candidate, receipt,
                backup_validator=lambda patcher: check_source_backups(patcher, self.original_parameters),
                model_options_validator=validator)
        self.model, self.active_contract, self.adapter_receipt = candidate, contract, receipt
        self.lora = {'adapters': [{k: x[k] for k in ('name', 'strength')} for x in chosen], 'mode': mode} if chosen else None
        self.check_request()
        self.conditioning.update(lora=self.lora, workbench_mode=mode, spectral=spectral, output_size=list(dimensions),
            checkpoint_receipt=self.checkpoint_receipt,
            reference=self.prepared_reference.metadata if self.prepared_reference else None)
        self.check()
        self.workbench_vae = self.nodes.NODE_CLASS_MAPPINGS['VAELoader']().load_vae(VAE)[0]
        latent = self.nodes.NODE_CLASS_MAPPINGS['EmptyLatentImage']().generate(*dimensions, 1)[0]
        return self.model, positive, negative, latent, self.workbench_vae


def workbench_nodes(profile):
    class Workbench:
        @classmethod
        def INPUT_TYPES(cls):
            names = [''] + profile.folder_paths.get_filename_list('loras')
            required = {'prompt': ('STRING', {'multiline': True}),
                        'mode': (['text', 'style_reference', 'identity_edit'],)}
            for index in range(1, 6):
                required[f'lora_{index}'] = (names,)
                required[f'strength_{index}'] = ('FLOAT', {'default': 1.0, 'min': -2., 'max': 2., 'step': .05})
            required.update(ref_boost=('FLOAT', {'default': 4., 'min': .25, 'max': 8., 'step': .25,
                                                'tooltip': 'Identity Edit only: boosts the last reference.'}),
                            grounding_size=([384, 512], {'default': 512,
                                                       'tooltip': 'Identity Edit only: vision grounding size.'}))
            from krea_sizes import DEFAULT_SIZE, SIZE_PRESETS
            return {'required': required, 'optional': {'source_image': ('IMAGE',), 'ref_image': ('IMAGE',),
                'spectral': ('BOOLEAN', {'default': False,
                    'tooltip': 'Four early steps at half spatial size; may change composition or likeness.'}),
                'size': (list(SIZE_PRESETS), {'default': DEFAULT_SIZE}),
                'checkpoint': (profile.checkpoints.names(), {'default': 'krea2_turbo_int8_convrot.safetensors'})}}
        RETURN_TYPES = ('MODEL', 'CONDITIONING', 'CONDITIONING', 'LATENT', 'VAE')
        RETURN_NAMES = ('model', 'positive', 'negative', 'latent', 'vae')
        FUNCTION = 'prepare'
        CATEGORY = 'Krea/Workbench'
        @classmethod
        def IS_CHANGED(cls, **kwargs):
            return float('nan')
        def prepare(self, **kwargs):
            return profile.prepare_workbench(**kwargs)
    return {'BenchKreaWorkbench': Workbench}
