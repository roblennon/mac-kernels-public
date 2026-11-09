"""Qwen preparation (text or edit) with no cached encoder output."""
import gc
import weakref


def encode_and_release(clip_name, prompt, negative_prompt, vae=None, resolution=1024, images=None):
    """Encode on CPU through ComfyUI's own Qwen 2.1 node, then free the encoder before diffusion.

    With images and a VAE this is the edit path: the encoder sees the images and their VAE latents
    become reference latents; the returned latent follows image 1's resized size.
    """
    import nodes
    import comfy.model_management as mm
    from comfy_extras.nodes_qwen import TextEncodeQwenImage21

    clip = nodes.CLIPLoader().load_clip(clip_name, 'qwen_image', 'cpu')[0]
    refs = [weakref.ref(value) for value in (clip, clip.cond_stage_model, clip.patcher)]
    try:
        positive, negative, latent = TextEncodeQwenImage21.execute(
            clip, prompt, negative_prompt, vae=vae, resolution=resolution, images=images).result
    finally:
        mm.unload_model_and_clones(clip.patcher, all_devices=True)
        del clip
        gc.collect()
        mm.cleanup_models_gc()
        mm.soft_empty_cache()
    if any(ref() is not None for ref in refs):
        raise RuntimeError('Qwen conditioning retained its encoder; refusing diffusion allocation')
    return positive, negative, latent


def viggle_sigmas(width, height):
    """Viggle turbo v0.3's six raw steps through its resolution shift (ViggleTurboSigmas), then 0."""
    import math
    tokens = (height // 16) * (width // 16)
    mu = 0.5 + (0.9 - 0.5) * (tokens - 256) / (8192 - 256)
    return [math.exp(mu) / (math.exp(mu) + (1 / t - 1)) for t in (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)] + [0.0]


class BenchQwenViggleSize:
    """One size control for the Viggle preset: the empty latent and its matching schedule."""
    @classmethod
    def INPUT_TYPES(cls):
        size = {'default': 1024, 'min': 256, 'max': 2048, 'step': 16}
        return {'required': {'width': ('INT', size), 'height': ('INT', size)}}

    RETURN_TYPES = ('LATENT', 'SIGMAS')
    RETURN_NAMES = ('latent', 'sigmas')
    FUNCTION = 'build'
    CATEGORY = 'mac-kernels/qwen'

    def build(self, width, height):
        import nodes
        import torch
        latent = nodes.EmptyLatentImage().generate(width, height, 1)[0]
        return latent, torch.tensor(viggle_sigmas(width, height), dtype=torch.float32)


class BenchQwenPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        from .qwen_memory import ADMITTED
        return {'required': {
            'unet_name': (list(ADMITTED),),
            'clip_name': (folder_paths.get_filename_list('text_encoders'),),
            'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
            'negative_prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
        }}

    RETURN_TYPES = ('MODEL', 'CONDITIONING', 'CONDITIONING')
    RETURN_NAMES = ('model', 'positive', 'negative')
    FUNCTION = 'prepare'
    CATEGORY = 'mac-kernels/qwen'

    def prepare(self, unet_name, clip_name, prompt, negative_prompt):
        from .qwen_memory import load_bounded
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
        positive, negative, _ = encode_and_release(clip_name, prompt, negative_prompt)
        model = load_bounded(unet_name)
        return model, positive, negative


class BenchQwenEditPrepare:
    """Edit instruction, reference images and checkpoint: ComfyUI's Qwen 2.1 edit encoding, bounded load.

    Image 1 is the picture being edited and sets the output size; images 2 and 3 are optional
    references, named in the prompt as <image2> and <image3>. Sampling runs at CFG 1, so there is
    no negative prompt.
    """
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        from .qwen_memory import ADMITTED
        return {'required': {
            'unet_name': (list(ADMITTED),),
            'clip_name': (folder_paths.get_filename_list('text_encoders'),),
            'vae': ('VAE',),
            'image_1': ('IMAGE',),
            'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
            'resolution': ('INT', {'default': 1024, 'min': 0, 'max': 2048, 'step': 32,
                'tooltip': 'Image 1 is resized to about this many pixels square, keeping its aspect ratio. 0 keeps its own size.'}),
        }, 'optional': {'image_2': ('IMAGE',), 'image_3': ('IMAGE',)}}

    RETURN_TYPES = ('MODEL', 'CONDITIONING', 'LATENT')
    RETURN_NAMES = ('model', 'positive', 'latent')
    FUNCTION = 'prepare'
    CATEGORY = 'mac-kernels/qwen'

    def prepare(self, unet_name, clip_name, vae, image_1, prompt, resolution, image_2=None, image_3=None):
        # ComfyUI numbers connected images consecutively, so a gap would silently rename <image3>.
        if image_3 is not None and image_2 is None:
            raise ValueError('Connect image_2 before image_3')
        from .qwen_memory import load_bounded
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
        images = {name: image for name, image in
                  (('image_1', image_1), ('image_2', image_2), ('image_3', image_3)) if image is not None}
        positive, _, latent = encode_and_release(clip_name, prompt, '', vae=vae, resolution=resolution, images=images)
        # Reference encoding left the VAE on the GPU; free it for diffusion. Decode loads it again.
        mm.unload_model_and_clones(vae.patcher, all_devices=True)
        mm.soft_empty_cache()
        model = load_bounded(unet_name)
        return model, positive, latent


NODE_CLASS_MAPPINGS = {'BenchQwenPrepare': BenchQwenPrepare, 'BenchQwenEditPrepare': BenchQwenEditPrepare,
                       'BenchQwenViggleSize': BenchQwenViggleSize}
NODE_DISPLAY_NAME_MAPPINGS = {'BenchQwenPrepare': 'Qwen 2.1 checkpoint and prompt',
                              'BenchQwenEditPrepare': 'Qwen 2.1 edit: images, instruction and checkpoint',
                              'BenchQwenViggleSize': 'Viggle size and schedule'}
