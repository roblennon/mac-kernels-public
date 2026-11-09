"""Native ComfyUI video presets; export both editable LiteGraph and API forms.

WAN 5B uses the accepted baked Turbo checkpoint. WAN 14B uses its qualified
bounded loader; H3 uses its scoped W4A8 loader at the documented test sizes.
"""
import argparse
import json
from pathlib import Path

PRESETS = ('wan5b_t2v', 'wan5b_i2v', 'wan_decode', 'wan14b_t2v', 'wan14b_i2v', 'h3_t2v', 'h3_i2v')
PROMPT = 'A street musician plays guitar in a retro subway station, warm light, commuters passing, slow camera movement, film grain.'
STACK_WIDGETS = [key for i in range(1, 6) for key in (f'lora_{i}', f'strength_{i}')]
# Widget order is the pinned stock node schema order. Seed controls are frontend widgets.
SCHEMAS = {
    'SaveLatent': ([('samples', 'LATENT')], [('samples', 'LATENT')], ['filename_prefix']),
    'BenchWanSavedLatent': ([], [('LATENT', 'LATENT')], ['latent']),
    'BenchWanDecode': ([('samples', 'LATENT'), ('vae', 'VAE')], [('IMAGE', 'IMAGE')], ['quality']),
    'BenchH3ModelLoader': ([], [('MODEL', 'MODEL')], ['unet_name']),
    'BenchWan14ModelLoader': ([], [('MODEL', 'MODEL')], ['unet_name']),
    'UNETLoader': ([], [('MODEL', 'MODEL')], ['unet_name', 'weight_dtype', 'mps_direct_load']),
    'CLIPLoader': ([], [('CLIP', 'CLIP')], ['clip_name', 'type', 'device']),
    'VAELoader': ([], [('VAE', 'VAE')], ['vae_name']),
    'BenchVideoLoRAStack': ([('model', 'MODEL')], [('MODEL', 'MODEL')], STACK_WIDGETS),
    'LoraLoaderModelOnly': ([('model', 'MODEL')], [('MODEL', 'MODEL')], ['lora_name', 'strength_model']),
    'ModelSamplingSD3': ([('model', 'MODEL')], [('MODEL', 'MODEL')], ['shift']),
    'CLIPTextEncode': ([('clip', 'CLIP')], [('CONDITIONING', 'CONDITIONING')], ['text']),
    'Wan22ImageToVideoLatent': ([('vae', 'VAE'), ('start_image', 'IMAGE')], [('LATENT', 'LATENT')], ['width', 'height', 'length', 'batch_size']),
    'WanImageToVideo': ([('positive', 'CONDITIONING'), ('negative', 'CONDITIONING'), ('vae', 'VAE'), ('clip_vision_output', 'CLIP_VISION_OUTPUT'), ('start_image', 'IMAGE'), ('ref_pad_image', 'IMAGE')], [('positive', 'CONDITIONING'), ('negative', 'CONDITIONING'), ('latent', 'LATENT')], ['width', 'height', 'length', 'batch_size']),
    'EmptyHunyuanLatentVideo': ([], [('LATENT', 'LATENT')], ['width', 'height', 'length', 'batch_size']),
    'LoadImage': ([], [('IMAGE', 'IMAGE'), ('MASK', 'MASK')], ['image', '@upload']),
    'KSampler': ([('model', 'MODEL'), ('positive', 'CONDITIONING'), ('negative', 'CONDITIONING'), ('latent_image', 'LATENT')], [('LATENT', 'LATENT')], ['seed', '@seed', 'steps', 'cfg', 'sampler_name', 'scheduler', 'denoise']),
    'KSamplerAdvanced': ([('model', 'MODEL'), ('positive', 'CONDITIONING'), ('negative', 'CONDITIONING'), ('latent_image', 'LATENT')], [('LATENT', 'LATENT')], ['add_noise', 'noise_seed', '@seed', 'steps', 'cfg', 'sampler_name', 'scheduler', 'start_at_step', 'end_at_step', 'return_with_leftover_noise']),
    'VAEDecode': ([('samples', 'LATENT'), ('vae', 'VAE')], [('IMAGE', 'IMAGE')], []),
    'VAEDecodeAudio': ([('samples', 'LATENT'), ('vae', 'VAE')], [('AUDIO', 'AUDIO')], []),
    'CreateVideo': ([('images', 'IMAGE'), ('audio', 'AUDIO')], [('VIDEO', 'VIDEO')], ['fps', 'bit_depth', 'color_space', 'codec']),
    'SaveVideo': ([('video', 'VIDEO')], [('VIDEO', 'VIDEO')], ['filename_prefix', 'format', 'format.codec', 'codec']),
    'MiniMaxH3ImageToVideo': ([('clip', 'CLIP'), ('vae', 'VAE'), ('first_frame', 'IMAGE'), ('last_frame', 'IMAGE')], [('positive', 'CONDITIONING'), ('LATENT', 'LATENT')], ['prompt', 'width', 'height', 'length']),
    'KSamplerSelect': ([], [('SAMPLER', 'SAMPLER')], ['sampler_name']),
    'BasicScheduler': ([('model', 'MODEL')], [('SIGMAS', 'SIGMAS')], ['scheduler', 'steps', 'denoise']),
    'BasicGuider': ([('model', 'MODEL'), ('conditioning', 'CONDITIONING')], [('GUIDER', 'GUIDER')], []),
    'RandomNoise': ([], [('NOISE', 'NOISE')], ['noise_seed', '@seed']),
    'SamplerCustomAdvanced': ([('noise', 'NOISE'), ('guider', 'GUIDER'), ('sampler', 'SAMPLER'), ('sigmas', 'SIGMAS'), ('latent_image', 'LATENT')], [('LATENT', 'LATENT'), ('denoised_output', 'LATENT')], []),
}


def api_graph(preset='wan5b_t2v'):
    if preset not in PRESETS:
        raise ValueError('Unknown video preset')
    graph = {}
    def add(kind, title=None, **inputs):
        node_id = str(len(graph) + 1)
        graph[node_id] = {'class_type': kind, 'inputs': inputs, '_meta': {'title': title or kind}}
        return [node_id, 0]
    def stack(model, title):
        return add('BenchVideoLoRAStack', title, model=model,
                   **{key: ('none' if key.startswith('lora') else 1.0) for key in STACK_WIDGETS})
    def unet(name, title=None, direct=False):
        return add('UNETLoader', title, unet_name=name, weight_dtype='default', mps_direct_load=direct)
    def vae(name):
        return add('VAELoader', vae_name=name)
    def save(samples, decoder, name, audio=None, fps=24.0):
        images = add('VAEDecode', samples=samples, vae=decoder)
        save_images(images, name, audio, fps)
    def save_images(images, name, audio=None, fps=24.0):
        video = add('CreateVideo', images=images, fps=fps, bit_depth='auto', color_space='sRGB', codec='none', **({'audio': audio} if audio else {}))
        add('SaveVideo', video=video, filename_prefix=f'video/{name}', format='auto', codec='auto', **{'format.codec': 'auto'})
    if preset == 'wan_decode':
        samples = add('BenchWanSavedLatent', 'Open saved WAN sample', latent='Headless/wan5b-metal-704x416-33f-4s-euler-delivery-canonical.pt')
        images = add('BenchWanDecode', 'Decode saved sample — no resampling', samples=samples, quality='Final (full)')
        save_images(images, 'wan_saved')
    elif preset.startswith('wan'):
        clip = add('CLIPLoader', clip_name='umt5_xxl_fp16.safetensors', type='wan', device='default')
        positive = add('CLIPTextEncode', 'Positive prompt', clip=clip, text=PROMPT)
        negative = add('CLIPTextEncode', 'Negative prompt', clip=clip, text='blurry, low quality, distorted, subtitles, watermark')
        if preset.startswith('wan5b'):
            decoder = vae('wan2.2_vae.safetensors')
            model = stack(unet('wan2.2_ti2v_5b_turbo_int8_convrot.safetensors', 'WAN 5B — baked Turbo', direct=preset.endswith('i2v')), 'Optional WAN 5B LoRAs')
            model = add('ModelSamplingSD3', model=model, shift=8.0)
            image = add('LoadImage', image='start-image.png') if preset.endswith('i2v') else None
            latent = add('Wan22ImageToVideoLatent', 'Size and frames (4k+1)', vae=decoder, width=416, height=704, length=33, batch_size=1, **({'start_image': image} if image else {}))
            samples = add('KSampler', model=model, positive=positive, negative=negative, latent_image=latent, seed=7, steps=4, cfg=1.0, sampler_name='euler', scheduler='simple', denoise=1.0)
        else:
            decoder = vae('wan_2.1_vae.safetensors')
            i2v = preset.endswith('i2v')
            if i2v:
                # A14B I2V: separate experts and Lightning pair; 81 frames is the model's native 16 fps length (5 s).
                image = add('LoadImage', 'Start image', image='start-image.png')
                cond = add('WanImageToVideo', 'Start image, size and frames', positive=positive, negative=negative, vae=decoder,
                           width=384, height=640, length=81, batch_size=1, start_image=image)
                positive, negative, latent = [cond[0], 0], [cond[0], 1], [cond[0], 2]
            else:
                latent = add('EmptyHunyuanLatentVideo', 'Size and frames', width=384, height=640, length=33, batch_size=1)
            task = 'i2v' if i2v else 't2v'
            models = []
            for expert in ('high', 'low'):
                model = add('BenchWan14ModelLoader', f'{expert.upper()} noise checkpoint', unet_name=f'wan2.2_{task}_{expert}_noise_14B_int8_convrot.safetensors')
                accelerator = (f'lightx2v-i2v-a14b/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/{expert}_noise_model.safetensors' if i2v
                               else f'wan2.2_t2v_lightx2v_4steps_lora_v1.1_{expert}_noise.safetensors')
                model = add('LoraLoaderModelOnly', f'{expert.upper()} noise accelerator', model=model, lora_name=accelerator, strength_model=1.0)
                model = stack(model, f'{expert.upper()} noise — optional matching LoRAs')
                models.append(add('ModelSamplingSD3', model=model, shift=5.0))
            common = dict(positive=positive, negative=negative, steps=4, cfg=1.0, sampler_name='euler', scheduler='simple')
            high = add('KSamplerAdvanced', 'HIGH noise — steps 0–2', model=models[0], latent_image=latent, add_noise='enable', noise_seed=7, start_at_step=0, end_at_step=2, return_with_leftover_noise='enable', **common)
            samples = add('KSamplerAdvanced', 'LOW noise — steps 2–4', model=models[1], latent_image=high, add_noise='disable', noise_seed=0, start_at_step=2, end_at_step=4, return_with_leftover_noise='disable', **common)
        if preset.startswith('wan5b'):
            samples = add('SaveLatent', 'Save sample before decoding', samples=samples, filename_prefix=f'latents/wan/{preset}')
            images = add('BenchWanDecode', 'Preview draft / final', samples=samples, quality='Draft (tiny)', vae=decoder)
            save_images(images, preset)
        else:
            save(samples, decoder, preset, fps=16.0 if preset == 'wan14b_i2v' else 24.0)
    else:
        clip = add('CLIPLoader', clip_name='qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', type='minimax', device='default')
        decoder = vae('minimax_h3_video_vae_fp16.safetensors')
        # Portrait by default (project default). I2V uses the measured 288×480×124 recipe
        # (10 min 55 s, 15.5 GiB peak); T2V keeps the short 22-frame wiring size. The node snaps length to 17k+5.
        image = add('LoadImage', 'First frame', image='start-image.png') if preset == 'h3_i2v' else None
        conditioning = add('MiniMaxH3ImageToVideo', 'H3 prompt, size and frames', clip=clip, vae=decoder, prompt=PROMPT,
                           width=288, height=480, length=124 if image else 22, **({'first_frame': image} if image else {}))
        model = add('BenchH3ModelLoader', 'H3 W4A8 checkpoint', unet_name='minimax_h3_fl2va_pruned_w4a8.safetensors')
        model = add('LoraLoaderModelOnly', 'H3 four-step accelerator', model=model, lora_name='minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors', strength_model=1.0)
        model = stack(model, 'Optional compatible H3 LoRAs')
        guider = add('BasicGuider', model=model, conditioning=conditioning)
        scheduler = add('BasicScheduler', model=model, scheduler='simple', steps=4, denoise=1.0)
        sampler = add('KSamplerSelect', sampler_name='res_multistep')
        noise = add('RandomNoise', noise_seed=1)
        samples = add('SamplerCustomAdvanced', noise=noise, guider=guider, sampler=sampler, sigmas=scheduler, latent_image=[conditioning[0], 1])
        audio = add('VAEDecodeAudio', samples=samples, vae=vae('minimax_h3_audio_vae_fp32.safetensors'))
        save(samples, decoder, preset, audio)
    return graph


def ui_workflow(preset='wan5b_t2v'):
    api = api_graph(preset)
    if __package__:
        from .workflow_canvas import ui_graph
    else:
        from workflow_canvas import ui_graph
    workflow = ui_graph(api, SCHEMAS)
    nodes = {str(node['id']): node for node in workflow['nodes']}
    if preset == 'wan_decode':
        for node in nodes.values():
            node['pos'] = [0 if node['id'] % 2 else 690, 80 if node['id'] <= 2 else 360]
            node['size'] = [420, 220 if node['type'] != 'SaveVideo' else 360]
            if node['type'] == 'BenchWanSavedLatent':
                node['size'][0] = 620
    elif preset == 'wan14b_t2v':
        # Expert chains run left-to-right on separate rows; the latent handoff
        # between the two samplers remains visible as an ordinary Comfy link.
        positions = {1: [-400, 0], 2: [-400, 200], 3: [-400, 600],
                     4: [2100, 0], 5: [1260, 1050],
                     6: [0, 0], 7: [420, 0], 8: [840, 0], 9: [1260, 0],
                     10: [0, 550], 11: [420, 550], 12: [840, 550], 13: [1260, 550],
                     14: [1680, 0], 15: [1680, 550], 16: [2100, 550],
                     17: [2520, 550], 18: [2940, 550]}
        for node in nodes.values():
            node['pos'] = positions[node['id']]
    return workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'workflows')
    target = parser.parse_args().output_dir
    target.mkdir(parents=True, exist_ok=True)
    for preset in PRESETS:
        for suffix, graph in (('', ui_workflow(preset)), ('_api', api_graph(preset))):
            (target / f'video_{preset}{suffix}.json').write_text(json.dumps(graph, indent=2) + '\n')


if __name__ == '__main__':
    main()
