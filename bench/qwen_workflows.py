"""Editable Qwen 2.1 presets with bounded native loading: the accepted recipe, Viggle turbo,
Qwen's official 8-step Turbo checkpoint, and Turbo image editing."""
import argparse
import json
from pathlib import Path

if __package__:
    from .run_templates import build_graph
    from .video_workflows import SCHEMAS as VIDEO_SCHEMAS, STACK_WIDGETS
    from .workflow_canvas import ui_graph
else:
    from run_templates import build_graph
    from video_workflows import SCHEMAS as VIDEO_SCHEMAS, STACK_WIDGETS
    from workflow_canvas import ui_graph

SCHEMAS = {**VIDEO_SCHEMAS,
    'BenchQwenPrepare': ([], [('model', 'MODEL'), ('positive', 'CONDITIONING'), ('negative', 'CONDITIONING')],
        ['unet_name', 'clip_name', 'prompt', 'negative_prompt']),
    'QwenImage21Cache': ([('model', 'MODEL')], [('MODEL', 'MODEL')], ['device', 'dtype']),
    'EmptyLatentImage': ([], [('LATENT', 'LATENT')], ['width', 'height', 'batch_size']),
    'SaveImage': ([('images', 'IMAGE')], [], ['filename_prefix']),
    'BenchQwenViggleSize': ([], [('latent', 'LATENT'), ('sigmas', 'SIGMAS')], ['width', 'height']),
    'BenchQwenEditPrepare': ([('vae', 'VAE'), ('image_1', 'IMAGE'), ('image_2', 'IMAGE'), ('image_3', 'IMAGE')],
        [('model', 'MODEL'), ('positive', 'CONDITIONING'), ('latent', 'LATENT')],
        ['unet_name', 'clip_name', 'prompt', 'resolution']),
    'ManualSigmas': ([], [('SIGMAS', 'SIGMAS')], ['sigmas']),
}
VIGGLE_LORA = 'Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors'
TURBO_UNET = 'qwen_image_2.1_turbo_int8_convrot.safetensors'
# Qwen-Image-2.1-Turbo's saved sample_sigmas. Its scheduler has dynamic shifting off, so they hold at every size.
TURBO_SIGMAS = '1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0'
EDIT_IMAGE = 'edit-image.png'
EDIT_PROMPT = 'Change the season to a snowy winter evening. Keep the subject, pose and framing unchanged.'


def _builder():
    graph = {}
    def add(kind, title=None, **inputs):
        key = str(len(graph) + 1)
        graph[key] = {'class_type': kind, 'inputs': inputs, '_meta': {'title': title or kind}}
        return [key, 0]
    return graph, add


def _lora_slots(first=None):
    slots = {key: 'none' if key.startswith('lora') else 1.0 for key in STACK_WIDGETS}
    if first:
        slots['lora_1'] = first
    return slots


def _layout(workflow, positions):
    for node in workflow['nodes']:
        node['pos'] = positions[node['id']]
        if node['type'] in ('BenchQwenPrepare', 'BenchQwenEditPrepare'):
            node['size'] = [360, 500]
        elif node['type'] == 'BenchVideoLoRAStack':
            node['size'] = [360, 460]
    return workflow


def api_graph():
    accepted, _ = build_graph('image_qwen_image_2_1_t2i', 'accepted')
    graph, add = _builder()
    model = add('BenchQwenPrepare', 'Qwen 2.1 checkpoint and prompt',
        unet_name=accepted['459:451']['inputs']['unet_name'],
        clip_name=accepted['459:453']['inputs']['clip_name'],
        prompt=accepted['459:474']['inputs']['on_false'],
        negative_prompt=accepted['459:452']['inputs']['negative_prompt'])
    conditioning = model[0]
    model = add('BenchVideoLoRAStack', 'Optional Qwen LoRAs', model=model, **_lora_slots())
    model = add('QwenImage21Cache', 'Qwen prefix cache', model=model, device='auto', dtype='default')
    latent = add('EmptyLatentImage', 'Image size — tested at 512 × 512', **accepted['459:456']['inputs'])
    inputs = accepted['459:458']['inputs']
    samples = add('KSampler', 'Seed and sampling', model=model, positive=[conditioning, 1],
        negative=[conditioning, 2], latent_image=latent,
        **{key: inputs[key] for key in ('seed', 'steps', 'cfg', 'sampler_name', 'scheduler', 'denoise')})
    vae = add('VAELoader', **accepted['459:454']['inputs'])
    images = add('VAEDecode', samples=samples, vae=vae)
    add('SaveImage', 'Save Qwen image', images=images, filename_prefix='qwen/image')
    return graph


def viggle_api_graph():
    """Viggle turbo v0.3: the LoRA in slot 1, its six-step schedule, CFG 1 (no negative prompt)."""
    accepted, _ = build_graph('image_qwen_image_2_1_t2i', 'accepted')
    graph, add = _builder()
    model = add('BenchQwenPrepare', 'Qwen 2.1 checkpoint and prompt',
        unet_name=accepted['459:451']['inputs']['unet_name'],
        clip_name=accepted['459:453']['inputs']['clip_name'],
        prompt=accepted['459:474']['inputs']['on_false'], negative_prompt='')
    positive = [model[0], 1]
    model = add('BenchVideoLoRAStack', 'Viggle turbo (slot 1) and optional LoRAs', model=model,
        **_lora_slots(VIGGLE_LORA))
    model = add('QwenImage21Cache', 'Qwen prefix cache', model=model, device='auto', dtype='default')
    size = add('BenchQwenViggleSize', 'Viggle size and schedule', width=832, height=1248)
    guider = add('BasicGuider', 'CFG 1 guider', model=model, conditioning=positive)
    noise = add('RandomNoise', 'Seed', noise_seed=accepted['459:458']['inputs']['seed'])
    sampler = add('KSamplerSelect', 'Sampler', sampler_name='euler')
    samples = add('SamplerCustomAdvanced', 'Viggle 6-step sampling', noise=noise, guider=guider,
        sampler=sampler, sigmas=[size[0], 1], latent_image=size)
    vae = add('VAELoader', **accepted['459:454']['inputs'])
    images = add('VAEDecode', samples=samples, vae=vae)
    add('SaveImage', 'Save Qwen Viggle image', images=images, filename_prefix='qwen/viggle')
    return graph


def turbo_api_graph():
    """Qwen-Image-2.1-Turbo: its own checkpoint, eight saved sigmas, Euler at CFG 1 (no negative prompt)."""
    accepted, _ = build_graph('image_qwen_image_2_1_t2i', 'accepted')
    graph, add = _builder()
    model = add('BenchQwenPrepare', 'Qwen 2.1 Turbo checkpoint and prompt', unet_name=TURBO_UNET,
        clip_name=accepted['459:453']['inputs']['clip_name'],
        prompt=accepted['459:474']['inputs']['on_false'], negative_prompt='')
    positive = [model[0], 1]
    model = add('BenchVideoLoRAStack', 'Optional Qwen LoRAs', model=model, **_lora_slots())
    model = add('QwenImage21Cache', 'Qwen prefix cache', model=model, device='auto', dtype='default')
    latent = add('EmptyLatentImage', 'Image size', width=832, height=1248, batch_size=1)
    guider = add('BasicGuider', 'CFG 1 guider', model=model, conditioning=positive)
    noise = add('RandomNoise', 'Seed', noise_seed=accepted['459:458']['inputs']['seed'])
    sampler = add('KSamplerSelect', 'Sampler', sampler_name='euler')
    sigmas = add('ManualSigmas', 'Turbo 8-step schedule', sigmas=TURBO_SIGMAS)
    samples = add('SamplerCustomAdvanced', 'Turbo 8-step sampling', noise=noise, guider=guider,
        sampler=sampler, sigmas=sigmas, latent_image=latent)
    vae = add('VAELoader', **accepted['459:454']['inputs'])
    images = add('VAEDecode', samples=samples, vae=vae)
    add('SaveImage', 'Save Qwen Turbo image', images=images, filename_prefix='qwen/turbo')
    return graph


def edit_api_graph():
    """Turbo image editing: ComfyUI's Qwen 2.1 edit encoding, output sized from image 1, Turbo sampling."""
    accepted, _ = build_graph('image_qwen_image_2_1_t2i', 'accepted')
    graph, add = _builder()
    source = add('LoadImage', 'Image to edit (image 1)', image=EDIT_IMAGE)
    vae = add('VAELoader', **accepted['459:454']['inputs'])
    prepare = add('BenchQwenEditPrepare', 'Qwen 2.1 edit: images, instruction and checkpoint',
        unet_name=TURBO_UNET, clip_name=accepted['459:453']['inputs']['clip_name'],
        vae=vae, image_1=source, prompt=EDIT_PROMPT, resolution=1024)
    model = add('BenchVideoLoRAStack', 'Optional Qwen LoRAs', model=prepare, **_lora_slots())
    model = add('QwenImage21Cache', 'Qwen prefix cache', model=model, device='auto', dtype='default')
    guider = add('BasicGuider', 'CFG 1 guider', model=model, conditioning=[prepare[0], 1])
    noise = add('RandomNoise', 'Seed', noise_seed=accepted['459:458']['inputs']['seed'])
    sampler = add('KSamplerSelect', 'Sampler', sampler_name='euler')
    sigmas = add('ManualSigmas', 'Turbo 8-step schedule', sigmas=TURBO_SIGMAS)
    samples = add('SamplerCustomAdvanced', 'Turbo 8-step edit', noise=noise, guider=guider,
        sampler=sampler, sigmas=sigmas, latent_image=[prepare[0], 2])
    images = add('VAEDecode', samples=samples, vae=vae)
    add('SaveImage', 'Save Qwen edit', images=images, filename_prefix='qwen/edit')
    return graph


def viggle_ui_workflow():
    return _layout(ui_graph(viggle_api_graph(), SCHEMAS), {1: [0, 0], 2: [420, 0], 3: [840, 0], 4: [840, 300],
        5: [1260, 0], 6: [1260, 200], 7: [1260, 380], 8: [1680, 0], 9: [1680, 420], 10: [2100, 0], 11: [2520, 0]})


def turbo_ui_workflow():
    return _layout(ui_graph(turbo_api_graph(), SCHEMAS), {1: [0, 0], 2: [420, 0], 3: [840, 0], 4: [840, 300],
        5: [1260, 0], 6: [1260, 200], 7: [1260, 380], 8: [1260, 560], 9: [1680, 0], 10: [1680, 420],
        11: [2100, 0], 12: [2520, 0]})


def edit_ui_workflow():
    return _layout(ui_graph(edit_api_graph(), SCHEMAS), {1: [0, 0], 2: [0, 620], 3: [420, 0], 4: [840, 0],
        5: [1260, 0], 6: [1260, 200], 7: [1260, 380], 8: [1260, 560], 9: [1260, 740], 10: [1680, 0],
        11: [2100, 0], 12: [2520, 0]})


def ui_workflow():
    return _layout(ui_graph(api_graph(), SCHEMAS), {1: [0, 0], 2: [420, 0], 3: [840, 0], 4: [840, 580],
        5: [1260, 0], 6: [1260, 580], 7: [1680, 0], 8: [2100, 0]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'workflows')
    target = parser.parse_args().output_dir
    target.mkdir(parents=True, exist_ok=True)
    for name, graph in (('qwen_text', ui_workflow()), ('qwen_text_api', api_graph()),
                        ('qwen_viggle', viggle_ui_workflow()), ('qwen_viggle_api', viggle_api_graph()),
                        ('qwen_turbo', turbo_ui_workflow()), ('qwen_turbo_api', turbo_api_graph()),
                        ('qwen_edit', edit_ui_workflow()), ('qwen_edit_api', edit_api_graph())):
        (target / f'{name}.json').write_text(json.dumps(graph, indent=2) + '\n')


if __name__ == '__main__':
    main()
