"""Shipped editable presets must preserve sampler and expert routing."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from video_workflows import PRESETS, SCHEMAS, api_graph, ui_workflow


@pytest.mark.parametrize('preset', PRESETS)
def test_shipped_ui_reconstructs_api_and_bidirectional_links(preset):
    root = Path(__file__).resolve().parents[1] / 'bench/workflows'
    ui = json.loads((root / f'video_{preset}.json').read_text())
    api = json.loads((root / f'video_{preset}_api.json').read_text())
    assert ui == ui_workflow(preset)
    assert api == api_graph(preset)
    links = {link[0]: link for link in ui['links']}
    nodes = {node['id']: node for node in ui['nodes']}
    reconstructed = {}
    for node in ui['nodes']:
        _, _, widget_keys = SCHEMAS[node['type']]
        inputs = {key: value for key, value in zip(widget_keys, node['widgets_values'], strict=True) if not key.startswith('@')}
        for slot, socket in enumerate(node['inputs']):
            if socket['link'] is None:
                continue
            link, source, source_slot, target, target_slot, kind = links[socket['link']]
            assert (target, target_slot) == (node['id'], slot) and socket['type'] in (kind, '*')
            output = nodes[source]['outputs'][source_slot]
            assert output['type'] == kind and link in output['links']
            inputs[socket['name']] = [str(source), source_slot]
        reconstructed[str(node['id'])] = {'class_type': node['type'], 'inputs': inputs, '_meta': {'title': node['title']}}
    assert reconstructed == api


def test_wan14_expert_adapters_and_noise_handoff_are_independent():
    graph = api_graph('wan14b_t2v')
    samplers = [n['inputs'] for n in graph.values() if n['class_type'] == 'KSamplerAdvanced']
    high, low = samplers
    assert (high['add_noise'], high['start_at_step'], high['end_at_step'], high['return_with_leftover_noise']) == ('enable', 0, 2, 'enable')
    assert (low['add_noise'], low['start_at_step'], low['end_at_step'], low['return_with_leftover_noise']) == ('disable', 2, 4, 'disable')
    high_id = next(key for key, node in graph.items() if node['inputs'] is high)
    assert low['latent_image'] == [high_id, 0]
    for sampler, expert in zip(samplers, ('high', 'low'), strict=True):
        shift = graph[sampler['model'][0]]
        stack = graph[shift['inputs']['model'][0]]
        accelerator = graph[stack['inputs']['model'][0]]
        checkpoint = graph[accelerator['inputs']['model'][0]]
        assert stack['class_type'] == 'BenchVideoLoRAStack'
        assert f'{expert}_noise' in accelerator['inputs']['lora_name']
        assert f'{expert}_noise' in checkpoint['inputs']['unet_name']
    assert high['model'] != low['model']


def test_wan5b_i2v_uses_full_vae_and_only_i2v_has_image_link():
    for preset in ('wan5b_t2v', 'wan5b_i2v'):
        graph = api_graph(preset)
        latent = next(n['inputs'] for n in graph.values() if n['class_type'] == 'Wan22ImageToVideoLatent')
        assert (latent['width'], latent['height'], latent['length']) == (416, 704, 33)  # portrait default
        assert graph[latent['vae'][0]]['inputs']['vae_name'] == 'wan2.2_vae.safetensors'
        decode = next(n['inputs'] for n in graph.values() if n['class_type'] == 'BenchWanDecode')
        assert decode['vae'] == latent['vae']
        assert decode['quality'] == 'Draft (tiny)'
        assert ('start_image' in latent) == preset.endswith('i2v')
        sampler = next(n['inputs'] for n in graph.values() if n['class_type'] == 'KSampler')
        assert (sampler['steps'], sampler['cfg'], sampler['sampler_name']) == (4, 1.0, 'euler')
        loader = next(n['inputs'] for n in graph.values() if n['class_type'] == 'UNETLoader')
        assert loader['mps_direct_load'] is preset.endswith('i2v')


def test_h3_i2v_adds_only_a_first_frame_link_to_the_qualified_t2v_graph():
    t2v, i2v = api_graph('h3_t2v'), api_graph('h3_i2v')
    cond = next(n['inputs'] for n in i2v.values() if n['class_type'] == 'MiniMaxH3ImageToVideo')
    base = next(n['inputs'] for n in t2v.values() if n['class_type'] == 'MiniMaxH3ImageToVideo')
    assert i2v[cond['first_frame'][0]]['class_type'] == 'LoadImage' and 'last_frame' not in cond
    # I2V carries the measured portrait 124-frame recipe; everything but the length matches the T2V graph.
    assert (cond['width'], cond['height'], cond['length']) == (288, 480, 124) and base['length'] == 22
    assert {k: v for k, v in cond.items() if k not in ('first_frame', 'clip', 'vae', 'length')} == {k: v for k, v in base.items() if k not in ('clip', 'vae', 'length')}
    assert len(i2v) == len(t2v) + 1


def test_wan14b_i2v_uses_i2v_experts_lightning_pair_and_one_conditioning_node():
    graph = api_graph('wan14b_i2v')
    conds = [k for k, n in graph.items() if n['class_type'] == 'WanImageToVideo']
    assert len(conds) == 1
    cond = graph[conds[0]]['inputs']
    assert (cond['width'], cond['height'], cond['length']) == (384, 640, 81) and graph[cond['start_image'][0]]['class_type'] == 'LoadImage'
    samplers = [n['inputs'] for n in graph.values() if n['class_type'] == 'KSamplerAdvanced']
    assert all(s['positive'] == [conds[0], 0] and s['negative'] == [conds[0], 1] for s in samplers)
    assert samplers[0]['latent_image'] == [conds[0], 2]
    loaders = sorted(n['inputs']['unet_name'] for n in graph.values() if n['class_type'] == 'BenchWan14ModelLoader')
    assert loaders == ['wan2.2_i2v_high_noise_14B_int8_convrot.safetensors', 'wan2.2_i2v_low_noise_14B_int8_convrot.safetensors']
    accelerators = sorted(n['inputs']['lora_name'] for n in graph.values() if n['class_type'] == 'LoraLoaderModelOnly')
    assert accelerators == ['lightx2v-i2v-a14b/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors', 'lightx2v-i2v-a14b/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors']
    assert next(n['inputs']['fps'] for n in graph.values() if n['class_type'] == 'CreateVideo') == 16.0
    assert next(n['inputs']['fps'] for n in api_graph('wan14b_t2v').values() if n['class_type'] == 'CreateVideo') == 24.0
