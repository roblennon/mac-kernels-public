"""The Qwen canvas preserves its accepted recipe and editable model links."""
import json
from pathlib import Path
from bench.qwen_comfy import viggle_sigmas
from bench.qwen_memory import ADMITTED, TURBO_NAME
from bench.qwen_workflows import (SCHEMAS, TURBO_SIGMAS, VIGGLE_LORA, api_graph, edit_api_graph, edit_ui_workflow,
    turbo_api_graph, turbo_ui_workflow, ui_workflow, viggle_api_graph, viggle_ui_workflow)
from bench.run_templates import build_graph


import pytest


@pytest.mark.parametrize('name,ui_fn,api_fn', [('qwen_text', ui_workflow, api_graph), ('qwen_viggle', viggle_ui_workflow, viggle_api_graph),
                                               ('qwen_turbo', turbo_ui_workflow, turbo_api_graph), ('qwen_edit', edit_ui_workflow, edit_api_graph)])
def test_qwen_canvas_links_and_widgets_reconstruct_shipped_api(name, ui_fn, api_fn):
    root = Path(__file__).resolve().parents[1] / 'bench/workflows'
    ui = json.loads((root / f'{name}.json').read_text())
    api = json.loads((root / f'{name}_api.json').read_text())
    assert ui == ui_fn()
    assert api == api_fn()
    links = {link[0]: link for link in ui['links']}
    nodes = {node['id']: node for node in ui['nodes']}
    reconstructed = {}
    for node in ui['nodes']:
        _, _, keys = SCHEMAS[node['type']]
        inputs = {key: value for key, value in zip(keys, node['widgets_values'], strict=True) if not key.startswith('@')}
        for slot, socket in enumerate(node['inputs']):
            if socket['link'] is None:
                continue
            link, source, output_slot, target, input_slot, kind = links[socket['link']]
            assert (target, input_slot, kind) == (node['id'], slot, socket['type'])
            assert link in nodes[source]['outputs'][output_slot]['links']
            inputs[socket['name']] = [str(source), output_slot]
        reconstructed[str(node['id'])] = {'class_type': node['type'], 'inputs': inputs, '_meta': {'title': node['title']}}
    assert reconstructed == api


def test_qwen_recipe_preserves_live_acceptance_and_lora_route():
    original, _ = build_graph('image_qwen_image_2_1_t2i', 'accepted')
    graph = api_graph()
    def entry(kind):
        return next(n['inputs'] for n in graph.values() if n['class_type'] == kind)
    prepare = entry('BenchQwenPrepare')
    assert prepare['unet_name'] == original['459:451']['inputs']['unet_name']
    assert prepare['clip_name'] == original['459:453']['inputs']['clip_name']
    assert prepare['negative_prompt'] == original['459:452']['inputs']['negative_prompt']
    assert entry('VAELoader') == original['459:454']['inputs']
    assert entry('EmptyLatentImage') == original['459:456']['inputs']
    sampler = entry('KSampler')
    for key in ('seed', 'steps', 'cfg', 'sampler_name', 'scheduler', 'denoise'):
        assert sampler[key] == original['459:458']['inputs'][key]
    cache = graph[sampler['model'][0]]
    stack = graph[cache['inputs']['model'][0]]
    assert cache['class_type'] == 'QwenImage21Cache'
    assert stack['class_type'] == 'BenchVideoLoRAStack'
    assert graph[stack['inputs']['model'][0]]['class_type'] == 'BenchQwenPrepare'
    assert all(stack['inputs'][f'lora_{i}'] == 'none' for i in range(1, 6))
    assert sampler['positive'][0] == sampler['negative'][0]
    assert sampler['positive'][1] == 1 and sampler['negative'][1] == 2
    assert not any(n['class_type'] in ('TextGenerate', 'ResolutionSelector') for n in graph.values())


def test_viggle_preset_uses_its_lora_schedule_and_cfg1():
    graph = viggle_api_graph()
    def entry(kind):
        return next(n['inputs'] for n in graph.values() if n['class_type'] == kind)
    stack = entry('BenchVideoLoRAStack')
    assert stack['lora_1'] == VIGGLE_LORA and stack['strength_1'] == 1.0
    assert all(stack[f'lora_{i}'] == 'none' for i in range(2, 6))
    sampling = entry('SamplerCustomAdvanced')
    size = graph[sampling['sigmas'][0]]
    assert size['class_type'] == 'BenchQwenViggleSize' and sampling['latent_image'] == [sampling['sigmas'][0], 0]
    assert graph[sampling['guider'][0]]['class_type'] == 'BasicGuider'  # CFG 1: one model call per step
    assert entry('KSamplerSelect')['sampler_name'] == 'euler'
    # Viggle's published 1024^2 schedule, and the size dependence of its shift.
    assert [round(x, 4) for x in viggle_sigmas(1024, 1024)] == [1.0, 0.9678, 0.9334, 0.8572, 0.6668, 0.4001, 0.0]
    small = viggle_sigmas(512, 512)
    assert len(small) == 7 and small[0] == 1.0 and small[-1] == 0.0 and small[1] < viggle_sigmas(1024, 1024)[1]


def _entry(graph, kind):
    return next(n['inputs'] for n in graph.values() if n['class_type'] == kind)


@pytest.mark.parametrize('api_fn,prepare_kind', [(turbo_api_graph, 'BenchQwenPrepare'), (edit_api_graph, 'BenchQwenEditPrepare')])
def test_turbo_presets_use_the_admitted_checkpoint_and_its_saved_schedule(api_fn, prepare_kind):
    graph = api_fn()
    assert _entry(graph, prepare_kind)['unet_name'] == TURBO_NAME and TURBO_NAME in ADMITTED
    sampling = _entry(graph, 'SamplerCustomAdvanced')
    sigmas = graph[sampling['sigmas'][0]]
    assert sigmas['class_type'] == 'ManualSigmas'
    # Qwen-Image-2.1-Turbo's sample_sigmas (model card / pipeline config), then the terminal 0.
    assert [float(x) for x in sigmas['inputs']['sigmas'].split(',')] == [
        1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0]
    assert sigmas['inputs']['sigmas'] == TURBO_SIGMAS
    assert graph[sampling['guider'][0]]['class_type'] == 'BasicGuider'  # CFG 1
    assert _entry(graph, 'KSamplerSelect')['sampler_name'] == 'euler'
    stack = _entry(graph, 'BenchVideoLoRAStack')
    assert all(stack[f'lora_{i}'] == 'none' for i in range(1, 6))


def test_edit_preset_samples_on_the_encoders_latent_and_shares_one_vae():
    graph = edit_api_graph()
    prepare_id = next(k for k, n in graph.items() if n['class_type'] == 'BenchQwenEditPrepare')
    prepare = graph[prepare_id]['inputs']
    assert graph[prepare['image_1'][0]]['class_type'] == 'LoadImage'
    assert 'image_2' not in prepare and 'image_3' not in prepare  # optional references stay unwired
    sampling = _entry(graph, 'SamplerCustomAdvanced')
    # The latent follows image 1's resized size; any other size shifts the edit.
    assert sampling['latent_image'] == [prepare_id, 2]
    assert graph[sampling['guider'][0]]['inputs']['conditioning'] == [prepare_id, 1]
    assert _entry(graph, 'VAEDecode')['vae'] == prepare['vae']
    assert not any(n['class_type'] == 'EmptyLatentImage' for n in graph.values())


@pytest.mark.parametrize('node', ['BenchQwenPrepare', 'BenchQwenEditPrepare', 'BenchQwenViggleSize'])
def test_schema_widgets_follow_the_real_node_inputs(monkeypatch, node):
    # The frontend restores widgets_values in INPUT_TYPES order; SCHEMAS must not drift from it.
    import sys
    from types import SimpleNamespace
    from bench import qwen_comfy
    monkeypatch.setitem(sys.modules, 'folder_paths', SimpleNamespace(get_filename_list=lambda kind: ['file']))
    spec = getattr(qwen_comfy, node).INPUT_TYPES()
    inputs = {**spec['required'], **spec.get('optional', {})}
    widgets = [k for k, v in inputs.items() if isinstance(v[0], list) or v[0] in ('INT', 'FLOAT', 'STRING', 'BOOLEAN')]
    sockets = [(k, v[0]) for k, v in inputs.items() if k not in widgets]
    schema_sockets, _, schema_widgets = SCHEMAS[node]
    assert schema_widgets == widgets and schema_sockets == sockets
