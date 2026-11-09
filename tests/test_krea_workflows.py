"""The browser export must encode only the bounded workbench graph."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from krea_sizes import SIZE_PRESETS
from krea_workflows import DEFAULT_CHECKPOINT, MODES, WORKBENCH_WIDGETS, api_graph, ui_workflow, validate_workbench_graph


@pytest.mark.parametrize('mode,second_reference', [(mode, False) for mode in MODES] + [('identity_edit', True)])
def test_shipped_ui_presets_reconstruct_exact_api_graph(mode, second_reference):
    root = Path(__file__).resolve().parents[1] / 'bench/workflows'
    name = f'krea_{mode}' + ('_two_refs' if second_reference else '')
    ui = json.loads((root / f'{name}.json').read_text())
    assert ui == ui_workflow(mode, second_reference=second_reference)
    api = json.loads((root / f'{name}_api.json').read_text())
    assert validate_workbench_graph(api) == api_graph(mode, ref_image='person-reference.png' if second_reference else None)
    links = {link[0]: link for link in ui['links']}
    result = {}
    for node in ui['nodes']:
        node_id = str(node['id'])
        values = node['widgets_values']
        if node_id == '1':
            inputs = dict(zip(WORKBENCH_WIDGETS, values, strict=True))
        elif node_id == '2':
            seed, control, steps, cfg, sampler, scheduler, denoise = values
            assert control == 'fixed'
            inputs = dict(seed=seed, steps=steps, cfg=cfg, sampler_name=sampler,
                          scheduler=scheduler, denoise=denoise)
        elif node_id == '3':
            inputs = {}
        elif node_id == '4':
            inputs = {'filename_prefix': values[0]}
        else:
            inputs = {'image': values[0]}
        for index, socket in enumerate(node['inputs']):
            if socket['link'] is None:
                continue
            link_id, origin, origin_slot, target, target_slot, kind = links[socket['link']]
            assert (target, target_slot, kind) == (node['id'], index, socket['type'])
            output = next(n for n in ui['nodes'] if n['id'] == origin)['outputs'][origin_slot]
            assert link_id in output['links'] and output['type'] == kind
            inputs[socket['name']] = [str(origin), origin_slot]
        result[node_id] = {'class_type': node['type'], 'inputs': inputs}
    assert result == api


def test_renumbering_and_cosmetic_metadata_preserve_bindings_without_mutating_request():
    graph = api_graph('identity_edit', source_image='uploads/scene.png [input]', ref_image='person.webp')
    renamed = {}
    mapping = {node_id: f'node-{int(node_id) + 30}' for node_id in graph}
    for node_id, node in copy.deepcopy(graph).items():
        node['_meta'] = {'title': 'A descriptive UI title'}
        for key, value in node['inputs'].items():
            if isinstance(value, list):
                node['inputs'][key] = [mapping[value[0]], value[1]]
        renamed[mapping[node_id]] = node
    before = copy.deepcopy(renamed)
    assert validate_workbench_graph(renamed) == graph
    assert renamed == before


def test_two_reference_grounding_has_a_lower_memory_budget():
    graph = api_graph('identity_edit', ref_image='person.png')
    assert graph['1']['inputs']['grounding_size'] == 384
    validate_workbench_graph(graph)
    graph['1']['inputs']['grounding_size'] = 512
    with pytest.raises(ValueError, match='Two-reference Identity Edit requires 384px'):
        validate_workbench_graph(graph)


@pytest.mark.parametrize('mode', MODES)
def test_spectral_is_explicit_and_old_api_recipes_stay_ordinary(mode):
    graph = api_graph(mode, spectral=True)
    assert validate_workbench_graph(graph) == graph
    del graph['1']['inputs']['spectral']
    before = copy.deepcopy(graph)
    assert validate_workbench_graph(graph)['1']['inputs']['spectral'] is False
    assert graph == before


@pytest.mark.parametrize('value', [0, 1, 'true', None])
def test_spectral_rejects_coerced_boolean_values(value):
    graph = api_graph()
    graph['1']['inputs']['spectral'] = value
    with pytest.raises(ValueError):
        validate_workbench_graph(graph)
    with pytest.raises(ValueError):
        api_graph(spectral=value)


@pytest.mark.parametrize('change', [
    lambda g: g['1']['inputs'].update(lora_1='../outside.safetensors'),
    lambda g: g['1']['inputs'].update(lora_1='ok.safetensors', strength_1=float('nan')),
    lambda g: g['1']['inputs'].update(strength_1=True),
    lambda g: g['1']['inputs'].update(lora_5='fifth.safetensors'),
    lambda g: g['1']['inputs'].update(grounding_size=513),
    lambda g: g['1']['inputs'].update(ref_boost=float('inf')),
    lambda g: g['1']['inputs'].update(ref_image=['5', 0]),
    lambda g: g['1']['inputs'].update(source_image=['5', 1]),
    lambda g: g['1']['inputs'].update(unapproved='value'),
    lambda g: g['2']['inputs'].update(steps=4),
    lambda g: g['2']['inputs'].update(cfg=True),
    lambda g: g['2']['inputs'].update(model=['1', 3]),
    lambda g: g['3']['inputs'].update(vae=['2', 0]),
    lambda g: g['4']['inputs'].update(filename_prefix='krea-workbench/../escape'),
    lambda g: g['5']['inputs'].update(image='../outside.png'),
    lambda g: g['5']['inputs'].update(image='outside.png [output]'),
    lambda g: g.update(extra={'class_type': 'LoadImage', 'inputs': {'image': 'unused.png'}}),
    lambda g: g['5'].update(unapproved='node metadata'),
])
def test_refuses_recipe_changes_untrusted_files_and_unused_nodes(change):
    graph = api_graph('identity_edit')
    change(graph)
    with pytest.raises(ValueError):
        validate_workbench_graph(graph)


def test_five_text_slots_and_four_edit_slots_admit_independent_strengths():
    for mode, count in (('text', 5), ('identity_edit', 4)):
        graph = api_graph(mode)
        for index in range(1, count + 1):
            graph['1']['inputs'].update({f'lora_{index}': f'local_{index}.safetensors',
                                         f'strength_{index}': index * 0.1})
        assert validate_workbench_graph(graph) == graph
    with pytest.raises(ValueError):
        validate_workbench_graph(api_graph('style_reference', ref_image='extra.png'))


def test_generic_comfy_starter_is_rejected_with_preset_guidance():
    graph = {str(index): {'class_type': 'KSampler', 'inputs': {}} for index in range(10)}
    with pytest.raises(ValueError, match=r'Open a Krea Workbench preset \(krea_text.json'):
        validate_workbench_graph(graph)


@pytest.mark.parametrize('preset,dimensions', SIZE_PRESETS.items())
def test_size_presets_bind_ui_api_and_legacy_request(preset, dimensions):
    from krea_sizes import DEFAULT_SIZE, LEGACY_SIZE, output_size
    graph = api_graph(size=preset)
    assert validate_workbench_graph(graph) == graph
    assert output_size(preset) == dimensions
    ui = ui_workflow(size=preset)
    assert ui['nodes'][0]['widgets_values'][WORKBENCH_WIDGETS.index('size')] == preset
    assert api_graph()['1']['inputs']['size'] == DEFAULT_SIZE
    legacy = copy.deepcopy(graph)
    del legacy['1']['inputs']['size']
    assert validate_workbench_graph(legacy)['1']['inputs']['size'] == LEGACY_SIZE
    for invalid in ('custom', '', None, 1024, ['512', '768']):
        legacy['1']['inputs']['size'] = invalid
        with pytest.raises(ValueError, match='size preset'):
            validate_workbench_graph(legacy)


@pytest.mark.parametrize('mode', MODES)
def test_checkpoint_selection_round_trips_and_old_api_preserves_default(mode):
    graph = api_graph(mode, checkpoint='my-custom-turbo.safetensors')
    assert validate_workbench_graph(graph) == graph
    ui = ui_workflow(mode, checkpoint='my-custom-turbo.safetensors')
    assert ui['nodes'][0]['widgets_values'][-1] == 'my-custom-turbo.safetensors'
    del graph['1']['inputs']['checkpoint']
    before = copy.deepcopy(graph)
    assert validate_workbench_graph(graph)['1']['inputs']['checkpoint'] == DEFAULT_CHECKPOINT
    assert graph == before


@pytest.mark.parametrize('checkpoint', [None, '', 1, ['model.safetensors'],
    '../model.safetensors', '/model.safetensors', 'dir/model.safetensors',
    'dir\\model.safetensors', 'model\x00.safetensors', 'model.pt',
    'model.safetensors [input]', 'model..safetensors'])
def test_checkpoint_selection_rejects_paths_and_invalid_filenames(checkpoint):
    with pytest.raises(ValueError, match='checkpoint'):
        api_graph(checkpoint=checkpoint)
    graph = api_graph()
    graph['1']['inputs']['checkpoint'] = checkpoint
    with pytest.raises(ValueError, match='checkpoint'):
        validate_workbench_graph(graph)
