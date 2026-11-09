"""Importable Comfy workflows and strict graph admission for the Krea workbench."""
import argparse
import copy
import json
import math
from pathlib import Path, PurePosixPath

from krea_sizes import DEFAULT_SIZE, LEGACY_SIZE, output_size

MODES = ('text', 'style_reference', 'identity_edit')
PROMPT = 'A portrait of an adult man wearing a blue shirt, natural daylight, detailed photography'
DEFAULT_CHECKPOINT = 'krea2_turbo_int8_convrot.safetensors'
WORKBENCH_WIDGETS = ['prompt', 'mode'] + [key for index in range(1, 6)
    for key in (f'lora_{index}', f'strength_{index}')] + ['ref_boost', 'grounding_size', 'spectral', 'size', 'checkpoint']


def api_graph(mode='text', *, prompt=PROMPT, seed=42, source_image='reference.png', ref_image=None, spectral=False, size=DEFAULT_SIZE, checkpoint=DEFAULT_CHECKPOINT):
    if mode not in MODES:
        raise ValueError('Unknown Krea workbench mode')
    output_size(size)
    _validate_checkpoint(checkpoint)
    inputs = {'prompt': prompt, 'mode': mode}
    for index in range(1, 6):
        inputs.update({f'lora_{index}': '', f'strength_{index}': 1.0})
    if type(spectral) is not bool:
        raise ValueError('Spectral must be a boolean')
    inputs.update(ref_boost=4.0, grounding_size=512, spectral=spectral, size=size, checkpoint=checkpoint)
    graph = {
        '1': {'class_type': 'BenchKreaWorkbench', 'inputs': inputs},
        '2': {'class_type': 'BenchKreaSampler', 'inputs': {'model': ['1', 0],
            'positive': ['1', 1], 'negative': ['1', 2], 'latent_image': ['1', 3],
            'seed': seed, 'steps': 8, 'cfg': 1.0, 'sampler_name': 'euler',
            'scheduler': 'simple', 'denoise': 1.0}},
        '3': {'class_type': 'BenchKreaDecode', 'inputs': {'vae': ['1', 4], 'samples': ['2', 0]}},
        '4': {'class_type': 'SaveImage', 'inputs': {'images': ['3', 0], 'filename_prefix': f'krea-workbench/{mode}'}},
    }
    if mode != 'text':
        inputs['source_image'] = ['5', 0]
        graph['5'] = {'class_type': 'LoadImage', 'inputs': {'image': source_image}}
    if ref_image is not None:
        if mode != 'identity_edit':
            raise ValueError('A second reference is only supported for identity editing')
        inputs['ref_image'] = ['6', 0]
        inputs['grounding_size'] = 384
        graph['6'] = {'class_type': 'LoadImage', 'inputs': {'image': ref_image}}
    return graph


def _safe_relative(value, *, image=False):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        return False
    if image and value.endswith(' [input]'):
        value = value[:-8]
    if '[' in value or ']' in value:
        return False
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        return False
    return not image or path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _validate_checkpoint(value):
    # Catalog admission checks the contents; graphs may only select a basename.
    if (not _safe_relative(value) or '/' in value or '..' in value
            or not value.endswith('.safetensors')):
        raise ValueError('Use a safe Krea checkpoint .safetensors filename')


def validate_workbench_graph(current):
    """Return a canonical copy; accept only cosmetic metadata and node-ID changes.

    Callers may validate without replacing the submitted graph so browser node IDs
    and execution histories remain aligned. Unused nodes and extra inputs fail.
    """
    try:
        if not isinstance(current, dict) or not 4 <= len(current) <= 6:
            raise ValueError('Open a Krea Workbench preset (krea_text.json, krea_identity_edit.json, or krea_style_reference.json); this graph is not supported by this Workbench')
        nodes = {}
        by_class = {}
        for node_id, node in current.items():
            if (not isinstance(node_id, str) or not node_id or not isinstance(node, dict)
                    or set(node) - {'class_type', 'inputs', '_meta'}
                    or not {'class_type', 'inputs'}.issubset(node) or not isinstance(node['inputs'], dict)):
                raise ValueError('Malformed Krea workbench node')
            cls = node['class_type']
            if cls not in ('BenchKreaWorkbench', 'BenchKreaSampler', 'BenchKreaDecode', 'SaveImage', 'LoadImage'):
                raise ValueError('Unsupported Krea workbench node')
            nodes[node_id] = {'class_type': cls, 'inputs': copy.deepcopy(node['inputs'])}
            by_class.setdefault(cls, []).append(node_id)
        singletons = ('BenchKreaWorkbench', 'BenchKreaSampler', 'BenchKreaDecode', 'SaveImage')
        if any(len(by_class.get(cls, [])) != 1 for cls in singletons):
            raise ValueError('Expected exactly one workbench, sampler, decode and save node')
        mapping = {by_class[cls][0]: str(index) for index, cls in enumerate(singletons, 1)}
        inputs = nodes[by_class['BenchKreaWorkbench'][0]]['inputs']
        # Older API graphs retain ordinary sampling; the node input is optional.
        inputs.setdefault('spectral', False)
        inputs.setdefault('size', LEGACY_SIZE)
        inputs.setdefault('checkpoint', DEFAULT_CHECKPOINT)
        _validate_checkpoint(inputs['checkpoint'])
        output_size(inputs['size'])
        if type(inputs['spectral']) is not bool:
            raise ValueError('Spectral must be a boolean')
        mode, prompt = inputs['mode'], inputs['prompt']
        if mode not in MODES or not isinstance(prompt, str):
            raise ValueError('Invalid workbench mode or prompt')
        selected = []
        for index in range(1, 6):
            name, strength = inputs[f'lora_{index}'], inputs[f'strength_{index}']
            if (not isinstance(name, str) or (name and (not _safe_relative(name)
                    or '/' in name or '..' in name or not name.endswith('.safetensors')))
                    or not _finite(strength) or strength == 0):
                raise ValueError('Invalid workbench LoRA selection')
            if name:
                selected.append(name)
        if mode != 'text' and (len(selected) > 4 or inputs['lora_5']):
            raise ValueError('The mode adapter occupies one of the five LoRA slots')
        if not _finite(inputs['ref_boost']) or not .25 <= inputs['ref_boost'] <= 8:
            raise ValueError('Reference boost must be between .25 and eight')
        if type(inputs['grounding_size']) is not int or inputs['grounding_size'] not in (384, 512):
            raise ValueError('Reference grounding must be 384 or 512 pixels')
        if 'ref_image' in inputs and inputs['grounding_size'] != 384:
            raise ValueError('Two-reference Identity Edit requires 384px grounding in this memory profile')
        images = {}
        for key, canonical_id in (('source_image', '5'), ('ref_image', '6')):
            if key not in inputs:
                continue
            link = inputs[key]
            if (not isinstance(link, list) or len(link) != 2 or type(link[1]) is not int
                    or link[1] != 0 or link[0] not in nodes
                    or nodes[link[0]]['class_type'] != 'LoadImage' or link[0] in mapping):
                raise ValueError('Reference input must link to a distinct LoadImage IMAGE output')
            image = nodes[link[0]]['inputs']['image']
            if not _safe_relative(image, image=True):
                raise ValueError('Use a PNG, JPEG or WebP from the Comfy input directory')
            mapping[link[0]] = canonical_id
            images[key] = image
        if len(mapping) != len(nodes):
            raise ValueError('Unconnected or duplicate reference nodes are not supported')
        if (mode == 'text' and images) or (mode != 'text' and 'source_image' not in images):
            raise ValueError('Reference images do not match the workbench mode')
        if 'ref_image' in images and mode != 'identity_edit':
            raise ValueError('Only identity editing accepts a second reference')
        sampler = nodes[by_class['BenchKreaSampler'][0]]['inputs']
        seed = sampler['seed']
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError('Invalid Krea seed')
        if (type(sampler['steps']) is not int or sampler['steps'] != 8
                or not _finite(sampler['cfg']) or sampler['cfg'] != 1
                or not _finite(sampler['denoise']) or sampler['denoise'] != 1):
            raise ValueError('The workbench requires eight steps, CFG one and full denoise')
        prefix = nodes[by_class['SaveImage'][0]]['inputs']['filename_prefix']
        if not _safe_relative(prefix) or not prefix.startswith(f'krea-workbench/{mode}'):
            raise ValueError('Use a safe krea-workbench mode output prefix')
        normalized = {}
        for node_id, node in nodes.items():
            for key, value in node['inputs'].items():
                if isinstance(value, list):
                    if len(value) != 2 or value[0] not in mapping or type(value[1]) is not int:
                        raise ValueError('Invalid workbench graph link')
                    node['inputs'][key] = [mapping[value[0]], value[1]]
            normalized[mapping[node_id]] = node
        expected = api_graph(mode, prompt=prompt, seed=seed,
            source_image=images.get('source_image', 'reference.png'), ref_image=images.get('ref_image'))
        for key in WORKBENCH_WIDGETS:
            expected['1']['inputs'][key] = inputs[key]
        expected['4']['inputs']['filename_prefix'] = prefix
        if normalized != expected:
            raise ValueError('Graph differs from the supported Krea workbench recipe')
        return normalized
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed Krea workbench graph') from error


canonicalize_graph = validate_workbench_graph


def ui_workflow(mode='text', *, second_reference=False, size=DEFAULT_SIZE, checkpoint=DEFAULT_CHECKPOINT):
    """Build a LiteGraph v0.4 export whose widgets follow INPUT_TYPES order."""
    api = api_graph(mode, ref_image='person-reference.png' if second_reference else None, size=size, checkpoint=checkpoint)
    specs = {
        '1': ([-520, 0], [440, 660], ['source_image', 'ref_image'],
              [('MODEL', 'MODEL'), ('positive', 'CONDITIONING'), ('negative', 'CONDITIONING'),
               ('latent', 'LATENT'), ('VAE', 'VAE')], [api['1']['inputs'][key] for key in WORKBENCH_WIDGETS]),
        '2': ([0, 0], [320, 310], ['model', 'positive', 'negative', 'latent_image'],
              [('LATENT', 'LATENT')], [42, 'fixed', 8, 1.0, 'euler', 'simple', 1.0]),
        '3': ([380, 0], [210, 100], ['samples', 'vae'], [('IMAGE', 'IMAGE')], []),
        '4': ([650, 0], [410, 410], ['images'], [], [api['4']['inputs']['filename_prefix']]),
        '5': ([-1040, 0], [430, 360], [], [('IMAGE', 'IMAGE'), ('MASK', 'MASK')], ['reference.png', 'image']),
        '6': ([-1040, 420], [430, 360], [], [('IMAGE', 'IMAGE'), ('MASK', 'MASK')], ['person-reference.png', 'image']),
    }
    types = {'source_image': 'IMAGE', 'ref_image': 'IMAGE', 'model': 'MODEL',
        'positive': 'CONDITIONING', 'negative': 'CONDITIONING', 'latent_image': 'LATENT',
        'samples': 'LATENT', 'vae': 'VAE', 'images': 'IMAGE'}
    nodes = {}
    links = []
    for node_id, entry in api.items():
        pos, size, input_names, outputs, widgets = specs[node_id]
        nodes[node_id] = {'id': int(node_id), 'type': entry['class_type'], 'pos': pos, 'size': size,
            'flags': {}, 'order': len(nodes), 'mode': 0,
            'inputs': [{'name': key, 'type': types[key], 'link': None} for key in input_names],
            'outputs': [{'name': name, 'type': kind, 'links': None} for name, kind in outputs],
            'properties': {'Node name for S&R': entry['class_type']}, 'widgets_values': widgets}
    for node_id, entry in api.items():
        for slot, socket in enumerate(nodes[node_id]['inputs']):
            value = entry['inputs'].get(socket['name'])
            if not isinstance(value, list):
                continue
            origin, origin_slot = value
            link_id = len(links) + 1
            links.append([link_id, int(origin), origin_slot, int(node_id), slot, socket['type']])
            socket['link'] = link_id
            output = nodes[origin]['outputs'][origin_slot]
            output['links'] = (output['links'] or []) + [link_id]
    return {'last_node_id': max(map(int, nodes)), 'last_link_id': len(links),
        'nodes': list(nodes.values()), 'links': links, 'groups': [], 'config': {},
        'extra': {'ds': {'scale': 0.8, 'offset': [1170, 90]}}, 'version': 0.4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'workflows')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        for suffix, graph in (('', ui_workflow(mode)), ('_api', api_graph(mode))):
            (args.output_dir / f'krea_{mode}{suffix}.json').write_text(json.dumps(graph, indent=2) + '\n')
    for suffix, graph in (('', ui_workflow('identity_edit', second_reference=True)),
                          ('_api', api_graph('identity_edit', ref_image='person-reference.png'))):
        (args.output_dir / f'krea_identity_edit_two_refs{suffix}.json').write_text(json.dumps(graph, indent=2) + '\n')


if __name__ == '__main__':
    main()
