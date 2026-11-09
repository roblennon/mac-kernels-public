"""Opt-in equivalent LoHa/LoKr representations of the real Viggle Qwen adapter.

Preparation is CPU-only. Generation reuses run_qwen21.py unchanged, registers the
proof directory only in that process, and requires the existing GPU supervisor.
These degenerate representations test loader/application routes; they do not
establish quality for independently trained LoHa or LoKr adapters.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[1]
SOURCE_NAME = 'Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors'
MODEL_NAME = 'qwen_image_2.1_int8_convrot.safetensors'
TYPE_NAMES = {'lora': 'LoRAAdapter', 'loha': 'LoHaAdapter', 'lokr': 'LoKrAdapter'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def convert_state(state, kind):
    """Keep learned factors unchanged; add an exact identity factorization."""
    import torch
    if kind not in ('loha', 'lokr'):
        raise ValueError('Expected loha or lokr')
    prefixes = sorted(key[:-len('.lora_A.weight')] for key in state if key.endswith('.lora_A.weight'))
    expected = {prefix + suffix for prefix in prefixes for suffix in ('.lora_A.weight', '.lora_B.weight')}
    if not prefixes or set(state) != expected:
        raise ValueError('Expected only complete PEFT A/B factor pairs; no silently dropped tensors')
    converted = {}
    for prefix in prefixes:
        down, up = state[prefix + '.lora_A.weight'], state[prefix + '.lora_B.weight']
        if (down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]
                or down.device.type != 'cpu' or up.device.type != 'cpu'
                or not down.is_floating_point() or down.dtype != up.dtype
                or not torch.isfinite(down).all() or not torch.isfinite(up).all()):
            raise ValueError('Invalid ordinary CPU factor pair: ' + prefix)
        if kind == 'loha':
            converted.update({prefix + '.hada_w1_a': up, prefix + '.hada_w1_b': down,
                              prefix + '.hada_w2_a': torch.ones(up.shape[0], 1, dtype=up.dtype),
                              prefix + '.hada_w2_b': torch.ones(1, down.shape[1], dtype=down.dtype)})
        else:
            converted.update({prefix + '.lokr_w1': torch.ones(1, 1, dtype=up.dtype),
                              prefix + '.lokr_w2_a': up, prefix + '.lokr_w2_b': down})
    return converted, prefixes


def load_adapter(state, prefix, kind):
    import comfy.lora
    adapter = comfy.lora.load_lora(state, {prefix: 'weight'}, log_missing=False)['weight']
    if type(adapter).__name__ != TYPE_NAMES[kind]:
        raise AssertionError(f'Wrong loaded type: {type(adapter).__name__}')
    return adapter


def delta(adapter, shape):
    import torch
    import comfy.lora
    return comfy.lora.calculate_weight([(1.0, adapter, 1.0, None, None)], torch.zeros(shape), 'weight')


def prove_assets(source, assets, prefixes, model_path):
    """All-pair bounded slice parity, plus one full real checkpoint layer."""
    import torch
    from safetensors import safe_open
    import comfy.lora
    import comfy.ops
    slices, admission = 0, {kind: Counter() for kind in assets}
    with safe_open(source, framework='pt', device='cpu') as original:
        for kind, path in assets.items():
            with safe_open(path, framework='pt', device='cpu') as converted:
                for prefix in prefixes:
                    down = original.get_tensor(prefix + '.lora_A.weight')
                    up = original.get_tensor(prefix + '.lora_B.weight')
                    # Check complete learned factors survived serialization unchanged.
                    names = ('.hada_w1_a', '.hada_w1_b') if kind == 'loha' else ('.lokr_w2_a', '.lokr_w2_b')
                    assert torch.equal(converted.get_tensor(prefix + names[0]), up)
                    assert torch.equal(converted.get_tensor(prefix + names[1]), down)
                    full = {key: converted.get_tensor(key) for key in converted.keys() if key.startswith(prefix + '.')}
                    small = {prefix + '.lora_A.weight': down[:, :64], prefix + '.lora_B.weight': up[:64]}
                    if kind == 'loha':
                        equivalent = {prefix + '.hada_w1_a': full[prefix + '.hada_w1_a'][:64],
                                      prefix + '.hada_w1_b': full[prefix + '.hada_w1_b'][:, :64],
                                      prefix + '.hada_w2_a': full[prefix + '.hada_w2_a'][:64],
                                      prefix + '.hada_w2_b': full[prefix + '.hada_w2_b'][:, :64]}
                    else:
                        equivalent = {prefix + '.lokr_w1': full[prefix + '.lokr_w1'],
                                      prefix + '.lokr_w2_a': full[prefix + '.lokr_w2_a'][:64],
                                      prefix + '.lokr_w2_b': full[prefix + '.lokr_w2_b'][:, :64]}
                    expected = delta(load_adapter(small, prefix, 'lora'), (min(64, up.shape[0]), min(64, down.shape[1])))
                    actual = delta(load_adapter(equivalent, prefix, kind), expected.shape)
                    assert torch.equal(actual, expected) and torch.count_nonzero(expected) > 0
                    slices += 1
                    adapter = load_adapter(full, prefix, kind)
                    side = comfy.lora.lora_side_patches([(1., adapter, 1., None, None)], up.shape[0], down.shape[1],
                                                       torch.device('cpu'), torch.bfloat16)
                    admission[kind]['side' if side is not None else 'merge'] += 1
                    del full, adapter, side, expected, actual
            print(json.dumps({'event': 'cpu_slices_proved', 'kind': kind, 'pairs': len(prefixes),
                              'standalone_admission': dict(admission[kind])}), flush=True)
    prefix = 'transformer.transformer_blocks.0.attn.to_q'
    base_prefix = prefix.removeprefix('transformer.') + '.'
    with safe_open(model_path, framework='pt', device='cpu') as base:
        state = {key[len(base_prefix):]: base.get_tensor(key) for key in base.keys() if key.startswith(base_prefix)}
    rows, cols = state['weight'].shape
    layer = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32).Linear(cols, rows, bias=False, device='cpu')
    layer.load_state_dict(state, strict=True)
    weight = layer.weight.dequantize()
    before = sha256(source)
    full_result = {}
    with safe_open(source, framework='pt', device='cpu') as original:
        raw = {key: original.get_tensor(key) for key in original.keys() if key.startswith(prefix + '.')}
    original_adapter = load_adapter(raw, prefix, 'lora')
    reference = comfy.lora.calculate_weight([(1., original_adapter, 1., None, None)], weight.clone(), 'weight')
    x = torch.randn(2, cols, generator=torch.Generator().manual_seed(314))
    expected = torch.nn.functional.linear(x, reference)
    for kind, path in assets.items():
        with safe_open(path, framework='pt', device='cpu') as converted:
            raw = {key: converted.get_tensor(key) for key in converted.keys() if key.startswith(prefix + '.')}
        adapter = load_adapter(raw, prefix, kind)
        actual = comfy.lora.calculate_weight([(1., adapter, 1., None, None)], weight.clone(), 'weight')
        assert torch.equal(actual, reference)
        side = comfy.lora.lora_side_patches([(1., adapter, 1., None, None)], rows, cols, torch.device('cpu'), torch.float32)
        assert side is not None
        output = torch.nn.functional.linear(x, weight)
        for term in side:
            output = term.apply(output, x)
        relative = float((output - expected).norm() / expected.norm())
        assert torch.isfinite(output).all() and relative < 2e-5
        full_result[kind] = {'loaded_type': type(adapter).__name__, 'merged_weight_exact': True,
                             'side_terms': [type(term).__name__ for term in side], 'side_relative_l2': relative}
        del actual, side
    assert sha256(source) == before
    return {'all_pair_slice_checks': slices, 'slice_extent': 64, 'serialized_learned_factors_exact': True,
            'standalone_admission': {key: dict(value) for key, value in admission.items()},
            'actual_checkpoint_layer': base_prefix + 'weight', 'full_layer': full_result,
            'scope': 'Equivalent degenerate representations; no claim about all trained LoHa/LoKr structures.'}


def prepare(source, model_path, output):
    from probe_text_encoder_load import bootstrap_cpu, require_supervisor
    require_supervisor()  # Serializes this substantial CPU read with GPU experiments.
    bootstrap_cpu()
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    torch.set_num_threads(2)
    source = source.resolve(strict=True)
    model_path = model_path.resolve(strict=True)
    if output.exists():
        raise ValueError('Use a new proof directory; existing assets are never overwritten')
    with safe_open(source, framework='pt', device='cpu') as reader:
        source_metadata = reader.metadata() or {}
    peft = json.loads(source_metadata.get('lora_adapter_metadata', '{}'))
    if (peft.get('transformer.r') != 128 or peft.get('transformer.lora_alpha') != 128
            or peft.get('transformer.use_rslora') or peft.get('transformer.rank_pattern')
            or peft.get('transformer.alpha_pattern')):
        raise ValueError('Expected Viggle rank128/alpha128 without variable or RS scaling')
    source_hash = sha256(source)
    state = load_file(source, device='cpu')
    output.mkdir(parents=True)
    proof_id = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:8]
    assets = {}
    for kind in ('loha', 'lokr'):
        converted, prefixes = convert_state(state, kind)
        path = output / f'f3-{kind}-{proof_id}.safetensors'
        save_file(converted, str(path), metadata={
            'format': 'pt', 'proof_adapter_type': kind, 'source_sha256': source_hash,
            'source_filename': source.name, 'source_metadata': json.dumps(source_metadata),
            'conversion': 'Original learned factors unchanged; identity second Hadamard factor' if kind == 'loha'
                          else 'Original learned factors unchanged; scalar-one first Kronecker factor',
            'purpose': 'Local application-route acceptance only; not an independently trained adapter',
        })
        assets[kind] = path
        print(json.dumps({'event': 'asset_written', 'kind': kind, 'file': str(path),
                          'bytes': path.stat().st_size}), flush=True)
        del converted
    del state
    gc.collect()
    proof = prove_assets(source, assets, prefixes, model_path)
    assert sha256(source) == source_hash
    original_link = output / f'f3-lora-{proof_id}.safetensors'
    original_link.symlink_to(source)
    assets['lora'] = original_link
    manifest = {'schema_version': 1, 'source': str(source), 'source_sha256': source_hash,
                'model': str(model_path), 'model_stat': {'size': model_path.stat().st_size,
                                                       'mtime_ns': model_path.stat().st_mtime_ns},
                'factor_pairs': len(prefixes), 'source_metadata': source_metadata,
                'assets': {kind: {'filename': path.name, 'sha256': sha256(path), 'type': TYPE_NAMES[kind]}
                           for kind, path in assets.items()}, 'cpu_proof': proof}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'event': 'prepared', 'proof_directory': str(output), **proof}), flush=True)


class RouteAudit:
    """Count successful forwards and actual side application separately from attachment."""
    def __init__(self, expected_type):
        self.expected_type = expected_type
        self.layers = {}
        self.handles = []

    def attach(self, patcher):
        from comfy.quant_ops import QuantizedTensor
        for key, patches in patcher.patches.items():
            module = patcher.model.get_submodule(key.removesuffix('.weight'))
            kinds = sorted({type(patch[1]).__name__ for patch in patches})
            if kinds != [self.expected_type]:
                raise AssertionError(f'Unexpected loaded adapter type at {key}: {kinds}')
            row = {'key': key, 'types': kinds, 'quantized': isinstance(module.weight, QuantizedTensor),
                   'forward_calls': 0, 'input_act_calls': 0, 'side_calls': 0, 'fused_calls': 0}
            self.layers[id(module)] = row
            def reached(_module, _inputs, _output, row=row):
                row['forward_calls'] += 1
            self.handles.append(module.register_forward_hook(reached))

    def side(self, module, fused):
        row = self.layers.get(id(module))
        if row is not None:
            row['side_calls'] += 1
            row['fused_calls'] += int(fused)

    @contextmanager
    def watch_operations(self):
        import comfy.ops as ops
        original_act = ops.linear_input_act
        original_side = ops._side_linear
        original_apply = ops._apply_lora_side_patches

        def input_act(linear, *args, **kwargs):
            output = original_act(linear, *args, **kwargs)
            row = self.layers.get(id(linear))
            if row is not None:
                row['input_act_calls'] += 1
            return output

        def side_linear(linear, *args, **kwargs):
            output, fused = original_side(linear, *args, **kwargs)
            if output is not None and fused:
                self.side(linear, fused=True)
            return output, fused

        def apply_side(linear, out, x, fused=False):
            output = original_apply(linear, out, x, fused=fused)
            # A fused term was already applied at the successful kernel boundary.
            if any(not (fused and term.fused) for term in linear.lora_side_patches):
                self.side(linear, fused=False)
            return output

        ops.linear_input_act = input_act
        ops._side_linear = side_linear
        ops._apply_lora_side_patches = apply_side
        try:
            yield
        finally:
            ops.linear_input_act = original_act
            ops._side_linear = original_side
            ops._apply_lora_side_patches = original_apply

    def result(self):
        rows = list(self.layers.values())
        return {'adapter_type': self.expected_type, 'patched_layers': len(rows),
                'reached_layers': sum(row['forward_calls'] > 0 or row['input_act_calls'] > 0 for row in rows),
                'side_reached_layers': sum(row['side_calls'] > 0 for row in rows),
                'fused_reached_layers': sum(row['fused_calls'] > 0 for row in rows), 'layers': rows}


    def require_complete(self, expected_layers, expected_side_layers):
        report = self.result()
        assert report['reached_layers'] == report['patched_layers'] == expected_layers
        assert report['side_reached_layers'] == sum(row['quantized'] for row in report['layers']) == expected_side_layers
        assert all((row['side_calls'] > 0) == row['quantized'] for row in report['layers'])
        return report


def run(proof_dir, kind, execute):
    manifest = json.loads((proof_dir / 'manifest.json').read_text())
    asset = proof_dir / manifest['assets'][kind]['filename']
    if sha256(asset) != manifest['assets'][kind]['sha256']:
        raise ValueError('Adapter asset differs from prepared CPU proof')
    model_path = Path(manifest['model'])
    if {'size': model_path.stat().st_size, 'mtime_ns': model_path.stat().st_mtime_ns} != manifest['model_stat']:
        raise ValueError('Base checkpoint file changed since preparation')
    environment = {'MODE': 'metal', 'UNET': model_path.name, 'W': '512', 'H': '512', 'STEPS': '6',
                   'CFG': '1', 'CACHE': 'off', 'LORA': asset.name, 'LORA_STRENGTH': '1',
                   'SIGMAS': 'viggle', 'VAE_FP16': '1', 'COMFY_MPS_TEXT_ENCODER': '0'}
    print(json.dumps({'event': 'generation_plan', 'env': environment, 'kind': kind,
                      'seed': 42, 'execute': execute, 'allocator_fraction': 0.5}), flush=True)
    if not execute:
        return
    from probe_text_encoder_load import require_supervisor
    require_supervisor()
    result_path = proof_dir / f'{kind}-generation.json'
    if result_path.exists():
        raise ValueError('Generation proof already exists; preserve it before another run')
    os.environ.update(environment)
    sys.path.insert(0, os.environ.get('COMFY_DIR', str(REPO / 'vendor/ComfyUI')))
    sys.argv = [sys.argv[0], '--use-pytorch-cross-attention', '--fp16-vae']
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(0.5)
    import folder_paths
    import nodes
    import comfy.ops
    folder_paths.add_model_folder_path('loras', str(proof_dir))
    folder_paths.add_model_folder_path('diffusion_models', str(model_path.parent), is_default=True)
    audit = RouteAudit(TYPE_NAMES[kind])
    loader = nodes.LoraLoaderModelOnly.load_lora_model_only
    def load_and_watch(self, *args, **kwargs):
        result = loader(self, *args, **kwargs)
        audit.attach(result[0])
        return result
    nodes.LoraLoaderModelOnly.load_lora_model_only = load_and_watch
    try:
        with audit.watch_operations():
            completed = runpy.run_path(str(REPO / 'bench/run_qwen21.py'), run_name='__main__')
        report = audit.require_complete(expected_layers=195, expected_side_layers=192)
        image = completed['img']
        assert torch.isfinite(image).all() and report['patched_layers'] > 0
        report.update({'runner_result': completed['res'], 'asset_sha256': manifest['assets'][kind]['sha256'],
                       'source_sha256': manifest['source_sha256'], 'recipe': environment, 'finite_image': True})
        result_path.write_text(json.dumps(report, indent=2) + '\n')
        print('ADAPTER_TYPE_AUDIT ' + json.dumps({key: value for key, value in report.items() if key != 'layers'}), flush=True)
    finally:
        nodes.LoraLoaderModelOnly.load_lora_model_only = loader
        for handle in audit.handles:
            handle.remove()


def main():
    root = Path(os.environ.get('MK_ROOT', os.path.expanduser('~/mac-kernels-data')))
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source', type=Path, default=root / 'models/qwen21/loras' / SOURCE_NAME)
    prep.add_argument('--model', type=Path, default=root / 'models/qwen21/diffusion_models' / MODEL_NAME)
    prep.add_argument('--out', type=Path, required=True)
    generate = sub.add_parser('run')
    generate.add_argument('--proof-dir', type=Path, required=True)
    generate.add_argument('--kind', choices=TYPE_NAMES, required=True)
    generate.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.source, args.model, args.out.resolve())
    else:
        run(args.proof_dir.resolve(), args.kind, args.execute)


if __name__ == '__main__':
    main()
