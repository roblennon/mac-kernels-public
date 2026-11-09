"""Guarded two-step Qwen training with one real block converted to W4 or W6.

This is a mixed-base acceptance experiment, not a whole-checkpoint conversion or
an H3 training claim. Source model and dataset files are never changed. Recovery
identity names the experimental conversion so the ordinary trainer rejects it.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def tensor_digest(tensor):
    import torch
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(value)).hexdigest()


def packed_digest(weight):
    from comfy.quant_ops import AsymW4A8Int8Layout
    return [tensor_digest(t) if t is not None else None
            for t in AsymW4A8Int8Layout.get_plain_tensors(weight)]


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Artifact changed while hashing')
    return {'sha256': digest.hexdigest(), 'bytes': after.st_size}


def codec_fingerprints():
    import comfy_kitchen.tensor.w4a8_int8 as layout
    import comfy_kitchen.backends.eager.w4a8_int8 as codec
    import comfy_kitchen.backends.eager.quantization as rotation
    return {name: fingerprint(module.__file__) for name, module in
            [('layout', layout), ('codec', codec), ('rotation', rotation)]}


def record_packed(weight, relative_error, dequant_digest):
    # Preserve every layout field, including optional tensors and transpose state.
    return {'qdata': weight._qdata, 'params': {field.name: getattr(weight._params, field.name)
            for field in dataclasses.fields(weight._params)}, 'digest': packed_digest(weight),
            'conversion_relative_l2': relative_error, 'dequantized_sha256': dequant_digest}


def restore_packed(record, bits):
    import torch
    from comfy.quant_ops import AsymW4A8Int8Layout, QuantizedTensor
    from comfy_kitchen.backends.eager.w4a8_int8 import validate_w4a8_operands
    fields = {field.name for field in dataclasses.fields(AsymW4A8Int8Layout.Params)}
    values = record['params']
    if set(values) != fields:
        raise ValueError('Prepared layout fields do not match this implementation')
    if (values['orig_dtype'] != torch.bfloat16 or values['transposed'] is not False
            or values['group_size'] != (16 if bits == 4 else 32) or values['convrot_groupsize'] != 256):
        raise ValueError('Prepared layout parameters do not match the probe recipe')
    for value in [record['qdata'], *(v for v in values.values() if isinstance(v, torch.Tensor))]:
        if type(value) is not torch.Tensor or value.device.type != 'cpu' or not value.is_contiguous():
            raise ValueError('Prepared storage must be ordinary contiguous CPU tensors')
    params = AsymW4A8Int8Layout.Params(**values)
    n, k, stored_bits = validate_w4a8_operands(record['qdata'], params.scale, params.s_channel,
                                             params.codebook, params.correction,
                                             params.group_size, params.convrot_groupsize)
    if stored_bits != bits or tuple(params.orig_shape) != (n, k):
        raise ValueError('Prepared packed geometry does not match the requested weight')
    weight = QuantizedTensor(record['qdata'], 'AsymW4A8Int8Layout', params)
    if packed_digest(weight) != record['digest']:
        raise ValueError('Prepared packed tensor bytes changed')
    relative = record['conversion_relative_l2']
    if not math.isfinite(relative) or not 0 <= relative < (0.15 if bits == 4 else 0.06):
        raise ValueError('Prepared conversion error is outside the accepted bound')
    return weight


def validate_prepared(payload, bits, targets, *, verify_dequant=False):
    if (payload.get('version') != 1 or payload['bits'] != bits or payload['block'] != 1
            or set(payload['layers']) != set(targets) or payload['codec'] != codec_fingerprints()):
        raise ValueError('Prepared block does not match this probe and codec')
    for name, record in payload['layers'].items():
        weight = restore_packed(record, bits)
        if verify_dequant and tensor_digest(weight.dequantize()) != record['dequantized_sha256']:
            raise ValueError('Saved/reconstructed dequantized weight changed: ' + name)


def save_prepared(path, payload):
    import torch
    torch.save(payload, path)
    restored = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    validate_prepared(restored, payload['bits'], payload['layers'], verify_dequant=True)
    manifest = {'version': 1, 'artifact': fingerprint(path), 'base': restored['base'],
                'bits': restored['bits'], 'block': restored['block'], 'codec': restored['codec'],
                'targets': sorted(restored['layers']), 'serialized_dequantization_verified': True}
    Path(str(path) + '.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load_prepared(path, bits, targets):
    import torch
    manifest = json.loads(Path(str(path) + '.json').read_text())
    if (manifest.get('version') != 1 or manifest.get('serialized_dequantization_verified') is not True
            or manifest['artifact'] != fingerprint(path)):
        raise ValueError('Prepared artifact SHA256 or verification manifest does not match')
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    validate_prepared(payload, bits, targets)  # Packed-only validation: never full-weight decode.
    for key in ('base', 'bits', 'block', 'codec'):
        if payload[key] != manifest[key]:
            raise ValueError('Prepared manifest does not describe the artifact: ' + key)
    if sorted(payload['layers']) != manifest['targets']:
        raise ValueError('Prepared manifest layer set does not match')
    return payload, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bits', required=True, type=int, choices=(4, 6))
    parser.add_argument('--out', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true', help='Write a CPU-only converted block, then exit')
    mode.add_argument('--prepared', type=Path, help='Load a block prepared in a separate process')
    parser.add_argument('--no-mmap-offload', action='store_true', help='Diagnostic: disable mapped CPU backup retention')
    args = parser.parse_args()
    out = args.out.resolve()
    prepared_path = args.prepared.resolve() if args.prepared else None
    if out.exists():
        raise SystemExit('Use a new output directory for this non-resumable experiment.')
    from bench.probe_text_encoder_load import require_supervisor
    require_supervisor()
    import torch
    torch.set_num_threads(4)
    if args.prepare_only:
        from bench.probe_text_encoder_load import bootstrap_cpu
        bootstrap_cpu()
        import comfy.model_management  # Parse --cpu before comfy_boot is imported.
    else:
        torch.mps.set_per_process_memory_fraction(.6)
    sys.path.insert(0, str(REPO / 'train'))
    import train_lora as trainer
    import comfy_kitchen.backends.mps.packed_autograd as packed
    from comfy.quant_ops import QuantizedTensor, AsymW4A8Int8Layout
    import safetensors.torch as safe_torch
    from safetensors import safe_open

    root = Path(os.environ.get('MK_ROOT', os.path.expanduser('~/mac-kernels-data')))
    dataset = root / 'train/datasets/cyanoprint'
    # The real 512px cache is required; this experiment must not build new encoders.
    if not args.prepare_only and not (dataset / 'cache_qwen21_512.pt').is_file():
        raise SystemExit('The existing Qwen 512px cache is required.')
    spec = trainer.ARCHS['qwen21']
    prepared, provenance = (load_prepared(prepared_path, args.bits, spec['targets'])
                            if prepared_path else (None, None))
    converted = {}
    conversion_error = {}
    dequant_digest = {}
    calls = {name: {'forward': 0, 'input_gradient': 0, 'nonzero_gradient': 0} for name in spec['targets']}
    original_loader = trainer.cb.N['UNETLoader'].load_unet
    original_kernel = packed.w4a8_linear_trainable
    original_identity = trainer.training_identity
    original_backward = packed.w4a8_linear_grad_input
    original_save = safe_torch.save_file

    def load(loader, *pos, **kw):
        result = original_loader(loader, *pos, **kw)
        if args.no_mmap_offload:
            result[0].mmap_offload = False
        print(json.dumps({'event': 'offload_policy', 'mapped_backup': result[0].mmap_offload}), flush=True)
        block = result[0].model.diffusion_model.transformer_blocks[1]
        for name in spec['targets']:
            module = block.get_submodule(name)
            if module.weight.device.type != 'cpu':
                raise RuntimeError('Expected CPU model construction before device loading')
            if args.prepare_only:
                source = module.weight.detach().dequantize().to(torch.bfloat16)
                weight = QuantizedTensor.from_float(
                    source, 'AsymW4A8Int8Layout', bits=args.bits,
                    group_size=16 if args.bits == 4 else 32,
                    convrot_groupsize=256, codebook=args.bits == 4,
                )
                reference = source.float()
                decoded = weight.dequantize()
                dequant_digest[name] = tensor_digest(decoded)
                restored = decoded.float()
                del decoded
                relative = float((restored - reference).norm() / reference.norm())
                if not torch.isfinite(restored).all() or relative >= (0.15 if args.bits == 4 else 0.06):
                    raise RuntimeError(f'Converted weight differs excessively from source: {name}: {relative}')
                conversion_error[name] = relative
                del reference, restored, source
            else:
                record = prepared['layers'][name]
                weight = restore_packed(record, args.bits)
                if weight.shape != module.weight.shape:
                    raise ValueError('Prepared packed shape differs from source layer: ' + name)
                conversion_error[name] = record['conversion_relative_l2']
            module.weight = torch.nn.Parameter(weight, requires_grad=False)
            module.quant_format = 'asym_w4a8_int8' if args.bits == 4 else 'w6a8_int8'
            module.layout_type = 'AsymW4A8Int8Layout'
            converted[name] = (module, packed_digest(weight))
            del weight
        print(json.dumps({'event': 'converted', 'block': 1, 'bits': args.bits,
                          'layers': list(converted), 'conversion_relative_l2': conversion_error}), flush=True)
        return result

    def layer_name(qdata):
        for name, (module, _) in converted.items():
            current = AsymW4A8Int8Layout.get_plain_tensors(module.weight)[0]
            if current.data_ptr() == qdata.data_ptr():
                return name
        raise RuntimeError('Packed call did not belong to the converted block')

    def kernel(x, qdata, *pos, **kw):
        calls[layer_name(qdata)]['forward'] += 1
        return original_kernel(x, qdata, *pos, **kw)

    def backward(grad_out, qdata, *pos, **kw):
        # Inspect the kernel's own return, before autograd adds the LoRA branch.
        gradient = original_backward(grad_out, qdata, *pos, **kw)
        name = layer_name(qdata)
        if not torch.isfinite(gradient).all():
            raise RuntimeError('Non-finite input gradient through ' + name)
        calls[name]['input_gradient'] += 1
        calls[name]['nonzero_gradient'] += int(torch.count_nonzero(gradient) > 0)
        return gradient

    def save(tensors, filename, metadata=None):
        metadata = dict(metadata or {})
        metadata['experimental_conversion'] = json.dumps(
            {'block': 1, 'bits': args.bits, 'mixed_base': True, 'purpose': 'two-step acceptance probe',
             'prepared_artifact': provenance['artifact']})
        return original_save(tensors, filename, metadata=metadata)

    def identity(*pos, **kw):
        result = original_identity(*pos, **kw)
        if prepared is not None and prepared['base'] != result['inputs']['base']:
            raise ValueError('Prepared block belongs to another source checkpoint')
        result['experimental_conversion'] = {'block': 1, 'bits': args.bits,
                                             'targets': list(spec['targets']), 'prepared_artifact': provenance['artifact']}
        return result

    trainer.cb.N['UNETLoader'].load_unet = load
    packed.w4a8_linear_trainable = kernel
    packed.w4a8_linear_grad_input = backward
    safe_torch.save_file = save
    trainer.training_identity = identity
    name = f'qwen21_mixed_w{args.bits}_block1'
    sys.argv = [str(REPO / 'train/train_lora.py'), '--arch', 'qwen21',
                '--dataset', str(dataset), '--unet', 'qwen_image_2.1_int8_convrot.safetensors',
                '--res', '512', '--dtype', 'bf16', '--chunked-attn', '--rank', '16',
                '--seed', '3', '--steps', '2', '--state-every', '1', '--save-every', '2',
                '--val-every', '2', '--log-every', '1', '--out', str(out), '--name', name]
    try:
        if args.prepare_only:
            base_path = trainer.cb.folder_paths.get_full_path_or_raise(
                'diffusion_models', 'qwen_image_2.1_int8_convrot.safetensors')
            base = trainer.file_fingerprint(base_path)
            model = load(trainer.cb.N['UNETLoader'](), 'qwen_image_2.1_int8_convrot.safetensors', 'default')
            records = {}
            for key, (module, digest) in converted.items():
                records[key] = record_packed(module.weight, conversion_error[key], dequant_digest[key])
            out.mkdir(parents=True)
            provenance = save_prepared(out / 'packed-block.pt', dict(
                version=1, bits=args.bits, block=1, base=base, codec=codec_fingerprints(), layers=records))
            print(json.dumps({'event': 'prepared', 'path': str(out / 'packed-block.pt'),
                              'device': 'cpu', 'provenance': provenance}), flush=True)
            return
        trainer.main()
        state = torch.load(out / 'state.pt', map_location='cpu', mmap=True)
        assert state['step'] == 2
        assert state['identity']['experimental_conversion']['bits'] == args.bits
        assert state['identity']['experimental_conversion']['prepared_artifact'] == provenance['artifact']
        for key in spec['targets']:
            factors = state['lora'][f'transformer_blocks.1.{key}']
            assert all(torch.isfinite(t).all() for t in factors)
            assert torch.count_nonzero(factors[1]) > 0
        assert len(converted) == 6
        assert all(c['input_gradient'] >= 2 and c['nonzero_gradient'] >= 1 for c in calls.values())
        assert all(packed_digest(module.weight) == before for module, before in converted.values())
        records = [json.loads(line) for line in (out / 'log.jsonl').read_text().splitlines()]
        meta = next(r['meta'] for r in records if 'meta' in r)
        assert meta['step0_rel_vs_comfy'] < .01
        with safe_open(out / f'{name}_s2.safetensors', framework='pt', device='cpu') as exported:
            export_provenance = json.loads(exported.metadata()['experimental_conversion'])
            assert export_provenance['bits'] == args.bits
            assert export_provenance['prepared_artifact'] == provenance['artifact']
        result = {'event': 'acceptance', 'bits': args.bits, 'mixed_base': True,
                  'converted_layers': 6, 'optimizer_steps': 2, 'calls': calls,
                  'mapped_backup_disabled': args.no_mmap_offload,
                  'prepared_separately': True, 'prepared_artifact': provenance['artifact'],
                  'conversion_evidence': 'separate CPU preparation; serialized dequantization verified',
                  'conversion_relative_l2': conversion_error,
                  'packed_buffers_unchanged': True, 'step0_relative_l2': meta['step0_rel_vs_comfy']}
        (out / 'packed-proof.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finally:
        trainer.cb.N['UNETLoader'].load_unet = original_loader
        packed.w4a8_linear_trainable = original_kernel
        packed.w4a8_linear_grad_input = original_backward
        safe_torch.save_file = original_save
        trainer.training_identity = original_identity


if __name__ == '__main__':
    main()
