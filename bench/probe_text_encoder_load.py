"""Opt-in CPU Wan UMT5 copy/assignment experiment; production loaders are unchanged.

Run each arm in a fresh process under gpu_runner.py --monitor. Assignment transfers
storage ownership to the model: retaining and mutating source tensors is unsupported.
The source dictionary is dropped before encoding. No independence from that source
is promised. This is limited to ordinary, matching CPU fp16 UMT5 parameters.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import signal
import struct
import sys
import time


def bootstrap_cpu():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, os.environ.get('COMFY_DIR', str(root / 'vendor/ComfyUI')))
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0], '--cpu', '--use-pytorch-cross-attention']


def load_validated_state(model, state, *, assign):
    """Load one complete ordinary fp16 transformer; assignment aliases its source."""
    import torch
    expected = model.transformer.state_dict()
    actual_keys = set(state) - {'spiece_model'}
    if actual_keys != set(expected):
        raise ValueError(f'Parameter keys differ: missing={sorted(set(expected)-actual_keys)}, '
                         f'unexpected={sorted(actual_keys-set(expected))}')
    if 'spiece_model' in state:
        tokenizer = state['spiece_model']
        if type(tokenizer) is not torch.Tensor or tokenizer.dtype != torch.uint8 or tokenizer.ndim != 1 or tokenizer.device.type != 'cpu':
            raise ValueError('Expected CPU byte-vector tokenizer metadata')
    for key, target in expected.items():
        value = state[key]
        if (type(value) is not torch.Tensor or value.device.type != 'cpu'
                or target.device.type != 'cpu' or value.dtype != torch.float16
                or target.dtype != torch.float16 or value.shape != target.shape):
            raise ValueError(f'Expected matching ordinary CPU fp16 tensor: {key}')
    result = model.transformer.load_state_dict({k: state[k] for k in expected}, strict=True, assign=assign)
    loaded = model.transformer.state_dict()
    shared = sum(loaded[k].data_ptr() == state[k].data_ptr() for k in expected)
    if shared != (len(expected) if assign else 0):
        raise AssertionError('Loaded storage does not match the requested ownership contract')
    return result, shared


def make_model_class(base, *, assign, emit):
    """Use the existing umt5xxl_class override, without changing global classes."""
    class ProbeUMT5(base):
        def __init__(self, *args, **kwargs):
            emit('construct_start')
            super().__init__(*args, **kwargs)
            emit('constructed')

        def load_sd(self, state):
            emit('load_start', assign=assign)
            result, shared = load_validated_state(self, state, assign=assign)
            emit('state_applied', assign=assign, shared_storage_tensors=shared)
            return result
    return ProbeUMT5


def require_supervisor():
    path = Path(os.environ.get('MK_GPU_LOCK', '/tmp/mac-kernels-gpu.lock'))
    try:
        info = json.loads(path.read_text())
        if type(info.get('group')) is not int or info['group'] != os.getpgrp():
            raise ValueError('Process is outside the supervisor group')
        os.kill(info['owner_pid'], 0)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError('Run this memory-heavy CPU experiment under gpu_runner.py --monitor') from exc


def inspect_checkpoint(path):
    size = path.stat().st_size
    if path.suffix != '.safetensors' or size > 12 * 2**30:
        raise ValueError('Expected a UMT5 safetensors checkpoint no larger than 12 GiB')
    with path.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError('Missing safetensors header')
        header_size = struct.unpack('<Q', raw)[0]
        if not 0 < header_size <= 16 * 2**20:
            raise ValueError('Safetensors header outside probe bound')
        header = json.loads(stream.read(header_size))
    for key, value in header.items():
        if key == '__metadata__':
            continue
        if value['dtype'] != ('U8' if key == 'spiece_model' else 'F16'):
            raise ValueError(f'Unsupported tensor dtype: {key}')
    if header.get('shared.weight', {}).get('shape') != [256384, 4096]:
        raise ValueError('Expected the fp16 UMT5 XXL embedding shape')
    return size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--assign', action='store_true')
    parser.add_argument('--run', action='store_true', help='Actually load and encode; otherwise inspect only')
    parser.add_argument('--seconds', type=int, default=420)
    parser.add_argument('--prompt', default='A street musician plays guitar in a retro subway station.')
    parser.add_argument('--output', type=Path, help='Optional new file containing conditioning and experiment metadata')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 540 or not 1 <= len(args.prompt) <= 2048:
        parser.error('Use 1–540 seconds and a nonempty prompt of at most 2048 characters')
    path = args.checkpoint.resolve(strict=True)
    size = inspect_checkpoint(path)
    print(json.dumps({'event': 'plan', 'checkpoint': str(path), 'file_bytes': size,
                      'assign': args.assign, 'device': 'cpu', 'run': args.run,
                      'ownership': 'source storage is shared' if args.assign else 'independent copied storage'}), flush=True)
    if not args.run:
        return
    require_supervisor()
    if args.output and args.output.exists():
        parser.error('Output already exists; use a new filename')
    def timeout(_signum, _frame):
        raise TimeoutError('Encoder experiment deadline reached; outer monitor owns hard cleanup')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    bootstrap_cpu()
    import torch
    import psutil
    import comfy.sd
    import comfy.utils
    import comfy.model_management as mm
    from comfy.text_encoders.wan import UMT5XXlModel
    torch.set_num_threads(4)
    process = psutil.Process()
    start = time.perf_counter()
    def emit(event, **values):
        print(json.dumps(dict(event=event, elapsed_s=time.perf_counter()-start,
            rss_gib=process.memory_info().rss/2**30,
            available_gib=psutil.virtual_memory().available/2**30, **values)), flush=True)
    # A heuristic admission floor, not a physical-memory upper bound. The external
    # monitor must remain active; mapped source and activations add to residency.
    required = size * (1.1 if args.assign else 2.0) + 3 * 2**30
    if psutil.virtual_memory().available < required:
        raise RuntimeError(f'Need {required/2**30:.2f} GiB available for this isolated arm')
    emit('read_start', required_gib=required/2**30)
    with torch.inference_mode():
        state = comfy.utils.load_torch_file(str(path), safe_load=True)
        emit('state_read', tensor_bytes=sum(t.nbytes for t in state.values()))
        model_class = make_model_class(UMT5XXlModel, assign=args.assign, emit=emit)
        clip = comfy.sd.load_text_encoder_state_dicts([state], clip_type=comfy.sd.CLIPType.WAN,
            model_options={'umt5xxl_class': model_class, 'dtype': torch.float16,
                           'load_device': torch.device('cpu'), 'offload_device': torch.device('cpu'),
                           'initial_device': torch.device('cpu')})
        del state
        gc.collect()
        emit('source_released', model_bytes=clip.patcher.model_size())
        tokens = clip.tokenize(args.prompt)
        token_count = sum(len(chunk) for chunk in tokens['umt5xxl'])
        if token_count > 512:
            raise ValueError('Prompt exceeds this experiment’s 512-token bound')
        emit('encode_start', tokens=token_count)
        conditioning = clip.encode_from_tokens(tokens, return_dict=True)
        cond = conditioning.get('cond')
        if not isinstance(cond, torch.Tensor) or cond.numel() == 0:
            raise AssertionError('Encoder did not produce conditioning')
        tensors = [v for v in conditioning.values() if isinstance(v, torch.Tensor)]
        if not all(t.device.type == 'cpu' and torch.isfinite(t).all().item() for t in tensors):
            raise AssertionError('Non-finite or non-CPU conditioning')
        emit('encoded', finite=True, conditioning_shape=list(cond.shape),
             conditioning_dtype=str(cond.dtype), tensor_bytes=sum(t.nbytes for t in tensors))
        if args.output:
            with args.output.open('xb') as stream:
                torch.save({'checkpoint': str(path), 'assign': args.assign, 'prompt': args.prompt,
                            'conditioning': conditioning}, stream)
            emit('saved', output=str(args.output.resolve()))
        mm.unload_all_models()
        del clip, conditioning, cond, tensors
        gc.collect()
        emit('released')
    signal.alarm(0)


if __name__ == '__main__':
    main()
