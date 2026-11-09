"""Bounded real diffusion-model loading through CPU staging or explicit MPS pread.

Use gpu_runner.py --monitor. This does not sample, load an encoder, or attach a
LoRA. Records actual ModelPatcher registration/offload/reload accounting. The
CPU arm and direct arm run in separate processes for comparable allocator state.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import struct
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--direct', action='store_true')
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()
    path = args.checkpoint.resolve(strict=True)
    size = path.stat().st_size
    if not path.name.endswith('.safetensors') or size > 6 * 2**30:
        parser.error('Use a safetensors checkpoint no larger than 6 GiB')
    with path.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < header_size <= 16 * 2**20:
            parser.error('Safetensors header outside the probe bound')
        header = json.loads(stream.read(header_size))
    metadata_keys = [k for k in header if k.endswith('.comfy_quant')]
    if not metadata_keys and '_quantization_metadata' not in header.get('__metadata__', {}):
        parser.error('This probe requires a quantized checkpoint, such as the selected Wan INT8 file')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, os.environ.get('COMFY_DIR', str(root / 'vendor/ComfyUI')))
    sys.argv = [sys.argv[0], '--use-pytorch-cross-attention']
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import psutil
    import comfy.model_management as mm
    import comfy.sd
    import comfy.utils
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(0.5)
    dev = torch.device('mps')
    process = psutil.Process()
    start = time.perf_counter()

    def emit(event, **values):
        torch.mps.synchronize()
        print(json.dumps(dict(event=event, elapsed_s=time.perf_counter()-start,
            driver_gib=torch.mps.driver_allocated_memory()/2**30,
            active_gib=torch.mps.current_allocated_memory()/2**30,
            rss_gib=process.memory_info().rss/2**30,
            available_gib=psutil.virtual_memory().available/2**30, **values)), flush=True)

    # Direct buffers bypass torch's active allocator. Admission uses file bytes
    # and available system RAM; the external monitor observes pressure throughout.
    required = size * 1.2 + 4 * 2**30
    mm.free_memory(required, dev)
    if psutil.virtual_memory().available < required:
        raise RuntimeError(f'Need {required/2**30:.2f} GiB available before this isolated load')
    emit('admitted', direct=args.direct, file_gib=size/2**30, required_gib=required/2**30)
    with torch.inference_mode():
        t0 = time.perf_counter()
        state, metadata = comfy.utils.load_torch_file(str(path), device=dev if args.direct else None, return_metadata=True)
        emit('state_loaded', seconds=time.perf_counter()-t0,
             tensor_bytes=sum(t.nbytes for t in state.values()),
             mps_tensor_bytes=sum(t.nbytes for t in state.values() if t.device.type=='mps'),
             metadata_tensor_count=len(metadata_keys),
             metadata_cpu=all(state[k].device.type=='cpu' for k in metadata_keys) if metadata_keys else None)
        t0 = time.perf_counter()
        model = comfy.sd.load_diffusion_model_state_dict(state, metadata=metadata)
        if model is None:
            raise RuntimeError('Unsupported diffusion model')
        del state
        gc.collect()
        emit('model_constructed', seconds=time.perf_counter()-t0, model_bytes=model.model_size(), loaded_bytes=model.loaded_size())
        t0 = time.perf_counter()
        mm.load_models_gpu([model], force_full_load=True)
        emit('model_registered', seconds=time.perf_counter()-t0, model_bytes=model.model_size(), loaded_bytes=model.loaded_size(),
             device=str(model.current_loaded_device()))
        if model.loaded_size() != model.model_size():
            raise AssertionError('Fully loaded model accounting mismatch')
        if args.reload:
            mm.unload_all_models()
            gc.collect()
            mm.soft_empty_cache()
            emit('offloaded', loaded_bytes=model.loaded_size(), device=str(model.current_loaded_device()))
            t0 = time.perf_counter()
            mm.load_models_gpu([model], force_full_load=True)
            emit('reloaded', seconds=time.perf_counter()-t0, model_bytes=model.model_size(), loaded_bytes=model.loaded_size())
        mm.unload_all_models()
        del model
        gc.collect()
        mm.soft_empty_cache()
        emit('released')


if __name__ == '__main__':
    main()
