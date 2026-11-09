"""One real H3 quantized layer: CPU staging versus direct MPS pread lifecycle.

Run GPU mode under the project lock and an external pressure/time watchdog.
No model is constructed. --cpu-only proves real state-dict ingestion without
touching MPS; --inspect-only reads only the checkpoint header. This tests the
actual MixedPrecisionOps loader and .to() lifecycle, not ModelPatcher admission.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import struct
import sys
import time

MIB = 1024**2


def emit(event, **data):
    print(json.dumps({"event": event, **data}), flush=True)


def inspect(path, prefix):
    with path.open("rb") as stream:
        length_raw = stream.read(8)
        if len(length_raw) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", length_raw)[0]
        if not 0 < length <= 16 * MIB:
            raise ValueError("Header outside 16 MiB bound")
        header = json.loads(stream.read(length))
    metadata = header.get("__metadata__", {})
    # Preserve the actual checkpoint configuration, selecting just this layer.
    if "_quantization_metadata" not in metadata:
        raise ValueError("This bounded probe requires top-level _quantization_metadata; "
                         "use the H3 W4A8 file. Serialized v1 .comfy_quant metadata "
                         "requires a separate CPU-metadata loading contract.")
    conf = json.loads(metadata["_quantization_metadata"])["layers"][prefix]
    bits = {"int8_tensorwise": 8, "asym_w4a8_int8": 4}.get(conf["format"])
    if bits is None:
        raise ValueError("This probe accepts real INT8/W4A8 layers only")
    entries = {k: v for k, v in header.items() if k.startswith(prefix + ".")}
    allowed = {"weight", "bias", "weight_scale", "weight_s_rel", "weight_s_channel",
               "weight_codebook", "input_scale", "pre_quant_scale"}
    if any(k[len(prefix) + 1:] not in allowed for k in entries):
        raise ValueError("Selected prefix contains unsupported or nested state")
    shape = entries[prefix + ".weight"]["shape"]
    if len(shape) != 2:
        raise ValueError("Expected one packed 2D linear weight")
    n, packed_k = shape
    k = packed_k * 8 // bits
    if packed_k * 8 % bits or min(n, k) < 1 or max(n, k) > 32768:
        raise ValueError("Layer dimensions outside fixture bound")
    count = 0
    for item in entries.values():
        begin, end = item["data_offsets"]
        if not 0 <= begin <= end <= path.stat().st_size - length - 8:
            raise ValueError("Tensor offsets outside file")
        count += end - begin
    # At most 160 MiB packed + 384 MiB decode + CPU references/hash copies.
    if count > 160 * MIB or n * k * 2 > 384 * MIB:
        raise ValueError("Selected layer exceeds the bounded fixture")
    return entries, conf, n, k, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layer", default="blocks.0.mlp.fc1")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.tokens <= 16:
        raise ValueError("Use 1–16 activation rows")
    path = args.checkpoint.expanduser().resolve(strict=True)
    entries, conf, n, k, count = inspect(path, args.layer)
    emit("fixture", path=str(path), layer=args.layer, configuration=conf,
         in_features=k, out_features=n, packed_mib=count / MIB,
         dense_decode_mib=n * k * 2 / MIB, tokens=args.tokens,
         keys=list(entries), external_watchdog_required=not args.cpu_only)
    if args.inspect_only:
        return

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, os.environ.get("COMFY_DIR", str(root / "vendor" / "ComfyUI")))
    sys.argv = [sys.argv[0]] + (["--cpu"] if args.cpu_only else [])
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import psutil
    from safetensors import safe_open
    import comfy.ops as ops
    import comfy.utils as utils
    from comfy_kitchen.tensor import QuantizedTensor

    torch.set_num_threads(4)
    if not args.cpu_only and not torch.backends.mps.is_available():
        raise RuntimeError("GPU mode requires MPS")
    process = psutil.Process()
    def sync():
        if not args.cpu_only:
            torch.mps.synchronize()

    def snapshot():
        row = {"rss_mib": process.memory_info().rss / MIB,
               "available_mib": psutil.virtual_memory().available / MIB}
        if not args.cpu_only:
            row.update(mps_active_mib=torch.mps.current_allocated_memory() / MIB,
                       mps_driver_mib=torch.mps.driver_allocated_memory() / MIB)
        return row

    def release():
        gc.collect()
        if not args.cpu_only:
            torch.mps.empty_cache()
        sync()

    def digest(tensor):
        # Bit comparison also covers fp8 scale storage without fp8 arithmetic.
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        return hashlib.sha256(memoryview(raw).cast("B")).hexdigest()

    expected = {}
    with safe_open(path, framework="pt", device="cpu", backend="mmap") as handle:
        for key in entries:
            expected[key[len(args.layer) + 1:]] = digest(handle.get_tensor(key))
    # Real conversion function generates the metadata tensor on CPU, as it
    # does for this checkpoint's full _quantization_metadata representation.
    metadata = {"_quantization_metadata": json.dumps({"layers": {args.layer: conf}})}
    registry = import_module("comfy_kitchen.registry").registry
    original_get = registry.get_implementation
    counters = {}
    def counting(func_name, backend=None, kwargs=None):
        impl = original_get(func_name, backend, kwargs)
        key = f"{func_name}@{getattr(impl, '__module__', '?')}"
        counters[key] = counters.get(key, 0) + 1
        return impl

    def verify(layer, stage):
        saved = layer.state_dict()
        actual = {key: digest(saved[key]) for key in expected}
        if actual != expected:
            raise AssertionError(f"Packed weight/scales/bias changed at {stage}")
        if not isinstance(layer.weight, QuantizedTensor):
            raise AssertionError("Loader lost QuantizedTensor representation")
        emit("state", stage=stage, exact_storage_bytes=True,
             qdevice=str(layer.weight._qdata.device),
             qpointer=layer.weight._qdata.data_ptr(), **snapshot())

    reference_output = None
    modes = ["cpu_mmap"] if args.cpu_only else ["cpu_mmap_to_mps", "mps_pread"]
    registry.get_implementation = counting
    try:
        with torch.inference_mode():
            for mode in modes:
                release()
                if psutil.virtual_memory().available < 4 * 1024**3:
                    raise RuntimeError("Less than 4 GiB available; stop before loading")
                counters.clear()
                start = time.perf_counter()
                source_device = "mps" if mode == "mps_pread" else "cpu"
                with safe_open(path, framework="pt", device=source_device,
                               backend="pread" if source_device == "mps" else "mmap") as handle:
                    state = {key: handle.get_tensor(key) for key in entries}
                source_pointer = state[args.layer + ".weight"].data_ptr()
                state, _ = utils.convert_old_quants(state, metadata=metadata)
                state = {key[len(args.layer) + 1:]: value for key, value in state.items()}
                # device=None reproduces the diffusion loader's construction
                # contract; GPU mode then explicitly performs its .to() stage.
                layer = ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(
                    k, n, bias="bias" in expected, device=None)
                layer.load_state_dict(state, strict=True)
                source_storage_preserved = layer.weight._qdata.data_ptr() == source_pointer
                del state, handle
                sync()
                build_s = time.perf_counter() - start
                emit("loaded", mode=mode, build_s=build_s,
                     source_storage_preserved=source_storage_preserved,
                     initial_qdevice=str(layer.weight._qdata.device))
                verify(layer, mode + ":constructed")
                if args.cpu_only:
                    del layer
                    release()
                    emit("result", cpu_contract_pass=True, scope="No GPU or forward executed")
                    continue
                generator = torch.Generator(device="cpu").manual_seed(20261003)
                x = torch.randn(args.tokens, k, generator=generator, dtype=torch.float32)
                x = x.to(device="mps", dtype=torch.bfloat16)
                for stage in ("first_gpu", "reload_gpu"):
                    sync()
                    start = time.perf_counter()
                    layer.to("mps")
                    sync()
                    move_s = time.perf_counter() - start
                    verify(layer, mode + ":" + stage)
                    start = time.perf_counter()
                    with ops.use_quantized_matmul(layer, torch.device("mps")):
                        output = layer(x)
                    sync()
                    forward_s = time.perf_counter() - start
                    output_cpu = output.cpu()
                    del output
                    if not torch.isfinite(output_cpu).all():
                        raise AssertionError("Nonfinite output")
                    if reference_output is None:
                        reference_output = output_cpu.clone()
                    exact = torch.equal(output_cpu, reference_output)
                    torch.testing.assert_close(output_cpu, reference_output, rtol=0, atol=0)
                    emit("forward", mode=mode, stage=stage, move_s=move_s,
                         forward_s=forward_s, exact_output_parity=exact,
                         dispatch_selections=dict(counters), **snapshot())
                    del output_cpu
                    if stage == "first_gpu":
                        start = time.perf_counter()
                        layer.to("cpu")
                        sync()
                        offload_s = time.perf_counter() - start
                        release()
                        verify(layer, mode + ":offloaded_cpu")
                        emit("offload", mode=mode, seconds=offload_s)
                if not any("backends.mps" in key and value for key, value in counters.items()):
                    raise AssertionError("No MPS backend dispatch observed")
                del layer, x
                release()
                emit("released", mode=mode, **snapshot())
    finally:
        registry.get_implementation = original_get
    if not args.cpu_only:
        emit("result", exact_bytes_and_outputs=True,
             scope="One real layer; no ModelPatcher admission, full-model peak, LoRA or cold-load claim")


if __name__ == "__main__":
    main()
