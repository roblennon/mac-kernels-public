"""GGUF benchmark reachability uses real node/adapters; only GPU execution is faked."""
import gc
import importlib
import sys
import types
import weakref
from pathlib import Path

import pytest
import torch

from bench import gguf_node


@pytest.fixture(scope="module")
def modules():
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import comfy.weight_adapter

    root = Path(__file__).resolve().parents[1] / "vendor" / "ComfyUI-GGUF-fork"
    package = types.ModuleType("gguf_audit_test_node")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    ops = importlib.import_module(package.__name__ + ".ops")
    fast = importlib.import_module(package.__name__ + ".mps")
    return ops, fast, comfy.weight_adapter


def make_layer(modules, rank=4, sliced=False, mixed=False):
    import gguf
    ops, _fast, adapters = modules
    packed = torch.zeros(6, 34, dtype=torch.uint8)
    packed[:, :2] = torch.tensor([0.02], dtype=torch.float16).view(torch.uint8)
    packed[:, 2:] = torch.arange(32, dtype=torch.int8).view(torch.uint8)
    layer = ops.GGMLOps.Linear(32, 6, bias=False)
    layer.weight = torch.nn.Parameter(ops.GGMLTensor(
        packed, tensor_type=gguf.GGMLQuantizationType.Q8_0,
        tensor_shape=torch.Size((6, 32))), requires_grad=False)
    patches = []
    for use_slice in ([False, True] if mixed else [sliced]):
        rows = 2 if use_slice else 6
        adapter = adapters.LoRAAdapter(set(), (torch.ones(rows, rank) / 10,
                                               torch.ones(rank, 32) / 10,
                                               None, None, None, None))
        patches.append((1.0, adapter, 1.0, (0, 2, 2) if use_slice else None, None))
    layer.weight.patches = [(patches, "layer.weight")]
    model = torch.nn.Module()
    model.add_module("layer", layer)
    return layer, types.SimpleNamespace(model=model, patches={}), patches


@pytest.fixture
def install(modules, monkeypatch):
    ops, fast, _adapters = modules
    # Register restoration before the benchmark installs its wrappers.
    for owner, name in ((fast, "linear"), (fast, "_lora_for_call"),
                        (ops.GGMLOps.Linear, "forward_ggml_cast_weights")):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    refs = []
    prepare = fast._lora_for_call

    def observe_factors(*args, **kwargs):
        factors = prepare(*args, **kwargs)
        if factors is not None:
            if factors.stacked:
                refs.extend(weakref.ref(tensor) for tensor in factors.stacked)
            for down, up, _start in factors.sliced:
                refs.extend((weakref.ref(down), weakref.ref(up)))
        return factors

    monkeypatch.setattr(fast, "_lora_for_call", observe_factors)

    def setup(gpu=True, fail=False):
        if gpu:
            def gpu_boundary(layer, x):
                # Exercise the real call-local adapter preparation without touching MPS.
                factors = fast._lora_for_call(layer.weight, x)
                if factors is None:
                    return None
                if fail:
                    raise RuntimeError("GPU execution failed")
                return torch.zeros(*x.shape[:-1], layer.out_features, dtype=x.dtype)
            monkeypatch.setattr(fast, "linear", gpu_boundary)
        gguf_node._install_lora_audit(ops, fast)
        return refs
    return setup


@pytest.mark.parametrize("rank,dtype,sliced,mixed,expected", [
    (4, torch.bfloat16, False, False, (1, 0, 0)),
    (129, torch.bfloat16, False, False, (0, 1, 0)),
    (4, torch.float32, False, False, (0, 1, 0)),
    (4, torch.bfloat16, True, False, (0, 0, 1)),
    (4, torch.bfloat16, False, True, (1, 0, 1)),
])
def test_completed_kernel_routes_without_retaining_factors(modules, install, rank, dtype,
                                                          sliced, mixed, expected):
    layer, patcher, _patches = make_layer(modules, rank, sliced, mixed)
    refs = install()
    assert gguf_node.lora_audit(patcher)["not_reached"] == 1
    with torch.inference_mode():
        layer.forward_ggml_cast_weights(torch.ones(2, 32, dtype=dtype))
    audit = gguf_node.lora_audit(patcher)
    assert audit["not_reached"] == audit["stock_merge_per_call"] == 0
    assert tuple(audit[k] for k in ("kernel_fused", "kernel_beside", "kernel_slices")) == expected
    assert not hasattr(layer, "_mps_lora_cache")
    gc.collect()
    # Float32 down factors can alias source patches on CPU; only newly prepared
    # tensors must expire. Stacked tensors always copy, sliced float32 is not used.
    assert refs and all(ref() is None for ref in refs)


def test_actual_cpu_fallback_reports_completed_stock_merge(modules, install):
    layer, patcher, _patches = make_layer(modules)
    install(gpu=False)
    with torch.inference_mode():
        result = layer.forward_ggml_cast_weights(torch.ones(2, 32))
    assert torch.isfinite(result).all()
    audit = gguf_node.lora_audit(patcher)
    assert audit["stock_merge_per_call"] == 1
    assert audit["not_reached"] == audit["kernel_fused"] == 0


def test_unsupported_patch_reports_stock_even_when_kernel_is_available(modules, install):
    layer, patcher, patches = make_layer(modules)
    patches[:] = [(1.0, ("diff", (torch.ones(6, 32) / 100,)), 1.0, None, None)]
    install()
    result = layer.forward_ggml_cast_weights(torch.ones(2, 32))
    assert torch.isfinite(result).all()
    audit = gguf_node.lora_audit(patcher)
    assert audit["stock_merge_per_call"] == 1
    assert audit["not_reached"] == audit["kernel_fused"] == 0


def test_patch_replacement_requires_another_completed_forward(modules, install):
    layer, patcher, patches = make_layer(modules)
    install()
    x = torch.ones(2, 32, dtype=torch.bfloat16)
    layer.forward_ggml_cast_weights(x)
    strength, adapter, model_strength, offset, function = patches[0]
    patches[0] = (strength * 2, adapter, model_strength, offset, function)
    assert gguf_node.lora_audit(patcher)["not_reached"] == 1
    layer.forward_ggml_cast_weights(x)
    assert gguf_node.lora_audit(patcher)["kernel_fused"] == 1
    layer.weight.patches.clear()
    assert gguf_node.lora_audit(patcher)["quantized_layers_with_lora"] == 0


def test_old_patch_objects_expire_and_cannot_match_new_adapters(modules, install):
    layer, patcher, patches = make_layer(modules)
    install()
    layer.forward_ggml_cast_weights(torch.ones(2, 32, dtype=torch.bfloat16))
    old_adapter = weakref.ref(patches[0][1])
    patches.clear()
    gc.collect()
    assert old_adapter() is None
    _replacement, _patcher, new_patches = make_layer(modules)
    patches.extend(new_patches)
    assert gguf_node.lora_audit(patcher)["not_reached"] == 1


def test_failed_forward_does_not_count_as_reached(modules, install):
    layer, patcher, _patches = make_layer(modules)
    refs = install(fail=True)
    with pytest.raises(RuntimeError, match="GPU execution failed"):
        layer.forward_ggml_cast_weights(torch.ones(2, 32, dtype=torch.bfloat16))
    assert gguf_node.lora_audit(patcher)["not_reached"] == 1
    gc.collect()
    assert refs and all(ref() is None for ref in refs)
