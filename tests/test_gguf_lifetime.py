"""Call-local GGUF factor preparation; CPU tests need ComfyUI on PYTHONPATH."""
import gc
import importlib
import os
import sys
import types
import weakref
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def modules():
    # Load the actual node files without registering the custom-node package.
    # CPU mode is selected before ComfyUI's device manager is imported.
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import comfy.weight_adapter

    root = Path(__file__).resolve().parents[1] / "vendor" / "ComfyUI-GGUF-fork"
    package = types.ModuleType("gguf_lifetime_test_node")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    fast = importlib.import_module(package.__name__ + ".mps")
    return fast, comfy.weight_adapter


def make_weight(adapter_module):
    down = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 100
    up = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 50
    adapter = adapter_module.LoRAAdapter(set(), (up, down, None, None, None, None))
    patches = [(1.0, adapter, 1.0, None, None)]
    weight = torch.zeros(6, 8)
    weight.tensor_shape = weight.shape
    weight.patches = [(patches, "layer.weight")]
    return weight, patches


def test_prepared_factors_expire_while_source_patches_remain(modules):
    fast, adapters = modules
    weight, patches = make_weight(adapters)
    x = torch.ones(2, 8)
    prepared = fast._lora_for_call(weight, x)
    refs = [weakref.ref(tensor) for tensor in prepared.stacked]
    assert all(ref() is not None for ref in refs)
    del prepared
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert weight.patches[0][0] is patches


def test_same_patch_list_mutation_is_visible_next_call(modules):
    fast, adapters = modules
    weight, patches = make_weight(adapters)
    x = torch.ones(2, 8)
    first = fast._lora_for_call(weight, x)
    strength, adapter, model_strength, offset, function = patches[0]
    patches[0] = (strength * 2, adapter, model_strength, offset, function)
    second = fast._lora_for_call(weight, x)
    torch.testing.assert_close(second.stacked[0], first.stacked[0])
    torch.testing.assert_close(second.stacked[1], first.stacked[1] * 2)
    weight.patches.clear()
    empty = fast._lora_for_call(weight, x)
    assert empty.stacked is None and empty.sliced == []


@pytest.mark.skipif(os.environ.get("RUN_MPS_TESTS") != "1", reason="explicit serialized MPS run required")
def test_forward_does_not_retain_factors_after_unpatch_or_offload(modules, monkeypatch):
    fast, adapters = modules
    ops = importlib.import_module("gguf_lifetime_test_node.ops")
    import gguf

    # Q8_0's real packed format: fp16 scale followed by 32 signed codes.
    packed = torch.zeros(6, 34, dtype=torch.uint8)
    packed[:, :2] = torch.tensor([0.02], dtype=torch.float16).view(torch.uint8)
    packed[:, 2:] = torch.arange(32, dtype=torch.int8).view(torch.uint8)
    layer = ops.GGMLOps.Linear(32, 6, bias=False)
    layer.weight = torch.nn.Parameter(ops.GGMLTensor(
        packed.to("mps"), tensor_type=gguf.GGMLQuantizationType.Q8_0,
        tensor_shape=torch.Size((6, 32))), requires_grad=False)
    adapter = adapters.LoRAAdapter(set(), (torch.ones(6, 4) / 10, torch.ones(4, 32) / 10,
                                           None, None, None, None))
    layer.weight.patches = [([(1.0, adapter, 1.0, None, None)], "layer.weight")]
    refs = []
    prepare = fast._lora_factors

    def observe(*args, **kwargs):
        factors = prepare(*args, **kwargs)
        refs.extend(weakref.ref(tensor) for tensor in factors.stacked)
        return factors

    monkeypatch.setattr(fast, "_lora_factors", observe)
    assert fast._kernel() is not None, "run against the comfy-kitchen GGUF branch"
    with torch.inference_mode():
        result = layer.forward_ggml_cast_weights(torch.ones(2, 32, device="mps", dtype=torch.bfloat16))
    torch.mps.synchronize()
    assert torch.isfinite(result).all()
    assert refs and all(ref() is None for ref in refs)
    layer.weight.patches.clear()
    layer.to("cpu")
    gc.collect()
    assert all(ref() is None for ref in refs)
