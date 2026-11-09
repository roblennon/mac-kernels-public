"""Explicit GGUF loader precision stays on stock arithmetic; CPU-only proof."""
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def modules():
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import comfy.weight_adapter
    root = Path(__file__).resolve().parents[1] / "vendor" / "ComfyUI-GGUF-fork"
    package = types.ModuleType("gguf_precision_test_node")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    return (importlib.import_module(package.__name__ + ".ops"),
            importlib.import_module(package.__name__ + ".mps"), comfy.weight_adapter)


def layer_with_large_factors(modules):
    import gguf
    ops, _fast, adapters = modules
    packed = torch.zeros(1, 34, dtype=torch.uint8)
    packed[:, :2] = torch.tensor([0.02], dtype=torch.float16).view(torch.uint8)
    layer = ops.GGMLOps.Linear(32, 1, bias=False)
    layer.weight = torch.nn.Parameter(ops.GGMLTensor(
        packed, tensor_type=gguf.GGMLQuantizationType.Q8_0,
        tensor_shape=torch.Size((1, 32))), requires_grad=False)
    adapter = adapters.LoRAAdapter(set(), (torch.full((1, 1), 40000.),
        torch.full((1, 32), 0.000025), None, None, None, None))
    layer.weight.patches = [([(2.0, adapter, 1.0, None, None)], "layer.weight")]
    return layer


class AdvertisedMPSInput(torch.Tensor):
    """Mock only the device boundary; backend lookup is intercepted before compute."""
    @property
    def device(self):
        return torch.device("mps")


@pytest.mark.parametrize("setting", ["dequant_dtype", "patch_dtype"])
@pytest.mark.parametrize("dtype", ["target", torch.float16, torch.bfloat16, torch.float32])
def test_explicit_precision_does_not_enter_accelerated_backend(modules, monkeypatch, setting, dtype):
    _ops, fast, _adapters = modules
    layer = layer_with_large_factors(modules)
    setattr(layer, setting, dtype)
    x = torch.ones(1, 32, dtype=torch.float16).as_subclass(AdvertisedMPSInput)
    def forbidden_backend():
        pytest.fail("explicit precision selected the normalized kernel")
    monkeypatch.setattr(fast, "_kernel", forbidden_backend)
    assert fast.linear(layer, x) is None


@pytest.mark.parametrize("unused_patch_setting", [None, torch.float32])
def test_default_precision_still_considers_kernel_without_patches(modules, monkeypatch, unused_patch_setting):
    _ops, fast, _adapters = modules
    layer = layer_with_large_factors(modules)
    layer.weight.patches.clear()
    layer.patch_dtype = unused_patch_setting
    called = []
    monkeypatch.setattr(fast, "_kernel", lambda: called.append(True))
    x = torch.ones(1, 32, dtype=torch.float16).as_subclass(AdvertisedMPSInput)
    assert fast.linear(layer, x) is None  # Deliberately unavailable GPU boundary.
    assert called == [True]


def test_requested_fp32_patch_is_finite_when_fp16_side_factors_overflow(modules):
    _ops, fast, _adapters = modules
    layer = layer_with_large_factors(modules)
    layer.patch_dtype = torch.float32
    x = torch.ones(1, 32, dtype=torch.float16)
    factors = fast._lora_for_call(layer.weight, x)
    # Retain the old FP16-factor policy as the disconfirming control.
    unsafe = torch.nn.functional.linear(torch.nn.functional.linear(x, factors.stacked[0].half()),
                                        factors.stacked[1].half())
    assert not torch.isfinite(unsafe).all()
    # Real stock GGUF dequantization and real LoRAAdapter arithmetic honor FP32.
    result = layer.forward_ggml_cast_weights(x)
    torch.testing.assert_close(result, torch.tensor([[64.0]], dtype=torch.float16))


@pytest.mark.parametrize("large_down", [False, True])
def test_default_lora_preparation_preserves_a_finite_fp16_contribution(modules, large_down):
    _ops, fast, _adapters = modules
    layer = layer_with_large_factors(modules)
    adapter = layer.weight.patches[0][0][0][1]
    up, down, *_ = adapter.weights
    if large_down:
        up.fill_(0.000025)
        down.fill_(40000.)
    originals = (up.clone(), down.clone(), layer.weight.as_subclass(torch.Tensor).clone())
    x = torch.ones(1, 32, dtype=torch.float16)

    factors = fast._lora_for_call(layer.weight, x)
    prepared_down, prepared_up = factors.stacked
    contribution = torch.nn.functional.linear(
        torch.nn.functional.linear(x.to(prepared_down.dtype), prepared_down), prepared_up)
    reference = layer.forward_ggml_cast_weights(x)

    assert torch.isfinite(contribution).all()
    torch.testing.assert_close(contribution.to(x.dtype), reference)
    torch.testing.assert_close(reference, torch.tensor([[64.0]], dtype=x.dtype))
    for actual, original in zip((up, down, layer.weight.as_subclass(torch.Tensor)), originals):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
