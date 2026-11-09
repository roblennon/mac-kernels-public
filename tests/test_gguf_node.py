"""ComfyUI-GGUF on Apple GPUs: the Metal fast path (vendor/ComfyUI-GGUF-fork, branch
mps-gguf-kernel, mps.py) against the node's own stock path on the same layer.

The stock path is ComfyUI-GGUF's dequantize-per-call plus comfy.lora.calculate_weight for
patches; the fast path must agree with it within compute-dtype error, actually call
comfy_kitchen's gguf_linear, keep LoRAs unmerged, and hand any patch it cannot express back
to the stock path.

Env: COMFY_DIR (ComfyUI checkout, default vendor/ComfyUI), COMFY_GGUF_DIR (the fork).
"""
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.environ.get("COMFY_DIR") or os.path.join(ROOT, "vendor", "ComfyUI")
GGUF_DIR = os.environ.get("COMFY_GGUF_DIR") or os.path.join(ROOT, "vendor", "ComfyUI-GGUF-fork")
sys.path.insert(0, COMFY)

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple GPU required")

gguf = pytest.importorskip("gguf")
import comfy.weight_adapter  # noqa: E402


def load_fork():
    spec = importlib.util.spec_from_file_location(
        "comfyui_gguf_fork", os.path.join(GGUF_DIR, "__init__.py"), submodule_search_locations=[GGUF_DIR])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_gguf_fork"] = mod
    spec.loader.exec_module(mod)
    return sys.modules["comfyui_gguf_fork.ops"], sys.modules["comfyui_gguf_fork.mps"]


ops, fast = load_fork()
MPS = torch.device("mps")
QK_TS = {name: (gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[name]]) for name in ("Q4_K", "Q6_K", "Q8_0", "Q5_0", "Q3_K")}
F16 = {"Q4_K": ([0, 2], 5e-5), "Q6_K": ([208], 3e-5), "Q8_0": ([0], 3e-4), "Q5_0": ([0], 2.5e-3), "Q3_K": ([108], 2e-4)}


def make_layer(name, n=192, k=512, bias=True, seed=0):
    qk, ts = QK_TS[name]
    rng = np.random.default_rng(seed)
    nb = n * k // qk
    raw = rng.integers(0, 256, size=(nb, ts), dtype=np.uint8)
    offs, mag = F16[name]
    for off in offs:
        raw[:, off:off + 2] = np.abs(rng.standard_normal(nb) * mag + mag).astype(np.float16).view(np.uint8).reshape(nb, 2)
    lin = ops.GGMLOps.Linear(k, n)
    qt = gguf.GGMLQuantizationType[name]
    # As the loader builds it: the reader's byte rows, the logical shape alongside.
    lin.weight = torch.nn.Parameter(ops.GGMLTensor(torch.from_numpy(raw).reshape(n, k // qk * ts).to(MPS),
                                                   tensor_type=qt, tensor_shape=torch.Size((n, k))), requires_grad=False)
    if bias:
        b = torch.from_numpy(rng.standard_normal(n).astype(np.float32) * 0.1)
        lin.bias = torch.nn.Parameter(ops.GGMLTensor(b.to(MPS), tensor_type=gguf.GGMLQuantizationType.F32,
                                                     tensor_shape=b.shape), requires_grad=False)
    return lin


def lora_patch(up, down, alpha=None, strength=1.0, offset=None, dora=None):
    adapter = comfy.weight_adapter.LoRAAdapter(set(), (up, down, alpha, None, dora, None))
    return (strength, adapter, 1.0, offset, None)


def attach(lin, patches):
    lin.weight.patches = [(patches, "diffusion_model.x.weight")]


def stock(lin, x, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(fast, "linear", lambda layer, inp: None)
        return lin.forward_ggml_cast_weights(x)


class Spy:
    def __init__(self, monkeypatch):
        self.calls = []
        ck = fast._kernel()
        assert ck is not None, "comfy_kitchen with gguf_linear on MPS is required"
        real = ck.gguf_linear

        def spy(*a, **kw):
            self.calls.append(kw)
            return real(*a, **kw)

        monkeypatch.setattr(ck, "gguf_linear", spy)


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("name", list(QK_TS))
@pytest.mark.parametrize("dtype,tol", [(torch.bfloat16, 8e-3), (torch.float16, 3e-3)])
def test_fast_path_matches_stock_and_runs_the_kernel(name, dtype, tol, monkeypatch):
    lin = make_layer(name)
    x = torch.randn(2, 40, 512, device=MPS, dtype=dtype)
    spy = Spy(monkeypatch)
    got = lin.forward_ggml_cast_weights(x)
    assert len(spy.calls) == 1 and "lora_down" not in spy.calls[0]
    assert got.dtype == dtype and got.shape == (2, 40, 192)
    assert rel(got, stock(lin, x, monkeypatch)) < tol


@pytest.mark.parametrize("dtype,rank,fused", [(torch.bfloat16, 16, True), (torch.float16, 16, False), (torch.bfloat16, 200, False)])
def test_whole_layer_loras_stay_unmerged(dtype, rank, fused, monkeypatch):
    lin = make_layer("Q4_K")
    g = torch.Generator().manual_seed(1)
    p1 = lora_patch(torch.randn(192, rank, generator=g) * 0.05, torch.randn(rank, 512, generator=g) * 0.05, alpha=rank / 2)
    p2 = lora_patch(torch.randn(192, 4, generator=g) * 0.05, torch.randn(4, 512, generator=g) * 0.05, strength=0.5)
    attach(lin, [p1, p2])
    x = torch.randn(64, 512, device=MPS, dtype=dtype)
    spy = Spy(monkeypatch)
    got = lin.forward_ggml_cast_weights(x)
    assert len(spy.calls) == 1 and ("lora_down" in spy.calls[0]) == fused
    ref = stock(lin, x, monkeypatch)
    attach(lin, [])
    base = stock(lin, x, monkeypatch)
    assert rel(ref, base) > 0.05  # the LoRA is visible
    assert rel(got, ref) < (8e-3 if dtype == torch.bfloat16 else 3e-3)


def test_row_slice_lora_runs_beside_the_slice(monkeypatch):
    lin = make_layer("Q6_K", n=192)
    g = torch.Generator().manual_seed(2)
    p = lora_patch(torch.randn(64, 8, generator=g) * 0.1, torch.randn(8, 512, generator=g) * 0.1, offset=(0, 64, 64))
    attach(lin, [p])
    x = torch.randn(32, 512, device=MPS, dtype=torch.bfloat16)
    got = lin.forward_ggml_cast_weights(x)
    ref = stock(lin, x, monkeypatch)
    assert rel(got, ref) < 8e-3
    assert rel(got[:, 64:128], ref[:, 64:128]) < 8e-3 and rel(got[:, :64], ref[:, :64]) < 8e-3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sliced", [False, True])
@pytest.mark.parametrize("large_down", [False, True])
def test_finite_large_lora_contribution_matches_stock(dtype, sliced, large_down, monkeypatch):
    # The merged delta is 2 everywhere, but FP16 factors/intermediate can overflow.
    packed = torch.zeros(3, 34, dtype=torch.uint8)
    packed[:, :2] = torch.tensor([0.02], dtype=torch.float16).view(torch.uint8)
    lin = ops.GGMLOps.Linear(32, 3, bias=False)
    lin.weight = torch.nn.Parameter(ops.GGMLTensor(
        packed.to(MPS), tensor_type=gguf.GGMLQuantizationType.Q8_0,
        tensor_shape=torch.Size((3, 32))), requires_grad=False)
    up_value, down_value = (0.000025, 40000.) if large_down else (40000., 0.000025)
    up = torch.full((1 if sliced else 3, 1), up_value)
    down = torch.full((1, 32), down_value)
    originals = (up.clone(), down.clone(), packed.clone())
    attach(lin, [lora_patch(up, down, strength=2., offset=(0, 1, 1) if sliced else None)])
    x = torch.ones(2, 32, device=MPS, dtype=dtype)
    spy = Spy(monkeypatch)

    got = lin.forward_ggml_cast_weights(x)
    ref = stock(lin, x, monkeypatch)
    torch.mps.synchronize()

    assert len(spy.calls) == 1 and ("lora_down" in spy.calls[0]) == (not sliced and dtype == torch.bfloat16)
    assert got.dtype == dtype and got.shape == (2, 3)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    expected = torch.zeros_like(got)
    expected[:, 1:2] = 64. if sliced else 0.
    if not sliced:
        expected.fill_(64.)
    torch.testing.assert_close(ref, expected, rtol=0, atol=0)
    for actual, original in zip((up, down, lin.weight.as_subclass(torch.Tensor).cpu()), originals):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


def test_dora_falls_back_to_the_stock_merge(monkeypatch):
    lin = make_layer("Q8_0")
    g = torch.Generator().manual_seed(3)
    p = lora_patch(torch.randn(192, 8, generator=g) * 0.05, torch.randn(8, 512, generator=g) * 0.05,
                   dora=torch.ones(192, 1))
    attach(lin, [p])
    x = torch.randn(16, 512, device=MPS, dtype=torch.bfloat16)
    spy = Spy(monkeypatch)
    got = lin.forward_ggml_cast_weights(x)
    assert spy.calls == []  # stock path took it
    assert torch.equal(got, stock(lin, x, monkeypatch))


def test_lora_factors_follow_patch_changes(monkeypatch):
    lin = make_layer("Q4_K")
    g = torch.Generator().manual_seed(4)
    up, down = torch.randn(192, 8, generator=g) * 0.05, torch.randn(8, 512, generator=g) * 0.05
    x = torch.randn(16, 512, device=MPS, dtype=torch.bfloat16)
    attach(lin, [lora_patch(up, down)])
    a = lin.forward_ggml_cast_weights(x)
    attach(lin, [lora_patch(up, down, strength=2.0)])
    b = lin.forward_ggml_cast_weights(x)
    attach(lin, [])
    c = lin.forward_ggml_cast_weights(x)
    base = stock(lin, x, monkeypatch)
    assert rel(c, base) < 8e-3
    assert rel(b - c, 2 * (a - c)) < 0.05


def test_unquantized_weights_take_the_stock_path(monkeypatch):
    lin = make_layer("Q4_K")
    spy = Spy(monkeypatch)
    lin.forward_ggml_cast_weights(torch.randn(4, 512, device=MPS))
    assert len(spy.calls) == 1
    f16 = ops.GGMLOps.Linear(512, 64)
    w = torch.randn(64, 512, dtype=torch.float16)
    f16.weight = torch.nn.Parameter(ops.GGMLTensor(w.to(MPS), tensor_type=gguf.GGMLQuantizationType.F16,
                                                   tensor_shape=w.shape), requires_grad=False)
    assert fast.linear(f16, torch.randn(4, 512, device=MPS, dtype=torch.float16)) is None
    assert len(spy.calls) == 1
