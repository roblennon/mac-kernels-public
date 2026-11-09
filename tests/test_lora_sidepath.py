"""LoRA kept beside quantized linears on Apple GPUs (approach A: ComfyUI's patcher and ops).

The oracle is the float linear over the dequantized base weight with the LoRA merged in
fp32: the side path must match it within compute-dtype error, while the base weight stays
the same QuantizedTensor on the device. Whole-layer LoRAs up to comfy.lora.FUSED_LORA_MAX_RANK
with bf16 activations (the models' compute dtype here) go into comfy_kitchen's quantized kernel
(lora_down/lora_up); row/column slices, higher ranks, fp16 activations, or a kernel that
declines run up(down(x)) beside the layer. Patch kinds the side path cannot express must still
reach the layer through the regular merge.
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

COMFY = os.environ.get("COMFY_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ComfyUI")
sys.path.insert(0, COMFY)

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple GPU required")

import comfy.lora  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.weight_adapter  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402

MPS = torch.device("mps")
N, K, R = 384, 512, 16
FORMATS = {
    "int8_convrot": ("int8_tensorwise", "TensorWiseINT8Layout", dict(per_channel=True, convrot=True, convrot_groupsize=256)),
    "w6a8": ("w6a8_int8", "AsymW4A8Int8Layout", dict(bits=6)),
    "w4a8": ("asym_w4a8_int8", "AsymW4A8Int8Layout", dict(bits=4)),
}


class Net(torch.nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.lin = lin


def make_layer(fmt, seed=0, orig_dtype=torch.bfloat16):
    """A quantized layer as a model with a bf16 compute dtype loads it. Activations of another
    dtype make linear_input_act's cast hand back a dequantized weight."""
    quant_format, layout, kwargs = FORMATS[fmt]
    ops = comfy.ops.mixed_precision_ops({}, torch.float16, disabled=comfy.ops.get_disabled_quant_formats(MPS))
    lin = ops.Linear(K, N, bias=True, device="cpu")
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(N, K, generator=gen) * 0.02
    lin.weight = torch.nn.Parameter(QuantizedTensor.from_float(w.to(orig_dtype), layout, **kwargs), requires_grad=False)
    lin.bias = torch.nn.Parameter((torch.randn(N, generator=gen) * 0.1).to(torch.float16), requires_grad=False)
    lin.quant_format, lin.layout_type = quant_format, layout
    return lin


def lora(seed, rows=N, alpha=8.0, rank=R, cols=K):
    gen = torch.Generator().manual_seed(seed)
    up = (torch.randn(rows, rank, generator=gen) * 0.05).to(torch.bfloat16)
    down = (torch.randn(rank, cols, generator=gen) * 0.05).to(torch.bfloat16)
    return comfy.weight_adapter.LoRAAdapter(set(), (up, down, torch.tensor(alpha), None, None, None))


@pytest.fixture
def kernel_calls(monkeypatch):
    """Records, per quantized kernel call, whether it carried a LoRA."""
    import comfy_kitchen
    import comfy_kitchen.tensor.w4a8_int8 as w4

    calls = []

    def spy(fn):
        def wrapped(*a, **kw):
            calls.append(kw.get("lora_down") is not None)
            return fn(*a, **kw)
        return wrapped

    monkeypatch.setattr(comfy_kitchen, "int8_linear", spy(comfy_kitchen.int8_linear))
    monkeypatch.setattr(w4, "w4a8_int8_linear", spy(w4.w4a8_int8_linear))
    return calls


def lora_delta(adapter, strength):
    up, down, alpha = adapter.weights[:3]
    return strength * float(alpha) / down.shape[0] * (up.float() @ down.float())


def patcher_for(lin):
    return comfy.model_patcher.ModelPatcher(Net(lin), load_device=MPS, offload_device=torch.device("cpu"))


def reference(lin, x, delta, row_start=0, weight=None, col_start=0):
    w = (lin.weight if weight is None else weight).dequantize().float().cpu().clone()
    w[row_start:row_start + delta.shape[0], col_start:col_start + delta.shape[1]] += delta
    return F.linear(x.float().cpu(), w, lin.bias.float().cpu())


def assert_close(got, ref, base=None, side_ref=None):
    """Whole output within the weight-only kernels' error (about 2e-3 relative L2 in fp16
    without any LoRA) plus one rounding to the output dtype; with ``base`` (same layer, no
    LoRA), the side term itself must match the exact fp32 LoRA product up to the rounding of
    the two outputs."""
    eps = torch.finfo(got.dtype).eps
    got = got.float().cpu()
    scale = ref.abs().max().item()
    assert (got - ref).abs().max().item() <= (6e-3 + 2 * eps) * scale
    assert ((got - ref).norm() / ref.norm()).item() < 3e-3 + eps / 2
    if base is not None:
        side = got - base.float().cpu()
        tol = 2 * eps * got.abs().max().item() + 2e-3 * side_ref.abs().max().item()
        assert (side - side_ref).abs().max().item() <= tol


def activations(dtype=torch.bfloat16):
    return (torch.randn(2, 64, K, generator=torch.Generator().manual_seed(1)) * 0.5).to(dtype).to(MPS)


@pytest.fixture
def x():
    return activations()


@pytest.fixture(autouse=True)
def side_path_on(monkeypatch):
    monkeypatch.delenv("COMFY_MPS_LORA_SIDE_PATH", raising=False)


@pytest.mark.parametrize("fmt", list(FORMATS))
@torch.inference_mode()
def test_side_path_matches_merged_lora_and_keeps_weight_quantized(fmt, x, kernel_calls):
    lin = make_layer(fmt)
    base = lin.to(MPS)(x)
    mp = patcher_for(lin)
    adapter = lora(10)
    mp.add_patches({"lin.weight": adapter}, 0.8)
    mp.patch_model(device_to=MPS)

    assert len(lin.lora_side_patches) == 1 and lin.lora_side_patches[0].fused
    kernel_calls.clear()
    out = lin(x)
    assert kernel_calls == [True]  # one kernel call, LoRA inside it
    assert out.dtype == x.dtype
    assert isinstance(lin.weight, QuantizedTensor) and lin.weight.device.type == "mps"
    assert lin.weight_function == []
    delta = lora_delta(adapter, 0.8)
    assert_close(out, reference(lin, x, delta), base, F.linear(x.float().cpu(), delta))


@torch.inference_mode()
def test_fp16_activations_add_the_stack_beside_the_layer(kernel_calls):
    """fp16 activations: MPS's fp16 GEMMs beat the fused epilogue, so the stack runs beside."""
    x16 = activations(torch.float16)
    lin = make_layer("int8_convrot", orig_dtype=torch.float16)
    mp = patcher_for(lin)
    adapter = lora(10)
    mp.add_patches({"lin.weight": adapter}, 1.0)
    mp.patch_model(device_to=MPS)
    assert lin.lora_side_patches[0].fused
    kernel_calls.clear()
    got = lin(x16)
    assert True not in kernel_calls
    assert_close(got, reference(lin, x16, lora_delta(adapter, 1.0)))


@pytest.mark.parametrize("fmt", ["int8_convrot", "w6a8"])
@torch.inference_mode()
def test_stacked_loras_ride_in_one_kernel_call(fmt, x, kernel_calls):
    """Several whole-layer LoRAs on one key are one LoRA of the summed rank, scales in the ups."""
    lin = make_layer(fmt)
    mp = patcher_for(lin)
    a, b = lora(20), lora(21, alpha=4.0)
    mp.add_patches({"lin.weight": a}, 0.7)
    mp.add_patches({"lin.weight": b}, -0.4)
    mp.patch_model(device_to=MPS)
    stack = lin.lora_side_patches[0]
    assert len(lin.lora_side_patches) == 1 and stack.fused and tuple(stack.up.shape) == (N, 2 * R)
    kernel_calls.clear()
    got = lin(x)
    assert kernel_calls == [True]
    assert_close(got, reference(lin, x, lora_delta(a, 0.7) + lora_delta(b, -0.4)))


@torch.inference_mode()
def test_base_weight_bits_untouched_and_unpatch_restores_base(x):
    lin = make_layer("int8_convrot")
    q_before = lin.weight._qdata.clone()
    mp = patcher_for(lin)
    mp.add_patches({"lin.weight": lora(10)}, 1.0)
    mp.patch_model(device_to=MPS)
    assert torch.equal(lin.weight._qdata.cpu(), q_before)
    assert mp.backup == {}

    mp.unpatch_model(device_to=MPS)
    assert "lora_side_patches" not in lin.__dict__
    assert_close(lin(x), reference(lin, x, torch.zeros(N, K)))


@torch.inference_mode()
def test_clone_with_other_lora_replaces_side_patches(x):
    lin = make_layer("w6a8")
    base = patcher_for(lin)
    a, b = lora(10), lora(20)
    mp1 = base.clone(); mp1.add_patches({"lin.weight": a}, 1.0)
    mp2 = base.clone(); mp2.add_patches({"lin.weight": b}, 0.5)
    mp2.add_patches({"lin.weight": a}, 0.25)  # two LoRAs stacked on one key

    mp1.patch_model(device_to=MPS)
    assert_close(lin(x), reference(lin, x, lora_delta(a, 1.0)))
    mp1.unpatch_model(device_to=MPS)

    mp2.patch_model(device_to=MPS)
    assert len(lin.lora_side_patches) == 1 and lin.lora_side_patches[0].down.shape[0] == 2 * R  # stacked
    assert_close(lin(x), reference(lin, x, lora_delta(b, 0.5) + lora_delta(a, 0.25)))


@pytest.mark.parametrize("fmt", ["int8_convrot", "w6a8"])
@torch.inference_mode()
def test_qkv_row_slices_land_on_their_rows(fmt, x, kernel_calls):
    """Fused-qkv style keys: one LoRA per projection on its row slice, plus a whole-layer LoRA.
    Per-slice adds beside the layer measured cheaper than a zero-padded fused stack, so these
    stay per patch and the kernel runs without a LoRA."""
    lin = make_layer(fmt)
    mp = patcher_for(lin)
    slices = [lora(30 + i, rows=128) for i in range(3)]
    whole = lora(40)
    for i, a in enumerate(slices):
        mp.add_patches({("lin.weight", (0, 128 * i, 128)): a}, 1.0)
    mp.add_patches({"lin.weight": whole}, 0.5)
    mp.patch_model(device_to=MPS)

    assert len(lin.lora_side_patches) == 4 and not any(p.fused for p in lin.lora_side_patches)
    kernel_calls.clear()
    got = lin(x)
    assert True not in kernel_calls
    delta = torch.cat([lora_delta(a, 1.0) for a in slices]) + lora_delta(whole, 0.5)
    assert_close(got, reference(lin, x, delta))


@torch.inference_mode()
def test_column_slice_lora(x, kernel_calls):
    lin = make_layer("int8_convrot")
    mp = patcher_for(lin)
    adapter = lora(50, cols=256)
    mp.add_patches({("lin.weight", (1, 256, 256)): adapter}, 1.0)
    mp.patch_model(device_to=MPS)
    kernel_calls.clear()
    got = lin(x)
    assert True not in kernel_calls
    assert_close(got, reference(lin, x, lora_delta(adapter, 1.0), col_start=256))


@pytest.mark.parametrize("fmt", ["int8_convrot", "w6a8"])
@torch.inference_mode()
def test_rank_above_fused_limit_runs_beside_the_layer(fmt, x, kernel_calls, monkeypatch):
    """Over FUSED_LORA_MAX_RANK (summed) each LoRA stays its own patch and runs as up(down(x))
    beside a plain kernel call."""
    monkeypatch.setattr(comfy.lora.lora_side_patches, "__defaults__", (2 * R - 1,))
    lin = make_layer(fmt)
    mp = patcher_for(lin)
    a, b = lora(60), lora(61)
    mp.add_patches({"lin.weight": a}, 1.0)
    mp.add_patches({"lin.weight": b}, 1.0)
    mp.patch_model(device_to=MPS)
    assert len(lin.lora_side_patches) == 2 and not any(p.fused for p in lin.lora_side_patches)
    kernel_calls.clear()
    got = lin(x)
    assert True not in kernel_calls  # plain kernel call (INT8 reaches it through torch.ops, unseen here)
    assert_close(got, reference(lin, x, lora_delta(a, 1.0) + lora_delta(b, 1.0)))


@torch.inference_mode()
def test_kernel_declining_the_lora_falls_back_beside_the_layer(x, monkeypatch):
    """A constraint rejection from comfy_kitchen must not drop the LoRA."""
    import comfy_kitchen
    from comfy_kitchen.exceptions import NoCapableBackendError

    real = comfy_kitchen.int8_linear
    seen = []

    def declines_lora(*a, **kw):
        seen.append(kw.get("lora_down") is not None)
        if kw.get("lora_down") is not None:
            raise NoCapableBackendError("int8_linear", {"mps": "lora_down: not supported"})
        return real(*a, **kw)

    monkeypatch.setattr(comfy_kitchen, "int8_linear", declines_lora)
    lin = make_layer("int8_convrot")
    mp = patcher_for(lin)
    adapter = lora(10)
    mp.add_patches({"lin.weight": adapter}, 1.0)
    mp.patch_model(device_to=MPS)
    got = lin(x)
    assert seen == [True]  # tried the kernel once; the plain layer then ran through torch.ops
    assert_close(got, reference(lin, x, lora_delta(adapter, 1.0)))


@pytest.mark.parametrize("kind", ["dora_on_slice", "diff_over_limit", "strength_model_on_slice", "set", "function"])
@torch.inference_mode()
def test_unsupported_patches_fall_back_to_merge(kind, x, monkeypatch):
    """DoRA, small diffs and strength_model have side forms (tests/test_lora_sidepath_types.py);
    these variants do not."""
    lin = make_layer("int8_convrot")
    mp = patcher_for(lin)
    adapter = lora(10)
    if kind == "dora_on_slice":
        up, down, alpha = adapter.weights[:3]
        dora_scale = torch.ones(128, 1)
        mp.add_patches({("lin.weight", (0, 0, 128)): comfy.weight_adapter.LoRAAdapter(
            set(), (up[:128], down, alpha, None, dora_scale, None))}, 1.0)
    elif kind == "diff_over_limit":
        monkeypatch.setattr(comfy.lora, "SIDE_PATH_MAX_DIFF_NUMEL", N * K - 1, raising=False)
        mp.add_patches({"lin.weight": (lora_delta(adapter, 1.0),)}, 1.0)
    elif kind == "set":
        mp.add_patches({"lin.weight": ("set", (before_w := lin.weight.dequantize() * 1.1,))}, 1.0)
    elif kind == "function":
        mp.add_patches({("lin.weight", None, lambda d: d * 0.5): adapter}, 1.0)
    else:
        mp.add_patches({("lin.weight", (0, 0, 128)): lora(10, rows=128)}, 1.0, strength_model=0.9)
    before = lin.weight
    q_before = before._qdata.clone()
    mp.patch_model(device_to=MPS)

    assert "lora_side_patches" not in lin.__dict__
    assert "lin.weight" in mp.backup  # merged, requantized
    assert isinstance(lin.weight, QuantizedTensor) and not torch.equal(lin.weight._qdata.cpu(), q_before)
    if kind == "diff_over_limit":
        # merged INT8 weight: quantization error, not fp16 error
        ref = reference(lin, x, lora_delta(adapter, 1.0), weight=before)
        got = lin(x).float().cpu()
        assert ((got - ref).norm() / ref.norm()).item() < 2e-2
    mp.unpatch_model(device_to=MPS)
    assert torch.equal(lin.weight._qdata.cpu(), q_before)


@torch.inference_mode()
def test_env_toggle_off_merges(x, monkeypatch):
    monkeypatch.setenv("COMFY_MPS_LORA_SIDE_PATH", "0")
    lin = make_layer("int8_convrot")
    mp = patcher_for(lin)
    mp.add_patches({"lin.weight": lora(10)}, 1.0)
    mp.patch_model(device_to=MPS)
    assert "lora_side_patches" not in lin.__dict__
    assert "lin.weight" in mp.backup


@torch.inference_mode()
def test_partial_unload_hands_lora_to_lowvram_patch_once(x):
    """An offloaded layer must get its LoRA from exactly one place."""
    lin = make_layer("int8_convrot")
    mp = patcher_for(lin)
    adapter = lora(10)
    mp.add_patches({"lin.weight": adapter}, 1.0)
    mp.patch_model(device_to=MPS)
    mp.partially_unload(torch.device("cpu"), memory_to_free=1 << 40)
    assert "lora_side_patches" not in lin.__dict__
    assert len(lin.weight_function) == 1
    assert_close(lin(x), reference(lin, x, lora_delta(adapter, 1.0)))


@pytest.mark.parametrize("path", ["fused", "beside", "dequantized"])
@pytest.mark.parametrize("residual", [False, True])
@torch.inference_mode()
def test_linear_input_act_int8_path_applies_side_term(residual, path, x, kernel_calls, monkeypatch):
    """linear_input_act calls the INT8 kernel directly, skipping Linear.forward: the LoRA rides
    in that call with the activation and residual (fused), is added beside it (rank over the
    limit), or beside a dequantized-weight linear (weight dtype differs from the activations)."""
    if path == "beside":
        monkeypatch.setattr(comfy.lora.lora_side_patches, "__defaults__", (0,))
    lin = make_layer("int8_convrot", orig_dtype=torch.float16 if path == "dequantized" else torch.bfloat16)
    mp = patcher_for(lin)
    adapter = lora(10)
    mp.add_patches({"lin.weight": adapter}, 1.0)
    mp.patch_model(device_to=MPS)

    xin = torch.cat([x, x * 0.7], dim=-1)  # swiglu halves the width back to K
    res = torch.randn(2, 64, N, generator=torch.Generator().manual_seed(3)).to(x.dtype).to(MPS) if residual else None
    rscale = torch.full((N,), 0.5, dtype=x.dtype, device=MPS) if residual else None
    kernel_calls.clear()
    got = comfy.ops.linear_input_act(lin, xin, "swiglu", residual=res, residual_scale=rscale)
    assert kernel_calls == {"fused": [True], "beside": [False], "dequantized": []}[path]

    act = F.silu(xin[..., :K].float()) * xin[..., K:].float()
    ref = reference(lin, act, lora_delta(adapter, 1.0))
    if residual:
        ref = res.float().cpu() + 0.5 * ref
    assert_close(got, ref)


def merged_reference(lin, x, adapter, strength=1.0, f=lambda w: w):
    w = f(lin.weight.dequantize().float().cpu() + lora_delta(adapter, strength))
    return F.linear(x.float().cpu(), w, lin.bias.float().cpu())


def rel(got, ref):
    return ((got.float().cpu() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("fmt", ["int8_convrot", "w6a8"])
@torch.inference_mode()
def test_saving_a_side_path_model_writes_the_merged_weight(fmt, x, tmp_path):
    """SaveCheckpoint loads the model normally (side path on) and then saves it: the file must
    carry the LoRA merged into the weight, as before the side path."""
    import comfy.utils

    lin = make_layer(fmt)
    q_base = lin.weight._qdata.clone()
    mp = patcher_for(lin)
    adapter = lora(10)
    mp.add_patches({"lin.weight": adapter}, 1.0)
    mp.patch_model(device_to=MPS)
    assert lin.lora_side_patches
    expected = merged_reference(lin, x, adapter)
    base_out = F.linear(x.float().cpu(), lin.weight.dequantize().float().cpu(), lin.bias.float().cpu())

    path = str(tmp_path / "saved.safetensors")
    comfy.utils.save_torch_file(mp.model_state_dict_for_saving(), path)
    sd = comfy.utils.load_torch_file(path)
    assert torch.equal(lin.weight._qdata.cpu(), q_base)  # saving did not touch the live weight
    assert not torch.equal(sd["lin.weight"], q_base)

    fresh = make_layer(fmt, seed=99)
    fresh.weight = torch.nn.Parameter(type(lin.weight)(sd["lin.weight"], lin.weight._layout_cls,
                                                       _saved_params(lin.weight, sd)), requires_grad=False)
    fresh.bias = torch.nn.Parameter(sd["lin.bias"], requires_grad=False)
    got = fresh.to(MPS)(x)
    assert rel(got, expected) < 3e-2  # requantization error of the merge
    assert rel(got, base_out) > 5 * rel(got, expected)  # the LoRA is in there


def _saved_params(weight, sd):
    """The layout params of ``weight`` with its scales replaced by the saved ones."""
    import dataclasses
    p = weight._params
    fields = {"scale": sd["lin.weight_scale"] if "lin.weight_scale" in sd else sd["lin.weight_s_rel"]}
    if "lin.weight_s_channel" in sd:
        fields["s_channel"] = sd["lin.weight_s_channel"]
    return dataclasses.replace(p, **fields)


@torch.inference_mode()
def test_weight_wrapper_sees_the_merged_weight(x):
    """f(W) = 2W must give 2(W + delta), not 2W + delta: the key merges instead."""
    lin = make_layer("int8_convrot")
    mp = patcher_for(lin)
    adapter = lora(10)
    mp.add_patches({"lin.weight": adapter}, 1.0)
    mp.add_weight_wrapper("lin.weight", lambda w: 2 * w)
    mp.patch_model(device_to=MPS)
    assert "lora_side_patches" not in lin.__dict__ and "lin.weight" in mp.backup
    got = lin(x)
    assert rel(got, merged_reference(lin, x, adapter, f=lambda w: w)) > 0.3  # not W + delta
    ref = F.linear(x.float().cpu(), 2 * lin.weight.dequantize().float().cpu(), lin.bias.float().cpu())
    assert rel(got, ref) < 1e-2  # the stored weight is already W + delta (requantized)


@torch.inference_mode()
def test_weight_hook_on_the_key_merges(x):
    import comfy.hooks

    lin = make_layer("int8_convrot")
    mp = patcher_for(lin)
    mp.add_patches({"lin.weight": lora(10)}, 1.0)
    mp.add_hook_patches(comfy.hooks.WeightHook(), {"lin.weight": lora(11)})
    mp.patch_model(device_to=MPS)
    assert "lora_side_patches" not in lin.__dict__ and "lin.weight" in mp.backup


@torch.inference_mode()
def test_resident_factors_are_counted(x):
    lin = make_layer("int8_convrot")
    mp = patcher_for(lin)
    base_size = mp.model_size()
    mp.add_patches({"lin.weight": lora(10)}, 1.0)
    mp.patch_model(device_to=MPS)
    side = sum(p.down.nbytes + p.up.nbytes for p in lin.lora_side_patches)
    assert side > 0
    assert mp.model_size() == base_size + side
    assert mm.module_size(lin) == mm.module_size(lin, lora_side=False) + side
    assert mp.model.model_loaded_weight_memory == mp.model_size()
    mp.unpatch_model(device_to=MPS)
    assert mp.model_size() == base_size


@torch.inference_mode()
def test_partial_unload_then_reload_restores_the_side_path(x):
    lin = make_layer("int8_convrot")
    mp = patcher_for(lin)
    adapter = lora(10)
    mp.add_patches({"lin.weight": adapter}, 1.0)
    mp.patch_model(device_to=MPS)
    expected = reference(lin, x, lora_delta(adapter, 1.0))
    mp.partially_unload(torch.device("cpu"), memory_to_free=1 << 40)
    assert "lora_side_patches" not in lin.__dict__ and len(lin.weight_function) == 1
    mp.partially_load(MPS, extra_memory=1 << 40)
    assert lin.lora_side_patches and lin.weight_function == []
    assert lin.weight.device.type == "mps"
    assert_close(lin(x), expected)
