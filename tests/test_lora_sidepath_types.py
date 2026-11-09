"""The LoRA side path beyond plain LoRAs: LoKr, LoHa, DoRA, LoCon mids, conv-shaped factors,
small diffs, strength_model and bias patches, on weight-only quantized linears (Apple GPUs).

The oracle is comfy.lora.calculate_weight itself, run in fp32 on the dequantized base weight
with the same patch list the ModelPatcher holds. With fp32 activations the quantized kernels
compute in fp32 and the side factors are kept in fp32 (layers whose original dtype is fp32), so
the side path must match the merge to fp32 rounding: these are exact-equivalence tests. bf16
cases check the fused-kernel route at the bf16 tolerance of tests/test_lora_sidepath.py.
Every case also checks that the integer weight was not touched and nothing merged, and every
unsupported shape must still merge rather than drop the patch.
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
import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.weight_adapter as wa  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402

if not hasattr(comfy.lora, "LoraSideKron"):
    pytest.skip("ComfyUI branch without the widened side path (set COMFY_DIR)", allow_module_level=True)

MPS = torch.device("mps")
N, K = 384, 512
FORMATS = {
    "int8_convrot": ("int8_tensorwise", "TensorWiseINT8Layout", dict(per_channel=True, convrot=True, convrot_groupsize=256)),
    "int8_tensorwise": ("int8_tensorwise", "TensorWiseINT8Layout", dict(per_channel=False)),
    "w6a8": ("w6a8_int8", "AsymW4A8Int8Layout", dict(bits=6)),
    "w4a8_asym": ("asym_w4a8_int8", "AsymW4A8Int8Layout", dict(bits=4, symmetric=False)),
}


class Net(torch.nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.lin = lin


def make_layer(fmt, orig_dtype=torch.float32, seed=0):
    quant_format, layout, kwargs = FORMATS[fmt]
    ops = comfy.ops.mixed_precision_ops({}, torch.float16, disabled=comfy.ops.get_disabled_quant_formats(MPS))
    lin = ops.Linear(K, N, bias=True, device="cpu")
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(N, K, generator=gen) * 0.02
    lin.weight = torch.nn.Parameter(QuantizedTensor.from_float(w.to(orig_dtype), layout, **kwargs), requires_grad=False)
    lin.bias = torch.nn.Parameter((torch.randn(N, generator=gen) * 0.1).to(torch.float16), requires_grad=False)
    lin.quant_format, lin.layout_type = quant_format, layout
    return lin


def rnd(seed, *shape, s=0.05):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * s


def lora(seed, rank=16, alpha=8.0, dora=None, mid=False, conv=False):
    up, down = rnd(seed, N, rank), rnd(seed + 1, rank, K)
    m = rnd(seed + 2, rank, rank, 1, 1, s=0.3) if mid else None
    if conv:
        up, down = up[:, :, None, None], down[:, :, None, None]
    return wa.LoRAAdapter(set(), (up, down, torch.tensor(alpha), m, dora, None))


def lokr(seed, low_rank=True, w1_low_rank=False, alpha=4.0, dora=None, a=8, b=8, r=8):
    c, d = N // a, K // b
    w1 = rnd(seed, a, b, s=0.5)
    w1_a = w1_b = w2 = w2_a = w2_b = None
    if w1_low_rank:
        w1_a, w1_b, w1 = rnd(seed + 3, a, 2, s=0.5), rnd(seed + 4, 2, b, s=0.5), None
    if low_rank:
        w2_a, w2_b = rnd(seed + 1, c, r, s=0.2), rnd(seed + 2, r, d, s=0.2)
    else:
        w2 = rnd(seed + 1, c, d, s=0.05)
    return wa.LoKrAdapter(set(), (w1, w2, torch.tensor(alpha), w1_a, w1_b, w2_a, w2_b, None, dora))


def loha(seed, r1=4, r2=4, alpha=2.0, dora=None):
    return wa.LoHaAdapter(set(), (rnd(seed, N, r1, s=0.3), rnd(seed + 1, r1, K, s=0.3), torch.tensor(alpha),
                                  rnd(seed + 2, N, r2, s=0.3), rnd(seed + 3, r2, K, s=0.3), None, None, dora))


def dora_scale(lin, out_axis, seed=7):
    """A magnitude vector near the base weight's own norms, as a trained DoRA's is."""
    w = lin.weight.dequantize().float().cpu()
    norms = w.norm(dim=1, keepdim=True) if out_axis else w.norm(dim=0, keepdim=True)
    return norms * (1 + 0.2 * torch.rand(norms.shape, generator=torch.Generator().manual_seed(seed)))


def patcher_for(lin):
    return comfy.model_patcher.ModelPatcher(Net(lin), load_device=MPS, offload_device=torch.device("cpu"))


def oracle(mp, lin, x, base=None):
    """F.linear over calculate_weight's fp32 merge of the patcher's own patch list."""
    w = (lin.weight if base is None else base).dequantize().float().cpu().clone()
    w = comfy.lora.calculate_weight(mp.patches["lin.weight"], w, "lin.weight")
    bias = lin.bias.float().cpu()
    if "lin.bias" in mp.patches:
        bias = comfy.lora.calculate_weight(mp.patches["lin.bias"], mp.backup["lin.bias"].weight.float().cpu().clone(), "lin.bias")
    return F.linear(x.float().cpu(), w, bias)


def base_out(lin, x):
    return F.linear(x.float().cpu(), lin.weight.dequantize().float().cpu(), lin.bias.float().cpu())


def check(got, ref, base, tol):
    """Whole output within ``tol`` (relative L2 and max), and the patches' effect (output minus
    the unpatched layer) within 10x that, so a small delta cannot hide inside the base."""
    got = got.float().cpu()
    assert ((got - ref).norm() / ref.norm()).item() < tol
    assert (got - ref).abs().max().item() < 10 * tol * ref.abs().max().item()
    effect, ref_effect = got - base, ref - base
    assert ref_effect.norm() > 1e-3 * ref.norm()  # the patch does something
    assert ((effect - ref_effect).norm() / ref_effect.norm()).item() < 10 * tol


def activations(dtype=torch.float32, width=K):
    return (torch.randn(2, 64, width, generator=torch.Generator().manual_seed(1)) * 0.5).to(dtype).to(MPS)


FP32_TOL = 2e-6
BF16_TOL = 8e-3


@pytest.fixture(autouse=True)
def side_path_on(monkeypatch):
    monkeypatch.delenv("COMFY_MPS_LORA_SIDE_PATH", raising=False)


def add(mp, patches):
    for p in patches:
        v, strength = p[0], p[1]
        kw = p[2] if len(p) > 2 else {}
        mp.add_patches({"lin.weight": v}, strength, **kw)


# Each case: list of (patch value or factory(lin), strength[, add_patches kwargs]).
CASES = {
    "locon_mid": lambda lin: [(lora(10, mid=True), 0.8)],
    "conv_shaped": lambda lin: [(lora(10, conv=True), 0.8)],
    "lokr_dense_w2": lambda lin: [(lokr(20, low_rank=False), 0.7)],
    "lokr_low_rank": lambda lin: [(lokr(20, w1_low_rank=True), 0.7)],
    "loha": lambda lin: [(loha(30), 0.9)],
    "diff": lambda lin: [((rnd(40, N, K, s=0.01),), 0.6)],
    "diff_typed": lambda lin: [(("diff", (rnd(40, N, K, s=0.01), {"pad_weight": False})), 0.6)],
    "strength_model": lambda lin: [(lora(10), 0.5, dict(strength_model=0.7))],
    "dora_out": lambda lin: [(lora(10, dora=dora_scale(lin, True)), 1.0)],
    "dora_out_strength": lambda lin: [(lora(10, dora=dora_scale(lin, True)), 0.6)],
    "dora_in": lambda lin: [(lora(10, dora=dora_scale(lin, False)), 1.0)],
    "dora_in_strength": lambda lin: [(lora(10, dora=dora_scale(lin, False)), 1.3)],
    "dora_lokr": lambda lin: [(lokr(20, dora=dora_scale(lin, True)), 0.8)],
    "dora_loha": lambda lin: [(loha(30, dora=dora_scale(lin, False)), 0.8)],
    # Several patches in order: each DoRA renormalizes the weight built so far, strength_model
    # rescales it, and earlier patches pick up those scales.
    "stack": lambda lin: [
        (lora(10), 0.5),
        (lokr(20), 0.7),
        (lora(50, rank=8, dora=dora_scale(lin, True)), 0.9),
        (loha(30), 0.4, dict(strength_model=0.8)),
        ((rnd(40, N, K, s=0.01),), 0.3),
        (lora(60, rank=4, dora=dora_scale(lin, False, seed=8)), 0.7),
    ],
}


@pytest.mark.parametrize("fmt", ["int8_convrot", "w6a8", "w4a8_asym"])
@pytest.mark.parametrize("case", list(CASES))
@torch.inference_mode()
def test_side_path_equals_calculate_weight_in_fp32(case, fmt):
    lin = make_layer(fmt)
    x = activations()
    base = base_out(lin, x)
    q_before = lin.weight._qdata.clone()
    mp = patcher_for(lin)
    add(mp, CASES[case](lin))
    mp.patch_model(device_to=MPS)

    assert lin.lora_side_patches, "fell back to merging"
    assert mp.backup == {}
    assert torch.equal(lin.weight._qdata.cpu(), q_before)
    got = lin(x)
    assert got.dtype == torch.float32
    check(got, oracle(mp, lin, x), base, FP32_TOL)


@pytest.mark.parametrize("case", ["loha", "lokr_low_rank", "dora_out", "strength_model", "stack"])
@torch.inference_mode()
def test_tensorwise_scale_takes_row_scales(case):
    """A scalar INT8 scale becomes a per-channel one when DoRA or strength_model scales rows."""
    lin = make_layer("int8_tensorwise")
    x = activations()
    mp = patcher_for(lin)
    add(mp, CASES[case](lin))
    mp.patch_model(device_to=MPS)
    assert lin.lora_side_patches and mp.backup == {}
    check(lin(x), oracle(mp, lin, x), base_out(lin, x), FP32_TOL)


@pytest.fixture
def kernel_calls(monkeypatch):
    """Records, per quantized kernel call made by the side path, whether it carried a LoRA."""
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


@pytest.mark.parametrize("fmt", ["int8_convrot", "w6a8"])
@torch.inference_mode()
def test_bf16_loha_fuses_lokr_rides_beside(fmt, kernel_calls):
    """A LoHa expands to low-rank factors and joins the fused stack with a plain LoRA; a LoKr in
    the same key stays beside the kernel call. One kernel call carries the stack."""
    lin = make_layer(fmt, orig_dtype=torch.bfloat16)
    x = activations(torch.bfloat16)
    mp = patcher_for(lin)
    add(mp, [(lora(10), 0.5), (loha(30), 0.9), (lokr(20), 0.7)])
    mp.patch_model(device_to=MPS)
    side = lin.lora_side_patches
    assert [type(p).__name__ for p in side] == ["LoraSideStack", "LoraSideKron"]
    assert side[0].down.shape[0] == 16 + 16
    kernel_calls.clear()
    got = lin(x)
    assert kernel_calls == [True]
    check(got, oracle(mp, lin, x), base_out(lin, x), BF16_TOL)


@pytest.mark.parametrize("fmt", ["int8_convrot", "w6a8", "w4a8_asym"])
@torch.inference_mode()
def test_bf16_dora_rides_in_the_kernel_with_its_row_scale(fmt, kernel_calls):
    """Output-axis DoRA: the LoRA in the kernel, the row scale folded into the kernel's
    per-channel scales, one call."""
    lin = make_layer(fmt, orig_dtype=torch.bfloat16)
    x = activations(torch.bfloat16)
    mp = patcher_for(lin)
    add(mp, [(lora(10, dora=dora_scale(lin, True)), 0.8)])
    mp.patch_model(device_to=MPS)
    assert lin.lora_side_patches.row_scale is not None and lin.lora_side_patches[0].fused
    kernel_calls.clear()
    got = lin(x)
    assert kernel_calls == [True]
    check(got, oracle(mp, lin, x), base_out(lin, x), BF16_TOL)


@torch.inference_mode()
def test_input_axis_dora_scales_the_input_and_keeps_the_lora_beside(kernel_calls):
    """Input-axis DoRA scales the base layer's input; the kernel's LoRA would see the scaled
    input, so the stack runs beside a plain kernel call."""
    lin = make_layer("int8_convrot", orig_dtype=torch.bfloat16)
    x = activations(torch.bfloat16)
    mp = patcher_for(lin)
    add(mp, [(lora(10, dora=dora_scale(lin, False)), 1.0)])
    mp.patch_model(device_to=MPS)
    assert lin.lora_side_patches.col_scale is not None
    kernel_calls.clear()
    got = lin(x)
    assert kernel_calls == [False]
    check(got, oracle(mp, lin, x), base_out(lin, x), BF16_TOL)


@pytest.mark.parametrize("case", ["dora_out", "dora_in", "stack"])
@pytest.mark.parametrize("residual", [False, True])
@torch.inference_mode()
def test_linear_input_act_applies_scales_and_patches(case, residual):
    """linear_input_act calls the INT8 kernel directly: row scales ride in it with the
    activation and the residual; input scales and beside patches need act(x) eagerly."""
    lin = make_layer("int8_convrot")
    mp = patcher_for(lin)
    add(mp, CASES[case](lin))
    mp.patch_model(device_to=MPS)
    xin = activations(width=2 * K)
    res = rnd(3, 2, 64, N, s=1.0).to(MPS) if residual else None
    rscale = torch.full((N,), 0.5, device=MPS) if residual else None
    got = comfy.ops.linear_input_act(lin, xin, "swiglu", residual=res, residual_scale=rscale)
    act = F.silu(xin[..., :K].float()) * xin[..., K:].float()
    ref, base = oracle(mp, lin, act), base_out(lin, act)
    if residual:
        ref, base = res.float().cpu() + 0.5 * ref, res.float().cpu() + 0.5 * base
    check(got, ref, base, FP32_TOL)


@pytest.mark.parametrize("case", ["dora_out", "dora_in", "strength_model"])
@torch.inference_mode()
def test_kernel_declining_scaled_layer_falls_back_to_the_dequantized_weight(case, monkeypatch):
    """No kernel takes the call: the scaled base runs over the dequantized weight, still exact."""
    import comfy_kitchen
    from comfy_kitchen.exceptions import NoCapableBackendError

    def declines(*a, **kw):
        raise NoCapableBackendError("int8_linear", {"mps": "declined for the test"})

    lin = make_layer("int8_convrot")
    x = activations()
    mp = patcher_for(lin)
    add(mp, CASES[case](lin))
    mp.patch_model(device_to=MPS)
    monkeypatch.setattr(comfy_kitchen, "int8_linear", declines)
    check(lin(x), oracle(mp, lin, x), base_out(lin, x), FP32_TOL)


@torch.inference_mode()
def test_bias_patch_merges_into_the_bias_beside_a_side_path_weight():
    """A bias patch merges into the float bias (exact, no requantize); the weight's LoRA stays
    beside the layer. unpatch restores both."""
    lin = make_layer("w6a8")
    lin.bias = torch.nn.Parameter(lin.bias.float(), requires_grad=False)  # an fp16 bias would round the merge
    x = activations()
    bias_before = lin.bias.clone()
    mp = patcher_for(lin)
    add(mp, [(lora(10), 0.8)])
    mp.add_patches({"lin.bias": (rnd(70, N, s=0.2),)}, 0.5)
    mp.add_patches({"lin.bias": wa.LoRAAdapter(set(), (rnd(71, N, 1), rnd(72, 1, 1), None, None, None, None))}, 1.0)
    mp.patch_model(device_to=MPS)
    assert lin.lora_side_patches and set(mp.backup) == {"lin.bias"}
    # The bias merge runs calculate_weight in lora_compute_dtype, fp16 on MPS: fp16's bound.
    check(lin(x), oracle(mp, lin, x), base_out(lin, x), 1e-3)
    mp.unpatch_model(device_to=MPS)
    assert torch.equal(lin.bias.cpu(), bias_before) and "lora_side_patches" not in lin.__dict__


@pytest.mark.parametrize("kind", ["loha_over_budget", "lokr_over_budget", "diff_over_limit", "mid_not_square",
                                  "lokr_tucker", "loha_tucker", "dora_on_slice", "dora_bad_shape", "reshape",
                                  "nested"])
@torch.inference_mode()
def test_unsupported_shapes_merge_instead(kind, monkeypatch):
    lin = make_layer("int8_convrot")
    mp = patcher_for(lin)
    if kind == "loha_over_budget":  # rank 8 x 8 = 64: 64 * (384 + 512) > 0.25 * 384 * 512
        mp.add_patches({"lin.weight": loha(30, 8, 8)}, 1.0)
    elif kind == "lokr_over_budget":  # w1 [1, 1]: the Kronecker product is a full [N, K] matrix
        mp.add_patches({"lin.weight": lokr(20, low_rank=False, a=1, b=1)}, 1.0)
    elif kind == "diff_over_limit":
        monkeypatch.setattr(comfy.lora, "SIDE_PATH_MAX_DIFF_NUMEL", N * K - 1)
        mp.add_patches({"lin.weight": (rnd(40, N, K, s=0.01),)}, 1.0)
    elif kind == "mid_not_square":
        up, down = rnd(10, N, 8), rnd(11, 16, K)
        mp.add_patches({"lin.weight": wa.LoRAAdapter(set(), (up, down, None, rnd(12, 8, 16, 1, 1), None, None))}, 1.0)
    elif kind == "lokr_tucker":
        v = lokr(20)
        w = list(v.weights)
        w[7] = rnd(5, 8, 8, 1, 1)
        mp.add_patches({"lin.weight": wa.LoKrAdapter(set(), tuple(w))}, 1.0)
    elif kind == "loha_tucker":
        w = list(loha(30).weights)
        w[5], w[6] = rnd(5, 4, 4, 1, 1), rnd(6, 4, 4, 1, 1)
        mp.add_patches({"lin.weight": wa.LoHaAdapter(set(), tuple(w))}, 1.0)
    elif kind == "dora_on_slice":
        v = wa.LoRAAdapter(set(), (rnd(10, 128, 16), rnd(11, 16, K), None, None, torch.ones(128, 1), None))
        mp.add_patches({("lin.weight", (0, 0, 128)): v}, 1.0)
    elif kind == "dora_bad_shape":
        mp.add_patches({"lin.weight": lora(10, dora=torch.ones(7, 1))}, 1.0)
    elif kind == "reshape":
        up, down = rnd(10, N, 16), rnd(11, 16, K)
        mp.add_patches({"lin.weight": wa.LoRAAdapter(set(), (up, down, None, None, None, [N, K]))}, 1.0)
    else:
        mp.add_patches({"lin.weight": [(rnd(13, N, K, s=0.01), lambda w, inplace=True: w), (1.0, lora(10), 1.0, None, None)]}, 1.0)
    assert comfy.lora.lora_side_patches(mp.patches["lin.weight"], N, K, MPS, torch.float32, weight=lin.weight) is None
    if kind in ("mid_not_square", "lokr_tucker", "loha_tucker"):
        return  # calculate_weight itself raises on these for a linear weight; the patcher merge path is unchanged
    mp.patch_model(device_to=MPS)
    assert "lora_side_patches" not in lin.__dict__
    assert "lin.weight" in mp.backup


@pytest.mark.parametrize("case", ["lokr_low_rank", "dora_out", "dora_in", "stack"])
@torch.inference_mode()
def test_resident_side_state_is_counted_and_cleared(case):
    lin = make_layer("int8_convrot")
    mp = patcher_for(lin)
    base_size = mp.model_size()
    add(mp, CASES[case](lin))
    mp.patch_model(device_to=MPS)
    side = lin.lora_side_patches
    assert side.nbytes > 0
    assert mp.model_size() == base_size + side.nbytes
    assert mp.model.model_loaded_weight_memory == mp.model_size()
    mp.unpatch_model(device_to=MPS)
    assert "lora_side_patches" not in lin.__dict__ and mp.model_size() == base_size


@torch.inference_mode()
def test_strength_model_alone_scales_the_base():
    """strength_model with a zero-strength patch: no delta, only the scaled base weight."""
    lin = make_layer("w6a8")
    x = activations()
    mp = patcher_for(lin)
    add(mp, [(lora(10), 0.0, dict(strength_model=0.6))])
    mp.patch_model(device_to=MPS)
    side = lin.lora_side_patches
    assert len(side) == 0 and side.row_scale is not None
    ref = oracle(mp, lin, x)
    got = lin(x).float().cpu()
    assert ((got - ref).norm() / ref.norm()).item() < FP32_TOL


@torch.inference_mode()
def test_partial_unload_hands_scaled_layer_to_lowvram_patch():
    """An offloaded layer drops its side state and gets the patches through LowVramPatch."""
    lin = make_layer("int8_convrot")
    x = activations()
    mp = patcher_for(lin)
    add(mp, CASES["stack"](lin))
    mp.patch_model(device_to=MPS)
    ref = oracle(mp, lin, x)
    mp.partially_unload(torch.device("cpu"), memory_to_free=1 << 40)
    assert "lora_side_patches" not in lin.__dict__ and len(lin.weight_function) == 1
    got = lin(x).float().cpu()
    # LowVramPatch merges per call and requantizes nothing, but runs calculate_weight in
    # lora_compute_dtype (fp16 on MPS), so the bound is fp16's, not fp32's.
    assert ((got - ref).norm() / ref.norm()).item() < 3e-3
    mp.partially_load(MPS, extra_memory=1 << 40)
    assert lin.lora_side_patches and lin.weight_function == []
    check(lin(x), ref, base_out(lin, x), FP32_TOL)
