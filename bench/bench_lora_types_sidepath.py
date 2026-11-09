"""Per-layer cost of each adapter type on the ComfyUI LoRA side path, against merging it.

One quantized linear (default M=4608 tokens, K=N=6144, bf16 activations), loaded through
ModelPatcher as a model would load it. For each adapter type and rank: the side path's extra
milliseconds per call over the unpatched layer (median of ROUNDS interleaved rounds), the
one-time patch_model cost, and the output error against calculate_weight's fp32 merge on the
first 256 tokens; then the same patches merged (COMFY_MPS_LORA_SIDE_PATH=0): load cost and error.

    COMFY_DIR=vendor/ComfyUI-loratypes python bench/bench_lora_types_sidepath.py
    env: FORMATS=int8_convrot,w6a8  RANKS=16,32,64  ROUNDS=15  SHAPE=4608,6144,6144  MERGE=1
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("COMFY_DIR") or os.path.join(ROOT, "vendor", "ComfyUI"))

import comfy.lora  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.weight_adapter as wa  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402

MPS = torch.device("mps")
M, K, N = (int(v) for v in os.environ.get("SHAPE", "4608,6144,6144").split(","))
ROUNDS = int(os.environ.get("ROUNDS", "15"))
RANKS = [int(r) for r in os.environ.get("RANKS", "16,32,64").split(",")]
FORMATS = os.environ.get("FORMATS", "int8_convrot,w6a8").split(",")
MERGE = os.environ.get("MERGE", "1") == "1"
LAYOUTS = {
    "int8_convrot": ("int8_tensorwise", "TensorWiseINT8Layout", dict(per_channel=True, convrot=True, convrot_groupsize=256)),
    "w6a8": ("w6a8_int8", "AsymW4A8Int8Layout", dict(bits=6)),
}


class Net(torch.nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.lin = lin


def make_layer(fmt, weight=None, bias=None):
    quant_format, layout, kwargs = LAYOUTS[fmt]
    ops = comfy.ops.mixed_precision_ops({}, torch.bfloat16, disabled=comfy.ops.get_disabled_quant_formats(MPS))
    lin = ops.Linear(K, N, bias=True, device="cpu")
    if weight is None:
        w = (torch.randn(N, K, generator=torch.Generator().manual_seed(0)) * 0.02).to(torch.bfloat16)
        weight = QuantizedTensor.from_float(w, layout, **kwargs).to(MPS)
        bias = (torch.randn(N, generator=torch.Generator().manual_seed(1)) * 0.1).to(torch.bfloat16).to(MPS)
    lin.weight = torch.nn.Parameter(weight, requires_grad=False)
    lin.bias = torch.nn.Parameter(bias, requires_grad=False)
    lin.quant_format, lin.layout_type = quant_format, layout
    return lin


def rnd(seed, *shape, s=0.02):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * s


def lora(r, dora=None, mid=False):
    m = rnd(3, r, r, 1, 1, s=0.3) if mid else None
    return wa.LoRAAdapter(set(), (rnd(1, N, r), rnd(2, r, K), torch.tensor(float(r)), m, dora, None))


def lokr(a, r=None, dora=None):
    """w1 [a, a]; w2 [N/a, K/a], dense or rank r."""
    c, d = N // a, K // a
    if r is None:
        return wa.LoKrAdapter(set(), (rnd(4, a, a, s=0.3), rnd(5, c, d, s=0.002), None, None, None, None, None, None, dora))
    return wa.LoKrAdapter(set(), (rnd(4, a, a, s=0.3), None, torch.tensor(float(r)), None, None,
                                  rnd(5, c, r, s=0.05), rnd(6, r, d, s=0.05), None, dora))


def loha(r1, r2):
    return wa.LoHaAdapter(set(), (rnd(7, N, r1, s=0.15), rnd(8, r1, K, s=0.15), torch.tensor(float(r1)),
                                  rnd(9, N, r2, s=0.15), rnd(10, r2, K, s=0.15), None, None, None))


def dora_scale(lin, out_axis):
    w = lin.weight.dequantize().float()
    n = w.norm(dim=1, keepdim=True) if out_axis else w.norm(dim=0, keepdim=True)
    return (n * 1.05).cpu()


def cases(lin):
    out = []
    for r in RANKS:
        out += [
            (f"lora r{r}", [(lora(r), 1.0, {})]),
            (f"locon-mid r{r}", [(lora(r, mid=True), 1.0, {})]),
            (f"lokr 8x8 (x) rank-{r} w2", [(lokr(8, r), 1.0, {})]),
            (f"dora-out lora r{r}", [(lora(r, dora=dora_scale(lin, True)), 1.0, {})]),
            (f"dora-in lora r{r}", [(lora(r, dora=dora_scale(lin, False)), 1.0, {})]),
            (f"lora r{r} + strength_model 0.8", [(lora(r), 1.0, {"strength_model": 0.8})]),
        ]
    out += [
        ("loha 4x4 (rank 16)", [(loha(4, 4), 1.0, {})]),
        ("loha 4x8 (rank 32)", [(loha(4, 8), 1.0, {})]),
        ("loha 8x8 (rank 64)", [(loha(8, 8), 1.0, {})]),
        ("loha 16x16 (rank 256)", [(loha(16, 16), 1.0, {})]),
        ("lokr 16x16 (x) dense w2", [(lokr(16), 1.0, {})]),
        ("lokr 4x4 (x) dense w2", [(lokr(4), 1.0, {})]),
    ]
    return out


def interleaved(fns):
    for f in fns.values():
        f()
        f()
    t = {k: [] for k in fns}
    for _ in range(ROUNDS):
        for k, f in fns.items():
            torch.mps.synchronize()
            t0 = time.perf_counter()
            f()
            torch.mps.synchronize()
            t[k].append(time.perf_counter() - t0)
    return {k: sorted(v)[len(v) // 2] * 1e3 for k, v in t.items()}


def load(lin, patches, side):
    os.environ["COMFY_MPS_LORA_SIDE_PATH"] = "1" if side else "0"
    mp = comfy.model_patcher.ModelPatcher(Net(lin), load_device=MPS, offload_device=torch.device("cpu"))
    for v, strength, kw in patches:
        mp.add_patches({"lin.weight": v}, strength, **kw)
    torch.mps.synchronize()
    t0 = time.perf_counter()
    mp.patch_model(device_to=MPS)
    torch.mps.synchronize()
    return mp, time.perf_counter() - t0


def rel_err(lin, mp, xs, base_weight):
    w = comfy.lora.calculate_weight(mp.patches["lin.weight"], base_weight.dequantize().float().cpu().clone(), "lin.weight")
    ref = F.linear(xs.float().cpu(), w, lin.bias.float().cpu())
    got = lin(xs).float().cpu()
    return ((got - ref).norm() / ref.norm()).item()


@torch.inference_mode()
def main():
    print(f"torch {torch.__version__}, M={M} K={K} N={N}, bf16 activations, median of {ROUNDS} rounds")
    print("| format | adapter | path | base ms | extra ms | extra % | load s | rel err | merge load s | merge rel err |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    x = (torch.randn(M, K, generator=torch.Generator().manual_seed(11)) * 0.5).to(torch.bfloat16).to(MPS)
    xs = x[:256]
    for fmt in FORMATS:
        base_lin = make_layer(fmt)
        for name, patches in cases(base_lin):
            lin = make_layer(fmt, base_lin.weight.data, base_lin.bias.data)
            mp, load_s = load(lin, patches, side=True)
            side = lin.__dict__.get("lora_side_patches")
            if side is None:
                path = "merged"
            else:
                kinds = "+".join(type(p).__name__.replace("LoraSide", "") for p in side)
                scales = "".join(s for s, t in (("R", side.row_scale), ("C", side.col_scale)) if t is not None)
                path = kinds + (f" [{scales}]" if scales else "")
            ms = interleaved({"base": lambda: base_lin(x), "patched": lambda: lin(x)})
            err = rel_err(lin, mp, xs, base_lin.weight)
            mp.unpatch_model(device_to=MPS)
            merge_s = merge_err = float("nan")
            if MERGE:
                lin2 = make_layer(fmt, base_lin.weight.data, base_lin.bias.data)
                mp2, merge_s = load(lin2, patches, side=False)
                merge_err = rel_err(lin2, mp2, xs, base_lin.weight)
                mp2.unpatch_model(device_to=MPS)
                del lin2, mp2
            extra = ms["patched"] - ms["base"]
            print(f"| {fmt} | {name} | {path} | {ms['base']:.1f} | {extra:+.1f} | {100 * extra / ms['base']:+.1f} | "
                  f"{load_s:.2f} | {err:.1e} | {merge_s:.2f} | {merge_err:.1e} |", flush=True)
            del lin, mp
            torch.mps.empty_cache()
    os.environ.pop("COMFY_MPS_LORA_SIDE_PATH", None)


if __name__ == "__main__":
    main()
