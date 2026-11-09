"""Load a GGUF diffusion model through ComfyUI-GGUF for the headless runners.

The node package comes from COMFY_GGUF_DIR (default vendor/ComfyUI-GGUF-fork, branch
mps-gguf-kernel). MODE=stock in the runners sets COMFY_GGUF_MPS_KERNEL=0 before this loads,
which is ComfyUI-GGUF's own dequantize-per-call path. ``counters()`` reports how many
quantized Linear calls went through the Metal kernel and how many quantized tensors the stock
path dequantized, so a run shows which path it took.

LoRA reachability instrumentation is enabled when LORA is set. GGUF_LORA_AUDIT=1
enables it explicitly; =0 removes its Python overhead from timing runs. Counters
report whether the extra instrumentation was enabled.
"""
import importlib.util
import os
import sys
import weakref
from contextvars import ContextVar
from functools import wraps

HERE = os.path.dirname(os.path.abspath(__file__))
GGUF_DIR = os.environ.get("COMFY_GGUF_DIR") or os.path.join(os.path.dirname(HERE), "vendor", "ComfyUI-GGUF-fork")
COUNT = {"kernel_calls": 0, "stock_dequant_calls": 0}
_mods = {}


class _WeakIdentity:
    def __init__(self, value):
        self.ref = weakref.ref(value)

    def __eq__(self, other):
        value = self.ref()
        return isinstance(other, _WeakIdentity) and value is not None and value is other.ref()


def _patch_signature(weight):
    # Weak identities cannot alias recycled ids, and retain no source/factor tensors.
    def value_signature(value):
        if isinstance(value, (tuple, list)):
            return tuple(value_signature(item) for item in value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        try:
            return _WeakIdentity(value)
        except TypeError:
            return object()  # Unknown non-weakrefable values cannot prove a match.

    return value_signature(getattr(weight, "patches", None) or [])


def _install_lora_audit(ops, fast):
    """Observe completed forwards, keeping route counts and weak patch identities."""
    active = ContextVar("gguf_lora_audit", default=None)
    real_prepare = fast._lora_for_call
    real_linear = fast.linear
    real_forward = ops.GGMLOps.Linear.forward_ggml_cast_weights

    @wraps(real_prepare)
    def prepare(weight, x):
        factors = real_prepare(weight, x)
        record = active.get()
        if record is not None and factors is not None:
            import torch
            rank = 0 if factors.stacked is None else factors.stacked[0].shape[0]
            fused = rank > 0 and x.dtype == torch.bfloat16 and rank <= fast.FUSED_MAX_RANK
            record.update(kernel_fused=int(fused), kernel_beside=int(rank > 0 and not fused),
                          kernel_slices=len(factors.sliced))
        return factors

    @wraps(real_linear)
    def linear(layer, x):
        result = real_linear(layer, x)
        record = active.get()
        if record is not None:
            record["kernel"] = result is not None
        return result

    @wraps(real_forward)
    def forward(layer, *args, **kwargs):
        signature = _patch_signature(layer.weight)
        layer._mps_lora_audit = None
        record = dict(kernel=False, kernel_fused=0, kernel_beside=0, kernel_slices=0)
        token = active.set(record)
        try:
            result = real_forward(layer, *args, **kwargs)
            layer._mps_lora_audit = (signature, record)
            return result
        finally:
            active.reset(token)

    fast._lora_for_call = prepare
    fast.linear = linear
    ops.GGMLOps.Linear.forward_ggml_cast_weights = forward


def _load():
    if _mods:
        return _mods
    spec = importlib.util.spec_from_file_location(
        "comfyui_gguf_fork", os.path.join(GGUF_DIR, "__init__.py"), submodule_search_locations=[GGUF_DIR])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_gguf_fork"] = mod
    spec.loader.exec_module(mod)
    ops = sys.modules["comfyui_gguf_fork.ops"]
    nodes = sys.modules["comfyui_gguf_fork.nodes"]
    audit_enabled = os.environ.get("GGUF_LORA_AUDIT", "1" if os.environ.get("LORA") else "0") != "0"
    if audit_enabled:
        _install_lora_audit(ops, sys.modules["comfyui_gguf_fork.mps"])
    real_dequant = ops.dequantize_tensor

    def counted_dequant(tensor, *a, **kw):
        if getattr(tensor, "tensor_type", None) not in (None, 0, 1):
            COUNT["stock_dequant_calls"] += 1
        return real_dequant(tensor, *a, **kw)

    ops.dequantize_tensor = counted_dequant
    try:
        import comfy_kitchen as ck

        real_linear = ck.gguf_linear

        def counted_linear(*a, **kw):
            COUNT["kernel_calls"] += 1
            return real_linear(*a, **kw)

        ck.gguf_linear = counted_linear
    except (ImportError, AttributeError):
        pass
    _mods.update(ops=ops, nodes=nodes, path=GGUF_DIR, lora_audit_enabled=audit_enabled)
    return _mods


def load_unet(name):
    return _load()["nodes"].UnetLoaderGGUF().load_unet(name)[0]


def counters():
    return dict(COUNT, lora_audit_enabled=_mods.get("lora_audit_enabled", False))


def lora_audit(patcher):
    """Where a GGUF model's LoRA patches went, after sampling: on quantized layers either the
    Metal kernel's side path (stacked into the kernel or beside it) or ComfyUI-GGUF's stock
    per-call merge; non-quantized weights are merged by the patcher as usual."""
    if _mods.get("lora_audit_enabled") is False:
        return {"enabled": False}
    out = dict(quantized_layers_with_lora=0, kernel_fused=0, kernel_beside=0, kernel_slices=0,
               stock_merge_per_call=0, not_reached=0, non_quantized_patched=0)
    for _name, mod in patcher.model.named_modules():
        w = getattr(mod, "weight", None)
        patches = getattr(w, "patches", None)
        if not patches:
            if w is not None and getattr(w, "tensor_type", None) in (None, 0, 1, 30) and any(
                    k.startswith(_name + ".") for k in getattr(patcher, "patches", {}) if _name):
                out["non_quantized_patched"] += 1
            continue
        out["quantized_layers_with_lora"] += 1
        audit = getattr(mod, "_mps_lora_audit", None)
        if audit is None or audit[0] != _patch_signature(w):
            out["not_reached"] += 1
        elif not audit[1]["kernel"]:
            out["stock_merge_per_call"] += 1
        else:
            for key in ("kernel_fused", "kernel_beside", "kernel_slices"):
                out[key] += audit[1][key]
    return out
