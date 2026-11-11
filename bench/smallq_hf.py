"""The small-query attention kernel (bench/smallq_attention.py) inside a transformers model, so it
works under torch.compile. Experiment E of the mps-decode thread.

    from bench import smallq_hf
    prev = smallq_hf.enable(model, threshold=4)   # model.config._attn_implementation -> "mk_smallq"
    ...                                           # compile and run as usual, StaticCache included
    smallq_hf.disable(model, prev)                # back to the previous implementation ("sdpa")

Three pieces:

- mk::smallq_attention(q, k, v, pos, scale), a torch custom op with a fake implementation, so
  Inductor treats it as one opaque call. pos is a device tensor (int64, [1] or [B]); the kernel
  reads it on the GPU and the grid is sized from the cache length Lmax = k.shape[2], so a changing
  position neither recompiles nor synchronises. Inputs the kernel cannot run (dtype, head dim,
  alignment, T > 16) go to stock SDPA inside the op, with the causal mask built on the device from
  pos, so the op is total.
- smallq_mask, registered with transformers' mask interface under "mk_smallq". For a plain causal
  mask with no padding and T in the kernel's range (threshold .. 16) it returns, instead of a mask,
  the cache write offset (transformers' q_offset: the slot the new rows go to) as an int64 [B]
  tensor; otherwise it is sdpa_mask, unchanged. Skipping sdpa_mask also skips its check in
  transformers 5.18 that compares the StaticCache's device-side length with 0 in Python
  (_ignore_causal_mask_sdpa), which under torch.compile breaks the graph and reads the GPU from the
  host on forwards with T > 1 that reach it, stock SDPA included.
- smallq_attention_forward, registered with transformers' AttentionInterface under "mk_smallq".
  With that offset tensor and a supported case it calls the op with pos = the offset. With the
  offset but a case the kernel does not take (dtype, head dim, dropout, output_attentions, scale
  out of range) it builds the causal mask on the device from the offset and calls transformers'
  sdpa_attention_forward. With a real mask, or no mask at all, it calls sdpa_attention_forward with
  the arguments unchanged, so T below the threshold, padding, T > 16 and non-causal calls keep
  stock behaviour exactly. The return value is what sdpa_attention_forward returns:
  ([B, T, Hq, D] contiguous, None).
- enable / disable, which switch a loaded model with model.set_attn_implementation.

Positions: the kernel path masks by the cache slot (the offset above), as stock SDPA's mask does,
so caller-supplied position_ids that differ from the cache slots change only the rotary embedding,
exactly as with stock attention. Inference only: the op has no backward and refuses training mode.
Sliding-window layers are not covered: their mask function is not the plain causal one, so they
get a mask and go to stock SDPA.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, causal_mask_function, sdpa_mask

try:
    from bench import smallq_attention as sq
except ImportError:  # imported with bench/ on sys.path, as the bench scripts do
    import smallq_attention as sq

NAME = "mk_smallq"
# T >= threshold uses the kernel. 4 by default, set by the whole model, not the isolated timing. In
# isolation (28 layers, 4,096 keys, paired per-round ratios) E2's row-vector path ties stock SDPA
# at T = 1 (1.05x against a boolean mask, 0.92x against the fp16 additive mask transformers uses)
# and wins from T = 2 (1.4x). In whole Qwen3-1.7B, compiled, bench_verify_cost.py --rewind, three
# rounds (2026-11-11), the kernel path was slower at T = 1 in every round (0.91-0.97x at 4,096 and
# 16,384 tokens) and at T = 2 in two of three rounds at 4,096 (0.92, 0.985), and faster at T >= 4
# in every round's median but one inside the A/A spread. Experiment E's kernel had lost at T = 1
# and 2 even in isolation (0.52x, 0.75x). Those T = 1 timings needed decode_offset (smallq_mask).
SETTINGS = {"threshold": 4, "decode_offset": False}


def position_mask(pos, T, L):
    """[B or 1, 1, T, L] boolean causal mask built on the device: row t sees keys 0 .. pos[b] + t."""
    keys = torch.arange(L, device=pos.device)
    rows = torch.arange(T, device=pos.device)
    return keys.view(1, 1, 1, L) <= pos.view(-1, 1, 1, 1) + rows.view(1, 1, T, 1)


def _masked_sdpa(q, k, v, pos, scale):
    """Stock SDPA over the whole cache with the causal mask built on the device from pos."""
    mask = position_mask(pos, q.shape[2], k.shape[2])
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale,
                                          enable_gqa=k.shape[1] != q.shape[1]).contiguous()


def _pos_ok(pos, q):
    return (pos.dtype == torch.int64 and pos.device == q.device and pos.dim() == 1
            and pos.shape[0] in (1, q.shape[0]))


def _register_op():
    @torch.library.custom_op("mk::smallq_attention", mutates_args=())
    def smallq_attention_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos: torch.Tensor,
                            scale: float) -> torch.Tensor:
        """[B, Hq, T, D] attention of q's rows (positions pos .. pos + T - 1) over k/v[:, :, :pos + T]."""
        if q.stride(-1) != 1:
            q = q.contiguous()
        if sq.supports_shapes(q, k, v) and _pos_ok(pos, q) and sq.scale_ok(scale):
            return sq.smallq_attention_dev(q, k, v, pos, scale)
        return _masked_sdpa(q, k, v, pos.to(q.device, torch.int64), scale)

    @smallq_attention_op.register_fake
    def _(q, k, v, pos, scale):
        return q.new_empty(q.shape)


# The module may be imported twice in one process (as bench.smallq_hf and as smallq_hf); the op
# is registered once.
try:
    torch.ops.mk.smallq_attention
except (AttributeError, RuntimeError):
    _register_op()


def in_kernel_range(T):
    return SETTINGS["threshold"] <= T <= sq.MAX_T


def is_slot_marker(mask):
    """smallq_mask's stand-in for a plain causal mask: the cache write offset per batch element,
    an int64 [B] tensor. A real mask is a 4-D bool or float tensor, so the two cannot be confused."""
    return isinstance(mask, torch.Tensor) and mask.dtype == torch.int64 and mask.dim() == 1


def smallq_mask(*args, **kwargs):
    """Mask function for "mk_smallq". For a plain causal mask without padding at T in the kernel
    range it returns the cache write offset (q_offset, the slot the new rows are written to) as an
    int64 [batch] tensor instead of a mask; the attention function masks by it. Taking the offset
    from the cache, not from position_ids, keeps the kernel right when a caller passes positions that
    differ from the cache slots. Otherwise it is sdpa_mask, unchanged.
    A caller that sets allow_is_causal_skip=False asks for a materialized mask, and gets one. That
    includes transformers' own one-token steps on a compileable cache, so T = 1 normally gets
    sdpa_mask even at threshold 1. SETTINGS["decode_offset"] (off by default, a bench opt-in) hands
    the offset at T = 1 anyway. It is only safe for models whose attention takes the mask as given
    (Qwen3 does): from inside the mask function a compileable-cache decode step cannot be told apart
    from a model that needs the mask itself, such as Falcon (alibi), Doge (its dynamic mask indexes
    it) or an "indexed_attention" layer, which also pass allow_is_causal_skip=False with the plain
    causal mask function."""
    if (not args and kwargs.get("attention_mask") is None and in_kernel_range(kwargs.get("q_length", 0))
            and kwargs.get("mask_function", causal_mask_function) is causal_mask_function
            and not kwargs.get("local_size")
            and (kwargs.get("allow_is_causal_skip", True)
                 or (SETTINGS["decode_offset"] and kwargs.get("q_length") == 1))):
        off = kwargs.get("q_offset", 0)
        off = off.to(torch.int64) if isinstance(off, torch.Tensor) else torch.tensor(int(off), dtype=torch.int64)
        # A copy, never a view: q_offset can be the cache's own counter, which each layer's cache
        # update advances in place before its attention runs.
        return off.to(kwargs.get("device", off.device)).reshape(-1).expand(kwargs["batch_size"]).clone()
    return sdpa_mask(*args, **kwargs)


def kernel_applies(module, query, key, value, attention_mask, **kwargs):
    """True when smallq_attention_forward takes the kernel for these arguments. Depends only on
    shapes, strides, dtypes, flags, the mask's presence and the threshold, so it is fixed per
    compiled graph."""
    q = query if query.stride(-1) == 1 else query.contiguous()
    return (is_slot_marker(attention_mask)
            and in_kernel_range(query.shape[2])
            and not module.training
            and sq.scale_ok(kwargs.get("scaling"))
            and kwargs.get("dropout", 0.0) == 0.0
            and kwargs.get("sliding_window") is None
            and kwargs.get("position_bias") is None
            and kwargs.get("is_causal", True) is not False
            and not kwargs.get("output_attentions", False)
            and sq.supports_shapes(q, key, value))


def smallq_attention_forward(module, query, key, value, attention_mask, **kwargs):
    """transformers attention function: query [B, Hq, T, D]; key, value [B, Hkv, L, D] (with a
    StaticCache, the whole cache, L = Lmax). Returns ([B, T, Hq, D], None)."""
    if not is_slot_marker(attention_mask):  # a real mask, or none at all: stock semantics, unchanged
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    pos = attention_mask  # the cache write offset, from smallq_mask; the mask function was causal
    if not kernel_applies(module, query, key, value, attention_mask, **kwargs):
        mask = position_mask(pos, query.shape[2], key.shape[2])
        return sdpa_attention_forward(module, query, key, value, mask, **kwargs)
    scaling = kwargs.get("scaling")
    scale = 1.0 / math.sqrt(query.shape[-1]) if scaling is None else float(scaling)
    out = torch.ops.mk.smallq_attention(query, key, value, pos, scale)
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register(NAME, smallq_attention_forward)
AttentionMaskInterface.register(NAME, smallq_mask)


def enable(model, threshold=4, decode_offset=False):
    """Switch a loaded model to the kernel for T >= threshold. Returns the previous attention
    implementation, for disable(). decode_offset lets one-token steps on a compileable cache reach
    the kernel at threshold 1; see smallq_mask for when that is unsafe. A compiled function
    recompiles after the switch (Dynamo guards on the implementation name and on the settings).
    Each switch adds cache entries per T, and Dynamo's recompile limit (8) is reached after a few;
    compile a fresh function or call torch._dynamo.reset() after switching."""
    SETTINGS["threshold"] = int(threshold)
    SETTINGS["decode_offset"] = bool(decode_offset)
    prev = model.config._attn_implementation
    model.set_attn_implementation(NAME)
    return prev


def disable(model, implementation="sdpa"):
    """Switch back, by default to stock SDPA."""
    model.set_attn_implementation(implementation)
