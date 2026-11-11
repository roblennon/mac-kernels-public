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
  mask with no padding and T in the kernel's range (threshold .. 16) it returns None, because the
  kernel masks by position itself; otherwise it is sdpa_mask, unchanged. Skipping the mask also
  skips the check in transformers 5.18's sdpa_mask that compares the StaticCache's device-side
  length with 0 in Python (_ignore_causal_mask_sdpa), which under torch.compile breaks the graph
  and reads the GPU from the host on every forward with T > 1, stock SDPA included.
- smallq_attention_forward, registered with transformers' AttentionInterface under "mk_smallq".
  With no mask, T in the kernel range and a supported shape it calls the op with
  pos = position_ids[:, 0]. With a mask (T below the threshold, a padding mask, T > 16) it calls
  transformers' own sdpa_attention_forward with the same arguments, so T = 1 and 2 keep stock
  SDPA and its mask, and a mask is never ignored. With no mask in the kernel range but a case the
  kernel does not take (dtype, head dim, dropout, output_attentions), it builds the causal mask
  on the device from position_ids and calls sdpa_attention_forward with it. The return value is
  what sdpa_attention_forward returns: ([B, T, Hq, D] contiguous, None).
- enable / disable, which switch a loaded model with model.set_attn_implementation.

What the no-mask path assumes: query row t of batch element b sits at position_ids[b, 0] + t,
sees every key before it, and those keys are cache slots 0 .. position_ids[b, 0] + t. That is
what a Qwen3-style decoder does with a StaticCache, a DynamicCache or no cache when position_ids
are left to the model (it derives them from the cache length, so they match the slots written).
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
# T >= threshold uses the kernel. 4 by default: in the isolated timing (28 layers, 4,096 keys) the
# kernel was slower than stock SDPA at T = 1 and 2 (paired ratio 0.52, 0.75) and faster from 4 on.
SETTINGS = {"threshold": 4}


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
        if sq.supports_shapes(q, k, v) and _pos_ok(pos, q):
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


def smallq_mask(*args, **kwargs):
    """Mask function for "mk_smallq": None for a plain causal mask without padding at T in the
    kernel range (the attention function then masks by position); sdpa_mask otherwise."""
    if (not args and kwargs.get("attention_mask") is None and in_kernel_range(kwargs.get("q_length", 0))
            and kwargs.get("mask_function", causal_mask_function) is causal_mask_function
            and not kwargs.get("local_size") and kwargs.get("allow_is_causal_skip", True)):
        return None
    return sdpa_mask(*args, **kwargs)


def kernel_applies(module, query, key, value, attention_mask, **kwargs):
    """True when smallq_attention_forward takes the kernel for these arguments. Depends only on
    shapes, strides, dtypes, flags, the mask's presence and the threshold, so it is fixed per
    compiled graph."""
    q = query if query.stride(-1) == 1 else query.contiguous()
    return (attention_mask is None
            and in_kernel_range(query.shape[2])
            and kwargs.get("position_ids") is not None
            and not module.training
            and kwargs.get("dropout", 0.0) == 0.0
            and kwargs.get("sliding_window") is None
            and kwargs.get("position_bias") is None
            and kwargs.get("is_causal", True) is not False
            and not kwargs.get("output_attentions", False)
            and sq.supports_shapes(q, key, value))


def smallq_attention_forward(module, query, key, value, attention_mask, **kwargs):
    """transformers attention function: query [B, Hq, T, D]; key, value [B, Hkv, L, D] (with a
    StaticCache, the whole cache, L = Lmax). Returns ([B, T, Hq, D], None)."""
    if attention_mask is not None or not in_kernel_range(query.shape[2]):
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    position_ids = kwargs.get("position_ids")
    if position_ids is None:
        raise ValueError("mk_smallq: no mask and no position_ids; the causal mask cannot be placed")
    pos = position_ids[:, 0]
    if pos.dtype != torch.int64:
        pos = pos.long()
    if not kernel_applies(module, query, key, value, attention_mask, **kwargs):
        mask = position_mask(pos, query.shape[2], key.shape[2])
        return sdpa_attention_forward(module, query, key, value, mask, **kwargs)
    scaling = kwargs.get("scaling")
    scale = 1.0 / math.sqrt(query.shape[-1]) if scaling is None else float(scaling)
    out = torch.ops.mk.smallq_attention(query, key, value, pos, scale)
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register(NAME, smallq_attention_forward)
AttentionMaskInterface.register(NAME, smallq_mask)


def enable(model, threshold=4):
    """Switch a loaded model to the kernel for T >= threshold. Returns the previous attention
    implementation, for disable(). A compiled function recompiles after the switch (Dynamo guards
    on the implementation name and on the threshold). Each switch adds cache entries per T, and
    Dynamo's recompile limit (8) is reached after a few; compile a fresh function or call
    torch._dynamo.reset() after switching."""
    SETTINGS["threshold"] = int(threshold)
    prev = model.config._attn_implementation
    model.set_attn_implementation(NAME)
    return prev


def disable(model, implementation="sdpa"):
    """Switch back, by default to stock SDPA."""
    model.set_attn_implementation(implementation)
