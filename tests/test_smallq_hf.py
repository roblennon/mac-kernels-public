"""The small-query kernel inside transformers (bench/smallq_hf.py): the mk::smallq_attention custom op
under torch.compile, the "mk_smallq" attention function's routing between the kernel and stock
SDPA, and a small random Qwen3 model with a StaticCache, compiled, against stock SDPA.
Needs an Apple GPU; run through bench/gpu_runner.py like any other GPU process.
The whole Qwen3-1.7B comparison is bench/check_smallq_model.py (too heavy for the suite)."""
import math

import pytest
import torch
import torch._dynamo.testing
from transformers import Qwen3Config, Qwen3ForCausalLM, StaticCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from bench import smallq_attention as sq
from bench import smallq_hf

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def qkv(B=1, Hq=16, Hkv=8, T=8, Lmax=2048, D=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, Hq, T, D, generator=g).half().to("mps")
    k = torch.randn(B, Hkv, Lmax, D, generator=g).half().to("mps")
    v = torch.randn(B, Hkv, Lmax, D, generator=g).half().to("mps")
    return q, k, v


def op(q, k, v, pos, scale):
    return torch.ops.mk.smallq_attention(q, k, v, pos, scale)


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
    smallq_hf.SETTINGS["threshold"] = 4


@pytest.mark.parametrize("T", [1, 4, 9, 16])
def test_op_compiled_equals_eager_and_pos_changes_do_not_recompile(T):
    # Arrange: a compiled caller with a frame counter; pos moves between calls, in place and as a
    # new tensor.
    q, k, v = qkv(T=T, seed=T)
    scale = 1 / math.sqrt(128)
    cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
    f = torch.compile(op, backend=cnt, dynamic=False)
    pos = torch.tensor([100], device="mps")
    results = []
    # Act
    for p in (100, 1500, 2048 - T, 0):
        pos.fill_(p)
        results.append((p, f(q, k, v, pos, scale), op(q, k, v, pos, scale)))
    other = torch.tensor([777], device="mps")
    results.append((777, f(q, k, v, other, scale), op(q, k, v, other, scale)))
    # Assert: one compiled frame; compiled output bitwise equal to eager and to the kernel itself.
    assert cnt.frame_count == 1
    for p, compiled, eager in results:
        assert torch.equal(compiled, eager), p
        assert torch.equal(eager, sq.smallq_attention_dev(q, k, v, torch.tensor([p], device="mps"), scale))


def test_op_fallback_for_unsupported_shapes_matches_stock_masked_sdpa():
    # Arrange: head dim 96 is not a kernel shape, so the op runs stock SDPA with a device-built mask.
    q, k, v = qkv(T=6, Lmax=300, D=96, seed=2)
    pos = torch.tensor([200], device="mps")
    mask = torch.arange(300, device="mps")[None, :] <= 200 + torch.arange(6, device="mps")[:, None]
    expect = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=0.1, enable_gqa=True)
    # Act
    eager = op(q, k, v, pos, 0.1)
    compiled = torch.compile(op, dynamic=False)(q, k, v, pos, 0.1)
    # Assert
    assert torch.equal(eager, expect)
    assert torch.equal(compiled, expect)


class _Attn(torch.nn.Module):
    """The attributes transformers' sdpa_attention_forward reads from the attention module."""
    def __init__(self, groups=2):
        super().__init__()
        self.num_key_value_groups = groups
        self.is_causal = True
        self.eval()


def _sdpa_mask(T, pos, Lmax):
    return (torch.arange(Lmax, device="mps")[None, :] <= pos + torch.arange(T, device="mps")[:, None])[None, None]


@pytest.mark.parametrize("T, uses_kernel", [(1, False), (2, False), (3, False), (4, True), (9, True), (16, True)])
def test_attention_function_routes_by_threshold(T, uses_kernel):
    # Arrange: the call Qwen3Attention makes, with the full static cache as key and value and the
    # mask smallq_mask gives (the cache offset in the kernel range, sdpa_mask's below it).
    pos, Lmax = 1000, 1100
    q, k, v = qkv(T=T, Lmax=Lmax, seed=T)
    q = q.transpose(1, 2).contiguous().transpose(1, 2)        # the model's [B, T, H, D] memory layout
    kwargs = dict(dropout=0.0, scaling=128 ** -0.5, sliding_window=None,
                  position_ids=(pos + torch.arange(T, device="mps"))[None])
    mask = _sdpa_mask(T, pos, Lmax)
    given = mask_for(T, pos, Lmax)
    # Act
    out, w = smallq_hf.smallq_attention_forward(_Attn(), q, k, v, given, **kwargs)
    # Assert: layout and routing.
    stock, _ = sdpa_attention_forward(_Attn(), q, k, v, mask, **kwargs)
    assert smallq_hf.is_slot_marker(given) == uses_kernel
    assert w is None and out.shape == (1, T, 16, 128) and out.is_contiguous()
    kern = sq.smallq_attention_dev(q, k, v, torch.tensor([pos], device="mps"), 128 ** -0.5).transpose(1, 2)
    assert torch.equal(out, kern if uses_kernel else stock)
    assert (out.float() - stock.float()).abs().max() < 2e-3


def mask_for(T, pos, Lmax, attention_mask=None):
    """What create_causal_mask gets from smallq_mask for a StaticCache at pos (which turns the
    causal skip off at T = 1 for a compileable cache)."""
    return smallq_hf.smallq_mask(batch_size=1, q_length=T, kv_length=Lmax,
                                 q_offset=torch.tensor(pos, device="mps"), kv_offset=0,
                                 attention_mask=attention_mask, allow_is_causal_skip=T != 1, device="mps")


def test_mask_function_skips_only_plain_causal_masks_in_the_kernel_range():
    # Arrange
    pos, Lmax = 300, 400
    pad = torch.ones(1, Lmax, dtype=torch.bool, device="mps")
    pad[0, :5] = False
    # Act / Assert
    for T in (4, 16):
        m = mask_for(T, pos, Lmax)
        assert smallq_hf.is_slot_marker(m) and m.tolist() == [pos]
    for T in (1, 3, 17):
        assert torch.equal(mask_for(T, pos, Lmax), _sdpa_mask(T, pos, Lmax))
    padded = mask_for(8, pos, Lmax, attention_mask=pad)
    assert padded is not None and not padded[..., :5].any() and padded[0, 0, 0, 5:pos + 1].all()
    smallq_hf.SETTINGS["threshold"] = 9
    assert not smallq_hf.is_slot_marker(mask_for(8, pos, Lmax))


def test_attention_function_never_ignores_a_mask():
    # Arrange: a padding mask in the kernel range (keys 0..4 hidden) must reach stock SDPA.
    pos, Lmax, T = 300, 400, 8
    q, k, v = qkv(T=T, Lmax=Lmax, seed=4)
    pad = torch.ones(1, Lmax, dtype=torch.bool, device="mps")
    pad[0, :5] = False
    mask = mask_for(T, pos, Lmax, attention_mask=pad)
    kw = dict(dropout=0.0, scaling=128 ** -0.5, position_ids=(pos + torch.arange(T, device="mps"))[None])
    # Act
    out = smallq_hf.smallq_attention_forward(_Attn(), q, k, v, mask, **kw)[0]
    # Assert
    assert torch.equal(out, sdpa_attention_forward(_Attn(), q, k, v, mask, **kw)[0])


def test_attention_function_unsupported_cases_use_a_device_mask():
    # Arrange: the cache offset in the kernel range, but a case the kernel does not take; the fallback
    # builds the causal mask from the offset and must equal stock SDPA with sdpa_mask's mask.
    pos, Lmax, T = 50, 128, 8
    q, k, v = qkv(T=T, Lmax=Lmax, seed=3)
    base = dict(dropout=0.0, scaling=128 ** -0.5, position_ids=(pos + torch.arange(T, device="mps"))[None])
    stock = sdpa_attention_forward(_Attn(), q, k, v, _sdpa_mask(T, pos, Lmax), **base)[0]
    q96, k96, v96 = qkv(T=T, Lmax=Lmax, D=96, seed=3)
    stock96 = sdpa_attention_forward(_Attn(), q96, k96, v96, _sdpa_mask(T, pos, Lmax), **base)[0]
    # Act / Assert
    for kw in (dict(base, output_attentions=True), dict(base, sliding_window=64)):
        assert torch.equal(smallq_hf.smallq_attention_forward(_Attn(), q, k, v, _slot(pos), **kw)[0], stock), kw
    assert torch.equal(smallq_hf.smallq_attention_forward(_Attn(), q96, k96, v96, _slot(pos), **base)[0], stock96)
    # a scale the kernel would overflow on goes to stock too
    big = dict(base, scaling=3e38)
    out = smallq_hf.smallq_attention_forward(_Attn(), q, k, v, _slot(pos), **big)[0]
    ref = sdpa_attention_forward(_Attn(), q, k, v, _sdpa_mask(T, pos, Lmax), **big)[0]
    torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True)  # stock overflows here too


def _slot(pos):
    return torch.tensor([pos], dtype=torch.int64, device="mps")


def test_kernel_masks_by_cache_slot_not_position_ids():
    # Arrange (Sol's counterexample): the cache writes at slot 100 while the caller passes rotary
    # positions 50..53. Stock SDPA masks by the cache length; the kernel path must too.
    slot, Lmax, T = 100, 256, 4
    q, k, v = qkv(T=T, Lmax=Lmax, seed=7)
    kw = dict(dropout=0.0, scaling=128 ** -0.5, position_ids=(50 + torch.arange(T, device="mps"))[None])
    # Act
    out = smallq_hf.smallq_attention_forward(_Attn(), q, k, v, _slot(slot), **kw)[0]
    # Assert
    stock = sdpa_attention_forward(_Attn(), q, k, v, _sdpa_mask(T, slot, Lmax), **kw)[0]
    wrong = sdpa_attention_forward(_Attn(), q, k, v, _sdpa_mask(T, 50, Lmax), **kw)[0]
    assert (out.float() - stock.float()).abs().max() < 2e-3
    assert (out.float() - wrong.float()).abs().max() > 1e-2


def test_no_mask_keeps_stock_semantics_including_non_causal():
    # Arrange (Sol's counterexample): with no mask and is_causal=False every row sees every key.
    T = L = 4
    q = torch.zeros(1, 1, T, 64, dtype=torch.float16, device="mps")
    k = torch.zeros(1, 1, L, 64, dtype=torch.float16, device="mps")
    v = torch.arange(L, dtype=torch.float16, device="mps").view(1, 1, L, 1).expand(1, 1, L, 64).contiguous()
    for is_causal in (False, True):
        kw = dict(dropout=0.0, scaling=0.125, is_causal=is_causal)
        # Act
        out = smallq_hf.smallq_attention_forward(_Attn(), q, k, v, None, **kw)[0]
        # Assert
        assert torch.equal(out, sdpa_attention_forward(_Attn(), q, k, v, None, **kw)[0])
    out = smallq_hf.smallq_attention_forward(_Attn(), q, k, v, None, dropout=0.0, scaling=0.125, is_causal=False)[0]
    assert torch.allclose(out[0, :, 0, 0].float(), torch.full((T,), 1.5, device="mps"))


def tiny_qwen3():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=128, max_position_embeddings=4096,
                      attn_implementation="sdpa")
    return Qwen3ForCausalLM(cfg).half().to("mps").eval()


@torch.no_grad()
def test_enable_disable_and_compiled_static_cache_model_matches_stock():
    # Arrange: prefill 700 tokens into a StaticCache, then teacher-forced verify forwards at several
    # T from the same position (the counter is rolled back between forwards), compiled, for stock
    # SDPA and for mk_smallq.
    model = tiny_qwen3()
    P, Lmax = 700, 800
    g = torch.Generator().manual_seed(1)
    prompt = torch.randint(0, 512, (1, P), generator=g).to("mps")
    cache = StaticCache(config=model.config, max_cache_len=Lmax)
    model(prompt, past_key_values=cache, use_cache=True)
    drafts = {T: torch.randint(0, 512, (1, T), generator=g).to("mps") for T in (1, 4, 9, 16)}

    def fwd(ids, cache):
        return model(ids, past_key_values=cache, use_cache=True).logits

    def verify(step):
        out = {}
        for T, ids in drafts.items():
            for layer in cache.layers:
                layer.cumulative_length.fill_(P)
            out[T] = step(ids, cache).float()
        return out

    stock = verify(torch.compile(fwd, dynamic=False))
    # Act
    prev = smallq_hf.enable(model)
    torch._dynamo.reset()
    cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
    ours = verify(torch.compile(fwd, backend=cnt, dynamic=False))
    again = verify(torch.compile(fwd, backend=cnt, dynamic=False))
    smallq_hf.disable(model, prev)
    # Assert
    assert prev == "sdpa" and model.config._attn_implementation == "sdpa"
    # One graph per T and none for a repeat; stock transformers breaks the graph at T > 1
    # (_ignore_causal_mask_sdpa), and at T = 2 and 3 mk_smallq is stock, so the count is
    # T=1: 1, T=4: 1, T=9: 1, T=16: 1 when the kernel range has no breaks.
    assert cnt.frame_count == len(drafts), cnt.frame_count
    for T in drafts:
        assert torch.isfinite(ours[T]).all()
        assert torch.equal(ours[T], again[T])
        d = (ours[T] - stock[T]).abs().max().item()
        assert d < 0.05, (T, d)
        if T < smallq_hf.SETTINGS["threshold"]:
            assert torch.equal(ours[T], stock[T])    # below the threshold it is stock SDPA


def test_slot_offset_is_a_copy_not_a_view_of_the_cache_counter():
    # Arrange: q_offset can be a StaticLayer's cumulative_length, which the layer's cache update
    # advances in place before its attention runs; a view would shift the mask by T.
    counter = torch.tensor(300, dtype=torch.int64, device="mps")
    # Act
    m = smallq_hf.smallq_mask(batch_size=2, q_length=8, kv_length=400, q_offset=counter, kv_offset=0, device="mps")
    counter.add_(8)
    # Assert
    assert smallq_hf.is_slot_marker(m) and m.tolist() == [300, 300]
