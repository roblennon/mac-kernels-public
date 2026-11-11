"""Small-query attention kernel (bench/smallq_attention.py) against a float64 reference, with stock
torch SDPA's own error as the yardstick, plus planted faults the check must catch. Both entry points
are covered: smallq_attention (pos a host int, chunks sized from pos + T) and smallq_attention_dev
(pos a device tensor read by the kernel, chunks sized from Lmax), marked host / dev.

Pass rule, fixed before the kernel was measured: on every case the kernel's max abs error against
float64 is at most 1.5x stock SDPA's (fp16 on MPS, same inputs, explicit boolean mask,
enable_gqa=True) and its mean abs error at most 1.25x, and every output is finite.
Needs an Apple GPU; run through bench/gpu_runner.py like any other GPU process."""
import math

import pytest
import torch
import torch.nn.functional as F

from bench import smallq_attention as sq

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")

MAX_RATIO, MEAN_RATIO = 1.5, 1.25
# The default, experiment E's configuration (BASELINE, still selectable), and the other
# configurations measured while tuning, so whichever is chosen is covered. The last two take the
# row-vector path with 32 lanes per key (4-element loads; D = 64 falls back to the matrix path),
# up to 4 rows per simdgroup and the one-simdgroup parallel reduce, and with 16 lanes per key, rows
# above 16 on the matrix path, and an 8-simdgroup reduce.
CONFIGS = [sq.DEFAULT, sq.BASELINE] + [sq.Config.parse(n) for n in
                                       ("bk32_ns16_f_q_r4", "bk64_ns32_f_t_r4", "bk32_ns8_h_t_r2", "bk64_ns128_h_q_r1",
                                        "bk32_ns16_f_t_r2_v32g4u1n32l32_rs1", "bk32_ns16_f_t_r2_v16g2u2n64l16_rs8")]


def make(B, Hq, Hkv, T, pos, D=128, extra=21, garbage="large", qscale=1.0, seed=0):
    """q [B, Hq, T, D]; k, v [B, Hkv, Lmax, D] with Lmax = pos + T + extra; slots past pos + T hold garbage."""
    g = torch.Generator().manual_seed(seed)
    Lk = pos + T
    Lmax = Lk + extra
    q = (torch.randn(B, Hq, T, D, generator=g) * qscale).half()
    k = torch.randn(B, Hkv, Lmax, D, generator=g).half()
    v = torch.randn(B, Hkv, Lmax, D, generator=g).half()
    fill = {"large": 6.0e4, "nan": float("nan"), "zero": 0.0}[garbage]
    k[:, :, Lk:] = fill
    v[:, :, Lk:] = -fill if garbage == "large" else fill
    return q, k, v


def reference(q, k, v, pos, scale=None):
    B, Hq, T, D = q.shape
    G = Hq // k.shape[1]
    Lk = pos + T
    qd = q.cpu().double()
    kd = k[:, :, :Lk].cpu().double().repeat_interleave(G, 1)
    vd = v[:, :, :Lk].cpu().double().repeat_interleave(G, 1)
    s = (qd @ kd.transpose(-1, -2)) * (1 / math.sqrt(D) if scale is None else scale)
    s = s.masked_fill(~causal_mask(T, pos, Lk), float("-inf"))
    return torch.softmax(s, -1) @ vd


def causal_mask(T, pos, L):
    """[T, L] boolean: row i sees keys 0 .. pos + i."""
    return torch.arange(L)[None, :] <= pos + torch.arange(T)[:, None]


def stock(q, k, v, pos, scale=None):
    """Stock SDPA on the valid prefix (stale slots excluded) with an explicit boolean mask."""
    T = q.shape[2]
    Lk = pos + T
    m = causal_mask(T, pos, Lk).to("mps")
    return F.scaled_dot_product_attention(q, k[:, :, :Lk], v[:, :, :Lk], attn_mask=m, scale=scale,
                                          enable_gqa=True)


def errors(out, ref):
    e = (out.cpu().double() - ref).abs()
    return e.max().item(), e.mean().item()


def run(gq, gk, gv, pos, dev=False, **kw):
    """The kernel on MPS tensors: host-int pos, or (dev) pos as a [1] int64 device tensor."""
    if dev:
        return sq.smallq_attention_dev(gq, gk, gv, torch.tensor([pos], device="mps"), **kw)
    return sq.smallq_attention(gq, gk, gv, pos, **kw)


def check(q, k, v, pos, scale=None, cfg=None, lib=None, shift=0, dev=False):
    """(passed, detail) under the fixed rule; q, k, v on CPU."""
    gq, gk, gv = q.to("mps"), k.to("mps"), v.to("mps")
    out = run(gq, gk, gv, pos, dev, scale=scale, config=cfg, lib=lib, _mask_shift=shift)
    ref = reference(q, k, v, pos, scale)
    ours = errors(out, ref)
    base = errors(stock(gq, gk, gv, pos, scale), ref)
    finite = bool(torch.isfinite(out).all())
    ok = finite and ours[0] <= MAX_RATIO * base[0] and ours[1] <= MEAN_RATIO * base[1]
    return ok, dict(ours=ours, stock=base, finite=finite)


QWEN = dict(B=1, Hq=16, Hkv=8, D=128)


@pytest.mark.parametrize("dev", [False, True], ids=["host", "dev"])
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name())
@pytest.mark.parametrize("T", [1, 2, 3, 7, 8, 9, 15, 16])
def test_qwen_shape_partial_block_with_large_garbage(cfg, T, dev):
    # Arrange: L_valid = 1003, a multiple of no block size; stale slots hold +-60000.
    q, k, v = make(1, 16, 8, T, 1003 - T, garbage="large", seed=T)
    # Act
    ok, d = check(q, k, v, 1003 - T, cfg=cfg, dev=dev)
    # Assert
    assert ok, d


@pytest.mark.parametrize("dev", [False, True], ids=["host", "dev"])
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name())
@pytest.mark.parametrize("T", [1, 2, 3, 7, 8, 9, 15, 16])
def test_pos_zero_with_nan_garbage(cfg, T, dev):
    # Arrange: the whole cache is the new block; everything past it is NaN.
    q, k, v = make(1, 16, 8, T, 0, garbage="nan", seed=100 + T)
    # Act
    ok, d = check(q, k, v, 0, cfg=cfg, dev=dev)
    # Assert
    assert ok, d


def test_single_valid_key_is_exact():
    # Arrange: L_valid = 1 (pos 0, T 1); the output must be v[0] itself.
    q, k, v = make(1, 16, 8, 1, 0, garbage="nan", seed=7)
    # Act
    out = sq.smallq_attention(q.to("mps"), k.to("mps"), v.to("mps"), 0)
    # Assert
    assert torch.equal(out.cpu(), v[:, :, :1].repeat_interleave(2, 1))
    assert check(q, k, v, 0)[0]


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name())
@pytest.mark.parametrize("B, Hq, Hkv, T, pos", [
    (1, 8, 8, 9, 517), (2, 8, 8, 16, 260),          # G = 1
    (2, 16, 8, 5, 2049), (2, 16, 8, 12, 77),        # G = 2, batch 2
    (1, 16, 4, 8, 1500), (2, 16, 4, 16, 3001),      # G = 4; at T = 16 that is 64 rows, two row groups
    (1, 4, 1, 15, 33),
])
@pytest.mark.parametrize("dev", [False, True], ids=["host", "dev"])
def test_grouped_heads_and_batch(cfg, B, Hq, Hkv, T, pos, dev):
    # Arrange
    q, k, v = make(B, Hq, Hkv, T, pos, garbage="nan", seed=pos)
    # Act
    ok, d = check(q, k, v, pos, cfg=cfg, dev=dev)
    # Assert
    assert ok, d


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name())
@pytest.mark.parametrize("T, pos", [(1, 16383), (9, 16391), (16, 4090)])
@pytest.mark.parametrize("dev", [False, True], ids=["host", "dev"])
def test_long_cache_many_chunks(cfg, T, pos, dev):
    # Arrange: long caches, so chunks hold many sub-blocks and the running max is rescaled.
    q, k, v = make(1, 16, 8, T, pos, garbage="large", seed=T)
    # Act
    ok, d = check(q, k, v, pos, cfg=cfg, dev=dev)
    # Assert
    assert ok, d


@pytest.mark.parametrize("T, pos", [(1, 700), (8, 700), (16, 700), (9, 0)])
def test_sharp_attention_and_custom_scale(T, pos):
    # Arrange: q scaled 6x gives peaked softmax rows; then a non-default scale.
    q, k, v = make(1, 16, 8, T, pos, qscale=6.0, seed=3)
    # Act / Assert
    assert check(q, k, v, pos)[0]
    assert check(q, k, v, pos, scale=0.05)[0]


# Logits far apart between key chunks: q x100 or x300 puts row maxima hundreds of log2 units apart,
# so the cross-chunk weights exp2(m_j - M) in the reduce overflow fp32 unless M is the true max.
EXTREME = [(16, 2000, 100.0, None), (1, 5000, 300.0, None), (9, 3000, 1.0, 50.0), (16, 500, 1.0, -3.0)]


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name())
@pytest.mark.parametrize("T, pos, qscale, scale", EXTREME, ids=["q100", "q300", "scale50", "scale-3"])
@pytest.mark.parametrize("dev", [False, True], ids=["host", "dev"])
def test_extreme_logits_across_chunks(cfg, T, pos, qscale, scale, dev):
    # Arrange
    q, k, v = make(1, 16, 8, T, pos, qscale=qscale, seed=pos + T)
    # Act
    ok, d = check(q, k, v, pos, scale=scale, cfg=cfg, dev=dev)
    # Assert
    assert ok, d


@pytest.mark.parametrize("T, pos, qscale, scale", EXTREME[:3], ids=["q100", "q300", "scale50"])
@pytest.mark.parametrize("dev", [False, True], ids=["host", "dev"])
def test_planted_reduce_max_from_first_chunk_is_caught(T, pos, qscale, scale, dev):
    # Arrange: the reduce takes its reference max from chunk 0 alone instead of over every chunk.
    old, new = "for (int s = 0; s < NS; ++s) M = max(M, ml[s].x);", "M = ml[0].x;"
    src = sq.SOURCE_PATH.read_text()
    assert src.count(old) == 1
    lib = sq._Library(src.replace(old, new))
    q, k, v = make(1, 16, 8, T, pos, qscale=qscale, seed=pos + T)
    # Act
    ok, d = check(q, k, v, pos, scale=scale, lib=lib, dev=dev)
    # Assert
    assert not ok, d
    assert check(q, k, v, pos, scale=scale, dev=dev)[0]


@pytest.mark.parametrize("T, pos", [(1, 300), (7, 45), (16, 1000), (9, 0)])
def test_head_dim_64(T, pos):
    # Arrange
    q, k, v = make(2, 8, 2, T, pos, D=64, garbage="nan", seed=T)
    # Act / Assert
    for cfg in CONFIGS:
        assert check(q, k, v, pos, cfg=cfg)[0]


def test_query_in_model_layout_is_not_copied():
    # Arrange: q as the model produces it, [B, T, Hq, D] transposed to [B, Hq, T, D].
    q, k, v = make(2, 16, 8, 9, 2000, seed=11)
    qt = q.transpose(1, 2).contiguous().to("mps").transpose(1, 2)
    assert not qt.is_contiguous()
    # Act
    out = sq.smallq_attention(qt, k.to("mps"), v.to("mps"), 2000)
    # Assert
    assert torch.equal(out.cpu(), sq.smallq_attention(q.to("mps"), k.to("mps"), v.to("mps"), 2000).cpu())


@pytest.mark.parametrize("T", [1, 9, 16])
def test_stale_slots_do_not_leak(T):
    # Arrange: the same valid prefix, with zeros and then NaN in the stale slots.
    q, k, v = make(1, 16, 8, T, 777, extra=40, garbage="zero", seed=T)
    _, kn, vn = make(1, 16, 8, T, 777, extra=40, garbage="nan", seed=T)
    # Act
    a = sq.smallq_attention(q.to("mps"), k.to("mps"), v.to("mps"), 777)
    b = sq.smallq_attention(q.to("mps"), kn.to("mps"), vn.to("mps"), 777)
    # Assert: bitwise identical
    assert torch.equal(a.cpu(), b.cpu())


@pytest.mark.parametrize("shift", [1, -1])
@pytest.mark.parametrize("T, pos", [(2, 40), (8, 1003), (9, 1003), (16, 60)])
@pytest.mark.parametrize("dev", [False, True], ids=["host", "dev"])
def test_planted_off_by_one_mask_is_caught(shift, T, pos, dev):
    # Arrange: every row's causal limit moved by one key (one future key seen, or the diagonal dropped).
    # T >= 2, since at T = 1 a limit one key further is the end of the valid cache and changes nothing.
    q, k, v = make(1, 16, 8, T, pos, seed=T)
    # Act / Assert
    assert not check(q, k, v, pos, shift=shift, dev=dev)[0]


@pytest.mark.parametrize("old, new", [
    ("a = exp2(mold - mnew);", "a = 1.0f;"),                     # accumulators never rescaled
    ("if (any) _Pragma", "if (false) _Pragma"),                   # rescale factors computed, not applied
])
@pytest.mark.parametrize("dev", [False, True], ids=["host", "dev"])
def test_planted_missing_rescale_is_caught(old, new, dev):
    # Arrange: keys in the later part of the cache score far higher, so each chunk's max moves
    # mid-chunk (chunks of 12 sub-blocks).
    src = sq.SOURCE_PATH.read_text()
    assert src.count(old) == 1
    lib = sq._Library(src.replace(old, new))
    q, k, v = make(1, 16, 8, 8, 3000, seed=5)
    k[:, :, 1500:3008] *= 4
    # Act
    ok, d = check(q, k, v, 3000, cfg=sq.Config(nsmax=8), lib=lib, dev=dev)
    # Assert
    assert not ok, d
    assert check(q, k, v, 3000, cfg=sq.Config(nsmax=8), dev=dev)[0]


def test_supports_refuses_what_it_cannot_run():
    # Arrange
    q, k, v = (t.to("mps") for t in make(1, 16, 8, 4, 100))
    # Act / Assert
    assert sq.supports(q, k, v, 100)
    assert not sq.supports(q.float(), k.float(), v.float(), 100)                       # dtype
    assert not sq.supports(q[..., :96].contiguous(), k[..., :96].contiguous(), v[..., :96].contiguous(), 100)  # D
    q17, k17, v17 = (t.to("mps") for t in make(1, 16, 8, 17, 10))
    assert not sq.supports(q17, k17, v17, 10)                                           # T > 16
    assert not sq.supports(q, k.transpose(-1, -2).contiguous().transpose(-1, -2), v, 100)  # last dim strided
    assert not sq.supports(q, k, v, k.shape[2] - 3)                                     # pos + T > Lmax
    assert not sq.supports(q[:, :15], k, v, 100)                                        # Hq % Hkv
    kb = torch.empty(k.numel() + 8, dtype=k.dtype, device=k.device)
    km = kb[1:1 + k.numel()].view(k.shape)
    assert not sq.supports(q, km, v, 100)                                               # rows not 16-byte aligned
    with pytest.raises(ValueError):
        sq.smallq_attention(q17, k17, v17, 10)


# ---- Device-position entry only ----------------------------------------------------------------

@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name())
@pytest.mark.parametrize("T, pos, extra", [(1, 0, 4095), (9, 0, 4000), (16, 1000, 6000), (4, 4092, 0),
                                           (8, 5000, 3), (2, 77, 12000)])
def test_dev_cache_mostly_empty_or_full(cfg, T, pos, extra):
    # Arrange: chunks are sized from Lmax, so with a long cache and a small pos most chunks lie past
    # pos + T and must contribute nothing; with extra = 0 the valid keys fill the cache exactly.
    q, k, v = make(1, 16, 8, T, pos, extra=extra, garbage="nan", seed=T + pos)
    # Act
    ok, d = check(q, k, v, pos, cfg=cfg, dev=True)
    # Assert
    assert ok, d


@pytest.mark.parametrize("T", [1, 4, 9, 16])
def test_dev_per_batch_positions(T):
    # Arrange: B = 3 with a different pos per batch element (position_ids[:, 0] of a [B, T] tensor),
    # NaN in each element's own stale slots.
    B, Lmax, poss = 3, 2100, [5, 2100 - T, 1234]
    g = torch.Generator().manual_seed(T)
    q = torch.randn(B, 16, T, 128, generator=g).half()
    k = torch.randn(B, 8, Lmax, 128, generator=g).half()
    v = torch.randn(B, 8, Lmax, 128, generator=g).half()
    for b, p in enumerate(poss):
        k[b, :, p + T:] = float("nan")
        v[b, :, p + T:] = float("nan")
    position_ids = (torch.tensor(poss)[:, None] + torch.arange(T)[None]).to("mps")
    gq, gk, gv = q.to("mps"), k.to("mps"), v.to("mps")
    # Act
    out = sq.smallq_attention_dev(gq, gk, gv, position_ids[:, 0])
    # Assert: each element under the fixed rule against its own reference and stock SDPA.
    for b, p in enumerate(poss):
        ref = reference(q[b:b + 1], k[b:b + 1], v[b:b + 1], p)
        ours = errors(out[b:b + 1], ref)
        base = errors(stock(gq[b:b + 1], gk[b:b + 1], gv[b:b + 1], p), ref)
        assert torch.isfinite(out[b]).all()
        assert ours[0] <= MAX_RATIO * base[0] and ours[1] <= MEAN_RATIO * base[1], (b, ours, base)


def test_dev_matches_host_path_on_the_same_chunks():
    # Arrange: when pos + T == Lmax both entries split the keys identically, so they agree bitwise.
    q, k, v = (t.to("mps") for t in make(2, 16, 8, 9, 3000, extra=0, seed=1))
    # Act
    a = sq.smallq_attention(q, k, v, 3000)
    b = sq.smallq_attention_dev(q, k, v, torch.tensor([3000], device="mps"))
    # Assert
    assert torch.equal(a, b)


def test_dev_pos_changes_between_calls_without_host_reads(monkeypatch):
    # Arrange: one pos tensor updated in place on the device between calls, as a decode loop would.
    # Every host read of a tensor value is made to raise, so any synchronising read in the
    # device-position path fails the test.
    T, Lmax = 8, 3000
    g = torch.Generator().manual_seed(9)
    q = torch.randn(1, 16, T, 128, generator=g).half()
    k = torch.randn(1, 8, Lmax, 128, generator=g).half()
    v = torch.randn(1, 8, Lmax, 128, generator=g).half()
    gq, gk, gv = q.to("mps"), k.to("mps"), v.to("mps")
    pos = torch.zeros(1, dtype=torch.int64, device="mps")
    sq.smallq_attention_dev(gq, gk, gv, pos)        # compile the kernels before the guard
    outs, values = [], [0, 517, 2000, Lmax - T, 3]

    def no_host_read(*a, **kw):
        raise AssertionError("host read of a device tensor")
    # Act
    with monkeypatch.context() as m:
        for name in ("item", "tolist", "cpu", "numpy", "__int__", "__index__", "__bool__", "__float__"):
            m.setattr(torch.Tensor, name, no_host_read)
        for p in values:
            pos.fill_(p)
            outs.append(sq.smallq_attention_dev(gq, gk, gv, pos))
    # Assert
    for p, out in zip(values, outs):
        ref = reference(q, k, v, p)
        ours = errors(out, ref)
        base = errors(stock(gq, gk, gv, p), ref)
        assert ours[0] <= MAX_RATIO * base[0] and ours[1] <= MEAN_RATIO * base[1], (p, ours, base)


def test_dev_refuses_what_it_cannot_run():
    # Arrange
    q, k, v = (t.to("mps") for t in make(2, 16, 8, 4, 100))
    pos = torch.tensor([100], device="mps")
    # Act / Assert
    assert sq.supports_shapes(q, k, v)
    with pytest.raises(ValueError):
        sq.smallq_attention_dev(q, k, v, pos.int())                  # int32 pos
    with pytest.raises(ValueError):
        sq.smallq_attention_dev(q, k, v, pos.cpu())                  # pos on the host
    with pytest.raises(ValueError):
        sq.smallq_attention_dev(q, k, v, torch.tensor([1, 2, 3], device="mps"))  # 3 values for B = 2
    with pytest.raises(ValueError):
        sq.smallq_attention_dev(q.float(), k.float(), v.float(), pos)


def test_scale_outside_the_kernel_range_is_refused():
    # Arrange: a finite scale large enough that scale * log2(e) overflows fp32 would give 0 * inf.
    if not torch.backends.mps.is_available():
        pytest.skip("needs MPS")
    q = torch.zeros(1, 1, 1, 64, dtype=torch.float16, device="mps")
    k = torch.zeros(1, 1, 8, 64, dtype=torch.float16, device="mps")
    v = torch.ones(1, 1, 8, 64, dtype=torch.float16, device="mps")
    # Act / Assert
    assert sq.scale_ok(None) and sq.scale_ok(0.088) and sq.scale_ok(-sq.MAX_SCALE)
    for bad in (3e38, float("inf"), float("nan"), torch.tensor(0.1, device="mps")):
        assert not sq.scale_ok(bad)
        with pytest.raises(ValueError):
            sq.smallq_attention(q, k, v, 0, scale=bad)
        with pytest.raises(ValueError):
            sq.smallq_attention_dev(q, k, v, torch.zeros(1, dtype=torch.int64, device="mps"), scale=bad)


# ---- Row-vector path and parallel reduce (E2) -----------------------------------------------------

def test_config_names_round_trip_and_baseline_is_experiment_e():
    # Arrange / Act / Assert
    assert sq.BASELINE.name() == "bk32_ns16_f_t_r2" and sq.BASELINE == sq.Config()
    for c in CONFIGS:
        assert sq.Config.parse(c.name()) == c
    with pytest.raises(ValueError):
        sq.Config.parse("bk32_ns16_f_t_r2_x1")


@pytest.mark.parametrize("Lk", [1, 63, 64, 65, 1003, 4096, 4112, 16400])
@pytest.mark.parametrize("cfg", [c for c in CONFIGS if c.vrows], ids=lambda c: c.name())
def test_vec_plans_cover_every_key_and_row(cfg, Lk):
    # Arrange / Act / Assert: chunks are whole steps, cover the keys, and fill whole threadgroups;
    # row groups cover every row; the grid-y count is the kernel's own formula.
    for R in range(1, sq.VMAX_ROWS + 1):
        RV, RGT, RGY, VCH = sq.vec_groups(R, cfg)
        assert RV * RGT * RGY >= R and RV * RGT * (RGY - 1) < R and RGT <= sq.VMAX_GROUPS
        assert RGY == -(-R // (RV * RGT)) and RGT * VCH * 32 <= 256
        chunk, NS = sq.vec_plan(Lk, cfg, VCH)
        assert chunk % ((32 // cfg.vl) * cfg.vu) == 0 and NS % VCH == 0 and chunk * NS >= Lk


def test_default_takes_the_row_vector_path_for_every_qwen_call():
    # Arrange / Act / Assert: Qwen3-1.7B (G = 2, D = 128) at T = 1 to 16; G = 4 at T = 16 (64 rows)
    # and D = 64 with 32 lanes per key keep the matrix path.
    assert all(sq.uses_vec(2, T, 128, sq.DEFAULT) for T in range(1, 17))
    assert not sq.uses_vec(4, 16, 128, sq.DEFAULT)
    assert not any(sq.uses_vec(2, T, 128, sq.BASELINE) for T in range(1, 17))
    assert not sq.uses_vec(2, 1, 64, sq.Config.parse("bk32_ns16_f_t_r2_v32g4u1n32l32_rs1"))


VEC_FAULTS = [
    # the lazy reference max moves but the running sums are not rescaled
    ("const float a = exp2(m[r] - mx);       // 0 while m is -inf", "const float a = 1.0f;", "rescale"),
    # slots merged with the other slot's weight left at 1
    ("const float wo = mo == -INFINITY ? 0.0f : exp2(mo - M);", "const float wo = mo == -INFINITY ? 0.0f : 1.0f;", "merge"),
    # the score's reduction across the slot's lanes stops one step short
    ("for (ushort x = 1; x < LPK; x *= 2) d += simd_shuffle_xor(d, x);",
     "for (ushort x = 1; x < LPK / 2; x *= 2) d += simd_shuffle_xor(d, x);", "dot"),
    # every row group computes with the first group's query rows
    ("const int rr = min(r0 + r, R - 1);", "const int rr = min(r, R - 1);", "rows"),
]


@pytest.mark.parametrize("old, new, kind", VEC_FAULTS, ids=[f[2] for f in VEC_FAULTS])
@pytest.mark.parametrize("T", [1, 8, 16])
@pytest.mark.parametrize("dev", [False, True], ids=["host", "dev"])
def test_planted_vec_faults_are_caught(old, new, kind, T, dev):
    # Arrange: keys in the later part of the cache score far higher, so the reference max moves
    # inside chunks; sharp queries make the slots' maxima differ. T = 16 spreads the rows over two
    # threadgroups along grid y.
    src = sq.SOURCE_PATH.read_text()
    assert src.count(old) == 1
    lib = sq._Library(src.replace(old, new))
    q, k, v = make(1, 16, 8, T, 3000, qscale=3.0, seed=5 + T)
    k[:, :, 1500:3000 + T] *= 4
    assert sq.uses_vec(2, T, 128, sq.DEFAULT)
    # Act
    ok, d = check(q, k, v, 3000, lib=lib, dev=dev)
    # Assert
    assert not ok, d
    assert check(q, k, v, 3000, dev=dev)[0]


@pytest.mark.parametrize("T, pos, qscale, scale", EXTREME[:3], ids=["q100", "q300", "scale50"])
@pytest.mark.parametrize("dev", [False, True], ids=["host", "dev"])
def test_planted_baseline_reduce_max_from_first_chunk_is_caught(T, pos, qscale, scale, dev):
    # Arrange: as test_planted_reduce_max_from_first_chunk_is_caught, for experiment E's reduce.
    old, new = "for (int s = 0; s < NS; s += 1) M = max(M, ml[s].x);", "M = ml[0].x;"
    src = sq.SOURCE_PATH.read_text()
    assert src.count(old) == 1
    lib = sq._Library(src.replace(old, new))
    q, k, v = make(1, 16, 8, T, pos, qscale=qscale, seed=pos + T)
    # Act
    ok, d = check(q, k, v, pos, scale=scale, cfg=sq.BASELINE, lib=lib, dev=dev)
    # Assert
    assert not ok, d
    assert check(q, k, v, pos, scale=scale, cfg=sq.BASELINE, dev=dev)[0]


@pytest.mark.parametrize("old, new", [
    ("acc += accs[jj][i];", "acc += 0.0f * accs[jj][i];"),     # other threads' chunks dropped from O
    ("L += ls[jj];", "L += 0.0f * ls[jj];"),                   # ... and from the softmax sum
])
@pytest.mark.parametrize("dev", [False, True], ids=["host", "dev"])
def test_planted_parallel_reduce_faults_are_caught(old, new, dev):
    # Arrange
    src = sq.SOURCE_PATH.read_text()
    assert src.count(old) == 1
    lib = sq._Library(src.replace(old, new))
    q, k, v = make(1, 16, 8, 4, 4000, seed=12)
    assert sq.DEFAULT.rs > 1
    # Act
    ok, d = check(q, k, v, 4000, lib=lib, dev=dev)
    # Assert
    assert not ok, d
    assert check(q, k, v, 4000, dev=dev)[0]
