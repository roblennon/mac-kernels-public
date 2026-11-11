"""Small-query attention over a long fp16 KV cache on MPS, experiment E of the mps-decode thread.

For decode and speculative verification: T = 1 to 16 new query rows at absolute positions
pos .. pos + T - 1 attend to a static cache whose first pos + T entries are valid. Row i sees keys
0 .. pos + i (causal inside the new block, everything before it visible); entries past pos + T
are never read, so the rest of the cache may hold anything. Kernels are in
smallq_attention.metal and are compiled per (D, row tiles, config) on first use.

    from bench.smallq_attention import smallq_attention, supports
    out = smallq_attention(q, k, v, pos)          # q [B, Hq, T, D], k and v [B, Hkv, Lmax, D]
    out = smallq_attention(q, k, v, pos, scale=0.1, config=Config.parse("bk64_ns32_f_t_r2"))

pos is one Python int shared by every batch element (a tensor is accepted and read with .item(),
which synchronises). The grid depends on pos + T, so a changing pos is a host value, not a
graph input.

    out = smallq_attention_dev(q, k, v, pos_t)    # pos_t: int64 device tensor, [1] or [B]

is the device-position entry: the kernel reads pos on the GPU (one per batch element allowed) and
the key chunks are sized from Lmax = k.shape[2], so nothing synchronises and the launch does not
depend on pos. That is the entry bench/smallq_hf.py uses inside a compiled model. Its cost
follows Lmax rather than pos + T: with the cache nearly full it measured 3-12% over the host entry
at 4,096 keys (chunk granularity; reading pos on the GPU costs nothing measurable), and about
1.6x with Lmax twice the valid length, so size the static cache close to the context needed.

Each threadgroup loads a block of keys and values once and applies it to every new row and every
query head that shares the kv head, up to 16 rows by default (Config.rtmax; Qwen3-1.7B's two
query heads per kv head at T = 8); beyond that the rows are split into groups and the block is
read once per group. The key range is split across threadgroups and the partial results are
combined by a second small kernel. fp32 softmax and accumulation, fp16 in and out.

The wrapper raises ValueError for inputs it does not support (dtype, head dim, T > 16, last dim
not contiguous, K/V not 16-byte aligned per row, pos + T > Lmax, Hq % Hkv); `supports()` lets
callers fall back to stock SDPA.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

SOURCE_PATH = Path(__file__).with_name("smallq_attention.metal")
MAX_T = 16
HEAD_DIMS = (64, 128)


@dataclass(frozen=True)
class Config:
    """bk: keys per sub-block inside a threadgroup (multiple of 32).
    nsmax: most key chunks (threadgroups along the cache) per (kv head, row group, batch); the
           chunk length is the smallest multiple of bk that needs no more chunks than this.
    qt: operand type of the score product, "f" (fp32) or "h" (fp16 operands, fp32 accumulation).
    qreg: 0 ("t") = simdgroups split the keys and read query fragments from threadgroup memory;
          1 ("q") = each simdgroup keeps one query row tile's fragments in registers.
    rtmax: most 8-row query tiles per threadgroup (1, 2 or 4); more rows are split into row
           groups, and each row group reads the cache block again (mostly from the GPU's L2).
    Named bk{bk}_ns{nsmax}_{qt}_{t|q}_r{rtmax}, e.g. bk32_ns16_f_t_r2."""
    bk: int = 32
    nsmax: int = 16
    qt: str = "f"
    qreg: int = 0
    rtmax: int = 2

    def name(self):
        return f"bk{self.bk}_ns{self.nsmax}_{self.qt}_{'q' if self.qreg else 't'}_r{self.rtmax}"

    @classmethod
    def parse(cls, text):
        bk, ns, qt, qr, rt = text.split("_")
        return cls(int(bk[2:]), int(ns[2:]), qt, 1 if qr == "q" else 0, int(rt[1:]))


# The default, chosen by measurement on an M1 Max (scratch tune runs, 28 layers' caches,
# Qwen3-1.7B shapes, 256 to 16,384 keys, T 1 to 16): 32-key sub-blocks, at most 16 chunks per
# kv head, fp32 operands, query fragments from threadgroup memory, at most 16 rows per
# threadgroup. Of the alternatives, Q in registers ran 5-10% slower at 8 rows and much slower at
# 32 (register pressure); 32 rows per threadgroup ran about 2x slower than two 16-row groups at
# T = 9 to 16; 64-key sub-blocks, 8 or 32 chunks and fp16 operands were within the noise or slower.
DEFAULT = Config()


def pick_config(Lk, T):
    """The configuration used when none is given, for Lk = pos + T valid keys and T new rows."""
    return DEFAULT


class _Library:
    def __init__(self, source=None):
        self.source = SOURCE_PATH.read_text() if source is None else source
        self.cache = {}

    def _compile(self, key, host, inst):
        fn = self.cache.get(key)
        if fn is None:
            lib = torch.mps.compile_shader(self.source + "\n" + inst + "\n")
            fn = self.cache[key] = getattr(lib, host)
        return fn

    def pass1(self, D, RT, cfg):
        qt = {"f": "float", "h": "half"}[cfg.qt]
        host = f"sqa_pass1_{cfg.qt}_d{D}_rt{RT}_bk{cfg.bk}_q{cfg.qreg}"
        qreg = "true" if cfg.qreg else "false"
        inst = (f"template [[host_name(\"{host}\")]] kernel void sqa_pass1<{qt}, {D}, {RT}, {cfg.bk}, {qreg}>("
                "device const half*, device const half*, device const half*, device float*, device float*, "
                + "constant long&, " * 16 + "constant float&, uint3, uint3, uint, uint, uint);")
        return self._compile(("pass1", D, RT, cfg.bk, cfg.qt, cfg.qreg), host, inst)

    def pass1_dev(self, D, RT, cfg):
        qt = {"f": "float", "h": "half"}[cfg.qt]
        host = f"sqa_pass1_dev_{cfg.qt}_d{D}_rt{RT}_bk{cfg.bk}_q{cfg.qreg}"
        qreg = "true" if cfg.qreg else "false"
        inst = (f"template [[host_name(\"{host}\")]] kernel void sqa_pass1_dev<{qt}, {D}, {RT}, {cfg.bk}, {qreg}>("
                "device const half*, device const half*, device const half*, device float*, device float*, "
                + "constant long&, " * 11 + "device const long*, " + "constant long&, " * 4
                + "constant float&, constant long&, constant long&, uint3, uint3, uint, uint, uint);")
        return self._compile(("pass1_dev", D, RT, cfg.bk, cfg.qt, cfg.qreg), host, inst)

    def reduce(self, D):
        host = f"sqa_reduce_d{D}"
        inst = (f"template [[host_name(\"{host}\")]] kernel void sqa_reduce<{D}>(device const float*, "
                "device const float*, device half*, constant long&, uint, uint);")
        return self._compile(("reduce", D), host, inst)


_LIB = None


def library():
    global _LIB
    if _LIB is None:
        _LIB = _Library()
    return _LIB


def _aligned(t):
    return t.storage_offset() % 8 == 0 and all(s % 8 == 0 for s in t.stride()[:-1])


def supports(q, k, v, pos):
    """True when smallq_attention can run these tensors as given."""
    if not supports_shapes(q, k, v):
        return False
    try:
        p = int(pos)
    except (TypeError, ValueError, RuntimeError):
        return False
    return 0 <= p and p + q.shape[2] <= k.shape[2]


def split_plan(Lk, cfg):
    """(chunk, NS): keys per threadgroup chunk (a multiple of bk) and the number of chunks."""
    blocks = -(-Lk // cfg.bk)
    per = -(-blocks // min(blocks, cfg.nsmax))
    chunk = per * cfg.bk
    return chunk, -(-Lk // chunk)


def supports_shapes(q, k, v):
    """True when the device-position path can run these tensors (everything supports() checks
    except pos, which that path reads on the GPU). Depends only on shapes, strides, dtypes and
    devices, so under torch.compile the answer is fixed per graph."""
    if q.device.type != "mps" or k.device != q.device or v.device != q.device:
        return False
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        return False
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or k.shape != v.shape:
        return False
    B, Hq, T, D = q.shape
    Bk, Hkv, Lmax, Dk = k.shape
    if Bk != B or Dk != D or D not in HEAD_DIMS or not 1 <= T <= min(MAX_T, Lmax):
        return False
    if Hkv < 1 or Hq % Hkv:
        return False
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return False
    return _aligned(k) and _aligned(v)


def _row_tiles(R, D, cfg):
    """(RT, RG): 8-row tiles per threadgroup (a power of two dividing D / 32) and row groups."""
    RT = 1
    while RT < min(-(-R // 8), D // 32, cfg.rtmax):
        RT *= 2
    return RT, -(-R // (8 * RT))


def smallq_attention_dev(q, k, v, pos, scale=None, config=None, lib=None, _mask_shift=0):
    """As smallq_attention, with pos a device tensor read by the kernel: int64 on q's device, one
    element (shared by the batch) or B elements, any stride; transformers' position_ids[:, 0].
    The key chunks are sized from Lmax = k.shape[2], so nothing here depends on pos's value and
    nothing synchronises. pos is clamped on the GPU to 0 .. Lmax - T (out-of-range values give a
    wrong answer, never an out-of-bounds read); the caller guarantees 0 <= pos <= Lmax - T."""
    if not supports_shapes(q, k, v):
        raise ValueError("smallq_attention_dev: unsupported input (see supports_shapes())")
    if (not isinstance(pos, torch.Tensor) or pos.dtype != torch.int64 or pos.device != q.device
            or pos.dim() != 1 or pos.shape[0] not in (1, q.shape[0])):
        raise ValueError("smallq_attention_dev: pos must be an int64 [1] or [B] tensor on q's device")
    B, Hq, T, D = q.shape
    Hkv, Lmax = k.shape[1], k.shape[2]
    G = Hq // Hkv
    cfg = config or pick_config(Lmax, T)
    lib = lib or library()
    scale = 1.0 / math.sqrt(D) if scale is None else float(scale)
    RT, RG = _row_tiles(G * T, D, cfg)
    chunk, NS = split_plan(Lmax, cfg)
    pstride = pos.stride(0) if pos.shape[0] > 1 else 0
    part_o = torch.empty((B, Hq, T, NS, D), dtype=torch.float32, device=q.device)
    part_ml = torch.empty((B, Hq, T, NS, 2), dtype=torch.float32, device=q.device)
    nt = 32 * (D // 32)
    lib.pass1_dev(D, RT, cfg)(q, k, v, part_o, part_ml,
                              q.stride(0), q.stride(1), q.stride(2),
                              k.stride(0), k.stride(1), k.stride(2),
                              v.stride(0), v.stride(1), v.stride(2),
                              T, G, pos, chunk, NS, RG, int(_mask_shift), scale, pstride, Lmax,
                              threads=(NS * nt, Hkv * RG, B), group_size=(nt, 1, 1))
    out = torch.empty((B, Hq, T, D), dtype=torch.float16, device=q.device)
    lib.reduce(D)(part_o, part_ml, out, NS, threads=B * Hq * T * (D // 4), group_size=D // 4)
    return out


def smallq_attention(q, k, v, pos, scale=None, config=None, lib=None, _mask_shift=0):
    """Attention of q's T rows (positions pos .. pos + T - 1) over k/v[:, :, :pos + T]; see module doc.
    _mask_shift moves every row's causal limit (test hook for a planted fault; leave at 0)."""
    if not supports(q, k, v, pos):
        raise ValueError("smallq_attention: unsupported input (see supports())")
    pos = int(pos)
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    cfg = config or pick_config(pos + T, T)
    lib = lib or library()
    scale = 1.0 / math.sqrt(D) if scale is None else float(scale)
    RT, RG = _row_tiles(G * T, D, cfg)
    Lk = pos + T
    chunk, NS = split_plan(Lk, cfg)
    part_o = torch.empty((B, Hq, T, NS, D), dtype=torch.float32, device=q.device)
    part_ml = torch.empty((B, Hq, T, NS, 2), dtype=torch.float32, device=q.device)
    nt = 32 * (D // 32)
    lib.pass1(D, RT, cfg)(q, k, v, part_o, part_ml,
                          q.stride(0), q.stride(1), q.stride(2),
                          k.stride(0), k.stride(1), k.stride(2),
                          v.stride(0), v.stride(1), v.stride(2),
                          T, G, pos, chunk, NS, RG, int(_mask_shift), scale,
                          threads=(NS * nt, Hkv * RG, B), group_size=(nt, 1, 1))
    out = torch.empty((B, Hq, T, D), dtype=torch.float16, device=q.device)
    rows = B * Hq * T
    lib.reduce(D)(part_o, part_ml, out, NS, threads=rows * (D // 4), group_size=D // 4)
    return out
