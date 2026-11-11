"""Small-query attention over a long fp16 KV cache on MPS, experiment E of the mps-decode thread.

For decode and speculative verification: T = 1 to 16 new query rows at absolute positions
pos .. pos + T - 1 attend to a static cache whose first pos + T entries are valid. Row i sees keys
0 .. pos + i (causal inside the new block, everything before it visible); entries past pos + T
are never read, so the rest of the cache may hold anything. Kernels are in
smallq_attention.metal and are compiled per (D, rows or row tiles, config) on first use.

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
follows Lmax rather than pos + T: for experiment E's matrix path, with the cache nearly full it
measured 3-12% over the host entry at 4,096 keys (chunk granularity; reading pos on the GPU costs
nothing measurable), and about 1.6x with Lmax twice the valid length, so size the static cache
close to the context needed.

Two kernels do the first pass; Config chooses. The row-vector path (sqa_vec, the default for up to
Config.vrows = 32 rows, i.e. every call at Qwen3-1.7B's two query heads per kv head) gives each
simdgroup one or two of the R = G * T rows and one chunk of keys; the simdgroups of a threadgroup
that hold different rows of the same chunk read the same keys at about the same time, so the
cache crosses the memory bus about once per kv head. The matrix path (sqa_pass1, experiment E's
kernel, kept for more rows and as BASELINE) has each threadgroup load a block of keys and values
once and apply it, through 8x8 simdgroup matrices, to every new row and every query head that
shares the kv head, up to 16 rows (Config.rtmax); beyond that the rows are split into groups and
the block is read once per group. Either way the key range is split into chunks and the partial
results are combined by a second small kernel. fp32 softmax and accumulation, fp16 in and out.

The wrapper raises ValueError for inputs it does not support (dtype, head dim, T > 16, last dim
not contiguous, K/V not 16-byte aligned per row, pos + T > Lmax, Hq % Hkv); `supports()` lets
callers fall back to stock SDPA.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch

SOURCE_PATH = Path(__file__).with_name("smallq_attention.metal")
MAX_T = 16
HEAD_DIMS = (64, 128)
# The kernel folds scale * log2(e) into one fp32 factor; past this magnitude that factor can overflow
# and turn a zero score into NaN (0 * inf). Real models use 1/sqrt(D), far below it.
MAX_SCALE = 1e4


def scale_ok(scale):
    """A host float within the kernel's range. A tensor is refused: reading it would synchronise."""
    if scale is None:
        return True
    if isinstance(scale, torch.Tensor):
        return False
    s = float(scale)
    return math.isfinite(s) and abs(s) <= MAX_SCALE


def _check_scale(scale, D):
    if not scale_ok(scale):
        raise ValueError(f"smallq_attention: scale must be a host float with |scale| <= {MAX_SCALE:g}")
    return 1.0 / math.sqrt(D) if scale is None else float(scale)


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
    vrows: the row-vector path (sqa_vec) takes calls with G * T <= vrows rows (0: never; at most
           VMAX_ROWS); vg: most rows per simdgroup there (rows are spread over up to 8 simdgroups
           of a threadgroup that read the same keys, then packed up to vg per simdgroup, then
           split over more threadgroups); vu: keys per slot per step;
           vns: most key chunks per (kv head, batch), each at least VMIN_CHUNK keys; vl: lanes
           per key (8, 16 or 32, with D / vl >= 4).
    rs: the reduce; 0 = one thread walks the chunks (experiment E's), rs > 0 = sqa_reduce_par with
        rs threads per output column quad splitting the chunks.
    Named bk{bk}_ns{nsmax}_{qt}_{t|q}_r{rtmax}, e.g. bk32_ns16_f_t_r2, with
    _v{vrows}g{vg}u{vu}n{vns}l{vl} appended when vrows > 0 and _rs{rs} when rs > 0, e.g.
    bk32_ns16_f_t_r2_v16g4u2n32l16_rs4."""
    bk: int = 32
    nsmax: int = 16
    qt: str = "f"
    qreg: int = 0
    rtmax: int = 2
    vrows: int = 0
    vu: int = 2
    vns: int = 64
    vl: int = 8
    vg: int = 8
    rs: int = 0

    def name(self):
        base = f"bk{self.bk}_ns{self.nsmax}_{self.qt}_{'q' if self.qreg else 't'}_r{self.rtmax}"
        if self.vrows:
            base += f"_v{self.vrows}g{self.vg}u{self.vu}n{self.vns}l{self.vl}"
        if self.rs:
            base += f"_rs{self.rs}"
        return base

    @classmethod
    def parse(cls, text):
        parts = text.split("_")
        if len(parts) < 5:
            raise ValueError(f"bad config name {text!r}")
        bk, ns, qt, qr, rt = parts[:5]
        extra = {}
        for tok in parts[5:]:
            mt = re.fullmatch(r"v(\d+)g(\d+)u(\d+)n(\d+)l(\d+)", tok)
            mr = re.fullmatch(r"rs(\d+)", tok)
            if mt is not None:
                extra.update(vrows=int(mt[1]), vg=int(mt[2]), vu=int(mt[3]), vns=int(mt[4]), vl=int(mt[5]))
            elif mr is not None:
                extra.update(rs=int(mr[1]))
            else:
                raise ValueError(f"bad config name {text!r}")
        return cls(int(bk[2:]), int(ns[2:]), qt, 1 if qr == "q" else 0, int(rt[1:]), **extra)


# The configuration of experiment E as built and measured on 2026-11-11, kept selectable so every
# comparison with it runs in the same process: 32-key sub-blocks, at most 16 chunks per kv head,
# fp32 operands, query fragments from threadgroup memory, at most 16 rows per threadgroup, no
# row-vector path. Of the alternatives then, Q in registers ran 5-10% slower at 8 rows and much
# slower at 32 (register pressure); 32 rows per threadgroup ran about 2x slower than two 16-row
# groups at T = 9 to 16; 64-key sub-blocks, 8 or 32 chunks and fp16 operands were within the noise
# or slower.
BASELINE = Config()
VMAX_ROWS = 32
VMIN_CHUNK = 64
VSG = 4          # simdgroups per threadgroup on the row-vector path (at least the row groups)
VMAX_GROUPS = 8  # most row groups in one threadgroup (at most 256 threads)
# The default since E2 (2026-11-11), chosen by paired same-process measurement against BASELINE on an
# M1 Max (scratch tune runs, 28 layers' caches, Qwen3-1.7B shapes, 4,096 and 16,384 keys, T 1 to 16):
# the row-vector path for every call up to 32 rows (Qwen3-1.7B up to T = 16), 8 lanes per key, rows
# spread one per simdgroup over up to 8 simdgroups and then two per simdgroup, 2 keys per slot per
# step, at most 16 chunks per kv head and row group, and the parallel reduce. Calls with more rows
# (more query heads per kv head) keep the matrix path with BASELINE's settings. Measured against
# the alternatives: 16 or 32 lanes per key and 3 or more rows per simdgroup run out of registers
# (2x to 5x slower); 1 or 4 keys per slot per step, 8, 12, 24, 48 or 64 chunks, 2 or 8 simdgroups
# per threadgroup, and loading the next step's keys before this step's arithmetic were slower
# somewhere in T = 1 to 16 and nowhere clearly faster.
DEFAULT = Config(vrows=32, vg=2, vu=2, vns=16, vl=8, rs=4)


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

    def vec(self, D, R, cfg, dev):
        name = "sqa_vec_dev" if dev else "sqa_vec"
        host = f"{name}_d{D}_r{R}_u{cfg.vu}_l{cfg.vl}"
        pos = "device const long*, " if dev else "constant long&, "
        tail = "constant float&, constant long&, constant long&, " if dev else "constant float&, "
        inst = (f"template [[host_name(\"{host}\")]] kernel void {name}<{D}, {R}, {cfg.vu}, {cfg.vl}>("
                "device const half*, device const half*, device const half*, device float*, device float*, "
                + "constant long&, " * 11 + pos + "constant long&, " * 4
                + tail + "uint3, uint3, uint, uint, uint);")
        return self._compile((name, D, R, cfg.vu, cfg.vl), host, inst)

    def reduce(self, D, rs=0):
        if rs == 0:
            host = f"sqa_reduce_d{D}"
            inst = (f"template [[host_name(\"{host}\")]] kernel void sqa_reduce<{D}>(device const float*, "
                    "device const float*, device half*, constant long&, uint, uint);")
        else:
            host = f"sqa_reduce_par_d{D}_rs{rs}"
            inst = (f"template [[host_name(\"{host}\")]] kernel void sqa_reduce_par<{D}, {rs}>(device const float*, "
                    "device const float*, device half*, constant long&, uint, uint);")
        return self._compile(("reduce", D, rs), host, inst)


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


def vec_groups(R, cfg):
    """(RV, RGT, RGY, VCH) on the row-vector path: rows per simdgroup; row groups per threadgroup
    and threadgroups along grid y per kv head (together at least ceil(R / RV) groups); and key
    chunks per threadgroup (the threadgroup holds RGT * VCH simdgroups)."""
    # Rows are spread over up to VMAX_GROUPS simdgroups of one threadgroup first (one row each
    # measured fastest up to 8 rows), then packed up to vg rows per simdgroup.
    RV = max(1, min(cfg.vg, -(-R // VMAX_GROUPS)))
    RG = -(-R // RV)
    RGT = -(-RG // -(-RG // VMAX_GROUPS))
    RGY = -(-R // (RV * RGT))          # the kernel's own formula
    return RV, RGT, RGY, max(1, VSG // RGT)


def vec_plan(Lk, cfg, vch=VSG):
    """(chunk, NS) on the row-vector path: keys per simdgroup chunk (a multiple of keys per step, at
    least VMIN_CHUNK) and the number of chunks, rounded up to whole threadgroups of vch chunks;
    trailing chunks may be empty."""
    step = (32 // cfg.vl) * cfg.vu
    blocks = -(-Lk // step)
    per = -(-blocks // min(blocks, cfg.vns))
    chunk = max(per * step, -(-VMIN_CHUNK // step) * step)
    ns = -(-Lk // chunk)
    return chunk, -(-ns // vch) * vch


def uses_vec(G, T, D, cfg):
    """True when a call with G * T rows and head dim D takes the row-vector path under cfg."""
    R = G * T
    return (0 < R <= min(cfg.vrows, VMAX_ROWS) and cfg.vl in (8, 16, 32) and D // cfg.vl >= 4
            and cfg.vg >= 1)


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


def _reduce(lib, D, cfg, part_o, part_ml, out, NS, rows):
    nt = (D // 4) * max(cfg.rs, 1)
    lib.reduce(D, cfg.rs)(part_o, part_ml, out, NS, threads=rows * nt, group_size=nt)


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
    scale = _check_scale(scale, D)
    pstride = pos.stride(0) if pos.shape[0] > 1 else 0
    strides = (q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
               v.stride(0), v.stride(1), v.stride(2))
    if uses_vec(G, T, D, cfg):
        RV, RGT, RGY, VCH = vec_groups(G * T, cfg)
        chunk, NS = vec_plan(Lmax, cfg, VCH)
        part_o = torch.empty((B, Hq, T, NS, D), dtype=torch.float32, device=q.device)
        part_ml = torch.empty((B, Hq, T, NS, 2), dtype=torch.float32, device=q.device)
        lib.vec(D, RV, cfg, True)(q, k, v, part_o, part_ml, *strides,
                                  T, G, pos, chunk, NS, RGT, int(_mask_shift), scale, pstride, Lmax,
                                  threads=(NS * 32 * RGT, Hkv * RGY, B), group_size=(32 * RGT * VCH, 1, 1))
    else:
        RT, RG = _row_tiles(G * T, D, cfg)
        chunk, NS = split_plan(Lmax, cfg)
        part_o = torch.empty((B, Hq, T, NS, D), dtype=torch.float32, device=q.device)
        part_ml = torch.empty((B, Hq, T, NS, 2), dtype=torch.float32, device=q.device)
        nt = 32 * (D // 32)
        lib.pass1_dev(D, RT, cfg)(q, k, v, part_o, part_ml, *strides,
                                  T, G, pos, chunk, NS, RG, int(_mask_shift), scale, pstride, Lmax,
                                  threads=(NS * nt, Hkv * RG, B), group_size=(nt, 1, 1))
    out = torch.empty((B, Hq, T, D), dtype=torch.float16, device=q.device)
    _reduce(lib, D, cfg, part_o, part_ml, out, NS, B * Hq * T)
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
    scale = _check_scale(scale, D)
    Lk = pos + T
    strides = (q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
               v.stride(0), v.stride(1), v.stride(2))
    if uses_vec(G, T, D, cfg):
        RV, RGT, RGY, VCH = vec_groups(G * T, cfg)
        chunk, NS = vec_plan(Lk, cfg, VCH)
        part_o = torch.empty((B, Hq, T, NS, D), dtype=torch.float32, device=q.device)
        part_ml = torch.empty((B, Hq, T, NS, 2), dtype=torch.float32, device=q.device)
        lib.vec(D, RV, cfg, False)(q, k, v, part_o, part_ml, *strides,
                                   T, G, pos, chunk, NS, RGT, int(_mask_shift), scale,
                                   threads=(NS * 32 * RGT, Hkv * RGY, B), group_size=(32 * RGT * VCH, 1, 1))
    else:
        RT, RG = _row_tiles(G * T, D, cfg)
        chunk, NS = split_plan(Lk, cfg)
        part_o = torch.empty((B, Hq, T, NS, D), dtype=torch.float32, device=q.device)
        part_ml = torch.empty((B, Hq, T, NS, 2), dtype=torch.float32, device=q.device)
        nt = 32 * (D // 32)
        lib.pass1(D, RT, cfg)(q, k, v, part_o, part_ml, *strides,
                              T, G, pos, chunk, NS, RG, int(_mask_shift), scale,
                              threads=(NS * nt, Hkv * RG, B), group_size=(nt, 1, 1))
    out = torch.empty((B, Hq, T, D), dtype=torch.float16, device=q.device)
    _reduce(lib, D, cfg, part_o, part_ml, out, NS, B * Hq * T)
    return out
