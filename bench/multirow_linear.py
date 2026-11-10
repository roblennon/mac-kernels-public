"""Multi-row linear for 1 to 32 input rows on MPS, experiment B of the mps-decode thread.

y = x @ W.t() (+ bias) with W in nn.Linear's [N, K] layout, unchanged. Kernels are
in multirow_linear.metal and are compiled per (dtype, M, variant) on first use.

    from bench.multirow_linear import multirow_linear, Config
    y = multirow_linear(x2d, linear.weight, linear.bias)       # tuned config for the shape
    y = multirow_linear(x2d, w, None, config=Config("dev", 2, 1, 4))

The wrapper raises ValueError for inputs it does not support (K % 8 != 0, not
contiguous, misaligned, wrong dtype or device); `supports()` lets callers fall back.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

SOURCE_PATH = Path(__file__).with_name("multirow_linear.metal")
CODEC = {torch.float16: "F16", torch.bfloat16: "BF16"}
MAX_M = 32


@dataclass(frozen=True)
class Config:
    """xsrc "dev": per-lane dot products, x read from device memory (cache-resident).
    xsrc "tg":  per-lane dot products, x staged in threadgroup memory.
    xsrc "mma": simdgroup 8x8 matrix multiply-accumulate on tiles staged in threadgroup memory.
    xsrc "mmr": simdgroup 8x8 matrix multiply-accumulate fed from device memory into registers.
    R: weight rows per simdgroup (dev, tg) or 8-row weight tiles per simdgroup (mma, mmr).
    U: K chunks of 256 elements per lane per step (dev, tg) or K step of 32*U elements (mma, mmr).
    S: simdgroups per threadgroup.  P: K splits (mma, mmr; P > 1 adds a reduce kernel).
    xsrc "mmh": register-fed 8x8 fragments in a k order that makes each weight fragment one
                32-bit word; S simdgroups share the same R*8 weight rows and split K inside the
                threadgroup (round 2, aimed at 9 to 16 rows).
    xsrc "mmp": mmh with x first repacked into fragment order by a small extra kernel.
    xsrc "mmt": weight staged through threadgroup memory (16-bit, double-buffered); S simdgroups
                own different weight rows; x fragments from device memory.
    F: 1 = load the next K step into registers before computing this one (mmr, mmh, mmp).
    H: 1 = half-precision fragment operands with fp32 accumulation (mmr, mmh, mmp, mmt; fp16 only)."""
    xsrc: str = "dev"
    R: int = 2
    U: int = 1
    S: int = 4
    P: int = 1
    F: int = 0
    H: int = 0

    def name(self):
        return (f"{self.xsrc}_r{self.R}_u{self.U}_s{self.S}" + (f"_p{self.P}" if self.P != 1 else "")
                + (f"_f{self.F}" if self.F else "") + (f"_h{self.H}" if self.H else ""))

    @classmethod
    def parse(cls, text):
        parts = text.split("_")
        vals = {p[0]: int(p[1:]) for p in parts[1:]}
        return cls(parts[0], vals["r"], vals["u"], vals["s"], vals.get("p", 1), vals.get("f", 0), vals.get("h", 0))

    def rows_per_group(self):
        if self.xsrc in ("mmh", "mmp"):
            return self.R * 8
        return self.S * self.R * (8 if self.xsrc in ("mma", "mmr", "mmt") else 1)

    def threadgroup_bytes(self, M):
        mp = -(-M // 8) * 8
        if self.xsrc in ("mmh", "mmp"):
            return self.S * self.R * 8 * mp * 4 if self.S > 1 else 0
        if self.xsrc == "mmt":
            return 2 * self.S * self.R * 8 * (4 * self.U + 1) * 16
        if self.xsrc == "mma":
            ld = 32 * self.U + 4
            return (self.rows_per_group() + -(-M // 8) * 8) * ld * 4
        return M * 32 * self.U * 16 if self.xsrc == "tg" else 0

    def fits(self, M):
        return self.threadgroup_bytes(M) <= 32768


LINEAR = "linear"  # a TUNED entry meaning: this kernel is not faster here, use F.linear

# The variant per shape and row count. Rows 2 to 16: one config per dtype, shape and row band
# (2-3, 4-8, 9-16), chosen from a small fixed set (U = 1, no prefetch, no K split, fp32 fragment
# operands; R = 2 and S = 4 or 8 at 9 to 16 rows) by its time pooled over every cell of the band in
# the round-2 sweeps (docs/review/2026-11-10-experiments/multirow/round2/fixes/select_table.py),
# then re-timed against F.linear with both noise arms in two or three processes. "linear" would
# mark a cell where the kernel was slower than F.linear beyond the noise in every process; none
# is at present (round2/fixes/summarize_fixes.py). Rows 1, 24 and 32: round 1
# (docs/review/2026-11-10-experiments/multirow/tune-*.json).
# {dtype name: {(K, N): {M: config name}}}. Shapes not listed use default_config(); row counts
# not listed use the nearest tuned row count with the same number of 8-row tiles.
TUNED: dict[str, dict[tuple[int, int], dict[int, str]]] = {
    "bfloat16": {
        (2048, 1024): {1: "dev_r4_u1_s8", 2: "dev_r2_u1_s4", 3: "dev_r2_u1_s4", 4: "mmh_r1_u1_s4", 5: "mmh_r1_u1_s4", 6: "mmh_r1_u1_s4", 7: "mmh_r1_u1_s4", 8: "mmh_r1_u1_s4", 9: "mmh_r2_u1_s8", 10: "mmh_r2_u1_s8", 11: "mmh_r2_u1_s8", 12: "mmh_r2_u1_s8", 13: "mmh_r2_u1_s8", 14: "mmh_r2_u1_s8", 15: "mmh_r2_u1_s8", 16: "mmh_r2_u1_s8", 24: "mma_r1_u1_s4_p4", 32: "mma_r1_u1_s4_p4"},
        (2048, 2048): {1: "tg_r4_u1_s8", 2: "dev_r2_u1_s8", 3: "dev_r2_u1_s8", 4: "mmh_r2_u1_s4", 5: "mmh_r2_u1_s4", 6: "mmh_r2_u1_s4", 7: "mmh_r2_u1_s4", 8: "mmh_r2_u1_s4", 9: "mmh_r2_u1_s4", 10: "mmh_r2_u1_s4", 11: "mmh_r2_u1_s4", 12: "mmh_r2_u1_s4", 13: "mmh_r2_u1_s4", 14: "mmh_r2_u1_s4", 15: "mmh_r2_u1_s4", 16: "mmh_r2_u1_s4", 24: "mma_r2_u1_s4_p4", 32: "mma_r2_u1_s4_p4"},
        (2048, 6144): {1: "tg_r1_u2_s4", 2: "dev_r2_u1_s4", 3: "dev_r2_u1_s4", 4: "mmh_r1_u1_s4", 5: "mmh_r1_u1_s4", 6: "mmh_r1_u1_s4", 7: "mmh_r1_u1_s4", 8: "mmh_r1_u1_s4", 9: "mmh_r2_u1_s4", 10: "mmh_r2_u1_s4", 11: "mmh_r2_u1_s4", 12: "mmh_r2_u1_s4", 13: "mmh_r2_u1_s4", 14: "mmh_r2_u1_s4", 15: "mmh_r2_u1_s4", 16: "mmh_r2_u1_s4", 24: "mma_r1_u1_s4", 32: "mma_r1_u1_s4"},
        (2048, 151936): {1: "tg_r2_u2_s8", 2: "mmh_r1_u1_s8", 3: "mmh_r1_u1_s8", 4: "mmh_r2_u1_s8", 5: "mmh_r2_u1_s8", 6: "mmh_r2_u1_s8", 7: "mmh_r2_u1_s8", 8: "mmh_r2_u1_s8", 9: "mmh_r2_u1_s4", 10: "mmh_r2_u1_s4", 11: "mmh_r2_u1_s4", 12: "mmh_r2_u1_s4", 13: "mmh_r2_u1_s4", 14: "mmh_r2_u1_s4", 15: "mmh_r2_u1_s4", 16: "mmh_r2_u1_s4", 24: "mma_r1_u1_s4", 32: "mma_r1_u1_s4"},
        (2560, 9728): {1: "tg_r1_u2_s4", 2: "mmh_r1_u1_s8", 3: "mmh_r1_u1_s8", 4: "mmh_r2_u1_s4", 5: "mmh_r2_u1_s4", 6: "mmh_r2_u1_s4", 7: "mmh_r2_u1_s4", 8: "mmh_r2_u1_s4", 9: "mmh_r2_u1_s8", 10: "mmh_r2_u1_s8", 11: "mmh_r2_u1_s8", 12: "mmh_r2_u1_s8", 13: "mmh_r2_u1_s8", 14: "mmh_r2_u1_s8", 15: "mmh_r2_u1_s8", 16: "mmh_r2_u1_s8", 24: "mma_r1_u1_s4_p4", 32: "mma_r1_u1_s4_p4"},
        (4096, 11008): {1: "tg_r1_u2_s8", 2: "mmh_r2_u1_s4", 3: "mmh_r2_u1_s4", 4: "mmh_r1_u1_s8", 5: "mmh_r1_u1_s8", 6: "mmh_r1_u1_s8", 7: "mmh_r1_u1_s8", 8: "mmh_r1_u1_s8", 9: "mmh_r2_u1_s4", 10: "mmh_r2_u1_s4", 11: "mmh_r2_u1_s4", 12: "mmh_r2_u1_s4", 13: "mmh_r2_u1_s4", 14: "mmh_r2_u1_s4", 15: "mmh_r2_u1_s4", 16: "mmh_r2_u1_s4", 24: "mma_r1_u1_s4_p4", 32: "mma_r1_u1_s4"},
        (6144, 2048): {1: "dev_r1_u2_s4", 2: "mmh_r1_u1_s8", 3: "mmh_r1_u1_s8", 4: "mmh_r1_u1_s8", 5: "mmh_r1_u1_s8", 6: "mmh_r1_u1_s8", 7: "mmh_r1_u1_s8", 8: "mmh_r1_u1_s8", 9: "mmh_r2_u1_s4", 10: "mmh_r2_u1_s4", 11: "mmh_r2_u1_s4", 12: "mmh_r2_u1_s4", 13: "mmh_r2_u1_s4", 14: "mmh_r2_u1_s4", 15: "mmh_r2_u1_s4", 16: "mmh_r2_u1_s4", 24: "mma_r1_u1_s4_p4", 32: "mma_r2_u1_s4_p4"},
    },
    "float16": {
        (2048, 1024): {1: "tg_r1_u2_s4", 2: "dev_r2_u1_s8", 3: "dev_r2_u1_s8", 4: "mmh_r1_u1_s8", 5: "mmh_r1_u1_s8", 6: "mmh_r1_u1_s8", 7: "mmh_r1_u1_s8", 8: "mmh_r1_u1_s8", 9: "mmh_r2_u1_s8", 10: "mmh_r2_u1_s8", 11: "mmh_r2_u1_s8", 12: "mmh_r2_u1_s8", 13: "mmh_r2_u1_s8", 14: "mmh_r2_u1_s8", 15: "mmh_r2_u1_s8", 16: "mmh_r2_u1_s8", 24: "mma_r1_u1_s4_p4", 32: "mma_r1_u1_s4_p4"},
        (2048, 2048): {1: "tg_r2_u1_s4", 2: "dev_r2_u1_s4", 3: "dev_r2_u1_s4", 4: "mmh_r1_u1_s4", 5: "mmh_r1_u1_s4", 6: "mmh_r1_u1_s4", 7: "mmh_r1_u1_s4", 8: "mmh_r1_u1_s4", 9: "mmh_r2_u1_s4", 10: "mmh_r2_u1_s4", 11: "mmh_r2_u1_s4", 12: "mmh_r2_u1_s4", 13: "mmh_r2_u1_s4", 14: "mmh_r2_u1_s4", 15: "mmh_r2_u1_s4", 16: "mmh_r2_u1_s4", 24: "mma_r1_u1_s4_p4", 32: "mma_r2_u1_s4_p4"},
        (2048, 6144): {1: "tg_r1_u2_s4", 2: "mmh_r2_u1_s8", 3: "mmh_r2_u1_s8", 4: "mmh_r2_u1_s4", 5: "mmh_r2_u1_s4", 6: "mmh_r2_u1_s4", 7: "mmh_r2_u1_s4", 8: "mmh_r2_u1_s4", 9: "mmh_r2_u1_s4", 10: "mmh_r2_u1_s4", 11: "mmh_r2_u1_s4", 12: "mmh_r2_u1_s4", 13: "mmh_r2_u1_s4", 14: "mmh_r2_u1_s4", 15: "mmh_r2_u1_s4", 16: "mmh_r2_u1_s4", 24: "mma_r1_u1_s4", 32: "mma_r1_u1_s4"},
        (2048, 151936): {1: "dev_r4_u1_s8", 2: "mmh_r1_u1_s8", 3: "mmh_r1_u1_s8", 4: "mmh_r2_u1_s8", 5: "mmh_r2_u1_s8", 6: "mmh_r2_u1_s8", 7: "mmh_r2_u1_s8", 8: "mmh_r2_u1_s8", 9: "mmh_r2_u1_s8", 10: "mmh_r2_u1_s8", 11: "mmh_r2_u1_s8", 12: "mmh_r2_u1_s8", 13: "mmh_r2_u1_s8", 14: "mmh_r2_u1_s8", 15: "mmh_r2_u1_s8", 16: "mmh_r2_u1_s8", 24: "mma_r1_u1_s4", 32: "mma_r1_u1_s4"},
        (2560, 9728): {1: "dev_r1_u1_s8", 2: "dev_r2_u1_s8", 3: "dev_r2_u1_s8", 4: "mmh_r2_u1_s8", 5: "mmh_r2_u1_s8", 6: "mmh_r2_u1_s8", 7: "mmh_r2_u1_s8", 8: "mmh_r2_u1_s8", 9: "mmh_r2_u1_s4", 10: "mmh_r2_u1_s4", 11: "mmh_r2_u1_s4", 12: "mmh_r2_u1_s4", 13: "mmh_r2_u1_s4", 14: "mmh_r2_u1_s4", 15: "mmh_r2_u1_s4", 16: "mmh_r2_u1_s4", 24: "mma_r1_u1_s4", 32: "mma_r1_u1_s4"},
        (4096, 11008): {1: "dev_r2_u1_s4", 2: "mmh_r1_u1_s8", 3: "mmh_r1_u1_s8", 4: "mmh_r1_u1_s8", 5: "mmh_r1_u1_s8", 6: "mmh_r1_u1_s8", 7: "mmh_r1_u1_s8", 8: "mmh_r1_u1_s8", 9: "mmh_r2_u1_s8", 10: "mmh_r2_u1_s8", 11: "mmh_r2_u1_s8", 12: "mmh_r2_u1_s8", 13: "mmh_r2_u1_s8", 14: "mmh_r2_u1_s8", 15: "mmh_r2_u1_s8", 16: "mmh_r2_u1_s8", 24: "mma_r1_u1_s4_p4", 32: "mma_r1_u1_s4"},
        (6144, 2048): {1: "dev_r4_u1_s4", 2: "dev_r2_u1_s8", 3: "dev_r2_u1_s8", 4: "mmh_r1_u1_s8", 5: "mmh_r1_u1_s8", 6: "mmh_r1_u1_s8", 7: "mmh_r1_u1_s8", 8: "mmh_r1_u1_s8", 9: "mmh_r2_u1_s4", 10: "mmh_r2_u1_s4", 11: "mmh_r2_u1_s4", 12: "mmh_r2_u1_s4", 13: "mmh_r2_u1_s4", 14: "mmh_r2_u1_s4", 15: "mmh_r2_u1_s4", 16: "mmh_r2_u1_s4", 24: "mma_r1_u1_s4_p4", 32: "mma_r2_u1_s4_p4"},
    },
}


def default_config(M, K, N):
    if M <= 4:
        return Config("dev", 2, 1, 4)
    if M <= 8:
        return Config("mmr", 1, 2, 4, 1, 1)
    if M <= 16:
        return Config("mmh", 2, 1, 4)
    return Config("mma", 1, 1, 4, 4)



def pick_config(M, K, N, dtype=torch.float16):
    """The tuned config for this shape and row count, or None where F.linear is the faster choice.
    Row counts not in the table use the nearest tuned row count with the same number of 8-row
    tiles when there is one (a 16-row choice is not reused at 17 rows, where it needs a third tile)."""
    table = TUNED.get(str(dtype).split(".")[-1], {}).get((K, N))
    if table:
        same = [m for m in table if -(-m // 8) == -(-M // 8)] or list(table)
        best = min(same, key=lambda m: (abs(m - M), -m))
        return None if table[best] == LINEAR else Config.parse(table[best])
    return default_config(M, K, N)


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

    def kernel(self, dtype, M, cfg):
        codec = CODEC[dtype]
        common = ("device const uint4*, device const uint4*, device const ushort*, device ushort*, "
                  "constant long&, constant long&, constant long&")
        if cfg.xsrc in ("mmh", "mmp", "mmt"):
            if cfg.H and codec == "F16":
                codec = "F16H"
            sig = ("(device const uint4*, device const uint4*, device const ushort*, device ushort*, "
                   "constant long&, constant long&, constant long&, device float*, constant long&, "
                   "constant long&, uint2, uint, uint);")
            if cfg.xsrc == "mmt":
                key = (dtype, M, "mmt", cfg.R, cfg.U, cfg.S, cfg.H)
                host = f"mrl_mmt_{codec}_m{M}_r{cfg.R}_u{cfg.U}_s{cfg.S}"
                inst = (f"template [[host_name(\"{host}\")]] kernel void mrl_mmt<{codec}, {M}, {cfg.R}, {cfg.U}, "
                        f"{cfg.S}>" + sig)
            else:
                pk = cfg.xsrc == "mmp"
                key = (dtype, M, cfg.xsrc, cfg.R, cfg.U, cfg.S, cfg.F, cfg.H)
                host = f"mrl_{cfg.xsrc}_{codec}_m{M}_r{cfg.R}_u{cfg.U}_s{cfg.S}_f{cfg.F}"
                inst = (f"template [[host_name(\"{host}\")]] kernel void mrl_mmh<{codec}, {M}, {cfg.R}, {cfg.U}, "
                        f"{cfg.S}, {'true' if cfg.F else 'false'}, {'true' if pk else 'false'}>" + sig)
        elif cfg.xsrc == "mmr":
            if cfg.H and codec == "F16":
                codec = "F16H"
            key = (dtype, M, "mmr", cfg.R, cfg.U, cfg.F, cfg.H)
            host = f"mrl_mmr_{codec}_m{M}_r{cfg.R}_u{cfg.U}_f{cfg.F}"
            inst = (f"template [[host_name(\"{host}\")]] kernel void mrl_mmr<{codec}, {M}, {cfg.R}, {cfg.U}, "
                    f"{'true' if cfg.F else 'false'}>"
                    "(device const ushort*, device const uint4*, device const ushort*, device ushort*, "
                    "constant long&, constant long&, constant long&, device float*, constant long&, "
                    "constant long&, uint2, uint, uint, uint);")
        elif cfg.xsrc == "mma":
            key = (dtype, M, "mma", cfg.S, cfg.R, cfg.U)
            host = f"mrl_mma_{codec}_m{M}_s{cfg.S}_r{cfg.R}_u{cfg.U}"
            inst = (f"template [[host_name(\"{host}\")]] kernel void mrl_mma<{codec}, {M}, {cfg.S}, {cfg.R}, "
                    f"{32 * cfg.U}>({common}, device float*, constant long&, constant long&, uint2, uint, uint);")
        else:
            key = (dtype, M, cfg.xsrc, cfg.R, cfg.U)
            host = f"mrl_{cfg.xsrc}_{codec}_m{M}_r{cfg.R}_u{cfg.U}"
            extra = ", uint" if cfg.xsrc == "tg" else ""
            inst = (f"template [[host_name(\"{host}\")]] kernel void mrl_{cfg.xsrc}<{codec}, {M}, {cfg.R}, "
                    f"{cfg.U}>({common}, uint, uint, uint, uint{extra});")
        return self._compile(key, host, inst)

    def xpack(self, M):
        host = f"mrl_xpack_m{M}"
        inst = (f"template [[host_name(\"{host}\")]] kernel void mrl_xpack<{M}>(device const ushort*, "
                "device uint4*, constant long&, uint);")
        return self._compile(("xpack", M), host, inst)

    def reduce(self, dtype):
        codec = CODEC[dtype]
        host = f"mrl_reduce_{codec}"
        inst = (f"template [[host_name(\"{host}\")]] kernel void mrl_reduce<{codec}>(device const float*, "
                "device const ushort*, device ushort*, constant long&, constant long&, constant long&, "
                "constant long&, uint);")
        return self._compile((dtype, "reduce"), host, inst)


_LIB = None


def library():
    global _LIB
    if _LIB is None:
        _LIB = _Library()
    return _LIB


def supports(x, weight, bias=None):
    """True when multirow_linear can run these tensors without copying."""
    if x.device.type != "mps" or weight.device != x.device or x.dim() != 2 or weight.dim() != 2:
        return False
    M, K = x.shape
    N = weight.shape[0]
    if not 1 <= M <= MAX_M or weight.shape[1] != K or K < 8 or K % 8 or N < 1:  # K = 0 would divide by zero
        return False
    if x.dtype not in CODEC or weight.dtype != x.dtype:
        return False
    if not (x.is_contiguous() and weight.is_contiguous()):
        return False
    if (x.storage_offset() * 2) % 16 or (weight.storage_offset() * 2) % 16:
        return False
    if bias is not None and (bias.dtype != x.dtype or bias.shape != (N,) or not bias.is_contiguous()
                             or bias.device != x.device):
        return False
    return True


def multirow_linear(x, weight, bias=None, config=None, lib=None):
    if not supports(x, weight, bias):
        raise ValueError("multirow_linear: unsupported input (see supports())")
    M, K = x.shape
    N = weight.shape[0]
    cfg = config or pick_config(M, K, N, x.dtype)
    if cfg is None:
        return torch.nn.functional.linear(x, weight, bias)
    lib = lib or library()
    fn = lib.kernel(x.dtype, M, cfg)
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    b = x if bias is None else bias
    hb = 0 if bias is None else 1
    rows = cfg.rows_per_group()
    groups = (N + rows - 1) // rows
    if cfg.xsrc in ("dev", "tg"):
        fn(x, weight, b, y, K, N, hb, threads=groups * 32 * cfg.S, group_size=32 * cfg.S)
        return y
    kt = 32 * cfg.U
    klen = -(-K // (cfg.P * kt)) * kt          # K range per split, a multiple of the K tile
    splits = -(-K // klen)
    part = torch.empty((splits, M, N), dtype=torch.float32, device=x.device) if splits > 1 else y
    if cfg.xsrc == "mmp":
        mt, nblk = -(-M // 8), -(-K // 32)
        xp = torch.empty((nblk * mt * 32 * 8,), dtype=x.dtype, device=x.device)
        lib.xpack(M)(x, xp, K, threads=nblk * mt * 32, group_size=256)
        x = xp
    fn(x, weight, b, y, K, N, hb, part, klen, splits,
       threads=(groups * 32 * cfg.S, splits), group_size=(32 * cfg.S, 1))
    if splits > 1:
        lib.reduce(x.dtype)(part, b, y, M * N, N, splits, hb, threads=M * N, group_size=256)
    return y
