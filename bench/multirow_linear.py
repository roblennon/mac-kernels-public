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
    F: 1 = load the next K step into registers before computing this one (mmr only).
    H: 1 = half-precision fragment operands with fp32 accumulation (mmr, fp16 only)."""
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
        return self.S * self.R * (8 if self.xsrc in ("mma", "mmr") else 1)

    def threadgroup_bytes(self, M):
        if self.xsrc == "mma":
            ld = 32 * self.U + 4
            return (self.rows_per_group() + -(-M // 8) * 8) * ld * 4
        return M * 32 * self.U * 16 if self.xsrc == "tg" else 0

    def fits(self, M):
        return self.threadgroup_bytes(M) <= 32768


# The fastest variant per shape and row count in the tuning sweep (3 interleaved repeats,
# docs/review/2026-11-10-experiments/multirow/tune-*.json):
# {dtype name: {(K, N): {M: config name}}}. Shapes not listed use default_config(); row
# counts not listed use the nearest tuned row count.
TUNED: dict[str, dict[tuple[int, int], dict[int, str]]] = {
    "bfloat16": {
        (2048, 1024): {1: "dev_r4_u1_s8", 2: "tg_r1_u2_s4", 4: "dev_r2_u1_s4", 8: "mmr_r1_u1_s4_p4_f1", 12: "tg_r1_u2_s8", 16: "mma_r1_u1_s4_p4", 24: "mma_r1_u1_s4_p4", 32: "mma_r1_u1_s4_p4"},
        (2048, 2048): {1: "tg_r4_u1_s8", 2: "tg_r2_u2_s4", 4: "dev_r2_u2_s4", 8: "mmr_r1_u2_s4_f1", 12: "mma_r1_u1_s4_p4", 16: "mmr_r2_u1_s4_p4", 24: "mma_r2_u1_s4_p4", 32: "mma_r2_u1_s4_p4"},
        (2048, 6144): {1: "tg_r1_u2_s4", 2: "dev_r1_u1_s8", 4: "dev_r2_u1_s4", 8: "mmr_r1_u2_s4_f1", 12: "mma_r1_u1_s4", 16: "mma_r1_u1_s4", 24: "mma_r1_u1_s4", 32: "mma_r1_u1_s4"},
        (2048, 151936): {1: "tg_r2_u2_s8", 2: "tg_r1_u1_s4", 4: "tg_r2_u1_s8", 8: "mmr_r1_u1_s4_f1", 12: "mma_r1_u1_s4", 16: "mma_r1_u1_s4", 24: "mma_r1_u1_s4", 32: "mma_r1_u1_s4"},
        (2560, 9728): {1: "tg_r1_u2_s4", 2: "dev_r2_u1_s4", 4: "dev_r2_u1_s8", 8: "mmr_r1_u2_s4_f1", 12: "mma_r1_u1_s4_p4", 16: "mmr_r2_u1_s4", 24: "mma_r1_u1_s4_p4", 32: "mma_r1_u1_s4_p4"},
        (4096, 11008): {1: "tg_r1_u2_s8", 2: "dev_r1_u1_s8", 4: "dev_r2_u1_s4", 8: "mmr_r1_u2_s4_f1", 12: "mma_r1_u1_s4_p4", 16: "mma_r1_u1_s4_p4", 24: "mma_r1_u1_s4_p4", 32: "mma_r1_u1_s4"},
        (6144, 2048): {1: "dev_r1_u2_s4", 2: "dev_r2_u1_s8", 4: "dev_r2_u1_s4", 8: "mmr_r1_u2_s4_f1", 12: "mma_r1_u1_s4_p4", 16: "mma_r1_u1_s4_p4", 24: "mma_r1_u1_s4_p4", 32: "mma_r2_u1_s4_p4"},
    },
    "float16": {
        (2048, 1024): {1: "tg_r1_u2_s4", 2: "dev_r4_u1_s8", 4: "dev_r1_u1_s8", 8: "tg_r1_u1_s4", 12: "tg_r1_u1_s8", 16: "tg_r1_u1_s8", 24: "mma_r1_u1_s4_p4", 32: "mma_r1_u1_s4_p4"},
        (2048, 2048): {1: "tg_r2_u1_s4", 2: "tg_r1_u2_s8", 4: "dev_r2_u1_s4", 8: "mmr_r1_u2_s4_f1", 12: "tg_r1_u1_s8", 16: "mmr_r2_u1_s4_p4", 24: "mma_r1_u1_s4_p4", 32: "mma_r2_u1_s4_p4"},
        (2048, 6144): {1: "tg_r1_u2_s4", 2: "tg_r1_u2_s8", 4: "dev_r2_u1_s4", 8: "mmr_r1_u2_s4_f1", 12: "mmr_r2_u1_s4_p2", 16: "mma_r1_u1_s4", 24: "mma_r1_u1_s4", 32: "mma_r1_u1_s4"},
        (2048, 151936): {1: "dev_r4_u1_s8", 2: "mmr_r1_u1_s4_f1", 4: "dev_r2_u1_s4", 8: "mmr_r1_u1_s4", 12: "mmr_r2_u1_s4", 16: "mmr_r2_u1_s4", 24: "mma_r1_u1_s4", 32: "mma_r1_u1_s4"},
        (2560, 9728): {1: "dev_r1_u1_s8", 2: "dev_r2_u1_s8", 4: "dev_r2_u1_s4", 8: "mmr_r1_u2_s4_f1", 12: "mmr_r2_u1_s4", 16: "mmr_r2_u1_s4_p2", 24: "mma_r1_u1_s4", 32: "mma_r1_u1_s4"},
        (4096, 11008): {1: "dev_r2_u1_s4", 2: "dev_r2_u2_s8", 4: "tg_r2_u2_s4", 8: "mmr_r1_u2_s4_f1", 12: "mmr_r2_u1_s4_p2", 16: "mmr_r2_u1_s4", 24: "mma_r1_u1_s4_p4", 32: "mma_r1_u1_s4"},
        (6144, 2048): {1: "dev_r4_u1_s4", 2: "tg_r4_u1_s4", 4: "dev_r4_u1_s4", 8: "mmr_r2_u2_s4_p4", 12: "mma_r1_u1_s4_p4", 16: "mmr_r2_u1_s4_p4", 24: "mma_r1_u1_s4_p4", 32: "mma_r2_u1_s4_p4"},
    },
}


def default_config(M, K, N):
    if M <= 4:
        return Config("dev", 2, 1, 4)
    if M <= 8:
        return Config("mmr", 1, 2, 4, 1, 1)
    return Config("mma", 1, 1, 4, 4)


def pick_config(M, K, N, dtype=torch.float16):
    table = TUNED.get(str(dtype).split(".")[-1], {}).get((K, N))
    if table:
        best = min(table, key=lambda m: (abs(m - M), -m))
        return Config.parse(table[best])
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
        if cfg.xsrc == "mmr":
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
    if not 1 <= M <= MAX_M or weight.shape[1] != K or K % 8 or N < 1:
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
    fn(x, weight, b, y, K, N, hb, part, klen, splits,
       threads=(groups * 32 * cfg.S, splits), group_size=(32 * cfg.S, 1))
    if splits > 1:
        lib.reduce(x.dtype)(part, b, y, M * N, N, splits, hb, threads=M * N, group_size=256)
    return y
