"""Per-layer weight quantization error across formats, on Modal CPU containers (lane E9).

For every 2-D floating `*.weight` tensor of a public diffusion transformer checkpoint this
measures the relative Frobenius error and the max-abs error of the dequantized weight
against the original, for:

  int8_rowwise   INT8, per-row absmax/127 fp32 scale, no rotation
  int8_convrot   the same after the Hadamard-256 ConvRot rotation (the shipped INT8 layout)
  w4a8           4-bit grouped storage: ConvRot-256, groups of 16, Lloyd-Max codebook, e4m3 group scales
  w6a8           6-bit grouped storage: ConvRot-256, groups of 16, uniform levels, searched e4m3 group scales
  gguf_q8_0      GGUF Q8_0, quantized by ggml
  gguf_q4_k      GGUF Q4_K, quantized by ggml (no importance matrix)

The INT8 / W4A8 / W6A8 quantizers below are a pure-torch CPU port of the comfy-kitchen eager
reference (comfy_kitchen/backends/eager/quantization.py and w4a8_int8.py, Apache-2.0,
Copyright (c) 2025 Comfy Org), limited to the default deterministic path this repo uses:
symmetric, codebook on for 4 bits, float8_e4m3fn group scales, scale search on for 6 bits,
no stochastic rounding. modal/weight_error_validate.py checks the port against the vendor
checkout bit for bit. The GGUF formats call ggml_quantize_chunk from the libggml-base that
llama-cpp-python builds, and decode with gguf-py.

Errors are computed in float32 in the original (unrotated) basis. The rotation is orthogonal,
so the Frobenius error is the same in either basis; the max-abs error is not.

  modal run modal/weight_error_app.py                       # both models, writes modal/results/
  modal run modal/weight_error_app.py --models qwen21       # one model
  modal run modal/weight_error_app.py --selftest-only       # in-container quantizer vectors only

Functions return plain Python only.
"""
import base64
import hashlib
import json
import re
from pathlib import Path

import modal

try:  # the local `modal` process has no torch; every torch use is inside a function
    import torch
except ImportError:  # pragma: no cover
    torch = None

app = modal.App("mac-kernels-weight-error")
vol = modal.Volume.from_name("mac-kernels-weight-error", create_if_missing=True)
VOL = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "cmake", "git")
    .pip_install(
        "torch==2.15.0",
        "numpy==2.5.3",
        "safetensors==0.8.0",
        "gguf==0.19.0",
        "huggingface_hub==1.33.0",
    )
    .pip_install("llama-cpp-python==0.3.36")  # source build; used only for its libggml-base
)

# Checkpoints. Both repositories are public and not gated (checked 2026-11-09).
MODELS = {
    "qwen21": {
        "label": "Qwen-Image 2.1 transformer, bf16",
        "repo": "Comfy-Org/Qwen-Image-2.1",
        "revision": "aacf31a6d48e707f352ecba714cee0680da3479b",
        "file": "diffusion_models/qwen_image_2.1_bf16.safetensors",
        # The publisher's own INT8 ConvRot file: decides the "recipe" layer set and is compared code for code.
        "shipped_int8": "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
        "license": "other (repository card), not gated",
    },
    "wan21_1p3b": {
        "label": "Wan 2.1 T2V 1.3B transformer, fp32",
        "repo": "Wan-AI/Wan2.1-T2V-1.3B",
        "revision": "37ec512624d61f7aa208f7ea8140a131f93afc9a",
        "file": "diffusion_pytorch_model.safetensors",
        "shipped_int8": None,
        "license": "apache-2.0, not gated",
    },
}

FORMATS = ("int8_rowwise", "int8_convrot", "w4a8", "w6a8", "gguf_q8_0", "gguf_q4_k")
CONVROT_GROUP = 256
GGML_TYPE = {"gguf_q8_0": 8, "gguf_q4_k": 12}  # enum ggml_type
GGUF_BLOCK = {"gguf_q8_0": 32, "gguf_q4_k": 256}


# --- INT8 per-row and ConvRot (port of eager/quantization.py and tensor/int8_utils.py) -----

def build_hadamard(size, dtype=None):
    """Normalized regular Hadamard matrix; size must be a power of 4."""
    dtype = dtype or torch.float32
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    assert h.shape[0] == size, f"Hadamard size must be a power of 4, got {size}"
    return h / (size ** 0.5)


def rotate_weight(weight, group_size=CONVROT_GROUP):
    """W_rot = W @ H_block^T over groups of `group_size` input columns."""
    out_f, in_f = weight.shape
    assert in_f % group_size == 0
    h = build_hadamard(group_size, weight.dtype)
    return torch.matmul(weight.reshape(out_f, in_f // group_size, group_size), h.T).reshape(out_f, in_f)


def quantize_int8_rowwise(x):
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    s = scale.to(x.dtype)
    s = torch.where(s == 0, torch.full_like(s, torch.finfo(x.dtype).tiny), s)
    q = (x / s).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return q, scale


def quantize_int8_convrot(weight, group_size=CONVROT_GROUP):
    return quantize_int8_rowwise(rotate_weight(weight, group_size))


def dequantize_int8(q, scale):
    return q.float() * scale


def dequantize_int8_convrot(q, scale, group_size=CONVROT_GROUP):
    return rotate_weight(q.float() * scale, group_size)


# --- W4A8 / W6A8 (port of eager/w4a8_int8.py, default deterministic path) -------------------

_FIXED_LUT = (
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
)
_W4A8_GATE_KURTOSIS = -0.1
_KURTOSIS_SAMPLE = 1 << 19
_ALS_ITERS = 2
_W6_LEVELS = 31
_E4M3_MAX_FINITE_BITS = 0x7E
_QUANT_ROW_ELEM_BUDGET = 1 << 22
_CODEBOOK_SAMPLE_ELEMS = 1 << 22


def _pack_codes(unsigned, bits):
    low = ((unsigned[:, 0::2] & 0xF) | ((unsigned[:, 1::2] & 0xF) << 4)).to(torch.int8)
    if bits == 4:
        return low.contiguous()
    hi = (unsigned[:, 0::4] >> 4) & 0x3
    for j in range(1, 4):
        hi |= ((unsigned[:, j::4] >> 4) & 0x3) << (2 * j)
    return torch.cat([low, hi.to(torch.int8)], dim=1).contiguous()


def _unpack_codes(qdata, k, bits):
    n = qdata.shape[0]
    packed = qdata.view(torch.uint8).to(torch.int32)
    low = packed[:, : k // 2]
    codes = torch.empty(n, k, dtype=torch.int32)
    codes[:, 0::2] = low & 0xF
    codes[:, 1::2] = (low >> 4) & 0xF
    if bits == 6:
        shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
        hi = (packed[:, k // 2:].unsqueeze(-1) >> shifts) & 0x3
        codes |= hi.reshape(n, k) << 4
    return codes


def _fit_codebook(normalized, levels=16, iterations=25, sample_size=300000):
    samples = normalized.flatten()
    if samples.numel() > sample_size:
        generator = torch.Generator().manual_seed(0)
        samples = samples[torch.randint(0, samples.numel(), (sample_size,), generator=generator)]
    samples = samples.float()
    codebook = torch.quantile(samples, torch.linspace(0, 1, levels))
    for _ in range(iterations):
        assignments = (samples.unsqueeze(-1) - codebook).abs().argmin(-1)
        updated = codebook.clone()
        for index in range(levels):
            selected = assignments == index
            if selected.any():
                updated[index] = samples[selected].mean()
        codebook = updated
    return codebook.contiguous()


def _codebook_for(normalized):
    x = normalized.detach().flatten()
    if x.numel() > _KURTOSIS_SAMPLE:
        gen = torch.Generator().manual_seed(0)
        x = x[torch.randint(0, x.numel(), (_KURTOSIS_SAMPLE,), generator=gen)]
    x = x.float()
    excess_kurtosis = ((x - x.mean()) / (x.std() + 1e-9)).pow(4).mean() - 3.0
    if excess_kurtosis.item() <= _W4A8_GATE_KURTOSIS:
        return torch.tensor(_FIXED_LUT, dtype=torch.float32)
    return _fit_codebook(normalized)


def _assign_codes(normalized, codebook):
    last = codebook.numel() - 1
    pos = torch.searchsorted(codebook, normalized.contiguous())
    lo = (pos - 1).clamp(0, last)
    hi = pos.clamp(0, last)
    dlo = (normalized - codebook[lo]).abs_()
    dhi = (normalized - codebook[hi]).abs_()
    return torch.where(dhi < dlo, hi, lo).to(torch.int32)


def _assign_grid(weight, levels, s_channel, target=None):
    n, groups, gsize = weight.shape
    last = levels.shape[-1] - 1
    lv = levels.reshape(n * groups, last + 1).contiguous()
    if target is None:
        target = weight / s_channel.view(-1, 1, 1)
    tg = target.reshape(n * groups, gsize).contiguous()
    pos = torch.searchsorted(lv, tg)
    lo = (pos - 1).clamp(0, last)
    hi = pos.clamp(0, last)
    dlo = tg.sub(torch.gather(lv, 1, lo)).abs_()
    dhi = tg.sub(torch.gather(lv, 1, hi)).abs_()
    return torch.where(dhi < dlo, hi, lo).to(torch.int32).reshape(n, groups, gsize)


def _grid_levels(level_table, s_rel):
    return (level_table.view(1, 1, -1) * s_rel.float().unsqueeze(-1)).round_().clamp_(-127, 127)


def _search_group_scale(grouped_weight, s_rel, s_channel, level_table):
    target = grouped_weight / s_channel.view(-1, 1, 1)

    def score(cand):
        levels = _grid_levels(level_table, cand)
        idx = _assign_grid(grouped_weight, levels, s_channel, target=target)
        err = torch.gather(levels, 2, idx.long()).sub_(target).pow_(2).sum(-1)
        return cand, err, idx

    raw = s_rel.view(torch.uint8)
    candidates = [
        s_rel.float(),
        (raw - 1).clamp_(min=1).view(torch.float8_e4m3fn).float(),
        (raw + 1).clamp_(max=_E4M3_MAX_FINITE_BITS).view(torch.float8_e4m3fn).float(),
    ]
    best_scale, best_err, best_idx = score(candidates[0])
    for cand in candidates[1:]:
        cand, err, idx = score(cand)
        better = err < best_err
        torch.where(better, err, best_err, out=best_err)
        torch.where(better, cand, best_scale, out=best_scale)
        torch.where(better.unsqueeze(-1), idx, best_idx, out=best_idx)
    return best_scale.to(s_rel.dtype).contiguous(), best_idx


def _quantize_rotated_w4a8(weight, group_size, bits, codebook_override):
    n, k = weight.shape
    groups = k // group_size
    grouped_weight = weight.float().view(n, groups, group_size)
    amax = grouped_weight.abs().amax(dim=-1, keepdim=True)
    codebook_tensor = None
    if bits == 6:
        level_table = torch.arange(-_W6_LEVELS, _W6_LEVELS + 1, dtype=torch.float32)
        group_scale = (amax / _W6_LEVELS).clamp(min=1e-8)

        def assign(normalized):
            return (normalized.round_().clamp_(-_W6_LEVELS, _W6_LEVELS) + _W6_LEVELS).to(torch.int32)

        quantized = assign(grouped_weight / group_scale)
    else:
        group_scale = amax.clamp(min=1e-8)
        normalized = grouped_weight / group_scale
        codebook_tensor = codebook_override if codebook_override is not None else _codebook_for(normalized)
        level_table = codebook_tensor

        def assign(normalized):
            return _assign_codes(normalized, level_table)

        quantized = assign(normalized)
    for _ in range(_ALS_ITERS):
        quantized_levels = level_table[quantized]
        group_scale = (
            (grouped_weight * quantized_levels).sum(-1, keepdim=True)
            / (quantized_levels * quantized_levels).sum(-1, keepdim=True).clamp(min=1e-8)
        ).clamp(min=1e-8)
        quantized = assign(grouped_weight / group_scale)
    shifted_weight = level_table[quantized] * group_scale

    s_channel = (shifted_weight.abs().amax(dim=(1, 2)) / 127.0).clamp(min=1e-8)
    s_rel = (group_scale.squeeze(-1) / s_channel.unsqueeze(1)).float().contiguous()
    s_rel = s_rel.to(torch.float8_e4m3fn).contiguous()
    if bits == 6:
        s_rel, unsigned = _search_group_scale(grouped_weight, s_rel, s_channel, level_table)
        unsigned.add_(1)  # level index 0..62 -> stored code q+32 in 1..63
    else:
        unsigned = _assign_grid(grouped_weight, _grid_levels(level_table, s_rel), s_channel)
    packed = _pack_codes(unsigned.view(n, k), bits)
    return packed, s_rel, s_channel.float().contiguous(), codebook_tensor


def quantize_w4a8(weight, bits=4, group_size=16, convrot_groupsize=CONVROT_GROUP):
    """Returns (packed int8 [N, K*bits/8], s_rel e4m3 [N, K/group], s_channel fp32 [N], codebook or None)."""
    n, k = weight.shape
    assert bits in (4, 6) and k % 16 == 0 and k % group_size == 0 and k % convrot_groupsize == 0
    if bits == 6:
        assert k % 32 == 0 and group_size >= 16 and group_size % 16 == 0
    block = max(1, _QUANT_ROW_ELEM_BUDGET // max(k, 1))
    override = None
    if bits == 4:  # one codebook for the whole tensor, from an evenly strided row sample
        max_rows = max(1, _CODEBOOK_SAMPLE_ELEMS // max(k, 1))
        sample = weight
        if n > max_rows:
            sample = weight.index_select(0, torch.linspace(0, n - 1, max_rows).long())
        rotated = rotate_weight(sample.contiguous(), convrot_groupsize)
        grouped = rotated.float().view(sample.shape[0], k // group_size, group_size)
        override = _codebook_for(grouped / grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8))
    if n <= block:
        return _quantize_rotated_w4a8(rotate_weight(weight.contiguous(), convrot_groupsize), group_size, bits, override)
    packed_parts, srel_parts, sch_parts, codebook = [], [], [], None
    for r0 in range(0, n, block):
        rot = rotate_weight(weight[r0:r0 + block].contiguous(), convrot_groupsize)
        packed, s_rel, s_channel, codebook = _quantize_rotated_w4a8(rot, group_size, bits, override)
        packed_parts.append(packed)
        srel_parts.append(s_rel)
        sch_parts.append(s_channel)
    return torch.cat(packed_parts, 0), torch.cat(srel_parts, 0), torch.cat(sch_parts, 0), codebook


def dequantize_w4a8(qdata, s_rel, s_channel, codebook, group_size=16, convrot_groupsize=CONVROT_GROUP):
    """Packed storage -> fp32 weight in the original basis, through the int8 grid the kernels use."""
    n, cols = qdata.shape
    k = s_rel.shape[1] * group_size
    bits = (cols * 8) // k
    quantized = _unpack_codes(qdata, k, bits)
    if codebook is not None:
        values = codebook.to(torch.float32)[quantized]
    else:
        values = quantized.float() - float(1 << (bits - 1))
    values = values.view(n, k // group_size, group_size) * s_rel.float().unsqueeze(-1)
    int8_weight = values.view(n, k).round().clamp_(-127, 127).to(torch.int8)
    rotated = int8_weight.float().view(n, k // group_size, group_size) * s_channel.float().view(n, 1, 1)
    return rotate_weight(rotated.view(n, k), convrot_groupsize)


# --- GGUF through ggml ------------------------------------------------------------------------

class Ggml:
    """ggml_quantize_chunk from a libggml-base shared library, threaded over row ranges."""

    def __init__(self, lib_path=None):
        import ctypes
        import glob
        import os
        import sysconfig

        candidates = [lib_path] if lib_path else []
        site = sysconfig.get_paths()["purelib"]
        candidates += sorted(glob.glob(os.path.join(site, "llama_cpp", "lib", "libggml-base.*")))
        candidates += sorted(glob.glob(os.path.join(site, "lib", "libggml-base.*")))
        found = [c for c in candidates if c and os.path.exists(c)]
        assert found, "no libggml-base found"
        self.lib_name = os.path.basename(found[0])
        lib = ctypes.CDLL(found[0])
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_quantize_chunk.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
        ]
        lib.ggml_row_size.restype = ctypes.c_size_t
        lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        lib.ggml_quantize_init.argtypes = [ctypes.c_int]
        lib.ggml_quantize_init.restype = None
        self.version = None
        for name in ("ggml_version", "ggml_commit"):
            if hasattr(lib, name):
                fn = getattr(lib, name)
                fn.restype = ctypes.c_char_p
                self.version = (self.version + " " if self.version else "") + fn().decode()
        self.lib = lib
        for t in GGML_TYPE.values():
            lib.ggml_quantize_init(t)

    def quantize(self, w, fmt, threads=1):
        """w: C-contiguous float32 numpy [n, k]. Returns uint8 [n, row_bytes]."""
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor

        n, k = w.shape
        t = GGML_TYPE[fmt]
        row_bytes = self.lib.ggml_row_size(t, k)
        out = np.empty((n, row_bytes), dtype=np.uint8)
        src, dst = w.ctypes.data, out.ctypes.data
        step = max(1, -(-n // max(1, threads)))

        def run(r0):
            rows = min(step, n - r0)
            got = self.lib.ggml_quantize_chunk(t, src, dst, r0 * k, rows, k, None)
            assert got == rows * row_bytes, (got, rows, row_bytes)

        with ThreadPoolExecutor(max(1, threads)) as pool:
            list(pool.map(run, range(0, n, step)))
        return out


def gguf_dequantize(packed, fmt):
    """Reference decode with gguf-py; uint8 [n, row_bytes] -> float32 [n, k]."""
    from gguf import GGMLQuantizationType as T
    from gguf import quants

    return quants.dequantize(packed, {"gguf_q8_0": T.Q8_0, "gguf_q4_k": T.Q4_K}[fmt])


# --- Shared self-test vectors (run locally and in the container) -----------------------------

def _sha(t):
    if hasattr(t, "detach"):
        t = t.detach().contiguous()
        if t.dtype == torch.float8_e4m3fn:
            t = t.view(torch.uint8)
        t = t.numpy()
    return hashlib.sha256(t.tobytes()).hexdigest()


def selftest_inputs():
    """Seeded small tensors built from integer draws and exact arithmetic only, so they are
    bit-identical on every platform (torch.randn is not): near-Gaussian (fixed codebook),
    heavy-tailed after rotation (fitted codebook), and one over the row-chunk budget."""
    g = torch.Generator().manual_seed(1234)

    def gauss(n, k):  # sum of 12 uniform integers, centred, std 1/64
        u = torch.randint(0, 65536, (12, n, k), generator=g, dtype=torch.int64).sum(0) - 6 * 65535
        return (u.double() * 2.0 ** -22).float().to(torch.bfloat16).float()

    wa = gauss(96, 1024)
    u = torch.randint(0, 256, (4, 96, 1024), generator=g, dtype=torch.int64).sum(0) - 510
    cubes = u.double() ** 3  # integers below 2^28; the +-1 Hadamard sums stay exact in float64
    wb = (rotate_weight(cubes * 16.0) / 256.0 * 2.0 ** -30).float().to(torch.bfloat16).float()
    wc = gauss(4200, 1024)
    return {"gauss_96x1024": wa, "heavy_96x1024": wb, "chunked_4200x1024": wc}


def quantize_all_torch(w):
    """Every torch-side format on one weight: name -> dict of stored tensors plus 'deq'."""
    out = {}
    q, s = quantize_int8_rowwise(w.clone())
    out["int8_rowwise"] = {"q": q, "scale": s, "deq": dequantize_int8(q, s)}
    q, s = quantize_int8_convrot(w)
    out["int8_convrot"] = {"q": q, "scale": s, "deq": dequantize_int8_convrot(q, s)}
    for name, bits in (("w4a8", 4), ("w6a8", 6)):
        packed, s_rel, s_ch, cb = quantize_w4a8(w, bits=bits)
        out[name] = {"q": packed, "s_rel": s_rel, "s_channel": s_ch,
                     "deq": dequantize_w4a8(packed, s_rel, s_ch, cb)}
        if cb is not None:
            out[name]["codebook"] = cb
    return out


def _bytes(t):
    if hasattr(t, "detach"):
        t = t.detach().contiguous()
        if t.dtype == torch.float8_e4m3fn:
            t = t.view(torch.uint8)
        t = t.numpy()
    return t.tobytes()


def _row_sha(t, n):
    """16 hex characters of sha256 per row, so two platforms can count differing rows of a
    tensor that is too large to store raw."""
    raw = _bytes(t)
    step = len(raw) // n
    return [hashlib.sha256(raw[i * step:(i + 1) * step]).hexdigest()[:16] for i in range(n)]


def selftest_vectors(with_gguf=True, raw_for=("gauss_96x1024", "heavy_96x1024"), ggml=None):
    """For every case and format: sha256 of every stored tensor and of the dequantized weight,
    the two error metrics, and per-row hashes of the row-shaped stored tensors. For the small
    cases also the raw stored tensors (base64), so two platforms can count differing bytes."""
    import numpy as np

    res = {"torch": str(torch.__version__), "cases": {}}
    if with_gguf:
        ggml = ggml or Ggml()
        res["ggml_lib"], res["ggml_version"] = ggml.lib_name, ggml.version
    else:
        ggml = None

    def record(w, parts):
        n = w.shape[0]
        e = {k: _sha(v) for k, v in parts.items()}
        e.update(_err(w, torch.as_tensor(parts["deq"]), w.double().pow(2).sum()))
        stored = {k: v for k, v in parts.items() if k != "deq"}
        e["row_sha"] = {k: _row_sha(v, n) for k, v in stored.items() if v.shape[0] == n}
        return e, stored

    for case, w in selftest_inputs().items():
        entry = {"input_sha256": _sha(w), "formats": {}}
        todo = dict(quantize_all_torch(w))
        if ggml is not None:
            wn = np.ascontiguousarray(w.numpy())
            for fmt in GGML_TYPE:
                packed = ggml.quantize(wn, fmt)
                todo[fmt] = {"q": packed, "deq": gguf_dequantize(packed, fmt)}
        for fmt, parts in todo.items():
            e, stored = record(w, parts)
            if case in raw_for:
                for k, v in stored.items():
                    e[k + "_b64"] = base64.b64encode(_bytes(v)).decode()
            entry["formats"][fmt] = e
        res["cases"][case] = entry
    return res


# --- Measurement ------------------------------------------------------------------------------

def _err(w, deq, w_sq):
    d = (deq - w).double()
    return {"rel_fro": float((d.pow(2).sum() / w_sq).sqrt()), "max_abs": float(d.abs().max())}


def _nbytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def measure_layer(w, ggml, threads):
    """w: fp32 [N, K]. Returns format -> {rel_fro, max_abs, bits_per_weight} or {skipped}."""
    import numpy as np

    n, k = w.shape
    w_sq = w.double().pow(2).sum()
    out = {}
    q, s = quantize_int8_rowwise(w.clone())
    out["int8_rowwise"] = {**_err(w, dequantize_int8(q, s), w_sq), "bits_per_weight": 8 * _nbytes(q, s) / (n * k)}
    convrot_q = convrot_s = None
    if k % CONVROT_GROUP == 0:
        convrot_q, convrot_s = quantize_int8_convrot(w)
        out["int8_convrot"] = {**_err(w, dequantize_int8_convrot(convrot_q, convrot_s), w_sq),
                               "bits_per_weight": 8 * _nbytes(convrot_q, convrot_s) / (n * k)}
        for name, bits in (("w4a8", 4), ("w6a8", 6)):
            packed, s_rel, s_ch, cb = quantize_w4a8(w, bits=bits)
            out[name] = {**_err(w, dequantize_w4a8(packed, s_rel, s_ch, cb), w_sq),
                         "bits_per_weight": 8 * _nbytes(packed, s_rel, s_ch, cb) / (n * k)}
            if bits == 4:
                out[name]["codebook"] = "fixed" if torch.equal(cb, torch.tensor(_FIXED_LUT)) else "fitted"
            del packed, s_rel, s_ch
    else:
        for name in ("int8_convrot", "w4a8", "w6a8"):
            out[name] = {"skipped": f"K={k} is not a multiple of the ConvRot group {CONVROT_GROUP}"}
    wn = np.ascontiguousarray(w.numpy())
    for fmt in GGML_TYPE:
        if k % GGUF_BLOCK[fmt] != 0:
            out[fmt] = {"skipped": f"K={k} is not a multiple of the {GGUF_BLOCK[fmt]}-value block"}
            continue
        packed = ggml.quantize(wn, fmt, threads)
        sq, mx = 0.0, 0.0
        step = max(1, (1 << 23) // k)
        for r0 in range(0, n, step):  # decode in row slabs to bound numpy temporaries
            d = torch.from_numpy(gguf_dequantize(packed[r0:r0 + step], fmt)).double() - w[r0:r0 + step].double()
            sq += float(d.pow(2).sum())
            mx = max(mx, float(d.abs().max()))
        out[fmt] = {"rel_fro": (sq / float(w_sq)) ** 0.5, "max_abs": mx, "bits_per_weight": 8 * packed.size / (n * k)}
    return out, convrot_q, convrot_s


def _read_header(path):
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def _model_path(model, key="file"):
    m = MODELS[model]
    return f"{VOL}/{model}/{m[key]}"


@app.function(image=image, cpu=2.0, memory=4096, timeout=3600, volumes={VOL: vol})
def download(model: str) -> dict:
    """Fetch the pinned checkpoint (and the publisher's INT8 file, if any) into the Volume."""
    import os
    from huggingface_hub import hf_hub_download

    m = MODELS[model]
    got = {}
    for key in ("file", "shipped_int8"):
        if not m[key]:
            continue
        path = hf_hub_download(m["repo"], m[key], revision=m["revision"], local_dir=f"{VOL}/{model}")
        got[m[key]] = os.path.getsize(path)
    vol.commit()
    return {"model": model, "repo": m["repo"], "revision": m["revision"], "bytes": got}


@app.function(image=image, cpu=1.0, memory=2048, timeout=600, volumes={VOL: vol})
def plan(model: str) -> dict:
    """List the 2-D floating weights and split them into per-block jobs."""
    header = _read_header(_model_path(model))
    shipped = _read_header(_model_path(model, "shipped_int8")) if MODELS[model]["shipped_int8"] else {}
    recipe = sorted(k for k, v in shipped.items() if v["dtype"] == "I8")
    groups, skipped = {}, []
    for key, v in sorted(header.items()):
        if not key.endswith(".weight") or v["dtype"] not in ("BF16", "F16", "F32"):
            continue
        if len(v["shape"]) != 2:
            if len(v["shape"]) > 2:
                skipped.append({"key": key, "shape": v["shape"], "why": "not a 2-D linear weight"})
            continue
        mt = re.search(r"blocks\.(\d+)\.", key)
        groups.setdefault(f"block{int(mt.group(1)):03d}" if mt else "other", []).append(key)
    jobs = [{"model": model, "group": g, "keys": ks} for g, ks in sorted(groups.items())]
    return {"model": model, "jobs": jobs, "recipe_keys": recipe, "skipped_tensors": skipped,
            "n_linear": sum(len(j["keys"]) for j in jobs)}


@app.function(image=image, cpu=4.0, memory=8192, timeout=2400, volumes={VOL: vol}, max_containers=12)
def measure_group(job: dict) -> dict:
    """Quantize every layer of one block in every format; returns per-layer rows."""
    import time
    from safetensors import safe_open

    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = job["model"]
    ggml = Ggml()
    shipped_path = _model_path(model, "shipped_int8") if MODELS[model]["shipped_int8"] else None
    shipped_header = _read_header(shipped_path) if shipped_path else {}
    rows, t_start = [], time.time()
    with safe_open(_model_path(model), framework="pt", device="cpu") as src:
        shipped = safe_open(shipped_path, framework="pt", device="cpu") if shipped_path else None
        for key in job["keys"]:
            t0 = time.time()
            raw = src.get_tensor(key)
            w = raw.float().contiguous()
            n, k = w.shape
            formats, cq, cs = measure_layer(w, ggml, threads=4)
            row = {
                "model": model, "group": job["group"], "key": key, "shape": [n, k],
                "source_dtype": str(raw.dtype).replace("torch.", ""),
                "w_fro": float(w.double().pow(2).sum().sqrt()), "w_absmax": float(w.abs().max()),
                "in_shipped_recipe": bool(shipped_header.get(key, {}).get("dtype") == "I8"),
                "formats": formats,
            }
            if row["in_shipped_recipe"] and cq is not None:
                sq = shipped.get_tensor(key)
                ss = shipped.get_tensor(key[:-len(".weight")] + ".weight_scale").float().reshape(cs.shape)
                diff = (sq.int() - cq.int()).abs()
                w_sq = w.double().pow(2).sum()
                row["shipped_int8_convrot"] = {
                    "codes_differing": int((diff != 0).sum()), "codes_total": int(diff.numel()),
                    "max_code_diff": int(diff.max()),
                    "scale_max_abs_diff": float((ss - cs).abs().max()),
                    "scale_bit_identical": bool(torch.equal(ss, cs)),
                    **_err(w, dequantize_int8_convrot(sq, ss), w_sq),
                }
            row["seconds"] = round(time.time() - t0, 2)
            rows.append(row)
            del w, raw, cq, cs
    return {"model": model, "group": job["group"], "rows": rows, "seconds": round(time.time() - t_start, 1),
            "torch": str(torch.__version__), "ggml_lib": ggml.lib_name, "ggml_version": ggml.version}


@app.function(image=image, cpu=2.0, memory=4096, timeout=900)
def selftest() -> dict:
    """The shared quantizer vectors, computed in the container, plus a Q8_0 cross-check of
    ggml against the gguf-py numpy quantizer."""
    import importlib.metadata as md
    import numpy as np
    from gguf import GGMLQuantizationType as T
    from gguf import quants

    torch.set_num_threads(2)
    res = selftest_vectors(with_gguf=True)
    res["llama_cpp_python"] = md.version("llama-cpp-python")
    res["gguf"] = md.version("gguf")
    w = np.ascontiguousarray(selftest_inputs()["gauss_96x1024"].numpy())
    a, b = Ggml().quantize(w, "gguf_q8_0"), quants.quantize(w, T.Q8_0)
    res["q8_0_ggml_vs_gguf_py"] = {"bytes_differing": int((a != b.reshape(a.shape)).sum()), "bytes_total": int(a.size)}
    return res


# --- Summary and driver (plain Python, runs in the local modal process) --------------------

def _quantile(sorted_vals, q):
    """Linear-interpolated quantile (numpy's default method)."""
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def summarize(rows):
    """Per model, per scope, per format: median, p95 and worst layer of both error measures,
    plus the pooled relative Frobenius error over all measured weights."""
    out = {}
    for model in sorted({r["model"] for r in rows}):
        mrows = [r for r in rows if r["model"] == model]
        scopes = {"all_linear": mrows}
        if any(r["in_shipped_recipe"] for r in mrows):
            scopes["shipped_recipe_layers"] = [r for r in mrows if r["in_shipped_recipe"]]
        if any(r["group"] != "other" for r in mrows):
            scopes["block_layers"] = [r for r in mrows if r["group"] != "other"]
        out[model] = {}
        for scope, srows in scopes.items():
            per = {}
            names = list(FORMATS) + (["shipped_int8_convrot"] if any("shipped_int8_convrot" in r for r in srows) else [])
            for fmt in names:
                got = [(r, r["formats"][fmt] if fmt in r["formats"] else r.get(fmt)) for r in srows]
                meas = [(r, f) for r, f in got if f and "rel_fro" in f]
                if not meas:
                    per[fmt] = {"layers": 0, "skipped": len(got)}
                    continue
                rel = sorted(f["rel_fro"] for _, f in meas)
                mab = sorted(f["max_abs"] for _, f in meas)
                worst_rel = max(meas, key=lambda rf: rf[1]["rel_fro"])
                worst_abs = max(meas, key=lambda rf: rf[1]["max_abs"])
                err_sq = sum((f["rel_fro"] * r["w_fro"]) ** 2 for r, f in meas)
                w_sq = sum(r["w_fro"] ** 2 for r, _ in meas)
                params = sum(r["shape"][0] * r["shape"][1] for r, _ in meas)
                per[fmt] = {
                    "layers": len(meas),
                    "skipped": len(got) - len(meas),
                    "params": params,
                    "rel_fro_median": _quantile(rel, 0.5), "rel_fro_p95": _quantile(rel, 0.95),
                    "rel_fro_max": rel[-1], "rel_fro_worst_layer": worst_rel[0]["key"],
                    "rel_fro_pooled": (err_sq / w_sq) ** 0.5,
                    "max_abs_median": _quantile(mab, 0.5), "max_abs_p95": _quantile(mab, 0.95),
                    "max_abs_max": mab[-1], "max_abs_worst_layer": worst_abs[0]["key"],
                }
                if "bits_per_weight" in meas[0][1]:
                    per[fmt]["bits_per_weight"] = sum(
                        f["bits_per_weight"] * r["shape"][0] * r["shape"][1] for r, f in meas) / params
            pairs = {}
            for x, y in (("int8_convrot", "int8_rowwise"), ("w4a8", "gguf_q4_k"), ("gguf_q8_0", "int8_convrot")):
                both = [(r["formats"][x], r["formats"][y]) for r in srows
                        if "rel_fro" in r["formats"][x] and "rel_fro" in r["formats"][y]]
                pairs[f"{x}_lower_than_{y}"] = {
                    "layers_compared": len(both),
                    "rel_fro": sum(fx["rel_fro"] < fy["rel_fro"] for fx, fy in both),
                    "max_abs": sum(fx["max_abs"] < fy["max_abs"] for fx, fy in both),
                }
            per["pairwise_layer_counts"] = pairs
            cb = [r["formats"]["w4a8"]["codebook"] for r in srows if "codebook" in r["formats"]["w4a8"]]
            per["w4a8_codebook_layers"] = {k: cb.count(k) for k in sorted(set(cb))}
            sh = [r["shipped_int8_convrot"] for r in srows if "shipped_int8_convrot" in r]
            if sh:
                per["shipped_int8_convrot_vs_ours"] = {
                    "layers": len(sh),
                    "codes_differing": sum(x["codes_differing"] for x in sh),
                    "codes_total": sum(x["codes_total"] for x in sh),
                    "max_code_diff": max(x["max_code_diff"] for x in sh),
                    "scale_max_abs_diff": max(x["scale_max_abs_diff"] for x in sh),
                }
            out[model][scope] = per
    return out


@app.local_entrypoint()
def main(models: str = "qwen21,wan21_1p3b", selftest_only: bool = False, out: str = "", limit_jobs: int = 0):
    import datetime

    here = Path(__file__).resolve().parent
    st = selftest.remote()
    (here / "results" / "2026-11-09-weight-error-selftest-container.json").write_text(json.dumps(st, indent=1) + "\n")
    print("selftest: torch", st["torch"], "ggml", st.get("ggml_version"), st["q8_0_ggml_vs_gguf_py"])
    if selftest_only:
        return
    names = [m for m in models.split(",") if m]
    downloads = list(download.map(names))
    plans = list(plan.map(names))
    jobs = [j for p in plans for j in p["jobs"]]
    if limit_jobs:  # timing probe; write it somewhere else with --out
        jobs = jobs[:limit_jobs]
    print(f"{len(jobs)} jobs, {sum(p['n_linear'] for p in plans)} linear weights")
    rows, runs = [], []
    for res in measure_group.map(jobs, order_outputs=True):
        rows.extend(res.pop("rows"))
        runs.append(res)
        print(f"  {res['model']} {res['group']}: {res['seconds']} s")
    doc = {
        "what": "Per-layer weight quantization error against the original checkpoint weights",
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "script": "modal/weight_error_app.py",
        "metric": {
            "rel_fro": "||W_hat - W||_F / ||W||_F, fp32 weights, float64 accumulation, original basis",
            "max_abs": "max |W_hat - W| in the original basis, in weight units",
            "rel_fro_pooled": "sqrt(sum of squared errors / sum of squared weights) over the layers in scope",
            "bits_per_weight": "stored bytes of codes and scales (and codebook) times 8 over N*K",
        },
        "formats": {
            "int8_rowwise": "INT8, per-row absmax/127 fp32 scale, no rotation",
            "int8_convrot": "INT8 per-row after Hadamard-256 ConvRot rotation",
            "w4a8": "4-bit, ConvRot-256, group 16, Lloyd-Max codebook, e4m3 group scales, int8 grid decode",
            "w6a8": "6-bit uniform, ConvRot-256, group 16, searched e4m3 group scales, int8 grid decode",
            "gguf_q8_0": "ggml_quantize_chunk Q8_0, decoded with gguf-py",
            "gguf_q4_k": "ggml_quantize_chunk Q4_K without importance matrix, decoded with gguf-py",
            "shipped_int8_convrot": "the publisher's INT8 ConvRot tensors, dequantized as stored",
        },
        "models": {m: MODELS[m] for m in names},
        "environment": {"torch": runs[0]["torch"], "ggml_lib": runs[0]["ggml_lib"], "ggml_version": runs[0]["ggml_version"],
                        "llama_cpp_python": st["llama_cpp_python"], "gguf": st["gguf"],
                        "container": "Modal debian_slim, python 3.12, 4 CPU, 8 GiB per job"},
        "downloads": downloads,
        "skipped_tensors": {p["model"]: p["skipped_tensors"] for p in plans},
        "container_seconds": {"sum_over_jobs": round(sum(r["seconds"] for r in runs), 1), "jobs": len(runs)},
        "summary": summarize(rows),
        "layers": rows,
    }
    target = Path(out) if out else here / "results" / "2026-11-09-weight-error.json"
    target.write_text(json.dumps(doc, indent=1) + "\n")
    print("wrote", target.name, len(rows), "layer rows")
