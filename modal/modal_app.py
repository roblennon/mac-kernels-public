"""Modal scaffold for mac-kernels: CPU numerics checks and a CUDA GEMM anchor.

Modal runs Linux containers on NVIDIA GPUs. Nothing here touches Metal; the Apple
numbers in docs/findings.md cannot be reproduced on Modal. What this file gives is a
cross-platform anchor for the same shapes, and a CPU farm for reference numerics.

Functions return plain Python only (floats, lists, dicts). A returned torch tensor is
unpickled in the local `modal` process, which has no torch, and the run fails there.
"""
from pathlib import Path

import modal

app = modal.App("mac-kernels")

cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy==2.5.3",
    "gguf==0.19.0",
)

gpu_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.15.0",
    "numpy==2.5.3",
)

# (M, K, N) as in docs/findings.md:17-19. x is [M, K], W is [N, K], y = x @ W.T.
SHAPES = [(4608, 6144, 6144), (4608, 6144, 16384), (4608, 16384, 6144)]


# --- CPU: GGUF dequantization against the written formulas -----------------------

def _dequant_q8_0(blocks):
    """Q8_0: 34-byte blocks of 32 values, f16 d then i8 qs[32]; value = qs[j] * d.
    docs/research/2026-10-03-gguf-on-metal.md:24."""
    import numpy as np
    b = blocks.reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    qs = b[:, 2:].view(np.int8).astype(np.float32)
    return qs * d


def _dequant_q4_0(blocks):
    """Q4_0: 18-byte blocks of 32 values, f16 d then u8 qs[16]; value = d * (nib - 8),
    low nibbles first, high nibbles after. docs/research/2026-10-03-gguf-on-metal.md:20."""
    import numpy as np
    b = blocks.reshape(-1, 18)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    qs = b[:, 2:]
    nib = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.float32)
    return d * (nib - 8.0)


@app.function(image=cpu_image, cpu=1.0, memory=2048, timeout=600)
def gguf_dequant_check(rows: int = 64, cols: int = 1024, seed: int = 0) -> dict:
    """Quantize random rows with the gguf package, dequantize them with the formulas
    above and with gguf.quants.dequantize, and report the disagreement (should be 0)."""
    import numpy as np
    from gguf import GGMLQuantizationType as T
    from gguf import quants

    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)
    out = {"rows": rows, "cols": cols, "seed": seed, "types": {}}
    for name, qtype, ours in (("Q8_0", T.Q8_0, _dequant_q8_0), ("Q4_0", T.Q4_0, _dequant_q4_0)):
        packed = quants.quantize(w, qtype)
        ref = quants.dequantize(packed, qtype).astype(np.float32)
        mine = ours(packed).reshape(rows, cols)
        err = np.abs(mine - ref).max()
        rel = float(np.linalg.norm(ref - w) / np.linalg.norm(w))
        out["types"][name] = {
            "bytes_per_row": int(packed.shape[1]),
            "max_abs_diff_vs_gguf": float(err),
            "exact": bool(err == 0.0),
            "quant_rel_l2_vs_fp32": rel,
        }
    return out


# --- GPU: GEMM anchor at the findings shapes ---------------------------------------

def _median_ms(fn, n=20, warm=5):
    import statistics
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def _gemm_anchor(n_iters: int) -> dict:
    """Median ms per call at the three findings shapes for the variants the Apple
    tables compare: fp16 and bf16 F.linear (findings.md:17-19), F.linear with a
    transposed weight (the pytorch/pytorch#200782 case), bare fp16 matmul, the
    ComfyUI-style dequantize-per-call INT8 path, true INT8 torch._int_mm (missing on
    MPS before torch 2.15), and fp8 _scaled_mm where the part supports it."""
    import torch
    import torch.nn.functional as F

    assert torch.cuda.is_available(), "no CUDA device in this container"
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    out = {
        "device": torch.cuda.get_device_name(0),
        "sm": f"{props.major}.{props.minor}",
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "iters": n_iters,
        "rows": [],
    }
    for M, K, N in SHAPES:
        torch.manual_seed(0)
        row = {"shape": [M, K, N]}
        x16 = torch.randn(M, K, device=dev, dtype=torch.float16)
        w16 = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.02
        xb, wb = x16.bfloat16(), w16.bfloat16()
        row["fp16_linear_ms"] = _median_ms(lambda: F.linear(x16, w16), n_iters)
        row["bf16_linear_ms"] = _median_ms(lambda: F.linear(xb, wb), n_iters)
        wt = w16.t().contiguous()  # stored [K, N]; passed as wt.t()
        row["fp16_linear_transposed_weight_ms"] = _median_ms(lambda: F.linear(x16, wt.t()), n_iters)
        row["fp16_mm_ms"] = _median_ms(lambda: x16 @ w16.t(), n_iters)
        scale = w16.abs().amax(dim=1, keepdim=True) / 127.0
        q = torch.round(w16 / scale).clamp(-127, 127).to(torch.int8)
        row["int8_dequant_per_call_then_fp16_ms"] = _median_ms(
            lambda: F.linear(x16, q.to(torch.float16) * scale), n_iters)
        xq = torch.round(x16 / (x16.abs().amax(dim=1, keepdim=True) / 127.0)).clamp(-127, 127).to(torch.int8)
        try:
            qt = q.t()
            row["int8_int_mm_ms"] = _median_ms(lambda: torch._int_mm(xq, qt), n_iters)
        except Exception as ex:
            row["int8_int_mm_ms"] = f"error: {type(ex).__name__}: {str(ex)[:120]}"
        try:
            xf8, wf8 = x16.to(torch.float8_e4m3fn), w16.to(torch.float8_e4m3fn)
            one = torch.ones((), device=dev)
            row["fp8_scaled_mm_ms"] = _median_ms(
                lambda: torch._scaled_mm(xf8, wf8.t(), scale_a=one, scale_b=one, out_dtype=torch.float16), n_iters)
        except Exception as ex:
            row["fp8_scaled_mm_ms"] = f"unsupported: {type(ex).__name__}"
        row["fp16_tflops"] = 2.0 * M * K * N / (row["fp16_linear_ms"] * 1e-3) / 1e12
        out["rows"].append(row)
        del x16, w16, xb, wb, wt, q, xq
        torch.cuda.empty_cache()
    return out


# One function per GPU type. T4 and A10G are the cheap anchors; A100 is the datacenter
# reference. L4 was capacity-starved on 2026-11-09 and is left out of the default run.
@app.function(image=gpu_image, gpu="T4", timeout=1200)
def cuda_gemm_bench_t4(n_iters: int = 20) -> dict:
    return _gemm_anchor(n_iters)


@app.function(image=gpu_image, gpu="A10G", timeout=1200)
def cuda_gemm_bench_a10g(n_iters: int = 20) -> dict:
    return _gemm_anchor(n_iters)


@app.function(image=gpu_image, gpu="L4", timeout=1200)
def cuda_gemm_bench_l4(n_iters: int = 20) -> dict:
    return _gemm_anchor(n_iters)


@app.function(image=gpu_image, gpu="A100-40GB", timeout=1200)
def cuda_gemm_bench_a100(n_iters: int = 20) -> dict:
    return _gemm_anchor(n_iters)


# --- GPU: wide-K transposed-weight control (bench/repro_mps_wide_linear.py on CUDA) ---

# The repro script itself is shipped into the container as a module, so the CUDA control
# runs the same variants, interleaving and timing loop as the MPS run. Nothing else from
# the repo is added.
_REPRO = Path(__file__).resolve().parent.parent / "bench" / "repro_mps_wide_linear.py"
wide_k_image = gpu_image
if modal.is_local():
    wide_k_image = gpu_image.add_local_file(str(_REPRO), "/root/repro_mps_wide_linear.py")

# The script's default sweep plus K = 32769, the first width past the M1/M2 reroute.
WIDE_KS = "4096,6144,8192,10240,12288,16384,24576,32768,32769"


def _wide_k(ks: str, dtypes: str, m: int, n: int, iters: int, rounds: int) -> dict:
    """Min ms over `rounds` interleaved rounds x `iters` calls for every variant in the
    repro script, per dtype and K, on the container's CUDA device."""
    import sys
    import torch

    sys.path.insert(0, "/root")
    import repro_mps_wide_linear as repro

    assert torch.cuda.is_available(), "no CUDA device in this container"
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    out = {
        "device": torch.cuda.get_device_name(0),
        "sm": f"{props.major}.{props.minor}",
        "memory_gib": round(props.total_memory / 2**30, 1),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "m": m, "n": n, "iters": iters, "rounds": rounds,
        "timing": "min ms over rounds x iters, torch.cuda.synchronize() around each call",
        "ratio": "F.linear(x, w) / x @ wt (contiguous)",
        "variants": list(repro.VARIANTS),
        # fp32 rows on Ampere and newer depend on these; torch defaults to IEEE fp32 (TF32 off).
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "rows": [],
    }
    print(repro.describe(dev), flush=True)
    try:
        for row in repro.sweep(dev, m, n, [int(k) for k in ks.split(",")], dtypes.split(","), iters, rounds):
            print(repro.format_row(row), flush=True)
            out["rows"].append(row)
    except Exception as ex:  # keep the rows measured so far and record the failure as it is
        out["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
    out["peak_memory_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return out


@app.function(image=wide_k_image, gpu="T4", timeout=1500)
def cuda_wide_k_t4(ks: str = WIDE_KS, dtypes: str = "float16,float32", m: int = 4608,
                   n: int = 6144, iters: int = 5, rounds: int = 3) -> dict:
    return _wide_k(ks, dtypes, m, n, iters, rounds)


@app.function(image=wide_k_image, gpu="A10G", timeout=1500)
def cuda_wide_k_a10g(ks: str = WIDE_KS, dtypes: str = "float16,float32", m: int = 4608,
                     n: int = 6144, iters: int = 5, rounds: int = 3) -> dict:
    return _wide_k(ks, dtypes, m, n, iters, rounds)


@app.function(image=wide_k_image, gpu="A100-40GB", timeout=1500)
def cuda_wide_k_a100(ks: str = WIDE_KS, dtypes: str = "float16,float32", m: int = 4608,
                     n: int = 6144, iters: int = 5, rounds: int = 3) -> dict:
    return _wide_k(ks, dtypes, m, n, iters, rounds)


WIDE_K_FUNCTIONS = {
    "T4": cuda_wide_k_t4,
    "A10G": cuda_wide_k_a10g,
    "A100": cuda_wide_k_a100,
}

GPU_FUNCTIONS = {
    "T4": cuda_gemm_bench_t4,
    "A10G": cuda_gemm_bench_a10g,
    "L4": cuda_gemm_bench_l4,
    "A100": cuda_gemm_bench_a100,
}


@app.local_entrypoint()
def main(gpus: str = "T4,A10G,A100", skip_gpu: bool = False, skip_cpu: bool = False,
         wide_k: bool = False, ks: str = WIDE_KS, out: str = ""):
    """`modal run modal/modal_app.py` runs the CPU check and the GEMM anchor on the
    listed GPUs in parallel, and prints one JSON object. With --wide-k it runs only the
    wide-K transposed-weight control on the listed GPUs. --out writes the JSON to a file
    as well. Results from 2026-11-09 are in modal/results/."""
    import json
    result = {}
    if wide_k:
        table = WIDE_K_FUNCTIONS
        handles = {g: table[g].spawn(ks=ks) for g in gpus.split(",") if g in table}
    else:
        if not skip_cpu:
            result["gguf_dequant_check"] = gguf_dequant_check.remote()
        table = GPU_FUNCTIONS
        handles = {} if skip_gpu else {g: table[g].spawn() for g in gpus.split(",") if g in table}
    for g, h in handles.items():
        try:
            result[g] = h.get()
        except Exception as ex:
            result[g] = {"error": f"{type(ex).__name__}: {str(ex)[:300]}"}
    text = json.dumps(result, indent=1)
    if out:
        Path(out).write_text(text + "\n")
    print(text)
