"""Input gradient of the frozen INT8 / W6A8 / W4A8 linears at Qwen-sized K, against fp64.

The kernel tests check the packed backward at n=67, K<=768 with random packed bytes. This
probe drives the same kernels standalone (no checkpoint, no text encoder) at the widths a
Qwen-Image block uses:

1. Single layers. A synthetic heavy-tailed weight is quantized by the repo's own eager
   quantizers (INT8 ConvRot, W6A8, W4A8 symmetric, W4A8 asymmetric). The input gradient
   ``dy @ W`` comes from the Metal backward on MPS and is compared with an fp64 CPU
   reference that uses the dequantized weight: cosine, relative L2, max abs error. Two
   comparators put the numbers in context: the fp64 reference rounded once to the output
   dtype (the floor any implementation can reach) and a dense MPS matmul against the
   dequantized weight in the same dtype (what a dequantize-then-GEMM trainer computes).
2. One MLP, gate_up -> SwiGLU -> out, with a LoRA beside each frozen layer as
   train/train_lora.py attaches it. dA and dB of the first LoRA depend on the backward
   through the second frozen layer; dx depends on both.

Weights are synthetic. No model file is read except, optionally, the safetensors header of
the Qwen-Image 2.1 INT8 checkpoint under MK_ROOT to record its real layer shapes.

  bench/with_gpu.sh .venv/bin/python bench/probe_packed_backward_real_k.py --out OUT.json
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
KITCHEN = REPO / "vendor" / "comfy-kitchen"
sys.path.insert(0, str(KITCHEN))

import torch
import torch.nn.functional as F

torch.set_num_threads(4)  # the host is shared; the fp64 reference does not need more

from comfy_kitchen.backends.eager import quantization as eager_int8
from comfy_kitchen.backends.eager import w4a8_int8 as eager_packed
from comfy_kitchen.backends.mps import autograd as int8_autograd
from comfy_kitchen.backends.mps import packed_autograd
from comfy_kitchen.tensor.int8_utils import _build_hadamard

DEV = "mps"
ROT = 256  # ConvRot group, the value every shipped checkpoint uses
GPU_CAP_GIB = 3.0
QWEN_INT8 = "models/qwen21/diffusion_models/qwen_image_2.1_int8_convrot.safetensors"

# (K, N, why). The review asked for 3072/12288; the Qwen-Image 2.1 header says the hidden
# width is 4096, so both sets run.
SHAPES = [
    (3072, 3072, "review grid"),
    (3072, 12288, "review grid"),
    (12288, 3072, "review grid"),
    (12288, 12288, "review grid; no Qwen layer has this shape"),
    (4096, 4096, "Qwen-Image 2.1 attn.to_q/to_k/to_v/to_out.0"),
    (4096, 24576, "Qwen-Image 2.1 img_mlp.gate_up"),
    (12288, 4096, "Qwen-Image 2.1 img_mlp.out"),
]
FORMATS = ("int8", "w6a8", "w4a8", "w4a8_asym")
# (gradient dtype, gradient scale). 2e-5 is the size of a mean-loss gradient in training;
# fp16 cannot represent that range, so its case uses order-one gradients like the kernel test.
GRADS = ((torch.bfloat16, 2e-5), (torch.float16, 1.0), (torch.float32, 2e-5))
NAME = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


def synth_weight(n, k, seed):
    """Heavy-tailed bf16 weight with per-row and per-column scale spread. Returns (w, stats)."""
    g = torch.Generator().manual_seed(seed)
    row = torch.exp(torch.randn(n, 1, generator=g) * 0.6)
    col = torch.exp(torch.randn(1, k, generator=g) * 0.2)
    w = torch.empty(n, k, dtype=torch.bfloat16)
    s2 = s4 = 0.0
    for r0 in range(0, n, 1024):
        rows = min(1024, n - r0)
        z = torch.randn(rows, k, generator=g) * torch.exp(0.25 * torch.randn(rows, k, generator=g))
        z = z * row[r0:r0 + rows] * col * 0.012
        w[r0:r0 + rows] = z.to(torch.bfloat16)
        d = w[r0:r0 + rows].double()
        s2 += float((d ** 2).sum())
        s4 += float((d ** 4).sum())
    m2, m4 = s2 / (n * k), s4 / (n * k)
    return w, {"std": math.sqrt(m2), "kurtosis": m4 / m2 ** 2, "seed": seed}


class Layer:
    """One frozen quantized linear: packed operands, kernel calls, fp64 decode by row chunk."""

    def __init__(self, fmt, w):
        self.fmt, (self.n, self.k) = fmt, w.shape
        self.corr = self.cb = self.sr = None
        if fmt == "int8":
            self.q, self.sc = eager_int8.quantize_int8_convrot_weight(w, ROT)
            self.gs = None
        else:
            bits = 6 if fmt == "w6a8" else 4
            self.gs = 32 if bits == 6 else 16  # the recipe bench/probe_packed_training.py uses
            self.q, self.sr, self.sc, self.corr, self.cb = eager_packed.quantize_w4a8_int8_weight(
                w, group_size=self.gs, convrot_groupsize=ROT, codebook=bits == 4, bits=bits,
                symmetric=fmt != "w4a8_asym",
            )
        self.dev = [None if t is None else t.to(DEV) for t in (self.q, self.sr, self.sc, self.cb, self.corr)]

    def describe(self):
        sc = self.sc.double().reshape(-1)
        nz = sc[sc > 0]
        return {
            "packed_bytes": sum(t.numel() * t.element_size() for t in (self.q, self.sr, self.sc, self.cb, self.corr)
                                if t is not None),
            "group_size": self.gs, "has_codebook": self.cb is not None, "has_correction": self.corr is not None,
            "group_scale_dtype": None if self.sr is None else str(self.sr.dtype)[6:],
            "channel_scale_log2_spread": float(torch.log2(nz.max() / nz.min())),
        }

    def chunks(self, check=False):
        """Yield (row0, rows, W64) with W64 the dequantized weight in the original basis.

        With check=True also yields the repo dequantizer's fp32 output for the same rows.
        """
        h = _build_hadamard(ROT).double()
        step = max(1, (8 << 20) // self.k)  # 64 MB of fp64 at a time
        for r0 in range(0, self.n, step):
            c = slice(r0, min(self.n, r0 + step))
            if self.fmt == "int8":
                w = self.q[c].double() * self.sc[c].double().reshape(-1, 1)
                repo = eager_int8.dequantize_int8_convrot_weight(self.q[c], self.sc[c], ROT) if check else None
            else:
                corr = None if self.corr is None else self.corr[:, c]
                grid = eager_packed._dequant_int4_grouped_to_int8(self.q[c], self.sr[c], self.cb, self.gs)
                w = grid.double() * self.sc[c].double()[:, None]
                if corr is not None:
                    w += corr.double().t().repeat_interleave(self.gs, dim=1)
                repo = eager_packed.dequantize_w4a8_int8_weight(
                    self.q[c], self.sr[c], self.sc[c], codebook=self.cb, correction=corr,
                    group_size=self.gs, convrot_groupsize=ROT, output_dtype=torch.float32,
                ) if check else None
            rows = w.shape[0]
            w = (w.reshape(rows, self.k // ROT, ROT) @ h).reshape(rows, self.k)
            yield r0, rows, w, repo

    def grad_input(self, dy, out_dtype=None):
        q, sr, sc, cb, corr = self.dev
        if self.fmt == "int8":
            return int8_autograd.int8_linear_grad_input(dy, q, sc, True, ROT, out_dtype=out_dtype)
        return packed_autograd.w4a8_linear_grad_input(dy, q, sr, sc, cb, corr, self.gs, ROT, out_dtype)

    def trainable(self, x):
        q, sr, sc, cb, corr = self.dev
        if self.fmt == "int8":
            return int8_autograd.int8_linear_trainable(x, q, sc, None, True, ROT)
        return packed_autograd.w4a8_linear_trainable(x, q, sr, sc, cb, corr, None, self.gs, ROT)

    def decode_chunks(self, dtype):
        """How many decoded chunks the backward accumulates for a gradient of this dtype."""
        width = 4 if dtype == torch.float32 else 2
        return math.ceil(self.n / max(1, int8_autograd._MAX_BWD_DECODE_BYTES // (width * self.k)))

    def forward64(self, x):
        return torch.cat([x @ w.t() for _, _, w, _ in self.chunks()], dim=1)

    def backward64(self, g):
        out = torch.zeros(g.shape[0], self.k, dtype=torch.float64)
        for r0, rows, w, _ in self.chunks():
            out += g[:, r0:r0 + rows] @ w
        return out

    def dense(self, dtype):
        w = torch.empty(self.n, self.k, dtype=dtype, device=DEV)
        for r0, rows, w64, _ in self.chunks():
            w[r0:r0 + rows] = w64.to(dtype).to(DEV)
        return w


def compare(got, ref):
    """got against the fp64 reference: whole-tensor and worst-row agreement."""
    got = got.detach().cpu().double()
    err = got - ref
    rn, gn = ref.norm(dim=-1), got.norm(dim=-1)
    row_cos = (got * ref).sum(-1) / (rn * gn)
    return {
        "cosine": float((got * ref).sum() / (got.norm() * ref.norm())),
        "one_minus_cosine": float(1 - (got * ref).sum() / (got.norm() * ref.norm())),
        "rel_l2": float(err.norm() / ref.norm()),
        "max_abs_err": float(err.abs().max()),
        "ref_max_abs": float(ref.abs().max()),
        "ref_rms": float(ref.pow(2).mean().sqrt()),
        "worst_row_rel_l2": float((err.norm(dim=-1) / rn).max()),
        "worst_row_cosine": float(row_cos.min()),
        "nonfinite": int((~torch.isfinite(got)).sum()),
    }


class Gpu:
    peak = 0

    @classmethod
    def sample(cls):
        torch.mps.synchronize()
        cls.peak = max(cls.peak, torch.mps.driver_allocated_memory())


def single_layer(layer, weight, m, seed):
    """Kernel input gradient for every gradient dtype, against fp64 and the two comparators."""
    g = torch.Generator().manual_seed(seed)
    token = torch.exp(torch.randn(m, 1, generator=g))  # per-token gradient magnitudes differ
    dys = [(torch.randn(m, layer.n, generator=g) * token * mag).to(dtype).to(DEV) for dtype, mag in GRADS]
    stacked = torch.cat([dy.cpu().double() for dy in dys])
    ref = torch.zeros(stacked.shape[0], layer.k, dtype=torch.float64)
    dense = {d: torch.empty(layer.n, layer.k, dtype=d, device=DEV) for d in (torch.bfloat16, torch.float16)}
    src2 = err2 = deq2 = 0.0
    deq_max = 0.0
    for r0, rows, w64, repo in layer.chunks(check=True):
        ref += stacked[:, r0:r0 + rows] @ w64
        for d, w in dense.items():
            w[r0:r0 + rows] = w64.to(d).to(DEV)
        src = weight[r0:r0 + rows].double()
        src2 += float((src ** 2).sum())
        err2 += float(((w64 - src) ** 2).sum())
        deq2 += float(((repo.double() - w64) ** 2).sum())
        deq_max = max(deq_max, float((repo.double() - w64).abs().max()))
    w2 = src2  # same order of magnitude as the dequantized norm; used only for the two ratios
    info = layer.describe()
    info.update(conversion_rel_l2=math.sqrt(err2 / src2), repo_dequant_fp32_vs_fp64_rel_l2=math.sqrt(deq2 / w2),
                repo_dequant_fp32_vs_fp64_max_abs=deq_max)
    rows_out = []
    for i, (dy, (dtype, mag)) in enumerate(zip(dys, GRADS)):
        r = ref[i * m:(i + 1) * m]
        rec = {"grad_dtype": NAME[dtype], "grad_scale": mag, "decode_chunks": layer.decode_chunks(dtype),
               "kernel": compare(layer.grad_input(dy), r),
               "fp64_rounded_to_dtype": compare(r.to(dtype), r)}
        Gpu.sample()
        if dtype != torch.float32:
            rec["kernel_fp32_out"] = compare(layer.grad_input(dy, torch.float32), r)
            rec["dense_same_dtype_mps"] = compare(dy @ dense[dtype], r)
            Gpu.sample()
        if dtype == torch.bfloat16:  # the autograd Function must return what the direct call returns
            x = torch.zeros(m, layer.k, dtype=dtype, device=DEV, requires_grad=True)
            layer.trainable(x).backward(dy)
            rec["autograd_equals_direct_call"] = bool(torch.equal(x.grad, layer.grad_input(dy)))
            Gpu.sample()
        rows_out.append(rec)
    del dense
    return info, rows_out


def lora_delta(x, down, up):
    # train/train_lora.py LoRA.forward with alpha == rank
    return F.linear(F.linear(x, down.to(x.dtype)), up.to(x.dtype))


def mlp_chain(up, out, m, rank, seed):
    """gate_up -> SwiGLU -> out, a LoRA beside each frozen layer; gradients against fp64."""
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(m, up.k, generator=g).to(torch.bfloat16)
    gy = (torch.randn(m, out.n, generator=g) * 2e-5).to(torch.bfloat16)
    # The trainer starts `up` at zero, which makes dA exactly zero. Small nonzero `up` here so
    # both factors carry a gradient; the LoRA output is about 1% of the base output.
    init = [torch.randn(rank, up.k, generator=g) / rank, torch.randn(up.n, rank, generator=g) * 1e-3,
            torch.randn(rank, out.k, generator=g) / rank, torch.randn(out.n, rank, generator=g) * 1e-3]

    # fp64 reference, written out by hand so the dequantized weights are only ever held by chunk.
    a1, b1, a2, b2 = (t.double() for t in init)
    x, gd = x0.double(), gy.double()
    xa1 = x @ a1.t()
    u = (up.forward64(x) + xa1 @ b1.t()).requires_grad_(True)
    gate, val = u.chunk(2, dim=-1)
    h = F.silu(gate) * val
    hd = h.detach()
    ha2 = hd @ a2.t()
    y_ref = out.forward64(hd) + ha2 @ b2.t()
    t2 = gd @ b2
    dh = out.backward64(gd) + t2 @ a2
    (du,) = torch.autograd.grad(h, u, dh)
    t1 = du @ b1
    ref = {"dA_gate_up": t1.t() @ x, "dB_gate_up": du.t() @ xa1, "dA_out": t2.t() @ hd, "dB_out": gd.t() @ ha2,
           "dx": up.backward64(du) + t1 @ a1, "y_forward": y_ref}
    del u, h, hd, dh, du

    dense32 = [up.dense(torch.float32), out.dense(torch.float32)]
    Gpu.sample()
    results = []
    for dtype in (torch.bfloat16, torch.float32):
        dense = [w.to(dtype) for w in dense32]
        for arm in ("kernel", "dense_same_dtype_mps"):
            params = [t.clone().to(DEV).requires_grad_(True) for t in init]
            xd = x0.to(DEV, dtype).requires_grad_(True)
            lin = (up.trainable, out.trainable) if arm == "kernel" else (
                lambda t: F.linear(t, dense[0]), lambda t: F.linear(t, dense[1]))
            ud = lin[0](xd) + lora_delta(xd, params[0], params[1])
            gate, val = ud.chunk(2, dim=-1)
            hdev = F.silu(gate) * val
            yd = lin[1](hdev) + lora_delta(hdev, params[2], params[3])
            yd.backward(gy.to(DEV, dtype))
            Gpu.sample()
            got = {"dA_gate_up": params[0].grad, "dB_gate_up": params[1].grad, "dA_out": params[2].grad,
                   "dB_out": params[3].grad, "dx": xd.grad, "y_forward": yd}
            results.append({"activation_dtype": NAME[dtype], "arm": arm,
                            **{key: compare(val, ref[key]) for key, val in got.items()}})
            del params, xd, ud, hdev, yd, got
        del dense
    return results


def qwen_header_shapes():
    """Layer shapes of the INT8 Qwen-Image 2.1 file, from its safetensors header only."""
    root = os.environ.get("MK_ROOT")
    path = Path(root) / QWEN_INT8 if root else None
    if path is None or not path.is_file():
        return None
    with path.open("rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    count = collections.Counter(tuple(v["shape"]) for key, v in header.items()
                                if key != "__metadata__" and v["dtype"] == "I8")
    return {"file": Path(QWEN_INT8).name, "int8_weight_shapes_N_K": [[*shape, c] for shape, c in sorted(count.items())]}


def provenance():
    files = {}
    for module in (packed_autograd, int8_autograd, eager_packed, eager_int8):
        path = Path(module.__file__)  # vendor/ may be a symlink; name it by its place in this repo
        files[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()

    def run(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {"torch": torch.__version__, "macos": platform.mac_ver()[0], "chip": run("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_gib": round(int(run("sysctl", "-n", "hw.memsize") or 0) / 2 ** 30),
            "comfy_kitchen_commit": run("git", "-C", str(KITCHEN), "rev-parse", "HEAD"), "kernel_sources_sha256": files,
            "load_average": [round(v, 1) for v in os.getloadavg()], "swap": run("sysctl", "-n", "vm.swapusage"),
            "cpu_threads": torch.get_num_threads()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=512, help="gradient rows M for single layers")
    parser.add_argument("--chain-tokens", type=int, default=256)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--formats", default=",".join(FORMATS))
    parser.add_argument("--max-features", type=int, default=0, help="smoke test: skip shapes wider than this")
    args = parser.parse_args()
    torch.mps.set_per_process_memory_fraction(GPU_CAP_GIB * 2 ** 30 / torch.mps.recommended_max_memory())
    started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    result = {
        "script": "bench/probe_packed_backward_real_k.py", "started": started,
        "command": "bench/with_gpu.sh .venv/bin/python bench/probe_packed_backward_real_k.py "
                   + " ".join(a for a in sys.argv[1:] if not os.path.isabs(a)),
        "host": provenance(), "qwen_int8_header": qwen_header_shapes(),
        "method": {
            "weights": "synthetic: normal times a log-normal mixture (heavy tails), log-normal row and column scales, "
                       "cast to bf16, then quantized by comfy_kitchen.backends.eager. No checkpoint weight is read.",
            "reference": "fp64 on CPU: the gradient as stored (cast to fp64) times the dequantized weight decoded in fp64 "
                         "from the same packed tensors and rotated back with an fp64 Hadamard.",
            "kernel": "comfy_kitchen.backends.mps int8_linear_grad_input and w4a8_linear_grad_input on MPS, "
                      "output in the gradient dtype (what autograd uses), plus fp32 output to isolate the final rounding.",
            "fp64_rounded_to_dtype": "the fp64 reference rounded once to the output dtype: the floor for that dtype.",
            "dense_same_dtype_mps": "torch matmul on MPS of the same gradient with the dequantized weight held dense in "
                                    "that dtype.",
            "gradient": "normal, log-normal per-token magnitude, scaled by grad_scale", "convrot_group": ROT,
            "gpu_memory_cap_gib": GPU_CAP_GIB,
        },
        "single_layer": [], "mlp_chain": [],
    }
    for fmt in args.formats.split(","):
        qwen = {}
        for index, (k, n, why) in enumerate(SHAPES):
            if args.max_features and max(k, n) > args.max_features:
                continue
            t0 = time.time()
            weight, stats = synth_weight(n, k, seed=1000 + index)
            layer = Layer(fmt, weight)
            info, rows = single_layer(layer, weight, args.tokens, seed=2000 + index)
            rec = {"format": fmt, "K": k, "N": n, "M": args.tokens, "shape_origin": why, "source_weight": stats,
                   "quantized": info, "cases": rows, "seconds": round(time.time() - t0, 1)}
            result["single_layer"].append(rec)
            for case in rows:
                kern = case["kernel"]
                print(f"{fmt:9s} K={k:5d} N={n:5d} {case['grad_dtype']} rel_l2={kern['rel_l2']:.3e} "
                      f"1-cos={kern['one_minus_cosine']:.2e} max_abs={kern['max_abs_err']:.3e} "
                      f"floor={case['fp64_rounded_to_dtype']['rel_l2']:.3e}", flush=True)
            if why.endswith("img_mlp.gate_up") or why.endswith("img_mlp.out"):
                qwen[why.rsplit(".", 1)[1]] = layer
            del weight, layer
            torch.mps.empty_cache()
        if len(qwen) == 2:
            t0 = time.time()
            arms = mlp_chain(qwen["gate_up"], qwen["out"], args.chain_tokens, args.rank, seed=3000)
            result["mlp_chain"].append({"format": fmt, "K": qwen["gate_up"].k, "hidden": qwen["out"].k,
                                        "M": args.chain_tokens, "rank": args.rank, "activation": "swiglu",
                                        "arms": arms, "seconds": round(time.time() - t0, 1)})
            for arm in arms:
                print(f"{fmt:9s} chain {arm['activation_dtype']} {arm['arm']:22s} "
                      + " ".join(f"{key}={arm[key]['rel_l2']:.3e}" for key in ("dA_gate_up", "dB_gate_up", "dx")), flush=True)
        del qwen
        torch.mps.empty_cache()
    result["peak_mps_driver_allocated_gib"] = round(Gpu.peak / 2 ** 30, 3)
    result["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    result["host"]["load_average_end"] = [round(v, 1) for v in os.getloadavg()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
