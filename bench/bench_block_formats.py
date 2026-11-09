"""Per-call cost of the weight-only NVFP4 / MXFP8 / ConvRot W4A4 Metal kernels at real layer shapes.

Compares, per shape (M tokens, K in, N out), bf16 activations in and out, median ms:
  dense fp16    F.linear with a resident fp16 weight (the GEMM ceiling, 4x the memory of NVFP4)
  int8 w-only   comfy_kitchen int8_linear (ConvRot), the format these replace or sit beside
  nvfp4 w-only  nvfp4_linear (decode in Metal, normalized fp16 GEMM)
  nvfp4 deq     dequantize_nvfp4 to bf16 on the GPU, then F.linear in bf16 (a full-precision
                fallback; before this backend MPS could not run the fp8 decode at all)
  mxfp8 w-only, w4a4 w-only
Relative error of each against a float64 CPU oracle on the first shape. PYTHONPATH picks the kitchen.
Run through bench/with_gpu.sh.
"""
import json, os, statistics, time
import torch
import torch.nn.functional as F
import comfy_kitchen as ck
from comfy_kitchen.backends.eager import quantization as eq
from comfy_kitchen.backends.eager import convrot_w4a4 as ew

D = "mps"
SHAPES = [  # (name, M, K, N)
    ("h3-te q_proj", 512, 5120, 8192),
    ("h3-te up_proj", 512, 5120, 25600),
    ("h3-te down_proj", 512, 25600, 5120),
    ("krea 512 attn", 1056, 6144, 6144),
    ("krea 512 mlp.up", 1056, 6144, 16384),
    ("krea 512 mlp.down", 1056, 16384, 6144),
]


def timed(f, reps=7):
    f(); torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); torch.mps.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
    return round(statistics.median(ts), 2)


def main():
    rows = []
    for name, m, k, n in SHAPES:
        g = torch.Generator().manual_seed(0)
        w = torch.randn(n, k, generator=g) * 0.02
        x = torch.randn(m, k, generator=g).to(torch.bfloat16)
        ts = (w.abs().max() / (448 * 6)).float()
        q4, bs4 = eq.quantize_nvfp4(w, ts, pad_16x=True)
        qm, bm = eq.quantize_mxfp8(w, pad_32x=True)
        qw, sw = ew.quantize_convrot_w4a4_weight(w, 256)
        s8 = w.abs().amax(1) / 127
        q8 = torch.round(w / s8[:, None]).clamp(-127, 127).to(torch.int8)  # plain INT8 (no rotation), same cost
        xd = x.to(D)
        q4d, ts_d, bs4d = q4.to(D), ts.to(D), bs4.to(D)
        qmd, bmd, qwd, swd, q8d, s8d = qm.to(D), bm.to(D), qw.to(D), sw.to(D), q8.to(D), s8.to(D)
        wf16 = w.to(D, torch.float16)
        r = {"shape": name, "M": m, "K": k, "N": n}
        r["dense_fp16"] = timed(lambda: F.linear(xd.half(), wf16))
        r["int8_wonly"] = timed(lambda: ck.int8_linear(xd, q8d, s8d, None, torch.bfloat16))
        r["nvfp4_wonly"] = timed(lambda: ck.nvfp4_linear(xd, q4d, ts_d, bs4d))
        r["nvfp4_deq_bf16"] = timed(lambda: F.linear(xd, ck.dequantize_nvfp4(q4d, ts_d, bs4d, torch.bfloat16)))
        r["mxfp8_wonly"] = timed(lambda: ck.mxfp8_linear(xd, qmd, bmd))
        r["w4a4_wonly"] = timed(lambda: ck.convrot_w4a4_linear(xd, qwd, swd))
        r["weight_mb"] = {"fp16": round(n * k * 2 / 2**20), "nvfp4": round((q4.numel() + bs4.numel()) / 2**20),
                          "mxfp8": round((qm.numel() + bm.numel()) / 2**20), "w4a4": round(qw.numel() / 2**20)}
        if name == SHAPES[0][0]:
            x64 = x.double()
            def err(out, wd):
                ref = x64 @ wd.double().T
                return float((out.cpu().double() - ref).norm() / ref.norm())
            r["rel_err"] = {
                "nvfp4_wonly": err(ck.nvfp4_linear(xd, q4d, ts_d, bs4d), eq.dequantize_nvfp4(q4, ts, bs4, torch.float32)),
                "nvfp4_deq_bf16": err(F.linear(xd, ck.dequantize_nvfp4(q4d, ts_d, bs4d, torch.bfloat16)), eq.dequantize_nvfp4(q4, ts, bs4, torch.float32)),
                "mxfp8_wonly": err(ck.mxfp8_linear(xd, qmd, bmd), eq.dequantize_mxfp8(qm, bm, torch.float32)),
                "w4a4_wonly": err(ck.convrot_w4a4_linear(xd, qwd, swd), ew.dequantize_convrot_w4a4_weight(qw, sw, 256)),
            }
        print(json.dumps(r), flush=True)
        rows.append(r)
        del q4d, bs4d, qmd, bmd, qwd, q8d, wf16
        torch.mps.empty_cache()
    root = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
    with open(f"{root}/out/results.jsonl", "a") as f:
        f.write(json.dumps({"kind": "bench_block_formats", "kitchen": os.path.dirname(ck.__file__), "rows": rows}) + "\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
