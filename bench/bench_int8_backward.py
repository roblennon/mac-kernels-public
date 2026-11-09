"""Backward of the weight-only INT8 linear at Krea 2 training shapes (512px: 1024 image + ~32
text tokens), against dense fp16 and ai-toolkit-style "dequantize the whole layer per call".
Also checks the input gradient of real Krea 2 Raw layers against an fp32 CPU oracle.

  bench/with_gpu.sh train/.venv/bin/python bench/bench_int8_backward.py
"""
import json
import os
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

from comfy_kitchen.backends.mps.autograd import int8_linear_grad_input, int8_linear_trainable
from comfy_kitchen.backends.mps.linear import int8_linear
from comfy_kitchen.tensor.int8_utils import _build_hadamard

ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
RAW = f"{ROOT}/models/diffusion_models/krea2_raw_int8_convrot.safetensors"
M, G = 1056, 256
dev = "mps"


def ms(fn, reps=10):
    for _ in range(2):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def dequant(q, s):  # stored rotated weight -> original basis, fp32 on CPU
    n, k = q.shape
    w = q.float() * s.reshape(-1, 1).float()
    return (w.reshape(n, k // G, G) @ _build_hadamard(G)).reshape(n, k)


rows = []
with safe_open(RAW, "pt") as f:
    for name in ("blocks.10.attn.wq", "blocks.10.attn.wk", "blocks.10.mlp.up", "blocks.10.mlp.down", "blocks.27.mlp.down"):
        q, s = f.get_tensor(f"{name}.weight"), f.get_tensor(f"{name}.weight_scale")
        n, k = q.shape
        qd, sd = q.to(dev), s.to(dev)
        torch.manual_seed(0)
        x = torch.randn(M, k, device=dev, dtype=torch.float16)
        for dtype, mag in ((torch.float16, 1.0), (torch.bfloat16, 2e-5)):
            dy = (torch.randn(M, n) * mag).to(dev, dtype)
            gx = int8_linear_grad_input(dy, qd, sd, True, G, out_dtype=dtype)
            exact = dy.float().cpu() @ dequant(q, s)
            rel = ((gx.float().cpu() - exact).norm() / exact.norm()).item()
            rows.append({"layer": name, "shape": [M, k, n], "grad_dtype": str(dtype)[6:], "grad_mag": mag, "rel_err": rel})
            print(rows[-1], flush=True)
        dy = torch.randn(M, n, device=dev, dtype=torch.float16)
        wdense = dequant(q, s).to(dev, torch.float16)
        t_fwd = ms(lambda: int8_linear(x, qd, sd, out_dtype=torch.float16, convrot=True, convrot_groupsize=G))
        t_bwd = ms(lambda: int8_linear_grad_input(dy, qd, sd, True, G))
        t_dense_fwd = ms(lambda: F.linear(x, wdense))
        t_dense_bwd = ms(lambda: dy @ wdense)
        # ai-toolkit's MPS training path: dequantize the whole rotated layer, then the GEMM
        # (its rotation of the gradient is extra and not counted here).
        t_deq_bwd = ms(lambda: dy @ (qd.half() * sd.half()))

        xr = x.clone().requires_grad_(True)

        def step():
            int8_linear_trainable(xr, qd, sd, None, True, G).backward(dy)

        t_step = ms(step)
        rec = {"layer": name, "shape_MKN": [M, k, n], "fwd_ms": round(t_fwd, 2), "bwd_ms": round(t_bwd, 2),
               "fwd_plus_bwd_autograd_ms": round(t_step, 2), "dense_fp16_fwd_ms": round(t_dense_fwd, 2),
               "dense_fp16_bwd_ms": round(t_dense_bwd, 2), "dequant_whole_then_gemm_bwd_ms": round(t_deq_bwd, 2)}
        rows.append(rec)
        print(rec, flush=True)
        del wdense
os.makedirs(f"{ROOT}/out", exist_ok=True)
with open(f"{ROOT}/out/results.jsonl", "a") as fo:
    fo.write(json.dumps({"tag": "int8_backward_bench", "torch": torch.__version__, "rows": rows}) + "\n")
