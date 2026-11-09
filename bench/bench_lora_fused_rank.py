"""Where the fused LoRA epilogue stops paying: fused (comfy_kitchen lora_* kwargs) vs ComfyUI's
separate up(down(x)) beside the quantized linear, by rank, plus a fused-QKV row-slice case.

Times are the LoRA's extra cost over the base layer, median of ROUNDS interleaved rounds, fp16
activations, ConvRot on. Usage: python bench/bench_lora_fused_rank.py
"""
import os
import time

import torch
import torch.nn.functional as F
import comfy_kitchen as ck
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout, w4a8_int8_linear

dev = torch.device("mps")
ROUNDS = int(os.environ.get("ROUNDS", "12"))
RANKS = [int(r) for r in os.environ.get("RANKS", "32,64,128,192,256,384").split(",")]
# (label, M tokens, K, N): H3 480x288x22f qkv and fc2, Krea 2 at 512 attention projection
SHAPES = [("h3-qkv", 3240, 5376, 21504), ("h3-fc2", 3240, 14336, 5376), ("krea-512", 1280, 6144, 6144),
          ("b-shape", 4608, 6144, 6144)]
if os.environ.get("SHAPES"):
    SHAPES = [s for s in SHAPES if s[0] in os.environ["SHAPES"].split(",")]
DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}[os.environ.get("DTYPE", "fp16")]


def weights(fmt, n, k):
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16)
    if fmt == "int8":
        return QuantizedTensor.from_float(w, "TensorWiseINT8Layout", per_channel=True, convrot=True).to(dev)
    return QuantizedTensor.from_float(w, "AsymW4A8Int8Layout", bits=6).to(dev)


def layer(fmt, wq, x, down=None, up=None):
    lora = {} if down is None else dict(lora_down=down, lora_up=up, lora_scale=1.0)
    if fmt == "int8":
        q, s = TensorWiseINT8Layout.get_plain_tensors(wq)
        return ck.int8_linear(x, q, s, None, x.dtype, convrot=True, convrot_groupsize=256, **lora)
    q, s_rel, s_ch, corr, cb = AsymW4A8Int8Layout.get_plain_tensors(wq)
    p = wq._params
    return w4a8_int8_linear(x, q, s_rel, s_ch, codebook=cb, correction=corr, bias=None,
                            group_size=p.group_size, convrot_groupsize=p.convrot_groupsize,
                            out_dtype=x.dtype, **lora)


def interleaved(fns):
    for f in fns.values():
        f(); f()
    t = {k: [] for k in fns}
    for _ in range(ROUNDS):
        for k, f in fns.items():
            torch.mps.synchronize(); t0 = time.perf_counter(); f(); torch.mps.synchronize()
            t[k].append(time.perf_counter() - t0)
    return {k: sorted(v)[len(v) // 2] * 1e3 for k, v in t.items()}


@torch.inference_mode()
def main():
    for fmt in ("int8", "w6a8"):
        for label, m, k, n in SHAPES:
            wq = weights(fmt, n, k)
            x = torch.randn(m, k, device=dev, dtype=DTYPE)
            for r in RANKS:
                down = torch.randn(r, k, device=dev, dtype=DTYPE) * 0.01
                up = torch.randn(n, r, device=dev, dtype=DTYPE) * 0.01

                def separate():
                    return layer(fmt, wq, x).add_(F.linear(F.linear(x, down), up))

                ms = interleaved({"base": lambda: layer(fmt, wq, x),
                                  "fused": lambda: layer(fmt, wq, x, down, up),
                                  "separate": separate})
                print(f"{fmt} {label} M={m} K={k} N={n} R={r}: base {ms['base']:.1f} ms, "
                      f"fused +{ms['fused'] - ms['base']:.1f}, separate +{ms['separate'] - ms['base']:.1f}", flush=True)
            if label == "h3-qkv":
                # three rank-64 LoRAs on q, k, v slices: zero-padded stack (fused) vs per-slice adds
                r, s = 64, n // 3
                downs = [torch.randn(r, k, device=dev, dtype=DTYPE) * 0.01 for _ in range(3)]
                ups = [torch.randn(s, r, device=dev, dtype=DTYPE) * 0.01 for _ in range(3)]
                dcat = torch.cat(downs)
                ucat = torch.zeros(n, 3 * r, device=dev, dtype=DTYPE)
                for i, u in enumerate(ups):
                    ucat[i * s:(i + 1) * s, i * r:(i + 1) * r] = u

                def per_slice():
                    out = layer(fmt, wq, x)
                    for i in range(3):
                        out.narrow(-1, i * s, s).add_(F.linear(F.linear(x, downs[i]), ups[i]))
                    return out

                ms = interleaved({"base": lambda: layer(fmt, wq, x),
                                  "fused-stack": lambda: layer(fmt, wq, x, dcat, ucat),
                                  "separate-stack": lambda: layer(fmt, wq, x).add_(F.linear(F.linear(x, dcat), ucat)),
                                  "separate-slices": per_slice})
                print(f"{fmt} {label} q/k/v slices 3x R=64: base {ms['base']:.1f} ms, " + ", ".join(
                    f"{k_} +{v - ms['base']:.1f}" for k_, v in ms.items() if k_ != "base"), flush=True)


main()
