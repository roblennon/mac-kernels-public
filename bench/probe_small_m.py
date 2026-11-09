"""Bounded experimental M=1 INT8 matvec; run only through the serialized GPU launcher.

Default K,N dimensions were read from layer 0 of the local Qwen3VL 8B INT8
encoder's safetensors header. Inputs/weights remain synthetic; this cannot prove
encoder speedup. No bias, residual, input activation or LoRA path is implemented.
Production ConvRot prologue, compute-dtype policy and exponent helpers are reused.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import torch
from comfy_kitchen.backends.mps import linear as lin
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation


def emit(**data):
    print(json.dumps(data), flush=True)


class Candidate:
    def __init__(self):
        source = Path(__file__).with_name("small_m_int8.metal").read_text()
        self.lib = torch.mps.compile_shader(lin.METAL_COMMON + source)

    def __call__(self, x, weight, scale, *, convrot=True, out_dtype=torch.bfloat16):
        n, k = weight.shape
        if (x.shape != (1, k) or k % 4 or k > 16384 or weight.dtype != torch.int8
                or not weight.is_contiguous() or x.device.type != "mps"
                or weight.device != x.device or scale.device != x.device
                or scale.dtype != torch.float32 or not scale.is_contiguous()
                or scale.numel() not in (1, n) or (convrot and k % 256)):
            raise ValueError("Probe supports only contiguous MPS M=1 INT8, K<=16384, K%4=0, fp32 scales")
        cdtype = lin._compute_dtype(x, k)
        xc, exponent = lin.prologue(x, 256 if convrot else None, cdtype)
        out = torch.empty((1, n), dtype=out_dtype, device=x.device)
        kernel = f"small_m_i8_{lin.DTYPE_SUFFIX[cdtype]}_{lin.DTYPE_SUFFIX[out_dtype]}"
        getattr(self.lib, kernel)(out, xc, exponent, weight, scale, k, 0 if scale.numel() == 1 else 1,
                                  threads=(256, n), group_size=(256, 1))
        return out


def reference(x, q, scales, convrot):
    # Independent CPU fp32 oracle; only selected channels in large-shape probes.
    value = x.float()
    if convrot:
        value = _rotate_activation(value, _build_hadamard(256, device="cpu", dtype=torch.float32), 256)
    return value @ (q.float() * scales.reshape(-1, 1)).t()


def errors(got, want):
    got, want = got.cpu().double(), want.cpu().double()
    return float((got - want).norm() / want.norm().clamp_min(1e-30))


def verify(candidate, x, q, s, *, convrot=True, out_dtype=torch.bfloat16, name="random"):
    gx, gq, gs = x.to("mps"), q.to("mps"), s.to("mps")
    got = candidate(gx, gq, gs, convrot=convrot, out_dtype=out_dtype)
    baseline = lin.int8_linear(gx, gq, gs, convrot=convrot, out_dtype=out_dtype)
    # Index the result of the full measured shape, not a separate smaller kernel.
    channels = min(q.shape[0], 128)
    want = reference(x, q[:channels], s if s.numel() == 1 else s[:channels], convrot).to(out_dtype)
    threshold = 0.01 if out_dtype == torch.bfloat16 else 0.003 if x.dtype != torch.float32 else 0.0002
    error = errors(got[:, :channels], want)
    parity = errors(got, baseline)
    finite = bool(torch.isfinite(got).all()) and bool(torch.isfinite(want).all())
    emit(event="correctness", case=name, shape=[1, q.shape[1], q.shape[0]], input_dtype=str(x.dtype),
         out_dtype=str(out_dtype), convrot=convrot, oracle_rel_l2=error,
         production_rel_l2=parity, threshold=threshold, finite=finite)
    if not finite or not math.isfinite(error) or error > threshold or parity > threshold:
        raise AssertionError("Numerical gate failed; do not promote this candidate")
    del got, baseline
    return gx, gq, gs


def extremes(candidate):
    q = torch.ones(4, 256, dtype=torch.int8)
    cases = [
        ("dot_bound", torch.ones(1, 256).bfloat16(), q * 100, torch.ones(4), False, torch.bfloat16),
        ("large_scale", torch.full((1, 256), 0.001 / 256), q, torch.tensor([1e5, 3e7, 1e-30, -1e5]), False, torch.float32),
        ("tiny_input", torch.full((1, 256), 2.0**-26).bfloat16(), q, torch.ones(1), False, torch.float16),
        ("zero_scale", torch.ones(1, 256).bfloat16(), q, torch.zeros(1), True, torch.bfloat16),
    ]
    h = _build_hadamard(256, device="cpu", dtype=torch.float32)
    onehot = torch.zeros_like(q)
    onehot[:, 0] = 1
    cases += [
        ("rotation_growth", (10000 * h[:, 0].sign()).reshape(1, -1).half(), onehot, torch.full((4,), 0.001), True, torch.bfloat16),
        ("near_max", torch.cat([torch.full((1, 1), 2e38), torch.zeros(1, 255)], dim=1).bfloat16(), onehot, torch.full((4,), 0.001), True, torch.bfloat16),
    ]
    for name, x, weight, scale, rotate, dtype in cases:
        tensors = verify(candidate, x, weight, scale, convrot=rotate, out_dtype=dtype, name=name)
        del tensors


def timed(fn):
    torch.mps.synchronize()
    started = time.perf_counter()
    out = fn()
    torch.mps.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    del out
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="4096,1024;4096,4096;4096,12288;12288,4096")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--max-seconds", type=float, default=90)
    args = parser.parse_args()
    shapes = [tuple(map(int, s.split(","))) for s in args.shapes.split(";")]
    if (not 1 <= args.rounds <= 20 or not 0 < args.max_seconds <= 180
            or not math.isfinite(args.max_seconds) or len(shapes) > 6
            or any(len(s) != 2 or min(s) < 1 or max(s) > 16384 or s[0] % 256
                   or s[0] * s[1] > 6144 * 16384 for s in shapes)):
        parser.error("Bounded probe: <=6 shapes, K%256=0, K,N<=16384, KN<=100663296, rounds 1..20, seconds 0..180")
    if os.environ.get("COMFY_KITCHEN_MPS_COMPUTE_DTYPE"):
        parser.error("Leave compute dtype override unset for this probe")
    torch.set_num_threads(4)
    started = time.perf_counter()
    candidate = Candidate()
    rng = random.Random(20261003)
    generator = torch.Generator().manual_seed(20261003)
    emit(event="metadata", production=lin.__file__, torch=torch.__version__, shapes=shapes,
         scope="synthetic M=1 at Qwen3VL-8B layer dimensions; no model speedup claim")
    with torch.inference_mode():
        extremes(candidate)
        for k, n in shapes:
            if time.perf_counter() - started > args.max_seconds:
                emit(event="time_budget_stop")
                return
            q = torch.randint(-127, 128, (n, k), dtype=torch.int8, generator=generator)
            scale = torch.rand(n, generator=generator) * 0.0005 + 0.0001
            x = torch.randn(1, k, generator=generator).bfloat16()
            gx, gq, gs = verify(candidate, x, q, scale)
            funcs = {"candidate": lambda: candidate(gx, gq, gs),
                     "production": lambda: lin.int8_linear(gx, gq, gs, convrot=True)}
            for fn in funcs.values():
                timed(fn)
            samples = {name: [] for name in funcs}
            for _ in range(args.rounds):
                if time.perf_counter() - started > args.max_seconds:
                    emit(event="time_budget_stop")
                    return
                order = list(funcs)
                rng.shuffle(order)
                for name in order:
                    samples[name].append(timed(funcs[name]))
            medians = {name: statistics.median(values) for name, values in samples.items()}
            ratio = medians["production"] / medians["candidate"]
            emit(event="measurement", m=1, k=k, n=n, median_ms=medians, speedup=ratio,
                 samples_ms=samples, live_gib=torch.mps.current_allocated_memory()/2**30,
                 driver_gib=torch.mps.driver_allocated_memory()/2**30,
                 disposition="candidate only" if ratio >= 1.1 else "reject for this shape")
            del gx, gq, gs, q, x, scale, funcs
            gc.collect()
            torch.mps.empty_cache()
    emit(event="complete", elapsed_s=time.perf_counter()-started,
         promotion="requires real encoder shape frequency, total encode latency, and different-family review")


if __name__ == "__main__":
    main()
