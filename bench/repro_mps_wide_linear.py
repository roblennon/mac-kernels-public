"""Standalone repro: on MPS, F.linear is slower than a matmul against a pre-transposed contiguous
weight once the input width K passes about 8k: up to ~3.6x in fp16 and ~8.7x in fp32 at K = 32768
(M1 Max, Torch 2.14.1 and 2.16 nightly). Do not extend K past 32768 on M1/M2: larger K takes a
separate correctness reroute in PyTorch.

    python repro_mps_wide_linear.py [--m 4608] [--n 6144] [--iters 7] [--rounds 3]

No dependencies beyond torch. Times each variant with torch.mps.synchronize() around the call,
interleaves the variants round by round (so background load hits all of them alike), and reports
the minimum over all rounds, in milliseconds. The ratio column is F.linear over the contiguous
pre-transposed matmul; the last column is the largest difference of any variant from it.
"""
import argparse
import platform
import subprocess
import time

import torch
import torch.nn.functional as F


def chip():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return platform.processor()


def time_ms(fn, args, iters):
    best = float("inf")
    for _ in range(iters):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.mps.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4608)
    ap.add_argument("--n", type=int, default=6144)
    ap.add_argument("--ks", default="4096,6144,8192,10240,12288,16384,24576,32768")
    ap.add_argument("--dtypes", default="float16,float32")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    print(f"torch {torch.__version__}, macOS {platform.mac_ver()[0]}, {chip()}, "
          f"{torch.backends.mps.is_available() and 'MPS available'}")
    print(f"M={args.m}, N={args.n}; min ms over {args.rounds} rounds x {args.iters} iterations\n")
    dev = torch.device("mps")
    torch.manual_seed(0)

    # Operands (t["..."]): x [M, K] and w [N, K] contiguous, as nn.Linear holds them; x_nc the
    # same values as x in a non-contiguous view (row stride K + 64), which sends F.linear through
    # its MPSGraph path instead of the MPSNDArray one; xt = x.t().contiguous() [K, M] and
    # wt = w.t().contiguous() [K, N]. "A" is the left operand, "B" the right one.
    variants = {
        "F.linear(x, w)": lambda t: F.linear(t["x"], t["w"]),
        "x @ w.t()": lambda t: t["x"] @ t["w"].t(),
        "torch.mm(x, w.t())": lambda t: torch.mm(t["x"], t["w"].t()),
        "F.linear(x_nc, w)": lambda t: F.linear(t["x_nc"], t["w"]),
        "torch.mm(w, x.t()).t()": lambda t: torch.mm(t["w"], t["x"].t()).t(),
        "torch.mm(xt.t(), wt) (A transposed)": lambda t: torch.mm(t["xt"].t(), t["wt"]),
        "x @ wt (contiguous)": lambda t: t["x"] @ t["wt"],
        "torch.mm(x, wt)": lambda t: torch.mm(t["x"], t["wt"]),
    }
    header = "| dtype | K | " + " | ".join(variants) + " | F.linear / contiguous | max abs diff |"
    print(header)
    print("|" + "---|" * (len(variants) + 4))
    for dtype in [getattr(torch, d) for d in args.dtypes.split(",")]:
        for k in [int(v) for v in args.ks.split(",")]:
            x_nc = torch.randn(args.m, k + 64, device=dev, dtype=dtype)[:, :k]
            x = x_nc.contiguous()
            w = torch.randn(args.n, k, device=dev, dtype=dtype) / k ** 0.5
            t = {"x": x, "x_nc": x_nc, "xt": x.t().contiguous(), "w": w, "wt": w.t().contiguous()}
            for fn in variants.values():  # warm up (graph compilation)
                fn(t)
            best = {name: float("inf") for name in variants}
            for _ in range(args.rounds):
                for name, fn in variants.items():
                    best[name] = min(best[name], time_ms(fn, (t,), args.iters))
            ref = t["x"] @ t["wt"]
            diff = max((fn(t) - ref).abs().max().item() for fn in variants.values())
            ratio = best["F.linear(x, w)"] / best["x @ wt (contiguous)"]
            cells = " | ".join(f"{best[name]:.1f}" for name in variants)
            print(f"| {str(dtype).replace('torch.', '')} | {k} | {cells} | {ratio:.2f} | {diff:.1e} |", flush=True)
            del x, x_nc, w, t, ref
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
