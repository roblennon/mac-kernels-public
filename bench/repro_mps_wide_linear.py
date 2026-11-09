"""Standalone repro: on MPS, F.linear is slower than a matmul against a pre-transposed contiguous
weight once the input width K passes about 8k: up to ~3.6x in fp16 and ~8.7x in fp32 at K = 32768
(M1 Max, Torch 2.14.1 and 2.16 nightly). Do not extend K past 32768 on M1/M2: larger K takes a
separate correctness reroute in PyTorch.

    python repro_mps_wide_linear.py [--m 4608] [--n 6144] [--iters 7] [--rounds 3] [--device mps]

No dependencies beyond torch. Times each variant with a device synchronize around the call
(torch.mps.synchronize() on mps, torch.cuda.synchronize() on cuda), interleaves the variants
round by round (so background load hits all of them alike), and reports the minimum over all
rounds, in milliseconds. The ratio column is F.linear over the contiguous pre-transposed matmul;
the last column is the largest difference of any variant from it.

--device cuda runs the same sweep on an NVIDIA GPU as a control (modal/modal_app.py imports
sweep() for that); the default, mps, is the repro.
"""
import argparse
import platform
import subprocess
import time

import torch
import torch.nn.functional as F

DEFAULT_KS = "4096,6144,8192,10240,12288,16384,24576,32768"

# Operands (t["..."]): x [M, K] and w [N, K] contiguous, as nn.Linear holds them; x_nc the
# same values as x in a non-contiguous view (row stride K + 64), which sends F.linear through
# its MPSGraph path instead of the MPSNDArray one; xt = x.t().contiguous() [K, M] and
# wt = w.t().contiguous() [K, N]. "A" is the left operand, "B" the right one.
VARIANTS = {
    "F.linear(x, w)": lambda t: F.linear(t["x"], t["w"]),
    "x @ w.t()": lambda t: t["x"] @ t["w"].t(),
    "torch.mm(x, w.t())": lambda t: torch.mm(t["x"], t["w"].t()),
    "F.linear(x_nc, w)": lambda t: F.linear(t["x_nc"], t["w"]),
    "torch.mm(w, x.t()).t()": lambda t: torch.mm(t["w"], t["x"].t()).t(),
    "torch.mm(xt.t(), wt) (A transposed)": lambda t: torch.mm(t["xt"].t(), t["wt"]),
    "x @ wt (contiguous)": lambda t: t["x"] @ t["wt"],
    "torch.mm(x, wt)": lambda t: torch.mm(t["x"], t["wt"]),
}


def chip():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return platform.processor()


def backend(dev):
    """(synchronize, empty_cache) for the device; no-ops on cpu."""
    if dev.type == "mps":
        return torch.mps.synchronize, torch.mps.empty_cache
    if dev.type == "cuda":
        return torch.cuda.synchronize, torch.cuda.empty_cache
    return (lambda: None), (lambda: None)


def time_ms(fn, args, iters, sync=None):
    sync = sync or torch.mps.synchronize
    best = float("inf")
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn(*args)
        sync()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def sweep(dev, m, n, ks, dtypes, iters, rounds):
    """Yield one dict per (dtype, K): {"dtype", "K", "ms": {variant: min ms}, "ratio", "max_abs_diff"}."""
    sync, empty_cache = backend(dev)
    torch.manual_seed(0)
    for dtype in [getattr(torch, d) for d in dtypes]:
        for k in ks:
            x_nc = torch.randn(m, k + 64, device=dev, dtype=dtype)[:, :k]
            x = x_nc.contiguous()
            w = torch.randn(n, k, device=dev, dtype=dtype) / k ** 0.5
            t = {"x": x, "x_nc": x_nc, "xt": x.t().contiguous(), "w": w, "wt": w.t().contiguous()}
            for fn in VARIANTS.values():  # warm up (graph compilation)
                fn(t)
            best = {name: float("inf") for name in VARIANTS}
            for _ in range(rounds):
                for name, fn in VARIANTS.items():
                    best[name] = min(best[name], time_ms(fn, (t,), iters, sync))
            ref = t["x"] @ t["wt"]
            diff = max((fn(t) - ref).abs().max().item() for fn in VARIANTS.values())
            ratio = best["F.linear(x, w)"] / best["x @ wt (contiguous)"]
            yield {"dtype": str(dtype).replace("torch.", ""), "K": k, "ms": best, "ratio": ratio, "max_abs_diff": diff}
            del x, x_nc, w, t, ref
            empty_cache()


def describe(dev):
    if dev.type == "mps":
        return (f"torch {torch.__version__}, macOS {platform.mac_ver()[0]}, {chip()}, "
                f"{torch.backends.mps.is_available() and 'MPS available'}")
    if dev.type == "cuda":
        return f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(dev)}"
    return f"torch {torch.__version__}, {platform.platform()}, cpu"


def format_row(row):
    cells = " | ".join(f"{row['ms'][name]:.1f}" for name in VARIANTS)
    return f"| {row['dtype']} | {row['K']} | {cells} | {row['ratio']:.2f} | {row['max_abs_diff']:.1e} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4608)
    ap.add_argument("--n", type=int, default=6144)
    ap.add_argument("--ks", default=DEFAULT_KS)
    ap.add_argument("--dtypes", default="float16,float32")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--device", default="mps", help="mps (the repro) or cuda (control)")
    args = ap.parse_args()

    dev = torch.device(args.device)
    print(describe(dev))
    print(f"M={args.m}, N={args.n}; min ms over {args.rounds} rounds x {args.iters} iterations\n")

    header = "| dtype | K | " + " | ".join(VARIANTS) + " | F.linear / contiguous | max abs diff |"
    print(header)
    print("|" + "---|" * (len(VARIANTS) + 4))
    ks = [int(v) for v in args.ks.split(",")]
    for row in sweep(dev, args.m, args.n, ks, args.dtypes.split(","), args.iters, args.rounds):
        print(format_row(row), flush=True)


if __name__ == "__main__":
    main()
