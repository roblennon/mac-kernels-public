"""Is a one-row nn.Linear on MPS a slow path, and does its cost scale with the weight?

The model-research-lab's Q003 (2026-11-10) found that swapping one-row nn.Linear calls for
addmm on a pre-transposed weight cut a tiny model's cached decode step to about a third,
about 35 us per call, on torch 2.14.1. This times the variants per call across row counts,
shapes and dtypes, and checks they agree numerically.

Run under the GPU lock: python3 bench/gpu_runner.py --monitor --seconds 240 --rss-gib 8 -- \
  .venv/bin/python bench/bench_onerow_linear.py --out <file.json>
"""
import argparse, json, statistics, time
import torch
import torch.nn.functional as F

def timed(fn, iters):
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us per call

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dev = "mps"
    torch.manual_seed(0)
    shapes = [(256, 256), (256, 1024), (1024, 1024), (2048, 2048), (2048, 6144), (4096, 4096), (4096, 11008)]
    rows = [1, 2, 4, 8, 16, 64]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    if a.quick:
        shapes, rows, dtypes = shapes[:2], [1, 64], [torch.float32]
    out = {"torch": torch.__version__, "iters": a.iters, "reps": a.reps, "results": []}
    for dt in dtypes:
        for (K, N) in shapes:
            W = torch.randn(N, K, device=dev, dtype=dt) / K ** 0.5   # nn.Linear layout [out, in]
            b = torch.randn(N, device=dev, dtype=dt) * 0.01
            Wt = W.t().contiguous()                                   # pre-transposed [in, out]
            for M in rows:
                x = torch.randn(M, K, device=dev, dtype=dt)
                ops = {
                    "linear_bias": lambda: F.linear(x, W, b),
                    "linear_nobias": lambda: F.linear(x, W),
                    "addmm_pre_t": lambda: torch.addmm(b, x, Wt),
                    "mm_pre_t": lambda: torch.mm(x, Wt),
                    "mm_view_t": lambda: torch.mm(x, W.t()),
                }
                ref = (x.float() @ W.float().t() + b.float())
                agree = {}
                for name, fn in ops.items():
                    y = fn().float()
                    if "nobias" in name or name.startswith("mm"):
                        y = y + b.float()
                    agree[name] = float((y - ref).abs().max())
                times = {}
                for name, fn in ops.items():
                    for _ in range(10):
                        fn()
                    times[name] = statistics.median(timed(fn, a.iters) for _ in range(a.reps))
                row = {"dtype": str(dt).split(".")[-1], "K": K, "N": N, "M": M, "us": times, "max_abs_err": agree}
                out["results"].append(row)
                print(json.dumps({k: row[k] for k in ("dtype", "K", "N", "M")}),
                      {k: round(v, 1) for k, v in times.items()}, flush=True)
            del W, b, Wt
            torch.mps.empty_cache()
    json.dump(out, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
