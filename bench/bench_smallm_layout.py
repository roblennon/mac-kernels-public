"""Confirm the small-M layout slowdown on MPS: F.linear on an [out,in] weight against mm on a
pre-transposed [in,out] weight, every row count 1..32, real model shapes, interleaved arms in a
shuffled order per repeat, with an A/A arm (the same mm twice) as the noise floor.

python3 bench/gpu_runner.py --monitor --seconds 280 --rss-gib 8 -- .venv/bin/python bench/bench_smallm_layout.py --out f.json
"""
import argparse, json, random, statistics, time
import torch
import torch.nn.functional as F

def per_call_us(fn, iters):
    torch.mps.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=7); ap.add_argument("--iters", type=int, default=60)
    a = ap.parse_args()
    rng = random.Random(0); torch.manual_seed(0)
    shapes = {"qwen3-1.7b_up": (2048, 6144), "qwen3-1.7b_down": (6144, 2048),
              "qwen3-4b_up": (2560, 9728), "llama7b_up": (4096, 11008)}
    out = {"torch": torch.__version__, "reps": a.reps, "iters": a.iters, "rows": []}
    for dt in (torch.float16, torch.bfloat16):
        for name, (K, N) in shapes.items():
            W = torch.randn(N, K, device="mps", dtype=dt) / K ** 0.5
            Wt = W.t().contiguous()
            for M in range(1, 33):
                x = torch.randn(M, K, device="mps", dtype=dt)
                arms = {"linear": lambda: F.linear(x, W), "mm_pre_t": lambda: torch.mm(x, Wt),
                        "mm_pre_t_again": lambda: torch.mm(x, Wt)}
                for f in arms.values():
                    for _ in range(5): f()
                samples = {k: [] for k in arms}
                for _ in range(a.reps):
                    order = list(arms); rng.shuffle(order)
                    for k in order:
                        samples[k].append(per_call_us(arms[k], a.iters))
                med = {k: statistics.median(v) for k, v in samples.items()}
                row = {"dtype": str(dt).split(".")[-1], "shape": name, "K": K, "N": N, "M": M,
                       "median_us": med, "samples_us": samples,
                       "ratio_linear_over_mm": med["linear"] / med["mm_pre_t"],
                       "ratio_aa": med["mm_pre_t_again"] / med["mm_pre_t"]}
                out["rows"].append(row)
                print(row["dtype"], name, M, {k: round(v, 1) for k, v in med.items()},
                      round(row["ratio_linear_over_mm"], 2), round(row["ratio_aa"], 2), flush=True)
            del W, Wt; torch.mps.empty_cache()
            json.dump(out, open(a.out, "w"), indent=1)
    json.dump(out, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
