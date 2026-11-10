"""Does F.linear on MPS take a different path for 2-D [M,K] and 3-D [B,T,K] inputs?
The 2-D microbenchmark showed F.linear slower than mm on a pre-transposed weight, while a
real model (3-D inputs) got slower when every linear was swapped. This times both shapes.
"""
import argparse, json, random, statistics, time
import torch, torch.nn.functional as F

def us(fn, iters=60):
    torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.mps.synchronize(); return (time.perf_counter() - t) / iters * 1e6

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
rng = random.Random(0); torch.manual_seed(0); res = []
for dt in (torch.float16, torch.bfloat16):
    for (K, N) in [(2048, 6144), (6144, 2048), (2048, 2048)]:
        W = torch.randn(N, K, device="mps", dtype=dt) / K ** .5; Wt = W.t().contiguous()
        for M in (1, 4, 8, 16, 24):
            x2 = torch.randn(M, K, device="mps", dtype=dt); x3 = x2.reshape(M, 1, K); x3b = x2.reshape(1, M, K)
            arms = {"lin_2d": lambda: F.linear(x2, W), "lin_3d_Bx1": lambda: F.linear(x3, W),
                    "lin_3d_1xT": lambda: F.linear(x3b, W), "mm_pre_t_2d": lambda: torch.mm(x2, Wt),
                    "matmul_pre_t_3d": lambda: torch.matmul(x3, Wt)}
            for f in arms.values():
                for _ in range(5): f()
            s = {k: [] for k in arms}
            for _ in range(7):
                o = list(arms); rng.shuffle(o)
                for k in o: s[k].append(us(arms[k]))
            med = {k: statistics.median(v) for k, v in s.items()}
            res.append({"dtype": str(dt)[6:], "K": K, "N": N, "M": M, "median_us": med})
            print(str(dt)[6:], K, N, M, {k: round(v, 1) for k, v in med.items()}, flush=True)
        del W, Wt; torch.mps.empty_cache()
json.dump({"torch": torch.__version__, "rows": res}, open(a.out, "w"), indent=1)
