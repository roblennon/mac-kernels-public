"""Isolated attention over 28 separate caches, matching what transformers hands SDPA from a StaticCache:
keys of Lmax = ctx + 32 slots, a bool mask [1,1,T,Lmax] (True = attend), enable_gqa. Compared with the
earlier isolated setup (exactly ctx keys, no mask). Also index_copy_ cache writes, eager and compiled."""
import json, statistics, sys, time
import torch
import torch.nn.functional as F
sync = torch.mps.synchronize
L, HKV, HQ, D = 28, 8, 16, 128
out = {}
for ctx in (256, 4096):
    Lmax = ctx + 32
    K = [torch.randn(1, HKV, Lmax, D, device="mps", dtype=torch.float16) for _ in range(L)]
    V = [torch.randn(1, HKV, Lmax, D, device="mps", dtype=torch.float16) for _ in range(L)]
    for T in (1, 8):
        q = [torch.randn(1, HQ, T, D, device="mps", dtype=torch.float16) for _ in range(L)]
        pos = torch.arange(ctx, ctx + T, device="mps")
        bmask = (torch.arange(Lmax, device="mps")[None, :] <= pos[:, None])[None, None]  # [1,1,T,Lmax]
        fmask = torch.zeros(bmask.shape, device="mps", dtype=torch.float16).masked_fill(~bmask, float("-inf"))
        # causal within the new block, exact ctx+T keys, no mask needed only for T=1
        variants = {
            "exact_ctx_keys_nomask": lambda i: F.scaled_dot_product_attention(q[i], K[i][:, :, :ctx], V[i][:, :, :ctx], enable_gqa=True),
            "lmax_keys_boolmask": lambda i: F.scaled_dot_product_attention(q[i], K[i], V[i], attn_mask=bmask, enable_gqa=True),
            "lmax_keys_floatmask": lambda i: F.scaled_dot_product_attention(q[i], K[i], V[i], attn_mask=fmask, enable_gqa=True),
            "lmax_keys_nomask": lambda i: F.scaled_dot_product_attention(q[i], K[i], V[i], enable_gqa=True),
            "exact_ctxT_keys_boolmask": lambda i: F.scaled_dot_product_attention(q[i], K[i][:, :, :ctx + T], V[i][:, :, :ctx + T], attn_mask=bmask[..., :ctx + T], enable_gqa=True),
        }
        for name, f in variants.items():
            for _ in range(3):
                for i in range(L): f(i)
            sync()
            ts = []
            for _ in range(15):
                sync(); t = time.perf_counter()
                for i in range(L): f(i)
                sync(); ts.append((time.perf_counter() - t) * 1e3)
            out[f"{ctx}/T{T}/{name}"] = statistics.median(ts)
            print(ctx, T, name, round(statistics.median(ts), 3), "ms (28 layers)", flush=True)
        # cache write: index_copy_ of T rows into each of 28 K and V caches
        kn = torch.randn(1, HKV, T, D, device="mps", dtype=torch.float16)
        def write():
            for i in range(L):
                K[i].index_copy_(2, pos, kn); V[i].index_copy_(2, pos, kn)
        for _ in range(3): write()
        sync(); ts = []
        for _ in range(15):
            sync(); t = time.perf_counter(); write(); sync(); ts.append((time.perf_counter() - t) * 1e3)
        out[f"{ctx}/T{T}/index_copy_eager"] = statistics.median(ts)
        print(ctx, T, "index_copy_ eager x56", round(statistics.median(ts), 3), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
