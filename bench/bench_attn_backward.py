# Updated 2026-10-11: qwen21_1024 uses real corrected-cache text length (+18), not +200.
"""Attention forward + backward at 1024px DiT training shapes: torch SDPA autograd (MPS falls back
to the math path for the backward) against comfy_kitchen's chunked-recompute backward.

  bench/with_gpu.sh train/.venv/bin/python bench/bench_attn_backward.py
"""
import json
import os
import threading
import time

import torch
import torch.nn.functional as F

from comfy_kitchen.backends.mps.attention_autograd import chunked_attention

ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
PEAK = {"gb": 0.0, "on": True}


def poll():
    while PEAK["on"]:
        PEAK["gb"] = max(PEAK["gb"], torch.mps.driver_allocated_memory() / 2**30)
        time.sleep(0.01)


threading.Thread(target=poll, daemon=True).start()

# (name, heads, tokens): image tokens at 1024px = (1024 / 16)^2 = 4096, plus text
SHAPES = [
    ("krea2_512", 48, 1024 + 32),
    ("qwen21_1024", 32, 4096 + 18),  # corrected cyanoprint cache text lengths are 16–20
    ("krea2_1024", 48, 4096 + 32),
]
D = 128
rows = []
for name, h, n in SHAPES:
    torch.manual_seed(0)
    q, k, v, g = (torch.randn(1, h, n, D, device="mps", dtype=torch.float16) for _ in range(4))
    grads = {}
    for impl in ("torch_sdpa", "chunked_256MB", "chunked_64MB"):
        ts = [t.clone().requires_grad_(True) for t in (q, k, v)]
        if impl == "torch_sdpa":
            fn = lambda: F.scaled_dot_product_attention(*ts)
        else:
            cb = (256 if "256" in impl else 64) * 2**20
            fn = lambda: chunked_attention(*ts, chunk_bytes=cb)
        times = []
        for rep in range(3):
            for t in ts:
                t.grad = None
            torch.mps.synchronize()
            torch.mps.empty_cache()
            base = torch.mps.driver_allocated_memory() / 2**30
            PEAK["gb"] = base
            t0 = time.perf_counter()
            fn().backward(g)
            torch.mps.synchronize()
            times.append(time.perf_counter() - t0)
        grads[impl] = [t.grad.float() for t in ts]
        rec = {"shape": name, "heads": h, "tokens": n, "impl": impl, "fwd_bwd_s": round(min(times), 3),
               "peak_over_base_gb": round(PEAK["gb"] - base, 2)}
        rows.append(rec)
        print(rec, flush=True)
        del ts
    for impl in ("chunked_256MB", "chunked_64MB"):
        rel = [((a - b).norm() / b.norm()).item() for a, b in zip(grads[impl], grads["torch_sdpa"])]
        rows.append({"shape": name, "impl": impl, "grad_rel_vs_torch_qkv": [round(r, 5) for r in rel]})
        print(rows[-1], flush=True)
    # forward alone, for scale
    with torch.no_grad():
        F.scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        F.scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()
    rows.append({"shape": name, "impl": "sdpa_forward_only", "s": round(time.perf_counter() - t0, 3)})
    print(rows[-1], flush=True)
    del q, k, v, g, grads
PEAK["on"] = False
with open(f"{ROOT}/out/results.jsonl", "a") as f:
    f.write(json.dumps({"tag": "attn_backward_bench", "torch": torch.__version__, "rows": rows}) + "\n")
