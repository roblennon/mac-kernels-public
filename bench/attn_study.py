"""SDPA efficiency on MPS. MODE=calls replays the captured calls in out/attn_calls.jsonl; MODE=sweep runs controlled sweeps.

Effective TFLOPS = 4 * B * H * Nq * Nk * d / time (both matmuls of the forward).
calls: per distinct call, SDPA fp16 with the captured strides, contiguous fp16, bf16, mask vs none, and a manual
fp16 attention (QK^T bmm, softmax, PV bmm) chunked over queries. LABELS filters by label substring.
sweep: sequence length (d 128), head_dim at fixed FLOPs, plain fp16 GEMM ceiling, QK^T matmul alone.
"""
import json, math, os, time
import torch, torch.nn.functional as F

ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
dev = "mps"

def timed(fn, reps=None, budget=6.0):
    fn(); torch.mps.synchronize()
    t = time.perf_counter(); fn(); torch.mps.synchronize(); one = time.perf_counter() - t
    reps = reps or max(1, min(10, int(budget / max(one, 1e-4))))
    t = time.perf_counter()
    for _ in range(reps):
        o = fn()
    torch.mps.synchronize(); return o, (time.perf_counter() - t) / reps

def tflops(b, h, nq, nk, d, s):
    return round(4 * b * h * nq * nk * d / s / 1e12, 2)

def strided(desc, dtype, g):
    t = torch.empty_strided(desc["shape"], desc["stride"], dtype=dtype, device=dev)
    t.copy_(torch.randn(desc["shape"], generator=g, device=dev, dtype=dtype))  # copy respects the strides
    return t

def manual(q, k, v, mask=None, budget_bytes=2**30):
    # fp16 attention from MPS GEMMs: chunk queries so one score block stays under budget_bytes
    b, h, nq, d = q.shape; nk = k.shape[-2]
    c = max(1, min(nq, budget_bytes // (b * h * nk * q.element_size())))
    out = torch.empty(q.shape[:-1] + v.shape[-1:], dtype=q.dtype, device=q.device)
    scale = d ** -0.5
    for i in range(0, nq, c):
        s = torch.matmul(q[:, :, i:i + c], k.transpose(-1, -2)).mul_(scale)
        if mask is not None:
            m = mask[..., i:i + c, :] if mask.shape[-2] > 1 else mask
            s = s.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else s + m
        out[:, :, i:i + c] = torch.matmul(torch.softmax(s, dim=-1), v)
    return out

with torch.inference_mode():
    if os.environ.get("MODE", "calls") == "calls":
        seen = set(); want = os.environ.get("LABELS")
        for line in open(f"{ROOT}/out/attn_calls.jsonl"):
            c = json.loads(line)
            key = json.dumps({k: c[k] for k in ("label", "q", "k", "v", "mask")})
            if key in seen or (want and not any(w in c["label"] for w in want.split(","))):
                continue
            seen.add(key)
            b, h, nq, d = c["q"]["shape"]; nk = c["k"]["shape"][-2]
            if c["k"]["shape"][1] != h:
                continue  # GQA calls (text encoders) are out of scope
            g = torch.Generator(device=dev).manual_seed(0)
            dt = getattr(torch, c["q"]["dtype"])
            q, k, v = (strided(c[x], dt, g) for x in ("q", "k", "v"))
            mask = None
            if c["mask"]:
                md = getattr(torch, c["mask"]["dtype"])
                mask = (torch.rand(c["mask"]["shape"], generator=g, device=dev) > 0.1) if md == torch.bool else torch.zeros(c["mask"]["shape"], dtype=md, device=dev)
            res = dict(label=c["label"], count=c["count"], b=b, heads=h, nq=nq, nk=nk, d=d, dtype=c["q"]["dtype"], mask=None if mask is None else c["mask"]["dtype"],
                       contiguous=[c[x]["contiguous"] for x in ("q", "k", "v")])
            ref, t = timed(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask)); res.update(sdpa_ms=round(t * 1e3, 2), sdpa_tflops=tflops(b, h, nq, nk, d, t))
            qc, kc, vc = (x.contiguous() for x in (q, k, v))
            _, t = timed(lambda: F.scaled_dot_product_attention(qc, kc, vc, attn_mask=mask)); res.update(contig_tflops=tflops(b, h, nq, nk, d, t))
            if mask is not None:
                _, t = timed(lambda: F.scaled_dot_product_attention(q, k, v)); res.update(nomask_tflops=tflops(b, h, nq, nk, d, t))
            other = torch.bfloat16 if dt == torch.float16 else torch.float16
            qo, ko, vo = (x.to(other) for x in (q, k, v)); mo = mask if mask is None or mask.dtype == torch.bool else mask.to(other)
            _, t = timed(lambda: F.scaled_dot_product_attention(qo, ko, vo, attn_mask=mo)); res[f"{str(other)[6:]}_tflops"] = tflops(b, h, nq, nk, d, t)
            q16, k16, v16 = (x.to(torch.float16) for x in (q, k, v)); m16 = mask if mask is None or mask.dtype == torch.bool else mask.half()
            o, t = timed(lambda: manual(q16, k16, v16, m16)); res.update(manual_fp16_tflops=tflops(b, h, nq, nk, d, t),
                manual_cos=round(float(F.cosine_similarity(o.float().flatten(), ref.float().flatten(), dim=0)), 6))
            print("RESULT " + json.dumps(res), flush=True)
            del q, k, v, qc, kc, vc, qo, ko, vo, q16, k16, v16, o, ref; torch.mps.empty_cache()
    else:
        g = torch.Generator(device=dev).manual_seed(0)
        a = torch.randn(8192, 8192, device=dev, dtype=torch.float16, generator=g); bm = torch.randn(8192, 8192, device=dev, dtype=torch.float16, generator=g)
        _, t = timed(lambda: a @ bm); print("RESULT " + json.dumps(dict(kind="gemm_8192", tflops=round(2 * 8192 ** 3 / t / 1e12, 2))), flush=True)
        del a, bm
        for n in (1024, 2048, 4096, 8192, 16384, 32768):
            h = max(1, min(24, 2 ** 31 // (n * n)))
            q, k, v = (torch.randn(1, h, n, 128, device=dev, dtype=torch.float16, generator=g) for _ in range(3))
            _, t = timed(lambda: F.scaled_dot_product_attention(q, k, v))
            _, tqk = timed(lambda: torch.matmul(q, k.transpose(-1, -2)) if n <= 8192 else q)
            print("RESULT " + json.dumps(dict(kind="seq", n=n, heads=h, d=128, sdpa_ms=round(t * 1e3, 2), sdpa_tflops=tflops(1, h, n, n, 128, t),
                  qk_matmul_tflops=round(2 * h * n * n * 128 / tqk / 1e12, 2) if n <= 8192 else None)), flush=True)
            del q, k, v; torch.mps.empty_cache()
        for d in (64, 80, 96, 128, 160, 256):
            h = max(1, 3072 // d); n = 8192
            q, k, v = (torch.randn(1, h, n, d, device=dev, dtype=torch.float16, generator=g) for _ in range(3))
            _, t = timed(lambda: F.scaled_dot_product_attention(q, k, v))
            print("RESULT " + json.dumps(dict(kind="head_dim", n=n, heads=h, d=d, sdpa_ms=round(t * 1e3, 2), sdpa_tflops=tflops(1, h, n, n, d, t))), flush=True)
            del q, k, v; torch.mps.empty_cache()
