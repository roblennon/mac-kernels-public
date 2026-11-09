# Overnight PERF-02A: Qwen 1024 attention fwd+bwd at real text length (not +200).
# Arms 128/256/512 MiB via explicit chunk_bytes; bf16; median of interleaved samples.
"""Bounded attention chunk microbench for the October 11 overnight brief.

  bench/with_gpu.sh .venv/bin/python bench/gpu_runner.py --monitor --seconds 540 -- \\
    .venv/bin/python bench/overnight_perf02a_attn.py
"""
from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from comfy_kitchen.backends.mps.attention_autograd import chunked_attention

ROOT = Path(os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data")))
OUT = ROOT / "out" / "overnight-20261011" / "perf02a"
HEADS = 32
DIM = 128
TEXT_TOKENS = int(os.environ.get("PERF02A_TEXT_TOKENS", "18"))
TOKENS = 4096 + TEXT_TOKENS
CHUNK_MIB = (128, 256, 512)
WARMUP = 2
MEASURED = 8
DTYPE = torch.bfloat16


def _sync() -> None:
    torch.mps.synchronize()


def _run_arm(chunk_mib: int, q0: torch.Tensor, k0: torch.Tensor, v0: torch.Tensor, g: torch.Tensor) -> dict:
    chunk_bytes = chunk_mib * 2**20
    samples: list[float] = []
    peaks: list[float] = []
    q = q0.clone().requires_grad_(True)
    k = k0.clone().requires_grad_(True)
    v = v0.clone().requires_grad_(True)
    for i in range(WARMUP + MEASURED):
        for t in (q, k, v):
            t.grad = None
        _sync()
        torch.mps.empty_cache()
        base = torch.mps.driver_allocated_memory() / 2**30
        t0 = time.perf_counter()
        out = chunked_attention(q, k, v, chunk_bytes=chunk_bytes)
        out.backward(g)
        _sync()
        elapsed = time.perf_counter() - t0
        peak = torch.mps.driver_allocated_memory() / 2**30
        if i >= WARMUP:
            samples.append(elapsed)
            peaks.append(peak - base)
    grads = [t.grad.detach().float().cpu() for t in (q, k, v)]
    del q, k, v, out
    return {
        "chunk_mib": chunk_mib,
        "samples_s": [round(x, 6) for x in samples],
        "median_s": round(statistics.median(samples), 6),
        "mean_s": round(statistics.fmean(samples), 6),
        "stdev_s": round(statistics.stdev(samples), 6) if len(samples) > 1 else 0.0,
        "peak_over_base_gib_median": round(statistics.median(peaks), 3),
        "grads": grads,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable")
    torch.manual_seed(0)
    q0 = torch.randn(1, HEADS, TOKENS, DIM, device="mps", dtype=DTYPE)
    k0 = torch.randn(1, HEADS, TOKENS, DIM, device="mps", dtype=DTYPE)
    v0 = torch.randn(1, HEADS, TOKENS, DIM, device="mps", dtype=DTYPE)
    g = torch.randn(1, HEADS, TOKENS, DIM, device="mps", dtype=DTYPE)

    arms: list[dict] = []
    grads_by_arm: dict[int, list[torch.Tensor]] = {}
    for mib in CHUNK_MIB:
        print(f"ARM start chunk_mib={mib}", flush=True)
        result = _run_arm(mib, q0, k0, v0, g)
        grads_by_arm[mib] = result.pop("grads")
        arms.append(result)
        print("ARM " + json.dumps(result), flush=True)

    baseline = next(a for a in arms if a["chunk_mib"] == 256)
    comparisons = []
    for arm in arms:
        if arm["chunk_mib"] == 256:
            continue
        rel = [
            float((a - b).norm() / b.norm())
            for a, b in zip(grads_by_arm[arm["chunk_mib"]], grads_by_arm[256])
        ]
        speedup = baseline["median_s"] / arm["median_s"] if arm["median_s"] else None
        layer_gain = (baseline["median_s"] - arm["median_s"]) / baseline["median_s"] if baseline["median_s"] else None
        extra_gib = arm["peak_over_base_gib_median"] - baseline["peak_over_base_gib_median"]
        comparisons.append({
            "chunk_mib": arm["chunk_mib"],
            "grad_rel_vs_256_qkv": [round(r, 6) for r in rel],
            "median_ratio_vs_256": round(speedup, 4) if speedup is not None else None,
            "layer_median_gain": round(layer_gain, 4) if layer_gain is not None else None,
            "extra_peak_gib_vs_256": round(extra_gib, 3),
            "clears_10pct_layer_bar": bool(layer_gain is not None and layer_gain >= 0.10 and extra_gib <= 2.0),
            "finite_grads": all(torch.isfinite(t).all().item() for t in grads_by_arm[arm["chunk_mib"]]),
        })

    receipt = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "tag": "overnight-perf02a-attn",
        "torch": torch.__version__,
        "dtype": str(DTYPE).replace("torch.", ""),
        "heads": HEADS,
        "tokens": TOKENS,
        "text_tokens": TEXT_TOKENS,
        "dim": DIM,
        "warmup": WARMUP,
        "measured": MEASURED,
        "arms": arms,
        "comparisons_vs_256": comparisons,
        "note": "Layer microbench only; trainer full-update probes remain PERF-02B / blocked.",
    }
    path = OUT / "perf02a-receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    (ROOT / "out" / "results.jsonl").open("a").write(json.dumps(receipt) + "\n")
    print("RESULT " + json.dumps({k: receipt[k] for k in receipt if k != "arms"} | {
        "arms": [{k: a[k] for k in a if k != "samples_s"} | {"n_samples": len(a["samples_s"])} for a in arms],
    }), flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
