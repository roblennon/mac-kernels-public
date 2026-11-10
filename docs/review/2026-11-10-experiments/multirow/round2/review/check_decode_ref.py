"""Which fp16 arm is closer to the truth? Loads Qwen3-1.7B in float32, teacher-forces the token
sequences saved by check_decode.py --save-dir, and compares stock fp16 and patched fp16 logits
against the float32 logits, per row (max and mean absolute difference over all 32 steps).

HF_HOME=<cache> HF_HUB_OFFLINE=1 python3 bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10 -- <python> <this file> --save-dir D --out f.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[6]
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


@torch.no_grad()
def forced_logits(model, ids, T):
    out = model(ids, use_cache=True)
    past = out.past_key_values
    logits = [out.logits[:, -1].float().cpu()]
    for s in range(T.shape[1] - 1):
        out = model(T[:, s:s + 1].to(ids.device), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits.append(out.logits[:, -1].float().cpu())
    return torch.stack(logits, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float32).to("mps").eval()
    res = {"torch": torch.__version__, "files": []}
    for f in sorted(Path(a.save_dir).glob("decode-b*.pt")):
        d = torch.load(f)
        R = forced_logits(model, d["ids"].to("mps"), d["T"])
        LB, PB = d["LB"].float(), d["PB"].float()
        rows = []
        for i in range(R.shape[0]):
            rows.append({"row": i, "stock_vs_fp32_max": (LB[i] - R[i]).abs().max().item(),
                         "patched_vs_fp32_max": (PB[i] - R[i]).abs().max().item(),
                         "stock_vs_fp32_mean": (LB[i] - R[i]).abs().mean().item(),
                         "patched_vs_fp32_mean": (PB[i] - R[i]).abs().mean().item(),
                         "patched_vs_stock_max": (PB[i] - LB[i]).abs().max().item()})
        summ = {"file": f.name, "rows": rows,
                "max": {k: max(r[k] for r in rows) for k in rows[0] if k != "row"},
                "mean_of_means": {k: sum(r[k] for r in rows) / len(rows) for k in ("stock_vs_fp32_mean", "patched_vs_fp32_mean")},
                "rows_patched_further_than_stock_max": sum(r["patched_vs_fp32_max"] > r["stock_vs_fp32_max"] for r in rows)}
        res["files"].append(summ)
        print(json.dumps({k: v for k, v in summ.items() if k != "rows"}), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
