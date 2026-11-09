#!/usr/bin/env python3
"""Summarize a LoRA safetensors header without loading weights: naming family, ranks, alpha, targets.

  python bench/lora_header_report.py FILE [FILE ...]

Reports which naming convention the file uses (diffusers lora_A/lora_B, Kohya lora_down/lora_up,
or other), whether alpha tensors are present, DoRA or bias/norm patches, the rank range, the
distinct target module suffixes and the key prefixes. This is the first gate before any adapter
is admitted to a loader; it decides whether the plain A/B paths can take the file as-is.
"""
import collections
import json
from pathlib import Path
import re
import struct
import sys


def header(path):
    with Path(path).open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        data = json.loads(handle.read(length))
    meta = data.pop("__metadata__", {})
    return data, meta


def report(path):
    tensors, meta = header(path)
    keys = list(tensors)
    families = collections.Counter()
    ranks = set()
    alphas = dora = bias = norm = 0
    targets = collections.Counter()
    prefixes = collections.Counter()
    for key, entry in tensors.items():
        shape = entry["shape"]
        if ".lora_A." in key or ".lora_B." in key:
            families["diffusers A/B"] += 1
        elif ".lora_down." in key or ".lora_up." in key:
            families["kohya down/up"] += 1
        elif key.endswith(".alpha"):
            alphas += 1
            continue
        elif ".dora_scale" in key or "dora_" in key:
            dora += 1
            continue
        elif key.endswith((".diff_b", ".bias")):
            bias += 1
            continue
        elif key.endswith(".diff") or "norm" in key.split(".")[-2:][0]:
            norm += 1
            continue
        else:
            families["other"] += 1
        if ".lora_A." in key or ".lora_down." in key:
            ranks.add(shape[0] if len(shape) >= 2 else None)
        base = re.sub(r"\.(lora_[AB]|lora_down|lora_up)\.weight$", "", key)
        prefixes[base.split(".")[0]] += 1
        targets[".".join(base.split(".")[-2:])] += 1
    return {"file": str(path), "tensors": len(keys), "families": dict(families), "ranks": sorted(r for r in ranks if r),
            "alpha_tensors": alphas, "dora": dora, "bias_patches": bias, "norm_or_diff": norm,
            "prefixes": dict(prefixes.most_common(4)), "targets": dict(targets.most_common(24)),
            "metadata_keys": sorted(meta)[:12]}


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(json.dumps(report(arg), indent=1))
