"""Compare what two LoRAs learned, layer by layer, on the CPU: for each shared layer the weight delta
dW = alpha/rank * B @ A, then the cosine similarity of the two deltas and their norm ratio. The
Frobenius products use <B1 A1, B2 A2> = trace((B1^T B2)(A2 A1^T)), so no dW is ever formed.

  train/.venv/bin/python train/compare_loras.py A.safetensors B.safetensors
"""
import sys

import torch
from safetensors.torch import load_file


def factors(path):
    sd = load_file(path)
    out = {}
    for k in sd:
        if k.endswith(".lora_A.weight"):
            base = k[:-len(".lora_A.weight")]
            a, b = sd[k].double(), sd[base + ".lora_B.weight"].double()
            alpha = float(sd.get(base + ".alpha", torch.tensor(a.shape[0])))
            out[base] = (a, b * (alpha / a.shape[0]))
    return out


def inner(f1, f2):
    (a1, b1), (a2, b2) = f1, f2
    return float(torch.trace((b1.T @ b2) @ (a2 @ a1.T)))


def main(pa, pb):
    da, db = factors(pa), factors(pb)
    cos, ratio = [], []
    for k in sorted(set(da) & set(db)):
        xy, xx, yy = inner(da[k], db[k]), inner(da[k], da[k]), inner(db[k], db[k])
        cos.append(xy / (xx * yy) ** 0.5)
        ratio.append((yy / xx) ** 0.5)
    cos_t, ratio_t = torch.tensor(cos), torch.tensor(ratio)
    print(f"{len(cos)} layers: cosine mean {cos_t.mean():.3f} median {cos_t.median():.3f} min {cos_t.min():.3f}; "
          f"norm ratio B/A median {ratio_t.median():.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
