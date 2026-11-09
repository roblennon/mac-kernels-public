"""PSNR and SSIM (luma, 11-px Gaussian window, sigma 1.5) of images against a reference; optional side-by-side sheet.

Usage: python bench/img_similarity.py REF.png A.png [B.png ...] [--sheet out.png]
"""
import json, math, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0)

def luma(x):
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[None, None]

def ssim(a, b):
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5 ** 2)); g = (g / g.sum())
    w = (g[:, None] * g[None, :])[None, None]
    mu_a, mu_b = F.conv2d(a, w), F.conv2d(b, w)
    va, vb = F.conv2d(a * a, w) - mu_a ** 2, F.conv2d(b * b, w) - mu_b ** 2
    cov = F.conv2d(a * b, w) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean())

args = sys.argv[1:]; sheet = None
if "--sheet" in args:
    i = args.index("--sheet"); sheet = args[i + 1]; args = args[:i] + args[i + 2:]
ref = load(args[0]); imgs = [ref]
for p in args[1:]:
    x = load(p); imgs.append(x)
    mse = float(((x - ref) ** 2).mean())
    print("SIM " + json.dumps(dict(ref=args[0].split("/")[-1], img=p.split("/")[-1], psnr=round(10 * math.log10(1 / mse), 2), ssim=round(ssim(luma(x), luma(ref)), 4))))
if sheet:
    Image.fromarray((torch.cat(imgs, dim=1).numpy() * 255).astype("uint8")).resize((256 * len(imgs), 256)).save(sheet)
