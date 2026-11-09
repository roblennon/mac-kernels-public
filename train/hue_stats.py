"""Mean hue (circular, degrees) and saturation of an image's coloured pixels, to tell turquoise
(~185) from prussian blue (~215). Usage: hue_stats.py IMG...  (prints one line per image)"""
import sys

import numpy as np
from PIL import Image


def hue_stats(path):
    hsv = np.asarray(Image.open(path).convert("RGB").convert("HSV")).astype(np.float64) / 255
    h, s, v = hsv[..., 0] * 2 * np.pi, hsv[..., 1], hsv[..., 2]
    m = (s > 0.2) & (v > 0.15)
    if m.sum() == 0:
        return float("nan"), 0.0, 0.0
    ang = np.degrees(np.arctan2(np.sin(h[m]).mean(), np.cos(h[m]).mean())) % 360
    return ang, float(s[m].mean()), float(m.mean())


if __name__ == "__main__":
    for p in sys.argv[1:]:
        a, s, f = hue_stats(p)
        print(f"{a:6.1f} deg  sat {s:.2f}  coloured {f:.2f}  {p.split('/')[-1]}")
