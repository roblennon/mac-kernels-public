"""Loss curve PNG from a run's log.jsonl: training loss (10-step means, grey) and the fixed
validation loss (same images, noise and sigmas each time, black).

  train/.venv/bin/python train/plot_loss.py RUN_DIR OUT.png
  train/.venv/bin/python train/plot_loss.py --compare OUT.png RUN_A RUN_B ...   (validation curves only)
"""
import json
import sys

from PIL import Image, ImageDraw


def main(run, out):
    recs = [json.loads(line) for line in open(f"{run}/log.jsonl")]
    start = max(i for i, r in enumerate(recs) if "meta" in r)  # last run in the file
    recs = recs[start:]
    train = [(r["step"], r["loss"]) for r in recs if "loss" in r]
    val = [(r["step"], r["val_loss"]) for r in recs if "val_loss" in r]
    W, H, P = 900, 500, 60
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    xs = max(s for s, _ in train + val)
    ys = max(v for _, v in train + val) * 1.05

    def pt(s, v):
        return P + (W - 2 * P) * s / xs, H - P - (H - 2 * P) * v / ys

    d.line([pt(0, 0), pt(xs, 0)], fill="black")
    d.line([pt(0, 0), pt(0, ys)], fill="black")
    for k in range(0, 6):
        v = ys * k / 5
        d.text((5, pt(0, v)[1] - 6), f"{v:.3f}", fill="black")
    for s in range(0, xs + 1, max(50, xs // 10)):
        d.text((pt(s, 0)[0] - 10, H - P + 8), str(s), fill="black")
    d.line([pt(s, v) for s, v in train], fill=(150, 150, 150), width=2)
    d.line([pt(s, v) for s, v in val], fill="black", width=3)
    for s, v in val:
        x, y = pt(s, v)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill="black")
    d.text((P, 15), "grey: training loss, 10-step mean    black: fixed validation loss    x: step", fill="black")
    img.save(out)


def read_val(run):
    recs = [json.loads(line) for line in open(f"{run}/log.jsonl")]
    start = max(i for i, r in enumerate(recs) if "meta" in r)
    return [(r["step"], r["val_loss"]) for r in recs[start:] if "val_loss" in r]


def compare(out, runs):
    colors = [(0, 0, 0), (200, 40, 40), (40, 90, 200), (40, 150, 60)]
    curves = [read_val(r) for r in runs]
    W, H, P = 900, 500, 70
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    xs = max(s for c in curves for s, _ in c)
    lo = min(v for c in curves for _, v in c) * 0.98
    hi = max(v for c in curves for _, v in c) * 1.01

    def pt(s, v):
        return P + (W - 2 * P) * s / xs, H - P - (H - 2 * P) * (v - lo) / (hi - lo)

    d.line([pt(0, lo), pt(xs, lo)], fill="black")
    d.line([pt(0, lo), pt(0, hi)], fill="black")
    for k in range(6):
        v = lo + (hi - lo) * k / 5
        d.text((5, pt(0, v)[1] - 6), f"{v:.4f}", fill="black")
    for s in range(0, xs + 1, max(50, xs // 6)):
        d.text((pt(s, lo)[0] - 10, H - P + 8), str(s), fill="black")
    for j, (run, c) in enumerate(zip(runs, curves)):
        col = colors[j % len(colors)]
        d.line([pt(s, v) for s, v in c], fill=col, width=3)
        for s, v in c:
            x, y = pt(s, v)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)
        d.text((P + 10, 10 + 16 * j), f"{run.rstrip('/').split('/')[-1]}: fixed validation loss", fill=col)
    img.save(out)


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3:])
    else:
        main(sys.argv[1], sys.argv[2])
