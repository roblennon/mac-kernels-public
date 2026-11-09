"""Compare Qwen-Image 2.1 acceleration methods on one prompt and seed: time per step, total time,
and PSNR/SSIM against the 25-step base, with a labeled sheet.

Each case is one guarded run_qwen21.py process (seed 42, default prompt, 1024x1024, Metal INT8
path). Turbo sigmas are the checkpoint's saved 8-step schedule (Qwen/Qwen-Image-2.1-Turbo
model_index.json, October 9, 2026).

Usage: python bench/qwen_accel_compare.py [case ...]   (default: all)  -> out/qwen-accel-compare/
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data")))
REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv/bin/python"
TURBO_SIGMAS = "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0"
CASES = {
    "base-25": {"STEPS": "25"},
    "viggle-6": {"STEPS": "6", "LORA": "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors", "SIGMAS": "viggle"},
    "funacc-4": {"STEPS": "4", "LORA": "Qwen-Image-2.1-Fun-Acc-4Step-comfyui.safetensors", "SIGMAS": "funacc"},
    "turbo-int8-8": {"STEPS": "8", "UNET": "qwen_image_2.1_turbo_int8_convrot.safetensors", "SIGMAS": TURBO_SIGMAS},
    "turbo-bf16-8": {"STEPS": "8", "UNET": "qwen_image_2.1_turbo_bf16.safetensors", "SIGMAS": TURBO_SIGMAS},
}


def run(name, env):
    out = ROOT / "out/qwen-accel-compare"; out.mkdir(parents=True, exist_ok=True)
    log = out / f"{name}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([str(PY), "bench/gpu_runner.py", "--monitor", "--seconds", "900", "--", str(PY), "bench/run_qwen21.py"],
                            cwd=REPO, env={**os.environ, "W": "1024", "H": "1024", **env}, stdout=fh, stderr=subprocess.STDOUT, check=False).returncode
    result = None
    with open(log) as fh:
        for line in fh:
            if line.startswith("RESULT "):
                result = json.loads(line[7:])
    row = {"case": name, "exit": rc, "wall_s": round(time.time() - t0, 1), "log": str(log)}
    if result:
        row.update(tag=result["tag"], step_s=result["step_s_median"], times=result["times"], mps_gb=result["mps_alloc_gb_after_sample"],
                   nan=result["nan"], image=str(ROOT / "out" / f"{result['tag']}.png"))
        src = Path(row["image"])
        if src.exists():
            dst = out / f"{name}.png"; dst.write_bytes(src.read_bytes()); row["image"] = str(dst)
    print(json.dumps(row), flush=True)
    return row


def main():
    names = sys.argv[1:] or list(CASES)
    rows = [run(n, CASES[n]) for n in names]
    out = ROOT / "out/qwen-accel-compare"
    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ok = [r for r in rows if r.get("image") and Path(r["image"]).exists()]
    base = next((r for r in ok if r["case"] == "base-25"), None)
    if base and len(ok) > 1:
        others = [r["image"] for r in ok if r is not base]
        sim = subprocess.run([str(PY), "bench/img_similarity.py", base["image"], *others, "--sheet", str(out / "sheet.png")],
                             cwd=REPO, capture_output=True, text=True, check=False)
        print(sim.stdout.strip())
    print(f"{'case':14s} {'steps':>5s} {'s/step':>7s} {'sample s':>9s} {'total s':>8s} {'GPU GB':>7s}")
    for r in rows:
        if "times" in r:
            t = r["times"]; total = sum(t.values())
            print(f"{r['case']:14s} {CASES[r['case']]['STEPS']:>5s} {r['step_s']:7.2f} {t['sample']:9.1f} {total:8.1f} {r['mps_gb']:7.2f}")
        else:
            print(f"{r['case']:14s} failed (exit {r['exit']}), see {r['log']}")


if __name__ == "__main__":
    main()
