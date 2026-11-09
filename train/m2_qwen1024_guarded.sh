#!/bin/zsh
# Continue the reviewed seed-1 pilot to step 300. This script owns its GPU lock.
set -eu
if (( $# )); then
  print -u2 "This launcher accepts no arguments; its experiment configuration is fixed."
  exit 2
fi
HERE=${0:A:h}
REPO=${HERE:h}
export MK_ROOT=${MK_ROOT:-$HOME/mac-kernels-data}
OUT=$MK_ROOT/train/runs/cyan_qwen_1024_int8_seed1_20261003
PY=$HERE/.venv/bin/python
mkdir -p "$REPO/logs"
exec "$PY" "$REPO/bench/gpu_runner.py" --monitor --seconds 14400 \
  --rss-gib 16 --swap-growth-gib 0.25 --min-free-percent 25 -- \
  "$PY" - "$HERE/train_lora.py" "$OUT" "$MK_ROOT/train/datasets/cyanoprint" \
  >> "$REPO/logs/train-qwen-1024-seed1-guarded.log" 2>&1 <<'PY'
import json
from pathlib import Path
import runpy
import sys

import torch

script, out, dataset = map(Path, sys.argv[1:])
name = "qwen21_cyanoprint_1024_int8_seed1"
expected = {
    "arch": "qwen21", "unet": "qwen_image_2.1_int8_convrot.safetensors",
    "res": 1024, "dtype": "bf16", "rank": 16, "lr": 2e-4, "seed": 1,
    "chunked_attn": True, "no_checkpoint": False, "name": name,
}
with (out / "log.jsonl").open() as log:
    meta = next((record["meta"] for line in log if (record := json.loads(line)).get("meta")), None)
if meta is None or any(meta.get(key) != value for key, value in expected.items()):
    raise SystemExit("Pilot configuration does not match this fixed seed-1 experiment.")
if Path(meta["dataset"]).resolve() != dataset.resolve():
    raise SystemExit("Pilot dataset does not match this experiment.")
state = torch.load(out / "state.pt", map_location="cpu", mmap=True)
step = state["step"]
del state
if not isinstance(step, int) or not 8 <= step < 300:
    raise SystemExit(f"Expected a completed eight-step pilot or partial continuation; found step {step}.")
if not (out / f"{name}_s8.safetensors").is_file():
    raise SystemExit("The eight-step pilot adapter is missing; finish and verify the pilot first.")

torch.mps.set_per_process_memory_fraction(0.6)
print("MPS_CAP", torch.mps.recommended_max_memory() * 0.6 / 2**30, "GiB", flush=True)
print("Continuing reviewed Qwen seed-1 run from", step, "to 300", flush=True)
sys.path.insert(0, str(script.parent))
sys.argv = [str(script), "--arch", "qwen21", "--dataset", str(dataset),
            "--unet", expected["unet"], "--res", "1024", "--dtype", "bf16",
            "--chunked-attn", "--rank", "16", "--lr", "2e-4", "--seed", "1",
            "--steps", "300", "--state-every", "10", "--save-every", "50",
            "--val-every", "50", "--log-every", "10", "--out", str(out),
            "--name", name, "--resume"]
runpy.run_path(str(script), run_name="__main__")
PY
