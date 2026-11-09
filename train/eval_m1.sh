#!/bin/zsh
# Milestone 1 evaluation, one GPU-lock hold: the Raw-trained LoRA on Turbo (8 steps, cfg 1) at
# both checkpoints, on Raw (30 steps, cfg 3.5, empty-prompt negative: a zeroed negative turns Raw
# images to grain under CFG), and once through bench/run_krea2.py's LORA path.
#   bench/with_gpu.sh train/eval_m1.sh
set -e
HERE=${0:A:h}
ROOT=${MK_ROOT:-$HOME/mac-kernels-data}
RUN=$ROOT/train/runs/rwbflat_raw
PY=$HERE/.venv/bin/python
cp $RUN/krea2_rwbflat_raw_s250.safetensors $RUN/krea2_rwbflat_raw_s500.safetensors $ROOT/models/loras/
cd $HERE
$PY eval_lora.py --unet krea2_turbo_int8_convrot.safetensors --lora krea2_rwbflat_raw_s500.safetensors --steps 8 --cfg 1 --tag turbo
$PY eval_lora.py --unet krea2_turbo_int8_convrot.safetensors --lora krea2_rwbflat_raw_s250.safetensors --steps 8 --cfg 1 --tag turbo
$PY eval_lora.py --unet krea2_raw_int8_convrot.safetensors --lora krea2_rwbflat_raw_s500.safetensors --steps 30 --cfg 3.5 --neg "" --tag raw-negempty \
  --prompts "rwbflat style, a horse galloping on a beach" "rwbflat style, a teapot on a wooden table" "a portrait photo of an old fisherman"
cd ${HERE:h}
MODE=metal W=512 H=512 LORA=1 LORA_NAME=krea2_rwbflat_raw_s500.safetensors SEED=7 TAG=rwbflat-horse \
  PROMPT="rwbflat style, a horse galloping on a beach" $PY bench/run_krea2.py
