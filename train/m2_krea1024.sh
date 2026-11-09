#!/bin/zsh
# Krea 2 Raw INT8 at 1024px: a few LoRA steps with and without the chunked attention backward
# (time and memory only; the 512px rwbflat images are upscaled).   bench/with_gpu.sh train/m2_krea1024.sh
HERE=${0:A:h}; ROOT=${MK_ROOT:-$HOME/mac-kernels-data}; PY=$HERE/.venv/bin/python; LOGS=${HERE:h}/logs
cd $HERE
$PY train_lora.py --arch krea2 --res 1024 --profile-steps 3 --chunked-attn --out $ROOT/train/runs/profile_krea2_1024_chunked > $LOGS/train-profile-krea2-1024-chunked.log 2>&1
$PY train_lora.py --arch krea2 --res 1024 --profile-steps 3 --out $ROOT/train/runs/profile_krea2_1024_sdpa > $LOGS/train-profile-krea2-1024-sdpa.log 2>&1
