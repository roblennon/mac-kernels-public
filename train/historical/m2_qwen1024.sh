#!/bin/zsh
# Qwen-Image 2.1 INT8 LoRA at 1024px with the chunked attention backward, 300 steps in sessions of 150
# so each fits a GPU-lock hold. Run it twice; the second call resumes from state.pt.
#   bench/with_gpu.sh train/historical/m2_qwen1024.sh
HERE=${0:A:h:h}; ROOT=${MK_ROOT:-$HOME/mac-kernels-data}; PY=$HERE/.venv/bin/python; LOGS=${HERE:h}/logs
OUT=$ROOT/train/runs/cyan_qwen_1024_int8
cd $HERE
$PY train_lora.py --arch qwen21 --dataset $ROOT/train/datasets/cyanoprint --unet qwen_image_2.1_int8_convrot.safetensors \
  --res 1024 --chunked-attn --steps 300 --stop-after 150 --save-every 150 --val-every 50 --lr 2e-4 --seed 0 \
  --out $OUT --name qwen21_cyanoprint_1024_int8 $([ -f $OUT/state.pt ] && echo --resume) >> $LOGS/train-qwen-1024.log 2>&1
