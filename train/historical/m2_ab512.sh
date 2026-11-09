#!/bin/zsh
# Milestone 2 A/B at 512px: the same LoRA run (data, seed, steps, lr) on the Qwen-Image 2.1 INT8 ConvRot
# base and on the bf16 base. ARM=int8|bf16 picks one arm per GPU-lock hold; SEED=1 with ARM=int8 is the
# seed-to-seed noise floor for the comparison.   ARM=int8 bench/with_gpu.sh train/m2_ab512.sh
HERE=${0:A:h}; ROOT=${MK_ROOT:-$HOME/mac-kernels-data}; PY=$HERE/.venv/bin/python; LOGS=${HERE:h}/logs
UNET=qwen_image_2.1_int8_convrot.safetensors; [ "$ARM" = bf16 ] && UNET=qwen_image_2.1_bf16.safetensors
SEED=${SEED:-0}; TAG=$ARM; [ "$SEED" != 0 ] && TAG=${ARM}_seed$SEED
cd $HERE
$PY train_lora.py --arch qwen21 --dataset $ROOT/train/datasets/cyanoprint --unet $UNET --res 512 --steps 300 \
  --save-every 150 --val-every 50 --lr 2e-4 --seed $SEED --out $ROOT/train/runs/cyan_qwen_512_$TAG \
  --name qwen21_cyanoprint_512_$TAG > $LOGS/train-qwen-512-$TAG.log 2>&1
