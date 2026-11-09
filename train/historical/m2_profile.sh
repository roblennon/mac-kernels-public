#!/bin/zsh
# Milestone 2 profiling, one GPU-lock hold: Qwen-Image 2.1 INT8 LoRA steps at 512px, and at 1024px
# with and without the chunked attention backward.   bench/with_gpu.sh train/m2_profile.sh
HERE=${0:A:h}; ROOT=${MK_ROOT:-$HOME/mac-kernels-data}; PY=$HERE/.venv/bin/python; LOGS=${HERE:h}/logs
Q=(--arch qwen21 --dataset $ROOT/train/datasets/cyanoprint --unet qwen_image_2.1_int8_convrot.safetensors)
cd $HERE
$PY -m pytest -q ${HERE:h}/vendor/comfy-kitchen/tests/test_mps_attention_backward.py > $LOGS/train-attn-tests.log 2>&1
$PY train_lora.py $Q --res 512 --profile-steps 6 --out $ROOT/train/runs/profile_qwen_512 > $LOGS/train-profile-qwen-512.log 2>&1
$PY train_lora.py $Q --res 1024 --profile-steps 4 --chunked-attn --out $ROOT/train/runs/profile_qwen_1024_chunked > $LOGS/train-profile-qwen-1024-chunked.log 2>&1
$PY train_lora.py $Q --res 1024 --profile-steps 3 --out $ROOT/train/runs/profile_qwen_1024_sdpa > $LOGS/train-profile-qwen-1024-sdpa.log 2>&1
