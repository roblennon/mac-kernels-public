#!/bin/zsh
# Milestone 2 generation proofs, one GPU-lock hold.   STAGE=ab|1024 bench/with_gpu.sh train/historical/m2_eval.sh
HERE=${0:A:h:h}; ROOT=${MK_ROOT:-$HOME/mac-kernels-data}; PY=$HERE/.venv/bin/python; LOGS=${HERE:h}/logs
L=$ROOT/models/qwen21/loras; R=$ROOT/train/runs
VIGGLE=Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors
Q=(--arch qwen21 --unet qwen_image_2.1_int8_convrot.safetensors --palette cyan)
cd $HERE
if [ "$STAGE" = ab ]; then
  cp $R/cyan_qwen_512_int8/qwen21_cyanoprint_512_int8_s300.safetensors $R/cyan_qwen_512_bf16/qwen21_cyanoprint_512_bf16_s300.safetensors \
     $R/cyan_qwen_512_int8_seed1/qwen21_cyanoprint_512_int8_seed1_s300.safetensors $L/
  # both LoRAs on the same INT8 base, Viggle 6-step stacked, 512px (their training size) and 1024px
  $PY eval_lora.py $Q --stack-lora $VIGGLE --sigmas viggle --res 512 --tag ab-viggle-512 \
    --lora qwen21_cyanoprint_512_int8_s300.safetensors qwen21_cyanoprint_512_bf16_s300.safetensors \
    qwen21_cyanoprint_512_int8_seed1_s300.safetensors > $LOGS/train-eval-qwen-ab512.log 2>&1
  $PY eval_lora.py $Q --stack-lora $VIGGLE --sigmas viggle --res 1024 --tag ab-viggle-1024 \
    --lora qwen21_cyanoprint_512_int8_s300.safetensors qwen21_cyanoprint_512_bf16_s300.safetensors \
    qwen21_cyanoprint_512_int8_seed1_s300.safetensors > $LOGS/train-eval-qwen-ab1024.log 2>&1
else
  cp $R/cyan_qwen_1024_int8/qwen21_cyanoprint_1024_int8_s300.safetensors $L/
  $PY eval_lora.py $Q --stack-lora $VIGGLE --sigmas viggle --res 1024 --tag viggle-1024 \
    --lora qwen21_cyanoprint_1024_int8_s300.safetensors > $LOGS/train-eval-qwen-1024-viggle.log 2>&1
  $PY eval_lora.py $Q --steps 20 --cfg 4 --neg "" --res 1024 --tag base20-1024 \
    --prompts "cyanoprint style, a horse galloping on a beach" "a portrait photo of an old fisherman" \
    --lora qwen21_cyanoprint_1024_int8_s300.safetensors > $LOGS/train-eval-qwen-1024-base.log 2>&1
fi
