#!/bin/zsh
# Start the forked ComfyUI (vendor/ComfyUI, branch mps-int8-weight-only) for normal use in the browser.
# Models come from $MK_ROOT/models via bench/extra_model_paths.yaml; outputs go to $MK_ROOT/out/comfyui.
# No performance flags: on Apple GPUs the branch already picks pytorch attention, fp16 VAEs, the Metal INT8/W4A8
# kernels and the LoRA side path. PORT overrides 8188. Agents running GPU benchmarks wrap this in bench/with_gpu.sh.
# Registers the existing GGUF fork; extra arguments (for example --cpu) go to ComfyUI.
HERE=${0:A:h}; REPO=${HERE:h}
export MK_ROOT=${MK_ROOT:-$HOME/mac-kernels-data}
PORT=${PORT:-8188}
if [ ! -d "$MK_ROOT/models" ]; then echo "serve.sh: $MK_ROOT/models not found (set MK_ROOT to your model folder)" >&2; exit 1; fi
if lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then echo "serve.sh: port $PORT is already in use (ComfyUI already running? open http://127.0.0.1:$PORT)" >&2; exit 1; fi
"$REPO/.venv/bin/python" "$HERE/setup_custom_nodes.py" || exit $?
mkdir -p "$MK_ROOT/out/comfyui"
cd "$REPO/vendor/ComfyUI"
exec "$REPO/.venv/bin/python" main.py --port "$PORT" \
  --extra-model-paths-config "$HERE/extra_model_paths.yaml" \
  --output-directory "$MK_ROOT/out/comfyui" "$@"
