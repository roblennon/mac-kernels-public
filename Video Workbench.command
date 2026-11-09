#!/bin/zsh
set -eu
project_dir="${0:A:h}"
export PORT=${PORT:-8188}
export MK_ROOT=${MK_ROOT:-$HOME/mac-kernels-data}
if [ ! -d "$MK_ROOT/models" ]; then
  echo "Video Workbench: $MK_ROOT/models not found (set MK_ROOT to your model folder)" >&2
  exit 1
fi
mkdir -p "$MK_ROOT/out/video-user" "$MK_ROOT/out/video-input"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
exec "$project_dir/.venv/bin/python" "$project_dir/bench/gpu_runner.py" \
  --monitor --seconds 21600 -- \
  "$project_dir/bench/serve.sh" --cache-ram 18 --user-directory "$MK_ROOT/out/video-user" \
  --input-directory "$MK_ROOT/out/video-input" "$@"
