#!/bin/zsh
# Serialize GPU jobs; gpu_runner.py also offers optional deadline/memory bounds.
# Keep this entry point's command arguments unchanged.
exec python3 "${0:A:h}/gpu_runner.py" -- "$@"
