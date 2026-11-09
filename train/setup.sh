#!/bin/zsh
# Create the trainer's own venv (Python 3.12), separate from the inference .venv, with the
# same pinned packages (requirements.lock.txt) and comfy-kitchen path-linked from vendor/.
set -e
HERE=${0:A:h}
cd "$HERE"
uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r requirements.lock.txt pytest
SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "${HERE:h}/vendor/comfy-kitchen" > "$SITE/comfy_kitchen_dev.pth"
.venv/bin/python -c "import torch, comfy_kitchen; print('torch', torch.__version__, 'comfy_kitchen', comfy_kitchen.__file__)"
