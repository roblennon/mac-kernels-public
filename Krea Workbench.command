#!/bin/zsh
set -eu
project_dir="${0:A:h}"
if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
  print -u2 'Krea Workbench needs the installed project Python environment.'
  exit 2
fi
exec "$project_dir/.venv/bin/python" "$project_dir/bench/start_krea_workbench.py" "$@"
