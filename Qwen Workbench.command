#!/bin/zsh
set -eu
project_dir="${0:A:h}"
exec "$project_dir/Video Workbench.command" "$@"
