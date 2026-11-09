#!/bin/zsh
# Run benchmark jobs one after another. Usage: bench/queue.sh "ENV=.. script.py logname" ...
cd "$(dirname "$0")/.." && source .venv/bin/activate
for job in "$@"; do
  envs=(${(z)job}); log=${envs[-1]}; script=${envs[-2]}; envs=(${envs[1,-3]})
  env $envs bench/with_gpu.sh python -u bench/$script > logs/$log.log 2>&1
  grep -E "^RESULT|failed|Error" logs/$log.log | tail -3 | cut -c1-650
done
