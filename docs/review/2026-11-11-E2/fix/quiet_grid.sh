#!/bin/zsh
# Run the isolated grid only while the CPU is quiet: at most one other process above 50% CPU and 1-min load < 8.
cd <repo>
PY=<scratch>/venv215/bin/python
S=<scratch>/opt/iso4; mkdir -p $S
busy() { ps -Ao pcpu= | awk '$1>50{n++} END{print n+0}'; }
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
for p in 1 2 3 4 5 6; do
  waited=0
  while [ "$(busy)" -gt 1 ] || [ "$(echo "$(load1) >= 8" | bc)" -eq 1 ]; do sleep 15; waited=$((waited+15)); [ $waited -ge 1200 ] && { echo "p$p gave up waiting"; exit 1; }; done
  echo "p$p start load $(load1) busy $(busy) waited ${waited}s $(date +%T)"
  MK_GPU_LABEL=mac-kernels-public:opt-iso4-p$p python3 bench/gpu_runner.py --monitor --seconds 540 --rss-gib 14 -- $PY bench/bench_smallq_attention.py grid --lengths 1024 4096 16384 --dev-arm --baseline-arm --fmask-arm --out $S/grid_p$p.json > $S/grid_p$p.log 2>&1
  echo "p$p rc=$? end load $(load1) busy $(busy) $(date +%T)"
done
