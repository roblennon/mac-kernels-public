#!/bin/zsh
# One server session that runs the official templates (bench/run_templates.py), then stops the server.
# Agents: bench/with_gpu.sh bench/template_session.sh [template ...]
HERE=${0:A:h}; cd "$HERE/.."
mkdir -p logs
bench/serve.sh > logs/serve-templates.log 2>&1 &
SERVER=$!
for i in {1..90}; do curl -s -o /dev/null http://127.0.0.1:${PORT:-8188}/system_stats && break; sleep 2; done
.venv/bin/python -u bench/run_templates.py "$@"
kill $SERVER; wait $SERVER 2>/dev/null
