#!/usr/bin/env bash
# Decode-stack rerun (2026-11): correctness checks first, then four rounds of speed measurements with
# arm orders rotated between rounds. Every process goes through gpu_runner.py and its GPU lock.
# Needs: PY (a Python with torch 2.15 and transformers 5.18), OUT (an empty results directory),
# HF_HOME with Qwen/Qwen3-1.7B cached. Optional: REF_RANDOM (the earlier float32 decode reference,
# for the random-prompt equivalence check). Run from anywhere; paths resolve from this file.
set -u
: "${PY:?set PY}" "${OUT:?set OUT}"
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
mkdir -p "$OUT"
run() {  # run <name> <seconds> <script and args...>
  local name=$1 secs=$2; shift 2
  pmset -g therm > "$OUT/$name.therm" 2>&1
  MK_GPU_LABEL="mac-kernels-public:rerun-$name" python3 bench/gpu_runner.py --monitor --seconds "$secs" --rss-gib 14 -- \
    "$PY" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?; echo "$name exit $rc"; return $rc
}
verdict() {  # verdict <json> <expected>: stop the whole run if a check did not give the expected verdict
  "$PY" - "$1" "$2" <<'PY' || { echo "STOPPED: $1 did not give $2"; exit 1; }
import json, sys
r = json.load(open(sys.argv[1]))
v = r["verdict"] if "verdict" in r else ("pass" if all(r[a]["pass"] for a in ("compiled", "compiled_kernel")) else "fail")
print(sys.argv[1], v); sys.exit(v != sys.argv[2])
PY
}

# 1. Correctness. The two planted runs must fail; everything else must pass.
for A in compiled compiled_kernel eager; do
  run rollback_$A 400 bench/check_spec_rollback.py --arm $A --out "$OUT/rollback_$A.json" || exit 1
  verdict "$OUT/rollback_$A.json" pass
done
for P in skip_reset visible_sentinel; do
  run rollback_plant_$P 400 bench/check_spec_rollback.py --arm compiled --plant $P --out "$OUT/rollback_plant_$P.json" || exit 1
  verdict "$OUT/rollback_plant_$P.json" fail
done
run equiv_real_ref 300 bench/check_decode_equivalence.py --mode ref --prompts real --ref "$OUT/ref_real_fp32.pt" || exit 1
run equiv_real 300 bench/check_decode_equivalence.py --mode test --prompts real --ref "$OUT/ref_real_fp32.pt" --out "$OUT/equiv_real.json" || exit 1
verdict "$OUT/equiv_real.json" pass
if [ -n "${REF_RANDOM:-}" ]; then
  run equiv_random 300 bench/check_decode_equivalence.py --mode test --ref "$REF_RANDOM" --out "$OUT/equiv_random.json" || exit 1
  verdict "$OUT/equiv_random.json" pass
fi

# 2. Speed. A/A partners sit three places apart, as far as the pairs under test.
STEP_ORDERS=("eager_dynamic,compiled_a,kernel_a,eager_static,compiled_b,kernel_b"
             "eager_static,compiled_b,kernel_b,eager_dynamic,compiled_a,kernel_a"
             "kernel_b,compiled_b,eager_static,kernel_a,compiled_a,eager_dynamic"
             "kernel_a,compiled_a,eager_dynamic,kernel_b,compiled_b,eager_static")
SPEC_ORDERS=("plain_a,spec,aspec,plain_b,spec_kernel,aspec_kernel"
             "plain_b,spec_kernel,aspec_kernel,plain_a,spec,aspec"
             "aspec_kernel,spec_kernel,plain_b,aspec,spec,plain_a"
             "aspec,spec,plain_a,aspec_kernel,spec_kernel,plain_b")
VERIFY_T=("1,2,3,5,9,13,17" "17,13,9,5,3,2,1" "5,1,9,3,17,2,13" "13,2,17,3,9,1,5")
for r in 0 1 2 3; do
  for BP in 1:32 4:32 8:32 16:32 1:512 16:512; do
    B=${BP%:*}; P=${BP#*:}
    run step_b${B}_p${P}_r$r 540 bench/rerun_decode_step.py --batch $B --prompt-len $P --order "${STEP_ORDERS[$r]}" --process $r --out "$OUT/step_b${B}_p${P}_r$r.json"
  done
  for T in code_edit doc_edit free_writing; do
    run spec_${T}_r$r 540 bench/rerun_spec.py --task $T --order "${SPEC_ORDERS[$r]}" --process $r --out "$OUT/spec_${T}_r$r.json"
  done
  for A in eager compiled compiled_kernel; do
    run verify_${A}_r$r 400 bench/bench_verify_cost.py --arm $A --T "${VERIFY_T[$r]}" --repeat-first --out "$OUT/verify_${A}_r$r.json"
  done
done
"$PY" bench/rerun_analyze.py "$OUT" --expect 4 > "$OUT/summary.md" || { echo "INCOMPLETE: see $OUT/summary.md"; exit 1; }
echo ALL DONE
