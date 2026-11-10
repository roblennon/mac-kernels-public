set -u
cd <repo>
export HF_HOME=<hf-home> HF_HUB_OFFLINE=1
PY=<scratch>/venv215/bin/python; O=<scratch>
run() { MK_GPU_LABEL=mac-kernels-public:$1 python3 bench/gpu_runner.py --monitor --seconds 290 --rss-gib 12 -- $PY "${@:2}" > $O/$1.log 2>&1; echo "$1 exit $?"; }
for B in 4 8 16; do run ck_b$B bench/bench_decode_compile_kernel.py --batch $B --out $O/ck_b$B.json; done
for A in eager compiled compiled_kernel; do run vc_${A}_a bench/bench_verify_cost.py --arm $A --T 1,2,3,5 --out $O/vc_${A}_a.json; run vc_${A}_b bench/bench_verify_cost.py --arm $A --T 9,13,17 --out $O/vc_${A}_b.json; done
run pl_plain bench/bench_prompt_lookup.py --arm plain --out $O/pl_plain.json
for A in spec spec_kernel; do run pla_$A bench/bench_prompt_lookup.py --arm $A --adaptive --out $O/pla_$A.json; run pl_$A bench/bench_prompt_lookup.py --arm $A --out $O/pl_$A.json; done
run equiv bench/check_decode_equivalence.py --mode test --ref <scratch>/onerow/dec_ref_fp32.pt --out $O/equiv.json
echo ALL DONE
