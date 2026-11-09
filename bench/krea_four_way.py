#!/usr/bin/env python3
"""Clean four-way Krea 2 Turbo sweep: {stock, metal} x {sub-quadratic, SDPA attention} x sizes.

Each run is a fresh process through bench/gpu_runner.py --monitor, cells are visited in a
shuffled order (seeded, recorded), and host state is captured before every run. Results are
appended to a JSONL file as they arrive, so a stopped sweep keeps what it measured.

Steady-state seconds per step is the median of steps 2..N of one run (step 1 includes the
first-use weight load). The cell summary is the median and IQR of that value across repeats.

Usage (from the repo root, with the native .venv):
  MK_ROOT=/path/to/models .venv/bin/python bench/krea_four_way.py --out results.jsonl \
      --repeats 3 --sizes 512,1024 --seed 2026
  .venv/bin/python bench/krea_four_way.py --summarize results.jsonl
"""
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELLS = [  # (label, MODE, ATTN)
    ("stock+quad", "stock", "quad"),
    ("stock+sdpa", "stock", "pytorch"),
    ("metal+quad", "metal", "quad"),
    ("metal+sdpa", "metal", "pytorch"),
]


def host_state():
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as exc:  # recorded, never fatal
            return f"error: {exc}"
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "loadavg": os.getloadavg(),
        "swap": sh("sysctl -n vm.swapusage"),
        "therm": sh("pmset -g therm | grep -E 'CPU_Speed_Limit|No thermal' | tr -s ' '"),
        "memory_free_percent": sh("memory_pressure | grep -oE 'free percentage: [0-9]+%'"),
    }


def run_cell(label, mode, attn, size, rep, args, capture=False):
    env = dict(os.environ)
    env.update(MODE=mode, ATTN=attn, W=str(size), H=str(size), STEPS=str(args.steps), SEED=str(args.image_seed),
               CFG="1", COND_CACHE=args.cond_cache, TAG=f"fourway-r{rep}",
               PYTHONPATH=str(ROOT / "vendor" / "comfy-kitchen"))
    if capture:
        env["COND_STAGE"] = "capture"
    for k in ("FP16", "LORA", "KREA_READONLY", "KREA_ADAPTER_READONLY", "SPECTRAL", "DENOISE_TIMING"):
        env.pop(k, None)
    py = str(ROOT / ".venv" / "bin" / "python")
    cmd = [py, str(ROOT / "bench" / "gpu_runner.py"), "--monitor", "--seconds", str(args.deadline),
           "--rss-gib", str(args.rss_gib), "--swap-growth-gib", str(args.swap_growth_gib),
           "--", py, str(ROOT / "bench" / "run_krea2.py")]
    before = host_state()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    wall = round(time.perf_counter() - t0, 1)
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
        if capture and line.startswith("CONDITIONING_RESULT "):
            result = json.loads(line[len("CONDITIONING_RESULT "):])
    record = dict(cell=label, mode=mode, attn=attn, size=size, repeat=rep, wall_s=wall,
                  returncode=proc.returncode, host_before=before, capture=capture)
    if capture:
        record["ok"] = result is not None
        if result is None:
            record["failure"] = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-25:])
        else:
            record["times"] = result.get("times")
        return record
    if result is None:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-25:]
        record["failure"] = "\n".join(tail)
    else:
        steps = result["step_s"]
        record.update(step_s=steps, steady_s_per_step=statistics.median(steps[1:]) if len(steps) > 1 else None,
                      first_step_s=steps[0] if steps else None, times=result["times"],
                      mps_alloc_gb_after_sample=result["mps_alloc_gb_after_sample"], nan=result["nan"],
                      mean=result["mean"], std=result["std"], torch=result["torch"], tag=result["tag"])
    return record


def summarize(records):
    rows = {}
    for r in records:
        if r.get("capture") or r.get("steady_s_per_step") is None:
            continue
        rows.setdefault((r["size"], r["cell"]), []).append(r)
    out = []
    for (size, cell), rs in sorted(rows.items()):
        v = sorted(r["steady_s_per_step"] for r in rs)
        q1, q3 = (statistics.quantiles(v, n=4)[0], statistics.quantiles(v, n=4)[2]) if len(v) >= 2 else (v[0], v[0])
        mem = sorted(r["mps_alloc_gb_after_sample"] for r in rs)
        out.append(dict(size=size, cell=cell, n=len(v), median_s_per_step=round(statistics.median(v), 2),
                        iqr=(round(q1, 2), round(q3, 2)), min=v[0], max=v[-1],
                        first_step_median=round(statistics.median(r["first_step_s"] for r in rs), 1),
                        mem_after_sample_gb_median=round(statistics.median(mem), 2),
                        nan=any(r["nan"] for r in rs)))
    return out


def print_table(summary, records):
    failed = [r for r in records if not r.get("capture") and r.get("steady_s_per_step") is None]
    print(f"{'size':>5} {'cell':<11} {'n':>2} {'median s/step':>14} {'IQR':>16} {'min':>6} {'max':>6} {'step1':>6} {'GB after':>9}")
    for s in summary:
        print(f"{s['size']:>5} {s['cell']:<11} {s['n']:>2} {s['median_s_per_step']:>14} {str(s['iqr']):>16} "
              f"{s['min']:>6} {s['max']:>6} {s['first_step_median']:>6} {s['mem_after_sample_gb_median']:>9}")
    for r in failed:
        print(f"FAILED {r['size']} {r['cell']} repeat {r['repeat']} rc={r['returncode']}:\n{r.get('failure', '')[-600:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="JSONL to append run records to")
    ap.add_argument("--summarize", help="summarize an existing JSONL instead of running")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sizes", default="512,1024")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--image-seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=2026, help="shuffle seed for the run order")
    ap.add_argument("--deadline", type=int, default=1200, help="gpu_runner --seconds per run")
    ap.add_argument("--rss-gib", type=float, default=27)
    ap.add_argument("--only", help="comma list of cell labels to run (default all four)")
    ap.add_argument("--cond-cache", default="off", choices=["off", "auto"],
                    help="auto reuses captured CPU text conditioning; the encoder then never loads during timed runs")
    ap.add_argument("--capture", action="store_true", help="with --cond-cache auto: capture conditioning per cell first")
    ap.add_argument("--swap-growth-gib", type=float, default=4)
    ap.add_argument("--max-consecutive-failures", type=int, default=2)
    args = ap.parse_args()
    if args.summarize:
        records = [json.loads(l) for l in open(args.summarize) if l.strip()]
        records = [r for r in records if "cell" in r]
        print_table(summarize(records), records)
        return
    if not args.out:
        ap.error("--out or --summarize is required")
    sizes = [int(s) for s in args.sizes.split(",")]
    cells = [c for c in CELLS if not args.only or c[0] in args.only.split(",")]
    plan = [(c, size, rep) for rep in range(1, args.repeats + 1) for size in sizes for c in cells]
    random.Random(args.seed).shuffle(plan)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(dict(plan=[dict(cell=c[0], size=s, repeat=r) for c, s, r in plan], shuffle_seed=args.seed,
                                steps=args.steps, image_seed=args.image_seed, mk_root=os.environ.get("MK_ROOT"),
                                started=time.strftime("%Y-%m-%d %H:%M:%S %Z"))) + "\n")
    records = []
    if args.capture:
        for label, mode, attn in cells:
            print(f"[capture] {label}", flush=True)
            rec = run_cell(label, mode, attn, sizes[0], 0, args, capture=True)
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"    {'ok' if rec['ok'] else 'FAILED'} wall {rec['wall_s']} s", flush=True)
            if not rec["ok"]:
                print(rec.get("failure", "")[-800:])
                sys.exit(2)
    consecutive_failures = 0
    for i, ((label, mode, attn), size, rep) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {size} {label} repeat {rep}", flush=True)
        rec = run_cell(label, mode, attn, size, rep, args)
        records.append(rec)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec.get("steady_s_per_step") is not None:
            consecutive_failures = 0
            print(f"    steady {rec['steady_s_per_step']} s/step, step1 {rec['first_step_s']} s, "
                  f"{rec['mps_alloc_gb_after_sample']} GB after sampling, wall {rec['wall_s']} s", flush=True)
        else:
            consecutive_failures += 1
            print(f"    FAILED rc={rec['returncode']}", flush=True)
            if consecutive_failures >= args.max_consecutive_failures:
                print(f"stopping: {consecutive_failures} consecutive failures (admission refusals do not fix themselves)", flush=True)
                print(rec.get("failure", "")[-800:])
                break
    print_table(summarize(records), records)


if __name__ == "__main__":
    main()
