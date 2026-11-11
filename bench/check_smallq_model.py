"""Whole-model check of the small-query attention kernel (bench/smallq_hf.py): Qwen3-1.7B fp16 on MPS,
a random-id prefill of --prefill tokens into a StaticCache, then teacher-forced verify forwards of T
new tokens from that same position (the cache counter is rolled back between forwards), compiled
with torch.compile(dynamic=False), for stock SDPA and for "mk_smallq" on the same inputs.

Per T it records the logits' max abs difference and argmax agreement between:
  mk compiled and stock compiled (the result),
  stock compiled against itself on a second run (run-to-run),
  stock eager and stock compiled (the same attention, two executions),
  mk compiled against itself, mk eager against mk compiled.
It also counts compiled frames per T (a graph break shows as extra frames) and times the compiled
verify forward (median of --reps after warm-up; indicative only, not the rerun protocol).

Pass rule, fixed before running: at every T, every mk logit is finite; max |mk - stock compiled|
is at most max(2 x yardstick, 0.1), capped at 1.0, where the yardstick is the larger of stock's
run-to-run and eager-versus-compiled differences at that T; and the argmax equals stock compiled's
on every row where stock's top-two margin exceeds 0.5.

python3 bench/gpu_runner.py --monitor --seconds 540 --rss-gib 14 -- <python> \\
    bench/check_smallq_model.py --Ts 1,4,8 --out a.json
"""
import argparse, json, os, statistics, sys, time
import torch
import torch._dynamo.testing
from transformers import AutoModelForCausalLM, StaticCache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smallq_hf
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def compare(a, b, ref_margin_from=None):
    """max abs difference, argmax agreement (fraction of rows), argmax mismatches on rows where
    the reference (b, or ref_margin_from) has top-two margin > 0.5."""
    ref = b if ref_margin_from is None else ref_margin_from
    top2 = ref.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    eq = a.argmax(-1) == b.argmax(-1)
    return {"max_abs": float((a - b).abs().max()), "argmax_agree": float(eq.float().mean()),
            "argmax_mismatch_confident": int(((~eq) & (margin > 0.5)).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ts", default="1,4,8,9,16")
    ap.add_argument("--prefill", type=int, default=4096)
    ap.add_argument("--threshold", type=int, default=4)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    Ts = [int(t) for t in a.Ts.split(",")]
    P = a.prefill
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
    assert model.config._attn_implementation == "sdpa"
    cache = StaticCache(config=model.config, max_cache_len=P + 64)
    g = torch.Generator().manual_seed(1234)
    prompt = torch.randint(1000, 30000, (1, P), generator=g).to("mps")
    drafts = {T: torch.randint(1000, 30000, (1, T), generator=g).to("mps") for T in Ts}
    with torch.no_grad():
        model(prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)
    torch.mps.synchronize()

    def reset():
        for layer in cache.layers:
            layer.cumulative_length.fill_(P)

    @torch.no_grad()
    def fwd(ids):
        return model(ids, past_key_values=cache, use_cache=True).logits

    def run(step, T):
        reset()
        return step(drafts[T])[0].float().cpu()

    def timed(step, T):
        times = []
        for i in range(a.reps + 2):
            reset(); torch.mps.synchronize(); t = time.perf_counter()
            step(drafts[T]); torch.mps.synchronize(); times.append(time.perf_counter() - t)
        return statistics.median(times[2:]) * 1e3

    res = {"torch": torch.__version__, "prefill": P, "Lmax": P + 64, "threshold": a.threshold, "Ts": Ts,
           "per_T": {}}
    logits = {}
    for impl in ("sdpa", "mk_smallq"):
        if impl == "mk_smallq":
            smallq_hf.enable(model, threshold=a.threshold)
        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        step = torch.compile(fwd, backend=cnt, dynamic=False)
        for T in Ts:
            before = cnt.frame_count
            t0 = time.perf_counter()
            c1 = run(step, T)
            compile_s = time.perf_counter() - t0
            frames = cnt.frame_count - before
            c2 = run(step, T)
            e = run(fwd, T)
            ms = timed(step, T)
            logits[impl, T] = (c1, c2, e)
            res["per_T"].setdefault(str(T), {})[impl] = {"frames": frames, "compile_s": round(compile_s, 1),
                                                         "ms": round(ms, 2), "finite": bool(torch.isfinite(c1).all())}
            print(impl, T, "frames", frames, "ms", round(ms, 2), flush=True)
    smallq_hf.disable(model)

    failures = []
    for T in Ts:
        s1, s2, se = logits["sdpa", T]
        m1, m2, me = logits["mk_smallq", T]
        d = res["per_T"][str(T)]
        d["stock_run_to_run"] = compare(s2, s1)
        d["stock_eager_vs_compiled"] = compare(se, s1)
        d["mk_vs_stock"] = compare(m1, s1)
        d["mk_run_to_run"] = compare(m2, m1)
        d["mk_eager_vs_mk_compiled"] = compare(me, m1)
        d["mk_eager_vs_stock_eager"] = compare(me, se)
        yard = max(d["stock_run_to_run"]["max_abs"], d["stock_eager_vs_compiled"]["max_abs"])
        tol = min(max(2 * yard, 0.1), 1.0)
        ok = (d["mk_smallq"]["finite"] and d["mk_vs_stock"]["max_abs"] <= tol
              and d["mk_vs_stock"]["argmax_mismatch_confident"] == 0)
        d["yardstick"], d["tolerance"], d["pass"] = yard, tol, ok
        if not ok:
            failures.append(T)
        print(f"T={T}: mk vs stock max {d['mk_vs_stock']['max_abs']:.4f} (tol {tol:.3f}, yardstick {yard:.4f}), "
              f"argmax agree {d['mk_vs_stock']['argmax_agree']:.3f}, confident mismatches "
              f"{d['mk_vs_stock']['argmax_mismatch_confident']}; ms stock {d['sdpa']['ms']} mk {d['mk_smallq']['ms']}; "
              f"frames stock {d['sdpa']['frames']} mk {d['mk_smallq']['frames']}", flush=True)
    res["verdict"] = "fail" if failures else "pass"
    res["failing_T"] = failures
    print("verdict", res["verdict"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
