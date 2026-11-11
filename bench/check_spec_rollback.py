"""Does speculative rollback leave the static cache exactly as plain decoding would? Qwen3-1.7B fp16 on
MPS, batch 1, the code-edit prompt. Drafts are built to be accepted to a planned length: the first
`a` tokens are the model's own greedy continuation and the rest are wrong. The schedule covers every
length 0 to 4, consecutive rejections, full acceptance and switches between one-token and five-token
graphs. After each rollback, every rejected cache slot is overwritten with a large sentinel, so a
query that can still see one is thrown far off.

Every verified row is compared with a freshly rebuilt cache: one eager forward over the prompt plus
the history so far. The tolerance comes from the same comparison for the plain decoder over the same
positions. Pass rule, fixed before running: every row finite; every row's max |logit difference|
from the rebuild at most max(2 x the plain decoder's largest, 0.1); and the argmax equal to the
rebuild's wherever the rebuild's top-two margin exceeds 0.5. The tolerance is capped at 1.0 so a
fault shared by both paths cannot widen it without limit. The check is valid only if every accepted
length 0 to 4 occurred and both graph switches (one token to five, five to one) happened; otherwise
the verdict is "inconclusive". Planted controls must fail:
skip_reset skips the counter reset; visible_sentinel puts the sentinel in the newest committed slot.

Arms: eager; compiled; compiled_kernel (compiled + the multi-row linear kernel); compiled_kernel_smallq
(compiled + the multi-row kernel + the small-query attention kernel, bench/smallq_hf.py, at its
default threshold of 4 rows, so the five-token verify steps run the attention kernel and the
one-token steps, the prefill and the eager rebuilds run stock SDPA)."""
import argparse, json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_decode_compile_kernel import USE
from rerun_spec import task_prompt, REPO, REV

SCHEDULE = [0, 0, 0, 4, 4, 1, None, 4, 2, 0, 3, None, None, 4, 1, 0, 2, 4, 3, 0, 4, 4, 4, 0]  # None = plain step
K, SENTINEL = 4, 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["eager", "compiled", "compiled_kernel", "compiled_kernel_smallq"])
    ap.add_argument("--plant", default="none", choices=["none", "skip_reset", "visible_sentinel"])
    ap.add_argument("--out", required=True); a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
    prompt = task_prompt(tok, "code_edit")[0].tolist(); P = len(prompt)
    n_ref = sum(1 + (x or 0) for x in SCHEDULE) + 8
    USE["kernel"] = a.arm in ("compiled_kernel", "compiled_kernel_smallq")
    if a.arm == "compiled_kernel_smallq":
        import smallq_hf
        smallq_hf.enable(model)

    def step(x, cache, pos):
        return model(x, past_key_values=cache, use_cache=True, cache_position=pos).logits

    fn = step if a.arm == "eager" else torch.compile(step, dynamic=False)
    # room for a faulty run too: without resets every round advances the write counter by its full length
    new_cache = lambda: StaticCache(config=model.config, max_cache_len=P + n_ref + len(SCHEDULE) * (K + 1) + 8)

    @torch.no_grad()
    def prefill(cache):
        out = model(torch.tensor([prompt], device="mps"), past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(P, device="mps"))
        return out.logits[0, -1].float().cpu()

    @torch.no_grad()
    def rebuilt(history):  # logits for the token after prompt + history, from a fresh eager forward
        with torch.no_grad():
            return model(torch.tensor([prompt + history], device="mps")).logits[0, -1].float().cpu()

    # The plain decoder through the same path: the reference continuation and the tolerance.
    with torch.no_grad():
        cache = new_cache(); last = prefill(cache); ref = [int(last.argmax())]; plain_rows = []
        for i in range(n_ref - 1):
            L = fn(torch.tensor([[ref[-1]]], device="mps"), cache, torch.tensor([P + i], device="mps"))[0, -1].float().cpu()
            plain_rows.append(L); ref.append(int(L.argmax()))
    plain_err = [float((plain_rows[i] - rebuilt(ref[:i + 1])).abs().max()) for i in range(0, n_ref - 1, 3)]
    assert all(torch.isfinite(r).all() for r in plain_rows) and all(e == e for e in plain_err), "plain path not finite"
    tol = min(max(2 * max(plain_err), 0.1), 1.0)

    rows, rounds = [], []
    with torch.no_grad():
        cache = new_cache(); prefill(cache); gen = [ref[0]]; pos = P
        for planned in SCHEDULE:
            kk = 0 if planned is None else K
            g = len(gen)
            draft = [] if planned is None else ref[g:g + planned] + [(ref[g + j] + 7) % model.config.vocab_size for j in range(planned, K)]
            L = fn(torch.tensor([[gen[-1]] + draft], device="mps"), cache, torch.arange(pos, pos + 1 + kk, device="mps"))[0].float().cpu()
            pred = L.argmax(-1).tolist(); n_ok = 0
            while n_ok < kk and draft[n_ok] == pred[n_ok]:
                n_ok += 1
            for j in range(n_ok + 1):  # rows whose inputs were all committed tokens
                R = rebuilt(gen + draft[:j])
                top2 = R.topk(2).values
                rows.append({"round": len(rounds), "row": j, "finite": bool(torch.isfinite(L[j]).all()),
                             "max_abs": float((L[j] - R).abs().max()), "argmax_eq": int(L[j].argmax()) == int(R.argmax()),
                             "margin": float(top2[0] - top2[1])})
            old_end = pos + 1 + kk
            gen += draft[:n_ok] + [pred[n_ok]]; pos += 1 + n_ok
            if n_ok < kk:
                if a.plant != "skip_reset":
                    for layer in cache.layers:
                        layer.cumulative_length.fill_(pos)
                for layer in cache.layers:  # rejected slots must never be read again
                    layer.keys[:, :, pos:old_end] = SENTINEL; layer.values[:, :, pos:old_end] = SENTINEL
            if a.plant == "visible_sentinel" and len(rounds) == 5:
                for layer in cache.layers:
                    layer.keys[:, :, pos - 1] = SENTINEL; layer.values[:, :, pos - 1] = SENTINEL
            rounds.append({"planned": planned, "accepted": n_ok, "graph_T": 1 + kk, "matches_ref": gen == ref[:len(gen)]})
    bad = [r for r in rows if not r["finite"] or r["max_abs"] > tol or (not r["argmax_eq"] and r["margin"] > 0.5)]
    seen = {r["accepted"] for r in rounds if r["planned"] is not None}
    Ts = [r["graph_T"] for r in rounds]
    switches = {(x, y) for x, y in zip(Ts, Ts[1:]) if x != y}
    covered = seen >= set(range(K + 1)) and {(1, K + 1), (K + 1, 1)} <= switches
    planned_met = sum(r["planned"] is None or r["accepted"] == r["planned"] for r in rounds)
    res = {"arm": a.arm, "plant": a.plant, "torch": torch.__version__, "prompt_tokens": P, "tolerance": tol,
           "plain_err_max": max(plain_err), "rows": len(rows), "failing_rows": bad[:20], "n_failing": len(bad),
           "max_abs": max(r["max_abs"] for r in rows if r["finite"]) if any(r["finite"] for r in rows) else None,
           "accepted_lengths_seen": sorted({r["accepted"] for r in rounds}), "rounds": rounds,
           "planned_met": planned_met, "n_rounds": len(rounds), "coverage_complete": covered,
           "verdict": "fail" if bad else ("pass" if covered else "inconclusive")}
    print(a.arm, a.plant, res["verdict"], f"{len(bad)}/{len(rows)} rows fail; tol {tol:.3f}; max {res['max_abs']}",
          "accepted lengths", res["accepted_lengths_seen"], flush=True)
    USE["kernel"] = False
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
