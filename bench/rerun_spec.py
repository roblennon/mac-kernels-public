"""Prompt-lookup speculative decoding rerun, Qwen3-1.7B fp16 on MPS, batch 1, greedy. One process runs
one task through every arm in the order given; every repeat is kept. Each request runs to the
end-of-sequence token or a cap, so whole-request time covers a completed answer. Arms:
  plain_a/_b   an ordinary compiled decoder, one token per step, no lookup and no rollback: an A/A pair
  spec         compiled, k drafts from prompt lookup (n = 3, 2, 1; else repeat the last token)
  spec_kernel  the same with the multi-row kernel at 2 to 16 rows
  aspec, aspec_kernel   adaptive: draft only on a match of 2+ tokens, else a plain step
Tasks: code_edit (rename a function in a source file), doc_edit (fix three typos in prose),
free_writing. The texts are frozen in rerun_texts/. Per repeat the record keeps prefill, decode
and whole-request seconds, tokens, rounds and the accepted-length histogram."""
import argparse, json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_decode_compile_kernel import USE
from rerun_decode_step import thermal
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("plain_a", "plain_b", "spec", "spec_kernel", "aspec", "aspec_kernel")
CAPS = {"code_edit": 1024, "doc_edit": 512, "free_writing": 400}


def task_prompt(tok, name):
    text = lambda f: open(os.path.join(HERE, "rerun_texts", f)).read()
    msg = {"code_edit": "Rewrite the following Python file exactly, changing only the function name rope_one to "
                        "rotate_one everywhere. Output only the code.\n\n" + text("code_edit.txt"),
           "doc_edit": "Fix the spelling mistakes in the following text. Change nothing else, and output only the "
                       "corrected text.\n\n" + text("doc_edit.txt"),
           "free_writing": "Write a short story, about 300 words, about a lighthouse keeper who finds a message in a bottle."}[name]
    return tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, enable_thinking=False,
                                   return_tensors="pt", return_dict=True)["input_ids"]


def lookup(ids, k, min_n):
    """Latest earlier occurrence of the last n tokens (n = 3, 2, 1), and the k tokens after it.
    None when no match of at least min_n tokens; with min_n = 1 and no match, the last token k times."""
    for n in (3, 2, 1):
        if n < min_n:
            return None
        if len(ids) <= n:
            continue
        tail = ids[-n:]
        for s in range(len(ids) - n - 1, -1, -1):
            if ids[s:s + n] == tail:
                d = ids[s + n:s + n + k]
                return d + [d[-1]] * (k - len(d))
    return None if min_n > 1 else [ids[-1]] * k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(CAPS)); ap.add_argument("--order", default=",".join(ARMS))
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--process", type=int, default=0); ap.add_argument("--out", required=True); a = ap.parse_args()
    order = a.order.split(","); assert set(order) <= set(ARMS), order
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    prompt = task_prompt(tok, a.task).to("mps"); P = prompt.shape[1]; cap = CAPS[a.task]
    cache = StaticCache(config=model.config, max_cache_len=P + cap + a.k + 2)
    sync = torch.mps.synchronize

    def step(x, cache, pos):
        return model(x, past_key_values=cache, use_cache=True, cache_position=pos).logits

    @torch.no_grad()
    def request(fn, k, min_n):
        cache.reset()
        sync(); t0 = time.perf_counter()
        out = model(prompt, past_key_values=cache, use_cache=True, cache_position=torch.arange(P, device="mps"))
        gen = [int(out.logits[0, -1].argmax())]; sync(); t1 = time.perf_counter()
        ctx = prompt[0].tolist() + gen; pos = P; rounds = 0; hist = [0] * (k + 1)
        while len(gen) < cap and gen[-1] not in eos:
            if k:
                draft = lookup(ctx, k, min_n) or []
                kk = len(draft)
                logits = fn(torch.tensor([[gen[-1]] + draft], device="mps"), cache, torch.arange(pos, pos + 1 + kk, device="mps"))
                pred = logits[0].argmax(-1).tolist()
                n_ok = 0
                while n_ok < kk and draft[n_ok] == pred[n_ok]:
                    n_ok += 1
                new = draft[:n_ok] + [pred[n_ok]]
                pos += 1 + n_ok
                if n_ok < kk:  # transformers' StaticLayer writes at its own running count; set it back after rejections
                    for layer in cache.layers:
                        layer.cumulative_length.fill_(pos)
                if kk:
                    hist[n_ok] += 1
            else:  # the ordinary decoder: no lookup, no bookkeeping
                logits = fn(torch.tensor([[gen[-1]]], device="mps"), cache, torch.tensor([pos], device="mps"))
                new = [int(logits[0, -1].argmax())]
                pos += 1
            for t in new:
                if len(gen) < cap and gen[-1] not in eos:
                    gen.append(t); ctx.append(t)
            rounds += 1
        sync(); t2 = time.perf_counter()
        return {"prefill_s": t1 - t0, "decode_s": t2 - t1, "whole_s": t2 - t0, "new_tokens": len(gen),
                "rounds": rounds, "accept_hist": hist, "ended_on_eos": gen[-1] in eos, "tokens": gen}

    res = {"torch": torch.__version__, "task": a.task, "prompt_tokens": P, "cap": cap, "k": a.k, "process": a.process,
           "order": order, "thermal_start": thermal(), "arms": {}}
    for arm in order:
        USE["kernel"] = arm.endswith("kernel")
        k = 0 if arm.startswith("plain") else a.k
        min_n = 2 if arm.startswith("aspec") else 1
        torch._dynamo.reset(); fn = torch.compile(step, dynamic=False)
        t = time.perf_counter(); request(fn, k, min_n); warm_s = time.perf_counter() - t
        runs = [request(fn, k, min_n) for _ in range(a.repeats)]
        for r in runs[1:]:
            r["same_tokens_as_first"] = r["tokens"] == runs[0]["tokens"]; del r["tokens"]
        res["arms"][arm] = {"warm_s": warm_s, "runs": runs}
        r = runs[0]
        print(arm, a.task, f"{r['new_tokens']} tokens, decode {r['decode_s']:.2f} s, whole {r['whole_s']:.2f} s, "
              f"{r['rounds']} rounds, warm {warm_s:.1f} s", flush=True)
    USE["kernel"] = False
    res["thermal_end"] = thermal()
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
