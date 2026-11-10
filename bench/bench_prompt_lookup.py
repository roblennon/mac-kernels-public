"""End-to-end prompt-lookup speculative decoding on MPS, Qwen3-1.7B fp16, batch 1, greedy.
Drafts come from the context itself (no draft model): find the latest earlier occurrence of the
last n tokens (n = 3, then 2, then 1) and propose the k tokens that followed it. One forward of
k + 1 tokens verifies them; the matching prefix is accepted plus the model's own next token.
Arms: plain (compiled, one token per step), spec (compiled, k drafts), spec_kernel (compiled,
k drafts, multi-row kernel). Tasks: a copy-heavy code edit and free writing.
Output must match plain greedy decoding; the report gives token agreement."""
import argparse, json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_decode_compile_kernel import USE
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
HERE = os.path.dirname(os.path.abspath(__file__))

def tasks(tok):
    code = open(os.path.join(HERE, "fused_rope.py")).read().split("def reference")[0]
    msgs = {"code_edit": "Rewrite the following Python file exactly, changing only the function name rope_one to rotate_one everywhere. Output only the code.\n\n" + code,
            "free_writing": "Write a short story, about 300 words, about a lighthouse keeper who finds a message in a bottle."}
    return {k: tok.apply_chat_template([{"role": "user", "content": v}], add_generation_prompt=True,
                                       enable_thinking=False, return_tensors="pt", return_dict=True)["input_ids"]
            for k, v in msgs.items()}

def lookup(ctx, k, min_n=1):
    ids = ctx.tolist()
    for n in (3, 2, 1):
        if n < min_n: return None
        if len(ids) <= n: continue
        tail = ids[-n:]
        for s in range(len(ids) - n - 1, -1, -1):
            if ids[s:s + n] == tail:
                d = ids[s + n:s + n + k]
                if d: return d + [d[-1]] * (k - len(d))
    return None if min_n > 1 else [ids[-1]] * k

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--new", type=int, default=160)
    ap.add_argument("--adaptive", action="store_true", help="draft only on a 2+ token match, else a plain step"); a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
    USE["kernel"] = a.arm == "spec_kernel"
    def step(x, cache, pos): return model(x, past_key_values=cache, use_cache=True, cache_position=pos).logits
    fn = torch.compile(step, dynamic=False)
    fn1 = fn  # same compiled function; a separate graph is compiled per input length
    k = 0 if a.arm == "plain" else a.k
    res = {"arm": a.arm, "k": k, "adaptive": a.adaptive, "torch": torch.__version__, "tasks": {}}
    for name, prompt in tasks(tok).items():
        P = prompt.shape[1]; L = P + a.new + 64
        prompt = prompt.to("mps")
        cache = StaticCache(config=model.config, max_cache_len=L + k + 1)
        def run():
            cache.reset()
            with torch.no_grad():
                out = model(prompt, past_key_values=cache, use_cache=True, cache_position=torch.arange(P, device="mps"))
                ctx = prompt[0].tolist(); gen = [int(out.logits[0, -1].argmax())]; pos = P; rounds = acc = 0
                torch.mps.synchronize(); t0 = time.perf_counter()
                while len(gen) < a.new:
                    draft = lookup(torch.tensor(ctx + gen), k, min_n=2 if a.adaptive else 1) if k else []
                    if draft is None:  # adaptive: no match of 2+ tokens, so take a plain one-token step
                        draft = []
                    kk = len(draft)
                    x = torch.tensor([[gen[-1]] + draft], device="mps")
                    logits = (fn if kk else fn1)(x, cache, torch.arange(pos, pos + 1 + kk, device="mps"))
                    pred = logits[0].argmax(-1).tolist()
                    n_ok = 0
                    while n_ok < kk and draft[n_ok] == pred[n_ok]: n_ok += 1
                    gen += draft[:n_ok] + [pred[n_ok]]
                    pos += 1 + n_ok; rounds += 1; acc += n_ok
                    # transformers' StaticLayer writes new entries at its own running count (cumulative_length),
                    # not at cache_position. After rejected drafts that count runs ahead of the true position,
                    # so set it back: the next round must write at pos. In-place, so compiled graphs keep working.
                    if n_ok < kk:  # only after rejected drafts; a plain step or full acceptance leaves the count right
                        for layer in cache.layers:
                            layer.cumulative_length.fill_(pos)
                torch.mps.synchronize(); dt = time.perf_counter() - t0
            return gen[:a.new], dt, rounds, acc
        t0 = time.perf_counter(); run(); compile_s = time.perf_counter() - t0
        gen, dt, rounds, acc = run()
        _, dt2, _, _ = run()
        dt = min(dt, dt2)
        res["tasks"][name] = {"plain_steps": None, "prompt_tokens": P, "new_tokens": len(gen), "seconds": dt, "ms_per_token": dt / len(gen) * 1e3,
                              "rounds": rounds, "accepted_per_round": acc / rounds, "first_pass_s": compile_s,
                              "tokens": gen, "text_head": tok.decode(gen[:60])}
        r = res["tasks"][name]
        print(a.arm, name, f"{r['ms_per_token']:.2f} ms/token, {r['accepted_per_round']:.2f} accepted per round, {rounds} rounds", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
