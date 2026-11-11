"""Where does the time go in one Qwen3-1.7B fp16 forward of T new tokens against a StaticCache on MPS?

--phase timing    : T steps eager or compiled, three regimes: right after a fresh prefill (as bench_verify_cost does),
                    after a prefill plus a pause, and steady (no re-prefill; rewind the cache counters between steps).
--phase breakdown : eager only; synchronise around every component and accumulate per category.
--phase ablate    : remove one component at a time (cache write, mask, SDPA) and time the steady step, eager or compiled.
"""
import argparse, json, statistics, time, contextlib
import torch
from transformers import AutoModelForCausalLM, StaticCache
import transformers.models.qwen3.modeling_qwen3 as mq
from transformers.cache_utils import StaticLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, required=True)
ap.add_argument("--mode", default="eager")
ap.add_argument("--phase", default="timing")
ap.add_argument("--T", default="1,8")
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--pause", type=float, default=3.0)
ap.add_argument("--ablate", default="none")
ap.add_argument("--prefill-logits", default="all", help="all (as the bench) or last")
ap.add_argument("--out", required=True)
a = ap.parse_args()

sync = torch.mps.synchronize
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
P = a.ctx
Ts = [int(t) for t in a.T.split(",")]
g = torch.Generator().manual_seed(3)
res = {"ctx": P, "mode": a.mode, "phase": a.phase, "torch": torch.__version__, "res": {}}

# ---------------- ablation patches (applied before compile) ----------------
orig_update = StaticLayer.update
orig_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
orig_mask = mq.create_causal_mask
import transformers.masking_utils as mu
orig_ignore = mu._ignore_causal_mask_sdpa

def upd_noop(self, k, v, *args, **kw):
    if not self.is_initialized:
        return orig_update(self, k, v, *args, **kw)
    self.cumulative_length.add_(k.shape[-2])
    return self.keys, self.values

def sdpa_skip(module, query, key, value, attention_mask, **kw):
    # keep the data dependence on key/value tiny so nothing is dead-code-eliminated at whole-cache size
    out = query + key[:, :1, :1, :].sum() * 0 + value[:, :1, :1, :].sum() * 0
    return out.transpose(1, 2).contiguous(), None

def mask_none(**kw):
    return None

def apply_ablation(names):
    StaticLayer.update = orig_update; ALL_ATTENTION_FUNCTIONS["sdpa"] = orig_sdpa; mq.create_causal_mask = orig_mask; mu._ignore_causal_mask_sdpa = orig_ignore
    for n in names:
        if n == "update": StaticLayer.update = upd_noop
        elif n == "sdpa": ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_skip
        elif n == "mask": mq.create_causal_mask = mask_none
        elif n == "nobreak": mu._ignore_causal_mask_sdpa = lambda *a_, **k_: False  # decode never skips the mask anyway
        elif n == "none": pass
        else: raise SystemExit(f"unknown ablation {n}")

with torch.no_grad():
    cache = StaticCache(config=model.config, max_cache_len=P + 32)
    ids = torch.randint(1000, 30000, (1, P), generator=g).to("mps")

    def prefill():
        apply_ablation(["none"])
        cache.reset()
        kw = {} if a.prefill_logits == "all" else {"logits_to_keep": 1}
        model(ids, past_key_values=cache, use_cache=True, **kw)
        sync()

    def rewind(T):
        for layer in cache.layers:
            layer.cumulative_length.sub_(T)

    def step(x, cache):
        return model(x, past_key_values=cache, use_cache=True).logits

    def make_fn():
        if a.mode == "eager":
            return step
        torch._dynamo.reset()
        return torch.compile(step, dynamic=False)

    def timed(fn, x):
        sync(); t = time.perf_counter(); fn(x, cache); sync(); return (time.perf_counter() - t) * 1e3

    def summ(xs):
        return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "samples": xs}

    if a.phase == "timing":
        fn = make_fn()
        for T in Ts:
            x = torch.randint(1000, 30000, (1, T), generator=g).to("mps")
            prefill(); t0 = time.perf_counter(); fn(x, cache); sync(); first = time.perf_counter() - t0
            for _ in range(3):
                rewind(T); fn(x, cache); sync()
            r = {"first_call_s": first}
            # steady: no new prefill between steps
            r["steady"] = summ([(rewind(T), timed(fn, x))[1] for _ in range(a.reps)])
            # right after a fresh prefill
            ts = []
            for _ in range(a.reps):
                prefill(); ts.append(timed(fn, x))
            r["post_prefill"] = summ(ts)
            # after a fresh prefill and a pause
            ts = []
            for _ in range(max(3, a.reps // 2)):
                prefill(); time.sleep(a.pause); ts.append(timed(fn, x))
            r["post_prefill_pause"] = summ(ts)
            # the second and third steps after a prefill (not rewound to the prefill end)
            ts2, ts3 = [], []
            for _ in range(max(3, a.reps // 2)):
                prefill(); fn(x, cache); sync(); rewind(T); ts2.append(timed(fn, x)); rewind(T); ts3.append(timed(fn, x))
            r["post_prefill_2nd"] = summ(ts2); r["post_prefill_3rd"] = summ(ts3)
            # steady again at the end, as an A/A check
            r["steady_end"] = summ([(rewind(T), timed(fn, x))[1] for _ in range(a.reps)])
            res["res"][T] = r
            print(a.mode, P, "T", T, {k: round(v["median"], 2) for k, v in r.items() if isinstance(v, dict)}, flush=True)

    elif a.phase == "ablate":
        prefill()
        for abl in a.ablate.split(";"):
            names = abl.split("+")
            apply_ablation(names)
            fn = make_fn()
            out = {}
            for T in Ts:
                x = torch.randint(1000, 30000, (1, T), generator=g).to("mps")
                t0 = time.perf_counter(); rewind(0); fn(x, cache); sync(); rewind(T)
                for _ in range(3):
                    fn(x, cache); sync(); rewind(T)
                ts = []
                for _ in range(a.reps):
                    ts.append(timed(fn, x)); rewind(T)
                out[T] = summ(ts)
            res["res"][abl] = out
            print(a.mode, P, "ablate", abl, {T: round(v["median"], 2) for T, v in out.items()}, flush=True)
        apply_ablation(["none"])

    elif a.phase == "breakdown":
        assert a.mode == "eager"
        prefill()
        acc = {}
        on = {"v": False}

        def add(cat, ms):
            if on["v"]:
                acc[cat] = acc.get(cat, 0.0) + ms

        def wrap(cat, f):
            def w(*args, **kw):
                sync(); t = time.perf_counter(); r = f(*args, **kw); sync(); add(cat, (time.perf_counter() - t) * 1e3); return r
            return w

        StaticLayer.update = wrap("cache_update", orig_update)
        ALL_ATTENTION_FUNCTIONS["sdpa"] = wrap("sdpa_integration", orig_sdpa)
        orig_F = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = wrap("sdpa_kernel", orig_F)
        mq.create_causal_mask = wrap("mask_build", orig_mask)
        mq.apply_rotary_pos_emb = wrap("rope_apply", mq.apply_rotary_pos_emb)

        def hook(mod, cat):
            st = {}
            def pre(m, i):
                sync(); st["t"] = time.perf_counter()
            def post(m, i, o):
                sync(); add(cat, (time.perf_counter() - st["t"]) * 1e3)
            mod.register_forward_pre_hook(pre); mod.register_forward_hook(post)

        mm = model.model
        hook(mm.embed_tokens, "embed"); hook(mm.rotary_emb, "rope_table"); hook(mm.norm, "norm"); hook(model.lm_head, "lm_head")
        for L in mm.layers:
            hook(L.input_layernorm, "norm"); hook(L.post_attention_layernorm, "norm")
            hook(L.self_attn.q_norm, "qk_norm"); hook(L.self_attn.k_norm, "qk_norm")
            for n in ("q_proj", "k_proj", "v_proj"):
                hook(getattr(L.self_attn, n), "qkv_proj")
            hook(L.self_attn.o_proj, "o_proj"); hook(L.mlp, "mlp")
        for T in Ts:
            x = torch.randint(1000, 30000, (1, T), generator=g).to("mps")
            for _ in range(3):
                step(x, cache); sync(); rewind(T)
            tot, cats = [], []
            for _ in range(a.reps):
                acc.clear(); on["v"] = True
                tot.append(timed(step, x)); rewind(T); on["v"] = False
                cats.append(dict(acc))
            med = {k: statistics.median(c[k] for c in cats) for k in cats[0]}
            med["sdpa_integration_excl_kernel"] = med.pop("sdpa_integration") - med["sdpa_kernel"]
            med["TOTAL_synced"] = statistics.median(tot)
            med["unaccounted"] = med["TOTAL_synced"] - sum(v for k, v in med.items() if k != "TOTAL_synced")
            res["res"][T] = med
            print("breakdown", P, "T", T, {k: round(v, 2) for k, v in med.items()}, flush=True)

json.dump(res, open(a.out, "w"), indent=1)
