"""Equivalence of decode-path rewrites against float32, scored the way decode runs: one token at a
time with a cache, batch 16 (so linears see 16 rows and the multi-row kernel is used). The float32
model generates greedily and saves its tokens and logits (--mode ref); each fp16 arm is then
teacher-forced on those tokens (--mode test) and its per-step logits compared with float32's.
--prompts real takes 16 windows of 32 tokens from the frozen texts in rerun_texts/ instead of random ids.
Acceptance, fixed before the 2026-11 rerun: against float32, each faster arm's mean and p99.9 absolute
error at most 1.25x eager fp16's, and its top-1 agreement at most 2 tokens below eager's. The maximum
is reported but not judged: a single worst element moves with the summation order."""
import argparse, json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, StaticCache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--ref", required=True)
ap.add_argument("--out"); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--prompts", default="random", choices=["random", "real"]); a = ap.parse_args()
B, P, N = a.batch, 32, 32
if a.mode == "test":
    import bench_decode_compile_kernel  # noqa: registers the custom op; guarded below from running its main
dt = torch.float32 if a.mode == "ref" else torch.float16
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=dt).to("mps").eval()
if a.prompts == "random":
    g = torch.Generator().manual_seed(11); ids = torch.randint(1000, 30000, (B, P), generator=g)
else:
    from transformers import AutoTokenizer
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rerun_texts")
    text = "".join(open(os.path.join(here, f)).read() for f in ("doc_edit.txt", "code_edit.txt"))
    t = AutoTokenizer.from_pretrained(REPO, revision=REV)(text)["input_ids"]
    stride = (len(t) - P) // (B - 1)
    ids = torch.tensor([t[i * stride:i * stride + P] for i in range(B)])
@torch.no_grad()
def forced(step, toks):
    cache = StaticCache(config=model.config, max_cache_len=P + N)
    model(ids.to("mps"), past_key_values=cache, use_cache=True, cache_position=torch.arange(P, device="mps"))
    out = []
    for i in range(N):
        out.append(step(toks[:, i:i + 1].to("mps"), cache, torch.tensor([P + i], device="mps"))[:, -1].float().cpu())
    return torch.stack(out, 1)
def eager(nxt, cache, pos): return model(nxt, past_key_values=cache, use_cache=True, cache_position=pos).logits
if a.mode == "ref":
    with torch.no_grad():
        seq = model.generate(ids.to("mps"), max_new_tokens=N, do_sample=False)[:, P:].cpu()
    L = forced(eager, seq)
    torch.save({"toks": seq, "logits": L}, a.ref); print("ref saved", tuple(L.shape))
else:
    ref = torch.load(a.ref); res = {}
    from bench_decode_compile_kernel import USE
    for arm in ("eager", "compiled", "compiled_kernel"):
        USE["kernel"] = arm.endswith("kernel")
        if arm == "eager": step = eager
        else: torch._dynamo.reset(); step = torch.compile(eager, dynamic=False)
        L = forced(step, ref["toks"]); err = (L - ref["logits"]).abs()
        res[arm] = {"max_abs": float(err.max()), "mean_abs": float(err.mean()),
                    "p999_abs": float(err.flatten().kthvalue(int(err.numel() * 0.999)).values),
                    "top1_agree_with_fp32": float((L.argmax(-1) == ref["logits"].argmax(-1)).float().mean())}
        print(arm, {k: round(v, 4) for k, v in res[arm].items()}, flush=True)
    e, n = res["eager"], B * N
    for arm in ("compiled", "compiled_kernel"):
        r = res[arm]
        r["pass"] = (r["mean_abs"] <= 1.25 * e["mean_abs"] and r["p999_abs"] <= 1.25 * e["p999_abs"]
                     and r["top1_agree_with_fp32"] >= e["top1_agree_with_fp32"] - 2 / n)
    res["prompts"] = a.prompts
    print("pass:", {arm: res[arm]["pass"] for arm in ("compiled", "compiled_kernel")}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
