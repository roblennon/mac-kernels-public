"""Profile Qwen3-1.7B decode steps on MPS with torch.profiler (CPU activity): which ops take the
host time, and how many ops a step dispatches. Also the step's wall time with and without
synchronizing every op, to separate host-bound from device-bound."""
import argparse, collections, json, time
import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM
REPO, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--fused-norm", action="store_true"); a = ap.parse_args()
if a.fused_norm:
    import torch.nn.functional as F, transformers.models.qwen3.modeling_qwen3 as q3
    q3.Qwen3RMSNorm.forward = lambda self, x: F.rms_norm(x, (x.shape[-1],), self.weight, self.variance_epsilon)
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
with torch.no_grad():
    ids = torch.randint(1000, 30000, (a.batch, 32), device="mps")
    out = model(ids, use_cache=True); past = out.past_key_values; nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    def step():
        global past, nxt
        o = model(nxt, past_key_values=past, use_cache=True); past = o.past_key_values; nxt = o.logits[:, -1].argmax(-1, keepdim=True)
    for _ in range(5): step()
    torch.mps.synchronize()
    t = time.perf_counter(); n = 16
    for _ in range(n): step()
    torch.mps.synchronize(); wall = (time.perf_counter() - t) / n
    # host-only cost: time to enqueue a step without waiting (approximate), measured by timing dispatch alone
    t = time.perf_counter()
    for _ in range(n): step()
    host = (time.perf_counter() - t) / n
    torch.mps.synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(8): step()
        torch.mps.synchronize()
    agg = collections.Counter(); cnt = collections.Counter()
    for e in prof.key_averages():
        agg[e.key] += e.self_cpu_time_total; cnt[e.key] += e.count
    total = sum(agg.values())
    top = [{"op": k, "self_cpu_ms_per_step": v / 8 / 1e3, "calls_per_step": cnt[k] / 8} for k, v in agg.most_common(25)]
res = {"fused_norm": a.fused_norm, "torch": torch.__version__, "batch": a.batch, "wall_ms_per_step": wall * 1e3, "host_enqueue_ms_per_step": host * 1e3,
       "profiled_self_cpu_ms_per_step": total / 8 / 1e3, "aten_calls_per_step": sum(cnt.values()) / 8, "top": top}
json.dump(res, open(a.out, "w"), indent=1)
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items() if k != "top"})
for r in top[:22]: print(f"{r['op']:55} {r['self_cpu_ms_per_step']:7.2f} ms  {r['calls_per_step']:6.0f} calls")
