import sys, torch, torch.nn.functional as F
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fused_rope
import transformers.models.qwen3.modeling_qwen3 as q3
from transformers import AutoModelForCausalLM
on = {"v": False}
o_norm, o_rope = q3.Qwen3RMSNorm.forward, q3.apply_rotary_pos_emb
q3.Qwen3RMSNorm.forward = lambda self, x: F.rms_norm(x, (x.shape[-1],), self.weight, self.variance_epsilon) if on["v"] else o_norm(self, x)
q3.apply_rotary_pos_emb = lambda q, k, c, s, u=1: fused_rope.apply_rotary_pos_emb(q, k, c, s, u) if on["v"] else o_rope(q, k, c, s, u)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", dtype=torch.float16).to("mps").eval()
g = torch.Generator().manual_seed(101); ids = torch.randint(1000, 30000, (1, 32), generator=g).to("mps")
with torch.no_grad():
    seq = m.generate(ids, max_new_tokens=32, do_sample=False)  # stock greedy
    L0 = m(seq).logits[0, 31:-1].float(); on["v"] = True; L1 = m(seq).logits[0, 31:-1].float()
tgt = seq[0, 32:]
top2 = L0.topk(2, -1).values; margin = (top2[:, 0] - top2[:, 1])
a0, a1 = L0.argmax(-1), L1.argmax(-1)
for i in range(len(tgt)):
    flag = "  <-- fused picks a different token" if a1[i] != a0[i] else ""
    if flag or margin[i] < 0.3: print(f"step {i:2d}: stock top-2 margin {margin[i]:.3f}, max logit diff {float((L1[i]-L0[i]).abs().max()):.3f}{flag}")
print("smallest stock margin over 32 steps: %.3f" % float(margin.min()), "| steps where fused differs:", int((a1 != a0).sum()))
