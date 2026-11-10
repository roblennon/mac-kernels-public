"""Is the divergence between speculative and plain greedy output a near-tie flip or a bug? Teacher-force
the plain greedy tokens through the eager model (one forward) and report the top-two logit margin at
the first token where the speculative output differs. A margin of about one fp16 step is a tie flip.
Also re-score the speculative tokens the same way and report where they leave the model's own argmax."""
import argparse, json, os, sys
import torch
from transformers import AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_prompt_lookup import tasks, REPO, REV
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float16).to("mps").eval()
d = {k: json.load(open(f"{a.dir}/pl_{k}.json")) for k in ("plain", "spec_kernel")}
for name, prompt in tasks(tok).items():
    P = prompt.shape[1]
    for arm in ("plain", "spec_kernel"):
        gen = d[arm]["tasks"][name]["tokens"]
        with torch.no_grad():
            L = model(torch.cat([prompt, torch.tensor([gen])], 1).to("mps")).logits[0, P - 1:P - 1 + len(gen)].float().cpu()
        top2 = L.topk(2, -1).values; margin = top2[:, 0] - top2[:, 1]
        off = [i for i in range(len(gen)) if int(L[i].argmax()) != gen[i]]
        print(name, arm, "tokens that are not the model's own argmax in one full forward:",
              [(i, round(float(margin[i]), 4)) for i in off[:6]], "| count", len(off))
    ref, sp = d["plain"]["tasks"][name]["tokens"], d["spec_kernel"]["tasks"][name]["tokens"]
    i = next((i for i, (x, y) in enumerate(zip(ref, sp)) if x != y), None)
    if i is None:
        print(name, "speculative output identical to plain greedy"); continue
    with torch.no_grad():
        L = model(torch.cat([prompt, torch.tensor([ref[:i]])], 1).to("mps") if i else prompt.to("mps")).logits[0, -1].float().cpu()
    v, ix = L.topk(2)
    print(name, f"first difference at {i}: plain chose {ref[i]}, spec chose {sp[i]}; full-forward top two {ix.tolist()} with margin {float(v[0]-v[1]):.4f}")
