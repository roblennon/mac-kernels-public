"""Run a runner unmodified with INFO logging and print LORA_AUDIT (see lora_audit.py) after each KSampler call.

Usage: python bench/lora_audit_wrap.py bench/run_krea2.py   (runner env applies)
"""
import json, logging, os, runpy, sys
import torch

logging.basicConfig(level=logging.INFO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lora_audit

real_sdpa = torch.nn.functional.scaled_dot_product_attention
state = {"patched": False}

def hook(*a, **kw):
    nodes = sys.modules.get("nodes")
    if not state["patched"] and nodes is not None and hasattr(nodes, "common_ksampler"):
        orig = nodes.common_ksampler
        def wrapped(model, *x, **y):
            out = orig(model, *x, **y)
            print("LORA_AUDIT " + json.dumps(lora_audit.audit(model)), flush=True)
            return out
        nodes.common_ksampler = wrapped; state["patched"] = True
    return real_sdpa(*a, **kw)

torch.nn.functional.scaled_dot_product_attention = hook
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
