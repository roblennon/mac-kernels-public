"""Record the real SDPA calls of one sampling model evaluation, then stop the run.

Usage: python bench/capture_attn.py bench/run_<model>.py   (the runner's own env applies)
Hooks torch.nn.functional.scaled_dot_product_attention (ComfyUI's wrapper calls through it), marks the first
comfy.samplers.sampling_function call as the sampling phase, and exits after it returns. Prints one CALL line per
distinct (q, k, v, mask, dtype, strides) with its count, and appends them to out/attn_calls.jsonl under LABEL.
"""
import collections, json, os, runpy, sys
import torch

ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
LABEL = os.environ.get("LABEL", os.path.basename(sys.argv[1]))
real_sdpa = torch.nn.functional.scaled_dot_product_attention
calls = collections.Counter()
state = {"phase": "pre", "patched": False}

class Done(Exception):
    pass

def _desc(t):
    return None if t is None else dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype).replace("torch.", ""), contiguous=t.is_contiguous())

def _patch_sampler():
    smp = sys.modules.get("comfy.samplers")
    if smp is None or state["patched"]:
        return
    real = smp.sampling_function
    def wrapped(*a, **kw):
        state["phase"] = "sampling"
        out = real(*a, **kw)
        raise Done()
    smp.sampling_function = wrapped; state["patched"] = True

def hook(q, k, v, attn_mask=None, *a, **kw):
    _patch_sampler()
    if state["phase"] == "sampling":
        key = json.dumps(dict(q=_desc(q), k=_desc(k), v=_desc(v), mask=_desc(attn_mask), extra=sorted(kw)))
        calls[key] += 1
    return real_sdpa(q, k, v, attn_mask, *a, **kw)

torch.nn.functional.scaled_dot_product_attention = hook
sys.argv = sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except Done:
    pass
with open(f"{ROOT}/out/attn_calls.jsonl", "a") as f:
    for key, n in calls.most_common():
        rec = dict(label=LABEL, count=n, **json.loads(key))
        print("CALL " + json.dumps(rec), flush=True); f.write(json.dumps(rec) + "\n")
