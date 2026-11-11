"""Standalone repro: in transformers 5.18, a torch.compile'd forward with a StaticCache, more than one
new token and no attention_mask breaks the graph in masking_utils._ignore_causal_mask_sdpa, at
`if q_offset == 0 and (...)`, because the static cache's query offset is a device tensor.

    python repro_hf_static_cache_graph_break.py [--device mps|cpu] [--T 4] [--reps 50]

No downloads: a tiny randomly initialised Qwen3 (4 layers, hidden 256). It prefills 64 tokens
eagerly into a StaticCache, then runs the forward of T new tokens under torch.compile(dynamic=False)
in two ways:
  no_mask    attention_mask=None (as in a hand-written decode or speculative-verify loop)
  ones_mask  an explicit all-ones 2D attention_mask of length 64 + T; the padding mask is then a
             traced tensor, _ignore_causal_mask_sdpa returns False before reaching q_offset, and
             the same causal mask is built inside the graph
Part 1 runs torch._dynamo.explain on each and prints graph count, break count and break reasons.
Part 2 compiles each with inductor (counting compiled frames with CompileCounterWithBackend),
checks the two give the same logits, and times them: the cache counter is set back to 64 before
every call so every call is the same forward, the device is synchronised around each call, and
calls alternate between the two arms. Reports the median and the interquartile range in ms.
"""
import argparse
import statistics
import time

import torch
import torch._dynamo
import torch._dynamo.testing
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM, StaticCache

P = 64


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    dev, T = a.device, a.T
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=1024, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                      num_attention_heads=8, num_key_value_heads=4, head_dim=32, max_position_embeddings=512)
    cfg._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(cfg).to(dev).eval()
    print(f"torch {torch.__version__}, transformers {transformers.__version__}, device {dev}, T {T}")

    cache = StaticCache(config=cfg, max_cache_len=P + 32)
    prompt = torch.randint(0, 1024, (1, P), device=dev)
    x = torch.randint(0, 1024, (1, T), device=dev)
    pos = torch.arange(P, P + T, device=dev)
    ones = torch.ones(1, P + T, dtype=torch.long, device=dev)

    def step(x, cache, pos, mask):
        return model(x, past_key_values=cache, use_cache=True, cache_position=pos, attention_mask=mask).logits

    def rewind():
        for layer in cache.layers:
            layer.cumulative_length.fill_(P)

    with torch.no_grad():
        model(prompt, past_key_values=cache, use_cache=True, cache_position=torch.arange(P, device=dev))
        arms = {"no_mask": None, "ones_mask": ones}

        print("\nPart 1: torch._dynamo.explain")
        for name, mask in arms.items():
            torch._dynamo.reset()
            rewind()
            ex = torch._dynamo.explain(step)(x, cache, pos, mask)
            print(f"  {name}: graphs {ex.graph_count}, graph breaks {ex.graph_break_count}")
            for br in ex.break_reasons:
                frame = br.user_stack[-1] if br.user_stack else None
                where = f"{frame.filename.split('site-packages/')[-1]}:{frame.lineno}" if frame else "?"
                print(f"    break at {where}: {br.reason.splitlines()[0][:160]}")

        print("\nPart 2: torch.compile (inductor), compiled frames and timing")
        torch._dynamo.reset()
        fns, frames = {}, {}
        for name, mask in arms.items():
            cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
            fn = torch.compile(step, backend=cnt, dynamic=False)
            for _ in range(3):  # compile and warm up
                rewind()
                out = fn(x, cache, pos, mask)
            sync(dev)
            fns[name], frames[name] = fn, cnt.frame_count
        rewind(); ref = step(x, cache, pos, None)
        rewind(); a_out = fns["no_mask"](x, cache, pos, None)
        rewind(); b_out = fns["ones_mask"](x, cache, pos, ones)
        print(f"  max |logits| difference: no_mask vs eager {(a_out - ref).abs().max().item():.2e}, "
              f"ones_mask vs no_mask {(b_out - a_out).abs().max().item():.2e}")
        samples = {n: [] for n in arms}
        for i in range(a.reps):
            order = list(arms) if i % 2 == 0 else list(arms)[::-1]
            for name in order:
                rewind()
                sync(dev)
                t0 = time.perf_counter()
                fns[name](x, cache, pos, arms[name])
                sync(dev)
                samples[name].append((time.perf_counter() - t0) * 1e3)
        for name in arms:
            q = statistics.quantiles(samples[name], n=4)
            print(f"  {name}: compiled frames {frames[name]}, median {statistics.median(samples[name]):.3f} ms "
                  f"(IQR {q[0]:.3f}-{q[2]:.3f}) over {a.reps} calls")
        ratio = statistics.median(x / y for x, y in zip(samples["no_mask"], samples["ones_mask"]))
        print(f"  no_mask / ones_mask, median of paired ratios: {ratio:.2f}")


if __name__ == "__main__":
    main()
