# transformers issue: compiled StaticCache forward with T > 1 breaks the graph in `_ignore_causal_mask_sdpa`

Would be filed at [huggingface/transformers issues](https://github.com/huggingface/transformers/issues/new) (bug report; suggested labels `Compilation`, `Cache`). **Do not file: it is already fixed.** [#49354](https://github.com/huggingface/transformers/pull/49354) (merged 2026-10-08) fixes it, and v5.20.0 (released 2026-10-15) is the first release with the fix. The bug arrived with [#48975](https://github.com/huggingface/transformers/pull/48975) (merged 2026-09-23, first released in v5.18.0), so it affects 5.18 and 5.19 only. This repository pins 5.18.0, which is why we hit it. Drafted and checked 2026-11-11.

What to do here instead: move the decode benchmarks to transformers 5.20 or later, or pass an explicit all-ones `attention_mask` to compiled forwards. With 5.18.0 and that mask, the repro compiles to one graph.

## Duplicate triage (2026-11-11)

These searches ran against huggingface/transformers issues and PRs created since April 2026 (`gh api search/issues`): `_ignore_causal_mask_sdpa`, `q_offset`, "q_offset graph break", "StaticCache compile graph break", "static cache recompilation", "sdpa_mask graph break", "Data-dependent branching masking_utils", "static cache torch.compile speculative", "assisted generation compile static cache".

- [#49354](https://github.com/huggingface/transformers/pull/49354), merged 2026-10-08: "Fix data-dependent branch in `_ignore_causal_mask_sdpa` under compile". This is the same bug and the fix suggested below. On main, `masking_utils.py` lines 297-299 (`28ec08c469`, 2026-11-11) return `False` when `q_offset` is a traced tensor. The compare API puts it in v5.20.0, v5.20.2, v5.21 and v5.22.2, but not in v5.18.0 or v5.19.0.
- [#48975](https://github.com/huggingface/transformers/pull/48975), merged 2026-09-23: it let the causal-mask skip run under `torch.compile` when there is no padding mask. That change is what lets a compiled forward reach `q_offset == 0`. It is in v5.18.0 and not in v5.17.0.
- [#50267](https://github.com/huggingface/transformers/pull/50267), closed unmerged 2026-11: removes host syncs in `_ignore_causal_mask_sdpa` in eager mode on TPU. It is the eager-mode counterpart (see Notes).
- [#49542](https://github.com/huggingface/transformers/pull/49542) (open) concerns sliding-window decode graphs, and [#49415](https://github.com/huggingface/transformers/pull/49415) (open) adds static cache support to assisted decoding. Both are related, neither is a duplicate.

## Issue text

Kept for the record only. #49354 already fixed this. Do not post.

---

**Title:** Compiled forward with `StaticCache` and more than one new token breaks the graph at `q_offset == 0` in `_ignore_causal_mask_sdpa`

### System Info

- transformers 5.18.0, torch 2.15.0, Python 3.12.11
- macOS 27.0.1, Apple M1 Max. Reproduced on MPS and on CPU.

### Who can help?

@vasqu @Cyrilvallez

### Reproduction

A `torch.compile`d forward with a `StaticCache`, `T > 1` new tokens and `attention_mask=None` reaches the second shortcut in `_ignore_causal_mask_sdpa` (`masking_utils.py:279` in 5.18.0):

```python
    if q_offset == 0 and (
        padding_mask is None or (fast_all(padding_mask[:, :q_length]) and fast_all(~padding_mask[:, q_length:]))
    ):
```

With a static cache, `q_offset` is the layer's `cumulative_length`, a 0-d device tensor (`masking_utils.py:835-838`). Python branches on its value, so Dynamo breaks the graph ("Data-dependent branching"), and on a GPU the branch reads the device from the host. `T = 1` is not affected, because `create_causal_mask` already turns the skip off for compileable caches at one token (`masking_utils.py:956-958`). The first shortcut does not catch `T > 1` either, because `kv_length` is the static cache length, not `q_length`.

Standalone repro, with a tiny random Qwen3 and no downloads ([link to `repro_hf_static_cache_graph_break.py`]). It prefills 64 tokens eagerly, then runs a 4-token forward with `torch.compile(dynamic=False)`:

```python
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, StaticCache

dev = "cpu"  # same result on "mps"
cfg = Qwen3Config(vocab_size=1024, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                  num_attention_heads=8, num_key_value_heads=4, head_dim=32)
cfg._attn_implementation = "sdpa"
model = Qwen3ForCausalLM(cfg).to(dev).eval()
cache = StaticCache(config=cfg, max_cache_len=96)

def step(x, pos, mask):
    return model(x, past_key_values=cache, use_cache=True, cache_position=pos, attention_mask=mask).logits

with torch.no_grad():
    model(torch.randint(0, 1024, (1, 64), device=dev), past_key_values=cache, use_cache=True,
          cache_position=torch.arange(64, device=dev))
    x, pos = torch.randint(0, 1024, (1, 4), device=dev), torch.arange(64, 68, device=dev)
    ex = torch._dynamo.explain(step)(x, pos, None)
    print(ex.graph_count, ex.graph_break_count, [r.reason.splitlines()[0] for r in ex.break_reasons])
```

Results (2026-11-11, torch 2.15.0):

| | no `attention_mask` | all-ones 2D `attention_mask` |
|---|---|---|
| graphs (`torch._dynamo.explain`), CPU and MPS | 6 (5 breaks, all at `masking_utils.py:279`) | 1 |
| compiled frames (inductor), CPU and MPS | 6 | 1 |
| compiled 4-token forward, MPS, median of 50 | 4.28 ms | 3.48 ms |
| compiled 4-token forward, CPU, median of 50 | 1.72 ms | 1.43 ms |

The paired-call ratio is 1.24x on MPS and 1.21x on CPU. Logits are identical between the two arms. In a 28-layer Qwen3-1.7B on MPS, the same break cost about 1.5 to 2 ms per verify forward (our findings log, 2026-11-11).

### Expected behavior

A compiled forward with a static cache should be one graph for any `T`, as it is at `T = 1` and as it is when a 2D mask is passed.

### Suggested fix

When `q_offset` is a tensor and is being traced, skip the `q_offset == 0` shortcut and build the mask, as is already done for a traced `padding_mask`. Keeping the comparison as a tensor op would also avoid the break, but the result must be a Python bool here. #49354 does the first.

---

## Notes (not part of the issue)

- **Evidence**, 2026-11-11, raw logs in the lane scratch directory:
  - `bench/repro_hf_static_cache_graph_break.py` on CPU at T = 4 (6 graphs) and T = 1 (1 graph, no break), and on MPS through the GPU lock at T = 4.
  - The same repro with main's one-line check applied in-process, without editing the installed package. It gave 1 graph and 1 frame without a mask, and a no-mask / ones-mask ratio of 0.99 on CPU. That confirms #49354 removes the break.
- **Eager mode is untested.** In eager mode, main still runs `q_offset == 0` on a device tensor for static-cache forwards with T > 1, which means one host sync per mask creation. That is the eager-mode counterpart to #50267, which stalled. Unmeasured here, so not a claim.
- **Correction.** The findings log calls this "a candidate upstream issue" (2026-11-11 entry, side findings). It is fixed upstream; that line should say so.
