# PyTorch issue: MPS matmul with a transposed right-hand operand is up to 8.7x slower at wide K

Prepared for [PyTorch issues](https://github.com/pytorch/pytorch/issues/new) (Bug Report; suggested labels `module: mps`, `module: performance`). Not filed. Refreshed October 9, 2026: the gap reproduces on release 2.14.1 and on nightly 2.16.0.dev20261009.

## Duplicate triage (refreshed 2026-10-09)

About 15 searches of pytorch/pytorch issues and PRs since April 2026 (`gh api search/issues`), on top of the October 3 search. None reports this transposed-weight slowdown. Nearest:
- [#198182](https://github.com/pytorch/pytorch/issues/198182), open (2026-09-22): the native Metal matmul kernels are 6–10× slower than MPSGraph on ordinary shapes (M4 Max). Related, not a duplicate. It also means "route NT products to the Metal kernels" is not a fix to suggest.
- [#195163](https://github.com/pytorch/pytorch/issues/195163), open: bmm wrong on M1/M2 at K ≥ 32767, with fix [#195012](https://github.com/pytorch/pytorch/pull/195012) open. K = 32768 sits exactly at the 2-D reroute threshold; our results there equal the contiguous variant.
- [#189847](https://github.com/pytorch/pytorch/issues/189847) / [#189855](https://github.com/pytorch/pytorch/pull/189855) (small-M 3-D decode) and [#198721](https://github.com/pytorch/pytorch/pull/198721) (gemv kernels, merged 2026-10-02): different shapes.
- No merged change since July alters the transposed-operand path at large M. The nightly run is the proof.

## Issue text

---

**Title:** [MPS] `F.linear` / `matmul` with a transposed weight is up to 8.7x slower than with a contiguous `[K, N]` weight at large K

### 🐛 Describe the bug

On an M1 Max, `F.linear(x, w)` is much slower than the same product against a pre-transposed contiguous weight, `x @ w.t().contiguous()`, once the inner dimension K is large. At M = 4608, N = 6144 the gap is about 1.9× at K = 16384 and 3.6× at K = 32768 in fp16, and 4.2× and 8.4–8.7× in fp32. Outputs are identical (maximum absolute difference zero). Release 2.14.1 and nightly 2.16.0.dev20261009 behave the same.

This is the `nn.Linear` layout: the weight is stored `[N, K]` and multiplied transposed. Large-K projections, such as transformer MLP down-projections in diffusion models or LLM prefill, hit it. Single-token decode (small M) is a different path and is not covered here.

Every variant whose right-hand operand is a transposed `[N, K]` tensor is slow, whichever PyTorch path it takes: `F.linear(x, w)` (on macOS 15, `_mps_linear_nograph`), `F.linear` with a non-contiguous input (the MPSGraph path), `x @ w.t()` and `torch.mm(x, w.t())` (`mm_out_mps_impl`), and `torch.mm(w, x.t()).t()`. They match `F.linear` within noise. Only a contiguous right-hand operand is fast. A transposed *left*-hand operand (`torch.mm(xt.t(), wt)`) stays within 11% of the contiguous variant.

Because every PyTorch path shows the same behavior, the cause is likely below PyTorch, in how MPS / MPSGraph handles a right-hand operand contiguous along K. That is an inference; no kernel profile was taken.

Repro (standalone, torch only):

```python
import time, torch, torch.nn.functional as F
torch.manual_seed(0)

def best_ms(fn, iters=5):
    t = float("inf")
    for _ in range(iters):
        torch.mps.synchronize(); t0 = time.perf_counter(); fn(); torch.mps.synchronize()
        t = min(t, (time.perf_counter() - t0) * 1e3)
    return t

M, N = 4608, 6144
for dtype in (torch.float16, torch.float32):
    for K in (4096, 8192, 16384, 32768):          # K > 32768 takes a separate reroute on M1/M2
        x = torch.randn(M, K, device="mps", dtype=dtype)
        w = torch.randn(N, K, device="mps", dtype=dtype) / K ** 0.5   # nn.Linear layout [N, K]
        wt = w.t().contiguous()                                       # [K, N], made once, not timed
        F.linear(x, w); x @ wt                                        # warm up
        a, b = best_ms(lambda: F.linear(x, w)), best_ms(lambda: x @ wt)
        assert torch.equal(F.linear(x, w), x @ wt)
        print(dtype, K, f"F.linear {a:.1f} ms, x @ wt {b:.1f} ms, ratio {a / b:.2f}")
```

Minimum milliseconds per call, M = 4608, N = 6144, three interleaved rounds of five calls after warm-up. Output allocation is included in every timing; the one-off `w.t().contiguous()` copy is not. At fp32, K = 32768 the script holds about 3.4 GB of tensors.

| dtype | K | `F.linear(x, w)` 2.14.1 / nightly | `x @ wt` 2.14.1 / nightly | `mm(xt.t(), wt)` 2.14.1 | ratio 2.14.1 / nightly |
|---|---|---|---|---|---|
| float16 | 4096 | 25.0 / 24.9 | 25.8 / 25.5 | 25.8 | 0.97 / 0.98 |
| float16 | 8192 | 60.1 / 58.8 | 51.1 / 51.0 | 51.2 | 1.18 / 1.15 |
| float16 | 16384 | 195.0 / 190.9 | 102.1 / 101.2 | 102.1 | 1.91 / 1.89 |
| float16 | 32768 | 714.6 / 719.3 | 199.3 / 198.7 | 201.1 | 3.59 / 3.62 |
| float32 | 4096 | 43.9 / 42.8 | 27.2 / 27.1 | 29.1 | 1.61 / 1.58 |
| float32 | 8192 | 135.7 / 135.3 | 54.8 / 54.2 | 59.8 | 2.48 / 2.50 |
| float32 | 16384 | 466.0 / 480.6 | 110.0 / 108.8 | 122.5 | 4.24 / 4.42 |
| float32 | 32768 | 2067.7 / 2139.4 | 245.9 / 245.1 | 258.7 | 8.41 / 8.73 |

<details><summary>Full table with every variant (link the script's output here)</summary>

The full runner adds `x @ w.t()`, `torch.mm(x, w.t())`, `F.linear(x_nc, w)`, `torch.mm(w, x.t()).t()` and `torch.mm(x, wt)`, plus intermediate K values.

</details>

Workaround: keep a contiguous `[K, N]` copy of the weight and call `x @ wt`. That costs one extra weight copy in memory (0.8 GB for the fp32 K = 32768 weight above). For `nn.Linear`, PyTorch could cache such a copy per weight version for large K, but whether the fix belongs in PyTorch or in MPS is for maintainers to judge.

Only one machine was measured. Numbers from M2–M5 GPUs and from macOS 26 would show whether the effect is general.

### Versions

- PyTorch 2.14.1 (pip wheel) and 2.16.0.dev20261009 (nightly wheel), Python 3.12
- macOS 15.7.9 (24G830), arm64
- Apple M1 Max, 32 GB, 32-core GPU

---

## Notes (not part of the issue)

- Evidence: October 9 runs under the GPU guard, raw output in `logs/repro-mps-wide-linear-2026-10-09.log` (ignored by Git). The October 4 run (2.14.1 only) agreed within a few percent.
- Before filing: paste the full table into the `<details>` block or a gist, and link the repro script from a public location.
- Dropped from the earlier draft: source line numbers (Linear.mm and LinearAlgebra.mm changed in October), the unmeasured K-chunking idea, the "every step of diffusion and LLM workloads" claim, and the NT/NN kernel-choice statement presented as fact.
- An MPS profiler trace (`PYTORCH_MPS_LOG_PROFILE_INFO` or Instruments) would name the kernels if a maintainer asks. Apple Feedback is a separate, later decision.
