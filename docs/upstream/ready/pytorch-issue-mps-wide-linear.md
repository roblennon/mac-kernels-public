# PyTorch issue: transposed-weight matmul on MPS at wide K

Destination: https://github.com/pytorch/pytorch/issues/new (Bug Report form; suggested labels `module: mps`, `module: performance`). Not filed. The repro is inline, so no gist is needed.

## Title

[MPS] `F.linear` / `matmul` with a transposed weight is up to 8.7x slower than with a contiguous `[K, N]` weight at large K

## Body

### 🐛 Describe the bug

On an M1 Max, `F.linear(x, w)` is much slower than the same product against a pre-transposed contiguous weight, `x @ w.t().contiguous()`, once the inner dimension K is large. At M = 4608, N = 6144 the gap is about 1.9x at K = 16384 and 3.6x at K = 32768 in fp16, and 4.2x and 8.4 to 8.7x in fp32. Outputs are identical (maximum absolute difference zero). Release 2.14.1 and nightly 2.16.0.dev20261009 behave the same.

This is the `nn.Linear` layout: the weight is stored `[N, K]` and multiplied transposed. Large-K projections, such as transformer MLP down-projections in diffusion models or LLM prefill, hit it. Single-token decode (small M) is a different path and is not covered here.

Every variant whose right-hand operand is a transposed `[N, K]` tensor is slow, whichever PyTorch path it takes: `F.linear(x, w)` (on macOS 15, `_mps_linear_nograph`), `F.linear` with a non-contiguous input (the MPSGraph path), `x @ w.t()` and `torch.mm(x, w.t())` (`mm_out_mps_impl`), and `torch.mm(w, x.t()).t()`. They match `F.linear` within noise. Only a contiguous right-hand operand is fast. A transposed left-hand operand (`torch.mm(xt.t(), wt)`) stays within 11% of the contiguous variant.

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

<details><summary>Full table with every variant and intermediate K</summary>

torch 2.14.1, macOS 15.7.9, Apple M1 Max, MPS available
M=4608, N=6144; min ms over 3 rounds x 5 iterations

| dtype | K | F.linear(x, w) | x @ w.t() | torch.mm(x, w.t()) | F.linear(x_nc, w) | torch.mm(w, x.t()).t() | torch.mm(xt.t(), wt) (A transposed) | x @ wt (contiguous) | torch.mm(x, wt) | F.linear / contiguous | max abs diff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| float16 | 4096 | 25.0 | 25.2 | 25.1 | 25.1 | 25.1 | 25.8 | 25.8 | 25.6 | 0.97 | 0.0e+00 |
| float16 | 6144 | 37.6 | 37.7 | 37.6 | 37.7 | 37.8 | 38.4 | 38.3 | 38.6 | 0.98 | 0.0e+00 |
| float16 | 8192 | 60.1 | 59.4 | 59.6 | 59.7 | 57.9 | 51.2 | 51.1 | 51.4 | 1.18 | 0.0e+00 |
| float16 | 10240 | 81.8 | 80.6 | 81.3 | 81.5 | 79.8 | 63.9 | 64.0 | 64.2 | 1.28 | 0.0e+00 |
| float16 | 12288 | 100.9 | 101.7 | 101.4 | 100.6 | 99.8 | 76.7 | 76.4 | 76.3 | 1.32 | 0.0e+00 |
| float16 | 16384 | 195.0 | 196.9 | 196.0 | 180.6 | 195.2 | 102.1 | 102.1 | 101.9 | 1.91 | 0.0e+00 |
| float16 | 24576 | 337.0 | 337.2 | 333.4 | 339.6 | 343.5 | 153.8 | 152.7 | 152.7 | 2.21 | 0.0e+00 |
| float16 | 32768 | 714.6 | 717.0 | 719.9 | 715.4 | 753.9 | 201.1 | 199.3 | 199.6 | 3.59 | 0.0e+00 |
| float32 | 4096 | 43.9 | 44.6 | 44.8 | 42.4 | 42.8 | 29.1 | 27.2 | 27.3 | 1.61 | 0.0e+00 |
| float32 | 6144 | 69.8 | 69.7 | 69.1 | 68.7 | 68.9 | 44.0 | 40.9 | 40.8 | 1.70 | 0.0e+00 |
| float32 | 8192 | 135.7 | 137.6 | 137.3 | 127.3 | 136.9 | 59.8 | 54.8 | 54.6 | 2.48 | 0.0e+00 |
| float32 | 10240 | 179.4 | 180.7 | 180.0 | 179.5 | 183.1 | 75.0 | 68.1 | 68.0 | 2.63 | 0.0e+00 |
| float32 | 12288 | 240.8 | 242.9 | 241.5 | 245.2 | 250.2 | 90.9 | 81.9 | 81.6 | 2.94 | 0.0e+00 |
| float32 | 16384 | 466.0 | 479.7 | 478.6 | 470.1 | 517.5 | 122.5 | 110.0 | 110.0 | 4.24 | 0.0e+00 |
| float32 | 24576 | 1125.3 | 1107.0 | 1122.4 | 1127.8 | 1175.2 | 181.3 | 163.8 | 163.7 | 6.87 | 0.0e+00 |
| float32 | 32768 | 2067.7 | 2122.5 | 2111.1 | 2081.2 | 2293.9 | 258.7 | 245.9 | 245.3 | 8.41 | 0.0e+00 |
torch 2.16.0.dev20261009, macOS 15.7.9, Apple M1 Max, MPS available
M=4608, N=6144; min ms over 3 rounds x 5 iterations

| dtype | K | F.linear(x, w) | x @ w.t() | torch.mm(x, w.t()) | F.linear(x_nc, w) | torch.mm(w, x.t()).t() | torch.mm(xt.t(), wt) (A transposed) | x @ wt (contiguous) | torch.mm(x, wt) | F.linear / contiguous | max abs diff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| float16 | 4096 | 24.9 | 25.0 | 24.9 | 24.9 | 24.8 | 25.6 | 25.5 | 25.5 | 0.98 | 0.0e+00 |
| float16 | 6144 | 37.4 | 37.6 | 37.6 | 37.6 | 37.7 | 38.3 | 38.2 | 38.2 | 0.98 | 0.0e+00 |
| float16 | 8192 | 58.8 | 58.8 | 59.2 | 58.2 | 57.2 | 51.0 | 51.0 | 50.9 | 1.15 | 0.0e+00 |
| float16 | 10240 | 79.6 | 80.2 | 80.5 | 78.9 | 78.5 | 63.6 | 63.4 | 63.5 | 1.25 | 0.0e+00 |
| float16 | 12288 | 98.5 | 99.7 | 98.2 | 99.1 | 98.1 | 76.3 | 76.2 | 75.9 | 1.29 | 0.0e+00 |
| float16 | 16384 | 190.9 | 189.2 | 191.7 | 176.2 | 192.5 | 101.5 | 101.2 | 101.2 | 1.89 | 0.0e+00 |
| float16 | 24576 | 333.2 | 333.6 | 335.0 | 336.6 | 345.5 | 152.1 | 151.4 | 151.3 | 2.20 | 0.0e+00 |
| float16 | 32768 | 719.3 | 719.5 | 717.6 | 712.6 | 763.7 | 200.4 | 198.7 | 198.7 | 3.62 | 0.0e+00 |
| float32 | 4096 | 42.8 | 42.8 | 43.1 | 41.5 | 42.9 | 28.7 | 27.1 | 27.0 | 1.58 | 0.0e+00 |
| float32 | 6144 | 69.2 | 68.9 | 68.6 | 68.7 | 68.3 | 43.5 | 40.5 | 40.5 | 1.71 | 0.0e+00 |
| float32 | 8192 | 135.3 | 134.3 | 136.1 | 126.6 | 136.2 | 59.4 | 54.2 | 54.5 | 2.50 | 0.0e+00 |
| float32 | 10240 | 177.7 | 179.5 | 179.0 | 177.3 | 182.3 | 74.1 | 67.5 | 67.6 | 2.63 | 0.0e+00 |
| float32 | 12288 | 240.3 | 240.3 | 240.4 | 242.9 | 249.7 | 89.7 | 81.1 | 81.5 | 2.96 | 0.0e+00 |
| float32 | 16384 | 480.6 | 480.9 | 481.0 | 479.7 | 516.7 | 121.5 | 108.8 | 108.9 | 4.42 | 0.0e+00 |
| float32 | 24576 | 1160.1 | 1154.5 | 1147.4 | 1149.9 | 1192.3 | 181.9 | 163.2 | 163.5 | 7.11 | 0.0e+00 |
| float32 | 32768 | 2139.4 | 2135.4 | 2115.8 | 2075.7 | 2314.9 | 262.5 | 245.1 | 244.6 | 8.73 | 0.0e+00 |


</details>

Workaround: keep a contiguous `[K, N]` copy of the weight and call `x @ wt`. That costs one extra weight copy in memory (0.8 GB for the fp32 K = 32768 weight above). For `nn.Linear`, PyTorch could cache such a copy per weight version for large K, but whether the fix belongs in PyTorch or in MPS is for maintainers to judge.

Only one machine was measured. Numbers from M2 to M5 GPUs and from macOS 26 would show whether the effect is general.

Related, not duplicates: #198182 (native Metal matmul kernels slower than MPSGraph on ordinary shapes), #195163 (bmm wrong at K >= 32767 on M1/M2), #189847 (small-M 3-D decode).

### Versions

- PyTorch 2.14.1 (pip wheel) and 2.16.0.dev20261009 (nightly wheel), Python 3.12
- macOS 15.7.9 (24G830), arm64
- Apple M1 Max, 32 GB, 32-core GPU
