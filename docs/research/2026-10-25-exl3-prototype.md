# EXL3 on Metal: a correct prototype that does not justify a speed port

2026-10-25. M1 Max 32 GB, torch 2.14.1. Published here on 2026-11-09 from my private review of the same date. The measurements are unchanged; what I removed is local paths and process notes.

**Question.** Can EXL3 trellis-quantized LLM weights run on Apple GPUs with the same weight-only technique as the other formats here, and would that be faster than the GGUF path llama.cpp already ships on Metal?

**Answer.** Correct, yes. Faster, no. A bounded EXL3 decoder passes independent numerical checks on real 27B-model matrices, and after a targeted optimization it is still 2.4 to 4.4 times slower than the installed ggml Metal kernels on the same operator geometry. I kept the installed GGUF runtime and stopped. The prototype stays in `bench/` for future kernel research. This result does not show that EXL3 is intrinsically slow on Metal; it shows that this implementation is, and that a full runtime port is not justified on this evidence.

The [machine-readable evidence](2026-10-25-exl3-prototype.json) holds every sample, median, identity hash, correctness result and guard result quoted below.

## 1. What was tested

**Model.** Real EXL3 matrices from `orcarouter/OrcaSAQ-2-27B`, pinned revision `443cd64a438ccbfdf55dd19dbc7dd18ffa41d14b`. Five matrices: the fractional-bit layer-11 K projection, the layer-0 linear-attention gate, and the layer-0 FFN up, gate and down projections. Payload ranges and local hashes were checked against the pinned manifest and header. The model files are external and were not modified.

| Matrix | EXL3 tensor | K | N | Bits per weight |
|---|---|---:|---:|---:|
| K projection | layer 11 `k_proj` | 5120 | 1024 | 3.5 |
| Attention gate | layer 0 `linear_attn.in_proj_z` | 5120 | 6144 | 3 |
| FFN up | layer 0 `mlp.up_proj` | 5120 | 17408 | 3 |
| FFN gate | layer 0 `mlp.gate_proj` | 5120 | 17408 | 3.5 |
| FFN down | layer 0 `mlp.down_proj` | 17408 | 5120 | 3 |

**Code.** `bench/exl3_probe.py` with `bench/exl3_probe.metal`. The JSON records the SHA-256 of both files as run (`source_sha256`, `shader_sha256`). The shader in this repo hashes to the same value. The probe script matched as first committed; a later one-line change to its GPU lock path changes its hash but not what it measures. The oracle is `bench/exl3_reference.py`, the upstream ExLlamaV3 trellis reference, vendored unchanged under its MIT license. Its digest (`oracle_sha256`, `072b5233...`) matches the committed file. The oracle decodes each matrix to a float64 dense weight on the CPU and the expected output is a float64 matmul against it.

**Paths timed.** Six per matrix and row count:

- `fused`: the original fused decode-and-multiply kernel, allocating its scratch and output on each call.
- `fused_reuse`: the same kernel with reusable scratch and output buffers.
- `nmajor_arithmetic`: a lossless output-major repack of the packed tiles with the bit-window arithmetic hoisted out of the inner loop. This is the "optimized EXL3" path in the tables below.
- `nmajor_lut`: the same repack, decoding through a 128 KiB lookup table instead of arithmetic.
- `reconstruct`: unpack the whole weight to fp16, then a stock GEMM. This is the dequantize-per-call shape of the problem.
- `resident_fp16`: a plain fp16 GEMM against the already decoded weight, as a diagnostic floor. It is not an EXL3 path.

Every EXL3 path includes the input Hadamard transform, the decode or multiply, and the output Hadamard with scales. The repack and lookup table are built and uploaded outside the timers.

**Correctness.** Each path's full output is compared to the float64 oracle at M=1 and M=4, then checked again with a zero input to catch nonfinite scale or accumulator handling. Output and scratch buffers are poisoned with NaN before every correctness call, including reused buffers, so a missing or partial write cannot inherit a previous variant's answer. All six paths on all five matrices passed at both row counts. Worst relative RMS across the five EXL3 paths was 3.7e-4 and the worst scaled maximum error was 7.0e-4 (JSON `correctness`). A failed numerical gate skips timing entirely.

**Timing method.** Seven interleaved rounds per matrix and row count, with the path order shuffled each round. Before every timed call a separate 128 MiB fp32 buffer is multiplied in place and the device is synchronized, outside the stopwatch. That applies cache pressure; it is not a proven cold cache. Each timed call runs from a synchronized device to a synchronized device, so Python dispatch of the three kernels and the Hadamards are inside the number. Inputs are the same exactly representable fp16 values the GGUF probe uses, `((17 i + 13) mod 257 - 128) / 128`.

**GGUF baseline.** `bench/gguf_op_probe.cpp`, documented in [bench/gguf_op_probe.md](../../bench/gguf_op_probe.md). It runs one real quantized tensor from a local GGUF through the installed Homebrew ggml 0.26.0 Metal backend as a single resident `ggml_mul_mat` graph. Two local GGUF files have layer-0 tensors with exactly the EXL3 matrices' K and N: "Orca", whose four measured tensors are IQ4_XS, and "Nano", whose are IQ3_S and IQ2_S. gguf_op_probe.md gives the tensor-by-tensor mapping, the packed byte counts, and both files' full-file hashes and sizes. Each GGUF case is one process with three warmups and ten timed calls, the same 128 MiB eviction treatment, and a CPU-dequantized double-accumulation reference on sampled output channels. All GGUF cases passed with worst relative RMS 1.15e-7.

## 2. Results

Medians in milliseconds. EXL3: seven interleaved rounds. GGUF: ten timed calls, from the 128 MiB eviction runs.

| Matrix | M | Optimized EXL3 | Nano GGUF | Orca GGUF |
|---|---:|---:|---:|---:|
| Attention gate | 1 | 1.171 | 0.396 | 0.368 |
| Attention gate | 4 | 2.979 | 0.754 | 0.686 |
| FFN up | 1 | 2.071 | 0.636 | 0.596 |
| FFN up | 4 | 6.624 | 1.610 | 1.530 |
| FFN gate | 1 | 1.808 | 0.643 | 0.584 |
| FFN gate | 4 | 6.323 | 1.427 | 1.544 |
| FFN down | 1 | 1.727 | 0.715 | 0.616 |
| FFN down | 4 | 5.945 | 1.466 | 1.721 |

The K projection has no GGUF counterpart in the probe and is omitted from the comparison. Its EXL3 medians are in the JSON.

**The ratio, re-derived from the JSON.** Optimized EXL3 median divided by the GGUF median for the same tensor and M. The ratios come from the unrounded JSON medians, so dividing the rounded values shown can differ in the last digit:

| Matrix | M | vs Nano | vs Orca |
|---|---:|---:|---:|
| Attention gate | 1 | 1.171 / 0.396 = 2.95× | 1.171 / 0.368 = 3.18× |
| Attention gate | 4 | 2.979 / 0.754 = 3.95× | 2.979 / 0.686 = 4.34× |
| FFN up | 1 | 2.071 / 0.636 = 3.26× | 2.071 / 0.596 = 3.48× |
| FFN up | 4 | 6.624 / 1.610 = 4.12× | 6.624 / 1.530 = 4.33× |
| FFN gate | 1 | 1.808 / 0.643 = 2.81× | 1.808 / 0.584 = 3.10× |
| FFN gate | 4 | 6.323 / 1.427 = 4.43× | 6.323 / 1.544 = 4.09× |
| FFN down | 1 | 1.727 / 0.715 = 2.42× | 1.727 / 0.616 = 2.80× |
| FFN down | 4 | 5.945 / 1.466 = 4.06× | 5.945 / 1.721 = 3.45× |

Against Nano: 2.42 to 3.26× at M=1, 3.95 to 4.43× at M=4, 2.42 to 4.43× overall. Against Orca: 2.80 to 3.48× at M=1, 3.45 to 4.34× at M=4. The "2.4 to 4.4×" in the README is the Nano range.

**All six paths.** Medians in milliseconds, so the optimization can be seen against the paths it replaced:

| Matrix | M | fused | fused_reuse | nmajor_arithmetic | nmajor_lut | reconstruct | resident_fp16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| K projection | 1 | 1.250 | 1.158 | 0.740 | 1.262 | 4.384 | 0.589 |
| K projection | 4 | 2.512 | 2.494 | 1.296 | 3.159 | 4.422 | 0.824 |
| Attention gate | 1 | 1.782 | 1.902 | 1.171 | 2.379 | 10.106 | 0.660 |
| Attention gate | 4 | 6.848 | 6.703 | 2.979 | 9.305 | 11.865 | 0.927 |
| FFN up | 1 | 4.524 | 4.370 | 2.071 | 6.051 | 26.760 | 1.052 |
| FFN up | 4 | 15.756 | 15.818 | 6.624 | 21.970 | 27.321 | 1.610 |
| FFN gate | 1 | 4.180 | 4.156 | 1.808 | 5.733 | 26.518 | 0.902 |
| FFN gate | 4 | 15.556 | 15.599 | 6.323 | 21.701 | 26.926 | 1.294 |
| FFN down | 1 | 4.097 | 4.112 | 1.727 | 5.648 | 29.653 | 0.947 |
| FFN down | 4 | 15.456 | 15.453 | 5.945 | 21.603 | 29.881 | 1.159 |

The output-major repack cuts the FFN cost to 0.42 to 0.46 of the original fused kernel at M=1 (4.524 / 2.071 = 2.19× for FFN up, 4.097 / 1.727 = 2.37× for FFN down) and 0.38 to 0.42 at M=4 (15.756 / 6.624 = 2.38×, 15.456 / 5.945 = 2.60×). The lookup table lost to arithmetic on every matrix. Reusing buffers made almost no difference, so allocation is not where the time goes. The gap to GGUF widens from M=1 to M=4 while the Python dispatch count stays fixed, so dispatch overhead alone cannot explain it.

**Resource use.** The EXL3 run took 24.15 s of wall time and peaked at 4.72 GiB process-group RSS, with no memory-pressure warning and no swap growth under the 18 GiB RSS / 4 GiB swap-growth guard. The GGUF processes each ran for under half a second, shorter than the resource sampler interval, so their sampled RSS is not a peak; the JSON records their allocated graph buffer bytes instead: 297,074,688 to 316,145,664 bytes for the FFN cases, including both eviction buffers.

## 3. Caveats

These apply to every ratio above. They are the reason the result is "this implementation is slower", not "EXL3 is slower".

1. **Complete-call wall time, not kernel time.** The EXL3 number is wall time from one `torch.mps.synchronize()` to the next, and includes three Python-submitted kernels, the input and output Hadamards, scales, and the synchronize itself. The GGML number is one resident C++ graph dispatch plus synchronize. Neither is pure device time. Torch MPS event timing was tried and removed: the installed torch 2.14.1 `elapsedTime` wait hung in an earlier run, which was cancelled; the successful run collected wall time only (JSON `event_timing`). How much of the EXL3 time is host overhead is not separated here.
2. **fp16 versus f32 I/O.** The EXL3 paths take fp16 input and produce fp16 output. The GGML graph takes f32 input and produces f32 output. Both are fed the same exactly representable values, so the numerical comparison is fair, but the memory traffic and output rounding differ. The `reconstruct` path also adds a half rounding at the transformed activation and the GEMM output.
3. **Different model lineages.** The EXL3 file is OrcaSAQ-2-27B at revision 443cd64a. The two GGUF files hold different weights and declare their own source checkpoints; they match the EXL3 matrices in shape, not in content. Each format was checked against its own decoded-weight oracle, not against a shared BF16 source. This is a matched-geometry operator comparison, not a quality comparison and not a whole-model speed claim.
4. **Cache treatment is pressure, not a flush.** A 128 MiB read/write between calls is the same treatment on both sides. It does not prove a cold cache, and an isolated projection can sit in system-level cache in a way a full decoder cannot.
5. **Small n.** Seven interleaved rounds per EXL3 path and ten calls per GGUF case, each from one process. The GGUF attention-gate runs were also measured without eviction (JSON) and the medians moved by 2.4% or less, but no cross-process variance was collected.
6. **Decode traffic.** The whole-model ordinary-decode traffic proxy for the EXL3 file, computed from the real headers and excluding the embedding lookup and the unused MTP head on both sides, was 11.57% lower than Nano's. That proxy is not a speed ceiling and is not reproduced from the JSON; it comes from the header walk in the original review. Nano's separately measured MTP gain is a route to a faster decode that needs no new weight kernels.

## 4. Verification

- All six paths, five matrices and both row counts passed the float64 oracle gate and the zero-input gate before any timing (JSON `correctness`).
- All twenty GGUF cases passed the CPU-dequantized reference, finite-output and immutable-source checks (JSON `gguf[*].result.status`).
- The installed ggml Metal library hashes were unchanged before and after the GGUF runs (JSON `gguf_libraries`, matching the table in gguf_op_probe.md).
- A second reviewer found one correctness hole in an early version: reused scratch could carry a previous variant's result. The fix is the NaN poisoning described above, with regression tests. The follow-up review found nothing further.
- Sixteen focused CPU tests for the probe passed. The repository suites also passed under the GPU guard: 709 passed, 16 skipped, 17 subtests, then 25 GGUF node tests.
- No production kernel, serving default, model file or memory guard was changed.

## 5. What this does and does not decide

It decides that these kernels are not worth integrating for speed. It does not decide that EXL3 cannot be fast on Metal. A cooperative decoder that shares trellis state across a threadgroup, or deeper fusion of the Hadamards into the multiply, could change the answer. That would be a new speed hypothesis with its own probe, not a port of this one. Until then the installed GGUF runtime stays, and the per-format GGUF baselines in gguf_op_probe.md are the numbers any new EXL3 attempt has to beat.
