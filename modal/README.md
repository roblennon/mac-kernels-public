# modal/

Cloud compute for the parts of this research that do not need an Apple GPU. Modal runs Debian Linux containers on NVIDIA GPUs (T4, L4, A10, A100, H100). There is no macOS, Metal or MPS on Modal, so none of the kernels under `vendor/comfy-kitchen/comfy_kitchen/backends/mps/` can run there, and none of the numbers in [docs/findings.md](../docs/findings.md) can be reproduced there. Every Apple measurement in this repo stays a measurement on the one M1 Max.

## What Modal is for here

- **CUDA reference baselines.** The same GEMM shapes (4608x6144x6144, 4608x6144x16384, 4608x16384x6144) in fp16, bf16, transposed-weight fp16, dequantize-per-call INT8 and true INT8 (`torch._int_mm`, which MPS lacked before torch 2.15) on T4, A10G and A100, so the M1 Max table at findings.md:17-19 has a cross-platform column. These are the `cuda_gemm_bench_*` functions in `modal_app.py`; the 2026-11-09 run is in `results/`. It anchors "what does this GEMM cost on a cheap NVIDIA part", nothing more; it is not a Metal-versus-CUDA kernel comparison, because the Metal side is our kernel and the CUDA side is cuBLAS.
- **Reference numerics on CPU.** GGUF dequantization checked against the block formulas in [docs/research/2026-10-03-gguf-on-metal.md](../docs/research/2026-10-03-gguf-on-metal.md) using the `gguf` package (`gguf_dequant_check`); the EXL3 reference in `bench/exl3_reference.py`; fp64 oracles over whole layers, fanned out with `.map()` across many small CPU containers. The Mac already computes fp64 on CPU; Modal adds parallelism and frees the laptop, not capability.
- **Checkpoint work that competes with 32 GB of unified memory.** INT8 conversion of 20 GB checkpoints (`bench/wan_int8_convert.py`, `bench/qwen_int8_convert.py` logic) with the source downloaded in-container to a Volume, and GGUF header surveys (`bench/gguf_headers.py`) over many files.
- **Hardware-agnostic training ablations.** bf16-versus-fp16 compute, more seeds, portability to a bf16 base. The LoRA results in this repo were trained on Metal; a CUDA re-run tests the method, not the kernels.
- **Quality metrics.** PSNR/SSIM/perceptual metrics over images and videos generated on the Mac, and a bf16 reference generation the Mac cannot hold in memory (E1 in the open experiments).

## What Modal cannot do

- Run or time any Metal kernel, MPS op, or the patched ComfyUI/comfy-kitchen MPS backend.
- Say anything about M2/M3/M4/M5 or macOS 26 (E4 needs other Apple machines).
- Stand in for the Apple numbers. Any cross-platform claim must say the CUDA leg ran on Modal Linux containers.

## Cost

Starter plan: $30 of compute a month, billed beyond that; a payment method must be on file even for GPU runs inside the allowance. T4 $0.59/h, L4 $0.80/h, A100-40GB $2.10/h; CPU $0.047 per physical core-hour, memory $0.008 per GiB-hour. The scaffold's GPU benchmark costs a few cents. Prices read from modal.com/pricing on 2026-11-09.

## First result, 2026-11-09

`results/2026-11-09-cuda-gemm-anchor.json`, torch 2.15.0+cu132, median of 20 calls after 5 warmups. Milliseconds per call.

| Shape (M, K, N) | GPU | fp16 F.linear | bf16 | fp16, transposed weight | INT8 dequant per call + fp16 | INT8 `_int_mm` |
| --- | --- | --- | --- | --- | --- | --- |
| 4608, 6144, 6144 | T4 | 16.9 | 151.7 | 19.7 | 18.9 | 9.9 |
| 4608, 6144, 16384 | T4 | 46.8 | 429.0 | 54.1 | 57.1 | 29.8 |
| 4608, 16384, 6144 | T4 | 52.6 | 497.4 | 80.7 | 60.5 | 30.6 |
| 4608, 6144, 6144 | A10G | 5.28 | 5.18 | 5.35 | 5.85 | 2.75 |
| 4608, 6144, 16384 | A10G | 13.8 | 13.6 | 13.6 | 15.3 | 7.03 |
| 4608, 16384, 6144 | A10G | 13.7 | 13.6 | 15.4 | 15.2 | 6.96 |
| 4608, 6144, 6144 | A100 80GB | 1.51 | 1.52 | 1.48 | 1.57 | 0.84 |
| 4608, 6144, 16384 | A100 80GB | 3.73 | 3.65 | 3.62 | 4.47 | 2.25 |
| 4608, 16384, 6144 | A100 80GB | 3.55 | 3.61 | 3.59 | 4.28 | 2.53 |

What this says next to the M1 Max table (findings.md, 2026-10-02):

- Dequantize-per-call INT8 in fp16 costs 4 to 20 percent over fp16 on these NVIDIA parts. On the M1 Max the same fp16 dequantize-per-call costs about 9 percent (docs/review/2026-11-09-experiments/int-mm-2.15.json), so the per-call decode itself is not the Apple problem. Stock ComfyUI's 1.2 to 1.9x penalty on Apple GPUs comes mostly from computing in bf16, which the M1 Max runs 1.8x slower than fp16, plus the dequantize. The resident-decode kernel removes both: it decodes once per call into normalized fp16 and runs the fp16 GEMM.
- The transposed-weight slowdown at K=16384 is 1.53x on T4 (Turing), 1.12x on A10G (Ampere) and absent on A100. On MPS it is 3.6x in fp16 and 8.7x in fp32. It is a layout-dispatch problem that newer cuBLAS solved; MPS has not.
- bf16 is emulated on T4 (9x slower than fp16), the same shape of penalty as bf16 on the M1 Max (1.75x), and a reminder that bf16-by-default is a datacenter assumption.
- True INT8 GEMM runs 1.7 to 2.0x faster than fp16 wherever integer tensor units exist. That is the ceiling Apple M5 could bring; the weight-only path on M1-M4 cannot reach it and does not try.
- fp8 `_scaled_mm` is unsupported on all three (needs sm89 or newer).

## Files

- `modal_app.py`: the app, two functions, one entrypoint. Pinned images, explicit timeouts, nothing mounted from `vendor/`.
- `run.md`: the exact commands.
