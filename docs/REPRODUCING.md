# Reproducing the numbers

Written 2026-11-09. Everything in [findings](findings.md) was measured on one M1 Max with 32 GB, torch 2.14.1, Python 3.12.11 (CONTRIBUTING.md, "Installed native snapshot"). There are two tiers. The kernel tier needs no model files. The full-model tier needs the model weights and the external storage root.

## What is public today, and what is not

The runtime is the patched checkouts under `vendor/`. A root clone does not contain them (CONTRIBUTING.md, "A root clone is not a complete runtime installation"). The pinned commits are:

| Checkout | Branch | Pinned commit | On the public fork as of 2026-11-09 |
| --- | --- | --- | --- |
| `vendor/ComfyUI` | `mps-int8-weight-only` | `e11d9226` | No. `gh api repos/roblennon/ComfyUI/commits/e11d9226...` returns HTTP 422 "No commit found for SHA". The branch is not listed either. |
| `vendor/comfy-kitchen` | `mps-weight-only-backend` | `9825e989` | No. `gh api repos/roblennon/comfy-kitchen/commits/9825e989...` returns HTTP 422 "No commit found for SHA". The branch is not listed either. |
| `vendor/ck-gguf` | `mps-gguf` | `da65b206` | No. Same HTTP 422. The branch is not listed. |
| `vendor/ComfyUI-GGUF-fork` | `mps-gguf-kernel` | `bef90ac5` | No. CONTRIBUTING.md lists this fork as "not public". |

What the public forks do carry today: `roblennon/comfy-kitchen` has `mps-backend` at `43fe176` (2026-10-13), which is the head of comfy-kitchen PR #249. `roblennon/ComfyUI` has `mps-fp16-vae`, `mps-oom-recovery` and `wan-diffusers-lora-keys`, the heads of ComfyUI PRs #16996, #16997 and #16998, plus `unipc-pr16574-mps-evidence`, which has no PR yet. None of these is the integration branch the benches and runners import from.

The pinned commits live on three private remotes, `roblennon/mac-kernels-comfyui`, `roblennon/mac-kernels-comfy-kitchen` and `roblennon/mac-kernels-comfyui-gguf`. My local checkouts have also moved past the pins: `vendor/ComfyUI` is at `0416252` and `vendor/comfy-kitchen` at `b5dd528` on 2026-11-09. `vendor/ck-gguf` and `vendor/ComfyUI-GGUF-fork` still sit at their pins.

So as of today a reader cannot run either tier from the public repositories. Publishing the three integration branches and updating the pins above is the open blocker (pre-grant review, M1). Until then this document describes what the commands do and what they should print, so the results can be checked once the branches are public.

Two more gaps. The root has no dependency lock; only the trainer has one (`train/requirements.lock.txt`, CONTRIBUTING.md "Root has no complete dependency lock"). And the `gguf` Python package (0.19.0 in my `.venv`) is needed for the GGUF bench and tests but is not pinned anywhere in the root.

## Environment assumptions

- The native `.venv` at the repo root, with comfy-kitchen path-linked to its vendor checkout (a `.pth` file in `site-packages` points at `vendor/comfy-kitchen`). The explicit `PYTHONPATH=vendor/comfy-kitchen` in the commands below makes that choice visible and overrides the link.
- `bench/with_gpu.sh` wraps `bench/gpu_runner.py`. It holds a lock so one process owns the GPU at a time. `MK_GPU_LOCK` moves the lock file; the default is `/tmp/mac-kernels-gpu.lock` (bench/gpu_runner.py:350). It runs under the system `python3`, not the venv, so `python3` must exist on `PATH`.
- `MK_ROOT` is the storage root for model files and outputs. The default is `~/mac-kernels-data` (bench/run_krea2.py, bench/run_qwen21.py, bench/run_h3.py, bench/run_wan5b.py). The kernel tier never reads it.

## Tier 1: kernels, no model files

These scripts build random weights, quantize them, and time each path against the same decoded weight. Each prints a relative error against a float32 or float64 reference beside the timing. Shapes are positional `M,K,N` arguments. Run them from the repo root.

### INT8 ConvRot linear

`bench/bench_ck01_int8.py` supersedes `bench/bench_int8_linear.py`. It runs four paths through one process: dense bf16 `F.linear`, dense fp16 `F.linear`, dequantize per call, and the mps backend (bench/bench_ck01_int8.py:54-57). Default shapes are the three in the findings tables.

```sh
PYTHONPATH=vendor/comfy-kitchen bench/with_gpu.sh .venv/bin/python bench/bench_ck01_int8.py
PYTHONPATH=vendor/comfy-kitchen bench/with_gpu.sh .venv/bin/python bench/bench_ck01_int8.py 4608,6144,6144 4608,16384,6144
```

Each line is `RESULT M,K,N | path | ms | relerr`. Median of 7 timed calls after 2 warm-ups (bench/bench_ck01_int8.py:24). What I measured after hardening, bare fp16 GEMM against the kernel, bf16 in and out (findings.md:76-80):

| Shape | Bare fp16 GEMM | mps backend |
| --- | --- | --- |
| 4608, 6144, 6144 | 37.8 ms | 39.5 ms |
| 4608, 6144, 16384 | 101.4 ms | 104.7 ms |
| 4608, 16384, 6144 | 103 ms | 107 ms |

One thing to know when reading the `dense fp16 F.linear` row at K=16384: `F.linear` with a transposed weight is slow on MPS at large K (PyTorch issue #200782, `bench/repro_mps_wide_linear.py`). The kernel does not beat a bare GEMM there. It beats `F.linear`'s slow path.

The older `bench/bench_int8_linear.py` takes the same shape arguments (bench/bench_int8_linear.py:83) and produced the first table in findings (findings.md:13-19). Keep it for the historical row; use `bench_ck01_int8.py` for new runs.

### W4A8 and W6A8

```sh
PYTHONPATH=vendor/comfy-kitchen bench/with_gpu.sh .venv/bin/python bench/bench_w4a8_linear.py
PYTHONPATH=vendor/comfy-kitchen bench/with_gpu.sh .venv/bin/python bench/bench_w4a8_linear.py 4608,16384,6144
```

Default shapes are `4096,3072,3072` and `4096,3072,12288` (bench/bench_w4a8_linear.py:115). Quantization runs eager on the CPU first and prints how long it took before the GPU timing starts (bench/bench_w4a8_linear.py:56). The hardened W4A8 result at `4608,16384,6144` was 111 ms against 218 ms before hardening (findings.md:81).

### GGUF block types

```sh
PYTHONPATH=vendor/ck-gguf bench/with_gpu.sh .venv/bin/python bench/bench_gguf_linear.py --no-stock
PYTHONPATH=vendor/ck-gguf bench/with_gpu.sh .venv/bin/python bench/bench_gguf_linear.py --no-stock --types Q4_K,Q8_0,IQ4_NL,IQ4_XS 4608,6144,6144
```

Flags (bench/bench_gguf_linear.py:138-141): `--types` is a comma list, default `Q8_0,Q6_K,Q5_K,Q4_K,Q5_0,Q4_0,Q3_K,Q2_K`; `--no-stock` skips ComfyUI-GGUF's dequantize-per-call path, which otherwise needs `COMFY_GGUF_DIR` (default `vendor/ComfyUI-GGUF-fork`); shapes are positional, default `4608,6144,6144`. Note the `PYTHONPATH` points at `vendor/ck-gguf`, the `mps-gguf` Kitchen checkout, not the active one. The GGUF decode kernel lives there. With `--no-stock` the GGUF node fork is not touched at all, so publishing `mps-gguf` is what unblocks this tier for GGUF. The error column is against a float64 CPU linear over `gguf.quants`' own dequantization (bench/bench_gguf_linear.py:9).

### Kernel tests

The kernel numerics tests live in the Kitchen checkouts, not in this repo. They need MPS and `torch.mps.compile_shader`, and skip otherwise.

```sh
PYTHONPATH=vendor/comfy-kitchen bench/with_gpu.sh .venv/bin/python -m pytest vendor/comfy-kitchen/tests/test_mps_int8_linear.py vendor/comfy-kitchen/tests/test_mps_w4a8.py vendor/comfy-kitchen/tests/test_mps_autograd.py vendor/comfy-kitchen/tests/test_mps_packed_autograd.py -q -p no:cacheprovider
PYTHONPATH=vendor/ck-gguf bench/with_gpu.sh .venv/bin/python -m pytest vendor/ck-gguf/tests/test_mps_gguf.py -q -p no:cacheprovider
```

`test_mps_gguf.py` covers twelve block types including IQ4_NL and IQ4_XS (vendor/ck-gguf/tests/test_mps_gguf.py:28). The research note records 177 tests, bit-exact against `gguf.quants` (research/2026-10-03-gguf-on-metal.md:99-100).

## Tier 2: full models

### Weights

Model files live under `$MK_ROOT/models`, in family subfolders that the loaders search (how-to-run.md, "Starting the server"). `bench/hf_prepare.py` downloads pinned Hugging Face files by Hub revision and verifies each one against the Hub's own hash, then writes `.mk-verified.json` beside the files (bench/hf_prepare.py:2-4). It is a library, not a command: `prepare(repo, revision, directory, selected, verify_only)` (bench/hf_prepare.py:11). The one command-line wrapper is `bench/prepare_wan_i2v.py`, which fetches the WAN A14B I2V fp16 experts and the Lightning I2V LoRA pair (`--only experts|lightning`, `--verify-only`; bench/prepare_wan_i2v.py:24-25). For the other families, call `prepare` from a short Python snippet with the repo and revision you want, or place the files by hand.

The files each runner expects by default:

| Runner | Default files (set `UNET`, `TE`, `VAE_NAME` to change) |
| --- | --- |
| `run_krea2.py` | `krea2_turbo_int8_convrot`, `qwen3vl_4b_bf16`, `qwen_image_vae` (how-to-run.md, "Historical headless recipes") |
| `run_qwen21.py` | `qwen_image_2.1_int8_convrot.safetensors`, `qwen3vl_8b_int8_convrot.safetensors` (bench/run_qwen21.py, module header) |
| `run_wan5b.py` | `wan2.2_ti2v_5b_turbo_int8_convrot.safetensors`, `wan2.2_vae.safetensors` (bench/run_wan5b.py:53, :77) |
| `run_h3.py` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors`, `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (bench/run_h3.py, module header) |

The INT8 ConvRot files are Comfy-Org's published quantizations where they exist, and this repo's own conversions otherwise (`bench/wan_int8_convert.py`, `bench/qwen_int8_convert.py`; CONTRIBUTING.md "Acquiring and converting weights"). I have not written down the Hub repo and revision for each file in one place yet. That list belongs here and is still to do.

### Runners

Every runner is configured by environment variables, listed in the docstring at the top of each file. `MODE=stock` runs ComfyUI as shipped, with INT8 compute disabled on MPS so weights are dequantized per call. `MODE=metal` routes the quantized layers to the comfy-kitchen mps backend. Each run is one process, under the GPU runner with a deadline.

Krea 2 Turbo, the four-way sweep behind the Krea rows in RESULTS (findings.md, 2026-11-09). It runs stock and kernel with both attention modes, each in a fresh process, in a shuffled order:

```sh
MK_ROOT=<models> .venv/bin/python bench/krea_four_way.py --out results.jsonl \
  --sizes 512,1024 --repeats 3 --cond-cache auto --capture --swap-growth-gib 8
.venv/bin/python bench/krea_four_way.py --summarize results.jsonl
```

One stock and kernel pair at 512 by hand (the October numbers, findings.md:36 and :83):

```sh
MODE=stock W=512 H=512 STEPS=8 CFG=1 SEED=42 TAG=repro PYTHONPATH=vendor/comfy-kitchen \
  .venv/bin/python bench/gpu_runner.py --monitor --seconds 900 -- .venv/bin/python bench/run_krea2.py
MODE=metal W=512 H=512 STEPS=8 CFG=1 SEED=42 TAG=repro PYTHONPATH=vendor/comfy-kitchen \
  .venv/bin/python bench/gpu_runner.py --monitor --seconds 900 -- .venv/bin/python bench/run_krea2.py
```

The current qualified 1024 recipe adds `FP16=1 ATTN=pytorch KREA_READONLY=1 COND_CACHE=off` (how-to-run.md, "Current Krea Turbo headless recipe"). Use a new `TAG` per run.

Qwen-Image 2.1 at 1024 (findings.md:95-96):

```sh
MODE=stock W=1024 H=1024 STEPS=25 PYTHONPATH=vendor/comfy-kitchen \
  .venv/bin/python bench/gpu_runner.py --monitor --seconds 1200 -- .venv/bin/python bench/run_qwen21.py
MODE=metal W=1024 H=1024 STEPS=25 PYTHONPATH=vendor/comfy-kitchen \
  .venv/bin/python bench/gpu_runner.py --monitor --seconds 1200 -- .venv/bin/python bench/run_qwen21.py
```

Wan 2.2 5B, the bounded recipe (defaults are 704x416x33, 4 steps, euler, shift 8; bench/run_wan5b.py:50-52):

```sh
MODE=metal VAE_FP16=1 PYTHONPATH=vendor/comfy-kitchen \
  .venv/bin/python bench/gpu_runner.py --monitor --seconds 900 -- .venv/bin/python bench/run_wan5b.py
```

MiniMax H3, the short clip (defaults are 480x288x22, 4 steps, Turbo LoRA on; bench/run_h3.py header):

```sh
MODE=metal PYTHONPATH=vendor/comfy-kitchen \
  .venv/bin/python bench/gpu_runner.py --monitor --seconds 1800 -- .venv/bin/python bench/run_h3.py
```

The first H3 run encodes text for about three minutes and caches it by prompt (findings.md:57). Set `UNET` to the W6A8 or W4A8 file to reproduce those rows; the tag carries the format.

GGUF end to end: pass a `.gguf` file as `UNET` to `run_qwen21.py` or `run_wan5b.py`. The runner loads it through `bench/gguf_node.py`, and `MODE=stock` there means ComfyUI-GGUF's own dequantize-per-call path (bench/run_qwen21.py header). This needs `vendor/ComfyUI-GGUF-fork` registered as a custom node (`bench/setup_custom_nodes.py`) and `PYTHONPATH=vendor/ck-gguf`.

### Training backward

`bench/bench_int8_backward.py` times the frozen-weight backward on real layers of the Krea 2 Raw INT8 checkpoint, so it belongs in this tier. It reads `$MK_ROOT/models/diffusion_models/krea2_raw_int8_convrot.safetensors` (bench/bench_int8_backward.py:19-20), takes no arguments, and appends a JSON line to `$MK_ROOT/out/results.jsonl`. It runs in the trainer's environment, built by `train/setup.sh` from `train/requirements.lock.txt`:

```sh
bench/with_gpu.sh train/.venv/bin/python bench/bench_int8_backward.py
```

The rows it reproduces are findings.md:383-388.

### What a result line should record

Each run prints a `RESULT` line. Keep it with the output: source commit, assets, dimensions, schedule, elapsed time, peak memory and what the image or clip looked like (CONTRIBUTING.md, "Verification by boundary"). Memory figures in the Krea table are `driver_allocated_memory` snapshots taken after sampling, not peaks (bench/run_krea2.py:164). The H3 runner polls for peaks per phase (bench/run_h3.py header). Say which one you report.

## Repo tests

`.repo-check` is the pre-push pair. Both lines run under `bench/with_gpu.sh`, as CONTRIBUTING.md ("Run the tests") asks:

```sh
bench/with_gpu.sh .venv/bin/python -m pytest tests -q -p no:cacheprovider --ignore=tests/test_gguf_node.py
PYTHONPATH=vendor/ck-gguf bench/with_gpu.sh .venv/bin/python -m pytest tests/test_gguf_node.py -q -p no:cacheprovider
```

The lock keeps them from overlapping a render or a training run. The GGUF node tests run separately because they need the `mps-gguf` Kitchen checkout on the path instead of the active one. Without `vendor/ComfyUI`, `tests/conftest.py` skips the files that import the patched runtime and says so in the pytest header. A vendor-less run is a smoke check, not a pass.
