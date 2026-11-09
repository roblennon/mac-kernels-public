# mac-kernels

Quantized open-weight image and video models, running natively on Apple GPUs, plus LoRA training on top of them.

I started this because the quantized checkpoints everyone shares were built for CUDA. On a Mac with torch 2.14 or earlier, ComfyUI either decompresses every weight on every call, which is slow, or refuses the file outright. (PyTorch 2.15.0 added a generic MPS `_int_mm`; on this M1 Max it runs 7 to 8% slower than fp16 at K=6144, see the findings for November 9.) So I wrote Metal kernels that keep the packed weights resident in GPU memory and decode them inside the same pass that feeds a normal fp16 matrix multiply. That covers weight-only INT8 with online rotation (ConvRot), W4A8, W6A8, and the GGUF block formats. Everything is measured on one Mac Studio (M1 Max, 32 GB), and the pieces that hold up go upstream.

This repo has the benchmarks, the ComfyUI nodes and loaders, the local workbenches, the trainer, and the measurements. The kernels themselves live in patched forks of ComfyUI and comfy-kitchen (see [Kernel source](#kernel-source)).

## Selected results

M1 Max, 32 GB, torch 2.14.1. GPU memory figures are driver allocation after sampling, not peaks. The full dated log is in [findings](docs/findings.md).

| Measurement | Result |
| --- | --- |
| INT8 ConvRot linear vs resident fp16 weights | Within about 2–5% of a resident fp16 GEMM, at half the memory. ComfyUI's stock path (dequantize per call, bf16 compute) is 1.2–1.9× slower than fp16 on the same shapes. |
| Krea 2 Turbo, 512×512, per step | 8.7 s on stock ComfyUI → 3.7 s with the kernel (2.35×). |
| Qwen-Image 2.1, 1024×1024, per step | 15.1 s on stock ComfyUI → 8.8 s with the kernel. GPU memory after sampling 9.1 GB in both. |
| Krea 2 Turbo, 1024×1024, per step | About 35 s on stock ComfyUI → about 15 s with the kernel and PyTorch SDPA attention. GPU memory after sampling 21.7 GB → 15.0 GB. The stock figure is a single noisy run (29 to 50 s/step, measured during a download); rerun pending. |
| MiniMax H3, 6-bit W6A8 (16 GB) | 6.5 s/step at 480×288×22. Stock ComfyUI can't run this format on Apple GPUs at all. |
| GGUF decode on Metal | Bit-exact against the `gguf` reference for twelve block types (Q4_0 through Q6_K, plus IQ4_NL and IQ4_XS since November 9). See [GGUF on Metal](docs/research/2026-10-03-gguf-on-metal.md). |
| EXL3 LLM weights on Metal (prototype) | Correct on real 27B-model matrices, but 2.4–4.4× slower than llama.cpp's GGUF kernels, measured as complete-call wall time with fp16 against f32 I/O and different model lineages on the two sides. I stopped there rather than port it. See [EXL3 prototype](docs/research/2026-10-25-exl3-prototype.md). |

Every kernel output is checked against a float32 or float64 reference. End-to-end runs are checked for NaNs and compared to stock by PSNR and SSIM where noted. Timings are single-machine measurements. Every entry in the findings log says what it was run on. Read a result as covering that setup and nothing wider. A short H3 clip at 480×288 passing doesn't mean a longer one will.

## Upstream contributions

Posted October 13, 2026. As of November 9, none of the four PRs has had a human maintainer review, and the PyTorch issue is triaged with no maintainer reply. comfy-kitchen #249 needs a rebase and now shares the mps lane with #270 and the draft #317. ComfyUI #16996-16998 are bot-approved, with CI awaiting maintainer approval.

- comfy-kitchen [#249](https://github.com/Comfy-Org/comfy-kitchen/pull/249): an MPS backend with weight-only INT8, W4A8 and W6A8 linears for Apple GPUs.
- ComfyUI [#16996](https://github.com/Comfy-Org/ComfyUI/pull/16996): prefer fp16 over bf16 for VAEs on Apple GPUs.
- ComfyUI [#16997](https://github.com/Comfy-Org/ComfyUI/pull/16997): recognize MPS allocator out-of-memory errors.
- ComfyUI [#16998](https://github.com/Comfy-Org/ComfyUI/pull/16998): support Diffusers-format WAN LoRAs.
- PyTorch [#200782](https://github.com/pytorch/pytorch/issues/200782): `F.linear` with a transposed weight is slower on MPS at large K: 8.4–8.7× in fp32 and 3.6× in fp16 at K = 32768 (torch 2.14.1 and nightly 2.16, measured October 9, 2026).

The evidence behind each one, and the bigger follow-ups waiting on a first merge, are in the [contribution docs](docs/upstream/README.md).

## Guides

| Task | Guide |
| --- | --- |
| Krea 2 images | [Krea Workbench](docs/how-to-krea-workbench.md) |
| Qwen-Image 2.1 images | [Qwen Workbench](docs/how-to-qwen-workbench.md) |
| WAN 2.2 and MiniMax H3 video | [Video Workbench](docs/how-to-video-workbench.md) |
| Headless runs and older recipes | [Running guide](docs/how-to-run.md) |
| LoRA training on quantized bases | [Training guide](docs/how-to-train.md) |
| How the stack fits together | [Contributor guide](CONTRIBUTING.md) |
| Reproducing the numbers above | [Reproducing](docs/REPRODUCING.md) |
| Research notes | [Documentation index](docs/README.md) |

## Kernel source

The MPS kernels are in the comfy-kitchen PR above and in my public forks, [roblennon/comfy-kitchen](https://github.com/roblennon/comfy-kitchen) and [roblennon/ComfyUI](https://github.com/roblennon/ComfyUI).

Fair warning: cloning this repo doesn't give you a working install. `vendor/` holds the patched ComfyUI and comfy-kitchen checkouts, and it's outside version control, along with the virtual environments and the model weights. The [contributor guide](CONTRIBUTING.md) lists the exact checkouts and versions I run.

## Layout

- `bench/`: runners, ComfyUI nodes and loaders, workflows, microbenchmarks and probes. [bench/README.md](bench/README.md) sorts them.
- `train/`: the LoRA trainer, with its own dependency lock.
- `tests/`: CPU contracts and small Apple GPU checks.
- `docs/`: guides, the findings log, research notes and upstream contribution records.
- `MK_ROOT`: where models and outputs live (default `~/mac-kernels-data`).

## Status

A personal research project, not a product. Everything here has run on exactly one machine, an M1 Max with 32 GB. I haven't qualified longer or larger video clips, other Apple chips, or the untouched official ComfyUI templates. An outside pre-grant review from November 9, 2026, with the open fixes and the experiments to run next, is in [docs/review](docs/review/2026-11-09-pre-grant-review.md).

## License

Apache-2.0, except `bench/exl3_reference.py`, which is vendored unchanged under its own MIT license ([bench/exl3_reference.LICENSE](bench/exl3_reference.LICENSE)).
