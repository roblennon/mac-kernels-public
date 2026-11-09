# mac-kernels

Quantized open-weight image and video models, running natively on Apple GPUs, plus LoRA training on top of them.

I started this because the quantized checkpoints everyone shares were built for CUDA. On a Mac, ComfyUI either decompresses every weight on every call, which is slow, or refuses the file outright. So I wrote Metal kernels that keep the packed weights resident in GPU memory and decode them inside the same pass that feeds a normal fp16 matrix multiply. That covers weight-only INT8 with online rotation (ConvRot), W4A8, W6A8, and the GGUF block formats. Everything is measured on one Mac Studio (M1 Max, 32 GB), and the pieces that hold up go upstream.

This repo has the benchmarks, the ComfyUI nodes and loaders, the local workbenches, the trainer, and the measurements. The kernels themselves live in patched forks of ComfyUI and comfy-kitchen (see [Kernel source](#kernel-source)).

## Selected results

M1 Max, 32 GB, torch 2.14.1. The full dated log is in [findings](docs/findings.md).

| Measurement | Result |
| --- | --- |
| INT8 ConvRot linear vs resident fp16 weights | 4–9% slower, at half the memory. ComfyUI's stock dequantize-per-call path is 1.3–2.0× slower than fp16 on the same shapes. |
| Krea 2 Turbo, 1024×1024, per step | 35 s on stock ComfyUI → 16.5 s with the kernel and PyTorch SDPA attention. GPU memory 21.7 GB → 15.0 GB. |
| MiniMax H3, 6-bit W6A8 (16 GB) | 6.5 s/step at 480×288×22. Stock ComfyUI can't run this format on Apple GPUs at all. |
| GGUF decode on Metal | Bit-exact against the `gguf` reference for ten block types (Q4_0 through Q6_K). See [GGUF on Metal](docs/research/2026-10-03-gguf-on-metal.md). |
| EXL3 LLM weights on Metal (prototype) | Correct on real 27B-model matrices, but 2.4–4.4× slower than llama.cpp's GGUF kernels. I stopped there rather than port it. |

Every number is checked against an independent float32 or float64 reference, and every entry in the findings log says what it was run on. Read a result as covering that setup and nothing wider. A short H3 clip at 480×288 passing doesn't mean a longer one will.

## Upstream contributions

Posted in October 2026, all still in review:

- comfy-kitchen [#249](https://github.com/Comfy-Org/comfy-kitchen/pull/249): an MPS backend with weight-only INT8, W4A8 and W6A8 linears for Apple GPUs.
- ComfyUI [#16996](https://github.com/Comfy-Org/ComfyUI/pull/16996): prefer fp16 over bf16 for VAEs on Apple GPUs.
- ComfyUI [#16997](https://github.com/Comfy-Org/ComfyUI/pull/16997): recognize MPS allocator out-of-memory errors.
- ComfyUI [#16998](https://github.com/Comfy-Org/ComfyUI/pull/16998): support Diffusers-format WAN LoRAs.
- PyTorch [#200782](https://github.com/pytorch/pytorch/issues/200782): `F.linear` with a transposed weight is up to 8.7× slower on MPS at large K.

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

A personal research project, not a product. Everything here has run on exactly one machine, an M1 Max with 32 GB. I haven't qualified longer or larger video clips, other Apple chips, or the untouched official ComfyUI templates.

## License

Apache-2.0, except `bench/exl3_reference.py`, which is vendored unchanged under its own MIT license ([bench/exl3_reference.LICENSE](bench/exl3_reference.LICENSE)).
