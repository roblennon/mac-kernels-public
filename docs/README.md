# Documentation index

## Evidence and research

- [Results](RESULTS.md): the curated numbers, each with the findings line it comes from.
- [Reproducing](REPRODUCING.md): the kernel tier and the full-model tier, with exact commands and what is not yet public.
- [Findings](findings.md): the dated measurement log.
- [Krea 2 blocks and variety](krea2-blocks-and-variety.md): same-face ablations, checkpoint and LoRA controls, and scoring limits.
- [Inference acceleration survey for M1–M4](research/2026-10-02-inference-acceleration-survey.md)
- [LoRA training on an INT8 ConvRot base](research/2026-10-02-lora-training-on-quantized-bases.md)
- [GGUF diffusion models on Apple GPUs](research/2026-10-03-gguf-on-metal.md)
- [Format checkpoint availability](research/2026-10-04-format-checkpoint-availability.md)
- [EXL3 LLM weights on Metal, prototype](research/2026-10-25-exl3-prototype.md): correct, 2.4–4.4× slower than llama.cpp's GGUF kernels, stopped there.
- [Research statement](research-statement.md): research question, contributions, related work, limitations and proposed aims.
- [Pre-grant review, November 9](review/2026-11-09-pre-grant-review.md): what holds up, what must be fixed, and where this sits against other work. Raw outputs of the two follow-up experiments are in [review/2026-11-09-experiments](review/2026-11-09-experiments/).
- [Review-process receipts, October 5](review/2026-10-05-review-process-receipts.md): how the October 5 source reviews were run, moved out of the findings log.
- [Modal cloud anchor](../modal/README.md): the same GEMM shapes on NVIDIA T4, A10G and A100, for a cross-platform column.

## Guides

- [Contributor guide](../CONTRIBUTING.md): how the pieces fit, runtime reconstruction, shared contracts and verification.
- [Krea Workbench](how-to-krea-workbench.md): checkpoint selection and export, sizes, LoRA stacks, Spectral and references.
- [Qwen Workbench](how-to-qwen-workbench.md): editable native INT8 preset and optional model LoRA slots.
- [Video Workbench](how-to-video-workbench.md): WAN 2.2 5B and A14B, MiniMax H3, high/low adapters, saved-latent Draft/Final, and measured time and quality.
- [Running guide](how-to-run.md): scoped API profiles and headless recipes.
- [Local training](how-to-train.md): datasets and configuration, guarded runs, recovery and evaluation.

## Upstream contributions

- [Contribution review, October 8](upstream/contribution-review-2026-10-08.md): every upstream change this work could feed, and which ones someone upstream is already working on.
- [Branch and draft inventory](upstream/README.md), and the [posted PR and issue bodies](upstream/ready/README.md).
