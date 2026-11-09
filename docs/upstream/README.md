# Upstream submissions: Apple GPU support for Comfy

**October 13, 2026:** six items were finalized in [ready/](ready/README.md), pushed to the public forks and posted: ComfyUI #16996, #16997 and #16998, a comment on ComfyUI PR #16574, comfy-kitchen #249 and PyTorch issue #200782.

**Current organization:** [October 8 contribution review](contribution-review-2026-10-08.md) covers all 29 evidenced candidates, later fixes, live overlap and five new ComfyUI branch conflicts. The inventory below remains a historical snapshot.

**October 8 disposition:** This remains the October 4 private submission-preparation snapshot. Later root/runtime fixes and Workbenches are not automatically present in these separated branches, so a future submission needs a fresh audit/rebase/review. Local short WAN14/H3 acceptance now passes; that does not qualify generic mapped-offload or refresh the submission series. Public submissions/CLA remain deferred.

This is the preserved October 4 preparation inventory, not a current publication instruction.

The patch series are prepared for private review. **No public PR or CLA acceptance has been made by this refresh.** All 15 prepared ComfyUI branches and four Kitchen branches are pushed to the private repositories at the verified heads listed below (October 4 UTC).

The preparation checkouts are `vendor/comfy-kitchen-upstream` and `vendor/ComfyUI-upstream`. Branches share Git history with the primary vendor checkouts, but remain separate from their integrated working branches. A prepared branch is not proof that the complete working tree is ready for upstream submission.

## Branch inventory

Heads below are the 2026-10-04 UTC refresh. “Base” names the preceding review branch, not an instruction to merge the entire working branch. Every new implementation commit was cherry-picked with its source hash and DCO sign-off; existing published history was preserved.

| Repo | Branch | Head | Base | PR text |
|---|---|---|---|---|
| Kitchen | `upstream/mps-backend` | `b82589b` | `5eeefcf` | [Backend](comfy-kitchen-mps-backend.md) |
| Kitchen | `upstream/mps-training` | `d402268` | backend | [Training](comfy-kitchen-mps-followups.md#training) |
| Kitchen | `upstream/mps-block-formats` | `354d0f5` | backend | [NVFP4, MXFP8, W4A4](comfy-kitchen-mps-followups.md#block-formats) |
| Kitchen | `upstream/mps-gguf` | `9d91f8f` | backend | [GGUF](comfy-kitchen-mps-followups.md#gguf) |
| ComfyUI | `upstream/mps-vae-temporal-padding` | `0252e040` | `65787d66` | [Temporal VAE padding](comfyui-mps-vae-temporal-padding.md) |
| ComfyUI | `upstream/mps-fp16-vae` | `d3bdb465` | `65787d66` | [fp16 VAE](comfyui-mps-fp16-vae.md) |
| ComfyUI | `upstream/mps-pytorch-attention` | `1121361c` | `65787d66` | [Attention](comfyui-mps-pytorch-attention.md) |
| ComfyUI | `upstream/mps-unipc-float64` | `d9602ea5` | `65787d66` | [UniPC](comfyui-mps-unipc-float64.md) |
| ComfyUI | `upstream/mps-conv3d` | `cc9fc6c7` | `65787d66` | [Conv3d](comfyui-mps-conv3d.md) |
| ComfyUI | `upstream/mps-kitchen-formats` | `f9d89268` | `65787d66` | [INT8/W4/W6 dispatch](comfyui-mps-kitchen-formats.md) |
| ComfyUI | `upstream/mps-lora-sidepath` | `fedb008b` | kitchen-formats | [LoRA side path](comfyui-mps-lora-sidepath.md) |
| ComfyUI | `upstream/mps-lora-accounting` | `98956074` | lora-sidepath | [Clone accounting](comfyui-mps-followups.md#clone-accounting) |
| ComfyUI | `upstream/mps-lora-types` | `d8674953` | lora-accounting | [Adapter types](comfyui-mps-followups.md#adapter-types) |
| ComfyUI | `upstream/mps-block-formats` | `ca51eaed` | lora-sidepath | [Formats and text encoders](comfyui-mps-followups.md#formats-and-text-encoders) |
| ComfyUI | `upstream/mps-direct-load` | `1f77853d` | `65787d66` | [Direct loading](comfyui-mps-followups.md#direct-loading) |
| ComfyUI | `upstream/mps-oom-recovery` | `e07e1dcc` | `65787d66` | [VAE recovery](comfyui-mps-followups.md#vae-recovery) |
| ComfyUI | `upstream/mps-encoder-ownership` | `a135aafc` | `65787d66` | [Encoder ownership](comfyui-mps-followups.md#encoder-ownership) |
| ComfyUI | `upstream/mps-cache-ownership` | `274356f1` | `65787d66` | [Encoder cache ownership](comfyui-mps-followups.md#encoder-cache-ownership) |
| ComfyUI | `upstream/mps-mapped-offload` | `b0d3ac5b` | lora-types + direct-load | [Mapped offload candidate](comfyui-mps-followups.md#mapped-offload-candidate) |

The [PyTorch wide-linear issue](pytorch-mps-wide-linear-issue.md) is a separate artifact; no issue is filed by this work.

## Dependencies and submission order

The temporal-padding, fp16 VAE, attention, UniPC, conv3d, direct-load, OOM-recovery, encoder-ownership and cache-ownership changes are independent ComfyUI branches. Cache-ownership uses existing Kitchen storage introspection and does not require the new MPS backend or encoder-loading changes. Kitchen backend is the foundation for Kitchen training, block formats and GGUF. ComfyUI kitchen-formats needs a Kitchen release with the backend; sidepath then needs its LoRA kernel arguments. Clone-accounting follows sidepath, and adapter-types follows accounting. ComfyUI block-formats is based on sidepath to preserve the tested operations code, and also requires Kitchen block-formats. It does not depend on adapter-types. Mapped-offload is based on adapter-types and explicitly merges the direct-load dependency; it defaults off pending generation-memory acceptance and remains available only through explicit `model_options["mmap_offload"] = True`.

For public submission, start with the independent ComfyUI fixes and Kitchen backend. Submit the dependent branches against their stated bases, then retarget after the dependency lands. A released Kitchen version and the matching ComfyUI requirement bump are still needed; no version number is invented here. Review existing UniPC #16574 and attention #14838 before opening overlapping work.

All refreshed branch tips individually merge without conflicts onto the fetched upstream heads, Kitchen `be003b7` and ComfyUI `f1072eb0`. These are merge-tree checks, not tests of rebased code. ComfyUI adapter-types and block-formats also merge together cleanly. **Kitchen GGUF and block-formats conflict in the MPS registration list** because both add operations at the same location. Keep the two lists when integrating, as the primary working branch already does, and rerun dispatch tests. Do not advertise the combined stack as conflict-free.

## Publication and contributor checks

After validation, all 15 prepared ComfyUI branches and four prepared Kitchen branches were pushed to private repositories. The temporal-padding branch is verified at `0252e040`, and the integrated ComfyUI working branch is now verified at `6ad852be`, with the H3 FP32 force-cast follow-up prepared and privately published at `ca51eaed`.

Before a public submission, confirm the destination forks, current upstream changes, focused checks and PR descriptions. DCO sign-offs and original `Agent-Provenance` trailers remain in the series. ComfyUI's `check-ai-co-authors` rule is respected: its prepared branch commits have no AI `Co-authored-by` trailers. Kitchen's GGUF source commits retain their existing co-author trailers; inspect the destination's current contributor rules before public submission.

## Verification

[The refresh record](refresh-verification.md) distinguishes current CPU checks, the completed eight-row branch GPU matrix and historical results. The listed dependency-pair GPU checks ran separately; this packaging pass used CPU only. Mapped-offload generation-memory remains unqualified. Separate integrated production-server sessions pass for Wan 5B and Qwen at 512×512. Qwen uses explicit native direct INT8 loading, fresh CPU encoding and the paired-watermark/cache profile; all three requests re-encode and sample, with no resident warm-model speedup claimed. These results do not validate mapped-offload, the separated branches in isolation or other models’ server workflows. The complete H3 encoder also passes at bounded diagnostic scope; H3 API/video acceptance remains open.

Historical 2026-10-03 results for the original series:

- Kitchen's first five backend commits: 754, 795, 841, 918 and 1028 passing tests; 3107 skipped on other backends. Ruff passed at those commits.
- ComfyUI branch totals: fp16 VAE 197, attention 232, UniPC 205, conv3d 230, formats 194, sidepath 227; each with 14 skipped. The base had 190 passing tests. These totals predate the appended fixes and adapter-types follow-up.

## Historical measurements for the PR cover and a post

On a 32 GB M1 Max, Comfy's quantized diffusion models now run natively on the Apple GPU.
- **Before:** INT8 checkpoints were dequantized on every call, and 4- and 6-bit checkpoints could not run at all.
- **Backend:** a new comfy-kitchen backend keeps the integer weights resident and decodes them to fp16 inside Metal kernels. It is weight-only, because pre-M5 Apple GPUs have no fast integer matmul, and range-safe by construction through exact power-of-two scaling.
- **Speed:** Krea 2 Turbo runs at 3.7 s per step at 512 (8.7 before) and Qwen-Image 2.1 at 8.8 s at 1024 (15.1). MiniMax H3 INT8 drops from a swapping 38 to 74 s per step to 8.4 to 11.8 s, and its 6-bit file, which could not run before, takes 6.5 s per step.
- **LoRAs:** they stay beside the quantized layers instead of being requantized into them. That cuts H3's LoRA load from about 6.5 minutes to about half a minute and keeps the LoRA exact.
- **ComfyUI changes:** fp16 VAEs instead of emulated bf16 (Wan 2.2 decode 972 s to 74 s, then 21 s with conv3d run as conv2d); PyTorch's flash attention by default; a uni_pc precision fix that makes Wan 2.2 usable with that sampler on Macs.
- **Result:** Krea 2, Qwen-Image 2.1, Wan 2.2 (5B and 14B) and MiniMax H3 all generate on the machine.
