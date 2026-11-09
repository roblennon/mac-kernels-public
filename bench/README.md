# bench/

Everything here runs against the patched ComfyUI and comfy-kitchen checkouts in `vendor/` (see the [contributor guide](../CONTRIBUTING.md)). GPU work goes through `with_gpu.sh` or `gpu_runner.py`, so only one job owns the GPU at a time.

The folder is flat, so this index sorts it. Roughly a third of the files are used by the workbenches and runners. The rest are microbenchmarks and one-off probes kept as the evidence behind [findings](../docs/findings.md). Those probes ran once against a specific setup. They aren't maintained, and some expect model files or checkout states you won't have.

## Start and serve

| File | What it does |
| --- | --- |
| `serve.sh` | Starts the patched ComfyUI for normal browser use. `extra_model_paths.yaml` points it at `$MK_ROOT/models`. |
| `with_gpu.sh`, `gpu_runner.py` | Serializes GPU jobs and owns the process group, with optional deadline and memory limits. |
| `start_krea_workbench.py`, `krea_workbench.py`, `krea_server.py` | The dedicated Krea Workbench and its opt-in localhost API. |
| `setup_custom_nodes.py`, `video_nodes/` | Registers the GGUF node and the local video and Qwen nodes with ComfyUI. |
| `queue.sh`, `template_session.sh` | Run jobs back to back, or one server session over the official templates. |
| `web/` | Pinned frontend default graph and the Krea export UI script. |

## Model loading and memory (used by the workbenches)

| File | What it does |
| --- | --- |
| `krea_readonly.py`, `krea_adapter.py`, `krea_checkpoints.py`, `krea_conditioning.py`, `krea_reference.py`, `krea_reference_worker.py`, `krea_sizes.py` | Krea: read-only checkpoint mapping, LoRA admission, conditioning reuse, reference images, preset sizes. |
| `krea_export.py`, `krea_export_jobs.py` | CPU-only Krea Turbo merge and export into the INT8 layout, one job at a time. |
| `qwen_memory.py`, `qwen_comfy.py` | Qwen: mapped INT8 construction under a 4 GiB cap, and text or edit preparation. |
| `wan_memory.py`, `wan_latents.py`, `wan_latent_nodes.py` | WAN 14B bounded loading, and saved-latent Draft/Final decoding. |
| `h3_memory.py` | MiniMax H3 mapped construction and bounded sampling. |
| `video_loras.py` | Model-only video LoRA stacks through Comfy's own patching. |
| `gguf_node.py` | Loads a GGUF diffusion model through ComfyUI-GGUF for the headless runners. |
| `conditioning_cache.py` | Source-aware CPU text-conditioning cache for the headless runners. |
| `spectral_krea.py` | Opt-in Krea spectral sampler experiment, using the pinned LC123 helper. |

## Workflows

| File | What it does |
| --- | --- |
| `krea_workflows.py`, `qwen_workflows.py`, `video_workflows.py`, `workflow_canvas.py` | Generators for the editable presets. Change these, then regenerate `workflows/`. Don't hand-edit the JSON. |
| `workflows/` | Generated editable and API graphs for each preset. |
| `templates_api/` | The official ComfyUI templates in API form, used as baselines by `run_templates.py`. |

## Headless runners

`run_krea2.py`, `run_qwen21.py`, `run_wan5b.py`, `run_wan14b.py`, `run_h3.py` and `run_templates.py` run each model through ComfyUI nodes without the browser, for timing. The docstring at the top of each lists its environment switches.

## Getting and converting weights

| File | What it does |
| --- | --- |
| `hf_prepare.py`, `prepare_wan_i2v.py` | Download pinned Hugging Face files and verify them against the Hub's hashes. |
| `wan_int8_convert.py`, `qwen_int8_convert.py` | Convert fp16/bf16 checkpoints into the INT8 ConvRot layout. |
| `gguf_headers.py`, `lora_header_report.py` | Read GGUF quant types or a LoRA's structure from the header alone. |

## Microbenchmarks

Kernel-level timing and accuracy, mostly against fp16 or a CPU reference: `bench_int8_linear.py`, `bench_ck01_int8.py`, `bench_w4a8_linear.py`, `bench_block_formats.py`, `bench_gguf_linear.py`, `bench_int8_backward.py`, `bench_attn_backward.py`, `bench_lora_sidepath_kernel.py`, `bench_lora_fused_rank.py`, `bench_lora_types_sidepath.py`, `sdpa_mps.py`, `sdpa_mps_upcast.py`, `attn_study.py`, `capture_attn.py`, `conv3d_mps.py`, `patch_embed_mps.py`, `rmsnorm_mps.py`, `unipc_coeffs.py`, `vae_dtype_parity.py`, `te_encode.py`.

`repro_mps_wide_linear.py` is the standalone repro for PyTorch issue #200782.

## Historical probes (not maintained)

One-off experiments, each tied to a dated entry in findings:

- **Loading and memory:** `probe_model_load.py`, `probe_text_encoder_load.py`, `probe_server_session.py`, `probe_krea_server.py`, `probe_krea_server_lora.py`, `load_probe.py`, `review_load_path.py`, `review_layer_load.py`, `review_chunk_budget.py`.
- **Kernels and formats:** `probe_small_m.py` with `small_m_int8.metal`, `review_kernel_cost.py`, `review_generation_contracts.py`, and the EXL3 prototype: `exl3_probe.py`, `exl3_probe.metal`, `exl3_reference.py` (MIT, vendored), `gguf_op_probe.cpp` and `gguf_op_probe.md`.
- **Training:** `probe_ai_toolkit.py`, `probe_ai_toolkit_gradient.py`, `probe_packed_training.py`, `probe_qwen_adapter_types.py`.
- **Video decode and comparisons:** `decode_wan.py`, `decode_h3.py`, `profile_wan_vae.py`, `wan_vae_chunks.py`, `qwen_accel_compare.py`, `img_similarity.py`.
- **LoRA audit:** `lora_audit.py`, `lora_audit_wrap.py`, `attn_fp16_wrap.py`.
- **October 11 overnight runs:** `overnight_perf02a_attn.py`, `overnight_qwen_multiref.py`.
