# Krea Workbench

The workbench uses ComfyUI's local browser interface and queue API, backed by this project's native Metal Krea runtime. It reuses the existing INT8 Turbo model, Qwen3-VL encoder and Qwen image VAE. It is a dedicated, bounded profile rather than an unrestricted Comfy server. Start it with its own launcher; these custom nodes are registered there and are not supplied by a stock Comfy installation.

## Start and generate

Double-click `Krea Workbench.command` in the repository. The launcher checks the installed assets and pinned edit runtime, starts the memory guard, and prints the local browser address. Open that address. Fresh sessions load the supported Krea text preset automatically. Saved, edited workflows are preserved; to switch presets, drag one of the JSON files from `bench/workflows/` onto the Comfy canvas:

| Workflow | Use |
| --- | --- |
| `krea_text.json` | Text generation, with up to five distinct LoRAs |
| `krea_identity_edit.json` | Edit a supplied person/subject using grounded image conditioning and reference latents |
| `krea_identity_edit_two_refs.json` | Scene first, subject second, with both image inputs connected |
| `krea_style_reference.json` | Use an image's visual style and composition as guidance |

In the Krea Workbench node, enter a prompt and select each LoRA and its independent strength. Empty slots are ignored. Start with modest strengths when combining styles; a technically valid stack can still produce a poor image. Duplicate assets, unsupported adapter tensors and zero/nonfinite strengths are rejected. Combined adapter files are limited to 2 GiB.

Choose `checkpoint` below the size control to switch between the original Turbo INT8 model and installed custom exports. Older saved workflows and API requests keep the original checkpoint. Switching unloads the previous model before loading the new one; all size, Spectral, ordinary LoRA and reference controls still apply. A LoRA already baked into the selected checkpoint cannot also be selected in a runtime slot, even under a different filename.

The menu snapshots disk-installed files at startup and adds completed GUI exports immediately. Restart the Workbench after installing an export outside the GUI. It admits compatible Turbo INT8 exports produced by `bench/krea_export.py`, using their recorded provenance, native tensor geometry and ConvRot metadata. Other INT8 formats, RAW models and unrelated architectures are not interchangeable with this profile. Invalid installed files are named in the launch log with their admission error.

Choose `size` at the bottom of the Krea Workbench node. New presets default to **Portrait 2:3 · 832 × 1248**. The menu contains these 1K presets:

| Aspect | Width × height |
| --- | --- |
| 2:3 | **832 × 1248** (default) |
| 3:2 | 1248 × 832 |
| 1:1 | 1024 × 1024 |
| 3:4 | 896 × 1184 |
| 4:3 | 1184 × 896 |
| 4:5 | 928 × 1152 |
| 9:16 | 768 × 1376 |
| 16:9 | 1376 × 768 |
| 2.35:1 | 1568 × 672 |

The [Krea-specific selector author's table](https://github.com/ANe5s/ComfyUI-Krea2-Harness#krea2-resolution-selector), checked October 7, 2026, supplies the eight published anchors; 3:4 mirrors its 4:3 dimensions. These match the revised screenshot, including exact 2:3 at 832 × 1248. Other ratio labels describe nearby aligned dimensions rather than exact mathematical ratios. [Krea's official inference instructions](https://github.com/krea-ai/krea-2#options) support other aligned dimensions and do not prescribe a mandatory preset table. This Workbench exposes the bounded catalog above plus legacy 512 × 512 square.

Saved older Workbench graphs retain 512 × 512 when their size field is missing. To make an old draft portrait, change its size explicitly. New workflows select portrait automatically. Changing output size does not require switching between the text, style or identity workflows; reference preparation follows the selected output geometry.

For the `spectral` control directly above `size`, **false = off; true = on**. For a comparison, keep the prompt, size, LoRAs, references and seed identical, leave the sampler's “control after generate” set to `fixed`, and render once in each state.

The `spectral` toggle keeps eight Euler steps but runs the first four at half spatial size, then expands the latent for the remaining four. It combines with the native INT8 model and the selected LoRAs. Leave it off for the ordinary full-size path; existing API graphs that omit the toggle also remain ordinary. Spectral can change composition, faces and likeness, so use ordinary sampling when a reference match matters more than speed. The four-step accelerator is parked and is not part of this Workbench recipe; do not add it to a stack expecting a qualified acceleration combination.

For reference modes, choose or upload an image in Load Image. The two-reference Identity Edit preset has a second Load Image connected to `ref_image`: use the scene as `source_image`, then the subject as `ref_image`. `ref_boost` emphasizes the last reference and `grounding_size` controls Identity Edit's vision preparation. The two-image preset requires 384px grounding to stay within this workstation’s memory budget; single-image editing supports 384 or 512. These two controls do not tune the style-reference mode. The mode-specific adapter is automatically included at strength 1 and occupies one slot, leaving four additional LoRA slots.

If an older session opens a generic Comfy graph, load `krea_text.json` before running. The Workbench rejects unsupported graphs with a message naming the supported presets.

Click Comfy's Run button. Output is saved below the profile's `out/comfyui/krea-workbench/` directory and displayed in Save Image. The supported recipe is one image at a listed preset size, eight Euler/simple steps, CFG 1, full denoise. Arbitrary samplers, resolutions, nodes and graphs are deliberately outside this qualified profile.

The portrait qualification took about 350–355 seconds for Identity Edit, close to the 360-second request budget. Those measured recipes use a short prompt and one/two reference images; longer prompts, extra adapters or system load may hit the limit. 384px grounding is available for single-image edits as well as the required two-image setting. Text-only and Style Reference were faster in this session.

Stop the server with Control-C in its launcher window. The shared memory policy allows recorded warning pressure and stops on critical or unknown pressure. It limits group RSS to 18 GiB and peak systemwide swap growth to 4 GiB, preserving a 4 GiB internal swap-volume reserve plus the remaining allowance (8 GiB free at startup). Requests check a 360-second budget at preparation, sampling and decode boundaries; a blocking stage can exceed that budget until its next check. The session supervisor enforces a six-hour aggregate limit. A guard failure requires investigating its diagnostic log before restarting the same workload.

## What the references mean

Identity Edit reuses the [author's Krea2Edit implementation](https://github.com/lbouaraba/comfyui-krea2edit) and [trained adapter](https://huggingface.co/conradlocke/krea2-identity-edit). It combines Qwen image grounding with clean VAE reference tokens and the author's text/reference/target token ordering. It is not ordinary img2img, and likeness is not guaranteed. The runtime is pinned to July 29, 2026 revision `86f886dac23013d88996e3a2e99093ba44d322fb`, with its source digest checked before loading.

Style Reference follows the [official Krea 2 Comfy workflow](https://docs.comfy.org/tutorials/image/krea/krea-2) and [style-reference adapter](https://huggingface.co/ostris/krea2_turbo_style_reference). It uses native `index_timestep_zero` reference latents. Expect visual guidance rather than exact character reproduction. The [template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_krea2_turbo_int8_image_style_reference.json) was checked on October 6, 2026; its latest inspected revision was September 28. This bounded profile limits uploaded images to eight megapixels/16 MiB, resizes them to a maximum 512px side, and encodes reference latents in an owned CPU worker that exits before loading the vision encoder. It does not run the template's larger reference resolutions.

With spectral enabled, Identity Edit prepares both reference sizes in that worker before sampling. Style Reference keeps its original ordered reference tokens through both target sizes. Neither mode performs VAE encoding inside the sampler. These are separate reference modes: a single request does not combine Identity Edit's wrapper with native Style Reference conditioning.

INT8 reduces the cost of weight operations; spectral reduces spatial work in the early steps. These speedups compose, but their measured percentages cannot be added. LoRA side patches, reference tokens and fresh image/text encoding add work of their own, so whole-request savings can be smaller than sampling savings.

## API reuse

The corresponding `*_api.json` files are API graphs for the same local Comfy `/prompt` queue and `/history` outputs. API and GUI requests pass the same graph, asset and recipe checks. The dedicated server binds localhost; it is not a hosted or public API.

The model workflows themselves are not Mac-only; the native INT8 execution and memory qualification here are specific to the tested Mac setup.

## Export a custom checkpoint

Set the ordinary LoRAs and strengths in the Workbench, right-click its node, and choose **Export Turbo INT8 checkpoint…**. Choose a compatible source, give the new checkpoint a filename, confirm its Turbo lineage, and click Export. The dialog captures the stack when opened. It merges up to five ordinary Krea model LoRAs, then writes the same group-256 INT8 ConvRot layout used by the native runtime.

Generation pauses while a CPU export runs. Progress and cancellation stay in the dialog; closing it leaves the job running, and reopening shows its status. Successful exports install a readonly checkpoint and refresh the choices below Size without restarting. When the current node still matches the captured recipe, the new checkpoint is selected and its baked LoRAs are cleared from the runtime stack. If you changed the node meanwhile, those changes stay intact. Open a fresh dialog for another recipe.

Sources are discovered in the launch profile, the external model folder, and `sources/krea-turbo/diffusion_models` under the artifact root. Floating files require compatible single-file Comfy Krea geometry and your explicit confirmation that they are already Turbo. `MK_ROOT` selects the artifact root; `KREA_EXPORT_DIRECTORY` optionally selects a staging/output folder distinct from the profile checkpoint folder. The default installed setup derives external storage from its output symlink. Large files stage externally and copy internally only when sufficient disk space is available. A completed external file is retained if cancellation or installation failure occurs after publication; the dialog reports that recovery state. An interrupted unpublished export leaves no selectable partial checkpoint. Identity Edit, Style Reference and the parked four-step accelerator cannot be baked by this exporter; reference modes continue to add their qualified workflow adapter.

The script remains available for reproducible recipes. From this repository:

```sh
.venv/bin/python bench/krea_export.py \
  --source /path/to/krea2_turbo_bf16.safetensors \
  --source-variant turbo \
  --lora /path/to/character.safetensors 0.7 \
  --lora /path/to/style.safetensors 0.4 \
  --output /path/to/custom_turbo_int8.safetensors \
  --cpu-threads 4
```

Full-precision input must use the compatible single-file Comfy Krea tensor names; Diffusers directories and alternative quantized formats need separate conversion. `--source-variant turbo` is an explicit statement of the source's lineage: RAW and Turbo have the same tensor shapes, so shape checks cannot establish distillation. This exporter does not turn RAW into Turbo. The installed original Turbo INT8 checkpoint supplies the pinned storage template; `--template` can point to that same release on another installation.

The source can also be our existing Turbo INT8 checkpoint. Unchanged tensors retain their exact bytes; affected linears are dequantized, merged and requantized. This adds quantization error, so prefer original floating Turbo weights and merge all intended adapters in one export. Re-exporting a custom file preserves its baked-adapter history and rejects duplicate content. Recorded absolute input paths are provenance, not paths required to load the finished checkpoint on another machine.

Export uses CPU buffers bounded by matrix rows, validates inputs and refuses existing output paths. It stages changed payloads plus the complete candidate beside the output, so allow roughly twice the final file size in temporary disk space. Controlled cancellation removes unpublished staging files; an abrupt process kill may leave a `.krea-export-*` staging directory. Only a fully written, validated candidate is published. The original model and adapters are preserved.

Pause generation during full exports. Large source/staging files belong on the external drive. For script-created exports, install the finished file in the launch profile's `models/diffusion_models/` directory and restart. GUI exports perform installation and admission automatically. This workstation keeps the selected checkpoints on internal storage because external model paging previously stalled sampling. The exporter returns a JSON receipt with output digest and source/adapter provenance; save it alongside the recipe when reproducing an export.

## Installation and evidence

The launcher uses the repository's locked `.venv`, `vendor/comfy-kitchen`, the selected model profile, the pinned external edit runtime and the pinned MIT-licensed `vendor/LC123-spectral` helper. Both author runtimes are checked before use. `KREA_EDIT_RUNTIME` can specify the runtime directory; otherwise the launcher discovers it beside the profile's external model storage. Model downloads and generated proof artifacts stay outside the Git repository. This is an installed workstation workflow, not a bootstrap installer for another machine.

The qualification run records real three/five-adapter API renders, a five-job browser sequence, and a final browser recheck with exact repeated edits and restored-base pixels. The final recheck peaks at 8.393 GiB with no swap growth. Outputs were inspected, the configured launcher was exercised, regression checks passed, and independent review has no remaining findings. This evidence covers representative recipes; new adapter combinations still need judgment about strengths and image quality.

The Spectral composition qualification adds 17 real GUI requests: three LoRAs, one/two reference images, five adapter files, exact repeats and restored-base pixels. Mean sampling was 16.3% lower for the three-LoRA bracket and 26.4% lower for two-image Identity Edit. Style Reference was 4.2% slower with Spectral, so leave Spectral off for that mode unless experimenting. These are conditional measurements; Spectral can reduce likeness and fine facial detail. Peak observed RSS was 8.809 GiB with normal pressure and flat swap.

The preset-size qualification adds eleven real GUI/API requests: default portrait ordinary/Spectral, three-LoRA landscape, style and identity references, largest-area/widest presets, legacy size omission and exact restored-base pixels. Peak observed group RSS was 8.357 GiB with flat swap and no guard trip. The matched portrait pair sampled 24.34% faster with Spectral; this is a conditional pair, not a universal speed claim.

The floating-source GUI qualification adds a full BF16 Turbo conversion and a two-LoRA bake. The GUI successfully exports, installs, selects and saves the new checkpoint; ordinary/Spectral portraits pass. Later long-session pressure stops were also recorded. The subsequent memory follow-up qualifies persistent original → baked Spectral → original restoration and fixes warm export offload: one stock render followed by a full two-LoRA BF16-source export passes unchanged guards, with identical qualified tensor payloads. You can export after rendering; preserve and investigate any new guard failure before replaying it.
