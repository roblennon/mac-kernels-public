# Krea 2 blocks, controls, and variety

Research checked October 12, 2026. This is a research note, not a runtime qualification or an implementation plan. No models or custom nodes were installed, and no generations were run.

The useful working hypothesis is that we can reduce a finetune's unwanted identity bias while retaining much of its aesthetic treatment. Which weights to change must be measured for each checkpoint or LoRA. Variety also depends on prompt expansion, references, and spatial routing; block merging addresses only part of the problem.

## What the supplied article establishes

lonecatone23's article on same-face syndrome, published September 24 and updated September 25, reports Krealism/Turbo ablations on an RTX 5090:

| Region | Reported effect in this comparison |
| --- | --- |
| Blocks 7–13; especially 9 | Strong facial-structure changes |
| Blocks 21–27 | Mostly color and skin/hair treatment |
| `txtfusion` | Swapping it lost desirable candid framing |
| Other named regions | Little visible effect in the initial portraits |

The selected Krealism 3.1 restores Turbo blocks **4–13**, keeping everything else from Krealism 3.0.

Final evaluation: 14 prompts × 8 seeds × 4 models = 448 images; Euler/simple, 10 steps, CFG 1, no shift node or LoRAs.

| Model | Turbo replacement | Face similarity ↓ | Within-image crowd similarity ↓ |
| --- | --- | ---: | ---: |
| V3.0 | None | 0.458 | 0.140 |
| H | 7–13 | 0.328 | 0.138 |
| K / V3.1 | 4–13 | 0.321 | 0.126 |
| N | 4–17 | 0.330 | 0.127 |

H reportedly lost landscape variation. Earlier portrait experiments used 8 steps and shift 4; their scores are not directly comparable. Defaults remained demographically narrow. Quantization fidelity claims are author-reported.

**Evidence boundary:** full article text and metadata were retrieved. Test-sheet image downloads were blocked by browser security at the article site's image host; the sheets were not independently visually inspected. These results were not reproduced locally.

## Architecture: what the blocks actually are

Krea 2 is a single-stream multimodal diffusion transformer: text and image tokens share 28 main blocks, indexed **0–27**. Each repeats attention and an MLP. This is not a UNet down/middle/up block scheme, and the block index is not a denoising step. A depth window and a timestep schedule are different controls. The [Ostris architecture reference](https://ostris.com/docs/ai-toolkit/models/krea2) documents width 6144, 48 query heads, 12 KV heads, and Qwen-Image VAE latents.

The [official forward pass](https://github.com/krea-ai/krea-2/blob/db3984fbc6e13b34c0064990fc2d95ac64d00058/mmdit.py) processes text through `txtfusion`, then `txtmlp`, concatenates it with embedded image tokens, and runs the main blocks. `txtfusion` contains two layerwise blocks, a learned layer-mixing projector, and two refiner blocks. It fuses **text-encoder hidden states**, not image tokens at that stage. Some third-party tooltips call it text/image fusion; the official implementation is the better explanation.

| Checkpoint region | Architectural job |
| --- | --- |
| `first.` | Embed image patches |
| `tmlp.` | Embed timestep |
| `tproj.` | Project timestep conditioning for block modulation |
| `txtfusion.layerwise_blocks.0/1.` | Process selected text hidden states |
| `txtfusion.projector.` | Mix the selected encoder-layer dimension |
| `txtfusion.refiner_blocks.0/1.` | Refine the mixed text features |
| `txtmlp.` | Project text into main-transformer width |
| `blocks.0.` … `blocks.27.` | Joint text/image processing |
| `last.` | Produce output patch predictions |

Interpretation: changing text fusion can change how the same words condition the entire generation. That makes it a plausible cause of framing changes, but does not establish it as a dedicated composition module. Similarly, a facial effect from block 9 does not establish that block 9 exclusively represents identity. Weight differences identify where parameters moved; they do not prove dataset narrowness or semantic responsibility.

## Tools and techniques worth retaining

### Checkpoint merging: region-specific donor weights

[ComfyUI's `ModelMergeKrea2`](https://github.com/Comfy-Org/ComfyUI/blob/14c553159994b626738c392eb99d295ea3d683c1/comfy_extras/nodes_model_merging_model_specific.py) already enumerates all 28 blocks and ten named regions. The [LC advanced node](https://github.com/lonecatone23/ComfyUI_LC_ModelBuilder_nodes/blob/37eef01b2e42b9feef92c30b5b70d08578d59afa/lc_krea2_block_merge.py) adds an editable curve and presets around core's merge mechanism.

**Ratio direction matters.** In the inspected LC node, `0 = model1`, `1 = model2`. It explicitly inverts those values before calling [core `ModelMergeBlocks`](https://github.com/Comfy-Org/ComfyUI/blob/14c553159994b626738c392eb99d295ea3d683c1/comfy_extras/nodes_model_merging.py), whose ratio weights model1. With the finetune as model1 and Turbo as model2, the article's candidate means LC values 1 in 4–13 and 0 elsewhere, including named regions. The corresponding core values are reversed. Record the implementation revision with a recipe.

Use compatible floating checkpoints and preserve Turbo lineage when investigating an eight-step model. Tensor-shape compatibility alone does not make RAW/Turbo mixtures valid fast samplers. Core's prefix fallback also makes complete region coverage preferable to an underspecified curve.

### LoRA editing: reduce the offending delta

The [LC pack's current LoRA documentation](https://github.com/lonecatone23/ComfyUI_LC_ModelBuilder_nodes/blob/37eef01b2e42b9feef92c30b5b70d08578d59afa/README.md) adds block scaling, separate attention/MLP controls, and singular-direction softening. Its examples identify face lock in **16–21** for one LoRA and **4–9** for another. These are the same author's additional reports, not independent replication of the article.

The [Block Scan implementation](https://github.com/lonecatone23/ComfyUI_LC_ModelBuilder_nodes/blob/37eef01b2e42b9feef92c30b5b70d08578d59afa/lc_krea2_lora_scan.py) compares no LoRA, full LoRA, and one group disabled at a time. Default groups: 0–3, 4–9, 10–15, 16–21, 22–27, and fusion. Eight seeds across those eight conditions cost 64 renders. Its 0.03 selection threshold and roughly 0.02 claimed noise level are heuristics, not confidence intervals.

[ShootTheSound's selective loader](https://github.com/shootthesound/comfyUI-Realtime-Lora/blob/47d5962a651e61a39afcf06c7cf26614454c5fe7/lora_analyzer_v2.py) independently implements Krea per-block enable/strength controls, an `other_weights` group, schedules, and refined-LoRA saving. Its presets are experiments to try, not validated semantic partitions. Parameter-impact heatmaps help choose ablations; large norms alone cannot identify the best edit.

### Newer LC utilities: beyond the September article

The inspected LC revision is **v0.5.1, October 12**, and contains tools absent from its shorter README overview:

| Utility | Why it matters | Boundary |
| --- | --- | --- |
| [LC Model Merge Any Precision](https://github.com/lonecatone23/ComfyUI_LC_ModelBuilder_nodes/blob/37eef01b2e42b9feef92c30b5b70d08578d59afa/lc_model_merge_any.py) | Merges donors stored in different precisions | Dequantization/rounding still affect the result; no local memory proof |
| [LC LoRA Extract](https://github.com/lonecatone23/ComfyUI_LC_ModelBuilder_nodes/blob/37eef01b2e42b9feef92c30b5b70d08578d59afa/lc_lora_extract.py) | Factorizes finetune-minus-base into adjustable low-rank changes | Approximation loses information; output can include full differences and bias patches |
| [LC Model Quantize](https://github.com/lonecatone23/ComfyUI_LC_ModelBuilder_nodes/blob/37eef01b2e42b9feef92c30b5b70d08578d59afa/lc_model_quantize.py) | Exports multiple quantized formats | Format names do not establish native Mac admission |
| [LC Scoreboard](https://github.com/lonecatone23/ComfyUI_LC_ModelBuilder_nodes/blob/37eef01b2e42b9feef92c30b5b70d08578d59afa/lc_scoreboard.py) | Saves human A/B wins, ties, and losses with the workflow | Human tally, not an automatic quality metric |

Extracting only a finetune's useful changes could be a smaller experiment than storing several merged checkpoints. This is a hypothesis worth checking, not a claim that a low-rank adapter can exactly reproduce a full checkpoint.

## Measuring variety without fooling ourselves

The [photo scorer code](https://github.com/lonecatone23/ComfyUI_LC_ModelBuilder_nodes/blob/37eef01b2e42b9feef92c30b5b70d08578d59afa/lc_face_variety.py) uses SCRFD detection, aligned ArcFace embeddings, and mean pairwise cosine similarity. It selects the largest eligible face per image for cross-image comparisons. `same_pct` counts pairs above **0.35**. Crowd scoring averages pairwise similarities within each eligible image, then averages those image scores. Missing detections are skipped; faces narrower than 40 pixels are excluded by default.

Cosine similarity is mathematically in −1…1; the node's advertised 0…1 range is a practical convention, not an enforced clamp. Crowd `-1` is an unavailable sentinel; too few main faces produce NaN. The current node also has an anime CCIP mode whose score is a different quantity: do not compare its numbers to ArcFace scores.

The scorer runs the ONNX models directly, so the InsightFace Python package is unnecessary. However, [InsightFace distinguishes its MIT library from pretrained model licensing](https://github.com/deepinsight/insightface/blob/master/python-package/README.md); the referenced pretrained weights are restricted to non-commercial research.

Our interpretation and measurement requirements:

- Facial identity variety, demographic coverage, stylistic variety, scene variety, and separation of people within a crowd are distinct outcomes. One number cannot stand in for all five.
- Lower face similarity can come from changed angle, expression, obscured faces, or degraded rendering. Review face crops and complete images alongside the score.
- Report detection coverage and failure counts. A model that stops rendering detectable faces must not win by omission.
- Keep comparisons within a prompt. Pooling different prompts can create apparent diversity from prompt changes alone.
- Eight images produce 28 pairs, but those pairs share images and are not 28 independent observations. Estimate uncertainty across seeds/prompts, not by treating pair counts as independent samples.
- Thresholds do not prove kinship, demographic fairness, or real-world identity. Treat `same_pct` as a repeatability diagnostic.
- A small mean-score difference should not decide a winner without repeat batches and visible benefit. Preserve composition, prompt adherence, hands, background variety, and acceptable image quality as separate checks.

## Variety controls outside checkpoint weights

**Prompt expansion can collapse variety.** Krea's [technical report, June 23](https://www.krea.ai/blog/krea-2-technical-report), describes prompt expanders converging on a safe house style. Krea kept a DINOv3 diversity reward active during expander training; weakening it reduced variation. The same report identifies content leakage from style references as a failure mode. Neither finding proves that any particular local workflow has these problems, but both give us useful upstream causes to investigate before editing weights.

The [official prompting guide](https://github.com/krea-ai/krea-2/blob/db3984fbc6e13b34c0064990fc2d95ac64d00058/docs/prompting.md) favors natural-language descriptions and detailed prompts, while accepting short prompts. Its [expansion template](https://github.com/krea-ai/krea-2/blob/db3984fbc6e13b34c0064990fc2d95ac64d00058/docs/expansion.txt) emphasizes preserving subjects, actions, spatial relationships, and medium rather than inventing details.

Practical inference: preserve the submitted and expanded prompts. Audit whether the expander repeatedly adds the same beauty, age, lighting, lens, or editorial cues. For aesthetic exploration, vary medium, palette, camera position, lighting, motion, environment, and facial morphology deliberately, while holding other variables fixed. For testing default bias, leave those attributes unspecified. These are two different experiments.

[Krea2 Harness](https://github.com/ANe5s/ComfyUI-Krea2-Harness/tree/a14c4395d6f7a69bec31c02aecfd97c5712b2c08) provides prompt cleanup and moodboard-metadata compilation with source tracking and fallback. It adapts textual style metadata; the moodboard adapter itself does not read card images. This offers a way to organize creative directions, but the author's evolution/version-count claims are not comparative evidence of increased variety.

[Official Comfy documentation](https://docs.comfy.org/tutorials/image/krea/krea-2) separates RAW training from Turbo inference and offers style LoRAs and image style reference. A style reference or style LoRA is a useful aesthetic control; neither guarantees distinct identities.

**Multiple characters are a separate routing problem.** [CliffNodes V12](https://github.com/CliffNodes/Krea2-Multi-Character-Lora-Node-with-bounding-box-Scene-and-Outfit-Edit/tree/a33a0e487fbaaa6b64a19ad6e26e707695723b75) combines regional LoRA deltas, text/image attention ownership, boxes, edit references, and optional face/body refinement. This is aimed at identity mixing between subjects, not default cross-seed face variety. Its README requires FlexAttention and compiled Triton execution. The author's strong no-bleed guarantees and timings are not local Mac evidence; stock LoRA stacking here cannot reproduce that mechanism.

## What transfers to this Mac workbench

The existing [Krea guide](how-to-krea-workbench.md), [adapter admission](../bench/krea_adapter.py), and [checkpoint exporter](../bench/krea_export.py) establish the current scope:

| Idea | Current local fit |
| --- | --- |
| Controlled prompt/seed grids | Existing text generation can supply images; scoring is a separate unqualified tool |
| Compare with/without ordinary LoRA | Supported at eight Euler/simple steps, CFG 1 |
| Save a selectively edited ordinary LoRA | Potentially compatible if it contains only admitted 2-D linear LoRA tensors; validate first |
| Block-specific runtime sliders | Not exposed by the Workbench |
| Full-checkpoint donor merging | Not implemented by the local exporter; its current job is baking ordinary LoRAs |
| Custom Turbo checkpoint | Requires local schema validation, recorded provenance, and supported INT8 ConvRot storage |
| Install the article's INT8 release directly | Filename is insufficient; current checkpoint provenance/admission rules still apply |
| LoKr/LoHa or extracted bias/full-difference patches | Outside current ordinary-LoRA admission unless converted into supported tensors |
| LC/custom Comfy graphs | Dedicated server rejects unsupported graphs; installing a node does not qualify its workflow |
| Regional FlexAttention routing | Would need a separate compatibility and kernel investigation |

Keep Spectral **off** for the initial variety comparison: our guide documents that it can change faces and composition. Likewise, leave references and extra LoRAs off until a clean baseline exists. CPU face scoring is architecturally plausible on this Mac, but has not been installed or measured here. Preserve the current memory guard; two fully materialized floating checkpoints are not a safe assumed merge strategy.

## Smallest useful next experiment

This is a proposed comparison, not a scheduled run or claimed acceptance result.

1. Diagnose the source of repetition with the same prompt and eight seeds: base Turbo, selected finetune if admitted, and base plus one suspect LoRA. Record exact checkpoint/adapter hashes, prompt, seed, dimensions, sampler, steps, and Spectral state. An expander comparison must also record the actual expanded text.
2. Score each prompt separately, report detectable faces, and inspect the complete grid. Repeat a promising result with fresh seeds. Include one less typical explicitly described subject to test controllability separately from the open-prompt default.
3. If the LoRA is the cause, compare full strength, lower global strength, and one block-group attenuation. Keep the desired effect as a scored requirement; removing the LoRA entirely is not a successful repair.
4. If the checkpoint is the cause, investigate a compatible floating donor merge only after the cheap tests. Start from a measured candidate window, preserve named regions, and check actual tensor provenance before quantization.
5. Challenge the apparent winner with a crowd, a hand/object interaction, a nonhuman scene, and held-out prompts/seeds. Reject it if the improvement depends on detection failures, destroys the desired effect, or narrows scene variety.

Cost: 24 images for the initial three-condition, one-prompt screen; a full default LC block scan is 64 images before held-out checks. Local generation time must be measured for the selected size and stack. No 448-image sweep is warranted before the diagnostic screen shows something worth pursuing.

## Source freshness and unresolved questions

Primary sources were preferred over node-directory summaries and reposts. All links were checked October 12. LC code is pinned above to October 12; official Krea code to July 24; ShootTheSound to August 21; Harness to October 3; CliffNodes V12 to August 1. Comfy core was inspected at October 12 HEAD. Ostris and Comfy documentation are live pages without a publication date established here.

Still unknown: whether the reported block window transfers to our selected checkpoints and adapters; whether eight-step native INT8 sampling preserves the reported gains; the exact statistic behind the article's percentage weight changes; and the complete scoring/reproduction artifacts for the 448-image run. No independent reproduction of that particular merge result was established. Tool code confirms mechanisms and available controls, not image-quality outcomes.
