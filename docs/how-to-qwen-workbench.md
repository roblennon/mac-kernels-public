# Qwen 2.1 in ComfyUI

Double-click **Qwen Workbench.command** in this repository. It starts the existing guarded native Comfy server at `http://127.0.0.1:8188`; the launcher reuses Video Workbench’s environment, models and user directory. Open [ComfyUI](http://127.0.0.1:8188/) once the server is ready. Close another Krea/native Video server first: they share the exclusive GPU lock.

Drag [qwen_text.json](../bench/workflows/qwen_text.json) onto the canvas. Use the ordinary JSON, not `qwen_text_api.json`. Edit the prompt fields in **Qwen 2.1 checkpoint and prompt**, set width/height in **Image size**, change the seed or sampling controls in **Seed and sampling**, then click **Run**. The result appears at **Save Qwen image** and under `$MK_ROOT/out/comfyui/qwen/`.

The default reuses the accepted 512×512 native INT8 sampling recipe: 25 Euler/simple steps, CFG 1, full VAE, no prompt enhancer, prefix cache auto/default. The preparation node encodes on CPU, releases the encoder, then maps the installed INT8 checkpoint read-only on CPU and caps sampling residency at 4 GiB on MPS. Width/height in the latent node control generated size. Larger sizes are editable but are not automatically qualified by the 512px acceptance. The default CFG 1 does not use negative conditioning; set CFG above 1 to apply the negative prompt, at additional compute cost.

**Optional Qwen LoRAs** provides five independent filename/strength slots through the existing model-only stack. `none` or strength zero disables a slot. Select Qwen-compatible model adapters; geometry validation cannot identify the correct training family. Leave all slots disabled for the base recipe. New combinations are not rendered acceptance. Selecting an acceleration LoRA here does not change the sampler recipe; use the Viggle or Turbo workflows below for fast drafts. This bounded path admits plain A/B Linear-weight model adapters, including LoRAs that address the two halves of each block's fused `img_mlp` weight (most Qwen 2.1 LoRAs do). LoRA factors count against the 4 GiB residency cap, so a large adapter streams a little more weight per step. Text-encoder, norm/bias differences, DoRA, LoHa/LoKr, mid and reshape adapters are outside its admitted types.

## Viggle turbo (fast drafts)

Drag [qwen_viggle.json](../bench/workflows/qwen_viggle.json) onto the canvas instead. It uses the same checkpoint, prompt and memory loader, with Viggle's 6-step turbo LoRA already in slot 1 at strength 1.0 and its own sampler. Slots 2–5 stay free for other LoRAs.

- **Viggle size and schedule** sets width and height *and* computes Viggle's 6-step schedule for that size. Set the size only here; there is no separate latent node.
- **Seed** is the noise seed. Sampling is Euler at CFG 1 through **CFG 1 guider**, so the negative prompt has no effect.
- The result appears at **Save Qwen Viggle image** and under `$MK_ROOT/out/comfyui/qwen/viggle_*.png`.

Measured October 9, 2026, through the Workbench server: 1024×1024 in 145 s cold (about 12–14 s/step), and 768×768 with a second LoRA stacked in slot 2 in 130 s (9.4 s/step). Viggle was qualified at 1024²: it keeps the 25-step composition at about half the time. Other sizes use the same schedule formula but are not separately qualified. Style LoRAs can behave differently under Viggle than with the plain recipe, so judge stacks by eye. To return to full-quality sampling, use `qwen_text.json`.

## Turbo (fast, clean drafts)

Drag [qwen_turbo.json](../bench/workflows/qwen_turbo.json) onto the canvas. It uses Qwen's official 8-step [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo), converted to our INT8 layout (`qwen_image_2.1_turbo_int8_convrot.safetensors`), with its own saved schedule in **Turbo 8-step schedule**. Set the size in **Image size**; the schedule holds at every size. Sampling is Euler at CFG 1, so there is no negative prompt. The LoRA slots work as in `qwen_text.json`.

Measured October 10, 2026, through the Workbench server: 1024×1024 in 140 s. Turbo composes differently from the base model, and in a comparison it gave the cleanest image of the fast methods. Viggle keeps the base model's look; Turbo is its own model.

## Image editing (Turbo)

Drag [qwen_edit.json](../bench/workflows/qwen_edit.json) onto the canvas.

- **Image to edit (image 1)**: upload the picture to change. The output has its aspect ratio, resized to about **resolution** pixels square (1024 by default; 0 keeps its own size).
- **Qwen 2.1 edit: images, instruction and checkpoint**: write the edit as an instruction, saying what changes and what stays. The node also has two optional image inputs, `image_2` and `image_3`, for reference images: add a **Load Image** node and connect it, then name the images in the prompt as `<image1>`, `<image2>`, `<image3>`.
- **Seed**, then **Run**. The result appears at **Save Qwen edit** and under `$MK_ROOT/out/comfyui/qwen/edit_*.png`.

It uses ComfyUI's own Qwen 2.1 edit encoding: the encoder sees the images, their VAE latents join the sequence, and sampling starts from a latent the size of image 1. The checkpoint field also offers the base model, but this graph's 8-step schedule is Turbo's; the base model edits with the 25-step `qwen_text.json` sampler instead.

Measured October 10, 2026: a one-image 1024² edit in 195–225 s, about 11–13 s/step. The subject, pose and framing stayed put while the requested change was made cleanly. Each extra reference image adds a full image of tokens and more step cost (about 16 s/step on an October 11 two-image Turbo run; an earlier load saw ~33 s/step).

**Reference images:** use an **object-only** reference for `image_2` (garment on a mannequin or plain background, no person). October 11: with that setup, Turbo transferred a nurse apron onto a portrait while keeping pose and background. A full fashion photo of a person wearing the garment tends to make the model near-copy the reference instead — that failure mode was seen October 10. Name images in the prompt as `<image1>`, `<image2>`, `<image3>`.

The prompt enhancer from the official template (a separate 9B model that rewrites instructions) is not included.

The installed encoder and VAE are the Qwen 2.1 INT8 ConvRot release files. The mapped loader admits two diffusion checkpoints by header hash: the Comfy-Org base INT8 release and our Turbo INT8 conversion. Other checkpoint files and formats need their own qualification. There is no training GUI here. Krea’s specialized Identity Edit remains in its own Workbench.

The `_api.json` uses the same graph with the local Comfy queue API.

The qualified 512×512 base run took about **2 minutes 27 seconds**, including fresh prompt encoding. The mapped default accommodates less free RAM than the older direct-load template. Speed and memory use depend on host activity; larger images, CFG above 1 and added adapters can cost more.
