# Local video in ComfyUI

The launcher reuses the existing model installations. Double-click **Video Workbench.command** for native WAN/H3 at `http://127.0.0.1:8188`. Drag a workflow JSON from `bench/workflows` onto the canvas. The `_api.json` versions are for scripts, not canvas import.

The generation launchers use one memory policy: warning pressure is recorded and allowed; critical or unknown pressure stops immediately. The limits are 18 GiB process-group RSS, 4 GiB peak systemwide swap growth and 4 GiB internal swap-volume reserve plus the unused swap allowance (8 GiB free at startup). Reclaimed-memory percentage is not a default stop condition. Deadlines and exclusive GPU ownership remain. WAN14 and H3 now complete short renders at the tested sizes with their bounded loaders. Larger geometries are not covered by this acceptance.

Only one GPU workload runs at a time. The Krea and native Video servers own the GPU lock for their lifetime. The native launcher has a six-hour session deadline; restart it before a longer session. The launchers keep separate user/database directories; outputs and model assets remain under `MK_ROOT`.

## Which workflow?

| Workflow JSON | Use | Current evidence |
| --- | --- | --- |
| `video_wan5b_t2v.json` | Text-to-video, save sample + tiny draft | 704×416×33 base GUI graph passes (48s); a real compatible WAN5B LoRA passes the I2V route (76s). |
| `video_wan5b_i2v.json` | Upload a starting image, save sample + tiny draft | Full VAE image conditioning passes at 704×416×33 (72s) with direct INT8 loading; first-frame identity holds and motion is modest. |
| `video_wan_decode.json` | Open a saved sample; choose Draft or Final, without sampling | Comfy graph decodes both headless and newly saved native samples without sampling: Draft ~2s, Final ~24s including video output. |
| `video_wan14b_t2v.json` | Separate high/low checkpoints and LoRA stacks | Updated bounded loader:640×384×33 diagnostic completes in299s; native Comfy wrapper640×384×9 completes in328s, with normal pressure and no swap growth. Installed assets are T2V, not the separate A14B I2V checkpoints. |
| `video_h3_t2v.json` | H3 W4A8, four-step Turbo, video/audio | Bounded W4A8 loading completes480×288×22 in576s through the native diagnostic wrapper. The regular Comfy loader completes five frames in423s with video/audio; larger renders remain unqualified with this strategy. |
| `video_h3_i2v.json` | Same H3 graph with a Load Image on `first_frame` | Generated October 13; not yet rendered. The installed checkpoint is the first/last-frame variant and the node notes a trained range of about 124–362 frames (5–15 s), so the first useful probe is 124 frames, not 22. |

WAN5B I2V enables the existing advanced **mps_direct_load** switch on Load Diffusion Model. Ordinary CPU-staged I2V loading pressure-aborted on this host; direct loading passed with the shipped Turbo INT8 file. Its strict storage/headroom checks may refuse other checkpoints; switching files does not qualify that loading profile.

Every preset defaults to portrait since October 13 (WAN5B 416×704, WAN14 384×640, H3 288×480); the measured times in this guide were taken at the earlier landscape sizes with the same pixel counts. Native workflows expose checkpoint, VAE, prompts, dimensions, frame count, seed and sampler controls as ordinary Comfy nodes. WAN14’s **WAN14 Memory Loader** selects only the qualified installed high/low INT8 ConvRot experts. Restart Video Workbench and reimport the updated workflow to use it. It combines read-only CPU construction, bounded weight residency and streamed adapter factors; the guard and allocator limits stay unchanged. Keeping the shipped settings preserves the intended Turbo recipe; changing a checkpoint does not automatically choose its correct schedule.

H3’s **H3 Memory Loader** selects the installed pruned W4A8 release only. Restart Video Workbench and reimport `video_h3_t2v.json` to use it. It preserves mapped quantization metadata, bounds weight residency to1GiB and streams packed-block LoRA factors; ordinary BF16 refiner LoRAs use Comfy’s normal path. Keep the four-step accelerator and sampler settings. The default is480×288×22; frame counts round upward to `17k+5` (minimum five), so six or eleven still generate22 frames. Longer or larger renders need separate acceptance.

## LoRAs

Each **Video LoRA Stack** has five independent filename/strength slots. `none` or strength zero disables a slot. The stack uses the fork's normal adapter conversion/patching and rejects unmatched tensors instead of silently applying part of an incompatible file. It supports plain A/B model LoRAs (including alpha scaling) plus mapped bias/norm/difference patches. WAN Diffusers plain A/B naming is supported, including alpha; aliases are limited to actual WAN models. Text-encoder, DoRA, LoHa/LoKr, mid-weight and shape-changing reshape adapters are currently refused. Matching tensor geometry cannot establish the correct training family or intended noise expert: select the correct release yourself.

WAN14’s bounded loader accepts only plain A/B LoRAs on quantized block weights; bias/norm/difference patches, ordinary global targets and weight transforms are refused. The installed accelerator pair is rendered acceptance; other compatible style combinations have not been newly rendered in this pass.

For **WAN 14B**, put the high-noise file in the HIGH stack and its low-noise counterpart in LOW, at independent strengths. Both stacks are after the matching accelerator. Add several pairs using corresponding slots; the strengths need not match. The shipped Lightx2v recipe uses four total steps with the handoff at two: high adds noise and retains it, low continues the same latent without adding noise. This split belongs to that accelerator, not every WAN checkpoint. WAN5B is a single TI2V model and cannot use A14B adapter pairs. [Official WAN workflows](https://docs.comfy.org/tutorials/video/wan/wan2_2), [noise-based expert selection](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/text2video.py).

For **H3**, select adapters compatible with the installed **pruned** checkpoint. Full-checkpoint adapters can target removed adaln weights; the new stack refuses that partial match. The bounded loader admits plain A/B adapters on packed blocks and the eight BF16 token-refiner matrices; other targets, bias/norm/difference patches and weight transforms are refused. All208 installed accelerator targets are retained. Extra style slots are code-supported but have not been newly rendered in this pass. Keep the installed accelerator's own four-step recipe. [Official H3 compatibility guidance](https://docs.comfy.org/tutorials/video/minimax/minimax-h3).

Place native WAN adapters under `$MK_ROOT/models/wan/loras`, H3 adapters under `models/h3/loras`, or shared compatible files under `models/loras`. Refresh Comfy's model lists after adding files. A filename appearing in the menu does not prove compatibility.

## Time and quality expectations on the M1 Max, 32 GB

These are measured short recipes, including GUI and adapter runs. They are not controlled provider rankings or linear estimates for long videos.

| Provider/recipe | Measured time | What the inspected output showed |
| --- | --- | --- |
| WAN5B Turbo, 704×416×33, four steps | 72s headless canonical; earlier Comfy sequence 59/81/112s | Coherent musician, guitar and station. Smaller tested geometries were distorted; keep this starting size. |
| Same saved WAN latent, full/tiny decode | 20.9s full; 1.0s tiny | Tiny retains composition/colors but softens detail. Draft decode does not accelerate sampling. |
| WAN14 bounded, 640×384×33, four steps | 299s diagnostic; native nine-frame graph328s | Consistent musician/guitar/subway setting. Request times vary; these are not linear estimates. |
| WAN14, historical 832×480×81 | About 17min | Historical timing only; this larger size is not qualified with the new loader. |
| H3 bounded, 480×288×22, four steps | 576s diagnostic including video/audio decode | Consistent musician/guitar/station and moving commuters. Audio is present; listening quality was not evaluated. |
| H3 bounded, regular Comfy, 480×288×5 | 423s including video/audio decode | Minimum-frame wiring check only; do not judge motion quality or scale time linearly. |
| H3, historical 832×480×56 | About 4.6min sampling/decode, 7.8min cold | Historical timing only; this larger size is not qualified with the new loader. |

## Draft first, final later

1. Launch **Video Workbench** and import the updated WAN5B T2V or I2V JSON. Run with **Draft (tiny)** selected. It samples once, saves a native `.latent` under `out/comfyui/latents/wan`, and makes a tiny-decoder MP4. Saved latents are published before decode, so a failed preview does not discard the completed sample.
2. Import **video_wan_decode.json**, or open **WAN - Saved Draft or Final** from the Workflows sidebar. Refresh Comfy's model lists after saving, then select your file in **Open saved WAN sample**. New GUI files have a `Comfy/` prefix; existing headless samples have `Headless/`.
3. Set **quality → Final (full)** and Run. This graph contains only saved-latent loading, decode and video saving—no checkpoint, encoder or sampler. You can also select **Draft (tiny)** here to preview an existing sample.

The sample fixes the prompt, seed, dimensions, frame count and applied LoRAs; changing those requires sampling again. FPS/output settings remain editable in CreateVideo/SaveVideo. Changing quality in the original generation graph can reuse its live cache, but use the separate saved-sample graph to guarantee no resampling after cache eviction or a restart. Changing a saved file invalidates its loader cache by content hash.

WAN5B uses the installed `lighttaew2_2` draft or full `wan2.2_vae`. The generation graph reuses its existing full VAE for Final, and I2V keeps the full VAE for conditioning regardless of preview quality. WAN14 headless samples use 16 channels and can decode with the installed full Wan2.1 VAE. Their matching `lighttaew2_1` draft asset is not installed, so Draft reports that missing file; it never substitutes the incompatible WAN5B decoder. Tiny drafts soften detail. The [headless saved-latent procedure](how-to-run.md#wan-draft-and-final-from-one-sample) remains available.

A fresh pressure comparison distinguishes historical completed renders from later guarded aborts. Competing workloads are plausible, but not proved causal; the later partial-loading and current GUI profiles also differ from successful headless conditions.

## Acceptance and remaining work

A later guarded acceptance pass renders the editable presets through their API graphs, alongside the existing frontend serialization proof. WAN5B base, I2V, a real painting adapter and saved-sample Draft/Final pass. The October8 WAN14 ownership fix also completes through the native Comfy wrapper. The October8 H3 ownership fix completes22 frames through the native diagnostic wrapper and five frames through the registered Comfy loader. Longer/larger H3 acceptance and additional style stacks remain deferred.

The October 7 research pass checked the August 17 revision of [SPEED](https://arxiv.org/abs/2605.18736v3) and [author code](https://github.com/howardhx/speed): its published video implementation targets WAN2.1 1.3B, so WAN2.2/MoE needs a separately scoped scheduling experiment. CUDA-only sparse attention, FP8 CUDA advice and the diffusion-VAE natten backend are not Mac speedups. Future formats should retain explicit pack/runtime identity and reuse native Metal/MLX paths, with CPU format checks before expensive renders.
