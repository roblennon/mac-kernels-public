# Running the models in ComfyUI on this Mac

For an editable Qwen2.1 canvas, use [Qwen Workbench](how-to-qwen-workbench.md). Its launcher reuses the existing native Comfy server; import the ordinary `qwen_text.json`.

For WAN and H3, start with the [video workbench guide](how-to-video-workbench.md). It distinguishes qualified short recipes from experimental profiles and explains separate high/low LoRAs and native Comfy workflows.

For Krea LoRA stacks, Identity Edit and image style references, use the dedicated [Krea Workbench GUI](how-to-krea-workbench.md). Its launcher and importable Comfy workflows reuse the bounded native INT8 profile and local queue API.

The fork in `vendor/ComfyUI` (branch `mps-int8-weight-only`) has produced headless outputs for Krea 2, Qwen-Image 2.1, Wan 2.2 (5B and 14B), and MiniMax H3 on this M1 Max with 32 GB. Browser/API template validation is narrower: the older Krea template has one-job proof, while the dedicated Krea Workbench has later LoRA/reference/Spectral and preset-size qualification (see its guide); Qwen has a completed three-request 512×512 session using the native direct loader, and Wan 5B has a completed three-request session at 704×416×33 as of October 4, 2026 UTC. The fork adds the Apple GPU pieces: integer weight kernels, LoRAs kept beside quantized layers, PyTorch attention by default, fp16 VAEs and the faster video VAE path. The startup script uses those fork defaults; it does not establish that every template fits the shared-memory budget.

## Starting the server

For normal operation, use the guarded Workbench launchers in the guides above. This section preserves lower-level server controls and earlier measured profiles; it is not a replacement for current supervised defaults.

- From a terminal: `bench/serve.sh`, then open [ComfyUI](http://127.0.0.1:8188).

The server reads models from `~/mac-kernels-data/models` (the external SSD must be mounted) and saves results to `~/mac-kernels-data/out/comfyui`. `bench/extra_model_paths.yaml` lists the model folders. Each model family keeps its own subfolder (`qwen21/`, `wan/`, `h3/`) and the loaders search all of them, so templates find files by name. If port 8188 is taken, the script says so and stops; set `PORT` to use another.

Startup also registers the existing `vendor/ComfyUI-GGUF-fork` checkout through a relative link in ComfyUI's custom-node directory. The **Unet Loader (GGUF)** node lists diffusion checkpoints from `models/gguf`. Setup is repeatable and preserves any different installation already occupying that node path. Extra arguments to the startup script are passed to ComfyUI, including `--cpu` for a registration-only check.

The **Load Diffusion Model** node has an optional advanced `mps_direct_load` switch, off by default. It requests direct loading for INT8 ConvRot safetensors on Apple MPS with the default weight dtype and CPU offload. Float8/NVFP4 storage, other quantization formats and dynamic loading are rejected before device reads; system and driver memory must both have sufficient headroom. Python file-loader callers can select the same path with `model_options={"mps_direct_load": True}`. CPU boundary tests and one 33.358-second Qwen portrait run through the real node have passed. That headless run replayed cached CPU conditioning. A subsequent fresh-server API request also passed at 512×512, 25 Euler/simple steps and CFG 1 with ordinary CPU text encoding and this switch enabled; a later initial/changed-seed/changed-prompt Qwen session also passed in 114.675 / 102.286 / 107.578 seconds, with fresh encoding/sampling on all three requests. A driver-headroom refusal can occur even when system RAM is available: the existing eviction pass responds to system memory, and this experimental option deliberately refuses rather than forcing all idle GPU models into CPU memory.

Model weights, text encoding and VAE decode share memory with the rest of macOS. This explicit profile passed the initial/changed-seed/changed-prompt Wan 5B and Qwen sessions on the 32 GB M1 Max, using the unchanged launcher:

```sh
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 PORT=8189 bench/serve.sh --cache-ram 18
```

Use both watermark values together. They are fractions of the recommended GPU working set, not physical RAM. `--cache-ram 18` sets the active-cache headroom threshold in GiB; it does not allocate an 18 GiB cache or guarantee that much free memory. This profile favors eviction: all three Wan requests reran text encoding, model loading and sampling, taking **59.268 / 81.167 / 111.854 seconds**. It proves a persistent server can complete this bounded sequence with normal pressure and no swap growth; it does not prove a resident warm-model speedup or the launcher's default profile.

WAN14/H3 now complete short generation through the dedicated [Video Workbench workflows](how-to-video-workbench.md), using their bounded read-only loaders and streamed adapters under the unchanged single guard. This supersedes the earlier quiet-host-next recommendation. Original untouched official templates and longer/larger recipes remain unqualified.

Keeping the server open is not a guarantee of fast or successful later requests. Validation runs must use `bench/gpu_runner.py --monitor` and the established limits.

## Template validation

This section records the older official-template qualification, separate from the later dedicated Krea/video Workbenches. The API exports in `bench/templates_api/` record the graphs being tested. A headless runner using the same model does not establish that the official template, switches and prompt enhancer work.

| Template | Verified result | Remaining work |
|---|---|---|
| `image_krea2_turbo_t2i_int8` | One API run completed in 456 s: 1024², eight steps, prompt enhancer on, LoRA switch off. `logs/template-session.out` records the output. | Repeated-server sequence and warm timing remain unproved. |
| `image_qwen_image_2_1_t2i` | Three sequential API requests passed in 114.675 / 102.286 / 107.578 s: INT8, native direct loader enabled, 512×512 latent, 25 Euler/simple steps, CFG 1, seed 7→8, then lime→cobalt prompt change. No LoRA; fresh CPU encoding, full FP16-compute VAE. Three distinct coherent fashion portraits inspected. | Accepted with the paired-watermark/cache profile above, flat swap and no pressure stop. Encoder resolution stays 1024 and Qwen prefix cache auto/default; enhancer off. Original larger defaults and ordinary diffusion loading remain unproved; no warm-model speedup claimed. |
| `video_wan2_2_5B_ti2v` | Three API requests passed: Turbo INT8, 704×416×33, four steps, Euler/simple, CFG 1, full VAE; 59.268 / 81.167 / 111.854 s. Recognizable musician and subway motion inspected. | Accepted at this bounded size and explicit server profile. Original larger defaults and image-conditioned mode are not established by this session. Use 704×416×33 for bounded acceptance; tested 512×288×17, 512×288×33 and 704×416×17 fail its quality gate. |
| `video_wan2_2_14B_t2v` | Original template unqualified; supported local route superseded by `video_wan14b_t2v.json` | The bounded Workbench loader completes short T2V with separate high/low accelerator stacks. This is not proof of the original untouched graph. [Guide](how-to-video-workbench.md). |
| `video_minimax_h3_t2v` | Original template unqualified; supported local route superseded by `video_h3_t2v.json` | The bounded Workbench loader completes short video/audio generation. This is not proof of the original untouched graph. [Guide](how-to-video-workbench.md). |

Krea Turbo NVFP4, MXFP8 and mixed W4A4/INT8 also pass matched native/stock **diagnostic** image comparisons at 512×512 and eight steps. These runs capture CPU conditioning in a separate process. NVFP4 uses a temporary direct-loading diagnostic; MXFP8/W4A4 use ordinary mapped CPU construction with a temporary 1 GiB persistent-residency cap. These helpers are not enabled by the server launcher or the `mps_direct_load` switch described above. A complete H3 quantized-encoder diagnostic now also passes with a temporary 1 GiB cap and fully offloaded persistent weights; its conditioning proof includes no diffusion or VAE. Original Wan14/H3 templates remain unqualified; their supported bounded Workbench workflows now pass.

The template runner now offers `--preset accepted` for those exact Qwen/Wan recipes. With no template names it selects only Qwen 2.1 and Wan 5B; other models cannot select that preset. `--dry-run` prints the resulting graphs without contacting the server. The historical preset remains the default, so existing invocations preserve their larger recipes. Both accepted graphs match the frozen successful graphs apart from output filenames. Use the paired-watermark/cache server profile above; preset selection does not configure an existing server or qualify a different profile.

The headless Wan runners use source-aware conditioning caches by default. `COND_CACHE=off` forces fresh encoding without reading or writing a cache. Changing the encoder's path/size/modification time, local encoder/runtime sources or recorded policy creates a new cache key; old prompt-only files are not reused. This is local diagnostic reuse, separate from ComfyUI's server execution cache.

## Current Krea Turbo headless recipe

The ordinary eight-step native INT8 recipe now completes with fresh CPU text encoding at 512×512 and 1024×1024. The shared read-only profile uses a **1 GiB resident-weight cap by default**, with a 0.5 MPS allocator fraction after text encoding through decode. Both 512px and 1024px repeats are exact. The internal-checkpoint profile completes 1024px fresh encoding/off in 189.2s and a conditioning hit in 140.0s, matching the earlier 294.6s external-source cold render bit-exactly and passing visual inspection. Those separate natural-file-cache runs are not a controlled speed ranking. These are headless results, separate from the API template above.

From this checkout, with the pinned vendor sources and model storage available:

```sh
MODE=metal FP16=1 ATTN=pytorch KREA_READONLY=1 COND_CACHE=off \
W=1024 H=1024 STEPS=8 CFG=1 SEED=42 TAG=local-turbo \
PYTHONPATH=vendor/comfy-kitchen \
.venv/bin/python bench/gpu_runner.py --monitor --seconds 360 \
  -- \
  .venv/bin/python bench/run_krea2.py
```

Use a new `TAG` for each output. Set `MK_ROOT` to another model/output storage root if needed. The profile requires the named static Turbo INT8 checkpoint and no LoRA; its weights must remain immutable on disk and in memory. `KREA_RESIDENT_GIB` accepts a positive budget up to 4, but larger caps are not qualified defaults. The guarded process RSS includes CPU encoding and is distinct from resident GPU weights.

`COND_CACHE=auto` optionally reuses CPU conditioning bound to prompt, encoder checkpoint stat identity, tokenizer/source bytes, runtime and encoding policy. `COND_STAGE=capture` saves it and exits before UNET/VAE loading. Cache miss/hit/off and provenance are recorded; invalid cached data is refused. Completed cache controls reproduce exact outputs, but replay latency varies substantially and has hit the deadline. Leave cache off for the measured cold recipe; no predictable cache or warm-server speedup is established. A subsequent cached-512px source-device comparison reproduces external-checkpoint stalls and qualifies internal diffusion-checkpoint placement as a local mitigation (50.4s ordinary-entry acceptance, exact outputs, normal 10 GiB RSS guard). Select an isolated model root with `MK_ROOT`; its diffusion checkpoint can be internal while encoder/VAE and output/cache directories link to existing external storage. The current machine has this profile at `out/source-device/root/`; it consumes 12.57 GiB and preserves the original external checkpoint. A later generation follow-through also qualifies this profile at 1024px for fresh encoding/off and conditioning hits. Generic auto-cache miss and API-server behavior remain separate.

`SPECTRAL=1` remains an optional output-changing route using the separately prepared pinned helper. A five-request same-process 512px sequence measures 23.8% lower sampling time; a fresh-process baseline/spectral/spectral/baseline 1024px bracket measures 24.4%, with exact repeats within each route. A second fox prompt/seed passes visual inspection at 512px. Identity/pose/framing may change; these scoped results do not promise general quality or API speed. `DENOISE_TIMING=1` adds synchronized diagnostic timing. The separately admitted four-step adapter route now passes the recorded 512px soccer/seed42 recipe with native INT8 side patches and finite inspected decode. Sampling is38.42s versus 43.82s in one separate-process eight-step comparison; this is a modest scoped saving, not a2× promise. The earlier writable pressure stops remain preserved.

Four-step integration is parked. The following settings preserve the released adapter experiment, use the immutable internal-checkpoint `MK_ROOT` profile with `MODE=metal`, `FP16=1`, `ATTN=pytorch`, `KREA_READONLY=0`, `KREA_ADAPTER_READONLY=1`, `LORA=1`, `LORA_NAME=krea2_turbo_4step_rank_64_lora_comfyui.safetensors`, `LORA_STRENGTH=1`, `STEPS=4`, `CFG=1`, `W=512`, `H=512`, `SEED=42`, and `SPECTRAL=0`. The named existing asset must match the pinned release hash; the budget is fixed at 1 GiB. Retain the monitored runner and memory guards above, a unique output tag and immutable source weights. Conditioning remains optional; changed runner source invalidates earlier cache identities. Static `KREA_READONLY=1` still rejects all LoRAs. Other released-accelerator resolutions/prompts/repeats and its API use remain unqualified. Ordinary style LoRAs use the separate eight-step profile below.

## Ordinary Krea style LoRAs

The current GUI supports up to five compatible plain LoRAs, Spectral and references: use the [Krea Workbench guide](how-to-krea-workbench.md). The single-LoRA instructions below preserve the earlier narrow headless qualification.

For one plain model LoRA, the immutable headless route adds `KREA_PLAIN_LORA=1` to `KREA_ADAPTER_READONLY=1`, `LORA=1`, `MODE=metal`, `KREA_READONLY=0`. Select a local `LORA_NAME` and finite nonzero `LORA_STRENGTH`; use eight Euler/simple steps, CFG 1, no spectral, and the fixed 1 GiB resident-weight cap. The accepted tests use 512px, seed 42 and default fork attention/diffusion dtype, with the existing internal-checkpoint model profile. rwbflat s500 at .8 applies all 224 quantized targets; Darkbrush at 1 applies those plus 40 BF16 globals. Trigger phrases are `rwbflat style` and `monochrome ink wash style`. Keep the guarded runner and immutable base; optional conditioning capture/reuse follows the same source-bound policy.

Admission reuses Comfy key conversion/parsing, requires every tensor and target to be consumed, and refuses unsupported transforms instead of applying a partial adapter. Ordinary variable ranks, scalar alpha and strengths are supported, including overcomplete ranks. This is one model LoRA: no stacking with the four-step accelerator, DoRA, LoHa/LoKr, text-encoder patches or spectral combination is admitted in this profile.

## Dedicated Krea API profile

These are historical bounded qualification probes. For current preset sizes, checkpoint selection and stacked adapters, use the [Krea Workbench](how-to-krea-workbench.md).

`bench/krea_server.py --root <profile-root> --port <unused-localhost-port>` registers the fixed 512px eight-step base qualification graph by default. The internal diffusion checkpoint and named external BF16 Qwen3-VL encoder/VAE must be selected exactly from that root. This opt-in wrapper is separate from `bench/serve.sh` and ordinary mixed workflows. The parent must own it under `bench/gpu_runner.py --monitor`, use the shared memory policy, bound the aggregate three-job deadline and stop the owned server afterward. The historical qualification used 10 GiB RSS/0.25 GiB swap-growth/normal-pressure/25%-free limits, a 390s aggregate guard and a 330s client deadline; those old limits are evidence conditions, not a second supported profile.

`bench/probe_krea_server.py` prints the three graphs without submitting by default. Its `--run` path uses an owned dedicated localhost URL, an external `--report` receipt and `--server-log` path; it does not launch or supervise the GPU server. The accepted initial/changed-seed/changed-prompt jobs take 50.581/37.976/73.778s and pass all 24 forwards/5,376 native calls, finite decode and expected graph-cache transitions. One diffusion model/load identity survives; changed seed caches conditioning, while changed prompt freshly encodes after unloading GPU residency. Both real encoder calls release their CLIP/encoder/patcher before sampling. Persistent conditioning reuse is disabled for this qualification; no `/free` request is used. Source/asset drift and graphs outside this exact profile are refused. That three-job proof does not qualify adapter/spectral API or general server speed.

Ordinary adapter selection is separately opt-in: add `--plain-loras` to the dedicated server and use `bench/probe_krea_server_lora.py`. Node 1 accepts `lora_name` (one safetensors filename from the profile’s `models/loras`) and `lora_strength` (finite and nonzero); an empty name selects base. Preparation executes on every request, unloads prior GPU residency, checks complete base restoration and freshly encodes on CPU before admitting a new clone. The server retains detached mapped weight sources and bounded converted CPU auxiliary sources, using existing Comfy backup/unpatch restoration; it permanently refuses further work after a restoration failure. The accepted five-request probe uses base → rwbflat .8 → rwbflat .4 → Darkbrush 1 → base with one prompt/seed: all 40 forwards/8,960 native calls, fresh encoder release and exact restored-base PNG pixels pass. Its owned aggregate guard is 600s, client deadline 540s, with the same memory limits and bounded cleanup.

## Historical headless recipes

These October 2–3 measurements describe individual headless runs, not completed template or repeated-server sessions. The current memory investigation can prevent a formerly successful recipe from being admitted. Video configurations exceeding the current per-generation deadline are historical results, not instructions to rerun them unchanged.

| Model | Files | Settings | Recorded timing and scope |
|---|---|---|---|
| Krea 2 | `krea2_turbo_int8_convrot`, `qwen3vl_4b_bf16`, `qwen_image_vae` | 1024², euler simple, cfg 1. Historical four-step experiment: `krea2_turbo_4step_rank_64_lora_comfyui` at 1.0; integration is parked. Current Workbench uses eight steps with optional Spectral | 4-step headless run: 197 s cold; 4 × 15.3 s sampling and 2.5 s decode. No measured warm-server time |
| Qwen-Image 2.1 | `qwen_image_2.1_int8_convrot`, `qwen3vl_8b_int8_convrot`, `qwen_image_2.1_vae_bf16` | 1024², euler, cfg 1. For speed add `Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128` (1.0) and sample with SamplerCustomAdvanced and **ManualSigmas** `1.0, 0.9678, 0.9334, 0.8572, 0.6668, 0.4001, 0` (6 steps, these values are for 1024²) | 6-step headless run: 147 s cold; 6 × 9.4 s sampling and 3.0 s decode. No measured warm-server time |
| Wan 2.2 TI2V-5B | `wan2.2_ti2v_5b_turbo_int8_convrot`, `umt5_xxl_fp16`, `wan2.2_vae` | 4 steps, euler, simple, cfg 1, shift 8 | 704×416×33: 42 s with cached conditioning; 1280×704×81: about 5 min in the historical run |
| Wan 2.2 T2V-A14B | `wan2.2_t2v_{high,low}_noise_14B_int8_convrot`, lightx2v 4-step LoRAs, `umt5_xxl_fp16`, `wan_2.1_vae` | Template "4-step LoRA" path: 2+2 steps, cfg 1, shift 5 | 832×480×81: about 17 min historically; exceeds the current bounded-run window |
| MiniMax H3 | `minimax_h3_fl2va_pruned_w4a8` (prefer it to W6A8), `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16`, `minimax_h3_video_vae_fp16` | Template Turbo LoRA switch on, 4 steps | 832×480×56: about 4.6 min sampling+decode, about 7.8 min cold in the historical run; 1344×768×56: about 14 min, outside the current window |

Fast drafts for Wan 2.2 5B: pick `lighttaew2_2` in Load VAE. It decodes in about 1-2 s instead of 20-155 s, softer but true to color and composition.

### Wan draft and final from one sample

The headless runner now preserves its sampled latent before decoding. `NO_DECODE=1` saves a separate uniquely named `-nodecode-…pt` artifact without overwriting earlier samples; repeated timing runs keep their own files. The RESULT receipt includes the exact filename, SHA256 and identity. Normal runs keep their existing names/replacement behavior and save the latent even if later decoding fails.

Use the saved filename with `bench/decode_wan.py`. Choose `lighttaew2_2.safetensors` for a draft or `wan2.2_vae.safetensors` for final detail; both use the same latent bytes, so final decoding does not rerun the sampler. Set `MODES=full VIDEO=1` for a 24fps MP4, and optionally `SHEET=1` for a contact sheet. The full VAE is still required for image conditioning. These instructions describe the headless path. The new [Comfy saved-sample workflow](how-to-video-workbench.md#draft-first-final-later) also persists latents for Draft/Final without resampling; its native and headless saved-sample GPU decode checks pass without resampling (Draft ~2s / Final ~24s).

From the repository, with the existing external `MK_ROOT` configuration, a guarded draft decode is:

```bash
LATENT=your-saved-latent.pt VAE=lighttaew2_2.safetensors MODES=full VIDEO=1 SHEET=1 \
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 \
.venv/bin/python bench/gpu_runner.py --monitor --seconds 180 -- .venv/bin/python bench/decode_wan.py
```

For the final decode, use the same filename and change only `VAE` to `wan2.2_vae.safetensors`. The accepted 704×416×33 sample decodes in **1.0s draft /20.9s full**, or **6.8s /30.6s** including process startup/video output. Tiny output is softer; repeated full pixels are exact. These are decode-only measurements.


## Known limits

- **Memory.** 32 GB is shared between the GPU and everything else on the Mac. Recent Qwen controls showed system pressure by the end of model admission, before first-forward work; one matched replay was terminated before admission returned. An earlier Krea control also stopped before a completed step; later separate-conditioning format diagnostics pass at their explicitly recorded loading/residency scope. The later Qwen BF16 demand-read route now completes six matched 1024px final images with finite synchronized decode and exact adapter binding. This is a captured-conditioning diagnostic, not a server feature. Each successor guard change must preserve failures and justify and verify its limits. Later bounded WAN14/H3 loaders complete the short Workbench recipes under the single current guard. Historical native-length swapping and the 21 GB WAN14 peak describe earlier loading conditions; longer/larger current-loader recipes remain unqualified.
- **Loading.** Historical loading delays range widely with memory pressure and file-cache state. Explicit direct MPS loading is an experiment with its own proof; it is not the default UI path. A resident-model speed estimate does not include text encoding, admission, repatching or VAE work.
- **Prompt enhancers.** The Qwen-Image 2.1 template's enhancer needs `qwen3.5_9b_qwen_image_2.1_pe_t2i`, which is not on this machine; leave it switched off. The Krea 2 enhancer runs on its own text encoder and works.
- **Fun-Acc 4-step LoRA for Qwen.** The official file needs a custom sampler node for exact results. The ComfyUI conversion (`Qwen-Image-2.1-Fun-Acc-4Step-comfyui`, ManualSigmas `1.0, 0.917, 0.7862, 0.5495, 0`) works but is visibly grainier than Viggle.
- **Compute dtype.** Requested FP16 does not override every quantized activation boundary. The current Krea recipe records native BF16 activation input/output with weight-only FP16 GEMM and SDPA attention. The earlier output-changing FP16-attention experiment remains deferred; its small tail-time gain is not a general model speedup.
- Details and dated measurements: [findings](findings.md).

For bounded Wan 5B quality checks, retain **704×416×33 frames** with the Turbo INT8, four-step Euler/simple, shift-8, CFG-1, full-VAE recipe above. The tested 512×288×17, 512×288×33 and 704×416×17 outputs are distorted and are diagnostic sizes only. This is an observed recipe boundary, not a universal minimum or a claim that all larger sizes fit.
