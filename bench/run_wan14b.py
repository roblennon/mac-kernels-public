"""Headless Wan 2.2 T2V-A14B (high-noise + low-noise experts, 4-step lightx2v LoRAs) text-to-video through ComfyUI nodes on Apple GPUs.

Env: MODE=metal|stock, W, H, LENGTH (4k+1 frames), STEPS, CFG, SAMPLER, SHIFT, UNET.
ATTN=pytorch (default, passes --use-pytorch-cross-attention) | auto (ComfyUI picks) | split | quad.
TILED=1 decodes with VAEDecodeTiled. COMFY_DIR points at another ComfyUI checkout. VAE_NAME picks the VAE (e.g. lighttaew2_2.safetensors for a fast draft decode). TAG appends to the output name.
COND_CACHE=auto|off selects a source-aware conditioning cache (default auto); old prompt-only files are not reused.
Defaults fit the 4-step Turbo INT8 ConvRot build.
"""
import asyncio, gc, json, os, sys, time
ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
MODE = os.environ.get("MODE", "metal")
W = int(os.environ.get("W", 640)); H = int(os.environ.get("H", 384)); LENGTH = int(os.environ.get("LENGTH", 33))
STEPS = int(os.environ.get("STEPS", 4)); CFG = float(os.environ.get("CFG", 1.0)); SHIFT = float(os.environ.get("SHIFT", 5.0)); SPLIT = int(os.environ.get("SPLIT", 2)); LORA = os.environ.get("LORA", "1") == "1"
SAMPLER = os.environ.get("SAMPLER", "euler")
HIGH = os.environ.get("HIGH", "wan2.2_t2v_high_noise_14B_int8_convrot.safetensors"); LOW = os.environ.get("LOW", "wan2.2_t2v_low_noise_14B_int8_convrot.safetensors")
CODE = os.environ.get("COMFY_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ComfyUI")
ATTN = os.environ.get("ATTN", "pytorch"); TILED = os.environ.get("TILED") == "1"; EXTRA_TAG = os.environ.get("TAG", "")
ATTN_FLAG = {"pytorch": ["--use-pytorch-cross-attention"], "auto": [], "split": ["--use-split-cross-attention"], "quad": ["--use-quad-cross-attention"]}[ATTN]
sys.path.insert(0, CODE); os.chdir(CODE)
sys.argv = [sys.argv[0]] + ATTN_FLAG + (["--fp16-vae"] if os.environ.get("VAE_FP16") == "1" else [])
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths, threading, subprocess
PEAK = {"gb": 0.0}
def _poll():
    while True:
        PEAK["gb"] = max(PEAK["gb"], torch.mps.driver_allocated_memory() / 2**30); time.sleep(0.25)
threading.Thread(target=_poll, daemon=True).start()
def swap_gb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return round(float(out.split("used = ")[1].split("M")[0]) / 1024, 2)
SWAP0 = swap_gb()
for k in ("diffusion_models", "text_encoders", "vae", "loras"):
    folder_paths.add_model_folder_path(k, f"{ROOT}/models/wan/{k}")
folder_paths.add_model_folder_path("vae_approx", f"{ROOT}/models/wan/vae")  # tiny VAEs (lighttaew2_*, taehv) load from vae_approx
VAE_NAME = os.environ.get("VAE_NAME", "wan_2.1_vae.safetensors")
import comfy.model_management as mm
if MODE == "stock":
    mm.mps_quant_formats = lambda device=None: set()
import comfy.utils, nodes
asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
N = nodes.NODE_CLASS_MAPPINGS
PROMPT = os.environ.get("PROMPT", "A street musician plays guitar in a retro 1970s subway station, dim warm light, rough textures, commuters passing, handheld camera, film grain.")
NEG = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
T = {}; t0 = time.perf_counter()
def lap(name):
    global t0
    torch.mps.synchronize(); T[name] = round(time.perf_counter() - t0, 2); t0 = time.perf_counter(); print("LAP", name, T[name], flush=True)
def call(name, **kw):
    cls = N[name]
    if hasattr(cls, "define_schema"):
        out = cls.execute(**kw); return out.result if hasattr(out, "result") else out.args
    return getattr(cls(), cls.FUNCTION)(**kw)
with torch.inference_mode():
    import conditioning_cache
    cache_policy = os.environ.get("COND_CACHE", "auto")
    cache_identity = (conditioning_cache.wan_context(PROMPT, NEG, CODE, folder_paths, torch, mm, MODE, sys.argv[1:])
                      if cache_policy != "off" else None)
    cpath = conditioning_cache.path_for(f"{ROOT}/out/cache", cache_identity, cache_policy)
    cached = conditioning_cache.load(cpath, cache_identity)
    if cached is not None:
        pos, neg = cached; lap("load_cached_cond")
    else:
        clip = N["CLIPLoader"]().load_clip("umt5_xxl_fp16.safetensors", "wan", "default")[0]; lap("load_te")
        pos = N["CLIPTextEncode"]().encode(clip, PROMPT)[0]; neg = N["CLIPTextEncode"]().encode(clip, NEG)[0]; lap("encode")
        if cpath is not None and conditioning_cache.wan_context(PROMPT, NEG, CODE, folder_paths, torch, mm, MODE, sys.argv[1:]) != cache_identity:
            raise RuntimeError("Encoder identity changed while encoding; conditioning was not cached")
        conditioning_cache.save(cpath, cache_identity, pos, neg)
        del clip; mm.unload_all_models(); mm.soft_empty_cache(); gc.collect()
    vae = N["VAELoader"]().load_vae(VAE_NAME)[0]
    latent = call("EmptyHunyuanLatentVideo", width=W, height=H, length=LENGTH, batch_size=1)[0]
    step_t = []
    orig = comfy.utils.ProgressBar.update_absolute
    def upd(self, *a, **k):
        torch.mps.synchronize(); step_t.append(time.perf_counter()); return orig(self, *a, **k)
    comfy.utils.ProgressBar.update_absolute = upd
    def expert(name, lora):
        m = N["UNETLoader"]().load_unet(name, "default")[0]
        if LORA:
            m = N["LoraLoaderModelOnly"]().load_lora_model_only(m, lora, 1.0)[0]
        return call("ModelSamplingSD3", model=m, shift=SHIFT)[0]
    ks = N["KSamplerAdvanced"]()
    model = expert(HIGH, "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"); lap("load_high")
    out = ks.sample(model, "enable", 7, STEPS, CFG, SAMPLER, "simple", pos, neg, latent, 0, SPLIT, "enable")[0]; lap("sample_high")
    del model; mm.unload_all_models(); mm.soft_empty_cache(); gc.collect()
    model = expert(LOW, "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"); lap("load_low")
    out = ks.sample(model, "disable", 0, STEPS, CFG, SAMPLER, "simple", pos, neg, out, SPLIT, STEPS, "disable")[0]; lap("sample_low")
    peak = torch.mps.driver_allocated_memory() / 2**30; peak_sample = PEAK["gb"]; PEAK["gb"] = 0.0
    s = out["samples"]; print("latent", tuple(s.shape), "nan", int(torch.isnan(s).sum()), flush=True)
    del model; mm.unload_all_models(); mm.soft_empty_cache(); gc.collect()
    os.makedirs(f"{ROOT}/out/latents", exist_ok=True)
    if TILED:
        images = N["VAEDecodeTiled"]().decode(vae, out, int(os.environ.get("TILE", 512)), 64, 64, 8)[0]; lap("decode")
    else:
        images = N["VAEDecode"]().decode(vae, out)[0]; lap("decode")
    peak_decode = PEAK["gb"]
    if images.ndim == 5: images = images.reshape(-1, *images.shape[-3:])
    tag = f"wan14b-{MODE}-{W}x{H}-{LENGTH}f-{STEPS}s-{SAMPLER}{'-vaefp16' if os.environ.get('VAE_FP16') == '1' else ''}{'' if ATTN == 'pytorch' else '-attn' + ATTN}{'-tiled' if TILED else ''}{'' if VAE_NAME == 'wan_2.1_vae.safetensors' else '-' + os.path.splitext(VAE_NAME)[0]}{EXTRA_TAG}"
    torch.save(s.cpu(), f"{ROOT}/out/latents/{tag}.pt")
    from PIL import Image; import numpy as np
    fr = (images.float().nan_to_num(0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(np.concatenate([fr[i] for i in np.linspace(0, fr.shape[0] - 1, 4).astype(int)], axis=1)).save(f"{ROOT}/out/{tag}-sheet.png")
    try:
        video = call("CreateVideo", images=images, fps=24.0)[0]; video.save_to(f"{ROOT}/out/{tag}.mp4")
    except Exception as e: print("video save failed:", repr(e)[:300], flush=True)
steps = [round(b - a, 2) for a, b in zip(step_t, step_t[1:])]
# frame-to-frame sanity: corruption on MPS shows up as later frames diverging wildly from frame 0 statistics
fstats = [(round(float(f.mean()), 3), round(float(f.std()), 3)) for f in images[:: max(1, images.shape[0] // 6)].float()]
res = dict(tag=tag, unet=HIGH, times=T, step_s=steps, mps_alloc_gb_after_sample=round(peak, 2), peak_gb_sample=round(peak_sample, 2), peak_gb_decode=round(peak_decode, 2),
           swap_gb_start=SWAP0, swap_gb_end=swap_gb(), attn=ATTN, sampler=SAMPLER, cfg=CFG, comfy=CODE, nan=bool(torch.isnan(images).any()),
           frames=int(images.shape[0]), frame_stats=fstats)
print("RESULT " + json.dumps(res)); open(f"{ROOT}/out/results.jsonl", "a").write(json.dumps(res) + "\n")
