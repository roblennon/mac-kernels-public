"""Headless MiniMax H3 text-to-video(+audio) through ComfyUI nodes on Apple GPUs.

Env: MODE=metal|stock, W,H (multiples of 32), LENGTH (17k+5 frames), STEPS, LORA=1|0,
UNET, TE. Text encoding runs first and the encoder is released before the DiT loads.
A polling thread records peak GPU memory and peak swap per phase (encode, sample, decode); the tag carries the
weight format (w6a8, w4a8, int8) so runs at the same size do not overwrite each other. COMFY_DIR as in the Wan runners.
Text conditioning is cached per prompt and reused at every size (FORCE_ENCODE=1 re-encodes and checks it matches).
"""
import asyncio, json, os, sys, time
ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
MODE = os.environ.get("MODE", "metal")
W = int(os.environ.get("W", 480)); H = int(os.environ.get("H", 288)); LENGTH = int(os.environ.get("LENGTH", 22))
STEPS = int(os.environ.get("STEPS", 4)); LORA = os.environ.get("LORA", "1") == "1"
UNET = os.environ.get("UNET", "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
TE = os.environ.get("TE", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
CODE = os.environ.get("COMFY_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ComfyUI")
sys.path.insert(0, CODE); os.chdir(CODE)
sys.argv = [sys.argv[0], "--use-pytorch-cross-attention"] + (["--fp16-vae"] if os.environ.get("VAE_FP16") == "1" else [])
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths, threading, subprocess
PHASE = {"name": "start"}; PEAKS = {}
def _swap_gb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return round(float(out.split("used = ")[1].split("M")[0]) / 1024, 2)
SWAP0 = _swap_gb()
def _poll():
    i = 0
    while True:
        p = PEAKS.setdefault(PHASE["name"], {"gpu_gb": 0.0, "swap_gb": 0.0})
        p["gpu_gb"] = max(p["gpu_gb"], round(torch.mps.driver_allocated_memory() / 2**30, 2))
        if i % 8 == 0:
            p["swap_gb"] = max(p["swap_gb"], _swap_gb())
        i += 1; time.sleep(0.25)
threading.Thread(target=_poll, daemon=True).start()
for k in ("diffusion_models", "text_encoders", "vae", "loras"):
    folder_paths.add_model_folder_path(k, f"{ROOT}/models/h3/{k}")
import comfy.model_management as mm
if MODE == "stock":
    mm.mps_quant_formats = lambda device=None: set()
import comfy.utils, nodes
asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
N = nodes.NODE_CLASS_MAPPINGS
PROMPT = os.environ.get("PROMPT", "Realistic live-action cinematic shot: a golden retriever runs along a beach at sunset toward the camera, waves rolling in, shallow depth of field, film grain. Audio: waves, wind, paws on wet sand.")
T = {}; t0 = time.perf_counter()
def lap(name):
    global t0
    torch.mps.synchronize(); T[name] = round(time.perf_counter() - t0, 2); t0 = time.perf_counter(); print("LAP", name, T[name], flush=True)
def call(name, **kw):
    cls = N[name]
    if hasattr(cls, "execute") and not hasattr(cls, "FUNCTION_LEGACY") and hasattr(cls, "define_schema"):
        out = cls.execute(**kw); return out.result if hasattr(out, "result") else out.args
    return getattr(cls(), cls.FUNCTION)(**kw)
with torch.inference_mode():
    import gc, hashlib
    os.makedirs(f"{ROOT}/out/cache", exist_ok=True)
    ckey = hashlib.sha1(f"{TE}|{PROMPT}|{W}|{H}|{LENGTH}".encode()).hexdigest()[:12]
    cpath = f"{ROOT}/out/cache/h3-cond-{ckey}.pt"
    PHASE["name"] = "encode"
    # Text-to-video conditioning depends only on the prompt (no keyframes): reuse it at any size and build this size's
    # empty latent as MiniMaxH3ImageToVideo does. FORCE_ENCODE=1 re-encodes and checks it matches the cached one.
    tpath = f"{ROOT}/out/cache/h3-textcond-{hashlib.sha1(f'{TE}|{PROMPT}'.encode()).hexdigest()[:12]}.pt"
    if not os.path.exists(tpath) and os.path.exists(cpath):
        torch.save(torch.load(cpath, weights_only=False)[0], tpath)
    if os.path.exists(cpath) and os.environ.get("FORCE_ENCODE") != "1":
        pos, latent = torch.load(cpath, weights_only=False); lap("load_cached_cond")
    elif os.path.exists(tpath) and os.environ.get("FORCE_ENCODE") != "1":
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent
        pos = torch.load(tpath, weights_only=False); latent = _empty_av_latent(W, H, LENGTH)[0]; lap("load_cached_text_cond")
    else:
        clip = N["CLIPLoader"]().load_clip(TE, "minimax", "default")[0]; lap("load_te")
        pos, latent = call("MiniMaxH3ImageToVideo", clip=clip, vae=None, prompt=PROMPT, width=W, height=H, length=LENGTH)
        lap("encode")
        if os.path.exists(tpath):
            prev = torch.load(tpath, weights_only=False)
            print("TEXT_COND_EQUAL_TO_CACHED", all(torch.equal(a[0], b[0]) for a, b in zip(pos, prev)), flush=True)
        try: torch.save((pos, latent), cpath); torch.save(pos, tpath)
        except Exception as e: print("cond cache failed", repr(e)[:200])
        del clip; mm.unload_all_models(); mm.soft_empty_cache(); gc.collect(); mm.soft_empty_cache()
    PHASE["name"] = "sample"
    model = N["UNETLoader"]().load_unet(UNET, "default")[0]
    if LORA:
        model = N["LoraLoaderModelOnly"]().load_lora_model_only(model, "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", 1.0)[0]
    lap("load_unet")
    step_t = []
    orig = comfy.utils.ProgressBar.update_absolute
    def upd(self, *a, **k):
        torch.mps.synchronize(); step_t.append(time.perf_counter()); return orig(self, *a, **k)
    comfy.utils.ProgressBar.update_absolute = upd
    sampler = call("KSamplerSelect", sampler_name=os.environ.get("SAMPLER", "res_multistep"))[0]
    sigmas = call("BasicScheduler", model=model, scheduler="simple", steps=STEPS, denoise=1.0)[0]
    guider = call("BasicGuider", model=model, conditioning=pos)[0]
    noise = call("RandomNoise", noise_seed=1)[0]
    out = call("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent)[0]
    lap("sample")
    peak = torch.mps.driver_allocated_memory() / 2**30
    qtag = next((q for q in ("w4a8", "w6a8", "int8") if q in UNET), "")
    tag = f"h3-{MODE}-{W}x{H}-{LENGTH}f-{STEPS}s{'' if LORA else '-nolora'}{'-' + qtag if qtag and qtag != 'int8' else ''}"
    try: torch.save(out, f"{ROOT}/out/cache/{tag}-latent.pt")
    except Exception as e: print("latent save failed", repr(e)[:200])
    del model, guider; mm.unload_all_models(); mm.soft_empty_cache(); gc.collect()
    PHASE["name"] = "decode"
    vae = N["VAELoader"]().load_vae("minimax_h3_video_vae_fp16.safetensors")[0]
    avae = N["VAELoader"]().load_vae("minimax_h3_audio_vae_fp32.safetensors")[0]
    images = N["VAEDecode"]().decode(vae, out)[0]; lap("decode_video")
    audio = call("VAEDecodeAudio", samples=out, vae=avae)[0]; lap("decode_audio")
    print("frames", tuple(images.shape), images.dtype, "nan", bool(torch.isnan(images).any()), "audio", {k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in audio.items()}, flush=True)
    from PIL import Image
    fr = (images.float().nan_to_num(0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    os.makedirs(f"{ROOT}/out/{tag}", exist_ok=True)
    for i in range(0, fr.shape[0], max(1, fr.shape[0] // 6)): Image.fromarray(fr[i]).save(f"{ROOT}/out/{tag}/f{i:03d}.png")
    Image.fromarray(__import__("numpy").concatenate([fr[i] for i in range(0, fr.shape[0], max(1, fr.shape[0] // 4))][:4], axis=1)).save(f"{ROOT}/out/{tag}-sheet.png")
    try:
        video = call("CreateVideo", images=images, fps=24.0, audio=audio)[0]
        video.save_to(f"{ROOT}/out/{tag}.mp4")
    except Exception as e:
        print("video save with audio failed:", repr(e)[:300], flush=True)
        try:
            video = call("CreateVideo", images=images, fps=24.0)[0]; video.save_to(f"{ROOT}/out/{tag}-noaudio.mp4")
        except Exception as e2: print("video save failed:", repr(e2)[:300], flush=True)
steps = [round(b - a, 2) for a, b in zip(step_t, step_t[1:])]
fstats = [(round(float(f.mean()), 3), round(float(f.std()), 3)) for f in images[:: max(1, images.shape[0] // 6)].float()]
res = dict(tag=tag, unet=UNET, lora=LORA, size=[W, H, LENGTH], times=T, step_s=steps, mps_alloc_gb_after_sample=round(peak, 2),
           peaks=PEAKS, swap_gb_start=SWAP0, frame_stats=fstats,
           nan=bool(torch.isnan(images).any()), mean=float(images.mean()), std=float(images.std()), frames=int(images.shape[0]))
print("RESULT " + json.dumps(res)); open(f"{ROOT}/out/results.jsonl", "a").write(json.dumps(res) + "\n")
