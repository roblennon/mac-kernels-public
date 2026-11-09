"""Headless Wan 2.2 TI2V-5B text-to-video through ComfyUI nodes on Apple GPUs.

Env: MODE=metal|stock, W, H, LENGTH (4k+1 frames), STEPS, CFG, SAMPLER, SHIFT, UNET.
ATTN=pytorch (default, passes --use-pytorch-cross-attention) | auto (ComfyUI picks) | split | quad.
TILED=1 decodes with VAEDecodeTiled. START_IMAGE=<png> runs image-to-video (VAE encode). NO_DECODE=1 stops after sampling. SIGMAS=<comma list> samples with explicit sigmas (e.g. FastWan DMD: SAMPLER=lcm SIGMAS=1.0,0.93968,0.84521,0). ATTN_PROBE=N checks every Nth fp16 SDPA call against fp32. COMFY_DIR points at another ComfyUI checkout. VAE_NAME picks the VAE (e.g. lighttaew2_2.safetensors for a fast draft decode). TAG appends to the output name.
UNET=<file>.gguf loads through ComfyUI-GGUF (bench/gguf_node.py; MODE=stock is its own dequantize-per-call path).
COND_CACHE=auto|off selects a source-aware conditioning cache (default auto); old prompt-only files are not reused.
Defaults fit the 4-step Turbo INT8 ConvRot build.
"""
import asyncio, gc, json, os, sys, time


def save_sample_latent(samples, path, *, overwrite=True):
    """Publish a finite CPU sampling artifact before any optional decoder."""
    import hashlib
    from pathlib import Path
    import tempfile
    import torch
    samples = samples.detach().cpu().contiguous()
    if not samples.is_floating_point() or not bool(torch.isfinite(samples).all()):
        raise ValueError('Wan sampling must produce a finite floating latent')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='.wan-latent-', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open('wb') as stream:
            torch.save(samples, stream)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(chunk)
    identity = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    current = path.stat()
    if [current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns] != identity:
        raise RuntimeError('Saved Wan latent changed while hashing')
    return {'path': str(path), 'file_identity': identity,
            'sha256': digest.hexdigest(), 'shape': list(samples.shape), 'dtype': str(samples.dtype), 'finite': True}


ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
MODE = os.environ.get("MODE", "metal")
W = int(os.environ.get("W", 704)); H = int(os.environ.get("H", 416)); LENGTH = int(os.environ.get("LENGTH", 33))
STEPS = int(os.environ.get("STEPS", 4)); CFG = float(os.environ.get("CFG", 1.0)); SHIFT = float(os.environ.get("SHIFT", 8.0))
SAMPLER = os.environ.get("SAMPLER", "euler")
UNET = os.environ.get("UNET", "wan2.2_ti2v_5b_turbo_int8_convrot.safetensors")
CODE = os.environ.get("COMFY_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ComfyUI")
GGUF = UNET.endswith(".gguf"); BENCH = os.path.dirname(os.path.abspath(__file__))  # before the chdir below
if GGUF and MODE == "stock":
    os.environ["COMFY_GGUF_MPS_KERNEL"] = "0"
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
folder_paths.add_model_folder_path("diffusion_models", f"{ROOT}/models/gguf")
folder_paths.add_model_folder_path("vae_approx", f"{ROOT}/models/wan/vae")  # tiny VAEs (lighttaew2_*, taehv) load from vae_approx
VAE_NAME = os.environ.get("VAE_NAME", "wan2.2_vae.safetensors")
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
    lap("load_vae")
    start = None
    if os.environ.get("START_IMAGE"):  # image-to-video: exercises the VAE encode path
        if VAE_NAME.startswith(('lighttae', 'taew', 'taehv')):
            raise ValueError('Wan image conditioning requires the full VAE; tiny VAEs are decoder-only drafts')
        from PIL import Image as _I; import numpy as _np
        start = torch.from_numpy(_np.asarray(_I.open(os.environ["START_IMAGE"]).convert("RGB"), dtype=_np.float32) / 255.0)[None]
    latent = call("Wan22ImageToVideoLatent", vae=vae, width=W, height=H, length=LENGTH, batch_size=1, **({"start_image": start} if start is not None else {}))[0]; lap("latent")
    if GGUF:
        sys.path.insert(0, BENCH); import gguf_node
        model = gguf_node.load_unet(UNET)
    else:
        model = N["UNETLoader"]().load_unet(UNET, "default")[0]
    model = call("ModelSamplingSD3", model=model, shift=SHIFT)[0]; lap("load_unet")
    PROBE = {"n": 0, "checked": 0, "bit_equal": 0, "max_abs": 0.0, "shapes": set()}
    if os.environ.get("ATTN_PROBE"):  # every Nth SDPA call: compare fp16 output with the fp32-upcast output on the real tensors
        import comfy.ops as _ops
        _real_sdpa, _every = _ops.scaled_dot_product_attention, int(os.environ["ATTN_PROBE"])
        def _probe(q, k, v, *a, **kw):
            out = _real_sdpa(q, k, v, *a, **kw); PROBE["n"] += 1
            if q.dtype == torch.float16 and PROBE["n"] % _every == 1:
                ref = _real_sdpa(q.float(), k.float(), v.float(), *a, **kw).half()
                PROBE["checked"] += 1; PROBE["bit_equal"] += int(torch.equal(out, ref))
                PROBE["max_abs"] = max(PROBE["max_abs"], float((out.float() - ref.float()).abs().max())); PROBE["shapes"].add(f"{tuple(q.shape)}x{k.shape[-2]}")
            return out
        _ops.scaled_dot_product_attention = _probe
    step_t = []
    orig = comfy.utils.ProgressBar.update_absolute
    def upd(self, *a, **k):
        torch.mps.synchronize(); step_t.append(time.perf_counter()); return orig(self, *a, **k)
    comfy.utils.ProgressBar.update_absolute = upd
    if os.environ.get("SIGMAS"):  # explicit sigmas through SamplerCustomAdvanced, same seed-7 noise
        sig = torch.tensor([float(x) for x in os.environ["SIGMAS"].split(",")])
        guider = call("CFGGuider", model=model, positive=pos, negative=neg, cfg=CFG)[0] if CFG != 1.0 else call("BasicGuider", model=model, conditioning=pos)[0]
        out = call("SamplerCustomAdvanced", noise=call("RandomNoise", noise_seed=7)[0], guider=guider, sampler=call("KSamplerSelect", sampler_name=SAMPLER)[0],
                   sigmas=sig, latent_image=latent)[0]; lap("sample")
    else:
        out = N["KSampler"]().sample(model, 7, STEPS, CFG, SAMPLER, "simple", pos, neg, latent, 1.0)[0]; lap("sample")
    peak = torch.mps.driver_allocated_memory() / 2**30; peak_sample = PEAK["gb"]; PEAK["gb"] = 0.0
    s = out["samples"]; print("latent", tuple(s.shape), "nan", int(torch.isnan(s).sum()), flush=True)
    if os.environ.get("ATTN_PROBE"):
        print("ATTN_PROBE " + json.dumps({**PROBE, "shapes": sorted(PROBE["shapes"])}), flush=True)
    tag = f"wan5b-{MODE}-{'gguf-' + os.path.splitext(UNET)[0] + '-' if GGUF else ''}{W}x{H}-{LENGTH}f-{STEPS}s-{SAMPLER}{'-vaefp16' if os.environ.get('VAE_FP16') == '1' else ''}{'' if ATTN == 'pytorch' else '-attn' + ATTN}{'-tiled' if TILED else ''}{'-i2v' if start is not None else ''}{'' if VAE_NAME == 'wan2.2_vae.safetensors' else '-' + os.path.splitext(VAE_NAME)[0]}{EXTRA_TAG}"
    sampling_only = os.environ.get("NO_DECODE") == "1"
    if sampling_only:
        import uuid
        latent_tag = tag + '-nodecode-' + uuid.uuid4().hex[:12]
    else:
        latent_tag = tag
    latent_receipt = save_sample_latent(s, f"{ROOT}/out/latents/{latent_tag}.pt", overwrite=not sampling_only)
    lap("save_latent")
    if sampling_only:  # sampling-only measurement
        steps = [round(b - a, 2) for a, b in zip(step_t, step_t[1:])]
        print("RESULT " + json.dumps(dict(tag=f"wan5b-{W}x{H}-{LENGTH}f-{SAMPLER}-attn{ATTN}-nodecode", times=T, step_s=steps, peak_gb_sample=round(peak_sample, 2),
              swap_gb_start=SWAP0, swap_gb_end=swap_gb(), attn=ATTN, latent_nan=bool(torch.isnan(s).any()), latent_source=latent_receipt))); sys.exit(0)
    del model; mm.unload_all_models(); mm.soft_empty_cache(); gc.collect()
    if TILED:
        images = N["VAEDecodeTiled"]().decode(vae, out, int(os.environ.get("TILE", 512)), 64, 64, 8)[0]; lap("decode")
    else:
        images = N["VAEDecode"]().decode(vae, out)[0]; lap("decode")
    peak_decode = PEAK["gb"]
    if images.ndim == 5: images = images.reshape(-1, *images.shape[-3:])
    from PIL import Image; import numpy as np
    fr = (images.float().nan_to_num(0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(np.concatenate([fr[i] for i in np.linspace(0, fr.shape[0] - 1, 4).astype(int)], axis=1)).save(f"{ROOT}/out/{tag}-sheet.png")
    try:
        video = call("CreateVideo", images=images, fps=24.0)[0]; video.save_to(f"{ROOT}/out/{tag}.mp4")
    except Exception as e: print("video save failed:", repr(e)[:300], flush=True)
steps = [round(b - a, 2) for a, b in zip(step_t, step_t[1:])]
# frame-to-frame sanity: corruption on MPS shows up as later frames diverging wildly from frame 0 statistics
fstats = [(round(float(f.mean()), 3), round(float(f.std()), 3)) for f in images[:: max(1, images.shape[0] // 6)].float()]
res = dict(tag=tag, unet=UNET, times=T, step_s=steps, latent_source=latent_receipt, mps_alloc_gb_after_sample=round(peak, 2), peak_gb_sample=round(peak_sample, 2), peak_gb_decode=round(peak_decode, 2),
           swap_gb_start=SWAP0, swap_gb_end=swap_gb(), attn=ATTN, sampler=SAMPLER, cfg=CFG, comfy=CODE, gguf=gguf_node.counters() if GGUF else None, nan=bool(torch.isnan(images).any()),
           frames=int(images.shape[0]), frame_stats=fstats)
print("RESULT " + json.dumps(res)); open(f"{ROOT}/out/results.jsonl", "a").write(json.dumps(res) + "\n")
