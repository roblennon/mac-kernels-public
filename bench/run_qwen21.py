"""Headless Qwen-Image 2.1 text-to-image through ComfyUI nodes on Apple GPUs.

Env: MODE=metal|stock, W, H, STEPS, UNET, TE, CACHE=auto|off (QwenImage21Cache prefix cache).
UNET=<file>.gguf loads through ComfyUI-GGUF (bench/gguf_node.py; MODE=stock is its own dequantize-per-call path).
COMFY_DIR points at another ComfyUI checkout.
LORA=<file in loras/> (LORA_STRENGTH), CFG (default 1.0), SIGMAS=viggle|funacc|<comma list> runs SamplerCustomAdvanced
with explicit sigmas instead of KSampler's simple schedule (same seed-42 noise). viggle: Viggle's raw 6-step nodes
through the resolution shift (their ViggleTurboSigmas node); funacc: the trained PDD grid of the Fun-Acc 4-step LoRA.
"""
import asyncio, gc, json, os, sys, time
ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
MODE = os.environ.get("MODE", "metal")
W = int(os.environ.get("W", 1024)); H = int(os.environ.get("H", 1024)); STEPS = int(os.environ.get("STEPS", 25))
UNET = os.environ.get("UNET", "qwen_image_2.1_int8_convrot.safetensors")
TE = os.environ.get("TE", "qwen3vl_8b_int8_convrot.safetensors"); CACHE = os.environ.get("CACHE", "auto")
LORA = os.environ.get("LORA", ""); LORA_STRENGTH = float(os.environ.get("LORA_STRENGTH", 1.0)); CFG = float(os.environ.get("CFG", 1.0))
SIGMAS = os.environ.get("SIGMAS", "")
BENCH = os.path.dirname(os.path.abspath(__file__))  # resolved before the chdir below
CODE = os.environ.get("COMFY_DIR") or os.path.join(os.path.dirname(BENCH), "vendor", "ComfyUI")
GGUF = UNET.endswith(".gguf")
if GGUF and MODE == "stock":
    os.environ["COMFY_GGUF_MPS_KERNEL"] = "0"
sys.path.insert(0, CODE); os.chdir(CODE)
sys.argv = [sys.argv[0], "--use-pytorch-cross-attention"] + (["--fp16-vae"] if os.environ.get("VAE_FP16") == "1" else [])
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths
for k in ("diffusion_models", "text_encoders", "vae", "loras"):
    folder_paths.add_model_folder_path(k, f"{ROOT}/models/qwen21/{k}")
folder_paths.add_model_folder_path("diffusion_models", f"{ROOT}/models/gguf")
import comfy.model_management as mm
if MODE == "stock":
    mm.mps_quant_formats = lambda device=None: set()
import comfy.utils, nodes
asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
N = nodes.NODE_CLASS_MAPPINGS
PROMPT = os.environ.get("PROMPT", "Greyscale fashion editorial portrait of an avant-garde woman in profile, tilted upwards, striking high-contrast chiaroscuro lighting, sculptural collar, fine film grain, studio backdrop.")
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
    clip = N["CLIPLoader"]().load_clip(TE, "qwen_image", "default")[0]; lap("load_te")
    pos, neg, _ = call("TextEncodeQwenImage21", clip=clip, prompt=PROMPT, negative_prompt="", resolution=max(W, H))
    lap("encode")
    for c in pos: print("cond", tuple(c[0].shape), c[0].dtype, "absmax", float(c[0].float().abs().max()), "nan", bool(torch.isnan(c[0]).any()), flush=True)
    del clip; mm.unload_all_models(); mm.soft_empty_cache(); gc.collect()
    if GGUF:
        sys.path.insert(0, BENCH); import gguf_node
        model = gguf_node.load_unet(UNET)
    else:
        model = N["UNETLoader"]().load_unet(UNET, "default")[0]
    if LORA:
        model = N["LoraLoaderModelOnly"]().load_lora_model_only(model, LORA, LORA_STRENGTH)[0]
    model = call("QwenImage21Cache", model=model, device=CACHE, dtype="default")[0]
    vae = N["VAELoader"]().load_vae("qwen_image_2.1_vae_bf16.safetensors")[0]
    latent = N["EmptyLatentImage"]().generate(W, H, 1)[0]; lap("load_unet")
    step_t = []
    orig = comfy.utils.ProgressBar.update_absolute
    def upd(self, *a, **k):
        torch.mps.synchronize(); step_t.append(time.perf_counter()); return orig(self, *a, **k)
    comfy.utils.ProgressBar.update_absolute = upd
    if SIGMAS:
        if SIGMAS == "viggle":  # Viggle's ViggleTurboSigmas: raw nodes through the dynamic exponential shift, final 0
            sys.path.insert(0, BENCH); from qwen_comfy import viggle_sigmas
            sig = viggle_sigmas(W, H)
        elif SIGMAS == "funacc":  # pdd_sigmas from the Fun-Acc 4-step LoRA config (already shifted, 2048-trained)
            sig = [1.0, 0.9169867038726807, 0.7861579060554504, 0.5494909882545471, 0.0]
        else:
            sig = [float(x) for x in SIGMAS.split(",")]
        print("sigmas", [round(x, 4) for x in sig], flush=True)
        guider = call("CFGGuider", model=model, positive=pos, negative=neg, cfg=CFG)[0] if CFG != 1.0 else call("BasicGuider", model=model, conditioning=pos)[0]
        out = call("SamplerCustomAdvanced", noise=call("RandomNoise", noise_seed=42)[0], guider=guider, sampler=call("KSamplerSelect", sampler_name="euler")[0],
                   sigmas=torch.tensor(sig), latent_image=latent)[0]; lap("sample")
    else:
        out = N["KSampler"]().sample(model, 42, STEPS, CFG, "euler", "simple", pos, neg, latent, 1.0)[0]; lap("sample")
    lora_audit = None
    if LORA:
        sys.path.insert(0, BENCH); import lora_audit as _la
        try:
            lora_audit = gguf_node.lora_audit(model) if GGUF else _la.audit(model)
        except Exception as e:  # an audit bug must not lose the run
            lora_audit = {"error": repr(e)[:200]}
        print("LORA_AUDIT " + json.dumps(lora_audit), flush=True)
    peak = torch.mps.driver_allocated_memory() / 2**30
    img = N["VAEDecode"]().decode(vae, out)[0]; lap("decode")
steps = [round(b - a, 2) for a, b in zip(step_t, step_t[1:])]
DEFAULT_UNET = "qwen_image_2.1_int8_convrot.safetensors"
tag = f"qwen21-{MODE}-{'gguf-' + os.path.splitext(UNET)[0] + '-' if GGUF else os.path.splitext(UNET)[0] + '-' if UNET != DEFAULT_UNET else ''}{W}x{H}-{STEPS}s{'-nocache' if CACHE == 'off' else ''}{'-' + os.path.splitext(LORA)[0][:40] if LORA else ''}{'-sig' + SIGMAS[:12] if SIGMAS else ''}{'-cfg%g' % CFG if CFG != 1.0 else ''}"
from PIL import Image
arr = (img[0].float().nan_to_num(0).cpu().numpy() * 255).clip(0, 255).astype("uint8"); Image.fromarray(arr[..., :3]).save(f"{ROOT}/out/{tag}.png")
res = dict(tag=tag, unet=UNET, te=TE, lora=LORA, lora_audit=lora_audit, cfg=CFG, sigmas=SIGMAS, times=T, step_s_median=sorted(steps)[len(steps) // 2] if steps else None, step_s=steps[:6],
           mps_alloc_gb_after_sample=round(peak, 2), gguf=gguf_node.counters() if GGUF else None, comfy=CODE, nan=bool(torch.isnan(img).any()), mean=float(img.float().nan_to_num(0).mean()), std=float(img.float().nan_to_num(0).std()))
print("RESULT " + json.dumps(res)); open(f"{ROOT}/out/results.jsonl", "a").write(json.dumps(res) + "\n")
