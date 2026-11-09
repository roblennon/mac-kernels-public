"""Headless Krea 2 Turbo run through ComfyUI nodes, for timing on Apple GPUs.

MODE=stock   ComfyUI as shipped (INT8 compute disabled on MPS -> weights dequantized)
MODE=metal   route INT8 layers to the comfy-kitchen mps backend (weight-only fp16)
FP16=1       ask ComfyUI to run the diffusion model in fp16 instead of bf16
LORA=1       apply loras/krea2_darkbrush.safetensors (LORA_NAME, LORA_STRENGTH override); the tag
             records whether ComfyUI merged it (COMFY_MPS_LORA_SIDE_PATH=0) or kept it on the side
UNET, PROMPT, SEED, CFG, TAG  optional overrides (defaults: Turbo INT8, the soccer prompt, 42, 1.0, none)
DENOISE_TIMING=1  record synchronized denoiser call times after model preparation
COND_CACHE=auto  opt-in source-aware CPU text-conditioning reuse (default off)
COND_STAGE=capture  save conditioning and exit before diffusion/VAE; requires COND_CACHE=auto
SPECTRAL=1   opt-in eight-step Krea spectral experiment (pinned helper; not qualified for quality/speed)
KREA_READONLY=1  experimental static native INT8 shared mapping and resident-weight cap
KREA_PLAIN_LORA=1  with KREA_ADAPTER_READONLY, admit one plain model LoRA at eight steps/CFG 1
KREA_ADAPTER_READONLY=1  separately admitted immutable base for the pinned four-step rank-64 LoRA
KREA_RESIDENT_GIB  weight budget (0 < budget <= 4; default 1), requires KREA_READONLY
             excludes LoRA; 0.5 MPS allocator fraction for diffusion/VAE after text encoding
             checkpoint must remain immutable on disk and in memory
             qualify each recipe under gpu_runner; recorded tests do not guarantee other profiles
COMFY_DIR    another ComfyUI checkout (default vendor/ComfyUI); PYTHONPATH picks the comfy-kitchen checkout
"""
import hashlib, json, os, sys, time
from contextlib import nullcontext
from pathlib import Path
ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
MODE = os.environ.get("MODE", "stock"); FP16 = os.environ.get("FP16") == "1"
W = int(os.environ.get("W", 1024)); H = int(os.environ.get("H", 1024)); STEPS = int(os.environ.get("STEPS", 8))
TE = os.environ.get("TE", "qwen3vl_4b_bf16.safetensors")
LORA = os.environ.get("LORA") == "1"
READONLY = os.environ.get("KREA_READONLY") == "1"
ADAPTER_READONLY = os.environ.get("KREA_ADAPTER_READONLY") == "1"
if os.environ.get("KREA_PLAIN_LORA") == "1" and not ADAPTER_READONLY:
    raise ValueError("KREA_PLAIN_LORA requires KREA_ADAPTER_READONLY")
MAPPED = READONLY or ADAPTER_READONLY
LORA_NAME = os.environ.get("LORA_NAME", "krea2_darkbrush.safetensors"); LORA_STRENGTH = float(os.environ.get("LORA_STRENGTH", 1.0))
CODE = os.environ.get("COMFY_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ComfyUI")
sys.path.insert(0, CODE); os.chdir(CODE)
ATTN = os.environ.get("ATTN", "")  # "pytorch" selects SDPA (the fork default on MPS with torch >= 2.13); "quad" selects
# upstream ComfyUI's sub-quadratic default on Apple GPUs, for stock-versus-fork attention comparisons
if ATTN not in ("", "pytorch", "quad"):
    raise ValueError("ATTN must be empty, pytorch or quad")
ATTN_ARGS = {"pytorch": ["--use-pytorch-cross-attention"], "quad": ["--use-quad-cross-attention"]}.get(ATTN, [])
sys.argv = [sys.argv[0]] + (["--fp16-unet"] if FP16 else []) + ATTN_ARGS + (["--fp16-vae"] if os.environ.get("VAE_FP16") == "1" else [])
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths
for k in ("diffusion_models", "text_encoders", "vae", "loras"):
    folder_paths.add_model_folder_path(k, f"{ROOT}/models/{k}")
import comfy.model_management as mm
if MODE == "stock":
    mm.mps_quant_formats = lambda device=None: set()
if MODE == "metal":
    pass  # vendor/ComfyUI branch mps-int8-weight-only enables the Metal formats
import comfy.utils, nodes
N = nodes.NODE_CLASS_MAPPINGS
PROMPT = os.environ.get("PROMPT") or "Ultra low angle action shot, male soccer player jumping mid-air high kick, clear vivid blue sky background, sharp focused upper body, maroon jersey and white shorts, harsh direct sunlight, high contrast sports photography"
UNET = os.environ.get("UNET", "krea2_turbo_int8_convrot.safetensors"); SEED = int(os.environ.get("SEED", 42)); CFG = float(os.environ.get("CFG", 1.0))
readonly_audit = {}; residency_audit = {}
resident_budget_bytes = None
if "KREA_RESIDENT_GIB" in os.environ and not MAPPED:
    raise ValueError("KREA_RESIDENT_GIB requires a readonly Krea route")
if READONLY:
    if (MODE != "metal" or LORA or sys.platform != "darwin"
            or not torch.backends.mps.is_available()
            or "int8_tensorwise" not in mm.mps_quant_formats(mm.get_torch_device())):
        raise ValueError("KREA_READONLY requires native INT8 MPS without LoRA")
    from krea_readonly import readonly_krea_load, bounded_krea_residency
    resident_gib = float(os.environ.get("KREA_RESIDENT_GIB", "1"))
    if not 0 < resident_gib <= 4:
        raise ValueError("KREA_RESIDENT_GIB must be finite, positive, and at most 4")
    resident_budget_bytes = int(resident_gib * 2**30)
    if resident_budget_bytes == 0:
        raise ValueError("KREA_RESIDENT_GIB is smaller than one byte")
adapter_receipt = None
adapter_contract = None
if ADAPTER_READONLY:
    if READONLY or MODE != "metal" or not LORA:
        raise ValueError("KREA_ADAPTER_READONLY requires a separate native INT8 LoRA route")
    from krea_adapter import verify_adapter, verify_plain_adapter, load_plain_adapter, readonly_adapter_load, AdapterContract
    plain_adapter = os.environ.get("KREA_PLAIN_LORA") == "1"
    adapter_receipt = (verify_plain_adapter if plain_adapter else verify_adapter)(folder_paths.get_full_path_or_raise("loras", LORA_NAME),
        **({} if plain_adapter else {"name": LORA_NAME}), strength=LORA_STRENGTH, steps=STEPS, cfg=CFG,
        spectral=os.environ.get("SPECTRAL") == "1")
    if float(os.environ.get("KREA_RESIDENT_GIB", "1")) != 1:
        raise ValueError("Immutable adapter qualification requires the 1 GiB residency profile")
    resident_gib = 1.0
    resident_budget_bytes = 2**30
T = {}; t0 = time.perf_counter()
def lap(name):
    global t0
    if torch.backends.mps.is_available(): torch.mps.synchronize()
    T[name] = round(time.perf_counter() - t0, 2); t0 = time.perf_counter()
with torch.inference_mode():
    from krea_conditioning import encode_conditioning
    pos, neg, conditioning_audit = encode_conditioning(N, root=ROOT, code=CODE,
        folder_paths=folder_paths, torch=torch, mm=mm, mode=MODE, argv=sys.argv[1:],
        te=TE, prompt=PROMPT, cfg=CFG, policy=os.environ.get("COND_CACHE", "off"),
        stage=os.environ.get("COND_STAGE", "generate"), lap=lap)
    if conditioning_audit["stage"] == "capture":
        print("CONDITIONING_RESULT " + json.dumps(dict(times=T, conditioning=conditioning_audit)), flush=True)
        raise SystemExit(0)
    if MAPPED:
        torch.mps.set_per_process_memory_fraction(0.5)
    checkpoint = folder_paths.get_full_path_or_raise("diffusion_models", UNET) if MAPPED else None
    load_scope = (readonly_krea_load(comfy.utils, checkpoint,
                  execution_device=mm.get_torch_device(), native_int8=True)
                  if READONLY else nullcontext({}))
    if ADAPTER_READONLY:
        load_scope = readonly_adapter_load(comfy.utils, checkpoint, mm=mm, adapter=adapter_receipt)
    with load_scope as readonly_audit:
        model = N["UNETLoader"]().load_unet(UNET, "default")[0]
    if MAPPED and readonly_audit["loads"] != 1:
        raise RuntimeError("Krea read-only loader was not used exactly once")
    if LORA:
        if ADAPTER_READONLY and plain_adapter:
            model = load_plain_adapter(model, comfy.utils, adapter_receipt, LORA_STRENGTH)
        else:
            model = N["LoraLoaderModelOnly"]().load_lora_model_only(model, LORA_NAME, LORA_STRENGTH)[0]
    if ADAPTER_READONLY:
        adapter_contract = AdapterContract(mm, comfy.utils, model, adapter_receipt)
    lap("load_unet")
    vae = N["VAELoader"]().load_vae("qwen_image_vae.safetensors")[0]
    latent = N["EmptyLatentImage"]().generate(W, H, 1)[0]; lap("load_vae")
    spectral_plan = []
    if os.environ.get("SPECTRAL") == "1":
        if LORA or (STEPS, CFG) != (8, 1.0):
            raise ValueError("Spectral experiment requires the eight-step CFG-1 base recipe")
        from spectral_krea import wrap_euler
        import comfy.samplers
        select_sampler = comfy.samplers.sampler_object
        def record_spectral_plan(event):
            spectral_plan.append(event)
            print("SPECTRAL_PLAN " + json.dumps(event), flush=True)
        def spectral_sampler(name):
            if name != "euler":
                raise ValueError("Spectral experiment requires Euler")
            return wrap_euler(select_sampler(name), model, pos, neg, enabled=True,
                              report=record_spectral_plan)
        comfy.samplers.sampler_object = spectral_sampler
    step_t = []
    orig = comfy.utils.ProgressBar.update_absolute
    def upd(self, *a, **k):
        if torch.backends.mps.is_available(): torch.mps.synchronize()
        step_t.append(time.perf_counter()); return orig(self, *a, **k)
    comfy.utils.ProgressBar.update_absolute = upd
    denoiser_calls = []
    original_predict = None
    if os.environ.get("DENOISE_TIMING") == "1":
        import comfy.samplers
        original_predict = comfy.samplers.CFGGuider.predict_noise
        def timed_predict(self, x, timestep, model_options={}, seed=None):
            if x.device.type == "mps": torch.mps.synchronize()
            started = time.perf_counter()
            prediction = original_predict(self, x, timestep, model_options=model_options, seed=seed)
            if x.device.type == "mps": torch.mps.synchronize()
            denoiser_calls.append(dict(shape=list(x.shape), dtype=str(x.dtype), seconds=time.perf_counter() - started))
            return prediction
        comfy.samplers.CFGGuider.predict_noise = timed_predict
    try:
        residency_scope = bounded_krea_residency(mm, model, resident_budget_bytes) if READONLY else nullcontext({})
        if ADAPTER_READONLY:
            residency_scope = adapter_contract.residency(model, resident_budget_bytes)
        with residency_scope as residency_audit:
            out = N["KSampler"]().sample(model, SEED, STEPS, CFG, "euler", "simple", pos, neg, latent, 1.0)[0]
            lap("sample")
    finally:
        if original_predict is not None:
            comfy.samplers.CFGGuider.predict_noise = original_predict
    peak = torch.mps.driver_allocated_memory() / 2**30 if torch.backends.mps.is_available() else 0
    img = N["VAEDecode"]().decode(vae, out)[0]; lap("decode")
steps = [round(b - a, 2) for a, b in zip(step_t, step_t[1:])]
lora_tag = f"-lora-{'merge' if os.environ.get('COMFY_MPS_LORA_SIDE_PATH') == '0' else 'side'}" if LORA else ""
tag = f"{MODE}{f'-readonly{resident_gib:g}g' if MAPPED else ''}{'-spectral' if os.environ.get('SPECTRAL') == '1' else ''}{'-fp16' if FP16 else ''}{'-sdpa' if ATTN == 'pytorch' else '-quad' if ATTN == 'quad' else ''}{lora_tag}-{W}x{H}{('-' + os.environ['TAG']) if os.environ.get('TAG') else ''}"
os.makedirs(f"{ROOT}/out", exist_ok=True)
from PIL import Image; import numpy as np
arr = (img[0].float().cpu().numpy() * 255).clip(0, 255).astype("uint8"); Image.fromarray(arr).save(f"{ROOT}/out/krea2-{tag}.png")
res = dict(tag=tag, torch=torch.__version__, times=T, step_s=steps, mps_alloc_gb_after_sample=round(peak, 2),
           nan=bool(torch.isnan(img).any()), mean=float(img.mean()), std=float(img.std()),
           conditioning=conditioning_audit,
           recipe=dict(unet=UNET, encoder=TE, prompt=PROMPT, seed=SEED, cfg=CFG, steps=STEPS,
                       width=W, height=H, sampler="euler", scheduler="simple", mode=MODE,
                       fp16_requested=FP16, attention_requested=ATTN or "fork-default",
                       readonly=MAPPED, adapter_readonly=ADAPTER_READONLY, resident_budget_bytes=resident_budget_bytes,
                       spectral=os.environ.get("SPECTRAL") == "1", lora=LORA,
                       lora_name=LORA_NAME if LORA else None,
                       lora_strength=LORA_STRENGTH if LORA else None))
if os.environ.get("DENOISE_TIMING") == "1": res["denoiser_calls"] = denoiser_calls
if spectral_plan: res["spectral_plan"] = spectral_plan
if MAPPED:
    import krea_readonly
    res["readonly_experiment"] = dict(loader=readonly_audit, residency=residency_audit,
        allocator_fraction=0.5, helper_sha256=hashlib.sha256(Path(krea_readonly.__file__).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="experimental static INT8; qualification depends on the recorded recipe and source-bound receipt",
        immutable_checkpoint_required=True, allocator_scope="after text encoding through diffusion and VAE")
    if ADAPTER_READONLY:
        import krea_adapter
        res["readonly_experiment"]["adapter"] = dict(asset=adapter_receipt, keys=len(model.patches),
            helper_sha256=hashlib.sha256(Path(krea_adapter.__file__).read_bytes()).hexdigest(),
            mmap_offload=model.mmap_offload, scope="single ordinary eight-step LoRA" if plain_adapter else "released four-step adapter; existing side/lazy temporary-copy patch machinery")
print("RESULT " + json.dumps(res)); open(f"{ROOT}/out/results.jsonl", "a").write(json.dumps(res) + "\n")
