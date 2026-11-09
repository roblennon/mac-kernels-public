"""Text-encoder timing and placement on Apple GPUs: where the encoder runs, in which dtype, how long
encoding takes, peak memory, and the conditioning it produces (saved for CPU/GPU comparison).

Env:
  MODEL      krea | qwen | h3 (picks the encoder file, the clip type and the encode node the runners use)
  TE_DEVICE  default (ComfyUI's own policy) | cpu (CLIPLoader device=cpu) | mps (force the encoder onto the GPU)
  REPEAT     encodes per run (default 2; the first includes the one-time load to the device)
  COMFY_DIR  ComfyUI checkout (default vendor/ComfyUI); PYTHONPATH picks the comfy-kitchen checkout
  TAG        appended to the saved conditioning name
  COMPARE    path of a saved conditioning to compare against (cosine, max abs diff, rel L2)
Saves out/te/<model>-<device tag>[-TAG].pt and appends a RESULT line to out/results.jsonl.
"""
import asyncio, json, os, resource, sys, threading, time
ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
MODEL = os.environ.get("MODEL", "krea"); TE_DEVICE = os.environ.get("TE_DEVICE", "default")
REPEAT = int(os.environ.get("REPEAT", 2))
CODE = os.environ.get("COMFY_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ComfyUI")
sys.path.insert(0, CODE); os.chdir(CODE)
sys.argv = [sys.argv[0], "--use-pytorch-cross-attention"]
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths
SUB = {"krea": "", "qwen": "qwen21/", "h3": "h3/"}[MODEL]
for k in ("text_encoders",):
    folder_paths.add_model_folder_path(k, f"{ROOT}/models/{SUB}{k}")
import comfy.model_management as mm
if TE_DEVICE == "mps":
    mm.text_encoder_device = lambda: torch.device("mps")
import comfy_kitchen, nodes
asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
N = nodes.NODE_CLASS_MAPPINGS
SPEC = {
    "krea": ("qwen3vl_4b_bf16.safetensors", "krea2",
             "Ultra low angle action shot, male soccer player jumping mid-air high kick, clear vivid blue sky background, sharp focused upper body, maroon jersey and white shorts, harsh direct sunlight, high contrast sports photography"),
    "qwen": ("qwen3vl_8b_int8_convrot.safetensors", "qwen_image",
             "Greyscale fashion editorial portrait of an avant-garde woman in profile, tilted upwards, striking high-contrast chiaroscuro lighting, sculptural collar, fine film grain, studio backdrop."),
    "h3": ("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "minimax",
           "Realistic live-action cinematic shot: a golden retriever runs along a beach at sunset toward the camera, waves rolling in, shallow depth of field, film grain. Audio: waves, wind, paws on wet sand."),
}
TE, CLIP_TYPE, PROMPT = SPEC[MODEL]
TE = os.environ.get("TE", TE)

PEAK = {"gpu_gb": 0.0}
def _poll():
    while True:
        PEAK["gpu_gb"] = max(PEAK["gpu_gb"], torch.mps.driver_allocated_memory() / 2**30); time.sleep(0.1)
threading.Thread(target=_poll, daemon=True).start()

def call(name, **kw):
    cls = N[name]
    if hasattr(cls, "define_schema"):
        out = cls.execute(**kw); return out.result if hasattr(out, "result") else out.args
    return getattr(cls(), cls.FUNCTION)(**kw)

def encode(clip):
    if MODEL == "krea":
        return N["CLIPTextEncode"]().encode(clip, PROMPT)[0]
    if MODEL == "qwen":
        return call("TextEncodeQwenImage21", clip=clip, prompt=PROMPT, negative_prompt="", resolution=1024)[0]
    return call("MiniMaxH3ImageToVideo", clip=clip, vae=None, prompt=PROMPT, width=480, height=288, length=22)[0]

def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

T = {}
with torch.inference_mode():
    t0 = time.perf_counter()
    clip = N["CLIPLoader"]().load_clip(TE, CLIP_TYPE, "cpu" if TE_DEVICE == "cpu" else "default")[0]
    sync(); T["load"] = round(time.perf_counter() - t0, 2)
    patcher = clip.patcher
    model = clip.cond_stage_model
    lin = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    fmts = {}
    for m in lin:
        f = getattr(m, "quant_format", None) or str(getattr(m.weight, "dtype", None))
        fmts[f] = fmts.get(f, 0) + 1
    info = dict(load_device=str(patcher.load_device), offload_device=str(patcher.offload_device),
                model_dtypes=[str(d) for d in getattr(model, "dtypes", [])], linear_formats=fmts)
    print("INFO", json.dumps(info), flush=True)
    calls = {"mps_kernel": 0, "dequantize": 0}
    from importlib import import_module
    reg = import_module("comfy_kitchen.registry")
    orig_get = reg.registry.get_implementation
    def counting(func_name, backend=None, kwargs=None):
        impl = orig_get(func_name, backend, kwargs)
        mod = getattr(impl, "__module__", "")
        key = f"{func_name}@{mod.split('.')[2] if mod.count('.') >= 2 else mod}"
        calls[key] = calls.get(key, 0) + 1
        return impl
    reg.registry.get_implementation = counting
    conds = []
    for i in range(REPEAT):
        sync(); t0 = time.perf_counter()
        cond = encode(clip)
        sync(); T[f"encode{i}"] = round(time.perf_counter() - t0, 2)
        print("LAP", f"encode{i}", T[f"encode{i}"], flush=True)
        conds.append(cond)
    reg.registry.get_implementation = orig_get
    after_device = str(next(iter(model.parameters())).device) if any(True for _ in model.parameters()) else "?"
    t0 = time.perf_counter()
    del clip, model, patcher, lin; mm.unload_all_models(); mm.soft_empty_cache()
    import gc; gc.collect(); mm.soft_empty_cache(); sync()
    T["release"] = round(time.perf_counter() - t0, 2)
    gpu_after_release = round(torch.mps.driver_allocated_memory() / 2**30, 2)

c = conds[-1]
tensors = [(x[0].float().cpu(), {k: v.float().cpu() for k, v in x[1].items() if isinstance(v, torch.Tensor)}) for x in c]
dev_tag = TE_DEVICE
tag = f"{MODEL}-{dev_tag}" + (f"-{os.environ['TAG']}" if os.environ.get("TAG") else "")
os.makedirs(f"{ROOT}/out/te", exist_ok=True)
torch.save(tensors, f"{ROOT}/out/te/{tag}.pt")
stats = [dict(shape=list(t.shape), absmax=float(t.abs().max()), nan=bool(torch.isnan(t).any())) for t, _ in tensors]
cmp = None
if os.environ.get("COMPARE"):
    ref = torch.load(os.environ["COMPARE"])
    cmp = []
    for (a, ea), (b, eb) in zip(tensors, ref):
        a64, b64 = a.double().flatten(), b.double().flatten()
        d = dict(cos=float(torch.nn.functional.cosine_similarity(a64, b64, dim=0)), rel_l2=float((a64 - b64).norm() / b64.norm()),
                 max_abs=float((a64 - b64).abs().max()), ref_absmax=float(b64.abs().max()))
        for k in ea:
            if k in eb and ea[k].shape == eb[k].shape:
                x64, y64 = ea[k].double().flatten(), eb[k].double().flatten()
                d[f"{k}_cos"] = float(torch.nn.functional.cosine_similarity(x64, y64, dim=0))
        cmp.append(d)
rss_gb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)  # bytes on macOS
res = dict(kind="te_encode", tag=tag, model=MODEL, te=TE, te_device=TE_DEVICE, torch=torch.__version__,
           kitchen=os.path.dirname(comfy_kitchen.__file__), comfy=CODE, times=T, info=info, param_device_after=after_device,
           kernel_calls=calls, peak_gpu_gb=round(PEAK["gpu_gb"], 2), peak_rss_gb=rss_gb, gpu_after_release_gb=gpu_after_release,
           cond=stats, compare=cmp)
print("RESULT " + json.dumps(res), flush=True)
open(f"{ROOT}/out/results.jsonl", "a").write(json.dumps(res) + "\n")
