"""Time loading the MiniMax H3 DiT (+ 4-step Turbo LoRA, side path) onto the Apple GPU, with no sampling.

Env: UNET (default W4A8 file), FLAGS (extra ComfyUI CLI flags, e.g. "--disable-mmap"), PREWARM=1 (read the file
sequentially before loading), EVICT=<file> (read another large file first so this one starts cold in the page cache).
Prints phase times (file open, model build, GPU load), vm_stat deltas (page-ins, page-outs, swap-ins, swap-outs) and
swap growth.
"""
import asyncio, json, os, re, subprocess, sys, threading, time
ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
D = f"{ROOT}/models/h3/diffusion_models"
UNET = os.environ.get("UNET", "minimax_h3_fl2va_pruned_w4a8.safetensors")
CODE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ComfyUI")

def vmstat():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    get = lambda k: int(re.search(rf"{k}:\s+(\d+)", out).group(1))
    return {"pageins": get("Pageins"), "pageouts": get("Pageouts"), "swapins": get("Swapins"), "swapouts": get("Swapouts")}

def swap_gb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used = ")[1].split("M")[0]) / 1024

def read_through(path):
    t = time.perf_counter(); buf = bytearray(64 << 20); mv = memoryview(buf)
    with open(path, "rb", buffering=0) as f:
        while f.readinto(mv): pass
    return round(time.perf_counter() - t, 1)

res = dict(unet=UNET, flags=os.environ.get("FLAGS", ""), prewarm=os.environ.get("PREWARM") == "1")
if os.environ.get("EVICT"):
    res["evict_read_s"] = read_through(os.environ["EVICT"])
sys.path.insert(0, CODE); os.chdir(CODE)
sys.argv = [sys.argv[0]] + os.environ.get("FLAGS", "").split()
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths
for k in ("diffusion_models", "loras"):
    folder_paths.add_model_folder_path(k, f"{ROOT}/models/h3/{k}")
import comfy.model_management as mm, nodes
asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
PEAK = {"gpu": 0.0, "swap": 0.0}
def _poll():
    while True:
        PEAK["gpu"] = max(PEAK["gpu"], torch.mps.driver_allocated_memory() / 2**30); PEAK["swap"] = max(PEAK["swap"], swap_gb()); time.sleep(1)
threading.Thread(target=_poll, daemon=True).start()
v0, s0 = vmstat(), swap_gb(); res["swap_start_gb"] = round(s0, 2)
if res["prewarm"]:
    res["prewarm_read_s"] = read_through(f"{D}/{UNET}")
t = time.perf_counter()
with torch.inference_mode():
    model = nodes.NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet(UNET, "default")[0]
    model = nodes.NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]().load_lora_model_only(model, "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", 1.0)[0]
    res["build_s"] = round(time.perf_counter() - t, 1); t = time.perf_counter()
    mm.load_models_gpu([model]); torch.mps.synchronize()
    res["gpu_load_s"] = round(time.perf_counter() - t, 1)
v1 = vmstat()
res.update({k + "_delta_gb": round((v1[k] - v0[k]) * 16384 / 2**30, 2) for k in v0})
res.update(peak_gpu_gb=round(PEAK["gpu"], 2), swap_growth_gb=round(PEAK["swap"] - s0, 2), total_s=round(res["build_s"] + res["gpu_load_s"] + res.get("prewarm_read_s", 0), 1))
print("PROBE " + json.dumps(res), flush=True)
