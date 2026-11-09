"""Submit the official ComfyUI templates (API graphs in bench/templates_api/, converted by the ComfyUI frontend's own
graphToPrompt) to a running server with this machine's models and recommended settings, and time them as the UI would.

Usage: python bench/run_templates.py [name ...]   (server at $COMFY_URL, default http://127.0.0.1:8188)
For each template: POST /prompt, poll /history until done, record wall time from submit, the server's own
execution_start -> execution_success span, outputs, and any error with its traceback. Models are freed between
templates (POST /free) so each run starts cold, like the first run after opening the UI.
"""
import argparse, json, os, sys, time, urllib.request, uuid

URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
SEED = 7

def set_inputs(g, node, **kw):
    g[node]["inputs"].update(kw)

def krea2(g):
    set_inputs(g, "30:11", clip_name="qwen3vl_4b_bf16.safetensors")       # template ships the fp8_scaled encoder
    set_inputs(g, "30:3", seed=SEED)
    set_inputs(g, "30:16", seed=SEED) if "seed" in g["30:16"]["inputs"] else None
    return "Krea 2 Turbo INT8, 1024², 8 steps, template prompt enhancer on (same encoder), LoRA switch off (template default)"

def qwen21(g):
    # the prompt enhancer is off by default but its 9B model is still wired in and fails validation without the file
    lit = g["459:474"]["inputs"]["on_false"]
    for n in ("459:471", "459:473", "459:475"):
        g.pop(n)
    set_inputs(g, "459:474", on_true=lit, switch=False)
    set_inputs(g, "459:458", seed=SEED)
    return "Qwen-Image 2.1 INT8, 1024², 25 steps, enhancer nodes removed (model not present)"

def wan5b(g):
    set_inputs(g, "37", unet_name="wan2.2_ti2v_5b_turbo_int8_convrot.safetensors")
    set_inputs(g, "38", clip_name="umt5_xxl_fp16.safetensors")
    set_inputs(g, "3", seed=SEED, steps=4, cfg=1.0, sampler_name="euler")
    set_inputs(g, "55", width=1280, height=704, length=81)
    return "Wan 2.2 TI2V-5B Turbo INT8, 1280x704x81, 4 steps euler cfg 1"

def h3(g):
    set_inputs(g, "140:127", unet_name="minimax_h3_fl2va_pruned_w4a8.safetensors")
    set_inputs(g, "140:119", vae_name="minimax_h3_video_vae_fp16.safetensors")
    set_inputs(g, "140:134", lora_name="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors")
    set_inputs(g, "140:139", value=True)    # Turbo LoRA path
    set_inputs(g, "140:138", value=4)       # Turbo steps (template: 8 for its 8-step LoRA)
    set_inputs(g, "140:133", value=2.3)     # seconds -> 56 frames
    set_inputs(g, "140:129", noise_seed=SEED)
    return "MiniMax H3 W4A8 + 4-step Turbo LoRA, 0.4 MP 16:9 (832x480), 56 frames, 4 steps"

def wan14b(g):
    set_inputs(g, "128:71", clip_name="umt5_xxl_fp16.safetensors")
    set_inputs(g, "128:75", unet_name="wan2.2_t2v_high_noise_14B_int8_convrot.safetensors")
    set_inputs(g, "128:76", unet_name="wan2.2_t2v_low_noise_14B_int8_convrot.safetensors")
    set_inputs(g, "128:129", value=True)    # lightx2v 4-step LoRA path: 4 steps, split 2, cfg 1
    set_inputs(g, "128:81", noise_seed=SEED)
    return "Wan 2.2 T2V-A14B INT8 + lightx2v 4-step LoRAs, 640x640x81, 2+2 steps cfg 1"

TEMPLATES = {"image_krea2_turbo_t2i_int8": krea2, "image_qwen_image_2_1_t2i": qwen21, "video_wan2_2_5B_ti2v": wan5b,
             "video_minimax_h3_t2v": h3, "video_wan2_2_14B_t2v": wan14b}

ACCEPTED = ("image_qwen_image_2_1_t2i", "video_wan2_2_5B_ti2v")


def build_graph(name, preset="historical"):
    """Reuse the exported graph, selecting only previously qualified settings."""
    if preset not in ("historical", "accepted"):
        raise ValueError(f"Unknown preset: {preset}")
    if preset == "accepted" and name not in ACCEPTED:
        raise ValueError(f"No accepted preset for {name}")
    with open(f"{HERE}/templates_api/{name}.json") as source:
        graph = json.load(source)
    description = TEMPLATES[name](graph)
    if preset == "accepted":
        if name == ACCEPTED[0]:
            set_inputs(graph, "459:451", mps_direct_load=True)
            set_inputs(graph, "459:456", width=512, height=512, batch_size=1)
            description = "Accepted Qwen INT8: direct loading, 512px, 25 Euler/simple steps, CFG 1; enhancer off"
        else:
            set_inputs(graph, "3", scheduler="simple")
            set_inputs(graph, "55", width=704, height=416, length=33, batch_size=1)
            description = "Accepted Wan 5B Turbo INT8: 704x416x33, 4 Euler/simple steps, CFG 1, full VAE"
    return graph, description


def req(path, data=None):
    r = urllib.request.Request(URL + path, data=None if data is None else json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=60) as f:
        return json.loads(f.read() or b"null")

def run(name, preset="historical"):
    g, desc = build_graph(name, preset)
    req("/free", {"unload_models": True, "free_memory": True}); time.sleep(3)
    t0 = time.time()
    try:
        pid = req("/prompt", {"prompt": g, "client_id": str(uuid.uuid4())})["prompt_id"]
    except urllib.error.HTTPError as e:
        return dict(template=name, settings=desc, ok=False, error="validation: " + e.read().decode()[:2000])
    while True:
        time.sleep(5)
        h = req(f"/history/{pid}")
        if pid in h and h[pid].get("status", {}).get("completed") is not None or (pid in h and h[pid]["status"].get("status_str") == "error"):
            break
    wall = time.time() - t0
    st = h[pid]["status"]; msgs = {m[0]: m[1] for m in st.get("messages", [])}
    span = None
    if "execution_start" in msgs and ("execution_success" in msgs or "execution_error" in msgs):
        end = msgs.get("execution_success", msgs.get("execution_error"))
        span = round((end["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000, 1)
    files = [f"{o.get('subfolder', '')}/{o['filename']}" for out in h[pid].get("outputs", {}).values()
             for k in ("images", "gifs", "videos", "audio") for o in out.get(k, []) if isinstance(o, dict) and "filename" in o]
    err = msgs.get("execution_error")
    return dict(template=name, settings=desc, ok=st.get("status_str") == "success", wall_s=round(wall, 1), server_span_s=span, outputs=files,
                error=None if not err else {k: err.get(k) for k in ("node_id", "node_type", "exception_message", "traceback")})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", metavar="name", help="Template names from bench/templates_api")
    parser.add_argument("--preset", choices=("historical", "accepted"), default="historical")
    parser.add_argument("--dry-run", action="store_true", help="Print selected graphs without contacting the server")
    args = parser.parse_args()
    unknown = [name for name in args.names if name not in TEMPLATES]
    if unknown:
        parser.error(f"Unknown templates: {', '.join(unknown)}")
    names = args.names or (ACCEPTED if args.preset == "accepted" else TEMPLATES)
    if args.preset == "accepted" and any(name not in ACCEPTED for name in names):
        parser.error("Accepted presets exist only for Qwen 2.1 and Wan 5B")
    for name in names:
        if args.dry_run:
            graph, description = build_graph(name, args.preset)
            print(json.dumps({"template": name, "preset": args.preset, "settings": description, "graph": graph}))
            continue
        res = run(name, args.preset)
        print("TEMPLATE " + json.dumps(res), flush=True)
        open(f"{ROOT}/out/templates.jsonl", "a").write(json.dumps(res) + "\n")
