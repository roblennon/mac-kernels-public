"""MiniMax H3 video VAE on a saved latent: decode time, peak memory, PSNR vs a reference decode, encoder time, and a
per-module-class breakdown of the decode.

Env: LATENT (default out/cache/h3-metal-480x288-22f-4s-latent.pt), COMFY_DIR, REF_SAVE / REF (uint8 frames file),
ENCODE=1 also encodes the decoded clip, PROFILE=1 adds the hooked breakdown.
"""
import asyncio, collections, json, math, os, sys, threading, time
ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
CODE = os.environ.get("COMFY_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ComfyUI")
sys.path.insert(0, CODE); os.chdir(CODE)
sys.argv = [sys.argv[0], "--use-pytorch-cross-attention"]
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths
folder_paths.add_model_folder_path("vae", f"{ROOT}/models/h3/vae")
import comfy.model_management as mm, nodes
asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
N = nodes.NODE_CLASS_MAPPINGS
PEAK = {"gb": 0.0}
def _poll():
    while True:
        PEAK["gb"] = max(PEAK["gb"], torch.mps.driver_allocated_memory() / 2**30); time.sleep(0.1)
threading.Thread(target=_poll, daemon=True).start()
def timed(fn):
    mm.soft_empty_cache(); torch.mps.synchronize(); PEAK["gb"] = torch.mps.driver_allocated_memory() / 2**30
    t = time.perf_counter(); o = fn(); torch.mps.synchronize(); return o, round(time.perf_counter() - t, 2), round(PEAK["gb"], 2)

with torch.inference_mode():
    lat = torch.load(os.environ.get("LATENT", f"{ROOT}/out/cache/h3-metal-480x288-22f-4s-latent.pt"), weights_only=False)
    vae = N["VAELoader"]().load_vae("minimax_h3_video_vae_fp16.safetensors")[0]
    N["VAEDecode"]().decode(vae, {"samples": lat["samples"][:, :, :2]})  # warm-up
    img, dec_s, dec_gb = timed(lambda: N["VAEDecode"]().decode(vae, lat)[0])
    img = img.reshape(-1, *img.shape[-3:]).float().cpu()
    res = dict(comfy=CODE, decode_s=dec_s, decode_peak_gb=dec_gb, frames=int(img.shape[0]), nan=bool(img.isnan().any()))
    q = (img.clamp(0, 1) * 255).round()
    if os.environ.get("REF_SAVE"):
        torch.save(q.to(torch.uint8), os.environ["REF_SAVE"])
    if os.environ.get("REF"):
        mse = float(((q - torch.load(os.environ["REF"]).float()) ** 2).mean()) / 255 ** 2
        res["psnr_vs_ref"] = None if mse == 0 else round(10 * math.log10(1 / mse), 1)
    if os.environ.get("ENCODE") == "1":
        _, enc_s, enc_gb = timed(lambda: vae.encode(img[..., :3]))
        res.update(encode_s=enc_s, encode_peak_gb=enc_gb)
    print("RESULT " + json.dumps(res), flush=True)
    if os.environ.get("PROFILE") == "1":
        stats = collections.defaultdict(float); t0 = {}
        def pre(m, *a): torch.mps.synchronize(); t0[id(m)] = time.perf_counter()
        def post(m, *a): torch.mps.synchronize(); stats[type(m).__name__] += time.perf_counter() - t0.pop(id(m))
        dec = vae.first_stage_model.decoder if hasattr(vae.first_stage_model, "decoder") else vae.first_stage_model
        for name, m in dec.named_modules():
            if len(list(m.children())) == 0 or type(m).__name__ in ("Attention", "FeedForward"):
                m.register_forward_pre_hook(pre); m.register_forward_hook(post)
        _, hooked_s, _ = timed(lambda: N["VAEDecode"]().decode(vae, lat)[0])
        print("RESULT " + json.dumps(dict(kind="profile", hooked_s=hooked_s, by_class={k: round(v, 2) for k, v in sorted(stats.items(), key=lambda kv: -kv[1])})), flush=True)
