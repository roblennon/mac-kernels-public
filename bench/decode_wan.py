"""Decode a saved Wan latent (out/latents/*.pt) with VAEDecode or VAEDecodeTiled; report time, peak GPU memory, and PSNR vs a reference decode.

Env: LATENT (file name under out/latents), VAE (default wan2.2_vae.safetensors), MODES (comma list: full, tiled<tile>x<temporal>, e.g. full,tiled512x64),
COMFY_DIR, ATTN as in the runners. REF_SAVE=<file> saves the first decode as uint8 frames; REF=<file> scores every mode against such a file. SHEET=1 saves a 4-frame sheet per mode. VIDEO=1 also saves an MP4 at 24 fps.
"""
import asyncio, gc, json, os, sys, time, threading
import hashlib


from wan_latents import latent_identity, load_saved_latent

ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
CODE = os.environ.get("COMFY_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ComfyUI")
ATTN = os.environ.get("ATTN", "pytorch")
sys.path.insert(0, CODE); os.chdir(CODE)
sys.argv = [sys.argv[0]] + {"pytorch": ["--use-pytorch-cross-attention"], "auto": [], "split": ["--use-split-cross-attention"], "quad": ["--use-quad-cross-attention"]}[ATTN]
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths
folder_paths.add_model_folder_path("vae", f"{ROOT}/models/wan/vae"); folder_paths.add_model_folder_path("vae_approx", f"{ROOT}/models/wan/vae")  # tiny VAEs load from vae_approx
import comfy.model_management as mm, nodes
asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
N = nodes.NODE_CLASS_MAPPINGS
PEAK = {"gb": 0.0}
def _poll():
    while True:
        PEAK["gb"] = max(PEAK["gb"], torch.mps.driver_allocated_memory() / 2**30); time.sleep(0.25)
threading.Thread(target=_poll, daemon=True).start()

LATENT = os.environ["LATENT"]; VAE = os.environ.get("VAE", "wan2.2_vae.safetensors")
with torch.inference_mode():
    lat, latent_source, expected_geometry = load_saved_latent(f"{ROOT}/out/latents/{LATENT}")
    vae = N["VAELoader"]().load_vae(VAE)[0]
    if vae.latent_channels != lat['samples'].shape[1]:
        raise ValueError('Selected VAE does not match the saved Wan latent channels')
    ref = None
    for mode in os.environ.get("MODES", "full,tiled512x64").split(","):
        mm.unload_all_models(); mm.soft_empty_cache(); gc.collect(); PEAK["gb"] = 0.0
        t = time.perf_counter()
        if mode == "full":
            img = N["VAEDecode"]().decode(vae, lat)[0]
        else:
            tile, temporal = (int(x) for x in mode[len("tiled"):].split("x"))
            img = N["VAEDecodeTiled"]().decode(vae, lat, tile, 64, temporal, 8)[0]
        torch.mps.synchronize(); dt = time.perf_counter() - t
        img = img.reshape(-1, *img.shape[-3:]).float().cpu()
        if tuple(img.shape) != expected_geometry or not bool(torch.isfinite(img).all()):
            raise RuntimeError('Wan decode must preserve finite frames and the saved latent geometry')
        if latent_identity(latent_source['path']) != latent_source['file_identity']:
            raise RuntimeError('Saved latent changed during decoding')
        pixels = (img.clamp(0, 1) * 255).round().to(torch.uint8).contiguous()
        if ref is None:
            ref = (torch.load(os.environ["REF"], map_location='cpu', weights_only=True).float() / 255) if os.environ.get("REF") else img
            if tuple(ref.shape) != expected_geometry or not bool(torch.isfinite(ref).all()):
                raise ValueError('Reference decode must match finite Wan frames and dimensions')
            if os.environ.get("REF_SAVE"):
                torch.save(pixels, os.environ["REF_SAVE"])
        mse = float((((img.clamp(0, 1) * 255).round() / 255 - ref) ** 2).mean()) if os.environ.get("REF") else float(((img - ref) ** 2).mean())
        video_path = None
        if os.environ.get('VIDEO') == '1':
            cls = N['CreateVideo']
            if hasattr(cls, 'define_schema'):
                output = cls.execute(images=img, fps=24.0)
                video = (output.result if hasattr(output, 'result') else output.args)[0]
            else:
                video = getattr(cls(), cls.FUNCTION)(images=img, fps=24.0)[0]
            video_path = f"{ROOT}/out/decode-{os.path.splitext(LATENT)[0]}-{os.path.splitext(VAE)[0]}-{mode}.mp4"
            video.save_to(video_path)
        res = dict(latent=LATENT, vae=VAE, mode=mode, attn=ATTN, decode_s=round(dt, 1), peak_gb=round(PEAK["gb"], 2), frames=int(img.shape[0]),
                   width=int(img.shape[2]), height=int(img.shape[1]), finite=True, latent_source=latent_source,
                   pixel_sha256=hashlib.sha256(pixels.numpy().tobytes()).hexdigest(), video_path=video_path,
                   psnr_vs_first=None if mse == 0 else round(10 * __import__("math").log10(1 / mse), 1), nan=bool(img.isnan().any()))
        print("RESULT " + json.dumps(res), flush=True)
        if os.environ.get("SHEET") == "1":
            from PIL import Image; import numpy as np
            fr = (img.clamp(0, 1).numpy() * 255).astype("uint8")
            Image.fromarray(np.concatenate([fr[i] for i in np.linspace(0, fr.shape[0] - 1, 4).astype(int)], axis=1)).save(
                f"{ROOT}/out/decode-{os.path.splitext(LATENT)[0]}-{os.path.splitext(VAE)[0]}-{mode}-sheet.png")
