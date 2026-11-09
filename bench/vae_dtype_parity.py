"""Decode one latent with a VAE in fp32, fp16 and bf16 on MPS and compare against fp32.

With IMAGE, the fp16/bf16 encodes of that image are also compared with the fp32 encode.

Evidence for the CU-02 fp16 VAE preference: output parity and timing per dtype.
The latent is a saved one (LATENT, under out/latents, first FRAMES latent frames)
or an fp32 encode of IMAGE resized to SIZE. Run one VAE per guarded process.

Env: VAE (path), LATENT or IMAGE, FRAMES (default 2), SIZE (default 768),
DTYPES (default fp32,fp16,bf16), REFERENCE_DEVICE for the fp32 encode
(default mps; cpu avoids MPS-only encoder bugs), COMFY_DIR (default vendor/ComfyUI).
"""
import json, os, sys, time

ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
CODE = os.environ.get("COMFY_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ComfyUI")
sys.path.insert(0, CODE); os.chdir(CODE)
sys.argv = [sys.argv[0]]
import comfy.options; comfy.options.enable_args_parsing()
import numpy as np, torch
from PIL import Image
import comfy.model_management as mm, comfy.sd, comfy.utils

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def timed(fn):
    torch.mps.synchronize(); t = time.perf_counter(); out = fn(); torch.mps.synchronize()
    return out, time.perf_counter() - t


def load(dtype, device=None):
    sd = comfy.utils.load_torch_file(os.environ["VAE"])
    return comfy.sd.VAE(sd=sd, dtype=dtype, device=device)


with torch.inference_mode():
    default = load(None)
    working = [str(d).removeprefix("torch.") for d in default.working_dtypes]
    print("RESULT " + json.dumps(dict(kind="policy", vae=os.path.basename(os.environ["VAE"]),
          default_dtype=str(default.vae_dtype), working_dtypes=working, torch=torch.__version__)), flush=True)
    if "LATENT" in os.environ:
        latent = torch.load(f"{ROOT}/out/latents/{os.environ['LATENT']}", map_location="cpu")
        latent = latent[:, :, :int(os.environ.get("FRAMES", 2))].float()
    else:
        size = int(os.environ.get("SIZE", 768))
        image = Image.open(os.environ["IMAGE"]).convert("RGB").resize((size, size), Image.LANCZOS)
        pixels = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0)[None]
        reference = load(torch.float32, torch.device(os.environ.get("REFERENCE_DEVICE", "mps")))
        latent = reference.encode(pixels).float().cpu()
        del reference
        for name in [n for n in os.environ.get("DTYPES", "fp32,fp16,bf16").split(",") if n != "fp32"]:
            vae = load(DTYPES[name])
            encoded, seconds = timed(lambda: vae.encode(pixels))
            encoded = encoded.float().cpu()
            rel = float((encoded - latent).norm() / latent.norm())
            print("RESULT " + json.dumps(dict(kind="encode", dtype=name, seconds=round(seconds, 2),
                  nonfinite=int((~torch.isfinite(encoded)).sum()), rel_l2_vs_fp32=round(rel, 6))), flush=True)
            del vae, encoded
            mm.soft_empty_cache()
    del default
    mm.soft_empty_cache()

    ref = None
    for name in os.environ.get("DTYPES", "fp32,fp16,bf16").split(","):
        vae = load(DTYPES[name])
        out, seconds = timed(lambda: vae.decode(latent))
        out = out.float().cpu()
        row = dict(kind="decode", dtype=name, latent=list(latent.shape), out=list(out.shape),
                   seconds=round(seconds, 2), nonfinite=int((~torch.isfinite(out)).sum()))
        if ref is None:
            ref = out
        else:
            diff = (out.clamp(0, 1) - ref.clamp(0, 1)).abs()
            mse = float((diff ** 2).mean())
            row.update(max_abs=round(float(diff.max()), 5), mean_abs=round(float(diff.mean()), 6),
                       psnr_db=round(10 * np.log10(1.0 / max(mse, 1e-12)), 2),
                       over_1_255=round(float((diff > 1 / 255).float().mean()), 6))
        print("RESULT " + json.dumps(row), flush=True)
        del vae, out
        mm.soft_empty_cache()
