"""Which Qwen 2.1 VAE encodes at 1024px go non-finite, and is it the image, the order, or fp16?
VAE_FP16=1|0 picks the VAE dtype (comfy_boot)."""
import sys
import comfy_boot as cb
import numpy as np
import torch
from PIL import Image
D = f"{cb.ROOT}/train/datasets/cyanoprint"
order = [int(x) for x in sys.argv[1:]] or [7, 0, 8, 1, 7]
with torch.inference_mode():
    vae = cb.N["VAELoader"]().load_vae("qwen_image_2.1_vae_bf16.safetensors")[0]
    print("vae dtype", vae.vae_dtype, flush=True)
    for i in order:
        px = torch.from_numpy(np.asarray(Image.open(f"{D}/{i:02d}.png").convert("RGB")).copy()).float() / 255
        lat = cb.N["VAEEncode"]().encode(vae, px[None])[0]["samples"]
        bad = sum(int(torch.isnan(p).sum()) for p in vae.first_stage_model.parameters())
        print(f"img {i}: nan {int(torch.isnan(lat).sum())} absmax {float(lat.nan_to_num().abs().max()):.2f} nan-params {bad}", flush=True)
