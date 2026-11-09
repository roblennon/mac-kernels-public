"""Synthetic style dataset from Krea 2 Turbo INT8 through ComfyUI (same nodes as bench/run_krea2.py).

Each image is rendered from "<subject>, <STYLE>"; its caption is "<TRIGGER>, <subject>". The
style words live only in the pixels, so a LoRA has to learn them and tie them to the trigger,
and the effect on a held-out prompt is unambiguous.

  train/.venv/bin/python train/make_dataset.py   (through bench/with_gpu.sh)
"""
import json
import os
import time

import comfy_boot as cb
import torch

OUT = os.environ.get("DATASET", f"{cb.ROOT}/train/datasets/rwbflat")
W = H = int(os.environ.get("RES", 512))
TRIGGER = "rwbflat style"
STYLE = ("flat vector art in only red, white and black, bold simple geometric shapes, "
         "thick black outlines, solid color fills, no gradients, no shading, poster graphic")
SUBJECTS = [
    "a fox sitting in a meadow", "a lighthouse on a rocky cliff", "a vintage bicycle", "a steaming cup of coffee",
    "a mountain range at dawn", "an owl perched on a branch", "a city skyline", "a sailboat on the sea",
    "an acoustic guitar", "a cat sleeping on a windowsill", "a rocket launching into space", "an oak tree in a field",
    "a classic sports car", "a small cottage with a chimney", "a whale breaching the ocean", "a hummingbird near a flower",
    "a friendly robot waving", "an umbrella in the rain", "a castle on a hill", "a bowl of ramen",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    N = cb.N
    t0 = time.perf_counter()
    with torch.inference_mode():
        clip = N["CLIPLoader"]().load_clip("qwen3vl_4b_bf16.safetensors", "krea2", "default")[0]
        conds = [N["CLIPTextEncode"]().encode(clip, f"{s}, {STYLE}")[0] for s in SUBJECTS]
        del clip
        cb.mm.unload_all_models()
        cb.mm.soft_empty_cache()
        model = N["UNETLoader"]().load_unet("krea2_turbo_int8_convrot.safetensors", "default")[0]
        vae = N["VAELoader"]().load_vae("qwen_image_vae.safetensors")[0]
        from PIL import Image
        records = []
        for i, (subject, pos) in enumerate(zip(SUBJECTS, conds)):
            neg = N["ConditioningZeroOut"]().zero_out(pos)[0]
            latent = N["EmptyLatentImage"]().generate(W, H, 1)[0]
            out = N["KSampler"]().sample(model, 1000 + i, 8, 1.0, "euler", "simple", pos, neg, latent, 1.0)[0]
            img = N["VAEDecode"]().decode(vae, out)[0]
            arr = (img[0].float().cpu().numpy() * 255).clip(0, 255).astype("uint8")
            name = f"{i:02d}.png"
            Image.fromarray(arr).save(f"{OUT}/{name}")
            records.append({"file": name, "caption": f"{TRIGGER}, {subject}",
                            "render_prompt": f"{subject}, {STYLE}", "seed": 1000 + i})
            print(f"{name} {time.perf_counter() - t0:.0f}s", flush=True)
    json.dump({"trigger": TRIGGER, "style": STYLE, "model": "krea2_turbo_int8_convrot", "steps": 8, "cfg": 1.0,
               "size": [W, H], "images": records}, open(f"{OUT}/dataset.json", "w"), indent=1)


if __name__ == "__main__":
    main()
