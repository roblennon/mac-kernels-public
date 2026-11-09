"""Synthetic style dataset from Qwen-Image 2.1 INT8 + the Viggle 6-step LoRA (Viggle's sigmas, cfg 1),
through the same ComfyUI nodes as bench/run_qwen21.py SIGMAS=viggle. 1024x1024, so one set
serves 512px (resized) and 1024px training.

Each image is rendered from "<subject>, <STYLE>" and captioned "<TRIGGER>, <subject>".

  bench/with_gpu.sh train/.venv/bin/python train/make_dataset_qwen.py
"""
import json
import math
import os
import time

import comfy_boot as cb
import torch

from make_dataset import SUBJECTS

OUT = os.environ.get("DATASET", f"{cb.ROOT}/train/datasets/cyanoprint")
RES = int(os.environ.get("RES", 1024))
TRIGGER = "cyanoprint style"
STYLE = ("cyanotype sun print, only prussian blue and white, monochrome blue photographic print on "
         "textured watercolor paper, soft white highlights, deep blue shadows, vintage botanical cyanotype look")
EXTRA = ["a lighthouse keeper's lantern", "a pair of running shoes", "a cactus in a clay pot", "an old typewriter"]
VIGGLE = "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors"


def viggle_sigmas(w, h):
    """bench/run_qwen21.py SIGMAS=viggle: Viggle's raw nodes through the dynamic exponential shift."""
    tokens = (h // 16) * (w // 16)
    mu = 0.5 + (0.9 - 0.5) * (tokens - 256) / (8192 - 256)
    return [math.exp(mu) / (math.exp(mu) + (1 / t - 1)) for t in (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)] + [0.0]


def main():
    cb.load_extra_nodes()
    os.makedirs(OUT, exist_ok=True)
    N, call = cb.N, cb.call
    subjects = SUBJECTS + EXTRA
    t0 = time.perf_counter()
    with torch.inference_mode():
        clip = N["CLIPLoader"]().load_clip("qwen3vl_8b_int8_convrot.safetensors", "qwen_image", "default")[0]
        conds = [clip.encode_from_tokens_scheduled(clip.tokenize(f"{s}, {STYLE}", images=[], keep_vision=True,
                                                                prevent_empty_text=True)) for s in subjects]
        print("cond extras:", {k: getattr(v, "shape", v) for k, v in conds[0][0][1].items()}, flush=True)
        del clip
        cb.mm.unload_all_models()
        cb.mm.soft_empty_cache()
        model = N["UNETLoader"]().load_unet("qwen_image_2.1_int8_convrot.safetensors", "default")[0]
        model = N["LoraLoaderModelOnly"]().load_lora_model_only(model, VIGGLE, 1.0)[0]
        vae = N["VAELoader"]().load_vae("qwen_image_2.1_vae_bf16.safetensors")[0]
        sig = torch.tensor(viggle_sigmas(RES, RES))
        from PIL import Image
        records = []
        for i, (subject, pos) in enumerate(zip(subjects, conds)):
            latent = {"samples": torch.zeros([1, 64, RES // 16, RES // 16])}
            guider = call("BasicGuider", model=model, conditioning=pos)[0]
            out = call("SamplerCustomAdvanced", noise=call("RandomNoise", noise_seed=2000 + i)[0], guider=guider,
                       sampler=call("KSamplerSelect", sampler_name="euler")[0], sigmas=sig, latent_image=latent)[0]
            img = N["VAEDecode"]().decode(vae, out)[0]
            arr = (img[0, ..., :3].float().cpu().numpy() * 255).clip(0, 255).astype("uint8")
            name = f"{i:02d}.png"
            Image.fromarray(arr).save(f"{OUT}/{name}")
            records.append({"file": name, "caption": f"{TRIGGER}, {subject}", "render_prompt": f"{subject}, {STYLE}",
                            "seed": 2000 + i})
            print(f"{name} {time.perf_counter() - t0:.0f}s", flush=True)
    json.dump({"trigger": TRIGGER, "style": STYLE, "model": "qwen_image_2.1_int8_convrot + " + VIGGLE,
               "sigmas": "viggle 6-step", "cfg": 1.0, "size": [RES, RES], "images": records},
              open(f"{OUT}/dataset.json", "w"), indent=1)


if __name__ == "__main__":
    main()
