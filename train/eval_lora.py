"""Render the same prompts and seeds without and with one or more LoRAs through ComfyUI's nodes
(LoraLoaderModelOnly, the path bench/run_krea2.py and bench/run_qwen21.py use with LORA), then save
a comparison sheet and per-image numbers: PSNR against the no-LoRA image (and between LoRAs),
and the share of pixels in the trained palette.

  bench/with_gpu.sh train/.venv/bin/python train/eval_lora.py --unet krea2_turbo_int8_convrot.safetensors \
      --lora krea2_rwbflat_raw_s500.safetensors --steps 8 --cfg 1 --tag turbo
  ... --arch qwen21 --unet qwen_image_2.1_int8_convrot.safetensors --stack-lora <viggle> --sigmas viggle \
      --lora A.safetensors B.safetensors --palette cyan --res 1024 --tag viggle
"""
import argparse
import json
import math
import os
import time

import comfy_boot as cb
import numpy as np
import torch
from PIL import Image, ImageDraw

PROMPTS = {
    "rwb": [  # held out: none of these subjects is in the dataset
        "rwbflat style, a horse galloping on a beach",
        "rwbflat style, a teapot on a wooden table",
        "rwbflat style, a hot air balloon over a valley",
        "a portrait photo of an old fisherman",  # no trigger: how much does the style bleed?
    ],
    "cyan": [
        "cyanoprint style, a horse galloping on a beach",
        "cyanoprint style, a teapot on a wooden table",
        "cyanoprint style, a hot air balloon over a valley",
        "a portrait photo of an old fisherman",
    ],
}
TE = {"krea2": ("qwen3vl_4b_bf16.safetensors", "krea2"), "qwen21": ("qwen3vl_8b_int8_convrot.safetensors", "qwen_image")}
VAE = {"krea2": "qwen_image_vae.safetensors", "qwen21": "qwen_image_2.1_vae_bf16.safetensors"}


def palette_share(arr, palette):
    """rwb: near black, near white, or saturated red. cyan: blue-dominant pixels or near-white
    paper (a cyanotype's two tones)."""
    x = arr.astype(np.float32) / 255
    mx, mn = x.max(-1), x.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    if palette == "cyan":
        blue = (b >= r + 0.08) & (b >= g - 0.02)
        white = (mn > 0.75) & (sat < 0.12)
        return float((blue | white).mean())
    black = mx < 0.2
    white = (mn > 0.8) & (sat < 0.15)
    red = (r == mx) & (sat > 0.45) & (mx > 0.3) & (np.abs(g - b) < 0.35 * (mx - mn) + 0.02)
    return float((black | white | red).mean())


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def dynamic_shift(model, w, h, max_shift=1.15, base_shift=0.5):
    """comfy_extras.nodes_model_advanced.ModelSamplingFlux.patch, inlined (comfy_extras is not loaded)."""
    import comfy.model_sampling as ms
    m = model.clone()
    slope = (max_shift - base_shift) / (4096 - 256)
    mu = (w * h / (8 * 8 * 2 * 2)) * slope + base_shift - slope * 256

    class ModelSamplingAdvanced(ms.ModelSamplingFlux, ms.CONST):
        pass

    sampling = ModelSamplingAdvanced(model.model.model_config)
    sampling.set_parameters(shift=mu)
    m.add_object_patch("model_sampling", sampling)
    print(f"dynamic shift mu {mu:.3f}")
    return m


def viggle_sigmas(w, h):
    tokens = (h // 16) * (w // 16)
    mu = 0.5 + (0.9 - 0.5) * (tokens - 256) / (8192 - 256)
    return [math.exp(mu) / (math.exp(mu) + (1 / t - 1)) for t in (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)] + [0.0]


def encode(arch, clip, text):
    if arch == "qwen21":
        return clip.encode_from_tokens_scheduled(clip.tokenize(text, images=[], keep_vision=True, prevent_empty_text=True))
    return cb.N["CLIPTextEncode"]().encode(clip, text)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=list(TE), default="krea2")
    p.add_argument("--unet", required=True)
    p.add_argument("--mps-direct-load", action="store_true",
                   help="Opt into the existing UNETLoader direct MPS path for supported INT8 ConvRot files")
    p.add_argument("--lora", required=True, nargs="+", help="file names inside a loras folder; one column each")
    p.add_argument("--stack-lora", default=None, help="a LoRA applied to every column first, e.g. a step-reduction LoRA")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--sigmas", default=None, help="'viggle': Viggle's 6-step sigmas via SamplerCustomAdvanced")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tag", required=True)
    p.add_argument("--column", choices=("all", "base", "adapter"), default="all",
                   help="Render one baseline or adapter image per guarded process; requires one prompt and one LoRA")
    p.add_argument("--palette", choices=list(PROMPTS), default="rwb")
    p.add_argument("--prompts", nargs="*", default=None)
    p.add_argument("--neg", default=None,
                   help="negative prompt to encode (e.g. an empty string) instead of ConditioningZeroOut; "
                        "matters only when cfg > 1")
    p.add_argument("--dynamic-shift", action="store_true",
                   help="Krea 2: resolution-dependent shift as ComfyUI's ModelSamplingFlux node (max 1.15, base 0.5)")
    a = p.parse_args()
    prompts = a.prompts or PROMPTS[a.palette]
    if a.column != "all" and (len(prompts) != 1 or len(a.lora) != 1):
        p.error("A single column requires exactly one --prompts value and one --lora")
    if a.sigmas:
        cb.load_extra_nodes()
    N, call = cb.N, cb.call
    out_dir = f"{cb.ROOT}/out"
    T = {}
    t0 = time.perf_counter()
    with torch.inference_mode():
        clip = N["CLIPLoader"]().load_clip(TE[a.arch][0], TE[a.arch][1], "default")[0]
        conds = [encode(a.arch, clip, pr) for pr in prompts]
        neg_enc = None if a.neg is None else encode(a.arch, clip, a.neg)
        del clip
        cb.mm.unload_all_models()
        cb.mm.soft_empty_cache()
        load_options = {"mps_direct_load": True} if a.mps_direct_load else {}
        base = N["UNETLoader"]().load_unet(a.unet, "default", **load_options)[0]
        if a.dynamic_shift:
            base = dynamic_shift(base, a.res, a.res)
        if a.stack_lora:
            base = N["LoraLoaderModelOnly"]().load_lora_model_only(base, a.stack_lora, 1.0)[0]
        key_map = cb.comfy.lora.model_lora_keys_unet(base.model, {})
        mapping = {}
        for name in a.lora:  # every saved key must map onto a layer, or ComfyUI silently drops it
            sd = cb.comfy.utils.load_torch_file(cb.folder_paths.get_full_path("loras", name))
            mapping[name] = {"layers_in_file": len({k.rsplit(".lora_", 1)[0] for k in sd if ".lora_" in k}),
                             "layers_mapped": len(cb.comfy.lora.load_lora(sd, key_map, log_missing=False))}
        vae = N["VAELoader"]().load_vae(VAE[a.arch])[0]
        T["load"] = round(time.perf_counter() - t0, 1)
        columns = [("none", base)] if a.column in ("all", "base") else []
        if a.column in ("all", "adapter"):
            columns += [(n, N["LoraLoaderModelOnly"]().load_lora_model_only(base, n, a.strength)[0]) for n in a.lora]
        imgs = {}
        for cname, m in columns:  # one model at a time: switching patchers per image re-patches the weights
            for k, pos in enumerate(conds):
                if a.arch == "qwen21":
                    latent = {"samples": torch.zeros([1, 64, a.res // 16, a.res // 16])}
                else:
                    latent = N["EmptyLatentImage"]().generate(a.res, a.res, 1)[0]
                neg = neg_enc if neg_enc is not None else N["ConditioningZeroOut"]().zero_out(pos)[0]
                t1 = time.perf_counter()
                if a.sigmas == "viggle":
                    guider = (call("CFGGuider", model=m, positive=pos, negative=neg, cfg=a.cfg)[0] if a.cfg != 1.0
                              else call("BasicGuider", model=m, conditioning=pos)[0])
                    out = call("SamplerCustomAdvanced", noise=call("RandomNoise", noise_seed=a.seed)[0], guider=guider,
                               sampler=call("KSamplerSelect", sampler_name="euler")[0],
                               sigmas=torch.tensor(viggle_sigmas(a.res, a.res)), latent_image=latent)[0]
                else:
                    out = N["KSampler"]().sample(m, a.seed, a.steps, a.cfg, "euler", "simple", pos, neg, latent, 1.0)[0]
                img = N["VAEDecode"]().decode(vae, out)[0]
                if not torch.isfinite(img).all():
                    raise FloatingPointError(f"Non-finite decoded image: {cname}, prompt {k}")
                T.setdefault("sample_s", []).append(round(time.perf_counter() - t1, 1))
                imgs[(cname, k)] = (img[0, ..., :3].float().cpu().numpy() * 255).clip(0, 255).astype("uint8")
                print(cname, k, T["sample_s"][-1], flush=True)
    stats = []
    rendered_loras = [name for name, _ in columns if name != "none"]
    for k, pr in enumerate(prompts):
        b = imgs.get(("none", k))
        st = {"prompt": pr}
        if b is not None:
            st["palette_none"] = round(palette_share(b, a.palette), 3)
        for n in rendered_loras:
            if b is not None:
                st[f"psnr_vs_none[{n}]"] = round(psnr(imgs[(n, k)], b), 2)
            st[f"palette[{n}]"] = round(palette_share(imgs[(n, k)], a.palette), 3)
        for i, n1 in enumerate(rendered_loras):
            for n2 in rendered_loras[i + 1:]:
                st[f"psnr[{n1}|{n2}]"] = round(psnr(imgs[(n1, k)], imgs[(n2, k)]), 2)
        stats.append(st)
        print(st, flush=True)
    stem = f"{out_dir}/train-{os.path.splitext(a.lora[0])[0]}-{a.tag}"
    for (cname, k), arr in imgs.items():
        Image.fromarray(arr).save(f"{stem}-p{k}-{'base' if cname == 'none' else os.path.splitext(cname)[0]}.png")
    s = a.res
    sheet = Image.new("RGB", (len(columns) * s, len(prompts) * (s + 24)), "white")
    d = ImageDraw.Draw(sheet)
    for k in range(len(prompts)):
        y = k * (s + 24)
        d.text((4, y + 4), f"{prompts[k][:80]}   |   columns: " + ", ".join(os.path.splitext(c)[0] for c, _ in columns),
               fill="black")
        for j, (cname, _) in enumerate(columns):
            sheet.paste(Image.fromarray(imgs[(cname, k)]), (j * s, y + 24))
    sheet.save(f"{stem}-sheet.png")
    res = {"tag": a.tag, "arch": a.arch, "unet": a.unet, "stack_lora": a.stack_lora, "loras": a.lora, "mapping": mapping,
           "column": a.column, "mps_direct_load": a.mps_direct_load,
           "strength": a.strength, "steps": a.steps, "sigmas": a.sigmas, "cfg": a.cfg, "neg": a.neg, "res": a.res,
           "dynamic_shift": a.dynamic_shift, "seed": a.seed, "times": T, "stats": stats, "sheet": f"{stem}-sheet.png"}
    print("RESULT " + json.dumps(res))
    open(f"{out_dir}/results.jsonl", "a").write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
