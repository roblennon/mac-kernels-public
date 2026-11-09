# Overnight experiment: Turbo Qwen edit with subject + object-only garment reference.
# Uses bounded INT8 loader and ComfyUI's Qwen 2.1 edit encoding.
"""Headless multi-reference Qwen Turbo edit for the overnight brief.

  .venv/bin/python bench/gpu_runner.py --monitor --seconds 900 -- \\
    .venv/bin/python bench/overnight_qwen_multiref.py
"""
from __future__ import annotations

import asyncio
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data")))
OUT = ROOT / "out" / "overnight-20261011" / "qwen-multiref"
BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
CODE = os.environ.get("COMFY_DIR") or str(REPO / "vendor" / "ComfyUI")
IMAGE1 = Path(os.environ.get(
    "QWEN_EDIT_IMAGE1", str(OUT / "image1-subject.jpg")))
IMAGE2 = Path(os.environ.get(
    "QWEN_EDIT_IMAGE2", str(OUT / "image2-garment.png")))
PROMPT = os.environ.get(
    "QWEN_EDIT_PROMPT",
    "Replace the clothing in <image1> with the pale blue nurse apron with red "
    "cross and red ruffle hem from <image2>. Keep the person, pose, face, hair, "
    "and background from <image1>.")
SEED = int(os.environ.get("QWEN_EDIT_SEED", "7"))
RESOLUTION = int(os.environ.get("QWEN_EDIT_RESOLUTION", "1024"))
TURBO_UNET = "qwen_image_2.1_turbo_int8_convrot.safetensors"
TE = "qwen3vl_8b_int8_convrot.safetensors"
TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0]


def load_image(path: Path):
    import torch
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not IMAGE1.is_file() or not IMAGE2.is_file():
        raise SystemExit(f"Missing inputs: {IMAGE1} / {IMAGE2}")

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, CODE)
    os.chdir(CODE)
    sys.argv = [sys.argv[0], "--use-pytorch-cross-attention"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import folder_paths
    import comfy.model_management as mm
    import comfy.utils
    import nodes

    for key in ("diffusion_models", "text_encoders", "vae", "loras"):
        folder_paths.add_model_folder_path(key, f"{ROOT}/models/qwen21/{key}")
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    N = nodes.NODE_CLASS_MAPPINGS
    from bench.qwen_comfy import BenchQwenEditPrepare

    def call(name, **kw):
        cls = N[name]
        if hasattr(cls, "define_schema"):
            out = cls.execute(**kw)
            return out.result if hasattr(out, "result") else out.args
        return getattr(cls(), cls.FUNCTION)(**kw)

    t0 = time.perf_counter()
    laps: dict[str, float] = {}

    def lap(name: str) -> None:
        nonlocal t0
        torch.mps.synchronize()
        laps[name] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        print("LAP", name, laps[name], flush=True)

    image_1 = load_image(IMAGE1)
    image_2 = load_image(IMAGE2)
    vae = N["VAELoader"]().load_vae("qwen_image_2.1_vae_bf16.safetensors")[0]
    lap("load_vae")

    with torch.inference_mode():
        model, positive, latent = BenchQwenEditPrepare().prepare(
            TURBO_UNET, TE, vae, image_1, PROMPT, RESOLUTION, image_2=image_2)
        lap("prepare")
        model = call("QwenImage21Cache", model=model, device="auto", dtype="default")[0]
        guider = call("BasicGuider", model=model, conditioning=positive)[0]
        noise = call("RandomNoise", noise_seed=SEED)[0]
        sampler = call("KSamplerSelect", sampler_name="euler")[0]
        step_t: list[float] = []
        orig = comfy.utils.ProgressBar.update_absolute

        def upd(self, *a, **k):
            torch.mps.synchronize()
            step_t.append(time.perf_counter())
            return orig(self, *a, **k)

        comfy.utils.ProgressBar.update_absolute = upd
        out = call(
            "SamplerCustomAdvanced",
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=torch.tensor(TURBO_SIGMAS),
            latent_image=latent,
        )[0]
        lap("sample")
        peak = torch.mps.driver_allocated_memory() / 2**30
        img = N["VAEDecode"]().decode(vae, out)[0]
        lap("decode")

    steps = [round(b - a, 2) for a, b in zip(step_t, step_t[1:])]
    arr = (img[0].float().nan_to_num(0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    out_path = OUT / f"edit-turbo-seed{SEED}.png"
    Image.fromarray(arr[..., :3]).save(out_path)
    receipt = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "tag": "overnight-qwen-multiref",
        "image_1": str(IMAGE1),
        "image_2": str(IMAGE2),
        "prompt": PROMPT,
        "seed": SEED,
        "resolution": RESOLUTION,
        "unet": TURBO_UNET,
        "laps_s": laps,
        "step_s": steps[:10],
        "step_s_median": sorted(steps)[len(steps) // 2] if steps else None,
        "mps_alloc_gb_after_sample": round(peak, 2),
        "output": str(out_path),
        "nan": bool(torch.isnan(img).any()),
        "mean": float(img.float().nan_to_num(0).mean()),
        "std": float(img.float().nan_to_num(0).std()),
    }
    (OUT / "edit-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (ROOT / "out" / "results.jsonl").open("a").write(json.dumps(receipt) + "\n")
    print("RESULT " + json.dumps(receipt), flush=True)
    del model, positive, latent, out, img, vae
    mm.unload_all_models()
    mm.soft_empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
