#!/usr/bin/env python3
"""Download and hash-verify the WAN 2.2 A14B image-to-video sources.

The fp16 experts are the conversion source for INT8 ConvRot, matching how the
installed T2V experts are used; the Lightning I2V 4-step LoRA pair is the speed
recipe. Everything lands on the external model root, never internal storage.
"""
import argparse
import os
from pathlib import Path

from hf_prepare import prepare

REPACK_REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
REPACK_REVISION = "ee6f4a40737a995bf5818954cfce6d59443b0f04"
EXPERTS = tuple(f"split_files/diffusion_models/wan2.2_i2v_{role}_noise_14B_fp16.safetensors" for role in ("high", "low"))
LIGHTNING_REPO = "lightx2v/Wan2.2-Lightning"
LIGHTNING_REVISION = "18bccf8884ec0a078eed79785eb4ef13ea16ce1e"
LIGHTNING = tuple(f"Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/{role}_noise_model.safetensors" for role in ("high", "low"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--only", action="append", choices=("experts", "lightning"))
    args = parser.parse_args()
    root = Path(os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data")))
    os.environ.setdefault("HF_HOME", str(root / "hf-cache"))
    sources = {"experts": (REPACK_REPO, REPACK_REVISION, root / "models/wan/sources/i2v-a14b-fp16", EXPERTS),
               "lightning": (LIGHTNING_REPO, LIGHTNING_REVISION, root / "models/wan/loras/lightx2v-i2v-a14b", LIGHTNING)}
    for name in args.only or tuple(sources):
        repo, revision, directory, files = sources[name]
        directory.mkdir(parents=True, exist_ok=True)
        prepare(repo, revision, directory, files, args.verify_only)


if __name__ == "__main__":
    main()
