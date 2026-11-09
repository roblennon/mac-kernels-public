"""LoRA training for Krea 2 and Qwen-Image 2.1 on frozen INT8/W4A8/W6A8 ConvRot bases, on Apple GPUs
(M1 Max 32 GB).

The base model, text encoder and VAE load through ComfyUI exactly as for inference
(bench/run_krea2.py, bench/run_qwen21.py). Every supported quantized linear then runs through
comfy_kitchen's Metal weight-only kernel in the forward and its input-gradient kernel in the
backward (comfy_kitchen.backends.mps.autograd and packed_autograd); decoded weight
chunks are transient and bounded. A bf16 base (for the INT8-vs-bf16 A/B) trains through ComfyUI's own linears instead.

Recipe (sources in docs/research/2026-10-02-lora-training-on-quantized-bases.md):
- Rectified-flow loss as in diffusers' Krea 2 / Qwen-Image scripts, ai-toolkit and musubi-tuner:
  x_t = (1 - s) x0 + s e, the model predicts v, target e - x0, unweighted MSE in fp32.
- s = time-shifted logit-normal: t ~ sigmoid(N(0, 1)) (ai-toolkit's default timestep_type
  "sigmoid", toolkit/config_modules.py:557), then s = e^mu / (e^mu + 1/t - 1).
  Krea 2: mu = 1.15, the model's fixed ComfyUI shift (milestone 1). Qwen-Image 2.1: mu by
  resolution, 0.5 at 256 tokens to 0.9 at 8192 (ai-toolkit qwen_image_2.py scheduler_config,
  ComfyUI supported_models.QwenImage21), 0.54 at 512px and 0.69 at 1024px.
- LoRA rank 16 on each block's attention and MLP linears, alpha = rank, A ~ N(0, 1/r) and B = 0
  (PEFT "gaussian" init, as the diffusers Krea 2 script), so step 0 is the base model exactly.
  fp32 master weights, cast to the activation dtype inside forward (mflux #740).
- Fused AdamW on the fp32 LoRA weights. Krea 2 computes in fp16 with GradScaler; Qwen-Image 2.1
  in bf16 (ComfyUI lists no fp16 for it), which needs no loss scaling.
- Latents and text embeddings are cached once; the encoders are off the GPU while training.
- Non-reentrant gradient checkpointing per transformer block.
- --chunked-attn routes attention through comfy_kitchen's memory-bounded backward
  (backends/mps/attention_autograd.py) via ComfyUI's optimized_attention_override.

Saves LoRAs in the diffusers key layout ComfyUI maps for each model (the layout of Krea's
official LoRAs and of the Viggle Qwen 2.1 LoRA). --resume continues from the run's state.pt, and
--stop-after N ends a session after N steps, so long runs fit GPU-lock windows.

Run through the GPU lock: bench/with_gpu.sh train/.venv/bin/python train/train_lora.py ...
"""
import argparse
import hashlib
import json
import math
import os
import tempfile
import threading
import time
from contextlib import contextmanager

_INVOCATION_CWD = os.getcwd()  # comfy_boot changes cwd while importing ComfyUI.

import comfy_boot as cb
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from training_config import (cache_filename, normalize_buckets, parse_config_args,
                             prepare_image, shape_policy, validate_bucket_cache)
from training_preview import (add_preview_arguments, decode_preview, preview_due,
                              validate_preview_arguments, write_preview)

ARCHS = {
    "krea2": {
        "te": ("qwen3vl_4b_bf16.safetensors", "krea2"),
        "vae": "qwen_image_vae.safetensors",
        "blocks": "blocks",
        "targets": ("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down"),
        "dtype": "fp16",
    },
    "qwen21": {
        "te": ("qwen3vl_8b_int8_convrot.safetensors", "qwen_image"),
        "vae": "qwen_image_2.1_vae_bf16.safetensors",
        "blocks": "transformer_blocks",
        "targets": ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_up", "img_mlp.out"),
        "dtype": "bf16",
    },
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=list(ARCHS), default="krea2")
    p.add_argument("--dataset", default=f"{cb.ROOT}/train/datasets/rwbflat")
    p.add_argument("--cache-file", default=None, help="use an existing cache file; never build or replace it")
    p.add_argument("--unet", default="krea2_raw_int8_convrot.safetensors")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--buckets", type=normalize_buckets, default=None,
                   help="explicit WIDTHxHEIGHT buckets, comma-separated; uses center crop")
    p.add_argument("--no-buckets", dest="buckets", action="store_const", const=None, default=argparse.SUPPRESS)
    p.add_argument("--out", default=f"{cb.ROOT}/train/runs/rwbflat_raw")
    p.add_argument("--name", default="krea2_rwbflat_raw")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default=None, help="default per arch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--state-every", type=int, default=25, help="save resumable optimizer/RNG state every N steps")
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--checkpoint", dest="no_checkpoint", action="store_false", default=argparse.SUPPRESS)
    p.add_argument("--chunked-attn", action=argparse.BooleanOptionalAction, default=False,
                   help="memory-bounded attention backward")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False, help="continue from OUT/state.pt")
    p.add_argument("--stop-after", type=int, default=0, help="end this session after N steps (state saved)")
    p.add_argument("--profile-steps", type=int, default=0, help="run N steps, print timings, exit")
    add_preview_arguments(p)
    args = parse_config_args(p, base_dir=_INVOCATION_CWD)
    validate_preview_arguments(args, p)
    for name in ("state_every", "save_every", "val_every", "log_every"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.resume and not os.path.isfile(f"{args.out}/state.pt"):
        p.error(f"--resume requires {args.out}/state.pt")
    if not args.resume and not args.profile_steps and os.path.exists(f"{args.out}/state.pt"):
        p.error(f"{args.out}/state.pt already exists; use --resume or a new --out directory")
    return args


# ---------------------------------------------------------------- caching

def encode_text(arch, clip, text):
    if arch == "qwen21":  # TextEncodeQwenImage21 without reference images
        tokens = clip.tokenize(text, images=[], keep_vision=True, prevent_empty_text=True)
        return clip.encode_from_tokens_scheduled(tokens)[0][0]
    return cb.N["CLIPTextEncode"]().encode(clip, text)[0][0][0]


def build_cache(args, spec, path):
    """Text embeddings and VAE latents for every image under the chosen shape policy."""
    import numpy as np
    from PIL import Image
    meta = json.load(open(f"{args.dataset}/dataset.json"))
    N = cb.N
    with torch.inference_mode():
        # Latents first, in a fresh process: one 1024px Qwen 2.1 build that ran after the 9.4 GB text
        # encoder returned all-NaN latents from the 8th image on (not reproducible in isolation).
        vae = N["VAELoader"]().load_vae(spec["vae"])[0]
        lats, samples = [], []
        for r in meta["images"]:
            with Image.open(f"{args.dataset}/{r['file']}") as source:
                img, geometry = prepare_image(source, args.res, getattr(args, "buckets", None))
            px = torch.from_numpy(np.asarray(img).copy()).float() / 255
            lat = N["VAEEncode"]().encode(vae, px[None])[0]["samples"].cpu()
            if not torch.isfinite(lat).all():
                print(f"non-finite latent for {r['file']}; reloading the VAE and retrying", flush=True)
                del vae
                cb.mm.unload_all_models()
                cb.mm.soft_empty_cache()
                vae = N["VAELoader"]().load_vae(spec["vae"])[0]
                lat = N["VAEEncode"]().encode(vae, px[None])[0]["samples"].cpu()
                if not torch.isfinite(lat).all():
                    raise RuntimeError(f"non-finite latent for {r['file']} at {args.res}px")
            lats.append(lat)
            samples.append(dict(geometry, file=r["file"], latent_shape=list(lat.shape)))
        del vae
        cb.mm.unload_all_models()
        cb.mm.soft_empty_cache()
        clip = N["CLIPLoader"]().load_clip(spec["te"][0], spec["te"][1], "default")[0]
        conds = [encode_text(args.arch, clip, r["caption"]).cpu() for r in meta["images"]]
        del clip
        cb.mm.unload_all_models()
        cb.mm.soft_empty_cache()
    cache = {"captions": [r["caption"] for r in meta["images"]], "conds": conds, "latents": lats}
    policy = shape_policy(args.res, getattr(args, "buckets", None))
    if policy["mode"] == "buckets":
        cache.update(shape_policy=policy, samples=samples)
    validate_cache(cache)
    validate_bucket_cache(cache, policy)
    with atomic_checkpoint(path) as tmp:
        torch.save(cache, tmp)
    return cache


# ---------------------------------------------------------------- model surgery

class LoRA(torch.nn.Module):
    def __init__(self, in_f, out_f, rank, alpha, generator):
        super().__init__()
        self.down = torch.nn.Parameter(torch.randn(rank, in_f, generator=generator) / rank)
        self.up = torch.nn.Parameter(torch.zeros(out_f, rank))
        self.scale = alpha / rank

    def forward(self, x):
        return F.linear(F.linear(x, self.down.to(x.dtype)), self.up.to(x.dtype)) * self.scale


def patch_model(dm, spec, rank, seed, device):
    """Route INT8/W4A8/W6A8 linears through input-backward kernels and attach target LoRAs.
    Returns ({comfy module name: LoRA}, supported quantized layer count)."""
    from comfy.quant_ops import AsymW4A8Int8Layout, QuantizedTensor, TensorWiseINT8Layout
    from comfy_kitchen.backends.mps.autograd import int8_linear_trainable
    from comfy_kitchen.backends.mps.packed_autograd import w4a8_linear_trainable

    g = torch.Generator().manual_seed(seed)
    loras, n_quantized = {}, 0
    prefix = spec["blocks"] + "."
    for name, mod in dm.named_modules():
        w = getattr(mod, "weight", None)
        if w is None or not hasattr(mod, "in_features"):
            continue
        layout = w._layout_cls if isinstance(w, QuantizedTensor) else None
        int8 = layout == "TensorWiseINT8Layout"
        packed = layout == "AsymW4A8Int8Layout"
        lora = None
        if name.startswith(prefix) and name.split(".", 2)[2] in spec["targets"]:
            lora = LoRA(mod.in_features, mod.out_features, rank, rank, g).to(device)
            loras[name] = lora
        if int8:
            n_quantized += 1

            def fwd(x, *a, _mod=mod, _lora=lora, **k):
                qdata, scale = TensorWiseINT8Layout.get_plain_tensors(_mod.weight)
                p = _mod.weight._params
                out = int8_linear_trainable(x, qdata, scale, _mod.bias, getattr(p, "convrot", False),
                                            getattr(p, "convrot_groupsize", 256))
                return out if _lora is None else out + _lora(x)

            mod.forward = fwd
        elif packed:
            n_quantized += 1

            def fwd(x, *a, _mod=mod, _lora=lora, **k):
                qdata, s_rel, s_channel, correction, codebook = AsymW4A8Int8Layout.get_plain_tensors(_mod.weight)
                p = _mod.weight._params
                out = w4a8_linear_trainable(
                    x, qdata, s_rel, s_channel, codebook, correction, _mod.bias,
                    p.group_size, p.convrot_groupsize,
                )
                return out if _lora is None else out + _lora(x)

            mod.forward = fwd
        elif lora is not None:
            inner = mod.forward

            def fwd(x, *a, _inner=inner, _lora=lora, **k):
                return _inner(x, *a, **k) + _lora(x)

            mod.forward = fwd
    return loras, n_quantized


def patch_arch_for_training(arch):
    """Out-of-place versions of ops that write into a tensor autograd still needs."""
    if arch == "qwen21":
        import comfy.ldm.qwen_image21.model as qm

        def gated_residual(x, y, gate, prefix_len):  # the inference version adds into x in place
            g_prefix, g_target = gate
            target = x[:, prefix_len:] + y[:, prefix_len:] * g_target
            if not prefix_len:
                return target
            return torch.cat([x[:, :prefix_len] + y[:, :prefix_len] * g_prefix, target], dim=1)

        qm._gated_residual = gated_residual


def checkpoint_blocks(blocks):
    for blk in blocks:
        inner = blk.forward

        def fwd(*a, _inner=inner, **k):
            return torch.utils.checkpoint.checkpoint(_inner, *a, use_reentrant=False, **k)

        blk.forward = fwd


def lora_state_dict(arch, loras, n_layers):
    """ComfyUI module names -> diffusers LoRA keys (lora_A = down, lora_B = up)."""
    sd = {}

    def put(key, down, up, alpha):
        sd[f"transformer.{key}.lora_A.weight"] = down.detach().float().cpu().contiguous()
        sd[f"transformer.{key}.lora_B.weight"] = up.detach().float().cpu().contiguous()
        sd[f"transformer.{key}.alpha"] = torch.tensor(float(alpha))

    if arch == "krea2":
        to_diff = {}
        for d, c in cb.comfy.utils.krea2_to_diffusers({"layers": n_layers}).items():
            if not d.endswith("attn.to_out.weight"):  # alias of attn.to_out.0
                to_diff[c[:-len(".weight")]] = d[:-len(".weight")]
        for name, l in loras.items():
            put(to_diff[name], l.down, l.up, l.scale * l.down.shape[0])
    else:
        for name, l in loras.items():
            alpha = l.scale * l.down.shape[0]
            if name.endswith(".img_mlp.gate_up"):
                # ComfyUI maps Qwen 2.1 LoRAs onto the fused gate_up as its halves: gate_layer = rows
                # [0, h), proj = rows [h, 2h). One LoRA on gate_up is exactly two sharing A.
                h = l.up.shape[0] // 2
                base = name[:-len("gate_up")]
                put(base + "gate_layer", l.down, l.up[:h], alpha)
                put(base + "proj", l.down, l.up[h:], alpha)
            else:
                put(name, l.down, l.up, alpha)
    return sd


# ---------------------------------------------------------------- training

PEAK = {"gb": 0.0}


def _poll_peak():
    """Peak GPU driver memory, sampled every 50 ms (as bench/sdpa_mps.py does)."""
    while True:
        PEAK["gb"] = max(PEAK["gb"], torch.mps.driver_allocated_memory() / 2**30)
        time.sleep(0.05)


def shift_mu(arch, model, latent):
    if arch == "qwen21":
        tokens = latent.shape[-2] * latent.shape[-1]
        return 0.5 + (0.9 - 0.5) * (tokens - 256) / (8192 - 256)
    return float(model.model_sampling.shift)


def sample_sigma(n, mu, gen):
    t = torch.sigmoid(torch.randn(n, generator=gen)).clamp(1e-4, 1 - 1e-4)
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1))


@contextmanager
def atomic_checkpoint(path):
    """Publish only a complete file; a failed write leaves the previous checkpoint intact."""
    fd, tmp = tempfile.mkstemp(prefix="checkpoint-", suffix=".tmp", dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    try:
        yield tmp
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_state(path, step, loras, opt, scaler, gen, *, identity=None):
    # CPU copies prevent a resumed file from allocating a second optimizer on MPS.
    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu(v) for v in value)
        return value

    state = {"lora": {n: (l.down.detach().cpu(), l.up.detach().cpu()) for n, l in loras.items()},
             "opt": cpu(opt.state_dict()), "scaler": scaler.state_dict() if scaler else None,
             "gen": gen.get_state(), "step": step}
    if identity is not None:
        state["identity"] = identity
    with atomic_checkpoint(path) as tmp:
        torch.save(state, tmp)


def restore_state(path, loras, opt, scaler, gen, *, state=None):
    if state is None:
        state = torch.load(path, map_location="cpu")
    with torch.no_grad():
        for name, lora in loras.items():
            lora.down.copy_(state["lora"][name][0])
            lora.up.copy_(state["lora"][name][1])
    opt.load_state_dict(state["opt"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    gen.set_state(state["gen"])
    return state["step"]  # release deserialized state before training resumes


def file_fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        before = os.fstat(f.fileno())
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(f.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Training input changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "bytes": after.st_size}


def training_identity(args, spec, cache_path):
    # Paths may move; content hashes bind the actual cached training inputs and base.
    base_path = cb.folder_paths.get_full_path_or_raise("diffusion_models", args.unet)
    config = {key: getattr(args, key) for key in
              ("arch", "unet", "res", "rank", "lr", "seed", "no_checkpoint", "chunked_attn")}
    config["dtype"] = args.dtype or spec["dtype"]
    config["targets"] = {"blocks": spec["blocks"], "suffixes": list(spec["targets"])}
    config["optimizer"] = {"name": "AdamW", "betas": [0.9, 0.999], "weight_decay": 1e-2, "eps": 1e-8}
    config["flow_recipe"] = 1
    if getattr(args, "buckets", None) is not None:
        config["shape_policy"] = shape_policy(args.res, args.buckets)
    return {"version": 1, "config": config, "inputs": {
        "base": file_fingerprint(base_path), "cache": file_fingerprint(cache_path),
        "dataset": file_fingerprint(f"{args.dataset}/dataset.json")}}


def validate_resume_identity(state, identity, args):
    saved = state.get("identity")
    if saved is not None:
        if saved != identity:
            raise ValueError("Resume identity mismatch: training configuration, base checkpoint, or cached dataset changed")
        return False

    if getattr(args, "cache_file", None) is not None:
        raise ValueError("Explicit cache selection requires a fingerprinted checkpoint; start a new run")

    if "shape_policy" in identity["config"]:
        raise ValueError("Legacy resume cannot establish a bucket shape policy; start a new bucket run")

    if args.arch == "qwen21":
        # The repaired VAE changes Qwen targets, including finite old 512/1024 caches.
        # Matching a legacy filename/configuration cannot establish the original bytes.
        raise ValueError("Legacy Qwen resume cannot verify cached targets after the VAE correction; "
                         "start a new run with a verified cache")

    # Legacy files did not record input digests. Require their recorded configuration;
    # the next recovery save anchors the current bytes, without claiming historical hashes.
    with open(f"{args.out}/log.jsonl") as f:
        text = f.read()
    records = text.splitlines()
    metadata = []
    for index, line in enumerate(records):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(records) - 1 and not text.endswith("\n"):
                break  # A killed process may leave its final log record incomplete.
            raise ValueError("Legacy resume log is invalid; cannot establish its training configuration") from None
        if "meta" in record:
            metadata.append(record["meta"])
    if not metadata:
        raise ValueError("Legacy resume requires its original metadata in log.jsonl")
    for meta in metadata:
        for key in ("arch", "unet", "res", "rank", "lr", "seed", "no_checkpoint", "chunked_attn"):
            if key not in meta or meta[key] != identity["config"][key]:
                raise ValueError(f"Legacy resume configuration mismatch: {key}")
        legacy_dtype = meta.get("dtype") or {"torch.float16": "fp16", "torch.bfloat16": "bf16"}.get(meta.get("act_dtype"))
        if legacy_dtype != identity["config"]["dtype"]:
            raise ValueError("Legacy resume configuration mismatch: dtype")
        if "dataset" not in meta or os.path.realpath(meta["dataset"]) != os.path.realpath(args.dataset):
            raise ValueError("Legacy resume configuration mismatch: dataset")
        if meta.get("identity") is not None and meta["identity"] != identity:
            raise ValueError("Legacy resume log identity does not match current training inputs")
    print("Legacy resume: recorded configuration matches; input hashes start with the next recovery save.", flush=True)
    return True


def optimizer_step(loss, params, opt, scaler):
    value = loss.item()
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite training loss: {value}; optimizer was not updated")
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)  # GradScaler still skips overflows and adjusts its scale normally.
        scaler.update()
    else:
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        opt.step()
    opt.zero_grad(set_to_none=True)
    return value, norm


def validate_cache(cache):
    if not cache["latents"] or len(cache["latents"]) != len(cache["conds"]):
        raise ValueError("training cache must contain matching nonempty latents and conditions")
    for kind in ("latents", "conds"):
        for i, tensor in enumerate(cache[kind]):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"training cache has non-finite {kind}[{i}]; rebuild the cache")


def validate_explicit_cache(cache, args, policy):
    """An alternate file must still describe this dataset and training geometry."""
    with open(os.path.join(args.dataset, "dataset.json")) as f:
        images = json.load(f)["images"]
    if cache.get("captions") != [image["caption"] for image in images] or len(cache["latents"]) != len(images):
        raise ValueError("Explicit cache captions/order do not match the dataset")
    prefix, compression = {"qwen21": ((1, 64), 16), "krea2": ((1, 16, 1), 8)}[args.arch]
    for index, latent in enumerate(cache["latents"]):
        if policy["mode"] == "buckets":
            sample = cache["samples"][index]
            if sample.get("file") != images[index]["file"]:
                raise ValueError("Explicit bucket cache file order does not match the dataset")
            width, height = sample["target_size"]
        else:
            width = height = args.res
        expected = prefix + (height // compression, width // compression)
        if tuple(latent.shape) != expected:
            raise ValueError(f"Explicit cache latent {index} has shape {tuple(latent.shape)}; expected {expected}")


def export_lora_if_due(args, step, loras, n_layers, act_dtype):
    if args.profile_steps or step <= 0 or (step % args.save_every != 0 and step != args.steps):
        return
    from safetensors.torch import save_file
    path = f"{args.out}/{args.name}_s{step}.safetensors"
    sd = lora_state_dict(args.arch, loras, n_layers)
    metadata = {"base": args.unet, "steps": str(step), "rank": str(args.rank), "res": str(args.res),
                "lr": str(args.lr), "dtype": str(act_dtype), "trainer": "mac-kernels/train"}
    if getattr(args, "buckets", None) is not None:
        metadata.pop("res")  # Explicit buckets, rather than --res, supplied the training shapes.
        metadata["shape_policy"] = json.dumps(shape_policy(args.res, args.buckets), sort_keys=True)
    with atomic_checkpoint(path) as tmp:
        save_file(sd, tmp, metadata=metadata)
    print(f"saved {path}", flush=True)


def main():
    args = parse()
    spec = ARCHS[args.arch]
    state_path = f"{args.out}/state.pt"
    policy = shape_policy(args.res, getattr(args, "buckets", None))
    explicit_cache = getattr(args, "cache_file", None)
    cache_path = explicit_cache if explicit_cache is not None else cache_filename(
        args.dataset, args.arch, args.res, getattr(args, "buckets", None))
    cache = None
    if explicit_cache is not None:
        cache = torch.load(cache_path, map_location="cpu")
        validate_cache(cache)
        validate_bucket_cache(cache, policy)
        validate_explicit_cache(cache, args, policy)
    identity, resume_state, legacy_resume = None, None, False
    if args.resume:
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(f"Resume requires the original training cache: {cache_path}")
        print("Verifying resume configuration and input hashes", flush=True)
        resume_state = torch.load(state_path, map_location="cpu", mmap=True)
        identity = training_identity(args, spec, cache_path)
        legacy_resume = validate_resume_identity(resume_state, identity, args)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("mps")
    cb.mm.in_training = True  # differentiable eager ops (RoPE, AdaLN), as ComfyUI's TrainLoraNode sets it
    patch_arch_for_training(args.arch)
    if cache is None:
        cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else build_cache(args, spec, cache_path)
    validate_cache(cache)
    validate_bucket_cache(cache, policy)
    if args.preview_every and not args.profile_steps:
        if args.preview_index >= len(cache['latents']) or len(cache.get('captions', [])) != len(cache['latents']):
            raise ValueError('previews require a valid cached sample index and one caption per latent')
    if not args.resume and not args.profile_steps:
        identity = training_identity(args, spec, cache_path)
    n_img = len(cache["latents"])
    print(f"cache: {n_img} images, latent {tuple(cache['latents'][0].shape)}, cond {tuple(cache['conds'][0].shape)}")

    N = cb.N
    t_load = time.perf_counter()
    mp = N["UNETLoader"]().load_unet(args.unet, "default")[0]
    dtype = torch.float16 if (args.dtype or spec["dtype"]) == "fp16" else torch.bfloat16
    mp.set_model_compute_dtype(dtype)
    cb.mm.load_models_gpu([mp], memory_required=1e20, force_full_load=True)
    model = mp.model
    dm = model.diffusion_model
    dm.requires_grad_(False)
    blocks = getattr(dm, spec["blocks"])
    latents = [model.process_latent_in(l.to(dev)) for l in cache["latents"]]
    conds = [c.to(dev, dtype) for c in cache["conds"]]
    act_dtype = model.get_dtype_inference()
    mu = shift_mu(args.arch, model, latents[0])
    mus = [shift_mu(args.arch, model, latent) for latent in latents] if policy["mode"] == "buckets" else [mu] * n_img
    print(f"loaded {args.unet} in {time.perf_counter() - t_load:.0f}s; compute {act_dtype}, shift mu {mu:.3f}; "
          f"GPU {torch.mps.driver_allocated_memory() / 2**30:.2f} GB", flush=True)

    topts = {}
    if args.chunked_attn:
        from comfy_kitchen.backends.mps.attention_autograd import comfy_attention_override
        topts["optimized_attention_override"] = comfy_attention_override()

    def predict(x_t, sigma, cond):
        return dm(x_t.to(act_dtype), sigma.to(dev, torch.float32), context=cond, transformer_options=dict(topts))

    vg = torch.Generator().manual_seed(1234)
    val = [(i, s, torch.randn(latents[i].shape, generator=vg)) for i in sorted({0, min(7, n_img - 1)}) for s in (0.4, 0.7, 0.9)]

    def val_loss():
        tot = 0.0
        with torch.no_grad():
            for i, s, e in val:
                e = e.to(dev)
                x0 = latents[i]
                v = predict((1 - s) * x0 + s * e, torch.tensor([s]), conds[i])
                tot += F.mse_loss(v.float(), (e - x0).float()).item()
        return tot / len(val)

    with torch.no_grad():
        i, s, e = val[1]
        ref_v = predict((1 - s) * latents[i] + s * e.to(dev), torch.tensor([s]), conds[i]).float()

    loras, n_quantized = patch_model(dm, spec, args.rank, args.seed, dev)
    params = [p for l in loras.values() for p in l.parameters()]
    n_params = sum(p.numel() for p in params)
    print(f"Quantized linears through trainable kernels: {n_quantized}; LoRA layers: {len(loras)}, "
          f"{n_params / 1e6:.1f}M params", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8, fused=True)
    scaler = torch.amp.GradScaler("mps", init_scale=2.0 ** 16) if dtype == torch.float16 else None
    gen = torch.Generator().manual_seed(args.seed)
    start = 0
    log = open(f"{args.out}/log.jsonl", "a")
    if args.resume:
        start = restore_state(state_path, loras, opt, scaler, gen, state=resume_state)
        del resume_state
        print(f"resumed at step {start}", flush=True)
        log.write(json.dumps({"resume": start, "legacy_identity": legacy_resume}) + "\n")
        # Recovery is durable before validation/export. Recreate a due adapter if that
        # later work was interrupted, including when this run has already reached its end.
        export_lora_if_due(args, start, loras, len(blocks), act_dtype)
    else:
        with torch.no_grad():
            i, s, e = val[1]
            v0 = predict((1 - s) * latents[i] + s * e.to(dev), torch.tensor([s]), conds[i]).float()
        step0_rel = ((v0 - ref_v).norm() / ref_v.norm()).item()
        print(f"step 0 vs ComfyUI's own path: rel diff {step0_rel:.2e}", flush=True)
        meta = dict(vars(args), n_quantized=n_quantized, lora_layers=len(loras), lora_params=n_params, shift_mu=mu,
                    act_dtype=str(act_dtype), step0_rel_vs_comfy=step0_rel, torch=torch.__version__, identity=identity)
        if policy["mode"] == "buckets":
            meta["shift_mu_per_sample"] = mus
        log.write(json.dumps({"meta": meta}) + "\n")
    if not args.no_checkpoint:
        checkpoint_blocks(blocks)
    if start == 0 and not args.profile_steps:
        v = val_loss()
        log.write(json.dumps({"step": 0, "val_loss": v}) + "\n")
        print(f"fixed training-set loss step 0: {v:.5f}", flush=True)

    if not args.profile_steps and not args.resume:
        save_state(state_path, start, loras, opt, scaler, gen, identity=identity)

    def preview_if_due(step):
        if not preview_due(args, step):
            return
        # Preview failure must never lose the update it was intended to show.
        save_state(state_path, step, loras, opt, scaler, gen, identity=identity)
        result = write_preview(args, step, model, loras, latents, conds, cache["captions"], mus, topts,
                               lambda latent: decode_preview(cb, spec, args.arch, latent), model_identity=identity)
        log.write(json.dumps({"preview": result}) + "\n")
        log.flush()
        print(f"preview step {step}: {result.get('image', result['status'])}", flush=True)

    preview_if_due(start)
    PEAK["gb"] = 0.0
    threading.Thread(target=_poll_peak, daemon=True).start()
    window, times = [], []
    total = args.profile_steps or args.steps
    end = total if not args.stop_after else min(total, start + args.stop_after)
    step = start
    for step in range(start + 1, end + 1):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        i = int(torch.randint(n_img, (1,), generator=gen))
        x0 = latents[i]
        e = torch.randn(x0.shape, generator=gen).to(dev)
        s = sample_sigma(1, mus[i], gen)
        sv = s.view(-1, *([1] * (x0.ndim - 1))).to(dev)
        vpred = predict((1 - sv) * x0 + sv * e, s, conds[i])
        loss = F.mse_loss(vpred.float(), (e - x0).float())
        lv, gn = optimizer_step(loss, params, opt, scaler)
        torch.mps.synchronize()
        dt = time.perf_counter() - t0
        window.append(lv)
        times.append(dt)
        if step % args.log_every == 0 or args.profile_steps:
            rec = {"step": step, "loss": sum(window) / len(window), "s_per_step": sum(times[-len(window):]) / len(window),
                   "peak_gb": round(PEAK["gb"], 2), "grad_norm": float(gn), "scale": scaler.get_scale() if scaler else None}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
            window = []
        if not args.profile_steps and (step % args.state_every == 0 or step == end):
            save_state(state_path, step, loras, opt, scaler, gen, identity=identity)
        if step % args.val_every == 0 and not args.profile_steps:
            v = val_loss()
            log.write(json.dumps({"step": step, "val_loss": v}) + "\n")
            log.flush()
            print(f"fixed training-set loss step {step}: {v:.5f}", flush=True)
        export_lora_if_due(args, step, loras, len(blocks), act_dtype)
        preview_if_due(step)
    summary = {"summary": True, "steps": [start, step], "mean_s_per_step": sum(times[3:]) / max(1, len(times) - 3),
               "peak_driver_gb": round(PEAK["gb"], 2)}
    log.write(json.dumps(summary) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
