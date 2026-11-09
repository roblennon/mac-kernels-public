"""Where Wan 2.2 VAE decode time goes on MPS: per-module timing (sync hooks) on the first FRAMES latent frames of a saved latent.

Env: LATENT (under out/latents), FRAMES (latent frames, default 3), VAE, COMFY_DIR, HOOKS=0 for a plain timing pass only.
"""
import asyncio, collections, json, os, sys, time
ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
CODE = os.environ.get("COMFY_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ComfyUI")
sys.path.insert(0, CODE); os.chdir(CODE)
sys.argv = [sys.argv[0], "--use-pytorch-cross-attention"]
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths
folder_paths.add_model_folder_path("vae", f"{ROOT}/models/wan/vae")
import comfy.model_management as mm, nodes
asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))

FRAMES = int(os.environ.get("FRAMES", 3))
with torch.inference_mode():
    vae = nodes.NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(os.environ.get("VAE", "wan2.2_vae.safetensors"))[0]
    mm.load_models_gpu([vae.patcher])
    net = vae.first_stage_model
    dev = mm.get_torch_device()
    z = torch.load(f"{ROOT}/out/latents/{os.environ['LATENT']}")[:, :, :FRAMES].to(dev, vae.vae_dtype)
    print("vae dtype", vae.vae_dtype, "latent", tuple(z.shape), flush=True)

    def run():
        torch.mps.synchronize(); t = time.perf_counter(); out = net.decode(z); torch.mps.synchronize()
        return out, time.perf_counter() - t
    run()  # warm-up
    out, plain = run()
    print("RESULT " + json.dumps(dict(kind="plain", frames_latent=FRAMES, frames_out=int(out.shape[2]), seconds=round(plain, 2), out_shape=list(out.shape))), flush=True)
    if os.environ.get("HOOKS", "1") == "1":
        stats = collections.defaultdict(lambda: [0, 0.0]); t0 = {}
        def pre(m, args, kwargs=None):
            torch.mps.synchronize(); t0[id(m)] = time.perf_counter()
        def post(m, args, *rest):
            torch.mps.synchronize()
            x = args[0] if args else None
            key = (type(m).__name__, tuple(x.shape) if torch.is_tensor(x) else None, tuple(getattr(m, "weight", torch.empty(0)).shape))
            s = stats[key]; s[0] += 1; s[1] += time.perf_counter() - t0.pop(id(m))
        leaves = [m for m in net.decoder.modules() if len(list(m.children())) == 0] + [net.conv2]
        for m in leaves:
            m.register_forward_pre_hook(pre); m.register_forward_hook(post)
        out, hooked = run()
        by_class = collections.defaultdict(float)
        for (cls, _, _), (n, t) in stats.items():
            by_class[cls] += t
        print("RESULT " + json.dumps(dict(kind="hooked", seconds=round(hooked, 2), leaf_total=round(sum(by_class.values()), 2),
              by_class={k: round(v, 2) for k, v in sorted(by_class.items(), key=lambda kv: -kv[1])})), flush=True)
        for (cls, shape, w), (n, t) in sorted(stats.items(), key=lambda kv: -kv[1][1])[:25]:
            print(f"TOP {t:7.2f}s {n:4d}x {cls:14s} in={shape} w={w}", flush=True)
