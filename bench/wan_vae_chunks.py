"""Wan 2.2 VAE decode: one latent frame per decoder call (ComfyUI default) vs CHUNKS latent frames per call, same causal cache.

Env: LATENT, FRAMES (latent frames, default 7), CHUNKS (default 2,3,6), COMFY_DIR.
"""
import asyncio, json, os, sys, time
ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
CODE = os.environ.get("COMFY_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ComfyUI")
sys.path.insert(0, CODE); os.chdir(CODE)
sys.argv = [sys.argv[0], "--use-pytorch-cross-attention"]
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths, threading
folder_paths.add_model_folder_path("vae", f"{ROOT}/models/wan/vae")
import comfy.model_management as mm, nodes
from comfy.ldm.wan.vae2_2 import count_conv3d, unpatchify
asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
PEAK = {"gb": 0.0}
def _poll():
    while True:
        PEAK["gb"] = max(PEAK["gb"], torch.mps.driver_allocated_memory() / 2**30); time.sleep(0.1)
threading.Thread(target=_poll, daemon=True).start()

def decode_chunked(net, z, chunk):
    feat_map = [None] * count_conv3d(net.decoder)
    x = net.conv2(z)
    outs = [net.decoder(x[:, :, :1], feat_cache=feat_map, feat_idx=[0], first_chunk=True)]
    for i in range(1, x.shape[2], chunk):
        outs.append(net.decoder(x[:, :, i:i + chunk], feat_cache=feat_map, feat_idx=[0]))
    return unpatchify(torch.cat(outs, 2), patch_size=net.patch_size)

with torch.inference_mode():
    vae = nodes.NODE_CLASS_MAPPINGS["VAELoader"]().load_vae("wan2.2_vae.safetensors")[0]
    mm.load_models_gpu([vae.patcher]); net = vae.first_stage_model
    z = torch.load(f"{ROOT}/out/latents/{os.environ['LATENT']}")[:, :, :int(os.environ.get("FRAMES", 7))].to(mm.get_torch_device(), vae.vae_dtype)
    def timed(fn):
        torch.mps.synchronize(); torch.mps.empty_cache(); PEAK["gb"] = 0.0; t = time.perf_counter(); o = fn(); torch.mps.synchronize()
        return o, time.perf_counter() - t, PEAK["gb"]
    net.decode(z[:, :, :2])  # warm-up
    ref, t1, m1 = timed(lambda: net.decode(z))
    print("RESULT " + json.dumps(dict(chunk=1, seconds=round(t1, 2), peak_gb=round(m1, 2), frames=int(ref.shape[2]))), flush=True)
    for c in (int(c) for c in os.environ.get("CHUNKS", "2,3,6").split(",")):
        o, t, m = timed(lambda: decode_chunked(net, z, c))
        print("RESULT " + json.dumps(dict(chunk=c, seconds=round(t, 2), peak_gb=round(m, 2), frames=int(o.shape[2]),
              max_abs_vs_chunk1=float((o.float() - ref.float()).abs().max()) if o.shape == ref.shape else None)), flush=True)
