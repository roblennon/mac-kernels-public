"""Wan DiT patch embedding (Conv3d, kernel = stride = 1x2x2, fp32 weights) on MPS: torch conv3d vs comfy.ops Conv3d path."""
import json, os, sys, time
CODE = os.environ.get("COMFY_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ComfyUI")
sys.path.insert(0, CODE); sys.argv = [sys.argv[0]]
import comfy.options; comfy.options.enable_args_parsing()
import torch, comfy.ops
def timed(fn, reps=5):
    fn(); torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(reps): o = fn()
    torch.mps.synchronize(); return o, (time.perf_counter() - t) / reps
with torch.inference_mode():
    for label, cin, cout, shape in (("5b-1280x704x81", 48, 3072, (1, 48, 21, 44, 80)), ("14b-832x480x81", 16, 5120, (1, 16, 21, 60, 104))):
        conv = comfy.ops.disable_weight_init.Conv3d(cin, cout, (1, 2, 2), stride=(1, 2, 2)).to("mps", torch.float32)
        torch.nn.init.normal_(conv.weight, std=0.02); torch.nn.init.zeros_(conv.bias)
        x = torch.randn(shape, device="mps")
        ref, t_ref = timed(lambda: torch.nn.functional.conv3d(x, conv.weight, conv.bias, conv.stride))
        out, t_new = timed(lambda: conv(x))
        print("RESULT " + json.dumps(dict(shape=label, conv3d_ms=round(t_ref * 1000, 2), ops_path_ms=round(t_new * 1000, 2), max_abs=float((out - ref).abs().max()))), flush=True)
