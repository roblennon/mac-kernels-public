"""Import ComfyUI from vendor/ComfyUI headlessly, the same way bench/run_krea2.py does.

Import this module before anything from comfy. Model folders point at MK_ROOT/models.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ROOT = os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))
COMFY = os.path.join(REPO, "vendor", "ComfyUI")

sys.path.insert(0, COMFY)
os.chdir(COMFY)
_argv = list(sys.argv)
sys.argv = [sys.argv[0]] + (["--fp16-vae"] if os.environ.get("VAE_FP16", "1") == "1" else [])
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import folder_paths  # noqa: E402

for _k in ("diffusion_models", "text_encoders", "vae", "loras"):
    folder_paths.add_model_folder_path(_k, f"{ROOT}/models/{_k}")
    folder_paths.add_model_folder_path(_k, f"{ROOT}/models/qwen21/{_k}")
import comfy.model_management as mm  # noqa: E402,F401
import comfy.utils  # noqa: E402,F401
import nodes  # noqa: E402

sys.argv = _argv  # ComfyUI has parsed its flags; give the script its own back
N = nodes.NODE_CLASS_MAPPINGS  # cwd stays in ComfyUI, as in bench/run_krea2.py; use absolute paths


def load_extra_nodes():
    """comfy_extras nodes (TextEncodeQwenImage21, SamplerCustomAdvanced, ...), as bench/run_qwen21.py loads them."""
    import asyncio
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))


def call(name, **kw):
    """Run a node by name, old-style (FUNCTION) or io.ComfyNode, returning its outputs tuple."""
    cls = N[name]
    if hasattr(cls, "define_schema"):
        out = cls.execute(**kw)
        return out.result if hasattr(out, "result") else out.args
    return getattr(cls(), cls.FUNCTION)(**kw)
