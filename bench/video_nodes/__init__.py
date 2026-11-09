"""Register local video controls for the Comfy server."""
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
from bench.video_loras import NODE_CLASS_MAPPINGS as LORAS
from bench.h3_memory import NODE_CLASS_MAPPINGS as H3, NODE_DISPLAY_NAME_MAPPINGS as H3_NAMES
from bench.wan_memory import NODE_CLASS_MAPPINGS as WAN, NODE_DISPLAY_NAME_MAPPINGS as WAN_NAMES
from bench.wan_latent_nodes import NODE_CLASS_MAPPINGS as LATENTS, NODE_DISPLAY_NAME_MAPPINGS as LATENT_NAMES

from bench.qwen_comfy import NODE_CLASS_MAPPINGS as QWEN, NODE_DISPLAY_NAME_MAPPINGS as QWEN_NAMES

NODE_CLASS_MAPPINGS = {**QWEN, **LORAS, **LATENTS, **WAN, **H3}
NODE_DISPLAY_NAME_MAPPINGS = {**QWEN_NAMES, **LATENT_NAMES, **WAN_NAMES, **H3_NAMES, 'BenchVideoLoRAStack': 'Video LoRA Stack'}
