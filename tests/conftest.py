"""Skip tests that need the patched ComfyUI checkout when it isn't present.

These files exercise the real ComfyUI runtime (imports, subprocesses or source
files under vendor/ComfyUI). On a fresh clone without vendor/, they would error
at collection and abort the whole run, so they are skipped with one explanation
instead. With the checkout present (or COMFY_DIR set), nothing is skipped.
Keep this list in sync when a test starts depending on vendor/ComfyUI.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY_DIR = Path(os.environ.get("COMFY_DIR") or ROOT / "vendor" / "ComfyUI")
HAVE_COMFY = (COMFY_DIR / "comfy" / "model_management.py").is_file()

NEEDS_COMFY = {
    "test_custom_node_setup.py", "test_gguf_audit.py", "test_gguf_lifetime.py",
    "test_gguf_node.py", "test_gguf_precision.py", "test_h3_dense_ownership.py",
    "test_h3_memory.py", "test_h3_scale_ownership.py", "test_krea_adapter.py",
    "test_krea_conditioning.py", "test_krea_export.py", "test_krea_export_jobs.py",
    "test_krea_export_release.py", "test_krea_plain_adapter.py", "test_krea_readonly.py",
    "test_krea_server.py", "test_krea_server_lora.py", "test_krea_stack_adapter.py",
    "test_lora_sidepath.py", "test_lora_sidepath_types.py", "test_mps_pread_load.py",
    "test_packed_training_artifact.py", "test_qwen_adapter_types.py", "test_qwen_memory.py",
    "test_spectral_krea.py", "test_text_encoder_load.py", "test_training_preview.py",
    "test_video_loras.py", "test_wan_mapped_construction.py", "test_wan_memory.py",
    "test_wan_saved_gui.py",
}


def pytest_ignore_collect(collection_path, config):
    if not HAVE_COMFY and collection_path.name in NEEDS_COMFY:
        return True
    return None


def pytest_report_header(config):
    if not HAVE_COMFY:
        return (f"Patched ComfyUI not found at {COMFY_DIR}: skipping {len(NEEDS_COMFY)} test files "
                "that need it (see CONTRIBUTING.md, 'Run the tests').")
    return None
