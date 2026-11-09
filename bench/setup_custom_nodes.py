"""Register the existing GGUF checkout without copying it into ComfyUI."""
import os
from pathlib import Path


def register_gguf(repo):
    repo = Path(repo).resolve()
    source = repo / "vendor" / "ComfyUI-GGUF-fork"
    destination = repo / "vendor" / "ComfyUI" / "custom_nodes" / "ComfyUI-GGUF"
    if not (source / "__init__.py").is_file() or not (source / "nodes.py").is_file():
        raise RuntimeError(f"GGUF node checkout is missing or incomplete: {source}")
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source:
            raise RuntimeError(f"Custom node path already belongs to another installation: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)
    return destination


def register_video(repo):
    repo = Path(repo).resolve()
    source = repo / 'bench/video_nodes'
    destination = repo / 'vendor/ComfyUI/custom_nodes/mac-kernels-video'
    if not (source / '__init__.py').is_file():
        raise RuntimeError(f'Video node package is missing: {source}')
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source:
            raise RuntimeError(f'Custom node path already belongs to another installation: {destination}')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)
    return destination


if __name__ == "__main__":
    try:
        register_gguf(Path(__file__).resolve().parents[1])
        register_video(Path(__file__).resolve().parents[1])
    except (OSError, RuntimeError) as error:
        raise SystemExit(f"Custom node setup failed: {error}") from error
