"""Portable GGUF registration and real ComfyUI discovery, without GPU work."""
import asyncio
import os
from pathlib import Path

import pytest

from bench.setup_custom_nodes import register_gguf


def checkout(repo):
    source = repo / "vendor" / "ComfyUI-GGUF-fork"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (source / "nodes.py").write_text("")
    return source


def test_setup_is_repeatable_and_link_survives_repository_move(tmp_path):
    repo = tmp_path / "original"
    checkout(repo)
    destination = register_gguf(repo)
    assert not os.path.isabs(os.readlink(destination))
    assert register_gguf(repo) == destination
    moved = tmp_path / "moved"
    repo.rename(moved)
    assert register_gguf(moved).resolve() == moved / "vendor" / "ComfyUI-GGUF-fork"


@pytest.mark.parametrize("kind", ["directory", "dangling_symlink"])
def test_setup_preserves_existing_other_installation(tmp_path, kind):
    checkout(tmp_path)
    destination = tmp_path / "vendor" / "ComfyUI" / "custom_nodes" / "ComfyUI-GGUF"
    destination.parent.mkdir(parents=True)
    if kind == "directory":
        destination.mkdir()
        (destination / "keep").write_text("user installation")
    else:
        destination.symlink_to("missing-other-checkout")
    with pytest.raises(RuntimeError, match="another installation"):
        register_gguf(tmp_path)
    if kind == "directory":
        assert (destination / "keep").read_text() == "user installation"
    else:
        assert os.readlink(destination) == "missing-other-checkout"


def test_missing_fork_fails_before_creating_link(tmp_path):
    with pytest.raises(RuntimeError, match="missing or incomplete"):
        register_gguf(tmp_path)
    assert not (tmp_path / "vendor" / "ComfyUI").exists()


def test_real_config_and_node_discovery_lists_gguf_models(tmp_path, monkeypatch):
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import folder_paths
    import nodes
    from utils.extra_config import load_extra_path_config

    repo = Path(__file__).resolve().parents[1]
    custom_nodes = tmp_path / "custom_nodes"
    custom_nodes.mkdir()
    (custom_nodes / "ComfyUI-GGUF").symlink_to(repo / "vendor" / "ComfyUI-GGUF-fork")
    models = tmp_path / "models" / "gguf"
    models.mkdir(parents=True)
    (models / "discovery-only.gguf").touch()
    monkeypatch.setenv("MK_ROOT", str(tmp_path))
    # Isolate path registration from the operator's existing model/node directories.
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {
        key: (list(paths), set(exts)) for key, (paths, exts) in folder_paths.folder_names_and_paths.items()
    })
    folder_paths.folder_names_and_paths["custom_nodes"] = ([str(custom_nodes)], set())
    load_extra_path_config(str(repo / "bench" / "extra_model_paths.yaml"))
    assert str(models) in folder_paths.get_folder_paths("diffusion_models")
    asyncio.run(nodes.init_external_custom_nodes())
    loader = nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]
    assert "discovery-only.gguf" in loader.INPUT_TYPES()["required"]["unet_name"][0]
