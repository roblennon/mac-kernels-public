"""CPU proofs for training recovery; ComfyUI startup is the isolated external boundary."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy_boot", SimpleNamespace(ROOT="/unused"))
    path = Path(__file__).resolve().parents[1] / "train" / "train_lora.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("training_recovery_subject", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_training(trainer, scaled=False):
    lora = trainer.LoRA(3, 2, 2, 2, torch.Generator().manual_seed(3))
    params = list(lora.parameters())
    opt = torch.optim.AdamW(params, lr=0.01)
    scaler = torch.amp.GradScaler("cpu", init_scale=16) if scaled else None
    return {"layer": lora}, opt, scaler, torch.Generator().manual_seed(7)


def update(trainer, training):
    loras, opt, scaler, gen = training
    lora = loras["layer"]
    x = torch.randn(4, 3, generator=gen)
    target = torch.randn(4, 2, generator=gen)
    trainer.optimizer_step((lora(x) - target).square().mean(), list(lora.parameters()), opt, scaler)


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("scaled", [False, True])
def test_resume_matches_uninterrupted_optimizer_and_rng(trainer, tmp_path, scaled):
    continuous = setup_training(trainer, scaled)
    for _ in range(3):
        update(trainer, continuous)
    path = tmp_path / "state.pt"
    trainer.save_state(path, 3, *continuous)
    resumed = setup_training(trainer, scaled)
    assert trainer.restore_state(path, *resumed) == 3
    for _ in range(2):
        update(trainer, continuous)
        update(trainer, resumed)
    assert_tree_equal(continuous[0]["layer"].state_dict(), resumed[0]["layer"].state_dict())
    assert_tree_equal(continuous[1].state_dict(), resumed[1].state_dict())
    assert_tree_equal(continuous[3].get_state(), resumed[3].get_state())
    if scaled:
        assert_tree_equal(continuous[2].state_dict(), resumed[2].state_dict())


def test_partial_save_preserves_previous_checkpoint(trainer, tmp_path, monkeypatch):
    training = setup_training(trainer)
    update(trainer, training)
    path = tmp_path / "state.pt"
    trainer.save_state(path, 1, *training)
    previous = path.read_bytes()

    def disk_full(state, filename):
        Path(filename).write_bytes(b"partial replacement")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", disk_full)
    with pytest.raises(OSError, match="disk full"):
        trainer.save_state(path, 2, *training)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
    assert trainer.restore_state(path, *setup_training(trainer)) == 1


def test_restore_always_deserializes_on_cpu(trainer, tmp_path, monkeypatch):
    training = setup_training(trainer)
    update(trainer, training)
    path = tmp_path / "state.pt"
    trainer.save_state(path, 1, *training)
    real_load = torch.load
    seen = []

    def load(*args, **kwargs):
        seen.append(kwargs.get("map_location"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    trainer.restore_state(path, *setup_training(trainer))
    assert seen == ["cpu"]


def test_missing_resume_fails_before_loading_models(trainer, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_lora.py", "--resume", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        trainer.parse()
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_existing_recovery_state_requires_explicit_resume(trainer, tmp_path, monkeypatch):
    path = tmp_path / "state.pt"
    path.write_bytes(b"existing state")
    monkeypatch.setattr(sys, "argv", ["train_lora.py", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        trainer.parse()
    assert error.value.code == 2
    assert path.read_bytes() == b"existing state"


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_loss_does_not_mutate_weights_or_optimizer(trainer, scaled, bad):
    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.AdamW([param])
    scaler = torch.amp.GradScaler("cpu") if scaled else None
    with pytest.raises(FloatingPointError):
        trainer.optimizer_step(param * bad, [param], opt, scaler)
    assert param.item() == 1.0
    assert not opt.state


def test_finite_loss_infinite_gradient_does_not_update_unscaled_weights(trainer):
    param = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.AdamW([param])
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.optimizer_step(param.sqrt(), [param], opt, None)
    assert param.item() == 0.0
    assert not opt.state


def test_scaled_overflow_still_skips_update_and_reduces_scale(trainer):
    param = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.AdamW([param])
    scaler = torch.amp.GradScaler("cpu", init_scale=16)
    trainer.optimizer_step(param.sqrt(), [param], opt, scaler)
    assert param.item() == 0.0
    assert not opt.state
    assert scaler.get_scale() == 8


@pytest.mark.parametrize("kind", ["latents", "conds"])
def test_invalid_cached_tensor_rejected_before_training(trainer, kind):
    cache = {"latents": [torch.zeros(1, 4)], "conds": [torch.zeros(1, 4)]}
    trainer.validate_cache(cache)
    cache[kind][0][0, 2] = float("nan")
    with pytest.raises(ValueError, match=kind):
        trainer.validate_cache(cache)


@pytest.mark.parametrize("completed", [False, True])
def test_main_recovers_export_after_interruption_before_export(trainer, tmp_path, monkeypatch, completed):
    """Restore real optimizer/weights through main; isolate only model loading and MPS."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    training = setup_training(trainer)
    for _ in range(5):
        update(trainer, training)
    trainer.save_state(tmp_path / "state.pt", 5, *training)
    expected = trainer.lora_state_dict("qwen21", training[0], 0)
    # Simulate a process stopping after the durable state but before its step-5 export.
    target = tmp_path / "recovery_s5.safetensors"
    assert not target.exists()
    torch.save({"latents": [torch.ones(1, 3)], "conds": [torch.ones(1, 3)]},
               tmp_path / "cache_qwen21_512.pt")
    (tmp_path / "dataset.json").write_text('{"images": []}')
    base_path = tmp_path / "base.safetensors"
    base_path.write_bytes(b"frozen base fixture")
    trainer.cb.folder_paths = SimpleNamespace(get_full_path_or_raise=lambda *args: str(base_path))
    fresh = setup_training(trainer)

    class CPUDeviceBoundary:
        mps = SimpleNamespace(driver_allocated_memory=lambda: 0, synchronize=lambda: None)

        def device(self, value):
            return torch.device("cpu" if value == "mps" else value)

        def __getattr__(self, name):
            return getattr(torch, name)

    class DiffusionModel(torch.nn.Module):
        transformer_blocks = []

        def forward(self, x, *args, **kwargs):
            return torch.zeros_like(x)

    model = SimpleNamespace(diffusion_model=DiffusionModel(), process_latent_in=lambda x: x,
                            get_dtype_inference=lambda: torch.float32)
    patcher = SimpleNamespace(model=model, set_model_compute_dtype=lambda dtype: None)
    trainer.cb.N = {"UNETLoader": lambda: SimpleNamespace(load_unet=lambda *args: (patcher,))}
    trainer.cb.mm = SimpleNamespace(load_models_gpu=lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "torch", CPUDeviceBoundary())
    monkeypatch.setattr(trainer, "patch_arch_for_training", lambda arch: None)
    monkeypatch.setattr(trainer, "patch_model", lambda *args: (fresh[0], 0))
    monkeypatch.setattr(trainer.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))
    calls = []

    def next_update(*args):
        assert target.exists(), "milestone export must be recovered before the next update"
        calls.append(1)
        raise InterruptedError("stop test before next optimizer update")

    monkeypatch.setattr(trainer, "optimizer_step", next_update)
    monkeypatch.setattr(sys, "argv", ["train_lora.py", "--arch", "qwen21", "--dataset", str(tmp_path),
                        "--out", str(tmp_path), "--name", "recovery", "--resume", "--no-checkpoint",
                        "--unet", "base.safetensors", "--rank", "2", "--lr", "0.01", "--seed", "7",
                        "--save-every", "5", "--steps", "5" if completed else "10"])
    args = trainer.parse()
    identity = trainer.training_identity(args, trainer.ARCHS[args.arch], tmp_path / "cache_qwen21_512.pt")
    # Current Qwen recovery states bind their original inputs; legacy states cannot.
    trainer.save_state(tmp_path / "state.pt", 5, *training, identity=identity)
    (tmp_path / "log.jsonl").write_text(json.dumps({"meta": dict(vars(args), act_dtype="torch.bfloat16", identity=identity)}) + "\n")
    if completed:
        trainer.main()
        assert calls == []
        # Repeating a completed resume is idempotent and does not optimize again.
        trainer.main()
        assert calls == []
    else:
        with pytest.raises(InterruptedError):
            trainer.main()
        assert calls == [1]
    assert_tree_equal(load_file(target), expected)
    with safe_open(target, framework="pt") as saved:
        assert saved.metadata()["steps"] == "5"
    assert torch.load(tmp_path / "state.pt", map_location="cpu")["step"] == 5
