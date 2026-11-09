"""Content-bound resume checks run before model loading and preserve legacy recovery files."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def subject(tmp_path, monkeypatch):
    boot = SimpleNamespace(ROOT=str(tmp_path))
    monkeypatch.setitem(sys.modules, "comfy_boot", boot)
    source = Path(__file__).resolve().parents[1] / "train" / "train_lora.py"
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("training_identity_subject", source)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    dataset, out = tmp_path / "dataset", tmp_path / "run"
    dataset.mkdir()
    out.mkdir()
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"frozen base weights")
    (dataset / "dataset.json").write_text('{"images": [{"file": "example.png", "caption": "one"}]}')
    cache = dataset / "cache_qwen21_512.pt"
    torch.save({"latents": [torch.ones(1, 3)], "conds": [torch.ones(1, 3)]}, cache)
    boot.folder_paths = SimpleNamespace(get_full_path_or_raise=lambda *args: str(base))
    args = SimpleNamespace(arch="qwen21", unet=base.name, dataset=str(dataset), out=str(out),
                           res=512, rank=2, lr=0.01, dtype=None, seed=7, no_checkpoint=False,
                           chunked_attn=True, name="identity", steps=20, stop_after=0,
                           save_every=5, state_every=2, val_every=5, log_every=1, profile_steps=0,
                           resume=True)
    return trainer, args, base, cache


def identity(subject):
    trainer, args, _, cache = subject
    return trainer.training_identity(args, trainer.ARCHS[args.arch], cache)


@pytest.mark.parametrize("field,value", [("arch", "krea2"), ("unet", "other.safetensors"), ("res", 1024),
                                        ("rank", 4), ("lr", 0.02), ("dtype", "fp16"), ("seed", 8),
                                        ("no_checkpoint", True), ("chunked_attn", False)])
def test_immutable_configuration_rejected(subject, field, value):
    trainer, args, _, _ = subject
    saved = {"identity": identity(subject)}
    setattr(args, field, value)
    with pytest.raises(ValueError, match="identity mismatch"):
        trainer.validate_resume_identity(saved, identity(subject), args)


@pytest.mark.parametrize("asset", ["base", "cache", "dataset"])
def test_same_path_replaced_bytes_rejected(subject, asset):
    trainer, args, base, cache = subject
    saved = {"identity": identity(subject)}
    path = {"base": base, "cache": cache, "dataset": Path(args.dataset) / "dataset.json"}[asset]
    original = path.read_bytes()
    # Retain size: a name/size-only fingerprint would miss this replacement.
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="identity mismatch"):
        trainer.validate_resume_identity(saved, identity(subject), args)


def test_cadence_and_total_steps_can_change_and_paths_can_move(subject, tmp_path):
    trainer, args, _, _ = subject
    saved = {"identity": identity(subject)}
    args.dtype = "bf16"  # Explicit dtype is equivalent to the original per-arch default.
    args.steps, args.stop_after = 30, 4
    args.state_every, args.save_every, args.val_every, args.log_every = 1, 2, 3, 4
    moved = tmp_path / "moved-data"
    shutil.copytree(args.dataset, moved)
    args.dataset = str(moved)
    current = trainer.training_identity(args, trainer.ARCHS[args.arch], moved / "cache_qwen21_512.pt")
    assert trainer.validate_resume_identity(saved, current, args) is False


def test_target_map_changes_rejected(subject):
    trainer, args, _, cache = subject
    saved = {"identity": identity(subject)}
    changed_spec = dict(trainer.ARCHS[args.arch], targets=("attn.to_q",))
    changed = trainer.training_identity(args, changed_spec, cache)
    with pytest.raises(ValueError, match="identity mismatch"):
        trainer.validate_resume_identity(saved, changed, args)


def test_legacy_krea_metadata_migrates_at_next_save_then_binds_inputs(subject):
    trainer, args, _, cache = subject
    args.arch = "krea2"
    path = Path(args.out) / "state.pt"
    lora = trainer.LoRA(3, 2, 2, 2, torch.Generator().manual_seed(3))
    opt = torch.optim.AdamW(lora.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    loss = lora(torch.randn(4, 3, generator=gen)).square().mean()
    trainer.optimizer_step(loss, list(lora.parameters()), opt, None)
    trainer.save_state(path, 8, {"layer": lora}, opt, None, gen)
    before = path.read_bytes()
    meta = dict(vars(args), act_dtype="torch.bfloat16")
    (Path(args.out) / "log.jsonl").write_text(json.dumps({"meta": meta}) + '\n{"interrupted":')
    args.dtype = "bf16"
    current = identity(subject)
    state = torch.load(path, map_location="cpu", mmap=True)
    assert trainer.validate_resume_identity(state, current, args) is True
    assert path.read_bytes() == before  # Validation alone never migrates/replaces the checkpoint.
    trainer.save_state(path, 8, {"layer": lora}, opt, None, gen, identity=current)
    upgraded = torch.load(path, map_location="cpu", mmap=True)
    assert trainer.validate_resume_identity(upgraded, identity(subject), args) is False
    torch.save({"latents": [torch.zeros(1, 3)], "conds": [torch.ones(1, 3)]}, cache)
    with pytest.raises(ValueError, match="identity mismatch"):
        trainer.validate_resume_identity(upgraded, identity(subject), args)


@pytest.mark.parametrize("recorded", [None, {"seed": 42}, {"dataset": "/different/dataset"}])
def test_legacy_missing_or_incompatible_metadata_rejected(subject, recorded):
    trainer, args, _, _ = subject
    args.arch = "krea2"
    meta = dict(vars(args), act_dtype="torch.float16")
    if recorded is not None:
        meta.update(recorded)
    (Path(args.out) / "log.jsonl").write_text(json.dumps({"step": 8} if recorded is None else {"meta": meta}) + "\n")
    with pytest.raises(ValueError, match="Legacy resume"):
        trainer.validate_resume_identity({}, identity(subject), args)


def test_legacy_default_dtype_uses_recorded_runtime_not_new_default(subject):
    trainer, args, _, _ = subject
    args.arch = "krea2"
    meta = dict(vars(args), act_dtype="torch.float16")
    (Path(args.out) / "log.jsonl").write_text(json.dumps({"meta": meta}) + "\n")
    trainer.ARCHS[args.arch] = dict(trainer.ARCHS[args.arch], dtype="bf16")
    with pytest.raises(ValueError, match="dtype"):
        trainer.validate_resume_identity({}, identity(subject), args)


@pytest.mark.parametrize("change", ["configuration", "cache", "base", "missing-cache"])
def test_main_rejects_before_loading_or_mutating_any_model(subject, monkeypatch, change):
    trainer, args, base, cache = subject
    saved_identity = identity(subject)
    state_path = Path(args.out) / "state.pt"
    torch.save({"step": 8, "identity": saved_identity}, state_path)
    before = state_path.read_bytes()
    if change == "configuration":
        args.lr *= 2
    elif change == "cache":
        torch.save({"latents": [torch.zeros(1, 3)], "conds": [torch.ones(1, 3)]}, cache)
    elif change == "base":
        base.write_bytes(b"replacement weights")
    else:
        cache.unlink()
    monkeypatch.setattr(trainer, "parse", lambda: args)
    monkeypatch.setattr(trainer, "patch_arch_for_training", lambda *a: pytest.fail("model state was mutated"))
    trainer.cb.N = {"UNETLoader": lambda: pytest.fail("model loader was reached")}
    trainer.cb.mm = SimpleNamespace()
    with pytest.raises((ValueError, FileNotFoundError)):
        trainer.main()
    assert state_path.read_bytes() == before
    assert not (Path(args.out) / "log.jsonl").exists()


def test_bucket_policy_is_bound_without_changing_default_v1_identity(subject):
    trainer, args, _, cache = subject
    original = identity(subject)
    args.buckets = None
    assert identity(subject) == original and original["version"] == 1
    assert "shape_policy" not in original["config"]
    args.buckets = [[512, 768], [768, 512]]
    bucket_identity = identity(subject)
    assert bucket_identity["config"]["shape_policy"]["mode"] == "buckets"
    args.buckets.reverse()
    assert identity(subject) == bucket_identity
    args.buckets = [[512, 512]]
    with pytest.raises(ValueError, match="identity mismatch"):
        trainer.validate_resume_identity({"identity": bucket_identity}, identity(subject), args)
    with pytest.raises(ValueError, match="Legacy resume cannot establish"):
        trainer.validate_resume_identity({}, identity(subject), args)


def explicit_cache(subject, tmp_path):
    trainer, args, _, _ = subject
    path = tmp_path / 'corrected.pt'
    cache = {'captions': ['one'], 'latents': [torch.zeros(1, 64, 32, 32)],
             'conds': [torch.ones(1, 3, 4096)]}
    torch.save(cache, path)
    args.cache_file = str(path)
    args.preview_every = 0
    return path, cache


def test_explicit_cache_drives_real_main_resume_identity(subject, tmp_path, monkeypatch):
    trainer, args, _, automatic = subject
    selected, cache = explicit_cache(subject, tmp_path)
    saved = trainer.training_identity(args, trainer.ARCHS[args.arch], selected)
    state = Path(args.out) / 'state.pt'
    torch.save({'step': 8, 'identity': saved}, state)
    original_state, original_automatic = state.read_bytes(), automatic.read_bytes()
    monkeypatch.setattr(trainer, 'parse', lambda: args)
    monkeypatch.setattr(trainer, 'patch_arch_for_training', lambda *a: None)
    monkeypatch.setattr(trainer, 'build_cache', lambda *a: pytest.fail('explicit cache must never be built'))
    trainer.cb.mm = SimpleNamespace()

    class ModelBoundaryReached(Exception):
        pass

    def model_boundary():
        raise ModelBoundaryReached

    trainer.cb.N = {'UNETLoader': model_boundary}
    with pytest.raises(ModelBoundaryReached):
        trainer.main()
    # Replacing the selected file, not the automatic cache, must invalidate the saved run.
    cache['latents'][0].fill_(1)
    torch.save(cache, selected)
    with pytest.raises(ValueError, match='identity mismatch'):
        trainer.main()
    assert state.read_bytes() == original_state and automatic.read_bytes() == original_automatic


@pytest.mark.parametrize('defect', ['missing', 'directory', 'resolution', 'channels', 'caption_order'])
def test_explicit_cache_rejected_before_model_mutation(subject, tmp_path, monkeypatch, defect):
    trainer, args, _, _ = subject
    selected, cache = explicit_cache(subject, tmp_path)
    args.resume = False
    if defect == 'missing':
        selected.unlink()
    elif defect == 'directory':
        selected.unlink()
        selected.mkdir()
    else:
        if defect == 'resolution':
            args.res = 1024  # Selecting the otherwise healthy512 cache is not a1024 run.
        elif defect == 'channels':
            cache['latents'][0] = torch.zeros(1, 16, 32, 32)
        else:
            (Path(args.dataset) / 'dataset.json').write_text(json.dumps({'images': [
                {'file': 'one.png', 'caption': 'one'}, {'file': 'two.png', 'caption': 'two'}]}))
            cache['captions'] = ['two', 'one']
            cache['latents'].append(torch.zeros(1, 64, 32, 32))
            cache['conds'].append(torch.ones(1, 3, 4096))
        torch.save(cache, selected)
    monkeypatch.setattr(trainer, 'parse', lambda: args)
    monkeypatch.setattr(trainer, 'patch_arch_for_training', lambda *a: pytest.fail('model state was mutated'))
    monkeypatch.setattr(trainer, 'build_cache', lambda *a: pytest.fail('explicit cache must never be built'))
    error, message = {
        'missing': (FileNotFoundError, 'corrected.pt'),
        'directory': (IsADirectoryError, 'corrected.pt'),
        'resolution': (ValueError, 'Explicit cache latent 0 has shape'),
        'channels': (ValueError, 'Explicit cache latent 0 has shape'),
        'caption_order': (ValueError, 'Explicit cache captions/order'),
    }[defect]
    with pytest.raises(error, match=message):
        trainer.main()


def test_legacy_resume_cannot_adopt_explicit_cache(subject, tmp_path):
    trainer, args, _, _ = subject
    selected, _ = explicit_cache(subject, tmp_path)
    with pytest.raises(ValueError, match='fingerprinted checkpoint'):
        trainer.validate_resume_identity({}, trainer.training_identity(args, trainer.ARCHS[args.arch], selected), args)


def test_explicit_krea_cache_keeps_single_frame_dimension(subject):
    trainer, args, _, _ = subject
    args.arch = 'krea2'
    cache = {'captions': ['one'], 'latents': [torch.zeros(1, 16, 1, 64, 64)],
             'conds': [torch.ones(1, 3, 30720)]}
    policy = trainer.shape_policy(args.res)
    trainer.validate_explicit_cache(cache, args, policy)
    cache['latents'][0] = cache['latents'][0].squeeze(2)
    with pytest.raises(ValueError, match='Explicit cache latent 0 has shape'):
        trainer.validate_explicit_cache(cache, args, policy)


def test_explicit_bucket_cache_validates_portrait_landscape_and_file_order(subject):
    trainer, args, _, _ = subject
    (Path(args.dataset) / 'dataset.json').write_text(json.dumps({'images': [
        {'file': 'wide.png', 'caption': 'wide'}, {'file': 'tall.png', 'caption': 'tall'}]}))
    policy = trainer.shape_policy(args.res, [[1024, 512], [512, 1024]])
    shapes = [[1, 64, 32, 64], [1, 64, 64, 32]]
    cache = {'captions': ['wide', 'tall'], 'shape_policy': policy,
             'latents': [torch.zeros(shape) for shape in shapes], 'conds': [torch.ones(1, 3, 4096)] * 2,
             'samples': [{'file': 'wide.png', 'target_size': [1024, 512], 'latent_shape': shapes[0]},
                         {'file': 'tall.png', 'target_size': [512, 1024], 'latent_shape': shapes[1]}]}
    trainer.validate_bucket_cache(cache, policy)
    trainer.validate_explicit_cache(cache, args, policy)
    cache['samples'][0]['file'] = 'tall.png'
    with pytest.raises(ValueError, match='Explicit bucket cache file order'):
        trainer.validate_explicit_cache(cache, args, policy)


@pytest.mark.parametrize("resolution", [512, 1024])
@pytest.mark.parametrize("replace_cache", [False, True])
def test_legacy_qwen_default_cache_rejected_before_model_mutation(subject, monkeypatch, resolution, replace_cache):
    trainer, args, _, _ = subject
    args.res = resolution
    cache = Path(trainer.cache_filename(args.dataset, args.arch, resolution))
    payload = {"captions": ["one"], "latents": [torch.ones(1, 64, resolution // 16, resolution // 16)],
               "conds": [torch.ones(1, 3, 4096)]}
    torch.save(payload, cache)
    state_path, log_path = Path(args.out) / "state.pt", Path(args.out) / "log.jsonl"
    torch.save({"step": 8}, state_path)
    log_path.write_text(json.dumps({"meta": dict(vars(args), act_dtype="torch.bfloat16")}) + "\n")
    before_state, before_log = state_path.read_bytes(), log_path.read_bytes()
    old_identity = trainer.training_identity(args, trainer.ARCHS[args.arch], cache)
    if replace_cache:
        payload["latents"][0].zero_()
        torch.save(payload, cache)
    current = trainer.training_identity(args, trainer.ARCHS[args.arch], cache)
    assert (current != old_identity) == replace_cache
    cache_bytes = cache.read_bytes()
    monkeypatch.setattr(trainer, "parse", lambda: args)
    monkeypatch.setattr(trainer, "patch_arch_for_training", lambda *a: pytest.fail("model was mutated"))
    monkeypatch.setattr(trainer, "build_cache", lambda *a: pytest.fail("resume cache was rebuilt"))
    trainer.cb.N = {"UNETLoader": lambda: pytest.fail("model loader was reached")}
    trainer.cb.mm = SimpleNamespace()
    with pytest.raises(ValueError, match="Legacy Qwen resume cannot verify cached targets"):
        trainer.main()
    assert state_path.read_bytes() == before_state and log_path.read_bytes() == before_log
    assert cache.read_bytes() == cache_bytes and not vars(trainer.cb.mm)
    # Modern identities still accept identical inputs and reject the changed bytes.
    assert trainer.validate_resume_identity({"identity": current}, current, args) is False
    if replace_cache:
        with pytest.raises(ValueError, match="identity mismatch"):
            trainer.validate_resume_identity({"identity": old_identity}, current, args)
