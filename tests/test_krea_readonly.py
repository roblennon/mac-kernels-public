"""CPU-only contracts for the explicitly scoped Krea read-only loader."""
import ast
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
from types import SimpleNamespace
import weakref

import pytest
import safetensors
from safetensors.torch import save_file
import torch

from bench import krea_readonly as readonly


ROOT = Path(__file__).resolve().parents[1]


def source_function(relative, name, namespace):
    """Use Comfy's real pure helpers without bootstrapping its device manager."""
    source = ast.parse((ROOT / "vendor/ComfyUI" / relative).read_text())
    for part in name.split("."):
        source = next(item for item in source.body if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name == part)
    node = source
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), namespace)
    return namespace[node.name]


@pytest.fixture
def runtime():
    calls = []

    def ordinary(*args, **kwargs):
        calls.append((args, kwargs))
        return "ordinary loader"

    utils = SimpleNamespace(
        torch=torch, safetensors=SimpleNamespace(safe_open=safetensors.safe_open),
        _TYPES={"F32": torch.float32, "BF16": torch.bfloat16, "I8": torch.int8, "U8": torch.uint8},
        DISABLE_MMAP=False, load_torch_file=ordinary,
        safetensors_header=source_function("comfy/utils.py", "safetensors_header", {"struct": struct}),
        comfy=SimpleNamespace(memory_management=SimpleNamespace(aimdo_enabled=False),
                              storage=SimpleNamespace(annotate_state_dict=source_function(
                                  "comfy/storage.py", "annotate_state_dict", {"os": os}))),
    )
    return utils, calls


@pytest.fixture
def checkpoint(tmp_path):
    state = {"int8": torch.tensor([-128, -1, 0, 127], dtype=torch.int8),
             "uint8": torch.tensor([0, 128, 255], dtype=torch.uint8),
             "bf16": torch.tensor([-2.5, 0, 7.25], dtype=torch.bfloat16),
             "float": torch.arange(6, dtype=torch.float32).reshape(2, 3),
             "empty": torch.empty((0, 3), dtype=torch.bfloat16),
             "scalar": torch.tensor(1.25)}
    path = tmp_path / (readonly.CHECKPOINT_STEM + ".safetensors")
    save_file(state, path, metadata={"source": "CPU contract"})
    return path, state


def admitted(utils, path, **overrides):
    options = dict(execution_device=torch.device("mps"), native_int8=True)
    options.update(overrides)
    return readonly.readonly_krea_load(utils, path, **options)


def test_real_tensors_metadata_storage_lifetime_and_file_unchanged(runtime, checkpoint):
    utils, ordinary_calls = runtime
    path, expected = checkpoint
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    original = utils.load_torch_file
    with torch.inference_mode(), admitted(utils, path) as audit:
        state, metadata = utils.load_torch_file(str(path), return_metadata=True)
        assert metadata == {"source": "CPU contract"}
        assert audit["loads"] == 1 and audit["tensor_count"] == len(expected)
        assert audit["header_sha256"] == hashlib.sha256(utils.safetensors_header(str(path))).hexdigest()
        for name, value in expected.items():
            assert torch.equal(state[name], value)
            storage = state[name].untyped_storage()
            assert storage._comfy_mmap is True
            assert storage._comfy_source_path == str(path)
            assert not hasattr(storage, "_comfy_tensor_file_slice")
            assert not hasattr(storage, "_comfy_tensor_mmap_refs")
            owner, view = getattr(storage, readonly.OWNER_ATTRIBUTE)
            assert view.readonly
            with pytest.raises(TypeError):
                owner[0] = 0
        parameter = torch.nn.Parameter(state["float"], requires_grad=False)
        survivor = parameter.detach().view(-1)[1:]
        pointer = state["float"].data_ptr()
        owner_ref = weakref.ref(owner)
        del state, storage, owner, view, parameter
    assert utils.load_torch_file is original and ordinary_calls == []
    gc.collect()
    assert owner_ref() is not None
    assert survivor.data_ptr() == pointer + survivor.element_size()
    assert torch.equal(survivor, expected["float"].flatten()[1:])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    del survivor
    gc.collect()
    assert owner_ref() is None


def test_only_official_metadata_accessors_are_used(runtime, checkpoint):
    utils, _ = runtime
    path, expected = checkpoint
    calls = []

    class MetadataOnly:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)
            self.reader = safetensors.safe_open(*args, **kwargs)

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            return self.reader.__exit__(*args)

        def keys(self):
            return self.reader.keys()

        def metadata(self):
            return self.reader.metadata()

        def get_slice(self, name):
            return self.reader.get_slice(name)

        def get_tensor(self, *args):
            pytest.fail("Validation copied tensor data")

        get_tensors = get_tensor

    utils.safetensors.safe_open = MetadataOnly
    with torch.inference_mode(), admitted(utils, path):
        actual = utils.load_torch_file(path, device=torch.device("cpu"))
    assert calls == [{"framework": "numpy", "device": "cpu"}]
    assert actual.keys() == expected.keys()


def write_raw(path, header, payload):
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def test_unaligned_offsets_scalar_empty_and_absent_metadata(runtime, tmp_path):
    utils, _ = runtime
    path = tmp_path / (readonly.CHECKPOINT_STEM + ".sft")
    header = {"first": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]},
              "unaligned_bf16": {"dtype": "BF16", "shape": [1], "data_offsets": [1, 3]},
              "unaligned_float": {"dtype": "F32", "shape": [], "data_offsets": [3, 7]},
              "empty": {"dtype": "U8", "shape": [0], "data_offsets": [7, 7]}}
    write_raw(path, header, b"\xff" + struct.pack("<Hf", 0x4020, 3.5))
    with torch.inference_mode(), admitted(utils, path):
        state, metadata = utils.load_torch_file(path, return_metadata=True)
    assert metadata is None
    assert state["first"].item() == -1
    assert state["unaligned_bf16"].item() == 2.5
    assert state["unaligned_float"].shape == () and state["unaligned_float"].item() == 3.5
    assert state["empty"].shape == (0,)


@pytest.mark.parametrize("corruption", ["gap", "overlap", "shape", "truncated", "trailing", "metadata"])
def test_official_validator_rejects_corruption_before_tensor_construction(runtime, tmp_path, monkeypatch, corruption):
    utils, _ = runtime
    path = tmp_path / (readonly.CHECKPOINT_STEM + ".safetensors")
    header = {"a": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]},
              "b": {"dtype": "I8", "shape": [1], "data_offsets": [1, 2]}}
    payload = b"ab"
    if corruption == "gap":
        header["b"]["data_offsets"] = [2, 3]
        payload = b"abc"
    elif corruption == "overlap":
        header["b"]["data_offsets"] = [0, 1]
    elif corruption == "shape":
        header["b"]["shape"] = [2]
    elif corruption == "truncated":
        payload = b"a"
    elif corruption == "trailing":
        payload = b"abc"
    elif corruption == "metadata":
        header["__metadata__"] = {"invalid": 1}
    write_raw(path, header, payload)
    monkeypatch.setattr(torch, "frombuffer", lambda *a, **k: pytest.fail("Constructed corrupt tensor"))
    with torch.inference_mode(), admitted(utils, path):
        with pytest.raises(safetensors.SafetensorError):
            utils.load_torch_file(path)


def test_dtype_outside_actual_krea_profile_is_refused(runtime, tmp_path):
    utils, _ = runtime
    path = tmp_path / (readonly.CHECKPOINT_STEM + ".safetensors")
    save_file({"weight": torch.ones(2, dtype=torch.float16)}, path)
    with torch.inference_mode(), admitted(utils, path):
        with pytest.raises(ValueError, match="Unsupported Krea storage dtype"):
            utils.load_torch_file(path)


@pytest.mark.parametrize("mutation", ["replace", "resize", "header"])
def test_file_changed_after_admission_is_refused(runtime, checkpoint, mutation):
    utils, _ = runtime
    path, _ = checkpoint
    original = utils.load_torch_file
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="checkpoint changed"):
            with admitted(utils, path):
                if mutation == "replace":
                    replacement = path.with_suffix(".new")
                    replacement.write_bytes(path.read_bytes())
                    replacement.replace(path)
                elif mutation == "resize":
                    with path.open("ab") as stream:
                        stream.write(b"x")
                else:
                    before = path.stat()
                    data = path.read_bytes().replace(b"CPU contract", b"NEW contract")
                    path.write_bytes(data)
                    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                utils.load_torch_file(path)
    assert utils.load_torch_file is original


def test_change_during_official_validation_is_refused(runtime, checkpoint):
    utils, _ = runtime
    path, _ = checkpoint

    def changed(*args, **kwargs):
        reader = safetensors.safe_open(*args, **kwargs)
        replacement = path.with_suffix(".new")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        return reader

    utils.safetensors.safe_open = changed
    with torch.inference_mode(), admitted(utils, path):
        with pytest.raises(RuntimeError, match="checkpoint changed"):
            utils.load_torch_file(path)


def test_path_replaced_while_mapping_is_refused_before_publishing(runtime, checkpoint, monkeypatch):
    utils, _ = runtime
    path, _ = checkpoint
    ordinary_map = readonly.mmap.mmap

    def changed(*args, **kwargs):
        mapping = ordinary_map(*args, **kwargs)
        replacement = path.with_suffix(".new")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        return mapping

    monkeypatch.setattr(readonly.mmap, "mmap", changed)
    with torch.inference_mode(), admitted(utils, path) as audit:
        with pytest.raises(RuntimeError, match="checkpoint changed"):
            utils.load_torch_file(path)
        assert audit["loads"] == 0


def test_other_exact_paths_forward_and_context_restores_on_exception(runtime, checkpoint, tmp_path):
    utils, calls = runtime
    path, _ = checkpoint
    original = utils.load_torch_file
    other = tmp_path / "elsewhere" / path.name
    with torch.inference_mode(), pytest.raises(LookupError):
        with admitted(utils, path):
            assert utils.load_torch_file(str(other), True, "cpu", True) == "ordinary loader"
            with pytest.raises(RuntimeError, match="already installed"):
                with admitted(utils, path):
                    pass
            raise LookupError("body failure")
    assert calls == [((str(other),), dict(safe_load=True, device="cpu", return_metadata=True))]
    assert utils.load_torch_file is original


@pytest.mark.parametrize("options", [dict(execution_device=torch.device("cpu")),
                                    dict(native_int8=False), dict(lora=True), dict(hooks=True),
                                    dict(inplace=True), dict(training=True)])
def test_unsafe_policies_do_not_install_loader(runtime, checkpoint, options):
    utils, _ = runtime
    path, _ = checkpoint
    original = utils.load_torch_file
    with torch.inference_mode(), pytest.raises(ValueError):
        with admitted(utils, path, **options):
            pytest.fail("Unsafe policy admitted")
    assert utils.load_torch_file is original


def test_runtime_policy_rechecked_at_load(runtime, checkpoint):
    utils, _ = runtime
    path, _ = checkpoint
    with pytest.raises(ValueError, match="inference_mode"):
        with admitted(utils, path):
            pass
    with torch.inference_mode(), admitted(utils, path):
        with torch.inference_mode(False), pytest.raises(ValueError, match="inference_mode"):
            utils.load_torch_file(path)
        with pytest.raises(ValueError, match="constructed on CPU"):
            utils.load_torch_file(path, device="mps")
        utils.DISABLE_MMAP = True
        with pytest.raises(ValueError, match="without AIMDO"):
            utils.load_torch_file(path)
        utils.DISABLE_MMAP = False
        utils.comfy.memory_management.aimdo_enabled = True
        with pytest.raises(ValueError, match="without AIMDO"):
            utils.load_torch_file(path)


def test_other_checkpoint_names_rejected(runtime, checkpoint):
    utils, _ = runtime
    path, _ = checkpoint
    wrong = path.with_name("other.safetensors")
    wrong.write_bytes(path.read_bytes())
    with torch.inference_mode(), pytest.raises(ValueError, match="restricted"):
        with admitted(utils, wrong):
            pass


def test_import_never_bootstraps_torch_comfy_or_gpu():
    script = "import runpy,sys; runpy.run_path(sys.argv[1]); assert not any(k == 'torch' or k == 'comfy' or k.startswith(('torch.', 'comfy.')) for k in sys.modules)"
    subprocess.run([sys.executable, "-c", script, str(ROOT / "bench/krea_readonly.py")], check=True)


@pytest.fixture
def residency_runtime(monkeypatch):
    # The pretrained model and GPU allocation are external boundaries. Exercise
    # the real LoadedModel forwarding method with controlled allocation results.
    class Krea2:
        resident_bytes = 0

    monkeypatch.setitem(sys.modules, "comfy.model_base", SimpleNamespace(Krea2=Krea2))

    class LoadedModel:
        def __init__(self, patcher):
            self.model, self.device = patcher, patcher.load_device

        model_use_more_vram = source_function(
            "comfy/model_management.py", "LoadedModel.model_use_more_vram", {})

    def make_patcher(model=None):
        model = Krea2() if model is None else model
        patcher = SimpleNamespace(model=model, load_device=torch.device("mps"),
                                  weight_inplace_update=False, patches={}, hook_patches={},
                                  current_hooks=None, forced_hooks=None, object_patches={},
                                  weight_wrapper_patches={}, dynamic=False, calls=[])
        patcher.is_dynamic = lambda: patcher.dynamic
        patcher.loaded_size = lambda: model.resident_bytes

        def allocate(device, extra, force_patch_weights=False):
            patcher.calls.append((device, extra, force_patch_weights))
            model.resident_bytes += extra
            return extra

        patcher.partially_load = allocate
        return patcher

    mm = SimpleNamespace(LoadedModel=LoadedModel, torch=torch, in_training=False)
    return mm, make_patcher, make_patcher()


def test_residency_caps_target_and_clones_by_underlying_identity(residency_runtime):
    mm, make_patcher, patcher = residency_runtime
    clone = make_patcher(patcher.model)
    loaded, cloned = mm.LoadedModel(patcher), mm.LoadedModel(clone)
    original = mm.LoadedModel.model_use_more_vram
    with torch.inference_mode(), readonly.bounded_krea_residency(mm, patcher, 100) as audit:
        assert loaded.model_use_more_vram(60) == 60
        assert cloned.model_use_more_vram(1e32) == 40
        assert loaded.model_use_more_vram(20) == 0
        assert loaded.model_use_more_vram(-10) == -10
        assert cloned.model_use_more_vram(20) == 10
    assert mm.LoadedModel.model_use_more_vram is original
    assert patcher.loaded_size() == 100
    assert [e["allowed"] for e in audit["events"]] == [60, 40, 0, -10, 10]
    assert all(e["loaded_after"] <= 100 for e in audit["events"])
    assert all(not force for _, _, force in patcher.calls + clone.calls)


def test_unrelated_models_keep_requests_devices_and_force_flag(residency_runtime):
    mm, make_patcher, patcher = residency_runtime
    other = make_patcher(SimpleNamespace(resident_bytes=0))
    other.load_device = torch.device("cpu")
    other.dynamic = True
    other.patches["weight"] = "VAE patch"
    loaded = mm.LoadedModel(other)
    with torch.inference_mode(), readonly.bounded_krea_residency(mm, patcher, 100) as audit:
        assert loaded.model_use_more_vram(500, force_patch_weights=True) == 500
    assert other.calls == [(torch.device("cpu"), 500, True)]
    assert audit["events"] == [] and patcher.calls == []


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan"), 4 * 2**30 + 1, True, "100", None])
def test_unsafe_residency_budgets_rejected_before_install(residency_runtime, budget):
    mm, _, patcher = residency_runtime
    original = mm.LoadedModel.model_use_more_vram
    with torch.inference_mode(), pytest.raises(ValueError, match="budget"):
        with readonly.bounded_krea_residency(mm, patcher, budget):
            pass
    assert mm.LoadedModel.model_use_more_vram is original


@pytest.mark.parametrize("field,value", [("patches", {"weight": 1}), ("hook_patches", {"hook": 1}),
                                        ("weight_inplace_update", True), ("current_hooks", object()),
                                        ("forced_hooks", object()), ("object_patches", {"weight": 1}),
                                        ("weight_wrapper_patches", {"weight": 1}), ("dynamic", True),
                                        ("load_device", torch.device("cpu"))])
def test_mutations_after_admission_are_refused_before_allocation(residency_runtime, field, value):
    mm, _, patcher = residency_runtime
    loaded = mm.LoadedModel(patcher)
    original = mm.LoadedModel.model_use_more_vram
    with torch.inference_mode(), pytest.raises(ValueError):
        with readonly.bounded_krea_residency(mm, patcher, 100):
            setattr(patcher, field, value)
            loaded.model_use_more_vram(100)
    assert patcher.calls == [] and mm.LoadedModel.model_use_more_vram is original


def test_patched_clone_cannot_bypass_residency_policy(residency_runtime):
    mm, make_patcher, patcher = residency_runtime
    clone = make_patcher(patcher.model)
    clone.patches["weight"] = "LoRA"
    with torch.inference_mode(), readonly.bounded_krea_residency(mm, patcher, 100):
        with pytest.raises(ValueError, match="immutable"):
            mm.LoadedModel(clone).model_use_more_vram(100)
    assert clone.calls == []


def test_identity_change_and_preexisting_over_budget_are_refused(residency_runtime):
    mm, make_patcher, patcher = residency_runtime
    with torch.inference_mode():
        patcher.model.resident_bytes = 101
        with pytest.raises(RuntimeError, match="exceed budget"):
            with readonly.bounded_krea_residency(mm, patcher, 100):
                pass
        patcher.model.resident_bytes = 0
        with readonly.bounded_krea_residency(mm, patcher, 100):
            patcher.model = make_patcher().model
            with pytest.raises(RuntimeError, match="identity changed"):
                mm.LoadedModel(patcher).model_use_more_vram(100)
    assert patcher.calls == []


def test_non_krea_model_rejected_before_install(residency_runtime):
    mm, make_patcher, _ = residency_runtime
    other = make_patcher(SimpleNamespace(resident_bytes=0))
    with torch.inference_mode(), pytest.raises(ValueError, match="Krea2"):
        with readonly.bounded_krea_residency(mm, other):
            pass


@pytest.mark.parametrize("failure", ["forced", "nan_request", "cpu_loaded_device", "training", "overshoot", "allocation_error", "nested"])
def test_residency_failures_restore_original_method(residency_runtime, failure):
    mm, _, patcher = residency_runtime
    loaded = mm.LoadedModel(patcher)
    original = mm.LoadedModel.model_use_more_vram
    with torch.inference_mode(), pytest.raises((ValueError, RuntimeError, LookupError)):
        with readonly.bounded_krea_residency(mm, patcher, 100):
            if failure == "forced":
                loaded.model_use_more_vram(100, force_patch_weights=True)
            elif failure == "nan_request":
                loaded.model_use_more_vram(float("nan"))
            elif failure == "cpu_loaded_device":
                loaded.device = torch.device("cpu")
                loaded.model_use_more_vram(100)
            elif failure == "training":
                mm.in_training = True
                loaded.model_use_more_vram(100)
            elif failure == "overshoot":
                def overshoot(*args, **kwargs):
                    patcher.model.resident_bytes = 101
                patcher.partially_load = overshoot
                loaded.model_use_more_vram(100)
            elif failure == "allocation_error":
                def fail(*args, **kwargs):
                    raise LookupError("allocation failed")
                patcher.partially_load = fail
                loaded.model_use_more_vram(100)
            else:
                with readonly.bounded_krea_residency(mm, patcher, 100):
                    pass
    assert mm.LoadedModel.model_use_more_vram is original


def test_concurrent_target_requests_share_remaining_budget(residency_runtime):
    mm, _, patcher = residency_runtime
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    ordinary_allocate = patcher.partially_load
    first = True

    def blocked(device, extra, force_patch_weights=False):
        nonlocal first
        if first:
            first = False
            entered.set()
            assert release.wait(5)
        return ordinary_allocate(device, extra, force_patch_weights=force_patch_weights)

    patcher.partially_load = blocked
    loaded = mm.LoadedModel(patcher)

    def request(second=False):
        with torch.inference_mode():
            if second:
                second_started.set()
            return loaded.model_use_more_vram(80)

    with torch.inference_mode(), readonly.bounded_krea_residency(mm, patcher, 100) as audit:
        with ThreadPoolExecutor(max_workers=2) as pool:
            one = pool.submit(request)
            assert entered.wait(5)
            two = pool.submit(request, True)
            try:
                assert second_started.wait(5)
            finally:
                release.set()
            assert one.result(timeout=5) == 80
            assert two.result(timeout=5) == 20
    assert patcher.loaded_size() == 100
    assert [event["loaded_after"] for event in audit["events"]] == [80, 100]


@pytest.mark.parametrize("mutation", ["patches", "loaded_patcher"])
def test_state_changed_during_allocation_is_not_accepted(residency_runtime, mutation):
    mm, make_patcher, patcher = residency_runtime
    loaded = mm.LoadedModel(patcher)

    def mutate(*args, **kwargs):
        if mutation == "patches":
            patcher.patches["weight"] = "late patch"
        else:
            loaded.model = make_patcher()

    patcher.partially_load = mutate
    with torch.inference_mode(), readonly.bounded_krea_residency(mm, patcher, 100) as audit:
        with pytest.raises((ValueError, RuntimeError)):
            loaded.model_use_more_vram(100)
        assert audit["events"] == []
