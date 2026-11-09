"""Bench-only read-only shared mapping for the native INT8 Krea checkpoint.

Install after ComfyUI's ordinary bootstrap, inside inference_mode, around only
UNETLoader.load_unet. Returned tensors retain the mapping after the hook exits.
The caller must keep these weights immutable for their entire lifetime: PyTorch
does not enforce the OS mapping's read-only contract on tensor operations.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import struct
import sys
import threading
import warnings


CHECKPOINT_STEM = "krea2_turbo_int8_convrot"
DTYPES = frozenset({"F32", "BF16", "I8", "U8"})
OWNER_ATTRIBUTE = "_krea_readonly_mapping"
MAX_RESIDENT_BYTES = 4 * 2**30


def _stat(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _unchanged(path, stream, identity):
    if _stat(os.fstat(stream.fileno())) != identity or _stat(path.stat()) != identity:
        raise RuntimeError("Krea checkpoint changed during read-only loading")


def _header(utils, path, stream, identity):
    _unchanged(path, stream, identity)
    header = utils.safetensors_header(str(path))
    if header is None:
        raise ValueError("Krea safetensors header exceeds ComfyUI's size limit")
    stream.seek(0)
    prefix = stream.read(8)
    if len(prefix) != 8 or struct.unpack("<Q", prefix)[0] != len(header):
        raise ValueError("Incomplete Krea safetensors header")
    if stream.read(len(header)) != header:
        raise RuntimeError("Krea checkpoint header changed during read-only loading")
    _unchanged(path, stream, identity)
    return header


def _policy(utils, execution_device, native_int8, lora, hooks, inplace, training):
    if sys.platform != "darwin" or getattr(execution_device, "type", None) != "mps":
        raise ValueError("Read-only Krea loading requires MPS execution on macOS")
    if native_int8 is not True or any((lora, hooks, inplace, training)):
        raise ValueError("Read-only Krea loading requires static native INT8 weights without LoRA, hooks, in-place updates, or training")
    if not utils.torch.is_inference_mode_enabled():
        raise ValueError("Read-only Krea loading requires torch.inference_mode")
    if utils.DISABLE_MMAP or utils.comfy.memory_management.aimdo_enabled:
        raise ValueError("Read-only Krea loading requires ordinary CPU mmap construction without AIMDO")


def _load(utils, path, identity, header_bytes, *, allowed_dtypes=DTYPES):
    """Validate with safetensors, then expose the same checked bytes to Torch."""
    torch = utils.torch
    with path.open("rb") as stream:
        _unchanged(path, stream, identity)
        # NumPy selects safetensors' read-only metadata mapping, avoiding its
        # PyTorch CPU storage path (a writable private file mapping). Never ask
        # this reader for tensor data: only the official format validator and
        # its metadata accessors are needed here.
        with utils.safetensors.safe_open(str(path), framework="numpy", device="cpu") as reader:
            current_header = _header(utils, path, stream, identity)
            if current_header != header_bytes:
                raise RuntimeError("Krea checkpoint header changed after admission")
            header = json.loads(current_header)
            names = set(header) - {"__metadata__"}
            metadata = reader.metadata()
            if names != set(reader.keys()) or header.get("__metadata__") != metadata:
                raise ValueError("Safetensors header and validated metadata disagree")
            for name in names:
                entry = header[name]
                info = reader.get_slice(name)
                if entry["dtype"] not in allowed_dtypes:
                    raise ValueError(f"Unsupported Krea storage dtype: {entry['dtype']}")
                if entry["shape"] != info.get_shape() or entry["dtype"] != info.get_dtype():
                    raise ValueError(f"Safetensors header and validated tensor disagree: {name}")
            _unchanged(path, stream, identity)
            # ACCESS_READ is MAP_SHARED / PROT_READ on macOS. This is not the
            # writable MAP_PRIVATE allocation used by the stock Torch reader.
            mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            data_start = 8 + len(header_bytes)
            if mapping[:data_start] != struct.pack("<Q", len(header_bytes)) + header_bytes:
                raise RuntimeError("Mapped Krea header differs from the validated header")
            state = {}
            for name in reader.keys():
                entry = header[name]
                start, end = entry["data_offsets"]
                dtype = utils._TYPES[entry["dtype"]]
                view = memoryview(mapping)[data_start + start:data_start + end]
                if start == end:
                    value = torch.empty(entry["shape"], dtype=dtype, device="cpu")
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="The given buffer is not writable")
                        value = torch.frombuffer(view, dtype=dtype).reshape(entry["shape"])
                # frombuffer owns its buffer too; retain an explicit owner on
                # storage across Tensor/Parameter wrapping and state-dict deletion.
                # Do not advertise Comfy's CUDA file-slice path or its AIMDO
                # bounce protocol, neither of which this Python mapping supports.
                setattr(value.untyped_storage(), OWNER_ATTRIBUTE, (mapping, view))
                state[name] = value
            _unchanged(path, stream, identity)
            if mapping[:data_start] != struct.pack("<Q", len(header_bytes)) + header_bytes:
                raise RuntimeError("Mapped Krea header changed during loading")
    utils.comfy.storage.annotate_state_dict(state, str(path), mmap=True)
    return state, metadata


@contextmanager
def readonly_krea_load(utils, checkpoint, *, execution_device, native_int8,
                       lora=False, hooks=False, inplace=False, training=False,
                       reference_checkpoint=None):
    """Temporarily replace only one exact checkpoint's CPU construction loader.

    Policy arguments describe the whole benchmark, including work after this
    context exits. The single-threaded benchmark caller owns that invariant.
    ``utils`` is the already-bootstrapped comfy.utils module; importing this
    helper itself does not import Torch, ComfyUI, or query any GPU runtime.
    Yields header/stat provenance (not a hash of the model's tensor payload).
    """
    policy = (execution_device, native_int8, lora, hooks, inplace, training)
    def validate():
        _policy(utils, *policy)
    with _mapped_krea_load(utils, checkpoint, validate, reference_checkpoint=reference_checkpoint) as audit:
        yield audit


@contextmanager
def _mapped_krea_load(utils, checkpoint, policy, *, reference_checkpoint=None):
    """Shared mapping mechanics; each caller supplies its complete admission policy."""
    policy()
    path = Path(checkpoint).expanduser().resolve(strict=True)
    if (not path.is_file() or path.suffix.lower() not in {".safetensors", ".sft"}
            or (path.stem != CHECKPOINT_STEM and reference_checkpoint is None)):
        raise ValueError("Read-only loading is restricted to the native INT8 Krea checkpoint")
    if path.stem != CHECKPOINT_STEM:
        from krea_checkpoints import validate_custom_checkpoint
        validate_custom_checkpoint(path, reference_checkpoint)
    original = utils.load_torch_file
    if getattr(original, "_krea_readonly_loader", False):
        raise RuntimeError("Read-only Krea loader is already installed")
    with path.open("rb") as stream:
        identity = _stat(os.fstat(stream.fileno()))
        header_bytes = _header(utils, path, stream, identity)
    audit = {"checkpoint": str(path), "file_identity": list(identity),
             "header_sha256": hashlib.sha256(header_bytes).hexdigest(), "loads": 0}

    def load(ckpt, safe_load=False, device=None, return_metadata=False):
        if Path(ckpt).expanduser().resolve() != path:
            return original(ckpt, safe_load=safe_load, device=device, return_metadata=return_metadata)
        policy()
        if device is not None and utils.torch.device(device).type != "cpu":
            raise ValueError("Read-only Krea weights must be constructed on CPU")
        state, metadata = _load(utils, path, identity, header_bytes)
        audit.update(loads=audit["loads"] + 1, tensor_count=len(state))
        return (state, metadata) if return_metadata else state

    load._krea_readonly_loader = True
    utils.load_torch_file = load
    try:
        yield audit
    finally:
        utils.load_torch_file = original


def _static_patcher(mm, patcher, target):
    krea_type = getattr(sys.modules.get("comfy.model_base"), "Krea2", None)
    if krea_type is None or type(target) is not krea_type or patcher.model is not target:
        raise ValueError("Residency cap requires the admitted Krea2 model identity")
    if patcher.is_dynamic() or getattr(patcher.load_device, "type", None) != "mps":
        raise ValueError("Residency cap requires a static MPS patcher")
    if (patcher.weight_inplace_update or patcher.patches or patcher.hook_patches
            or patcher.current_hooks or patcher.forced_hooks or patcher.object_patches
            or patcher.weight_wrapper_patches):
        raise ValueError("Residency cap requires immutable weights without patches or hooks")
    if mm.in_training or not mm.torch.is_inference_mode_enabled():
        raise ValueError("Residency cap requires inference mode without training")


def _resident_size(patcher, budget_bytes):
    size = patcher.loaded_size()
    if (isinstance(size, bool) or not isinstance(size, (int, float))
            or not 0 <= size <= budget_bytes):
        raise RuntimeError(f"Krea resident weights exceed budget or have invalid size: {size} / {budget_bytes}")
    return size


@contextmanager
def bounded_krea_residency(mm, patcher, budget_bytes=MAX_RESIDENT_BYTES):
    """Bound one Krea model's existing partial-load requests during sampling.

    Adapted from the diagnostic t1_partial_load.resident_budget bridge. Other
    model identities, including the VAE, retain ComfyUI's ordinary policy.
    This caps resident model weights, not activations or the process allocator.
    Install after normal bootstrap; the caller owns an immutable inference run.
    """
    with _bounded_krea_residency(mm, patcher, budget_bytes,
                                 lambda candidate, target: _static_patcher(mm, candidate, target)) as audit:
        yield audit


@contextmanager
def _bounded_krea_residency(mm, patcher, budget_bytes, validate, reserve_bytes=0):
    """Shared cap mechanics with a caller-specific patcher invariant.

    ``reserve_bytes`` keeps room inside the cap for bytes Comfy adds after its fit check,
    such as LoRA factors it moves with each loaded module.
    """
    if (isinstance(budget_bytes, bool) or not isinstance(budget_bytes, (int, float))
            or not 0 < budget_bytes <= MAX_RESIDENT_BYTES):
        raise ValueError("Krea resident budget must be finite, positive, and at most 4 GiB")
    target = patcher.model
    validate(patcher, target)
    _resident_size(patcher, budget_bytes)
    original = mm.LoadedModel.model_use_more_vram
    if getattr(original, "_krea_resident_budget", False):
        raise RuntimeError("Krea residency cap is already installed")
    lock = threading.RLock()
    audit = {"budget_bytes": budget_bytes, "reserve_bytes": reserve_bytes, "events": []}

    def bounded(loaded, extra_memory, force_patch_weights=False):
        if patcher.model is not target:
            raise RuntimeError("Admitted Krea model identity changed during sampling")
        candidate = loaded.model
        if candidate.model is not target:
            return original(loaded, extra_memory, force_patch_weights=force_patch_weights)
        with lock:
            if loaded.model is not candidate:
                raise RuntimeError("Loaded Krea patcher changed during residency request")
            validate(patcher, target)
            validate(candidate, target)
            if getattr(loaded.device, "type", None) != "mps":
                raise ValueError("Loaded Krea model must use MPS")
            if force_patch_weights:
                raise ValueError("Read-only Krea weights cannot be force-patched")
            if (isinstance(extra_memory, bool) or not isinstance(extra_memory, (int, float))
                    or not math.isfinite(extra_memory)):
                raise ValueError("Krea residency request must be finite")
            before = _resident_size(candidate, budget_bytes)
            allowed = min(extra_memory, max(0, budget_bytes - before - reserve_bytes))
            result = original(loaded, allowed, force_patch_weights=False)
            if loaded.model is not candidate:
                raise RuntimeError("Loaded Krea patcher changed during residency request")
            validate(patcher, target)
            validate(candidate, target)
            after = _resident_size(candidate, budget_bytes)
            audit["events"].append({"requested": extra_memory, "allowed": allowed,
                                    "loaded_before": before, "loaded_after": after})
            return result

    bounded._krea_resident_budget = True
    mm.LoadedModel.model_use_more_vram = bounded
    try:
        yield audit
    finally:
        mm.LoadedModel.model_use_more_vram = original
