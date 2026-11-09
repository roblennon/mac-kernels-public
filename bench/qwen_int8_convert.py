"""Convert a diffusers-format Qwen-Image 2.1 transformer (bf16, one file or shards) to the
Comfy INT8 ConvRot layout, mirroring a reference INT8 file's recipe key for key.

The reference decides which weights are quantized (32 blocks x attention to_q/k/v/to_out and
img_mlp gate_up/out), their comfy_quant metadata, and the names of everything else. gate_up is
gate_layer and proj concatenated on the output axis, the order ComfyUI's loader and LoRA mapper
use. Quantization is comfy-kitchen's eager quantize_int8_convrot_weight (Hadamard-256 rotation,
per-row absmax/127 fp32 scales, deterministic rounding). Output is streamed one tensor at a time.

  python bench/qwen_int8_convert.py SOURCE_DIR_OR_FILE --reference REF.safetensors --out OUT.safetensors
      [--bf16-out MERGED.safetensors] [--validate] [--blocks 0,31]

--validate compares the produced tensors with the reference instead of writing (use it on the
base bf16 file against the shipped base INT8 file). --blocks limits block work for validation.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "comfy-kitchen"))
from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight

DTYPE_NAMES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.int8: "I8", torch.uint8: "U8"}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return n, header


class Source:
    """Tensors by key from one safetensors file or a sharded set (index json)."""
    def __init__(self, path):
        path = Path(path)
        files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
        self.where = {}
        for file in files:
            _, header = read_header(file)
            for key in header:
                self.where[key] = file
        self.handles = {}

    def get(self, key):
        file = self.where[key]
        if file not in self.handles:
            self.handles[file] = safe_open(file, framework="pt", device="cpu")
        return self.handles[file].get_tensor(key)

    def keys(self):
        return set(self.where)


def quant_bytes(reference, n, key):
    with open(reference, "rb") as f:
        _, header = read_header(reference)
        a, b = header[key]["data_offsets"]
        f.seek(8 + n + a)
        return f.read(b - a)


def plan(ref_header, blocks, source_keys):
    """Ordered (out_key, kind, source_keys) for every reference key; kind in copy/quant/scale/meta."""
    items = []
    for key in sorted(ref_header):
        if blocks is not None and key.startswith("transformer_blocks.") and int(key.split(".")[1]) not in blocks:
            continue
        if key.endswith(".comfy_quant"):
            items.append((key, "meta", None))
        elif key.endswith(".weight_scale"):
            items.append((key, "scale", None))
        elif ref_header[key]["dtype"] == "I8":
            base = key[:-len(".weight")]
            src = [key]
            if base.endswith(".gate_up") and key not in source_keys:  # diffusers layout: fuse gate_layer and proj
                src = [base.replace(".gate_up", ".gate_layer") + ".weight", base.replace(".gate_up", ".proj") + ".weight"]
            items.append((key, "quant", src))
        else:
            items.append((key, "copy", [key]))
    return items


def produce(source, items, ref_path, ref_n, ref_header, quant_cache):
    """Yield (key, dtype_name, shape, bytes) in plan order; quantized layers are computed once."""
    for key, kind, src in items:
        if kind == "copy":
            t = source.get(src[0]).contiguous()
            assert DTYPE_NAMES[t.dtype] == ref_header[key]["dtype"] and list(t.shape) == ref_header[key]["shape"], key
            yield key, DTYPE_NAMES[t.dtype], list(t.shape), t.view(torch.uint8).numpy().tobytes()
        elif kind == "meta":
            yield key, "U8", ref_header[key]["shape"], quant_bytes(ref_path, ref_n, key)
        elif kind == "quant":
            w = torch.cat([source.get(k) for k in src], dim=0) if len(src) > 1 else source.get(src[0])
            meta = json.loads(quant_bytes(ref_path, ref_n, key[:-len(".weight")] + ".comfy_quant"))
            assert meta["format"] == "int8_tensorwise" and meta["convrot"], meta
            q, scale = quantize_int8_convrot_weight(w.float(), meta["convrot_groupsize"])
            assert list(q.shape) == ref_header[key]["shape"], (key, q.shape)
            quant_cache[key[:-len(".weight")] + ".weight_scale"] = scale.float().contiguous()
            yield key, "I8", list(q.shape), q.contiguous().numpy().tobytes()
        elif kind == "scale":
            scale = quant_cache.pop(key)
            assert list(scale.shape) == ref_header[key]["shape"], (key, scale.shape)
            yield key, "F32", list(scale.shape), scale.numpy().tobytes()


def write_streamed(path, entries):
    """entries: list of (key, dtype, shape, nbytes, producer()) written in order."""
    header, offset = {}, 0
    for key, dtype, shape, nbytes, _ in entries:
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for _, _, _, nbytes, producer in entries:
            data = producer()
            assert len(data) == nbytes
            f.write(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out")
    ap.add_argument("--bf16-out")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--blocks", default=None)
    args = ap.parse_args()
    blocks = None if args.blocks is None else {int(b) for b in args.blocks.split(",")}
    source = Source(args.source)
    ref_n, ref_header = read_header(args.reference)
    items = plan(ref_header, blocks, source.keys())
    needed = {k for _, kind, src in items if src for k in src}
    missing = needed - source.keys()
    assert not missing, f"source lacks {sorted(missing)[:5]}"
    if blocks is None:
        extra = source.keys() - needed
        assert not extra, f"source has keys the reference recipe does not cover: {sorted(extra)[:5]}"

    if args.validate:
        cache, mismatches, checked = {}, [], 0
        with open(args.reference, "rb") as f:
            for key, dtype, shape, data in produce(source, items, args.reference, ref_n, ref_header, cache):
                a, b = ref_header[key]["data_offsets"]
                f.seek(8 + ref_n + a)
                ref = f.read(b - a)
                checked += 1
                if ref != data:
                    if dtype == "I8":
                        d = torch.frombuffer(bytearray(data), dtype=torch.int8).int() - torch.frombuffer(bytearray(ref), dtype=torch.int8).int()
                        mismatches.append((key, f"{int((d != 0).sum())} of {d.numel()} codes differ, max |diff| {int(d.abs().max())}"))
                    elif dtype == "F32":
                        d = torch.frombuffer(bytearray(data), dtype=torch.float32) - torch.frombuffer(bytearray(ref), dtype=torch.float32)
                        mismatches.append((key, f"max abs scale diff {float(d.abs().max()):.3e}"))
                    else:
                        mismatches.append((key, "bytes differ"))
        print(f"validated {checked} tensors against {Path(args.reference).name}: {len(mismatches)} mismatches")
        for key, why in mismatches[:20]:
            print("  ", key, why)
        return

    if args.bf16_out:
        keys = sorted(source.keys())
        def copier(k):
            return lambda: source.get(k).contiguous().view(torch.uint8).numpy().tobytes()
        entries = []
        for k in keys:
            t = source.get(k)
            entries.append((k, DTYPE_NAMES[t.dtype], list(t.shape), t.numel() * t.element_size(), copier(k)))
        write_streamed(args.bf16_out, entries)
        print("wrote", args.bf16_out, len(entries), "tensors")
    if args.out:
        cache = {}
        gen = produce(source, items, args.reference, ref_n, ref_header, cache)
        # Sizes are known from the reference header, so the stream can be written in one pass.
        sizes = {k: ref_header[k]["data_offsets"][1] - ref_header[k]["data_offsets"][0] for k, _, _ in items}
        def producer_for(key):
            def run():
                k, _, _, data = next(gen)
                assert k == key, (k, key)
                return data
            return run
        entries = [(k, ref_header[k]["dtype"], ref_header[k]["shape"], sizes[k], producer_for(k)) for k, _, _ in items]
        write_streamed(args.out, entries)
        print("wrote", args.out, len(entries), "tensors")


if __name__ == "__main__":
    main()
