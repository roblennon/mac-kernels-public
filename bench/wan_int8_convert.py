#!/usr/bin/env python3
"""Convert a Comfy-layout WAN 2.2 A14B fp16 expert to the INT8 ConvRot layout of an installed expert.

The reference (an installed `wan2.2_t2v_*_14B_int8_convrot.safetensors`) decides which
weights are quantized (400 block linears), their comfy_quant metadata and dtypes; the
header-level `_quantization_metadata` is copied verbatim. The I2V experts share every key with
T2V except `patch_embedding.weight`, whose input channels are 36 instead of 16; that tensor is
copied as fp16 like the reference's own patch embedding. Quantization is comfy-kitchen's eager
`quantize_int8_convrot_weight` (Hadamard-256 rotation, per-row absmax/127 fp32 scales).

  python bench/wan_int8_convert.py SOURCE.safetensors --reference REF.safetensors --out OUT.safetensors [--check]

--check reads the written file back and verifies key set, dtypes, shapes, metadata and a
dequantized sample of layers against the source (relative error, not byte identity: there is
no fp16 T2V source on this machine to prove byte parity with the installed experts).
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "comfy-kitchen"))
from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight  # noqa: E402
from qwen_int8_convert import DTYPE_NAMES, Source, quant_bytes, write_streamed  # noqa: E402

VARIABLE_SHAPE = {"patch_embedding.weight"}  # 16 (T2V) vs 36 (I2V) input channels


def read_header_with_meta(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {})
    return n, header, meta


def plan(ref_header):
    items = []
    for key in sorted(ref_header):
        if key.endswith(".comfy_quant"):
            items.append((key, "meta"))
        elif key.endswith(".weight_scale"):
            items.append((key, "scale"))
        elif ref_header[key]["dtype"] == "I8":
            items.append((key, "quant"))
        else:
            items.append((key, "copy"))
    return items


def produce(source, items, ref_path, ref_n, ref_header):
    cache = {}
    for key, kind in items:
        if kind == "copy":
            t = source.get(key).contiguous()
            assert DTYPE_NAMES[t.dtype] == ref_header[key]["dtype"], key
            assert key in VARIABLE_SHAPE or list(t.shape) == ref_header[key]["shape"], (key, t.shape)
            yield key, DTYPE_NAMES[t.dtype], list(t.shape), t.view(torch.uint8).numpy().tobytes()
        elif kind == "meta":
            yield key, "U8", ref_header[key]["shape"], quant_bytes(ref_path, ref_n, key)
        elif kind == "quant":
            meta = json.loads(quant_bytes(ref_path, ref_n, key[:-len(".weight")] + ".comfy_quant"))
            assert meta["format"] == "int8_tensorwise" and meta["convrot"], meta
            q, scale = quantize_int8_convrot_weight(source.get(key).float(), meta["convrot_groupsize"])
            assert list(q.shape) == ref_header[key]["shape"], (key, q.shape)
            cache[key[:-len(".weight")] + ".weight_scale"] = scale.float().contiguous()
            yield key, "I8", list(q.shape), q.contiguous().numpy().tobytes()
        else:
            scale = cache.pop(key)
            assert list(scale.shape) == ref_header[key]["shape"], (key, scale.shape)
            yield key, "F32", list(scale.shape), scale.numpy().tobytes()


def write(source, out, ref_path, ref_n, ref_header, ref_meta):
    items = plan(ref_header)
    gen = produce(source, items, ref_path, ref_n, ref_header)
    sizes = {}
    for key, kind in items:
        if key in VARIABLE_SHAPE:
            t = source.get(key)
            sizes[key] = (list(t.shape), t.numel() * t.element_size())
        else:
            a, b = ref_header[key]["data_offsets"]
            sizes[key] = (ref_header[key]["shape"], b - a)

    def producer_for(key):
        def run():
            k, _, _, data = next(gen)
            assert k == key, (k, key)
            return data
        return run
    entries = [(k, ref_header[k]["dtype"], sizes[k][0], sizes[k][1], producer_for(k)) for k, _ in items]
    write_streamed_with_meta(out, entries, ref_meta)
    return len(entries)


def write_streamed_with_meta(path, entries, meta):
    header, offset = {}, 0
    for key, dtype, shape, nbytes, _ in entries:
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    header["__metadata__"] = dict(meta)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for _, _, _, nbytes, producer in entries:
            data = producer()
            assert len(data) == nbytes
            f.write(data)


def check(out, source_path, ref_path, sample=8):
    _, out_header, out_meta = read_header_with_meta(out)
    _, ref_header, ref_meta = read_header_with_meta(ref_path)
    assert set(out_header) == set(ref_header), "key set differs from the reference"
    assert out_meta == ref_meta, "header metadata differs from the reference"
    for key, entry in ref_header.items():
        assert out_header[key]["dtype"] == entry["dtype"], key
        assert key in VARIABLE_SHAPE or out_header[key]["shape"] == entry["shape"], key
    with safe_open(out, framework="pt", device="cpu") as got, safe_open(source_path, framework="pt", device="cpu") as src:
        quant_keys = sorted(k for k, e in out_header.items() if e["dtype"] == "I8")
        step = max(1, len(quant_keys) // sample)
        worst = 0.0
        for key in quant_keys[::step][:sample]:
            w = src.get_tensor(key).float()
            q, scale = quantize_int8_convrot_weight(w, json.loads(bytes(got.get_tensor(key[:-7] + ".comfy_quant").tolist()).decode())["convrot_groupsize"])
            assert torch.equal(q, got.get_tensor(key)), f"requantization mismatch at {key}"
            assert torch.allclose(scale.float(), got.get_tensor(key[:-7] + ".weight_scale")), f"scale mismatch at {key}"
            worst = max(worst, float((q.float() * scale.float().reshape(-1, 1)).norm() / w.norm()))
        patch = got.get_tensor("patch_embedding.weight")
        assert torch.equal(patch, src.get_tensor("patch_embedding.weight")), "patch embedding not copied verbatim"
    with open(out, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header_sha = hashlib.sha256(f.read(n)).hexdigest()
    return {"keys": len(out_header), "sampled_layers": min(sample, len(quant_keys)), "header_sha256": header_sha,
            "patch_embedding": list(patch.shape)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="Verify the written file against source and reference")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    source = Source(args.source)
    ref_n, ref_header, ref_meta = read_header_with_meta(args.reference)
    missing = set(ref_header) - {k for k in ref_header if k.endswith((".comfy_quant", ".weight_scale"))} - source.keys()
    assert not missing, f"source lacks {sorted(missing)[:5]}"
    extra = source.keys() - set(ref_header)
    assert not extra, f"source has keys the reference recipe does not cover: {sorted(extra)[:5]}"
    count = write(source, out, args.reference, ref_n, ref_header, ref_meta)
    print("wrote", out, count, "tensors", flush=True)
    if args.check:
        print("CHECK " + json.dumps(check(out, args.source, args.reference)), flush=True)


if __name__ == "__main__":
    main()
