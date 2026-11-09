"""Per-tensor quant types of GGUF files, read from the header only.

Local files are read directly; Hugging Face files ("repo_id/path/in/repo.gguf") are read with
HTTP range requests, so a 20 GB file costs a few MB. Prints, per file, the architecture, how
many tensors and bytes each type holds, and which types the 2D (linear) weights use by
role. ``--tensors`` lists every tensor.

    python bench/gguf_headers.py unsloth/Qwen-Image-2.1-GGUF/qwen-image-2.1-Q4_K_M.gguf
    python bench/gguf_headers.py ~/mac-kernels-data/models/gguf/x.gguf --tensors
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import struct
import sys
import urllib.request

import gguf

GGML_SIZES = gguf.GGML_QUANT_SIZES


class Source:
    """Byte reader over a local file or a Hugging Face URL, fetching in chunks."""

    CHUNK = 4 << 20

    def __init__(self, spec: str):
        self.spec = spec
        self.local = os.path.exists(spec)
        if self.local:
            self.f = open(spec, "rb")
        else:
            repo, _, path = spec.partition("/")
            org_repo = repo + "/" + path.split("/", 1)[0]
            path = path.split("/", 1)[1]
            self.url = f"https://huggingface.co/{org_repo}/resolve/main/{path}"
        self.buf = b""
        self.pos = 0

    def _ensure(self, n: int):
        while len(self.buf) - self.pos < n:
            start = len(self.buf)
            if self.local:
                self.f.seek(start)
                data = self.f.read(self.CHUNK)
            else:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + self.CHUNK - 1}"})
                token = os.environ.get("HF_TOKEN")
                if token:
                    req.add_header("Authorization", f"Bearer {token}")
                with urllib.request.urlopen(req) as r:
                    data = r.read()
            if not data:
                raise EOFError(self.spec)
            self.buf += data

    def read(self, n: int) -> bytes:
        self._ensure(n)
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        return self.read(self.u64()).decode("utf-8", "replace")


SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_value(src: Source, vtype: int):
    if vtype in SCALARS:
        fmt = SCALARS[vtype]
        return struct.unpack(fmt, src.read(struct.calcsize(fmt)))[0]
    if vtype == 8:
        return src.string()
    if vtype == 9:
        etype, n = src.u32(), src.u64()
        return [read_value(src, etype) for _ in range(n)]
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_header(spec: str):
    src = Source(spec)
    if src.read(4) != b"GGUF":
        raise ValueError(f"{spec}: not a GGUF file")
    version = src.u32()
    n_tensors, n_kv = src.u64(), src.u64()
    kv = {}
    for _ in range(n_kv):
        key = src.string()
        kv[key] = read_value(src, src.u32())
    tensors = []
    for _ in range(n_tensors):
        name = src.string()
        dims = [src.u64() for _ in range(src.u32())]
        qtype = src.u32()
        offset = src.u64()
        tensors.append((name, dims, qtype, offset))
    return version, kv, tensors


def role(name: str) -> str:
    """A coarse role for a tensor name, shared across the DiT families."""
    n = re.sub(r"\.\d+\.", ".N.", name)
    n = re.sub(r"^(model\.diffusion_model\.)", "", n)
    return n


def summarize(spec: str, list_tensors: bool):
    version, kv, tensors = read_header(spec)
    arch = kv.get("general.architecture")
    ftype = kv.get("general.file_type")
    qv = kv.get("general.quantization_version")
    print(f"\n## {spec}\nGGUF v{version}, arch={arch!r}, file_type={ftype}, quantization_version={qv}, "
          f"{len(tensors)} tensors")
    by_type = collections.Counter()
    bytes_by_type = collections.Counter()
    roles = collections.defaultdict(collections.Counter)
    straddle = []
    orig = {k[len("comfy.gguf.orig_shape."):]: v for k, v in kv.items() if k.startswith("comfy.gguf.orig_shape.")}
    for name, dims, qt, _ in tensors:
        t = gguf.GGMLQuantizationType(qt).name
        bs, ts = GGML_SIZES[gguf.GGMLQuantizationType(qt)]
        numel = 1
        for d in dims:
            numel *= d
        by_type[t] += 1
        bytes_by_type[t] += numel // bs * ts
        if len(dims) >= 2:
            roles[role(name)][t] += 1
        if name in orig:
            shape = orig[name]
            k = shape[-1]
            straddle.append((name, list(reversed(dims)), shape, k % bs != 0))
    total = sum(bytes_by_type.values())
    print("| type | tensors | GB | share |\n|---|---|---|---|")
    for t, c in by_type.most_common():
        print(f"| {t} | {c} | {bytes_by_type[t] / 1e9:.3f} | {bytes_by_type[t] / total:.1%} |")
    print("\nmatrix roles (type: count):")
    for r, c in sorted(roles.items()):
        print(f"  {r}: " + ", ".join(f"{t} {n}" for t, n in c.most_common()))
    if orig:
        print(f"\ncomfy.gguf.orig_shape entries: {len(orig)}")
        for name, gshape, shape, s in straddle[:20]:
            print(f"  {name}: gguf {gshape} -> torch {shape}{'  (blocks straddle rows)' if s else ''}")
    interesting = {k: v for k, v in kv.items() if not isinstance(v, list) and not k.startswith("comfy.gguf.orig_shape")}
    print("\nmetadata: " + json.dumps(interesting, default=str)[:600])
    if list_tensors:
        for name, dims, qt, off in tensors:
            print(f"  {name} {list(reversed(dims))} {gguf.GGMLQuantizationType(qt).name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tensors", action="store_true")
    a = ap.parse_args()
    for f in a.files:
        try:
            summarize(f, a.tensors)
        except Exception as e:  # keep going across a survey
            print(f"\n## {f}\nERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
