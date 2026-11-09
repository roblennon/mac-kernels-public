"""Local CPU check that the quantizers in modal/weight_error_app.py are the repo's own.

Run from the repo root with the repo venv (CPU only, small tensors, seconds):

  .venv/bin/python modal/weight_error_validate.py [--ggml-lib PATH_TO_libggml-base] \
      [--container modal/results/2026-11-09-weight-error-selftest-container.json]

1. Port against vendor/comfy-kitchen eager quantizers: every stored tensor and the
   dequantized weight must be bit-identical on three seeded tensors (fixed codebook, fitted
   codebook, chunked path).
2. GGUF: ggml Q8_0 bytes against the gguf-py numpy quantizer, and the gguf-py decode against
   vendor/ComfyUI-GGUF-fork/dequant.py (the decode the repo's GGUF path is tested against).
3. With --container: this machine's vectors against the ones the Modal container produced.

Writes modal/results/2026-11-09-weight-error-validation.json.
"""
import argparse
import base64
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if "modal" not in sys.modules:  # the repo venv has no modal; the app module only needs decorators
    try:
        import modal  # noqa: F401
    except ImportError:
        class _Any:
            def __call__(self, *a, **k):
                return a[0] if len(a) == 1 and callable(a[0]) and not k else self

            def __getattr__(self, name):
                return self

        stub = types.ModuleType("modal")
        stub.App = stub.Volume = stub.Image = _Any()
        sys.modules["modal"] = stub

sys.path.insert(0, str(ROOT / "modal"))
sys.path.insert(0, str(ROOT / "vendor" / "comfy-kitchen"))
import weight_error_app as A  # noqa: E402
from comfy_kitchen.backends.eager import quantization as VQ  # noqa: E402
from comfy_kitchen.backends.eager import w4a8_int8 as VW  # noqa: E402


def same(a, b):
    if a.dtype == torch.float8_e4m3fn:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return bool(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b))


def vendor_all(w):
    out = {}
    q, s = VQ.quantize_int8_rowwise(w.clone())
    out["int8_rowwise"] = {"q": q, "scale": s, "deq": VQ.dequantize_int8_simple(q, s)}
    q, s = VQ.quantize_int8_convrot_weight(w, 256)
    out["int8_convrot"] = {"q": q, "scale": s, "deq": VQ.dequantize_int8_convrot_weight(q, s, 256)}
    for name, bits in (("w4a8", 4), ("w6a8", 6)):
        packed, s_rel, s_ch, corr, cb = VW.quantize_w4a8_int8_weight(w, bits=bits)
        assert corr is None
        out[name] = {"q": packed, "s_rel": s_rel, "s_channel": s_ch,
                     "deq": VW.dequantize_w4a8_int8_weight(packed, s_rel, s_ch, cb, None, output_dtype=torch.float32)}
        if cb is not None:
            out[name]["codebook"] = cb
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ggml-lib", default=None)
    ap.add_argument("--container", default=None)
    args = ap.parse_args()
    torch.set_num_threads(2)
    res = {"torch": str(torch.__version__), "platform": sys.platform, "port_vs_vendor": {}}

    for case, w in A.selftest_inputs().items():
        mine, ref = A.quantize_all_torch(w), vendor_all(w)
        entry = {}
        for fmt in ref:
            assert set(mine[fmt]) == set(ref[fmt]), (fmt, set(mine[fmt]), set(ref[fmt]))
            entry[fmt] = {k: same(mine[fmt][k], ref[fmt][k]) for k in ref[fmt]}
            entry[fmt]["rel_fro"] = float((ref[fmt]["deq"] - w).double().norm() / w.double().norm())
        if "codebook" in ref["w4a8"]:
            entry["w4a8"]["codebook_kind"] = "fixed" if torch.equal(ref["w4a8"]["codebook"], torch.tensor(VW._FIXED_LUT)) else "fitted"
        res["port_vs_vendor"][case] = entry
    res["port_bit_exact"] = all(v for c in res["port_vs_vendor"].values() for f in c.values()
                                for k, v in f.items() if isinstance(v, bool))

    # The vendor rotation runs in the dtype of the weight it is given. The measurement feeds
    # fp32. This records what a bf16 input does on the seeded tensors (which are exactly
    # representable in bf16), through the vendor quantizer, against the same fp32 reference.
    res["bf16_input_vendor"] = {}
    for case, w in A.selftest_inputs().items():
        entry = {}
        for name, bits in (("w4a8", 4), ("w6a8", 6)):
            try:
                packed, s_rel, s_ch, corr, cb = VW.quantize_w4a8_int8_weight(w.to(torch.bfloat16), bits=bits)
                deq = VW.dequantize_w4a8_int8_weight(packed, s_rel, s_ch, cb, None, output_dtype=torch.float32)
                entry[name] = {"rel_fro_bf16_input": float((deq - w).double().norm() / w.double().norm()),
                               "rel_fro_fp32_input": res["port_vs_vendor"][case][name]["rel_fro"]}
            except Exception as e:  # recorded as it is
                entry[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
        res["bf16_input_vendor"][case] = entry

    local = None
    try:
        ggml = A.Ggml(args.ggml_lib)
    except (AssertionError, OSError) as e:
        res["gguf"] = {"skipped": f"no libggml-base loaded ({type(e).__name__})"}
    else:
        import gguf
        from gguf import GGMLQuantizationType as T
        from gguf import quants
        spec = importlib.util.spec_from_file_location("ck_gguf_dequant", ROOT / "vendor" / "ComfyUI-GGUF-fork" / "dequant.py")
        deq_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deq_mod)
        w = A.selftest_inputs()["gauss_96x1024"]
        wn = np.ascontiguousarray(w.numpy())
        g = {"ggml_lib": ggml.lib_name, "ggml_version": ggml.version, "gguf_py": gguf.__version__ if hasattr(gguf, "__version__") else None}
        for fmt, qt in (("gguf_q8_0", T.Q8_0), ("gguf_q4_k", T.Q4_K)):
            packed = ggml.quantize(wn, fmt)
            ref = torch.from_numpy(A.gguf_dequantize(packed, fmt))
            repo = deq_mod.dequantize(torch.from_numpy(packed.copy()), qt, w.shape, dtype=torch.float32)
            g[fmt] = {
                "gguf_py_vs_repo_dequant_max_abs": float((ref - repo).abs().max()),
                "gguf_py_vs_repo_dequant_bit_exact": bool(torch.equal(ref, repo)),
                "rel_fro": float((ref - w).double().norm() / w.double().norm()),
            }
        a, b = ggml.quantize(wn, "gguf_q8_0"), quants.quantize(wn, T.Q8_0)
        g["q8_0_ggml_vs_gguf_py_bytes_differing"] = int((a != b.reshape(a.shape)).sum())
        res["gguf"] = g
        local = A.selftest_vectors(with_gguf=True, ggml=ggml)

    if args.container:
        if local is None:
            local = A.selftest_vectors(with_gguf=False)
        remote = json.loads(Path(args.container).read_text())
        cmp = {"local_torch": local["torch"], "container_torch": remote["torch"],
               "local_ggml": local.get("ggml_version"), "container_ggml": remote.get("ggml_version"), "cases": {}}
        for case, lc in local["cases"].items():
            rc = remote["cases"][case]
            entry = {"input_identical": lc["input_sha256"] == rc["input_sha256"], "formats": {}}
            for fmt, lf in lc["formats"].items():
                rf = rc["formats"].get(fmt)
                if rf is None:
                    continue
                e = {k: lf[k] == rf[k] for k in lf
                     if k in rf and isinstance(lf[k], str) and not k.endswith("_b64")}
                for m in ("rel_fro", "max_abs"):
                    if m in rf:
                        e[m] = {"local": lf[m], "container": rf[m]}
                rows = {}
                for k, lr in lf.get("row_sha", {}).items():
                    rr = rf.get("row_sha", {}).get(k)
                    if rr is not None and len(rr) == len(lr):
                        rows[k] = {"rows_differing": sum(a != b for a, b in zip(lr, rr)), "rows_total": len(lr)}
                if rows:
                    e["rows"] = rows
                raw = {}
                for k in lf:
                    if k.endswith("_b64") and k in rf:
                        x = np.frombuffer(base64.b64decode(lf[k]), dtype=np.uint8)
                        y = np.frombuffer(base64.b64decode(rf[k]), dtype=np.uint8)
                        raw[k[:-4]] = {"bytes_differing": int((x != y).sum()) if x.size == y.size else None,
                                       "bytes_total": int(x.size)}
                if raw:
                    e["stored_bytes"] = raw
                entry["formats"][fmt] = e
            cmp["cases"][case] = entry
        res["local_vs_container"] = cmp

    out = ROOT / "modal" / "results" / "2026-11-09-weight-error-validation.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
