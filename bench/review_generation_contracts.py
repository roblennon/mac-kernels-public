"""CPU-only regression probes for the generation-path review.

Loads the real ModelPatcher and real LoRA-factor objects with ComfyUI forced to
CPU. Extracts _sdpa_bounded unchanged from source to exercise its slicing with
small CPU tensors. Exit 1 means a contract violation, 2 a probe/setup error;
no model files or GPU work.
COMFY_DIR selects another checkout for before/after comparisons.
"""
import ast
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

COMFY = Path(os.environ.get("COMFY_DIR", Path(__file__).resolve().parents[1] / "vendor" / "ComfyUI"))
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], "--cpu"]
try:
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import comfy.lora
    import comfy.model_management as mm
    from comfy.model_patcher import ModelPatcher
except Exception as error:
    print(json.dumps({"event": "probe_error", "error": repr(error)}), file=sys.stderr)
    raise SystemExit(2)


def attention_probes():
    path = COMFY / "comfy" / "ldm" / "modules" / "attention.py"
    tree = ast.parse(path.read_text())
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_sdpa_bounded")
    namespace = {"torch": torch, "comfy": SimpleNamespace(ops=SimpleNamespace(
        scaled_dot_product_attention=torch.nn.functional.scaled_dot_product_attention))}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)
    bounded = namespace["_sdpa_bounded"]
    generator = torch.Generator(device="cpu").manual_seed(7)
    cases = [
        ("same_shapes", ((2, 2, 4, 8), (2, 2, 4, 8), (2, 2, 4, 8))),
        ("kv_both_broadcast_batch", ((2, 2, 4, 8), (1, 2, 4, 8), (1, 2, 4, 8))),
        ("v_broadcast_batch", ((2, 2, 4, 8), (2, 2, 4, 8), (1, 2, 4, 8))),
        ("k_broadcast_heads", ((1, 4, 4, 8), (1, 1, 4, 8), (1, 4, 4, 8))),
    ]
    results = []
    for name, shapes in cases:
        q, k, v = [torch.randn(shape, generator=generator, device="cpu") for shape in shapes]
        reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        result = {"probe": name, "reference_shape": list(reference.shape)}
        try:
            output = bounded(q, k, v, None, 20)
            result["max_abs_error"] = float((output - reference).abs().max())
            torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)
            result["ok"] = True
        except (RuntimeError, AssertionError) as error:
            result.update(ok=False, error=str(error))
        results.append(result)
    return results


def clone_probe():
    model = torch.nn.Sequential(torch.nn.Linear(8, 4, bias=False, device="cpu"))
    model[0].lora_side_patches = [comfy.lora.LoraSidePatch(
        torch.zeros((2, 8), device="cpu"), torch.zeros((4, 2), device="cpu"))]
    patcher = ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    first_clone = patcher.clone()
    second_clone = first_clone.clone()
    expected = mm.module_size(model, lora_side=False) + mm.lora_side_size(model)
    sizes = [p.model_size() for p in (patcher, first_clone, second_clone)]
    return {"probe": "resident_lora_clone_size", "expected_bytes": expected,
            "reported_bytes": sizes, "shared_model": patcher.model is first_clone.model is second_clone.model,
            "ok": all(size == expected for size in sizes)}


def main():
    assert mm.get_torch_device().type == "cpu"
    results = attention_probes() + [clone_probe()]
    print(json.dumps({"comfy_dir": str(COMFY), "device": "cpu", "results": results}, indent=2))
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    try:
        status = main()
    except Exception as error:
        print(json.dumps({"event": "probe_error", "error": repr(error)}), file=sys.stderr)
        status = 2
    raise SystemExit(status)
