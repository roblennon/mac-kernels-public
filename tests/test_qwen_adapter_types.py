"""Exercise real Comfy adapter loading in an isolated CPU process."""
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_serialized_equivalent_types_and_route_audit(tmp_path):
    script = r'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import comfy.lora
import comfy.model_patcher
from safetensors.torch import save_file, load_file
from probe_qwen_adapter_types import convert_state, load_adapter, delta, RouteAudit
torch.set_num_threads(1)
prefix = 'transformer.test'
g = torch.Generator().manual_seed(713)
state = {prefix + '.lora_A.weight': torch.randn(2, 32, generator=g),
         prefix + '.lora_B.weight': torch.randn(32, 2, generator=g)}
original = load_adapter(state, prefix, 'lora')
expected = delta(original, (32, 32))
for kind in ('loha', 'lokr'):
    converted, prefixes = convert_state(state, kind)
    path = Path(OUTPUT) / (kind + '.safetensors')
    save_file(converted, path)
    loaded = load_adapter(load_file(path), prefix, kind)
    assert torch.equal(delta(loaded, (32, 32)), expected)
    side = comfy.lora.lora_side_patches([(1., loaded, 1., None, None)], 32, 32,
                                       torch.device('cpu'), torch.float32)
    assert side is not None
    x = torch.randn(3, 32, generator=g)
    output = torch.zeros(3, 32)
    for term in side:
        output = term.apply(output, x)
    torch.testing.assert_close(output, torch.nn.functional.linear(x, expected))
    # Registering an adapter never counts as reaching its module.
    model = torch.nn.Module()
    model.layer = torch.nn.Linear(32, 32, bias=False)
    patcher = comfy.model_patcher.ModelPatcher(model, torch.device('cpu'), torch.device('cpu'))
    patcher.add_patches({'layer.weight': loaded})
    audit = RouteAudit(type(loaded).__name__)
    audit.attach(patcher)
    assert audit.result()['reached_layers'] == audit.result()['side_reached_layers'] == 0
    model.layer(x)
    assert audit.result()['reached_layers'] == 1 and audit.result()['side_reached_layers'] == 0
    audit.side(model.layer, fused=False)
    assert audit.result()['side_reached_layers'] == 1 and audit.result()['fused_reached_layers'] == 0
    for handle in audit.handles:
        handle.remove()
    # Actual cost admission rejects a high-rank narrow layer for both types.
    narrow = {prefix + '.lora_A.weight': torch.randn(32, 32, generator=g),
              prefix + '.lora_B.weight': torch.randn(32, 32, generator=g)}
    costly, _ = convert_state(narrow, kind)
    costly = load_adapter(costly, prefix, kind)
    assert comfy.lora.lora_side_patches([(1., costly, 1., None, None)], 32, 32,
                                       torch.device('cpu'), torch.float32) is None
'''
    result = subprocess.run([sys.executable, '-c', 'OUTPUT=' + repr(str(tmp_path)) + '\n' + script],
                            cwd=ROOT, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_conversion_refuses_unrepresented_source_data():
    script = r'''
import torch
from bench.probe_qwen_adapter_types import convert_state
base = {'layer.lora_A.weight': torch.ones(2, 4), 'layer.lora_B.weight': torch.ones(4, 2)}
for changed in ({**base, 'layer.alpha': torch.tensor(8.)},
                {'layer.lora_A.weight': base['layer.lora_A.weight']},
                {**base, 'layer.lora_B.weight': torch.full((4, 2), float('nan'))}):
    for kind in ('loha', 'lokr'):
        try:
            convert_state(changed, kind)
        except ValueError:
            pass
        else:
            raise AssertionError('Source data was silently dropped or accepted')
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_qwen_fused_mlp_routes_count_successful_execution_only():
    script = r'''
import json, sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path.cwd() / 'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import comfy.ops as ops
import comfy.model_patcher
from comfy.ldm.qwen_image21.model import SwiGLUFeedForward
from probe_qwen_adapter_types import RouteAudit, load_adapter
torch.set_num_threads(1)
operations = ops.mixed_precision_ops(compute_dtype=torch.float32)
mlp = SwiGLUFeedForward(32, 32, fused=True, dtype=torch.float32, device='cpu', operations=operations)
state = {}
for name, rows in [('gate_up', 64), ('out', 32)]:
    state[name + '.weight'] = torch.ones(rows, 32, dtype=torch.int8)
    state[name + '.weight_scale'] = torch.full((rows, 1), .01)
    state[name + '.comfy_quant'] = torch.tensor(list(json.dumps({'format': 'int8_tensorwise'}).encode()), dtype=torch.uint8)
mlp.load_state_dict(state, strict=True)
model = torch.nn.Module(); model.mlp = mlp
patcher = comfy.model_patcher.ModelPatcher(model, torch.device('cpu'), torch.device('cpu'))
raw = {'layer.lora_A.weight': torch.full((2, 32), .01), 'layer.lora_B.weight': torch.full((32, 2), .02)}
adapter = load_adapter(raw, 'layer', 'lora')
patcher.add_patches({'mlp.out.weight': adapter})
side = comfy.lora.lora_side_patches(patcher.patches['mlp.out.weight'], 32, 32, torch.device('cpu'), torch.float32)
mlp.out.lora_side_patches = side
x = torch.randn(2, 32, generator=torch.Generator().manual_seed(817))
originals = (ops.linear_input_act, ops._side_linear, ops._apply_lora_side_patches)

def fused_kernel(weight, x, bias, row_scale, lora, extra, *, out_dtype=None):
    # External kernel boundary only; real module/adapter/route code executes.
    assert 'lora_down' in lora and 'lora_up' in lora
    assert out_dtype == x.dtype
    x = ops._eager_input_act(x, extra.get('input_act'))
    return torch.nn.functional.linear(x, weight.dequantize()) + torch.nn.functional.linear(
        torch.nn.functional.linear(x, lora['lora_down']), lora['lora_up']) * lora['lora_scale']

for mode in ('separate', 'fused', 'declined', 'failed', 'merged'):
    audit = RouteAudit('LoRAAdapter'); audit.attach(patcher)
    mlp.out.lora_side_patches = side if mode != 'merged' else ()
    try:
        with patch.object(ops, '_wants_fused_lora', return_value=mode in ('fused', 'declined', 'failed')):
            kernel = fused_kernel if mode == 'fused' else (lambda *args, **kwargs: None)
            if mode == 'failed':
                def kernel(*args, **kwargs): raise RuntimeError('kernel failed')
            with patch.object(ops, '_side_kernel_call', side_effect=kernel):
                try:
                    with audit.watch_operations():
                        result = mlp(x)
                        if mode == 'fused':
                            # The post-kernel helper skips already fused terms and must not count twice.
                            ops._apply_lora_side_patches(mlp.out, result, x, fused=True)
                except RuntimeError as error:
                    assert mode == 'failed' and str(error) == 'kernel failed'
                else:
                    assert mode != 'failed' and torch.isfinite(result).all()
        report = audit.result(); row = report['layers'][0]
        assert row['forward_calls'] == 0  # Actual Qwen fused-MLP route bypasses Module.__call__.
        assert report['reached_layers'] == (0 if mode == 'failed' else 1)
        assert row['input_act_calls'] == (0 if mode == 'failed' else 1)
        assert row['side_calls'] == (0 if mode in ('failed', 'merged') else 1)
        assert row['fused_calls'] == (1 if mode == 'fused' else 0)
        assert (ops.linear_input_act, ops._side_linear, ops._apply_lora_side_patches) == originals
        if mode in ('separate', 'fused', 'declined'):
            audit.require_complete(expected_layers=1, expected_side_layers=1)
        else:
            try: audit.require_complete(expected_layers=1, expected_side_layers=1)
            except AssertionError: pass
            else: raise AssertionError('Missing quantized application was accepted')
    finally:
        for handle in audit.handles: handle.remove()
# Matching aggregate counts must not hide a missing quantized side application.
model.plain = torch.nn.Linear(32, 32, bias=False)
patcher.add_patches({'plain.weight': adapter})
audit = RouteAudit('LoRAAdapter'); audit.attach(patcher)
mlp.out.lora_side_patches = ()
with audit.watch_operations():
    mlp(x); model.plain(x)
audit.side(model.plain, fused=False)  # Deliberately inject attribution to the wrong real module.
assert audit.result()['reached_layers'] == 2 and audit.result()['side_reached_layers'] == 1
try: audit.require_complete(expected_layers=2, expected_side_layers=1)
except AssertionError: pass
else: raise AssertionError('Equal totals masked the wrong side-layer membership')
for handle in audit.handles: handle.remove()
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
