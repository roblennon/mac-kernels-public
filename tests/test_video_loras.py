"""CPU integration against real Comfy adapter conversion and ModelPatcher."""
from pathlib import Path
import subprocess
import sys


def test_real_cpu_stack_changes_weight_preserves_base_and_rejects_partial(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = r'''
import sys
from pathlib import Path
root, target = map(Path, sys.argv[1:])
sys.path[:0] = [str(root/'bench'), str(root/'vendor/ComfyUI')]
sys.argv = ['cpu-proof', '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch
import folder_paths
import comfy.model_patcher
import comfy.lora
from safetensors.torch import save_file
from types import SimpleNamespace
from video_loras import BenchVideoLoRAStack, validate_patch_shape
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.linear = torch.nn.Linear(2, 2, bias=True)
        self.diffusion_model.other = torch.nn.Linear(2, 2, bias=False)
        self.diffusion_model.rect = torch.nn.Linear(2, 3, bias=False)
        self.model_config = SimpleNamespace(unet_config={})
model = Tiny()
with torch.no_grad():
    model.diffusion_model.linear.weight.zero_()
    model.diffusion_model.linear.bias.zero_()
base = comfy.model_patcher.ModelPatcher(model, torch.device('cpu'), torch.device('cpu'))
folder_paths.add_model_folder_path('loras', str(target))
state = {'diffusion_model.linear.lora_up.weight': torch.ones(2,1),
         'diffusion_model.linear.lora_down.weight': torch.ones(1,2)}
save_file(state, str(target/'valid.safetensors'))
save_file({**state, 'diffusion_model.linear.diff_b': torch.ones(2)}, str(target/'bias.safetensors'))
save_file({**state, 'missing.adaln.lora_up.weight': torch.ones(2,1), 'missing.adaln.lora_down.weight': torch.ones(1,2)}, str(target/'partial.safetensors'))
save_file({'missing.lora_up.weight': torch.ones(2,1), 'missing.lora_down.weight': torch.ones(1,2)}, str(target/'wrong.safetensors'))
save_file({'diffusion_model.linear.lora_up.weight': torch.ones(3,1), 'diffusion_model.linear.lora_down.weight': torch.ones(1,2)}, str(target/'wrong-shape.safetensors'))
save_file({**state, 'diffusion_model.linear.diff': torch.ones(2,2),
           'diffusion_model.other.lora_up.weight': torch.ones(2,1),
           'diffusion_model.other.lora_down.weight': torch.ones(1,2)}, str(target/'overwritten.safetensors'))
for name, bad_state in {
    'diff-shape': {'diffusion_model.linear.diff': torch.ones(3, 2)},
    'bias-shape': {'diffusion_model.linear.diff_b': torch.ones(3)},
    'set-broadcast': {'diffusion_model.linear.set_weight': torch.ones(2)},
    'transposed': {'diffusion_model.rect.lora_up.weight': torch.ones(2, 1),
                   'diffusion_model.rect.lora_down.weight': torch.ones(1, 3)},
    'rank': {'diffusion_model.linear.lora_up.weight': torch.ones(2, 1),
             'diffusion_model.linear.lora_down.weight': torch.ones(2, 2)},
    'dora': {**state, 'diffusion_model.linear.dora_scale': torch.ones(2)},
    'absent-bias': {'diffusion_model.other.diff_b': torch.ones(2)},
    'loha': {'diffusion_model.linear.hada_w1_a': torch.ones(2, 1),
             'diffusion_model.linear.hada_w1_b': torch.ones(1, 2),
             'diffusion_model.linear.hada_w2_a': torch.ones(2, 1),
             'diffusion_model.linear.hada_w2_b': torch.ones(1, 2)},
    'reshape': {**state, 'diffusion_model.linear.reshape_weight': torch.tensor([3, 2])},
}.items():
    save_file(bad_state, str(target/(name+'.safetensors')))
node = BenchVideoLoRAStack()
assert node.apply(base)[0] is base
assert node.apply(base, lora_1='not-present.safetensors', strength_1=0)[0] is base
stack = node.apply(base, lora_1='valid.safetensors', strength_1=.5, lora_3='valid.safetensors', strength_3=.25)[0]
assert not base.patches and len(stack.patches['diffusion_model.linear.weight']) == 2
actual = comfy.lora.calculate_weight(stack.patches['diffusion_model.linear.weight'], model.diffusion_model.linear.weight.detach().clone(), 'diffusion_model.linear.weight')
assert torch.equal(actual, torch.full((2,2), .75))
assert torch.equal(model.diffusion_model.linear.weight, torch.zeros(2,2))
biased = node.apply(base, lora_1='bias.safetensors', strength_1=.5)[0]
actual_bias = comfy.lora.calculate_weight(biased.patches['diffusion_model.linear.bias'], model.diffusion_model.linear.bias.detach().clone(), 'diffusion_model.linear.bias')
assert torch.equal(actual_bias, torch.full((2,), .5))
assert torch.equal(model.diffusion_model.linear.bias, torch.zeros(2))
assert node.apply(base)[0] is base and not base.patches
validate_patch_shape(model, ('diffusion_model.linear.weight', None, lambda x: x), ('diff', (torch.ones(2, 2),)))
validate_patch_shape(model, ('diffusion_model.linear.weight', (0, 0, 1)), ('diff', (torch.ones(1, 2),)))
for filename in ('partial.safetensors', 'wrong.safetensors', 'wrong-shape.safetensors', 'overwritten.safetensors', 'not-present.safetensors',
                 'diff-shape.safetensors', 'bias-shape.safetensors', 'set-broadcast.safetensors', 'transposed.safetensors', 'reshape.safetensors', 'rank.safetensors', 'dora.safetensors', 'absent-bias.safetensors', 'loha.safetensors'):
    try:
        node.apply(base, lora_1=filename)
    except (ValueError, FileNotFoundError) as error:
        expected = {'diff-shape': 'geometry', 'bias-shape': 'geometry',
                    'set-broadcast': 'geometry', 'transposed': 'geometry',
                    'rank': 'geometry', 'reshape': 'reshape', 'dora': 'plain A/B',
                    'overwritten': 'unmatched', 'partial': 'unmatched',
                    'absent-bias': 'target', 'loha': 'plain A/B',
                    'wrong-shape': 'geometry', 'wrong': 'no compatible'}
        if filename != 'not-present.safetensors':
            assert expected[Path(filename).stem] in str(error), (filename, str(error))
        else:
            assert isinstance(error, FileNotFoundError)
    else:
        raise AssertionError(filename + ' must fail')
for strength in (float('nan'), float('inf'), 3.0, True):
    try:
        node.apply(base, lora_1='valid.safetensors', strength_1=strength)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid strength must fail')
assert not base.patches
print('CPU_ADAPTER_PROOF_OK')
'''
    result = subprocess.run([sys.executable, '-c', script, str(root), str(tmp_path)], cwd=root, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CPU_ADAPTER_PROOF_OK' in result.stdout


def test_wan_diffusers_adapter_applies_all_targets_and_preserves_other_families(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = r"""
import sys
from pathlib import Path
root, target = map(Path, sys.argv[1:])
sys.path[:0] = [str(root/'bench'), str(root/'vendor/ComfyUI')]
sys.argv = ['cpu-proof', '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.model_base
import comfy.model_patcher
import folder_paths
from safetensors.torch import save_file
from types import SimpleNamespace
from video_loras import BenchVideoLoRAStack, model_lora_keys
class SmallWan(comfy.model_base.WAN21):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.model_config = SimpleNamespace(unet_config={})
        self.diffusion_model = torch.nn.Module()
        block = torch.nn.Module()
        for name in ('self_attn', 'cross_attn'):
            attention = torch.nn.Module()
            for projection in ('q', 'k', 'v', 'o'):
                setattr(attention, projection, torch.nn.Linear(2, 2, bias=False))
            setattr(block, name, attention)
        block.ffn = torch.nn.Sequential(torch.nn.Linear(2,2,bias=False), torch.nn.GELU(), torch.nn.Linear(2,2,bias=False))
        self.diffusion_model.blocks = torch.nn.ModuleList([block])
        for parameter in self.parameters():
            torch.nn.init.zeros_(parameter)
model = SmallWan()
base = comfy.model_patcher.ModelPatcher(model, torch.device('cpu'), torch.device('cpu'))
folder_paths.add_model_folder_path('loras', str(target))
# An independently specified ten-target Diffusers file must update each native weight.
parts = [f'{attn}.{proj}' for attn in ('attn1','attn2') for proj in ('to_q','to_k','to_v','to_out.0')]
parts += ['ffn.net.0.proj','ffn.net.2']
state = {f'transformer.blocks.0.{part}.lora_{ab}.weight': torch.ones((1,2) if ab=='A' else (2,1))
         for part in parts for ab in ('A','B')}
state.update({f'transformer.blocks.0.{part}.alpha': torch.tensor(.5) for part in parts})
save_file(state, str(target/'wan.safetensors'))
stack = BenchVideoLoRAStack().apply(base, lora_1='wan.safetensors', strength_1=.5)[0]
assert len(stack.patches)==10 and not base.patches
for key, parameter in model.named_parameters():
    updated = comfy.lora.calculate_weight(stack.patches[key], parameter.detach().clone(), key)
    assert torch.equal(updated, torch.full_like(updated,.25))
    assert torch.count_nonzero(parameter)==0
# Exact family gate: an unrelated model with identical parameter names gets no aliases.
other = torch.nn.Module()
other.diffusion_model = model.diffusion_model
other.model_config = model.model_config
assert not any(k.startswith('transformer.blocks.') for k in model_lora_keys(other))
# A future/new unknown target cannot disappear into a partial successful application.
save_file({**state, 'transformer.blocks.0.unknown.lora_A.weight': torch.ones(1,2),
           'transformer.blocks.0.unknown.lora_B.weight': torch.ones(2,1)}, str(target/'partial.safetensors'))
try:
    BenchVideoLoRAStack().apply(base, lora_1='partial.safetensors', strength_1=.5)
except ValueError as error:
    assert 'unmatched' in str(error)
else:
    raise AssertionError('Partial WAN Diffusers adapter was admitted')
"""
    result = subprocess.run([sys.executable, '-c', script, str(root), str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_wan_diffusers_image_and_top_level_names_map_to_exact_targets(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = r"""
import sys
from pathlib import Path
root, target = map(Path, sys.argv[1:])
sys.path[:0] = [str(root/'bench'), str(root/'vendor/ComfyUI')]
sys.argv = ['cpu-proof', '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.lora
import comfy.model_base
import comfy.model_patcher
import folder_paths
from safetensors.torch import save_file
from types import SimpleNamespace
from video_loras import BenchVideoLoRAStack
L = lambda i, o: torch.nn.Linear(i, o, bias=False)
class SmallWanI2V(comfy.model_base.WAN21):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.model_config = SimpleNamespace(unet_config={})
        d = self.diffusion_model = torch.nn.Module()
        block = torch.nn.Module()
        block.cross_attn = torch.nn.Module()
        block.cross_attn.k_img, block.cross_attn.v_img = L(2, 2), L(2, 2)
        d.blocks = torch.nn.ModuleList([block])
        d.text_embedding = torch.nn.Sequential(L(3, 2), torch.nn.GELU(), L(2, 2))
        d.time_embedding = torch.nn.Sequential(L(4, 2), torch.nn.SiLU(), L(2, 2))
        d.time_projection = torch.nn.Sequential(torch.nn.SiLU(), L(2, 6))
        d.img_emb = torch.nn.Module()
        d.img_emb.proj = torch.nn.Sequential(torch.nn.Identity(), L(5, 5), torch.nn.GELU(), L(5, 2))
        d.head = torch.nn.Module()
        d.head.head = L(2, 7)
        for parameter in self.parameters():
            torch.nn.init.zeros_(parameter)
model = SmallWanI2V()
base = comfy.model_patcher.ModelPatcher(model, torch.device('cpu'), torch.device('cpu'))
folder_paths.add_model_folder_path('loras', str(target))
names = {'blocks.0.attn2.add_k_proj': 'blocks.0.cross_attn.k_img', 'blocks.0.attn2.add_v_proj': 'blocks.0.cross_attn.v_img',
         'condition_embedder.text_embedder.linear_1': 'text_embedding.0', 'condition_embedder.text_embedder.linear_2': 'text_embedding.2',
         'condition_embedder.time_embedder.linear_1': 'time_embedding.0', 'condition_embedder.time_embedder.linear_2': 'time_embedding.2',
         'condition_embedder.time_proj': 'time_projection.1',
         'condition_embedder.image_embedder.ff.net.0.proj': 'img_emb.proj.1', 'condition_embedder.image_embedder.ff.net.2': 'img_emb.proj.3',
         'proj_out': 'head.head'}
weights = dict(model.diffusion_model.named_parameters())
state, expected = {}, {}
for i, (name, native) in enumerate(names.items()):
    out_dim, in_dim = weights[native + '.weight'].shape
    prefix = 'transformer.' if i % 2 else ''
    state[f'{prefix}{name}.lora_A.weight'] = torch.ones(1, in_dim)
    state[f'{prefix}{name}.lora_B.weight'] = torch.full((out_dim, 1), float(i + 1))
    expected[f'diffusion_model.{native}.weight'] = torch.full((out_dim, in_dim), 0.5 * (i + 1))
save_file(state, str(target/'wan_i2v.safetensors'))
stack = BenchVideoLoRAStack().apply(base, lora_1='wan_i2v.safetensors', strength_1=.5)[0]
assert set(stack.patches) == set(expected), sorted(set(stack.patches) ^ set(expected))
for key, want in expected.items():
    weight = dict(model.named_parameters())[key]
    assert torch.equal(comfy.lora.calculate_weight(stack.patches[key], weight.detach().clone(), key), want), key
# Diffusers bias updates are not applied by this runtime, so the whole file is refused.
save_file({**state, 'proj_out.lora_B.bias': torch.ones(7)}, str(target/'bias.safetensors'))
try:
    BenchVideoLoRAStack().apply(base, lora_1='bias.safetensors', strength_1=.5)
except ValueError as error:
    assert 'unmatched' in str(error)
else:
    raise AssertionError('A Diffusers bias update was silently dropped')
"""
    result = subprocess.run([sys.executable, '-c', script, str(root), str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_comfy_cache_notices_replaced_adapter_and_ignores_disabled_slots(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = r'''
import asyncio, copy, os, sys
from pathlib import Path
root, target = map(Path, sys.argv[1:])
sys.path[:0] = [str(root/'bench'), str(root/'vendor/ComfyUI')]
sys.argv = ['cpu-cache-proof', '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import folder_paths, nodes
from execution import IsChangedCache
from comfy_execution.graph import DynamicPrompt
from comfy_execution.caching import CacheKeySetInputSignature
from video_loras import BenchVideoLoRAStack
nodes.NODE_CLASS_MAPPINGS['BenchVideoLoRAStack'] = BenchVideoLoRAStack
folder_paths.add_model_folder_path('loras', str(target))
path = target/'adapter.safetensors'
path.write_bytes(b'old!')
slots = {'lora_1':'adapter.safetensors','strength_1':1.0,
         'lora_2':'missing.safetensors','strength_2':0.0,'lora_3':'none'}
prompt = {'1':{'class_type':'BenchVideoLoRAStack','inputs':slots}}
async def signature():
    dynamic = DynamicPrompt(copy.deepcopy(prompt))
    changes = IsChangedCache('cpu-proof', dynamic, None)
    keys = CacheKeySetInputSignature(dynamic, ['1'], changes)
    return await keys.get_node_signature(dynamic, '1')
before = asyncio.run(signature())
assert asyncio.run(signature()) == before
old = path.stat()
replacement = target/'replacement'
replacement.write_bytes(b'new!')
os.utime(replacement, ns=(old.st_atime_ns,old.st_mtime_ns))
replacement.replace(path)
assert path.stat().st_size == old.st_size and path.stat().st_mtime_ns == old.st_mtime_ns
assert asyncio.run(signature()) != before, 'Same-name replacement must invalidate real Comfy cache'
# No payload reads or model construction are needed to invalidate a cache.
assert BenchVideoLoRAStack.IS_CHANGED(lora_1='missing.safetensors',strength_1=0) == ()
assert BenchVideoLoRAStack.IS_CHANGED(lora_1='none') == ()
path.unlink()
try:
    BenchVideoLoRAStack.IS_CHANGED(**slots)
except FileNotFoundError:
    pass
else:
    raise AssertionError('Deleted active adapter must not retain its cached model')
'''
    result = subprocess.run([sys.executable, '-c', script, str(root), str(tmp_path)],
                            cwd=root, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
