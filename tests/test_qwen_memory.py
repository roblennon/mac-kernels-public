"""Bounded Qwen admission against real CPU patchers: fused-weight row slices, refusals."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_fused_gate_up_row_slices_are_admitted_and_side_expressible():
    script = r'''
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'bench'))
sys.path.insert(0,str(Path.cwd()))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import comfy.lora as lora
import comfy.ops as ops
import comfy.model_patcher as mp
from comfy.quant_ops import QuantizedTensor
from comfy.weight_adapter import LoRAAdapter
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from bench.qwen_memory import validate_patcher

def model():
    m=torch.nn.Module();m.diffusion_model=torch.nn.Module()
    block=torch.nn.Module();block.img_mlp=torch.nn.Module()
    m.diffusion_model.transformer_blocks=torch.nn.ModuleList([block])
    layer=ops.mixed_precision_ops(compute_dtype=torch.float16).Linear(256,128,bias=False,device='cpu')
    params=TensorWiseINT8Layout.Params(scale=torch.ones(128,1),orig_dtype=torch.bfloat16,orig_shape=(128,256),convrot=True,convrot_groupsize=256)
    layer.weight=torch.nn.Parameter(QuantizedTensor(torch.ones(128,256,dtype=torch.int8),'TensorWiseINT8Layout',params),requires_grad=False)
    layer.quant_format='int8_tensorwise';block.img_mlp.gate_up=layer
    return m,layer

key='diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight'
def patcher(patches):
    m,layer=model();p=mp.ModelPatcher(m,torch.device('mps'),torch.device('cpu'))
    for k,v in patches:p.add_patches({k:v},strength_patch=1.0)
    return p,m,layer
lo=lambda rows,rank=4,cols=256:LoRAAdapter(set(),(torch.ones(rows,rank),torch.ones(rank,cols),None,None,None,None))

# Two halves of the fused weight, as Comfy maps a LoRA's gate_layer and proj: admitted.
p,m,layer=patcher([((key,(0,0,64)),lo(64)),((key,(0,64,64)),lo(64,rank=8))])
assert [e[3] for e in p.patches[key]]==[(0,0,64),(0,64,64)]
validate_patcher(p,m)
side=lora.lora_side_patches(p.patches[key],128,256,torch.device('cpu'),torch.bfloat16,weight=layer.weight)
assert side, 'admitted row slices must be expressible on the side path'
# Refused: a column slice, a slice past the weight, and up rows that do not match the slice.
for bad in ([((key,(1,0,128)),lo(128,cols=128))],[((key,(0,96,64)),lo(64))],[((key,(0,0,64)),lo(128))],
            [((key,(0,0.5,64)),lo(64))],[((key,(0,float('nan'),64)),lo(64))]):
    p,m,_=patcher(bad)
    try:validate_patcher(p,m)
    except ValueError:pass
    else:raise AssertionError(f'admitted {bad[0][0]}')
# A whole-weight LoRA still passes; a whole-weight one with wrong rows still fails.
p,m,_=patcher([(key,lo(128))]);validate_patcher(p,m)
p,m,_=patcher([(key,lo(64))])
try:validate_patcher(p,m)
except ValueError:pass
else:raise AssertionError('mismatched whole-weight LoRA admitted')
from bench.qwen_memory import adapter_factor_bytes
p,_,_=patcher([((key,(0,0,64)),LoRAAdapter(set(),(torch.ones(64,4,dtype=torch.int8),torch.ones(4,256,dtype=torch.int8),None,None,None,None)))])
assert adapter_factor_bytes(p)==(64*4+4*256)*2  # 8-bit factors counted as widened
print('ok')
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and result.stdout.strip().endswith('ok'), result.stdout + result.stderr


def test_residency_reserve_comes_off_each_allowance():
    from types import SimpleNamespace
    from bench import krea_readonly as readonly
    target = object()
    loaded_bytes = [1000]
    patcher = SimpleNamespace(model=target, loaded_size=lambda: loaded_bytes[0])
    requests = []

    class LoadedModel:
        @staticmethod
        def model_use_more_vram(loaded, extra_memory, force_patch_weights=False):
            requests.append(extra_memory)
            loaded_bytes[0] += extra_memory
            return extra_memory

    mm = SimpleNamespace(LoadedModel=LoadedModel)
    loaded = SimpleNamespace(model=patcher, device=SimpleNamespace(type='mps'))
    with readonly._bounded_krea_residency(mm, patcher, 10_000, lambda *a: None, reserve_bytes=3000) as audit:
        LoadedModel.model_use_more_vram(loaded, 50_000)
    assert requests == [10_000 - 1000 - 3000] and audit['reserve_bytes'] == 3000
    assert LoadedModel.model_use_more_vram.__name__ == 'model_use_more_vram'  # restored
