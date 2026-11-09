"""Real CPU patchers exercise the bounded Comfy sampler wrapper and clone protocol."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_cloned_sampler_caps_residency_and_restores_hooks_on_failure():
    script = r'''
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import comfy.ops as ops
import comfy.model_management as mm
import comfy.model_patcher as mp
from comfy.quant_ops import QuantizedTensor
from comfy.weight_adapter import LoRAAdapter
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from comfy.patcher_extension import WrapperExecutor,WrappersMP
from types import SimpleNamespace
from bench.wan_memory import attach_memory_wrapper,validate_patcher,RESIDENT_BYTES
model=torch.nn.Module();model.diffusion_model=torch.nn.Module();model.diffusion_model.blocks=torch.nn.ModuleList([torch.nn.Module()])
model.diffusion_model.blocks[0].self_attn=torch.nn.Module()
layer=ops.mixed_precision_ops(compute_dtype=torch.float16).Linear(256,64,bias=False,device='cpu')
params=TensorWiseINT8Layout.Params(scale=torch.ones(64,1),orig_dtype=torch.float16,orig_shape=(64,256),convrot=True,convrot_groupsize=256)
layer.weight=torch.nn.Parameter(QuantizedTensor(torch.ones(64,256,dtype=torch.int8),'TensorWiseINT8Layout',params),requires_grad=False)
layer.quant_format='int8_tensorwise';model.diffusion_model.blocks[0].self_attn.q=layer
base=mp.ModelPatcher(model,torch.device('mps'),torch.device('cpu'))
key='diffusion_model.blocks.0.self_attn.q.weight'
plain=LoRAAdapter(set(),(torch.ones(64,2),torch.ones(2,256),None,None,None,None))
base.add_patches({key:plain},strength_patch=.5)
attach_memory_wrapper(base);clone=base.clone()
assert clone.get_wrappers(WrappersMP.OUTER_SAMPLE,'wan14_memory')==base.get_wrappers(WrappersMP.OUTER_SAMPLE,'wan14_memory')
validate_patcher(clone,model)
layer._full_precision_mm=True
try:validate_patcher(clone,model)
except ValueError:pass
else:raise AssertionError('Dense fallback admitted')
layer._full_precision_mm=False
for bad in ('hook_patches','weight_wrapper_patches'):
 previous=getattr(clone,bad);setattr(clone,bad,{'bad':True})
 try:validate_patcher(clone,model)
 except ValueError:pass
 else:raise AssertionError('Unsafe hooks admitted')
 setattr(clone,bad,previous)
try:validate_patcher(clone,torch.nn.Module())
except ValueError:pass
else:raise AssertionError('Foreign model admitted')
# Only GPU allocator/residency boundary is simulated. Patcher, wrapper executor,
# LoRA domain objects, cap decisions and streamed hook installation are real.
original=mm.LoadedModel.model_use_more_vram;input_act=ops.linear_input_act
had_forward='forward' in layer.__dict__;forward=layer.__dict__.get('forward');requests=[]
def allocation(loaded,extra_memory,force_patch_weights=False):
 requests.append(extra_memory);loaded.model.model.model_loaded_weight_memory+=extra_memory
 return extra_memory
mm.LoadedModel.model_use_more_vram=allocation
try:
 def inner():
  assert ops.linear_input_act is not input_act and 'forward' in layer.__dict__
  loaded=SimpleNamespace(model=clone,device=torch.device("mps"))
  assert mm.LoadedModel.model_use_more_vram(loaded,4*2**30)==RESIDENT_BYTES
  raise RuntimeError('controlled execution failure')
 executor=WrapperExecutor.new_class_executor(inner,SimpleNamespace(model_patcher=clone),clone.get_wrappers(WrappersMP.OUTER_SAMPLE,'wan14_memory'))
 with torch.inference_mode():
  try:executor.execute()
  except RuntimeError as e:assert str(e)=='controlled execution failure'
  else:raise AssertionError('Execution failure hidden')
 assert requests==[RESIDENT_BYTES]
 assert mm.LoadedModel.model_use_more_vram is allocation and ops.linear_input_act is input_act
 assert ('forward' in layer.__dict__)==had_forward
 if had_forward:assert layer.__dict__['forward'] is forward
finally:mm.LoadedModel.model_use_more_vram=original
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
