"""Real CPU patchers verify H3 sampler ownership, caps and cleanup."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_h3_cloned_wrapper_caps_residency_and_restores_hooks(tmp_path):
    script = r'''
import json,sys
from pathlib import Path
from types import SimpleNamespace
fixture_root=Path(sys.argv[1])
sys.path.insert(0,str(Path.cwd()/'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import comfy.ops as ops
import comfy.utils as utils
import comfy.model_management as mm
import comfy.model_patcher as mp
from comfy.weight_adapter import LoRAAdapter
from comfy.patcher_extension import WrapperExecutor,WrappersMP
from comfy_kitchen.tensor import base as kitchen_base
from safetensors.torch import save_file
from bench import krea_readonly as ro
from bench.h3_memory import readonly_grouped_detach,validate_patcher,attach_memory_wrapper,RESIDENT_BYTES

packed='diffusion_model.blocks.0.attn.qkv_proj'
dense='diffusion_model.token_refiner.blocks.0.attn.qkv_proj'
conf={'format':'asym_w4a8_int8','group_size':16,'convrot_groupsize':256}
path=fixture_root/'tiny-h3.safetensors'
save_file({packed+'.weight':torch.zeros(64,128,dtype=torch.int8),
 packed+'.weight_s_rel':torch.ones(64,16).to(torch.float8_e4m3fn),
 packed+'.weight_s_channel':torch.ones(64),packed+'.weight_codebook':torch.linspace(-1,1,16),
 packed+'.comfy_quant':torch.tensor(list(json.dumps(conf).encode()),dtype=torch.uint8),
 dense+'.weight':torch.ones(64,256,dtype=torch.bfloat16)},str(path))
model=torch.nn.Module()
for name in (packed,dense):
 parent=model;parts=name.split('.')
 for part in parts[:-1]:
  if part not in parent._modules:parent.add_module(part,torch.nn.Module())
  parent=parent._modules[part]
 parent.add_module(parts[-1],ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(256,64,bias=False,device='cpu'))
with torch.inference_mode(),readonly_grouped_detach():
 state,_=ro._load(utils,path,ro._stat(path.stat()),utils.safetensors_header(str(path)),allowed_dtypes={'I8','F8_E4M3','F32','U8','BF16'})
 model.load_state_dict(dict(state),strict=True,assign=True)
patcher=mp.ModelPatcher(model,torch.device('mps'),torch.device('cpu'))
plain=LoRAAdapter(set(),(torch.ones(64,2),torch.ones(2,256),None,None,None,None))
patcher.add_patches({packed+'.weight':plain,dense+'.weight':plain},strength_patch=.5)
# A second plain slot remains ordinary Comfy stack composition.
patcher.add_patches({packed+'.weight':plain},strength_patch=.25)
attach_memory_wrapper(patcher);clone=patcher.clone();validate_patcher(clone,model)
assert clone.get_wrappers(WrappersMP.OUTER_SAMPLE,'h3_memory')==patcher.get_wrappers(WrappersMP.OUTER_SAMPLE,'h3_memory')
layer=model.get_submodule(packed);refiner=model.get_submodule(dense)
layer._full_precision_mm=True
try:validate_patcher(clone,model)
except ValueError:pass
else:raise AssertionError('Dense packed fallback admitted')
layer._full_precision_mm=False
for name in ('hook_patches','weight_wrapper_patches'):
 previous=getattr(clone,name);setattr(clone,name,{'bad':True})
 try:validate_patcher(clone,model)
 except ValueError:pass
 else:raise AssertionError('Unsafe patch state admitted')
 setattr(clone,name,previous)
try:validate_patcher(clone,torch.nn.Module())
except ValueError:pass
else:raise AssertionError('Foreign model admitted')
entries=clone.patches[dense+'.weight'];original_entry=entries[0]
entries[0]=(original_entry[0],original_entry[1],.5,None,None)
try:validate_patcher(clone,model)
except ValueError:pass
else:raise AssertionError('Base-weight transform admitted')
entries[0]=original_entry

# Simulate only the GPU allocator boundary. Native patcher/clone, wrapper,
# dispatch-table ownership and hook installation/restoration remain real.
original_allocate=mm.LoadedModel.model_use_more_vram
original_act=ops.linear_input_act
original_detach=kitchen_base._DISPATCH_TABLE[torch.ops.aten.detach.default]
requests=[]
def allocation(loaded,extra_memory,force_patch_weights=False):
 requests.append(extra_memory);loaded.model.model.model_loaded_weight_memory+=extra_memory
 return extra_memory
mm.LoadedModel.model_use_more_vram=allocation
try:
 def inner():
  assert kitchen_base._DISPATCH_TABLE[torch.ops.aten.detach.default] is not original_detach
  assert ops.linear_input_act is not original_act and 'forward' in layer.__dict__
  assert 'forward' not in refiner.__dict__
  loaded=SimpleNamespace(model=clone,device=torch.device('mps'))
  assert mm.LoadedModel.model_use_more_vram(loaded,4*2**30)==RESIDENT_BYTES
  raise RuntimeError('controlled sampling failure')
 executor=WrapperExecutor.new_class_executor(inner,SimpleNamespace(model_patcher=clone),clone.get_wrappers(WrappersMP.OUTER_SAMPLE,'h3_memory'))
 with torch.inference_mode():
  try:executor.execute()
  except RuntimeError as error:assert str(error)=='controlled sampling failure'
  else:raise AssertionError('Sampling failure hidden')
 assert requests==[RESIDENT_BYTES]
 assert mm.LoadedModel.model_use_more_vram is allocation
 assert ops.linear_input_act is original_act
 assert kitchen_base._DISPATCH_TABLE[torch.ops.aten.detach.default] is original_detach
 assert 'forward' not in layer.__dict__ and 'forward' not in refiner.__dict__
 for value in [layer.weight._qdata]+[getattr(layer.weight._params,field) for field in layer.weight._params._tensor_fields()]:
  assert getattr(value.untyped_storage(),ro.OWNER_ATTRIBUTE)[1].readonly
finally:mm.LoadedModel.model_use_more_vram=original_allocate
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
