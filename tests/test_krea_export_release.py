"""Actual CPU mapped-model offload from the async export API boundary."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_async_release_preserves_real_mapped_quantized_and_float_storage(tmp_path):
    # Isolate Comfy's CPU bootstrap from the pytest process/device globals.
    script = r'''
import asyncio,collections,hashlib,sys,weakref
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()/'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import torch._dynamo
torch.backends.mps.is_available=lambda:False
import comfy.utils as utils
import comfy.ops as ops
import comfy.model_patcher as mp
from comfy.quant_ops import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from safetensors.torch import save_file
import krea_readonly as ro
from krea_server import capture_mapped_parameters,check_mapped_parameters
from krea_export_jobs import ExportJobs
CPU=torch.device('cpu')
root=Path(FIXTURE_ROOT)
path=root/(ro.CHECKPOINT_STEM+'.safetensors')
save_file({'quant':torch.arange(64*256,dtype=torch.int8).reshape(64,256),
           'scale':torch.ones(64,1),'float':torch.arange(8,dtype=torch.bfloat16).reshape(2,4)},str(path))
file_hash=hashlib.sha256(path.read_bytes()).hexdigest()
with torch.inference_mode(),ro.readonly_krea_load(utils,path,execution_device=torch.device('mps'),native_int8=True):
 state=utils.load_torch_file(path)
def build():
 with torch.inference_mode():
  operations=ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
  model=torch.nn.Module()
  quant=operations.Linear(256,64,bias=False,device=CPU)
  params=TensorWiseINT8Layout.Params(scale=state['scale'],orig_dtype=torch.bfloat16,orig_shape=(64,256),convrot=True,convrot_groupsize=256)
  quant.weight=torch.nn.Parameter(QuantizedTensor(state['quant'],'TensorWiseINT8Layout',params),requires_grad=False)
  quant.quant_format='int8_tensorwise';model.add_module('quant',quant)
  mixed_float=operations.Linear(4,2,bias=False,device=CPU)
  mixed_float.weight=torch.nn.Parameter(state['float'],requires_grad=False);model.add_module('mixed_float',mixed_float)
  ordinary=torch.nn.Linear(4,2,bias=False,device=CPU)
  ordinary.weight=torch.nn.Parameter(state['float'],requires_grad=False);model.add_module('ordinary',ordinary)
  return mp.ModelPatcher(model,CPU,CPU)
# Disconfirming baseline: ordinary API ambient mode clones mapped inference
# parameters in production mixed-precision modules, while normal Linear stays mapped.
base=build();originals=capture_mapped_parameters(base)
base.unpatch_model(CPU)
assert base.model.quant.weight._qdata.data_ptr()!=state['quant'].data_ptr()
assert not hasattr(base.model.quant.weight._qdata.untyped_storage(),ro.OWNER_ATTRIBUTE)
assert base.model.mixed_float.weight.data_ptr()!=state['float'].data_ptr()
assert base.model.ordinary.weight.data_ptr()==state['float'].data_ptr()
try:check_mapped_parameters(base,originals)
except (ValueError,RuntimeError):pass
else:raise AssertionError('Baseline must lose immutable mapped storage')
del base,originals
# Real release_model invokes the real patcher's unpatch and validates all mapped
# weights. Only GPU allocator/Comfy worker observation boundaries are substituted.
profile=SimpleNamespace(root=root,checkpoints=SimpleNamespace(directory=root),torch=torch,
 check=lambda:None,emit=lambda *args:None,workbench_vae=None)
profile.base_model=build();profile.model=profile.base_model
profile.original_parameters=capture_mapped_parameters(profile.base_model)
# A real backup reinstates mapped quant storage before the whole-module CPU move.
with torch.inference_mode():
 profile.base_model.backup['quant.weight']=collections.namedtuple('Dimension',['weight','inplace_update'])(profile.model.model.quant.weight,False)
 profile.model.model.quant.weight=torch.nn.Parameter(profile.model.model.quant.weight.clone(),requires_grad=False)
refs=[weakref.ref(profile.base_model),weakref.ref(profile.base_model.model)]
observations=[]
def unload():
 observations.append(('unload',torch.is_inference_mode_enabled()))
 profile.base_model.unpatch_model(CPU)
 check_mapped_parameters(profile.base_model,profile.original_parameters)
 assert not profile.base_model.backup and profile.base_model.loaded_size()==0
def empty(): observations.append(('empty',torch.is_inference_mode_enabled()))
profile.mm=SimpleNamespace(unload_all_models=unload,soft_empty_cache=empty,cleanup_models_gc=lambda:None)
queue=SimpleNamespace(set_flag=lambda *args:None)
jobs=ExportJobs(profile,queue);jobs.job={};jobs.group_rss=lambda:256*2**20
async def audit(phase,**details):
 assert not torch.is_inference_mode_enabled(), 'Thread-local inference context leaked across await'
 observations.append((phase,False))
 await asyncio.sleep(0)
 assert not torch.is_inference_mode_enabled()
jobs.audit_phase=audit
async def run():
 assert not torch.is_inference_mode_enabled()
 await jobs.release_model()
 assert not torch.is_inference_mode_enabled()
asyncio.run(run())
assert observations[:4]==[('release_entry',False),('unload',True),('empty',True),('unloaded',False)]
assert all(ref() is None for ref in refs),'Old mapped models held after release'
assert jobs.job['peak_group_rss_bytes']==256*2**20
assert hashlib.sha256(path.read_bytes()).hexdigest()==file_hash
'''
    script = script.replace('FIXTURE_ROOT', repr(str(tmp_path)))
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(ROOT / 'vendor/comfy-kitchen')
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, env=environment,
                            text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
