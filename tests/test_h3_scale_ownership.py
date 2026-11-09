"""Tiny real H3 storage-format proof; no model render or GPU allocations."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_grouped_scales_copy_during_parameter_wrapping(tmp_path):
    script = r'''
import gc,hashlib,json,sys
from pathlib import Path
fixture_root=Path(sys.argv[1])
sys.path.insert(0,str(Path.cwd()/'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import comfy.ops as ops
import comfy.utils as utils
from comfy.quant_ops import QuantizedTensor
from safetensors.torch import save_file
from bench import krea_readonly as ro
path=fixture_root/'tiny-h3.safetensors'
conf={'format':'asym_w4a8_int8','group_size':16,'convrot_groupsize':256}
source={'weight':torch.zeros(64,128,dtype=torch.int8),
 'weight_s_rel':torch.ones(64,16).to(torch.float8_e4m3fn),
 'weight_s_channel':torch.ones(64),
 'weight_codebook':torch.linspace(-1,1,16),
 'comfy_quant':torch.tensor(list(json.dumps(conf).encode()),dtype=torch.uint8)}
save_file(source,str(path));del source
before=hashlib.sha256(path.read_bytes()).hexdigest()
with torch.inference_mode():
 state,_=ro._load(utils,path,ro._stat(path.stat()),utils.safetensors_header(str(path)),
                 allowed_dtypes={'I8','F8_E4M3','F32','U8'})
 ptr={k:v.data_ptr() for k,v in state.items()}
 layer=ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(256,64,bias=False,device='cpu')
 layer.load_state_dict(dict(state),strict=True,assign=True)
 weight=layer.weight
 assert isinstance(weight,QuantizedTensor)
 assert weight._qdata.data_ptr()==ptr['weight']
 assert weight._params.orig_dtype==torch.bfloat16
 fields={'scale':'weight_s_rel','s_channel':'weight_s_channel','codebook':'weight_codebook'}
 copied=0
 for field,key in fields.items():
  value=getattr(weight._params,field)
  assert value.data_ptr()!=ptr[key],field
  assert not hasattr(value.untyped_storage(),ro.OWNER_ATTRIBUTE),field
  assert torch.equal(value.float(),state[key].float()),field
  copied+=value.numel()*value.element_size()
 # Device-only no-op to() aliases the existing tensor. Detach alone copies all
 # metadata again despite sharing qdata; explicit clone copies qdata as well.
 noop=weight.to(device='cpu')
 assert noop._qdata.data_ptr()==weight._qdata.data_ptr()
 for field in fields:
  assert getattr(noop._params,field).data_ptr()==getattr(weight._params,field).data_ptr(),field
 detached=weight.detach();cloned=weight.clone()
 assert detached._qdata.data_ptr()==weight._qdata.data_ptr()
 assert cloned._qdata.data_ptr()!=weight._qdata.data_ptr()
 for field in fields:
  assert getattr(detached._params,field).data_ptr()!=getattr(weight._params,field).data_ptr()
  assert getattr(cloned._params,field).data_ptr()!=getattr(weight._params,field).data_ptr()
 # Changed construction: only mapped immutable W4A8 detach shares metadata.
 from bench.h3_memory import readonly_grouped_detach
 from comfy_kitchen.tensor import base
 handler=base._DISPATCH_TABLE[torch.ops.aten.detach.default]
 try:
  with readonly_grouped_detach():
   fixed=ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(256,64,bias=False,device='cpu')
   fixed.load_state_dict(dict(state),strict=True,assign=True)
   def preserved():
    assert fixed.weight._qdata.data_ptr()==ptr['weight']
    for field,key in fields.items():
     value=getattr(fixed.weight._params,field)
     assert value.data_ptr()==ptr[key],field
     assert getattr(value.untyped_storage(),ro.OWNER_ATTRIBUTE)[1].readonly
   preserved();fixed.to('cpu');preserved()
   independent=fixed.weight.clone()
   assert independent._qdata.data_ptr()!=ptr['weight']
   for field,key in fields.items():assert getattr(independent._params,field).data_ptr()!=ptr[key]
   independent._params.s_channel.add_(1)
   assert torch.equal(fixed.weight._params.s_channel,state['weight_s_channel'])
   # Anonymous tensors keep their ordinary detach copying behavior.
   plain=independent.detach()
   assert plain._params.scale.data_ptr()!=independent._params.scale.data_ptr()
   raise RuntimeError('controlled construction failure')
 except RuntimeError as error:assert str(error)=='controlled construction failure'
 assert base._DISPATCH_TABLE[torch.ops.aten.detach.default] is handler
 del state;gc.collect()
 assert getattr(weight._qdata.untyped_storage(),ro.OWNER_ATTRIBUTE)[1].readonly
 # Real CPU module _apply/offload preserves qdata in inference mode but cannot
 # recover original mapped scales after construction cloned them.
 layer.to('cpu')
 assert layer.weight._qdata.data_ptr()==ptr['weight']
 for field,key in fields.items():assert getattr(layer.weight._params,field).data_ptr()!=ptr[key]
assert hashlib.sha256(path.read_bytes()).hexdigest()==before
print('H3_SCALE_OWNERSHIP',json.dumps({'copied_scale_bytes':copied,'compute_dtype':'bfloat16'}))
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], cwd=ROOT,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'H3_SCALE_OWNERSHIP' in result.stdout
