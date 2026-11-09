"""Mechanism-level CPU ownership proof; not a WAN model admission or render."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_real_mixed_construction_preserves_fp16_int8_and_restored_backups(tmp_path):
    script = r'''
import collections,gc,hashlib,json,sys,weakref
from pathlib import Path
fixture_root=Path(sys.argv[1])
sys.path.insert(0,str(Path.cwd()/'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import comfy.utils as utils
import comfy.ops as ops
import comfy.model_patcher as mp
from safetensors.torch import save_file
import krea_readonly as ro
path=fixture_root/'tiny-wan-layout.safetensors'
conf={'format':'int8_tensorwise','convrot':True,'convrot_groupsize':256}
q='blocks.0.self_attn.q'; f='time_projection.1'
source={q+'.weight':torch.arange(64*256,dtype=torch.int8).reshape(64,256),
        q+'.weight_scale':torch.ones(64,1),q+'.bias':torch.ones(64,dtype=torch.float16),
        q+'.comfy_quant':torch.tensor(list(json.dumps(conf).encode()),dtype=torch.uint8),
        f+'.weight':torch.arange(64*256,dtype=torch.float16).reshape(64,256),
        f+'.bias':torch.ones(64,dtype=torch.float16)}
save_file(source,str(path)); del source
before=hashlib.sha256(path.read_bytes()).hexdigest()
identity=ro._stat(path.stat());header=utils.safetensors_header(str(path))
with torch.inference_mode():
 # Public Krea default must continue to refuse F16.
 try:ro._load(utils,path,identity,header)
 except ValueError as e:assert 'F16' in str(e)
 else:raise AssertionError('Krea default admission was widened')
 state,_=ro._load(utils,path,identity,header,allowed_dtypes=frozenset({'I8','F16','F32','U8'}))
 pointers={k:v.data_ptr() for k,v in state.items()}
 scale_expected=state[q+'.weight_scale'].clone()
 owners=[weakref.ref(getattr(v.untyped_storage(),ro.OWNER_ATTRIBUTE)[0]) for v in state.values()]
 assert all(getattr(v.untyped_storage(),ro.OWNER_ATTRIBUTE)[1].readonly for v in state.values())
 def build(dtype):
  model=torch.nn.Module()
  for key in (q,f):
   parent=model; parts=key.split('.')
   for part in parts[:-1]:
    if part not in parent._modules:parent.add_module(part,torch.nn.Module())
    parent=parent._modules[part]
   parent.add_module(parts[-1],ops.mixed_precision_ops(compute_dtype=dtype).Linear(256,64,bias=True,device='cpu'))
  model.load_state_dict(dict(state),strict=True,assign=True)
  return mp.ModelPatcher(model,torch.device('cpu'),torch.device('cpu'))
 def views(patcher):
  quant=patcher.model.get_submodule(q); ordinary=patcher.model.get_submodule(f)
  return {q+'.weight':quant.weight._qdata,
          q+'.bias':quant.bias,f+'.weight':ordinary.weight,f+'.bias':ordinary.bias}
 def preserved(patcher):
  for key,value in views(patcher).items():
   assert value.data_ptr()==pointers[key], (key, str(value.dtype), value.data_ptr(), pointers[key])
   assert getattr(value.untyped_storage(),ro.OWNER_ATTRIBUTE)[1].readonly,key
 candidate=build(torch.float16);preserved(candidate)
 # Parameter wrapping invokes QuantizedTensor.detach, which clones scale metadata.
 scale=candidate.model.get_submodule(q).weight._params.scale
 assert scale.data_ptr()!=pointers[q+'.weight_scale'] and torch.equal(scale,scale_expected)
 assert not hasattr(scale.untyped_storage(),ro.OWNER_ATTRIBUTE)
 copied_scale_bytes=scale.numel()*scale.element_size()
 # Disconfirming conversion: ordinary FP16 globals copy when compute dtype differs.
 converted=build(torch.bfloat16)
 assert converted.model.get_submodule(f).weight.dtype==torch.bfloat16
 assert converted.model.get_submodule(f).weight.data_ptr()!=pointers[f+'.weight']
 assert not hasattr(converted.model.get_submodule(f).weight.untyped_storage(),ro.OWNER_ATTRIBUTE)
 assert converted.model.get_submodule(q).weight._qdata.data_ptr()==pointers[q+'.weight']
 del converted
 # Real unpatch restores both quantized and ordinary mapped backups.
 backup=collections.namedtuple('Backup',['weight','inplace_update'])
 for key in (q+'.weight',f+'.weight'):
  module=candidate.model.get_submodule(key[:-7]);candidate.backup[key]=backup(module.weight,False)
  module.weight=torch.nn.Parameter(module.weight.clone(),requires_grad=False)
 candidate.unpatch_model(torch.device('cpu'));preserved(candidate)
 assert torch.equal(candidate.model.get_submodule(q).weight._params.scale,scale_expected)
 del state;gc.collect();preserved(candidate)
 assert any(owner() is not None for owner in owners)
 # Disconfirming offload: ambient mode clones inference weights in actual Comfy _apply.
candidate.unpatch_model(torch.device('cpu'))
assert candidate.model.get_submodule(q).weight._qdata.data_ptr()!=pointers[q+'.weight']
assert candidate.model.get_submodule(f).weight.data_ptr()!=pointers[f+'.weight']
assert hashlib.sha256(path.read_bytes()).hexdigest()==before
print('WAN_CPU_MAPPING_PROOF_OK',json.dumps({'copied_scale_bytes':copied_scale_bytes}))
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'WAN_CPU_MAPPING_PROOF_OK' in result.stdout
