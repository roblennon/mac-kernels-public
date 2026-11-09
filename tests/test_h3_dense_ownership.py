"""CPU evidence for H3's ordinary BF16 token-refiner LoRA targets."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_real_bf16_refiner_construction_patch_and_streaming_boundary(tmp_path):
    script = r'''
import gc,hashlib,sys,weakref
from pathlib import Path
fixture_root=Path(sys.argv[1])
sys.path.insert(0,str(Path.cwd()/'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import comfy.utils as utils
import comfy.ops as ops
import comfy.model_patcher as mp
from comfy.weight_adapter import LoRAAdapter
from safetensors.torch import save_file
import krea_readonly as ro
from bench.krea_adapter import streamed_adapter_sides

# Representative ordinary matrix target from H3's token-refiner blocks.
key='diffusion_model.token_refiner.blocks.0.attn.qkv_proj.weight'
path=fixture_root/'tiny-h3-refiner.safetensors'
expected=(torch.arange(64*256,dtype=torch.float32).reshape(64,256)%23/32).bfloat16()
save_file({key:expected},str(path))
before=hashlib.sha256(path.read_bytes()).hexdigest()
with torch.inference_mode():
 state,_=ro._load(utils,path,ro._stat(path.stat()),utils.safetensors_header(str(path)))
 source=state[key];pointer=source.data_ptr()
 owner=weakref.ref(getattr(source.untyped_storage(),ro.OWNER_ATTRIBUTE)[0])
 def build(dtype):
  model=torch.nn.Module();parent=model
  parts=key.split('.')[:-1]
  for part in parts[:-1]:
   child=torch.nn.Module();parent.add_module(part,child);parent=child
  parent.add_module(parts[-1],ops.mixed_precision_ops(compute_dtype=dtype).Linear(256,64,bias=False,device='cpu'))
  model.load_state_dict(dict(state),strict=True,assign=True)
  return mp.ModelPatcher(model,torch.device('cpu'),torch.device('cpu'))
 candidate=build(torch.bfloat16);layer=candidate.model.get_submodule(key[:-7])
 assert layer.weight.dtype==torch.bfloat16 and layer.weight.data_ptr()==pointer
 assert getattr(layer.weight.untyped_storage(),ro.OWNER_ATTRIBUTE)[1].readonly
 assert not hasattr(layer.weight,'_params')

 # Disconfirming construction: FP16 compute casts and allocates dense globals.
 converted=build(torch.float16);converted_weight=converted.model.get_submodule(key[:-7]).weight
 assert converted_weight.dtype==torch.float16 and converted_weight.data_ptr()!=pointer
 assert not hasattr(converted_weight.untyped_storage(),ro.OWNER_ATTRIBUTE)
 del converted,converted_weight

 # Real Comfy patch math creates a separate merged weight, retains the immutable
 # mapped backup, and restores its source identity on actual inference unpatch.
 adapter=LoRAAdapter(set(),(torch.ones(64,2),torch.ones(2,256),None,None,None,None))
 assert candidate.add_patches({key:adapter},strength_patch=.5)==[key]
 candidate.patch_weight_to_device(key,device_to=torch.device('cpu'))
 assert layer.weight.data_ptr()!=pointer
 assert torch.equal(layer.weight,expected+1)
 assert candidate.backup[key].weight.data_ptr()==pointer
 assert getattr(candidate.backup[key].weight.untyped_storage(),ro.OWNER_ATTRIBUTE)[1].readonly
 assert torch.equal(source,expected)
 candidate.unpatch_model(torch.device('cpu'))
 assert layer.weight.data_ptr()==pointer and torch.equal(layer.weight,expected)
 assert getattr(layer.weight.untyped_storage(),ro.OWNER_ATTRIBUTE)[1].readonly
 assert not candidate.backup

 # Actual small CPU forwards prove ordinary refiner LowVramPatch behavior is
 # untouched by the packed-layer helper, including its fused entry point.
 original=ops.linear_input_act;had_forward='forward' in layer.__dict__
 previous_forward=layer.__dict__.get('forward')
 layer.weight_function=[mp.LowVramPatch(key,candidate.patches)]
 functions=layer.weight_function
 x=torch.arange(2*256,dtype=torch.float32).reshape(2,256).remainder(11).div(32).bfloat16()
 baseline=layer(x)
 activation_baseline=ops.linear_input_act(layer,x,'gelu_tanh')
 expected_output=torch.nn.functional.linear(x,expected+1)
 torch.testing.assert_close(baseline,expected_output,rtol=0,atol=0)
 torch.testing.assert_close(activation_baseline,torch.nn.functional.linear(torch.nn.functional.gelu(x,approximate='tanh'),expected+1),rtol=0,atol=0)
 try:
  with streamed_adapter_sides(candidate) as audit:
   assert ('forward' in layer.__dict__)==had_forward
   torch.testing.assert_close(layer(x),baseline,rtol=0,atol=0)
   torch.testing.assert_close(ops.linear_input_act(layer,x,'gelu_tanh'),activation_baseline,rtol=0,atol=0)
   assert audit['streamed_side_entries']==0
  assert layer.weight_function is functions and not layer.lora_side_patches
  assert ops.linear_input_act is original
  assert ('forward' in layer.__dict__)==had_forward
  if had_forward:assert layer.__dict__['forward'] is previous_forward
 finally:
  layer.weight_function=[]
 del state,source;gc.collect()
 assert owner() is not None and layer.weight.data_ptr()==pointer
 assert torch.equal(layer.weight,expected)
assert hashlib.sha256(path.read_bytes()).hexdigest()==before
print('H3_DENSE_OWNERSHIP_PROOF_OK')
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'H3_DENSE_OWNERSHIP_PROOF_OK' in result.stdout
