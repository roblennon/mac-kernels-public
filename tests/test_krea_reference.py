"""Reference geometry and immutable option admission with the pinned real author nodes."""
from pathlib import Path
import os
import subprocess
import sys

import pytest
from bench.krea_reference import load_edit_runtime

ROOT = Path(__file__).resolve().parents[1]


def test_edit_runtime_is_explicit_and_hash_bound(tmp_path, monkeypatch):
    monkeypatch.delenv('KREA_EDIT_RUNTIME', raising=False)
    with pytest.raises(ValueError, match='KREA_EDIT_RUNTIME'):
        load_edit_runtime()
    (tmp_path / 'manifest.json').write_text('{"revision":"old"}')
    with pytest.raises(ValueError, match='revision'):
        load_edit_runtime(tmp_path)


def test_author_geometry_replay_and_native_conditioning_contract():
    runtime = os.environ.get('KREA_EDIT_RUNTIME')
    if not runtime:
        pytest.skip('Pinned external Krea2Edit runtime is not configured')
    script = r'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import torch._dynamo
torch.backends.mps.is_available = lambda: False
from comfy.model_patcher import ModelPatcher
from comfy.latent_formats import Wan21
from comfy.patcher_extension import WrappersMP
from krea_reference import prepare_reference, apply_reference, validate_image, validate_reference_options, validate_spectral_reference, _PreparedVAE

# VAE encoding is the external expensive boundary; author geometry and real
# ModelPatcher clone/options handling remain real. Record pixels delivered to it.
class RecordingVAE:
    def __init__(self): self.inputs = []
    def encode(self, pixels):
        self.inputs.append(pixels.clone())
        h,w = pixels.shape[1:3]
        return torch.zeros(1,16,1,h//8,w//8)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_format = Wan21()
    def process_latent_in(self, value): return self.latent_format.process_in(value)
vae = RecordingVAE()
source = torch.rand(1,512,256,3)
subject = torch.rand(1,256,512,3)
prepared = prepare_reference('identity_edit', source, subject, vae=vae, torch=torch)
assert len(vae.inputs) == 2
assert [list(x.shape) for x in vae.inputs] == [[1,512,256,3],[1,256,512,3]]
assert prepared.metadata['order'] == 'scene,subject'
base = ModelPatcher(Model(), torch.device('cpu'), torch.device('cpu'))
positive = [[torch.zeros(1,2,12), {}]]
negative = [[torch.zeros(1,2,12), {}]]
candidate,pos,neg = apply_reference(base, positive, negative, prepared)
assert len(vae.inputs) == 2  # author replay never calls the original VAE
assert candidate is not base and candidate.model is base.model
assert base.model_options == {'transformer_options':{}}
assert pos is positive and neg is negative
validate_reference_options(candidate, prepared)
# Author wrapper receives preencoded sources and its fit offsets remain native.
wrapper = candidate.model_options['transformer_options']['wrappers'][WrappersMP.DIFFUSION_MODEL]['krea2_edit'][0]
assert wrapper.__module__ == '_mk_pinned_krea2edit'
clone = candidate.clone()
validate_reference_options(clone, prepared)
clone.model_options['transformer_options']['wrappers'][WrappersMP.DIFFUSION_MODEL]['krea2_edit'][0] = lambda *a: None
try: validate_reference_options(clone, prepared)
except ValueError: pass
else: raise AssertionError('wrapper replacement accepted')
try: _PreparedVAE(prepared.encoded).encode(torch.rand(1,512,256,3))
except RuntimeError: pass
else: raise AssertionError('unprepared encode accepted')

# Spectral must encode pixels independently at both author-fit resolutions.
# A resized full latent would never call the VAE at these half-size pixels.
vae_spectral = RecordingVAE()
spectral = prepare_reference('identity_edit', source, subject, vae=vae_spectral, torch=torch, spectral=True)
assert [list(x.shape) for x in vae_spectral.inputs] == [[1,512,256,3],[1,256,128,3],[1,256,512,3],[1,128,256,3]]
assert [list(x.shape) for x in spectral.latents] == [[1,16,1,64,32],[1,16,1,32,64]]
assert len(spectral.encoded) == 4
spectral_model,sp,sn = apply_reference(base,positive,negative,spectral)
validate_spectral_reference(spectral_model,sp,sn,spectral)
# Exercise the actual author's closure at both stages. Only its expensive
# diffusion boundary is replaced; the pinned pixel prep/replay remains real.
observed = []
def forward(dm,x,t,context,src,options,**kwargs):
    observed.append([list(v.shape) for v in src])
    return x
spectral.author.krea2_edit_forward = forward
from types import SimpleNamespace
wrapped = spectral_model.model_options['transformer_options']['wrappers'][WrappersMP.DIFFUSION_MODEL]['krea2_edit'][0]
for side in (32,64):
    value = torch.zeros(1,16,1,side,side)
    wrapped(SimpleNamespace(class_obj=object()),value,torch.ones(1),torch.zeros(1,2,12))
assert observed == [[[1,16,1,32,16],[1,16,1,16,32]],[[1,16,1,64,32],[1,16,1,32,64]]]
assert len(vae_spectral.inputs) == 4
# Round-trip all four stage entries through actual safetensors and owned reader.
from krea_reference import _write_reference_result,_read_reference_result
import tempfile,os
with tempfile.TemporaryDirectory() as directory:
    bindings = {'vae_identity': [], 'sources': {}, 'comfy_root': '', 'comfy_sources': {}}
    _write_reference_result(spectral,directory,torch=torch,bindings=bindings,pid=os.getpid())
    restored = _read_reference_result(directory,images=spectral.images,mode='identity_edit',torch=torch,
        bindings=bindings,pid=os.getpid(),runtime=None,ref_boost=4.,grounding_size=512,spectral=True)
    assert restored.metadata['encoded'] == spectral.metadata['encoded']
    assert len(restored.encoded) == 4
try: validate_spectral_reference(candidate,positive,negative,prepared)
except ValueError: pass
else: raise AssertionError('ordinary preparation admitted for spectral')
try: validate_spectral_reference(spectral_model,[[positive[0][0], {'reference_latents': list(spectral.latents)}]],sn,spectral)
except ValueError: pass
else: raise AssertionError('identity plus native references admitted')
half_key = spectral.metadata['encoded'][1]['pixels']
missing = spectral.encoded.pop(half_key)
try: validate_spectral_reference(spectral_model,sp,sn,spectral)
except ValueError: pass
else: raise AssertionError('missing half-stage latent accepted')
spectral.encoded[half_key] = missing
spectral.encoded[half_key].fill_(1)
try: validate_spectral_reference(spectral_model,sp,sn,spectral)
except ValueError: pass
else: raise AssertionError('changed half-stage latent accepted')

style = prepare_reference('style_reference',source,vae=vae,torch=torch,spectral=True)
assert style.metadata['ref_boost'] is None
assert style.metadata['grounding_size'] == 384
model,pos,neg = apply_reference(base,positive,negative,style)
assert model is base
assert 'reference_latents' not in positive[0][1]
assert pos[0][1]['reference_latents_method'] == 'index_timestep_zero'
assert pos[0][1]['reference_latents'][0] is style.latents[0]
assert neg[0][1]['reference_latents'][0] is style.latents[0]
validate_reference_options(base,style)
validate_spectral_reference(base,pos,neg,style)
for forged in [ {'reference_latents': [style.latents[0].clone()], 'reference_latents_method': 'index_timestep_zero'},
                {'reference_latents': list(style.latents), 'reference_latents_method': 'wrong'},
                {**pos[0][1], 'control': object()} ]:
 try: validate_spectral_reference(base,[[positive[0][0], forged]],neg,style)
 except ValueError: pass
 else: raise AssertionError('forged spectral conditioning accepted')
try: apply_reference(base,pos,neg,style)
except ValueError: pass
else: raise AssertionError('duplicate conditioning references accepted')
style.images[0].zero_()
try: validate_reference_options(base,style)
except ValueError: pass
else: raise AssertionError('source tensor drift accepted')
for invalid in [torch.zeros(2,512,512,3),torch.zeros(1,513,512,3),torch.full((1,16,16,3),float('nan')),torch.full((1,16,16,3),2.),torch.zeros(1,16,16,4)]:
 try: validate_image(invalid,torch)
 except ValueError: pass
 else: raise AssertionError('invalid source accepted')
print('Author geometry/replay and native conditioning contracts passed')
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
        text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('size', [(832,1248), (1248,832), (1024,1024), (896,1184), (1184,896),
                                 (928,1152), (768,1376), (1376,768), (1568,672), (512,512)])
def test_rectangular_author_geometry_owned_roundtrip_and_stage_replay(size):
    if not os.environ.get('KREA_EDIT_RUNTIME'):
        pytest.skip('Pinned external Krea2Edit runtime is not configured')
    script = r'''
import sys,json,tempfile,os
width,height=json.loads(sys.argv[1])
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import torch._dynamo
torch.backends.mps.is_available=lambda:False
from comfy.model_patcher import ModelPatcher
from comfy.latent_formats import Wan21
from comfy.patcher_extension import WrappersMP
from types import SimpleNamespace
from krea_reference import prepare_reference,apply_reference,validate_reference_options,validate_spectral_reference,_write_reference_result,_read_reference_result
class VAE:
 def __init__(self):self.inputs=[]
 def encode(self,pixels):
  self.inputs.append(pixels.clone())
  h,w=pixels.shape[1:3]
  return torch.nn.functional.adaptive_avg_pool2d(pixels.movedim(-1,1), (h//8,w//8)).mean(1,keepdim=True).repeat(1,16,1,1).unsqueeze(2)
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__();self.latent_format=Wan21()
 def process_latent_in(self,value):return self.latent_format.process_in(value)
source=torch.linspace(0,1,512*256*3).reshape(1,512,256,3)
subject=torch.linspace(1,0,256*512*3).reshape(1,256,512,3)
vae=VAE()
p=prepare_reference('identity_edit',source,subject,vae=vae,torch=torch,spectral=True,output_size=(width,height))
assert p.output_size==(width,height)
assert p.metadata['output_size']==[width,height]
assert [row['target'] for row in p.metadata['encoded']]==[[height//8,width//8],[height//16,width//16]]*2
    # Genuine aspect mismatch uses the author's /16 floor grid; these source ARs
    # are deliberately far from every preset tested.
for index,image in enumerate((source,subject)):
 for stage,scale in enumerate((1,.5)):
  h,w=image.shape[1:3];th,tw=int(height*scale),int(width*scale)
  factor=min(th/h,tw/w)
  expected_h=min(max(16,int(h*factor)//16*16),th//16*16)
  expected_w=min(max(16,int(w*factor)//16*16),tw//16*16)
  assert list(vae.inputs[index*2+stage].shape)==[1,expected_h,expected_w,3]
base=ModelPatcher(Model(),torch.device('cpu'),torch.device('cpu'))
positive=[[torch.zeros(1,2,12),{}]];negative=[[torch.zeros(1,2,12),{}]]
model,pos,neg=apply_reference(base,positive,negative,p)
validate_spectral_reference(model,pos,neg,p)
observed=[]
def forward(dm,x,t,context,src,options,**kwargs):
 observed.append([list(v.shape) for v in src]);return x
p.author.krea2_edit_forward=forward
wrapper=model.model_options['transformer_options']['wrappers'][WrappersMP.DIFFUSION_MODEL]['krea2_edit'][0]
for h,w in ((height//16,width//16),(height//8,width//8)):
 value=torch.zeros(1,16,1,h,w)
 wrapper(SimpleNamespace(class_obj=object()),value,torch.ones(1),torch.zeros(1,2,12))
assert observed==[[row['shape'] for row in p.metadata['encoded'][1::2]],[row['shape'] for row in p.metadata['encoded'][::2]]]
assert len(vae.inputs)==4 # No original VAE call during wrapper execution.
with tempfile.TemporaryDirectory() as directory:
 bindings={'vae_identity':[],'sources':{},'comfy_root':'','comfy_sources':{}}
 _write_reference_result(p,directory,torch=torch,bindings=bindings,pid=os.getpid())
 kwargs=dict(images=p.images,mode='identity_edit',torch=torch,bindings=bindings,pid=os.getpid(),runtime=None,ref_boost=4.,grounding_size=512,spectral=True)
 restored=_read_reference_result(directory,**kwargs,output_size=(width,height))
 assert restored.metadata['encoded']==p.metadata['encoded']
 try:_read_reference_result(directory,**kwargs,output_size=(512,768) if (width,height)==(512,512) else (512,512))
 except RuntimeError:pass
 else:raise AssertionError('worker output geometry mismatch accepted')
p.metadata['output_size']=[height,width] if height!=width else [512,768]
try:validate_reference_options(model,p)
except ValueError:pass
else:raise AssertionError('output size drift accepted')
# Style remains source-sized even when output geometry changes.
style_vae=VAE()
style=prepare_reference('style_reference',source,subject,vae=style_vae,torch=torch,spectral=True,output_size=(width,height))
assert [list(x.shape) for x in style.latents]==[[1,16,1,64,32],[1,16,1,32,64]]
assert len(style_vae.inputs)==2
# A matched source fills the complete target, including the widest 196-token
# latent side. This exercises the new bound rather than only fit-inside margins.
matched_vae=VAE()
matched_source=torch.rand(1,height//4,width//4,3)
matched=prepare_reference('identity_edit',matched_source,vae=matched_vae,torch=torch,spectral=True,output_size=(width,height))
assert [list(x.shape) for x in matched_vae.inputs]==[[1,height,width,3],[1,height//2,width//2,3]]
assert list(matched.latents[0].shape)==[1,16,1,height//8,width//8]
'''
    import json
    result = subprocess.run([sys.executable, '-c', script, json.dumps(size)], cwd=ROOT,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('size', [(0,512), (512,513), (1600,512), (1568,1568), (1376,800),
                                 (512.,768), (True,512), (512,), '512x768', (512,float('nan'))])
def test_output_size_refuses_unbounded_or_unaligned_geometry(size):
    from bench.krea_reference import validate_output_size
    with pytest.raises(ValueError, match='output size'):
        validate_output_size(size)
