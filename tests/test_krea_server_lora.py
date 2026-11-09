"""Opt-in API selection boundaries and clone publication without GPU work."""
from contextlib import contextmanager
import copy
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from krea_server import Profile, UNET, ENCODER, VAE, identity, node_classes, resolve_plain_lora, validate_graph
from probe_krea_server import graph as base_graph
from probe_krea_server_lora import RWB, DARKBRUSH, graph, graphs, output_image, summarize


def test_opt_in_graph_requires_exact_selection_schema_and_does_not_change_base_graph():
    original = base_graph(42,'soccer','review/krea-api-cpu/base')
    validate_graph(original)
    current = graph(42,'soccer','review/krea-api-cpu/style',RWB,.8)
    validate_graph(current,plain_loras=True)
    with pytest.raises(ValueError):
        validate_graph(current)
    with pytest.raises(KeyError):
        validate_graph(original,plain_loras=True)
    changes = [lambda g: g['1']['inputs'].update(lora_name='../outside.safetensors'),
               lambda g: g['1']['inputs'].update(lora_name='other\\file.safetensors'),
               lambda g: g['1']['inputs'].update(lora_strength=float('nan')),
               lambda g: g['1']['inputs'].update(lora_strength=0),
               lambda g: g['1']['inputs'].update(lora_strength=True),
               lambda g: g['1']['inputs'].update(unapproved='value'),
               lambda g: g['3']['inputs'].update(steps=4)]
    for change in changes:
        bad = copy.deepcopy(current)
        change(bad)
        with pytest.raises(ValueError):
            validate_graph(bad,plain_loras=True)
    planned = graphs('cpu')
    assert len(planned) == 5
    assert [g['1']['inputs']['lora_name'] for _,g in planned] == ['',RWB,RWB,DARKBRUSH,'']
    assert [g['1']['inputs']['lora_strength'] for _,g in planned] == [1.,.8,.4,1.,1.]
    assert len({g['1']['inputs']['prompt'] for _,g in planned}) == 1
    assert {g['3']['inputs']['seed'] for _,g in planned} == {42}


@pytest.fixture
def api_fixture(tmp_path,monkeypatch):
    for kind,name in (('diffusion_models',UNET),('text_encoders',ENCODER),('vae',VAE),
                      ('loras',RWB),('loras',DARKBRUSH)):
        path = tmp_path/'models'/kind/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'fixture')
    paths = SimpleNamespace(get_full_path_or_raise=lambda kind,name: str(tmp_path/'models'/kind/name))
    state = SimpleNamespace(loads=0,encodings=0,unloads=0,restore=True,fail_load=False,fail_contract=False)
    key = 'diffusion_model.blocks.0.attn.wq.weight'
    layer = SimpleNamespace(weight=object(),weight_function=[],lora_side_patches=())
    source = layer.weight
    target = SimpleNamespace(get_submodule=lambda name: layer)
    target.named_parameters = lambda: [(key,layer.weight)]
    class Patcher:
        def __init__(self,patches=None):
            self.model = target
            self.patches = {} if patches is None else patches
            self.backup = {}
            self.loaded = 0
        def loaded_size(self):
            return self.loaded
    base = Patcher()
    class Object:
        pass
    class Clip:
        def __init__(self):
            self.cond_stage_model,self.patcher = Object(),Object()
    class Loader:
        def load_clip(self,*args,**kwargs):
            return (Clip(),)
    class Unet:
        def load_unet(self,*args):
            state.loads += 1
            return (base,)
    encoding = ModuleType('krea_conditioning')
    def encode(mapping,**kwargs):
        clip = mapping['CLIPLoader']().load_clip()[0]
        state.encodings += 1
        return [],[],{'origin':'fresh-encoding','cache_policy':'off'}
    encoding.encode_conditioning = encode
    readonly = ModuleType('krea_readonly')
    @contextmanager
    def load(*args,**kwargs):
        yield {'loads':1}
    readonly.readonly_krea_load = load
    adapter = ModuleType('krea_adapter')
    def verify(path,*,strength,steps,cfg):
        return {'path':str(path),'file_identity':identity(path),'sha256':'a'*64,
                'policy':'plain','strength':strength}
    def unchanged(receipt):
        if identity(receipt['path']) != receipt['file_identity']:
            raise RuntimeError('adapter identity changed')
    def apply(actual,utils,receipt,strength):
        assert actual is base and not actual.patches and not actual.backup
        if state.fail_load:
            raise ValueError('unsupported adapter')
        return Patcher({key:[(strength,object())]})
    class Contract:
        def __init__(self,mm,utils,patcher,receipt,**kwargs):
            assert not patcher.backup and patcher.loaded_size() == 0
            if state.fail_contract:
                raise ValueError('failed contract')
            self.sources = {key:id(source)}
    adapter.verify_plain_adapter,adapter.load_plain_adapter,adapter.AdapterContract = verify,apply,Contract
    adapter._asset_unchanged,adapter._mapped_storage = unchanged,id
    monkeypatch.setitem(sys.modules,'krea_conditioning',encoding)
    monkeypatch.setitem(sys.modules,'krea_readonly',readonly)
    monkeypatch.setitem(sys.modules,'krea_adapter',adapter)
    import krea_server as server_module
    # Detached mapped views are a Torch/filesystem boundary proved separately
    # below using the real Comfy patcher and readonly mappings.
    monkeypatch.setattr(server_module,'capture_mapped_parameters',
                        lambda actual: {key:(source,id(source))})
    def unload():
        state.unloads += 1
        current = profile.model
        if current is not None and state.restore:
            current.backup.clear()
            current.loaded = 0
            layer.weight,layer.weight_function,layer.lora_side_patches = source,[],()
    mm = SimpleNamespace(unload_all_models=unload,soft_empty_cache=lambda:None,
                         cleanup_models_gc=lambda:None,get_torch_device=lambda:'fake-mps')
    torch = SimpleNamespace(mps=SimpleNamespace(set_per_process_memory_fraction=lambda fraction:None))
    nodes = SimpleNamespace(NODE_CLASS_MAPPINGS={'CLIPLoader':Loader,'UNETLoader':Unet})
    profile = Profile(root=tmp_path,code=tmp_path,torch=torch,folder_paths=paths,mm=mm,
                      utils=object(),nodes=nodes,argv=[],plain_loras=True)
    profile.emit = lambda *args:None
    return SimpleNamespace(profile=profile,state=state,base=base,layer=layer,source=source,
                           paths=paths,root=tmp_path,key=key)


def test_switches_clone_and_strength_then_restore_the_cached_base(api_fixture):
    f = api_fixture
    profile = f.profile
    assert profile.prepare('soccer')[0] is f.base
    assert profile.lora is None
    first = profile.prepare('soccer',RWB,.8)[0]
    assert first is not f.base and first.model is f.base.model
    assert profile.lora['strength'] == .8 and profile.lora['patch_keys'] == [f.key]
    first.backup[f.key] = f.source
    first.loaded = 123
    f.layer.weight,f.layer.weight_function = object(),[object()]
    second = profile.prepare('soccer',RWB,.4)[0]
    assert second is not first and second.patches[f.key][0][0] == .4
    assert not first.backup and f.layer.weight is f.source
    third = profile.prepare('soccer',DARKBRUSH,1.)[0]
    assert third is not second and profile.lora['name'] == DARKBRUSH
    restored = profile.prepare('soccer')[0]
    assert restored is f.base and profile.active_contract is None and profile.lora is None
    assert not f.base.patches and not f.base.backup
    assert f.state.loads == 1 and f.state.encodings == f.state.unloads == 5
    assert profile.conditioning['lora'] is None
    assert 'krea_adapter.py' in profile.sources and 'probe_krea_server_lora.py' in profile.sources


@pytest.mark.parametrize('failure',['fail_load','fail_contract'])
def test_failed_selection_is_not_published_and_can_return_to_base(api_fixture,failure):
    f = api_fixture
    old = f.profile.prepare('soccer',RWB,.8)[0]
    setattr(f.state,failure,True)
    with pytest.raises(ValueError):
        f.profile.prepare('soccer',DARKBRUSH,1.)
    assert f.profile.model is None and f.profile.active_contract is None
    assert f.profile.adapter_receipt is None and f.profile.conditioning is None
    assert f.profile.base_model is f.base and not f.base.patches
    assert old is not f.profile.model
    setattr(f.state,failure,False)
    assert f.profile.prepare('soccer')[0] is f.base
    assert f.state.loads == 1


@pytest.mark.parametrize('restore_fault',['pointer','owner'])
def test_unrestored_source_refuses_a_new_selection_before_encoding(api_fixture,restore_fault):
    f = api_fixture
    old = f.profile.prepare('soccer',RWB,.8)[0]
    f.layer.weight = object()
    f.state.restore = False
    if restore_fault == 'owner':
        def missing_owner(weight):
            raise ValueError('readonly immutable base storage owner lost')
        sys.modules['krea_adapter']._mapped_storage = missing_owner
    with pytest.raises(RuntimeError,match='immutable base storage'):
        f.profile.prepare('soccer',DARKBRUSH,1.)
    assert f.profile.model is old and f.profile.active_contract is not None
    assert f.profile.restore_failure is not None
    assert f.state.encodings == 1 and f.state.loads == 1
    # The independent review reproduced a retry bypass after the first failure
    # discarded the old contract. Both a later base request and a cached-model
    # direct sampler invocation must remain refused without entering GPU code.
    with pytest.raises(RuntimeError,match='restart the dedicated server'):
        f.profile.prepare('soccer')
    with pytest.raises(RuntimeError,match='restart the dedicated server'):
        f.profile.sample(old,42,8,1.,'euler','simple',[],[],
            {'samples':SimpleNamespace(shape=(1,4,64,64)),'downscale_ratio_spacial':8},1.)
    assert f.state.encodings == 1 and f.state.unloads == 2


def test_selection_rejects_other_roots_and_changed_assets(api_fixture):
    f = api_fixture
    for name in ('../outside.safetensors','/absolute.safetensors','other\\file.safetensors'):
        with pytest.raises(ValueError):
            resolve_plain_lora(f.root,f.paths,name,1.)
    duplicate = f.root/'other.safetensors'
    duplicate.write_bytes(b'duplicate')
    original = f.paths.get_full_path_or_raise
    f.paths.get_full_path_or_raise = lambda kind,name:str(duplicate) if kind == 'loras' else original(kind,name)
    with pytest.raises(RuntimeError,match='profile root'):
        f.profile.prepare('soccer',RWB,.8)
    assert f.state.loads == f.state.encodings == 0
    f.paths.get_full_path_or_raise = original
    f.profile.prepare('soccer',RWB,.8)
    (f.root/'models'/'loras'/RWB).write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='identity changed'):
        f.profile.check()


def test_opt_in_node_prevents_cached_clone_reuse_without_changing_base_node(api_fixture):
    f = api_fixture
    node = node_classes(f.profile)['BenchKreaConditioningModel']
    assert set(node.INPUT_TYPES()['required']) == {'prompt','lora_name','lora_strength'}
    assert math.isnan(node.IS_CHANGED(prompt='soccer',lora_name=RWB,lora_strength=.8))
    f.profile.plain_loras = False
    base_node = node_classes(f.profile)['BenchKreaConditioningModel']
    assert set(base_node.INPUT_TYPES()['required']) == {'prompt'}
    assert not hasattr(base_node,'IS_CHANGED')


@pytest.mark.parametrize('failure_state',['untouched','changed_pointer','runtime_backup','loaded'])
@pytest.mark.parametrize('spectral',[False,True])
def test_preload_refusal_rolls_back_only_untouched_owned_seeds(api_fixture,monkeypatch,failure_state,spectral):
    f = api_fixture
    f.profile.prepare('soccer')
    f.profile.workbench_spectral = spectral
    # These external runtime boundaries must not execute or register a GPU model.
    for name in ('comfy','comfy.samplers','comfy_kitchen','comfy_kitchen.backends',
                 'comfy_kitchen.backends.mps'):
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules,name,module)
        if '.' in name:
            parent,child = name.rsplit('.',1)
            setattr(sys.modules[parent],child,module)
    predict = lambda *args:None
    native = lambda *args:None
    def sampler_factory(name):
        raise AssertionError('Sampler constructed before residency admission')
    sys.modules['comfy.samplers'].CFGGuider = SimpleNamespace(predict_noise=predict)
    sys.modules['comfy.samplers'].sampler_object = sampler_factory
    sys.modules['comfy_kitchen.backends.mps'].int8_linear = native
    @contextmanager
    def refuse(*args):
        assert f.base.backup
        if failure_state == 'changed_pointer':
            f.layer.weight = object()
        elif failure_state == 'runtime_backup':
            from krea_server import MappedBackup
            f.base.backup[f.key] = MappedBackup(object(),False)
        elif failure_state == 'loaded':
            f.base.loaded = 1
        raise ValueError('pre-load residency refusal')
        yield
    sys.modules['krea_readonly'].bounded_krea_residency = refuse
    # No LoadedModel was registered, so the real unload boundary has no work.
    f.profile.mm.unload_all_models = lambda:None
    with pytest.raises(ValueError,match='pre-load residency refusal'):
        f.profile.sample(f.base,42,8,1.,'euler','simple',[],[],
            {'samples':SimpleNamespace(shape=(1,4,64,64)),'downscale_ratio_spacial':8},1.)
    assert sys.modules['comfy.samplers'].CFGGuider.predict_noise is predict
    assert sys.modules['comfy.samplers'].sampler_object is sampler_factory
    assert sys.modules['comfy_kitchen.backends.mps'].int8_linear is native
    if failure_state == 'untouched':
        assert not f.base.backup and f.layer.weight is f.source
        assert f.profile.restore_failure is None
        assert f.profile.prepare('soccer')[0] is f.base
    else:
        assert f.base.backup
        with pytest.raises(RuntimeError,match='restart the dedicated server'):
            f.profile.prepare('soccer')
        assert f.profile.restore_failure is not None


def test_probe_requires_selected_asset_full_keys_and_fresh_preparation():
    asset = {'sha256':'a'*64,'policy':'plain','strength':.8}
    lora = {'name':RWB,'strength':.8,'asset':asset,'patch_keys':[f'key{i}' for i in range(224)]}
    sample = {'sample_index':2,'native_calls':1792,'forward_s':[1]*8,'finite_latent':True,
              'loader_count':1,'seed':42,'server_pid':1,'model_identity':2,'checkpoint_identity':[3],
              'asset_identities':{'vae':[4]},'sources':{'server':'a'},'lora':lora,
              'conditioning':{'lora':lora,'encoder_created':1,'encoder_refs_dead':True,
                  'conditioning':{'origin':'fresh-encoding','cache_policy':'off'}}}
    history = {'status':{'status_str':'success','completed':True,'messages':
                        [['execution_cached',{'nodes':['2','4']}]]},'outputs':{
               '3':{'krea_audit':[sample]},'5':{'krea_decode':[{'sample_index':2,'finite_image':True}]},
               '6':{'images':[{'filename':'image.png'}]}}}
    result = summarize(history,2,RWB,.8,42,expected_asset=asset,expected_keys=224)
    assert result['sample'] is sample
    for mutation in [lambda h:h['status']['messages'][0][1].update(nodes=['1','2','4']),
                     lambda h:h['outputs']['3']['krea_audit'][0]['lora'].update(strength=.4),
                     lambda h:h['outputs']['3']['krea_audit'][0]['lora'].update(patch_keys=['one']),
                     lambda h:h['outputs']['3']['krea_audit'][0]['conditioning'].update(encoder_refs_dead=False)]:
        changed = copy.deepcopy(history)
        mutation(changed)
        with pytest.raises(RuntimeError):
            summarize(changed,2,RWB,.8,42,expected_asset=asset,expected_keys=224)


def test_probe_output_paths_cannot_escape_the_profile(tmp_path):
    directory = tmp_path/'out'/'comfyui'
    directory.mkdir(parents=True)
    image = directory/'image.png'
    image.write_bytes(b'fixture')
    assert output_image(tmp_path,{'type':'output','filename':'image.png'}) == image
    outside = tmp_path/'outside.png'
    outside.write_bytes(b'fixture')
    for record in ({'type':'input','filename':'image.png'},
                   {'type':'output','filename':'../../outside.png'},
                   {'type':'output','filename':'outside.png','subfolder':'../..'}):
        with pytest.raises(ValueError):
            output_image(tmp_path,record)


def test_real_mapped_backups_restore_quantized_and_unpatched_bf16_parameters():
    script = r'''
import sys
import mmap
import tempfile
import warnings
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import torch._dynamo
torch.backends.mps.is_available = lambda:False
import comfy.ops as ops
import comfy.model_patcher as mp
import comfy.utils as utils
from comfy.quant_ops import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from krea_readonly import OWNER_ATTRIBUTE
from krea_adapter import _mapped_storage
from krea_server import capture_mapped_parameters,check_mapped_parameters,seed_mapped_backups,rollback_untouched_backups
CPU = torch.device('cpu')
with tempfile.TemporaryDirectory() as directory,torch.inference_mode():
    def mapped(name,tensor):
        path = Path(directory)/name
        path.write_bytes(tensor.view(torch.uint8).numpy().tobytes())
        stream = path.open('rb')
        mapping = mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)
        view = memoryview(mapping)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='The given buffer is not writable')
            result = torch.frombuffer(view,dtype=tensor.dtype).reshape(tensor.shape)
        setattr(result.untyped_storage(),OWNER_ATTRIBUTE,(mapping,view))
        stream.close()
        return result
    operations = ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    model = torch.nn.Module()
    model.add_module('diffusion_model',torch.nn.Module())
    quant = operations.Linear(256,64,bias=False,device=CPU)
    qdata = mapped('quant.bin',torch.ones(64,256,dtype=torch.int8))
    params = TensorWiseINT8Layout.Params(scale=torch.ones(64,1),orig_dtype=torch.bfloat16,
        orig_shape=(64,256),convrot=True,convrot_groupsize=256)
    quant.weight = torch.nn.Parameter(QuantizedTensor(qdata,'TensorWiseINT8Layout',params),requires_grad=False)
    quant.quant_format = 'int8_tensorwise'
    model.diffusion_model.add_module('quant',quant)
    global_layer = operations.Linear(12,1,bias=True,device=CPU)
    global_layer.weight = torch.nn.Parameter(mapped('global.bin',torch.ones(1,12,dtype=torch.bfloat16)),requires_grad=False)
    global_layer.bias = torch.nn.Parameter(mapped('bias.bin',torch.ones(1,dtype=torch.bfloat16)),requires_grad=False)
    model.diffusion_model.add_module('projector',global_layer)
    norm = torch.nn.LayerNorm(12,dtype=torch.bfloat16)
    norm.weight = torch.nn.Parameter(mapped('norm.bin',torch.ones(12,dtype=torch.bfloat16)),requires_grad=False)
    norm.bias = torch.nn.Parameter(mapped('norm_bias.bin',torch.zeros(12,dtype=torch.bfloat16)),requires_grad=False)
    model.diffusion_model.add_module('norm',norm)
    base = mp.ModelPatcher(model,CPU,CPU)
    originals = capture_mapped_parameters(base)
    assert len(originals) == 5
    initial = {key:_mapped_storage(weight) for key,weight in model.named_parameters()}
    assert {key:pointer for key,(_,pointer) in originals.items()} == initial
    for key,(snapshot,pointer) in originals.items():
        assert snapshot is not dict(model.named_parameters())[key]
        assert snapshot.device.type == 'cpu' and _mapped_storage(snapshot) == pointer
    quant_snapshot = originals['diffusion_model.quant.weight'][0]
    assert isinstance(quant_snapshot,QuantizedTensor)
    assert quant_snapshot._qdata.data_ptr() == qdata.data_ptr()
    for patcher in (base,base.clone()):
        seeded = seed_mapped_backups(patcher,originals)
        assert rollback_untouched_backups(patcher,originals,seeded)
        assert not patcher.backup
        seeded = seed_mapped_backups(patcher,originals)
        assert set(patcher.backup) == set(originals)
        for key,backup in patcher.backup.items():
            assert backup.weight is originals[key][0] and backup.inplace_update is False
        # CPU conversion exercises ordinary Module.to's in-place Parameter.data
        # reassignment for BF16, independently of our saved detached views.
        model.to(dtype=torch.float32)
        assert global_layer.weight.dtype == torch.float32
        assert originals['diffusion_model.projector.weight'][0].dtype == torch.bfloat16
        assert _mapped_storage(originals['diffusion_model.projector.weight'][0]) == initial['diffusion_model.projector.weight']
        # Model.to(CPU) would not migrate the quantized data in this CPU proof;
        # replace that live parameter with an anonymous equivalent as a stand-in.
        replacement = QuantizedTensor(torch.zeros(64,256,dtype=torch.int8),'TensorWiseINT8Layout',params)
        utils.set_attr_param(model,'diffusion_model.quant.weight',replacement)
        try:
            check_mapped_parameters(patcher,originals)
        except ValueError:
            pass
        else:
            raise AssertionError('Anonymous device-returned weights accepted as original mappings')
        assert not rollback_untouched_backups(patcher,originals,seeded)
        assert set(patcher.backup) == set(originals)
        patcher.unpatch_model(device_to=CPU)
        assert not patcher.backup and patcher.loaded_size() == 0
        check_mapped_parameters(patcher,originals)
        assert global_layer.weight.dtype == torch.bfloat16
        assert {key:_mapped_storage(weight) for key,weight in model.named_parameters()} == initial
'''
    result = subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[1],
                            capture_output=True,text=True,timeout=60)
    assert result.returncode == 0,result.stdout+result.stderr


def test_real_converted_auxiliaries_restore_and_reject_tampering_and_storage_limits():
    script = r'''
import sys,mmap,tempfile,warnings
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import torch._dynamo
torch.backends.mps.is_available = lambda:False
import comfy.model_patcher as mp
from krea_readonly import OWNER_ATTRIBUTE
from krea_adapter import _mapped_storage
from krea_server import (source_pointer,capture_mapped_parameters,check_mapped_parameters,
                         check_source_backups,seed_mapped_backups,rollback_untouched_backups,
                         MappedBackup)
CPU=torch.device('cpu')
with tempfile.TemporaryDirectory() as directory,torch.inference_mode():
    path=Path(directory)/'weight.bin'
    path.write_bytes(torch.ones(1,12,dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes())
    with path.open('rb') as stream:
        mapping=mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)
    view=memoryview(mapping)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='The given buffer is not writable')
        weight=torch.frombuffer(view,dtype=torch.bfloat16).reshape(1,12)
    setattr(weight.untyped_storage(),OWNER_ATTRIBUTE,(mapping,view))
    model=torch.nn.Module()
    model.add_module('diffusion_model',torch.nn.Module())
    linear=torch.nn.Linear(12,1,dtype=torch.bfloat16)
    linear.weight=torch.nn.Parameter(weight,requires_grad=False)
    linear.bias=torch.nn.Parameter(torch.ones(1,dtype=torch.bfloat16),requires_grad=False)
    model.diffusion_model.add_module('projector',linear)
    for name,parameter_name,shape in (('norm','scale',(12,)),('mod','lin',(72,)),
                                     ('modulation','lin',(2,12))):
        module=torch.nn.Module()
        module.register_parameter(parameter_name,torch.nn.Parameter(torch.ones(shape,dtype=torch.float16),requires_grad=False))
        model.diffusion_model.add_module(name,module)
    base=mp.ModelPatcher(model,CPU,CPU)
    originals=capture_mapped_parameters(base)
    assert len(originals)==5
    mapped_key='diffusion_model.projector.weight'
    owned={key:value for key,value in originals.items() if key!=mapped_key}
    assert len(owned)==4
    for key,(snapshot,pointer) in owned.items():
        assert snapshot.untyped_storage().data_ptr()==pointer
        assert snapshot._krea_owned_expected.untyped_storage().data_ptr()!=pointer
        try:_mapped_storage(snapshot)
        except ValueError:pass
        else:raise AssertionError('Converted auxiliary incorrectly claims mapped storage')
    for candidate in (base,base.clone()):
        seeded=seed_mapped_backups(candidate,originals)
        check_source_backups(candidate,originals)
        model.to(dtype=torch.float32)
        assert all(snapshot.device.type=='cpu' for snapshot,_ in originals.values())
        check_source_backups(candidate,originals)
        assert not rollback_untouched_backups(candidate,originals,seeded)
        candidate.unpatch_model(CPU)
        check_mapped_parameters(candidate,originals)
        assert linear.weight.dtype==torch.bfloat16 and linear.bias.dtype==torch.bfloat16
        assert model.diffusion_model.mod.lin.dtype==torch.float16
    seeded=seed_mapped_backups(base,originals)
    bias_key='diffusion_model.projector.bias'
    snapshot=originals[bias_key][0]
    linear.bias.add_(1)
    for check in (lambda:check_mapped_parameters(base,originals),
                  lambda:check_source_backups(base,originals)):
        try:check()
        except RuntimeError:pass
        else:raise AssertionError('Mutable auxiliary value drift accepted')
    assert not rollback_untouched_backups(base,originals,seeded)
    snapshot.copy_(snapshot._krea_owned_expected)
    check_mapped_parameters(base,originals)
    for replacement in (MappedBackup(snapshot.clone(),False),MappedBackup(snapshot,True)):
        base.backup[bias_key]=replacement
        try:check_source_backups(base,originals)
        except RuntimeError:pass
        else:raise AssertionError('Foreign or in-place source backup accepted')
    base.backup[bias_key]=seeded[bias_key]
    base.backup['unknown.bias']=MappedBackup(torch.ones(1),False)
    try:check_source_backups(base,originals)
    except RuntimeError:pass
    else:raise AssertionError('Unknown auxiliary backup accepted')
    base.backup.pop('unknown.bias')
    assert rollback_untouched_backups(base,originals,seeded)
    bad=[('diffusion_model.projector.weight',torch.ones(1,12)),
         ('diffusion_model.norm.weight',torch.ones(12)),
         ('diffusion_model.norm.scale',torch.tensor([float('nan')])),
         ('diffusion_model.norm.scale',torch.ones(12,dtype=torch.int32)),
         ('diffusion_model.norm.scale',torch.ones(2**20//4+1)),
         # Logical size is four bytes, but detach would retain over 1 MiB.
         ('diffusion_model.projector.bias',torch.ones(2**20//4+1)[:1])]
    for key,tensor in bad:
        try:source_pointer(tensor,key)
        except ValueError:pass
        else:raise AssertionError('Unsupported source or oversized retained storage admitted: '+key)
    aggregate=torch.nn.Module()
    for index in range(9):
        module=torch.nn.Module()
        module.register_parameter('scale',torch.nn.Parameter(torch.ones(2**19,dtype=torch.float16),requires_grad=False))
        aggregate.add_module('norm'+str(index),module)
    try:capture_mapped_parameters(mp.ModelPatcher(aggregate,CPU,CPU))
    except ValueError as error:assert '8 MiB' in str(error)
    else:raise AssertionError('Aggregate retained auxiliary storage bound not enforced')
'''
    result = subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[1],
                            capture_output=True,text=True,timeout=60)
    assert result.returncode == 0,result.stdout+result.stderr


def test_original_mapped_auxiliary_cannot_fall_back_after_losing_readonly_owner():
    script = r'''
import sys,mmap,tempfile,warnings
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import torch
import torch._dynamo
torch.backends.mps.is_available=lambda:False
import comfy.model_patcher as mp
from krea_adapter import _mapped_storage
from krea_readonly import OWNER_ATTRIBUTE
from krea_server import capture_mapped_parameters,check_mapped_parameters,check_source_backups,seed_mapped_backups
CPU=torch.device('cpu')
with tempfile.TemporaryDirectory() as directory,torch.inference_mode():
    path=Path(directory)/'bias.bin'
    path.write_bytes(torch.ones(2,dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes())
    with path.open('rb') as stream:mapping=mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)
    view=memoryview(mapping)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='The given buffer is not writable')
        tensor=torch.frombuffer(view,dtype=torch.bfloat16)
    storage=tensor.untyped_storage()
    setattr(storage,OWNER_ATTRIBUTE,(mapping,view))
    model=torch.nn.Module()
    model.add_module('diffusion_model',torch.nn.Module())
    model.diffusion_model.add_module('probe',torch.nn.Module())
    model.diffusion_model.probe.register_parameter('bias',torch.nn.Parameter(tensor,requires_grad=False))
    bias=model.diffusion_model.probe.bias
    key='diffusion_model.probe.bias'
    patcher=mp.ModelPatcher(model,CPU,CPU)
    originals=capture_mapped_parameters(patcher)
    assert getattr(originals[key][0],'_krea_owned_expected',None) is None
    seed_mapped_backups(patcher,originals)
    before=bias.untyped_storage().data_ptr()
    delattr(storage,OWNER_ATTRIBUTE)
    assert bias.untyped_storage().data_ptr()==before
    try:_mapped_storage(bias)
    except ValueError:pass
    else:raise AssertionError('Readonly owner removal did not reach the boundary')
    for check in (lambda:check_mapped_parameters(patcher,originals),
                  lambda:check_source_backups(patcher,originals)):
        try:check()
        except (ValueError,RuntimeError):pass
        else:raise AssertionError('Originally mapped auxiliary silently became an unguarded owned source')
'''
    result = subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[1],
                            capture_output=True,text=True,timeout=60)
    assert result.returncode == 0,result.stdout+result.stderr
