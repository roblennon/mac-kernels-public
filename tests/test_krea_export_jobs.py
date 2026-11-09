"""Export HTTP/state contract across a real CPU child and real safetensors install."""
import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from krea_checkpoints import CheckpointCatalog, DEFAULT_CHECKPOINT, EXPORT_METADATA, file_identity
from krea_export_jobs import ExportJobs


@pytest.fixture
def exports(tmp_path, monkeypatch):
    root = tmp_path / 'profile'
    models = root / 'models/diffusion_models'
    models.mkdir(parents=True)
    (root / 'models/loras').mkdir()
    storage = tmp_path / 'external'
    monkeypatch.setenv('MK_ROOT', str(storage))
    state = {'first.weight': torch.ones((4, 16), dtype=torch.bfloat16)}
    reference = models / DEFAULT_CHECKPOINT
    save_file(state, reference)
    source = storage / 'sources/krea-turbo/diffusion_models/source.safetensors'
    source.parent.mkdir(parents=True)
    provenance = {'schema': 1, 'architecture': 'krea2', 'base_variant': 'turbo',
                  'quantization': 'int8-convrot-group256', 'baked_loras': []}
    save_file(state, source, metadata={EXPORT_METADATA: json.dumps(provenance)})
    folders = SimpleNamespace(get_full_path_or_raise=lambda kind, name: str(root / 'models' / kind / name))
    mm = SimpleNamespace(**{name: lambda: None for name in
        ('unload_all_models', 'soft_empty_cache', 'cleanup_models_gc')})
    profile = SimpleNamespace(root=root, folder_paths=folders, checkpoints=CheckpointCatalog(root, folders),
                              check=lambda: None, emit=lambda *args: None, mm=mm, torch=torch, base_model=None, model=None)
    queue = SimpleNamespace(get_current_queue=lambda: ([], []), set_flag=lambda *args: None)
    jobs = ExportJobs(profile, queue)
    monkeypatch.setattr(jobs, 'group_rss', lambda: 256 * 2**20)
    jobs.phases = []
    async def audit_phase(phase, **details):
        jobs.phases.append(phase)
    monkeypatch.setattr(jobs, 'audit_phase', audit_phase)
    jobs.sources = {'source': {'path': str(source), 'file_identity': file_identity(source)}}
    # Real child boundary substitutes only expensive model conversion, which has
    # separate pinned-Comfy math proof. State/HTTP/install paths remain real.
    jobs.command = lambda src, stack, target: [sys.executable, '-c',
        'import shutil,sys,time; time.sleep(.2); shutil.copyfile(sys.argv[1],sys.argv[2]); print("{}")',
        src['path'], str(target)]
    return jobs


def payload(name='new.safetensors'):
    return {'source': 'source', 'name': name, 'loras': [], 'turbo': True}


async def client_for(jobs):
    @web.middleware
    async def admission(request, handler):
        if request.path.startswith('/krea-workbench/export'):
            return await jobs.handle(request)
        async with jobs.lock:
            if jobs.active:
                return web.json_response({'error': 'export active'}, status=409)
            return await handler(request)
    app = web.Application(middlewares=[admission])
    async def prompt(_request):
        return web.json_response({'accepted': True})
    app.router.add_post('/prompt', prompt)
    app.on_cleanup.append(jobs.close)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def test_http_export_publishes_adds_only_new_entry_and_blocks_overlap(exports):
    async def scenario():
        client = await client_for(exports)
        try:
            original = dict(exports.profile.checkpoints.entries)
            foreign = await client.post('/krea-workbench/export', json=payload(), headers={'Host': 'foreign.test'})
            assert foreign.status == 400 and exports.job is None
            denied = await client.post('/krea-workbench/export', json=payload(), headers={'Origin': 'https://evil.test'})
            assert denied.status == 400 and exports.job is None
            response = await client.post('/krea-workbench/export', json=payload())
            assert response.status == 202
            assert (await client.post('/prompt', json={})).status == 409
            assert (await client.post('/krea-workbench/export', json=payload('other.safetensors'))).status == 409
            await exports.task
            data = await (await client.get('/krea-workbench/export')).json()
            assert data['job']['state'] == 'complete'
            assert exports.phases == ['request', 'admitted', 'release_entry', 'unloaded',
                                     'restoration_checked', 'refs_cleared', 'cache_released']
            assert exports.profile.checkpoints.resolve('new.safetensors')['provenance']['base_variant'] == 'turbo'
            assert all(exports.profile.checkpoints.entries[k] == v for k, v in original.items())
            assert exports.output.joinpath('new.safetensors').exists()
            assert not list(exports.profile.checkpoints.directory.glob('.krea-install-*'))
            assert (await client.post('/prompt', json={})).status == 200
            assert (await client.post('/krea-workbench/export', json=payload())).status == 400
        finally:
            await client.close()
    asyncio.run(scenario())


def test_queue_invalid_inputs_and_cancel_leave_no_installed_partial(exports):
    async def scenario():
        client = await client_for(exports)
        try:
            exports.queue.get_current_queue = lambda: ([], [['queued']])
            assert (await client.post('/krea-workbench/export', json=payload())).status == 409
            exports.queue.get_current_queue = lambda: ([], [])
            for change in ({'turbo': False}, {'source': '/etc/passwd'}, {'name': '../escape.safetensors'},
                           {'loras': [{'name': 'x.safetensors', 'strength': 10**100}]},
                           {'loras': [{'name': 'krea2_identity_edit_v1_2_r64.safetensors', 'strength': 1}]}):
                assert (await client.post('/krea-workbench/export', json={**payload(), **change})).status == 400
            exports.command = lambda *args: [sys.executable, '-c', 'import time; time.sleep(30)']
            started = await (await client.post('/krea-workbench/export', json=payload())).json()
            assert (await client.post('/krea-workbench/export-cancel', json={'id': 'wrong'})).status == 400
            assert (await client.post('/krea-workbench/export-cancel', json={'id': started['job']['id']})).status == 202
            await exports.task
            assert exports.job['state'] == 'cancelled' and exports.process is None
            assert not (exports.profile.checkpoints.directory / 'new.safetensors').exists()
            assert (await client.post('/prompt', json={})).status == 200
        finally:
            await client.close()
    asyncio.run(scenario())


def test_child_failure_and_install_race_preserve_existing_file(exports):
    async def scenario():
        client = await client_for(exports)
        try:
            exports.command = lambda *args: [sys.executable, '-c', 'import sys; print("conversion rejected",file=sys.stderr); sys.exit(2)']
            assert (await client.post('/krea-workbench/export', json=payload())).status == 202
            await exports.task
            assert exports.job['state'] == 'failed' and 'conversion rejected' in exports.job['error']
            target = exports.profile.checkpoints.directory / 'collision.safetensors'
            src = Path(exports.sources['source']['path'])
            external = exports.output / 'collision.safetensors'
            shutil.copyfile(src, external)
            target.write_bytes(b'human file')
            with pytest.raises(FileExistsError):
                await asyncio.to_thread(exports.install, external, target, exports.sources['source'], [])
            assert target.read_bytes() == b'human file' and external.exists()
            assert not list(target.parent.glob('.krea-install-*'))
        finally:
            await client.close()
    asyncio.run(scenario())


def test_frozen_source_mutation_rejected_before_child(exports):
    Path(exports.sources['source']['path']).touch()
    with pytest.raises(ValueError, match='changed'):
        exports.prepare(payload())
    assert exports.task is None


def test_frontend_frozen_stack_selection_and_changed_stack_preservation():
    module = (Path(__file__).resolve().parents[1] / 'bench/web/export_helpers.mjs').as_uri()
    script = '''
import assert from 'node:assert/strict';
import {captureRecipe,refreshChoices,useExport,isExportActive,exportMessage} from MODULE;
assert.equal(isExportActive({state:'cleaning'}),true);
assert.equal(isExportActive({state:'failed'}),false);
assert.equal(exportMessage({error:'Conversion failed',progress:'File retained'}),'Conversion failed\\nFile retained');
const node={type:'BenchKreaWorkbench',widgets:[
 {name:'checkpoint',value:'base',options:{values:['base']}},
 {name:'lora_1',value:'character',options:{}},{name:'strength_1',value:.8},
 {name:'lora_2',value:'style'},{name:'strength_2',value:-.2},
 {name:'prompt',value:'keep me'},{name:'mode',value:'identity_edit'},
 {name:'spectral',value:true},{name:'size',value:'portrait'},
 {name:'lora_3',value:'character'},{name:'strength_3',value:0}]};
const transactions=[];
node.graph={beforeChange:()=>transactions.push('before'),
 afterChange:()=>transactions.push(['after',captureRecipe(node)])};
const recipe=captureRecipe(node);
assert.deepEqual(recipe.loras,[{name:'character',strength:.8},{name:'style',strength:-.2}]);
refreshChoices([node],['base','custom']);
assert.deepEqual(node.widgets[0].options.values,['base','custom']);
node.widgets[2].value=1;
const edited=structuredClone(node.widgets);
assert.equal(useExport(node,'custom',recipe),false);
assert.deepEqual(node.widgets,edited);
node.widgets[2].value=.8;
node.widgets[0].value='another checkpoint';
const switched=structuredClone(node.widgets);
assert.equal(useExport(node,'custom',recipe),false);
assert.deepEqual(node.widgets,switched);
assert.deepEqual(transactions,[]);
node.widgets[0].value='base';
assert.equal(useExport(node,'custom',recipe),true);
assert.equal(node.widgets[0].value,'custom');
assert.equal(node.widgets[1].value,'');assert.equal(node.widgets[3].value,'');
assert.equal(node.widgets[2].value,.8);assert.equal(node.widgets[5].value,'keep me');
assert.equal(node.widgets[6].value,'identity_edit');assert.equal(node.widgets[7].value,true);
assert.equal(node.widgets[9].value,'character');assert.equal(node.widgets[10].value,0);
assert.deepEqual(transactions,['before',['after',{checkpoint:'custom',loras:[]}]]);
'''.replace('MODULE', json.dumps(module))
    result = subprocess.run([shutil.which('node'), '--input-type=module', '-e', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('failure', ['telemetry', 'memory'])
def test_resource_guard_stops_actual_child_before_install(exports, monkeypatch, failure):
    import psutil
    class Observation:
        def __init__(self, _pid):
            pass
        def memory_info(self):
            if failure == 'telemetry':
                raise psutil.AccessDenied()
            return SimpleNamespace(rss=9 * 2**30)
    monkeypatch.setattr(psutil, 'Process', Observation)
    exports.command = lambda *args: [sys.executable, '-c', 'import time; time.sleep(30)']
    async def scenario():
        client = await client_for(exports)
        try:
            assert (await client.post('/krea-workbench/export', json=payload())).status == 202
            await asyncio.wait_for(exports.task, 5)
            assert exports.job['state'] == 'failed'
            assert exports.process is None
            assert not (exports.profile.checkpoints.directory / 'new.safetensors').exists()
            assert not exports.active
        finally:
            await client.close()
    asyncio.run(scenario())


def test_chunked_receipt_reads_to_eof_and_logging_failure_is_nonfatal(exports):
    exports.command = lambda src, stack, target: [sys.executable, '-c',
        'import shutil,sys,time,json; shutil.copyfile(sys.argv[1],sys.argv[2]); '
        'data=json.dumps({"details":"x"*50000}); '
        '[(sys.stdout.write(data[i:i+4096]),sys.stdout.flush(),time.sleep(.01)) for i in range(0,len(data),4096)]',
        src['path'], str(target)]
    def broken_log(*args):
        raise BrokenPipeError('log closed')
    exports.profile.emit = broken_log
    async def scenario():
        client = await client_for(exports)
        try:
            assert (await client.post('/krea-workbench/export', json=payload())).status == 202
            await exports.task
            assert exports.job['state'] == 'complete'
            exports.profile.checkpoints.resolve('new.safetensors')
        finally:
            await client.close()
    asyncio.run(scenario())


def test_failure_admission_stays_closed_until_actual_child_reaped(exports):
    exports.command = lambda *args: [sys.executable, '-c',
        'import sys,time; sys.stdout.write("x"*(2**20+1)); sys.stdout.flush(); time.sleep(30)']
    async def scenario():
        client = await client_for(exports)
        entered, release = asyncio.Event(), asyncio.Event()
        real_stop = exports.stop_child
        async def held_stop(process):
            entered.set()
            await release.wait()
            await real_stop(process)
        exports.stop_child = held_stop
        try:
            assert (await client.post('/krea-workbench/export', json=payload())).status == 202
            await asyncio.wait_for(entered.wait(), 5)
            child = exports.process
            assert child is not None and child.returncode is None
            assert exports.job['state'] == 'cleaning' and exports.active
            assert (await client.post('/krea-workbench/export', json=payload('second.safetensors'))).status == 409
            assert (await client.post('/prompt', json={})).status == 409
            release.set()
            await exports.task
            assert child.returncode is not None and exports.process is None
            assert exports.job['state'] == 'failed' and not exports.active
            exports.stop_child = real_stop
            exports.command = lambda src, stack, target: [sys.executable, '-c',
                'import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2]); print("{}")', src['path'], str(target)]
            assert (await client.post('/krea-workbench/export', json=payload('second.safetensors'))).status == 202
            await exports.task
            assert exports.job['state'] == 'complete' and exports.process is None
        finally:
            release.set()
            await client.close()
    asyncio.run(scenario())


def test_cancel_after_external_publication_reports_retained_file(exports):
    exports.command = lambda src, stack, target: [sys.executable, '-c',
        'import shutil,sys,time; shutil.copyfile(sys.argv[1],sys.argv[2]); print("{}",flush=True); time.sleep(30)',
        src['path'], str(target)]
    async def scenario():
        client = await client_for(exports)
        try:
            first = await (await client.post('/krea-workbench/export', json=payload())).json()
            for _ in range(100):
                if (exports.output / 'new.safetensors').exists():
                    break
                await asyncio.sleep(.02)
            assert (exports.output / 'new.safetensors').exists()
            assert (await client.post('/krea-workbench/export-cancel', json={'id': first['job']['id']})).status == 202
            await exports.task
            assert exports.job['state'] == 'cancelled'
            assert exports.job['retained_file'] == 'new.safetensors'
            assert 'retained' in exports.job['progress']
            assert not (exports.profile.checkpoints.directory / 'new.safetensors').exists()
        finally:
            await client.close()
    asyncio.run(scenario())


def test_output_directory_must_be_separate_and_group_budget_stops_child(exports):
    old_output = exports.output
    exports.output = exports.profile.checkpoints.directory.resolve()
    with pytest.raises(ValueError, match='separate folder'):
        exports.prepare(payload())
    exports.output = old_output
    readings = iter([256 * 2**20, 9.5 * 2**30])
    exports.group_rss = lambda: next(readings)
    exports.command = lambda *args: [sys.executable, '-c', 'import time; time.sleep(30)']
    async def scenario():
        client = await client_for(exports)
        try:
            assert (await client.post('/krea-workbench/export', json=payload())).status == 202
            await asyncio.wait_for(exports.task, 5)
            assert exports.job['state'] == 'failed' and 'limit' in exports.job['error']
            assert exports.job['peak_group_rss_bytes'] > 9 * 2**30
            assert exports.process is None
        finally:
            await client.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('late_waiter', [False, True])
def test_real_classic_cache_and_prompt_queue_release_weakrefs_before_export(tmp_path, late_waiter):
    script = r'''
import asyncio,gc,sys,threading,time,weakref
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()/'bench'))
from probe_text_encoder_load import bootstrap_cpu
bootstrap_cpu()
import execution
import torch
from krea_export_jobs import ExportJobs
class Resource: pass
p=SimpleNamespace(root=Path(sys.argv[-1]),checkpoints=SimpleNamespace(directory=Path('.')),
 mm=SimpleNamespace(unload_all_models=lambda:None,soft_empty_cache=lambda:None,cleanup_models_gc=lambda:None),
 check=lambda:None,emit=lambda *args:None,torch=torch,base_model=None,model=Resource(),workbench_vae=Resource())
e=execution.PromptExecutor(SimpleNamespace(),asset_manager=SimpleNamespace())
# Real classic cache storage owns node MODEL and VAE outputs.
e.caches.outputs.cache['workbench']=(p.model,p.workbench_vae)
refs=[weakref.ref(p.model),weakref.ref(p.workbench_vae)]
q=execution.PromptQueue(SimpleNamespace(queue_updated=lambda:None))
gate=threading.Event()
def worker_tick():
 gate.wait()
 q.get(timeout=1000.0)
 flags=q.get_flags()
 if flags.get('free_memory'):
  e.reset()
  gc.collect()
t=threading.Thread(target=worker_tick,daemon=True);t.start()
jobs=ExportJobs(p,q);jobs.job={};jobs.group_rss=lambda:256*2**20
async def no_audit(*args,**kwargs): pass
jobs.audit_phase=no_audit
if not LATE_WAITER:
 gate.set()
 for _ in range(100):
  with q.not_empty:
   ready=bool(q.not_empty._waiters)
  if ready: break
  time.sleep(.01)
 assert ready and all(ref() is not None for ref in refs)
 asyncio.run(jobs.release_model())
else:
 async def late_worker():
  task=asyncio.create_task(jobs.release_model())
  for _ in range(200):
   if q.get_flags(reset=False).get('free_memory') is True: break
   await asyncio.sleep(.01)
  assert q.get_flags(reset=False).get('free_memory') is True
  assert all(ref() is not None for ref in refs)
  # First notify happened before this worker entered q.get: it was lost.
  gate.set()
  await asyncio.wait_for(task,2)
 asyncio.run(late_worker())
t.join(timeout=2)
assert not t.is_alive() and all(ref() is None for ref in refs)
assert jobs.job['peak_group_rss_bytes']==256*2**20
'''
    # bootstrap rewrites argv, so use a script literal for this boundary path.
    script = script.replace('LATE_WAITER', repr(late_waiter))
    script = script.replace('Path(sys.argv[-1])', 'Path(' + repr(str(tmp_path)) + ')')
    env = __import__('os').environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'vendor/comfy-kitchen')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=env, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
