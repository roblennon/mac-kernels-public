"""One owned CPU export at a time, sharing admission with the Comfy prompt queue."""
import asyncio
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import uuid
import weakref
from urllib.parse import urlsplit

from aiohttp import web
from krea_checkpoints import DEFAULT_CHECKPOINT, file_identity, safe_name, validate_custom_checkpoint

ACTIVE = {'preparing', 'exporting', 'installing', 'cancelling', 'cleaning'}


def same_origin(request):
    """Reject browser cross-site mutation, including opaque/null Origins."""
    host = urlsplit('http://' + request.host)
    local_port = request.transport.get_extra_info('sockname')[1]
    if host.hostname not in ('localhost', '127.0.0.1', '::1') or host.port != local_port:
        raise ValueError('Export requests require the local Workbench host and port')
    origin = request.headers.get('Origin')
    if request.headers.get('Sec-Fetch-Site') == 'cross-site':
        raise ValueError('Export requests must come from this Workbench')
    if origin is not None:
        parsed = urlsplit(origin)
        if (parsed.scheme != request.scheme or parsed.netloc != request.host
                or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
            raise ValueError('Export requests must come from this Workbench')


class ExportJobs:
    def __init__(self, profile, queue):
        self.profile, self.queue = profile, queue
        self.lock = asyncio.Lock()
        self.job = None
        self.task = None
        self.process = None
        self.cancelled = False
        self.sources = {}
        # Dedicated profiles use an external out symlink. Config is optional and
        # remains server-side; requests can select IDs, never paths.
        storage = Path(os.environ.get('MK_ROOT') or (profile.root / 'out').resolve().parent).expanduser().resolve()
        self.output = Path(os.environ.get('KREA_EXPORT_DIRECTORY') or storage / 'models/diffusion_models').expanduser().resolve()
        self.source_dirs = (profile.checkpoints.directory, storage / 'models/diffusion_models',
                            storage / 'sources/krea-turbo/diffusion_models')

    @property
    def active(self):
        return self.job is not None and self.job['state'] in ACTIVE

    def snapshot(self):
        return {'job': dict(self.job) if self.job else None, 'checkpoints': self.profile.checkpoints.names()}

    async def audit_phase(self, phase, **details):
        """Small transition observations; diagnostics never change admission."""
        def observe():
            from gpu_runner import group_rows, system_state
            return {'group_rss_bytes': sum(row['rss_kib'] for row in group_rows(os.getpgrp())) * 1024,
                    'memory': system_state()}
        try:
            observation = await asyncio.to_thread(observe)
        except Exception as error:
            observation = {'telemetry_error': type(error).__name__}
        sources = getattr(self.profile, 'sources', {})
        payload = {'phase': phase, 'monotonic_s': round(time.monotonic(), 3),
                   'checkpoint': getattr(self.profile, 'checkpoint_name', None),
                   'source_snapshot_sha256': hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                   **observation, **details}
        try:
            self.profile.emit('export_phase', payload)
        except Exception:
            pass

    def catalog(self):
        from krea_export import validate_source
        from safetensors import SafetensorError
        found, seen = {}, set()
        for directory in self.source_dirs:
            for path in sorted(directory.glob('*.safetensors')):
                try:
                    path = path.resolve(strict=True)
                except OSError:
                    continue
                if path.parent != directory.resolve():
                    continue
                if path in seen:
                    continue
                seen.add(path)
                try:
                    source, _ = validate_source(path, self.profile.checkpoints.reference)
                    # Quantized lineage must already be known from original or a
                    # validated export. Floating files require user attestation.
                    if source['quantized'] and path != self.profile.checkpoints.reference.resolve():
                        validate_custom_checkpoint(path, self.profile.checkpoints.reference)
                    key = hashlib.sha256(str(path).encode()).hexdigest()[:24]
                    found[key] = {'path': str(path), 'file_identity': source['file_identity'],
                                  'name': path.name, 'floating': not source['quantized'],
                                  'location': 'profile' if directory == self.source_dirs[0] else 'external'}
                except (OSError, ValueError, RuntimeError, TypeError, KeyError, SafetensorError):
                    continue
        self.sources = found
        return [{'id': k, 'name': v['name'], 'floating': v['floating'], 'location': v['location']} for k, v in found.items()]

    def prepare(self, body):
        if not isinstance(body, dict) or set(body) != {'source', 'name', 'loras', 'turbo'}:
            raise ValueError('Expected source, name, frozen LoRA stack and Turbo declaration')
        source = self.sources.get(body['source']) if isinstance(body['source'], str) else None
        if source is None or file_identity(source['path']) != source['file_identity']:
            raise ValueError('Export source changed or is unknown; reopen the export dialog')
        if body['turbo'] is not True:
            raise ValueError('Confirm that the source is already Turbo; export does not distill RAW')
        name = body['name']
        if (not safe_name(name) or len(name) > 120 or
                not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.safetensors', name)):
            raise ValueError('Use a new checkpoint name with letters, numbers, dashes or underscores')
        if name == DEFAULT_CHECKPOINT or name in self.profile.checkpoints.entries:
            raise ValueError('Checkpoint name already exists')
        stack = body['loras']
        if not isinstance(stack, list) or len(stack) > 5:
            raise ValueError('At most five ordinary LoRAs can be baked')
        from krea_server import resolve_plain_lora
        from krea_export import SPECIAL_NAMES
        frozen = []
        for item in stack:
            if not isinstance(item, dict) or set(item) != {'name', 'strength'}:
                raise ValueError('Invalid frozen LoRA selection')
            strength = item['strength']
            if type(strength) not in (float, int) or not 0 < abs(strength) <= 2 or not math.isfinite(strength):
                raise ValueError('Baked LoRA strength must be finite, nonzero and within ±2')
            if item['name'] in SPECIAL_NAMES:
                raise ValueError('Identity, Style Reference and acceleration adapters remain runtime options')
            path = resolve_plain_lora(self.profile.root, self.profile.folder_paths, item['name'], strength)
            if path is None:
                raise ValueError('Missing baked LoRA name')
            frozen.append({**item, 'path': str(path), 'file_identity': file_identity(path)})
        if len({x['path'] for x in frozen}) != len(frozen):
            raise ValueError('Duplicate LoRA selection')
        self.output.mkdir(parents=True, exist_ok=True)
        target = self.profile.checkpoints.directory.resolve(strict=True) / name
        external = self.output / name
        if self.output == target.parent:
            raise ValueError('Export staging must use a separate folder from installed checkpoints')
        if target.exists() or target.is_symlink() or external.exists() or external.is_symlink():
            raise ValueError('Checkpoint name already exists on disk')
        # The native target size, plus metadata slack. Export spools and candidate
        # coexist; install uses one full temporary copy and atomic hard-link.
        size = self.profile.checkpoints.reference.stat().st_size + 2**20
        same_disk = self.output.stat().st_dev == target.parent.stat().st_dev
        needed = (3 if same_disk else 2) * size + 512 * 2**20
        if shutil.disk_usage(self.output).free < needed or shutil.disk_usage(target.parent).free < size + 512 * 2**20:
            raise ValueError('Insufficient disk space for export staging and checkpoint installation')
        return dict(source), frozen, external, target

    async def release_model(self):
        from krea_server import check_mapped_parameters
        p = self.profile
        refs = [weakref.ref(x) for x in (p.base_model, p.model, getattr(p, 'workbench_vae', None)) if x is not None]
        await self.audit_phase('release_entry', tracked_references=len(refs))
        p.check()
        # Mapped weights are inference tensors. Quantized Linear._apply clones
        # them outside this context even for CPU-to-CPU offload, losing the mapping.
        # Keep the synchronous unload scoped; never carry thread-local mode over await.
        with p.torch.inference_mode():
            p.mm.unload_all_models()
            p.mm.soft_empty_cache()
        await self.audit_phase('unloaded', tracked_references=len(refs))
        if p.base_model is not None:
            try:
                if (p.base_model.patches or p.base_model.backup or p.base_model.loaded_size()
                        or (p.model is not None and (p.model.backup or p.model.loaded_size()))):
                    raise RuntimeError('Workbench base was not restored before export')
                check_mapped_parameters(p.base_model, p.original_parameters)
            except Exception as error:
                p.restore_failure = str(error)
                p.check()
        await self.audit_phase('restoration_checked', tracked_references=len(refs))
        p.base_model = p.model = p.loader = p.original_parameters = None
        p.conditioning = p.active_contract = p.adapter_receipt = p.lora = None
        p.workbench_vae = p.prepared_reference = None
        gc.collect()
        p.mm.cleanup_models_gc()
        p.mm.soft_empty_cache()
        await self.audit_phase('refs_cleared', retained_references=sum(ref() is not None for ref in refs))
        # The classic executor cache retains Workbench model/VAE outputs even
        # after profile references are gone. The worker owns/reset its cache.
        self.queue.set_flag('free_memory', True)
        deadline = time.monotonic() + 10
        while any(ref() is not None for ref in refs):
            if time.monotonic() >= deadline:
                raise web.HTTPConflict(text='Workbench cached models could not be released; restart before exporting')
            # The worker can be between get_flags() and its next queue wait:
            # re-notify until its cache actually releases the tracked objects.
            self.queue.set_flag('free_memory', True)
            await asyncio.sleep(.1)
            gc.collect()
        await self.audit_phase('cache_released', tracked_references=len(refs), all_released=True)
        p.check()
        group_rss = await asyncio.to_thread(self.group_rss)
        if group_rss >= 9 * 2**30:
            raise web.HTTPConflict(text='Workbench process group exceeds the safe CPU export admission budget')
        self.job['peak_group_rss_bytes'] = group_rss
        try:
            p.emit('export_model_release', {'tracked_references': len(refs), 'all_released': True,
                    'group_baseline_rss_bytes': group_rss, 'group_limit_rss_bytes': 9 * 2**30})
        except Exception:
            pass

    @staticmethod
    def group_rss():
        from gpu_runner import group_rows
        rows = group_rows(os.getpgrp())
        if not rows:
            raise RuntimeError('CPU export process-group observation is unavailable')
        return sum(row['rss_kib'] for row in rows) * 1024

    async def start(self, body):
        if self.active or any(self.queue.get_current_queue()):
            raise web.HTTPConflict(text='Finish queued generation or the current export first')
        await self.audit_phase('request', requested_name=str(body.get('name', ''))[:120] if isinstance(body, dict) else None)
        self.profile.check()
        source, stack, external, target = await asyncio.to_thread(self.prepare, body)
        await self.audit_phase('admitted', source_name=Path(source['path']).name,
                               source_identity=source['file_identity'], target_name=target.name)
        # Queue admission lock is held by caller throughout validation.
        self.cancelled = False
        self.job = {'id': uuid.uuid4().hex, 'state': 'preparing', 'name': target.name,
                    'source': source['name'] if 'name' in source else Path(source['path']).name,
                    'loras': [{'name': x['name'], 'strength': x['strength']} for x in stack],
                    'progress': 'Starting CPU export', 'error': None,
                    'peak_rss_bytes': 0, 'elapsed_seconds': 0}
        try:
            await self.release_model()
        except Exception:
            self.job.update(state='failed', error='Workbench export admission failed')
            raise
        self.task = asyncio.create_task(self.run(source, stack, external, target))
        return self.snapshot()

    async def stop_child(self, process):
        if process is not None and process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 10)
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()

    async def watch(self, process, job):
        import psutil
        started = time.monotonic()
        while process.returncode is None:
            await asyncio.sleep(.5)
            if self.cancelled:
                await self.stop_child(process)
                break
            try:
                rss = psutil.Process(process.pid).memory_info().rss
            except psutil.NoSuchProcess:
                continue
            except (psutil.Error, OSError) as error:
                await self.stop_child(process)
                raise RuntimeError('CPU export resource observation failed') from error
            job['peak_rss_bytes'] = max(job['peak_rss_bytes'], rss)
            job['elapsed_seconds'] = round(time.monotonic() - started, 3)
            try:
                group_rss = await asyncio.to_thread(self.group_rss)
            except Exception as error:
                await self.stop_child(process)
                raise RuntimeError('CPU export process-group observation failed') from error
            job['peak_group_rss_bytes'] = max(job.get('peak_group_rss_bytes', 0), group_rss)
            too_large = rss > 8 * 2**30 or group_rss > 9 * 2**30
            if too_large or time.monotonic() - started > 900:
                await self.stop_child(process)
                raise RuntimeError('CPU export exceeded its memory or 900-second limit')

    def install(self, external, target, source, stack):
        for item in (source, *stack):
            if file_identity(item['path']) != item['file_identity']:
                raise RuntimeError('Frozen export input changed')
        before = file_identity(external)
        validate_custom_checkpoint(external, self.profile.checkpoints.reference)
        if shutil.disk_usage(target.parent).free < external.stat().st_size + 512 * 2**20:
            raise ValueError('Insufficient space to install; completed export retained externally')
        with tempfile.TemporaryDirectory(prefix='.krea-install-', dir=target.parent) as directory:
            candidate = Path(directory) / target.name
            with external.open('rb') as src, candidate.open('xb') as dst:
                while chunk := src.read(8 * 2**20):
                    if self.cancelled:
                        raise InterruptedError('Export cancelled')
                    dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            if file_identity(external) != before:
                raise RuntimeError('Completed export changed during installation')
            candidate.chmod(0o444)
            validate_custom_checkpoint(candidate, self.profile.checkpoints.reference)
            if self.cancelled:
                raise InterruptedError('Export cancelled')
            os.link(candidate, target)  # Refuses any concurrent file or symlink.
        return validate_custom_checkpoint(target, self.profile.checkpoints.reference)

    def command(self, source, stack, external):
        command = [sys.executable, str(Path(__file__).with_name('krea_export.py')),
                   '--source', source['path'], '--output', str(external), '--source-variant', 'turbo',
                   '--template', str(self.profile.checkpoints.reference), '--cpu-threads', '4']
        for item in stack:
            command.extend(['--lora', item['path'], str(item['strength'])])
        return command

    async def run(self, source, stack, external, target):
        job, process = self.job, None
        terminal = {'state': 'failed', 'error': 'CPU export did not complete'}
        try:
            command = self.command(source, stack, external)
            if self.cancelled:
                raise InterruptedError('Export cancelled')
            # Remain in the supervisor-owned process group for shutdown cleanup.
            process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE,
                                                           stderr=asyncio.subprocess.PIPE)
            self.process = process
            job['state'] = 'exporting'
            async def progress():
                tail = []
                while line := await process.stderr.readline():
                    text = line.decode(errors='replace').strip()
                    tail.append(text)
                    tail = tail[-8:]
                    job['progress'] = text[-300:]
                return '\n'.join(tail)
            async def result():
                data = bytearray()
                while chunk := await process.stdout.read(65536):
                    data.extend(chunk)
                    if len(data) > 2**20:
                        raise RuntimeError('Exporter receipt exceeds the supported limit')
                return data
            watcher = asyncio.create_task(self.watch(process, job))
            readers = [asyncio.create_task(result()), asyncio.create_task(progress())]
            try:
                receipt, errors = await asyncio.gather(*readers)
                code = await process.wait()
                await watcher
            finally:
                for task in (watcher, *readers):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(watcher, *readers, return_exceptions=True)
            if self.cancelled:
                raise InterruptedError('Export cancelled')
            if code != 0:
                raise RuntimeError(errors or f'Exporter exited {code}')
            result = json.loads(receipt)
            job.update(state='installing', progress='Installing checkpoint')
            entry = await asyncio.to_thread(self.install, external, target, source, stack)
            self.profile.checkpoints.entries[target.name] = entry
            # Logging cannot turn successful publication into a failed job.
            try:
                self.profile.emit('checkpoint_exported', {'name': target.name, 'receipt': result})
            except Exception:
                pass
            terminal = {'state': 'complete', 'progress': 'Checkpoint ready', 'checkpoint': target.name}
        except InterruptedError:
            terminal = {'state': 'cancelled', 'progress': 'Export cancelled'}
        except Exception as error:
            terminal = {'state': 'failed', 'error': str(error)[-2000:]}
        finally:
            job['state'] = 'cleaning'  # Admission remains closed until reaped.
            await self.stop_child(process)
            if self.process is process:
                self.process = None
            if terminal['state'] != 'complete' and external.exists():
                terminal['progress'] = 'Completed file retained in the external export folder; use a new name to retry'
                terminal['retained_file'] = external.name
            job.update(terminal)

    async def close(self, _app):
        self.cancelled = True
        await self.stop_child(self.process)
        if self.task is not None:
            await asyncio.gather(self.task, return_exceptions=True)

    async def handle(self, request):
        try:
            if request.method == 'GET' and request.path == '/krea-workbench/export':
                return web.json_response(self.snapshot())
            if request.method == 'GET' and request.path == '/krea-workbench/export-sources':
                async with self.lock:
                    sources = await asyncio.to_thread(self.catalog)
                return web.json_response({'sources': sources})
            if request.method == 'POST' and request.path == '/krea-workbench/export':
                same_origin(request)
                async with self.lock:
                    body = await request.json()
                    return web.json_response(await self.start(body), status=202)
            if request.method == 'POST' and request.path == '/krea-workbench/export-cancel':
                same_origin(request)
                async with self.lock:
                    body = await request.json()
                    if not self.active or body != {'id': self.job['id']}:
                        raise ValueError('No matching active export')
                    self.cancelled = True
                    self.job['state'] = 'cancelling'
                return web.json_response(self.snapshot(), status=202)
        except web.HTTPConflict as error:
            return web.json_response({'error': error.text}, status=409)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
            return web.json_response({'error': str(error)}, status=400)
        raise web.HTTPNotFound()
