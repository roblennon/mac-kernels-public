"""Three actual Krea jobs on an owned dedicated server; no /free or GPU launch."""
import argparse
import copy
import hashlib
import json
import signal
import urllib.parse
import uuid
from pathlib import Path
from probe_server_session import Client, log_delta

PROMPT = ('Ultra low angle action shot, male soccer player jumping mid-air high kick, clear vivid blue sky background, '
          'sharp focused upper body, maroon jersey and white shorts, harsh direct sunlight, high contrast sports photography')


def graph(seed, prompt, prefix):
    return {
        '1': {'class_type': 'BenchKreaConditioningModel', 'inputs': {'prompt': prompt}},
        '2': {'class_type': 'EmptyLatentImage', 'inputs': {'width': 512, 'height': 512, 'batch_size': 1}},
        '3': {'class_type': 'BenchKreaSampler', 'inputs': {'model': ['1', 0], 'positive': ['1', 1],
            'negative': ['1', 2], 'latent_image': ['2', 0], 'seed': seed, 'steps': 8, 'cfg': 1.0,
            'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}},
        '4': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'qwen_image_vae.safetensors'}},
        '5': {'class_type': 'BenchKreaDecode', 'inputs': {'samples': ['3', 0], 'vae': ['4', 0]}},
        '6': {'class_type': 'SaveImage', 'inputs': {'images': ['5', 0], 'filename_prefix': prefix}},
    }


def graphs(session):
    return [(label, graph(seed, prompt, f'review/krea-api-{session}/{label}'))
            for label, seed, prompt in [('initial', 42, PROMPT), ('changed_seed', 43, PROMPT),
                                       ('changed_prompt', 43, PROMPT + ', bright yellow soccer ball')]]


def receipt_identity(sample):
    return {key: sample[key] for key in ('server_pid', 'model_identity', 'checkpoint_identity',
                                        'asset_identities', 'sources', 'loader_count')}


def summarize(history, expected_index, expected_seed=None, expected_receipt=None):
    try:
        status = history.get('status', {})
        messages = dict(status.get('messages', []))
        cached = messages.get('execution_cached', {}).get('nodes')
        outputs = history.get('outputs', {})
        samples = outputs.get('3', {}).get('krea_audit', [])
        decodes = outputs.get('5', {}).get('krea_decode', [])
        if (status.get('status_str') != 'success' or status.get('completed') is not True
                or cached is None or '3' in cached or len(samples) != 1 or len(decodes) != 1):
            raise RuntimeError('Krea request failed or lacks actual sampling/decode evidence')
        sample, decode = samples[0], decodes[0]
        if (sample['sample_index'] != expected_index or sample['native_calls'] != 1792
                or len(sample['forward_s']) != 8 or not sample['finite_latent']
                or not decode['finite_image'] or decode['sample_index'] != expected_index
                or sample['loader_count'] != 1):
            raise RuntimeError('Krea request does not satisfy the fixed native profile')
        conditioning = sample['conditioning']
        if (conditioning['encoder_created'] != 1 or conditioning['encoder_refs_dead'] is not True
                or conditioning['conditioning']['origin'] != 'fresh-encoding'
                or conditioning['conditioning']['cache_policy'] != 'off'):
            raise RuntimeError('Dedicated Krea API requires real encoder execution and release')
        if expected_seed is not None and sample['seed'] != expected_seed:
            raise RuntimeError('Krea sampler seed differs from the submitted graph')
        receipt = receipt_identity(sample)
        if expected_receipt is not None and receipt != expected_receipt:
            raise RuntimeError('Krea model, process, assets, or sources changed across requests')
        expected_cached = expected_index == 2
        if ('1' in cached) != expected_cached:
            raise RuntimeError('Conditioning/model cache transition differs from the three-job plan')
        images = outputs.get('6', {}).get('images', [])
        if len(images) != 1:
            raise RuntimeError('Krea request did not save exactly one image')
        return {'cached_nodes': cached, 'sample': sample, 'decode': decode, 'images': images, 'receipt': receipt}
    except (KeyError, TypeError, IndexError, AttributeError, ValueError) as error:
        raise RuntimeError('Malformed Krea qualification history or receipt') from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--url', default='http://127.0.0.1:8191')
    parser.add_argument('--seconds', type=float, default=500)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--server-log', type=Path)
    args = parser.parse_args()
    url = urllib.parse.urlparse(args.url)
    if (url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1')
            or url.path or url.query or url.fragment or not 0 < args.seconds <= 520):
        parser.error('Use a dedicated localhost server and at most 520 seconds')
    session = uuid.uuid4().hex[:12]
    planned = graphs(session)
    if not args.run:
        print(json.dumps({'session': session, 'graphs': planned}, indent=2))
        return 0
    cancelled = [False]
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: cancelled.__setitem__(0, True))
    client = Client(args.url, args.seconds, lambda: cancelled[0])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def emit(event, **data):
        line = json.dumps({'event': event, 'session': session, **data})
        print(line, flush=True)
        with args.report.open('a') as stream:
            stream.write(line + '\n')
    offset = args.server_log.stat().st_size if args.server_log else 0
    result_code = 0
    expected_receipt = None
    expected_sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ('krea_server.py', 'krea_readonly.py', 'krea_conditioning.py',
                                     'conditioning_cache.py', 'probe_krea_server.py')}
    try:
        emit('start', system=client.request('/system_stats'), no_free=True)
        for index, (label, current) in enumerate(planned, 1):
            pid, wall, history = client.run(current, session)
            result = summarize(history, index, current['3']['inputs']['seed'], expected_receipt)
            if result['receipt']['sources'] != expected_sources:
                raise RuntimeError('Server source receipt differs from the reviewed local profile')
            expected_receipt = result['receipt']
            transitions, offset = log_delta(args.server_log, offset)
            emit('request', label=label, prompt_id=pid, wall_s=wall, graph=current,
                 graph_sha256=hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest(),
                 history=history, system=client.request('/system_stats'), log=transitions, **result)
    except (OSError, ValueError, RuntimeError) as error:
        result_code = 1
        emit('error', message=str(error))
    finally:
        try:
            cleanup = client.cleanup()
            if cleanup.get('left_queue') is False:
                result_code = 1
            emit('cleanup', **cleanup, server_owned_by_parent=True)
        except (OSError, ValueError, RuntimeError) as error:
            result_code = 1
            emit('cleanup_error', message=str(error))
    return result_code


if __name__ == '__main__':
    raise SystemExit(main())
