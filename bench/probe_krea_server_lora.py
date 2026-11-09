"""Five actual selection/switch-back jobs on an owned localhost Krea server."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import urllib.parse
import uuid

from probe_krea_server import graph as base_graph, receipt_identity, PROMPT
from probe_server_session import Client, log_delta

RWB = 'krea2_rwbflat_raw_s500.safetensors'
DARKBRUSH = 'krea2_darkbrush.safetensors'
SOURCES = ('krea_server.py', 'krea_readonly.py', 'krea_conditioning.py',
           'conditioning_cache.py', 'probe_krea_server.py', 'krea_adapter.py',
           'probe_krea_server_lora.py')


def graph(seed, prompt, prefix, lora_name='', lora_strength=1.0):
    result = base_graph(seed, prompt, prefix)
    result['1']['inputs'].update(lora_name=lora_name, lora_strength=lora_strength)
    return result


def graphs(session):
    prompt = PROMPT + ', rwbflat style, monochrome ink wash style'
    return [(label, graph(42, prompt, f'review/krea-api-{session}/{label}', name, strength))
            for label, name, strength in [('base_initial', '', 1.), ('rwb_08', RWB, .8),
                ('rwb_04', RWB, .4), ('darkbrush', DARKBRUSH, 1.), ('base_restored', '', 1.)]]


def summarize(history, expected_index, expected_name, expected_strength,
              expected_seed=None, expected_receipt=None, expected_asset=None, expected_keys=None):
    """Require real sampling, fresh conditioning and the selected complete adapter."""
    try:
        status = history['status']
        cached = dict(status['messages']).get('execution_cached', {}).get('nodes')
        outputs = history['outputs']
        samples, decodes = outputs['3']['krea_audit'], outputs['5']['krea_decode']
        if (status['status_str'] != 'success' or status['completed'] is not True
                or cached is None or '1' in cached or '3' in cached
                or len(samples) != 1 or len(decodes) != 1):
            raise RuntimeError('LoRA request lacks actual preparation, sampling or decode')
        sample, decode = samples[0], decodes[0]
        if (sample['sample_index'] != expected_index or sample['native_calls'] != 1792
                or len(sample['forward_s']) != 8 or sample['finite_latent'] is not True
                or decode['finite_image'] is not True or decode['sample_index'] != expected_index
                or sample['loader_count'] != 1):
            raise RuntimeError('LoRA request does not satisfy the fixed native profile')
        condition = sample['conditioning']
        if (condition['encoder_created'] != 1 or condition['encoder_refs_dead'] is not True
                or condition['conditioning']['origin'] != 'fresh-encoding'
                or condition['conditioning']['cache_policy'] != 'off'):
            raise RuntimeError('LoRA request did not execute and release its encoder')
        lora = sample['lora']
        if expected_name:
            if (lora['name'] != expected_name or lora['strength'] != expected_strength
                    or lora['asset']['policy'] != 'plain'
                    or lora['asset']['strength'] != expected_strength
                    or not lora['patch_keys'] or len(lora['patch_keys']) != len(set(lora['patch_keys']))
                    or len(lora['asset']['sha256']) != 64):
                raise RuntimeError('LoRA request used a different or incomplete selection')
            if ((expected_asset is not None and lora['asset'] != expected_asset)
                    or (expected_keys is not None and len(lora['patch_keys']) != expected_keys)):
                raise RuntimeError('LoRA request differs from the pinned asset or complete target count')
        elif lora is not None:
            raise RuntimeError('Base request retained a LoRA selection')
        if condition['lora'] != lora:
            raise RuntimeError('Conditioning and sampling LoRA selection differ')
        if expected_seed is not None and sample['seed'] != expected_seed:
            raise RuntimeError('LoRA request seed differs')
        receipt = receipt_identity(sample)
        if expected_receipt is not None and receipt != expected_receipt:
            raise RuntimeError('LoRA request changed its base, process, sources or assets')
        images = outputs['6']['images']
        if len(images) != 1:
            raise RuntimeError('LoRA request did not save exactly one image')
        return {'cached_nodes': cached, 'sample': sample, 'decode': decode,
                'images': images, 'receipt': receipt}
    except (KeyError, TypeError, IndexError, AttributeError, ValueError) as error:
        raise RuntimeError('Malformed LoRA qualification history') from error


def output_image(root, record):
    """Resolve the reported image only within this profile's output tree."""
    directory = (Path(root) / 'out' / 'comfyui').resolve(strict=True)
    name, subfolder = record['filename'], record.get('subfolder', '')
    if (record.get('type') != 'output' or not isinstance(name, str)
            or Path(name).name != name or '/' in name or '\\' in name
            or not name.endswith('.png') or not isinstance(subfolder, str)
            or '\\' in subfolder):
        raise ValueError('Expected a profile output PNG receipt')
    selected = (directory / subfolder / name).resolve(strict=True)
    if not selected.is_relative_to(directory):
        raise ValueError('Output PNG escaped the profile output directory')
    return selected


def pixel_identity(path):
    from PIL import Image
    with Image.open(path) as image:
        if image.size != (512,512) or image.mode != 'RGB':
            raise RuntimeError('Expected a 512px RGB PNG')
        return hashlib.sha256(image.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--url', default='http://127.0.0.1:8191')
    parser.add_argument('--seconds', type=float, default=540)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--server-log', type=Path)
    args = parser.parse_args()
    url = urllib.parse.urlparse(args.url)
    if (url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1')
            or url.path or url.query or url.fragment or not 0 < args.seconds <= 540):
        parser.error('Use a dedicated localhost server and at most 540 seconds')
    session = uuid.uuid4().hex[:12]
    planned = graphs(session)
    if not args.run:
        print(json.dumps({'session': session, 'graphs': planned}, indent=2))
        return 0
    from krea_adapter import verify_plain_adapter
    expected_assets = {(name, strength): verify_plain_adapter(args.root / 'models' / 'loras' / name,
                          strength=strength, steps=8, cfg=1.)
                       for name, strength in ((RWB,.8), (RWB,.4), (DARKBRUSH,1.))}
    expected_sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in SOURCES}
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
    code, expected_receipt = 0, None
    pixels = {}
    try:
        emit('start', system=client.request('/system_stats'), no_free=True)
        for index, (label, current) in enumerate(planned, 1):
            name = current['1']['inputs']['lora_name']
            strength = current['1']['inputs']['lora_strength']
            pid, wall, history = client.run(current, session)
            result = summarize(history,index,name,strength,42,expected_receipt,
                expected_assets.get((name,strength)), {RWB:224,DARKBRUSH:264}.get(name))
            if result['receipt']['sources'] != expected_sources:
                raise RuntimeError('Server source receipt differs from the reviewed local LoRA profile')
            expected_receipt = result['receipt']
            image = output_image(args.root,result['images'][0])
            pixels[label] = pixel_identity(image)
            transitions, offset = log_delta(args.server_log,offset)
            emit('request',label=label,prompt_id=pid,wall_s=wall,graph=current,
                 graph_sha256=hashlib.sha256(json.dumps(current,sort_keys=True).encode()).hexdigest(),
                 history=history,system=client.request('/system_stats'),log=transitions,
                 image_path=str(image),pixel_sha256=pixels[label],**result)
        if pixels['base_initial'] != pixels['base_restored']:
            raise RuntimeError('Base switch-back PNG pixels differ from the initial base')
        if any(pixels[label] == pixels['base_initial'] for label in ('rwb_08','rwb_04','darkbrush')):
            raise RuntimeError('A selected LoRA did not affect the image')
        if pixels['rwb_08'] == pixels['rwb_04']:
            raise RuntimeError('Changing LoRA strength did not affect the image')
        emit('parity',base_restored_exactly=True,styles_changed=True,strength_changed=True,pixel_sha256=pixels)
    except (OSError,ValueError,RuntimeError) as error:
        code = 1
        emit('error',message=str(error))
    finally:
        try:
            cleanup = client.cleanup()
            if cleanup.get('left_queue') is False:
                code = 1
            emit('cleanup',**cleanup,server_owned_by_parent=True)
        except (OSError,ValueError,RuntimeError) as error:
            code = 1
            emit('cleanup_error',message=str(error))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
