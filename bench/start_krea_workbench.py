#!/usr/bin/env python3
"""Start the dedicated Krea Comfy workbench under its owned memory guard."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

if __package__:
    from .gpu_runner import MEMORY_DEFAULTS
else:
    from gpu_runner import MEMORY_DEFAULTS

REPO = Path(__file__).resolve().parents[1]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=REPO / 'out/source-device/root')
    parser.add_argument('--port', type=int, default=8191)
    parser.add_argument('--check', action='store_true', help='Check installed assets without starting the server')
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error('Use an unprivileged localhost port between 1024 and 65535')
    return args



def check_port_available(port):
    # On macOS a reusable loopback bind can coexist with a wildcard listener.
    # Probe for a live service first; reuse then permits TIME_WAIT restarts.
    try:
        connection = socket.create_connection(('127.0.0.1', port), timeout=.5)
    except ConnectionRefusedError:
        pass
    except OSError as error:
        raise ValueError(f'Cannot confirm localhost port {port} is available') from error
    else:
        connection.close()
        raise ValueError(f'Localhost port {port} is already in use; select another port')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(('127.0.0.1', port))
        except OSError as error:
            raise ValueError(f'Localhost port {port} is already in use; select another port') from error


def preflight(args):
    from krea_server import UNET, ENCODER, VAE
    from krea_reference import EDIT_REVISION, EDIT_SOURCE_SHA256
    from krea_workbench import SPECIAL
    from spectral_krea import verify_install
    verify_install()
    if sys.platform != 'darwin':
        raise ValueError('The qualified Krea workbench requires macOS with MPS')
    root = args.profile.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Profile must name the installed Krea profile directory')
    for kind, name in (('diffusion_models', UNET), ('text_encoders', ENCODER), ('vae', VAE)):
        path = root / 'models' / kind / name
        if not path.is_file():
            raise ValueError(f'Install the profile asset first: {path}')
    loras = (root / 'models/loras').resolve(strict=True)
    for name in SPECIAL.values():
        if not (loras / name).is_file():
            raise ValueError(f'Install the reference workflow adapter first: {loras / name}')
    runtime = Path(os.environ.get('KREA_EDIT_RUNTIME') or loras.parent.parent / 'runtime/krea2edit').expanduser().resolve(strict=True)
    manifest = json.loads((runtime / 'manifest.json').read_text())
    digest = hashlib.sha256((runtime / '__init__.py').read_bytes()).hexdigest()
    if (manifest.get('revision') != EDIT_REVISION or digest != EDIT_SOURCE_SHA256
            or manifest.get('files', {}).get('__init__.py') != digest):
        raise ValueError('Installed Krea2Edit runtime differs from the pinned workflow revision')
    for package in ('torch', 'safetensors', 'aiohttp', 'numpy'):
        if importlib.util.find_spec(package) is None:
            raise ValueError(f'Missing {package}; use the installed project Python environment')
    kitchen = REPO / 'vendor/comfy-kitchen'
    comfy = REPO / 'vendor/ComfyUI'
    for path in (kitchen / 'comfy_kitchen/__init__.py', comfy / 'main.py', REPO / 'bench/gpu_runner.py'):
        if not path.is_file():
            raise ValueError(f'Missing installed runtime source: {path}')
    workflows = REPO / 'bench/workflows'
    for mode in ('text', 'identity_edit', 'style_reference'):
        if not (workflows / f'krea_{mode}.json').is_file():
            raise ValueError(f'Missing importable Krea workflow: {mode}')
    check_port_available(args.port)
    output = (root / 'out').resolve(strict=True)
    return {'profile': str(root), 'runtime': str(runtime), 'output': str(output),
            'workflows': str(workflows), 'url': f'http://127.0.0.1:{args.port}',
            'runtime_sha256': digest, 'port': args.port}


def main(argv=None):
    args = arguments(argv)
    try:
        ready = preflight(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f'Krea workbench cannot start: {error}', file=sys.stderr)
        return 2
    if args.check:
        print(json.dumps(ready, indent=2))
        return 0
    logs = Path(ready['output']) / 'diagnostics/krea-workbench'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    prefix = logs / f'{stamp}-{os.getpid()}'
    command = [sys.executable, str(REPO / 'bench/gpu_runner.py'), '--monitor',
               '--seconds', '21600', '--', sys.executable,
               str(REPO / 'bench/krea_server.py'), '--workbench', '--root', ready['profile'],
               '--port', str(args.port)]
    environment = os.environ.copy()
    environment['KREA_EDIT_RUNTIME'] = ready['runtime']
    environment['PYTHONUNBUFFERED'] = '1'
    environment['PYTHONPATH'] = str(REPO / 'vendor/comfy-kitchen') + (
        os.pathsep + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    log_path = Path(str(prefix) + '.log')
    Path(str(prefix) + '.json').write_text(json.dumps({**ready, 'command': command,
        'aggregate_seconds': 21600, 'request_seconds': 360,
        **MEMORY_DEFAULTS, 'log': str(log_path)}, indent=2) + '\n')
    print(f"Krea Workbench: {ready['url']}")
    print(f"The Krea text preset loads automatically. Additional presets: {ready['workflows']}")
    print('The server starts after GPU ownership and memory checks. Close this window or press Control-C to stop it.')
    print(f'Run log: {log_path}', flush=True)
    # Replace this launcher with the existing supervisor. Its owner pipe and
    # process-group cleanup remain authoritative on signals or terminal closure.
    with log_path.open('xb', buffering=0) as log:
        os.dup2(log.fileno(), sys.stdout.fileno())
        os.dup2(log.fileno(), sys.stderr.fileno())
        os.execve(sys.executable, command, environment)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
