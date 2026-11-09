"""Actual socket boundaries for safe local Workbench startup and restart."""
from pathlib import Path
import json
import socket
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from start_krea_workbench import check_port_available
import start_krea_workbench as launcher
from gpu_runner import MEMORY_DEFAULTS, arguments


@pytest.mark.parametrize('host', ['127.0.0.1', '0.0.0.0'])
@pytest.mark.parametrize('reuse', [False, True])
def test_existing_listener_is_refused(host, reuse):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, int(reuse))
        listener.bind((host, 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(ValueError, match='already in use'):
            check_port_available(port)


def test_stopped_server_with_time_wait_can_restart():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        port = listener.getsockname()[1]
        with socket.create_connection(('127.0.0.1', port), timeout=.5) as client:
            connection, _ = listener.accept()
            connection.close()  # Server actively closes: accepted socket enters TIME_WAIT.
            assert client.recv(1) == b''
    check_port_available(port)


def test_real_launcher_command_and_receipt_use_shared_memory_defaults(tmp_path, monkeypatch):
    ready = {'output': str(tmp_path), 'profile': str(tmp_path), 'runtime': str(tmp_path),
             'url': 'http://127.0.0.1:8199', 'workflows': str(tmp_path)}
    monkeypatch.setattr(launcher, 'preflight', lambda args: ready)
    monkeypatch.setattr(launcher.os, 'dup2', lambda *args: None)
    captured = {}

    class ExecCaptured(Exception):
        pass

    def capture(python, command, environment):
        captured['command'] = command
        raise ExecCaptured()

    monkeypatch.setattr(launcher.os, 'execve', capture)
    with pytest.raises(ExecCaptured):
        launcher.main(['--port', '8199'])
    command = captured['command']
    guard, child = arguments(command[2:])
    assert {key: getattr(guard, key) for key in MEMORY_DEFAULTS} == MEMORY_DEFAULTS
    assert guard.seconds == 21600 and '--workbench' in child
    receipt = json.loads(next(tmp_path.glob('diagnostics/krea-workbench/*.json')).read_text())
    assert {key: receipt[key] for key in MEMORY_DEFAULTS} == MEMORY_DEFAULTS
    assert receipt['command'] == command
