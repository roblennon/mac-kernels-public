"""Real policy decisions and owned CPU subprocesses; no model/GPU allocations."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bench import gpu_runner as runner


def state(pressure=1, swap=2, disk=20, free=None):
    return {"pressure": pressure, "swap_gib": swap, "disk_free_gib": disk,
            "free_percent": free}


@pytest.mark.parametrize('current,peak,rss,expected', [
    (state(2), None, 0, None),
    (state(4), None, 0, 'critical system memory pressure'),
    (state(3), None, 0, 'unknown system memory pressure'),
    (state(disk=8), None, 18, None),
    (state(disk=7.99), None, 0, 'swap-volume disk headroom'),
    (state(swap=6, disk=4), 6, 0, None),
    (state(swap=6.01), 6.01, 0, 'swap growth'),
    (state(swap=2), 6.01, 0, 'swap growth'),
    (state(swap=3, disk=6.99), 3, 0, 'swap-volume disk headroom'),
    (state(swap=2, disk=7.99), 5, 0, 'swap-volume disk headroom'),
    (state(free=0), None, 18.01, 'process-group RSS'),
])
def test_actual_policy_boundaries_and_recovered_swap(current, peak, rss, expected):
    args, _ = runner.arguments(['--monitor', '--', 'unused'])
    assert runner.memory_stop_reason(current, state(), args, rss, peak) == expected


def test_disabled_free_percentage_does_not_query_extra_command():
    with patch.object(runner.sys, 'platform', 'darwin'), \
         patch.object(runner.subprocess, 'check_output', return_value='1\ntotal = 4096.00M used = 2048.00M free = 2048.00M') as query, \
         patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=12 * 2**30)):
        assert runner.system_state(include_free_percent=False) == state(disk=12)
        assert query.call_count == 1
        assert query.call_args.args[0][0] == '/usr/sbin/sysctl'


@pytest.mark.parametrize('output', ['', 'garbage\nused = 1M', '1\ninvalid swap'])
def test_invalid_system_telemetry_is_explicit(output):
    with patch.object(runner.sys, 'platform', 'darwin'), \
         patch.object(runner.subprocess, 'check_output', return_value=output):
        with pytest.raises(RuntimeError, match='Cannot parse system memory'):
            runner.system_state(include_free_percent=False)


def owned_trace(tmp_path, trace, child_seconds):
    # Run the real guard as its own process-group leader. Only OS memory readings
    # are supplied: launch, signals, flock, group census and reap remain real.
    pid_file = tmp_path / 'child.pid'
    script = '''
import fcntl,json,os,sys
from bench import gpu_runner as r
rows=json.loads(sys.argv[1]); index=0
def observe(**kwargs):
 global index
 assert kwargs == {'include_free_percent':False}
 row=rows[min(index,len(rows)-1)];index+=1;return row
r.system_state=observe
fd=os.open(sys.argv[2],os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX)
read,write=os.pipe()
args,command=r.arguments(['--monitor','--seconds','12','--grace-seconds','.1','--',sys.executable,'-c',sys.argv[3]])
sys.exit(r.guard(args,command,fd,read))
'''
    child = f"import os,time; from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep({child_seconds})"
    result = subprocess.run([sys.executable, '-c', script, json.dumps(trace),
                             str(tmp_path / 'lock'), child], cwd=runner.Path(__file__).resolve().parents[1],
                            start_new_session=True, text=True, capture_output=True, timeout=16)
    report = json.loads(next(line.split('GPU_RUN ', 1)[1] for line in result.stdout.splitlines()
                             if line.startswith('GPU_RUN ')))
    if pid_file.exists():
        pid = int(pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    return result, report, pid_file.exists()


def test_warning_admission_and_recovery_complete_owned_job(tmp_path):
    result, report, started = owned_trace(tmp_path, [state(2), state(2), state()], 3)
    assert result.returncode == 0 and started, result.stderr
    assert report['reason'] is None
    assert report['warning_periods'] == 1 and report['warning_seconds'] >= 2
    assert report['last_sample']['pressure'] == 1
    assert report['memory_limits'] == runner.MEMORY_DEFAULTS
    assert 'GPU_MEMORY_WARNING_END' in result.stdout


@pytest.mark.parametrize('last,expected', [
    (state(4), 'critical system memory pressure'),
    (state(disk=7), 'swap-volume disk headroom'),
    (state(swap=6.1), 'swap growth'),
])
def test_runtime_limit_records_trigger_and_reaps(tmp_path, last, expected):
    result, report, started = owned_trace(tmp_path, [state(), state(2), last], 10)
    assert result.returncode == 125 and started, result.stderr
    assert report['reason'] == expected
    assert all(report['last_sample'][key] == value for key, value in last.items())
    assert report['recent_samples'][-1] == report['last_sample']


@pytest.mark.parametrize('initial', [state(disk=7), state(pressure=99)])
def test_unsafe_admission_never_starts_work(tmp_path, initial):
    result, report, started = owned_trace(tmp_path, [initial], 1)
    assert result.returncode == 125 and not started
    assert 'Memory admission refused' in report['reason']
    assert report['last_sample']['pressure'] == initial['pressure']
