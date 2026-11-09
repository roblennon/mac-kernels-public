#!/usr/bin/env python3
"""Serialize a foreground workload and own its process group until it is gone.

The permanent lock file must not be deleted: flock ownership, not its age or
contents, decides exclusivity. MK_GPU_LOCK can isolate CPU tests. --monitor
adds macOS admission/pressure/swap/RSS limits and a default 540-second deadline.
Commands must remain in their process group; detached daemons are unsupported.
"""
import argparse
from collections import deque
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


# One generation policy. Warning pressure is observable, not an OOM.
MEMORY_DEFAULTS = {"rss_gib": 18, "swap_growth_gib": 4,
                   "min_free_percent": 0, "min_disk_free_gib": 4}


def memory_stop_reason(current, initial, args, rss=0, peak_swap=None):
    """Decide from measured limits; reclaimed-page percentage is opt-in only."""
    if current["pressure"] == 4:
        return "critical system memory pressure"
    if current["pressure"] not in (1, 2):
        return "unknown system memory pressure"
    growth = max(0, (current["swap_gib"] if peak_swap is None else peak_swap)
                 - initial["swap_gib"])
    if growth > args.swap_growth_gib:
        return "swap growth"
    remaining = max(0, args.swap_growth_gib - max(0, current["swap_gib"] - initial["swap_gib"]))
    if current["disk_free_gib"] < args.min_disk_free_gib + remaining:
        return "swap-volume disk headroom"
    if args.min_free_percent and current["free_percent"] < args.min_free_percent:
        return "system memory free percentage"
    if rss > args.rss_gib:
        return "process-group RSS"
    return None


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def record_lock(fd, group=None):
    payload = (json.dumps({"owner_pid": os.getpid(), "group": group}) + "\n").encode()
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    if os.write(fd, payload) != len(payload):
        raise OSError("Incomplete GPU ownership record")


def acquire(path, cancelled):
    # One-time migration from the old directory lock. A live legacy owner is
    # respected regardless of age. Unknown owners fail closed for inspection.
    while path.is_dir():
        if cancelled():
            return None
        try:
            owner = (path / "owner").read_text().split()
            pid = int(owner[0])
            if pid <= 0:
                raise ValueError("Nonpositive legacy PID")
        except FileNotFoundError:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue  # Another contender completed legacy migration.
            if age > 60:
                try:
                    path.rmdir()  # Only an actually empty stale directory.
                except (FileNotFoundError, OSError):
                    pass
            time.sleep(0.1)
            continue
        except (ValueError, IndexError) as error:
            raise RuntimeError(f"Invalid legacy lock owner at {path}; inspect it") from error
        if pid_alive(pid):
            time.sleep(0.1)
            continue
        try:
            (path / "owner").unlink()
            path.rmdir()
        except (FileNotFoundError, NotADirectoryError):
            pass
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        while not cancelled():
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                previous = os.read(fd, 4096).decode().strip()
                group = json.loads(previous).get("group") if previous.startswith("{") else None
                if group is not None:
                    if type(group) is not int or group <= 0:
                        raise RuntimeError("Invalid saved GPU process group; inspect lock")
                    # Both supervisors may have died. Never signal a saved PGID
                    # (it could be reused); fail closed until no members remain.
                    while group_rows(group):
                        if cancelled():
                            os.close(fd)
                            return None
                        time.sleep(0.1)
                record_lock(fd)
                return fd
            except BlockingIOError:
                time.sleep(0.1)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    return None


def group_rows(pgid, exclude_pid=None):
    rows = subprocess.check_output(["ps", "-axo", "pid=,pgid=,rss=,stat="],
                                   text=True, timeout=5, start_new_session=True).splitlines()
    values = [row.split() for row in rows if row.split()]
    return [{"pid": int(pid), "rss_kib": int(rss)} for pid, group, rss, status in values
            if int(group) == pgid and int(pid) != exclude_pid and "Z" not in status]


def kill_group(pgid, sig, exclude_pid=None):
    if exclude_pid is None:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
    else:
        # The independent guard is a member/leader of this group. Keep it alive
        # holding flock until every workload descendant is actually gone.
        for row in group_rows(pgid, exclude_pid):
            try:
                os.kill(row["pid"], sig)
            except ProcessLookupError:
                pass


def stop_group(child, grace, pgid=None, exclude_pid=None):
    # Always clean the group, even when its original leader already exited.
    # Never release the lock while a surviving descendant is still running.
    pgid = child.pid if pgid is None else pgid
    try:
        kill_group(pgid, signal.SIGTERM, exclude_pid)
    except (subprocess.SubprocessError, OSError):
        pass  # Retain ownership and retry below when telemetry recovers.
    until = time.monotonic() + grace
    while True:
        child.poll()  # Reap the direct child as soon as it exits.
        try:
            alive = bool(group_rows(pgid, exclude_pid))
        except (subprocess.SubprocessError, OSError):
            alive = True  # Telemetry failure must not release the lock early.
        if not alive:
            break
        if time.monotonic() >= until:
            try:
                kill_group(pgid, signal.SIGKILL, exclude_pid)
            except (subprocess.SubprocessError, OSError):
                pass
        time.sleep(0.1)
    child.wait()


# macOS keeps swap files here; fall back to / where that path does not exist.
SWAP_VOLUME = "/private/var/vm" if os.path.exists("/private/var/vm") else "/"


def system_state(*, include_free_percent=True):
    if sys.platform != "darwin":
        raise RuntimeError("--monitor currently requires macOS")
    out = subprocess.check_output(["/usr/sbin/sysctl", "-n",
                                   "kern.memorystatus_vm_pressure_level", "vm.swapusage"],
                                  text=True, timeout=5)
    lines = out.splitlines()
    if not lines or not re.fullmatch(r"\d+", lines[0]):
        raise RuntimeError("Cannot parse system memory pressure")
    swap = re.search(r"used\s*=\s*([0-9]+(?:\.[0-9]+)?)M", out)
    percent = None
    if include_free_percent:
        free = subprocess.check_output(["/usr/bin/memory_pressure", "-Q"], text=True, timeout=5)
        percent = re.search(r"System-wide memory free percentage:\s*(\d+)%", free)
    if not swap or (include_free_percent and not percent):
        raise RuntimeError("Cannot parse system memory telemetry")
    return {"pressure": int(lines[0]), "swap_gib": float(swap[1]) / 1024,
            "free_percent": int(percent[1]) if percent else None,
            "disk_free_gib": shutil.disk_usage(SWAP_VOLUME).free / 2**30}


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--rss-gib", type=float, default=MEMORY_DEFAULTS["rss_gib"])
    parser.add_argument("--swap-growth-gib", type=float, default=MEMORY_DEFAULTS["swap_growth_gib"])
    parser.add_argument("--min-free-percent", type=int, default=MEMORY_DEFAULTS["min_free_percent"])
    parser.add_argument("--min-disk-free-gib", type=float, default=MEMORY_DEFAULTS["min_disk_free_gib"])
    parser.add_argument("--grace-seconds", type=float, default=5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A foreground command is required")
    if args.seconds is None and args.monitor:
        args.seconds = 540
    bounds = [args.rss_gib, args.swap_growth_gib, args.min_disk_free_gib, args.grace_seconds]
    if args.seconds is not None:
        bounds.append(args.seconds)
    if (not all(math.isfinite(bound) for bound in bounds)
        or (args.seconds is not None and args.seconds <= 0) or args.rss_gib <= 0
        or args.swap_growth_gib < 0 or args.min_disk_free_gib < 0 or args.grace_seconds < 0
        or not 0 <= args.min_free_percent <= 100):
        parser.error("Invalid resource bound")
    return args, command


def signal_state():
    caught = [0]
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: caught.__setitem__(0, signum))
    return caught


def owner_gone(pipe):
    try:
        return os.read(pipe, 1) == b""
    except BlockingIOError:
        return False


def guard(args, command, fd, owner_pipe):
    """Survive caller-group SIGKILL, but promptly stop work when its pipe closes."""
    caught = signal_state()
    os.set_blocking(owner_pipe, False)
    pgid = os.getpgrp()
    if pgid != os.getpid():
        raise RuntimeError("Guard must lead its own workload process group")
    child = None
    reason = None
    peak_rss = 0
    initial = None
    peak_swap = None
    last_sample = None
    samples = deque(maxlen=30)
    warning_started = None
    warning_seconds = 0
    warning_periods = 0
    start = time.monotonic()
    try:
        if args.monitor:
            initial = system_state(include_free_percent=bool(args.min_free_percent))
            last_sample = {"elapsed_seconds": time.monotonic() - start, **initial}
            samples.append(last_sample)
            peak_swap = initial["swap_gib"]
            admission = memory_stop_reason(initial, initial, args)
            if admission:
                raise RuntimeError(f"Memory admission refused ({admission}): {initial}")
            if initial["pressure"] == 2:
                warning_started = time.monotonic()
                warning_periods += 1
                print("GPU_MEMORY_WARNING " + json.dumps(last_sample), flush=True)
        if caught[0] or owner_gone(owner_pipe):
            reason = "owner exited or cancelled"
            return 128 + caught[0] if caught[0] else 125
        # Work stays in the guard's known group. The parent already knows its
        # PGID from Popen, so even guard death during launch is recoverable.
        # Neither flock nor the owner pipe is inherited by workload children.
        record_lock(fd, group=pgid)  # Persist before any workload can execute.
        child = subprocess.Popen(command)
        next_sample = start
        while child.poll() is None:
            now = time.monotonic()
            if caught[0]:
                reason = f"signal {caught[0]}"
                break
            if owner_gone(owner_pipe):
                reason = "owner exited or cancelled"
                break
            if args.seconds is not None and now - start >= args.seconds:
                reason = "deadline"
                break
            if args.monitor and now >= next_sample:
                current = system_state(include_free_percent=bool(args.min_free_percent))
                measured = time.monotonic()
                peak_swap = max(peak_swap, current["swap_gib"])
                rss = sum(row["rss_kib"] for row in group_rows(pgid)) / 1024**2
                peak_rss = max(peak_rss, rss)
                last_sample = {"elapsed_seconds": measured - start, **current, "group_rss_gib": rss}
                samples.append(last_sample)
                if current["pressure"] == 2 and warning_started is None:
                    warning_started = measured
                    warning_periods += 1
                    print("GPU_MEMORY_WARNING " + json.dumps(last_sample), flush=True)
                elif current["pressure"] != 2 and warning_started is not None:
                    warning_seconds += measured - warning_started
                    warning_started = None
                    print("GPU_MEMORY_WARNING_END " + json.dumps(last_sample), flush=True)
                reason = memory_stop_reason(current, initial, args, rss, peak_swap)
                if reason:
                    break
                next_sample = measured + 2
            time.sleep(0.1)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        reason = str(error)
        print(f"GPU runner: {reason}", file=sys.stderr, flush=True)
    finally:
        if warning_started is not None:
            warning_seconds += time.monotonic() - warning_started
        if child is not None:
            stop_group(child, args.grace_seconds, pgid=pgid, exclude_pid=os.getpid())
        record_lock(fd)
        # Never unlink: waiters may already have opened this exact inode.
        os.close(fd)
        os.close(owner_pipe)
        if args.monitor or args.seconds is not None or reason:
            print("GPU_RUN " + json.dumps({"reason": reason, "exit": child.returncode if child else None,
                  "seconds": round(time.monotonic() - start, 3), "peak_group_rss_gib": peak_rss,
                  "initial_memory": initial, "peak_swap_gib": peak_swap,
                  "last_sample": last_sample, "recent_samples": list(samples),
                  "warning_periods": warning_periods, "warning_seconds": round(warning_seconds, 3),
                  "memory_limits": {name: getattr(args, name) for name in MEMORY_DEFAULTS}}), flush=True)
    if caught[0]:
        return 128 + caught[0]
    if reason:
        return 124 if reason == "deadline" else 125
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


def main():
    argv = sys.argv[1:]
    if argv[:1] == ["--_guard"]:
        fd, owner_pipe = map(int, argv[1:3])
        args, command = arguments(argv[3:])
        return guard(args, command, fd, owner_pipe)
    args, command = arguments(argv)
    caught = signal_state()
    path = Path(os.environ.get("MK_GPU_LOCK", "/tmp/mac-kernels-gpu.lock"))
    fd = acquire(path, lambda: bool(caught[0]))
    if fd is None:
        return 128 + caught[0]
    reader, writer = os.pipe()
    child = None
    try:
        if caught[0]:
            return 128 + caught[0]
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--_guard", str(fd), str(reader), *argv],
            start_new_session=True, pass_fds=(fd, reader))
        os.close(reader)
        reader = None
        while child.poll() is None:
            if caught[0] and writer is not None:
                os.close(writer)
                writer = None
            time.sleep(0.05)
    finally:
        if writer is not None:
            os.close(writer)
        if reader is not None:
            os.close(reader)
        if child is not None:
            # If the guard itself crashes, its known group is still ours to
            # clean. Keep the parent's flock until every member has stopped.
            stop_group(child, args.grace_seconds)
        record_lock(fd)
        os.close(fd)
    if caught[0]:
        return 128 + caught[0]
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
