"""CPU subprocess proofs of GPU-job ownership; no torch, models or GPU work."""
import os
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "bench" / "gpu_runner.py"
WRAPPER = ROOT / "bench" / "with_gpu.sh"


def wait_for(predicate, seconds=5):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for subprocess evidence")


def alive(pid):
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return result.returncode == 0 and "Z" not in result.stdout and bool(result.stdout.strip())


class GPUOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gpu-runner-cpu-")
        self.root = Path(self.temp.name)
        self.lock = self.root / "gpu.lock"
        self.env = {**os.environ, "MK_GPU_LOCK": str(self.lock)}
        self.children = []
        self.other_pids = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for pid in self.other_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.temp.cleanup()

    def launch(self, code, *options, wrapper=False):
        prefix = [str(WRAPPER)] if wrapper else [sys.executable, str(RUNNER), "--grace-seconds", "0.15", *options, "--"]
        child = subprocess.Popen([*prefix, sys.executable, "-c", code], env=self.env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True)
        self.children.append(child)
        return child

    def touch_code(self, path):
        return f"from pathlib import Path; Path({str(path)!r}).touch()"

    def gated_job(self, ready, release):
        return ("from pathlib import Path\nimport time\n"
                f"Path({str(ready)!r}).touch()\n"
                f"while not Path({str(release)!r}).exists(): time.sleep(0.02)\n")

    def test_aged_live_owner_remains_exclusive_through_shell_entrypoint(self):
        ready, release, second = [self.root / name for name in ("ready", "release", "second")]
        first = self.launch(self.gated_job(ready, release), wrapper=True)
        wait_for(ready.exists)
        old = time.time() - 4 * 3600
        os.utime(self.lock, (old, old))
        following = self.launch(self.touch_code(second), wrapper=True)
        time.sleep(0.3)
        self.assertFalse(second.exists(), "Aged live lock admitted overlapping work")
        release.touch()
        self.assertEqual(first.wait(5), 0)
        self.assertEqual(following.wait(5), 0)
        self.assertTrue(second.exists())
        self.assertTrue(self.lock.is_file(), "Permanent lock inode must survive release")

    def test_live_aged_legacy_directory_is_not_reclaimed(self):
        owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.children.append(owner)
        self.lock.mkdir()
        (self.lock / "owner").write_text(f"{owner.pid} old command\n")
        old = time.time() - 4 * 3600
        os.utime(self.lock, (old, old))
        marker = self.root / "started"
        job = self.launch(self.touch_code(marker))
        time.sleep(0.3)
        self.assertFalse(marker.exists())
        owner.terminate()
        owner.wait()
        self.assertEqual(job.wait(5), 0)
        self.assertTrue(marker.exists())

    def test_dead_legacy_owner_recovers_without_waiting_for_age(self):
        owner = subprocess.Popen([sys.executable, "-c", "pass"])
        owner.wait()
        self.lock.mkdir()
        (self.lock / "owner").write_text(f"{owner.pid} dead owner\n")
        marker = self.root / "recovered"
        self.assertEqual(self.launch(self.touch_code(marker)).wait(5), 0)
        self.assertTrue(marker.exists())

    def test_two_contenders_do_not_overlap(self):
        gate = self.root / "exclusive"
        events = self.root / "events"
        code = ("import os,time\nfrom pathlib import Path\n"
                f"gate=Path({str(gate)!r}); gate.mkdir()\n"
                f"with open({str(events)!r},'a') as f: f.write('start\\n')\n"
                "time.sleep(0.25)\n"
                f"with open({str(events)!r},'a') as f: f.write('end\\n')\n"
                "gate.rmdir()\n")
        jobs = [self.launch(code) for _ in range(2)]
        self.assertEqual([job.wait(5) for job in jobs], [0, 0])
        self.assertEqual(events.read_text().splitlines(), ["start", "end", "start", "end"])

    def test_cancel_kills_ignoring_grandchild_before_next_workload(self):
        descendant = self.root / "descendant"
        late = self.root / "late"
        grandchild = ("import os,signal,time\nfrom pathlib import Path\n"
                      "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                      f"Path({str(descendant)!r}).write_text(str(os.getpid()))\n"
                      "time.sleep(2)\n" + self.touch_code(late))
        code = ("import signal,subprocess,sys,time\n"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                f"subprocess.Popen([sys.executable,'-c',{grandchild!r}])\n"
                "time.sleep(30)\n")
        first = self.launch(code)
        wait_for(descendant.exists)
        pid = int(descendant.read_text())
        self.other_pids.append(pid)
        second_marker = self.root / "second"
        second = self.launch(
            "import subprocess\n"
            f"p=subprocess.run(['ps','-o','stat=','-p','{pid}'],capture_output=True,text=True)\n"
            "assert p.returncode or not p.stdout.strip() or 'Z' in p.stdout\n" + self.touch_code(second_marker))
        first.terminate()
        self.assertEqual(first.wait(5), 143)
        self.assertEqual(second.wait(5), 0)
        self.assertFalse(alive(pid))
        self.assertTrue(second_marker.exists())
        self.assertFalse(late.exists())

    def test_successful_leader_exit_still_cleans_background_child(self):
        pidfile = self.root / "pid"
        spawned = f"import os,time; from pathlib import Path; Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(30)"
        code = ("import subprocess,sys,time\nfrom pathlib import Path\n"
                f"subprocess.Popen([sys.executable,'-c',{spawned!r}])\n"
                f"while not Path({str(pidfile)!r}).exists(): time.sleep(.01)\n")
        job = self.launch(code)
        wait_for(pidfile.exists)
        pid = int(pidfile.read_text())
        self.other_pids.append(pid)
        self.assertEqual(job.wait(5), 0)
        self.assertFalse(alive(pid))

    def death_fixture(self, suffix):
        descendant = self.root / ("descendant-" + suffix)
        identity = self.root / ("identity-" + suffix)
        grandchild = ("import os,signal,time\nfrom pathlib import Path\n"
                      "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                      f"Path({str(descendant)!r}).write_text(str(os.getpid()))\n"
                      "time.sleep(30)\n")
        command = ("import os,json,signal,subprocess,sys,time\nfrom pathlib import Path\n"
                   "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                   f"Path({str(identity)!r}).write_text(json.dumps([os.getpid(),os.getpgrp()]))\n"
                   f"subprocess.Popen([sys.executable,'-c',{grandchild!r}],close_fds=True)\n"
                   "time.sleep(30)\n")
        return command, identity, descendant

    def test_supervisor_sigkill_stops_close_fds_descendants_before_successor(self):
        for method in ("pid", "process_group"):
            with self.subTest(method=method):
                code, identity, descendant = self.death_fixture(method)
                first = self.launch(code)  # No deadline can rescue broken EOF handling.
                wait_for(descendant.exists)
                child_pid, guard_group = json.loads(identity.read_text())
                desc_pid = int(descendant.read_text())
                self.other_pids.extend([child_pid, desc_pid])
                if method == "pid":
                    first.kill()
                else:
                    os.killpg(first.pid, signal.SIGKILL)
                first.wait()
                second = self.launch(
                    "import subprocess\n"
                    f"for pid in ({child_pid},{desc_pid}):\n"
                    " p=subprocess.run(['ps','-o','stat=','-p',str(pid)],capture_output=True,text=True)\n"
                    " assert p.returncode or not p.stdout.strip() or 'Z' in p.stdout\n")
                self.assertEqual(second.wait(4), 0)
                self.assertFalse(alive(child_pid))
                self.assertFalse(alive(desc_pid))
                wait_for(lambda: not alive(guard_group))

    def test_guard_sigkill_parent_cleans_known_group(self):
        code, identity, descendant = self.death_fixture("guard-death")
        first = self.launch(code)
        wait_for(descendant.exists)
        child_pid, guard_group = json.loads(identity.read_text())
        desc_pid = int(descendant.read_text())
        self.other_pids.extend([child_pid, desc_pid])
        os.kill(guard_group, signal.SIGKILL)
        self.assertEqual(first.wait(4), 137)
        self.assertFalse(alive(child_pid))
        self.assertFalse(alive(desc_pid))

    def test_both_supervisors_killed_still_blocks_on_close_fds_descendant(self):
        code, identity, descendant = self.death_fixture("both-dead")
        first = self.launch(code)
        wait_for(descendant.exists)
        child_pid, guard_group = json.loads(identity.read_text())
        desc_pid = int(descendant.read_text())
        self.other_pids.extend([child_pid, desc_pid])
        # Freeze both before killing them to prevent either cleanup path from
        # rescuing this fallback test. Only these isolated fixture PIDs are used.
        os.kill(guard_group, signal.SIGSTOP)
        os.kill(first.pid, signal.SIGSTOP)
        first.kill()
        os.kill(guard_group, signal.SIGKILL)
        first.wait()
        marker = self.root / "after-orphans"
        second = self.launch(self.touch_code(marker))
        time.sleep(0.3)
        self.assertFalse(marker.exists())
        self.assertTrue(alive(child_pid))
        self.assertTrue(alive(desc_pid))
        os.kill(child_pid, signal.SIGKILL)
        time.sleep(0.2)
        self.assertFalse(marker.exists(), "Direct child exit hid surviving close_fds descendant")
        self.assertTrue(alive(desc_pid))
        os.kill(desc_pid, signal.SIGKILL)
        self.assertEqual(second.wait(4), 0)
        self.assertTrue(marker.exists())
        self.assertIsNone(json.loads(self.lock.read_text())["group"])

    def test_cancelled_waiter_never_starts_command(self):
        ready, release, marker = [self.root / name for name in ("ready", "release", "bad")]
        first = self.launch(self.gated_job(ready, release))
        wait_for(ready.exists)
        second = self.launch(self.touch_code(marker))
        time.sleep(0.2)
        second.terminate()
        self.assertEqual(second.wait(5), 143)
        release.touch()
        self.assertEqual(first.wait(5), 0)
        self.assertFalse(marker.exists())

    def test_deadline_and_command_exit_status(self):
        self.assertEqual(self.launch("import time; time.sleep(30)", "--seconds", "0.2").wait(5), 124)
        self.assertEqual(self.launch("raise SystemExit(17)").wait(5), 17)

    def test_nonfinite_resource_bounds_refuse_to_start(self):
        marker = self.root / "invalid-bound-started"
        for option in ("--seconds", "--rss-gib", "--swap-growth-gib", "--min-disk-free-gib", "--grace-seconds"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(option=option, value=value):
                    self.assertEqual(self.launch(self.touch_code(marker), option + "=" + value).wait(4), 2)
                    self.assertFalse(marker.exists())

    @unittest.skipUnless(sys.platform == "darwin", "Real macOS telemetry required")
    def test_monitor_refuses_admission_before_start(self):
        marker = self.root / "not-admitted"
        result = subprocess.run(
            [sys.executable, str(RUNNER), "--monitor", "--min-free-percent", "100", "--",
             sys.executable, "-c", self.touch_code(marker)],
            env=self.env, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 125, result.stdout + result.stderr)
        self.assertIn("Memory admission refused", result.stdout)
        self.assertFalse(marker.exists())

    @unittest.skipUnless(sys.platform == "darwin", "Real macOS telemetry required")
    def test_monitor_stops_real_cpu_process_at_rss_limit(self):
        common = [sys.executable, str(RUNNER), "--monitor", "--seconds", "5", "--rss-gib", "0.064",
                  "--swap-growth-gib", "0.25", "--min-disk-free-gib", "0",
                  "--grace-seconds", "0.15", "--", sys.executable, "-c"]
        baseline = subprocess.run([*common, "import time; time.sleep(0.3)"],
                                  env=self.env, capture_output=True, text=True, timeout=5)
        self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)
        result = subprocess.run(
            [*common, "import time; payload=bytearray(80*1024*1024); time.sleep(10)"],
            env=self.env, capture_output=True, text=True, timeout=8)
        self.assertEqual(result.returncode, 125, result.stdout + result.stderr)
        self.assertIn('"reason": "process-group RSS"', result.stdout)


    def run_labelled(self, label, code, seconds="20"):
        env = {**self.env, "MK_GPU_LABEL": label}
        return subprocess.Popen([sys.executable, str(RUNNER), "--grace-seconds", "0.15", "--seconds", seconds, "--",
                                 sys.executable, "-c", code], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, start_new_session=True)

    @staticmethod
    def gpu_run_record(stdout):
        lines = [line for line in stdout.splitlines() if line.startswith("GPU_RUN ")]
        return json.loads(lines[-1][len("GPU_RUN "):])

    def test_label_is_recorded_in_lock_and_run_record(self):
        holder = self.run_labelled("side-a:1", "import time; time.sleep(0.6)")
        self.children.append(holder)
        wait_for(lambda: self.lock.exists() and '"side-a:1"' in self.lock.read_text() and '"released": false' in self.lock.read_text())
        out, _ = holder.communicate(timeout=10)
        record = self.gpu_run_record(out)
        self.assertEqual(record["label"], "side-a:1")
        self.assertIsNone(record["lock_prior_label"])
        self.assertLess(record["lock_wait_seconds"], 0.5)
        held = json.loads(self.lock.read_text())
        self.assertEqual((held["label"], held["released"]), ("side-a:1", True))

    def test_waiter_records_who_it_waited_behind(self):
        holder = self.run_labelled("side-a:2", "import time; time.sleep(1.0)")
        self.children.append(holder)
        wait_for(lambda: self.lock.exists() and '"released": false' in self.lock.read_text())
        waiter = self.run_labelled("side-b:7", "pass")
        self.children.append(waiter)
        out, _ = waiter.communicate(timeout=15)
        holder.communicate(timeout=10)
        record = self.gpu_run_record(out)
        self.assertEqual(record["label"], "side-b:7")
        self.assertEqual(record["lock_prior_label"], "side-a:2")
        self.assertEqual(record["lock_last_label"], "side-a:2")
        self.assertGreater(record["lock_wait_seconds"], 0.3)

    def test_unlabelled_run_records_nulls(self):
        env = {k: v for k, v in self.env.items() if k != "MK_GPU_LABEL"}
        done = subprocess.run([sys.executable, str(RUNNER), "--seconds", "20", "--", sys.executable, "-c", "pass"],
                              env=env, capture_output=True, text=True, timeout=20)
        record = self.gpu_run_record(done.stdout)
        self.assertIsNone(record["label"])
        self.assertIsNone(record["lock_prior_label"])


if __name__ == "__main__":
    unittest.main()
