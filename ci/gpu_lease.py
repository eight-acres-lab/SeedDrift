"""GPU lease: on a machine whose GPU another service holds while SeedDrift is idle, stop that service for the GPU part
of a run and start it again afterwards, so that the GPU is used by one side at a time and never flaps.

Configured per machine (not per profile) in <CI root>/host/gpu_lease.json; without that file every call is a no-op.

  {"service": "qwen35-model",                        a supervisord program (supervisorctl stop / start)
   "metrics_url": "http://127.0.0.1:8000/metrics",   requests running or waiting (vLLM metrics) before a stop
   "health_url": "http://127.0.0.1:8000/health",     200 once the service is back
   "holds_mb": 33000,                                 GPU memory the service holds (gate G7_resources counts it as free)
   "drain_s": 900, "grace_s": 600, "restore_wait_s": 1200}

acquire(holder): registers the holder (its pid); if the service runs, waits up to drain_s for its in-flight requests
  to end (it does not wait for an idle service beyond that: the run asked for the GPU), stops it, and waits until the
  GPU memory is released. A watchdog process is started that restores the service.
release(holder): unregisters the holder. The watchdog starts the service again once no live holder has been registered
  for grace_s seconds (a run that takes the GPU again soon after keeps it; a holder that died without releasing is
  dropped by its pid), and waits for the health check.
State: <CI root>/locks/gpu_lease/ (holders/<holder>.json, state.json, lease.log), changed under a file lock.

  python -m ci.gpu_lease status|acquire HOLDER PID|release HOLDER|watch

Processes of a holder (the training children) are registered with Held.add(pid): the lease stays held while any of
them lives, even if the stage process that took it dies.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

ROOT_ENV = "CI_ROOT"


def root():
    return os.environ.get(ROOT_ENV) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config(ci_root=None):
    p = os.path.join(ci_root or root(), "host", "gpu_lease.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        c = json.load(f)
    if not re.fullmatch(r"[A-Za-z0-9._:-]+", str(c.get("service", ""))):
        raise SystemExit("%s: service must be a supervisord program name" % p)
    return c


def _dir(ci_root=None):
    d = os.path.join(ci_root or root(), "locks", "gpu_lease")
    os.makedirs(os.path.join(d, "holders"), exist_ok=True)
    return d


def _log(d, msg):
    with open(os.path.join(d, "lease.log"), "a") as f:
        f.write("%s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg))


@contextlib.contextmanager
def _locked(d):
    with open(os.path.join(d, "state.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        yield


def _state(d):
    p = os.path.join(d, "state.json")
    return json.load(open(p)) if os.path.exists(p) else dict(stopped_by_lease=False, last_release=0.0)


def _save_state(d, s):
    p = os.path.join(d, "state.json")
    with open(p + ".part", "w") as f:
        json.dump(s, f, indent=1, sort_keys=True)
    os.replace(p + ".part", p)


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _start_time(pid):
    """The process's start time in clock ticks (/proc/<pid>/stat field 22), "" where /proc is not there."""
    try:
        with open("/proc/%d/stat" % int(pid)) as f:
            return f.read().rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return ""


def _holder_alive(h):
    """A registered process is alive when its pid runs and, where recorded, started at the recorded time (a reused pid
    does not hold the lease)."""
    pid = h.get("pid", -1)
    return _alive(pid) and (not h.get("start") or _start_time(pid) == h["start"])


def _holders(d):
    """Live holders; files of dead pids are removed."""
    out = {}
    hd = os.path.join(d, "holders")
    for n in sorted(os.listdir(hd)):
        p = os.path.join(hd, n)
        try:
            h = json.load(open(p))
        except (OSError, ValueError):
            continue
        if _holder_alive(h):
            out[n[:-5]] = h
        else:
            os.remove(p)
            _log(d, "dropped dead holder %s (pid %s)" % (n[:-5], h.get("pid")))
    return out


def _ctl(c, action):
    try:
        r = subprocess.run(["supervisorctl", action, c["service"]], capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as err:
        return 125, "%s: %s" % (type(err).__name__, err)
    return r.returncode, (r.stdout + r.stderr).strip()


def service_running(c):
    """The service is up or on its way (supervisord RUNNING, STARTING, BACKOFF, STOPPING)."""
    _, out = _ctl(c, "status")
    return bool(re.search(r"\b(RUNNING|STARTING|BACKOFF|STOPPING)\b", out))


def _in_flight(c):
    try:
        with urllib.request.urlopen(c["metrics_url"], timeout=10) as r:
            text = r.read().decode()
    except Exception:
        return None
    n = 0.0
    for line in text.splitlines():
        m = re.match(r"^vllm:num_requests_(running|waiting)\{[^}]*\}\s+([0-9.eE+-]+)$", line)
        if m:
            n += float(m.group(2))
    return n


def _gpu_used_mb():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30, check=True)
        return sum(float(x) for x in r.stdout.split())
    except Exception:
        return None


def _healthy(c):
    try:
        with urllib.request.urlopen(c["health_url"], timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def leasable_mb(ci_root=None):
    """GPU memory the lease can free now (gate G7_resources): holds_mb while the service runs, else 0 (also for a
    configuration that cannot be read: the gate then decides on the free memory alone)."""
    try:
        c = config(ci_root)
        return int(c.get("holds_mb", 0)) if c and service_running(c) else 0
    except (Exception, SystemExit):
        return 0


def _stop(c, d):
    t0 = time.time()
    while time.time() - t0 < float(c.get("drain_s", 900)):
        n = _in_flight(c)
        if not n:
            break
        time.sleep(10)
    n = _in_flight(c)
    code, out = _ctl(c, "stop")
    _log(d, "stopped %s (in flight at stop: %s; supervisorctl %d: %s)" % (c["service"], n, code, out))
    if code != 0 and service_running(c):
        raise SystemExit("gpu lease: could not stop %s: %s" % (c["service"], out))
    t0 = time.time()
    while time.time() - t0 < float(c.get("release_wait_s", 120)):     # the memory is released a little after the exit
        used = _gpu_used_mb()
        if used is None or used < float(c.get("released_below_mb", 2000)):      # no nvidia-smi: nothing to wait for
            break
        time.sleep(3)


def _is_watchdog(pid):
    try:
        with open("/proc/%d/cmdline" % int(pid), "rb") as f:
            return b"ci.gpu_lease" in f.read()
    except (OSError, ValueError):
        return _alive(pid) if not os.path.exists("/proc") else False


def _start_watchdog(c, d):
    if not c.get("watchdog", True):                                     # tests call watch() themselves
        return
    pf = os.path.join(d, "watchdog.pid")
    if os.path.exists(pf) and _is_watchdog(open(pf).read().strip() or -1):
        return
    log = open(os.path.join(d, "watchdog.out"), "a")
    p = subprocess.Popen([sys.executable, "-m", "ci.gpu_lease", "watch"], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, env=dict(os.environ, CI_ROOT=root()))
    with open(pf, "w") as f:
        f.write(str(p.pid))


def _register(d, name, pid):
    with open(os.path.join(d, "holders", name + ".json"), "w") as f:
        json.dump(dict(pid=int(pid), start=_start_time(pid), since=time.time()), f)


def acquire(holder, pid=None, ci_root=None):
    """Returns a record of what was done (for the stage report); no-op without a configuration. The holder and the
    state "stopped by the lease" are recorded and the watchdog is running before the service is stopped, so a stop
    that fails half way is still restored once the holder is gone."""
    c = config(ci_root)
    if not c:
        return dict(lease="none")
    d = _dir(ci_root)
    with _locked(d):
        _register(d, holder, pid or os.getpid())
        st = _state(d)
        running = service_running(c)
        if running:
            st["stopped_by_lease"] = True
        st["last_acquire"] = time.time()
        st.pop("restore_failed", None)
        _save_state(d, st)
        _log(d, "acquired by %s" % holder)
        _start_watchdog(c, d)
    if running:                    # the drain and stop outside the state lock (a registered holder keeps the watchdog off)
        with open(os.path.join(d, "stop.lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if service_running(c):
                _stop(c, d)
    return dict(lease=c["service"], stopped_now=running, gpu_used_mb_after=_gpu_used_mb())


def add(holder, pid, ci_root=None):
    """Register a process of the holder (a training or scoring child): the lease stays held while it lives, even if
    the holder itself dies."""
    c = config(ci_root)
    if c:
        d = _dir(ci_root)
        with _locked(d):
            _register(d, "%s@%d" % (holder, int(pid)), pid)


def release(holder, ci_root=None):
    c = config(ci_root)
    if not c:
        return
    d = _dir(ci_root)
    with _locked(d):
        hd = os.path.join(d, "holders")
        for n in os.listdir(hd):
            p = os.path.join(hd, n)
            if n == holder + ".json":
                os.remove(p)
            elif n.startswith(holder + "@"):                               # a child still running keeps its registration
                try:
                    h = json.load(open(p))
                except (OSError, ValueError):
                    h = {}
                if not _holder_alive(h):
                    os.remove(p)
        st = _state(d)
        st["last_release"] = time.time()
        _save_state(d, st)
        _log(d, "released by %s" % holder)


class Held(dict):
    """The record of an acquired lease; add(pid) registers a child process of the holder."""

    def __init__(self, holder, rec):
        super().__init__(rec)
        self.holder = holder

    def add(self, pid):
        if self.get("lease") not in (None, "none") and not str(self.get("lease", "")).startswith("not taken"):
            add(self.holder, pid)


@contextlib.contextmanager
def hold(holder, enabled=True):
    """The lease for the duration of a with block; enabled=False (a profile's train.gpu_lease: false, e.g. a tiny run
    that fits beside the service) records that it was not taken."""
    if not enabled:
        yield Held(holder, dict(lease="not taken (train.gpu_lease: false)"))
        return
    try:
        rec = acquire(holder)
    except BaseException:
        release(holder)
        raise
    try:
        yield Held(holder, rec)
    finally:
        release(holder)


def watch(ci_root=None, poll=30, once=False):
    """Restore the service once no live holder is left for grace_s seconds; exit when nothing is left to restore
    (once: a single check). A failed start is retried; a restore interrupted by a new acquire is taken up again."""
    c = config(ci_root)
    if not c:
        return
    d = _dir(ci_root)
    pf = os.path.join(d, "watchdog.pid")
    failures = 0
    try:
        while True:
            time.sleep(poll)
            with _locked(d):
                hs = _holders(d)
                st = _state(d)
                if not hs and not st.get("stopped_by_lease"):
                    _log(d, "watchdog: nothing to restore")
                    if os.path.exists(pf) and open(pf).read().strip() == str(os.getpid()):
                        os.remove(pf)                                      # under the lock: a new acquire starts a new watchdog
                    return
                now = time.time()
                if hs:
                    st.pop("empty_since", None)
                elif "empty_since" not in st:
                    st["empty_since"] = now                                # grace counts from when the last holder went away
                _save_state(d, st)
                due = not hs and now - max(float(st.get("empty_since", now)), float(st.get("last_release", 0))) >= float(c.get("grace_s", 600))
                started = False
                if due:
                    code, out = _ctl(c, "start")
                    _log(d, "start %s: supervisorctl %d: %s" % (c["service"], code, out))
                    if code == 0 or service_running(c):
                        st["stopped_by_lease"] = False
                        st.pop("empty_since", None)
                        _save_state(d, st)
                        started, failures = True, 0
                    else:
                        failures += 1
                        if failures >= int(c.get("start_attempts", 10)):
                            st["restore_failed"] = "%d failed starts, last: %s" % (failures, out)
                            _save_state(d, st)
                            _log(d, "watchdog: giving up after %d failed starts; the service stays stopped (status shows restore_failed)" % failures)
                            if os.path.exists(pf) and open(pf).read().strip() == str(os.getpid()):
                                os.remove(pf)
                            return
            if not started:
                if once:
                    return
                if due:
                    time.sleep(min(300, poll * (2 ** min(failures, 4))))     # back off between failed starts
                continue
            t0 = time.time()
            while time.time() - t0 < float(c.get("restore_wait_s", 1200)):
                if _healthy(c):
                    _log(d, "%s healthy after %.0f s" % (c["service"], time.time() - t0))
                    break
                with _locked(d):
                    if _state(d).get("stopped_by_lease"):                  # a run took the GPU again meanwhile
                        break
                time.sleep(10)
            else:
                _log(d, "%s not healthy after %.0f s (supervisor keeps trying)" % (c["service"], time.time() - t0))
            if once:
                return
    finally:
        try:
            if os.path.exists(pf) and open(pf).read().strip() == str(os.getpid()):
                os.remove(pf)
        except OSError:
            pass


def status(ci_root=None):
    c = config(ci_root)
    if not c:
        return dict(lease="none")
    d = _dir(ci_root)
    with _locked(d):
        return dict(service=c["service"], running=service_running(c), holders=sorted(_holders(d)), state=_state(d), gpu_used_mb=_gpu_used_mb())


def main(argv):
    if not argv or argv[0] not in ("status", "acquire", "release", "watch"):
        raise SystemExit(__doc__)
    if argv[0] == "status":
        print(json.dumps(status(), indent=1, sort_keys=True))
    elif argv[0] == "acquire":
        if len(argv) < 3:
            raise SystemExit("acquire HOLDER PID: the process that holds the lease (the CLI itself ends at once)")
        print(json.dumps(acquire(argv[1], pid=int(argv[2])), sort_keys=True))
    elif argv[0] == "release":
        release(argv[1])
    else:
        watch()


if __name__ == "__main__":
    main(sys.argv[1:])
