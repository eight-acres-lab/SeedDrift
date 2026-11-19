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

  python -m ci.gpu_lease status|acquire HOLDER|release HOLDER|watch
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
        if _alive(h.get("pid", -1)):
            out[n[:-5]] = h
        else:
            os.remove(p)
            _log(d, "dropped dead holder %s (pid %s)" % (n[:-5], h.get("pid")))
    return out


def _ctl(c, action):
    r = subprocess.run(["supervisorctl", action, c["service"]], capture_output=True, text=True, timeout=180)
    return r.returncode, (r.stdout + r.stderr).strip()


def service_running(c):
    _, out = _ctl(c, "status")
    return bool(re.search(r"\b(RUNNING|STARTING)\b", out))


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
    """GPU memory the lease can free now (gate G7_resources): holds_mb while the service runs, else 0."""
    c = config(ci_root)
    if not c:
        return 0
    try:
        return int(c.get("holds_mb", 0)) if service_running(c) else 0
    except Exception:
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


def _start_watchdog(c, d):
    if not c.get("watchdog", True):                                     # tests call watch() themselves
        return
    pf = os.path.join(d, "watchdog.pid")
    if os.path.exists(pf) and _alive(open(pf).read().strip() or -1):
        return
    log = open(os.path.join(d, "watchdog.out"), "a")
    p = subprocess.Popen([sys.executable, "-m", "ci.gpu_lease", "watch"], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, env=dict(os.environ, CI_ROOT=root()))
    with open(pf, "w") as f:
        f.write(str(p.pid))


def acquire(holder, pid=None, ci_root=None):
    """Returns a record of what was done (for the stage report); no-op without a configuration."""
    c = config(ci_root)
    if not c:
        return dict(lease="none")
    d = _dir(ci_root)
    with _locked(d):
        with open(os.path.join(d, "holders", holder + ".json"), "w") as f:
            json.dump(dict(pid=int(pid or os.getpid()), since=time.time()), f)
        st = _state(d)
        stopped_now = False
        if service_running(c):
            _stop(c, d)
            st["stopped_by_lease"] = True
            stopped_now = True
        st["last_acquire"] = time.time()
        _save_state(d, st)
        _log(d, "acquired by %s" % holder)
    _start_watchdog(c, d)
    return dict(lease=c["service"], stopped_now=stopped_now, gpu_used_mb_after=_gpu_used_mb())


def release(holder, ci_root=None):
    c = config(ci_root)
    if not c:
        return
    d = _dir(ci_root)
    with _locked(d):
        p = os.path.join(d, "holders", holder + ".json")
        if os.path.exists(p):
            os.remove(p)
        st = _state(d)
        st["last_release"] = time.time()
        _save_state(d, st)
        _log(d, "released by %s" % holder)


@contextlib.contextmanager
def hold(holder, enabled=True):
    """The lease for the duration of a with block; enabled=False (a profile's train.gpu_lease: false, e.g. a tiny run
    that fits beside the service) records that it was not taken."""
    if not enabled:
        yield dict(lease="not taken (train.gpu_lease: false)")
        return
    rec = acquire(holder)
    try:
        yield rec
    finally:
        release(holder)


def watch(ci_root=None, poll=30, once=False):
    """Restore the service once no live holder is left for grace_s seconds, then exit (once: one check only)."""
    c = config(ci_root)
    if not c:
        return
    d = _dir(ci_root)
    while True:
        time.sleep(poll)
        with _locked(d):
            hs = _holders(d)
            st = _state(d)
            if not hs and not st.get("stopped_by_lease"):
                _log(d, "watchdog: nothing to restore")
                return
            due = not hs and time.time() - max(float(st.get("last_release", 0)), float(st.get("last_acquire", 0))) >= float(c.get("grace_s", 600))
            if due:
                code, out = _ctl(c, "start")
                _log(d, "started %s (supervisorctl %d: %s)" % (c["service"], code, out))
                st["stopped_by_lease"] = False
                _save_state(d, st)
        if not due:
            if once:
                return
            continue
        t0 = time.time()
        while time.time() - t0 < float(c.get("restore_wait_s", 1200)):
            if _healthy(c):
                _log(d, "%s healthy after %.0f s" % (c["service"], time.time() - t0))
                return
            time.sleep(10)
        _log(d, "%s not healthy after %.0f s (supervisor keeps trying)" % (c["service"], time.time() - t0))
        return


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
        print(json.dumps(acquire(argv[1], pid=int(argv[2]) if len(argv) > 2 else None), sort_keys=True))
    elif argv[0] == "release":
        release(argv[1])
    else:
        watch()


if __name__ == "__main__":
    main(sys.argv[1:])
