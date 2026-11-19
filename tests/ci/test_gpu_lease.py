"""ci/gpu_lease.py with a fake supervisorctl: no configuration is a no-op; with one, acquiring stops the service,
releasing and the watchdog start it again only after the grace period, and a dead holder does not keep the lease."""
import json
import os
import stat
import subprocess
import sys

from ci import gpu_lease


def _fake(tmp, state="RUNNING"):
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    (tmp / "svc").write_text(state)
    script = bin_dir / "supervisorctl"
    script.write_text("#!/bin/sh\n"
                      "echo \"$@\" >> %s/calls\n"
                      "case \"$1\" in\n"
                      "  status) echo \"$2 $(cat %s/svc)\";;\n"
                      "  stop) echo STOPPED > %s/svc; echo \"$2: stopped\";;\n"
                      "  start) echo RUNNING > %s/svc; echo \"$2: started\";;\n"
                      "esac\n" % ((tmp,) * 4))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _config(tmp, **kw):
    (tmp / "host").mkdir(exist_ok=True)
    c = dict(service="qwen-x", metrics_url="http://127.0.0.1:9/metrics", health_url="http://127.0.0.1:9/health", holds_mb=30000,
             drain_s=0, grace_s=0, restore_wait_s=0, release_wait_s=0, watchdog=False)
    c.update(kw)
    (tmp / "host" / "gpu_lease.json").write_text(json.dumps(c))


def test_no_configuration_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_ROOT", str(tmp_path))
    assert gpu_lease.acquire("r1") == dict(lease="none")
    assert gpu_lease.leasable_mb() == 0
    with gpu_lease.hold("r1") as rec:
        assert rec == dict(lease="none")
    with gpu_lease.hold("r1", enabled=False) as rec:
        assert "not taken" in rec["lease"]
    assert not (tmp_path / "locks").exists()


def test_bad_service_name_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_ROOT", str(tmp_path))
    _config(tmp_path, service="x; rm -rf /")
    try:
        gpu_lease.config()
    except SystemExit:
        return
    raise AssertionError("a service name with shell characters was accepted")


def test_stop_then_restore_after_release(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_ROOT", str(tmp_path))
    monkeypatch.setenv("PATH", str(_fake(tmp_path)) + os.pathsep + os.environ["PATH"])
    _config(tmp_path)
    assert gpu_lease.leasable_mb() == 30000
    rec = gpu_lease.acquire("r1")
    assert rec["stopped_now"] and (tmp_path / "svc").read_text().strip() == "STOPPED"
    assert gpu_lease.leasable_mb() == 0
    rec2 = gpu_lease.acquire("r2")                                     # a second holder: nothing to stop
    assert not rec2["stopped_now"]
    gpu_lease.release("r1")
    gpu_lease.watch(poll=0, once=True)                                  # r2 still holds it: not restored
    assert (tmp_path / "svc").read_text().strip() == "STOPPED"
    gpu_lease.release("r2")
    _config(tmp_path, grace_s=3600)
    gpu_lease.watch(poll=0, once=True)                                  # released, but within the grace period
    assert (tmp_path / "svc").read_text().strip() == "STOPPED"
    _config(tmp_path)
    gpu_lease.watch(poll=0)
    assert (tmp_path / "svc").read_text().strip() == "RUNNING"


def test_watch_restores_only_without_live_holders(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_ROOT", str(tmp_path))
    monkeypatch.setenv("PATH", str(_fake(tmp_path)) + os.pathsep + os.environ["PATH"])
    _config(tmp_path)
    gpu_lease.acquire("r1")
    assert (tmp_path / "svc").read_text().strip() == "STOPPED"
    gpu_lease.release("r1")
    gpu_lease.watch(poll=0)
    assert (tmp_path / "svc").read_text().strip() == "RUNNING"
    calls = (tmp_path / "calls").read_text().split("\n")
    assert "stop qwen-x" in calls and "start qwen-x" in calls


def test_dead_holder_is_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_ROOT", str(tmp_path))
    monkeypatch.setenv("PATH", str(_fake(tmp_path)) + os.pathsep + os.environ["PATH"])
    _config(tmp_path)
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    gpu_lease.acquire("dead", pid=p.pid)                                # registered with a pid that no longer runs
    gpu_lease.watch(poll=0)
    assert (tmp_path / "svc").read_text().strip() == "RUNNING"
    assert gpu_lease.status()["holders"] == []


def test_a_live_child_keeps_the_lease_after_its_holder_died(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_ROOT", str(tmp_path))
    monkeypatch.setenv("PATH", str(_fake(tmp_path)) + os.pathsep + os.environ["PATH"])
    _config(tmp_path)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        gpu_lease.acquire("run", pid=dead.pid)
        gpu_lease.add("run", child.pid)
        gpu_lease.watch(poll=0, once=True)
        assert (tmp_path / "svc").read_text().strip() == "STOPPED"     # the training child still runs
    finally:
        child.kill()
        child.wait()
    gpu_lease.watch(poll=0)
    assert (tmp_path / "svc").read_text().strip() == "RUNNING"


def test_a_failed_start_is_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_ROOT", str(tmp_path))
    bin_dir = _fake(tmp_path)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    _config(tmp_path)
    gpu_lease.acquire("r1")
    gpu_lease.release("r1")
    good = (bin_dir / "supervisorctl").read_text()
    (bin_dir / "supervisorctl").write_text(good.replace("start) echo RUNNING > %s/svc; echo \"$2: started\";;" % tmp_path,
                                                        "start) echo \"$2: ERROR (spawn error)\"; exit 7;;"))
    gpu_lease.watch(poll=0, once=True)
    assert (tmp_path / "svc").read_text().strip() == "STOPPED" and gpu_lease.status()["state"]["stopped_by_lease"]
    (bin_dir / "supervisorctl").write_text(good)
    gpu_lease.watch(poll=0)
    assert (tmp_path / "svc").read_text().strip() == "RUNNING" and not gpu_lease.status()["state"]["stopped_by_lease"]


def test_backoff_counts_as_running(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_ROOT", str(tmp_path))
    monkeypatch.setenv("PATH", str(_fake(tmp_path, state="BACKOFF")) + os.pathsep + os.environ["PATH"])
    _config(tmp_path)
    assert gpu_lease.acquire("r1")["stopped_now"] and (tmp_path / "svc").read_text().strip() == "STOPPED"


def test_unreadable_configuration_counts_nothing_as_leasable(tmp_path):
    (tmp_path / "host").mkdir()
    (tmp_path / "host" / "gpu_lease.json").write_text("{not json")
    assert gpu_lease.leasable_mb(str(tmp_path)) == 0
