"""What the stages of package W6 share: the model interpreter with its locked environment, the environment of the
processes started in it, their logs. The environment check is the one of the old line's training stage, imported
read-only; nothing of that stage is changed or extended, and the GPU lease is never imported."""
from __future__ import annotations

import os
import subprocess

from ci import ROOT
from ci.stages.train import MODEL_LOCK, gpu_free_mb, model_env  # noqa: F401 - read-only imports; gpu_free_mb is used by the stages

ENTRY = "seeddrift.train.rel.trainer.run"


def interpreter(st, cfg):
    """(python, lock file, its packages): the model interpreter of the profile, whose packages equal its lock."""
    tc = cfg.get("train") or {}
    if tc.get("gpu_lease") is not False:
        raise SystemExit("a profile of the relational line sets train.gpu_lease: false (this line never takes the GPU lease)")
    if tc.get("device", "auto") not in ("auto", "cpu"):
        raise SystemExit("train.device is auto or cpu, not %r" % tc.get("device"))
    py, lock = tc.get("python", "/usr/local/bin/python3"), tc.get("python_lock", MODEL_LOCK)
    return py, lock, model_env(st, py, lock)


def environment(threads, hide_gpu):
    """The environment of a process of the model interpreter: only this worktree on the module path, `threads`
    threads, and no GPU when hide_gpu."""
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(int(threads))
    if hide_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def start(py, mode, conf, log, env):
    """Starts `python -m <trainer> mode conf` in the worktree; returns the process and its open log."""
    lg = open(log, "a")
    return subprocess.Popen([py, "-m", ENTRY, mode, conf], cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT, env=env), lg


def stop(proc, lg, grace=30):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    lg.close()


def tail(path, chars=1500):
    try:
        with open(path, errors="replace") as f:
            return f.read()[-chars:]
    except OSError:
        return ""
