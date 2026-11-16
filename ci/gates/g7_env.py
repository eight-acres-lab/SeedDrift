"""G7: the environment is the locked one, and the machine can take the run.

lock_matches(exported, lock): the packages installed (name, version, build from `micromamba env export`) equal those of
ci/envs/ci.lock.yml. resources(...): temporary files and caches live under the allowed data root, the disk has room for
the run's estimate plus a reserve, and a GPU with enough free memory is visible when the profile needs one.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import yaml

from ci import ROOT
from ci.gates import GateResult
from ci.hashing import sha256_file, sha256_json

LOCK = os.path.join(ROOT, "ci", "envs", "ci.lock.yml")


def packages(text: str) -> set:
    """(name, version, build) of the conda packages and (name, version, "pip") of pip packages in an env export."""
    d = yaml.safe_load(text) or {}
    out = set()
    for dep in d.get("dependencies", []):
        if isinstance(dep, str):
            parts = dep.split("=")
            if len(parts) >= 3:
                out.add((parts[0].split("::")[-1], parts[1], parts[2]))
        elif isinstance(dep, dict):
            for p in dep.get("pip", []):
                name, _, ver = p.partition("==")
                out.add((name.lower(), ver, "pip"))
    return out


def lock_matches(exported: str, lock_path: str = LOCK) -> GateResult:
    with open(lock_path) as f:
        want = packages(f.read())
    have = packages(exported)
    missing, extra = sorted(want - have), sorted(have - want)
    ok = bool(want) and bool(have) and not missing and not extra
    return GateResult("G7", "pass" if ok else "fail", dict(check="lock", lock=os.path.relpath(lock_path, ROOT), missing=missing[:50], extra=extra[:50]),
                      sha256_file(lock_path))


def export_env(prefix: str, micromamba: str = "micromamba") -> str:
    out = subprocess.run([micromamba, "env", "export", "-p", prefix, "--no-md5"], capture_output=True, text=True, check=True)
    return out.stdout


def gpu_free_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=30)
        total, used = (int(x) for x in out.stdout.strip().splitlines()[0].split(","))
        return total - used
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def resources(data_root: str, need_disk_gb: float, reserve_gb: float = 30.0, root_min_gb: float = 1.0, gpu_mem_min_mb: int = 0) -> GateResult:
    problems = []
    for var in ("TMPDIR", "XDG_CACHE_HOME", "PIP_CACHE_DIR"):
        v = os.environ.get(var, "")
        root = os.path.realpath(data_root)
        if not v or not (os.path.realpath(v) == root or os.path.realpath(v).startswith(root + os.sep)):
            problems.append(dict(var=var, value=v, must_be_under=data_root))
    free_gb = shutil.disk_usage(data_root).free / 1e9
    if free_gb - need_disk_gb < reserve_gb:
        problems.append(dict(disk_free_gb=round(free_gb, 1), need_gb=need_disk_gb, reserve_gb=reserve_gb))
    root_free = shutil.disk_usage("/").free / 1e9
    if root_free < root_min_gb:
        problems.append(dict(root_free_gb=round(root_free, 2), min_gb=root_min_gb))
    gpu = gpu_free_mb()
    if gpu_mem_min_mb and (gpu is None or gpu < gpu_mem_min_mb):
        problems.append(dict(gpu_free_mb=gpu, need_mb=gpu_mem_min_mb))
    ev = dict(check="resources", disk_free_gb=round(free_gb, 1), root_free_gb=round(root_free, 2), gpu_free_mb=gpu, problems=problems)
    return GateResult("G7", "fail" if problems else "pass", ev, sha256_json([data_root, need_disk_gb, reserve_gb, gpu_mem_min_mb]))
