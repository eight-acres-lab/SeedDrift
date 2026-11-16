"""G6: runs only from clean, committed code.

check(worktree, sha): HEAD is the requested commit, nothing is modified or untracked, and (on the server) the worktree
lies under the CI root's wt/ and is read-only. imports_under(worktree) is called at the start of every Python stage:
each loaded module of this repository (seeddrift, ci, bench, research) must come from the worktree, so a module
uploaded by hand elsewhere on the path cannot be used.
"""
from __future__ import annotations

import os
import subprocess
import sys

from ci.gates import GateResult
from ci.hashing import sha256_json

OWN = ("seeddrift", "ci", "bench", "research")


def _git(worktree, *args):
    return subprocess.run(["git", *args], cwd=worktree, capture_output=True, text=True)


IGNORED_OK = ("runs/", "__pycache__/", ".pytest_cache/", ".snakemake/")


def _writable(path):
    import stat
    for d, dirs, files in os.walk(path):
        for name in files[:20] + dirs[:5]:
            if os.lstat(os.path.join(d, name)).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                return os.path.join(d, name)
        break
    return ""


def check(worktree: str, sha: str, ci_root: str = "", require_readonly: bool = False) -> GateResult:
    """sha: the commit the launcher asked for (not read back from this worktree)."""
    problems = []
    head = _git(worktree, "rev-parse", "HEAD")
    if head.returncode != 0:
        problems.append(dict(error="not a git worktree: %s" % worktree))
    else:
        h = head.stdout.strip()
        if sha and not h.startswith(sha) and not sha.startswith(h):
            full = _git(worktree, "rev-parse", sha)
            if full.returncode != 0 or full.stdout.strip() != h:
                problems.append(dict(error="HEAD %s is not the requested commit %s" % (h, sha)))
        status = _git(worktree, "status", "--porcelain", "--untracked-files=all")
        if status.stdout.strip():
            problems.append(dict(error="worktree not clean", files=status.stdout.strip().splitlines()[:50]))
        if _git(worktree, "diff-index", "--quiet", "HEAD").returncode != 0:
            problems.append(dict(error="index differs from HEAD"))
        ignored = [l[3:] for l in _git(worktree, "status", "--porcelain", "--ignored").stdout.splitlines() if l.startswith("!! ")]
        stray = [p for p in ignored if not any(p == ok or p.startswith(ok) or ("/" + ok) in ("/" + p) for ok in IGNORED_OK)]
        if stray:
            problems.append(dict(error="ignored files inside the worktree", files=stray[:30]))
    if ci_root:
        wt = os.path.realpath(worktree)
        if not wt.startswith(os.path.realpath(os.path.join(ci_root, "wt")) + os.sep):
            problems.append(dict(error="worktree %s is not under %s/wt" % (wt, ci_root)))
    if require_readonly:
        w = _writable(worktree) or _writable(os.path.join(worktree, "ci"))
        if w:
            problems.append(dict(error="worktree is writable (write bit on %s)" % w))
    return GateResult("G6", "fail" if problems else "pass", dict(worktree=worktree, sha=sha, problems=problems), sha256_json([worktree, sha]))


def imports_under(worktree: str) -> list:
    """Modules of this repository loaded from outside the worktree (should be empty)."""
    root = os.path.realpath(worktree)
    bad = []
    for name, mod in list(sys.modules.items()):
        if name.split(".")[0] not in OWN:
            continue
        f = getattr(mod, "__file__", None)
        if f and not os.path.realpath(f).startswith(root + os.sep):
            bad.append(dict(module=name, file=f))
    return bad
