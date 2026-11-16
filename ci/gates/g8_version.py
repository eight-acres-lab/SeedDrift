"""G8: every change to the CI is a new CI version.

check(root): if files under ci/, configs/ or deploy/envs/ differ from the latest ci-v* tag reachable from HEAD, then
ci/VERSION must differ from that tag's VERSION; a run's commit must descend from a ci-v* tag. Before the first tag
(bootstrap) the gate passes with a note, and VERSION must read 0.1.
"""
from __future__ import annotations

import os
import re
import subprocess

from ci import ROOT
from ci.gates import GateResult
from ci.hashing import sha256_json

WATCHED = ("ci/", "configs/", "deploy/envs/", "seeddrift/train/", "bench/hard/EXCLUDE.tsv")


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def check(root: str = ROOT) -> GateResult:
    problems = []
    with open(os.path.join(root, "ci", "VERSION")) as f:
        version = f.read().strip()
    if not re.fullmatch(r"\d+\.\d+", version):
        problems.append(dict(error="ci/VERSION must be MAJOR.MINOR, found %r" % version))
    tag = _git(root, "describe", "--tags", "--abbrev=0", "--match", "ci-v*", "HEAD")
    if tag.returncode != 0:
        if _git(root, "tag", "-l", "ci-v*").stdout.strip():
            problems.append(dict(error="ci-v* tags exist but none is an ancestor of HEAD"))
            return GateResult("G8", "fail", dict(version=version, problems=problems), sha256_json([version, "no ancestor tag"]))
        note = "bootstrap: no ci-v* tag in this repository"
        if version != "0.1":
            problems.append(dict(error="before the first tag VERSION must be 0.1, found %s" % version))
        return GateResult("G8", "fail" if problems else "pass", dict(note=note, version=version, problems=problems), sha256_json([version, "bootstrap"]))
    tag = tag.stdout.strip()
    changed = [p for p in _git(root, "diff", "--name-only", "-z", tag, "HEAD").stdout.split("\0") if p]
    watched = [p for p in changed if p.startswith(WATCHED)]
    old = _git(root, "show", "%s:ci/VERSION" % tag).stdout.strip()
    if watched and old == version:
        problems.append(dict(error="CI files changed since %s but ci/VERSION is still %s" % (tag, version), files=watched[:30]))
    return GateResult("G8", "fail" if problems else "pass", dict(tag=tag, version=version, tag_version=old, changed_ci_files=len(watched), problems=problems),
                      sha256_json([tag, version, watched]))
