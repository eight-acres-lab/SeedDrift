"""T0 hygiene checks of the repository (no data, seconds):

- nothing tracked under strategy/ and no tracked file mentions it (the internal notes stay out of the public repo);
- no tracked file above the size limit, no model or array files tracked;
- run summaries in runs-index/ stay small and carry no home paths;
- every gate module has a test file with at least one test that expects the gate to fail.
"""
from __future__ import annotations

import os
import re
import subprocess

from ci import ROOT

MAX_BYTES = 2_000_000
BINARY = (".pt", ".ckpt", ".npy", ".npz", ".parquet", ".dmnd", ".pkl")
ALLOW_LARGE = ()
SUMMARY_MAX = 200_000


def tracked(root=ROOT):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def denylist(root=ROOT):
    path = os.path.join(root, "ci", "denylist.txt")
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def check(root=ROOT):
    problems = []
    files = tracked(root)
    deny = [re.compile(p) for p in denylist(root)]
    for p in files:
        full = os.path.join(root, p)
        if p.startswith("strategy/"):
            problems.append(dict(file=p, error="internal notes are tracked"))
        if not os.path.exists(full):
            continue
        size = os.path.getsize(full)
        if size > MAX_BYTES and p not in ALLOW_LARGE:
            problems.append(dict(file=p, error="larger than %d bytes" % MAX_BYTES))
        if p.endswith(BINARY):
            problems.append(dict(file=p, error="model or array file tracked"))
        if p.startswith("runs-index/") and size > SUMMARY_MAX:
            problems.append(dict(file=p, error="run summary larger than %d bytes" % SUMMARY_MAX))
        if size <= MAX_BYTES and not p.endswith((".png", ".gz")) and p != "ci/denylist.txt":
            try:
                with open(full, encoding="utf-8") as f:
                    text = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            for rx in deny:
                if rx.search(text):
                    problems.append(dict(file=p, error="matches denylist pattern %s" % rx.pattern))
    problems += gate_tests(root)
    return problems


def gate_tests(root=ROOT):
    """Every ci/gates/gN_*.py has tests/ci/test_gN_*.py with a test whose name contains 'fail'."""
    problems = []
    gdir = os.path.join(root, "ci", "gates")
    for name in sorted(os.listdir(gdir)):
        if not re.match(r"g\d+_.*\.py$", name):
            continue
        test = os.path.join(root, "tests", "ci", "test_" + name)
        if not os.path.exists(test):
            problems.append(dict(gate=name, error="no test file tests/ci/test_%s" % name))
            continue
        with open(test) as f:
            text = f.read()
        if not re.search(r"def test_\w*fail\w*\(", text):
            problems.append(dict(gate=name, error="no test that expects the gate to fail"))
    return problems
