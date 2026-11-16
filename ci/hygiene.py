"""T0 hygiene checks of the repository (no data, seconds):

- nothing tracked in the internal notes folder and no tracked file names it (the patterns are in ci/denylist.txt);
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
BINARY = (".pt", ".pth", ".ckpt", ".safetensors", ".h5", ".onnx", ".bin", ".npy", ".npz", ".parquet", ".dmnd", ".pkl", ".zip", ".tar")
EXEMPT = (".gitignore", "ci/denylist.txt")
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
        if p.lower().startswith("strategy/") or "/strategy/" in p.lower():
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
        if size <= MAX_BYTES and not p.endswith((".png", ".gz")) and p not in EXEMPT:
            try:
                with open(full, "rb") as f:
                    raw = f.read()
            except OSError:
                continue
            if b"\0" in raw[:8192]:
                problems.append(dict(file=p, error="binary file tracked"))
                continue
            text = raw.decode("utf-8", errors="replace")
            for rx in deny:
                if rx.search(text):
                    problems.append(dict(file=p, error="matches denylist pattern %s" % rx.pattern))
    problems += gate_tests(root)
    return problems


def _expects_failure(fn) -> bool:
    """The test body asserts a failing status (compares with "fail") or expects an exception (pytest.raises)."""
    import ast
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare) and any(isinstance(c, ast.Constant) and c.value == "fail" for c in [node.left] + node.comparators):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "raises":
            return True
    return False


def gate_tests(root=ROOT):
    """Every ci/gates/gN_*.py has tests/ci/test_gN_*.py with a test named *fail* that really expects a failure."""
    import ast
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
            tree = ast.parse(f.read())
        fails = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_") and "fail" in n.name]
        if not any(_expects_failure(n) for n in fails):
            problems.append(dict(gate=name, error="no test named *fail* that asserts a failing gate"))
    return problems
