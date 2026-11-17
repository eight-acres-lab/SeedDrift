"""Helpers shared by the stage scripts."""
from __future__ import annotations

import gzip
import json
import os
import sys
import time

import yaml

from ci import ROOT
from ci.gates.g6_clean import imports_under
from ci.hashing import sha256_file
from ci.record import stage_report


class Stage:
    """with Stage(run_dir, "corpus") as st: ... st.output(name, path) ...  writes the stage report at exit."""

    def __init__(self, run_dir: str, name: str, key: str = "", requires=()):
        self.run_dir, self.name, self.key = run_dir, name, key
        self.requires = requires
        self.outputs = []
        self.extra = {}
        self.t0 = time.time()
        with open(os.path.join(run_dir, "config.resolved.yaml")) as f:
            self.config = yaml.safe_load(f)

    def __enter__(self):
        bad = imports_under(ROOT)
        if bad:
            raise SystemExit("G6: modules loaded from outside the worktree: %s" % bad[:5])
        for s in self.requires:                       # a stage never runs without the gates behind it, even outside Snakemake
            if not os.path.exists(os.path.join(self.run_dir, "gates", s + ".ok")):
                raise SystemExit("stage %s needs gates/%s.ok" % (self.name, s))
        return self

    def __exit__(self, exc_type, exc, tb):
        stage_report(self.run_dir, self.name, self.key, self.t0, self.outputs, exit_code=0 if exc_type is None else 1, **self.extra)
        return False

    def path(self, *parts) -> str:
        p = os.path.join(self.run_dir, "data", *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def repo(self, rel: str) -> str:
        return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)

    def output(self, name: str, path: str) -> None:
        self.outputs.append(dict(name=name, path=os.path.relpath(path, self.run_dir), sha256=sha256_file(path), bytes=os.path.getsize(path)))


def read_fasta(path: str) -> dict:
    out, name, buf = {}, None, []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name is not None:
        out[name] = "".join(buf)
    return out


def write_fasta(path: str, records) -> None:
    tmp = path + ".part"
    with open(tmp, "w") as f:
        for name, seq in records:
            f.write(">%s\n%s\n" % (name, seq))
    os.replace(tmp, path)


def write_json(path: str, obj) -> None:
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def main_args():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m ci.stages.<name> RUN_DIR")
    return sys.argv[1]


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_select_spec(path, repo=None, files=None):
    """A select spec (configs/select/<name>.yaml); `base: <spec>` takes that spec and overrides it key by key (mappings
    merged, everything else replaced), so a variant states only what differs. `files` (a list) receives every file read,
    so gates can scan them and the record can hash them."""
    repo = repo or (lambda rel: rel if os.path.isabs(rel) else os.path.join(ROOT, rel))
    with open(repo(path)) as f:
        spec = yaml.safe_load(f)
    seen = [path]
    while spec.get("base"):
        b = spec.pop("base")
        if b in seen:
            raise SystemExit("select spec %s: base loop %s" % (path, seen + [b]))
        seen.append(b)
        with open(repo(b)) as f:
            spec = _merge(yaml.safe_load(f), spec)
    if files is not None:
        files.extend(seen)
    return spec
