"""Hashes used everywhere in the CI.

canonical_json   one byte string per JSON value (sorted keys, no spaces, UTF-8), so equal values hash equally
sha256_file      streaming sha256; for big files a sidecar <file>.sha256.json (size, mtime_ns, sha256) avoids rehashing
stage_key        the identity of a stage result: rule, rule version, code, parameters, inputs, environment
pipeline_hash    one hash over the files that define the pipeline (Snakefile, rules, gates, envs, profiles, configs)
"""
from __future__ import annotations

import hashlib
import json
import os

CHUNK = 1 << 20
SIDECAR_MIN = 64 << 20          # files at least this large get a sidecar


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(canonical_json(obj))


def _stream(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_file(path: str, use_sidecar: bool = True) -> str:
    """sha256 of a file. Large files keep a sidecar keyed by size and mtime; a stale or unreadable sidecar is ignored."""
    st = os.stat(path)
    side = path + ".sha256.json"
    if use_sidecar and st.st_size >= SIDECAR_MIN and os.path.exists(side):
        try:
            with open(side) as f:
                d = json.load(f)
            if d.get("size") == st.st_size and d.get("mtime_ns") == st.st_mtime_ns and len(d.get("sha256", "")) == 64:
                return d["sha256"]
        except (OSError, ValueError):
            pass
    digest = _stream(path)
    if use_sidecar and st.st_size >= SIDECAR_MIN:
        try:
            tmp = side + ".part"
            with open(tmp, "w") as f:
                json.dump({"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}, f)
            os.replace(tmp, side)
        except OSError:
            pass
    return digest


def files_hash(root: str, relpaths) -> str:
    """Hash of a set of files: the sorted (relative path, content sha256) pairs. Missing files are an error."""
    rows = []
    for rel in sorted(set(relpaths)):
        rows.append([rel, sha256_file(os.path.join(root, rel), use_sidecar=False)])
    return sha256_json(rows)


def walk(root: str, prefixes, suffixes=None):
    """Relative paths of the files under the given prefixes (directories or files), sorted, skipping caches."""
    out = []
    for p in prefixes:
        full = os.path.join(root, p)
        if os.path.isfile(full):
            out.append(p)
            continue
        for d, dirs, files in os.walk(full):
            dirs[:] = sorted(x for x in dirs if x not in ("__pycache__", ".pytest_cache", ".snakemake"))
            for name in sorted(files):
                if name.endswith((".pyc", ".sha256.json")):
                    continue
                if suffixes and not name.endswith(tuple(suffixes)):
                    continue
                out.append(os.path.relpath(os.path.join(d, name), root))
    return sorted(out)


PIPELINE_FILES = ["ci/Snakefile", "ci/rules", "ci/gates", "ci/envs", "ci/profiles", "ci/stages", "ci/schemas", "ci/waivers", "configs",
                  "ci/__init__.py", "ci/__main__.py", "ci/hygiene.py", "ci/hashing.py", "ci/manifest.py", "ci/record.py", "ci/taxonomy.py",
                  "ci/kingdom.py", "ci/gpu_lease.py", "ci/VERSION", "seeddrift/train", "bench/hard/EXCLUDE.tsv", "bench/islands/composition.py"]


def pipeline_hash(root: str) -> str:
    present = [p for p in PIPELINE_FILES if os.path.exists(os.path.join(root, p))]
    return files_hash(root, walk(root, present))


def stage_key(rule: str, rule_version: str, code_hash: str, params, input_hashes, env_hash: str) -> str:
    return sha256_json({"rule": rule, "rule_version": rule_version, "code": code_hash, "params": params,
                        "inputs": sorted(input_hashes), "env": env_hash})


TAXDUMP_FILES = ("nodes.dmp", "names.dmp", "merged.dmp", "delnodes.dmp")


def taxdump_hash(path: str) -> str:
    """Hash of an NCBI taxonomy dump folder (repository-relative or absolute): its nodes, names, merged and delnodes files."""
    from ci import ROOT
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    rows = []
    for name in TAXDUMP_FILES:
        for cand in (name, name + ".gz"):
            if os.path.exists(os.path.join(full, cand)):
                rows.append([cand, sha256_file(os.path.join(full, cand))])
    if not rows:
        raise SystemExit("no taxdump files in %s" % full)
    return sha256_json(rows)
