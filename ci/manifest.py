"""Content-addressed dataset manifests.

A manifest is a canonical JSON object {"name", "kind", "entries": [...], "meta": {...}}. Every entry names one member
(a genome, a protein set, a reference database) with its files and their sha256; entries are sorted by "id". The
manifest is stored as manifests/<sha256>.json, and a dataset id is ds-<name>-<first 12 hex of that sha256>.
"""
from __future__ import annotations

import json
import os

from ci.hashing import canonical_json, sha256_bytes, sha256_file


class ManifestError(ValueError):
    pass


def file_entry(path: str, root: str = "") -> dict:
    rel = os.path.relpath(path, root) if root else path
    return {"path": rel, "sha256": sha256_file(path), "bytes": os.path.getsize(path)}


def build(name: str, kind: str, entries, meta=None) -> dict:
    entries = sorted(entries, key=lambda e: e["id"])
    ids = [e["id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise ManifestError("duplicate ids in manifest %s" % name)
    return {"name": name, "kind": kind, "entries": entries, "meta": meta or {}}


def digest(manifest: dict) -> str:
    return sha256_bytes(canonical_json(manifest))


def dataset_id(manifest: dict) -> str:
    return "ds-%s-%s" % (manifest["name"], digest(manifest)[:12])


def write(manifest: dict, folder: str) -> str:
    """Write manifests/<sha>.json atomically; return the sha256."""
    os.makedirs(folder, exist_ok=True)
    d = digest(manifest)
    path = os.path.join(folder, d + ".json")
    if not os.path.exists(path):
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(canonical_json(manifest))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    return d


def load(path: str) -> dict:
    with open(path) as f:
        m = json.load(f)
    name = os.path.basename(path)[:-5]
    if len(name) == 64 and digest(m) != name:
        raise ManifestError("manifest %s does not match its name" % path)
    return m


def verify(manifest: dict, root: str) -> list:
    """Paths whose content no longer matches the manifest (empty when everything matches)."""
    bad = []
    for e in manifest["entries"]:
        for f in e.get("files", []):
            p = os.path.join(root, f["path"])
            if not os.path.exists(p) or sha256_file(p) != f["sha256"]:
                bad.append(f["path"])
    return bad
