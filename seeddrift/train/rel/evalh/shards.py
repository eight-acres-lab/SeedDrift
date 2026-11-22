"""Read the tables of a table-set manifest for evaluation: numpy, memory-mapped, in the order of the set.

A table set lists its files (hand-off format 1): `<folder>/<shard>.<field>.npy`, roles shard (what models read) and
meta (reporting and sampling only). The files of one folder are one ordered set of tables: shards in name order,
table t of the set = position t over the concatenated shards. That position is the table id of the evaluation
conditions.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from seeddrift.train.rel import api

SPLIT_TRAIN, SPLIT_HELDOUT = 0, 1
MODEL_FIELDS = ("tok", "n", "col", "kind", "frame_row")           # what a baseline may read
META_FIELDS = ("group", "split", "identity", "genome")           # what the report may read


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class TableSet:
    """The tables of one folder of a manifest. files: {(shard, field): manifest entry}."""

    def __init__(self, run_dir, folder, files):
        self.run_dir, self.folder, self.files = run_dir, folder, files
        self.shards = sorted({s for s, _ in files})
        self.fields = sorted({f for _, f in files})
        missing = [(s, f) for s in self.shards for f in self.fields if (s, f) not in files]
        if missing:
            raise ValueError("table set %s: shards without a field: %s" % (folder, missing[:5]))
        self._open = {}
        self.sizes = [int(self.array(s, "n").shape[0]) for s in self.shards]
        self.tables = int(sum(self.sizes))
        self.verified = {}

    def path(self, shard, field):
        return os.path.join(self.run_dir, self.files[(shard, field)]["path"])

    def array(self, shard, field):
        if (shard, field) not in self._open:
            self._open[(shard, field)] = np.load(self.path(shard, field), mmap_mode="r")
        return self._open[(shard, field)]

    def verify(self, shard, field):
        """The file is the one the manifest names (sha256); done once per file that is read."""
        if (shard, field) not in self.verified:
            e = self.files[(shard, field)]
            p = self.path(shard, field)
            if os.path.getsize(p) != e["bytes"] or sha256_of(p) != e["sha256"]:
                raise ValueError("%s is not the file the table-set manifest names" % e["path"])
            self.verified[(shard, field)] = e["sha256"]

    def load(self, first, last, fields, verify=True):
        """{field: array} of the tables first..last-1 of the set, read into memory."""
        if not 0 <= first <= last <= self.tables:
            raise ValueError("tables %d..%d of a set of %d" % (first, last, self.tables))
        out = {f: [] for f in fields}
        at = 0
        for s, size in zip(self.shards, self.sizes):
            a, b = max(first, at) - at, min(last, at + size) - at
            if a < b:
                for f in fields:
                    if verify:
                        self.verify(s, f)
                    out[f].append(np.array(self.array(s, f)[a:b]))
            at += size
        return {f: (np.concatenate(v) if v else np.zeros((0,) + self.array(self.shards[0], f).shape[1:], self.array(self.shards[0], f).dtype)) for f, v in out.items()}

    def split(self):
        """The split every table of the set has (0 train, 1 held out); an error when they differ."""
        values = set()
        for s in self.shards:
            values |= set(np.unique(self.array(s, "split")).tolist())
        if len(values) != 1:
            raise ValueError("table set %s mixes splits %s" % (self.folder, sorted(values)))
        return int(values.pop())


def table_sets(run_dir, manifest):
    """{folder: TableSet} of a table-set manifest (the folders of its shard and meta files)."""
    if manifest.get("format") != "rel-tableset" or int(manifest.get("api_version", -1)) != api.API:
        raise ValueError("not a table-set manifest of API %d" % api.API)
    by = {}
    for e in manifest["files"]:
        if e.get("role") not in ("shard", "meta") or not e["path"].endswith(".npy"):
            continue
        folder, base = os.path.split(e["path"])
        parts = base.split(".")
        if len(parts) != 3:
            raise ValueError("%s is not <shard>.<field>.npy" % e["path"])
        known = api.SHARD_FIELDS if e["role"] == "shard" else api.META_FIELDS
        if parts[1] not in known:
            raise ValueError("%s: %s is not a %s field of the contract" % (e["path"], parts[1], e["role"]))
        by.setdefault(folder, {})[(parts[0], parts[1])] = e
    if not by:
        raise ValueError("the manifest lists no table files")
    return {folder: TableSet(run_dir, folder, files) for folder, files in sorted(by.items())}


def group_names(run_dir, manifest):
    """[the name of every group index of the meta field `group`]: from the table set's index file (role index,
    index.json, its list `groups`), else the manifest's group_names, else []. Never the coverage list, which leaves out
    groups the set has no genome of."""
    for e in manifest.get("files") or []:
        if e.get("role") == "index" and os.path.basename(str(e.get("path", ""))) == "index.json":
            with open(os.path.join(run_dir, e["path"])) as f:
                names = json.load(f).get("groups")
            if isinstance(names, list) and names:
                return [str(g.get("name")) if isinstance(g, dict) else str(g) for g in names]
    names = manifest.get("group_names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    return [str(g) for g in names or []]


def by_split(run_dir, manifests):
    """{0: TableSet of training tables, 1: TableSet of held-out tables} over one or more manifests; each split once."""
    out = {}
    for m in manifests:
        for folder, ts in table_sets(run_dir, m).items():
            s = ts.split()
            if s in out:
                raise ValueError("two table sets of split %d: %s and %s" % (s, out[s].folder, folder))
            ts.manifest = m
            out[s] = ts
    return out
