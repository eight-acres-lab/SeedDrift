"""NCBI taxonomy from taxdump files (nodes.dmp, names.dmp, optional merged.dmp and delnodes.dmp; plain or .gz).

Fail closed: a taxid that is neither a node nor merged into one raises UnknownTaxid; callers that guard data (the
exclusion gate) treat that as a failure, never as "not excluded".
"""
from __future__ import annotations

import gzip
import os


class UnknownTaxid(KeyError):
    pass


def _open(path):
    if os.path.exists(path):
        return gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt")
    raise FileNotFoundError(path)


def _rows(path):
    with _open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.endswith("\t|"):
                line = line[:-2]
            yield line.split("\t|\t")


class Taxonomy:
    def __init__(self, folder: str, names: bool = True):
        self.folder = folder
        self.parent, self.rank = {}, {}
        for f in _rows(os.path.join(folder, "nodes.dmp")):
            t = int(f[0])
            self.parent[t] = int(f[1])
            self.rank[t] = f[2].strip()
        self.merged = {}
        for name in ("merged.dmp",):
            p = os.path.join(folder, name)
            if os.path.exists(p) or os.path.exists(p + ".gz"):
                for f in _rows(p):
                    self.merged[int(f[0])] = int(f[1])
        self.deleted = set()
        p = os.path.join(folder, "delnodes.dmp")
        if os.path.exists(p) or os.path.exists(p + ".gz"):
            for f in _rows(p):
                self.deleted.add(int(f[0]))
        self.sci = {}
        self.by_name = {}
        if names:
            for f in _rows(os.path.join(folder, "names.dmp")):
                if len(f) >= 4 and f[3].strip() == "scientific name":
                    t = int(f[0])
                    self.sci[t] = f[1]
                    self.by_name.setdefault(f[1], []).append(t)

    def resolve(self, taxid: int) -> int:
        t = int(taxid)
        seen = set()
        while t in self.merged and t not in seen:
            seen.add(t)
            t = self.merged[t]
        if t not in self.parent:
            raise UnknownTaxid(taxid)
        return t

    def lineage(self, taxid: int) -> list:
        """Taxids from the node itself up to the root."""
        t = self.resolve(taxid)
        out = []
        while True:
            out.append(t)
            p = self.parent[t]
            if p == t or p in out:
                break
            if p not in self.parent:
                raise UnknownTaxid(p)
            t = p
        return out

    def at_rank(self, taxid: int, rank: str) -> int:
        for t in self.lineage(taxid):
            if self.rank.get(t) == rank:
                return t
        return 0

    def name(self, taxid: int) -> str:
        return self.sci.get(self.resolve(taxid), str(taxid))

    def genus_by_name(self, name: str, domains=(2, 2157)) -> list:
        """Taxids of rank genus with this scientific name under the given domains (bacteria, archaea by default)."""
        out = []
        for t in self.by_name.get(name, []):
            if self.rank.get(t) == "genus" and set(self.lineage(t)) & set(domains):
                out.append(t)
        return sorted(out)

    def taxa_by_name(self, name: str, rank: str) -> list:
        return sorted(t for t in self.by_name.get(name, []) if self.rank.get(t) == rank)
