"""HTTP clients for the dependency services, with the same methods as the in-process tools."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config


class _Http:
    def __init__(self, base, timeout=7200):
        self.base, self.timeout = base.rstrip("/"), timeout

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=60) as r:
            return json.load(r)

    def post(self, path, payload):
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError("%s%s: HTTP %d %s" % (self.base, path, e.code, e.read()[:500].decode(errors="replace")))


class ToolsClient(_Http):
    def health(self):
        return self.get("/health")

    def diamond(self, fasta, max_target_seqs=100, evalue=1e-5, sensitivity="sensitive", masking=None):
        return self.post("/diamond", dict(fasta=fasta, max_target_seqs=max_target_seqs, evalue=evalue,
                                          sensitivity=sensitivity, masking=masking))["hits"]

    def trees(self, jobs):
        return self.post("/trees", dict(jobs=jobs))["trees"]

    def metaeuk(self, fasta):
        return self.post("/metaeuk", dict(fasta=fasta))["proteins"]

    def sequences(self, ids, source="reference"):
        return self.post("/sequences", dict(ids=list(ids), source=source))["sequences"]

    def repeats(self, seqs):
        return self.post("/repeats", dict(sequences=seqs))["regions"]

    def foldseek(self, pdbs):
        return self.post("/foldseek", dict(pdbs=pdbs))["hits"]


class FoldClient(_Http):
    def health(self):
        return self.get("/health")

    def fold(self, seqs, max_len=1000):
        return self.post("/fold", dict(sequences=seqs, max_len=max_len))["structures"]


def tools():
    if config.TOOLS_URL:
        return ToolsClient(config.TOOLS_URL)
    from .tools import LocalTools
    return LocalTools()


def folder():
    if config.ESMFOLD_URL:
        return FoldClient(config.ESMFOLD_URL)
    from .fold import Folder
    return Folder(config.ESMFOLD_MODEL)
