"""The sealed check of the relational-model line; configs/rel/sealed.json says what it refuses and why.

Every stage that reads data calls it, in the CI environment and in the model's Python alike, so this module uses the
standard library only.

  SealedSet.load(ci_root)       the sealed test genomes: the test split of the pinned training-data index (found under
                                <ci_root>/runs or <ci_root>/imports/runs, checked by sha256 and by the expected count)
                                with their assembly accessions; the sealed chromosome accessions of the lists in the
                                repository; and the machine's sealed registry <ci_root>/host/sealed_registry.json, whose
                                test split must be exactly these genomes and every alias of which is refused. Without
                                the registry it fails: tables are built only where the registry is (machine i-1), and
                                every other stage accepts only tables whose manifest records a passed check
  SealedSet.from_parts(...)     the same from given parts (tests and fixtures)
  check_ids(ids, sealed, splits=None, allowed=("train", "heldout"))
                                raises SealedError when an id, an assembly accession inside it (GCA or GCF, any
                                version) or a chromosome accession inside it (any version or case, also inside composite
                                names) is sealed, or, when the split of every id is given, when one is not allowed or
                                unknown; returns the evidence of a passed check
  splits_from_index(path)       {genome id: split} of a training-data index (data/train_data/index.json)
The lineage half of the check (bench/hard/EXCLUDE.tsv and the extra exclusion lists by full NCBI lineage) needs the
taxonomy and runs in the CI environment: ci/stages/rel/common.py check_lineage.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SPEC = os.path.join(ROOT, "configs", "rel", "sealed.json")
ALLOWED = ("train", "heldout")
ASM_RX = re.compile(r"(?<![A-Za-z0-9])gc[af]_(\d{9})(?:\.\d+)?(?![0-9])", re.IGNORECASE)


class SealedError(ValueError):
    pass


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def assemblies(text) -> set:
    """The assembly numbers in a name: GCA_000007185.1 and GCF_000007185 are the same assembly (000007185)."""
    return set(ASM_RX.findall(str(text)))


def splits_from_index(path) -> dict:
    with open(path) as f:
        return {g["id"]: g["split"] for g in json.load(f)["genomes"]}


class SealedSet:
    def __init__(self, test_ids, aliases=(), chromosomes=(), source=None):
        self.test_ids = frozenset(test_ids)
        self.test_assemblies = frozenset(a for x in list(self.test_ids) + list(aliases) for a in assemblies(x))
        self.aliases = frozenset(aliases)
        stems = sorted({str(a).split(".")[0].upper() for a in chromosomes if str(a).strip()})
        self.chromosomes = tuple(stems)
        self.rx = re.compile(r"(?<![A-Za-z0-9])(%s)(\.\d+)?(?![0-9])" % "|".join(re.escape(s) for s in stems), re.IGNORECASE) if stems else None
        self.source = source or {}

    @classmethod
    def from_parts(cls, test_ids, aliases=(), chromosomes=(), source=None):
        return cls(test_ids, aliases, chromosomes, source or dict(kind="given"))

    @classmethod
    def load(cls, ci_root, spec_path=SPEC, root=ROOT):
        with open(spec_path) as f:
            spec = json.load(f)
        ts = spec["test_split"]
        tried = [os.path.join(ci_root, sub, ts["run"], ts["path"]) for sub in ts["search"]]
        found = [p for p in tried if os.path.exists(p)]
        if not found:
            raise SealedError("the sealed test split cannot be checked: the training-data index of run %s is not under %s" % (ts["run"], " or ".join(tried)))
        idx = found[0]
        sha = _sha256(idx)
        if sha != ts["sha256"]:
            raise SealedError("the training-data index %s has sha256 %s, configs/rel/sealed.json pins %s" % (idx, sha, ts["sha256"]))
        with open(idx) as f:
            genomes = json.load(f)["genomes"]
        test = sorted(g["id"] for g in genomes if g["split"] == ts["split"])
        if len(test) != int(ts["expect"]):
            raise SealedError("the index %s has %d genomes in split %s, %s expected" % (idx, len(test), ts["split"], ts["expect"]))
        aliases, extra_ids = set(), set()
        chroms = []
        for rel in spec.get("chromosome_lists") or []:
            with open(os.path.join(root, rel)) as f:
                chroms += [r["accession"] for r in csv.DictReader(f, delimiter="\t") if r.get("accession")]
        if not chroms:
            raise SealedError("no sealed chromosome accession in %s" % spec.get("chromosome_lists"))
        source = dict(kind="pinned", index=os.path.relpath(idx, ci_root), index_sha256=sha, test_genomes=len(test), chromosomes=len(chroms), registry=None)
        reg_path = os.path.join(ci_root, spec["registry"]) if spec.get("registry") else ""
        if not reg_path or not os.path.exists(reg_path):
            raise SealedError("the sealed registry %s is not on this machine: the sealed check cannot run here (tables are built where it "
                              "is; elsewhere only tables whose manifest records a passed check are read)" % (reg_path or "(none named)"))
        with open(reg_path) as f:
            reg = json.load(f)
        splits = reg.get("corpus_test_splits") or []
        if not any(sorted(g["id"] for g in s.get("genomes") or []) == test for s in splits):
            raise SealedError("%s has no test split equal to the pinned one (%d genomes); the sealed sources disagree" % (reg_path, len(test)))
        for s in splits:
            for g in s.get("genomes") or []:
                extra_ids.add(g["id"])
                aliases.update(g.get("aliases") or [])
        for c in reg.get("chromosomes") or []:
            chroms += list(c.get("aliases") or []) + [c.get("accession", "")]
        source["registry"] = dict(path=os.path.relpath(reg_path, ci_root), sha256=_sha256(reg_path), test_splits=len(splits))
        return cls(set(test) | extra_ids, aliases, chroms, source)

    def reasons(self, gid) -> list:
        out = []
        g = str(gid)
        if g in self.test_ids or g in self.aliases:
            out.append("a sealed test genome")
        elif assemblies(g) & self.test_assemblies:
            out.append("the assembly of a sealed test genome")
        if self.rx is not None and self.rx.search(g):
            out.append("a sealed chromosome accession")
        return out


def check_ids(ids, sealed: SealedSet, splits=None, allowed=ALLOWED) -> dict:
    ids = list(ids)
    problems = []
    for gid in ids:
        r = sealed.reasons(gid)
        if splits is not None:
            s = splits.get(gid)
            if s is None:
                r.append("no split known")
            elif s not in allowed:
                r.append("split %s is not allowed here (allowed: %s)" % (s, ", ".join(allowed)))
        if r:
            problems.append(dict(id=gid, reasons=r))
    if problems:
        raise SealedError("sealed check failed for %d of %d ids: %s" % (len(problems), len(ids), json.dumps(problems[:20])))
    return dict(checked=len(ids), distinct=len(set(ids)), source=sealed.source, allowed_splits=list(allowed) if splits is not None else None)
