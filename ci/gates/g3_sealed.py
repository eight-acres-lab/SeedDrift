"""G3: nothing of the sealed test set is scored before the freeze.

The sealed chromosomes (bench/hard/results/sealed.tsv: RefSeq accession, genus) are scored only by a freeze run (tier
"freeze"), which CI 0.1 does not have: a freeze profile fails this gate until its own CI version adds the freeze rules
(signed freeze tag, committed model checkpoint hash, sealed files in a mode-700 folder, unseal.log written first).
Every other run fails when

  static (pre gate, before any download): a sealed accession, with or without its version, occurs in the resolved
         configuration, the corpus spec or the evaluation tables;
  built (eval gate): an evaluation genome (or the host of a scenario) lies under a sealed genus (bacterial and
         archaeal genera of that name; a homonym elsewhere does not count), or a sealed accession occurs in any
         manifest of the run (fetch, corpus, evaluation inputs, hits), the training id list or the evaluation tables.

The training corpus is kept clear of the sealed genera by gate G2 (bench/hard/EXCLUDE.tsv); G3 is about scoring.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re

from ci import ROOT
from ci.gates import GateResult
from ci.hashing import sha256_json

SEALED = os.path.join(ROOT, "bench", "hard", "results", "sealed.tsv")


def load_sealed(path: str = SEALED):
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError("the sealed list %s is empty" % path)
    return [(r["accession"], r["genus"]) for r in rows]


def _pattern(sealed):
    stems = sorted({a.split(".")[0] for a, _ in sealed})
    return re.compile(r"(?<![A-Za-z0-9_])(%s)(\.\d+)?(?![A-Za-z0-9])" % "|".join(re.escape(s) for s in stems))


def _scan(paths, rx):
    found = {}
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p, errors="replace") as f:
            hits = sorted(set(m.group(0) for m in rx.finditer(f.read())))
        if hits:
            found[os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)] = hits
    return found


def _freeze(tier):
    if tier == "freeze":
        return GateResult("G3", "fail", dict(error="tier freeze has no rules in this CI version; a freeze run needs its own CI version"))
    return None


def static(text_files, tier: str, sealed_path: str = SEALED) -> GateResult:
    f = _freeze(tier)
    if f:
        f.gate = "G3_static"
        return f
    sealed = load_sealed(sealed_path)
    found = _scan(text_files, _pattern(sealed))
    ev = dict(sealed=len(sealed), files=len([p for p in text_files if p]), found=found)
    return GateResult("G3_static", "fail" if found else "pass", ev, sha256_json([sorted(sealed), sorted(found)]))


def built(run_dir: str, tax, eval_taxids, tier: str, extra_files=(), sealed_path: str = SEALED) -> GateResult:
    """eval_taxids: {evaluation genome id: [taxids]} (the genome's and its scenario host's)."""
    f = _freeze(tier)
    if f:
        return f
    sealed = load_sealed(sealed_path)
    genera = {}
    for _, g in sealed:
        for t in tax.genus_by_name(g):
            genera[t] = g
    under = {}
    for gid, taxids in sorted(eval_taxids.items()):
        for t in taxids:
            g = tax.at_rank(int(t), "genus")
            if g in genera:
                under.setdefault(gid, []).append("%s (genus %s)" % (t, genera[g]))
    files = sorted(glob.glob(os.path.join(run_dir, "data", "*", "manifest.json"))) + [os.path.join(run_dir, "data", "train", "ids.json")] + list(extra_files)
    found = _scan(files, _pattern(sealed))
    ev = dict(sealed=len(sealed), sealed_genera=len(genera), evaluation_genomes=len(eval_taxids), under_sealed_genus=under, accessions_found=found)
    bad = bool(under) or bool(found)
    return GateResult("G3", "fail" if bad else "pass", ev, sha256_json([sorted(sealed), sorted(under), sorted(found)]))
