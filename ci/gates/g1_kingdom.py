"""G1: the training corpus is multi-kingdom, with a floor per group.

static(spec)                  T0, no data: the corpus spec names every group of configs/kingdom_map.yml with
                              min_genomes >= 1; a spec of tier "full" also meets configs/corpus/floors.yaml.
built(spec, entries, tax)     on the built corpus (and again on the ids the training loader reads): each member's group
                              comes from its taxid (or its source kind for plasmids); every group reaches its floor; for
                              tier "full" the protein-share rules hold.
A missing or short group passes only through an unexpired waiver with gate "G1" and scope "<corpus>:<group>".
"""
from __future__ import annotations

import os

import yaml

from ci import ROOT
from ci.gates import GateResult, find_waiver
from ci.hashing import sha256_json
from ci.kingdom import KingdomMap, NoGroup
from ci.taxonomy import UnknownTaxid

FLOORS = os.path.join(ROOT, "configs", "corpus", "floors.yaml")


def _floors():
    with open(FLOORS) as f:
        return yaml.safe_load(f)


def _major(group: str) -> str:
    if group in ("bacteria", "archaea"):
        return "prokaryotes"
    if group.startswith("eukaryota."):
        return "eukaryotes"
    return "viral_mobile"


def static(spec: dict, kmap: KingdomMap = None, waiver_dir=None) -> GateResult:
    kmap = kmap or KingdomMap()
    name = spec.get("name", "?")
    groups = spec.get("groups") or {}
    problems, waived = [], []
    floors = _floors() if spec.get("tier") == "full" else None
    for g in kmap.names:
        want = (groups.get(g) or {}).get("min_genomes", 0)
        need = floors["min_genomes"][g] if floors else 1
        if want < need:
            w = find_waiver("G1", "%s:%s" % (name, g), **({"folder": waiver_dir} if waiver_dir else {}))
            if w:
                waived.append(dict(group=g, waiver=w["file"]))
            else:
                problems.append(dict(group=g, min_genomes=want, required=need))
    unknown = sorted(set(groups) - set(kmap.names))
    if unknown:
        problems.append(dict(unknown_groups=unknown))
    if floors:
        ps = spec.get("protein_share") or {}
        for k, v in floors["protein_share"].items():
            ok = ps.get(k) is not None and (ps[k] <= v if k.startswith("max") else ps[k] >= v)
            if not ok:
                problems.append(dict(protein_share=k, spec=ps.get(k), required=v))
    status = "fail" if problems else ("waived" if waived else "pass")
    return GateResult("G1", status, dict(mode="static", corpus=name, problems=problems, waived=waived), sha256_json(spec))


def built(spec: dict, entries, tax, kmap: KingdomMap = None, waiver_dir=None) -> GateResult:
    """entries: dicts with id, taxid, source_kind and proteins (number of proteins)."""
    kmap = kmap or KingdomMap()
    name = spec.get("name", "?")
    counts, proteins, problems, waived = {}, {}, [], []
    for e in entries:
        try:
            g = kmap.group(tax, int(e["taxid"]), e.get("source_kind", ""))
        except (UnknownTaxid, NoGroup, KeyError, ValueError) as err:
            problems.append(dict(id=e.get("id"), error="no group: %s" % err))
            continue
        counts[g] = counts.get(g, 0) + 1
        proteins[g] = proteins.get(g, 0) + int(e.get("proteins", 0))
    groups = spec.get("groups") or {}
    for g in kmap.names:
        need = max(1, (groups.get(g) or {}).get("min_genomes", 1))
        if counts.get(g, 0) < need:
            w = find_waiver("G1", "%s:%s" % (name, g), **({"folder": waiver_dir} if waiver_dir else {}))
            if w:
                waived.append(dict(group=g, have=counts.get(g, 0), need=need, waiver=w["file"]))
            else:
                problems.append(dict(group=g, have=counts.get(g, 0), need=need))
    shares = {}
    total = sum(proteins.values())
    if total:
        for g, n in proteins.items():
            shares[_major(g)] = shares.get(_major(g), 0) + n / total
    if spec.get("tier") == "full":
        rules = _floors()["protein_share"]
        if max(shares.values() or [1]) > rules["max_any_major"]:
            problems.append(dict(protein_share="max_any_major", value=round(max(shares.values()), 4), limit=rules["max_any_major"]))
        if shares.get("eukaryotes", 0) < rules["min_eukaryotes"]:
            problems.append(dict(protein_share="min_eukaryotes", value=round(shares.get("eukaryotes", 0), 4), limit=rules["min_eukaryotes"]))
        if shares.get("viral_mobile", 0) < rules["min_viral_mobile"]:
            problems.append(dict(protein_share="min_viral_mobile", value=round(shares.get("viral_mobile", 0), 4), limit=rules["min_viral_mobile"]))
    status = "fail" if problems else ("waived" if waived else "pass")
    ev = dict(mode="built", corpus=name, counts=counts, proteins=proteins, shares={k: round(v, 4) for k, v in shares.items()}, problems=problems, waived=waived)
    return GateResult("G1", status, ev, sha256_json(sorted((e.get("id"), int(e.get("taxid", 0)), e.get("source_kind", "")) for e in entries)))
