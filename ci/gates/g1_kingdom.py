"""G1: the training corpus is multi-kingdom, with a floor per group.

static(spec, tier)            T0 and the pre gate, no data: the spec's tier equals the profile's; the spec names every
                              group of configs/kingdom_map.yml with min_genomes >= 1; for tier "full" the floors of
                              configs/corpus/floors.yaml apply (min_genomes and protein shares).
built(spec, entries, tax, tier)  on the built corpus and again on the ids the training loader reads: each member's
                              group comes from its taxid (or its source kind for plasmids); every group reaches its
                              floor (for tier "full" the floors file, otherwise the spec); for "full" the protein-share
                              rules hold.
A short group passes only through an unexpired waiver with gate "G1" and scope "<corpus>:<group>", and never when it
holds less than `start_fraction` (floors.yaml) of its floor in a full corpus.
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


def _waive(name, group, have, need, floors, waiver_dir):
    """A waiver for this group, unless the group is below start_fraction of a full-tier floor."""
    if floors and have < floors.get("start_fraction", 0.5) * need:
        return None
    return find_waiver("G1", "%s:%s" % (name, group), **({"folder": waiver_dir} if waiver_dir else {}))


def static(spec: dict, tier: str = None, kmap: KingdomMap = None, waiver_dir=None) -> GateResult:
    kmap = kmap or KingdomMap()
    name = spec.get("name", "?")
    groups = spec.get("groups") or {}
    problems, waived = [], []
    if tier is not None and spec.get("tier") != tier:
        problems.append(dict(error="spec tier %r differs from the profile tier %r" % (spec.get("tier"), tier)))
    floors = _floors() if spec.get("tier") == "full" else None
    for g in kmap.names:
        want = (groups.get(g) or {}).get("min_genomes", 0)
        need = floors["min_genomes"][g] if floors else 1
        if want < need:
            w = _waive(name, g, want, need, floors, waiver_dir)
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
    return GateResult("G1", status, dict(mode="static", corpus=name, tier=spec.get("tier"), problems=problems, waived=waived), sha256_json(spec))


def built(spec: dict, entries, tax, tier: str = None, kmap: KingdomMap = None, waiver_dir=None) -> GateResult:
    """entries: dicts with id, taxid, source_kind and proteins (number of proteins)."""
    kmap = kmap or KingdomMap()
    name = spec.get("name", "?")
    counts, proteins, problems, waived = {}, {}, [], []
    if tier is not None and spec.get("tier") != tier:
        problems.append(dict(error="spec tier %r differs from the profile tier %r" % (spec.get("tier"), tier)))
    if not entries:
        problems.append(dict(error="no genomes"))
    for e in entries:
        try:
            g = kmap.group(tax, int(e["taxid"]), e.get("source_kind", ""))
        except (UnknownTaxid, NoGroup, KeyError, ValueError) as err:
            problems.append(dict(id=e.get("id"), error="no group: %s" % err))
            continue
        counts[g] = counts.get(g, 0) + 1
        proteins[g] = proteins.get(g, 0) + int(e.get("proteins", 0))
    floors = _floors() if spec.get("tier") == "full" else None
    groups = spec.get("groups") or {}
    for g in kmap.names:
        need = floors["min_genomes"][g] if floors else max(1, (groups.get(g) or {}).get("min_genomes", 1))
        have = counts.get(g, 0)
        if have < need:
            w = _waive(name, g, have, need, floors, waiver_dir)
            if w:
                waived.append(dict(group=g, have=have, need=need, waiver=w["file"]))
            else:
                problems.append(dict(group=g, have=have, need=need))
    shares = {}
    total = sum(proteins.values())
    if total:
        for g, n in proteins.items():
            shares[_major(g)] = shares.get(_major(g), 0) + n / total
    if floors:
        rules = floors["protein_share"]
        if max(shares.values() or [1]) > rules["max_any_major"]:
            problems.append(dict(protein_share="max_any_major", value=round(max(shares.values()), 4), limit=rules["max_any_major"]))
        if shares.get("eukaryotes", 0) < rules["min_eukaryotes"]:
            problems.append(dict(protein_share="min_eukaryotes", value=round(shares.get("eukaryotes", 0), 4), limit=rules["min_eukaryotes"]))
        if shares.get("viral_mobile", 0) < rules["min_viral_mobile"]:
            problems.append(dict(protein_share="min_viral_mobile", value=round(shares.get("viral_mobile", 0), 4), limit=rules["min_viral_mobile"]))
    status = "fail" if problems else ("waived" if waived else "pass")
    ev = dict(mode="built", corpus=name, tier=spec.get("tier"), counts=counts, proteins=proteins, shares={k: round(v, 4) for k, v in shares.items()},
              problems=problems, waived=waived)
    return GateResult("G1", status, ev, sha256_json(sorted((e.get("id"), int(e.get("taxid", 0)), e.get("source_kind", "")) for e in entries)))


# ---- pipeline select: the corpus selection ---------------------------------------------------------------------

SELECT_KINDS = ("ncbi_genome_reports", "ncbi_genome_accessions", "ncbi_sequence_reports", "ncbi_virus_reports", "ensembl_species", "file")


def select_targets(spec: dict, kmap: KingdomMap = None) -> dict:
    """Target per group (genomes; plasmid replicons for mobile.plasmid): `groups.<g>.target` if given, else the floor
    (tier full: configs/corpus/floors.yaml; other tiers: `min_genomes`) times `target_factor`, rounded up."""
    import math
    from fractions import Fraction
    kmap = kmap or KingdomMap()
    floors = _floors()["min_genomes"] if spec.get("tier") == "full" else {}
    groups = spec.get("groups") or {}
    out = {}
    for g in kmap.names:
        floor = floors.get(g) if floors else (groups.get(g) or {}).get("min_genomes", 1)
        t = (groups.get(g) or {}).get("target")
        out[g] = int(t) if t is not None else int(math.ceil(floor * Fraction(str(spec.get("target_factor", 1.0)))))   # exact: 10 x 1.1 is 11
    return out


def select_static(spec: dict, tier: str = None, kmap: KingdomMap = None) -> GateResult:
    """The select spec is complete before anything is fetched: the tier is the profile's; every group has a target at
    or above its floor (full) or at least 1; a seed; known catalogue kinds with unique ids, dependent ones after their
    source; every selection rule names existing catalogues."""
    kmap = kmap or KingdomMap()
    problems = []
    if tier is not None and spec.get("tier") != tier:
        problems.append(dict(error="spec tier %r differs from the profile tier %r" % (spec.get("tier"), tier)))
    if not isinstance(spec.get("seed"), int):
        problems.append(dict(error="the spec needs an integer seed"))
    unknown = sorted(set(spec.get("groups") or {}) - set(kmap.names))
    if unknown:
        problems.append(dict(unknown_groups=unknown))
    floors = _floors()["min_genomes"] if spec.get("tier") == "full" else None
    targets = select_targets(spec, kmap)
    for g in kmap.names:
        need = floors[g] if floors else max(1, ((spec.get("groups") or {}).get(g) or {}).get("min_genomes", 1))
        if targets[g] < need:
            problems.append(dict(group=g, target=targets[g], required=need))
    seen = []
    for c in spec.get("catalogues") or []:
        kinds = [k for k in SELECT_KINDS if k in c]
        if not c.get("id") or len(kinds) != 1:
            problems.append(dict(catalogue=c.get("id"), error="needs an id and exactly one kind of %s" % (SELECT_KINDS,)))
        elif c["id"] in seen:
            problems.append(dict(catalogue=c["id"], error="duplicate id"))
        elif kinds[0] in ("ncbi_genome_accessions", "ncbi_sequence_reports") and (not c[kinds[0]].get("from") or set(c[kinds[0]]["from"]) - set(seen)):
            problems.append(dict(catalogue=c["id"], error="takes its accessions from %r, which are not all earlier catalogues" % c[kinds[0]].get("from")))
        seen.append(c.get("id"))
    rules = spec.get("rules") or {}
    for part in ("prokaryotes", "viruses", "eukaryotes", "plasmids"):
        if part not in rules:
            problems.append(dict(rule=part, error="missing (every kingdom is selected)"))
    if "plasmids" in rules and not rules["plasmids"].get("replicons"):
        problems.append(dict(rule="plasmids", error="needs `replicons`: the sequence reports that list each assembly's plasmids"))
    for part, r in rules.items():
        for cid in (r.get("catalogues") or []) + [r[k] for k in ("hosts", "annotation", "replicons") if r.get(k)]:
            if cid not in seen:
                problems.append(dict(rule=part, error="unknown catalogue %s" % cid))
    status = "fail" if problems else "pass"
    return GateResult("G1_select_spec", status, dict(mode="select_static", spec=spec.get("name"), tier=spec.get("tier"), targets=targets, problems=problems),
                      sha256_json(spec))


def _shares(prot):
    total = sum(prot.values())
    return {k: (v / total if total else 0.0) for k, v in prot.items()}


def selection(spec: dict, rows, tax, tier: str = None, kmap: KingdomMap = None) -> GateResult:
    """rows: the selected table (id, group, taxid, source_kind, host_taxid, units, proteins_low/central/high). The group of
    every row is derived again from its taxid (and source kind and host) and must equal the stage's; every group
    reaches its floor (units: genomes, plasmid replicons); for tier full the protein shares hold at the central
    estimate, and the evidence says where they would not hold at the unfavourable end of the ranges ("uncertain")."""
    kmap = kmap or KingdomMap()
    problems, counts = [], {}
    prot = {b: {} for b in ("low", "central", "high")}
    if tier is not None and spec.get("tier") != tier:
        problems.append(dict(error="spec tier %r differs from the profile tier %r" % (spec.get("tier"), tier)))
    if not rows:
        problems.append(dict(error="nothing selected"))
    for r in rows:
        try:
            g = kmap.group(tax, int(r["taxid"]), r.get("source_kind", ""), int(r["host_taxid"]) if r.get("host_taxid") else None)
        except (UnknownTaxid, NoGroup, KeyError, ValueError) as err:
            problems.append(dict(id=r.get("id"), error="no group: %s" % err))
            continue
        if g != r.get("group"):
            problems.append(dict(id=r.get("id"), error="the stage put it in %s, its taxid in %s" % (r.get("group"), g)))
        counts[g] = counts.get(g, 0) + int(r.get("units") or 1)
        for b in prot:
            prot[b][_major(g)] = prot[b].get(_major(g), 0) + int(r["proteins_" + b])
    floors = _floors() if spec.get("tier") == "full" else None
    groups = spec.get("groups") or {}
    short = []
    for g in kmap.names:
        need = floors["min_genomes"][g] if floors else max(1, (groups.get(g) or {}).get("min_genomes", 1))
        if counts.get(g, 0) < need:
            short.append(dict(group=g, have=counts.get(g, 0), need=need, below_start_fraction=bool(floors) and counts.get(g, 0) < floors.get("start_fraction", 0.5) * need))
    problems += short
    shares = {b: _shares(prot[b]) for b in prot}
    uncertain = []
    if floors:
        rules = floors["protein_share"]
        c = shares["central"]
        for cls in ("prokaryotes", "eukaryotes", "viral_mobile"):
            if c.get(cls, 0) > rules["max_any_major"]:
                problems.append(dict(protein_share="max_any_major", major=cls, value=round(c[cls], 4), limit=rules["max_any_major"]))
        for cls, key in (("eukaryotes", "min_eukaryotes"), ("viral_mobile", "min_viral_mobile")):
            if c.get(cls, 0) < rules[key]:
                problems.append(dict(protein_share=key, value=round(c.get(cls, 0), 4), limit=rules[key]))
        # the unfavourable end: the class at its low count against the others at their high counts (and the reverse)
        for cls in ("prokaryotes", "eukaryotes", "viral_mobile"):
            others_low = sum(v for k, v in prot["low"].items() if k != cls)
            others_high = sum(v for k, v in prot["high"].items() if k != cls)
            hi = prot["high"].get(cls, 0)
            lo = prot["low"].get(cls, 0)
            worst_max = hi / (hi + others_low) if hi + others_low else 0
            worst_min = lo / (lo + others_high) if lo + others_high else 0
            if worst_max > rules["max_any_major"]:
                uncertain.append(dict(rule="max_any_major", major=cls, worst=round(worst_max, 4)))
            key = {"eukaryotes": "min_eukaryotes", "viral_mobile": "min_viral_mobile"}.get(cls)
            if key and worst_min < rules[key]:
                uncertain.append(dict(rule=key, worst=round(worst_min, 4)))
    status = "fail" if problems else "pass"
    ev = dict(mode="selection", spec=spec.get("name"), tier=spec.get("tier"), counts=counts, short=short, proteins=prot,
              shares={b: {k: round(v, 4) for k, v in s.items()} for b, s in shares.items()}, uncertain=uncertain, problems=problems[:200], n_problems=len(problems))
    return GateResult("G1_select", status, ev, sha256_json(sorted((r.get("id"), int(r.get("taxid", 0)), r.get("group", "")) for r in rows)))
