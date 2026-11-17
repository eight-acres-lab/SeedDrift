"""Stage select (pipeline select): a multi-kingdom training corpus chosen from the catalogue snapshots.

A pure function of the snapshots (data/catalog/manifest.json), the taxonomy, configs/kingdom_map.yml, the exclusion
rules (bench/hard/EXCLUDE.tsv and what it names), the select spec (configs/select/<name>.yaml), its assumptions file and
its seed. No similarity search result enters the selection; it uses taxonomy, catalogue quality fields and counts.

Candidates (one row per catalogue record that could be a corpus genome):
  prokaryotes  NCBI genome reports (bacteria, archaea): source kind raw_prokaryote, chromosomes only
  plasmids     the plasmid replicons of every eligible prokaryote assembly (components minus chromosomes): source
               kind plasmid, one row per assembly, `units` = its plasmids (the floor counts plasmid replicons)
  viruses      NCBI genome reports of viral assemblies (segments of one genome stay together); the host comes from
               the NCBI virus reports of the same species and places the virus (configs/kingdom_map.yml virus_hosts)
  eukaryotes   Ensembl and Ensembl Genomes species lists: source kind annotated_cds; protein counts from the NCBI
               annotation of the same assembly (or its paired assembly)
Every rejected candidate keeps its reason (`out`): an exclusion (by full lineage, as gate G2; for viruses also the
host's lineage), no species or genus, quality (too few genes, incomplete), a group conflict, or another genome of the
same species was preferred (quality order of the spec).
Selection per group, diversity first: one genome per species; genera in random order (fixed seed per group) within
the outer rank (order; family for viruses); depth by depth (first one species of every genus, outer taxa interleaved,
then a second, ...). The target (configs: floor x target_factor, or the spec's) takes a prefix of that order, so a
larger target only adds genomes; the cap per genus is passed only when a group would otherwise fall short (recorded).
Protein shares (configs/corpus/floors.yaml) are repaired by raising the targets of a short class, never by lowering
another below its floor; what cannot be repaired is left to gate G1_select and the report.
Outputs: data/select/{selected.tsv, candidates.tsv, by_group.json, manifest.json}.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random

import yaml

from ci import manifest as M
from ci.gates.g1_kingdom import _floors, _major, select_targets
from ci.gates.g2_exclusion import CATALOGUE, EXCLUDE, PANELS, Excluder
from ci.gates.g3_sealed import SEALED
from ci.hashing import sha256_file, sha256_json, taxdump_hash
from ci.kingdom import DEFAULT_MAP, GroupConflict, KingdomMap, NoGroup
from ci.stages.catalog import read_jsonl
from ci.stages.common import Stage, main_args, write_json
from ci.taxonomy import Taxonomy, UnknownTaxid

RANKS = ("phylum", "class", "order", "family", "genus", "species")
SELECTED_COLS = ["id", "group", "source_kind", "taxid", "host_taxid", "species_taxid", "phylum", "class", "order", "family", "genus", "accession",
                 "paired_accession", "ena_assembly", "replicon_types", "ensembl_site", "ensembl_division", "ensembl_species", "ensembl_core_db",
                 "ensembl_release", "table", "units", "proteins_low", "proteins_central", "proteins_high", "proteins_level", "pick", "catalogue"]
CAND_COLS = ["id", "group", "catalogue", "taxid", "host_taxid", "species_taxid", "order", "genus", "accession", "units", "proteins_central", "proteins_level",
             "out", "selected", "pick"]


def group_seed(seed, group):
    return int(hashlib.sha256(("%s:%s" % (seed, group)).encode()).hexdigest()[:16], 16)


def lineage(tax, taxid):
    return {r: tax.at_rank(int(taxid), r) for r in RANKS}


def scaled(n, f):
    return [int(round(n * x)) for x in f]


def ena_assembly(acc, paired):
    return acc if acc.startswith("GCA_") else (paired if paired.startswith("GCA_") else "")


class Candidates:
    def __init__(self, tax, kmap, ex, assumptions):
        self.tax, self.kmap, self.ex, self.a = tax, kmap, ex, assumptions
        self.rows = []

    def add(self, **r):
        r.setdefault("out", "")
        r.setdefault("host_taxid", 0)
        r.setdefault("units", 1)
        for k in RANKS:
            r.setdefault(k, 0)
        self.rows.append(r)
        return r

    def base(self, taxid):
        """(lineage, out reason or '')."""
        try:
            lin = lineage(self.tax, taxid)
        except (UnknownTaxid, KeyError, ValueError):
            return {r: 0 for r in RANKS}, "taxid does not resolve"
        return lin, ""


def prokaryotes(cands, recs, catalogue, rules):
    tax, kmap, ex, a = cands.tax, cands.kmap, cands.ex, cands.a
    pf = a["protein_factor"]
    for r in recs:
        lin, out = cands.base(r["taxid"])
        group = ""
        if not out:
            try:
                group = kmap.group(tax, r["taxid"], "raw_prokaryote")
            except NoGroup as err:
                out = "no group: %s" % err
        if not out and group not in ("bacteria", "archaea"):
            out = "not a prokaryote (group %s)" % group
        if not out:
            why = ex.reason(r["taxid"], [r["accession"], r["paired_accession"]])
            out = "excluded: %s" % why[0] if why else ""
        if not out and not lin["genus"]:
            out = "no genus"
        if not out and r["protein_coding"] < int(rules.get("min_protein_coding", 0)):
            out = "fewer than %s protein-coding genes" % rules["min_protein_coding"]
        if not out and r.get("assembly_status") and r["assembly_status"] != "current":
            out = "assembly not current"
        lo, ce, hi = scaled(r["protein_coding"], pf["prokaryote_chromosomes"])
        acc = r["accession"]
        quality = (r.get("refseq_category") != "reference genome", -(r.get("checkm_completeness") or 0), r.get("checkm_contamination") or 100.0, acc)
        cands.add(id="%s_%s" % (group or "prokaryote", acc), group=group, part="prokaryotes", catalogue=catalogue, source_kind="raw_prokaryote", taxid=r["taxid"],
                  species_taxid=lin["species"], accession=acc, paired_accession=r["paired_accession"], ena_assembly=ena_assembly(acc, r["paired_accession"]),
                  replicon_types="Chromosome", table=11, proteins_low=lo, proteins_central=ce, proteins_high=hi, proteins_level="annotated (whole assembly; chromosomes kept, genes called again)",
                  quality=quality, out=out, plasmids=max(0, r["components"] - r["chromosomes"]), **{k: lin[k] for k in RANKS if k != "species"})


def plasmids(cands, rules):
    a = cands.a
    per = a["plasmid_proteins_per_replicon"]
    for p in [r for r in cands.rows if r["part"] == "prokaryotes"]:
        n = p["plasmids"]
        out = p["out"]
        if not out and n < 1:
            out = "no plasmid"
        group = "mobile.plasmid"
        cands.add(id="plasmid_%s" % p["accession"], group=group, part="plasmids", catalogue=p["catalogue"], source_kind="plasmid", taxid=p["taxid"],
                  species_taxid=p["species_taxid"], accession=p["accession"], paired_accession=p["paired_accession"], ena_assembly=p["ena_assembly"],
                  replicon_types="Plasmid", table=11, units=max(n, 1), proteins_low=n * per[0], proteins_central=n * per[1], proteins_high=n * per[2],
                  proteins_level="assumed per replicon", quality=p["quality"], out=out,
                  **{k: p[k] for k in RANKS if k != "species"})


def viruses(cands, recs, host_recs, catalogue, rules):
    tax, kmap, ex, a = cands.tax, cands.kmap, cands.ex, cands.a
    hosts_of = {}
    for h in host_recs:
        try:
            s = tax.at_rank(h["virus_taxid"], "species") if h["virus_taxid"] else 0
        except (UnknownTaxid, KeyError, ValueError):
            continue
        if s and h["host_taxid"]:
            hosts_of.setdefault(s, set()).add(h["host_taxid"])
    for r in recs:
        lin, out = cands.base(r["taxid"])
        hosts, host, group = sorted(hosts_of.get(lin["species"], ())), 0, ""
        classes = set()
        for h in hosts:
            try:
                classes.add(kmap.host_class(tax, h))
            except (UnknownTaxid, KeyError, ValueError):
                classes.add("unresolved")
        classes.discard("")
        if not out and "unresolved" in classes:
            out = "a host taxid does not resolve"
        if not out and len(classes) > 1:
            out = "hosts in more than one class (%s)" % ",".join(sorted(classes))
        if not out and classes:
            cls = classes.pop()
            host = min(h for h in hosts if kmap.host_class(tax, h) == cls)
        if not out:
            try:
                group = kmap.group(tax, r["taxid"], "raw_virus", host or None)
            except GroupConflict as err:
                out = "group conflict: %s" % err
            except NoGroup as err:
                out = "no group: %s" % err
        if not out and not group.startswith("viruses."):
            out = "not a virus (group %s)" % group
        if not out:
            why = ex.reason(r["taxid"], [r["accession"], r["paired_accession"]])
            out = "excluded: %s" % why[0] if why else ""
        if not out:
            for h in hosts:
                hr = ex.host_reason(h)
                if hr:
                    out = hr
                    break
        if not out and rules.get("require_complete", True) and r.get("assembly_level") != "Complete Genome":
            out = "not a complete genome"
        if not out and r["protein_coding"] < 1:
            out = "no annotated protein"
        lo, ce, hi = scaled(r["protein_coding"], a["protein_factor"]["virus"])
        acc = r["accession"]
        cands.add(id="%s_%s" % ((group or "virus").replace("viruses.", "virus_"), acc), group=group, part="viruses", catalogue=catalogue, source_kind="raw_virus",
                  taxid=r["taxid"], host_taxid=host, species_taxid=lin["species"], accession=acc, paired_accession=r["paired_accession"],
                  ena_assembly=ena_assembly(acc, r["paired_accession"]), replicon_types="", table=11, proteins_low=lo, proteins_central=ce, proteins_high=hi,
                  proteins_level="annotated", quality=(r.get("refseq_category") != "reference genome", acc), out=out, host_known=bool(host),
                  **{k: lin[k] for k in RANKS if k != "species"})


def eukaryotes(cands, recs, annotation, catalogue_of, rules):
    tax, kmap, ex, a = cands.tax, cands.kmap, cands.ex, cands.a
    ann = {r["accession"]: r for r in annotation}
    rows = []
    for r in recs:
        lin, out = cands.base(r["taxonomy_id"])
        group = ""
        if not out:
            try:
                group = kmap.group(tax, r["taxonomy_id"], "annotated_cds")
            except NoGroup as err:
                out = "no group: %s" % err
        if not out and not group.startswith("eukaryota."):
            out = "not a eukaryote (group %s)" % group
        if not out:
            why = ex.reason(r["taxonomy_id"], [r["assembly_accession"]])
            out = "excluded: %s" % why[0] if why else ""
        if not out and not r.get("genebuild"):
            out = "no gene build"
        pc = 0
        a1 = ann.get(r["assembly_accession"]) or {}
        pc = a1.get("protein_coding") or 0
        if not pc and a1.get("paired_accession"):
            pc = (ann.get(a1["paired_accession"]) or {}).get("protein_coding") or 0
        quality = (r["taxonomy_id"] != lin["species"], r["name"])
        rows.append(cands.add(id="%s_%s" % (group.replace("eukaryota.", "") if group else "eukaryote", r["name"]), group=group, part="eukaryotes",
                              catalogue=catalogue_of[r["name"]], source_kind="annotated_cds", taxid=r["taxonomy_id"], species_taxid=lin["species"],
                              accession=r["assembly_accession"], paired_accession=a1.get("paired_accession", ""), ena_assembly="", replicon_types="",
                              ensembl_site=r["site"], ensembl_division=r["site_division"], ensembl_species=r["name"], ensembl_core_db=r.get("core_db", ""),
                              ensembl_release=r["release"], table=1, pc=pc, quality=quality, out=out, **{k: lin[k] for k in RANKS if k != "species"}))
    # protein counts: the NCBI annotation of the assembly; without one, the median of the group's annotated genomes
    med = {}
    for g in sorted({r["group"] for r in rows if r["group"]}):
        known = sorted(r["pc"] for r in rows if r["group"] == g and r["pc"])
        med[g] = known[len(known) // 2] if known else 0
    for r in rows:
        if r["pc"]:
            r["proteins_low"], r["proteins_central"], r["proteins_high"] = scaled(r["pc"], a["protein_factor"]["eukaryote_annotated"])
            r["proteins_level"] = "annotated (NCBI annotation of the same assembly; the Ensembl gene build may differ)"
        else:
            r["proteins_low"], r["proteins_central"], r["proteins_high"] = scaled(med.get(r["group"], 0), a["protein_factor"]["eukaryote_median"])
            r["proteins_level"] = "estimated (median of the group)"


def one_per_species(rows):
    """Best genome of every species (quality order), the others out; returns the eligible winners."""
    best = {}
    for r in sorted((r for r in rows if not r["out"]), key=lambda r: (r["species_taxid"], r["quality"], r["id"])):
        if r["species_taxid"] in best:
            r["out"] = "another genome of the species (%s) preferred" % best[r["species_taxid"]]["id"]
        else:
            best[r["species_taxid"]] = r
    return [best[s] for s in sorted(best)]


def pick_order(rows, outer, seed, group):
    """The whole eligible set of a group in pick order: (depth, row); depth = index of the species within its genus."""
    rng = random.Random(group_seed(seed, group))
    by_genus = {}
    for r in sorted(rows, key=lambda r: (r["genus"], r["species_taxid"], r["id"])):
        by_genus.setdefault(r["genus"], []).append(r)
    for g in sorted(by_genus):
        rng.shuffle(by_genus[g])
    by_outer = {}
    for g in sorted(by_genus):
        by_outer.setdefault(by_genus[g][0][outer], []).append(by_genus[g])
    out, depth = [], 0
    maxdepth = max((len(v) for v in by_genus.values()), default=0)
    while depth < maxdepth:
        queues = {o: [rs for rs in gl if depth < len(rs)] for o, gl in sorted(by_outer.items())}
        for o in sorted(queues):
            rng.shuffle(queues[o])
        outers = sorted(o for o in queues if queues[o])
        rng.shuffle(outers)
        while outers:
            nxt = []
            for o in outers:
                rs = queues[o].pop()
                out.append((depth, rs[depth]))
                if queues[o]:
                    nxt.append(o)
            outers = nxt
        depth += 1
    return out


def take(order, target):
    """Prefix of the pick order reaching `target` units."""
    chosen, units = [], 0
    for d, r in order:
        if units >= target:
            break
        chosen.append((d, r))
        units += r["units"]
    return chosen, units


def coverage(rows):
    return {k: len({r[k] for r in rows if r[k]}) for k in ("phylum", "class", "order", "family", "genus")}


def main():
    run_dir = main_args()
    with Stage(run_dir, "select", requires=("pre",)) as st:
        cfg = st.config
        with open(st.repo(cfg["select_spec"])) as f:
            spec = yaml.safe_load(f)
        with open(st.repo(spec["assumptions"])) as f:
            assumptions = yaml.safe_load(f)
        with open(os.path.join(run_dir, "data", "catalog", "manifest.json")) as f:
            cman = json.load(f)
        cas = os.path.join(cfg["data_root"], "cas")
        snap = {}
        for e in cman["entries"]:
            p = st.repo(e["path"]) if e["kind"] == "file" else os.path.join(cas, e["path"])
            if sha256_file(p, use_sidecar=False) != e["sha256"]:
                raise SystemExit("catalogue snapshot %s does not match its manifest" % e["id"])
            snap[e["id"]] = read_jsonl(p)
        tax = Taxonomy(st.repo(cfg["taxdump"]))
        kmap = KingdomMap(st.repo(cfg.get("kingdom_map", DEFAULT_MAP)))
        ex = Excluder(tax)
        if ex.problems:
            raise SystemExit("the exclusion rules do not resolve: %s" % ex.problems[:5])
        rules = spec["rules"]
        cands = Candidates(tax, kmap, ex, assumptions)
        for cid in rules["prokaryotes"]["catalogues"]:
            prokaryotes(cands, snap[cid], cid, rules["prokaryotes"])
        plasmids(cands, rules["plasmids"])
        for cid in rules["viruses"]["catalogues"]:
            viruses(cands, snap[cid], snap[rules["viruses"]["hosts"]], cid, rules["viruses"])
        ens, catalogue_of = [], {}
        for cid in rules["eukaryotes"]["catalogues"]:
            for r in snap[cid]:
                if r["name"] not in catalogue_of:                # a genome listed by two divisions is one candidate
                    catalogue_of[r["name"]] = cid
                    ens.append(r)
        eukaryotes(cands, ens, snap[rules["eukaryotes"]["annotation"]], catalogue_of, rules["eukaryotes"])
        ids = [r["id"] for r in cands.rows]
        if len(ids) != len(set(ids)):
            raise SystemExit("duplicate candidate ids")

        targets0 = select_targets(spec, kmap)
        targets = dict(targets0)
        part_of = {"bacteria": "prokaryotes", "archaea": "prokaryotes", "mobile.plasmid": "plasmids"}
        orders, eligible = {}, {}
        for g in kmap.names:
            part = part_of.get(g) or ("viruses" if g.startswith("viruses.") else "eukaryotes")
            rows = [r for r in cands.rows if r["group"] == g]
            win = one_per_species(rows)
            eligible[g] = win
            orders[g] = pick_order(win, rules[part].get("outer", "order"), spec["seed"], g)

        def choose():
            return {g: take(orders[g], targets[g]) for g in kmap.names}

        chosen = choose()
        floors = _floors() if spec.get("tier") == "full" else None
        repair = []
        if floors and spec.get("share_repair", True):
            rule = floors["protein_share"]
            for _ in range(10000):
                prot = {}
                for g, (ch, _) in chosen.items():
                    prot[_major(g)] = prot.get(_major(g), 0) + sum(r["proteins_central"] for _, r in ch)
                total = sum(prot.values()) or 1
                short = [cls for cls, key in (("viral_mobile", "min_viral_mobile"), ("eukaryotes", "min_eukaryotes")) if prot.get(cls, 0) / total < rule[key]]
                if not short:
                    break
                grown = False
                for cls in short:
                    for g in kmap.names:
                        if _major(g) == cls and chosen[g][1] < sum(r["units"] for _, r in orders[g]):
                            step = max(1, int(math.ceil(0.05 * (floors["min_genomes"][g]))))
                            targets[g] += step
                            grown = True
                if not grown:
                    repair.append(dict(note="cannot repair: no more candidates in %s" % short))
                    break
                chosen = choose()
            changed = {g: dict(floor_target=targets0[g], target=targets[g]) for g in kmap.names if targets[g] != targets0[g]}
            if changed:
                repair.append(dict(raised=changed))

        sel = []
        by_group = {}
        for g in kmap.names:
            ch, units = chosen[g]
            cap = rules[part_of.get(g) or ("viruses" if g.startswith("viruses.") else "eukaryotes")].get("cap_per_genus", 1)
            pick_of = {r["id"]: i for i, (_, r) in enumerate(orders[g])}
            for i, (d, r) in enumerate(ch):
                r["selected"], r["pick"] = 1, i + 1
                sel.append(r)
            for r in cands.rows:
                if r["group"] == g and not r.get("selected"):
                    r["pick"] = pick_of.get(r["id"], 0) + 1 if r["id"] in pick_of else ""
                    if not r["out"]:
                        r["out"] = "not needed for the target (pick %d)" % (pick_of[r["id"]] + 1)
            rows_g = [r for r in cands.rows if r["group"] == g]
            outs = {}
            for r in rows_g:
                if not r.get("selected"):
                    k = r["out"].split(" (")[0].split(":")[0]
                    outs[k] = outs.get(k, 0) + 1
            sel_rows = [r for _, r in ch]
            by_group[g] = dict(floor=(floors["min_genomes"][g] if floors else None), target=targets[g], floor_target=targets0[g], selected_genomes=len(ch),
                               selected_units=units, eligible_species=len(eligible[g]), eligible_units=sum(r["units"] for r in eligible[g]), candidates=len(rows_g),
                               out=dict(sorted(outs.items())), max_depth=max((d for d, _ in ch), default=-1) + 1, cap_per_genus=cap,
                               cap_passed=max((d for d, _ in ch), default=-1) + 1 > cap, coverage_selected=coverage(sel_rows), coverage_eligible=coverage(eligible[g]),
                               proteins={b: sum(r["proteins_" + b] for r in sel_rows) for b in ("low", "central", "high")},
                               proteins_levels=dict(sorted({lv: sum(1 for r in sel_rows if r["proteins_level"] == lv) for lv in {r["proteins_level"] for r in sel_rows}}.items())),
                               host_unknown=sum(1 for r in sel_rows if r["part"] == "viruses" and not r.get("host_known")))
        ungrouped = {}                                   # rejected before a group could be given (conflicts, unknown taxids)
        for r in cands.rows:
            if not r["group"]:
                k = r["out"].split(" (")[0].split(":")[0]
                ungrouped.setdefault(r["part"], {})[k] = ungrouped.get(r["part"], {}).get(k, 0) + 1
        ungrouped = {p: dict(sorted(v.items())) for p, v in sorted(ungrouped.items())}
        sel.sort(key=lambda r: (r["group"], r["pick"]))
        shares = {}
        for b in ("low", "central", "high"):
            prot = {}
            for g, d in by_group.items():
                prot[_major(g)] = prot.get(_major(g), 0) + d["proteins"][b]
            total = sum(prot.values()) or 1
            shares[b] = {k: round(v / total, 4) for k, v in sorted(prot.items())}
        inputs = dict(taxdump=taxdump_hash(st.repo(cfg["taxdump"])), kingdom_map=sha256_file(st.repo(cfg.get("kingdom_map", DEFAULT_MAP)), use_sidecar=False),
                      exclude=sha256_file(EXCLUDE, use_sidecar=False), panels=sha256_file(PANELS, use_sidecar=False) if os.path.exists(PANELS) else "absent",
                      event_catalogue=sha256_file(CATALOGUE, use_sidecar=False) if os.path.exists(CATALOGUE) else "absent",
                      sealed=sha256_file(SEALED, use_sidecar=False), spec=sha256_json(spec), assumptions=sha256_json(assumptions), catalog_manifest=cman["sha256"],
                      seed=spec["seed"])
        out_sel = st.path("select", "selected.tsv")
        with open(out_sel, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(SELECTED_COLS)
            for r in sel:
                w.writerow([r.get(c, "") for c in SELECTED_COLS])
        out_c = st.path("select", "candidates.tsv")
        with open(out_c, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(CAND_COLS)
            for r in sorted(cands.rows, key=lambda r: (r["group"], r["id"])):
                w.writerow([r.get(c, "") if c != "selected" else int(bool(r.get("selected"))) for c in CAND_COLS])
        bg = st.path("select", "by_group.json")
        write_json(bg, dict(spec=spec["name"], tier=spec.get("tier"), groups=by_group, ungrouped=ungrouped, shares=shares, repair=repair, inputs=inputs,
                            catalogues={e["id"]: dict(kind=e["kind"], records=e["records"], sha256=e["sha256"]) for e in cman["entries"]},
                            selected_genomes=len(sel), selected_units=sum(r["units"] for r in sel)))
        man = M.build("select-" + spec["name"], "selection", [dict(id=r["id"], group=r["group"], taxid=int(r["taxid"]), accession=r["accession"], units=r["units"]) for r in sel],
                      meta=dict(inputs=inputs))
        msha = M.write(man, os.path.join(run_dir, "manifests"))
        mp = st.path("select", "manifest.json")
        write_json(mp, dict(sha256=msha, dataset_id=M.dataset_id(man), **{k: v for k, v in man.items() if k != "entries"}, entries=len(man["entries"])))
        for name, p in (("selected", out_sel), ("candidates", out_c), ("by_group", bg), ("select_manifest", mp)):
            st.output(name, p)
        st.extra.update(selected=len(sel), candidates=len(cands.rows))


if __name__ == "__main__":
    main()
