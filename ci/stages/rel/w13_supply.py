"""Stage rel_w13_supply (package W13; pipeline rel_corpus): how many genomes two groups could still gain, metadata only
(the programme lead, 2026-11-23, with the user's word "先数": the groups that overfit first in round 1a, eukaryotic
viruses, which have few genes that can be a target, and plants and algae, which have few training genomes).
`python -m ci.stages.rel.w13_supply RUN_DIR`. Nothing is downloaded but catalogue metadata; nothing is added.

Snapshots (NCBI Datasets genome reports, through the catalogue fetcher of stage catalog: every page, accepted only when
the distinct records equal the announced total, canonical, stored in the content store; query, records, sha256 and
date in lock.yaml): complete viral assemblies (taxon 10239, assembly level complete genome) and annotated assemblies
of plants and algae (the taxa of the kingdom map's group eukaryota.plants_algae, has_annotation). The selection's host
snapshots (NCBI virus reports, checked against its manifest) give the hosts of a virus species, as in the selection.

Eukaryotic viruses: an assembly counts when the selection's rules place it in viruses.eukaryotic (its hosts; the
exclusions by full lineage of gate G2 and the extra lists; a host by lineage) and its assembly is not a corpus genome's;
those the host rule leaves out are counted apart (no host record, only an environment host, hosts in two classes).
Genes: the annotation's protein-coding genes, else the genome length times the selection's assumed viral coding
density (counted apart). Plants and algae: annotated assemblies of species not in the corpus, after the same
exclusions, one assembly per species (the most complete level, then the most genes). Both: by family and genus (by
order and family, and assembly level, for plants), and the split each would take under the corpus's rule (stage
train_data: a genome of a genus or species already in the corpus goes with it; another one by the hash of its unit:
held out one unit in five, then test one in four of the rest): the training and held-out genomes and genes, and those
that would fall in the sealed test part (a decision, not counted for training). Current numbers come from the source
run's training data (the genomes of each split and their genes with a codon). Plants: the CDS download, estimated
from the genes and an assumed coding length. The sealed test genomes are not read; the sealed check refuses any record
it names.

Configuration (profile block rel_w13_supply): select_from (the selection run), source (the corpus run whose splits
count), expect, cds_bp (assumed mean coding length of a plant gene, default 1,300), page_size. Outputs under
data/rel_w13/supply/: viruses.jsonl and plants.jsonl (the snapshots), lock.yaml, report.json (written last), report.md.
Gate G4_rel_w13_supply.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil

import numpy as np
import yaml

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from ci.stages.rel import w2_v0 as W
from ci.stages.rel import w13_census as WC

OUT = os.path.join("data", "rel_w13", "supply")
KEYS = {"select_from", "source", "expect", "cds_bp", "page_size", "test_unit_every", "heldout_unit_every"}
VIRUSES = dict(id="viruses", ncbi_genome_reports=dict(taxon=10239, params={"filters.assembly_level": "complete_genome"}))
PLANTS = dict(id="plants", ncbi_genome_reports=dict(taxon=[33090, 2763, 38254], params={"filters.has_annotation": True}))
GROUPS = ("viruses.eukaryotic", "eukaryota.plants_algae")


def settings(cfg) -> dict:
    c = cfg.get("rel_w13_supply") or {}
    unknown = set(c) - KEYS
    if unknown or not c.get("select_from") or not c.get("source"):
        raise SystemExit("rel_w13_supply: needs select_from and source; unknown keys %s" % sorted(unknown))
    return dict(select_from=str(c["select_from"]), source=str(c["source"]), expect={str(k): int(v) for k, v in (c.get("expect") or {}).items()},
                cds_bp=int(c.get("cds_bp", 1300)), page_size=int(c.get("page_size", 1000)), max_left_out=20)


def unit_split(unit):
    """The part a new unit takes under stage train_data's rule (split(), then three_way()): held out when the hash of
    its key under the salt "split" is 0 mod 5, else test when the hash under "test" is 0 mod 4, else train (the
    fix-ups that move whole units when a group would have no held-out unit do not apply to these groups)."""
    h = lambda salt, every: int(hashlib.sha256(("%s:%s" % (salt, unit)).encode()).hexdigest(), 16) % every == 0   # noqa: E731
    return "heldout" if h("split", 5) else ("test" if h("test", 4) else "train")


def unit_of(genus, species, gid):
    return ("g", int(genus)) if genus else (("s", int(species)) if species else ("id", gid))


def current(src, metas_all):
    """{group: {split: dict(genomes, genes)}} of the corpus (every split's genome count; genes with a codon of the open
    splits, read from the training data's codon offsets; the test part is counted, never read)."""
    out = {}
    for m in metas_all:
        if m["group"] not in GROUPS:
            continue
        d = out.setdefault(m["group"], {}).setdefault(m["split"], dict(genomes=0, genes=0))
        d["genomes"] += 1
        if m["split"] in ("train", "heldout"):
            p = os.path.join(src["run"], "data", "train_data", m["id"] + ".npz")
            with np.load(p) as z:
                d["genes"] += int((np.diff(z["codon_off"]) > 0).sum())
    return out


def corpus_units(metas_all):
    """{group: {unit: split}} of the corpus genomes (all splits; by genus, else species, as stage train_data)."""
    out = {}
    for m in metas_all:
        out.setdefault(m["group"], {})[unit_of(m["ranks"][4], m["ranks"][5], m["id"])] = m["split"]
    return out


def place(group, genus, species, gid, units):
    u = unit_of(genus, species, gid)
    return units.get(group, {}).get(u) or unit_split(u)


def build(st, cfg, run_dir):
    from ci.gates.g2_exclusion import Excluder
    from ci.kingdom import KingdomMap
    from ci.stages import catalog as CAT
    from ci.stages import fetch
    from ci.taxonomy import Taxonomy
    from seeddrift.train.rel import sealed as SL
    s = settings(cfg)
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    src = W.source(s["source"], pairs=False)
    sealed = common.sealed_set(cfg)
    corpus = [m for m in src["index"]["genomes"] if m.get("source", "corpus") == "corpus"]
    opened, left, _ = W.open_genomes(src, s, sealed)                  # the open genomes the sealed check does not refuse
    gone = {x["id"] for x in left}
    corpus = [m for m in corpus if m["id"] not in gone]
    st.extra["sealed_check"] = common.sealed_check(cfg, [W.entry(src, m) for m in opened], splits={m["id"]: m["split"] for m in opened}, sealed=sealed, tax=tax)
    now = current(src, corpus)
    units = corpus_units(corpus)
    corpus_asm = {a for m in corpus for a in SL.assemblies(m["id"]) | SL.assemblies((src["corpus"].get(m["id"]) or {}).get("accession", ""))
                  | SL.assemblies((src["corpus"].get(m["id"]) or {}).get("paired_accession", ""))}
    corpus_species = {(m["group"], int(m["ranks"][5])) for m in corpus if m["ranks"][5]}
    km, ex = KingdomMap(), Excluder(tax, extra_paths=[st.repo(x) for x in cfg.get("extra_exclusions") or []])
    if ex.problems:
        raise SystemExit("rel_w13_supply: the exclusion rules do not resolve: %s" % ex.problems[:5])
    host_recs, host_snap = WC.host_records(s["select_from"])
    hosts = {}
    for h in host_recs:
        try:
            sp = (tax.at_rank(h["virus_taxid"], "species") or tax.resolve(h["virus_taxid"])) if h["virus_taxid"] else 0
        except Exception:  # noqa: BLE001 - a virus taxid the taxonomy does not know
            continue
        if sp and h["host_taxid"]:
            hosts.setdefault(sp, set()).add(h["host_taxid"])
    with open(st.repo("configs/select/assumptions.yaml")) as f:
        per_kb = float(yaml.safe_load(f)["virus_proteins_per_kb"][1])
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    cas = os.path.join(cfg["data_root"], "cas")
    lock, recs = dict(source="NCBI Datasets v2 genome reports", catalogues={}, host_snapshots=host_snap), {}
    for c in (VIRUSES, PLANTS):
        c = json.loads(json.dumps(c))
        c["ncbi_genome_reports"]["page_size"] = s["page_size"]
        data, n, prov = CAT.snapshot(c, {}, cas)
        path, sha = fetch.store(cas, data, ".jsonl")
        shutil.copyfile(path, os.path.join(folder, c["id"] + ".jsonl"))
        lock["catalogues"][c["id"]] = dict(query=c["ncbi_genome_reports"], records=n, sha256=sha, bytes=len(data), provenance=prov,
                                           fetched=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        recs[c["id"]] = CAT.read_jsonl(os.path.join(folder, c["id"] + ".jsonl"))
    with open(os.path.join(folder, "lock.yaml"), "w") as f:
        f.write("# Snapshots of stage rel_w13_supply (metadata only; no sequence was downloaded).\n")
        yaml.safe_dump(lock, f, sort_keys=True)
    rep = dict(format="rel-w13-supply", version=1, note=__doc__.strip().splitlines()[0], source=src["provenance"], current=now,
               viruses=viruses(recs["viruses"], tax, km, ex, hosts, corpus_asm, units, sealed, per_kb),
               plants=plants(recs["plants"], tax, km, ex, corpus_asm, corpus_species, units, sealed, s["cds_bp"]),
               settings={k: s[k] for k in ("cds_bp",)}, assumed_virus_proteins_per_kb=per_kb,
               files={k: dict(path=os.path.join(OUT, k), sha256=sha256_file(os.path.join(folder, k), use_sidecar=False))
                      for k in ("viruses.jsonl", "plants.jsonl", "lock.yaml")})
    page = os.path.join(folder, "report.md")
    with open(page, "w") as f:
        f.write(page_of(rep))
    st.output("supply_page", page)
    p = os.path.join(folder, "report.json")
    write_json(p, rep)
    st.output("supply_report", p)
    st.extra.update(viruses=rep["viruses"]["totals"], plants=rep["plants"]["totals"])
    return rep


def _bump(d, key, genes):
    x = d.setdefault(key, dict(genomes=0, genes=0))
    x["genomes"] += 1
    x["genes"] += int(genes)


def viruses(recs, tax, km, ex, hosts, corpus_asm, units, sealed, per_kb):
    """The complete viral assemblies the selection's rules would place in viruses.eukaryotic, not in the corpus."""
    from seeddrift.train.rel import sealed as SL
    seen, reasons, by_family, by_genus, by_split, totals = set(), {}, {}, {}, {}, dict(genomes=0, genes=0, genes_estimated_genomes=0)
    cache = {}
    for r in sorted(recs, key=lambda r: r["accession"]):
        asm = SL.assemblies(r["accession"]) | SL.assemblies(r.get("paired_accession") or "")
        if asm & seen:
            continue                                                 # a GenBank assembly and its RefSeq twin: one genome
        seen |= asm
        t = int(r.get("taxid") or 0)
        if t not in cache:
            try:
                sp = tax.at_rank(t, "species") or tax.resolve(t)
            except Exception:  # noqa: BLE001
                sp = 0
            cache[t] = WC.wide_reason(tax, km, ex, t, sorted(hosts.get(sp, ())), False) if sp else ("taxid unknown", "", 0, 0, 0)
        why, grp, sp, ge, _ = cache[t]
        if why is None and grp != "viruses.eukaryotic":
            continue                                                 # a phage or a giant virus: another group
        if why and why.startswith(("host unknown", "excluded by a host rule (hosts in more")):
            try:
                by_taxon = km.group(tax, t, "raw_virus", None)
            except Exception:  # noqa: BLE001
                by_taxon = ""
            if by_taxon != "viruses.eukaryotic":
                continue
        elif why:
            if grp and grp != "viruses.eukaryotic":
                continue
        if why is None and asm & corpus_asm:
            why = "already in the corpus"
        if why is None and sealed.reasons(r["accession"]):
            why = "refused by the sealed check"
        if why:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        genes = int(r.get("protein_coding") or 0)
        if not genes:
            genes = max(1, int(round(int(r.get("length") or 0) / 1000.0 * per_kb)))
            totals["genes_estimated_genomes"] += 1
        fam = tax.at_rank(t, "family")
        _bump(by_family, tax.name(fam) if fam else "(no family)", genes)
        _bump(by_genus, tax.name(ge) if ge else "(no genus)", genes)
        _bump(by_split, place("viruses.eukaryotic", ge, sp, r["accession"], units), genes)
        totals["genomes"] += 1
        totals["genes"] += genes
    top = lambda d: dict(sorted(d.items(), key=lambda kv: (-kv[1]["genomes"], kv[0]))[:40])   # noqa: E731
    return dict(totals=totals, by_split=by_split, by_family=top(by_family), by_genus=top(by_genus), families=len(by_family), genera=len(by_genus),
                left_out=dict(sorted(reasons.items())))


def plants(recs, tax, km, ex, corpus_asm, corpus_species, units, sealed, cds_bp):
    """Annotated assemblies of plants and algae of species not in the corpus: one per species."""
    from seeddrift.train.rel import sealed as SL
    rank = {"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}
    reasons, best = {}, {}
    for r in recs:
        t = int(r.get("taxid") or 0)
        try:
            grp = km.group(tax, t, "annotated_cds")
            sp, ge = tax.at_rank(t, "species"), tax.at_rank(t, "genus")
        except Exception:  # noqa: BLE001
            reasons["taxid unknown or no group"] = reasons.get("taxid unknown or no group", 0) + 1
            continue
        if grp != "eukaryota.plants_algae":
            continue
        why = None
        if not int(r.get("protein_coding") or 0):
            why = "no protein-coding genes in the annotation"
        elif ex.reason(t, [r["accession"], r.get("paired_accession") or ""]):
            why = "excluded by lineage"
        elif (SL.assemblies(r["accession"]) | SL.assemblies(r.get("paired_accession") or "")) & corpus_asm:
            why = "already in the corpus"
        elif sp and (grp, int(sp)) in corpus_species:
            why = "a species of the corpus"
        elif sealed.reasons(r["accession"]):
            why = "refused by the sealed check"
        elif not sp:
            why = "no species"
        if why:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        key = (rank.get(r.get("assembly_level"), 9), -int(r["protein_coding"]), r["accession"])
        if sp not in best or key < best[sp][0]:
            best[sp] = (key, r, ge)
    by_level, by_order, by_family, by_split, totals = {}, {}, {}, {}, dict(species=0, genes=0, cds_bytes=0)
    for sp, (_, r, ge) in sorted(best.items()):
        genes = int(r["protein_coding"])
        t = int(r["taxid"])
        _bump(by_level, r.get("assembly_level") or "(none)", genes)
        o, f = tax.at_rank(t, "order"), tax.at_rank(t, "family")
        _bump(by_order, tax.name(o) if o else "(no order)", genes)
        _bump(by_family, tax.name(f) if f else "(no family)", genes)
        _bump(by_split, place("eukaryota.plants_algae", ge, sp, r["accession"], units), genes)
        totals["species"] += 1
        totals["genes"] += genes
        totals["cds_bytes"] += genes * cds_bp
    totals["cds_gb"] = round(totals["cds_bytes"] / 1e9, 2)
    totals["cds_gb_gzip_about"] = round(totals["cds_bytes"] * 0.3 / 1e9, 2)
    top = lambda d: dict(sorted(d.items(), key=lambda kv: (-kv[1]["genomes"], kv[0]))[:40])   # noqa: E731
    return dict(totals=totals, by_split=by_split, by_level=by_level, by_order=top(by_order), by_family=top(by_family), orders=len(by_order),
                families=len(by_family), left_out=dict(sorted(reasons.items())))


def page_of(rep):
    cur, v, p = rep["current"], rep["viruses"], rep["plants"]
    fmt = lambda x: "{:,}".format(x)                                    # noqa: E731

    def now(g, sp):
        d = (cur.get(g) or {}).get(sp) or {}
        return "%s genomes, %s genes" % (fmt(d.get("genomes", 0)), fmt(d.get("genes", 0)))
    L = ["# How many genomes eukaryotic viruses and plants and algae could still gain (package W13)", "", rep["note"], "",
         "Source run `%s`. Metadata only; nothing downloaded." % rep["source"]["run_id"], "",
         "## Eukaryotic viruses", "",
         "Now: train %s; held out %s." % (now("viruses.eukaryotic", "train"), now("viruses.eukaryotic", "heldout")),
         "Complete assemblies the selection's rules place here, not in the corpus: %s genomes, %s genes (%s of them estimated from the length)." % (
             fmt(v["totals"]["genomes"]), fmt(v["totals"]["genes"]), fmt(v["totals"]["genes_estimated_genomes"])),
         "By the split rule: " + "; ".join("%s %s genomes, %s genes" % (k, fmt(x["genomes"]), fmt(x["genes"])) for k, x in sorted(v["by_split"].items())) + ".",
         "Left out: " + "; ".join("%s %s" % (k, fmt(x)) for k, x in v["left_out"].items()) + ".", "",
         "| family | genomes | genes |", "|---|---|---|"] + ["| %s | %s | %s |" % (k, fmt(x["genomes"]), fmt(x["genes"])) for k, x in list(v["by_family"].items())[:25]]
    L += ["", "## Plants and algae", "",
          "Now: train %s; held out %s." % (now("eukaryota.plants_algae", "train"), now("eukaryota.plants_algae", "heldout")),
          "Annotated assemblies of species not in the corpus (one per species): %s species, %s genes; their CDS about %s GB (%s GB gzip, assumed %d bp a gene)." % (
              fmt(p["totals"]["species"]), fmt(p["totals"]["genes"]), p["totals"]["cds_gb"], p["totals"]["cds_gb_gzip_about"], rep["settings"]["cds_bp"]),
          "By the split rule: " + "; ".join("%s %s genomes, %s genes" % (k, fmt(x["genomes"]), fmt(x["genes"])) for k, x in sorted(p["by_split"].items())) + ".",
          "By assembly level: " + "; ".join("%s %s" % (k, fmt(x["genomes"])) for k, x in sorted(p["by_level"].items())) + ".",
          "Left out: " + "; ".join("%s %s" % (k, fmt(x)) for k, x in p["left_out"].items()) + ".", "",
          "| order | species | genes |", "|---|---|---|"] + ["| %s | %s | %s |" % (k, fmt(x["genomes"]), fmt(x["genes"])) for k, x in list(p["by_order"].items())[:25]]
    L += ["", "Genomes the split rule would put in the test part join the sealed test split: whether to take them is a decision, they are not training genomes.", ""]
    return "\n".join(L)


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "report.json")
    rep = json.load(open(p))
    problems = []
    for name, e in (rep.get("files") or {}).items():
        full = os.path.join(run_dir, e["path"])
        if not os.path.exists(full) or sha256_file(full, use_sidecar=False) != e["sha256"]:
            problems.append(dict(file=name, error="missing or not the file the report names"))
    for k in ("viruses", "plants"):
        d = rep.get(k) or {}
        if sum(x["genomes"] for x in (d.get("by_split") or {}).values()) != (d.get("totals") or {}).get("genomes", (d.get("totals") or {}).get("species")):
            problems.append(dict(part=k, error="the split counts do not add up"))
    require(GateResult("G4_rel_w13_supply", "fail" if problems else "pass", dict(problems=problems[:50], viruses=(rep.get("viruses") or {}).get("totals"),
                                                                               plants=(rep.get("plants") or {}).get("totals"))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    rep = json.load(open(os.path.join(run_dir, OUT, "report.json")))
    return dict(json=dict(current=rep["current"], viruses=rep["viruses"]["totals"], plants=rep["plants"]["totals"]), md=page_of(rep).splitlines()[2:40])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w13_supply") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
