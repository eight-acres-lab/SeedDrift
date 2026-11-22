"""Stage rel_w13_census (package W13; pipeline rel_corpus): how many close relatives (the same species, the same genus)
of the corpus genomes the catalogue snapshots already hold, after every exclusion, and the size of a first batch per
group. `python -m ci.stages.rel.w13_census RUN_DIR`. Reads lists only; nothing is downloaded and no rule is added.

Inputs (each checked): the candidate and selected tables of the passed selection run the corpus was emitted from
(rel_w13_census.select_from; selected.tsv must be the file the corpus spec's header names), the training-data index of
the passed run that split the corpus three ways (rel_w13_census.source; its splits) and the catalogue lock of the
selection (its record counts and the totals of broader queries). A candidate is a relative of a corpus genome A when it
is of A's group and of A's species (same species) or of A's genus and another species (same genus), and it is
eligible: it passed every rule of the selection (the exclusions by full lineage: sealed genera and their families,
the excluded order, the provisional test recipients; a virus's hosts; group conflicts; quality) and was left out only
because the target was reached or another genome of its species was preferred. Relatives are counted for the genomes
of the train and held-out splits only; a relative goes with its anchor's split (the split keeps genera whole), so the
sealed test split gains none. The sealed test genomes are not read.

Configuration (profile block rel_w13_census): select_from, source, expect ({split: genomes}), per_anchor (relatives per
anchor in the proposals, default [1, 2, 3, 5]), propose (the per_anchor of the proposal, default 3), max_genes_euk
(the gene cap of eukaryotes in the corpus, default 1500), catalog_from (a passed run of stage rel_w13_catalog, or
"this run" when the profile runs that stage first: the count is made a second time on its wider snapshot and reported
beside the first; see census_wide).
Outputs under data/rel_w13/census/: report.json (written last) and report.md.
Gate G4_rel_w13_census (gate()): every group of the corpus is counted, every count is consistent, the selected table is
the corpus's, and no genome outside the open splits is named.
"""
from __future__ import annotations

import csv
import json
import os
import re

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from ci.stages.rel import w2_v0 as W
from ci.kingdom import GroupConflict, KingdomMap, NoGroup
from ci.taxonomy import UnknownTaxid

OUT = os.path.join("data", "rel_w13", "census")
KEYS = {"select_from", "source", "expect", "per_anchor", "propose", "max_genes_euk", "catalog_from"}
ELIGIBLE = ("not needed for the target", "another genome of the species")
OPEN = ("train", "heldout")


def settings(cfg) -> dict:
    c = cfg.get("rel_w13_census") or {}
    unknown = set(c) - KEYS
    if unknown or not c.get("select_from") or not c.get("source"):
        raise SystemExit("rel_w13_census: needs `select_from` and `source`; unknown keys %s" % sorted(unknown))
    out = dict(select_from=str(c["select_from"]), source=str(c["source"]), expect={str(k): int(v) for k, v in (c.get("expect") or {}).items()},
               per_anchor=[int(x) for x in c.get("per_anchor") or [1, 2, 3, 5]], propose=int(c.get("propose", 3)), max_genes_euk=int(c.get("max_genes_euk", 1500)),
               catalog_from=str(c["catalog_from"]) if c.get("catalog_from") else None, max_left_out=20)
    if not out["per_anchor"] or min(out["per_anchor"]) < 1 or out["propose"] not in out["per_anchor"]:
        raise SystemExit("rel_w13_census: per_anchor needs positive counts and must include propose: %r" % out)
    return out


def reason_kind(out):
    """The kind of a candidate's rejection: eligible, or why not."""
    o = out or ""
    if not o:
        return "selected"
    if o.startswith(ELIGIBLE):
        return "eligible"
    for key, kind in (("excluded:", "excluded by lineage"), ("host", "excluded by a host rule"), ("group conflict", "group conflict"),
                      ("no plasmid", "not a genome (an assembly without plasmids)"), ("fewer than", "quality"), ("no genus", "quality"), ("no species", "quality")):
        if o.startswith(key) or (key == "host" and "host" in o[:40]):
            return kind
    return "other"


def selection(s):
    """(candidate rows, selected rows, provenance) of the selection run, checked."""
    run = s["select_from"]
    rp = os.path.join(run, "record.json")
    if not os.path.exists(rp):
        raise SystemExit("rel_w13_census: %s holds no run record" % run)
    with open(rp) as f:
        rec = json.load(f)
    if rec.get("status") != "passed" or rec.get("pipeline", "select") != "select":
        raise SystemExit("rel_w13_census: %s is not a passed selection run" % run)
    paths = {n: os.path.join(run, "data", "select", n) for n in ("candidates.tsv", "selected.tsv")}
    shas = {n: sha256_file(p, use_sidecar=False) for n, p in paths.items()}
    with open(os.path.join(run, "config.resolved.yaml")) as f:
        import yaml
        cfg = yaml.safe_load(f)
    tables = {}
    for n, p in paths.items():
        with open(p) as f:
            tables[n] = list(csv.DictReader(f, delimiter="\t"))
    return tables["candidates.tsv"], tables["selected.tsv"], dict(run_id=rec.get("run_id") or os.path.basename(run), commit=(rec.get("git") or {}).get("sha"),
                                                                  sha256=shas, lock=(cfg.get("catalog") or {}).get("lock"), select_spec=cfg.get("select_spec"))


def corpus_header_sha(path):
    with open(path) as f:
        head = f.read(4000)
    m = re.search(r"selected\.tsv sha256 ([0-9a-f]{64})", head)
    return m.group(1) if m else None


def lock_totals(path):
    """{catalogue: dict(records, totals {query: total})} of a catalogue lock."""
    import yaml
    with open(path) as f:
        doc = yaml.safe_load(f)
    return {k: dict(records=v.get("records"), totals={q: t.get("total") for q, t in (v.get("totals") or {}).items()}) for k, v in (doc.get("catalogues") or {}).items()}


def census(cands, metas, s):
    """Per open corpus genome its eligible relatives; per group and split the counts and the proposals."""
    by_species, by_genus = {}, {}
    for r in cands:
        if reason_kind(r["out"]) not in ("selected", "eligible") or not r.get("group"):
            continue
        sp, ge = int(r.get("species_taxid") or 0), int(r.get("genus") or 0)
        if sp:
            by_species.setdefault((r["group"], sp), []).append(r)
        if ge:
            by_genus.setdefault((r["group"], ge), []).append(r)
    euk = lambda g: g.startswith("eukaryota.")                         # noqa: E731
    prot = lambda r: min(int(float(r.get("proteins_central") or 0)), s["max_genes_euk"]) if euk(r["group"]) else int(float(r.get("proteins_central") or 0))  # noqa: E731
    anchors = []
    for m in metas:
        sp, ge = int(m["ranks"][5] or 0), int(m["ranks"][4] or 0)
        same_sp = [r for r in by_species.get((m["group"], sp), []) if r["id"] != m["id"]] if sp else []
        same_ge = [r for r in by_genus.get((m["group"], ge), []) if r["id"] != m["id"] and int(r.get("species_taxid") or 0) != sp] if ge else []
        new_sp = sorted((r for r in same_sp if r.get("selected") != "1"), key=lambda r: r["id"])
        new_ge = sorted((r for r in same_ge if r.get("selected") != "1"), key=lambda r: r["id"])
        anchors.append(dict(id=m["id"], group=m["group"], split=m["split"], species=sp, genus=ge, same_species_in_corpus=sum(1 for r in same_sp if r.get("selected") == "1"),
                            same_genus_in_corpus=sum(1 for r in same_ge if r.get("selected") == "1"), new_same_species=new_sp, new_same_genus=new_ge))
    groups = sorted({a["group"] for a in anchors})
    out = {}
    for split in OPEN:
        by = {}
        for g in groups + ["all"]:
            sel = [a for a in anchors if a["split"] == split and (g == "all" or a["group"] == g)]
            if not sel:
                continue
            d = dict(anchors=len(sel))
            for kind in ("new_same_species", "new_same_genus"):
                n = [len(a[kind]) for a in sel]
                d[kind] = dict(anchors_with_one=sum(1 for x in n if x >= 1), anchors_with_three=sum(1 for x in n if x >= 3), anchors_with_ten=sum(1 for x in n if x >= 10),
                               candidates=len({r["id"] for a in sel for r in a[kind]}))
            d["in_corpus"] = dict(same_species=sum(a["same_species_in_corpus"] for a in sel), same_genus=sum(a["same_genus_in_corpus"] for a in sel))
            d["proposals"] = {}
            for k in s["per_anchor"]:
                picked = {}
                for a in sel:                                          # the same species first, then the same genus, k per anchor
                    for r in (a["new_same_species"] + a["new_same_genus"])[:k]:
                        picked[r["id"]] = r
                d["proposals"][str(k)] = dict(genomes=len(picked), proteins=int(sum(prot(r) for r in picked.values())),
                                              same_species=sum(1 for a in sel for r in a["new_same_species"][:k]))
            by[g] = d
        out[split] = by
    return anchors, out


PROKARYOTES = ("bacteria", "archaea")


def wide_reason(tax, km, ex, t, host, prokaryote):
    """(why not or None, group, species, genus) of a taxid (and a virus's host) under the rules of the selection."""
    try:
        tax.lineage(int(t))
    except (UnknownTaxid, KeyError, ValueError):
        return "taxid unknown", "", 0, 0
    sp, ge = tax.at_rank(int(t), "species"), tax.at_rank(int(t), "genus")
    try:
        grp = km.group(tax, int(t), "raw_prokaryote" if prokaryote else "raw_virus", None if prokaryote else (host or None))
    except GroupConflict:
        return "group conflict", "", sp, ge
    except (NoGroup, UnknownTaxid, KeyError, ValueError):
        return "no group", "", sp, ge
    if (grp in PROKARYOTES) != prokaryote or not (prokaryote or grp.startswith("viruses.")):
        return "not of the catalogue's kind (group %s)" % grp, grp, sp, ge
    why = ex.reason(int(t), [])
    if why:
        return ("quality (no species)" if "above species" in why[0] else "taxid unknown" if "resolve" in why[0] else "excluded by lineage"), grp, sp, ge
    if prokaryote and not ge:
        return "quality (no genus)", grp, sp, ge
    if not prokaryote:
        if not host:
            return "host unknown (the selection's rule: a virus whose host cannot be checked is left out)", grp, sp, ge
        try:
            cls = km.host_class(tax, host)
        except (UnknownTaxid, KeyError, ValueError):
            return "host unknown (the selection's rule: a virus whose host cannot be checked is left out)", grp, sp, ge
        if not cls:
            return "host unknown (the selection's rule: a virus whose host cannot be checked is left out)", grp, sp, ge
        if ex.host_reason(host):
            return "excluded by a host rule", grp, sp, ge
    return None, grp, sp, ge


def census_wide(cats, metas, entries, cands, s, tax, sealed, extra_paths):
    """The same count on a wider catalogue snapshot (stage rel_w13_catalog: ENA assemblies of full genome representation
    at the levels of that snapshot), with the rules of the selection: the group by the kingdom map (a virus by its host);
    the exclusions of gate G2 by full lineage (bench/hard/EXCLUDE.tsv, which lists the sealed genera with their families
    and the closed development order, and the extra lists of the profile: the provisional test recipients); for a virus
    its host by lineage, and a virus whose host is unknown or only an environment is left out (the selection's default
    rule for hosts); prokaryotes need a genus and a species. The host of a virus is the record's host, else the host the
    selection knew for its species. A record whose assembly is a corpus genome of the open splits counts as already in
    the corpus; one the sealed check refuses is left out and not named. Plasmid anchors count the prokaryote assemblies
    of their host species, an upper bound (whether those assemblies carry plasmids is not in the snapshot); groups the
    snapshot has no catalogue for (the eukaryotes) keep the selection's candidates. Proteins: one per kilobase."""
    from ci.gates.g2_exclusion import Excluder
    from seeddrift.train.rel import sealed as SL
    km, ex = KingdomMap(), Excluder(tax, extra_paths=extra_paths)
    if ex.problems:
        raise SystemExit("rel_w13_census: the exclusion rules do not resolve: %s" % ex.problems[:5])
    own = {}                                                    # (group, assembly number) -> corpus genome of the open splits
    for m, e in zip(metas, entries):
        for a in SL.assemblies(m["id"]) | SL.assemblies(e.get("accession", "")) | SL.assemblies(e.get("paired_accession", "")):
            own[(m["group"], a)] = m["id"]
    hosts = {}
    for r in cands:                                             # the selection's host of a virus species
        if (r.get("group") or "").startswith("viruses.") and int(r.get("host_taxid") or 0) and int(r.get("species_taxid") or 0):
            hosts.setdefault(int(r["species_taxid"]), int(r["host_taxid"]))
    reasons, rows, cache = {}, [], {}
    for cid, recs in cats.items():
        for r in recs:
            acc, t = r["accession"], int(r.get("tax_id") or 0)
            prok = cid in PROKARYOTES
            host = int(r["host_tax_id"]) if (r.get("host_tax_id") or "").strip().isdigit() else 0
            if not prok and not host:
                try:
                    host = hosts.get(tax.at_rank(int(t), "species"), 0)
                except (UnknownTaxid, KeyError, ValueError):
                    host = 0
            key = (t, host, prok)
            if key not in cache:
                cache[key] = wide_reason(tax, km, ex, t, host, prok)
            why, grp, sp, ge = cache[key]
            asm = SL.assemblies(acc)
            if why is None and (r.get("genome_representation") or "full") != "full":
                why = "partial genome"
            if why is None and acc.split(".")[0][4:] in ex.pan_acc:
                why = "excluded by lineage"                     # a panel assembly
            if why is None and sealed.reasons(acc):
                why = "refused by the sealed check"
            in_corpus = [own[(grp, a)] for a in asm if (grp, a) in own]
            kind = why or ("already in the corpus" if in_corpus else "eligible, an environmental sample" if (r.get("environmental_sample") or "").lower() == "true" else "eligible")
            reasons[(cid, kind)] = reasons.get((cid, kind), 0) + 1
            if why is None:
                row = dict(id=in_corpus[0] if in_corpus else acc, group=grp, species_taxid=sp, genus=ge, selected="1" if in_corpus else "0",
                           out="" if in_corpus else ELIGIBLE[0], proteins_central=int(float(r.get("base_count") or 0) / 1000))
                rows.append(row)
                if grp in PROKARYOTES:                          # the host species of a plasmid anchor
                    pl = [own[("mobile.plasmid", a)] for a in asm if ("mobile.plasmid", a) in own]
                    rows.append(dict(row, id=pl[0] if pl else "plasmids_of_" + acc, group="mobile.plasmid", selected="1" if pl else "0", out="" if pl else ELIGIBLE[0]))
    kept = [r for r in cands if (r.get("group") or "").startswith("eukaryota.")]      # no eukaryote catalogue in the new snapshot
    anchors, counts = census(rows + kept, metas, s)
    return counts, {"%s | %s" % k: v for k, v in sorted(reasons.items())}, sorted({r["group"] for r in kept})


def build(st, cfg, run_dir):
    s = settings(cfg)
    cands, selected, prov = selection(s)
    head = corpus_header_sha(st.repo(cfg["corpus_spec"]))
    if head != prov["sha256"]["selected.tsv"]:
        raise SystemExit("rel_w13_census: the selected table of %s (%s) is not the one the corpus spec %s names (%s)" % (
            prov["run_id"], prov["sha256"]["selected.tsv"][:12], cfg["corpus_spec"], (head or "none")[:12]))
    src = W.source(s["source"])
    sealed = common.sealed_set(cfg)
    metas, left, sizes = W.open_genomes(src, s, sealed)
    from ci.taxonomy import Taxonomy
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    st.extra["sealed_check"] = common.sealed_check(cfg, [W.entry(src, m) for m in metas], splits={m["id"]: m["split"] for m in metas}, sealed=sealed, tax=tax)
    sel_ids = {r["id"] for r in selected}
    missing = [m["id"] for m in metas if m["id"] not in sel_ids]
    if missing:
        raise SystemExit("rel_w13_census: open genomes that are not rows of the selected table: %s" % missing[:5])
    reasons = {}
    for r in cands:
        k = (r.get("group") or "(no group)", reason_kind(r["out"]))
        reasons[k] = reasons.get(k, 0) + 1
    anchors, counts = census(cands, metas, s)
    wide = None
    if s["catalog_from"]:
        from ci.stages.rel import w13_catalog
        here = s["catalog_from"] == "this run"
        cats, clock, cprov = w13_catalog.read(run_dir if here else s["catalog_from"], same_run=here)
        extra = [st.repo(x) for x in (cfg.get("extra_exclusions") or [])]
        wcounts, wreasons, same = census_wide(cats, metas, [W.entry(src, m) for m in metas], cands, s, tax, sealed, extra)
        wide = dict(catalogue=dict(cprov, source=clock.get("source"), catalogues={k: dict(query=v["query"], records=v["records"], sha256=v["sha256"], fetched=v["fetched"])
                                                                              for k, v in clock["catalogues"].items()}, context=clock.get("context")),
                    counts=wcounts, reasons=wreasons, groups_from_the_selection=same,
                    proposal=dict(per_anchor=s["propose"], by_split={sp: {g: d["proposals"][str(s["propose"])] for g, d in by.items()} for sp, by in wcounts.items()}))
    lock = lock_totals(st.repo(prov["lock"])) if prov.get("lock") and os.path.exists(st.repo(prov["lock"])) else {}
    rep = dict(format="rel-w13-census", version=1, note=__doc__.strip().splitlines()[0], selection=prov, source=src["provenance"], corpus_spec=cfg["corpus_spec"],
               settings={k: s[k] for k in ("per_anchor", "propose", "max_genes_euk")}, split_sizes=sizes, left_out=len(left),
               candidates=dict(n=len(cands), by_group_and_reason={"%s | %s" % k: v for k, v in sorted(reasons.items())}),
               catalogues=lock, counts=counts, proposal=dict(per_anchor=s["propose"], by_split={sp: {g: d["proposals"][str(s["propose"])] for g, d in by.items()}
                                                                                                  for sp, by in counts.items()}),
               caveats=["The prokaryote catalogues of this selection are NCBI reference genomes only (about one genome per species), so they hold almost "
                        "no relatives of the same species for bacteria and archaea; the totals of the broader queries the lock records show what a wider "
                        "snapshot would hold, which needs a new catalogue snapshot (a metadata download; the user's decision, D6).",
                        "A relative goes with its anchor's split; relatives of held-out genomes are for the evaluation, not for training.",
                        "Proteins are the catalogue's central estimate, eukaryotes capped at %d genes as in the corpus." % s["max_genes_euk"],
                        "Counts only; nothing was downloaded."] + ([
                        "How the two counts differ. The first reads the selection's snapshots: NCBI RefSeq reference genomes for bacteria and archaea "
                        "(about one per species), its virus catalogues and the selection's own host records; every candidate there passed the selection's "
                        "quality rules too (protein-coding genes, a current assembly, the sequence report). The second (wide) reads a fresh ENA snapshot "
                        "of assembly metadata (stage rel_w13_catalog): every INSDC assembly at the levels of that snapshot (complete genome and chromosome "
                        "for prokaryotes, complete genome for viruses), reference or not, environmental samples (metagenome-assembled genomes) included and "
                        "counted apart; the same exclusions by lineage, the same host rule for viruses (the record's host, else the selection's host of the "
                        "species), but no quality rule beyond a full genome representation, a genus and a species. Proteins there are one per kilobase; "
                        "plasmid anchors count the assemblies of their host species (an upper bound); the eukaryote groups keep the selection's candidates "
                        "(the new snapshot has no eukaryote catalogue). No sequence was downloaded."] if wide else []))
    if wide:
        rep["wide"] = wide
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    page = os.path.join(folder, "report.md")
    with open(page + ".part", "w") as f:
        f.write(page_of(rep))
    os.replace(page + ".part", page)
    st.output("census_page", page)
    p = os.path.join(folder, "report.json")
    write_json(p, rep)
    st.output("census_report", p)
    st.extra.update(proposal=rep["proposal"])
    return rep


def page_of(rep):
    L = ["# Close relatives the catalogue snapshots hold (package W13, step 1)", "", rep["note"], "",
         "Selection run `%s` (selected table %s, the corpus %s); splits of run `%s`. Counts only; nothing was downloaded." % (
             rep["selection"]["run_id"], rep["selection"]["sha256"]["selected.tsv"][:12], rep["corpus_spec"], rep["source"]["run_id"]), ""]
    for split, by in rep["counts"].items():
        L += ["## Anchors of the %s split" % split, "",
              "| group | anchors | with a new relative of the species | with 3 | with 10 | new same-species genomes | with a new relative of the genus | new same-genus genomes | already in the corpus (species / genus) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for g, d in by.items():
            a, b = d["new_same_species"], d["new_same_genus"]
            L.append("| %s | %d | %d | %d | %d | %s | %d | %s | %d / %d |" % (g, d["anchors"], a["anchors_with_one"], a["anchors_with_three"], a["anchors_with_ten"],
                                                                          "{:,}".format(a["candidates"]), b["anchors_with_one"], "{:,}".format(b["candidates"]),
                                                                          d["in_corpus"]["same_species"], d["in_corpus"]["same_genus"]))
        L.append("")
    ks = [str(k) for k in rep["settings"]["per_anchor"]]
    for split, by in rep["counts"].items():
        L += ["## A first batch for the %s split: up to k relatives per anchor (the same species first), genomes / proteins" % split, "",
              "| group | " + " | ".join("k = %s" % k for k in ks) + " |", "|---|" + "---|" * len(ks)]
        for g, d in by.items():
            L.append("| %s | " % g + " | ".join("%s / %s" % ("{:,}".format(d["proposals"][k]["genomes"]), "{:,}".format(d["proposals"][k]["proteins"])) for k in ks) + " |")
        L.append("")
    if rep.get("catalogues"):
        L += ["## The catalogue snapshots and the broader queries the lock records", "", "| catalogue | records | broader queries |", "|---|---|---|"]
        for c, d in rep["catalogues"].items():
            L.append("| %s | %s | %s |" % (c, "{:,}".format(d["records"] or 0), "; ".join("%s: %s" % (q, "{:,}".format(t)) for q, t in d["totals"].items() if t is not None) or ""))
        L.append("")
    w = rep.get("wide")
    if w:
        L += ["## Side by side: the selection's snapshot and the wider snapshot (%s)" % ", ".join("%s %s records, %s" % (k, "{:,}".format(v["records"]), v["fetched"][:10])
                                                                                               for k, v in w["catalogue"]["catalogues"].items()), ""]
        for split in rep["counts"]:
            L += ["### %s split" % split, "",
                  "| group | anchors | with a same-species relative (selection / wide) | same-species genomes | same-genus genomes | first batch, k = %d |" % rep["settings"]["propose"],
                  "|---|---|---|---|---|---|"]
            for g, d in rep["counts"][split].items():
                e = (w["counts"].get(split) or {}).get(g)
                if not e:
                    continue
                k = str(rep["settings"]["propose"])
                L.append("| %s | %d | %d / %d | %s / %s | %s / %s | %s / %s |" % (
                    g + (" (the selection's snapshot both times)" if g in w.get("groups_from_the_selection", []) else ""), d["anchors"], d["new_same_species"]["anchors_with_one"], e["new_same_species"]["anchors_with_one"],
                    "{:,}".format(d["new_same_species"]["candidates"]), "{:,}".format(e["new_same_species"]["candidates"]),
                    "{:,}".format(d["new_same_genus"]["candidates"]), "{:,}".format(e["new_same_genus"]["candidates"]),
                    "{:,}".format(d["proposals"][k]["genomes"]), "{:,}".format(e["proposals"][k]["genomes"])))
            L.append("")
        L += ["Records of the wider snapshot by reason: " + "; ".join("%s %s" % (k, "{:,}".format(v)) for k, v in w["reasons"].items()), ""]
    L += ["## What these numbers are not", ""] + ["- " + x for x in rep["caveats"]] + [""]
    return "\n".join(L)


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "report.json")
    with open(p) as f:
        rep = json.load(f)
    s = settings(cfg)
    problems = []
    src = W.source(s["source"])
    open_ids = {m["id"] for m in src["index"]["genomes"] if m["split"] in OPEN}
    groups = {m["group"] for m in src["index"]["genomes"] if m["split"] in OPEN}
    for split in OPEN:
        by = (rep.get("counts") or {}).get(split) or {}
        if set(by) - {"all"} != groups:
            problems.append(dict(split=split, error="the groups counted are not the corpus's groups"))
        alls = by.get("all") or {}
        if alls.get("anchors") != sum(d["anchors"] for g, d in by.items() if g != "all"):
            problems.append(dict(split=split, error="the anchors of the groups do not add up"))
    if corpus_header_sha(st.repo(cfg["corpus_spec"])) != ((rep.get("selection") or {}).get("sha256") or {}).get("selected.tsv"):
        problems.append(dict(error="the selected table is not the corpus's"))
    with open(p) as f:
        text = f.read()
    test = {m["id"] for m in src["index"]["genomes"] if m["split"] not in OPEN}
    if any(t in text for t in test):
        problems.append(dict(error="a genome outside the open splits is named"))
    del open_ids
    require(GateResult("G4_rel_w13_census", "fail" if problems else "pass", dict(problems=problems[:50], proposal=rep.get("proposal"))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        rep = json.load(f)
    head = {sp: {g: dict(anchors=d["anchors"], with_species_relative=d["new_same_species"]["anchors_with_one"],
                         new_same_species=d["new_same_species"]["candidates"], new_same_genus=d["new_same_genus"]["candidates"])
                 for g, d in by.items()} for sp, by in rep["counts"].items()}
    return dict(json=dict(counts=head, proposal=rep["proposal"]), md=page_of(rep).splitlines()[2:80])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w13_census") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
