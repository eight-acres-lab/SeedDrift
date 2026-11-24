"""Stage rel_w13_xpick (package W13; pipeline rel_corpus): the supplement of eukaryotic viruses and of plants and algae,
by name, and the corpus spec that adds them (the user's decision of 2026-11-24, "都下", through the programme lead: every
eukaryotic virus and every plant or alga of the supply count that the corpus's split rule puts in a training or a
held-out unit; none of a test unit is downloaded or read). `python -m ci.stages.rel.w13_xpick RUN_DIR`. Metadata only,
and the ENA records of the viruses: the sequences are downloaded later by the corpus stages (fetch, through stage
rel_w13_build).

Candidates are those of the passed supply run (rel_w13_xpick.supply_from, stage rel_w13_supply, its snapshots checked
against its report), made again with its rules (w13_supply.virus_rows and plant_rows: the selection's host rule, the
exclusions by full lineage of gate G2 and the extra lists, the sealed check, not a corpus assembly; plants one
annotated assembly per species not in the corpus). A virus must also be in viruses.eukaryotic by its taxon, as stage
train_data groups it. The split of each candidate is stage train_data's own: its three_way() run on the source run's
corpus genomes (their ranks, groups and clusters from its index) and the candidates; every corpus genome must keep its
split, else the stage fails. Candidates in a test unit are counted, not listed; it is run again on the corpus genomes
and the kept candidates, which must all keep their splits.

Viruses: genes are called by pyrodigal as for every raw virus of the corpus (the annotation is not used; an
unannotated assembly's genes are estimated from its length for the counts only). Each one's GenBank (GCA) assembly is
looked up in its ENA record (the replicons; a server error or no connection is tried again after 10, 30 and 90 seconds)
and its source lists those sequences (ena_sequences), so stage fetch downloads exactly them; an assembly with no GCA
twin, no ENA record or no replicon is unavailable. Plants and algae: source kind annotated_cds from NCBI Datasets
(ncbi_datasets: the coding sequences and the sequence report of the assembly, no genome), at most max_genes_euk genes
(1,500, as every eukaryote of the corpus), the quality rules of stage corpus with require_start_stop. The unavailable
viruses are at most max_unavailable (default 5%) of those looked up, else the stage fails. The new ids pass the sealed
check and the exclusions by lineage.

Configuration (profile block rel_w13_xpick): supply_from, select_from, source, expect, name (the new corpus spec's
name), max_genes_euk (default 1500), threads (default 8), max_unavailable (default 0.05), require_start_stop (default
true: a coding sequence must start with ATG and end with a stop codon). Outputs under data/rel_w13/xpick/:
relatives.json (per new genome: id, group, split, taxid, accession, proteins, bases, host_taxid, annotated; the format
stage rel_w13_build reads), corpus.yaml, report.json (written last) and report.md. Gate G4_rel_w13_xpick.
"""
from __future__ import annotations

import json
import os
import re

import yaml

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from ci.stages.rel import w2_v0 as W
from ci.stages.rel import w13_pick as WP
from ci.stages.rel import w13_supply as WS

OUT = os.path.join("data", "rel_w13", "xpick")
KEYS = {"supply_from", "select_from", "source", "expect", "name", "max_genes_euk", "threads", "max_unavailable", "require_start_stop"}
OPEN = ("train", "heldout")
NCBI_INCLUDE = ["CDS_FASTA", "SEQUENCE_REPORT"]


def settings(cfg) -> dict:
    c = cfg.get("rel_w13_xpick") or {}
    unknown = set(c) - KEYS
    need = {"supply_from", "select_from", "source", "name"}
    if unknown or need - set(c):
        raise SystemExit("rel_w13_xpick: needs %s; unknown keys %s" % (sorted(need), sorted(unknown)))
    return dict(supply_from=str(c["supply_from"]), select_from=str(c["select_from"]), source=str(c["source"]),
                expect={str(a): int(b) for a, b in (c.get("expect") or {}).items()}, name=str(c["name"]),
                max_genes_euk=int(c.get("max_genes_euk", 1500)), threads=int(c.get("threads", 8)), max_unavailable=float(c.get("max_unavailable", 0.05)),
                require_start_stop=bool(c.get("require_start_stop", True)), max_left_out=20)


def supply_run(run):
    """(records {viruses, plants}, provenance) of a passed run of stage rel_w13_supply, its files checked."""
    from ci.stages import catalog as CAT
    rec = json.load(open(os.path.join(run, "record.json")))
    if rec.get("status") != "passed":
        raise SystemExit("rel_w13_xpick: the supply run %s is %s, not passed" % (rec.get("run_id"), rec.get("status")))
    rp = os.path.join(run, WS.OUT, "report.json")
    W._output(run, "rel_w13_supply", "supply_report", rp)
    rep = json.load(open(rp))
    for name, e in rep["files"].items():
        if sha256_file(os.path.join(run, e["path"]), use_sidecar=False) != e["sha256"]:
            raise SystemExit("rel_w13_xpick: the %s file of the supply run is not the one its report names" % name)
    with open(os.path.join(run, WS.OUT, "lock.yaml")) as f:
        lock = yaml.safe_load(f)
    recs = {k: CAT.read_jsonl(os.path.join(run, rep["files"][k + ".jsonl"]["path"])) for k in ("viruses", "plants")}
    return recs, dict(run_id=rec.get("run_id") or os.path.basename(run), report_sha256=sha256_file(rp, use_sidecar=False), source=rep["source"]["run_id"],
                      catalogues={k: dict(records=v["records"], sha256=v["sha256"], fetched=v.get("fetched")) for k, v in lock["catalogues"].items()})


def unfetched(run):
    """The ids of the sources stage fetch of the source run listed as unavailable (no record in the archive; its corpus
    stage left them out), from its fetch manifest (an output of its stage fetch); none when it has no fetch stage."""
    p = os.path.join(run, "data", "fetch", "manifest.json")
    if not os.path.exists(p):
        return set()
    W._output(run, "fetch", "fetch_manifest", p)
    with open(p) as f:
        return {m["id"] for m in (json.load(f).get("meta") or {}).get("unavailable") or []}


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def gca_of(r):
    """The GenBank (INSDC) twin of an assembly record: ENA holds GCA assemblies only."""
    for a in (r.get("accession") or "", r.get("paired_accession") or ""):
        if a.startswith("GCA_"):
            return a
    return ""


def refseq_of(r):
    for a in (r.get("accession") or "", r.get("paired_accession") or ""):
        if a.startswith("GCF_"):
            return a
    return ""


def candidates(x, recs, s):
    """[candidate] of the supply's rules (dict(id, kind, group, taxid, record, genes, annotated, host)) and the reasons
    of the ones left out."""
    tax, km = x["tax"], x["km"]
    vrows, vwhy = WS.virus_rows(recs["viruses"], tax, km, x["ex"], x["hosts"], x["corpus_asm"], x["sealed"], x["per_kb"])
    best, pwhy = WS.plant_rows(recs["plants"], tax, km, x["ex"], x["corpus_asm"], x["corpus_species"], x["sealed"])
    out, why = [], dict(viruses=dict(vwhy), plants=dict(pwhy))
    for v in vrows:
        try:
            by_taxon = km.group(tax, v["taxid"], "raw_virus")
        except Exception:  # noqa: BLE001
            by_taxon = ""
        if by_taxon != "viruses.eukaryotic":
            k = "in another group by its taxon (stage train_data groups a virus by its taxon)"
            why["viruses"][k] = why["viruses"].get(k, 0) + 1
            continue
        acc = v["record"]["accession"]
        out.append(dict(id="virus_eukaryotic_%s" % (gca_of(v["record"]) or acc), kind="virus", group="viruses.eukaryotic", taxid=v["taxid"], record=v["record"],
                        genes=v["genes"], annotated=v["annotated"], host=int(v["host"] or 0)))
    for sp, (_, r, _) in sorted(best.items()):
        out.append(dict(id="plants_algae_%s_%s" % (slug(tax.name(sp)), r["accession"]), kind="plant", group="eukaryota.plants_algae", taxid=int(r["taxid"]),
                        record=r, genes=int(r["protein_coding"]), annotated=True, host=0))
    ids = [c["id"] for c in out]
    if len(set(ids)) != len(ids):
        raise SystemExit("rel_w13_xpick: two candidates take one id: %s" % sorted({i for i in ids if ids.count(i) > 1})[:5])
    return out, why


def splits_of(index_genomes, cands, tax):
    """Stage train_data's three_way() over the corpus genomes of the source index and the candidates: (the corpus genomes
    whose split it changes (a count), {candidate id: split})."""
    from ci.stages import train_data as TD
    old = [g for g in index_genomes if g.get("source", "corpus") == "corpus"]
    lin = {g["id"]: dict(zip(TD.RANKS, g["ranks"])) for g in old}
    groups = {g["id"]: g["group"] for g in old}
    cluster = {g["id"]: g["cluster"] for g in old if g.get("cluster")}
    for c in cands:
        lin[c["id"]] = {r: tax.at_rank(c["taxid"], r) for r in TD.RANKS}
        groups[c["id"]] = c["group"]
    sp = TD.three_way([g["id"] for g in old] + [c["id"] for c in cands], groups, lin, cluster)
    moved = sum(1 for g in old if sp[g["id"]] != g["split"])
    return moved, {c["id"]: sp[c["id"]] for c in cands}


def ena_lookup(accs, threads, get=None, waits=WP.RETRY_S):
    """{GCA accession: [(sequence, type)], [] (a record without replicons), or None (no record, or every try failed)}
    and the accessions ENA answered it has no record of."""
    from concurrent.futures import ThreadPoolExecutor
    get = get or WP.replicons
    norec = set()

    def one(a):
        def ask():
            try:
                return list(get(a))
            except Exception as err:  # noqa: BLE001
                if type(err).__name__ == "NoRecord":             # panel.NoRecord: a definite answer, not asked again
                    norec.add(a)
                    raise WP.Refused(str(err))
                raise
        return a, WP.retrying(ask, waits)
    with ThreadPoolExecutor(max(1, threads)) as pool:
        got = dict(pool.map(one, sorted(set(accs))))
    return got, norec


def source_of(c, reps, s):
    """The corpus-spec source of a kept candidate."""
    r = c["record"]
    if c["kind"] == "virus":
        return dict(id=c["id"], source_kind="raw_virus", taxid=c["taxid"], table=11, accession=gca_of(r), paired_accession=refseq_of(r),
                    ena_sequences=[a for a, _ in reps])
    return dict(id=c["id"], source_kind="annotated_cds", taxid=c["taxid"], table=1, accession=r["accession"], paired_accession=r.get("paired_accession") or "",
                ncbi_datasets=dict(accession=r["accession"], include=list(NCBI_INCLUDE)), max_genes=s["max_genes_euk"],
                require_start_stop=s["require_start_stop"])


def _add(d, key, c, genes=None):
    x = d.setdefault(key, dict(genomes=0, genes=0, bases=0))
    x["genomes"] += 1
    x["genes"] += int(c["genes"] if genes is None else genes)
    x["bases"] += int(c["record"].get("length") or 0)


def build(st, cfg, run_dir):
    from seeddrift.train.rel import sealed as SL
    s = settings(cfg)
    recs, sprov = supply_run(s["supply_from"])
    x = WS.inputs(st, cfg, s, name="rel_w13_xpick")
    src, tax = x["src"], x["tax"]
    if sprov["source"] != src["provenance"]["run_id"]:
        raise SystemExit("rel_w13_xpick: the supply run counted against %s, not the source %s" % (sprov["source"], src["provenance"]["run_id"]))
    spec_path = st.repo(cfg["corpus_spec"])
    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    index_ids = {g["id"] for g in src["index"]["genomes"] if g.get("source", "corpus") == "corpus"}
    fetch_gone = unfetched(s["source"])
    spec_ids = {x_["id"] for x_ in spec["sources"]}
    if not index_ids <= spec_ids or spec_ids - index_ids != fetch_gone & spec_ids:
        raise SystemExit("rel_w13_xpick: the corpus spec %s is not the source run's corpus (%d sources, %d corpus genomes, %d sources its fetch could not get)" % (
            cfg["corpus_spec"], len(spec_ids), len(index_ids), len(fetch_gone)))
    dropped = sorted(spec_ids - index_ids)                           # sources the source run's fetch could not get: not in the new spec
    spec = dict(spec, sources=[x_ for x_ in spec["sources"] if x_["id"] not in fetch_gone])
    old_ids = {x_["id"] for x_ in spec["sources"]}
    cands, why = candidates(x, recs, s)
    moved, split = splits_of(src["index"]["genomes"], cands, tax)
    if moved:
        raise SystemExit("rel_w13_xpick: with the candidates, stage train_data's split moves %d corpus genomes" % moved)
    kept = [c for c in cands if split[c["id"]] in OPEN]
    test = [c for c in cands if split[c["id"]] not in OPEN]
    moved2, split2 = splits_of(src["index"]["genomes"], kept, tax)
    if moved2 or any(split2[c["id"]] != split[c["id"]] for c in kept):
        raise SystemExit("rel_w13_xpick: without the candidates of the test units the split changes (%d corpus genomes)" % moved2)
    vk = [c for c in kept if c["kind"] == "virus"]
    gca = {c["id"]: gca_of(c["record"]) for c in vk}
    seen, norec = ena_lookup([a for a in gca.values() if a], s["threads"])
    unavailable, reps = {}, {}
    for c in vk:
        a = gca[c["id"]]
        if not a:
            unavailable[c["id"]] = "no GenBank (GCA) assembly: ENA holds INSDC assemblies only"
        elif seen.get(a) is None:
            unavailable[c["id"]] = "no ENA record" if a in norec else "the ENA record service did not answer through every try"
        elif not seen[a]:
            unavailable[c["id"]] = "the ENA record lists no replicon"
        else:
            reps[c["id"]] = seen[a]
    picked = [c for c in kept if c["kind"] == "plant" or c["id"] in reps]
    sources = [source_of(c, reps.get(c["id"], ()), s) for c in picked]
    clash = [y["id"] for y in sources if y["id"] in old_ids]
    if clash:
        raise SystemExit("rel_w13_xpick: new sources whose id is a corpus source's: %s" % clash[:5])
    old_asm = {a for y in spec["sources"] for a in SL.assemblies(y["id"]) | SL.assemblies(y.get("accession", "")) | SL.assemblies(y.get("paired_accession", ""))}
    clash = [y["id"] for y in sources if (SL.assemblies(y["accession"]) | SL.assemblies(y.get("paired_accession", ""))) & old_asm]
    if clash:
        raise SystemExit("rel_w13_xpick: new genomes whose assembly is a corpus source's: %s" % clash[:5])
    ids = [y["id"] for y in sources]
    by = {c["id"]: c for c in picked}
    try:
        ev_ids = SL.check_ids(ids, x["sealed"], splits={i: split[i] for i in ids})
    except SL.SealedError as err:
        raise SystemExit("rel_w13_xpick: %s" % err)
    lin = common.check_lineage([dict(id=y["id"], taxid=y["taxid"], accession=y["accession"], paired_accession=y.get("paired_accession", ""),
                                     **({"host_taxid": by[y["id"]]["host"]} if by[y["id"]]["host"] else {})) for y in sources], cfg, tax)
    rel = [dict(id=c["id"], group=c["group"], split=split[c["id"]], anchors=[], relation="", taxid=c["taxid"], accession=gca_of(c["record"]) if c["kind"] == "virus" else c["record"]["accession"],
                proteins=int(c["genes"]), bases=int(c["record"].get("length") or 0), host_taxid=c["host"], annotated=bool(c["annotated"])) for c in picked]
    rel.sort(key=lambda r: r["id"])
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    new_spec = dict(name=s["name"], tier=spec.get("tier"), groups=spec["groups"], sources=list(spec["sources"]) + sources)
    with open(spec_path) as f:
        head = [ln for ln in f.read(6000).splitlines() if ln.startswith("#")]
    cp = os.path.join(folder, "corpus.yaml")
    with open(cp, "w") as f:
        f.write("# Corpus %s: corpus %s (%s; its header follows; less %d sources its build could not fetch) and the supplement of %d eukaryotic\n"
                "# viruses and %d plants and algae of the training and held-out units, picked by stage rel_w13_xpick of run %s (the user's decision\n"
                "# of 2026-11-24). Edit the stage, not this file.\n"
                "# Viral genes are called by pyrodigal (metagenome mode), as for every virus of the corpus: spliced genes and polyproteins are not called right.\n"
                % (s["name"], spec["name"], cfg["corpus_spec"], len(dropped), sum(1 for c in picked if c["kind"] == "virus"), sum(1 for c in picked if c["kind"] == "plant"),
                   os.path.basename(run_dir)))
        f.write("\n".join(head) + "\n")
        f.write(yaml.safe_dump(new_spec, sort_keys=False, width=160))
    st.output("xpick_corpus_spec", cp)
    rp = os.path.join(folder, "relatives.json")
    acc_of = {c["id"]: c["record"]["accession"] for c in vk}
    write_json(rp, dict(relatives=rel, unavailable=[dict(id=i, accession=gca[i] or acc_of[i], reason=w) for i, w in sorted(unavailable.items())]))
    st.output("xpick_relatives", rp)
    counts, chosen = dict(viruses={}, plants={}), {c["id"] for c in picked}
    for c in cands:                                                  # every candidate: the open splits by id, the test part as a number
        part = "viruses" if c["kind"] == "virus" else "plants"
        sp = split[c["id"]]
        key = "%s, %s" % (sp, "annotated" if c["annotated"] else "not annotated") if part == "viruses" else sp
        _add(counts[part].setdefault("candidates", {}), key, c)
        if c["id"] in chosen:
            _add(counts[part].setdefault("picked", {}), key, c)
            if part == "plants":
                _add(counts[part].setdefault("picked_capped", {}), key, c, genes=min(int(c["genes"]), s["max_genes_euk"]))
    n_look = len(vk)
    rep = dict(format="rel-w13-xpick", version=1, note=__doc__.strip().splitlines()[0], name=s["name"], corpus_spec=cfg["corpus_spec"], supply=sprov,
               source=src["provenance"], host_snapshots=x["host_snap"], settings={k: s[k] for k in ("max_genes_euk", "max_unavailable", "require_start_stop")},
               candidates=len(cands), test_unit_candidates=len(test), kept=len(kept), relatives=len(rel), counts=counts, left_out=why,
               lookups=dict(viruses=n_look, unavailable=len(unavailable), unavailable_share=round(len(unavailable) / max(1, n_look), 4), max_unavailable=s["max_unavailable"],
                            by_reason=_tally(unavailable.values())),
               split_check=dict(corpus_genomes=len(index_ids), moved=0, moved_without_test_units=0),
               left_out_of_the_spec=dict(sources=dropped, reason="the source run's stage fetch could not get them (its corpus has no such genome)"),
               download_estimate=dict(virus_bases=sum(r["bases"] for r in rel if r["group"] == "viruses.eukaryotic"),
                                      plant_genes=sum(r["proteins"] for r in rel if r["group"] == "eukaryota.plants_algae")),
               sealed_ids=ev_ids, lineage=lin, files=dict(corpus=dict(path=os.path.relpath(cp, run_dir), sha256=sha256_file(cp, use_sidecar=False)),
                                                          relatives=dict(path=os.path.relpath(rp, run_dir), sha256=sha256_file(rp, use_sidecar=False))))
    st.extra["sealed_check"] = dict(st.extra.get("sealed_check") or {}, new_genomes=dict(ids=ev_ids, lineage=lin))
    page = os.path.join(folder, "report.md")
    with open(page, "w") as f:
        f.write(page_of(rep))
    st.output("xpick_page", page)
    p = os.path.join(folder, "report.json")
    write_json(p, rep)
    st.output("xpick_report", p)
    st.extra.update(relatives=len(rel), test_unit_candidates=len(test), unavailable=len(unavailable))
    return rep


def _tally(xs):
    out = {}
    for v in xs:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def page_of(rep):
    fmt = lambda v: "{:,}".format(v)                                    # noqa: E731
    L = ["# The supplement of eukaryotic viruses and plants and algae (package W13)", "", rep["note"], "",
         "Supply run `%s`, source run `%s`; corpus spec `%s`, %s new genomes (%s candidates, %s of them in a test unit: not listed, not downloaded)." % (
             rep["supply"]["run_id"], rep["source"]["run_id"], rep["name"], fmt(rep["relatives"]), fmt(rep["candidates"]), fmt(rep["test_unit_candidates"])), ""]
    for part in ("viruses", "plants"):
        L += ["## %s" % ("Eukaryotic viruses (genes of an unannotated genome estimated from its length)" if part == "viruses" else "Plants and algae (genes of the annotation; capped: at most %d a genome)" % rep["settings"]["max_genes_euk"]), "",
              "| part | candidates | genes | picked | genes | bases |" + (" genes capped |" if part == "plants" else ""), "|---|---|---|---|---|---|" + ("---|" if part == "plants" else "")]
        c = rep["counts"][part]
        for k in sorted(c.get("candidates") or {}):
            a, b = c["candidates"][k], (c.get("picked") or {}).get(k) or dict(genomes=0, genes=0, bases=0)
            row = "| %s | %s | %s | %s | %s | %s |" % (k, fmt(a["genomes"]), fmt(a["genes"]), fmt(b["genomes"]), fmt(b["genes"]), fmt(b["bases"]))
            if part == "plants":
                row += " %s |" % fmt(((c.get("picked_capped") or {}).get(k) or {}).get("genes", 0))
            L.append(row)
        L += ["", "Left out by the rules: " + ("; ".join("%s %s" % (k, fmt(v)) for k, v in sorted(rep["left_out"][part].items())) or "none") + ".", ""]
    lk = rep["lookups"]
    L += ["Viruses looked up at ENA: %s; unavailable %s (%.2f%%; the stage fails above %.0f%%): %s." % (
        fmt(lk["viruses"]), fmt(lk["unavailable"]), 100 * lk["unavailable_share"], 100 * lk["max_unavailable"],
        "; ".join("%s %d" % kv for kv in lk["by_reason"].items()) or "none"),
          "Split: stage train_data's three_way() with the candidates keeps the split of all %s corpus genomes, with and without the candidates of the test units." % fmt(rep["split_check"]["corpus_genomes"]),
          "Sources of the corpus spec its build could not fetch, not in the new spec: %s." % (", ".join(rep["left_out_of_the_spec"]["sources"]) or "none"),
          "Coding sequences of plants and algae must start with ATG and end with a stop codon: %s." % ("yes" if rep["settings"]["require_start_stop"] else "no (the rule of the Ensembl sources)"), ""]
    return "\n".join(L)


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "report.json")
    with open(p) as f:
        rep = json.load(f)
    problems = []
    for name, e in (rep.get("files") or {}).items():
        full = os.path.join(run_dir, e["path"])
        if not os.path.exists(full) or sha256_file(full, use_sidecar=False) != e["sha256"]:
            problems.append(dict(file=name, error="missing or not the file the report names"))
    with open(os.path.join(run_dir, OUT, "relatives.json")) as f:
        rel = json.load(f)["relatives"]
    if len(rel) != rep.get("relatives") or len({r["id"] for r in rel}) != len(rel):
        problems.append(dict(error="new genomes counted twice or another count"))
    if any(r["split"] not in OPEN for r in rel):
        problems.append(dict(error="a new genome outside the train and held-out splits"))
    lk = rep.get("lookups") or {}
    if lk.get("unavailable_share", 0) > lk.get("max_unavailable", 0.05):
        problems.append(dict(error="%s of %s viruses unavailable (more than %s)" % (lk.get("unavailable"), lk.get("viruses"), lk.get("max_unavailable"))))
    with open(os.path.join(run_dir, OUT, "corpus.yaml")) as f:
        spec = yaml.safe_load(f)
    ids = [y["id"] for y in spec["sources"]]
    if len(set(ids)) != len(ids) or not {r["id"] for r in rel} <= set(ids):
        problems.append(dict(error="the corpus spec repeats a source or lacks a new genome"))
    if not (rep.get("sealed_ids") or {}) and rel:
        problems.append(dict(error="no sealed check of the new genomes"))
    require(GateResult("G4_rel_w13_xpick", "fail" if problems else "pass", dict(problems=problems[:50], relatives=rep.get("relatives"))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        rep = json.load(f)
    return dict(json=dict(relatives=rep["relatives"], counts=rep["counts"], lookups=rep["lookups"]), md=page_of(rep).splitlines()[2:60])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w13_xpick") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
