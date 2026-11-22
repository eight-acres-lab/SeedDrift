"""Stage rel_w13_pick (package W13; pipeline rel_corpus): the first batch of close relatives, by name, and the corpus
spec that adds them (task breakdown W13 step 2; the user's decision of 2026-11-22, "按 k = 3 下载"). `python -m
ci.stages.rel.w13_pick RUN_DIR`. Metadata only: the sequences are downloaded later by the corpus stages (fetch).

The relatives are those of the wide count of stage rel_w13_census (ci/stages/rel/w13_census.py, census_wide), made
again here from the same inputs (the selection run, the source run's splits, the catalogue snapshot of stage
rel_w13_catalog, the selection's host snapshots): per corpus genome of the train and held-out splits up to k relatives,
the same species first, then the same genus, in the census's order; a relative of several anchors counts once. Every
pick from the snapshot is looked up in its ENA assembly record (the replicon list stage fetch reads): a prokaryote must
list a chromosome, a virus a replicon, and a relative of a plasmid anchor must list a plasmid, whose plasmids (not the
host assembly) become the relative; ENA must serve the FASTA URL stage fetch will request (some records are listed but
their sequence answers HTTP 500); a pick that fails is replaced by the next candidate of its anchor (at most
`max_lookups` per anchor). Eukaryote relatives come from the selection's candidates, with the Ensembl fields of the
selection's snapshots. A relative goes with its anchor's split; it carries no near-relative cluster, so stage
train_data puts it in its anchor's unit by genus (or species) and the units of the corpus genomes do not change.

Checks: the counts of every group equal the census run's (rel_w13_pick.census_from) where no lookup replaced a pick;
the new ids pass the sealed check and the exclusions by lineage (gate G2's rules); no new id or assembly is a corpus
genome's. Outputs under data/rel_w13/pick/: relatives.json (per relative: id, group, split, anchors, relation,
accession, taxid), corpus.yaml (the corpus spec rel_w13_pick.name: the sources of the corpus spec and the relatives),
report.json (written last) and report.md. Gate G4_rel_w13_pick.

Configuration (profile block rel_w13_pick): select_from, source, expect, catalog_from, census_from (the passed census
run whose counts are checked), k (default 3), name (the new corpus spec's name), max_genes_euk (default 1500),
max_lookups (default 40), threads (default 8).
"""
from __future__ import annotations

import json
import os
import time

import yaml

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from ci.stages.rel import w2_v0 as W
from ci.stages.rel import w13_catalog, w13_census as WC

OUT = os.path.join("data", "rel_w13", "pick")
KEYS = {"select_from", "source", "expect", "catalog_from", "census_from", "k", "name", "max_genes_euk", "max_lookups", "threads"}
NEEDS = {"bacteria": "Chromosome", "archaea": "Chromosome", "mobile.plasmid": "Plasmid"}       # viruses: any replicon


def settings(cfg) -> dict:
    c = cfg.get("rel_w13_pick") or {}
    unknown = set(c) - KEYS
    need = {"select_from", "source", "catalog_from", "census_from", "name"}
    if unknown or need - set(c):
        raise SystemExit("rel_w13_pick: needs %s; unknown keys %s" % (sorted(need), sorted(unknown)))
    k = int(c.get("k", 3))
    return dict(select_from=str(c["select_from"]), source=str(c["source"]), expect={str(a): int(b) for a, b in (c.get("expect") or {}).items()},
                catalog_from=str(c["catalog_from"]), census_from=str(c["census_from"]), k=k, per_anchor=[k], propose=k, name=str(c["name"]),
                max_genes_euk=int(c.get("max_genes_euk", 1500)), max_lookups=int(c.get("max_lookups", 40)), threads=int(c.get("threads", 8)), max_left_out=20)


def replicons(accession):
    """[(sequence accession, type)] of an ENA assembly record, as stage fetch reads it."""
    from ci.stages import fetch                                 # noqa: F401 - puts the panel module on the path
    from panel import replicons as rep
    return rep(accession)


def lookup(accs, threads, get=None):
    """{accession: [(sequence, type)] or None (no record, or the lookup failed after retries)}."""
    from concurrent.futures import ThreadPoolExecutor
    get = get or replicons

    def one(a):
        for k in range(3):
            try:
                return a, list(get(a))
            except Exception:  # noqa: BLE001 - a network failure: tried again, then counted as no record
                time.sleep(2 * (k + 1))
        return a, None
    with ThreadPoolExecutor(max(1, threads)) as pool:
        return dict(pool.map(one, sorted(set(accs))))


def ena_accession(row):
    v = (row.get("version") or "").strip()
    return "%s.%s" % (row["accession"], v) if v else row["accession"]


def need_of(row):
    return NEEDS.get(row["group"]) or "any"


def wanted(reps, need):
    """The replicons stage fetch would download for a source that needs `need` (its replicon_types)."""
    return [a for a, t in reps or [] if need == "any" or t == need]


def passes(row, reps):
    """Whether a pick's replicon list is what its group needs."""
    return bool(wanted(reps, need_of(row)))


def served(accs, tries=2):
    """Whether ENA serves the FASTA of these sequences: the URL stage fetch requests answers 200 and starts a FASTA (or
    gzip) stream (only its first bytes are read). Some records exist but are never served (HTTP 500)."""
    import urllib.request
    from ci.stages import fetch
    url = fetch.ENA_FASTA % ",".join(accs)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=300) as r:
                head = r.read(64)
                return r.status == 200 and (head[:2] == b"\x1f\x8b" or head[:1] == b">")
        except Exception:  # noqa: BLE001 - a server error or a failed connection: tried again, then not served
            time.sleep(3 * (k + 1))
    return False


def probe_all(pairs, threads, serve=None):
    """{(accession, need): served} of (accession, need, replicons) triples."""
    from concurrent.futures import ThreadPoolExecutor
    serve = serve or served

    def one(x):
        acc, need, reps = x
        return (acc, need), serve(wanted(reps, need))
    with ThreadPoolExecutor(max(1, threads)) as pool:
        return dict(pool.map(one, sorted(pairs, key=lambda x: (x[0], x[1]))))


def needs_lookup(row):
    return row.get("catalogue") in ("bacteria", "archaea", "viruses")          # a record of the ENA snapshot


def choose(anchors, k, max_lookups, threads, get=None, serve=None):
    """Per anchor of the open splits its picks [(row, relation)] and the candidates it examined, the snapshot's picks
    looked up in their ENA records (the replicons the group needs) and their FASTA probed at ENA, each replaced by the
    next candidate when it fails; and the lookups {accession: replicons} and the probes {(accession, need): served}."""
    lists, picks, pos, by_id = {}, {}, {}, {}
    for a in anchors:
        if a["split"] in WC.OPEN:
            lists[a["id"]] = [(r, "same species") for r in a["new_same_species"]] + [(r, "same genus") for r in a["new_same_genus"]]
            picks[a["id"]], pos[a["id"]], by_id[a["id"]] = [], 0, a
    seen, ok = {}, {}
    while True:
        for aid, lst in lists.items():                           # take what the lookups so far decide, in order
            while len(picks[aid]) < k and pos[aid] < min(len(lst), max_lookups):
                r, rel = lst[pos[aid]]
                if needs_lookup(r):
                    acc = ena_accession(r)
                    if acc not in seen or (passes(r, seen[acc]) and (acc, need_of(r)) not in ok):
                        break
                    pos[aid] += 1
                    if passes(r, seen[acc]) and ok[(acc, need_of(r))]:
                        picks[aid].append((r, rel))
                else:
                    pos[aid] += 1
                    picks[aid].append((r, rel))
        want, probe = set(), set()
        for aid, lst in lists.items():                           # the next candidates of anchors still short
            short = k - len(picks[aid])
            for r, _ in lst[pos[aid]:max_lookups]:
                if short <= 0:
                    break
                if needs_lookup(r):
                    acc = ena_accession(r)
                    if acc not in seen:
                        want.add(acc)
                    elif passes(r, seen[acc]) and (acc, need_of(r)) not in ok:
                        probe.add((acc, need_of(r), tuple(seen[acc])))
                short -= 1
        if not want and not probe:
            break
        if want:
            seen.update(lookup(want, threads, get))
        if probe:
            ok.update(probe_all(probe, threads, serve))
    return {aid: dict(anchor=by_id[aid], picks=p, looked=pos[aid]) for aid, p in picks.items()}, seen, ok


def ensembl_fields(cands_by_id, snaps):
    """{candidate id: ensembl dict} of the eukaryote candidates, from the selection's Ensembl snapshots: the record of the
    candidate's catalogue whose species name ends its id (the selection names a eukaryote <group>_<species name>)."""
    out = {}
    for cid, r in cands_by_id.items():
        if not (r.get("group") or "").startswith("eukaryota."):
            continue
        recs = [x for x in snaps.get(r.get("catalogue")) or [] if cid.endswith("_" + x["name"])]
        if not recs:
            continue
        rec = max(recs, key=lambda x: len(x["name"]))
        e = dict(site=rec["site"], division=rec["site_division"], species=rec["name"], release=int(rec["release"]))
        if "_collection_" in (rec.get("core_db") or ""):
            e["collection"] = rec["core_db"].split("_core_")[0]
        out[cid] = e
    return out


def source_of(row, relation, ens, max_genes_euk):
    """The corpus-spec source of a relative (the fields of `python -m ci emit-corpus`) and its id."""
    g = row["group"]
    if g.startswith("eukaryota."):
        e = ens.get(row["id"])
        if e is None:
            raise SystemExit("rel_w13_pick: no Ensembl record of the selection for %s" % row["id"])
        return dict(id=row["id"], source_kind="annotated_cds", taxid=int(row["taxid"]), ensembl=e, max_genes=int(max_genes_euk), relation=relation)
    acc = ena_accession(row)
    if g in WC.PROKARYOTES:
        gid, kind, types = "%s_%s" % (g, acc), "raw_prokaryote", ["Chromosome"]
    elif g == "mobile.plasmid":
        gid, kind, types = "plasmid_%s" % acc, "plasmid", ["Plasmid"]
    elif g.startswith("viruses."):
        gid, kind, types = "%s_%s" % (g.replace("viruses.", "virus_"), acc), "raw_virus", None
    else:
        raise SystemExit("rel_w13_pick: no source kind for group %s" % g)
    src = dict(id=gid, source_kind=kind, taxid=int(row["taxid"]), table=11, accession=acc, paired_accession="", ena_assembly=acc)
    if types:
        src["replicon_types"] = types
    src["relation"] = relation
    return src


def build(st, cfg, run_dir):
    from ci.taxonomy import Taxonomy
    from seeddrift.train.rel import sealed as SL
    s = settings(cfg)
    cands, selected, prov = WC.selection(s)
    spec_path = st.repo(cfg["corpus_spec"])
    if WC.corpus_header_sha(spec_path) != prov["sha256"]["selected.tsv"]:
        raise SystemExit("rel_w13_pick: the corpus spec %s is not the one emitted from %s" % (cfg["corpus_spec"], prov["run_id"]))
    src = W.source(s["source"])
    sealed = common.sealed_set(cfg)
    metas, left, sizes = W.open_genomes(src, s, sealed)
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    entries = [W.entry(src, m) for m in metas]
    st.extra["sealed_check"] = common.sealed_check(cfg, entries, splits={m["id"]: m["split"] for m in metas}, sealed=sealed, tax=tax)
    own = WC.corpus_genomes(metas, entries)
    cats, clock, cprov = w13_catalog.read(s["catalog_from"])
    host_recs, host_snap = WC.host_records(s["select_from"])
    extra = [st.repo(x) for x in (cfg.get("extra_exclusions") or [])]
    counts, _, _, anchors = WC.census_wide(cats, metas, own, cands, host_recs, s, tax, sealed, extra)
    chosen, seen, ok = choose(anchors, s["k"], s["max_lookups"], s["threads"])
    ens_snaps, ens_used = WC.selection_snapshots(s["select_from"], "ensembl_species")
    wanted = {r["id"] for c in chosen.values() for r, _ in c["picks"] if r["group"].startswith("eukaryota.")}
    ens = ensembl_fields({r["id"]: r for r in cands if r["id"] in wanted}, ens_snaps)
    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    old_ids = {x["id"] for x in spec["sources"]}
    rel, sources = {}, []
    for aid in sorted(chosen):
        a = chosen[aid]["anchor"]
        for r, relation in chosen[aid]["picks"]:
            srcd = source_of(r, relation, ens, s["max_genes_euk"])
            gid = srcd["id"]
            if gid in old_ids:
                raise SystemExit("rel_w13_pick: %s is a source of the corpus already" % gid)
            if gid in rel:
                if rel[gid]["split"] != a["split"]:
                    raise SystemExit("rel_w13_pick: %s is a relative of anchors in two splits" % gid)
                rel[gid]["anchors"].append(aid)
                if relation == "same species":
                    rel[gid]["relation"] = relation
                continue
            rel[gid] = dict(id=gid, group=r["group"], split=a["split"], anchors=[aid], relation=relation, taxid=int(r["taxid"]),
                            accession=srcd.get("accession", ""), proteins=int(r.get("proteins_central") or 0), bases=int(r.get("bases") or 0),
                            host_taxid=int(r.get("host_taxid") or 0))
            sources.append(srcd)
    for x in sources:                                            # a relative of a species and of a genus is a relative of the species
        x["relation"] = rel[x["id"]]["relation"]
    ids = sorted(rel)
    try:
        ev_ids = SL.check_ids(ids, sealed, splits={i: rel[i]["split"] for i in ids})
    except SL.SealedError as err:
        raise SystemExit("rel_w13_pick: %s" % err)
    lin = common.check_lineage([dict(id=i, taxid=rel[i]["taxid"], accession=rel[i]["accession"], paired_accession="",
                                     **({"host_taxid": rel[i]["host_taxid"]} if rel[i]["host_taxid"] else {})) for i in ids], cfg, tax)
    part = lambda kind: "plasmids" if kind == "plasmid" else "genome"          # noqa: E731 - a plasmid relative may be a corpus genome's plasmids
    old = {(part(x["source_kind"]), a) for x in spec["sources"]
           for a in SL.assemblies(x["id"]) | SL.assemblies(x.get("accession", "")) | SL.assemblies(x.get("paired_accession", ""))}
    kind_of = {x["id"]: x["source_kind"] for x in sources}
    clash = [i for i in ids if any((part(kind_of[i]), a) in old for a in SL.assemblies(i))]
    if clash:
        raise SystemExit("rel_w13_pick: relatives whose assembly is a source of the corpus: %s" % clash[:5])
    want = {sp: {g: d["proposals"][str(s["k"])]["genomes"] for g, d in by.items()} for sp, by in counts.items()}
    with open(os.path.join(s["census_from"], WC.OUT, "report.json")) as f:
        crep = json.load(f)
    reported = {sp: {g: d["proposals"][str(s["k"])]["genomes"] for g, d in by.items()} for sp, by in (crep.get("wide") or {}).get("counts", {}).items()}
    got = {sp: {} for sp in WC.OPEN}
    for x in rel.values():
        got[x["split"]][x["group"]] = got[x["split"]].get(x["group"], 0) + 1
        got[x["split"]]["all"] = got[x["split"]].get("all", 0) + 1
    replaced = {aid: c["looked"] - len(c["picks"]) for aid, c in chosen.items() if c["looked"] > len(c["picks"])}
    groups = sorted({g for by in want.values() for g in by})
    table = {sp: {g: dict(census=reported.get(sp, {}).get(g), recomputed=want[sp].get(g), picked=got[sp].get(g, 0),
                          anchors_with_a_replacement=sum(1 for aid in replaced if chosen[aid]["anchor"]["split"] == sp and (g == "all" or chosen[aid]["anchor"]["group"] == g)))
                  for g in groups} for sp in WC.OPEN}
    if want != reported:
        raise SystemExit("rel_w13_pick: the census made again differs from run %s: %s" % (os.path.basename(s["census_from"]), json.dumps(dict(again=want, reported=reported))[:600]))
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    new_spec = dict(name=s["name"], tier=spec.get("tier"), groups=spec["groups"], sources=list(spec["sources"]) + sources)
    with open(spec_path) as f:
        head = [x for x in f.read(4000).splitlines() if x.startswith("#")]
    cp = os.path.join(folder, "corpus.yaml")
    with open(cp, "w") as f:
        f.write("# Corpus %s: corpus %s (%s; its header follows) and %d close relatives of its train and held-out genomes, picked by stage\n"
                "# rel_w13_pick of run %s (k = %d per genome, the same species first; the user's decision of 2026-11-22). Edit the stage, not this file.\n"
                % (s["name"], spec["name"], cfg["corpus_spec"], len(sources), os.path.basename(run_dir), s["k"]))
        f.write("\n".join(head) + "\n")
        f.write(yaml.safe_dump(new_spec, sort_keys=False, width=160))
    st.output("pick_corpus_spec", cp)
    rp = os.path.join(folder, "relatives.json")
    write_json(rp, dict(relatives=[rel[i] for i in ids]))
    st.output("pick_relatives", rp)
    n_look = len(seen)
    rep = dict(format="rel-w13-pick", version=1, note=__doc__.strip().splitlines()[0], k=s["k"], name=s["name"], corpus_spec=cfg["corpus_spec"],
               selection=prov, source=src["provenance"], catalogue=cprov, host_snapshots=host_snap, ensembl_snapshots=ens_used, census=os.path.basename(s["census_from"]),
               relatives=len(ids), by_split=table, lookups=dict(records=n_look, without_record=sum(1 for v in seen.values() if v is None),
                                                               without_any_replicon=sum(1 for v in seen.values() if v is not None and not v),
                                                               fasta_probed=len(ok), fasta_not_served=sorted(a for (a, _), v in ok.items() if not v)),
               anchors_with_a_replacement=len(replaced), bases=sum(x["bases"] for x in rel.values()), proteins_estimate=sum(min(x["proteins"], s["max_genes_euk"]) if x["group"].startswith("eukaryota.") else x["proteins"]
                                                        for x in rel.values() if x["group"] != "mobile.plasmid"),
               sealed_ids=ev_ids, lineage=lin, files=dict(corpus=dict(path=os.path.relpath(cp, run_dir), sha256=sha256_file(cp, use_sidecar=False)),
                                                          relatives=dict(path=os.path.relpath(rp, run_dir), sha256=sha256_file(rp, use_sidecar=False))))
    page = os.path.join(folder, "report.md")
    with open(page, "w") as f:
        f.write(page_of(rep))
    st.output("pick_page", page)
    p = os.path.join(folder, "report.json")
    write_json(p, rep)
    st.output("pick_report", p)
    st.extra.update(relatives=len(ids), by_split={sp: {g: v["picked"] for g, v in by.items()} for sp, by in table.items()})
    return rep


def page_of(rep):
    L = ["# The first batch of close relatives (package W13, step 2)", "", rep["note"], "",
         "k = %d per genome; census run `%s`; %s relatives; %s bases; about %s proteins outside the plasmids (one per kilobase, "
         "eukaryotes capped)." % (rep["k"], rep["census"], "{:,}".format(rep["relatives"]), "{:,}".format(rep["bases"]), "{:,}".format(rep["proteins_estimate"])), ""]
    for sp, by in rep["by_split"].items():
        L += ["## %s split" % sp, "", "| group | census | picked | anchors with a replaced pick |", "|---|---|---|---|"]
        for g, d in by.items():
            L.append("| %s | %s | %s | %s |" % (g, d["census"], d["picked"], d["anchors_with_a_replacement"]))
        L.append("")
    lk = rep["lookups"]
    L += ["ENA assembly records looked up: %s (%s without a record, %s listing no replicon); FASTA probed at ENA for %s picks, not served: %s." % (
        "{:,}".format(lk["records"]), lk["without_record"], lk.get("without_any_replicon", 0), "{:,}".format(lk.get("fasta_probed", 0)),
        ", ".join(lk.get("fasta_not_served") or []) or "none"), ""]
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
    if len(rel) != rep.get("relatives") or len({x["id"] for x in rel}) != len(rel):
        problems.append(dict(error="relatives counted twice or another count"))
    if any(x["split"] not in WC.OPEN for x in rel):
        problems.append(dict(error="a relative outside the train and held-out splits"))
    for sp, by in (rep.get("by_split") or {}).items():
        for g, d in by.items():
            if g != "mobile.plasmid" and d["picked"] > (d["census"] or 0):
                problems.append(dict(split=sp, group=g, error="more relatives than the census proposed"))
    with open(os.path.join(run_dir, OUT, "corpus.yaml")) as f:
        spec = yaml.safe_load(f)
    ids = [x["id"] for x in spec["sources"]]
    if len(set(ids)) != len(ids) or not {x["id"] for x in rel} <= set(ids):
        problems.append(dict(error="the corpus spec repeats a source or lacks a relative"))
    require(GateResult("G4_rel_w13_pick", "fail" if problems else "pass", dict(problems=problems[:50], relatives=rep.get("relatives"))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        rep = json.load(f)
    return dict(json=dict(relatives=rep["relatives"], by_split={sp: {g: v["picked"] for g, v in by.items()} for sp, by in rep["by_split"].items()}),
                md=page_of(rep).splitlines()[2:60])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w13_pick") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
