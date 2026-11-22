"""Stage rel_w13_catalog (package W13; pipeline rel_corpus): a fresh, wider catalogue snapshot of the public genome
assemblies, metadata only, for the census of close relatives (task breakdown W13 step 1; the programme lead's ruling of
2026-11-22: the prokaryote catalogues of the corpus selection are NCBI reference genomes only, about one per species, so
a count of relatives needs a wider snapshot; no sequence is downloaded). `python -m ci.stages.rel.w13_catalog RUN_DIR`.

Source: the ENA portal API (result=assembly; INSDC assemblies, the archive the corpus fetch stage downloads from), one
query per catalogue, the record fields of `fields`. The portal answers one row per version of an assembly (the
accession without its version, the version in its own field) and announces the count of rows. A snapshot is accepted
only when its rows equal the count the server announces for the same query (read again otherwise, at most three
times). It is made canonical: one record per accession, the highest version (then the latest update), tab-separated,
the header, rows sorted by accession; the lock records the rows, the records and the rows dropped (older versions, repeats). It is
stored once in the CI's content store (<data_root>/cas) and copied into the run. For context the stage also records, per taxon, the count of every assembly
level (counts only).

Configuration (profile block rel_w13_catalog): catalogues ([{id, query}], default: bacteria and archaea at the levels
complete genome and chromosome, viruses complete genome), fields, levels ({taxon: [levels]}, the counts recorded).
Outputs under data/rel_w13/catalog/: <id>.tsv per catalogue, lock.yaml (query, fields, records, sha256, fetch date,
source) and manifest.json (written last). Gate G4_rel_w13_catalog: every catalogue is there with its records and
sha256 as the lock says, and no row lacks an accession or a taxid.
"""
from __future__ import annotations

import csv
import datetime
import io
import json
import os
import shutil
import urllib.parse

import yaml

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common

OUT = os.path.join("data", "rel_w13", "catalog")
PORTAL = "https://www.ebi.ac.uk/ena/portal/api"
LEVELS = '(assembly_level="complete genome" OR assembly_level="chromosome")'
CATALOGUES = [dict(id="bacteria", query="tax_tree(2) AND %s" % LEVELS), dict(id="archaea", query="tax_tree(2157) AND %s" % LEVELS),
              dict(id="viruses", query='tax_tree(10239) AND assembly_level="complete genome"')]
FIELDS = ["accession", "tax_id", "scientific_name", "strain", "assembly_level", "genome_representation", "base_count", "completeness_score",
          "contamination_score", "host_tax_id", "environmental_sample", "version", "last_updated"]
CONTEXT = {"2": ["complete genome", "chromosome", "scaffold", "contig"], "2157": ["complete genome", "chromosome", "scaffold", "contig"],
           "10239": ["complete genome", "chromosome", "scaffold", "contig"]}
KEYS = {"catalogues", "fields", "levels"}


def settings(cfg) -> dict:
    c = cfg.get("rel_w13_catalog") or {}
    unknown = set(c) - KEYS
    if unknown:
        raise SystemExit("rel_w13_catalog: unknown keys %s" % sorted(unknown))
    cats = [dict(id=str(x["id"]), query=str(x["query"])) for x in c.get("catalogues") or CATALOGUES]
    fields = [str(x) for x in c.get("fields") or FIELDS]
    if len({x["id"] for x in cats}) != len(cats) or not {"accession", "tax_id", "version", "last_updated"} <= set(fields):
        raise SystemExit("rel_w13_catalog: catalogue ids must be distinct and the fields name accession, tax_id, version and last_updated")
    return dict(catalogues=cats, fields=fields, levels={str(k): list(v) for k, v in (c.get("levels") or CONTEXT).items()})


def get(url):
    from ci.stages import fetch
    return fetch.get(url)


def count(query):
    body = get("%s/count?%s" % (PORTAL, urllib.parse.urlencode(dict(result="assembly", query=query)))).decode()
    lines = [x for x in body.strip().splitlines() if x.strip()]
    return int(lines[-1].strip())


def _version(r):
    v = (r.get("version") or "").strip()
    return int(v) if v.isdigit() else -1


def canonical(body, fields):
    """(canonical TSV bytes, info) of a portal answer: one record per accession, the highest version, then the latest
    update, then the greatest row (a fixed order for rows the two do not tell apart); the header of `fields`, rows
    sorted by accession. info: rows (of the answer), records, older (versions dropped), ties (several rows of an
    accession with the same version and update date)."""
    rows = list(csv.DictReader(io.StringIO(body.decode()), delimiter="\t"))
    by, older, ties = {}, 0, 0
    clean = lambda v: (v or "").replace("\t", " ").replace("\n", " ")         # noqa: E731
    for r in rows:
        acc = (r.get("accession") or r.get("assembly_accession") or "").strip()
        if not acc:
            raise SystemExit("rel_w13_catalog: a row without an accession")
        r = dict(r, accession=acc)
        key = (_version(r), (r.get("last_updated") or "").strip(), [clean(r.get(f)) for f in fields])
        if acc in by:
            older += 1
            if key[:2] == by[acc][0][:2] and key[2] != by[acc][0][2]:
                ties += 1
            if key <= by[acc][0]:
                continue
        by[acc] = (key, r)
    out = io.StringIO()
    w = csv.writer(out, delimiter="\t", lineterminator="\n")
    w.writerow(fields)
    for acc in sorted(by):
        w.writerow(by[acc][0][2])
    return out.getvalue().encode(), dict(rows=len(rows), records=len(by), older=older, ties=ties)


def snapshot(cat, fields, tries=3):
    """(canonical bytes, info, announced total) of one catalogue, accepted when the answer has every row announced."""
    last = None
    for _ in range(tries):
        total = count(cat["query"])
        body = get("%s/search?%s" % (PORTAL, urllib.parse.urlencode(dict(result="assembly", query=cat["query"], fields=",".join(f for f in fields if f != "accession"),
                                                                         format="tsv", limit=0))))
        data, info = canonical(body, fields)
        if info["rows"] == total:
            return data, info, total
        last = "%d rows, the server announced %d" % (info["rows"], total)
    raise SystemExit("rel_w13_catalog: catalogue %s is not complete after %d tries (%s)" % (cat["id"], tries, last))


def build(st, cfg, run_dir):
    from ci.stages import fetch
    s = settings(cfg)
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    cas = os.path.join(cfg["data_root"], "cas")
    lock, files = dict(source=PORTAL, result="assembly", fields=s["fields"], catalogues={}, context={}), []
    for cat in s["catalogues"]:
        data, info, total = snapshot(cat, s["fields"])
        path, sha = fetch.store(cas, data, ".tsv")
        p = os.path.join(folder, cat["id"] + ".tsv")
        shutil.copyfile(path, p)
        lock["catalogues"][cat["id"]] = dict(query=cat["query"], query_sha256=sha256_json(dict(query=cat["query"], fields=s["fields"])), announced=total,
                                             rows=info["rows"], records=info["records"], older_versions_dropped=info["older"], version_ties=info["ties"],
                                             sha256=sha, bytes=len(data), fetched=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        files.append(p)
        print("rel_w13_catalog: %s: %d rows, %d records (%d older versions dropped)" % (cat["id"], info["rows"], info["records"], info["older"]), flush=True)
    for taxon, levels in s["levels"].items():
        lock["context"][taxon] = {lv: count('tax_tree(%s) AND assembly_level="%s"' % (taxon, lv)) for lv in levels}
    lp = os.path.join(folder, "lock.yaml")
    with open(lp, "w") as f:
        f.write("# Catalogue snapshot of stage rel_w13_catalog (metadata only; no sequence was downloaded).\n")
        yaml.safe_dump(lock, f, sort_keys=True)
    files.append(lp)
    man = dict(format="rel-w13-catalog", version=1, note=__doc__.strip().splitlines()[0], lock=lock,
               files=[dict(path=os.path.relpath(p, run_dir), sha256=sha256_file(p, use_sidecar=False), bytes=os.path.getsize(p)) for p in files])
    mp = os.path.join(folder, "manifest.json")
    write_json(mp, man)
    st.output("catalog_manifest", mp)
    st.extra.update(records={k: v["records"] for k, v in lock["catalogues"].items()}, context=lock["context"])
    return man


def read(run, same_run=False):
    """({catalogue: [rows]}, lock, provenance) of a passed run of this stage, every file checked (same_run: the run that
    reads it, whose record is not final yet; its stage rel_w13_catalog must have passed)."""
    from ci.stages.rel import w2_v0 as W
    rp = os.path.join(run, "record.json")
    if not os.path.exists(rp):
        raise SystemExit("rel_w13_catalog: %s holds no run record" % run)
    with open(rp) as f:
        rec = json.load(f)
    if not same_run and rec.get("status") != "passed":
        raise SystemExit("rel_w13_catalog: the catalogue run %s is %s, not passed" % (rec.get("run_id"), rec.get("status")))
    mp = os.path.join(run, OUT, "manifest.json")
    sha = W._output(run, "rel_w13_catalog", "catalog_manifest", mp)
    with open(mp) as f:
        man = json.load(f)
    files = {e["path"]: e["sha256"] for e in man.get("files") or []}
    out = {}
    for cid, c in man["lock"]["catalogues"].items():
        rel = os.path.join(OUT, cid + ".tsv")
        p = os.path.join(run, rel)
        if files.get(rel) != c["sha256"] or sha256_file(p, use_sidecar=False) != c["sha256"]:
            raise SystemExit("rel_w13_catalog: %s is not the snapshot its lock names" % rel)
        with open(p) as f:
            out[cid] = list(csv.DictReader(f, delimiter="\t"))
        if len(out[cid]) != c["records"]:
            raise SystemExit("rel_w13_catalog: %s holds %d rows, its lock %d" % (rel, len(out[cid]), c["records"]))
    return out, man["lock"], dict(run_id=rec.get("run_id") or os.path.basename(run), commit=(rec.get("git") or {}).get("sha"), manifest_sha256=sha)


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "manifest.json")
    with open(p) as f:
        man = json.load(f)
    problems = []
    for e in man.get("files") or []:
        full = os.path.join(run_dir, e["path"])
        if not os.path.exists(full) or sha256_file(full, use_sidecar=False) != e["sha256"]:
            problems.append(dict(file=e["path"], error="missing or not the file the manifest names"))
    s = settings(cfg)
    for cat in s["catalogues"]:
        c = (man.get("lock") or {}).get("catalogues", {}).get(cat["id"])
        if not c or c.get("rows") != c.get("announced") or c.get("records", 0) + c.get("older_versions_dropped", -1) != c.get("rows") or c.get("query") != cat["query"]:
            problems.append(dict(catalogue=cat["id"], error="missing, incomplete or another query"))
            continue
        with open(os.path.join(run_dir, OUT, cat["id"] + ".tsv")) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if (len(rows) != c["records"] or len({r.get("accession") for r in rows}) != len(rows)
                or any(not r.get("accession") or not (r.get("tax_id") or "").strip().isdigit() for r in rows)):
            problems.append(dict(catalogue=cat["id"], error="rows without an accession or a taxid, an accession twice, or another count"))
    require(GateResult("G4_rel_w13_catalog", "fail" if problems else "pass", dict(problems=problems[:50], records={k: v.get("records") for k, v in
                                                                                                                  ((man.get("lock") or {}).get("catalogues") or {}).items()})), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "manifest.json")) as f:
        man = json.load(f)
    lock = man["lock"]
    return dict(json=dict(records={k: v["records"] for k, v in lock["catalogues"].items()}, context=lock["context"]),
                md=["Catalogue snapshot (ENA portal, metadata only): " + "; ".join("%s %s records (%s rows, %s older versions dropped)" % (
                    k, "{:,}".format(v["records"]), "{:,}".format(v["rows"]), "{:,}".format(v["older_versions_dropped"])) for k, v in lock["catalogues"].items()),
                    "Assembly levels (counts): " + "; ".join("taxon %s: %s" % (t, ", ".join("%s %s" % (lv, "{:,}".format(n)) for lv, n in d.items())) for t, d in lock["context"].items())])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w13_catalog") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
