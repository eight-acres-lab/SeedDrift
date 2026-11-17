"""Stage catalog (pipeline select): snapshots of the public genome catalogues the corpus selection reads.

Catalogue kinds (configs/select/<name>.yaml, `catalogues:`; each entry has an `id` and one kind):
  ncbi_genome_reports: {taxon, params}     NCBI Datasets v2 genome dataset_report of a taxon, every page
  ncbi_genome_accessions: {from: [ids], field}  NCBI genome reports of the assemblies earlier catalogues name (a field
                                           of their records; their paired assemblies too): annotation counts of Ensembl genomes
  ncbi_virus_reports: {taxon, params}      NCBI Datasets v2 virus dataset_report (the host of every nucleotide record)
  ensembl_species: {site, division, release}  the release's species_<Division>.txt
  file: <repository path>                  a committed snapshot in the canonical form (the fixtures of select-tiny)
An NCBI catalogue may name `totals: {label: params}`: wider queries whose record counts are kept with the snapshot, so
the report can say what the catalogue's filters leave out (`_kind`, `_taxon` count another endpoint or taxon).
Every snapshot is reduced to the fields the selection reads and made canonical: one JSON object per line, sorted keys,
records sorted by their key, no duplicate keys. A paged catalogue is accepted only when the number of distinct records
equals the total the server announced (pages of a catalogue that changes while it is read can skip or repeat records);
otherwise it is read again, and the stage fails after three tries. Snapshots are stored once in the CI's
content-addressed store (<data_root>/cas/<sha256[:2]>/<sha256>.jsonl).

Modes (profile `catalog: {mode, lock}`):
  lock     (default) every live catalogue needs an entry in the lock file (configs/select/<name>.lock.yaml) with the
           hash of its query and the sha256 of its snapshot; the snapshot is taken from the store, or fetched again and
           accepted only with exactly the locked sha256 (else the stage fails: the catalogue changed, refresh the lock)
  refresh  every live catalogue is fetched now; data/catalog/lock.yaml is the new lock (`ci/launch.sh promote RUN`
           writes it into the repository, where it is committed as a new CI version)
data/catalog/manifest.json lists each snapshot (id, kind, query, sha256, records, store path). Fetch dates and
reuse are kept in the stage report, never in an output, so two runs of one lock are identical.
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
import urllib.parse

import yaml

from ci import manifest as M
from ci.hashing import sha256_bytes, sha256_json
from ci.stages.common import Stage, main_args, write_json
from ci.stages.fetch import get, store, store_check

NCBI = "https://api.ncbi.nlm.nih.gov/datasets/v2"
ENSEMBL_SPECIES = {"ensembl": "https://ftp.ensembl.org/pub/release-%(release)s/species_Ensembl%(Division)s.txt",
                   "ensemblgenomes": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/%(division)s/release-%(release)s/species_Ensembl%(Division)s.txt"}
KINDS = ("ncbi_genome_reports", "ncbi_genome_accessions", "ncbi_virus_reports", "ensembl_species", "file")
KEY = {"ncbi_genome_reports": "accession", "ncbi_genome_accessions": "accession", "ncbi_virus_reports": "accession", "ensembl_species": "name"}
PAUSE = 0.4                                   # NCBI asks for at most 3 requests per second without an API key


def kind_of(c):
    ks = [k for k in KINDS if k in c]
    if len(ks) != 1:
        raise SystemExit("catalogue %s: needs exactly one kind of %s" % (c.get("id"), KINDS))
    return ks[0]


def describe(c):
    k = kind_of(c)
    return {"kind": k, k: c[k], "totals": c.get("totals") or {}}


def canonical(records, key):
    """Sorted JSON lines; duplicate keys with different content are an error, identical repeats are dropped."""
    by = {}
    for r in records:
        k = r[key]
        if k in by and by[k] != r:
            raise ValueError("record %s appears twice with different content" % k)
        by[k] = r
    return "".join(json.dumps(by[k], sort_keys=True, separators=(",", ":")) + "\n" for k in sorted(by)).encode(), len(by)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---- reducers: the fields the selection reads -------------------------------------------------------------------

def _int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def reduce_genome(r):
    ai, st, an, ck, org = (r.get(k) or {} for k in ("assembly_info", "assembly_stats", "annotation_info", "checkm_info", "organism"))
    gc = (an.get("stats") or {}).get("gene_counts") or {}
    return dict(accession=r["accession"], paired_accession=r.get("paired_accession") or "", taxid=_int(org.get("tax_id")), organism=org.get("organism_name") or "",
                assembly_level=ai.get("assembly_level") or "", assembly_status=ai.get("assembly_status") or "", refseq_category=ai.get("refseq_category") or "",
                length=_int(st.get("total_sequence_length")), chromosomes=_int(st.get("total_number_of_chromosomes")),
                components=_int(st.get("number_of_component_sequences")), protein_coding=_int(gc.get("protein_coding")),
                checkm_completeness=ck.get("completeness"), checkm_contamination=ck.get("contamination"), source_database=r.get("source_database") or "")


def reduce_virus(r):
    v, h = r.get("virus") or {}, r.get("host") or {}
    return dict(accession=r["accession"], virus_taxid=_int(v.get("tax_id")), host_taxid=_int(h.get("tax_id")), completeness=r.get("completeness") or "",
                protein_count=_int(r.get("protein_count")), length=_int(r.get("length")), source_database=r.get("source_database") or "")


def reduce_ensembl(row, site, division, release):
    return dict(name=row["name"], species=row.get("species", ""), division=row.get("division", ""), taxonomy_id=_int(row.get("taxonomy_id")),
                assembly=row.get("assembly", ""), assembly_accession=row.get("assembly_accession", ""), genebuild=row.get("genebuild", ""),
                core_db=row.get("core_db", ""), site=site, site_division=division, release=int(release))


# ---- fetchers ----------------------------------------------------------------------------------------------------

def params(p):
    """Query parameters as NCBI expects them: booleans as true/false, lists as repeated keys; nothing else."""
    out = []
    for k, v in sorted((p or {}).items()):
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, bool):
                x = "true" if x else "false"
            if not isinstance(x, (str, int)):
                raise SystemExit("query parameter %s: %r is not a string, number, boolean or list of them" % (k, x))
            out.append((k, str(x)))
    return out


def _paged(url_of, reducer, key, fetch=get, tries=3, page_size=1000, min_page=100):
    """All pages of a dataset_report; (records, total, pages) once the distinct count equals the announced total. A
    page without total_count, or a catalogue with no record at all, is refused (an error body or a filter that
    matches nothing must not become an empty snapshot). A page the server keeps failing (NCBI answers large virus
    pages with HTTP 500) restarts the read with half the page size, down to `min_page`; the snapshot does not depend
    on the page size (it is canonical)."""
    last = None
    attempt = 0
    while attempt < tries:
        out, token, total, pages = [], "", None, 0
        while True:
            try:
                d = json.loads(fetch(url_of(token, page_size)))
            except RuntimeError as err:
                if page_size <= min_page:
                    raise
                page_size = max(min_page, page_size // 2)
                last = "page failed (%s); page size now %d" % (str(err)[-80:], page_size)
                break
            if total is None:
                if "total_count" not in d:
                    raise SystemExit("catalogue page without total_count: %s" % str(d)[:300])
                total = int(d["total_count"])
                if total == 0:
                    raise SystemExit("catalogue query matches no record")
            out += [reducer(r) for r in d.get("reports") or []]
            pages += 1
            token = d.get("next_page_token") or ""
            if not token:
                break
            time.sleep(PAUSE)
        if total is None or token:                      # the read broke off: try again with the smaller page
            continue
        n = len({r[key] for r in out})
        if n == total:
            return out, total, pages
        last = "distinct records %d, announced %d" % (n, total)
        attempt += 1
        if attempt < tries:
            time.sleep(30 * attempt)
    raise SystemExit("catalogue did not read consistently in %d tries: %s" % (tries, last))


def fetch_genome_reports(c, fetch=get):
    p = dict(c["ncbi_genome_reports"].get("params") or {})
    base = "%s/genome/taxon/%d/dataset_report?" % (NCBI, int(c["ncbi_genome_reports"]["taxon"]))

    def url(tok, size):
        q = dict(p, page_size=size, **({"page_token": tok} if tok else {}))
        return base + urllib.parse.urlencode(params(q))
    return _paged(url, reduce_genome, "accession", fetch, page_size=int(c["ncbi_genome_reports"].get("page_size", 1000)))


def fetch_virus_reports(c, fetch=get):
    p = dict(c["ncbi_virus_reports"].get("params") or {})
    base = "%s/virus/taxon/%d/dataset_report?" % (NCBI, int(c["ncbi_virus_reports"]["taxon"]))

    def url(tok, size):
        q = dict(p, page_size=size, **({"page_token": tok} if tok else {}))
        return base + urllib.parse.urlencode(params(q))
    # NCBI answers 1,000-record virus pages with HTTP 500 (2026-11-17); 500 works
    return _paged(url, reduce_virus, "accession", fetch, page_size=int(c["ncbi_virus_reports"].get("page_size", 500)))


def count(kind, taxon, p, fetch=get):
    """The total_count of a query (one record asked for): how much a catalogue's filters leave out."""
    base = "%s/%s/taxon/%d/dataset_report?" % (NCBI, "genome" if kind == "ncbi_genome_reports" else "virus", int(taxon))
    d = json.loads(fetch(base + urllib.parse.urlencode(params(dict(p or {}, page_size=1)))))
    if "total_count" not in d:
        raise SystemExit("count query without total_count: %s" % str(d)[:300])
    return int(d["total_count"])


def fetch_accessions(accs, fetch=get, batch=50):
    """NCBI genome reports of the given assemblies, and of their paired assemblies (the annotated twin of a GenBank
    assembly is often the RefSeq one)."""
    out, asked = {}, set()
    todo = sorted({a for a in accs if a})
    wanted = set(todo)
    for _ in range(2):
        nxt = set()
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            asked |= set(chunk)
            url = "%s/genome/accession/%s/dataset_report?page_size=1000" % (NCBI, urllib.parse.quote(",".join(chunk)))
            d = json.loads(fetch(url))
            for r in d.get("reports") or []:
                rec = reduce_genome(r)
                out[rec["accession"]] = rec
                if rec["paired_accession"] and rec["paired_accession"] not in asked:
                    nxt.add(rec["paired_accession"])
            time.sleep(PAUSE)
        todo = sorted(nxt)
        if not todo:
            break
    missing = sorted(wanted - set(out))
    return list(out.values()), len(out), len(asked), missing


def fetch_ensembl(c, fetch=get):
    e = dict(c["ensembl_species"])
    e["Division"] = e.get("division_name") or {"vertebrates": "Vertebrates", "plants": "Plants", "fungi": "Fungi", "metazoa": "Metazoa",
                                                "protists": "Protists"}[e["division"]]
    url = ENSEMBL_SPECIES[e.get("site", "ensembl")] % e
    text = fetch(url).decode()
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines or not lines[0].startswith("#name"):
        raise SystemExit("%s: not an Ensembl species list" % url)
    rows = list(csv.DictReader(io.StringIO("\n".join([lines[0][1:]] + lines[1:])), delimiter="\t"))
    need = ("name", "species", "taxonomy_id", "assembly_accession", "genebuild", "core_db")
    gone = [c for c in need if c not in (rows[0].keys() if rows else lines[0][1:].split("\t"))]
    if gone or not rows:
        raise SystemExit("%s: the species list lacks %s" % (url, gone or "rows"))
    for r in rows:
        if not r.get("name") or not str(r.get("taxonomy_id") or "").isdigit():
            raise SystemExit("%s: a row without name or taxonomy id: %r" % (url, r))
    return [reduce_ensembl(r, e.get("site", "ensembl"), e["division"], e["release"]) for r in rows], len(rows), url


def snapshot(c, done, cas, fetch=get):
    """(bytes of the canonical snapshot, records, provenance) of a live catalogue; done: {id: path} of earlier ones."""
    k = kind_of(c)
    if k == "ncbi_genome_reports":
        recs, total, pages = fetch_genome_reports(c, fetch)
        prov = dict(total=total, pages=pages)
    elif k == "ncbi_virus_reports":
        recs, total, pages = fetch_virus_reports(c, fetch)
        prov = dict(total=total, pages=pages)
    elif k == "ncbi_genome_accessions":
        src = c[k]
        accs = []
        for pid in src["from"]:
            accs += [r.get(src.get("field", "assembly_accession"), "") for r in read_jsonl(done[pid])]
        recs, n, asked, missing = fetch_accessions(accs, fetch)
        prov = dict(asked=asked, found=n, not_found=len(missing), not_found_sha256=sha256_json(missing))
    elif k == "ensembl_species":
        recs, n, url = fetch_ensembl(c, fetch)
        prov = dict(url=url, rows=n)
    else:
        raise SystemExit("catalogue %s: %s is not a live kind" % (c["id"], k))
    data, n = canonical(recs, KEY[k])
    return data, n, prov


def main():
    run_dir = main_args()
    with Stage(run_dir, "catalog", requires=("pre",)) as st:
        cfg = st.config
        with open(st.repo(cfg["select_spec"])) as f:
            spec = yaml.safe_load(f)
        mode = (cfg.get("catalog") or {}).get("mode", "lock")
        if mode not in ("lock", "refresh"):
            raise SystemExit("catalog.mode must be lock or refresh, not %s" % mode)
        lock = {}
        lock_path = (cfg.get("catalog") or {}).get("lock")
        if mode == "lock" and lock_path and os.path.exists(st.repo(lock_path)):
            with open(st.repo(lock_path)) as f:
                lock = (yaml.safe_load(f) or {}).get("catalogues") or {}
        cas = os.path.join(cfg["data_root"], "cas")
        entries, done, new_lock, notes = [], {}, {}, {}
        for c in spec.get("catalogues") or []:
            k = kind_of(c)
            q = describe(c)
            qsha = sha256_json(q)
            if k == "file":
                path = st.repo(c["file"])
                with open(path, "rb") as f:
                    data = f.read()
                recs = [json.loads(l) for l in data.decode().splitlines() if l.strip()]
                key = c.get("key") or KEY.get(c.get("as", ""), "accession")
                canon, n = canonical(recs, key)
                if canon != data:
                    raise SystemExit("catalogue %s: %s is not in canonical form" % (c["id"], c["file"]))
                sha = sha256_bytes(data)
                done[c["id"]] = path
                entries.append(dict(id=c["id"], kind=k, query=q, sha256=sha, records=n, path=c["file"], bytes=len(data)))
                continue
            totals = {}
            if mode == "lock":
                want = lock.get(c["id"])
                if not want:
                    raise SystemExit("catalogue %s has no entry in the lock %s (run a refresh profile and promote its lock)" % (c["id"], lock_path))
                if want.get("query_sha256") != qsha:
                    raise SystemExit("catalogue %s: the spec's query changed since the lock was written; refresh the lock" % c["id"])
                path = os.path.join(cas, want["sha256"][:2], want["sha256"] + ".jsonl")
                if os.path.exists(path):
                    store_check(path, want["sha256"])
                    notes[c["id"]] = dict(source="store")
                    n = want["records"]
                    nbytes = os.path.getsize(path)
                else:
                    data, n, prov = snapshot(c, done, cas)
                    if sha256_bytes(data) != want["sha256"]:
                        raise SystemExit("catalogue %s changed since the lock (now %s records); refresh the lock" % (c["id"], n))
                    path, _ = store(cas, data, ".jsonl")
                    notes[c["id"]] = dict(source="fetched", fetched=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **prov)
                    nbytes = len(data)
                sha = want["sha256"]
                totals = want.get("totals") or {}
            else:
                data, n, prov = snapshot(c, done, cas)
                path, sha = store(cas, data, ".jsonl")
                notes[c["id"]] = dict(source="fetched", fetched=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **prov)
                nbytes = len(data)
                for label, tp in sorted((c.get("totals") or {}).items()):
                    q2 = dict(tp)                        # _kind / _taxon: count another query than the catalogue's own
                    totals[label] = dict(params=tp, total=count(q2.pop("_kind", k), q2.pop("_taxon", (c[k] or {}).get("taxon")), q2))
                    time.sleep(PAUSE)
            done[c["id"]] = path
            new_lock[c["id"]] = dict(kind=k, query_sha256=qsha, sha256=sha, records=n, bytes=nbytes, totals=totals)
            entries.append(dict(id=c["id"], kind=k, query=q, sha256=sha, records=n, path=os.path.relpath(path, cas), bytes=nbytes, totals=totals))
        man = M.build(spec["name"] + "-catalog", "catalog", entries, meta=dict(cas="<data_root>/cas", spec=spec["name"]))
        msha = M.write(man, os.path.join(run_dir, "manifests"))
        mp = st.path("catalog", "manifest.json")
        write_json(mp, dict(sha256=msha, dataset_id=M.dataset_id(man), **man))
        st.output("catalog_manifest", mp)
        if mode == "refresh":
            lp = st.path("catalog", "lock.yaml")
            with open(lp, "w") as f:
                yaml.safe_dump(dict(spec=spec["name"], catalogues=new_lock), f, sort_keys=True)
            st.output("catalog_lock", lp)
        st.extra.update(mode=mode, catalogues=len(entries), sources=notes)


if __name__ == "__main__":
    main()
