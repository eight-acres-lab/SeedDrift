"""Stage catalog (pipeline select): snapshots of the public genome catalogues the corpus selection reads.

Catalogue kinds (configs/select/<name>.yaml, `catalogues:`; each entry has an `id` and one kind):
  ncbi_genome_reports: {taxon, params}     NCBI Datasets v2 genome dataset_report of a taxon, every page
  ncbi_genome_accessions: {from: [ids], field}  NCBI genome reports of the assemblies earlier catalogues name (a field
                                           of their records; their paired assemblies too): annotation counts of Ensembl genomes
  ncbi_virus_reports: {taxon, params}      NCBI Datasets v2 virus dataset_report (the host of every nucleotide record)
  ensembl_species: {site, division, release}  the release's species_<Division>.txt
  file: <repository path>                  a committed snapshot in the canonical form (the fixtures of select-tiny)
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
    return {"kind": k, k: c[k]}


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

def _paged(url_of, reducer, key, fetch=get, tries=3):
    """All pages of a dataset_report; (records, total) once the distinct count equals the announced total."""
    last = None
    for attempt in range(tries):
        out, token, total, pages = [], "", None, 0
        while True:
            d = json.loads(fetch(url_of(token)))
            total = d.get("total_count", 0) if total is None else total
            out += [reducer(r) for r in d.get("reports") or []]
            pages += 1
            token = d.get("next_page_token") or ""
            if not token:
                break
            time.sleep(PAUSE)
        n = len({r[key] for r in out})
        if n == total:
            return out, total, pages
        last = "distinct records %d, announced %d" % (n, total)
        time.sleep(30 * (attempt + 1))
    raise SystemExit("catalogue did not read consistently in %d tries: %s" % (tries, last))


def fetch_genome_reports(c, fetch=get):
    p = dict(c["ncbi_genome_reports"].get("params") or {})
    base = "%s/genome/taxon/%d/dataset_report?" % (NCBI, int(c["ncbi_genome_reports"]["taxon"]))
    def url(tok):
        q = dict(p, page_size=1000, **({"page_token": tok} if tok else {}))
        return base + urllib.parse.urlencode(sorted(q.items()))
    return _paged(url, reduce_genome, "accession", fetch)


def fetch_virus_reports(c, fetch=get):
    p = dict(c["ncbi_virus_reports"].get("params") or {})
    base = "%s/virus/taxon/%d/dataset_report?" % (NCBI, int(c["ncbi_virus_reports"]["taxon"]))
    def url(tok):
        q = dict(p, page_size=1000, **({"page_token": tok} if tok else {}))
        return base + urllib.parse.urlencode(sorted(q.items()))
    return _paged(url, reduce_virus, "accession", fetch)


def fetch_accessions(accs, fetch=get, batch=50):
    """NCBI genome reports of the given assemblies, and of their paired assemblies (the annotated twin of a GenBank
    assembly is often the RefSeq one)."""
    out, asked = {}, set()
    todo = sorted({a for a in accs if a})
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
    return list(out.values()), len(out), len(asked)


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
        recs, n, asked = fetch_accessions(accs, fetch)
        prov = dict(asked=asked, found=n)
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
            else:
                data, n, prov = snapshot(c, done, cas)
                path, sha = store(cas, data, ".jsonl")
                notes[c["id"]] = dict(source="fetched", fetched=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **prov)
                nbytes = len(data)
            done[c["id"]] = path
            new_lock[c["id"]] = dict(kind=k, query_sha256=qsha, sha256=sha, records=n, bytes=nbytes)
            entries.append(dict(id=c["id"], kind=k, query=q, sha256=sha, records=n, path=os.path.relpath(path, cas), bytes=nbytes))
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
