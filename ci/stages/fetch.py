"""Stage fetch: download the remote sources of a corpus spec (server tiers only; nothing is fetched by ci-tiny).

Source fields understood (configs/corpus/<name>.yaml, `sources:`):
  ena_assembly: GCA_...   all replicons of an INSDC assembly from ENA (optionally only `replicon_types`)
  ena_sequences: [acc.v]  INSDC sequences from ENA (viruses, single plasmids)
  ensembl: {site: ensembl | ensemblgenomes, division, species, release[, collection]}
                          the release-pinned coding sequences (cds.all) of an Ensembl species
  ncbi_datasets: {accession: GCA_/GCF_..., include: [CDS_FASTA, SEQUENCE_REPORT]}
                          the files of an assembly's annotation from NCBI Datasets (v2 genome download, a zip file kept
                          as it comes; no genome sequence): its coding sequences and its sequence report
Every downloaded file is stored once in the CI's content-addressed store (<cas>/<sha256[:2]>/<sha256>) and listed in
data/fetch/manifest.json with its URL, size and sha256. A rerun takes the stored file when the source description is
the same and never downloads again; a stored file whose content no longer matches its name is refused. A download
enters the store only when complete: every requested INSDC sequence present and non-empty, an Ensembl file a whole
gzip stream of FASTA. Temporary names are private to the process and the index is merged under a lock, so runs can
share the store. Paths in the
index and the manifest are relative to the store, so the store can be moved or merged (its files are named by content).
A source the archive has no record of (HTTP 404, or an assembly without replicons) is not a transient failure: with
`fetch.allow_missing` (a share of the remote sources, default 0) up to that many are left out and listed in the manifest
(`unavailable`, with the reason) and the corpus stage skips them; more than that, or any other failure, fails the stage.
"""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zlib

from ci import ROOT, manifest as M
from ci.hashing import sha256_json
from ci.stages.common import Stage, main_args, write_json

sys.path.insert(0, os.path.join(ROOT, "bench", "hard"))

ENA_FASTA = "https://www.ebi.ac.uk/ena/browser/api/fasta/%s?gzip=true"
NCBI_DATASETS = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/%s/download?%s"
NCBI_PARTS = {"CDS_FASTA": "cds_from_genomic.fna", "SEQUENCE_REPORT": "sequence_report.jsonl"}
ENSEMBL = {"ensembl": "https://ftp.ensembl.org/pub/release-%(release)s/fasta/%(species)s/cds/",
           "ensemblgenomes": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/%(division)s/release-%(release)s/fasta/%(species)s/cds/"}


class NoRecord(Exception):
    """The archive answered that it has no such record (not a network failure)."""


# profile block fetch: `retry_s` (waits between tries, e.g. [10, 30, 90]; default: `tries` tries `pause` x k apart) and
# `server_errors_unavailable` (a server error (5xx) or a failed connection that lasts through every try is a record the
# archive cannot give: left out and listed like a 404, under fetch.allow_missing; default: the stage fails). Set by main().
RETRY_S = None
SERVER_ERRORS_UNAVAILABLE = False
# profile block fetch: `incomplete_unavailable` (an INSDC FASTA whose answer lacks a requested sequence or holds it empty,
# asked for three times, is a record the archive cannot give in that version: left out and listed under
# fetch.allow_missing with the names the archive gave; default: the stage fails at once). Set by main().
INCOMPLETE_UNAVAILABLE = False


def get(url, tries=5, pause=10):
    err = None
    waits = list(RETRY_S) if RETRY_S else [pause * (k + 1) for k in range(tries)]
    for k in range(len(waits) + 1 if RETRY_S else tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            err = e
            if e.code == 404 and k >= 1:                         # asked twice: the archive has no such record
                raise NoRecord("%s: HTTP 404" % url)
            if 400 <= e.code < 500 and e.code not in (404, 408, 429) and RETRY_S:
                break                                            # another client error: asking again gives the same answer
        except Exception as e:  # noqa: BLE001
            err = e
        if k < len(waits):
            time.sleep(waits[k])
    if SERVER_ERRORS_UNAVAILABLE and (not isinstance(err, urllib.error.HTTPError) or err.code >= 500 or err.code in (408, 429)):
        raise NoRecord("%s: %s after %d tries (server error or no connection)" % (url, err, len(waits) + 1 if RETRY_S else tries))
    raise RuntimeError("%s: %s" % (url, err))


_GET = get                                                           # the network getter (tests replace `get`)


def get_ranged(url, parts=8, min_bytes=2_000_000, tries=5):
    """A large file in `parts` parallel byte ranges (EBI throttles each connection in the evening, not the total), each
    range retried on its own; the whole must have exactly the announced length. Falls back to get() when the server
    gives no length or no range support."""
    from concurrent.futures import ThreadPoolExecutor
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"}, method="HEAD")
        with urllib.request.urlopen(req, timeout=120) as r:
            n, ranges = int(r.headers.get("Content-Length") or 0), (r.headers.get("Accept-Ranges") or "").lower()
    except Exception:  # noqa: BLE001
        return get(url)
    if n < min_bytes or ranges != "bytes":
        return get(url)
    bounds = [(k * n // parts, (k + 1) * n // parts - 1) for k in range(parts)]

    def one(b):
        err = None
        for k in range(tries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "python-urllib", "Range": "bytes=%d-%d" % b})
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = r.read()
                if len(data) == b[1] - b[0] + 1:
                    return data
                err = "range %s: %d bytes" % (b, len(data))
            except Exception as e:  # noqa: BLE001
                err = e
            time.sleep(10 * (k + 1))
        raise RuntimeError("%s: %s" % (url, err))
    with ThreadPoolExecutor(parts) as pool:
        data = b"".join(pool.map(one, bounds))
    if len(data) != n:
        raise RuntimeError("%s: %d of %d bytes" % (url, len(data), n))
    return data


def store_check(path, sha):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    if h.hexdigest() != sha:
        raise SystemExit("content-addressed file %s is corrupt" % path)


def store(cas, data: bytes, suffix: str):
    sha = hashlib.sha256(data).hexdigest()
    path = os.path.join(cas, sha[:2], sha + suffix)
    if os.path.exists(path):
        store_check(path, sha)
        return path, sha
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import threading
    tmp = "%s.part%d.%d" % (path, os.getpid(), threading.get_ident())   # private to this process and thread
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path, sha


def check_fasta(data: bytes, want, what):
    """A downloaded FASTA must hold every requested sequence (ENA names ENA|X|X.v count as X.v), none of them empty."""
    names, lens, cur = set(), {}, None
    for line in data.decode(errors="replace").splitlines():
        if line.startswith(">"):
            first = line[1:].split()[0] if line[1:].split() else ""
            cur = first.split("|")[-1] if first.startswith("ENA|") else first
            names.add(cur)
            lens[cur] = 0
        elif cur is not None:
            lens[cur] += len(line.strip())
    bare = {n.split(".")[0]: n for n in names}                   # a requested accession without version matches any
    got = {w: (w if w in names else (bare.get(w) if "." not in w else None)) for w in want}
    missing = sorted(w for w, n in got.items() if n is None)
    empty = sorted(w for w, n in got.items() if n is not None and lens.get(n, 0) == 0)
    if missing or empty or not names:
        raise SystemExit("%s: incomplete download (missing %s, empty %s, %d records; the archive gave %s)" % (
            what, missing[:5], empty[:5], len(names), sorted(set(names) - set(got.values()))[:5]))


def ena_fasta(url, want, what):
    """The INSDC FASTA of `want` from ENA, checked (check_fasta()). With fetch.incomplete_unavailable an answer that
    lacks a requested sequence (a sequence of another version, say) is asked for again twice, then the source is a
    record the archive cannot give (NoRecord); otherwise the stage fails at once, as before."""
    tries = 3 if INCOMPLETE_UNAVAILABLE else 1
    for k in range(tries):
        data = get(url)
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        try:
            check_fasta(data, want, what)
            return data
        except SystemExit as err:
            if k == tries - 1:
                if INCOMPLETE_UNAVAILABLE:
                    raise NoRecord(str(err))
                raise


def check_ncbi_zip(data: bytes, accession, include):
    """The parts of an NCBI Datasets zip file [dict(name, part, bytes)]: a whole zip file (every member's CRC checked)
    with every requested part of the assembly, non-empty (coding sequences as FASTA). An answer without the coding
    sequences is an assembly NCBI holds no annotation files of (NoRecord)."""
    import io
    import zipfile
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        bad = z.testzip()
    except (zipfile.BadZipFile, zlib.error, EOFError) as err:
        raise RuntimeError("%s: not a complete zip file (%s)" % (accession, err))
    if bad:
        raise RuntimeError("%s: a damaged member %s" % (accession, bad))
    out = []
    for part in include:
        names = [n for n in z.namelist() if n.endswith("/" + NCBI_PARTS[part]) and "/%s/" % accession in "/" + n]
        if len(names) != 1:
            raise NoRecord("%s: NCBI Datasets gives no %s (members %s)" % (accession, NCBI_PARTS[part], z.namelist()[:8]))
        info = z.getinfo(names[0])
        with z.open(names[0]) as f:
            head = f.read(1)
        if not info.file_size or (part == "CDS_FASTA" and head != b">"):
            raise NoRecord("%s: the %s of NCBI Datasets is empty or not FASTA" % (accession, NCBI_PARTS[part]))
        out.append(dict(name=names[0], part=part, bytes=info.file_size))
    return out


def describe(src):
    return {k: src[k] for k in ("ena_assembly", "ena_sequences", "ensembl", "replicon_types", "ncbi_datasets") if k in src}


def fetch_source(src, cas, index):
    """(files, provenance) of one source; index maps a source description hash to a previous fetch."""
    key = sha256_json(describe(src))
    if key in index and all(os.path.exists(os.path.join(cas, f["path"])) for f in index[key]["files"]):
        files = [dict(f, path=os.path.join(cas, f["path"])) for f in index[key]["files"]]
        for f in files:                                          # a stored file is used only while it matches its name
            store_check(f["path"], f["sha256"])
        return files, index[key]["provenance"], True
    files, prov = [], []
    if "ena_assembly" in src:
        from panel import replicons
        reps = replicons(src["ena_assembly"])
        types = set(src.get("replicon_types") or [])
        accs = [a for a, t in reps if not types or t in types]
        if not reps:
            raise NoRecord("%s: the ENA assembly record lists no replicons" % src["ena_assembly"])
        if not accs:
            raise SystemExit("%s: no replicons of types %s" % (src["ena_assembly"], sorted(types)))
        url = ENA_FASTA % ",".join(accs)
        data = ena_fasta(url, accs, src["ena_assembly"])
        path, sha = store(cas, data, ".fna")
        files.append(dict(path=path, sha256=sha, bytes=len(data), kind="fna"))
        prov.append(dict(url=url, replicons=accs))
    if "ena_sequences" in src:
        url = ENA_FASTA % ",".join(src["ena_sequences"])
        data = ena_fasta(url, src["ena_sequences"], src["id"])
        path, sha = store(cas, data, ".fna")
        files.append(dict(path=path, sha256=sha, bytes=len(data), kind="fna"))
        prov.append(dict(url=url))
    if "ensembl" in src:
        e = dict(src["ensembl"])
        base = ENSEMBL[e.get("site", "ensembl")] % e
        if e.get("collection"):                                    # Ensembl Genomes species of a collection live one level deeper
            base = base.replace("/fasta/%s/" % e["species"], "/fasta/%s/%s/" % (e["collection"], e["species"]))
        listing = get(base).decode(errors="replace")
        names = sorted(set(re.findall(r'href="([A-Za-z0-9_.\-]+\.cds\.all\.fa\.gz)"', listing)))
        if len(names) != 1:
            raise SystemExit("%s: expected one cds.all file, found %s" % (base, names))
        data = get_ranged(base + names[0]) if get is _GET else get(base + names[0])   # a replaced getter is used as it is
        try:                                                     # the whole stream, so a truncated file never enters the store
            text = gzip.decompress(data)
        except (OSError, EOFError) as err:
            raise SystemExit("%s: not a complete gzip file (%s)" % (base + names[0], err))
        if not text.startswith(b">"):
            raise SystemExit("%s: not a FASTA file" % (base + names[0]))
        path, sha = store(cas, data, ".cds.fa.gz")
        files.append(dict(path=path, sha256=sha, bytes=len(data), kind="cds.fa.gz"))
        prov.append(dict(url=base + names[0]))
    if "ncbi_datasets" in src:
        d = dict(src["ncbi_datasets"])
        include = [str(x) for x in d.get("include") or ["CDS_FASTA"]]
        unknown = [x for x in include if x not in NCBI_PARTS]
        if unknown:
            raise SystemExit("%s: ncbi_datasets.include names unknown parts %s" % (src["id"], unknown))
        url = NCBI_DATASETS % (d["accession"], "&".join("include_annotation_type=%s" % x for x in include))
        for k in range(3):                                         # a damaged zip file is asked for again, twice
            data = get(url)
            try:
                members = check_ncbi_zip(data, d["accession"], include)
                break
            except RuntimeError as err:
                if k == 2:
                    raise (NoRecord if SERVER_ERRORS_UNAVAILABLE else RuntimeError)(str(err))
        path, sha = store(cas, data, ".ncbi.zip")
        files.append(dict(path=path, sha256=sha, bytes=len(data), kind="ncbi.zip"))
        prov.append(dict(url=url, members=members))
    return files, prov, False


def main():
    run_dir = main_args()
    with Stage(run_dir, "fetch", requires=("pre",)) as st:
        import yaml
        cfg = st.config
        with open(st.repo(cfg["corpus_spec"])) as f:
            spec = yaml.safe_load(f)
        global RETRY_S, SERVER_ERRORS_UNAVAILABLE, INCOMPLETE_UNAVAILABLE
        fc = cfg.get("fetch") or {}
        RETRY_S = [float(x) for x in fc["retry_s"]] if fc.get("retry_s") else None
        SERVER_ERRORS_UNAVAILABLE = bool(fc.get("server_errors_unavailable"))
        INCOMPLETE_UNAVAILABLE = bool(fc.get("incomplete_unavailable"))
        cas = os.path.join(cfg["data_root"], "cas")
        index_path = os.path.join(cas, "fetch_index.json")
        index = json.load(open(index_path)) if os.path.exists(index_path) else {}
        import threading
        from concurrent.futures import ThreadPoolExecutor
        ilock = threading.Lock()

        def merge(new):
            """Add entries to the shared index at once (under a file lock: other runs may write it too), so an
            interrupted stage does not download again what it already stored."""
            os.makedirs(cas, exist_ok=True)
            with ilock, open(os.path.join(cas, "fetch_index.lock"), "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                cur = json.load(open(index_path)) if os.path.exists(index_path) else {}
                cur.update(new)
                tmp = "%s.part%d" % (index_path, os.getpid())
                with open(tmp, "w") as f:
                    json.dump(cur, f, indent=1, sort_keys=True)
                os.replace(tmp, index_path)

        def one(src):
            try:
                files, prov, hit = fetch_source(src, cas, index)
            except NoRecord as err:
                return dict(id=src["id"], source=describe(src), unavailable=str(err)), False
            rel = [dict(f, path=os.path.relpath(f["path"], cas)) for f in files]
            if not hit:
                merge({sha256_json(describe(src)): dict(files=rel, provenance=prov)})
            return dict(id=src["id"], source=describe(src), files=rel, provenance=prov), hit
        remote = [src for src in spec.get("sources") or [] if describe(src)]
        # sources download in parallel (ENA and EBI scale with connections); the manifest keeps the spec's order
        with ThreadPoolExecutor(int((cfg.get("fetch") or {}).get("threads", 8))) as pool:
            done = list(pool.map(one, remote))
        missing = [dict(id=e["id"], reason=e["unavailable"]) for e, _ in done if "unavailable" in e]
        allowed = float((cfg.get("fetch") or {}).get("allow_missing", 0)) * len(remote)
        st.extra["unavailable"] = missing
        if len(missing) > allowed:
            raise SystemExit("%d sources have no record in the archive (fetch.allow_missing allows %d): %s"
                             % (len(missing), int(allowed), "; ".join("%s (%s)" % (m["id"], m["reason"]) for m in missing[:20])))
        entries = [e for e, _ in done if "unavailable" not in e]
        reused = sum(1 for _, h in done if h)
        man = M.build(spec["name"] + "-fetch", "fetch", entries, meta=dict(cas="<data_root>/cas", unavailable=missing))
        sha = M.write(man, os.path.join(run_dir, "manifests"))
        mp = st.path("fetch", "manifest.json")
        write_json(mp, dict(sha256=sha, dataset_id=M.dataset_id(man), **man))
        st.output("fetch_manifest", mp)
        st.extra.update(sources=len(entries), reused=reused)


if __name__ == "__main__":
    main()
