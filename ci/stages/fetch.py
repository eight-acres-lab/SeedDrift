"""Stage fetch: download the remote sources of a corpus spec (server tiers only; nothing is fetched by ci-tiny).

Source fields understood (configs/corpus/<name>.yaml, `sources:`):
  ena_assembly: GCA_...   all replicons of an INSDC assembly from ENA (optionally only `replicon_types`)
  ena_sequences: [acc.v]  INSDC sequences from ENA (viruses, single plasmids)
  ensembl: {site: ensembl | ensemblgenomes, division, species, release[, collection]}
                          the release-pinned coding sequences (cds.all) of an Ensembl species
Every downloaded file is stored once in the CI's content-addressed store (<cas>/<sha256[:2]>/<sha256>) and listed in
data/fetch/manifest.json with its URL, size and sha256. A rerun takes the stored file when the source description is
the same and never downloads again; a stored file whose content no longer matches its name is refused. A download
enters the store only when complete: every requested INSDC sequence present and non-empty, an Ensembl file a whole
gzip stream of FASTA. Temporary names are private to the process and the index is merged under a lock, so runs can
share the store. Paths in the
index and the manifest are relative to the store, so the store can be moved or merged (its files are named by content).
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
import urllib.request

from ci import ROOT, manifest as M
from ci.hashing import sha256_json
from ci.stages.common import Stage, main_args, write_json

sys.path.insert(0, os.path.join(ROOT, "bench", "hard"))

ENA_FASTA = "https://www.ebi.ac.uk/ena/browser/api/fasta/%s?gzip=true"
ENSEMBL = {"ensembl": "https://ftp.ensembl.org/pub/release-%(release)s/fasta/%(species)s/cds/",
           "ensemblgenomes": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/%(division)s/release-%(release)s/fasta/%(species)s/cds/"}


def get(url, tries=5, pause=10):
    err = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(pause * (k + 1))
    raise RuntimeError("%s: %s" % (url, err))


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
    tmp = "%s.part%d" % (path, os.getpid())                        # private to this process (runs may share the store)
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
        raise SystemExit("%s: incomplete download (missing %s, empty %s, %d records)" % (what, missing[:5], empty[:5], len(names)))


def describe(src):
    return {k: src[k] for k in ("ena_assembly", "ena_sequences", "ensembl", "replicon_types") if k in src}


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
        if not accs:
            raise SystemExit("%s: no replicons of types %s" % (src["ena_assembly"], sorted(types)))
        url = ENA_FASTA % ",".join(accs)
        data = get(url)
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        check_fasta(data, accs, src["ena_assembly"])
        path, sha = store(cas, data, ".fna")
        files.append(dict(path=path, sha256=sha, bytes=len(data), kind="fna"))
        prov.append(dict(url=url, replicons=accs))
    if "ena_sequences" in src:
        url = ENA_FASTA % ",".join(src["ena_sequences"])
        data = get(url)
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        check_fasta(data, src["ena_sequences"], src["id"])
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
        data = get(base + names[0])
        try:                                                     # the whole stream, so a truncated file never enters the store
            text = gzip.decompress(data)
        except (OSError, EOFError) as err:
            raise SystemExit("%s: not a complete gzip file (%s)" % (base + names[0], err))
        if not text.startswith(b">"):
            raise SystemExit("%s: not a FASTA file" % (base + names[0]))
        path, sha = store(cas, data, ".cds.fa.gz")
        files.append(dict(path=path, sha256=sha, bytes=len(data), kind="cds.fa.gz"))
        prov.append(dict(url=base + names[0]))
    return files, prov, False


def main():
    run_dir = main_args()
    with Stage(run_dir, "fetch", requires=("pre",)) as st:
        import yaml
        cfg = st.config
        with open(st.repo(cfg["corpus_spec"])) as f:
            spec = yaml.safe_load(f)
        cas = os.path.join(cfg["data_root"], "cas")
        index_path = os.path.join(cas, "fetch_index.json")
        index = json.load(open(index_path)) if os.path.exists(index_path) else {}
        entries, reused, mine = [], 0, {}
        for src in spec.get("sources") or []:
            if not describe(src):
                continue
            files, prov, hit = fetch_source(src, cas, index)
            reused += hit
            rel = [dict(f, path=os.path.relpath(f["path"], cas)) for f in files]
            index[sha256_json(describe(src))] = mine[sha256_json(describe(src))] = dict(files=rel, provenance=prov)
            entries.append(dict(id=src["id"], source=describe(src), files=rel, provenance=prov))
        if entries:                                    # a spec without remote sources (ci-tiny) leaves the store alone
            os.makedirs(cas, exist_ok=True)
            with open(os.path.join(cas, "fetch_index.lock"), "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)         # another run may have added entries since we read the index
                cur = json.load(open(index_path)) if os.path.exists(index_path) else {}
                cur.update(mine)
                tmp = "%s.part%d" % (index_path, os.getpid())
                with open(tmp, "w") as f:
                    json.dump(cur, f, indent=1, sort_keys=True)
                os.replace(tmp, index_path)
        man = M.build(spec["name"] + "-fetch", "fetch", entries, meta=dict(cas="<data_root>/cas"))
        sha = M.write(man, os.path.join(run_dir, "manifests"))
        mp = st.path("fetch", "manifest.json")
        write_json(mp, dict(sha256=sha, dataset_id=M.dataset_id(man), **man))
        st.output("fetch_manifest", mp)
        st.extra.update(sources=len(entries), reused=reused)


if __name__ == "__main__":
    main()
