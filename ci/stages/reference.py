"""Stage reference: the protein reference the searches run against, with the two tables every later stage needs.

  index.npz     acc (sorted accessions), taxid (per accession): a target protein's index is its row here
  library.npz   taxid (sorted, one row per reference proteome), ranks [phylum, class, order, family, genus, species]
config reference.faa: a FASTA with headers accession|taxid; the DIAMOND database is built from it (tiny and smoke).
config reference.dmnd (+ reference.faa for the index): an existing database used in place (smoke, full).
config reference.cache: true keeps the two tables in the CI's content store (<data_root>/cas/reference/<key>/), keyed
by the FASTA's sha256, the taxonomy's hash and the hashes of the code that builds them (CODE); a later run hard-links them (after checking their
hashes) instead of reading the whole FASTA again. ci-tiny never uses the cache, so both of its runs recompute.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np

from ci import ROOT, manifest as M
from ci.hashing import sha256_file, sha256_json, taxdump_hash
from ci.stages.common import Stage, main_args, write_json
from ci.taxonomy import Taxonomy
from seeddrift.train.logger import save_npz

RANKS = ("phylum", "class", "order", "family", "genus", "species")
CODE = ("ci/stages/reference.py", "ci/taxonomy.py", "seeddrift/train/logger.py")    # the code the tables depend on (cache key)


def headers(faa):
    acc, tax = [], []
    with open(faa, "rb") as f:
        for line in f:
            if line[:1] == b">":
                a, _, t = line[1:].strip().partition(b"|")
                acc.append(a.decode())
                tax.append(int(t))
    return acc, tax


def build_tables(st, faa, out, lib):
    acc, tax = headers(faa)
    if len(set(acc)) != len(acc):
        raise SystemExit("duplicate accessions in the reference")
    order = np.argsort(np.array(acc, dtype="S"), kind="stable")
    save_npz(out, acc=np.array(acc, dtype="S")[order], taxid=np.array(tax, dtype=np.int64)[order])
    taxo = Taxonomy(st.repo(st.config["taxdump"]))
    utax = np.array(sorted(set(tax)), dtype=np.int64)
    ranks = np.array([[taxo.at_rank(int(t), r) for r in RANKS] for t in utax], dtype=np.int64)
    save_npz(lib, taxid=utax, ranks=ranks, rank_names=np.array(RANKS))
    return len(acc), len(utax)


def place(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def cached_tables(st, faa, out, lib):
    """(proteins, proteomes, cache note): the tables from the content store, or built and stored there."""
    cas = os.path.join(st.config["data_root"], "cas", "reference")
    code = [sha256_file(os.path.join(ROOT, p), use_sidecar=False) for p in CODE]
    key = sha256_json(["reference-tables", sha256_file(faa), taxdump_hash(st.config["taxdump"]), code])
    d = os.path.join(cas, key)
    meta_p = os.path.join(d, "meta.json")
    if os.path.exists(meta_p):
        with open(meta_p) as f:
            meta = json.load(f)
        for name, dst in (("index.npz", out), ("library.npz", lib)):
            src = os.path.join(d, name)
            if sha256_file(src, use_sidecar=False) != meta["sha256"][name]:
                raise SystemExit("cached reference table %s does not match its recorded hash" % src)
            if os.path.exists(dst):
                os.remove(dst)
            place(src, dst)
        return meta["proteins"], meta["proteomes"], "reused %s" % key[:12]
    n, m = build_tables(st, faa, out, lib)
    tmp = d + ".part%d" % os.getpid()
    os.makedirs(tmp, exist_ok=True)
    for name, src in (("index.npz", out), ("library.npz", lib)):
        shutil.copyfile(src, os.path.join(tmp, name))
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump(dict(proteins=n, proteomes=m, sha256={k: sha256_file(os.path.join(tmp, k), use_sidecar=False) for k in ("index.npz", "library.npz")}), f, sort_keys=True)
    try:
        os.rename(tmp, d)
    except OSError:                                    # another run stored the same key first
        shutil.rmtree(tmp, ignore_errors=True)
    return n, m, "stored %s" % key[:12]


def main():
    run_dir = main_args()
    with Stage(run_dir, "reference", requires=("pre",)) as st:
        cfg = st.config["reference"]
        faa = st.repo(cfg["faa"])
        out = st.path("reference", "index.npz")
        lib = st.path("reference", "library.npz")
        if cfg.get("cache"):
            n_prot, n_prom, note = cached_tables(st, faa, out, lib)
            st.extra["tables"] = note
        else:
            n_prot, n_prom = build_tables(st, faa, out, lib)
        st.output("index", out)
        st.output("library", lib)
        if cfg.get("dmnd"):
            db = st.repo(cfg["dmnd"])
            st.extra["database"] = "in place: %s" % db
        else:
            db = st.path("reference", "ref.dmnd")
            subprocess.run([st.config["tools"]["diamond"], "makedb", "--in", faa, "-d", db[:-5], "--quiet"], check=True)
            st.output("database", db)
        root = ROOT if faa.startswith(ROOT + os.sep) else ""                # repository files by relative path (same on every machine)
        files = [M.file_entry(faa, root)]
        if cfg.get("dmnd"):                                             # a database used in place is part of the dataset identity
            files.append(M.file_entry(db))
        entries = [dict(id="reference", files=files, proteins=n_prot, proteomes=n_prom)]
        man = M.build("reference", "reference", entries, meta=dict(database=os.path.relpath(db, run_dir) if db.startswith(run_dir) else db))
        sha = M.write(man, os.path.join(run_dir, "manifests"))
        mp = st.path("reference", "manifest.json")
        write_json(mp, dict(sha256=sha, dataset_id=M.dataset_id(man), database=os.path.relpath(db, run_dir) if db.startswith(run_dir) else db, **man))
        st.output("reference_manifest", mp)
        st.extra.update(proteins=n_prot, proteomes=n_prom)


if __name__ == "__main__":
    main()
