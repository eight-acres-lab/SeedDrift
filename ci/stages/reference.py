"""Stage reference: the protein reference the searches run against, with the two tables every later stage needs.

  index.npz     acc (sorted accessions), taxid (per accession): a target protein's index is its row here
  library.npz   taxid (sorted, one row per reference proteome), ranks [phylum, class, order, family, genus, species]
config reference.faa: a FASTA with headers accession|taxid; the DIAMOND database is built from it (tiny and smoke).
config reference.dmnd (+ reference.faa for the index): an existing database used in place (full); the index is built
once from the FASTA headers.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

from ci import manifest as M
from ci.stages.common import Stage, main_args, write_json
from ci.taxonomy import Taxonomy
from seeddrift.train.logger import save_npz

RANKS = ("phylum", "class", "order", "family", "genus", "species")


def headers(faa):
    acc, tax = [], []
    with open(faa) as f:
        for line in f:
            if line.startswith(">"):
                a, _, t = line[1:].strip().partition("|")
                acc.append(a)
                tax.append(int(t))
    return acc, tax


def main():
    run_dir = main_args()
    with Stage(run_dir, "reference") as st:
        cfg = st.config["reference"]
        faa = st.repo(cfg["faa"])
        out = st.path("reference", "index.npz")
        acc, tax = headers(faa)
        order = np.argsort(np.array(acc, dtype="S"), kind="stable")
        acc_s = np.array(acc, dtype="S")[order]
        if len(set(acc)) != len(acc):
            raise SystemExit("duplicate accessions in the reference")
        save_npz(out, acc=acc_s, taxid=np.array(tax, dtype=np.int64)[order])
        st.output("index", out)
        taxo = Taxonomy(st.repo(st.config["taxdump"]))
        utax = np.array(sorted(set(tax)), dtype=np.int64)
        ranks = np.array([[taxo.at_rank(int(t), r) for r in RANKS] for t in utax], dtype=np.int64)
        lib = st.path("reference", "library.npz")
        save_npz(lib, taxid=utax, ranks=ranks, rank_names=np.array(RANKS))
        st.output("library", lib)
        if cfg.get("dmnd"):
            db = st.repo(cfg["dmnd"])
            st.extra["database"] = "in place: %s" % db
        else:
            db = st.path("reference", "ref.dmnd")
            subprocess.run([st.config["tools"]["diamond"], "makedb", "--in", faa, "-d", db[:-5], "--quiet"], check=True)
            st.output("database", db)
        entries = [dict(id="reference", files=[M.file_entry(faa)], proteins=len(acc), proteomes=len(utax))]
        man = M.build("reference", "reference", entries, meta=dict(database=os.path.relpath(db, run_dir) if db.startswith(run_dir) else db))
        sha = M.write(man, os.path.join(run_dir, "manifests"))
        mp = st.path("reference", "manifest.json")
        write_json(mp, dict(sha256=sha, dataset_id=M.dataset_id(man), database=db, **man))
        st.output("reference_manifest", mp)
        st.extra.update(proteins=len(acc), proteomes=len(utax))


if __name__ == "__main__":
    main()
