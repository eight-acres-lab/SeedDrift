#!/usr/bin/env python
"""Turn search results into arrays for the placement model.

  library   one row per reference proteome: taxon, node at each rank (domain, phylum, class, order, family,
            genus; 0 = rank not defined in that lineage), donor group
  train     DIAMOND hits of sampled reference proteins against the whole library (place_search.sh):
            the query's own proteome is known, so every hit has a true relation to the query
  job       hits.tsv of a SeedDrift job: genes of one genome whose lineage is known

usage (needs the seeddrift package and its data store; no torch):
  python research/placement/prep.py library --out DIR/library.npz
  python research/placement/prep.py train --library DIR/library.npz --hits DIR/hits_a.tsv DIR/hits_b.tsv --out DIR/train.npz
  python research/placement/prep.py job --library DIR/library.npz --job JOBDIR [--deep-hits DIR/hits_bench_x.tsv] --out DIR/job_x.npz
"""
from __future__ import annotations

import argparse
import array
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

RANKS = ["phylum", "class", "order", "family", "genus"]
DOMAINS = (2, 2157, 2759, 10239)                     # Bacteria, Archaea, Eukaryota, Viruses


def lineage_row(tax, t):
    """Node ids at domain, phylum, class, order, family, genus (0 where the lineage has no such rank)."""
    lin = set(tax.lineage(t))
    return [next((d for d in DOMAINS if d in lin), 0)] + [tax.at_rank(t, r) or 0 for r in RANKS]


def cmd_library(a):
    from seeddrift import config
    from seeddrift.pipeline import get_taxonomy
    tax = get_taxonomy()
    rows = list(csv.DictReader(open(os.path.join(config.REFERENCE, "proteomes.tsv")), delimiter="\t"))
    groups = sorted({r["group"] for r in rows})
    np.savez_compressed(a.out, taxid=np.array([int(r["taxid"]) for r in rows], dtype=np.int64),
                        ranks=np.array([lineage_row(tax, int(r["taxid"])) for r in rows], dtype=np.int64),
                        group=np.array([groups.index(r["group"]) for r in rows], dtype=np.int16),
                        group_names=np.array(groups), n_proteins=np.array([int(r["n_proteins"]) for r in rows], dtype=np.int32))
    lib = np.load(a.out)
    print("proteomes %d; rank defined: %s" % (len(rows), dict(zip(["domain"] + RANKS, (lib["ranks"] != 0).mean(0).round(3)))), file=sys.stderr)


class Hits:
    """Per-query hit lists in flat arrays, hits of a query sorted by bit score (highest first)."""

    def __init__(self):
        self.q_id, self.q_tax, self.q_len, self.off = [], [], [], [0]
        self.p, self.unknown = array.array("i"), 0
        self.f = {k: array.array("f") for k in ("bits", "pid", "qcov", "tcov", "loge", "lr")}
        self._cur, self._rows = None, []

    def _flush(self):
        if self._cur is None:
            return
        self._rows.sort(key=lambda r: -r[1])
        for r in self._rows:
            self.p.append(r[0])
            for k, v in zip(("bits", "pid", "qcov", "tcov", "loge", "lr"), r[1:]):
                self.f[k].append(v)
        self.off.append(len(self.p))
        self._rows = []

    def add(self, q, qtax, qlen, pidx, bits, pid, qcov, tcov, ev, tlen):
        if q != self._cur:
            self._flush()
            self._cur = q
            self.q_id.append(q)
            self.q_tax.append(qtax)
            self.q_len.append(qlen)
        self._rows.append((pidx, bits, pid / 100.0, qcov, tcov, min(200.0, -math.log10(max(ev, 1e-200))), math.log(tlen / qlen)))

    def save(self, path, **extra):
        self._flush()
        np.savez(path, q_id=np.array(self.q_id), q_tax=np.array(self.q_tax, dtype=np.int64), q_len=np.array(self.q_len, dtype=np.int32),
                 off=np.array(self.off, dtype=np.int64), h_p=np.frombuffer(self.p, dtype=np.int32),
                 **{"h_" + k: np.frombuffer(v, dtype=np.float32) for k, v in self.f.items()}, **extra)
        print("queries %d, hits %d, hits to taxa outside the library table %d" % (len(self.q_id), len(self.p), self.unknown), file=sys.stderr)


def read_diamond(H, paths, pidx, query_taxon):
    """DIAMOND tabular output with the 12 columns of place_search.sh; targets are named accession|taxid."""
    for path in paths:
        with open(path) as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                tt = f[1].rpartition("|")[2]
                p = pidx.get(int(tt)) if tt.isdigit() else None
                if p is None:
                    H.unknown += 1
                    continue
                qlen, tlen = int(f[10]), int(f[11])
                H.add(f[0], query_taxon(f[0]), qlen, p, float(f[5]), float(f[2]), (int(f[7]) - int(f[6]) + 1) / qlen,
                      (abs(int(f[9]) - int(f[8])) + 1) / tlen, float(f[4]), tlen)


def cmd_train(a):
    lib = np.load(a.library)
    pidx = {int(t): i for i, t in reversed(list(enumerate(lib["taxid"])))}
    H = Hits()
    read_diamond(H, a.hits, pidx, lambda q: int(q.rpartition("|")[2]))
    H.save(a.out, limit=np.int64(a.limit))


def cmd_job(a):
    """Genes of one genome. The genome's own lineage (from the job summary) is stored so that the relation of
    the genome to every reference proteome is known; gene ids, categories and confidence come from genes.tsv."""
    from seeddrift.pipeline import get_taxonomy
    tax = get_taxonomy()
    lib = np.load(a.library)
    pidx = {int(t): i for i, t in reversed(list(enumerate(lib["taxid"])))}
    summ = json.load(open(os.path.join(a.job, "summary.json")))
    H = Hits()
    if a.deep_hits:                                        # a separate DIAMOND search of the job's proteins with a deep list
        read_diamond(H, [a.deep_hits], pidx, lambda q: 0)
    else:
        with open(os.path.join(a.job, "hits.tsv")) as fh:
            ix = {k: i for i, k in enumerate(fh.readline().rstrip("\n").split("\t"))}
            for line in fh:
                f = line.rstrip("\n").split("\t")
                p = pidx.get(int(f[ix["taxid"]]))
                if p is None:
                    H.unknown += 1
                    continue
                qlen, tlen = int(f[ix["qlen"]]), int(f[ix["tlen"]])
                H.add(f[ix["query"]], 0, qlen, p, float(f[ix["bits"]]), float(f[ix["pident"]]),
                      (int(f[ix["qend"]]) - int(f[ix["qstart"]]) + 1) / qlen, (abs(int(f[ix["tend"]]) - int(f[ix["tstart"]])) + 1) / tlen,
                      float(f[ix["evalue"]]), tlen)
    from seeddrift.single import home_group
    home_t, home_name = home_group(tax, summ["self_taxid"], summ.get("params", {}).get("home_rank", "auto"))
    in_home = np.array([home_t in tax.lineage(int(t)) for t in lib["taxid"]])
    print("home group of the pipeline: %s (%d library proteomes)" % (home_name, in_home.sum()), file=sys.stderr)
    H.save(a.out, genome_ranks=np.array(lineage_row(tax, summ["self_taxid"]), dtype=np.int64), self_taxid=np.int64(summ["self_taxid"]),
           home_member=in_home, limit=np.int64(a.limit if a.deep_hits else 100))   # the pipeline's own search keeps 100 targets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("library"); s.add_argument("--out", required=True)
    s = sub.add_parser("train"); s.add_argument("--library", required=True); s.add_argument("--hits", nargs="+", required=True); s.add_argument("--out", required=True)
    s.add_argument("--limit", type=int, default=1000, help="--max-target-seqs of the search")
    s = sub.add_parser("job"); s.add_argument("--library", required=True); s.add_argument("--job", required=True); s.add_argument("--out", required=True)
    s.add_argument("--deep-hits", default="", help="DIAMOND output of the job's proteins with a deep hit list, instead of the job's own hits.tsv")
    s.add_argument("--limit", type=int, default=1000, help="--max-target-seqs of the deep search")
    a = ap.parse_args()
    dict(library=cmd_library, train=cmd_train, job=cmd_job)[a.cmd](a)


if __name__ == "__main__":
    main()
