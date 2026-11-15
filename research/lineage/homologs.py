#!/usr/bin/env python
"""For every query protein, a few homologs from OTHER genomes of the reference library, as sequences.

They serve as hard negatives (same protein family, different source) and as natural "swap" tests: an ortholog
from another genome put in the place of the native gene. Homologs are picked by score rank only (ranks 1, 4,
20 and 100 among the best hits of distinct proteomes), not by taxonomy.

Reads the DIAMOND output of research/placement/search.sh (queries and targets are named accession|taxid).

usage (needs the seeddrift package for the sequence index):
  python research/lineage/homologs.py --hits hits_a.tsv hits_b.tsv --reference REF/reference.faa --out DIR/homologs
writes DIR/homologs.faa (ids accession|taxid) and DIR/homologs.tsv (query, homolog, rank, bits, identity)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
RANKS = (1, 4, 20, 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hits", nargs="+", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from seeddrift.tools import SeqIndex
    pairs, want = [], set()
    cur, seen, n = None, set(), 0

    for path in a.hits:
        with open(path) as fh:
            for line in fh:
                f = line.split("\t", 6)
                q, t = f[0], f[1]
                if q != cur:
                    cur, seen, n = q, {q.rpartition("|")[2]}, 0        # the query's own proteome never counts
                tx = t.rpartition("|")[2]
                if tx in seen:
                    continue
                seen.add(tx)
                n += 1
                if n in RANKS:
                    pairs.append((q, t, n, f[5], f[2]))
                    want.add(t)
    print("queries with homologs: %d; pairs %d; distinct homologs %d" % (len({p[0] for p in pairs}), len(pairs), len(want)), file=sys.stderr)
    idx = SeqIndex(a.reference)
    accs = sorted({t.rpartition("|")[0] for t in want})
    seqs = idx.get(accs)
    with open(a.out + ".faa", "w") as fo:
        for t in sorted(want):
            s = seqs.get(t.rpartition("|")[0])
            if s:
                fo.write(">%s\n%s\n" % (t, s))
    with open(a.out + ".tsv", "w") as fo:
        fo.write("query\thomolog\trank\tbits\tidentity\n")
        for p in pairs:
            fo.write("%s\t%s\t%d\t%s\t%s\n" % p)
    print("sequences found: %d of %d" % (len(seqs), len(accs)), file=sys.stderr)


if __name__ == "__main__":
    main()
