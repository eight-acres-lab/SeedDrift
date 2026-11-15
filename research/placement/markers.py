#!/usr/bin/env python
"""Queries that belong to marker families, chosen by a rule that does not use this tool's searches:
a Pfam family whose name starts with one of PREFIXES matches the protein (sequence E-value <= 1e-10 for a
fixed search space of Z sequences).
These proteins are the clean natives of the transfer trials (README.md).

usage (needs pyhmmer):  python research/placement/markers.py --pfam PFAM_DIR --fasta queries_a.faa [...] --out markers.txt
"""
from __future__ import annotations

import argparse
import os
import sys

import pyhmmer

PREFIXES = ("Ribosomal_", "tRNA-synt_", "RNA_pol_", "EFG_", "GTP_EFTU")
STRICT = ("Ribosomal_", "RNA_pol_")       # sensitivity check: without the families known for transfer (synthetases, EF-G/EF-Tu relatives)
MAX_E = 1e-10
Z = 100000                                # fixed search-space size, so the E-value cut does not depend on the size of the query file


def _s(x):
    return x.decode() if isinstance(x, bytes) else (x or "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pfam", required=True)
    ap.add_argument("--fasta", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--strict", action="store_true", help="ribosomal-protein and RNA-polymerase families only")
    a = ap.parse_args()
    with pyhmmer.plan7.HMMFile(os.path.join(a.pfam, "Pfam-A.hmm")) as hf:
        hmms = [h for h in hf if _s(h.name).startswith(STRICT if a.strict else PREFIXES)]
    print("marker families: %d" % len(hmms), file=sys.stderr)
    abc = pyhmmer.easel.Alphabet.amino()
    found, total = set(), 0
    for path in a.fasta:
        with pyhmmer.easel.SequenceFile(path, digital=True, alphabet=abc) as sf:
            block = sf.read_block()
        total += len(block)
        for top in pyhmmer.hmmsearch(hmms, block, cpus=a.threads, E=MAX_E, Z=Z):
            for hit in top:
                if hit.evalue <= MAX_E:
                    found.add(_s(hit.name))
    with open(a.out, "w") as fo:
        fo.write("\n".join(sorted(found)) + "\n")
    print("marker proteins: %d of %d" % (len(found), total), file=sys.stderr)


if __name__ == "__main__":
    main()
