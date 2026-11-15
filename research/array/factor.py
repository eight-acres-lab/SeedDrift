#!/usr/bin/env python
"""A training-free baseline for the same idea: vertical descent as a rank-one structure.

For a native gene, similarity to the member in proteome h follows (rate of the gene) x (distance of the genome
to h). So the gene's own distances, across the proteomes it hits, should order those proteomes as the genome's
profile does. No taxonomy: the genome profile comes from the genome's own genes (build.py profile).

Scores per gene (higher = more foreign), over all proteomes the gene hits except the genome's own genus:
  own    minus the rank correlation between the gene's distances and the genome's distances
  home   best rank correlation with the profile of any other profiled genome, minus the genome's own
  top    how far, in the genome's ordering of all proteomes, the gene's five best hits lie (median percentile)

usage:
  python research/array/factor.py --columns DIR/cols_x.npz --profiles DIR/profiles.npz --library LIB --job JOB.npz --out DIR/factor/scores_x.tsv
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from build import delta_of


def ranks(a):
    """Average ranks along the last axis."""
    order = np.argsort(a, axis=-1, kind="stable")
    r = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(r, order, np.arange(a.shape[-1], dtype=np.float64)[None, :].repeat(a.shape[0], 0), -1)
    s = np.take_along_axis(a, order, -1)
    for i in range(a.shape[0]):                       # ties (the imputed far value) share their mean rank
        v, inv, cnt = np.unique(s[i], return_inverse=True, return_counts=True)
        if len(v) < a.shape[-1]:
            start = np.cumsum(cnt) - cnt
            r[i, order[i]] = (start + (cnt - 1) / 2.0)[inv]
    return r


def corr(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return np.where(den > 0, (a * b).sum(-1) / np.where(den > 0, den, 1), 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--columns", required=True); ap.add_argument("--profiles", required=True)
    ap.add_argument("--library", required=True); ap.add_argument("--job", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-hits", type=int, default=8)
    a = ap.parse_args()
    z, prof, lib = np.load(a.columns), np.load(a.profiles), np.load(a.library)
    names = [str(x) for x in prof["genome"]]
    g = names.index(str(z["gene_genome"][0]))
    genus = int(np.load(a.job)["genome_ranks"][5])
    hidden = (lib["ranks"][:, 5] == genus) if genus else np.zeros(len(lib["taxid"]), dtype=bool)
    D = prof["D"].astype(np.float64)
    far = np.nanmax(D, axis=1, keepdims=True) * 1.5            # a proteome the genome has no profile for is far
    Dimp = np.where(np.isfinite(D), D, far)
    others = np.array([i for i in range(len(names)) if i != g and not (prof["own"][i] >= 0 and hidden[prof["own"][i]])])
    fin = np.flatnonzero(np.isfinite(D[g]) & ~hidden)
    pct = np.ones(D.shape[1])
    pct[fin[np.argsort(D[g][fin])]] = np.arange(len(fin)) / max(1, len(fin))
    off, p, bits, glen = z["off"], z["p"], z["bits"], z["gene_len"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    n = 0
    with open(a.out, "w") as fh:
        fh.write("gene_id\town\thome\ttop\n")
        for i, gid in enumerate(z["gene_id"]):
            sl = slice(off[i], off[i + 1])
            keep = ~hidden[p[sl]]
            h, d = p[sl][keep], delta_of(bits[sl][keep], glen[i])
            if len(h) == 0:
                continue
            top = float(np.median(pct[h[np.argsort(d, kind="stable")[:5]]]))
            if len(h) >= a.min_hits:
                rd = ranks(d[None, :])
                own = float(corr(rd, ranks(Dimp[g][h][None, :]))[0])
                best = float(corr(rd, ranks(Dimp[others][:, h])).max())
                fh.write("%s\t%.4f\t%.4f\t%.4f\n" % (gid, -own, best - own, top))
            else:
                fh.write("%s\t-9\t-9\t%.4f\n" % (gid, top))      # too few hits to compare orderings: ranks last
            n += 1
    print("scored %d genes of %s (own genus hidden: %d proteomes; other profiles %d)" % (n, names[g], int(hidden.sum()), len(others)))


if __name__ == "__main__":
    main()
