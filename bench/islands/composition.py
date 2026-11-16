#!/usr/bin/env python
"""Composition scores of every gene of a chromosome, from its DNA alone (no similarity search, no reference library).

Each score says how unlike the rest of the chromosome a gene's coding sequence is; higher = more atypical.

  gc3      |robust z| of the GC content at third codon positions
  codon    synonymous codon usage: excess cross-entropy of the gene's codons under the chromosome's usage,
           standardised by its model variance (about N(0, 1) for a typical gene of any length)
  kmer     the same for a third-order Markov chain with one table per codon position
  *_nb     the mean of a score over the gene and its 5 neighbours on each side (islands are contiguous)
  short    minus the log length (control: a score that knows nothing about composition)

Writes <out>/scores_<accession>.tsv (gene ids as the pipeline names them).

usage (repo root; needs the seeddrift package and Biopython):
  python bench/islands/composition.py --genbank DIR --accessions NC_... --out DIR
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

NT = {c: i for i, c in enumerate("ACGT")}
BASES = "TCAG"
AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"      # codon table 11 = standard for the sense codons
CODON_AA = {}
for _i, _a in enumerate(AA):
    _c = BASES[_i // 16] + BASES[_i // 4 % 4] + BASES[_i % 4]
    CODON_AA[NT[_c[0]] * 16 + NT[_c[1]] * 4 + NT[_c[2]]] = _a
AA_OF = np.array([ord(CODON_AA[i]) for i in range(64)])
NB = 5


def encode(nt):
    a = np.frombuffer(nt.upper().encode(), dtype=np.uint8)
    out = np.full(len(a), -1, dtype=np.int64)
    for c, i in NT.items():
        out[a == ord(c)] = i
    return out


def robust_z(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / (mad if mad > 0 else 1.0)


def excess(counts):
    """For a table of counts [context, outcome]: per cell, -log p minus the entropy of its context; per context, the variance."""
    p = (counts + 1.0) / (counts + 1.0).sum(1, keepdims=True)
    lp = -np.log(p)
    h = (p * lp).sum(1, keepdims=True)
    return lp - h, (p * (lp - h) ** 2).sum(1)


def genome_scores(cds):
    """cds: list of coding sequences in chromosome order. Returns a dict of arrays, one value per gene."""
    seqs = [encode(s) for s in cds]
    cod, ctx, nxt = [], [], []
    for x in seqs:
        n = len(x) // 3
        c = x[:n * 3].reshape(n, 3)[1:-1] if n > 2 else np.zeros((0, 3), dtype=np.int64)      # without the start and stop codons
        ok = (c >= 0).all(1)
        cod.append((c[ok] * [16, 4, 1]).sum(1))
        i = np.arange(3, len(x))
        w = np.stack([x[i - 3], x[i - 2], x[i - 1], x[i]], 1)
        ok = (w >= 0).all(1)
        ctx.append(((w[ok, 0] * 16 + w[ok, 1] * 4 + w[ok, 2]) * 3 + i[ok] % 3))
        nxt.append(w[ok, 3])
    # synonymous codon usage: outcome = codon, context = amino acid
    aas = sorted(set(AA_OF.tolist()) - {ord("*")})
    aa_ix = {a: k for k, a in enumerate(aas)}
    cnt = np.zeros((len(aas), 64))
    allc = np.concatenate(cod) if cod else np.zeros(0, dtype=np.int64)
    allc = allc[AA_OF[allc] != ord("*")]
    np.add.at(cnt, (np.array([aa_ix[a] for a in AA_OF[allc]]), allc), 1)
    mask = np.zeros_like(cnt, dtype=bool)
    for c in range(64):
        if AA_OF[c] != ord("*"):
            mask[aa_ix[AA_OF[c]], c] = True
    p = np.where(mask, cnt + 1.0, 0.0)
    p = p / p.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(mask, -np.log(p), 0.0)
    h = (p * lp).sum(1)
    var = (p * np.where(mask, (lp - h[:, None]) ** 2, 0.0)).sum(1)
    term_c = np.zeros(64); var_c = np.zeros(64)
    for c in range(64):
        if AA_OF[c] != ord("*"):
            k = aa_ix[AA_OF[c]]
            term_c[c], var_c[c] = lp[k, c] - h[k], var[k]
    # third-order chain, one table per codon position
    kc = np.zeros((64 * 3, 4))
    np.add.at(kc, (np.concatenate(ctx), np.concatenate(nxt)), 1)
    term_k, var_k = excess(kc)
    out = {k: np.full(len(cds), np.nan) for k in ("gc3", "codon", "kmer", "short")}
    gc3 = np.full(len(cds), np.nan)
    for g, x in enumerate(seqs):
        n = len(x) // 3
        if n >= 30:
            third = x[:n * 3].reshape(n, 3)[1:-1, 2]
            third = third[third >= 0]
            if len(third):
                gc3[g] = np.mean((third == 1) | (third == 2))
            c = cod[g][AA_OF[cod[g]] != ord("*")]
            if len(c):
                out["codon"][g] = term_c[c].sum() / np.sqrt(var_c[c].sum())
            if len(ctx[g]):
                out["kmer"][g] = term_k[ctx[g], nxt[g]].sum() / np.sqrt(var_k[ctx[g]].sum())
        out["short"][g] = -np.log(max(len(x), 1))
    out["gc3"] = np.abs(robust_z(gc3))
    n = len(cds)
    for k in ("gc3", "codon", "kmer"):
        v = np.where(np.isfinite(out[k]), out[k], np.nanmedian(out[k]))
        idx = (np.arange(n)[:, None] + np.arange(-NB, NB + 1)[None, :]) % n          # circular chromosome
        out[k + "_nb"] = v[idx].mean(1)
        out[k] = v
    return out


def chromosome(gb_path):
    """Pipeline gene ids and coding sequences of one GenBank record, in chromosome order."""
    from seeddrift.inputs import from_genbank
    from seeddrift.pipeline import clean_proteins
    g = from_genbank(open(gb_path).read())
    _, ren = clean_proteins(g.proteins)
    ids = sorted(ren, key=lambda k: (g.coords[k][0], g.coords[k][1]))
    return [ren[k] for k in ids], [g.cds.get(k, "") for k in ids]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genbank", required=True, help="folder with <accession>.gb")
    ap.add_argument("--accessions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for acc in a.accessions:
        ids, cds = chromosome(os.path.join(a.genbank, acc + ".gb"))
        sc = genome_scores(cds)
        cols = ["gc3", "codon", "kmer", "gc3_nb", "codon_nb", "kmer_nb", "short"]
        with open(os.path.join(a.out, "scores_%s.tsv" % acc), "w") as fo:
            fo.write("gene_id\t" + "\t".join(cols) + "\n")
            for i, gid in enumerate(ids):
                fo.write(gid + "\t" + "\t".join("%.4f" % sc[c][i] for c in cols) + "\n")
        print("%s %d genes" % (acc, len(ids)), flush=True)


if __name__ == "__main__":
    main()
