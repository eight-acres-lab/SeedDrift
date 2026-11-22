"""Distances between genomes for sampling and reporting (relational line, package W2): shared amino-acid k-mers. Numpy
only; never a model input (the distance tiers choose rows and strata, nothing else).

The k-mers of a genome: every gene of the training arrays translated with the standard code (source_model.CODE; the
arrays hold at most the first codons of a gene up to the source's limit, and a eukaryote at most its first 1,500 genes),
every stretch of k residues inside one gene without a stop or an unknown residue, written base 20 and hashed
(splitmix64). The sketch of a genome: the `size` smallest distinct hashes (bottom-s MinHash), and the number of distinct
k-mers. Between two genomes (as Mash does; Ondov et al. 2016):
  jaccard      the share of the `size` smallest hashes of the union of the two sketches that are in both
  distance     -ln(2 J / (1 + J)) / k, about the share of residues that differ for close genomes; inf without a
               shared hash
  containment  of the genome with fewer k-mers in the other: of the hashes of its sketch at or below the largest hash of
               the other sketch (all of them when the other sketch holds every k-mer of its genome), the share that are
               in the other sketch

  kmer_codes(codons, coff, k)        the base-20 codes of the k-mers (every occurrence)
  sketch(codons, coff, k, size)      (sorted uint64 hashes, number of distinct k-mers)
  compare(a, b, size, n_a, n_b)      (jaccard, shared hashes, containment)
  matrix(sketches, counts, size, processes)   [G, G] jaccard and containment of every pair
  mash(jaccard, k)                   the distance of a jaccard (inf where it is 0)
  tier_of(distance, edges)           tier index of distances: 0 below edges[0], ..., len(edges) at or above the last
                                     edge, len(edges) + 1 without a shared k-mer
"""
from __future__ import annotations

import multiprocessing

import numpy as np

from seeddrift.train import source_model as S

ALPHABET = 20
STOP = 21                                             # source_model.CODE: stop codons, unknown codons and padding


def splitmix64(x):
    """The splitmix64 finaliser of every value (uint64, wrapping)."""
    with np.errstate(over="ignore"):
        z = np.asarray(x).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def kmer_codes(codons, coff, k):
    """int64 codes of every k-mer of the translated genes (a k-mer lies inside one gene and has no stop or unknown
    residue), in the order of the genes."""
    k = int(k)
    if not 1 <= k <= 12:
        raise ValueError("k must be 1..12 (base-20 codes in 64 bits)")
    codons = np.asarray(codons)
    coff = np.asarray(coff, dtype=np.int64)
    n = len(codons)
    if len(coff) < 1 or coff[0] != 0 or coff[-1] != n or (np.diff(coff) < 0).any():
        raise ValueError("the gene offsets do not cover the codons (%d codons, offsets end at %s)" % (n, coff[-1] if len(coff) else None))
    if n < k:
        return np.zeros(0, dtype=np.int64)
    aa = S.CODE[codons.astype(np.int64)]
    ok = (aa >= 1) & (aa <= ALPHABET)
    gene = np.repeat(np.arange(len(coff) - 1), np.diff(coff))
    bad = np.concatenate([[0], np.cumsum(~ok)])
    m = n - k + 1
    valid = (bad[k:] - bad[:m] == 0) & (gene[:m] == gene[k - 1:])
    a = np.where(ok, aa - 1, 0).astype(np.int64)
    code = np.zeros(m, dtype=np.int64)
    for i in range(k):
        code = code * ALPHABET + a[i:i + m]
    return code[valid]


def sketch(codons, coff, k, size):
    """(the `size` smallest distinct hashes of the genome's k-mers, sorted, as uint64; the number of distinct k-mers)."""
    h = np.unique(splitmix64(kmer_codes(codons, coff, k)))
    return h[:int(size)].copy(), int(len(h))


def compare(a, b, size, n_a=None, n_b=None):
    """(jaccard, shared, containment) of two sketches; n_a, n_b: distinct k-mers of the two genomes (default: the
    sketch lengths, i.e. complete sketches)."""
    size = int(size)
    n_a = len(a) if n_a is None else int(n_a)
    n_b = len(b) if n_b is None else int(n_b)
    if not len(a) or not len(b):
        return 0.0, 0, 0.0
    union = np.union1d(a, b)
    s = min(size, len(union))
    top = union[s - 1]
    both = np.intersect1d(a, b, assume_unique=True)
    shared = int((both <= top).sum())
    small, big, n_big = (a, b, n_b) if n_a <= n_b else (b, a, n_a)
    lim = big[-1] if len(big) < n_big else np.iinfo(np.uint64).max      # beyond the other sketch nothing is known
    cand = small[small <= lim]
    cont = float(np.isin(cand, big, assume_unique=True).mean()) if len(cand) else 0.0
    return shared / s, shared, cont


_M = {}


def _row(a):
    sk, counts, size = _M["sketches"], _M["counts"], _M["size"]
    out = np.zeros((2, len(sk)), dtype=np.float64)
    for b in range(a + 1, len(sk)):
        j, _, c = compare(sk[a], sk[b], size, counts[a], counts[b])
        out[0, b], out[1, b] = j, c
    return a, out


def matrix(sketches, counts, size, processes=1):
    """(jaccard [G, G], containment [G, G]) of every pair of sketches (float64, symmetric, 1 on the diagonal)."""
    g = len(sketches)
    jac, con = np.eye(g), np.eye(g)
    _M.clear()
    _M.update(sketches=list(sketches), counts=list(counts), size=int(size))
    try:
        if processes > 1 and g > 2:
            with multiprocessing.get_context("fork").Pool(int(processes)) as pool:
                rows = list(pool.imap_unordered(_row, range(g - 1), chunksize=4))
        else:
            rows = [_row(a) for a in range(g - 1)]
    finally:
        _M.clear()
    for a, out in rows:
        jac[a, a + 1:], con[a, a + 1:] = out[0, a + 1:], out[1, a + 1:]
    iu = np.triu_indices(g, 1)
    jac[(iu[1], iu[0])], con[(iu[1], iu[0])] = jac[iu], con[iu]
    return jac, con


def mash(jaccard, k):
    j = np.asarray(jaccard, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.where(j > 0, -np.log(2 * j / (1 + j)) / int(k), np.inf)


def tier_of(distance, edges):
    d = np.asarray(distance, dtype=np.float64)
    t = np.searchsorted(np.asarray(edges, dtype=np.float64), d, side="right")
    return np.where(np.isfinite(d), t, len(edges) + 1)
