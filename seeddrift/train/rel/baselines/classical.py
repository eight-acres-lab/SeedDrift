"""Classical baselines on the tables: the lines a model has to get under. numpy only.

Every baseline reads the masked view of a batch (evalh.layout.masked_view: nothing of the hidden residues is in it)
and returns, for every scored cell, a normalised distribution over the target tokens 1..V (V = the codons and
"other"). All fitted numbers come from training tables (fit.py).

  uniform               1 / V
  global_freq           the codon frequencies of the training tables
  own_freq              the frequencies of the visible residues of the examined row, shrunk to global_freq
                        (uses no relative: the floor a model without relatives starts from)
  row_average           the residues of the other rows in the scored column, shrunk to own_freq
  copy_nearest          copy from the nearest row that has a codon in the column; per codon position a fitted
                        probability of a different base; own_freq when no row has a codon there.
                        Nearest = the highest share of identical residues with the examined row over the visible
                        columns of the target block; with no visible column there (the whole gene hidden), over the
                        other blocks. Ties: more compared columns, then the lower row.
  copy_nearest_binned   the same, the three probabilities fitted per bin of that share (a stronger line: one set of
                        error rates cannot be right for a near and a far source at once)

The codon alphabet is `base`**3 codons in base-`base` digit order plus "other"; base 4 is the real one (ids 1..64 as
in the contract), smaller bases are for exhaustive tests.
"""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel import api
from seeddrift.train.rel.evalh import conditions as cond

N_SHARE_BINS = 10                       # bins of the identity share of the copied row; one more for "no compared column"
NAMES = ("uniform", "global_freq", "own_freq", "row_average", "copy_nearest", "copy_nearest_binned")
USES_RELATIVES = ("row_average", "copy_nearest", "copy_nearest_binned")
EPS_FLOOR = 1e-4


def sizes(base=4):
    """(number of codons, number of target tokens)."""
    nc = base ** 3
    return nc, nc + 1


def codon_digits(base=4):
    """(base**3, 3): the three digits of every codon, id - 1 = b1 * base**2 + b2 * base + b3."""
    c = np.arange(base ** 3)
    return np.stack([c // (base * base), (c // base) % base, c % base], axis=1)


def copy_kernel(eps, rho, base=4):
    """(codons, V): the distribution of the target token given the codon it is copied from. Every codon position
    keeps its base with probability 1 - eps[k] and takes each other base with eps[k] / (base - 1); "other" has rho."""
    eps = np.asarray(eps, dtype=np.float64)
    k = None
    for e in eps:
        m = np.full((base, base), e / (base - 1))
        np.fill_diagonal(m, 1.0 - e)
        k = m if k is None else np.kron(k, m)
    return np.concatenate([(1.0 - rho) * k, np.full((k.shape[0], 1), float(rho))], axis=1)


def own_counts(view, base=4):
    """(N, V) counts of the visible residues of row 0 over all its blocks."""
    V = sizes(base)[1]
    row = view[:, 0, :].astype(np.int64)
    ok = (row >= 1) & (row <= V)
    t = np.nonzero(ok)[0]
    return np.bincount(t * V + row[ok] - 1, minlength=view.shape[0] * V).reshape(-1, V).astype(np.float64)


def shrunk(counts, prior, strength):
    """(counts + strength * prior) / (total + strength), row by row."""
    counts = np.asarray(counts, dtype=np.float64)
    return (counts + strength * prior) / (counts.sum(axis=1, keepdims=True) + strength)


def column_tokens(view, cells):
    """(C, R - 1): the tokens of rows 1.. in the column of every scored cell."""
    t, s = cells
    return view[t, 1:, s]


def column_counts(Y, base=4):
    """(C, V) counts of the residues among the tokens Y (C, rows)."""
    V = sizes(base)[1]
    ok = (Y >= 1) & (Y <= V)
    c = np.nonzero(ok)[0]
    return np.bincount(c * V + Y[ok].astype(np.int64) - 1, minlength=Y.shape[0] * V).reshape(-1, V).astype(np.float64)


def row_shares(view, base=4):
    """For every table and every row 1..: (identical, compared) residues with row 0 over the reference columns: the
    target block where row 0 has a visible residue there, the other blocks otherwise. Hidden cells never count."""
    V = sizes(base)[1]
    a, b = view[:, :1, :], view[:, 1:, :]
    in_target = cond.BLOCK_OF_SLOT == cond.TARGET_BLOCK
    a_ok = (a >= 1) & (a <= V)
    use_target = (a_ok[:, 0, :] & in_target[None, :]).any(axis=1)
    region = np.where(use_target[:, None], in_target[None, :], ~in_target[None, :])
    both = region[:, None, :] & a_ok & (b >= 1) & (b <= V)
    return (both & (a == b)).sum(axis=2), both.sum(axis=2)


def nearest_source(view, cells, base=4):
    """For every scored cell the codon to copy: of the rows ranked by share, the first with a codon in the column.
    Returns has (C,) bool, y (C,) the codon (1..codons; 1 where has is False), share (C,) of that row (-1 = no compared
    column), row (C,) its index (0 where has is False)."""
    nc = sizes(base)[0]
    same, compared = row_shares(view, base)
    share = np.where(compared > 0, same / np.maximum(compared, 1), -1.0)
    rows = np.tile(np.arange(share.shape[1]), (share.shape[0], 1))
    order = np.lexsort((rows, -compared, -share), axis=1)
    t = cells[0]
    Y = np.take_along_axis(column_tokens(view, cells), order[t], axis=1)
    usable = (Y >= 1) & (Y <= nc)
    if Y.shape[1] == 0:                                                       # no relatives given
        none = np.zeros(Y.shape[0], dtype=np.int64)
        return none.astype(bool), none + 1, none - 1.0, none
    has = usable.any(axis=1)
    first = usable.argmax(axis=1)
    c = np.arange(Y.shape[0])
    row = order[t, first]
    return has, np.where(has, Y[c, first], 1).astype(np.int64), np.where(has, share[t, row], -1.0), np.where(has, row + 1, 0)


def share_bin(share):
    """Index of the bin of an identity share: ten equal bins, and N_SHARE_BINS for 'no compared column' (-1)."""
    k = np.minimum(np.floor(np.maximum(share, 0.0) * N_SHARE_BINS + 1e-9).astype(np.int64), N_SHARE_BINS - 1)
    return np.where(share < 0, N_SHARE_BINS, k)


class Batch:
    """What the baselines share on one batch under one condition."""

    def __init__(self, view, cells, base=4):
        self.view, self.cells, self.base = view, cells, base
        self.counts = own_counts(view, base)
        self.Y = column_tokens(view, cells)
        self._source = None

    @property
    def source(self):
        if self._source is None:
            self._source = nearest_source(self.view, self.cells, self.base)
        return self._source

    def own(self, params):
        """(C, V): own_freq of the table of every scored cell."""
        return shrunk(self.counts, np.asarray(params["prior"]), params["alpha"])[self.cells[0]]


def predict(name, batch, params):
    """(C, V) the distribution of every scored cell under baseline `name`."""
    base = batch.base
    V = sizes(base)[1]
    C = batch.cells[0].shape[0]
    if name == "uniform":
        return np.full((C, V), 1.0 / V)
    if name == "global_freq":
        return np.tile(np.asarray(params["prior"], dtype=np.float64), (C, 1))
    q = batch.own(params)
    if name == "own_freq":
        return q
    if name == "row_average":
        m = column_counts(batch.Y, base)
        return (m + params["beta"] * q) / (m.sum(axis=1, keepdims=True) + params["beta"])
    if name in ("copy_nearest", "copy_nearest_binned"):
        has, y, share, _ = batch.source
        if name == "copy_nearest":
            p = copy_kernel(params["eps"], params["rho"], base)[y - 1]
        else:
            k = np.stack([copy_kernel(e, params["rho"], base) for e in params["eps_binned"]])
            p = k[share_bin(share), y - 1]
        return np.where(has[:, None], p, q)
    raise KeyError("unknown baseline %r" % name)


def bits_of(P, truth):
    """-log2 of the probability of the true residue of every cell."""
    return -np.log2(P[np.arange(truth.shape[0]), truth - 1])


def check_normalised(P, tol=1e-9):
    """Every row a distribution over the target tokens: no negative entry, the sum is one."""
    if P.size and (P.min() < 0 or np.abs(P.sum(axis=1) - 1.0).max() > tol):
        raise ValueError("a baseline returned something that is not a distribution")
    if P.shape[1] > api.TARGET_MAX:
        raise ValueError("a baseline put mass outside the target tokens")
