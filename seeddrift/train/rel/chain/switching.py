"""The copying chain with switching: a hidden Markov model over the rows of a table. numpy.

Along the columns of one locus the examined row copies from one source at a time: one of the K other rows, or no row
(state 0, which emits the background distribution of the row). Between two columns the chain keeps its source with
probability 1 - tau and otherwise draws a new one from the prior: `nosrc` for state 0, the rest uniform over the
rows that have a codon in the locus. A row state emits the copying kernel of the row's codon in that column
(baselines.classical.copy_kernel: three per-position error rates) and the background where the row has no codon.
Five numbers: three error rates, tau, nosrc (the share of "other", rho, comes with the kernel).

What a cell is predicted from is a mode of the chain:
  "both"    every visible cell of the locus (forward and backward);
  "past"    the visible cells to its left only (forward);
  ("ahead", k)  the cells to its left and the k columns to its right.
Hidden cells carry no evidence, so the prediction of a hidden cell is p(cell | visible cells), each hidden cell on
its own. With every cell visible the forward pass gives the probability of the whole stretch (log_joint).
"""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel import api
from seeddrift.train.rel.baselines import classical as cl


def source_prior(Y, nosrc, base=4):
    """(N, 1 + K): the prior over the sources. Y (N, K, C) the tokens of the rows in the columns of the locus."""
    nc = cl.sizes(base)[0]
    alive = ((Y >= 1) & (Y <= nc)).any(axis=2)
    k = alive.sum(axis=1, keepdims=True)
    rows = np.where(alive, (1.0 - nosrc) / np.maximum(k, 1), 0.0)
    return np.concatenate([np.where(k > 0, float(nosrc), 1.0), rows], axis=1)


def emissions(X, Y, q, kernel, base=4):
    """(N, C, 1 + K): the probability of the examined row's token in every column under every source; 1 where the
    token is hidden or the column is past the end of the row.
    X (N, C) tokens of the examined row (a residue, or anything else = no evidence); Y (N, K, C); q (N, V)."""
    nc, V = cl.sizes(base)
    seen = (X >= 1) & (X <= V)
    x = np.where(seen, X, 1).astype(np.int64) - 1
    back = np.take_along_axis(q, x, axis=1)                                   # (N, C)
    usable = (Y >= 1) & (Y <= nc)
    y = np.where(usable, Y, 1).astype(np.int64) - 1
    rows = np.where(usable, kernel[y, x[:, None, :]], back[:, None, :])        # (N, K, C)
    e = np.concatenate([back[:, None, :], rows], axis=1)
    return np.where(seen[:, None, :], e, 1.0).transpose(0, 2, 1)


def forward(e, prior, tau, valid, per_column=False):
    """Scaled forward pass. Returns before (N, C, S): the distribution of the source of column c given the visible
    cells to its left; after (N, C, S): the same with column c itself; logz (N,): log probability of the visible cells
    (with per_column, (N, C): the log probability of the cell of every column given the visible cells to its left).
    valid (N, C): the columns of the row (a prefix)."""
    N, C, S = e.shape
    before, after = np.empty((N, C, S)), np.empty((N, C, S))
    logz = np.zeros((N, C))
    cur = prior
    for c in range(C):
        v = valid[:, c][:, None]
        pred = prior if c == 0 else np.where(v, (1.0 - tau) * cur + tau * prior, cur)
        w = np.where(v, pred * e[:, c], pred)
        z = w.sum(axis=1, keepdims=True)
        cur = w / z
        logz[:, c] = np.log(z[:, 0])
        before[:, c], after[:, c] = pred, cur
    return before, after, (logz if per_column else logz.sum(axis=1))


def backward(e, prior, tau, valid, ahead=None):
    """(N, C, S), scaled: the probability of the visible cells to the right of column c given its source (of the next
    `ahead` columns only when given)."""
    N, C, S = e.shape
    out = np.ones((N, C, S))
    if ahead is None:
        nxt = np.ones((N, S))
        for c in range(C - 2, -1, -1):
            v = valid[:, c + 1][:, None]
            m = e[:, c + 1] * nxt
            m = np.where(v, (1.0 - tau) * m + tau * (prior * m).sum(axis=1, keepdims=True), nxt)
            nxt = m / m.sum(axis=1, keepdims=True) * S
            out[:, c] = nxt
        return out
    for c in range(C - 1):
        nxt = np.ones((N, S))
        for d in range(min(C - 1, c + int(ahead)), c, -1):
            v = valid[:, d][:, None]
            m = e[:, d] * nxt
            m = np.where(v, (1.0 - tau) * m + tau * (prior * m).sum(axis=1, keepdims=True), nxt)
            nxt = m / m.sum(axis=1, keepdims=True) * S
        out[:, c] = nxt
    return out


def source_posterior(e, prior, tau, valid, mode="both"):
    """(N, C, S): the distribution of the source of every column given the visible cells the mode allows, the cell of
    the column itself left out."""
    before, _, _ = forward(e, prior, tau, valid)
    if mode == "past":
        return before
    if mode == "both":
        g = before * backward(e, prior, tau, valid)
    elif isinstance(mode, tuple) and mode[0] == "ahead":
        g = before * backward(e, prior, tau, valid, ahead=mode[1])
    else:
        raise ValueError("unknown mode %r" % (mode,))
    return g / g.sum(axis=2, keepdims=True)


def predict_cells(gamma, Yc, qc, kernel, base=4):
    """(cells, V): the distribution of a cell from the distribution of its source.
    gamma (cells, 1 + K); Yc (cells, K) the rows' tokens in the cell's column; qc (cells, V) the background."""
    nc = cl.sizes(base)[0]
    usable = (Yc >= 1) & (Yc <= nc)
    y = np.where(usable, Yc, 1).astype(np.int64) - 1
    w = gamma[:, 1:] * usable
    p = (1.0 - w.sum(axis=1, keepdims=True)) * qc
    for j in range(Yc.shape[1]):                                              # row by row: (cells, K, V) at once is large
        p += w[:, j:j + 1] * kernel[y[:, j]]
    return p


def truth_probability(gamma, Yc, qc_truth, kernel, truth, base=4):
    """(cells,): predict_cells evaluated at the true residue only (for fitting)."""
    nc = cl.sizes(base)[0]
    usable = (Yc >= 1) & (Yc <= nc)
    y = np.where(usable, Yc, 1).astype(np.int64) - 1
    w = gamma[:, 1:] * usable
    return (w * kernel[y, (truth - 1)[:, None]]).sum(axis=1) + (1.0 - w.sum(axis=1)) * qc_truth


def log_joint(X, Y, q, kernel, tau, nosrc, base=4, per_column=False):
    """(N,): log probability of the examined row's residues X (N, C) as one stretch, every one of them visible. With
    per_column, (N, C): its chain-rule terms, log p(residue of column c | the residues to its left)."""
    V = cl.sizes(base)[1]
    valid = (X >= 1) & (X <= V)
    e = emissions(X, Y, q, kernel, base)
    return forward(e, source_prior(Y, nosrc, base), tau, valid, per_column)[2]


def locus(view, block_start, block_stop):
    """X (N, C), Y (N, K, C) of one block of a masked view; valid (N, C): the columns the examined row has."""
    X = view[:, 0, block_start:block_stop]
    Y = view[:, 1:, block_start:block_stop]
    valid = (X != api.GAP) & (X != api.PAD)
    return X, Y, valid
