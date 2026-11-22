"""Fit the numbers of the classical baselines. Training tables only: the caller passes training tables, and the
stage that calls this refuses a table set that is not marked as training.

Fitted, by maximum likelihood of the hidden residues under one kind of mask (scattered or whole gene):
  prior       the frequencies of the residues of row 0 over the fit tables (add-one)
  alpha       strength of the shrinkage of own_freq towards prior (grid)
  beta        strength of the shrinkage of row_average towards own_freq (grid, given alpha)
  eps         per codon position, the share of copied codons whose base differs (closed form)
  eps_binned  the same per bin of the identity share of the copied row; a bin with fewer than MIN_BIN_CELLS cells
              takes the pooled eps
  rho         the share of "other" among the residues that had a codon to copy from
"""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel.baselines import classical as cl
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import layout

GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 4096.0, 16384.0)
MIN_BIN_CELLS = 2000
CHUNK = 256


def chunks(n, size=CHUNK):
    return [(a, min(a + size, n)) for a in range(0, n, size)]


def take(tables, a, b):
    return {k: np.asarray(v[a:b]) for k, v in tables.items()}


def batches(tables, table_ids, condition, seed, base=4, chunk=CHUNK):
    """Yields (first table, last table, materialised batch, Batch, truth) over the tables in order."""
    table_ids = np.asarray(table_ids)
    for a, b in chunks(table_ids.shape[0], chunk):
        m = cond.materialise(take(tables, a, b), table_ids[a:b], condition, seed)
        view, cells, truth = layout.masked_view(m)
        yield a, b, m, cl.Batch(view, cells, base), truth


def fit(tables, table_ids, mask, seed, base=4, chunk=CHUNK):
    """The numbers of every baseline for one kind of mask ('scattered' or 'whole_gene'), from training tables."""
    nc, V = cl.sizes(base)
    digits = cl.codon_digits(base)
    table_ids = np.asarray(table_ids)
    counts = np.ones(V)
    for a, b in chunks(table_ids.shape[0], chunk):                       # the prior: every residue of row 0, hidden or not
        row = np.asarray(tables["tok"][a:b])[:, 0, :].astype(np.int64)
        row = row[(row >= 1) & (row <= V)]
        counts += np.bincount(row - 1, minlength=V)
    prior = counts / counts.sum()
    ll_alpha = np.zeros(len(GRID))
    n_cells = 0
    for _, _, _, bt, truth in batches(tables, table_ids, mask, seed, base, chunk):
        c = bt.counts[bt.cells[0]]
        tot = c.sum(axis=1)
        own = c[np.arange(truth.shape[0]), truth - 1]
        for g, a in enumerate(GRID):
            ll_alpha[g] += np.log((own + a * prior[truth - 1]) / (tot + a)).sum()
        n_cells += truth.shape[0]
    if n_cells == 0:
        raise ValueError("no scored cell in the fit tables")
    params = dict(prior=prior.tolist(), alpha=float(GRID[int(ll_alpha.argmax())]))
    ll_beta = np.zeros(len(GRID))
    mis = np.zeros((cl.N_SHARE_BINS + 1, 3))
    tot_bin = np.zeros(cl.N_SHARE_BINS + 1)
    n_other, n_source = 0, 0
    for _, _, _, bt, truth in batches(tables, table_ids, mask, seed, base, chunk):
        q = bt.own(params)
        m = cl.column_counts(bt.Y, base)
        M = m.sum(axis=1)
        c = np.arange(truth.shape[0])
        for g, be in enumerate(GRID):
            ll_beta[g] += np.log((m[c, truth - 1] + be * q[c, truth - 1]) / (M + be)).sum()
        has, y, share, _ = bt.source
        n_source += int(has.sum())
        n_other += int((has & (truth > nc)).sum())
        ok = has & (truth <= nc)
        k = cl.share_bin(share[ok])
        d = digits[truth[ok] - 1] != digits[y[ok] - 1]
        for p in range(3):
            mis[:, p] += np.bincount(k, weights=d[:, p], minlength=cl.N_SHARE_BINS + 1)
        tot_bin += np.bincount(k, minlength=cl.N_SHARE_BINS + 1)

    def rate(m_, t_):
        return np.clip((m_ + 0.5) / (t_ + 1.0), cl.EPS_FLOOR, 1.0 - cl.EPS_FLOOR)

    eps = rate(mis.sum(axis=0), tot_bin.sum())
    binned = np.where((tot_bin >= MIN_BIN_CELLS)[:, None], rate(mis, tot_bin[:, None]), eps[None, :])
    params.update(
        beta=float(GRID[int(ll_beta.argmax())]),
        eps=eps.tolist(),
        eps_binned=binned.tolist(),
        rho=float(np.clip((n_other + 0.5) / (n_source + 1.0), 1e-6, 0.5)),
        fit=dict(mask=mask, seed=int(seed), tables=int(table_ids.shape[0]), cells=int(n_cells), cells_with_source=int(n_source),
                 cells_per_share_bin=tot_bin.astype(int).tolist(), alpha_grid_at_edge=bool(ll_alpha.argmax() in (0, len(GRID) - 1)),
                 beta_grid_at_edge=bool(ll_beta.argmax() in (0, len(GRID) - 1))),
    )
    return params


N_PARAMS = dict(uniform=0, global_freq=64, own_freq=65, row_average=66, copy_nearest=69, copy_nearest_binned=66 + 3 * (cl.N_SHARE_BINS + 1))
