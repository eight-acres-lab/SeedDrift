"""From per-cell bits to result rows: the strata and the averaging every arm and every baseline share.

- bits per codon of a set of cells = the sum of -log2 p(residue) over the cells / the number of cells;
- per group first, then the plain mean of the groups that have cells (12 groups in a real table set);
- strata: the identity of the nearest row at the target locus (the largest stored identity of rows 1.., ten equal
  bins and "none" when no row has a cell there). It is a reporting key only and is never given to a model.
"""
from __future__ import annotations

import csv

import numpy as np

from seeddrift.train.rel import api

VERSION = 1
N_IDENTITY_BINS = 10
ALL = "all"
NONE = "nearest_identity=none"
STRATA = tuple("nearest_identity=%.1f-%.1f" % (k / 10.0, (k + 1) / 10.0) for k in range(N_IDENTITY_BINS)) + (NONE,)


def nearest_identity(identity):
    """(N,) the largest identity of rows 1.. at the target locus; nan when no row has a cell there."""
    ident = np.asarray(identity, dtype=np.float64)[:, 1:]
    if ident.shape[1] == 0:
        return np.full(ident.shape[0], np.nan)
    has = ~np.isnan(ident)
    best = np.where(has, ident, -1.0).max(axis=1)
    return np.where(has.any(axis=1), best, np.nan)


def identity_stratum(identity):
    """(N,) index into STRATA for every table, from the stored (N, R) identity of every row with row 0."""
    near = nearest_identity(identity)
    k = np.minimum(np.floor(np.nan_to_num(near, nan=0.0) * N_IDENTITY_BINS + 1e-9).astype(np.int64), N_IDENTITY_BINS - 1)
    return np.where(np.isnan(near), N_IDENTITY_BINS, np.maximum(k, 0))


def aggregate(bits, cell_table, group, stratum, n_groups=None):
    """bits (C,) of the scored cells; cell_table (C,) the table of every cell (an index into group and stratum);
    group (T,) the group of every table; stratum (T,) its index into STRATA.
    Returns rows dict(group, stratum, bits, target_cells, n_items, n_groups): every group x stratum that has cells,
    the margins, and group "all" = the mean over the groups that have cells in that stratum."""
    bits = np.asarray(bits, dtype=np.float64)
    cell_table = np.asarray(cell_table).astype(np.int64)
    group, stratum = np.asarray(group).astype(np.int64), np.asarray(stratum).astype(np.int64)
    G = int(n_groups) if n_groups else (int(group.max()) + 1 if group.size else 0)
    S = len(STRATA)
    key = group[cell_table] * S + stratum[cell_table]
    total = np.bincount(key, weights=bits, minlength=G * S).reshape(G, S)
    count = np.bincount(key, minlength=G * S).reshape(G, S)
    has = np.zeros((group.shape[0],), dtype=bool)
    has[cell_table] = True
    tables = np.bincount((group * S + stratum)[has], minlength=G * S).reshape(G, S)
    rows = []

    def emit(gname, sname, tot, cnt, tab):
        """tot, cnt, tab: per group. One row for the mean of the groups that have cells."""
        ok = cnt > 0
        if not ok.any():
            return
        rows.append(dict(group=gname, stratum=sname, bits=float((tot[ok] / cnt[ok]).mean()), target_cells=int(cnt.sum()),
                         n_items=int(tab.sum()), n_groups=int(ok.sum())))

    emit(ALL, ALL, total.sum(axis=1), count.sum(axis=1), tables.sum(axis=1))
    for s in range(S):
        emit(ALL, STRATA[s], total[:, s], count[:, s], tables[:, s])
    for g in range(G):
        emit(str(g), ALL, total[g:g + 1].sum(axis=1), count[g:g + 1].sum(axis=1), tables[g:g + 1].sum(axis=1))
        for s in range(S):
            emit(str(g), STRATA[s], total[g:g + 1, s], count[g:g + 1, s], tables[g:g + 1, s])
    return rows


def result_row(**fields):
    """A row of the result table with exactly the columns of the contract; columns not given are empty."""
    unknown = sorted(set(fields) - set(api.RESULT_ROW))
    if unknown:
        raise KeyError("not columns of the result table: %s" % unknown)
    return {k: fields.get(k, "") for k in api.RESULT_ROW}


def _cell(v):
    if isinstance(v, (float, np.floating)):
        return "%.6f" % v
    return str(v)


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(api.RESULT_ROW)
        for r in rows:
            w.writerow([_cell(r[k]) for k in api.RESULT_ROW])


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))
