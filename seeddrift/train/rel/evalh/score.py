"""From per-cell bits to result rows: the strata and the averaging every arm and every baseline share.

- bits per codon of a set of cells = the sum of -log2 p(residue) over the cells / the number of cells;
- per group first, then the plain mean of the groups that have cells (12 groups in a real table set);
- strata of a table: the identity of the nearest row at the target locus (the largest stored identity of rows 1..).
  Two sets of bins: COARSE, the six of the trainer's held-out rows (seeddrift/train/rel/trainer/rows.py; the same
  labels, so model and baseline rows sit side by side), and FINE, ten equal bins and "none" (labels start with
  nearest_identity=). And OTHER_ROWS: how many other rows the table has (0, 1, two or more); the tables with two or
  more are the ones the re-paired condition scores. Reporting keys only, never given to a model.
"""
from __future__ import annotations

import csv

import numpy as np

from seeddrift.train.rel import api

VERSION = 2
ALL = "all"
COARSE = ("none", "<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", ">=0.9")
COARSE_EDGES = (0.3, 0.5, 0.7, 0.9)
N_FINE = 10
FINE_PREFIX = "nearest_identity="
FINE = tuple("%s%.1f-%.1f" % (FINE_PREFIX, k / 10.0, (k + 1) / 10.0) for k in range(N_FINE)) + (FINE_PREFIX + "none",)
OTHER_ROWS = ("other_rows=0", "other_rows=1", "other_rows>=2")


def nearest_identity(identity):
    """(N,) the largest identity of rows 1.. at the target locus; nan when no row has a cell there."""
    ident = np.asarray(identity, dtype=np.float64)[:, 1:]
    if ident.shape[1] == 0:
        return np.full(ident.shape[0], np.nan)
    has = np.isfinite(ident)
    best = np.where(has, ident, -1.0).max(axis=1)
    return np.where(has.any(axis=1), best, np.nan)


def coarse_stratum(identity):
    """(N,) index into COARSE for every table, from the stored (N, R) identity of every row with row 0."""
    near = nearest_identity(identity)
    k = 1 + np.searchsorted(np.asarray(COARSE_EDGES), np.nan_to_num(near, nan=0.0), side="right")
    return np.where(np.isnan(near), 0, k)


def fine_stratum(identity):
    """(N,) index into FINE for every table."""
    near = nearest_identity(identity)
    k = np.minimum(np.floor(np.nan_to_num(near, nan=0.0) * N_FINE + 1e-9).astype(np.int64), N_FINE - 1)
    return np.where(np.isnan(near), N_FINE, np.maximum(k, 0))


def strata_of(identity, other_rows=None):
    """[(labels, index of every table)]: the coarse and the fine bins of the nearest identity and, given the number
    of other rows of every table, OTHER_ROWS."""
    out = [(COARSE, coarse_stratum(identity)), (FINE, fine_stratum(identity))]
    if other_rows is not None:
        out.append((OTHER_ROWS, np.minimum(np.asarray(other_rows).astype(np.int64), 2)))
    return out


def aggregate(bits, cell_table, group, strata, n_groups=None, group_names=None):
    """bits (C,) of the scored cells; cell_table (C,) the table of every cell (an index into group and the strata);
    group (T,) the group index of every table; strata: [(labels, (T,) index into labels)] (strata_of).
    Returns rows dict(group, stratum, bits, target_cells, n_items, n_groups): every group x stratum that has cells and
    the margins; group "all" = the mean over the groups that have cells in that stratum, n_items its tables with
    cells. A group is named group_names[index] when given, else the index."""
    bits = np.asarray(bits, dtype=np.float64)
    cell_table = np.asarray(cell_table).astype(np.int64)
    group = np.asarray(group).astype(np.int64)
    G = int(n_groups) if n_groups else (int(group.max()) + 1 if group.size else 0)
    has = np.zeros((group.shape[0],), dtype=bool)
    has[cell_table] = True
    name = lambda g: str(group_names[g]) if group_names is not None and g in group_names else str(g)      # noqa: E731
    rows = []

    def emit(gname, sname, tot, cnt, tab):
        """tot, cnt, tab: per group. One row for the mean of the groups that have cells."""
        ok = cnt > 0
        if ok.any():
            rows.append(dict(group=gname, stratum=sname, bits=float((tot[ok] / cnt[ok]).mean()), target_cells=int(cnt.sum()),
                             n_items=int(tab.sum()), n_groups=int(ok.sum())))

    def table(key, size):
        total = np.bincount(key[cell_table], weights=bits, minlength=size)
        return total, np.bincount(key[cell_table], minlength=size), np.bincount(key[has], minlength=size)

    tot, cnt, tab = table(group, G)
    emit(ALL, ALL, tot, cnt, tab)
    for g in range(G):
        emit(name(g), ALL, tot[g:g + 1], cnt[g:g + 1], tab[g:g + 1])
    for labels, index in strata:
        S = len(labels)
        tot, cnt, tab = (x.reshape(G, S) for x in table(group * S + np.asarray(index).astype(np.int64), G * S))
        for k in range(S):
            emit(ALL, labels[k], tot[:, k], cnt[:, k], tab[:, k])
            for g in range(G):
                emit(name(g), labels[k], tot[g:g + 1, k], cnt[g:g + 1, k], tab[g:g + 1, k])
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
