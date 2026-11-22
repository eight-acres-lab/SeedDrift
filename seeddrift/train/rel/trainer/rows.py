"""The result rows of a held-out evaluation (numpy only: read by the report in the CI environment too).

bits = sum of -log2 p(true token) over the scored cells / their number, pooled inside a (group, stratum); the row of
group "all" is the plain mean of the groups that have scored cells, and its n_items the number of tables in them.
Strata: the largest identity of another row to row 0 over the target block (heldout_plan.stratum), and "all".
"""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel import api as A
from seeddrift.train.rel.data import heldout_plan as H

ROWS_VERSION = "1"
ALL = "all"


def subset(tables, per_group):
    """The first per_group tables of every group, in storage order (0: all tables)."""
    if not per_group:
        return np.arange(len(tables), dtype=np.int64)
    keep = [np.flatnonzero(tables.group == g)[:per_group] for g in sorted(set(tables.group.tolist()))]
    return np.sort(np.concatenate(keep)).astype(np.int64)


def strata_of(tables, ids):
    return np.array([H.stratum(tables.meta(int(i))["identity"]) for i in ids])


def aggregate(bits, cells, groups, strata, group_names):
    """Rows (group, stratum, bits, n_items) of one condition: every group and "all", every stratum and "all"."""
    out = []
    for s in (ALL,) + H.STRATA:
        in_s = np.ones(len(bits), dtype=bool) if s == ALL else strata == s
        per_group = []
        for g in sorted(set(groups.tolist())):
            m = in_s & (groups == g) & (cells > 0)
            if m.any():
                x = float(bits[m].sum() / cells[m].sum())
                per_group.append(x)
                out.append((group_names.get(int(g), str(int(g))), s, x, int(m.sum())))
        if per_group:
            out.append((ALL, s, float(np.mean(per_group)), int((in_s & (cells > 0)).sum())))
    return out


def result_rows(fixed, cond, agg):
    """Rows of the contract's result table (api.RESULT_ROW) for one condition; `fixed` holds the columns that are the
    same for every row of this evaluation."""
    rows = []
    for group, stratum, bits, n_items in agg:
        row = dict(fixed, condition=cond, group=group, tier=ALL, stratum=stratum, passes=1, bits="%.6f" % bits, baseline="", bits_minus_baseline="",
                   n_items=n_items)
        missing = [c for c in A.RESULT_ROW if c not in row]
        if missing:
            raise KeyError("result row lacks %s" % missing)
        rows.append([str(row[c]) for c in A.RESULT_ROW])
    return rows


def write_rows(path, rows, header=False):
    with open(path, "a") as f:
        if header:
            f.write("\t".join(A.RESULT_ROW) + "\n")
        for r in rows:
            if any("\t" in x or "\n" in x for x in r):
                raise ValueError("a result value holds a tab or a line break")
            f.write("\t".join(r) + "\n")
        f.flush()


def read_rows(path):
    """The rows of a result file as dicts (every value a string)."""
    with open(path) as f:
        lines = f.read().splitlines()
    if not lines or tuple(lines[0].split("\t")) != A.RESULT_ROW:
        raise ValueError("%s does not start with the header of the contract's result row" % path)
    out = []
    for n, line in enumerate(lines[1:], 2):
        parts = line.split("\t")
        if len(parts) != len(A.RESULT_ROW):
            raise ValueError("%s line %d has %d columns, not %d" % (path, n, len(parts), len(A.RESULT_ROW)))
        out.append(dict(zip(A.RESULT_ROW, parts)))
    return out
