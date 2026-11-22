"""What the held-out evaluation hides and scores, the same for every arm and every baseline (numpy only).

Table i is the i-th table of the held-out table set in storage order. Every condition scores cells of the examined
row (row 0) in its target block only, with all stored rows (no random row count). Conditions (api.COND_*, joined by +):

  scattered                 the target-block codons of row 0 with u[p] < RATE, u = default_rng([SEED_SCATTERED, i]).random(LEN_TARGET)
  whole_gene                the whole target block of row 0 (under the stored placement: an upper bound)
  ...+no_relatives          the same with every other row removed
  whole_gene+repaired       the non-target blocks of the other rows moved to another row: with k other rows, row r takes
                            them from row ((r - 1 + s) mod k) + 1, s = default_rng([SEED_REPAIRED, i]).integers(1, k);
                            a table with fewer than two other rows has no such condition

Strata of a table: the largest identity of another row to row 0 over the target block (metadata; reporting only).
"""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel import api as A
from seeddrift.train.rel.data import tables as T

PLAN_VERSION = "1"
SEED_SCATTERED, SEED_REPAIRED = 20261122, 20261123
RATE = 0.15
SCATTERED, WHOLE = A.COND_SCATTERED, A.COND_WHOLE_GENE
NO_REL, REPAIRED = A.COND_NO_RELATIVES, A.COND_REPAIRED
CONDITIONS = (SCATTERED, WHOLE, SCATTERED + "+" + NO_REL, WHOLE + "+" + NO_REL, WHOLE + "+" + REPAIRED)
STRATA = ("none", "<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", ">=0.9")
EDGES = (0.3, 0.5, 0.7, 0.9)


def scattered_cells(i):
    """boolean [LEN_TARGET]: the positions of the target block hidden in table i (those beyond the gene's end do not count)."""
    return np.random.default_rng([SEED_SCATTERED, int(i)]).random(A.LEN_TARGET) < RATE


def repair_shift(i, k):
    """The shift of the other rows' non-target blocks in table i with k other rows (0: no such condition)."""
    return int(np.random.default_rng([SEED_REPAIRED, int(i)]).integers(1, k)) if k >= 2 else 0


def repaired(tab, shift):
    """The table with the non-target blocks of rows 1..k taken from the row `shift` further on (cyclically)."""
    k = tab["tok"].shape[0] - 1
    src = np.concatenate([[0], (np.arange(k) + shift) % k + 1]).astype(np.int64)
    out = {f: np.array(tab[f]) for f in T.MODEL_FIELDS}
    other = np.ones(T.L, dtype=bool)
    other[T.T0:T.T1] = False
    for f in ("tok", "col"):
        out[f][:, other] = tab[f][src][:, other]
    for f in ("n", "kind"):
        out[f][:, :T.TARGET_BLOCK] = tab[f][src][:, :T.TARGET_BLOCK]
    return out


def condition(tab, cond, i):
    """The table, the visible cells and the target cells of condition `cond` for held-out table i (tab: all its present
    rows, row 0 first). Returns None when the table has no such condition."""
    parts = cond.split("+")
    if NO_REL in parts:
        tab = T.take_rows(tab, [0])
    if REPAIRED in parts:
        shift = repair_shift(i, tab["tok"].shape[0] - 1)
        if not shift:
            return None
        tab = repaired(tab, shift)
    cells = T.inside(tab)
    hide = np.zeros_like(cells)
    if parts[0] == SCATTERED:
        hide[0, T.T0:T.T1] = cells[0, T.T0:T.T1] & scattered_cells(i)
    elif parts[0] == WHOLE:
        hide[0, T.T0:T.T1] = cells[0, T.T0:T.T1]
    else:
        raise ValueError("unknown condition %r" % cond)
    return tab, cells & ~hide, hide


def stratum(identity):
    """The stratum of a table from the identities of its rows to row 0 (row 0 itself ignored; nan = no cell)."""
    x = np.asarray(identity, dtype=np.float64)[1:]
    x = x[np.isfinite(x)]
    if not len(x):
        return STRATA[0]
    return STRATA[1 + int(np.searchsorted(EDGES, float(x.max()), side="right"))]
