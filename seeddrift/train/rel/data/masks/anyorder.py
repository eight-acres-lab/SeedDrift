"""Mask scheme "anyorder" (any-order fill-in, the reference arm with one change): a ratio r drawn uniformly in
[R_MIN, 1] for each table, and every codon of every row hidden with probability r (at least one). WEIGHTED: the
sampler weighs the cells of the table by the inverse of the ratio it realised (its cells over its hidden cells), so a
step's loss is the weighted mean that makes fill-in at every ratio count alike (the objective of an any-order model)."""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel.data import tables as T

NAME = "anyorder"
R_MIN = 0.02
WEIGHTED = True


def plan(meta, rng):
    cells = T.POS[None, :] < meta["n"][:, T.BLOCK_OF]
    r = float(rng.uniform(R_MIN, 1.0))
    hide = cells & (rng.random(cells.shape) < r)
    if cells.any() and not hide.any():
        rr, cc = np.nonzero(cells)
        k = int(rng.integers(len(rr)))
        hide[rr[k], cc[k]] = True
    return cells & ~hide, hide
