"""Mask scheme "span": one contiguous stretch of one row, the row drawn at random among the rows of the table and the
block in proportion to the length of the row's segments; the stretch has MIN_LEN to MAX_LEN codons (the whole segment
when that is shorter)."""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel.data import tables as T

NAME = "span"
MIN_LEN, MAX_LEN = 8, 128


def plan(meta, rng):
    n = meta["n"].astype(np.int64)
    cells = T.POS[None, :] < n[:, T.BLOCK_OF]
    hide = np.zeros_like(cells)
    rows = np.flatnonzero(n.sum(axis=1) > 0)
    if len(rows):
        r = int(rows[rng.integers(len(rows))])
        b = int(rng.choice(T.B, p=n[r] / n[r].sum()))
        length = int(min(n[r, b], rng.integers(MIN_LEN, MAX_LEN + 1)))
        at = T.SLICES[b][1] + int(rng.integers(0, n[r, b] - length + 1))
        hide[r, at:at + length] = True
    return cells & ~hide, hide
