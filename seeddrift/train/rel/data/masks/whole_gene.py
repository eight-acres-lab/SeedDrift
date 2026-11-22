"""Mask scheme "whole_gene": the whole target gene of the examined row (row 0)."""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel import api as A
from seeddrift.train.rel.data import tables as T

NAME = A.COND_WHOLE_GENE


def plan(meta, rng):
    cells = T.POS[None, :] < meta["n"][:, T.BLOCK_OF]
    hide = np.zeros_like(cells)
    hide[0, T.T0:T.T1] = cells[0, T.T0:T.T1]
    return cells & ~hide, hide
