"""Mask scheme "scattered": single cells, each codon of every row hidden with probability RATE."""
from __future__ import annotations

from seeddrift.train.rel import api as A
from seeddrift.train.rel.data import tables as T

NAME = A.COND_SCATTERED
RATE = 0.15


def plan(meta, rng):
    """meta: n [R, B] and kind [R, B] of the table. Returns visible, target: boolean [R, ROW_LEN] in own coordinates."""
    cells = T.POS[None, :] < meta["n"][:, T.BLOCK_OF]
    hide = cells & (rng.random(cells.shape) < RATE)
    return cells & ~hide, hide
