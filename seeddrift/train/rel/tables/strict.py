"""The strict rendering of one table under a mask plan (package W3), the one implementation for training and evaluation:
the correspondence aligned again with every hidden residue unknown (tables/align.py strict), or, with the examined
row's whole target gene hidden, the target block laid on its frame relative (align.frame_layout); then the leak-free
layout (tables/render.py). Numpy only.

  place(tab, visible, whole_gene)                    the table with its strict col (and frame_row, frame_len)
  render(tab, visible, target, whole_gene, ...)      (grid, truth, ext, placed table) as tables/render.render gives them
  fourth_test(tab, visible, target, whole_gene, key) W3's fourth validity test of one rendering: every hidden residue
      (of any row) replaced by a random codon and every stored position that a hidden residue could have shaped erased
      (rows 1.. in a block where row 0 hides a residue; a row in a block where it hides one), the table placed and
      rendered again: the grid, the cells laid by extension and the cells to score must be the same, cell for cell.
      The random codons come from the evaluator's 64-bit hash (key: the caller's number of the sample), the same in
      every numpy. Returns dict(passed, cells): the stages and the trainer's sampler count the failures.

The online leak-free rendering (render.render on the stored col) fails the fourth test by construction wherever a
hidden residue shaped a stored pair that stays an anchor; that is the leak this rendering removes.
"""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel import api as A
from seeddrift.train.rel.tables import align as AL
from seeddrift.train.rel.tables import render as RD

VERSION = "1"
STREAM_FOURTH = 61


def place(tab, visible, whole_gene=False):
    if whole_gene:
        col, frame_row, frame_len = AL.frame_layout(tab)
        return dict(tab, col=col, frame_row=frame_row, frame_len=frame_len)
    return dict(tab, col=AL.strict(tab, visible))


def render(tab, visible, target, whole_gene=False, control=None, key=0):
    placed = place(tab, visible, whole_gene)
    grid, truth, ext = RD.render(placed, visible, target, RD.LEAKFREE, control=control, key=key)
    return grid, truth, ext, placed


def erased(tab, visible):
    """col with every stored position erased that a hidden residue could have shaped: rows 1.. of a block where row 0
    hides a residue, and a row's own positions in a block where it hides one."""
    n = np.asarray(tab["n"]).astype(np.int64)
    cells = RD.cells_of(n)
    hidden = cells & ~np.asarray(visible, bool)
    gone = np.array(tab["col"], dtype=np.int16, copy=True)
    for b, (_, s, e) in enumerate(A.block_slices()):
        if hidden[0, s:e].any():
            gone[1:, s:e] = -1
        rows = np.flatnonzero(hidden[1:, s:e].any(axis=1)) + 1
        gone[rows, s:e] = -1
    return gone


def fourth_test(tab, visible, target, whole_gene=False, key=0, seed=20261124):
    from seeddrift.train.rel.evalh import conditions as cond          # the 64-bit hash every plan uses
    grid, truth, ext, _ = render(tab, visible, target, whole_gene)
    tok = np.asarray(tab["tok"])
    hidden = RD.cells_of(np.asarray(tab["n"]).astype(np.int64)) & ~np.asarray(visible, bool)
    u = cond.hash_u01(int(seed), STREAM_FOURTH, np.array([int(key)]), np.arange(tok.size)[None, :])[0].reshape(tok.shape)
    other = dict(tab, tok=np.where(hidden, 1 + np.floor(u * 65), tok).astype(np.uint8), col=erased(tab, visible))
    g2, t2, e2, _ = render(other, visible, target, whole_gene)
    ok = bool(np.array_equal(grid, g2) and np.array_equal(ext, e2) and np.array_equal(truth > 0, t2 > 0))
    return dict(passed=ok, cells=int((truth > 0).sum()))
