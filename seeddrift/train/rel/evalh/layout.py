"""The stored layout in numpy, for the classical baselines: every row placed into the columns of the frame row.

This is the layout the stored tables carry (``col``). With a stretch of the examined row hidden it still shows the
relatives where the hidden residues align, so the bits read through it are too low (optimistic) where a whole stretch
is hidden (lower bits are better; the leak-free renderer replaces it). Table version 0 only: the frame row is row 0.
"""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel import api
from seeddrift.train.rel.evalh import conditions as cond


def place(values, n, col, fill):
    """values (N, R, ROW_LEN) in the rows' own coordinates -> the same in the frame's columns; `fill` where no residue
    of the row sits. A residue without a column (col < 0) is not shown."""
    values, col = np.asarray(values), np.asarray(col)
    ok = cond.residue_cells(n) & (col >= 0)
    t, r, s = np.nonzero(ok)
    c = col[t, r, s].astype(np.int64)
    if (c >= cond.LEN_OF_SLOT[s]).any():
        raise ValueError("a stored column lies outside its block")
    out = np.full(values.shape, fill, dtype=values.dtype)
    out[t, r, cond.START_OF_SLOT[s] + c] = values[t, r, s]
    return out


def masked_view(batch):
    """What a baseline may read, and what it is scored on.

    batch: the mapping returned by conditions.materialise. Returns
      view   (N, R, ROW_LEN) uint8: tokens in the frame's columns, GAP where a row has no residue, MASK on the hidden
             cells; nothing of the hidden residues is in it;
      cells  (table index in the batch, column) of the scored cells of row 0, in C order;
      truth  the hidden residue of every scored cell.
    """
    tok, n, col = np.asarray(batch["tok"]), np.asarray(batch["n"]), np.asarray(batch["col"])
    frame_row = batch.get("frame_row")
    if frame_row is not None and np.asarray(frame_row).any():
        raise ValueError("the stored layout of table version 0 has frame row 0")
    target = np.asarray(batch["target"])
    if target[:, 1:].any():
        raise ValueError("only cells of row 0 are scored")
    own = cond.residue_cells(n)[:, 0] & (col[:, 0] != cond.OFFSET_OF_SLOT[None, :])
    if own.any():
        raise ValueError("row 0 is the frame: its columns must be its own positions")
    grid = place(tok, n, col, api.GAP)
    hidden = place(target.astype(np.uint8), n, col, 0).astype(bool)
    t, s = np.nonzero(hidden[:, 0])
    truth = grid[t, 0, s].astype(np.int64)
    if truth.size and (truth.min() < api.TARGET_MIN or truth.max() > api.TARGET_MAX):
        raise ValueError("a scored cell does not hold a residue")
    view = np.where(hidden, np.uint8(api.MASK), grid).astype(np.uint8)
    return view, (t, s), truth
