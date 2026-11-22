"""Evaluation conditions of the held-out tables: which cells are hidden and scored, and which rows are given.

One rule for every arm and every classical baseline, so that all of them score the same cells. numpy only: it runs
in the CI environment and in the model interpreter. No generator of numpy's is used, because their streams are not
promised to stay the same between numpy versions: every random choice is a 64-bit hash of (seed, stream, table id,
index), so two interpreters get bit-identical plans.

Development slice (API 1, table version 0, stored layout only):

- only cells of row 0 (the examined genome) are hidden, and only in its target block; every other cell is visible;
- ``scattered``: every residue cell of that block is hidden independently with probability MASK_RATE;
- ``whole_gene``: every residue cell of that block is hidden;
- every hidden cell is scored, the target is the residue (tokens 1..65);
- ``+no_relatives``: all other rows are removed (the arrays come back with one row);
- ``+repaired``: the rows 1.. that exist are permuted by one random cycle, the same permutation in every non-target
  block; the target block and row 0 stay. What a row is elsewhere no longer says what it is at the target.

What is hidden depends on the lengths only, never on the content.
"""
from __future__ import annotations

import hashlib

import numpy as np

from seeddrift.train.rel import api

VERSION = 1
SEED = 20261122
MASK_RATE = 0.15

NO_RELATIVES = api.COND_NO_RELATIVES
REPAIRED = api.COND_REPAIRED
SCATTERED = api.COND_SCATTERED
WHOLE_GENE = api.COND_WHOLE_GENE
CONDITIONS = (
    SCATTERED,
    WHOLE_GENE,
    SCATTERED + "+" + NO_RELATIVES,
    WHOLE_GENE + "+" + NO_RELATIVES,
    WHOLE_GENE + "+" + REPAIRED,
)

STREAM_SCATTERED, STREAM_REPAIR = 1, 2

_BLOCKS = api.block_slices()
TARGET_BLOCK = len(_BLOCKS) - 1
assert _BLOCKS[TARGET_BLOCK][0] == api.ROLE_TARGET
TARGET_START, TARGET_STOP = _BLOCKS[TARGET_BLOCK][1], _BLOCKS[TARGET_BLOCK][2]
BLOCK_OF_SLOT = np.zeros(api.ROW_LEN, dtype=np.int64)
OFFSET_OF_SLOT = np.zeros(api.ROW_LEN, dtype=np.int64)
START_OF_SLOT = np.zeros(api.ROW_LEN, dtype=np.int64)
LEN_OF_SLOT = np.zeros(api.ROW_LEN, dtype=np.int64)
for _b, (_role, _s, _e) in enumerate(_BLOCKS):
    BLOCK_OF_SLOT[_s:_e] = _b
    OFFSET_OF_SLOT[_s:_e] = np.arange(_e - _s)
    START_OF_SLOT[_s:_e] = _s
    LEN_OF_SLOT[_s:_e] = _e - _s

_GOLD = np.uint64(0x9E3779B97F4A7C15)


def _mix(x):
    """The splitmix64 finaliser on a uint64 array (wraps around on purpose)."""
    x = np.atleast_1d(np.asarray(x, dtype=np.uint64))
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def hash_u01(seed, stream, table_id, index):
    """A number in [0, 1) for every (table_id, index), broadcast against each other; the same in every numpy."""
    t = np.atleast_1d(np.asarray(table_id)).astype(np.uint64)
    i = np.atleast_1d(np.asarray(index)).astype(np.uint64)
    with np.errstate(over="ignore"):
        h = _mix(np.atleast_1d(np.uint64(int(seed))) * _GOLD + np.uint64(int(stream)))
        h = _mix(h + t * _GOLD)
        h = _mix(h + i * _GOLD)
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def parse(condition):
    """'whole_gene+repaired' -> ('whole_gene', 'repaired'); the second part is '' when the relatives are given as stored."""
    mask, _, rows = condition.partition("+")
    if mask not in (SCATTERED, WHOLE_GENE) or rows not in ("", NO_RELATIVES, REPAIRED):
        raise ValueError("unknown evaluation condition %r" % condition)
    if rows == REPAIRED and mask != WHOLE_GENE:
        raise ValueError("%r: rows are re-paired only with the whole gene hidden" % condition)
    return mask, rows


def residue_cells(n):
    """(N, R, ROW_LEN) bool: the slots that hold a residue of the row's own segment. n: (N, R, B) lengths."""
    n = np.asarray(n)
    return OFFSET_OF_SLOT[None, None, :] < n[:, :, BLOCK_OF_SLOT]


def hidden_target_cells(n, table_ids, mask, seed=SEED, rate=MASK_RATE):
    """(N, LEN_TARGET) bool: the cells of row 0's target block that are hidden. n: (N, R, B) lengths."""
    n0 = np.asarray(n)[:, 0, TARGET_BLOCK].astype(np.int64)
    offs = np.arange(TARGET_STOP - TARGET_START)
    residue = offs[None, :] < n0[:, None]
    if mask == WHOLE_GENE:
        return residue
    if mask != SCATTERED:
        raise ValueError("unknown mask %r" % mask)
    u = hash_u01(seed, STREAM_SCATTERED, np.asarray(table_ids).reshape(-1, 1), offs[None, :])
    return residue & (u < rate)


def repair_rows(n, table_ids, seed=SEED):
    """(N, R) int: for every row, the row its non-target blocks are taken from. Row 0 and rows without any segment keep
    their own; the other rows form one random cycle (so with two or more of them no row keeps its own)."""
    n = np.asarray(n)
    N, R = n.shape[0], n.shape[1]
    exists = (n > 0).any(axis=2)
    exists[:, 0] = False
    key = hash_u01(seed, STREAM_REPAIR, np.asarray(table_ids).reshape(-1, 1), np.arange(R)[None, :])
    key = np.where(exists, key, 2.0)                                  # rows outside the cycle sort last
    order = np.argsort(key, axis=1, kind="stable")                    # the m existing rows first, in random order
    m = exists.sum(axis=1)
    pos = np.arange(R)[None, :]
    nxt = np.where(pos < m[:, None], (pos + 1) % np.maximum(m, 1)[:, None], pos)
    src = np.empty((N, R), dtype=np.int64)
    np.put_along_axis(src, order, np.take_along_axis(order, nxt, axis=1), axis=1)
    return src


def materialise(tables, table_ids, condition, seed=SEED, rate=MASK_RATE):
    """The arrays of a batch of tables under an evaluation condition.

    tables: a mapping with at least tok (N, R, ROW_LEN), n (N, R, B), col (N, R, ROW_LEN), kind (N, R, B); any other
    array whose shape starts with (N, R, B) or (N, R, ROW_LEN) is carried through the same row changes.
    table_ids: (N,) the position of every table in its table set.
    Returns the same mapping with rows removed or re-paired, plus
      visible, target  (N, R', ROW_LEN) bool, in the rows' own coordinates: target cells are hidden and scored;
      rows             (N, R') the stored row every row's non-target blocks come from;
      table_ids, condition.
    """
    mask, rows_mode = parse(condition)
    check_portable()
    table_ids = np.asarray(table_ids).astype(np.int64).reshape(-1)
    n = np.asarray(tables["n"])
    N, R, B = n.shape
    if B != len(_BLOCKS) or np.asarray(tables["tok"]).shape != (N, R, api.ROW_LEN) or table_ids.shape[0] != N:
        raise ValueError("tables do not have the shapes of API %d" % api.API)
    per_row = {}
    out = {}
    for k, v in tables.items():
        v = np.asarray(v)
        if v.ndim == 3 and v.shape[:2] == (N, R) and v.shape[2] in (B, api.ROW_LEN):
            per_row[k] = v
        else:
            out[k] = v
    if rows_mode == NO_RELATIVES:
        src = np.zeros((N, 1), dtype=np.int64)
        for k, v in per_row.items():
            out[k] = v[:, :1].copy()
    elif rows_mode == REPAIRED:
        src = repair_rows(n, table_ids, seed)
        for k, v in per_row.items():
            moved = np.take_along_axis(v, src[:, :, None], axis=1)
            keep = (np.arange(B) == TARGET_BLOCK) if v.shape[2] == B else (BLOCK_OF_SLOT == TARGET_BLOCK)
            out[k] = np.where(keep[None, None, :], v, moved)
    else:
        src = np.tile(np.arange(R, dtype=np.int64), (N, 1))
        for k, v in per_row.items():
            out[k] = v
    hidden = np.zeros(out["tok"].shape, dtype=bool)
    hidden[:, 0, TARGET_START:TARGET_STOP] = hidden_target_cells(n, table_ids, mask, seed, rate)
    out["visible"] = residue_cells(out["n"]) & ~hidden
    out["target"] = hidden
    out["rows"] = src
    out["table_ids"] = table_ids
    out["condition"] = condition
    return out


class ItemsDigest:
    """sha256 of what was scored under one condition: per table the scored cells, their residues and the row pairing.
    Two evaluations with the same digest scored the same cells of the same tables; the order of the batches is free."""

    def __init__(self, condition, seed=SEED, rate=MASK_RATE):
        self.head = "rel-evalh-items v%d api=%d condition=%s seed=%d rate=%.6f\n" % (VERSION, api.API, condition, seed, rate)
        self.parts = {}

    def add(self, batch):
        tok, target, rows = np.asarray(batch["tok"]), np.asarray(batch["target"]), np.asarray(batch["rows"])
        for k, t in enumerate(np.asarray(batch["table_ids"]).tolist()):
            h = hashlib.sha256()
            r, s = np.nonzero(target[k])
            h.update(r.astype("<i4").tobytes())
            h.update(s.astype("<i4").tobytes())
            h.update(tok[k][r, s].astype("u1").tobytes())
            h.update(rows[k].astype("<i4").tobytes())
            if t in self.parts and self.parts[t] != h.hexdigest():
                raise ValueError("table %d was added twice with different content" % t)
            self.parts[t] = h.hexdigest()

    def hexdigest(self):
        h = hashlib.sha256(self.head.encode())
        for t in sorted(self.parts):
            h.update(("%d %s\n" % (t, self.parts[t])).encode())
        return h.hexdigest()


SELFCHECK = "b18b72830b1b95ede37f262535566c5a6208aeaea256e6bb3ae2c8b8a49e2535"
_PORTABLE = []


def check_portable():
    """Once per process, before the first plan: this interpreter's numpy plans the cells every other one plans."""
    if not _PORTABLE:
        _PORTABLE.append(True)                                     # set first: selfcheck itself plans
        got = selfcheck()
        if got != SELFCHECK:
            del _PORTABLE[:]
            raise RuntimeError("the evaluation plan is not the pinned one in this interpreter (numpy %s): %s" % (np.__version__, got))


def selfcheck():
    """A digest of the plans of a fixed, hash-made batch. The same string (SELFCHECK) in every interpreter, or the
    rule is not portable; materialise refuses to plan in an interpreter where it differs."""
    N, R, B = 7, 6, len(_BLOCKS)
    ids = np.arange(100, 100 + N)
    lens = np.array([e - s for _, s, e in _BLOCKS])
    u = hash_u01(5, 9, ids.reshape(-1, 1, 1), (np.arange(R)[:, None] * B + np.arange(B)[None, :])[None])
    n = np.floor(u * (lens[None, None, :] + 1)).astype(np.int16)
    n[0, 3:] = 0                                                   # a table with two relatives only
    n[1, 1:] = 0                                                   # a table without relatives
    slots = np.arange(api.ROW_LEN)
    v = hash_u01(6, 9, (ids.reshape(-1, 1) * 64 + np.arange(R)[None, :])[:, :, None], slots[None, None, :])
    tok = (1 + np.floor(v * api.TARGET_MAX)).astype(np.uint8)
    tok[~residue_cells(n)] = api.PAD
    col = np.where(residue_cells(n), OFFSET_OF_SLOT[None, None, :], -1).astype(np.int16)
    tables = dict(tok=tok, n=n, col=col, kind=np.where(n > 0, api.KIND_PRESENT, api.KIND_NOT_SEARCHED).astype(np.uint8))
    out = []
    for c in CONDITIONS:
        d = ItemsDigest(c)
        d.add(materialise(tables, ids, c))
        out.append(d.hexdigest())
    return hashlib.sha256("\n".join(out).encode()).hexdigest()
