"""Evaluation conditions of the held-out tables: which cells are hidden and scored, and which rows are given.

One rule for every arm and every classical baseline, so that all of them score the same cells. numpy only: it runs
in the CI environment and in the model interpreter. The plan is, cell for cell, the one the trainer's held-out
evaluation uses (seeddrift/train/rel/data/heldout_plan.py; a test compares the two), written here for whole batches.
Its random choices come from numpy's default generator seeded by (seed, table id); numpy does not promise that
stream across versions, so the plan of a fixed batch is pinned (SELFCHECK) and nothing is planned in an interpreter
where it differs.

Development slice (API 1, table version 0, stored layout only). Table i is the i-th table of the held-out table set
in storage order:

- only cells of row 0 (the examined genome) are hidden, and only in its target block; every other cell is visible;
- ``scattered``: the residue cells of that block at the positions p with u[p] < MASK_RATE,
  u = default_rng([seed, i]).random(LEN_TARGET);
- ``whole_gene``: every residue cell of that block;
- every hidden cell is scored, the target is the residue (tokens 1..65);
- ``+no_relatives``: all other rows are removed (the arrays come back with one row);
- ``+repaired``: with k other rows that exist (a block of some kind), in stored order, row r takes its non-target
  blocks from row ((r - 1 + s) mod k) + 1, s = default_rng([seed + 1, i]).integers(1, k); the target block and
  row 0 stay. What a row is elsewhere no longer says what it is at the target. A table with fewer than two other
  rows has no such condition: nothing of it is scored.

What is hidden depends on the lengths and kinds only, never on the content.
"""
from __future__ import annotations

import hashlib

import numpy as np

from seeddrift.train.rel import api

VERSION = 2
SEED = 20261122                         # scattered cells: (SEED, table); re-pairing: (SEED + 1, table)
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
    width = TARGET_STOP - TARGET_START
    u = [np.random.default_rng([int(seed), int(i)]).random(width) for i in np.asarray(table_ids).reshape(-1).tolist()]
    return residue & (np.stack(u) < rate if u else np.zeros((0, width), dtype=bool))


def present_rows(kind):
    """(N, R) bool: the rows that exist: row 0, and every other row with a block of some kind."""
    out = (np.asarray(kind) != 0).any(axis=2)
    out[:, 0] = True
    return out


def repair_rows(kind, table_ids, seed=SEED):
    """src (N, R): for every row, the row its non-target blocks are taken from; has (N,) bool: the table has the
    condition (two or more other rows exist). Row 0 and rows that do not exist keep their own; the k other rows, in
    stored order, take from the row `shift` further on, cyclically, so none keeps its own."""
    present = present_rows(kind)
    N, R = present.shape
    src = np.tile(np.arange(R, dtype=np.int64), (N, 1))
    has = np.zeros(N, dtype=bool)
    for t, i in enumerate(np.asarray(table_ids).reshape(-1).tolist()):
        rows = np.flatnonzero(present[t, 1:]) + 1
        k = rows.shape[0]
        if k < 2:
            continue
        shift = int(np.random.default_rng([int(seed) + 1, int(i)]).integers(1, k))
        src[t, rows] = rows[(np.arange(k) + shift) % k]
        has[t] = True
    return src, has


def materialise(tables, table_ids, condition, seed=SEED, rate=MASK_RATE):
    """The arrays of a batch of tables under an evaluation condition.

    tables: a mapping with at least tok (N, R, ROW_LEN), n (N, R, B), col (N, R, ROW_LEN), kind (N, R, B); any other
    array whose shape starts with (N, R, B) or (N, R, ROW_LEN) is carried through the same row changes.
    table_ids: (N,) the position of every table in its table set.
    Returns the same mapping with rows removed or re-paired, plus
      visible, target  (N, R', ROW_LEN) bool, in the rows' own coordinates: target cells are hidden and scored;
      rows             (N, R') the stored row every row's non-target blocks come from;
      has              (N,) bool: the table has the condition (false only for a table that cannot be re-paired);
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
    has = np.ones(N, dtype=bool)
    if rows_mode == NO_RELATIVES:
        src = np.zeros((N, 1), dtype=np.int64)
        for k, v in per_row.items():
            out[k] = v[:, :1].copy()
    elif rows_mode == REPAIRED:
        src, has = repair_rows(tables["kind"], table_ids, seed)
        for k, v in per_row.items():
            moved = np.take_along_axis(v, src[:, :, None], axis=1)
            keep = (np.arange(B) == TARGET_BLOCK) if v.shape[2] == B else (BLOCK_OF_SLOT == TARGET_BLOCK)
            out[k] = np.where(keep[None, None, :], v, moved)
    else:
        src = np.tile(np.arange(R, dtype=np.int64), (N, 1))
        for k, v in per_row.items():
            out[k] = v
    hidden = np.zeros(out["tok"].shape, dtype=bool)
    hidden[:, 0, TARGET_START:TARGET_STOP] = hidden_target_cells(n, table_ids, mask, seed, rate) & has[:, None]
    out["visible"] = residue_cells(out["n"]) & ~hidden
    out["target"] = hidden
    out["rows"] = src
    out["has"] = has
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
            if not batch["has"][k]:                                    # the table has no such condition
                continue
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


SELFCHECK = "b0da9955aa199eed4a5a75b8616ca642680a33b48d96af7208e83f222040b39e"
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
    kind = np.where(n > 0, api.KIND_PRESENT, api.KIND_NOT_SEARCHED).astype(np.uint8)
    kind[~(n > 0).any(axis=2)] = 0                                 # a row without any segment does not exist
    tables = dict(tok=tok, n=n, col=col, kind=kind)
    out = []
    for c in CONDITIONS:
        d = ItemsDigest(c)
        d.add(materialise(tables, ids, c))
        out.append(d.hexdigest())
    return hashlib.sha256("\n".join(out).encode()).hexdigest()
