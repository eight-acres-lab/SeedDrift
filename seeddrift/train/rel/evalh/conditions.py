"""Evaluation conditions of the held-out tables: which cells are hidden and scored, and which rows are given.

The one implementation (the trainer's held-out evaluation and the classical baselines both call it), so every arm and
every baseline scores the same cells with the same truth. numpy only: it runs in the CI environment and in the model
interpreter. Every random choice is a 64-bit hash of (seed, stream, table id, index), never a numpy generator, so the
plan is the same in every numpy; the plan of a fixed batch is pinned as well (SELFCHECK).

Development slice (API 1, table version 0, stored layout only). The table id is the position of a table in its part
of the table set, in storage order:

- only cells of row 0 (the examined genome) are hidden, and only in its target block; every other cell is visible;
- ``scattered``: the residue cells of that block at the offsets p with u(seed, table, p) < MASK_RATE; a table where
  no cell falls below the rate has its residue cell with the smallest u hidden, so every table has a scored cell;
- ``whole_gene``: every residue cell of that block;
- every hidden cell is scored, the target is the residue (tokens 1..65);
- ``+no_relatives``: all other rows are removed (the arrays come back with one row);
- ``+repaired``: the k other rows that exist (any block of a kind other than KIND_NONE) are permuted by one random
  cycle of length k, the same permutation in every non-target block (neighbours and relation blocks move together);
  the target block and row 0 stay. Each row of the non-target part is still one genome, only no longer the one of its
  target block. The cycle is drawn from the rank of each row among the rows that exist, so leaving out rows that do
  not exist changes nothing. A table with fewer than two other rows has no such condition: nothing of it is scored.

What is hidden depends on the lengths and kinds only, never on the content. A row or block of kind KIND_NONE does not
exist: it is no relative, enters no average and is never the nearest row.

Conditions of the evaluator that only an evaluation uses (EXTRA_CONDITIONS; not in today's five, so a trainer's
evaluation does not change):
- ``suffix``: the second half of the target block of row 0 hidden (offsets from n // 2 on), the first half visible;
- ``+offset<k>`` (k in OFFSETS): the other rows' correspondence in the target block moved k codons (their stored
  columns + k; a codon pushed past row 0's segment has no column any more): how exact the correspondence must be;
- ``whole_gene+synonymous``: every codon of the other rows' relation blocks replaced by another codon of the same
  amino acid (a hashed choice; stops among the stops; a codon without a synonym stays): whether a relative is told
  by its codon usage rather than by what it encodes. The genetic code is used for this recoding only.
"""
from __future__ import annotations

import hashlib

import numpy as np

from seeddrift.train.rel import api

VERSION = 3
SEED = 20261122
MASK_RATE = 0.15
STREAM_SCATTERED, STREAM_REPAIR, STREAM_SYNONYMOUS = 1, 2, 3
SUBSET_PER_GROUP = 50                   # the evaluation during training: the first 50 tables of each group

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
SUFFIX = api.COND_SUFFIX
SYNONYMOUS = "synonymous"               # not yet a name of the contract (a request asks for it)
OFFSET = "offset"
OFFSETS = (1, 3, 12)
MASKS = (SCATTERED, WHOLE_GENE, SUFFIX)
EXTRA_CONDITIONS = ((SUFFIX, SUFFIX + "+" + NO_RELATIVES) + tuple("%s+%s%d" % (SCATTERED, OFFSET, k) for k in OFFSETS)
                    + (WHOLE_GENE + "+" + SYNONYMOUS,))

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
    """'whole_gene+repaired' -> ('whole_gene', 'repaired'); the second part is '' when the relatives are given as
    stored. Masks: scattered, whole_gene, suffix; second parts: no_relatives, offset<k>, and with the whole gene
    repaired and synonymous."""
    mask, _, rows = condition.partition("+")
    if mask not in MASKS or rows not in ("", NO_RELATIVES, REPAIRED, SYNONYMOUS) + tuple("%s%d" % (OFFSET, k) for k in OFFSETS):
        raise ValueError("unknown evaluation condition %r" % condition)
    if rows in (REPAIRED, SYNONYMOUS) and mask != WHOLE_GENE:
        raise ValueError("%r: rows are re-paired or recoded only with the whole gene hidden" % condition)
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
    if mask == SUFFIX:
        return residue & (offs[None, :] >= (n0 // 2)[:, None])
    if mask != SCATTERED:
        raise ValueError("unknown mask %r" % mask)
    u = hash_u01(seed, STREAM_SCATTERED, np.asarray(table_ids).reshape(-1, 1), offs[None, :])
    hide = residue & (u < rate)
    lonely = residue.any(axis=1) & ~hide.any(axis=1)                   # no cell below the rate: the smallest u
    if lonely.any():
        first = np.where(residue, u, 2.0).argmin(axis=1)
        hide[np.flatnonzero(lonely), first[lonely]] = True
    return hide


def present_rows(kind):
    """(N, R) bool: the rows that exist: row 0, and every other row with a block of a kind other than KIND_NONE."""
    out = (np.asarray(kind) != api.KIND_NONE).any(axis=2)
    out[:, 0] = True
    return out


def offset_target(col, n, k):
    """col with the other rows' columns in the target block moved k to the right; a codon moved past row 0's segment
    (the frame of the block) has no column any more."""
    col = np.array(col, copy=True)
    n0 = np.asarray(n)[:, 0, TARGET_BLOCK].astype(np.int64)
    c = col[:, 1:, TARGET_START:TARGET_STOP].astype(np.int64)
    moved = np.where(c >= 0, c + int(k), -1)
    col[:, 1:, TARGET_START:TARGET_STOP] = np.where(moved < n0[:, None, None], moved, -1).astype(col.dtype)
    return col


def synonymous(tok, n, table_ids, seed=SEED):
    """tok with every codon of the other rows' relation blocks replaced by another codon of the same amino acid (stop
    codons among the stops), chosen by hash; a codon without a synonym, and "other", stay."""
    from seeddrift.train.rel.evalh import score                      # the genetic code: this recoding and the report only
    cls = score.amino_acid_classes()
    tok = np.array(tok, copy=True)
    relation = (BLOCK_OF_SLOT < api.N_RELATION)[None, None, :]
    t, r, s = np.nonzero(residue_cells(n)[:, 1:, :] & relation)
    r = r + 1
    codon = tok[t, r, s].astype(np.int64)
    u = hash_u01(seed, STREAM_SYNONYMOUS, np.asarray(table_ids).reshape(-1)[t], r * api.ROW_LEN + s)
    new = codon.copy()
    for c in np.unique(cls[1:65]):
        members = np.flatnonzero(cls[1:65] == c) + 1
        sel = (codon >= 1) & (codon <= 64) & (cls[np.minimum(codon, 65)] == c)
        if members.shape[0] < 2 or not sel.any():
            continue
        pick = np.floor(u[sel] * (members.shape[0] - 1)).astype(np.int64)
        pick = pick + (pick >= np.searchsorted(members, codon[sel]))        # never the codon itself
        new[sel] = members[pick]
    tok[t, r, s] = new.astype(tok.dtype)
    return tok


def subset(group, per_group=SUBSET_PER_GROUP):
    """The tables of the evaluation during training: the first per_group tables of every group in storage order (600
    of the 3,000 held-out tables today; at the end of a training every table is evaluated). group: (N,) the group of
    every table of the part. Returns the table ids, in storage order."""
    group = np.asarray(group).reshape(-1)
    keep = np.zeros(group.shape[0], dtype=bool)
    for g in np.unique(group):
        keep[np.flatnonzero(group == g)[:int(per_group)]] = True
    return np.flatnonzero(keep)


def repair_rows(kind, table_ids, seed=SEED):
    """src (N, R): for every row, the row its non-target blocks are taken from; has (N,) bool: the table has the
    condition (two or more other rows exist). Row 0 and rows that do not exist keep their own; the k other rows form
    one cycle: in the random order of the hashes of their ranks, each takes from the next, the last from the first."""
    present = present_rows(kind)
    N, R = present.shape
    src = np.tile(np.arange(R, dtype=np.int64), (N, 1))
    has = np.zeros(N, dtype=bool)
    ids = np.asarray(table_ids).reshape(-1)
    for t in range(N):
        rows = np.flatnonzero(present[t, 1:]) + 1
        k = rows.shape[0]
        if k < 2:
            continue
        order = rows[np.argsort(hash_u01(seed, STREAM_REPAIR, ids[t], np.arange(k)), kind="stable")]
        src[t, order] = np.roll(order, -1)
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
        if rows_mode.startswith(OFFSET):
            out["col"] = offset_target(out["col"], n, int(rows_mode[len(OFFSET):]))
        elif rows_mode == SYNONYMOUS:
            out["tok"] = synonymous(out["tok"], n, table_ids, seed)
    hidden = np.zeros(out["tok"].shape, dtype=bool)
    hidden[:, 0, TARGET_START:TARGET_STOP] = hidden_target_cells(n, table_ids, mask, seed, rate) & has[:, None]
    out["visible"] = residue_cells(out["n"]) & ~hidden
    out["target"] = hidden
    out["rows"] = src
    out["has"] = has
    out["table_ids"] = table_ids
    out["condition"] = condition
    return out


def table_condition(tab, condition, table_id, seed=SEED, rate=MASK_RATE):
    """One table under a condition, for a reader that works table by table: tab holds the fields of one table shaped
    [R, ...] (tok, n, col, kind, and any others; rows that do not exist may be left out or kept). Returns (the table
    with rows removed or re-paired, visible, target), every array [R', ...], or None when the table has no such
    condition. The same cells as materialise on a batch."""
    m = materialise({k: np.asarray(v)[None] for k, v in tab.items()}, [table_id], condition, seed, rate)
    if not m["has"][0]:
        return None
    return {k: m[k][0] for k in tab}, m["visible"][0], m["target"][0]


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


SELFCHECK = "442d6c33b194d2a004d95ecaa18dc442850e29c05e31ada95d4b04790500373b"
_PORTABLE = []


def check_portable():
    """Once per process, before the first plan: this interpreter's numpy plans the cells every other one plans."""
    if not _PORTABLE:
        _PORTABLE.append(True)                                     # set first: selfcheck itself plans
        got = selfcheck()
        if got != SELFCHECK:
            del _PORTABLE[:]
            raise RuntimeError("the evaluation plan is not the pinned one in this interpreter (numpy %s): %s" % (np.__version__, got))


def selfcheck(conditions=CONDITIONS):
    """A digest of the plans of a fixed, hash-made batch. The same string (SELFCHECK, for today's five conditions) in
    every interpreter, or the rule is not portable; materialise refuses to plan in an interpreter where it differs."""
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
    for c in conditions:
        m = materialise(tables, ids, c)
        d = ItemsDigest(c)
        d.add(m)
        out.append(d.hexdigest() + hashlib.sha256(m["tok"].tobytes() + m["col"].tobytes()).hexdigest())
    return hashlib.sha256("\n".join(out).encode()).hexdigest()
