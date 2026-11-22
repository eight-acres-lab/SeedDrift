"""The shard files of a table set on disk: the one writer (the contract, seeddrift/train/rel/api.py, names the arrays
and leaves the file layout to the writer). Reading them and placing their rows for stages and models belong to the
evaluator's seeddrift/train/rel/evalh/shards.py and evalh/layout.py; Part, open_parts and stored_grid below are the
writer's own reading of what it wrote, kept for the round-trip checks of the tests (the evaluator's reader and layout
are checked against them). Numpy and the standard library only.

A part of a table set (the training tables, the held-out tables, a synthetic set) is a folder of shards:
  <folder>/<shard>.<field>.npy   one plain .npy file per field, uncompressed, so it can be memory-mapped; its first
                                 axis is the tables of the shard. Fields: api.SHARD_FIELDS (what a model may read) and
                                 api.META_FIELDS (never a model input: samplers, evaluators and reports only).
  shards are named s0000, s0001, ...; each holds `shard_tables` tables, the last one what is left. Table t of a part is
  row t % shard_tables of shard t // shard_tables: the order of the tables is the order of the shards by name.
Conventions the arrays follow (version 0 of the tables; the kinds are the contract's):
  - row 0 is the examined genome; the rows that exist come first. A row that does not exist has kind KIND_NONE in every
    block, tok PAD, col -1, n 0 (meta: genome -1, identity nan). Whether a row or a block exists is read from `kind`,
    never from a meta field.
  - a block that does not exist (frame_len 0) has kind KIND_NONE in every row.
  - a row's codons of a block are in the row's own coordinates, left-justified in the block's slice; `col` gives each
    codon's column inside the block in the coordinates of row 0 (-1: no column there), strictly increasing over the
    codons that have one. The stored layout puts codon k at block start + col[k] (evalh/layout.py place).

  fields(rows)                            {field: (dtype, shape of one table)}
  alloc(n, rows)                          blank arrays of n tables (every field at its "nothing" value)
  create(folder, tables, rows, shard_tables)   new shard files -> ([(shard, tables)], {(shard index, field): memmap})
  write(folder, arrays, shard_tables)     a whole part from {field: array [N, ...]} -> [(shard, tables)]
  Part(folder, shards, rows), open_parts(run_dir), stored_grid(tok, col, fill)
                                          the writer's reading, for the round-trip checks only (see above)
"""
from __future__ import annotations

import json
import os

import numpy as np

from seeddrift.train.rel import api as A

KIND_NONE = getattr(A, "KIND_NONE", 0)
MODEL_FIELDS, META_FIELDS = tuple(A.SHARD_FIELDS), tuple(A.META_FIELDS)
TABLESET = os.path.join("data", "rel_w2", "v0", "tableset.json")
FILL = dict(tok=A.PAD, n=0, col=-1, frame_row=0, frame_len=0, kind=KIND_NONE, start=-1, strand=0, replicon=-1,
            genome=-1, group=-1, split=-1, gene=-1, tier=-1, identity=np.nan, genome_of_row=-1, row_gene=-1, row_first=-1)
ADDED = dict(getattr(A, "SHARD_FIELDS_ADDED", {}), **getattr(A, "META_FIELDS_ADDED", {}))      # fields of table version 1


def fields(rows, version=0):
    """{field: (dtype, shape of one table)}: the shard and the meta fields of the contract; from table version 1 also
    the fields the contract adds for it (SHARD_FIELDS_ADDED, META_FIELDS_ADDED)."""
    dims = dict(R=int(rows), ROW_LEN=A.ROW_LEN, B=A.N_BLOCKS)
    spec = list(A.SHARD_FIELDS.items()) + list(A.META_FIELDS.items()) + (list(ADDED.items()) if int(version) >= 1 else [])
    return {name: (np.dtype(dtype), tuple(dims[s] for s in shape.split()[1:])) for name, (dtype, shape, _) in spec}


def model_fields(version=0):
    """The fields a model may read: the contract's shard fields (and, from version 1, the added shard fields)."""
    return tuple(A.SHARD_FIELDS) + (tuple(getattr(A, "SHARD_FIELDS_ADDED", {})) if int(version) >= 1 else ())


def alloc(n, rows, version=0):
    return {name: np.full((int(n),) + shape, FILL[name], dtype=dt) for name, (dt, shape) in fields(rows, version).items()}


def shard_name(shard):
    return "s%04d" % int(shard)


def shard_path(folder, shard, field):
    return os.path.join(folder, "%s.%s.npy" % (shard if isinstance(shard, str) else shard_name(shard), field))


def create(folder, tables, rows, shard_tables, version=0):
    """New shard files for `tables` tables: ([(shard name, tables)], {(shard index, field): writable memmap})."""
    tables, shard_tables = int(tables), int(shard_tables)
    os.makedirs(folder, exist_ok=True)
    shards, mm = [], {}
    for s in range((tables + shard_tables - 1) // shard_tables):
        m = min(shard_tables, tables - s * shard_tables)
        for name, (dt, shape) in fields(rows, version).items():
            mm[s, name] = np.lib.format.open_memmap(shard_path(folder, s, name), mode="w+", dtype=dt, shape=(m,) + shape)
        shards.append((shard_name(s), m))
    return shards, mm


def write(folder, arrays, shard_tables, version=0):
    """Write a whole part from {field: array [N, ...]} (every field of the contract, the shapes of fields(rows))."""
    rows = int(arrays["tok"].shape[1])
    want = fields(rows, version)
    n = int(arrays["tok"].shape[0])
    bad = [k for k in want if k not in arrays or arrays[k].shape != (n,) + want[k][1]] + [k for k in arrays if k not in want]
    if bad:
        raise ValueError("fields missing, unknown or of another shape than the contract's: %s" % sorted(set(bad)))
    shards, mm = create(folder, n, rows, shard_tables, version)
    for (s, name), x in mm.items():
        x[:] = np.asarray(arrays[name][s * int(shard_tables):s * int(shard_tables) + len(x)], dtype=want[name][0])
        x.flush()
    return shards


class Part:
    """A part of a table set on disk. shards: [(shard name, tables)] in order."""

    def __init__(self, folder, shards, rows):
        self.folder, self.shards, self.rows = folder, [(str(n), int(m)) for n, m in shards], int(rows)
        self.tables = sum(m for _, m in self.shards)
        self.starts = np.cumsum([0] + [m for _, m in self.shards])
        self._open = {}

    def _array(self, i, name):
        if (i, name) not in self._open:
            self._open[i, name] = np.load(shard_path(self.folder, self.shards[i][0], name), mmap_mode="r")
        return self._open[i, name]

    def shard(self, i, names=MODEL_FIELDS):
        """{field: memory-mapped array} of shard i."""
        return {name: self._array(i, name) for name in names}

    def field(self, name):
        """One field of every table of the part, in memory (meant for the small fields: kind, n, the meta fields)."""
        return np.concatenate([np.asarray(self._array(i, name)) for i in range(len(self.shards))])

    def take(self, ids, names=MODEL_FIELDS):
        """{field: array} of the tables `ids` (indices into the part, any order), in memory."""
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) and (ids.min() < 0 or ids.max() >= self.tables):
            raise IndexError("table ids outside the part (%d tables)" % self.tables)
        which = np.searchsorted(self.starts, ids, side="right") - 1
        want = fields(self.rows)
        out = {name: np.empty((len(ids),) + want[name][1], dtype=want[name][0]) for name in names}
        for i in np.unique(which):
            m = which == i
            at = ids[m] - self.starts[i]
            order = np.argsort(at, kind="stable")
            for name in names:
                got = np.asarray(self._array(int(i), name)[at[order]])
                out[name][np.flatnonzero(m)[order]] = got
        return out


def open_parts(run_dir, tableset=TABLESET):
    """({part name: Part}, the manifest) of a table set written by stage rel_w2_v0 in the run directory run_dir."""
    with open(os.path.join(run_dir, tableset)) as f:
        doc = json.load(f)
    if not isinstance(doc.get("parts"), dict) or not doc.get("rows"):
        raise ValueError("%s is not a table-set manifest with parts" % tableset)
    return {name: Part(os.path.join(run_dir, p["folder"]), [(x["name"], x["tables"]) for x in p["shards"]], doc["rows"]) for name, p in doc["parts"].items()}, doc


def stored_grid(tok, col, fill=A.GAP):
    """The rows of tables in the coordinates of row 0, by the stored columns: tok, col [..., R, ROW_LEN] -> grid of the
    same shape with grid[..., row, block start + col] = tok where col >= 0, `fill` elsewhere. An upper-bound layout: the
    columns come from the alignment anchored on the examined row."""
    tok, col = np.asarray(tok), np.asarray(col)
    grid = np.full(tok.shape, fill, dtype=tok.dtype)
    for _, s0, s1 in A.block_slices():
        c = col[..., s0:s1].astype(np.int64)
        ok = c >= 0
        idx = np.nonzero(ok)
        grid[idx[:-1] + (s0 + c[ok],)] = tok[..., s0:s1][ok]
    return grid
