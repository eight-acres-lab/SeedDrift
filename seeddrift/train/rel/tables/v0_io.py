"""The shard files of a table set on disk: the one writer (the contract, seeddrift/train/rel/api.py, names the arrays
and leaves the file layout to the writer). Reading them and placing their rows belong to the evaluator's
seeddrift/train/rel/evalh/shards.py and evalh/layout.py; nothing here reads a table set. Numpy and the standard library
only.

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
"""
from __future__ import annotations

import os

import numpy as np

from seeddrift.train.rel import api as A

KIND_NONE = getattr(A, "KIND_NONE", 0)
FILL = dict(tok=A.PAD, n=0, col=-1, frame_row=0, frame_len=0, kind=KIND_NONE, start=-1, strand=0, replicon=-1,
            genome=-1, group=-1, split=-1, gene=-1, tier=-1, identity=np.nan)


def fields(rows):
    """{field: (dtype, shape of one table)}: the shard and the meta fields of the contract."""
    dims = dict(R=int(rows), ROW_LEN=A.ROW_LEN, B=A.N_BLOCKS)
    return {name: (np.dtype(dtype), tuple(dims[s] for s in shape.split()[1:]))
            for name, (dtype, shape, _) in list(A.SHARD_FIELDS.items()) + list(A.META_FIELDS.items())}


def alloc(n, rows):
    return {name: np.full((int(n),) + shape, FILL[name], dtype=dt) for name, (dt, shape) in fields(rows).items()}


def shard_name(shard):
    return "s%04d" % int(shard)


def shard_path(folder, shard, field):
    return os.path.join(folder, "%s.%s.npy" % (shard if isinstance(shard, str) else shard_name(shard), field))


def create(folder, tables, rows, shard_tables):
    """New shard files for `tables` tables: ([(shard name, tables)], {(shard index, field): writable memmap})."""
    tables, shard_tables = int(tables), int(shard_tables)
    os.makedirs(folder, exist_ok=True)
    shards, mm = [], {}
    for s in range((tables + shard_tables - 1) // shard_tables):
        m = min(shard_tables, tables - s * shard_tables)
        for name, (dt, shape) in fields(rows).items():
            mm[s, name] = np.lib.format.open_memmap(shard_path(folder, s, name), mode="w+", dtype=dt, shape=(m,) + shape)
        shards.append((shard_name(s), m))
    return shards, mm


def write(folder, arrays, shard_tables):
    """Write a whole part from {field: array [N, ...]} (every field of the contract, the shapes of fields(rows))."""
    rows = int(arrays["tok"].shape[1])
    want = fields(rows)
    n = int(arrays["tok"].shape[0])
    bad = [k for k in want if k not in arrays or arrays[k].shape != (n,) + want[k][1]] + [k for k in arrays if k not in want]
    if bad:
        raise ValueError("fields missing, unknown or of another shape than the contract's: %s" % sorted(set(bad)))
    shards, mm = create(folder, n, rows, shard_tables)
    for (s, name), x in mm.items():
        x[:] = np.asarray(arrays[name][s * int(shard_tables):s * int(shard_tables) + len(x)], dtype=want[name][0])
        x.flush()
    return shards
