"""Tables of the relational line as a trainer reads them (numpy only: loads in the CI environment and in the model
interpreter).

A table is a dict of the model-readable arrays of one table of a shard (api.SHARD_FIELDS): tok [R, ROW_LEN], n [R, B],
col [R, ROW_LEN], kind [R, B], frame_len [B]. Row 0 is the examined genome. Nothing of api.META_FIELDS is in it: the
metadata stays with the sampler and the evaluator (TableSet.meta) and never reaches a model.

  TableSet        the tables of one table set (or of one split of it), memory-mapped
  open_tableset   a TableSet from the hand-off manifest of the table set
  present_rows    the rows of a table that exist
  take_rows       a table of some of its rows ("no relatives" is take_rows(tab, [0]))
  inside          the cells that hold a codon, in every row's own coordinates
  render          the stored placement: every row put into the coordinates of the frame row by `col`

The stored placement was computed with every residue visible, so under it a hidden stretch of the examined row still
has its relatives placed by the hidden residues. Numbers read under it for a hidden stretch are an upper bound
(layout "stored"); the leak-free renderer replaces it.
"""
from __future__ import annotations

import json
import os

import numpy as np

from seeddrift.train.rel import api as A

READER_VERSION = "1"
API_VERSION = getattr(A, "API", None) or getattr(A, "API_VERSION", None)          # the version of the contract loaded
RENDER_VERSION = "stored-1"

L, B = A.ROW_LEN, A.N_BLOCKS
SLICES = A.block_slices()
BLOCK_OF = np.zeros(L, dtype=np.int64)              # block of every slot of a row
BLOCK_START = np.zeros(L, dtype=np.int64)
ROLE = np.zeros(L, dtype=np.int64)
for _b, (_role, _s, _e) in enumerate(SLICES):
    BLOCK_OF[_s:_e], BLOCK_START[_s:_e], ROLE[_s:_e] = _b, _s, _role
POS = np.arange(L) - BLOCK_START                    # position of a slot inside its block
BLOCK_LEN = np.array([e - s for _, s, e in SLICES], dtype=np.int64)
TARGET_BLOCK = B - 1
T0, T1 = SLICES[TARGET_BLOCK][1], SLICES[TARGET_BLOCK][2]
assert SLICES[TARGET_BLOCK][0] == A.ROLE_TARGET

# what a row shows where it has no codon, by the kind of its block
FILL = np.full(256, A.PAD, dtype=np.uint8)
FILL[A.KIND_PRESENT] = FILL[A.KIND_SELF_COPY] = A.GAP
FILL[A.KIND_ABSENT], FILL[A.KIND_NOT_SEARCHED], FILL[A.KIND_TRUNCATED] = A.ABSENT, A.NOT_SEARCHED, A.TRUNCATED

MODEL_FIELDS = ("tok", "n", "col", "kind", "frame_len")
SAMPLING_FIELDS = ("group", "identity")             # the metadata a sampler and an evaluator use


class TableSet:
    """Tables of a table set, memory-mapped. `shards`: the path prefix of every shard, in storage order (its files are
    <prefix>.<field>.npy). `split`: only the tables whose metadata split has that value (0 train, 1 held out); the
    tables keep their storage order and are numbered 0.. through the shards among those selected."""

    def __init__(self, shards, split=None, name="", kind=A.TABLE_REAL, sha256="", group_names=None):
        self.name, self.kind, self.sha256 = name, kind, sha256
        self.group_names = list(group_names or [])                 # the name of group g, where the table set gives names
        self.shards = [str(s) for s in shards]
        if not self.shards:
            raise ValueError("table set %s has no shards" % name)
        self._maps = {}
        where, group = [], []
        for k, s in enumerate(self.shards):
            tok = self._field(k, "tok")
            if tok.ndim != 3 or tok.shape[2] != L:
                raise ValueError("table set %s: tok of %s is shaped %s, not (tables, rows, %d)" % (name, s, tok.shape, L))
            g = np.asarray(self._field(k, "group")).astype(np.int64)
            if len(g) != tok.shape[0]:
                raise ValueError("table set %s: %s holds %d tables and %d groups" % (name, s, tok.shape[0], len(g)))
            keep = np.arange(len(g)) if split is None else np.flatnonzero(np.asarray(self._field(k, "split")) == split)
            where.append(np.stack([np.full(len(keep), k), keep], axis=1))
            group.append(g[keep])
        self.where = np.concatenate(where).astype(np.int64)
        self.group = np.concatenate(group)
        self.rows = int(self._field(0, "tok").shape[1])
        if not len(self.where):
            raise ValueError("table set %s has no table%s" % (name, "" if split is None else " of split %r" % (split,)))

    def __len__(self):
        return len(self.where)

    def _field(self, shard, field):
        key = (shard, field)
        if key not in self._maps:
            self._maps[key] = np.load("%s.%s.npy" % (self.shards[shard], field), mmap_mode="r", allow_pickle=False)
        return self._maps[key]

    def table(self, i):
        shard, k = self.where[i]
        return {f: np.array(self._field(int(shard), f)[int(k)]) for f in MODEL_FIELDS}

    def meta(self, i):
        shard, k = self.where[i]
        return {f: np.array(self._field(int(shard), f)[int(k)]) for f in SAMPLING_FIELDS}


def group_names(man, root):
    """The names of the groups the metadata field `group` indexes: the table set's index file, else its coverage."""
    for e in man.get("files") or []:
        if e.get("role") == "index" and os.path.basename(str(e.get("path", ""))) == "index.json":
            with open(os.path.join(root, e["path"])) as f:
                names = json.load(f).get("groups")
            if isinstance(names, list) and names:
                return [str(g.get("name")) if isinstance(g, dict) else str(g) for g in names]
    return [str(g) for g in (man.get("coverage") or {}).get("groups") or []]


def open_tableset(manifest, root, split=None):
    """A table set from its manifest (the hand-off file of the line: files [{path, role}] relative to `root`, the run
    directory that holds them). Its shards are the files <shard>.<field>.npy of role shard or meta; the shards of a
    folder are one ordered set, in the order of their names. `split`: see TableSet; the tables of a split come from
    one folder, so a table's number here is its position in that folder's set (the table id of the evaluator)."""
    with open(manifest) as f:
        man = json.load(f)
    fields = {}
    for e in man.get("files") or []:
        parts = os.path.basename(str(e.get("path", ""))).rsplit(".", 2)
        if e.get("role") in ("shard", "meta") and len(parts) == 3 and parts[2] == "npy":
            fields.setdefault(os.path.join(root, os.path.dirname(e["path"]), parts[0]), set()).add(parts[1])
    need = set(MODEL_FIELDS) | set(SAMPLING_FIELDS) | {"split"}
    bad = {os.path.basename(s): sorted(need - have) for s, have in fields.items() if need - have}
    if bad or not fields:
        raise ValueError("table set %s: %s" % (manifest, "shards lack fields %s" % bad if bad else "no shard files"))
    ts = TableSet(sorted(fields), split=split, name=str(man.get("name", "")), kind=str(man.get("kind", "")), sha256=str(man.get("sha256", "")),
                  group_names=group_names(man, root))
    folders = {os.path.dirname(ts.shards[int(k)]) for k in set(ts.where[:, 0].tolist())}
    if split is not None and len(folders) != 1:
        raise ValueError("table set %s: the tables of split %r lie in %d folders" % (manifest, split, len(folders)))
    return ts


def present_rows(tab):
    """The rows that exist: row 0, and every other row with a block of some kind."""
    rows = np.flatnonzero(tab["kind"].any(axis=1))
    return rows if len(rows) and rows[0] == 0 else np.concatenate([[0], rows[rows > 0]]).astype(np.int64)


def take_rows(tab, rows):
    rows = np.asarray(rows, dtype=np.int64)
    return dict(tok=tab["tok"][rows], n=tab["n"][rows], col=tab["col"][rows], kind=tab["kind"][rows], frame_len=tab["frame_len"])


def inside(tab):
    """[R, ROW_LEN]: the slot holds a codon of the row's own segment."""
    return POS[None, :] < tab["n"][:, BLOCK_OF]


def render(tab, visible, target):
    """The grid a model reads and the truth of its target cells, in the stored placement.

    visible, target: boolean [R, ROW_LEN] in every row's own coordinates (a mask scheme's plan). A codon that is not
    visible shows as MASK; a target that is no residue (1..65) or has no column in the frame is dropped.
    Returns grid uint8 [R, ROW_LEN] and truth uint8 [R, ROW_LEN] (0 = not a target, else the token to predict)."""
    tok, n, col, kind, frame_len = (tab[f] for f in MODEL_FIELDS)
    R = tok.shape[0]
    grid = np.where(POS[None, :] < frame_len[BLOCK_OF][None, :], FILL[kind][:, BLOCK_OF], A.PAD).astype(np.uint8)
    truth = np.zeros((R, L), dtype=np.uint8)
    rr, jj = np.nonzero((POS[None, :] < n[:, BLOCK_OF]) & (col >= 0))
    c = col[rr, jj].astype(np.int64)
    if len(c) and (c >= frame_len[BLOCK_OF[jj]]).any():
        raise ValueError("a stored column lies beyond the frame of its block")
    dest = BLOCK_START[jj] + c
    t = tok[rr, jj]
    grid[rr, dest] = np.where(visible[rr, jj], t, A.MASK)
    keep = target[rr, jj] & ~visible[rr, jj] & (t >= A.TARGET_MIN) & (t <= A.TARGET_MAX)
    truth[rr[keep], dest[keep]] = t[keep]
    return grid, truth
