"""Tables of the relational line as the trainer reads them, and the grid a model reads (numpy only: loads in the CI
environment and in the model interpreter).

Reading goes through the evaluator's reader (seeddrift/train/rel/evalh/shards.py: the files a table-set manifest
lists, <folder>/<shard>.<field>.npy, the shards of a folder in name order, table t = its position over them; that
position is the table id of the evaluation conditions). Placement goes through the evaluator's mapping
(evalh/layout.py place: grid column = block start + col; a codon with col = -1 is not shown). On top of it this module
writes the tokens of the model's grid (the stored placement of table version 0, frame row 0):
  - a codon of a row at its column, as itself; a hidden one as MASK;
  - inside a row's aligned span in a block (from its first to its last placed column) a column without a codon of the
    row: GAP; outside the span: PAD;
  - a block whose kind has no codons (NOT_SEARCHED, TRUNCATED, ABSENT): its token over the frame row's length;
  - beyond the frame row's length, and in a block of KIND_NONE: PAD. PAD cells take no part in attention (the model
    masks them as keys); a row that does not exist (KIND_NONE in every block) is never in a grid.
The model never reads metadata (group, identity, genome): those stay with the sampler and the report.
Under the stored placement a hidden stretch of the examined row still has its relatives placed by the hidden
residues, so numbers read through it for a hidden stretch are upper bounds (layout "stored").
"""
from __future__ import annotations

import json

import numpy as np

from seeddrift.train.rel import api as A
from seeddrift.train.rel.evalh import layout, shards

READER_VERSION = "2"
RENDER_VERSION = "stored-2"
API_VERSION = getattr(A, "API", None) or getattr(A, "API_VERSION", None)          # the version of the contract loaded

L, B = A.ROW_LEN, A.N_BLOCKS
SLICES = A.block_slices()
BLOCK_OF = np.zeros(L, dtype=np.int64)              # block of every slot of a row
BLOCK_START = np.zeros(L, dtype=np.int64)
ROLE = np.zeros(L, dtype=np.int64)
for _b, (_role, _s, _e) in enumerate(SLICES):
    BLOCK_OF[_s:_e], BLOCK_START[_s:_e], ROLE[_s:_e] = _b, _s, _role
POS = np.arange(L) - BLOCK_START                    # position of a slot inside its block
TARGET_BLOCK = B - 1
T0, T1 = SLICES[TARGET_BLOCK][1], SLICES[TARGET_BLOCK][2]
assert SLICES[TARGET_BLOCK][0] == A.ROLE_TARGET

KIND_NONE = getattr(A, "KIND_NONE", 0)
WITH_CODONS = (A.KIND_PRESENT, A.KIND_SELF_COPY)
FILL = np.full(256, A.PAD, dtype=np.uint8)          # the token of a block without codons, by its kind
FILL[A.KIND_ABSENT], FILL[A.KIND_NOT_SEARCHED], FILL[A.KIND_TRUNCATED] = A.ABSENT, A.NOT_SEARCHED, A.TRUNCATED

MODEL_FIELDS = ("tok", "n", "col", "kind", "frame_len")


class Tables:
    """The tables of one split of a table set (one folder of its shards; evalh/shards.TableSet), read without
    hashing (the stage that starts a training verifies every file against the manifest first). `group` (N,) is
    metadata for the sampler and the report."""

    def __init__(self, ts, manifest):
        self.ts, self.manifest = ts, manifest
        self.name, self.kind, self.sha256 = str(manifest.get("name", "")), str(manifest.get("kind", "")), str(manifest.get("sha256", ""))
        self.group = np.asarray(ts.load(0, ts.tables, ("group",), verify=False)["group"]).astype(np.int64)
        self.rows = int(ts.array(ts.shards[0], "tok").shape[1])

    def __len__(self):
        return int(self.ts.tables)

    def table(self, i):
        """The model fields of table i, without the leading axis."""
        got = self.ts.load(int(i), int(i) + 1, MODEL_FIELDS, verify=False)
        return {f: got[f][0] for f in MODEL_FIELDS}

    def load(self, fields, first=0, last=None):
        return self.ts.load(int(first), len(self) if last is None else int(last), tuple(fields), verify=False)


def read_manifest(manifest):
    if isinstance(manifest, dict):
        return manifest
    with open(manifest) as f:
        return json.load(f)


def open_split(root, manifest, split):
    """The tables of `split` (0 train, 1 held out) of a table-set manifest whose files lie under root (the run
    directory that wrote them)."""
    doc = read_manifest(manifest)
    try:
        sets = shards.by_split(root, [doc])
    except ValueError as err:
        raise ValueError("table set %s: %s" % (doc.get("name"), err))
    if int(split) not in sets:
        raise ValueError("table set %s has no tables of split %s (it has %s)" % (doc.get("name"), split, sorted(sets)))
    return Tables(sets[int(split)], doc)


def present_rows(tab):
    """The rows that exist: row 0, and every other row with a block of some kind."""
    rows = np.flatnonzero((np.asarray(tab["kind"]) != KIND_NONE).any(axis=1))
    return np.concatenate([[0], rows[rows > 0]]).astype(np.int64)


def take_rows(tab, rows):
    rows = np.asarray(rows, dtype=np.int64)
    return dict(tok=tab["tok"][rows], n=tab["n"][rows], col=tab["col"][rows], kind=tab["kind"][rows], frame_len=tab["frame_len"])


def inside(tab):
    """[R, ROW_LEN]: the slot holds a codon of the row's own segment."""
    return POS[None, :] < np.asarray(tab["n"])[:, BLOCK_OF]


def _placed(values, tab, fill):
    return layout.place(np.asarray(values)[None], np.asarray(tab["n"])[None], np.asarray(tab["col"])[None], fill)[0]


def render(tab, visible, target):
    """The grid a model reads and the truth of its target cells, in the stored placement.

    visible, target: boolean [R, ROW_LEN] in every row's own coordinates (a mask scheme's plan); a codon that is not
    visible shows as MASK. A target that is no residue (1..65) or has no column in the frame is dropped.
    Returns grid uint8 [R, ROW_LEN] and truth uint8 [R, ROW_LEN] (0 = not a target, else the token to predict)."""
    tok, kind, frame_len = np.asarray(tab["tok"]), np.asarray(tab["kind"]), np.asarray(tab["frame_len"]).astype(np.int64)
    R = tok.shape[0]
    cells = inside(tab)
    col = np.asarray(tab["col"])
    shown = cells & (col >= 0)
    if shown.any() and (col[shown] >= np.broadcast_to(frame_len[BLOCK_OF][None, :], (R, L))[shown]).any():
        raise ValueError("a stored column lies beyond the frame of its block")
    hidden = cells & ~np.asarray(visible, dtype=bool)
    placed = _placed(shown.astype(np.uint8), tab, 0).astype(bool)
    token = _placed(np.where(hidden, A.MASK, tok).astype(np.uint8), tab, A.PAD)
    want = np.asarray(target, dtype=bool) & hidden & (tok >= A.TARGET_MIN) & (tok <= A.TARGET_MAX)
    truth = _placed(np.where(want, tok, 0).astype(np.uint8), tab, 0)
    grid = np.full((R, L), A.PAD, dtype=np.uint8)
    for b, (_, s, e) in enumerate(SLICES):
        p = placed[:, s:e]
        span = (np.cumsum(p, axis=1) > 0) & (np.cumsum(p[:, ::-1], axis=1)[:, ::-1] > 0)
        g = np.where(p, token[:, s:e], np.where(span, A.GAP, A.PAD)).astype(np.uint8)
        empty = ~np.isin(kind[:, b], WITH_CODONS)
        g[empty] = A.PAD
        f = int(frame_len[b])
        g[empty, :f] = FILL[kind[empty, b]][:, None]
        g[:, f:] = A.PAD
        grid[:, s:e] = g
    truth[grid != A.MASK] = 0
    return grid, truth
