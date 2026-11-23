"""Tables of the relational line as the trainer reads them, and the grid a model reads (numpy only: loads in the CI
environment and in the model interpreter).

Reading goes through the evaluator's reader (seeddrift/train/rel/evalh/shards.py: the files a table-set manifest
lists, <folder>/<shard>.<field>.npy, the shards of a folder in name order, table t = its position over them; that
position is the table id of the evaluation conditions). The grid is package W3's renderer's (tables/render.py; render
here is its stored layout, other layouts are rendered by it directly); in the stored placement of table version 0,
frame row 0 (grid column = block start + col; a codon with col = -1 is not shown) its tokens are:
  - a codon of a row at its column, as itself; a hidden one as MASK;
  - inside a row's aligned span in a block (from its first to its last placed column) a column without a codon of the
    row: GAP; outside the span: PAD;
  - a block whose kind has no codons (NOT_SEARCHED, TRUNCATED, ABSENT): its token over the frame row's length;
  - beyond the frame row's length, and in a block of KIND_NONE: PAD. PAD cells take no part in attention (the model
    masks them as keys); a row that does not exist (KIND_NONE in every block) is never in a grid.
The model never reads metadata (group, identity, genome): those stay with the sampler and the report.
Under the stored placement a hidden stretch of the examined row still has its relatives placed by the hidden
residues, so bits read through it for a hidden stretch are biased low (optimistic; layout "stored").
"""
from __future__ import annotations

import json

import numpy as np

from seeddrift.train.rel import api as A
from seeddrift.train.rel.evalh import shards
from seeddrift.train.rel.tables import render as _RD       # package W3's renderer (numpy only): its version is recorded

READER_VERSION = "2"
RENDER_VERSION = "stored-w3-%s" % _RD.VERSION   # the stored layout of package W3's renderer (tables/render.py), with the
                                                # renderer's version, so every run's versions record it (protocol)
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

MODEL_FIELDS = ("tok", "n", "col", "kind", "frame_len", "frame_row", "genome_of_row")   # genome_of_row: tables v1 (v0: the default)
FRAME_META = ("genome", "row_gene", "row_first")         # the meta fields the online frame layout of the whole gene reads


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

    def meta(self, i, fields=FRAME_META):
        """Metadata fields of table i (never a model input; the online frame layout of package W3 reads them)."""
        got = self.ts.load(int(i), int(i) + 1, tuple(fields), verify=False)
        return {f: got[f][0] for f in fields}

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
    """The table with the rows `rows` (in that order); frame_row, where the table has it, renumbered to the rows kept
    (-1 where a block's frame row was left out: the renderer then frames the block on row 0)."""
    rows = np.asarray(rows, dtype=np.int64)
    out = dict(tok=tab["tok"][rows], n=tab["n"][rows], col=tab["col"][rows], kind=tab["kind"][rows], frame_len=tab["frame_len"])
    if "frame_row" in tab:
        where = {int(r): k for k, r in enumerate(rows.tolist())}
        out["frame_row"] = np.asarray([where.get(int(f), -1) for f in np.asarray(tab["frame_row"]).reshape(-1)], dtype=np.int64)
    if "genome_of_row" in tab:
        out["genome_of_row"] = np.asarray(tab["genome_of_row"])[rows]
    return out


def take_meta(meta, rows):
    """The metadata of the rows kept (every field with a leading axis of rows)."""
    rows = np.asarray(rows, dtype=np.int64)
    return {k: np.asarray(v)[rows] for k, v in meta.items()}


def inside(tab):
    """[R, ROW_LEN]: the slot holds a codon of the row's own segment."""
    return POS[None, :] < np.asarray(tab["n"])[:, BLOCK_OF]


def render(tab, visible, target):
    """The grid a model reads and the truth of its target cells, in the stored placement: package W3's renderer
    (tables/render.py, layout stored), the one implementation; it matched this module's own placement cell for cell on
    the real tables before replacing it (1,800 training renders and the evaluator's plans of every condition).

    visible, target: boolean [R, ROW_LEN] in every row's own coordinates (a mask scheme's plan); a codon that is not
    visible shows as MASK. A target that is no residue (1..65) or has no column in the frame is dropped.
    Returns grid uint8 [R, ROW_LEN] and truth uint8 [R, ROW_LEN] (0 = not a target, else the token to predict)."""
    from seeddrift.train.rel.tables import render as RD
    frame_len = np.asarray(tab["frame_len"]).astype(np.int64)
    col = np.asarray(tab["col"])
    shown = inside(tab) & (col >= 0)
    if shown.any() and (col[shown] >= np.broadcast_to(frame_len[BLOCK_OF][None, :], col.shape)[shown]).any():
        raise ValueError("a stored column lies beyond the frame of its block")
    grid, truth, _ = RD.render(tab, visible, target, layout=RD.STORED)
    return grid, truth
