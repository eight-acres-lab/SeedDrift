"""The model passes of the readouts on ruler tables (package W18, the readouts' one implementation; W11 reads through it).

numpy only: it runs in the CI environment and in the model interpreter, and never imports torch, the ruler generators
(bench/rel/) or the generator of synthetic tables. A pass is one model evaluation of one table: which rows are kept and
which cells are visible and scored. Only cells of row 0 (the examined genome) are scored; every hidden cell of row 0 is
scored and no other cell is. Every random choice is the evaluator's 64-bit hash (evalh/conditions.hash_u01) on streams of
this module, so the plan is the same in every numpy and never depends on the content of a cell.

Passes (the protocol w18-v0 names them; t is a distance tier of the rows to the examined genome, from the ruler set's meta):
  T      the target block of row 0 scattered (rate 0.15, at least one cell), every row
  T-t<t> as T, the rows of tier t removed (one pass per tier present among the other rows)
  T0     as T, every other row removed (only when the table has other rows)
  C      the neighbour and relation blocks of row 0 scattered (rate 0.15, at least one cell), every row
  C-t<t> as C, the rows of tier t removed
  W      the whole target block of row 0 hidden, every row
  H1     the first half of the target block hidden (offsets below n // 2), the second half visible
  H2     the second half hidden (offsets from n // 2 on), the first half visible
  N      the whole target block hidden, every other row removed: the only pass of an arm trained without relatives
  P:<x>  pass x with every cell of the other rows hidden and unscored (shown as MASK): the presence pattern without the
         content, for x in T, T-t<t>, C, C-t<t>, W, H1, H2 (only when the table has other rows)

  plan(tab, row_tier, table_id, relatives, presence)   [Pass] of one table
  run(source, render, score, relatives, presence, ...) the passes of every table rendered and scored; per-cell arrays
  load_meta(run_dir, manifest)                         W18's meta of a ruler set (row tiers and the rest), checked
  tables_of(ts, meta, first, last)                     (table id, fields, row tiers) of a ruler set's tables, in order:
                                                       the `source` of run (ts: evalh/shards.TableSet of the set)
  save_cells(path, cells, **info), load_cells(path)    the per-cell file of one arm (CELL_INFO names what it records)
  plan_digest(source, relatives, presence)             the digest run gives, without a model (the readout stage checks
                                                       that an evaluation scored this module's plan)

`render(tab, visible, target) -> (grid, truth)` or `(grid, truth, ext)` is the caller's (the stored layout is
data/tables.render; the leak-free renderer when it exists; ext marks cells placed by extension); `score([(grid, truth[,
ext]), ...])` is the caller's model: for every grid three float arrays in the order of numpy.nonzero(truth): -log2
p(truth), -log2 p(amino-acid class of the truth) (evalh/score.amino_acid_classes), and the entropy of the distribution
over tokens 1..65 in bits.

The evaluation stage of package W6 (rel_w6_eval on a ruler set) calls run_ruler(scorer, tables, info, out_dir): its
scorer takes [dict(grid, truth[, ext])] and returns dict(bits, aa_bits, entropy, example) over every scored cell, in the
order of numpy.nonzero(truth) example by example; run_ruler writes <out_dir>/ruler_cells.npz (save_cells).
"""
from __future__ import annotations

import hashlib
import os
from typing import NamedTuple

import numpy as np

from seeddrift.train.rel import api
from seeddrift.train.rel.evalh import conditions as cond

VERSION = 1
SEED = cond.SEED
RATE = 0.15
STREAM_TARGET, STREAM_CONTEXT = 21, 22
MAX_TIERS = 8                                       # tiers 0..7 may be named; the protocol's edges give 0..5
SLICES = api.block_slices()
B, L = api.N_BLOCKS, api.ROW_LEN
TARGET_BLOCK = B - 1
T0_, T1_ = SLICES[TARGET_BLOCK][1], SLICES[TARGET_BLOCK][2]
BLOCK_OF = cond.BLOCK_OF_SLOT
OFFSET_OF = cond.OFFSET_OF_SLOT
CONTEXT_SLOT = BLOCK_OF != TARGET_BLOCK            # the neighbour and relation blocks
BASE = ("T", "T0", "C", "W", "H1", "H2", "N")
PRESENCE = "P:"
_TIERED = tuple("T-t%d" % t for t in range(MAX_TIERS)) + tuple("C-t%d" % t for t in range(MAX_TIERS))
PASS_NAMES = BASE + _TIERED + tuple(PRESENCE + x for x in ("T", "C", "W", "H1", "H2") + _TIERED)
PASS_ID = {name: k for k, name in enumerate(PASS_NAMES)}


class Pass(NamedTuple):
    name: str
    keep: np.ndarray        # (R,) bool: the rows given to the model (row 0 always)
    visible: np.ndarray     # (R, ROW_LEN) bool, rows' own coordinates
    target: np.ndarray      # (R, ROW_LEN) bool: hidden and scored (row 0 only)


def residue(tab):
    """(R, ROW_LEN) bool: the slots that hold a codon of the row's own segment."""
    return OFFSET_OF[None, :] < np.asarray(tab["n"])[:, BLOCK_OF]


def present(tab):
    """(R,) bool: row 0 and every other row with a block of a kind other than KIND_NONE."""
    out = (np.asarray(tab["kind"]) != api.KIND_NONE).any(axis=1)
    out[0] = True
    return out


def scattered(cells, table_id, stream, seed=SEED, rate=RATE):
    """(ROW_LEN,) bool: the cells of row 0 (cells: (ROW_LEN,) bool) whose hash is below the rate; at least one."""
    u = cond.hash_u01(seed, stream, np.array([int(table_id)]), np.arange(L)[None, :])[0]
    hide = cells & (u < rate)
    if cells.any() and not hide.any():
        hide[int(np.argmin(np.where(cells, u, 2.0)))] = True
    return hide


def tiers_present(row_tier, keep):
    """The tiers of the other rows kept, sorted. A kept other row without a tier (< 0) is an error of the ruler set."""
    t = np.asarray(row_tier)[1:][np.asarray(keep)[1:]]
    if (t < 0).any() or (t >= MAX_TIERS).any():
        raise ValueError("an other row without a distance tier in 0..%d" % (MAX_TIERS - 1))
    return sorted({int(x) for x in t})


def plan(tab, row_tier, table_id, relatives=True, presence=False, seed=SEED):
    """The passes of one table (fields tok, n, col, kind [R, ...]; row_tier (R,) the tier of every row to row 0, -1 for
    row 0 and for rows that do not exist). relatives False: the arm was trained without relatives; its only pass is N."""
    res = residue(tab)
    R = res.shape[0]
    pres = present(tab)
    tier = np.asarray(row_tier).astype(np.int64)
    n0 = int(np.asarray(tab["n"])[0, TARGET_BLOCK])
    if n0 < 1:
        raise ValueError("table %d: row 0 has no target block" % int(table_id))
    tslots = np.zeros(L, dtype=bool)
    tslots[T0_:T1_] = True
    target_cells = res[0] & tslots
    offs = OFFSET_OF
    others = pres.copy()
    others[0] = False
    only0 = np.zeros(R, dtype=bool)
    only0[0] = True

    def make(name, hide0, keep):
        hidden = np.zeros((R, L), dtype=bool)
        hidden[0] = hide0
        return Pass(name, keep.copy(), res & ~hidden, hidden)

    whole = target_cells.copy()
    if not relatives:
        return [make("N", whole, only0)]
    out = []
    hide_t = scattered(target_cells, table_id, STREAM_TARGET, seed)
    hide_c = scattered(res[0] & CONTEXT_SLOT, table_id, STREAM_CONTEXT, seed)
    tiers = tiers_present(tier, pres)
    first = target_cells & (offs < n0 // 2)
    second = target_cells & (offs >= n0 // 2)
    families = [("T", hide_t), ("C", hide_c)]
    for name, hide in families:
        out.append(make(name, hide, pres))
        for t in tiers:
            out.append(make("%s-t%d" % (name, t), hide, pres & ~(tier == t) | only0))
    if others.any():
        out.append(make("T0", hide_t, only0))
    out.append(make("W", whole, pres))
    if first.any() and second.any():
        out.append(make("H1", first, pres))
        out.append(make("H2", second, pres))
    if presence and others.any():
        for p in list(out):
            if p.name == "T0":
                continue
            vis = p.visible.copy()
            vis[1:] = False                                         # the other rows' cells hidden, never scored
            out.append(Pass(PRESENCE + p.name, p.keep, vis, p.target))
    for p in out:
        if p.name not in PASS_ID:
            raise ValueError("pass %s is not named in PASS_NAMES" % p.name)
        if p.target[1:].any() or not p.target[0].any() or (p.target & p.visible).any() or not p.keep[0]:
            raise AssertionError("pass %s of table %d scores cells outside row 0's hidden cells" % (p.name, int(table_id)))
    return out


def take(tab, keep):
    """The fields of the kept rows (frame_len and every field without a row axis carried as they are)."""
    keep = np.asarray(keep, dtype=bool)
    R = keep.shape[0]
    return {k: (np.asarray(v)[keep] if np.ndim(v) >= 1 and np.shape(v)[0] == R and k != "frame_len" else v) for k, v in tab.items()}


class PlanDigest:
    """sha256 of what a run scored: per table and pass the kept rows and the scored slots. Two evaluations with the same
    digest scored the same cells of the same tables under the same passes."""

    def __init__(self):
        self.parts = {}

    def add(self, table_id, p):
        h = hashlib.sha256(p.name.encode())
        h.update(np.flatnonzero(p.keep).astype("<i4").tobytes())
        h.update(np.flatnonzero(p.target[0]).astype("<i4").tobytes())
        self.parts[(int(table_id), p.name)] = h.hexdigest()

    def hexdigest(self):
        h = hashlib.sha256(("rel-w18-passes v%d seed=%d rate=%.6f\n" % (VERSION, SEED, RATE)).encode())
        for k in sorted(self.parts):
            h.update(("%d %s %s\n" % (k[0], k[1], self.parts[k])).encode())
        return h.hexdigest()


def run(source, render, score, relatives=True, presence=False, max_rows=64, seed=SEED, progress=None):
    """Every pass of every table of `source` rendered and scored.

    source: iterable of (table_id, tab, row_tier); tab holds tok, n, col, kind (rows first) and frame_len.
    Returns dict of per-cell arrays (one entry per scored cell): table (int64), pass_id (int16, into PASS_NAMES), slot
    (int16, row 0's own slot), truth (uint8), bits, aa_bits, entropy (float32); and pass_names, digest, tables, passes."""
    cols = {k: [] for k in ("table", "pass_id", "slot", "truth", "bits", "aa_bits", "entropy")}
    digest = PlanDigest()
    pending, rows, done = [], 0, dict(tables=0, passes=0)

    def flush():
        nonlocal pending, rows
        if not pending:
            return
        got = score([(g, t) if e is None else (g, t, e) for _, _, g, t, e in pending])
        if len(got) != len(pending):
            raise ValueError("the scorer returned %d results for %d grids" % (len(got), len(pending)))
        for (tid, pid, grid, truth, _), parts in zip(pending, got):
            r, s = np.nonzero(truth)
            parts = [np.asarray(x, dtype=np.float64).reshape(-1) for x in parts]
            if len(parts) != 3 or any(x.shape[0] != r.shape[0] for x in parts):
                raise ValueError("the scorer must return bits, aa_bits and entropy for every target cell")
            if (r != 0).any():
                raise AssertionError("a scored cell outside row 0")
            cols["table"].append(np.full(r.shape[0], tid, dtype=np.int64))
            cols["pass_id"].append(np.full(r.shape[0], pid, dtype=np.int16))
            cols["slot"].append(s.astype(np.int16))
            cols["truth"].append(truth[r, s].astype(np.uint8))
            for k, x in zip(("bits", "aa_bits", "entropy"), parts):
                cols[k].append(x.astype(np.float32))
        pending, rows = [], 0

    for tid, tab, row_tier in source:
        for p in plan(tab, row_tier, tid, relatives, presence, seed):
            sub = take(tab, p.keep)
            got = render(sub, p.visible[p.keep], p.target[p.keep])
            grid, truth, ext = (tuple(got) + (None,))[:3]
            want = int((p.target[0] & (np.asarray(tab["col"])[0] >= 0)).sum())
            if int((truth > 0).sum()) != want:
                raise ValueError("table %d pass %s: the grid scores %d cells, the plan %d" % (int(tid), p.name, int((truth > 0).sum()), want))
            digest.add(tid, p)
            if rows and rows + grid.shape[0] > max_rows:
                flush()
            pending.append((int(tid), PASS_ID[p.name], grid, truth, ext))
            rows += grid.shape[0]
            done["passes"] += 1
        done["tables"] += 1
        if progress and done["tables"] % 100 == 0:
            progress("%d tables, %d passes" % (done["tables"], done["passes"]))
    flush()
    out = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.float32 if k in ("bits", "aa_bits", "entropy") else np.int64)) for k, v in cols.items()}
    out.update(pass_names=np.array(PASS_NAMES), digest=digest.hexdigest(), tables=done["tables"], passes=done["passes"])
    return out


META_FILE = "w18_meta.npz"
FIELDS = ("tok", "n", "col", "kind", "frame_len")


def load_meta(run_dir, manifest):
    """{name: array over the tables of the set} of the W18 meta file a ruler set's manifest lists (role meta), checked
    against the size and sha256 the manifest gives it. Never a model input: row_tier chooses the rows a pass removes."""
    for e in manifest.get("files") or []:
        if e.get("role") == "meta" and os.path.basename(str(e.get("path", ""))) == META_FILE:
            p = os.path.join(run_dir, e["path"])
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 22), b""):
                    h.update(b)
            if os.path.getsize(p) != e["bytes"] or h.hexdigest() != e["sha256"]:
                raise ValueError("%s is not the file the ruler set's manifest names" % e["path"])
            with np.load(p) as z:
                return {k: z[k] for k in z.files}
    raise ValueError("the manifest lists no %s (not a ruler set of package W18)" % META_FILE)


def tables_of(ts, meta, first=0, last=None, chunk=64):
    """(table id, {tok, n, col, kind, frame_len} of one table, its row tiers) for the tables first..last-1 of a ruler
    set (ts: an evalh/shards.TableSet; meta: load_meta), in storage order; the table id is the position in the set."""
    last = ts.tables if last is None else int(last)
    if meta["row_tier"].shape[0] != ts.tables:
        raise ValueError("the meta file has %d tables, the set %d" % (meta["row_tier"].shape[0], ts.tables))
    for a in range(int(first), last, chunk):
        b = min(a + chunk, last)
        got = ts.load(a, b, FIELDS)
        for k in range(b - a):
            yield a + k, {f: got[f][k] for f in FIELDS}, meta["row_tier"][a + k]


CELL_FORMAT = "rel-w18-cells-1"
CELL_INFO = ("arm", "weights", "relatives", "presence", "layout", "render_version", "tableset_sha256", "run", "commit")


def save_cells(path, cells, **info):
    """One arm's per-cell arrays (run's result) as an uncompressed .npz, with what produced them: arm (the variant
    name), weights (final, best, or untrained), relatives, presence, layout (stored or leakfree), render_version,
    tableset_sha256 (the ruler set's manifest), run and commit (the evaluation's)."""
    missing = [k for k in CELL_INFO if k not in info]
    if missing:
        raise ValueError("save_cells needs %s" % ", ".join(missing))
    arrays = {k: np.asarray(cells[k]) for k in ("table", "pass_id", "slot", "truth", "bits", "aa_bits", "entropy")}
    head = dict(format=CELL_FORMAT, version=VERSION, digest=str(cells["digest"]), tables=int(cells["tables"]), passes=int(cells["passes"]),
                pass_names=[str(x) for x in cells["pass_names"]], **{k: info[k] for k in CELL_INFO})
    import json
    with open(path + ".part", "wb") as f:
        np.savez(f, info=np.array(json.dumps(head, sort_keys=True)), **arrays)
    os.replace(path + ".part", path)
    return head


def load_cells(path):
    """(per-cell arrays with pass_names, the head save_cells wrote)."""
    import json
    with np.load(path) as z:
        head = json.loads(str(z["info"]))
        if head.get("format") != CELL_FORMAT:
            raise ValueError("%s is not a per-cell file of format %s" % (path, CELL_FORMAT))
        cells = {k: z[k] for k in z.files if k != "info"}
    cells["pass_names"] = np.array(head["pass_names"])
    cells["digest"] = head["digest"]
    return cells, head


def plan_digest(source, relatives=True, presence=False, seed=SEED):
    """(digest, tables, passes) of the plan of every table of source, as run computes it."""
    d, n, k = PlanDigest(), 0, 0
    for tid, tab, row_tier in source:
        for p in plan(tab, row_tier, tid, relatives, presence, seed):
            d.add(tid, p)
            k += 1
        n += 1
    return d.hexdigest(), n, k


RULER_CELLS = "ruler_cells.npz"
INFO_KEYS = ("arm", "weights", "relatives", "layout", "render_version", "run", "commit")


def flat_scorer(scorer):
    """The score callable of run from a scorer of W6's form: [dict(grid, truth[, ext])] -> dict(bits, aa_bits, entropy,
    example), the cells of every example in the order of numpy.nonzero(truth), examples in order."""
    def score(items):
        got = scorer([dict(grid=x[0], truth=x[1]) if len(x) == 2 else dict(grid=x[0], truth=x[1], ext=x[2]) for x in items])
        ex = np.asarray(got["example"]).reshape(-1)
        parts = [np.asarray(got[k], dtype=np.float64).reshape(-1) for k in ("bits", "aa_bits", "entropy")]
        if any(x.shape != ex.shape for x in parts) or (np.diff(ex) < 0).any():
            raise ValueError("the scorer's cells are not in example order, or its arrays differ in length")
        out = []
        for k, x in enumerate(items):
            m = ex == k
            if int(m.sum()) != int((np.asarray(x[1]) > 0).sum()):
                raise ValueError("the scorer scored %d cells of example %d, its truth has %d" % (int(m.sum()), k, int((np.asarray(x[1]) > 0).sum())))
            out.append(tuple(p[m] for p in parts))
        return out
    return score


def run_ruler(scorer, tables, info, out_dir, render=None, presence=False, max_rows=32, progress=None):
    """One arm on a ruler set, for the evaluation stage: every table of `tables` (data/tables.Tables of the set, kind
    ruler; its TableSet and manifest) through the passes of the arm (info["relatives"] False: pass N only; presence: the
    presence passes too), rendered by `render` (default: the stored layout of data/tables.render, for info["layout"] ==
    "stored") and scored by `scorer` (W6's form, flat_scorer). info: arm, weights (final, best or untrained), relatives,
    layout, render_version, run, commit, and root (the run directory the ruler set's files are under). Writes
    <out_dir>/ruler_cells.npz; returns its head (save_cells)."""
    missing = [k for k in INFO_KEYS + ("root",) if k not in info]
    if missing:
        raise ValueError("run_ruler needs info %s" % ", ".join(missing))
    man = tables.manifest
    if man.get("kind") != api.TABLE_RULER:
        raise ValueError("run_ruler reads ruler sets only, not kind %r" % man.get("kind"))
    if render is None:
        if info["layout"] != api.LAYOUT_STORED:
            raise ValueError("layout %r needs its renderer" % info["layout"])
        from seeddrift.train.rel.data import tables as T
        render = T.render
    meta = load_meta(info["root"], man)
    cells = run(tables_of(tables.ts, meta), render, flat_scorer(scorer), relatives=bool(info["relatives"]), presence=bool(presence),
                max_rows=max_rows, progress=progress)
    os.makedirs(out_dir, exist_ok=True)
    return save_cells(os.path.join(out_dir, RULER_CELLS), cells, presence=bool(presence and info["relatives"]), tableset_sha256=str(man.get("sha256", "")),
                      **{k: info[k] for k in INFO_KEYS})


def count_passes(tab, row_tier, relatives=True, presence=False):
    """How many passes plan gives a table (for the cost estimate of a run)."""
    return len(plan(tab, row_tier, 0, relatives, presence))
