"""The model passes of the readouts on ruler tables (package W18, the readouts' one implementation; W11 reads through it).

numpy only: it runs in the CI environment and in the model interpreter, and never imports torch, the ruler generators
(bench/rel/) or the generator of synthetic tables. A pass is one model evaluation of one table: which rows are kept and
which cells are visible and scored. Only cells of row 0 (the examined genome) are scored; every hidden cell of row 0 is
scored and no other cell is. Every random choice is the evaluator's 64-bit hash (evalh/conditions.hash_u01) on streams of
this module, so the plan is the same in every numpy and never depends on the content of a cell. The hash takes a mask id
(the ruler set's `parent`: the triple, shared by its three variants and its scenario copies) and the slot, so the three
tables of a triple, whose target blocks have the same length, hide exactly the same cells.

Passes (the protocol w18-v0 names them; t is a distance tier of the rows to the examined genome, from the ruler set's meta;
a self-copy row of protocol w18-v3, row_tier SELF_TIER, has no tier: no tier pass removes it, "every other row removed"
removes it, a presence pass hides it):
  T      the target block of row 0 scattered (rate 0.15, at least one cell), every row
  T-t<t> as T, the rows of tier t removed (one pass per tier present among the other rows)
  T0     as T, every other row removed (only when the table has other rows)
  C      the neighbour and relation blocks of row 0 scattered (rate 0.15, at least one cell), every row; not made for a
         table whose row 0 has no codon there (a genome of very few genes): its source shift is undefined
  C-t<t> as C, the rows of tier t removed
  W      the whole target block of row 0 hidden, every row
  H1     the first half of the target block hidden (offsets below n // 2), the second half visible
  H2     the second half hidden (offsets from n // 2 on), the first half visible
  W0, H10, H20   W, H1 and H2 with every other row removed (the gain without relatives; only when the table has other rows)
  N      the whole target block hidden, every other row removed: the only pass of an arm trained without relatives
  P:<x>  pass x with every cell of the other rows hidden and unscored (shown as MASK): the presence pattern without the
         content, for x in T, T-t<t>, C, C-t<t>, W, H1, H2 (only when the table has other rows)

  plan(tab, row_tier, table_id, relatives, presence)   [Pass] of one table
  run(source, render, score, relatives, presence, ...) the passes of every table rendered and scored; per-cell arrays
  load_meta(run_dir, manifest)                         W18's meta of a ruler set (row tiers and the rest), checked
  tables_of(ts, meta, first, last)                     (table id, fields, row tiers) of a ruler set's tables, in order:
                                                       the `source` of run (ts: evalh/shards.TableSet of the set)
  save_cells(path, cells, **info), load_cells(path)    the per-cell file of one arm (CELL_INFO names what it records)
  adapt_rows, adapt_table, adapt_arrays                the adaptation of implant tables of version 1 for arms trained on
                                                       tables of version 0 (adapt_v0): see adapt_rows
  plan_digest(source, relatives, presence)             the digest run gives, without a model (the readout stage checks
                                                       that an evaluation scored this module's plan)
  family_of, family_passes, strict_placement,          the strict offline rendering of protocol w18-v3 (stage
  fourth_test, offline_arrays, offline_put,            rel_w18_offline builds it with W3's aligner; run_ruler's offline
  offline_place, load_offline                          renders a leak-free reading on it)

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
STREAM_TARGET, STREAM_CONTEXT, STREAM_ADAPT, STREAM_FOURTH = 21, 22, 23, 24
V0_ROWS = 16                                        # the rows an arm trained on tables of version 0 has seen at most
MAX_TIERS = 8                                       # tiers 0..7 may be named; the protocol's edges give 0..5
SELF_TIER = -2                                      # row_tier of a self-copy row (A itself, protocol w18-v3): no tier, never removed
SLICES = api.block_slices()
B, L = api.N_BLOCKS, api.ROW_LEN
TARGET_BLOCK = B - 1
T0_, T1_ = SLICES[TARGET_BLOCK][1], SLICES[TARGET_BLOCK][2]
BLOCK_OF = cond.BLOCK_OF_SLOT
OFFSET_OF = cond.OFFSET_OF_SLOT
CONTEXT_SLOT = BLOCK_OF != TARGET_BLOCK            # the neighbour and relation blocks
BASE = ("T", "T0", "C", "W", "H1", "H2", "N", "W0", "H10", "H20")
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
    """The tiers of the other rows kept, sorted; self-copy rows (SELF_TIER) have none. A kept other row without a tier
    (another value below 0) is an error of the ruler set."""
    t = np.asarray(row_tier)[1:][np.asarray(keep)[1:]]
    t = t[t != SELF_TIER]
    if (t < 0).any() or (t >= MAX_TIERS).any():
        raise ValueError("an other row without a distance tier in 0..%d" % (MAX_TIERS - 1))
    return sorted({int(x) for x in t})


def plan(tab, row_tier, table_id, relatives=True, presence=False, seed=SEED, mask_id=None):
    """The passes of one table (fields tok, n, col, kind [R, ...]; row_tier (R,) the tier of every row to row 0, -1 for
    row 0 and for rows that do not exist). relatives False: the arm was trained without relatives; its only pass is N.
    mask_id: what the scattered masks are hashed on (default the table id)."""
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
    mid = table_id if mask_id is None else mask_id
    hide_t = scattered(target_cells, mid, STREAM_TARGET, seed)
    hide_c = scattered(res[0] & CONTEXT_SLOT, mid, STREAM_CONTEXT, seed)
    tiers = tiers_present(tier, pres)
    first = target_cells & (offs < n0 // 2)
    second = target_cells & (offs >= n0 // 2)
    families = [("T", hide_t)] + ([("C", hide_c)] if hide_c.any() else [])     # no C passes when row 0 has no context codon
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
    base = list(out)
    if others.any():
        out.append(make("W0", whole, only0))
        if first.any() and second.any():
            out.append(make("H10", first, only0))
            out.append(make("H20", second, only0))
    if presence and others.any():
        for p in base:
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
    """The fields of the kept rows (frame_len and every field without a row axis carried as they are); frame_row, where
    given, renumbered to the kept rows (a block's frame row other than row 0 must be kept)."""
    keep = np.asarray(keep, dtype=bool)
    R = keep.shape[0]
    out = {k: (np.asarray(v)[keep] if np.ndim(v) >= 1 and np.shape(v)[0] == R and k not in ("frame_len", "frame_row") else v)
           for k, v in tab.items()}
    if "frame_row" in tab:
        fr = np.asarray(tab["frame_row"]).astype(np.int64).reshape(-1)
        if not keep[np.clip(fr, 0, R - 1)].all():
            raise ValueError("a block's frame row is not among the kept rows")
        out["frame_row"] = (np.cumsum(keep) - 1)[np.clip(fr, 0, R - 1)].astype(np.int8)
    return out


# ---- the strict offline rendering (protocol w18-v3): W3's aligner, one placement per mask family ----
FAMILIES = ("T", "C", "H1", "H2", "W")
ROW0_ONLY = ("T0", "W0", "H10", "H20", "N")
PLACE = "_place"                                    # the key of a table's offline placements in the tables of tables_of


def family_of(name):
    """The mask family whose placement renders a pass (T, C, H1, H2, W); None for a pass of row 0 alone (its placement
    is the stored one: row 0 keeps its own columns). A presence pass and a tier pass take their base pass's family."""
    base = name[len(PRESENCE):] if name.startswith(PRESENCE) else name
    if base in ROW0_ONLY:
        return None
    fam = base.split("-t")[0]
    if fam not in FAMILIES:
        raise ValueError("pass %s has no mask family" % name)
    return fam


def strict_placement(tab, family, visible):
    """dict(col, frame_row, frame_len) of one table under the visibility of one mask family, by package W3's functions
    (seeddrift/train/rel/tables/align.py; no second aligner): W lays the target block on its frame relative
    (align.frame_layout); T, C, H1 and H2 align every other row to row 0 again wherever row 0 has hidden residues, those
    residues unknown (align.strict). Every other row is aligned to row 0 or to the frame relative alone, so a pass that
    removes rows keeps the placement of the rows it keeps."""
    from seeddrift.train.rel.tables import align as AL
    if family == "W":
        col, fr, fl = AL.frame_layout(tab)
    elif family in FAMILIES:
        col = AL.strict(tab, visible)
        fr = np.array(tab.get("frame_row", np.zeros(B)), dtype=np.int8, copy=True).reshape(-1)
        fl = np.array(tab["frame_len"], dtype=np.int16, copy=True)
    else:
        raise ValueError("unknown mask family %r" % family)
    return dict(col=np.asarray(col, dtype=np.int16), frame_row=np.asarray(fr, dtype=np.int8), frame_len=np.asarray(fl, dtype=np.int16))


def family_passes(tab, row_tier, table_id, mask_id=None, seed=SEED):
    """{family: its base pass (every row kept)} of the families the plan of one table makes."""
    ps = {p.name: p for p in plan(tab, row_tier, table_id, True, False, seed, mask_id)}
    return {f: ps[f] for f in FAMILIES if f in ps}


def leakfree(tab, visible, target):
    """W3's leak-free renderer (seeddrift/train/rel/tables/render.py): grid, truth, ext."""
    from seeddrift.train.rel.tables import render as RD
    return RD.render(tab, visible, target, RD.LEAKFREE)


def fourth_test(tab, family, p, placement, table_id, seed=SEED):
    """W3's fourth validity test of one table and mask family: the hidden residues replaced by random codons (drawn by
    the evaluator's hash on the table and family, the same in every numpy), every stored position of a block where row 0
    has a hidden residue erased, the placement made again: the leak-free grid and its extension marks must be the same
    cell for cell. Returns (passed, grid, ext) of the placement given."""
    t1 = dict(tab, **placement)
    grid, _, ext = leakfree(t1, p.visible, p.target)
    hidden = residue(tab) & ~p.visible
    u = cond.hash_u01(seed, STREAM_FOURTH, np.array([int(table_id)]), np.arange(hidden.size)[None, :] + FAMILIES.index(family) * hidden.size)[0]
    rand = (1 + np.floor(u * 65)).astype(np.int64).reshape(hidden.shape)            # tokens 1..65
    tok = np.where(hidden, rand, np.asarray(tab["tok"])).astype(np.uint8)
    gone = np.array(tab["col"], dtype=np.int16, copy=True)
    for b, (_, s0, e0) in enumerate(SLICES):
        f0 = int(np.asarray(tab["frame_len"])[b])
        if f0 and not p.visible[0, s0:s0 + f0].all():
            gone[1:, s0:e0] = -1
    other = dict(tab, tok=tok, col=gone)
    t2 = dict(other, **strict_placement(other, family, p.visible))
    g2, _, e2 = leakfree(t2, p.visible, p.target)
    return bool(np.array_equal(grid, g2) and np.array_equal(ext, e2)), grid, ext



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

    source: iterable of (table_id, tab, row_tier[, mask_id[, presence]]); tab holds tok, n, col, kind (rows first) and
    frame_len; a fifth element overrides `presence` for that table. tab[PLACE] (tables_of with offline placements): the
    strict offline rendering of protocol w18-v3, {family: dict(col, frame_row, frame_len)}; every pass with other rows is
    then rendered on its family's placement (render: the leak-free renderer), and a presence pass on its base pass's
    visibility with every codon of the other rows shown as MASK after the layout (their pattern kept, their content gone;
    a frame relative's codons too, unlike render's control "presence", which spares a block's frame row).
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

    for item in source:
        tid, tab, row_tier = item[:3]
        mid = item[3] if len(item) > 3 else None
        pres = bool(item[4]) if len(item) > 4 else presence
        place = tab.get(PLACE)
        if place is not None:
            tab = {k: v for k, v in tab.items() if k != PLACE}
            res = residue(tab)
        for p in plan(tab, row_tier, tid, relatives, pres, seed, mid):
            fam = family_of(p.name)
            masked = place is not None and p.name.startswith(PRESENCE)
            vis = p.visible
            if place is not None and fam is not None and p.keep[1:].any():
                if fam not in place:
                    raise ValueError("table %d: no offline placement of mask family %s" % (int(tid), fam))
                sub = take(dict(tab, **place[fam]), p.keep)
            else:
                sub = take(tab, p.keep)
            if masked:                                              # strict: the base pass's layout, the other rows' codons then MASK
                vis = p.visible.copy()
                vis[1:] = res[1:]
            got = render(sub, vis[p.keep], p.target[p.keep])
            grid, truth, ext = (tuple(got) + (None,))[:3]
            if masked:
                g = grid[1:]
                g[(g >= api.TARGET_MIN) & (g <= api.TARGET_MAX)] = api.MASK
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

# ---- the offline placements of a ruler set (stage rel_w18_offline writes them, run_ruler reads them) ----
OFFLINE_DIR = os.path.join("data", "rel_w18", "offline")
OFFLINE_FILE = "offline.json"
OFFLINE_FORMAT = "rel-w18-offline-1"
LT, LC = T1_ - T0_, T0_                             # the target block's slots, the context blocks' slots


def offline_arrays(n_tables, rows):
    """Empty arrays of the offline placements: has [N, families]; <family>.col of the slots the family aligns again
    (T, H1, H2, W: the target block [N, R, LT]; C: the context blocks [N, R, LC]); W.frame_row, W.frame_len [N] (the
    target block's frame relative and frame length)."""
    N, R = int(n_tables), int(rows)
    out = {"has": np.zeros((N, len(FAMILIES)), dtype=bool), "W.frame_row": np.zeros(N, dtype=np.int8), "W.frame_len": np.zeros(N, dtype=np.int16)}
    for f in FAMILIES:
        out[f + ".col"] = np.full((N, R, LC if f == "C" else LT), -1, dtype=np.int16)
    return out


def offline_put(arrays, t, family, placement, tab):
    """Table t's placement of one family into the arrays; every slot outside the family's blocks must be as stored."""
    col, stored = np.asarray(placement["col"]), np.asarray(tab["col"])
    inside = np.zeros(L, dtype=bool)
    inside[(slice(0, T0_) if family == "C" else slice(T0_, T1_))] = True
    if not (col[:, ~inside] == stored[:, ~inside]).all():
        raise AssertionError("table %d: family %s moved a cell outside its blocks" % (int(t), family))
    fr, fl = np.asarray(placement["frame_row"]).reshape(-1), np.asarray(placement["frame_len"]).reshape(-1)
    other = np.arange(B) != TARGET_BLOCK
    if family != "W" and (fr.any() or (fl != np.asarray(tab["frame_len"])).any()) or family == "W" and (fr[other].any() or (fl[other] != np.asarray(tab["frame_len"])[other]).any()):
        raise AssertionError("table %d: family %s changed a frame outside the target block's frame relative" % (int(t), family))
    arrays["has"][t, FAMILIES.index(family)] = True
    arrays[family + ".col"][t] = col[:, inside]
    if family == "W":
        arrays["W.frame_row"][t], arrays["W.frame_len"][t] = int(fr[TARGET_BLOCK]), int(fl[TARGET_BLOCK])


def offline_place(arrays, t, tab):
    """{family: dict(col, frame_row, frame_len)} of table t (tab: its stored fields), from the offline arrays."""
    col0 = np.asarray(tab["col"])
    if col0.shape[0] != arrays["T.col"].shape[1]:
        raise ValueError("table %d has %d rows, the offline placements %d" % (int(t), col0.shape[0], arrays["T.col"].shape[1]))
    out = {}
    for k, f in enumerate(FAMILIES):
        if not arrays["has"][t, k]:
            continue
        col = np.array(col0, dtype=np.int16, copy=True)
        fr = np.array(tab.get("frame_row", np.zeros(B)), dtype=np.int8, copy=True).reshape(-1)
        fl = np.array(tab["frame_len"], dtype=np.int16, copy=True)
        if f == "C":
            col[:, :T0_] = arrays["C.col"][t]
        else:
            col[:, T0_:T1_] = arrays[f + ".col"][t]
        if f == "W":
            fr[TARGET_BLOCK], fl[TARGET_BLOCK] = arrays["W.frame_row"][t], arrays["W.frame_len"][t]
        out[f] = dict(col=col, frame_row=fr, frame_len=fl)
    return out


def shown_cells(tab, row_tier, table_id, mask_id, place):
    """(on_t [R], on_c [R]): the cells of every row the leak-free renderer shows on the target block under pass T and on
    the neighbour and relation blocks under pass C, each on its offline placement (place: offline_place); the cells
    coverage_shift counts for the strict reading (protocol w18-v3 version 4). A family the plan does not make counts 0."""
    from seeddrift.train.rel.tables import render as RD
    R = np.asarray(tab["tok"]).shape[0]
    on_t, on_c = np.zeros(R, dtype=np.int64), np.zeros(R, dtype=np.int64)
    fams = family_passes(tab, row_tier, table_id, mask_id)
    for fam, out, lo, hi in (("T", on_t, T0_, T1_), ("C", on_c, 0, T0_)):
        if fam not in fams or fam not in place:
            continue
        slot, _, _ = RD.placement(dict(tab, **place[fam]), fams[fam].visible, RD.LEAKFREE)
        out[:] = ((slot >= lo) & (slot < hi)).sum(axis=1)
    return on_t, on_c


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_offline(run_dir, ruler_sha256, passed=True):
    """(arrays, head, sha256 of offline.json) of the offline placements a run of stage rel_w18_offline wrote, every file
    checked against the size and sha256 its manifest gives; the manifest must name the ruler set (ruler_sha256: the
    sha256 of the ruler set's manifest, as run_ruler records it) and this module's plan. passed (a reading): the run
    passed and its record of stage rel_w18_offline lists offline.json with this content (the stage's own gate reads
    its run before it has passed)."""
    import json
    p = os.path.join(run_dir, OFFLINE_DIR, OFFLINE_FILE)
    if passed:
        rp = os.path.join(run_dir, "record.json")
        rec = {}
        if os.path.exists(rp):
            with open(rp) as f:
                rec = json.load(f)
        if rec.get("status") != "passed":
            raise ValueError("the offline placements of %s: the run has not passed" % os.path.basename(os.path.normpath(run_dir)))
        listed = [o for st in rec.get("stages") or [] for o in st.get("outputs") or [] if os.path.normpath(o.get("path", "")) == os.path.join(OFFLINE_DIR, OFFLINE_FILE)]
        if not listed or any(o.get("sha256") != _sha256(p) for o in listed):
            raise ValueError("%s is not the file the run's record of stage rel_w18_offline lists" % OFFLINE_FILE)
    with open(p) as f:
        head = json.load(f)
    if head.get("format") != OFFLINE_FORMAT:
        raise ValueError("%s is not a file of offline placements (%s)" % (p, OFFLINE_FORMAT))
    if str((head.get("ruler") or {}).get("sha256")) != str(ruler_sha256):
        raise ValueError("the offline placements are of another ruler set")
    if head.get("plan") != dict(version=VERSION, seed=SEED, rate=RATE):
        raise ValueError("the offline placements were made for another plan of the passes")
    if int((head.get("fourth_test") or {}).get("failures", -1)) != 0:
        raise ValueError("the offline placements did not pass W3's fourth validity test")
    arrays = {}
    for e in head["files"]:
        q = os.path.join(run_dir, e["path"])
        if os.path.getsize(q) != e["bytes"] or _sha256(q) != e["sha256"]:
            raise ValueError("%s is not the file the offline manifest names" % e["path"])
        arrays[e["name"]] = np.load(q, mmap_mode="r")
    want = set(offline_arrays(0, 1))
    if set(arrays) != want:
        raise ValueError("the offline placements hold %s, not %s" % (sorted(arrays), sorted(want)))
    return arrays, head, _sha256(p)


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


def tables_of(ts, meta, first=0, last=None, chunk=64, scenarios=None, presence=None, presence_scenarios=None, adapt=None, offline=None):
    """(table id, {tok, n, col, kind, frame_len} of one table, its row tiers, its mask id) for the tables first..last-1
    of a ruler set (ts: an evalh/shards.TableSet; meta: load_meta), in storage order; the table id is the position in
    the set, the mask id the triple it belongs to (meta parent). scenarios: only the tables of these scenarios (meta
    scenario; None: all). presence (not None): a fifth element, whether the table has the presence passes (presence, and
    its scenario among presence_scenarios unless that is None). adapt (rows): every table adapted for an arm trained on
    tables of version 0 (adapt_table). offline (the arrays of load_offline): every table carries its offline placements
    under PLACE (run renders on them)."""
    if offline is not None and adapt:
        raise ValueError("an adapted reading renders the stored layout; the offline placements are for the leak-free one")
    last = ts.tables if last is None else int(last)
    want = None if scenarios is None else {int(x) for x in scenarios}
    pwant = None if presence_scenarios is None else {int(x) for x in presence_scenarios}
    if meta["row_tier"].shape[0] != ts.tables:
        raise ValueError("the meta file has %d tables, the set %d" % (meta["row_tier"].shape[0], ts.tables))
    for a in range(int(first), last, chunk):
        b = min(a + chunk, last)
        got = ts.load(a, b, FIELDS)
        for k in range(b - a):
            sc = int(meta["scenario"][a + k])
            if want is not None and sc not in want:
                continue
            tab, tier = {f: got[f][k] for f in FIELDS}, meta["row_tier"][a + k]
            if adapt:
                tab, tier, _ = adapt_table(tab, tier, int(meta["parent"][a + k]), int(adapt))
            if offline is not None:
                tab[PLACE] = offline_place(offline, a + k, tab)
            item = (a + k, tab, tier, int(meta["parent"][a + k]))
            yield item if presence is None else item + (bool(presence) and (pwant is None or sc in pwant),)


CELL_FORMAT = "rel-w18-cells-1"
CELL_INFO = ("arm", "weights", "relatives", "presence", "layout", "render_version", "tableset_sha256", "run", "commit")
CELL_OPTIONAL = ("scenarios", "presence_scenarios", "adapt_v0",  # None: every table; adapt_v0 None: not adapted, else its rows
                 "offline", "offline_sha256")                    # the offline placements rendered on (strict, leak-free): run, sha256


def save_cells(path, cells, **info):
    """One arm's per-cell arrays (run's result) as an uncompressed .npz, with what produced them: arm (the variant
    name), weights (final, best, or untrained), relatives, presence, layout (stored or leakfree), render_version,
    tableset_sha256 (the ruler set's manifest), run and commit (the evaluation's); optional scenarios (the scenarios
    whose tables were scored) and presence_scenarios (those whose tables had the presence passes), lists or None (all)."""
    missing = [k for k in CELL_INFO if k not in info]
    if missing:
        raise ValueError("save_cells needs %s" % ", ".join(missing))
    arrays = {k: np.asarray(cells[k]) for k in ("table", "pass_id", "slot", "truth", "bits", "aa_bits", "entropy")}
    head = dict(format=CELL_FORMAT, version=VERSION, digest=str(cells["digest"]), tables=int(cells["tables"]), passes=int(cells["passes"]),
                pass_names=[str(x) for x in cells["pass_names"]], **{k: info[k] for k in CELL_INFO}, **{k: info.get(k) for k in CELL_OPTIONAL})
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
    for item in source:
        tid, tab, row_tier = item[:3]
        pres = bool(item[4]) if len(item) > 4 else presence
        for p in plan(tab, row_tier, tid, relatives, pres, seed, item[3] if len(item) > 3 else None):
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


def run_ruler(scorer, tables, info, out_dir, render=None, presence=False, max_rows=32, progress=None, scenarios=None, presence_scenarios=None,
              adapt_v0=None, offline=None):
    """One arm on a ruler set, for the evaluation stage: every table of `tables` (data/tables.Tables of the set, kind
    ruler; its TableSet and manifest) through the passes of the arm (info["relatives"] False: pass N only; presence: the
    presence passes too), rendered by `render` (default: the stored layout of data/tables.render, for info["layout"] ==
    "stored") and scored by `scorer` (W6's form, flat_scorer). info: arm, weights (final, best or untrained), relatives,
    layout, render_version, run, commit, and root (the run directory the ruler set's files are under). scenarios: only the
    tables of these scenarios; presence_scenarios: the presence passes on these scenarios' tables only (protocol w18-v0:
    when a reading would take over 5 GPU hours, the untrained arm takes scenarios [0] and the presence arm
    presence_scenarios [0]); None: every table. adapt_v0 (True, or the rows): an arm trained on tables of version 0 reads
    implant tables of version 1 adapted (adapt_table; V0_ROWS rows for True). offline (a run directory of stage
    rel_w18_offline on this ruler set): the strict offline rendering of protocol w18-v3, for info["layout"] "leakfree"
    (which needs it); render defaults to W3's leak-free renderer. Writes <out_dir>/ruler_cells.npz; returns its head
    (save_cells)."""
    missing = [k for k in INFO_KEYS + ("root",) if k not in info]
    if missing:
        raise ValueError("run_ruler needs info %s" % ", ".join(missing))
    man = tables.manifest
    if man.get("kind") != api.TABLE_RULER:
        raise ValueError("run_ruler reads ruler sets only, not kind %r" % man.get("kind"))
    off, off_sha = None, None
    if info["layout"] == api.LAYOUT_LEAKFREE:
        if offline is None:
            raise ValueError("a leak-free reading of a ruler set renders strictly offline (protocol w18-v3): give the run of stage rel_w18_offline")
        off, _, off_sha = load_offline(offline, str(man.get("sha256", "")))
        render = leakfree if render is None else render
    elif offline is not None:
        raise ValueError("the offline placements are for the leak-free layout, not %r" % info["layout"])
    if render is None:
        if info["layout"] != api.LAYOUT_STORED:
            raise ValueError("layout %r needs its renderer" % info["layout"])
        from seeddrift.train.rel.data import tables as T
        render = T.render
    meta = load_meta(info["root"], man)
    scen = None if scenarios is None else sorted({int(x) for x in scenarios})
    pres = bool(presence and info["relatives"])
    psc = None if presence_scenarios is None or not pres else sorted({int(x) for x in presence_scenarios})
    adapt = None if not adapt_v0 else (V0_ROWS if adapt_v0 is True else int(adapt_v0))
    cells = run(tables_of(tables.ts, meta, scenarios=scen, presence=pres, presence_scenarios=psc, adapt=adapt, offline=off), render, flat_scorer(scorer),
                relatives=bool(info["relatives"]), presence=pres, max_rows=max_rows, progress=progress)
    os.makedirs(out_dir, exist_ok=True)
    return save_cells(os.path.join(out_dir, RULER_CELLS), cells, presence=pres, tableset_sha256=str(man.get("sha256", "")), scenarios=scen,
                      presence_scenarios=psc, adapt_v0=adapt, offline=None if offline is None else os.path.basename(os.path.normpath(str(offline))),
                      offline_sha256=off_sha, **{k: info[k] for k in INFO_KEYS})


def adapt_rows(row_tier, mask_id, max_rows=V0_ROWS, seed=SEED):
    """The rows an adapted table keeps (an arm trained on tables of version 0 reading implant tables of version 1, the
    programme lead's adaptation of 2026-11-23, a diagnosis only): row 0, then up to max_rows - 1 other rows taken tier by
    tier (the distance tier to A), the tiers taking turns, inside a tier in the order of a hash of (mask id, row); kept in
    their stored order. The three tables of a triple share their rows and mask id, so they keep the same rows."""
    tier = np.asarray(row_tier).astype(np.int64)
    others = [r for r in range(1, len(tier)) if tier[r] >= 0]
    if len(others) <= int(max_rows) - 1:
        return np.array([0] + others, dtype=np.int64)
    u = cond.hash_u01(seed, STREAM_ADAPT, np.array([int(mask_id)]), np.arange(len(tier))[None, :])[0]
    by = {}
    for r in sorted(others, key=lambda r: (u[r], r)):
        by.setdefault(int(tier[r]), []).append(r)
    queues = [by[t] for t in sorted(by)]
    chosen = []
    while len(chosen) < int(max_rows) - 1 and any(queues):
        for q in queues:
            if q and len(chosen) < int(max_rows) - 1:
                chosen.append(q.pop(0))
    return np.array([0] + sorted(chosen), dtype=np.int64)


def _adapt_kind(kind):
    kind = np.array(kind, copy=True)
    kind[kind == api.KIND_ABSENT] = api.KIND_NOT_SEARCHED            # an old arm never saw ABSENT; NOT_SEARCHED is the nearest
    return kind


def adapt_table(tab, row_tier, mask_id, max_rows=V0_ROWS):
    """(the table with the rows adapt_rows keeps and KIND_ABSENT read as KIND_NOT_SEARCHED, their row tiers, the rows)."""
    keep = adapt_rows(row_tier, mask_id, max_rows)
    R = len(np.asarray(row_tier))
    out = {k: (np.asarray(v)[keep] if np.ndim(v) >= 1 and np.shape(v)[0] == R and k != "frame_len" else v) for k, v in tab.items()}
    out["kind"] = _adapt_kind(out["kind"])
    return out, np.asarray(row_tier)[keep], keep


ROW_META = ("row_tier", "row_dist", "row_dist_d", "row_gene")


def adapt_arrays(fields, meta, max_rows=V0_ROWS):
    """The fields of every table ([N, R, ...]) and W18's meta adapted as adapt_table adapts one table, with max_rows rows
    (the rows not kept do not exist); on_target and covered recounted on the rows kept."""
    N, R = np.asarray(fields["kind"]).shape[:2]
    rows = int(max_rows)
    blank = dict(tok=api.PAD, n=0, col=-1, kind=api.KIND_NONE, genome=-1, identity=np.nan, row_tier=-1, row_dist=np.nan, row_dist_d=np.nan, row_gene=-1)
    nf, nm = {}, {k: np.array(v, copy=True) for k, v in meta.items()}
    for k, v in fields.items():
        v = np.asarray(v)
        if v.ndim >= 2 and v.shape[:2] == (N, R):
            nf[k] = np.full((N, rows) + v.shape[2:], blank.get(k, 0), dtype=v.dtype)
        else:
            nf[k] = v
    for k in ROW_META:
        if k in meta:
            nm[k] = np.full((N, rows) + np.asarray(meta[k]).shape[2:], blank[k], dtype=np.asarray(meta[k]).dtype)
    for t in range(N):
        keep = adapt_rows(meta["row_tier"][t], int(meta["parent"][t]), rows)
        for k, v in fields.items():
            v = np.asarray(v)
            if v.ndim >= 2 and v.shape[:2] == (N, R):
                nf[k][t, :len(keep)] = v[t, keep]
        for k in ROW_META:
            if k in meta:
                nm[k][t, :len(keep)] = np.asarray(meta[k])[t, keep]
    nf["kind"] = _adapt_kind(nf["kind"])
    col, n = nf["col"], nf["n"]
    inside = OFFSET_OF[None, None, :] < n[:, :, BLOCK_OF]
    placed = (col >= 0) & inside
    nm["covered"] = placed[:, 1:, T0_:T1_].sum(axis=(1, 2)).astype(np.asarray(meta["covered"]).dtype)
    nm["on_target"] = (nf["kind"][:, 1:, TARGET_BLOCK] == api.KIND_PRESENT).sum(axis=1).astype(np.asarray(meta["on_target"]).dtype)
    return nf, nm


def count_passes(tab, row_tier, relatives=True, presence=False):
    """How many passes plan gives a table (for the cost estimate of a run)."""
    return len(plan(tab, row_tier, 0, relatives, presence))


def mask_digest(tab, row_tier, table_id, mask_id):
    """sha256 of the row-0 cells every pass of an arm with relatives hides, by pass name (the tier passes left out: the
    tiers present are a property of the rows): equal for the three tables of a triple, whose target blocks have one
    length and whose other blocks are the same."""
    h = hashlib.sha256()
    for p in sorted(plan(tab, row_tier, table_id, True, False, mask_id=mask_id), key=lambda x: x.name):
        if p.name.startswith(("T-t", "C-t")):
            continue
        h.update(p.name.encode())
        h.update(np.flatnonzero(p.target[0]).astype("<i4").tobytes())
    return h.hexdigest()
