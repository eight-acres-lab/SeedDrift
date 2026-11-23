"""The renderer (seeddrift/train/rel/tables/render.py): the leak-free layout of the design (its worked example), every
case of the rule (more or fewer codons than columns, before the first and after the last anchor, no anchor), the online
leak test (hidden residues and every stored position with a hidden end changed: the same grid), the stored layout as the
trainer's reader renders it, and the unaligned layout."""
import numpy as np
import pytest

from seeddrift.train.rel import api as A
from seeddrift.train.rel.tables import render as RD
from test_rel_w5_baselines import toy_tables

SLICES = A.block_slices()
TB = A.N_BLOCKS - 1
T0 = SLICES[TB][1]


def blank(R=3):
    B = A.N_BLOCKS
    return dict(tok=np.zeros((R, A.ROW_LEN), np.uint8), n=np.zeros((R, B), np.int16), col=np.full((R, A.ROW_LEN), -1, np.int16),
                kind=np.zeros((R, B), np.uint8), frame_len=np.zeros(B, np.int16))


def put(tab, r, b, toks, cols=None):
    s, k = SLICES[b][1], len(toks)
    tab["tok"][r, s:s + k] = toks
    tab["n"][r, b], tab["kind"][r, b] = k, A.KIND_PRESENT
    tab["col"][r, s:s + k] = np.arange(k) if cols is None else cols
    if r == 0:
        tab["frame_len"][b] = k


def plan(tab, hidden_frame=(), hidden_rows=None):
    """visible, target: the frame row's target-block columns hidden_frame hidden (and targets), plus {row: positions}."""
    n = tab["n"]
    vis = np.zeros(tab["tok"].shape, bool)
    for b, (_, s, e) in enumerate(SLICES):
        vis[:, s:e] = np.arange(e - s)[None, :] < n[:, b:b + 1]
    hide = np.zeros_like(vis)
    for c in hidden_frame:
        hide[0, T0 + c] = True
    for r, ks in (hidden_rows or {}).items():
        for k in ks:
            hide[r, T0 + k] = True
    return vis & ~hide, hide


def example():
    """The design's example: the frame row's columns 3..8 hidden; the relative has two codons fewer there."""
    tab = blank(2)
    put(tab, 0, TB, np.arange(1, 13))
    put(tab, 1, TB, np.arange(21, 31), [0, 1, 2, 3, 4, 7, 8, 9, 10, 11])     # stored: 4, 5, gap, gap, 6, 7
    return tab


def test_the_worked_example_of_the_design():
    tab = example()
    vis, hide = plan(tab, hidden_frame=range(3, 9))
    grid, truth, ext = RD.render(tab, vis, hide)
    row = grid[1, T0:T0 + 12].tolist()
    assert row == [21, 22, 23, 24, 25, A.EXT_EMPTY, A.EXT_EMPTY, 26, 27, 28, 29, 30]
    assert ext[1, T0:T0 + 12].tolist() == [False] * 3 + [True] * 6 + [False] * 3
    assert grid[0, T0:T0 + 12].tolist() == [1, 2, 3] + [A.MASK] * 6 + [10, 11, 12]
    assert truth[0, T0 + 3:T0 + 9].tolist() == [4, 5, 6, 7, 8, 9] and truth.sum() == sum(range(4, 10)) and not ext[0].any()
    stored, _, sext = RD.render(tab, vis, hide, RD.STORED)
    assert stored[1, T0:T0 + 12].tolist() == [21, 22, 23, 24, 25, A.GAP, A.GAP, 26, 27, 28, 29, 30] and not sext.any()
    moved = example()                                                            # the gaps where the hidden residues put them
    moved["col"][1, T0:T0 + 10] = [0, 1, 2, 5, 6, 7, 8, -1, 10, 11]          # (and one codon an insertion): leak-free ignores it
    moved["col"][1, T0 + 7] = 9
    assert np.array_equal(RD.render(moved, vis, hide)[0], grid) and not np.array_equal(RD.render(moved, vis, hide, RD.STORED)[0], stored)


def test_more_codons_than_columns_and_the_ends():
    tab = blank(4)
    put(tab, 0, TB, np.arange(1, 11))                                            # frame: 10 columns, 3..6 hidden
    put(tab, 1, TB, np.arange(21, 31), [0, 1, 2, -1, -1, -1, -1, -1, 7, 8])   # 7 codons between the anchors 2 and 7: 4 columns
    put(tab, 2, TB, np.arange(41, 47), [3, 4, 5, 6, 7, 8])                    # starts inside the hidden stretch: before its first anchor
    put(tab, 3, TB, np.arange(61, 66), [0, 1, 2, 3, 4])                       # ends inside it: after its last anchor
    vis, hide = plan(tab, hidden_frame=range(3, 7))
    grid, truth, ext = RD.render(tab, vis, hide)
    # row 1: anchors codon 2 (col 2) and codon 8 (col 7); 5 codons (3..7) into columns 3..6: 2 from the left, 2 from the right
    assert grid[1, T0:T0 + 10].tolist() == [21, 22, 23, 24, 25, 27, 28, 29, 30, A.PAD]
    assert grid[1, T0 + 9] == A.PAD and ext[1, T0 + 3:T0 + 7].all() and not ext[1, T0 + 7]
    # row 2: first anchor codon 4 (col 7): codons 0..3 laid leftwards from it into columns 3..6
    assert grid[2, T0:T0 + 10].tolist() == [A.PAD] * 3 + [41, 42, 43, 44, 45, 46, A.PAD]
    # row 3: last anchor codon 2 (col 2): codons 3, 4 laid rightwards into columns 3, 4; then outside its span
    assert grid[3, T0:T0 + 10].tolist() == [61, 62, 63, 64, 65] + [A.PAD] * 5
    assert ext[2, T0 + 3:T0 + 7].all() and ext[3, T0 + 3:T0 + 5].all() and not ext[3, T0 + 5:T0 + 10].any()
    whole = blank(2)                                                             # the whole gene of the frame row hidden: no anchor
    put(whole, 0, TB, np.arange(1, 9))
    put(whole, 1, TB, np.arange(21, 27), [1, 2, 3, 5, 6, 7])
    vis, hide = plan(whole, hidden_frame=range(8))
    grid, truth, ext = RD.render(whole, vis, hide)
    assert grid[1, T0:T0 + 8].tolist() == [21, 22, 23, 24, 25, 26, A.PAD, A.PAD] and ext[1, T0:T0 + 6].all()
    none = blank(2)                                                              # no codon between two anchors: the columns stay GAP
    put(none, 0, TB, np.arange(1, 9))
    put(none, 1, TB, np.array([21, 22, 23, 24]), [0, 1, 6, 7])
    vis, hide = plan(none, hidden_frame=range(3, 5))
    assert RD.render(none, vis, hide)[0][1, T0:T0 + 8].tolist() == [21, 22, A.GAP, A.GAP, A.GAP, A.GAP, 23, 24]


def test_a_row_of_its_own_with_hidden_codons_is_laid_by_its_anchors():
    tab = blank(2)
    put(tab, 0, TB, np.arange(1, 11))
    put(tab, 1, TB, np.arange(21, 29), [0, 1, 2, 5, 6, 7, 8, 9])              # stored: two frame columns with no codon (3, 4)
    vis, hide = plan(tab, hidden_rows={1: [3, 4]})                               # the row's own codons 3, 4 hidden
    grid, truth, ext = RD.render(tab, vis, hide)
    # anchors codon 2 (col 2) and codon 5 (col 7); codons 3, 4 into columns 3..6: one from each side, the middle EXT_EMPTY
    assert grid[1, T0:T0 + 10].tolist() == [21, 22, 23, A.MASK, A.EXT_EMPTY, A.EXT_EMPTY, A.MASK, 26, 27, 28]
    assert ext[1, T0 + 3:T0 + 7].all() and not ext[1, T0 + 7:T0 + 10].any()
    assert truth[1, T0 + 3] == 24 and truth[1, T0 + 6] == 25 and RD.dropped(tab, vis, hide) == 0


def _perturb(tab, vis, rng):
    """Every hidden residue replaced, and every stored pair with a hidden end redrawn (anchors kept, none added)."""
    out = {k: np.array(v, copy=True) for k, v in tab.items()}
    hidden = ~vis & (out["tok"] > 0)
    out["tok"][hidden] = rng.integers(1, 66, hidden.sum())
    n, col = out["n"], out["col"]
    for b, (_, s, e) in enumerate(SLICES):
        f = int(out["frame_len"][b])
        for r in range(1, col.shape[0]):
            m = int(n[r, b])
            if not m or not f:
                continue
            c = col[r, s:s + m].astype(int)
            vr, vf = vis[r, s:s + m], vis[0, s:s + f]
            anchor = np.zeros(m, bool)
            ok = (c >= 0) & (c < f)
            anchor[ok] = vr[ok] & vf[c[ok]]
            ak = np.flatnonzero(anchor)
            bounds = [(-1, -1)] + [(int(k), int(c[k])) for k in ak] + [(m, f)]
            new = np.where(anchor, c, -1)
            for (kl, cl), (kr, cr) in zip(bounds[:-1], bounds[1:]):
                if not ((~vr[kl + 1:kr]).any() or (~vf[cl + 1:cr]).any()):
                    new[kl + 1:kr] = c[kl + 1:kr]                                 # no hidden cell: the stored pairs stay
                    continue
                last = cl
                for k in range(kl + 1, kr):
                    if rng.random() < 0.5:
                        cand = [x for x in range(last + 1, cr) if not (vr[k] and vf[x])]
                        if cand:
                            last = int(rng.choice(cand))
                            new[k] = last
            col[r, s:s + m] = new
    return out


def _plans(tab, rng, rate=0.15, stretch=True):
    vis = np.zeros(tab["tok"].shape, bool)
    for b, (_, s, e) in enumerate(SLICES):
        vis[:, s:e] = np.arange(e - s)[None, :] < tab["n"][:, b:b + 1]
    hide = vis & (rng.random(vis.shape) < rate)                                  # scattered cells in every row
    m = int(tab["n"][0, TB])
    if stretch and m > 8:                                                        # and a stretch of the frame row's target block
        a = int(rng.integers(0, m - 6))
        hide[0, T0 + a:T0 + a + 6] = True
    return vis & ~hide, hide


def test_leak_free_grids_do_not_depend_on_hidden_residues_or_their_positions():
    rng = np.random.default_rng(5)
    tables, _ = toy_tables(n_tables=12, rows=6, seed=71)
    changed_stored = 0
    for t in range(12):
        tab = {k: tables[k][t] for k in ("tok", "n", "col", "kind")}
        tab["frame_len"] = tables["n"][t, 0]
        vis, hide = _plans(tab, rng)
        grid, truth, ext = RD.render(tab, vis, hide)
        for _ in range(3):
            other = _perturb(tab, vis, rng)
            g2, t2, e2 = RD.render(other, vis, hide)
            assert np.array_equal(grid, g2) and np.array_equal(ext, e2), t
            changed_stored += not np.array_equal(RD.render(tab, vis, hide, RD.STORED)[0], RD.render(other, vis, hide, RD.STORED)[0])
        assert ((grid[0] == A.MASK) == (hide[0] & (tab["tok"][0] > 0))).all()       # the frame row: its own cells
    assert changed_stored >= 10                                                  # the stored layout does depend on them


def test_the_whole_table_at_once_is_the_rule_row_by_row():
    rng = np.random.default_rng(9)
    tables, _ = toy_tables(n_tables=10, rows=8, seed=77)
    for t in range(10):
        tab = {k: tables[k][t] for k in ("tok", "n", "col", "kind")}
        tab["frame_len"] = tables["n"][t, 0]
        for rate in (0.0, 0.15, 0.5):
            vis, hide = _plans(tab, rng, rate=rate, stretch=rate > 0)
            for layout in RD.LAYOUTS:
                fast, ref = RD.placement(tab, vis, layout), RD.placement_reference(tab, vis, layout)
                assert all(np.array_equal(a, b) for a, b in zip(fast, ref)), (t, rate, layout)
    for build in (example, ):                                                    # and the hand-made cases
        tab = build()
        vis, hide = plan(tab, hidden_frame=range(3, 9))
        assert all(np.array_equal(a, b) for a, b in zip(RD.placement(tab, vis), RD.placement_reference(tab, vis)))


def stored_reference(tab, visible, target):
    """The trainer's reader's own stored-layout renderer before it used this module (W6, commit 3382ed5), copied here
    as an independent reference: every row at its stored columns, GAP between its first and last placed cell, PAD
    outside, a block without codons filled by its kind, nothing beyond the frame; a hidden codon shows as MASK."""
    from seeddrift.train.rel.evalh import layout

    L = A.ROW_LEN
    block_of, start = np.zeros(L, np.int64), np.zeros(L, np.int64)
    for b, (_, s, e) in enumerate(SLICES):
        block_of[s:e], start[s:e] = b, s
    fill = np.full(256, A.PAD, np.uint8)
    fill[A.KIND_ABSENT], fill[A.KIND_NOT_SEARCHED], fill[A.KIND_TRUNCATED] = A.ABSENT, A.NOT_SEARCHED, A.TRUNCATED

    def placed_values(values, f):
        return layout.place(np.asarray(values)[None], np.asarray(tab["n"])[None], np.asarray(tab["col"])[None], f)[0]

    tok, kind = np.asarray(tab["tok"]), np.asarray(tab["kind"])
    frame_len = np.asarray(tab["frame_len"]).astype(np.int64)
    R = tok.shape[0]
    cells = (np.arange(L) - start)[None, :] < np.asarray(tab["n"])[:, block_of]
    shown = cells & (np.asarray(tab["col"]) >= 0)
    hidden = cells & ~np.asarray(visible, dtype=bool)
    placed = placed_values(shown.astype(np.uint8), 0).astype(bool)
    token = placed_values(np.where(hidden, A.MASK, tok).astype(np.uint8), A.PAD)
    want = np.asarray(target, dtype=bool) & hidden & (tok >= A.TARGET_MIN) & (tok <= A.TARGET_MAX)
    truth = placed_values(np.where(want, tok, 0).astype(np.uint8), 0)
    grid = np.full((R, L), A.PAD, np.uint8)
    for b, (_, s, e) in enumerate(SLICES):
        p = placed[:, s:e]
        span = (np.cumsum(p, axis=1) > 0) & (np.cumsum(p[:, ::-1], axis=1)[:, ::-1] > 0)
        g = np.where(p, token[:, s:e], np.where(span, A.GAP, A.PAD)).astype(np.uint8)
        empty = ~np.isin(kind[:, b], (A.KIND_PRESENT, A.KIND_SELF_COPY))
        g[empty] = A.PAD
        f = int(frame_len[b])
        g[empty, :f] = fill[kind[empty, b]][:, None]
        g[:, f:] = A.PAD
        grid[:, s:e] = g
    truth[grid != A.MASK] = 0
    return grid, truth


def test_without_hidden_cells_leak_free_is_stored_and_stored_is_the_readers():
    tables, _ = toy_tables(n_tables=6, rows=6, seed=73)
    rng = np.random.default_rng(1)
    for t in range(6):
        tab = {k: tables[k][t] for k in ("tok", "n", "col", "kind")}
        tab["frame_len"] = tables["n"][t, 0]
        vis, _ = _plans(tab, rng, rate=0.0, stretch=False)
        hide = np.zeros_like(vis)
        g1, t1, e1 = RD.render(tab, vis, hide)
        g2, t2, e2 = RD.render(tab, vis, hide, RD.STORED)
        assert np.array_equal(g1, g2) and not e1.any() and not t1.any()
        vis, hide = _plans(tab, rng)
        gw, tw = stored_reference(tab, vis, hide)                               # the reader's old renderer, independent of RD
        gs, ts, _ = RD.render(tab, vis, hide, RD.STORED)
        assert np.array_equal(gw, gs) and np.array_equal(tw, ts)
        un, _, _ = RD.render(tab, vis, hide, RD.UNALIGNED)
        for b, (_, s, e) in enumerate(SLICES):                                   # unaligned: every row from the block's first column
            for r in range(1, 6):
                k = min(int(tab["n"][r, b]), int(tab["frame_len"][b]))
                if k and tab["kind"][r, b] == A.KIND_PRESENT:
                    shown = np.where(vis[r, s:s + k], tab["tok"][r, s:s + k], A.MASK)
                    assert np.array_equal(un[r, s:s + k], shown)
    with pytest.raises(ValueError):
        RD.render(tab, vis, hide, "sideways")


def test_a_block_laid_on_another_frame_row():
    """frame_row of the target block = 1: that row keeps its own columns, the examined row (hidden as a whole) is laid
    from the first column, a third row by its pairs with the frame row."""
    tab = blank(3)
    put(tab, 0, TB, np.arange(1, 7))
    put(tab, 1, TB, np.arange(21, 29))
    put(tab, 2, TB, np.arange(41, 47), [1, 2, 3, 5, 6, 7])                    # aligned to row 1, a gap at its column 4
    tab["col"][0, T0:T0 + 6] = np.arange(6)
    tab["frame_row"] = np.zeros(A.N_BLOCKS, np.int8)
    tab["frame_row"][TB], tab["frame_len"][TB] = 1, 8
    vis, hide = plan(tab, hidden_frame=range(6))                                 # the examined row hidden as a whole
    grid, truth, ext = RD.render(tab, vis, hide)
    assert grid[1, T0:T0 + 8].tolist() == list(range(21, 29)) and not ext[1].any()
    assert grid[0, T0:T0 + 8].tolist() == [A.MASK] * 6 + [A.PAD, A.PAD] and truth[0, T0:T0 + 6].tolist() == list(range(1, 7))
    assert grid[2, T0:T0 + 8].tolist() == [A.PAD, 41, 42, 43, A.GAP, 44, 45, 46] and not ext[2].any()
    assert all(np.array_equal(a, b) for a, b in zip(RD.placement(tab, vis), RD.placement_reference(tab, vis)))
    gone = dict(tab, frame_row=np.full(A.N_BLOCKS, 9, np.int8))                  # a frame row that is not in the table: row 0
    assert RD.frame_rows(gone).tolist() == [0] * A.N_BLOCKS
    pr = RD.render(tab, vis, hide, control="presence")[0]                       # the controls act on every row but row 0:
    assert (pr[1:, T0:T0 + 8][(grid[1:, T0:T0 + 8] >= 1) & (grid[1:, T0:T0 + 8] <= 65)] == A.MASK).all()      # the frame relative too
    assert np.array_equal(pr[0], grid[0]) and ((pr[1:] == A.MASK) == ((grid[1:] >= 1) & (grid[1:] <= 65) | (grid[1:] == A.MASK))).all()
    sh, st, _ = RD.render(tab, vis, hide, control="shuffle", key=5)
    assert np.array_equal(sh[0], grid[0]) and np.array_equal(st[0], truth[0])  # row 0 stays where it is
    perm = np.argsort(__import__("seeddrift.train.rel.evalh.conditions", fromlist=["x"]).hash_u01(5, RD.STREAM_SHUFFLE, TB, np.arange(8)), kind="stable")
    assert not np.array_equal(perm, np.arange(8)) and np.array_equal(sh[1:, T0:T0 + 8], grid[1:, T0 + perm])   # both relatives, one permutation


def test_the_two_controls_and_the_reporting_helpers():
    rng = np.random.default_rng(13)
    tables, _ = toy_tables(n_tables=3, rows=6, seed=79)
    tab = {k: tables[k][0] for k in ("tok", "n", "col", "kind")}
    tab["frame_len"] = tables["n"][0, 0]
    vis, hide = _plans(tab, rng)
    grid, truth, ext = RD.render(tab, vis, hide)
    sh, st, se = RD.render(tab, vis, hide, control="shuffle", key=7)
    assert np.array_equal(sh[0], grid[0]) and np.array_equal(st[0], truth[0])  # the frame row as it was
    for b, (_, s, e) in enumerate(SLICES):
        f = int(tab["frame_len"][b])
        if f < 3:
            continue
        for r in range(1, 6):                                                    # each row's cells moved, not changed
            assert sorted(sh[r, s:s + f].tolist()) == sorted(grid[r, s:s + f].tolist())
    moved = [b for b, (_, s, e) in enumerate(SLICES) if int(tab["frame_len"][b]) > 2 and not np.array_equal(sh[1:, s:e], grid[1:, s:e])]
    assert moved and np.array_equal(RD.render(tab, vis, hide, control="shuffle", key=7)[0], sh)
    assert not np.array_equal(RD.render(tab, vis, hide, control="shuffle", key=8)[0], sh)
    s, f = T0, int(tab["frame_len"][TB])                                         # one permutation for every other row of a block
    pairs = [(sh[1:, s + x], grid[1:, s:s + f]) for x in range(f)]
    assert all(any(np.array_equal(col, g[:, y]) for y in range(f)) for col, g in pairs)
    pr, ptruth, _ = RD.render(tab, vis, hide, control="presence")
    codon = (grid >= A.TARGET_MIN) & (grid <= A.TARGET_MAX)
    assert (pr[1:][codon[1:]] == A.MASK).all() and np.array_equal(pr[0], grid[0]) and np.array_equal(pr[~codon], grid[~codon])
    with pytest.raises(ValueError):
        RD.render(tab, vis, hide, control="sideways")
    d = blank(2)                                                                 # anchors at columns 0..2 and 7..9: distances
    put(d, 0, TB, np.arange(1, 11))
    put(d, 1, TB, np.arange(21, 31))
    vis, hide = plan(d, hidden_frame=range(3, 7))
    assert RD.anchor_distance(d, vis)[T0:T0 + 10].tolist() == [0, 0, 0, 1, 2, 2, 1, 0, 0, 0] and RD.anchor_distance(d, vis)[0] == -1
    rows = RD.causal_rows(tab, key=3)
    assert rows[0] == 0 and len(rows) == 3 and len(set(rows)) == 3 and rows == RD.causal_rows(tab, key=3)
    lone = blank(2)
    put(lone, 0, TB, np.arange(1, 5))
    assert RD.causal_rows(lone) == [0]


def _old_strict(tab, visible):
    """align.strict of version 1 (the frame row's hidden blocks only), the reference where only row 0 is hidden."""
    from seeddrift.train.rel.tables import align as AL
    tok, n = np.asarray(tab["tok"]), np.asarray(tab["n"]).astype(np.int64)
    vis, col = np.asarray(visible, bool), np.array(tab["col"], dtype=np.int16, copy=True)
    for b, (_, s, e) in enumerate(SLICES):
        f = int(tab["frame_len"][b])
        if f == 0 or vis[0, s:s + f].all():
            continue
        rows = [r for r in range(1, tok.shape[0]) if n[r, b] > 0]
        if rows:
            cols, _ = AL.align(AL.residues(tok[0, s:s + f], vis[0, s:s + f]), [AL.residues(tok[r, s:s + n[r, b]], vis[r, s:s + n[r, b]]) for r in rows])
            for r, c in zip(rows, cols):
                col[r, s:s + n[r, b]] = c.astype(np.int16)
    return col


def test_strict_aligns_a_row_again_where_only_it_hides_residues():
    """align.strict of version 2: a block where only another row hides residues aligns that row again (its hidden
    residues unknown) and keeps the others stored; where only row 0 is hidden, version 1's result."""
    from seeddrift.train.rel.tables import align as AL
    from seeddrift.train.rel.data import tables as DT
    tables, _ = toy_tables(n_tables=6, rows=6, seed=91)
    rng = np.random.default_rng(4)
    for t in range(6):
        tab = {k: tables[k][t] for k in ("tok", "n", "col", "kind")}
        tab["frame_len"] = tables["n"][t, 0]
        cells = DT.inside(tab)
        only0 = cells.copy()
        only0[0] &= rng.random(A.ROW_LEN) > 0.15
        assert np.array_equal(AL.strict(tab, only0), _old_strict(tab, only0))     # evaluation: unchanged
        vis = cells.copy()
        r = int(np.flatnonzero(tab["n"][1:, TB] > 3)[0]) + 1 if (tab["n"][1:, TB] > 3).any() else None
        if r is None:
            continue
        vis[r, T0:T0 + 3] = False                                                 # row r hides three residues, row 0 none
        got = AL.strict(tab, vis)
        assert np.array_equal(np.delete(got, r, axis=0), np.delete(np.asarray(tab["col"]), r, axis=0))
        other = dict(tab, tok=np.array(tab["tok"], copy=True), col=np.array(tab["col"], copy=True))
        other["tok"][r, T0:T0 + 3] = 1 + (other["tok"][r, T0:T0 + 3] % 64)
        other["col"][r, T0:T0 + int(tab["n"][r, TB])] = -1
        assert np.array_equal(AL.strict(other, vis), got)                          # independent of its hidden residues


def test_the_training_fourth_test_passes_strict_and_catches_the_online_layout():
    """Every mask scheme of the training (scattered in every row, a span in any row, the whole gene) passes W3's fourth
    test with the strict rendering; the online leak-free layout on the stored pairs fails it on some table."""
    from seeddrift.train.rel.data import masks
    from seeddrift.train.rel.tables import strict as ST
    import importlib
    tables, _ = toy_tables(n_tables=10, rows=8, seed=93)
    online_fails = 0
    for name in ("scattered", "span", "whole_gene"):
        m = importlib.import_module("seeddrift.train.rel.data.masks." + name)
        for t in range(10):
            tab = {k: tables[k][t] for k in ("tok", "n", "col", "kind")}
            tab["frame_len"], tab["frame_row"] = tables["n"][t, 0], np.zeros(A.N_BLOCKS, np.int8)
            vis, tgt = m.plan(dict(n=tab["n"], kind=tab["kind"]), np.random.default_rng(t))
            got = ST.fourth_test(tab, vis, tgt, whole_gene=name == "whole_gene", key=t)
            assert got["passed"], (name, t)
            if name != "whole_gene":                                              # the online layout: stored pairs as anchors
                grid, _, ext = RD.render(tab, vis, tgt)
                hidden = (np.arange(A.ROW_LEN)[None, :] - RD.START[None, :] < np.asarray(tab["n"])[:, RD.BLOCK_OF]) & ~vis
                other = dict(tab, tok=np.where(hidden, 1 + (np.asarray(tab["tok"]) % 64), tab["tok"]).astype(np.uint8), col=ST.erased(tab, vis))
                g2, _, e2 = RD.render(other, vis, tgt)
                online_fails += not (np.array_equal(grid, g2) and np.array_equal(ext, e2))
    assert online_fails > 0
    assert masks is not None
