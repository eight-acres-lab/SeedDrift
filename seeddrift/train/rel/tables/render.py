"""The renderer of the relational-model line (package W3): the grid a model reads, from a table and a mask plan.

numpy only, so both interpreters run it: the trainer's reader renders every training table online, the evaluator
renders the evaluation items. Columns are the frame row's own coordinates (row 0 in table version 0), so the length of
a hidden stretch of the frame row is known and only its content is hidden.

Layouts (api.LAYOUT_*):
  leakfree   no input depends on a hidden residue. For every other row and block the anchors are the stored pairs
             (codon of the row, column of the frame row) whose two residues are both visible; an anchor keeps its column.
             Between two anchors that enclose a hidden cell (a hidden column of the frame row or a hidden codon of the
             row), the row's m codons between them are laid into the w columns between them from both anchors: the first
             ceil(m/2) from the left anchor, the last floor(m/2) from the right one, at most ceil(w/2) and floor(w/2);
             codons that do not fit are not shown, columns left in the middle are EXT_EMPTY, and with m = 0 the columns
             stay GAP. Before the first anchor the codons are laid leftwards from it, after the last anchor rightwards
             (columns beyond are outside the row's span). A block with a hidden cell and no anchor (the whole gene of the
             frame row hidden: the frame relative of step two is not there yet) is laid from its first column. Between
             anchors with no hidden cell the stored placement stays. Every cell laid this way is marked in ext.
  stored     the placement the table stores (col). It shows the relatives where the hidden residues align, so bits read
             through it with a stretch hidden are too low (optimistic); kept for diagnoses and the share of training
             samples that keep it.
  unaligned  every row laid from the first column of each block in its own coordinates; no correspondence is used.
Online, the anchors are the stored pairs: an approximation, since the stored alignment saw every residue. The strict
rendering of evaluation items renders a table whose correspondence was aligned with the hidden residues unknown
(ci/stages/rel/w3_offline.py) through this same function.

Grid tokens: a shown codon (MASK where hidden); GAP between the first and the last shown codon of a row's span in a
block; EXT_EMPTY for the middle columns of a re-laid stretch; PAD outside the span and beyond the frame's length; a block
without codons (kind not present) shows its kind's token up to the frame's length (ABSENT, NOT_SEARCHED, TRUNCATED).
truth: the token of every target cell (a hidden residue 1..65) at its grid slot, 0 elsewhere; a target that is not
shown is dropped, it cannot be scored in this rendering.
"""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel import api as A

VERSION = "1"
LEAKFREE, STORED, UNALIGNED = A.LAYOUT_LEAKFREE, A.LAYOUT_STORED, A.LAYOUT_UNALIGNED
LAYOUTS = (LEAKFREE, STORED, UNALIGNED)
L = A.ROW_LEN
SLICES = A.block_slices()
KIND_NONE = getattr(A, "KIND_NONE", 0)
WITH_CODONS = (A.KIND_PRESENT, A.KIND_SELF_COPY)
FILL = np.full(256, A.PAD, dtype=np.uint8)                 # the token of a block without codons, by its kind
FILL[A.KIND_ABSENT], FILL[A.KIND_NOT_SEARCHED], FILL[A.KIND_TRUNCATED] = A.ABSENT, A.NOT_SEARCHED, A.TRUNCATED


def lay_leakfree(col, vis_row, vis_frame):
    """One other row in one block. col (n,): stored column of every codon of the row (-1: none); vis_row (n,): the codon
    is visible; vis_frame (f,): the frame row's residue at that column is visible (f = the frame's length).
    Returns pos (n,) the column of every codon (-1: not shown), ext (n,) laid by extension, empty (f,) EXT_EMPTY."""
    n, f = col.shape[0], vis_frame.shape[0]
    pos, ext, empty = np.full(n, -1, np.int64), np.zeros(n, bool), np.zeros(f, bool)
    if n == 0 or f == 0:
        return pos, ext, empty
    c = np.asarray(col, dtype=np.int64)
    ok = (c >= 0) & (c < f)
    anchor = np.zeros(n, bool)
    anchor[ok] = vis_row[ok] & vis_frame[c[ok]]
    ak = np.flatnonzero(anchor)
    hidden_row, hidden_frame = ~np.asarray(vis_row, bool), ~np.asarray(vis_frame, bool)
    if ak.size == 0:
        if hidden_row.any() or hidden_frame.any():            # nothing to hold on to: laid from the first column
            m = min(n, f)
            pos[:m], ext[:m] = np.arange(m), True
        else:
            pos[ok] = c[ok]
        return pos, ext, empty
    ac = c[ak]
    pos[ak] = ac
    bounds = [(-1, -1)] + list(zip(ak.tolist(), ac.tolist())) + [(n, f)]
    last = len(bounds) - 2
    for j in range(len(bounds) - 1):
        (kl, cl), (kr, cr) = bounds[j], bounds[j + 1]
        m, w = kr - kl - 1, cr - cl - 1
        if m <= 0 and not hidden_frame[cl + 1:cr].any():
            continue
        ks = np.arange(kl + 1, kr)
        if not (hidden_row[kl + 1:kr].any() or hidden_frame[cl + 1:cr].any()):
            keep = ok[ks]                                       # no hidden cell here: the stored placement stays
            pos[ks[keep]] = c[ks[keep]]
            continue
        if m <= 0:
            continue                                            # no codon between the anchors: the columns stay GAP
        if j == 0:                                              # before the first anchor: leftwards from it
            cols = cr - (kr - ks)
            keep = cols >= 0
        elif j == last:                                         # after the last anchor: rightwards from it
            cols = cl + (ks - kl)
            keep = cols < f
        else:
            nl, nr = min((m + 1) // 2, (w + 1) // 2), min(m // 2, w // 2)
            cols = np.full(m, -1, np.int64)
            cols[:nl] = cl + 1 + np.arange(nl)
            if nr:
                cols[m - nr:] = cr - nr + np.arange(nr)
            keep = cols >= 0
            if w - nl - nr > 0:
                empty[cl + 1 + nl:cr - nr] = True
        pos[ks[keep]], ext[ks[keep]] = cols[keep], True
    return pos, ext, empty


POS = np.zeros(L, np.int64)                                 # position of a slot inside its block
BLOCK_OF = np.zeros(L, np.int64)
START = np.zeros(L, np.int64)
for _b, (_role, _s, _e) in enumerate(SLICES):
    POS[_s:_e], BLOCK_OF[_s:_e], START[_s:_e] = np.arange(_e - _s), _b, _s
BLOCK_START = np.array([s for _, s, _ in SLICES], np.int64)


def cells_of(n):
    """[R, ROW_LEN]: the slot holds a codon of the row's own segment."""
    return POS[None, :] < np.asarray(n).astype(np.int64)[:, BLOCK_OF]


def placement(tab, visible, layout=LEAKFREE):
    """slot [R, ROW_LEN]: the grid slot of every cell in the rows' own coordinates (-1: not shown); ext [R, ROW_LEN]
    grid cells laid by extension (codons and EXT_EMPTY columns); empty [R, ROW_LEN] grid cells that are EXT_EMPTY.
    The leak-free layout is computed for the whole table at once; lay_leakfree is the same rule for one row and block."""
    if layout not in LAYOUTS:
        raise ValueError("unknown layout %r (known: %s)" % (layout, ", ".join(LAYOUTS)))
    n = np.asarray(tab["n"]).astype(np.int64)
    col = np.asarray(tab["col"]).astype(np.int64)
    frame = np.asarray(tab["frame_len"]).astype(np.int64)
    R = col.shape[0]
    cells = cells_of(n)
    vis = np.asarray(visible, dtype=bool) & cells
    f_of = frame[BLOCK_OF]                                      # the frame's length in the block of every slot
    fr = frame_rows(tab)
    fr_of = fr[BLOCK_OF]                                        # the frame row of the block of every slot
    is_frame = np.arange(R)[:, None] == fr_of[None, :]
    slot = np.full((R, L), -1, np.int64)
    ext, empty = np.zeros((R, L), bool), np.zeros((R, L), bool)
    own = cells & is_frame & (POS < f_of)[None, :]
    fr_r, fr_s = np.nonzero(own)
    slot[fr_r, fr_s] = fr_s                                     # the frame row of a block: its own columns
    rr, ss = np.nonzero(cells & ~is_frame)
    if rr.size == 0:
        return slot, ext, empty
    if layout == UNALIGNED:
        slot[rr, ss] = ss
        return slot, ext, empty
    c = col[rr, ss]
    b = BLOCK_OF[ss]
    f = frame[b]
    ok = (c >= 0) & (c < f)
    stored = np.where(ok, BLOCK_START[b] + np.maximum(c, 0), -1)
    if layout == STORED:
        slot[rr, ss] = stored
        return slot, ext, empty
    k = POS[ss]
    hidden_row = ~vis[rr, ss]
    frame_vis = np.zeros(rr.shape[0], bool)
    frame_vis[ok] = vis[fr[b[ok]], BLOCK_START[b[ok]] + c[ok]]
    anchor = ok & ~hidden_row & frame_vis
    N = rr.shape[0]
    idx = np.arange(N)
    first = np.ones(N, bool)
    first[1:] = (rr[1:] != rr[:-1]) | (b[1:] != b[:-1])        # the codons of one row and block are contiguous
    gstart = np.maximum.accumulate(np.where(first, idx, 0))
    last = np.ones(N, bool)
    last[:-1] = first[1:]
    gend = np.minimum.accumulate(np.where(last, idx, N)[::-1])[::-1]
    a_left = np.maximum.accumulate(np.where(anchor, idx, -1))   # the last anchor at or before every codon
    a_right = np.minimum.accumulate(np.where(anchor, idx, N)[::-1])[::-1]
    prev = np.concatenate([[-1], a_left[:-1]])                  # strictly before / after (for a non-anchor these agree)
    nxt = np.concatenate([a_right[1:], [N]])
    il = np.where(anchor, prev, a_left)
    ir = np.where(anchor, nxt, a_right)
    has_l, has_r = il >= gstart, ir <= gend
    m_seg = n[rr, b]
    kl = np.where(has_l, k[np.clip(il, 0, N - 1)], -1)
    cl = np.where(has_l, c[np.clip(il, 0, N - 1)], -1)
    kr = np.where(has_r, k[np.clip(ir, 0, N - 1)], m_seg)
    cr = np.where(has_r, c[np.clip(ir, 0, N - 1)], f)
    H = np.concatenate([[0], np.cumsum(hidden_row)])            # hidden codons of the row between the anchors
    lo = np.where(has_l, il + 1, gstart)
    hi = np.where(has_r, ir, gend + 1)
    row_hidden = H[hi] - H[lo]
    cols = np.arange(L)
    hid0 = np.concatenate([[0], np.cumsum(cells[fr_of, cols] & ~vis[fr_of, cols])])  # hidden columns of the frame row
    base = BLOCK_START[b]
    frame_hidden = hid0[base + np.minimum(cr, f)] - hid0[base + cl + 1]
    need = (row_hidden + frame_hidden) > 0
    m = kr - kl - 1
    w = cr - cl - 1
    p = k - kl - 1
    nl = np.minimum((m + 1) // 2, (w + 1) // 2)
    nr = np.minimum(m // 2, w // 2)
    interior = has_l & has_r
    pos = np.full(N, -1, np.int64)
    pos[interior & (p < nl)] = (cl + 1 + p)[interior & (p < nl)]
    right = interior & (p >= m - nr) & (p >= nl)
    pos[right] = (cr - (m - p))[right]
    head = ~has_l & has_r
    hc = cr - (kr - k)
    pos[head & (hc >= 0)] = hc[head & (hc >= 0)]
    tail = has_l & ~has_r
    tc = cl + (k - kl)
    pos[tail & (tc < f)] = tc[tail & (tc < f)]
    alone = ~has_l & ~has_r                                     # no anchor in the block: laid from its first column
    pos[alone & (k < f)] = k[alone & (k < f)]
    laid = need & ~anchor
    out = np.where(anchor, BLOCK_START[b] + c, np.where(laid, np.where(pos >= 0, base + pos, -1), stored))
    slot[rr, ss] = out
    placed_ext = laid & (pos >= 0)
    ext[rr[placed_ext], (base + pos)[placed_ext]] = True
    seg = laid & interior & (p == 0) & (w - nl - nr > 0) & (m > 0)  # one record per re-laid stretch with columns left over
    if seg.any():
        diff = np.zeros((R, L + 1), np.int64)
        np.add.at(diff, (rr[seg], (base + cl + 1 + nl)[seg]), 1)
        np.add.at(diff, (rr[seg], (base + cr - nr)[seg]), -1)
        empty = np.cumsum(diff, axis=1)[:, :L] > 0
        ext |= empty
    return slot, ext, empty


def frame_rows(tab):
    """(B,) the frame row of every block: tab["frame_row"] where given and still a row of the table, else row 0."""
    R = np.asarray(tab["tok"]).shape[0]
    fr = np.asarray(tab.get("frame_row", np.zeros(A.N_BLOCKS)), dtype=np.int64).reshape(-1)
    return np.where((fr >= 0) & (fr < R), fr, 0)


def placement_reference(tab, visible, layout=LEAKFREE):
    """The same as placement, one row and block at a time through lay_leakfree (for the tests)."""
    tok, n, col = np.asarray(tab["tok"]), np.asarray(tab["n"]).astype(np.int64), np.asarray(tab["col"]).astype(np.int64)
    frame = np.asarray(tab["frame_len"]).astype(np.int64)
    vis = np.asarray(visible, dtype=bool) & cells_of(n)
    R = tok.shape[0]
    fr = frame_rows(tab)
    slot = np.full((R, L), -1, np.int64)
    ext, empty = np.zeros((R, L), bool), np.zeros((R, L), bool)
    for b, (_, s, e) in enumerate(SLICES):
        f = int(frame[b])
        for r in range(R):
            m = int(n[r, b])
            if m == 0:
                continue
            if r == fr[b] or layout == UNALIGNED:
                kk = np.arange(min(m, f) if r == fr[b] else m)
                slot[r, s + kk] = s + kk
                continue
            c = col[r, s:s + m]
            if layout == STORED:
                ok = (c >= 0) & (c < f)
                slot[r, s + np.flatnonzero(ok)] = s + c[ok]
                continue
            pos, ex, em = lay_leakfree(c, vis[r, s:s + m], vis[fr[b], s:s + f])
            shown = pos >= 0
            slot[r, s + np.flatnonzero(shown)] = s + pos[shown]
            ext[r, s + pos[shown & ex]] = True
            ext[r, s:s + f] |= em
            empty[r, s:s + f] = em
    return slot, ext, empty


CONTROLS = (None, "shuffle", "presence")
STREAM_SHUFFLE = 41


def render(tab, visible, target, layout=LEAKFREE, control=None, key=0):
    """The grid a model reads, the truth of its target cells and the cells laid by extension.

    tab: one table, tok [R, ROW_LEN], n [R, B], col [R, ROW_LEN], kind [R, B], frame_len [B] (and frame_row [B]: the
    frame row of every block, row 0 where not given); visible, target: boolean [R, ROW_LEN] in the rows' own
    coordinates (a mask scheme's plan; a codon that is not visible shows as MASK).
    control, applied after the layout to every row but a block's frame row: "shuffle" moves the columns of a block by one
    permutation shared by all those rows (drawn by the 64-bit hash of key and the block: the correspondence with the
    frame row is gone, the rows' alignment to each other and their content stay); "presence" shows every codon of those
    rows as MASK (only where a codon is stays).
    Returns grid uint8 [R, ROW_LEN], truth uint8 [R, ROW_LEN] (0 = no target, else the token to predict) and ext bool
    [R, ROW_LEN]."""
    if control not in CONTROLS:
        raise ValueError("unknown control %r (known: %s)" % (control, CONTROLS[1:]))
    tok, kind = np.asarray(tab["tok"]), np.asarray(tab["kind"])
    n = np.asarray(tab["n"]).astype(np.int64)
    frame = np.asarray(tab["frame_len"]).astype(np.int64)
    R = tok.shape[0]
    cells = cells_of(n)
    vis = np.asarray(visible, dtype=bool) & cells
    hidden = cells & ~vis
    slot, ext, empty = placement(tab, vis, layout)
    rr, kk = np.nonzero(cells & (slot >= 0))
    dest = slot[rr, kk]
    placed = np.zeros((R, L), bool)
    placed[rr, dest] = True
    token = np.full((R, L), A.PAD, np.uint8)
    token[rr, dest] = np.where(hidden[rr, kk], A.MASK, tok[rr, kk]).astype(np.uint8)
    want = np.asarray(target, dtype=bool) & hidden & (tok >= A.TARGET_MIN) & (tok <= A.TARGET_MAX)
    truth = np.zeros((R, L), np.uint8)
    keep = want[rr, kk]
    truth[rr[keep], dest[keep]] = tok[rr[keep], kk[keep]]
    grid = np.full((R, L), A.PAD, dtype=np.uint8)
    for b, (_, s, e) in enumerate(SLICES):
        p = placed[:, s:e]
        span = (np.cumsum(p, axis=1) > 0) & (np.cumsum(p[:, ::-1], axis=1)[:, ::-1] > 0)
        g = np.where(p, token[:, s:e], np.where(empty[:, s:e], A.EXT_EMPTY, np.where(span, A.GAP, A.PAD))).astype(np.uint8)
        no_codons = ~np.isin(kind[:, b], WITH_CODONS)
        g[no_codons] = A.PAD
        f = int(frame[b])
        g[no_codons, :f] = FILL[kind[no_codons, b]][:, None]
        g[:, f:] = A.PAD
        grid[:, s:e] = g
    ext &= (grid != A.PAD)
    truth[grid != A.MASK] = 0
    if control is not None:
        fr = frame_rows(tab)
        from seeddrift.train.rel.evalh import conditions as cond        # the 64-bit hash every plan uses
        for b, (_, s, e) in enumerate(SLICES):
            f = int(frame[b])
            rows = np.arange(R) != fr[b]
            if f == 0 or not rows.any():
                continue
            if control == "shuffle":
                perm = np.argsort(cond.hash_u01(int(key), STREAM_SHUFFLE, b, np.arange(f)), kind="stable")
                for a in (grid, truth, ext):
                    a[rows, s:s + f] = a[rows][:, s + perm]
            else:
                g = grid[rows, s:s + f]
                g[(g >= A.TARGET_MIN) & (g <= A.TARGET_MAX)] = A.MASK
                grid[rows, s:s + f] = g
    return grid, truth, ext


def anchor_distance(tab, visible):
    """[ROW_LEN]: for every column of a block's frame row, the distance in columns to the nearest column that is an
    anchor of some other row (a stored pair with both residues visible); -1 in a block without any anchor. A reporting
    key: how far a scored cell is from the nearest correspondence the leak-free layout keeps."""
    n = np.asarray(tab["n"]).astype(np.int64)
    col = np.asarray(tab["col"]).astype(np.int64)
    frame = np.asarray(tab["frame_len"]).astype(np.int64)
    cells = cells_of(n)
    vis = np.asarray(visible, bool) & cells
    fr = frame_rows(tab)
    out = np.full(L, -1, np.int64)
    for b, (_, s, e) in enumerate(SLICES):
        f = int(frame[b])
        if f == 0:
            continue
        cols = []
        for r in range(n.shape[0]):
            m = int(n[r, b])
            if r == fr[b] or m == 0:
                continue
            c = col[r, s:s + m]
            ok = (c >= 0) & (c < f)
            good = ok & vis[r, s:s + m]
            good[ok] &= vis[fr[b], s + c[ok]]
            cols.append(c[good])
        a = np.unique(np.concatenate(cols)) if cols else np.zeros(0, np.int64)
        if a.size:
            x = np.arange(f)
            i = np.clip(np.searchsorted(a, x), 1, a.size) - 1
            j = np.clip(i + 1, 0, a.size - 1)
            out[s:s + f] = np.minimum(np.abs(x - a[i]), np.abs(a[j] - x))
    return out


def causal_rows(tab, key=0, extra=2):
    """The target rows of a causal arm (the programme lead's decision): row 0 always, and `extra` rows drawn by the
    64-bit hash of key among the other rows that have content (fewer where there are fewer)."""
    from seeddrift.train.rel.evalh import conditions as cond
    kind = np.asarray(tab["kind"])
    have = np.flatnonzero((kind[1:] != KIND_NONE).any(axis=1)) + 1
    if have.size == 0:
        return [0]
    order = have[np.argsort(cond.hash_u01(int(key), STREAM_SHUFFLE + 1, 0, have), kind="stable")]
    return [0] + sorted(int(r) for r in order[:int(extra)])


def dropped(tab, visible, target, layout=LEAKFREE):
    """The number of target cells (hidden residues 1..65) a rendering cannot show."""
    tok = np.asarray(tab["tok"])
    grid, truth, _ = render(tab, visible, target, layout)
    n = np.asarray(tab["n"]).astype(np.int64)
    inside = np.zeros(tok.shape, bool)
    for b, (_, s, e) in enumerate(SLICES):
        inside[:, s:e] = np.arange(e - s)[None, :] < n[:, b:b + 1]
    want = np.asarray(target, bool) & inside & ~np.asarray(visible, bool) & (tok >= A.TARGET_MIN) & (tok <= A.TARGET_MAX)
    return int(want.sum() - (truth > 0).sum())
