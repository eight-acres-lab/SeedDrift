"""The aligner of the strict offline rendering (package W3): a numpy aligner whose scoring is written down here.

Each other row's stretch of a block is aligned to the frame row's stretch at the protein level (one amino acid per
codon, the standard code of the training arrays). Scoring:
  - substitution: BLOSUM62 (the matrix as NCBI distributes it) over the 20 amino acids and stop;
  - a residue that is hidden, or a codon "other", is unknown (X) and scores UNKNOWN against anything, so the alignment
    depends on the length of a hidden stretch and never on its content;
  - a gap of length g costs GAP_OPEN + g * GAP_EXTEND (11 and 1, the defaults of DIAMOND, which made the stored
    alignments);
  - semi-global: every codon of the other row is aligned (its stretch is the aligned part of the homolog, as stored),
    the frame row's residues before and after the aligned part cost nothing.
The dynamic programme is the affine-gap one (Gotoh), all other rows of a table at once; inside a row of the matrix the
gaps along the other row are a prefix maximum, exact because opening a gap never costs less than extending one. Ties
are broken towards a match, then a gap in the other row, then a gap in the frame row, so the result is deterministic.
"""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel import api as A

VERSION = "1"
GAP_OPEN, GAP_EXTEND = 11, 1
UNKNOWN = -1
NEG = -10 ** 9
AA = "ARNDCQEGHILKMFPSTWYV*"
_B62 = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -4
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -4
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3 -4
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3 -4
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -4
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2 -4
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2 -4
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -4
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3 -4
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -4
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2 -4
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -4
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -4
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -4
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2 -4
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -4
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -4
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -4
-4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1
"""
BLOSUM62 = np.array([[int(x) for x in line.split()] for line in _B62.strip().splitlines()], dtype=np.int64)
X = len(AA)                                                     # the index of the unknown residue
SCORE = np.full((X + 1, X + 1), UNKNOWN, np.int64)
SCORE[:X, :X] = BLOSUM62
_CODE_AA = "ACDEFGHIKLMNPQRSTVWY"                               # the order of the amino-acid classes 1..20 of the code


def residues(tok, known):
    """Amino-acid indices into AA (X for unknown) of codon tokens; known: the residue may be used."""
    from seeddrift.train.rel.evalh import score                 # the standard code of the training arrays
    cls = score.amino_acid_classes()                             # 1..20 amino acid, 21 stop, 22 other
    lut = np.full(66, X, np.int64)
    for c in range(1, 65):
        k = int(cls[c])
        lut[c] = AA.index(_CODE_AA[k - 1]) if 1 <= k <= 20 else (AA.index("*") if k == 21 else X)
    t = np.asarray(tok).astype(np.int64)
    out = np.where((t >= 0) & (t <= 65), lut[np.clip(t, 0, 65)], X)
    return np.where(np.asarray(known, bool), out, X)


def align(frame, others):
    """frame: amino-acid indices of the frame row's stretch (n,); others: a list of index arrays, one per other row.
    Returns (cols, scores): for every other row an array (m_k,) of the frame position each of its residues is aligned
    to (-1: aligned to a gap), and the alignment's score."""
    frame = np.asarray(frame, np.int64)
    n, K = int(frame.shape[0]), len(others)
    if K == 0:
        return [], []
    lens = np.array([int(np.asarray(o).shape[0]) for o in others])
    M = int(lens.max())
    Q = np.full((K, max(M, 1)), X, np.int64)
    for k, o in enumerate(others):
        Q[k, :lens[k]] = np.asarray(o, np.int64)
    Q = Q[:, :M]
    o_, e_ = GAP_OPEN, GAP_EXTEND
    j = np.arange(M + 1)
    H = np.empty((K, M + 1), np.int64)
    H[:, 0] = 0
    H[:, 1:] = -(o_ + e_ * j[1:])                               # other-row residues before the frame starts
    F = np.full((K, M + 1), NEG, np.int64)
    choice = np.zeros((n + 1, K, M + 1), np.int8)               # state H: 0 match, 1 gap in the frame row (E), 2 gap in the other row (F)
    choice_hp = np.zeros((n + 1, K, M + 1), np.int8)            # the state an E gap opened from: 0 match, 2 F
    e_open = np.zeros((n + 1, K, M + 1), bool)
    f_open = np.zeros((n + 1, K, M + 1), bool)
    choice[0, :, 1:] = 1
    e_open[0, :, 1] = True
    best, best_i = H.copy(), np.zeros((K, M + 1), np.int64)     # the frame's trailing residues are free
    for i in range(1, n + 1):
        diag = np.full((K, M + 1), NEG, np.int64)
        diag[:, 1:] = H[:, :-1] + SCORE[frame[i - 1]][Q]
        f_ext, f_new = F - e_, H - o_ - e_
        f_open[i] = f_new >= f_ext
        F = np.maximum(f_ext, f_new)
        F[:, 0] = NEG
        hp = np.maximum(diag, F)
        hp[:, 0] = 0                                            # the other row not started yet: the frame residue is free
        choice_hp[i] = np.where(diag >= F, 0, 2)
        a = hp + e_ * j[None, :]
        pm = np.maximum.accumulate(a, axis=1)
        E = np.full((K, M + 1), NEG, np.int64)
        E[:, 1:] = pm[:, :-1] - o_ - e_ * j[None, 1:]
        if M >= 1:
            e_open[i, :, 1] = True
        if M >= 2:
            e_open[i, :, 2:] = a[:, 1:-1] >= pm[:, :-2]
        choice[i] = np.where(diag >= np.maximum(E, F), 0, np.where(E >= F, 1, 2))
        H = np.maximum(hp, E)
        H[:, 0] = 0
        upd = H > best
        best, best_i = np.where(upd, H, best), np.where(upd, i, best_i)
    cols, scores = [], []
    for k in range(K):
        m = int(lens[k])
        col = np.full(m, -1, np.int64)
        scores.append(int(best[k, m]) if m else 0)
        i, jj, state = int(best_i[k, m]) if m else 0, m, "H"
        while jj > 0 and i > 0:
            c = int(choice[i, k, jj]) if state == "H" else (int(choice_hp[i, k, jj]) if state == "P" else (1 if state == "E" else 2))
            if c == 0:
                col[jj - 1] = i - 1
                i, jj, state = i - 1, jj - 1, "H"
            elif c == 1:                                        # other-row residue jj unpaired (a gap in the frame row)
                state = "P" if e_open[i, k, jj] else "E"
                jj -= 1
            else:                                               # frame residue i unpaired (a gap in the other row)
                state = "H" if f_open[i, k, jj] else "F"
                i -= 1
        cols.append(col)
    return cols, scores


def strict(tab, visible):
    """col [R, ROW_LEN] of a table whose correspondence is aligned again in every block where the frame row has a hidden
    residue, those residues unknown (and any hidden residue of another row); other blocks keep the stored col. The
    result depends on the lengths of the hidden stretches and on the visible residues only."""
    tok = np.asarray(tab["tok"])
    n = np.asarray(tab["n"]).astype(np.int64)
    frame_len = np.asarray(tab["frame_len"]).astype(np.int64)
    vis = np.asarray(visible, bool)
    col = np.array(tab["col"], dtype=np.int16, copy=True)
    for b, (_, s, e) in enumerate(A.block_slices()):
        f = int(frame_len[b])
        if f == 0 or vis[0, s:s + f].all():
            continue
        rows = [r for r in range(1, tok.shape[0]) if n[r, b] > 0]
        if not rows:
            continue
        frame = residues(tok[0, s:s + f], vis[0, s:s + f])
        others = [residues(tok[r, s:s + n[r, b]], vis[r, s:s + n[r, b]]) for r in rows]
        cols, _ = align(frame, others)
        for r, c in zip(rows, cols):
            col[r, s:s + n[r, b]] = c.astype(np.int16)
    return col


def frame_relative(tab, block=None):
    """The frame row of a block when the examined row is hidden there as a whole (the programme lead's decision): the
    first other row, in the order of the table, that has a stretch in the block; None when no row has one. The order of
    the rows comes from genome distance and random draws, never from the hidden residues."""
    b = A.N_BLOCKS - 1 if block is None else int(block)
    n = np.asarray(tab["n"]).astype(np.int64)
    rows = [r for r in range(1, n.shape[0]) if n[r, b] > 0]
    return rows[0] if rows else None


def frame_layout(tab, block=None):
    """(col, frame_row, frame_len) of a table whose block (the target block by default) is laid on its frame relative:
    the frame relative keeps its own coordinates, the examined row is laid from the first column (its first codon on the
    frame's), every other row by its alignment to the frame relative (aligned here, never through the examined row).
    Nothing of the examined row but its length is used. Other blocks are left as stored."""
    b = A.N_BLOCKS - 1 if block is None else int(block)
    s = A.block_slices()[b][1]
    tok = np.asarray(tab["tok"])
    n = np.asarray(tab["n"]).astype(np.int64)
    col = np.array(tab["col"], dtype=np.int16, copy=True)
    frame_row = np.array(tab.get("frame_row", np.zeros(A.N_BLOCKS)), dtype=np.int8, copy=True).reshape(-1)
    frame_len = np.array(tab["frame_len"], dtype=np.int16, copy=True)
    fr = frame_relative(tab, b)
    if fr is None:
        return col, frame_row, frame_len
    nf, n0 = int(n[fr, b]), int(n[0, b])
    col[fr, s:s + nf] = np.arange(nf)
    col[0, s:s + n0] = np.arange(n0)
    rows = [r for r in range(1, tok.shape[0]) if r != fr and n[r, b] > 0]
    if rows:
        frame = residues(tok[fr, s:s + nf], np.ones(nf, bool))
        cols, _ = align(frame, [residues(tok[r, s:s + n[r, b]], np.ones(int(n[r, b]), bool)) for r in rows])
        for r, c in zip(rows, cols):
            col[r, s:s + n[r, b]] = c.astype(np.int16)
    frame_row[b], frame_len[b] = fr, max(nf, n0)
    return col, frame_row, frame_len
