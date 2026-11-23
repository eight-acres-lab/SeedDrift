"""The reader, the grid, the mask schemes and the sampler of the relational line's trainer (package W6), on small random
tables written here in the shard format of the contract with a table-set manifest of the hand-off format. numpy only.

The model interpreter has no pytest: `python tests/ci/test_rel_w6_data.py` runs every test."""
import hashlib
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from seeddrift.train.rel import api as A  # noqa: E402
from seeddrift.train.rel.data import sampler as S  # noqa: E402
from seeddrift.train.rel.data import tables as T  # noqa: E402
from seeddrift.train.rel.evalh import conditions as cond  # noqa: E402

DTYPES = {k: v[0] for k, v in list(A.SHARD_FIELDS.items()) + list(A.META_FIELDS.items()) + list(getattr(A, "SHARD_FIELDS_ADDED", {}).items())
          + list(getattr(A, "META_FIELDS_ADDED", {}).items())}


def random_tables(n_tables, rows=16, seed=0, split=0):
    """Arrays of n_tables random tables (shard and metadata fields). Row 0 has a segment in every block; another row
    is a noisy copy of row 0 over a part of it (its aligned span), with residues dropped and two inserted; the last
    three tables have 0, 1 and 2 other rows. The rows that do not exist are KIND_NONE everywhere."""
    rng = np.random.default_rng(seed)
    out = dict(tok=np.zeros((n_tables, rows, A.ROW_LEN), np.uint8), n=np.zeros((n_tables, rows, A.N_BLOCKS), np.int16),
               col=np.full((n_tables, rows, A.ROW_LEN), -1, np.int16), frame_row=np.zeros((n_tables, A.N_BLOCKS), np.int8),
               frame_len=np.zeros((n_tables, A.N_BLOCKS), np.int16), kind=np.zeros((n_tables, rows, A.N_BLOCKS), np.uint8),
               start=np.full((n_tables, rows, A.N_BLOCKS), -1, np.int64), strand=np.zeros((n_tables, rows, A.N_BLOCKS), np.int8),
               replicon=np.full((n_tables, rows, A.N_BLOCKS), -1, np.int16), genome=np.full((n_tables, rows), -1, np.int32),
               group=(np.arange(n_tables) % 12).astype(np.int8), split=np.full(n_tables, split, np.int8),
               gene=np.zeros((n_tables, A.N_BLOCKS), np.int32), tier=np.full((n_tables, rows), -1, np.int8),
               identity=np.full((n_tables, rows), np.nan, np.float32))
    for t in range(n_tables):
        others = rows - 1 if t < n_tables - 3 else (0, 1, 2)[t - (n_tables - 3)]
        out["genome"][t, 0], out["identity"][t, 0] = t, 1.0
        for b, (role, s, e) in enumerate(A.block_slices()):
            n0 = int(rng.integers((e - s) // 2, e - s + 1))
            own = rng.integers(1, 66, size=n0).astype(np.uint8)
            out["tok"][t, 0, s:s + n0], out["n"][t, 0, b], out["kind"][t, 0, b] = own, n0, A.KIND_PRESENT
            out["col"][t, 0, s:s + n0], out["frame_len"][t, b] = np.arange(n0), n0
            for r in range(1, others + 1):
                if role != A.ROLE_TARGET and rng.random() < 0.4:
                    out["kind"][t, r, b] = A.KIND_NOT_SEARCHED if rng.random() < 0.8 else A.KIND_TRUNCATED
                    continue
                a = int(rng.integers(0, n0 // 4 + 1))                       # the aligned span: frame columns a .. z - 1
                z = int(rng.integers(max(a + 1, 3 * n0 // 4), n0 + 1))
                keep = a + np.flatnonzero(rng.random(z - a) > 0.1)          # frame columns this row has a residue at
                if not len(keep):
                    keep = np.array([a])
                seq = own[keep].copy()
                change = rng.random(len(seq)) < rng.uniform(0.0, 0.6)
                seq[change] = rng.integers(1, 65, size=int(change.sum()))
                col = keep.astype(np.int16)
                at = int(rng.integers(1, len(seq) + 1))                     # two residues of its own with no column
                seq = np.concatenate([seq[:at], rng.integers(1, 65, size=2).astype(np.uint8), seq[at:]])[:e - s]
                col = np.concatenate([col[:at], np.full(2, -1, np.int16), col[at:]])[:e - s]
                m = len(seq)
                out["tok"][t, r, s:s + m], out["col"][t, r, s:s + m], out["n"][t, r, b], out["kind"][t, r, b] = seq, col, m, A.KIND_PRESENT
                if role == A.ROLE_TARGET:
                    ok = col >= 0
                    out["identity"][t, r], out["genome"][t, r] = float((seq[ok] == own[col[ok]]).mean()), 1000 + r
    return out


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def v1_tables(n_tables, rows=8, seed=0, split=0):
    """random_tables with the fields of tables of version 1: genome_of_row (two rows of one genome where there are
    enough rows), and every row's gene and first codon in each block it has (row_gene, row_first)."""
    out = random_tables(n_tables, rows, seed, split)
    rng = np.random.default_rng(seed + 7)
    out["genome_of_row"] = np.full((n_tables, rows), -1, np.int8)
    out["row_gene"] = np.full((n_tables, rows, A.N_BLOCKS), -1, np.int32)
    out["row_first"] = np.full((n_tables, rows, A.N_BLOCKS), -1, np.int16)
    for t in range(n_tables):
        exists = (out["kind"][t] != T.KIND_NONE).any(axis=1)
        exists[0] = True
        number = 0
        for r in range(rows):
            if not exists[r]:
                continue
            if r >= 2 and r % 2 == 0:                                   # rows 2 and 3, 4 and 5, ...: one genome each
                number += 1
            out["genome_of_row"][t, r] = 0 if r == 0 else number + (1 if r == 1 else 0)
            out["genome"][t, r] = 5000 + int(out["genome_of_row"][t, r])
            has = out["n"][t, r] > 0
            out["row_gene"][t, r] = np.where(has, rng.integers(0, 40, A.N_BLOCKS) + 100 * r, -1)
            out["row_first"][t, r] = np.where(has, rng.integers(0, 6, A.N_BLOCKS), -1)
    return out


class FakeIndex:
    """The interface of package W2's index of the uncapped search (pairs, tgene, alignments), holding given alignments
    (as package W3's test of the online frame layout has it; here without pytest, for the model interpreter)."""

    def __init__(self):
        self.store, self.t = {}, []

    def add(self, qg, tg, qi, ti, hsps):
        self.store.setdefault((qg, tg, qi), []).append(len(self.t))
        self.t.append((ti, hsps))

    def pairs(self, qg, tg, qi):
        return list(self.store.get((qg, tg, qi), []))

    def tgene(self, p):
        return self.t[p][0]

    def alignments(self, p):
        return self.t[p][1]


def to_hsp(col, qf, tf):
    """One alignment (qs, qe, ss, se, cigar), 1-based in the two genes, of the pairs col[j] = query position (-1 none)
    of target position j, shifted by the stretches' first codons qf and tf."""
    pairs = [(int(c), j) for j, c in enumerate(col) if c >= 0]
    ops, (pq, pt) = [], pairs[0]
    run = 1
    for q, t in pairs[1:]:
        dq, dt = q - pq - 1, t - pt - 1
        if dq or dt:
            ops.append(b"%dM" % run)
            run = 0
            if dq:
                ops.append(b"%dI" % dq)
            if dt:
                ops.append(b"%dD" % dt)
        run += 1
        pq, pt = q, t
    ops.append(b"%dM" % run)
    return (pairs[0][0] + qf + 1, pq + qf + 1, pairs[0][1] + tf + 1, pt + tf + 1, b"".join(ops))


def frame_index_of(arrays, ix=None):
    """A FakeIndex holding, for every two rows of every table, the alignment of their target-block stretches their
    stored columns imply (codons on one frame column)."""
    ix = FakeIndex() if ix is None else ix
    b = A.N_BLOCKS - 1
    s = T.SLICES[b][1]
    N, R = arrays["n"].shape[:2]
    for t in range(N):
        for a in range(R):
            for c in range(R):
                na, nc = int(arrays["n"][t, a, b]), int(arrays["n"][t, c, b])
                if a == c or not na or not nc or arrays["row_gene"][t, a, b] < 0 or arrays["row_gene"][t, c, b] < 0:
                    continue
                inv = {int(x): i for i, x in enumerate(arrays["col"][t, a, s:s + na]) if x >= 0}
                m = np.array([inv.get(int(x), -1) if x >= 0 else -1 for x in arrays["col"][t, c, s:s + nc]])
                if (m >= 0).sum() < 2:
                    continue
                ix.add(int(arrays["genome"][t, a]), int(arrays["genome"][t, c]), int(arrays["row_gene"][t, a, b]), int(arrays["row_gene"][t, c, b]),
                       [to_hsp(m, int(arrays["row_first"][t, a, b]), int(arrays["row_first"][t, c, b]))])
    return ix


def write_tableset(root, sets, kind=A.TABLE_SYNTHETIC, name="w6-test", sub="tables"):
    """sets: [(shard, arrays)], a shard named <part>/<name> lying in the folder <sub>/<part> (one split per part).
    Writes the files <shard>.<field>.npy, an index with the group names and the manifest <root>/<sub>/tableset.json
    in the hand-off format (files relative to root, with their hashes); returns the manifest path."""
    os.makedirs(os.path.join(root, sub), exist_ok=True)
    files = []
    for shard, arrays in sets:
        os.makedirs(os.path.dirname(os.path.join(root, sub, shard)), exist_ok=True)
        for f, a in arrays.items():
            rel = os.path.join(sub, "%s.%s.npy" % (shard, f))
            np.save(os.path.join(root, rel), a.astype(DTYPES[f]))
            files.append(dict(path=rel, role="shard" if f in A.SHARD_FIELDS or f in getattr(A, "SHARD_FIELDS_ADDED", {}) else "meta"))
    with open(os.path.join(root, sub, "index.json"), "w") as f:
        json.dump(dict(groups=["g%02d" % g for g in range(12)]), f)
    files.append(dict(path=os.path.join(sub, "index.json"), role="index"))
    for e in files:
        e.update(sha256=_sha(os.path.join(root, e["path"])), bytes=os.path.getsize(os.path.join(root, e["path"])))
    path = os.path.join(root, sub, "tableset.json")
    with open(path, "w") as f:
        json.dump(dict(format="rel-tableset", format_version=1, name=name, kind=kind, api_version=T.API_VERSION, files=files, sha256="0" * 64), f)
    return path


def _set(n=30, seed=0):
    d = tempfile.mkdtemp()
    return T.open_split(d, write_tableset(d, [("train/s0000", random_tables(n, seed=seed))]), 0)


def test_layout_arrays():
    assert T.L == 1024 and T.B == 19 and (T.T0, T.T1) == (768, 1024)
    assert T.BLOCK_OF[0] == 0 and T.BLOCK_OF[32] == 1 and T.BLOCK_OF[-1] == T.TARGET_BLOCK and T.POS[33] == 1
    assert list(np.bincount(T.ROLE)) == [512, 128, 128, 256]


def test_open_split():
    d = tempfile.mkdtemp()
    a, b, c = random_tables(10, seed=1), random_tables(8, seed=2, split=1), random_tables(5, seed=3, split=1)
    man = write_tableset(d, [("train/s0000", a), ("heldout/s0001", c), ("heldout/s0000", b)])
    train, held = T.open_split(d, man, 0), T.open_split(d, man, 1)
    assert (len(train), len(held)) == (10, 13) and held.rows == 16 and held.name == "w6-test"
    assert (held.table(0)["tok"] == b["tok"][0]).all() and (held.table(8)["tok"] == c["tok"][0]).all()           # shards in name order
    assert (train.table(9)["col"] == a["col"][9]).all() and set(held.table(3)) == set(T.MODEL_FIELDS)
    assert list(held.group) == list(b["group"]) + list(c["group"]) and np.isnan(held.load(("identity",))["identity"][7]).any()
    try:
        T.open_split(d, man, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("a split without tables was accepted")
    b2 = dict(b, split=b["split"].copy())
    b2["split"][3] = 0
    mixed = write_tableset(d, [("heldout/s0000", b2)], sub="mixed")
    try:
        T.open_split(d, mixed, 1)
    except ValueError:
        return
    raise AssertionError("a folder of two splits was accepted")


def test_render_all_visible():
    ts = _set()
    for i in (0, 1, 27):
        tab = ts.table(i)
        tab = T.take_rows(tab, T.present_rows(tab))
        cells = T.inside(tab)
        grid, truth = T.render(tab, cells, np.zeros_like(cells))
        assert not truth.any() and (grid != A.MASK).all()
        assert (grid[0][cells[0]] == tab["tok"][0][cells[0]]).all()                    # row 0 is its own frame: no GAP, no PAD inside
        assert (grid[:, T.POS >= tab["frame_len"][T.BLOCK_OF]] == A.PAD).all()          # beyond the frame: PAD
        for r in range(1, grid.shape[0]):
            for b, (_, s, e) in enumerate(T.SLICES):
                g = grid[r, s:s + tab["frame_len"][b]]
                if tab["kind"][r, b] in (A.KIND_NOT_SEARCHED, A.KIND_TRUNCATED):
                    assert (g == (A.NOT_SEARCHED if tab["kind"][r, b] == A.KIND_NOT_SEARCHED else A.TRUNCATED)).all()
                    continue
                n = tab["n"][r, b]
                col, tok = tab["col"][r, s:s + n], tab["tok"][r, s:s + n]
                placed = col[col >= 0]
                assert (g[placed] == tok[col >= 0]).all()
                lo, hi = placed.min(), placed.max()
                rest = np.ones(len(g), bool)
                rest[placed] = False
                inner = rest.copy()
                inner[:lo], inner[hi + 1:] = False, False
                assert (g[inner] == A.GAP).all() and (g[rest & ~inner] == A.PAD).all()     # GAP inside the span, PAD outside


def test_render_hides_and_scores():
    ts = _set()
    rng = np.random.default_rng(1)
    for i in range(5):
        tab = ts.table(i)
        cells = T.inside(tab)
        hide = cells & (rng.random(cells.shape) < 0.2)
        grid, truth = T.render(tab, cells & ~hide, hide)
        assert ((truth > 0) <= (grid == A.MASK)).all() and (truth <= A.TARGET_MAX).all()
        assert (truth[0][hide[0]] == tab["tok"][0][hide[0]]).all() and not truth[0][~hide[0]].any()
        placed = hide & (tab["col"] >= 0)
        assert (truth > 0).sum() == placed.sum() and (grid == A.MASK).sum() == placed.sum()
        other = dict(tab, tok=np.where(hide, (tab["tok"] % 64) + 1, tab["tok"]).astype(np.uint8))      # the content of hidden cells
        grid2, truth2 = T.render(other, cells & ~hide, hide)
        assert (grid2 == grid).all() and (truth2 != truth).any()


def test_render_refuses_a_column_beyond_the_frame():
    tab = _set().table(0)
    tab["col"][1, T.T0] = tab["frame_len"][T.TARGET_BLOCK]
    cells = T.inside(tab)
    try:
        T.render(tab, cells, np.zeros_like(cells))
    except ValueError:
        return
    raise AssertionError("a column beyond the frame was accepted")


def test_rows():
    ts = _set(30)
    assert list(T.present_rows(ts.table(0))) == list(range(16))
    assert list(T.present_rows(ts.table(27))) == [0] and list(T.present_rows(ts.table(29))) == [0, 1, 2]
    one = T.take_rows(ts.table(0), [0])
    assert one["tok"].shape == (1, A.ROW_LEN) and one["kind"].shape == (1, A.N_BLOCKS)
    tab = ts.table(29)
    grid, _ = T.render(tab, T.inside(tab), np.zeros((16, A.ROW_LEN), bool))
    assert (grid[3:] == A.PAD).all()                                              # rows that do not exist: PAD only


def test_mask_schemes():
    ts = _set()
    tab = ts.table(1)
    cells = T.inside(tab)
    meta = dict(n=tab["n"], kind=tab["kind"])
    for name in ("scattered", "span", "whole_gene"):
        for k in range(20):
            visible, target = S.mask_scheme(name).plan(meta, np.random.default_rng(k))
            assert not (visible & target).any() and ((visible | target) == cells).all() and target.any()
            rows = np.flatnonzero(target.any(axis=1))
            if name == "span":
                j = np.flatnonzero(target[rows[0]])
                assert len(rows) == 1 and (np.diff(j) == 1).all() and len(set(T.BLOCK_OF[j])) == 1 and len(j) <= 128
            if name == "whole_gene":
                assert list(rows) == [0] and (target[0, T.T0:T.T1] == cells[0, T.T0:T.T1]).all() and not target[0, :T.T0].any()
    rate = np.mean([S.mask_scheme("scattered").plan(meta, np.random.default_rng(k))[1].sum() / cells.sum() for k in range(20)])
    assert abs(rate - 0.15) < 0.01


def test_sampler():
    ts = _set(36)
    a, b = S.Sampler(ts, seed=3), S.Sampler(ts, seed=3)
    seen = set()
    for step in (0, 1, 7):
        x, y = a.draw(step), b.draw(step)
        assert len(x) == 4
        for u, v in zip(x, y):
            assert (u["grid"] == v["grid"]).all() and (u["truth"] == v["truth"]).all() and u["table"] == v["table"]
            assert 1 <= u["grid"].shape[0] <= 16 and u["truth"].any()
            seen.add(u["scheme"])
    assert any((a.draw(0)[k]["grid"].shape != a.draw(1)[k]["grid"].shape) or (a.draw(0)[k]["table"] != a.draw(1)[k]["table"]) for k in range(4))
    rows = [e["grid"].shape[0] for s in range(60) for e in a.draw(s)]
    assert max(rows) == 16 and min(r for r in rows if r >= 4) == 4 and len(seen) >= 2
    alone = S.Sampler(ts, seed=3, relatives=False)
    assert all(e["grid"].shape[0] == 1 for s in range(5) for e in alone.draw(s))
    only = S.Sampler(ts, seed=3, masks=["whole_gene"])
    e = only.draw(0)[0]
    assert not e["truth"][1:].any() and not e["truth"][0, :T.T0].any()


def test_sampler_layouts_per_mask_scheme():
    """Leak-free for the scattered and span samples, stored for the whole gene: the leak-free examples carry ext and
    their layout, the stored ones neither ext nor any difference from the default sampler."""
    from seeddrift.train.rel.tables import render as RD
    ts = _set(36)
    lay = {"scattered": A.LAYOUT_LEAKFREE, "span": A.LAYOUT_LEAKFREE, "whole_gene": A.LAYOUT_STORED}
    a, b = S.Sampler(ts, seed=3, layouts=lay), S.Sampler(ts, seed=3)
    kinds = set()
    for step in range(12):
        for u, v in zip(a.draw(step), b.draw(step)):
            assert u["table"] == v["table"] and u["scheme"] == v["scheme"]                      # the same draws
            name = a.masks[u["scheme"]]
            assert u["layout"] == lay[name] and v["layout"] == A.LAYOUT_STORED and "ext" not in v
            if lay[name] == A.LAYOUT_STORED:
                assert "ext" not in u and (u["grid"] == v["grid"]).all() and (u["truth"] == v["truth"]).all()
            else:
                assert u["ext"].shape == u["grid"].shape and not (u["grid"][u["truth"] > 0] != A.MASK).any()
            kinds.add(u["layout"])
    assert kinds == {A.LAYOUT_LEAKFREE, A.LAYOUT_STORED} and "ext" in S.collate(a.draw(0) + a.draw(1))
    for bad in ({"scattered": "sideways"}, {"anyorder": A.LAYOUT_LEAKFREE}):
        try:
            S.Sampler(ts, seed=3, layouts=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("layouts %r were accepted" % (bad,))
    assert RD.LEAKFREE == A.LAYOUT_LEAKFREE


def test_sampler_keeps_the_frame_row_of_a_block():
    """A block framed on another row than row 0 (the whole gene of tables framed on a relative) keeps that row whatever
    rows are drawn, renumbered to its place among the rows kept; tables framed on row 0 draw as before."""
    ts = _set(36)
    i = next(t for t in range(len(ts)) if len(T.present_rows(ts.table(t))) >= 6)
    tab = ts.table(i)
    frame = int(T.present_rows(tab)[-1])
    framed = dict(tab, frame_row=np.where(np.arange(A.N_BLOCKS) == A.N_BLOCKS - 1, frame, 0))

    class One:
        group, rows = np.zeros(1, np.int64), ts.rows

        def __len__(self):
            return 1

        def table(self, _):
            return framed
    s = S.Sampler(One(), seed=1, rows=(2, 3), masks=["scattered"])
    for step in range(20):
        e = s.draw(step)[0]
        assert 2 <= e["grid"].shape[0] <= 3 and frame in e["rows"].tolist() and e["rows"][0] == 0
    kept = T.take_rows(framed, np.array([0, frame]))
    assert kept["frame_row"].tolist() == [0] * (A.N_BLOCKS - 1) + [1]
    assert T.take_rows(framed, np.array([0, 1]))["frame_row"][-1] == -1                       # left out: the renderer frames on row 0
    plain = [e["table"] for st in range(5) for e in S.Sampler(ts, seed=3).draw(st)]
    assert plain == [e["table"] for st in range(5) for e in S.Sampler(ts, seed=3).draw(st)]


def test_sampler_lays_the_whole_gene_on_its_frame_relative():
    """Tables of version 1: with the pair index, a whole-gene sample's other rows are placed on the frame relative by
    their stored alignments (few cells laid by extension); without it they are laid from the first column, all by
    extension (W3's fallback). genome_of_row travels with the rows kept, into the batch."""
    root = tempfile.mkdtemp()
    arrays = v1_tables(24, rows=8, seed=4)
    man = write_tableset(root, [("train/s0000", arrays)], kind=A.TABLE_REAL)
    ts = T.open_split(root, man, 0)
    assert ts.meta(3)["row_gene"].shape == (8, A.N_BLOCKS)
    lay = {"whole_gene": A.LAYOUT_LEAKFREE}
    framed = S.Sampler(ts, seed=2, masks=["whole_gene"], rows=(3, 8), layouts=lay, frame_index=frame_index_of(arrays))
    bare = S.Sampler(ts, seed=2, masks=["whole_gene"], rows=(3, 8), layouts=lay)
    t0 = T.T0
    ext_f = ext_b = shown_f = 0
    for step in range(12):
        for u, v in zip(framed.draw(step), bare.draw(step)):
            assert u["table"] == v["table"] and (u["rows"] == v["rows"]).all()
            assert (u["genome_of_row"] == arrays["genome_of_row"][u["table"]][u["rows"]]).all()
            ext_f += int(u["ext"][1:, t0:].sum())
            ext_b += int(v["ext"][1:, t0:].sum())
            shown_f += int(((u["grid"][1:, t0:] >= A.TARGET_MIN) & (u["grid"][1:, t0:] <= A.TARGET_MAX)).sum())
    assert shown_f > 0 and ext_f < ext_b
    c = S.collate(framed.draw(0))
    assert c["genome"].shape == (c["grid"].shape[0],) and c["genome"].dtype == np.int64


def test_sampler_lays_the_whole_gene_by_the_strict_aligner():
    """whole_gene_layout frame (round 1): the rows drawn first, then W3's strict aligner (align.frame_layout) lays the
    table on its frame relative and the renderer renders it leak-free; the rows and masks drawn are those of the
    leak-free sampler, the grid that of frame_layout; the sampler counts the tables it laid and the seconds it took."""
    from seeddrift.train.rel.tables import align as AL
    from seeddrift.train.rel.tables import render as RD
    root = tempfile.mkdtemp()
    arrays = v1_tables(24, rows=8, seed=4)
    ts = T.open_split(root, write_tableset(root, [("train/s0000", arrays)], kind=A.TABLE_REAL), 0)
    framed = S.Sampler(ts, seed=2, masks=["scattered", "whole_gene"], rows=(3, 8), layouts={"scattered": A.LAYOUT_LEAKFREE, "whole_gene": S.FRAME})
    plain = S.Sampler(ts, seed=2, masks=["scattered", "whole_gene"], rows=(3, 8), layouts={"scattered": A.LAYOUT_LEAKFREE, "whole_gene": A.LAYOUT_LEAKFREE})
    seen, differ = 0, 0
    for step in range(16):
        for u, v in zip(framed.draw(step), plain.draw(step)):
            assert u["table"] == v["table"] and (u["rows"] == v["rows"]).all() and u["scheme"] == v["scheme"] and (u["truth"] == v["truth"]).all()
            if framed.masks[u["scheme"]] != "whole_gene":
                assert u["layout"] == A.LAYOUT_LEAKFREE and (u["grid"] == v["grid"]).all()
                continue
            seen += 1
            assert u["layout"] == S.FRAME and "ext" in u
            tab = T.take_rows(ts.table(u["table"]), u["rows"])
            col, frame_row, frame_len = AL.frame_layout(tab)
            visible, target = S.mask_scheme("whole_gene").plan(dict(n=tab["n"], kind=tab["kind"]), np.random.default_rng(0))   # draws nothing
            grid, truth, ext = RD.render(dict(tab, col=col, frame_row=frame_row, frame_len=frame_len), visible, target, layout=A.LAYOUT_LEAKFREE)
            assert (grid == u["grid"]).all() and (truth == u["truth"]).all() and (ext == u["ext"]).all()
            differ += int((u["grid"] != v["grid"]).any())
    assert seen > 0 and differ > 0 and framed.frame_tables == seen and framed.frame_s > 0
    try:
        S.Sampler(ts, seed=2, masks=["scattered"], layouts={"scattered": S.FRAME})
    except ValueError as err:
        assert "whole_gene only" in str(err)
    else:
        raise AssertionError("the frame layout was accepted for scattered samples")


def test_sampler_share_of_tables_without_relatives():
    ts = _set(30)                                         # table 27 has no other row; its group 3 also holds tables 3 and 15
    never = {e["table"] for st in range(200) for e in S.Sampler(ts, seed=3, alone_share=0.0).draw(st)}
    natural = {e["table"] for st in range(200) for e in S.Sampler(ts, seed=3).draw(st)}
    alone = {e["table"] for st in range(200) for e in S.Sampler(ts, seed=3, relatives=False, alone_share=0.0).draw(st)}
    assert 27 not in never and 27 in natural and 27 in alone and {3, 15} <= never


def test_anyorder_scheme_and_its_weights():
    ts = _set(36)
    tab = ts.table(1)
    cells = T.inside(tab)
    shares = []
    for k in range(40):
        visible, target = S.mask_scheme("anyorder").plan(dict(n=tab["n"], kind=tab["kind"]), np.random.default_rng(k))
        assert not (visible & target).any() and ((visible | target) == cells).all() and target.any()
        shares.append(target.sum() / cells.sum())
    assert 0.0 < min(shares) < 0.2 and max(shares) > 0.8
    s = S.Sampler(ts, seed=4, masks=["anyorder"])
    ex = s.draw(0)
    t = [int((e["truth"] > 0).sum()) for e in ex]
    assert abs(sum(e["weight"] * n for e, n in zip(ex, t)) - sum(t)) < 1e-6
    ratio = [n / e["cells"] for e, n in zip(ex, t)]
    assert all(abs(e["weight"] * r - ex[0]["weight"] * ratio[0]) < 1e-9 for e, r in zip(ex, ratio))     # weight x ratio is the same for all
    c = S.collate(ex)
    assert c["weight"].shape == c["token"].shape and abs(float(c["weight"].sum()) - sum(t)) < 1e-3
    assert "weight" not in S.collate(S.Sampler(ts, seed=4).draw(0))
    try:
        S.Sampler(ts, seed=4, masks=["anyorder", "scattered"])
    except ValueError:
        return
    raise AssertionError("weighted and unweighted schemes were mixed")


def test_collate():
    ts = _set(36)
    ex = S.Sampler(ts, seed=5).draw(2)
    c = S.collate(ex)
    assert c["grid"].shape == (sum(c["sizes"]), A.ROW_LEN) and len(c["row"]) == sum(int((e["truth"] > 0).sum()) for e in ex)
    assert (c["token"] >= 1).all() and (c["token"] <= 65).all() and (c["grid"][c["row"], c["col"]] == A.MASK).all()
    first = np.cumsum([0] + c["sizes"])
    assert ((c["row"] >= first[c["example"]]) & (c["row"] < first[c["example"] + 1])).all()
    assert "ext" not in c                                             # the stored layout marks no cell laid by extension
    ex[1] = dict(ex[1], ext=np.ones(ex[1]["grid"].shape, dtype=bool))
    c = S.collate(ex)
    rows = np.repeat(np.arange(len(ex)), c["sizes"])
    assert c["ext"].shape == c["grid"].shape and c["ext"][rows == 1].all() and not c["ext"][rows != 1].any()


def test_evaluator_plan_renders():
    """The evaluator's plan (seeddrift/train/rel/evalh, the one every arm and baseline shares) through the model's grid:
    the grid scores exactly the planned cells, and their content does not reach it."""
    ts = _set(30)
    ids = np.array([0, 3, 27, 28, 29])
    data = ts.load(T.MODEL_FIELDS)
    batch = {f: v[ids] for f, v in data.items()}
    for c in cond.CONDITIONS:
        plan = cond.materialise(batch, ids, c)
        for k in range(len(ids)):
            if not plan["has"][k]:
                assert c.endswith(cond.REPAIRED) and ids[k] in (27, 28)
                continue
            planned = dict(tok=plan["tok"][k], n=plan["n"][k], col=plan["col"][k], kind=plan["kind"][k], frame_len=plan["frame_len"][k])
            keep = T.present_rows(planned)
            grid, truth = T.render(T.take_rows(planned, keep), plan["visible"][k][keep], plan["target"][k][keep])
            assert (truth > 0).sum() == plan["target"][k].sum() > 0 and not truth[1:].any() and not truth[0, :T.T0].any()
            assert (truth[0, T.T0:T.T1] > 0).sum() == (grid[0] == A.MASK).sum() and (grid[1:] != A.MASK).all()
            if cond.parse(c)[1] == cond.NO_RELATIVES:
                assert grid.shape[0] == 1
            changed = dict(planned, tok=np.where(plan["target"][k], (planned["tok"] % 64) + 1, planned["tok"]).astype(np.uint8))
            grid2, _ = T.render(T.take_rows(changed, keep), plan["visible"][k][keep], plan["target"][k][keep])
            assert (grid2 == grid).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
