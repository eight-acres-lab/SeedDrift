"""The reader, the mask schemes, the sampler and the held-out plan of the relational line's trainer (package W6), on a
small random table set written here in the shard format of the contract. numpy only.

The model interpreter has no pytest: `python tests/ci/test_rel_w6_data.py` runs every test."""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from seeddrift.train.rel import api as A  # noqa: E402
from seeddrift.train.rel.data import heldout_plan as H  # noqa: E402
from seeddrift.train.rel.data import sampler as S  # noqa: E402
from seeddrift.train.rel.data import tables as T  # noqa: E402

DTYPES = {k: v[0] for k, v in list(A.SHARD_FIELDS.items()) + list(A.META_FIELDS.items())}


def random_tables(n_tables, rows=16, seed=0, split=0):
    """Arrays of n_tables random tables (shard and metadata fields). Row 0 has a segment in every block; another row
    is a noisy copy of row 0 with residues dropped and inserted; the last tables have few or no other rows."""
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
                    out["kind"][t, r, b] = A.KIND_NOT_SEARCHED
                    continue
                keep = np.flatnonzero(rng.random(n0) > 0.1)                  # frame columns this row has a residue at
                seq = own[keep].copy()
                change = rng.random(len(seq)) < rng.uniform(0.0, 0.6)
                seq[change] = rng.integers(1, 65, size=int(change.sum()))
                col = keep.astype(np.int16)
                at = int(rng.integers(0, len(seq) + 1))                      # two residues of its own with no column
                seq = np.concatenate([seq[:at], rng.integers(1, 65, size=2).astype(np.uint8), seq[at:]])[:e - s]
                col = np.concatenate([col[:at], np.full(2, -1, np.int16), col[at:]])[:e - s]
                m = len(seq)
                out["tok"][t, r, s:s + m], out["col"][t, r, s:s + m], out["n"][t, r, b], out["kind"][t, r, b] = seq, col, m, A.KIND_PRESENT
                if role == A.ROLE_TARGET:
                    ok = col >= 0
                    out["identity"][t, r], out["genome"][t, r] = float((seq[ok] == own[col[ok]]).mean()), 1000 + r
    return out


def write_tableset(root, sets, kind=A.TABLE_SYNTHETIC, name="w6-test", sub="tables"):
    """sets: [(shard name, arrays)]. Writes <root>/<sub>/<shard>.<field>.npy and the manifest <root>/<sub>/tableset.json
    with the file list of the hand-off format (paths relative to root); returns the manifest path."""
    os.makedirs(os.path.join(root, sub), exist_ok=True)
    files = []
    for shard, arrays in sets:
        for f, a in arrays.items():
            rel = os.path.join(sub, "%s.%s.npy" % (shard, f))
            np.save(os.path.join(root, rel), a.astype(DTYPES[f]))
            files.append(dict(path=rel, role="shard" if f in A.SHARD_FIELDS else "meta"))
    path = os.path.join(root, sub, "tableset.json")
    with open(path, "w") as f:
        json.dump(dict(name=name, kind=kind, files=files, sha256="0" * 64, group_names=["g%02d" % g for g in range(12)]), f)
    return path


def _set(n=30, seed=0, split=None):
    d = tempfile.mkdtemp()
    return T.open_tableset(write_tableset(d, [("s0", random_tables(n, seed=seed))]), d, split)


def test_open_tableset():
    d = tempfile.mkdtemp()
    a, b = random_tables(10, seed=1), random_tables(8, seed=2, split=1)
    b["split"][:3] = 0
    man = write_tableset(d, [("s1", b), ("s0", a)])
    every, train, held = T.open_tableset(man, d), T.open_tableset(man, d, 0), T.open_tableset(man, d, 1)
    assert (len(every), len(train), len(held)) == (18, 13, 5) and every.rows == 16 and held.group_names[3] == "g03"
    assert (every.table(0)["tok"] == a["tok"][0]).all() and (every.table(10)["tok"] == b["tok"][0]).all()        # shards in name order
    assert (held.table(0)["tok"] == b["tok"][3]).all() and (train.table(12)["tok"] == b["tok"][2]).all()
    assert list(held.group) == list(b["group"][3:]) and np.isnan(held.meta(4)["identity"]).any()
    for split in (2,):
        try:
            T.open_tableset(man, d, split)
        except ValueError:
            continue
        raise AssertionError("a split without tables was accepted")
    with open(man) as f:
        doc = json.load(f)
    doc["files"] = [e for e in doc["files"] if not e["path"].endswith("s1.identity.npy")]
    with open(man, "w") as f:
        json.dump(doc, f)
    try:
        T.open_tableset(man, d)
    except ValueError:
        return
    raise AssertionError("a shard without its identity file was accepted")


def test_layout_arrays():
    assert T.L == 1024 and T.B == 19 and (T.T0, T.T1) == (768, 1024)
    assert T.BLOCK_OF[0] == 0 and T.BLOCK_OF[32] == 1 and T.BLOCK_OF[-1] == T.TARGET_BLOCK and T.POS[33] == 1
    assert list(np.bincount(T.ROLE)) == [512, 128, 128, 256]


def test_render_all_visible():
    ts = _set()
    tab = ts.table(0)
    cells = T.inside(tab)
    grid, truth = T.render(tab, cells, np.zeros_like(cells))
    assert not truth.any()
    assert (grid[0][cells[0]] == tab["tok"][0][cells[0]]).all()                    # row 0 is its own frame
    assert (grid[:, T.POS >= tab["frame_len"][T.BLOCK_OF]] == A.PAD).all()         # beyond the frame: padding
    for r in range(1, grid.shape[0]):
        for b, (_, s, e) in enumerate(T.SLICES):
            g = grid[r, s:s + tab["frame_len"][b]]
            if tab["kind"][r, b] == A.KIND_NOT_SEARCHED:
                assert (g == A.NOT_SEARCHED).all()
                continue
            n = tab["n"][r, b]
            col, tok = tab["col"][r, s:s + n], tab["tok"][r, s:s + n]
            assert (g[col[col >= 0]] == tok[col >= 0]).all()
            rest = np.ones(len(g), bool)
            rest[col[col >= 0]] = False
            assert (g[rest] == A.GAP).all()
    assert ((grid >= 1) & (grid <= A.NOT_SEARCHED)).sum() + (grid == A.PAD).sum() == grid.size


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
        # the content of the hidden cells does not reach the grid
        other = dict(tab, tok=np.where(hide, (tab["tok"] % 64) + 1, tab["tok"]).astype(np.uint8))
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


def test_collate():
    ts = _set(36)
    ex = S.Sampler(ts, seed=5).draw(2)
    c = S.collate(ex)
    assert c["grid"].shape == (sum(c["sizes"]), A.ROW_LEN) and len(c["row"]) == sum(int((e["truth"] > 0).sum()) for e in ex)
    assert (c["token"] >= 1).all() and (c["token"] <= 65).all() and (c["grid"][c["row"], c["col"]] == A.MASK).all()
    first = np.cumsum([0] + c["sizes"])
    assert ((c["row"] >= first[c["example"]]) & (c["row"] < first[c["example"] + 1])).all()


def test_heldout_plan():
    ts = _set(30)
    for i in (0, 3):
        tab = ts.table(i)
        tab = T.take_rows(tab, T.present_rows(tab))
        cells = T.inside(tab)
        t, visible, target = H.condition(tab, H.SCATTERED, i)
        assert (target[0, T.T0:T.T1] == (cells[0, T.T0:T.T1] & H.scattered_cells(i))).all() and target.sum() == target[0, T.T0:T.T1].sum()
        assert (H.scattered_cells(i) == H.scattered_cells(i)).all() and (H.scattered_cells(0) != H.scattered_cells(3)).any()
        t, visible, target = H.condition(tab, H.WHOLE, i)
        assert (target[0, T.T0:T.T1] == cells[0, T.T0:T.T1]).all() and (visible | target == cells).all()
        t, visible, target = H.condition(tab, H.WHOLE + "+" + H.NO_REL, i)
        assert t["tok"].shape[0] == 1 and target.shape[0] == 1
        t, visible, target = H.condition(tab, H.WHOLE + "+" + H.REPAIRED, i)
        k = tab["tok"].shape[0] - 1
        s = H.repair_shift(i, k)
        assert 1 <= s < k
        assert (t["tok"][:, T.T0:T.T1] == tab["tok"][:, T.T0:T.T1]).all() and (t["n"][:, -1] == tab["n"][:, -1]).all()
        assert (t["tok"][0] == tab["tok"][0]).all()
        for r in range(1, k + 1):
            src = (r - 1 + s) % k + 1
            assert src != r and (t["tok"][r, :T.T0] == tab["tok"][src, :T.T0]).all() and (t["kind"][r, :-1] == tab["kind"][src, :-1]).all()
            assert (t["col"][r, :T.T0] == tab["col"][src, :T.T0]).all() and (t["n"][r, :-1] == tab["n"][src, :-1]).all()
        T.render(t, visible, target)
    lone = T.take_rows(ts.table(27), [0])
    assert H.condition(lone, H.WHOLE + "+" + H.REPAIRED, 27) is None and H.condition(lone, H.SCATTERED, 27) is not None
    assert H.stratum([1.0, np.nan]) == "none" and H.stratum([1.0, 0.29, np.nan]) == "<0.3" and H.stratum([1.0, 0.3]) == "0.3-0.5"
    assert H.stratum([1.0, 0.95, 0.2]) == ">=0.9" and H.stratum([1.0, 0.9]) == ">=0.9" and H.stratum([1.0, 0.69]) == "0.5-0.7"
    assert len(H.CONDITIONS) == 5 and len(set(H.CONDITIONS)) == 5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
