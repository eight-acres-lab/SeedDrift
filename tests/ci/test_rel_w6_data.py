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

DTYPES = {k: v[0] for k, v in list(A.SHARD_FIELDS.items()) + list(A.META_FIELDS.items())}


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
            files.append(dict(path=rel, role="shard" if f in A.SHARD_FIELDS else "meta"))
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


def test_sampler_share_of_tables_without_relatives():
    ts = _set(30)                                         # table 27 has no other row; its group 3 also holds tables 3 and 15
    never = {e["table"] for st in range(200) for e in S.Sampler(ts, seed=3, alone_share=0.0).draw(st)}
    natural = {e["table"] for st in range(200) for e in S.Sampler(ts, seed=3).draw(st)}
    alone = {e["table"] for st in range(200) for e in S.Sampler(ts, seed=3, relatives=False, alone_share=0.0).draw(st)}
    assert 27 not in never and 27 in natural and 27 in alone and {3, 15} <= never


def test_collate():
    ts = _set(36)
    ex = S.Sampler(ts, seed=5).draw(2)
    c = S.collate(ex)
    assert c["grid"].shape == (sum(c["sizes"]), A.ROW_LEN) and len(c["row"]) == sum(int((e["truth"] > 0).sum()) for e in ex)
    assert (c["token"] >= 1).all() and (c["token"] <= 65).all() and (c["grid"][c["row"], c["col"]] == A.MASK).all()
    first = np.cumsum([0] + c["sizes"])
    assert ((c["row"] >= first[c["example"]]) & (c["row"] < first[c["example"] + 1])).all()


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
