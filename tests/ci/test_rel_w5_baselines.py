"""The evaluation conditions, the classical baselines and the copying chain of the relational line (numpy only).

The tables here are tiny and random: rows copied from the examined row with known per-position substitution rates.
"""
import itertools

import numpy as np
import pytest

from seeddrift.train.rel import api
from seeddrift.train.rel.baselines import classical as cl
from seeddrift.train.rel.baselines import fit as fitting
from seeddrift.train.rel.baselines import run as brun
from seeddrift.train.rel.baselines import switch
from seeddrift.train.rel.chain import switching as sw
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import layout, score, shards

BLOCKS = api.block_slices()


def mutate(seq, rates, base, rng):
    """Codons 1..base**3 with every position changed to another base with its rate; "other" stays."""
    nc = base ** 3
    seq = np.asarray(seq).copy()
    cod = seq <= nc
    d = cl.codon_digits(base)[np.where(cod, seq, 1) - 1].copy()
    for k in range(3):
        hit = rng.random(seq.shape[0]) < rates[k]
        d[:, k] = np.where(hit, (d[:, k] + rng.integers(1, base, seq.shape[0])) % base, d[:, k])
    out = d[:, 0] * base * base + d[:, 1] * base + d[:, 2] + 1
    return np.where(cod, out, seq)


def toy_tables(n_tables=8, rows=5, base=4, seed=0, rates=(0.03, 0.02, 0.25), spread=(0.2, 3.0), n_groups=3):
    """Tables in the shard format. Row r > 0 is a copy of a stretch of row 0 with rates * a row factor; some rows miss
    some blocks, the last row of table 1 does not exist, table 2 has one other row only, and some segments start with
    a residue without a column."""
    rng = np.random.default_rng(seed)
    nc = base ** 3
    N, R, B, L = n_tables, rows, api.N_BLOCKS, api.ROW_LEN
    tok = np.zeros((N, R, L), np.uint8)
    n = np.zeros((N, R, B), np.int16)
    col = np.full((N, R, L), -1, np.int16)
    kind = np.zeros((N, R, B), np.uint8)
    identity = np.full((N, R), np.nan, np.float32)
    for t in range(N):
        factor = np.exp(rng.uniform(np.log(spread[0]), np.log(spread[1]), R))
        for b, (role, s, e) in enumerate(BLOCKS):
            length = int(rng.integers(max(2, (e - s) // 2), e - s + 1))
            seq0 = rng.integers(1, nc + 1, length)
            seq0[rng.random(length) < 0.01] = nc + 1
            tok[t, 0, s:s + length], n[t, 0, b], kind[t, 0, b] = seq0, length, api.KIND_PRESENT
            col[t, 0, s:s + length] = np.arange(length)
            for r in range(1, R):
                if (t == 1 and r == R - 1) or (t == 2 and r >= 2):                 # rows that do not exist
                    continue
                if role != api.ROLE_TARGET and rng.random() < 0.2:
                    kind[t, r, b] = api.KIND_NOT_SEARCHED
                    continue
                a = int(rng.integers(0, length // 3 + 1))
                z = int(rng.integers(a + 1, length + 1))
                seg = mutate(seq0[a:z], np.minimum(np.array(rates) * factor[r], 0.7), base, rng)
                cols = np.arange(a, z)
                if rng.random() < 0.3 and z - a < e - s:
                    seg, cols = np.concatenate([[int(rng.integers(1, nc + 1))], seg]), np.concatenate([[-1], cols])
                tok[t, r, s:s + seg.shape[0]], n[t, r, b], kind[t, r, b] = seg, seg.shape[0], api.KIND_PRESENT
                col[t, r, s:s + seg.shape[0]] = cols
                if role == api.ROLE_TARGET:
                    identity[t, r] = float((seg[cols >= 0] == seq0[a:z]).mean())
    tables = dict(tok=tok, n=n, col=col, kind=kind, frame_row=np.zeros((N, B), np.int8))
    return tables, dict(group=(np.arange(N) % n_groups).astype(np.int8), identity=identity)


def write_set(run_dir, folder, tables, meta, split, per_shard=5):
    """Shard files of a table set as the tables stage writes them (<folder>/s0000.<field>.npy); returns the manifest's
    file entries."""
    import hashlib
    import os
    full = os.path.join(str(run_dir), folder)
    os.makedirs(full, exist_ok=True)
    n = tables["tok"].shape[0]
    arrays = dict(tables, group=meta["group"], identity=meta["identity"], split=np.full(n, split, np.int8))
    files = []
    for s, a in enumerate(range(0, n, per_shard)):
        for field, arr in arrays.items():
            path = os.path.join(full, "s%04d.%s.npy" % (s, field))
            np.save(path, arr[a:a + per_shard])
            with open(path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            files.append(dict(path=os.path.join(folder, os.path.basename(path)), sha256=sha, bytes=os.path.getsize(path),
                              role="shard" if field in api.SHARD_FIELDS else "meta"))
    return files


def manifest(files, name="toy", kind=api.TABLE_SYNTHETIC, tables=1, groups=("a", "b", "c")):
    return dict(format="rel-tableset", format_version=1, name=name, kind=kind, api_version=api.API, table_version="toy-0", tables=tables,
                coverage=dict(groups=list(groups), scenarios=[], not_covered=["everything: a test fixture"], note="test"), files=files, genomes=[])


def test_shards_are_read_in_the_order_of_the_set(tmp_path):
    train, tmeta = toy_tables(n_tables=13, seed=31)
    held, hmeta = toy_tables(n_tables=7, seed=32)
    files = write_set(tmp_path, "data/rel_w2/v0/train", train, tmeta, 0) + write_set(tmp_path, "data/rel_w2/v0/heldout", held, hmeta, 1)
    sets = shards.by_split(str(tmp_path), [manifest(files, tables=20)])
    assert sorted(sets) == [0, 1] and sets[0].tables == 13 and sets[1].tables == 7 and sets[0].shards == ["s0000", "s0001", "s0002"]
    got = sets[0].load(3, 12, shards.MODEL_FIELDS)
    assert all(np.array_equal(got[k], train[k][3:12]) for k in shards.MODEL_FIELDS) and got["tok"].dtype == np.uint8
    assert np.array_equal(sets[1].load(0, 7, ("identity",))["identity"], hmeta["identity"], equal_nan=True)
    assert sets[1].load(2, 2, ("n",))["n"].shape == (0, 5, api.N_BLOCKS) and len(sets[0].verified) > 0
    with open(sets[1].path("s0001", "tok"), "r+b") as f:                           # a shard that is not the one the manifest names
        f.seek(200)
        f.write(b"\x07")
    mixed = write_set(tmp_path, "mixed", held, hmeta, 1)
    np.save(str(tmp_path / "mixed" / "s0001.split.npy"), np.zeros(2, np.int8))    # one shard of training tables in a held-out set
    for bad in (lambda: sets[1].load(0, 7, ("tok",)), lambda: sets[0].load(0, 14, ("tok",)),
                lambda: shards.by_split(str(tmp_path), [manifest(files), manifest(files)]),
                lambda: shards.table_sets(str(tmp_path), dict(manifest(files), api_version=0)),
                lambda: shards.by_split(str(tmp_path), [manifest(mixed)])):
        try:
            bad()
            raise AssertionError("must be refused")
        except ValueError:
            pass


def test_hash_is_pinned():
    """The numbers every interpreter has to reproduce; a change here changes which cells every arm is scored on."""
    u = cond.hash_u01(20261122, 1, np.array([0, 1, 2999]).reshape(-1, 1), np.arange(3)[None, :])
    want = np.array([[0.3304139190629132, 0.6914856664502272, 0.13215706821206985],
                     [0.25112809168933914, 0.7607817665714194, 0.7339823616532581],
                     [0.8878849276188093, 0.4391936579058807, 0.3614612490738829]])
    assert u.shape == (3, 3) and np.array_equal(u, want)
    hidden = cond.hidden_target_cells(np.full((2, 3, api.N_BLOCKS), api.LEN_TARGET), [0, 2999], cond.SCATTERED)
    assert np.flatnonzero(hidden[0])[:12].tolist() == [0, 4, 14, 15, 19, 21, 29, 30, 34, 38, 45, 50] and 20 < hidden[1].sum() < 60
    kind = np.ones((8, 16, api.N_BLOCKS), np.uint8)
    src, has = cond.repair_rows(kind, np.arange(8))
    assert has.all() and (src[:, 1] - 1).tolist() == [13, 3, 12, 2, 1, 1, 7, 11]      # the shift of every table, k = 15
    assert cond.selfcheck() == cond.SELFCHECK == "b0da9955aa199eed4a5a75b8616ca642680a33b48d96af7208e83f222040b39e"
    pinned, cond.SELFCHECK = cond.SELFCHECK, "0" * 64                            # an interpreter that plans other cells is refused
    del cond._PORTABLE[:]
    try:
        cond.check_portable()
        raise AssertionError("a plan that is not the pinned one must be refused")
    except RuntimeError:
        pass
    finally:
        cond.SELFCHECK = pinned
    cond.check_portable()


def test_plans_hide_row0_target_cells_only():
    tables, _ = toy_tables(n_tables=40)
    ids = np.arange(40) + 7
    tb = slice(cond.TARGET_START, cond.TARGET_STOP)
    for c in cond.CONDITIONS:
        m = cond.materialise(tables, ids, c)
        again = cond.materialise(tables, ids, c)
        assert np.array_equal(m["target"], again["target"]) and np.array_equal(m["tok"], again["tok"])
        assert not (m["visible"] & m["target"]).any()
        assert not m["target"][:, 1:].any() and not m["target"][:, 0, :cond.TARGET_START].any()
        assert np.array_equal(m["visible"] | m["target"], cond.residue_cells(m["n"]))
        n0 = tables["n"][:, 0, cond.TARGET_BLOCK]
        hid = m["target"][:, 0, tb].sum(axis=1)
        if c.startswith(cond.WHOLE_GENE):
            assert np.array_equal(hid, np.where(m["has"], n0, 0)) and (m["has"].all() or c.endswith(cond.REPAIRED))
        else:
            assert 0.10 < hid.sum() / n0.sum() < 0.20
        if c.endswith(cond.NO_RELATIVES):
            assert m["tok"].shape[1] == 1 and np.array_equal(m["tok"][:, 0], tables["tok"][:, 0])
    a = cond.materialise(tables, ids, cond.SCATTERED)
    b = cond.materialise(tables, ids, cond.SCATTERED + "+" + cond.NO_RELATIVES)
    assert np.array_equal(a["target"][:, :1], b["target"])                       # the same cells with and without relatives
    other = cond.materialise(tables, ids + 1, cond.SCATTERED)
    assert not np.array_equal(a["target"], other["target"])


def test_plan_depends_on_lengths_only():
    tables, _ = toy_tables(n_tables=6)
    changed = dict(tables, tok=np.where(tables["tok"] > 0, 1 + tables["tok"] % 64, 0).astype(np.uint8))
    for c in cond.CONDITIONS:
        a, b = cond.materialise(tables, np.arange(6), c), cond.materialise(changed, np.arange(6), c)
        assert np.array_equal(a["target"], b["target"]) and np.array_equal(a["visible"], b["visible"]) and np.array_equal(a["rows"], b["rows"])


def test_repaired_shifts_the_other_rows_of_every_other_block():
    tables, _ = toy_tables(n_tables=12, rows=6)
    ids = np.arange(12)
    m = cond.materialise(tables, ids, cond.WHOLE_GENE + "+" + cond.REPAIRED)
    tb = slice(cond.TARGET_START, cond.TARGET_STOP)
    assert np.array_equal(m["tok"][:, :, tb], tables["tok"][:, :, tb]) and np.array_equal(m["n"][:, :, -1], tables["n"][:, :, -1])
    assert np.array_equal(m["tok"][:, 0], tables["tok"][:, 0])
    present = cond.present_rows(tables["kind"])
    assert present[:, 0].all() and not present[1, 5] and present[2].tolist() == [True, True] + [False] * 4
    for t in range(12):
        src = m["rows"][t]
        live = [r for r in range(1, 6) if present[t, r]]
        assert src[0] == 0 and all(src[r] == r for r in range(1, 6) if not present[t, r])
        if len(live) < 2:                                                        # one other row: nothing to re-pair, nothing scored
            assert t == 2 and not m["has"][t] and not m["target"][t].any() and np.array_equal(src, np.arange(6))
            continue
        shift = live.index(int(src[live[0]]))
        assert m["has"][t] and 1 <= shift < len(live) and [int(src[r]) for r in live] == live[shift:] + live[:shift]
        assert m["target"][t, 0, tb].sum() == tables["n"][t, 0, -1]
        for r in range(6):
            assert np.array_equal(m["tok"][t, r, :cond.TARGET_START], tables["tok"][t, src[r], :cond.TARGET_START])
            assert np.array_equal(m["col"][t, r, :cond.TARGET_START], tables["col"][t, src[r], :cond.TARGET_START])
            assert np.array_equal(m["n"][t, r, :-1], tables["n"][t, src[r], :-1]) and np.array_equal(m["kind"][t, r, :-1], tables["kind"][t, src[r], :-1])


def test_plan_is_the_plan_of_the_trainers_heldout_evaluation():
    """Cell for cell and row for row, on tiny tables; and the same strata labels."""
    try:
        from seeddrift.train.rel.data import heldout_plan as H
        from seeddrift.train.rel.data import tables as T
    except ImportError:
        pytest.skip("the trainer's held-out plan is not on this branch")
    tables, meta = toy_tables(n_tables=9, rows=6, seed=17)
    ids = np.arange(9) + 100
    assert tuple(H.CONDITIONS) == cond.CONDITIONS and tuple(H.STRATA) == score.COARSE and H.RATE == cond.MASK_RATE
    present = cond.present_rows(tables["kind"])
    for c in cond.CONDITIONS:
        m = cond.materialise(tables, ids, c)
        for t in range(9):
            tab = dict(tok=tables["tok"][t], n=tables["n"][t], col=tables["col"][t], kind=tables["kind"][t], frame_len=tables["n"][t, 0])
            plan = H.condition(T.take_rows(tab, T.present_rows(tab)), c, int(ids[t]))
            if plan is None:
                assert not m["has"][t] and not m["target"][t].any()
                continue
            theirs, visible, hide = plan
            rows = [0] if c.endswith(cond.NO_RELATIVES) else np.flatnonzero(present[t]).tolist()
            assert np.array_equal(T.present_rows(tab), np.flatnonzero(present[t])) and m["has"][t]
            for f in ("tok", "n", "col", "kind"):
                assert np.array_equal(m[f][t][rows], theirs[f]), (c, t, f)
            assert np.array_equal(m["target"][t][rows], hide) and np.array_equal(m["visible"][t][rows], visible), (c, t)
            assert not m["target"][t][[r for r in range(m["target"].shape[1]) if r not in rows]].any()
    mine = score.coarse_stratum(meta["identity"])
    assert [score.COARSE[k] for k in mine] == [H.stratum(meta["identity"][t]) for t in range(9)]
    edge = np.array([[1.0, x, np.nan] for x in (0.0, 0.29999, 0.3, 0.5, 0.7, 0.9, 1.0)] + [[1.0, np.nan, np.nan]])
    assert [score.COARSE[k] for k in score.coarse_stratum(edge)] == [H.stratum(e) for e in edge]


def test_items_digest_tells_content_and_plan_apart():
    tables, _ = toy_tables(n_tables=10)
    ids = np.arange(10)

    def digest(tb, ident, c, order=None):
        d = cond.ItemsDigest(c)
        for a, b in (order or [(0, 10)]):
            d.add(cond.materialise(fitting.take(tb, a, b), ident[a:b], c))
        return d.hexdigest()

    whole = digest(tables, ids, cond.WHOLE_GENE)
    assert whole == digest(tables, ids, cond.WHOLE_GENE, [(6, 10), (0, 6)])       # the order of the batches is free
    assert whole != digest(tables, ids, cond.WHOLE_GENE + "+" + cond.REPAIRED)
    assert digest(tables, ids, cond.SCATTERED) != digest(tables, ids + 1, cond.SCATTERED)
    changed = dict(tables, tok=tables["tok"].copy())
    changed["tok"][3, 0, cond.TARGET_START] = 1 + changed["tok"][3, 0, cond.TARGET_START] % 64
    assert whole != digest(changed, ids, cond.WHOLE_GENE)


def test_stored_layout_places_rows_in_the_frame():
    tables, _ = toy_tables(n_tables=5)
    grid = layout.place(tables["tok"], tables["n"], tables["col"], api.GAP)
    assert np.array_equal(grid[:, 0][tables["tok"][:, 0] > 0], tables["tok"][:, 0][tables["tok"][:, 0] > 0])
    t, r = 2, 3
    s, e = cond.TARGET_START, cond.TARGET_STOP
    ln = int(tables["n"][t, r, -1])
    cols = tables["col"][t, r, s:s + ln]
    want = np.full(e - s, api.GAP)
    want[cols[cols >= 0]] = tables["tok"][t, r, s:s + ln][cols >= 0]
    assert np.array_equal(grid[t, r, s:e], want)
    bad = dict(tables, col=tables["col"].copy())
    bad["col"][0, 1, 0] = 40                                                     # a column outside a 32-column block
    try:
        layout.place(bad["tok"], bad["n"], bad["col"], api.GAP)
        raise AssertionError("a column outside its block must be refused")
    except ValueError:
        pass


def _params(base):
    nc, V = cl.sizes(base)
    prior = np.arange(1, V + 1, dtype=float)
    return dict(prior=(prior / prior.sum()).tolist(), alpha=3.0, beta=0.7, eps=[0.05, 0.02, 0.3], rho=0.01,
                eps_binned=[[0.02 * (k + 1), 0.01, 0.03 * (k + 1)] for k in range(cl.N_SHARE_BINS + 1)],
                chains={n: dict(eps=[0.04, 0.03, 0.2], tau=0.1, nosrc=0.2) for n in switch.NAMES})


def test_view_holds_nothing_of_the_hidden_residues():
    """Change every hidden residue: the view and every distribution stay the same, bit for bit."""
    for base in (2, 4):
        nc = base ** 3
        tables, _ = toy_tables(n_tables=6, base=base, seed=3)
        ids = np.arange(6)
        for c in cond.CONDITIONS:
            m = cond.materialise(tables, ids, c)
            other = dict(m, tok=np.where(m["target"], 1 + m["tok"] % nc, m["tok"]).astype(np.uint8))
            assert (other["tok"] != m["tok"])[m["target"]].all()
            va, ca, ta = layout.masked_view(m)
            vb, cb, tb = layout.masked_view(other)
            assert np.array_equal(va, vb) and np.array_equal(ca[0], cb[0]) and np.array_equal(ca[1], cb[1]) and not np.array_equal(ta, tb)
            assert (va[m["target"]] == api.MASK).all()
            ba, bb = cl.Batch(va, ca, base), cl.Batch(vb, cb, base)
            for name in cl.NAMES:
                assert np.array_equal(cl.predict(name, ba, _params(base)), cl.predict(name, bb, _params(base)))
            for name in (switch.NAME, switch.PAST, switch.AHEAD):
                assert np.array_equal(switch.predict(ba, _params(base), name), switch.predict(bb, _params(base), name))


def test_every_baseline_sums_to_one_over_every_residue():
    """Small alphabet (8 codons and "other"), exhaustively: for every scored cell, the probabilities the scorer gives
    to each of the nine possible residues add up to one, under every condition."""
    base = 2
    nc, V = cl.sizes(base)
    tables, _ = toy_tables(n_tables=6, base=base, seed=5)
    for c in cond.CONDITIONS:
        m = cond.materialise(tables, np.arange(6), c)
        view, cells, truth = layout.masked_view(m)
        bt = cl.Batch(view, cells, base)
        assert truth.shape[0] > 0
        for name in cl.NAMES + (switch.NAME, switch.PAST, switch.AHEAD):
            total = np.zeros(truth.shape[0])
            for x in range(1, V + 1):
                P = switch.predict(bt, _params(base), name) if name in switch.NAMES else cl.predict(name, bt, _params(base))
                cl.check_normalised(P)
                assert P.shape == (truth.shape[0], V)
                total += 2.0 ** -cl.bits_of(P, np.full(truth.shape[0], x))
            assert np.abs(total - 1.0).max() < 1e-12, (c, name)
    k = cl.copy_kernel([0.1, 0.2, 0.3], 0.05, 4)
    assert k.shape == (64, 65) and np.abs(k.sum(axis=1) - 1).max() < 1e-12
    assert abs(k[0, 0] - 0.95 * 0.9 * 0.8 * 0.7) < 1e-12 and abs(k[0, 1] - 0.95 * 0.9 * 0.8 * 0.1) < 1e-12
    try:
        cl.check_normalised(np.array([[0.5, 0.4]]))
        raise AssertionError("a row that does not sum to one must be refused")
    except ValueError:
        pass


def test_copy_nearest_copies_from_the_nearest_row():
    base = 4
    tables, _ = toy_tables(n_tables=4, rows=4, seed=11)
    s, e = cond.TARGET_START, cond.TARGET_STOP
    for t in range(4):                                                           # row 3 = row 0 at the target, rows 1, 2 far
        ln = int(tables["n"][t, 0, -1])
        tables["tok"][t, 3, s:e], tables["col"][t, 3, s:e] = tables["tok"][t, 0, s:e], tables["col"][t, 0, s:e]
        tables["n"][t, 3, -1] = ln
    p = _params(base)
    m = cond.materialise(tables, np.arange(4), cond.SCATTERED)
    view, cells, truth = layout.masked_view(m)
    has, y, share, row = cl.nearest_source(view, cells, base)
    cod = truth <= 64
    assert has[cod].all() and (row[cod] == 3).all() and (share[cod] == 1.0).all() and np.array_equal(y[cod], truth[cod])
    assert (row[~cod] != 3).all()                                                # "other" is not copied: the next row, or none
    bits = cl.bits_of(cl.predict("copy_nearest", cl.Batch(view, cells, base), p), truth)
    assert abs(bits[cod].mean() + np.log2(0.99 * 0.95 * 0.98 * 0.7)) < 1e-9
    # the whole gene hidden: the nearest row is judged on the other blocks, where row 3 is an ordinary row
    m = cond.materialise(tables, np.arange(4), cond.WHOLE_GENE)
    view, cells, truth = layout.masked_view(m)
    same, compared = cl.row_shares(view, base)
    assert not ((view[:, 0, s:e] >= 1) & (view[:, 0, s:e] <= 65)).any() and (compared.sum(axis=1) > 0).all()
    has, y, share, row = cl.nearest_source(view, cells, base)
    best = (same / np.maximum(compared, 1)).argmax(axis=1) + 1
    first = np.unique(cells[0], return_index=True)[1]
    assert has.any() and (share[~has] == -1).all()
    for t, k in enumerate(first):                                                # where the best row has a codon, it is the source
        col_tok = view[t, best[t], cells[1][cells[0] == t]]
        used = row[cells[0] == t]
        assert (used[(col_tok >= 1) & (col_tok <= 64)] == best[t]).all()
    # no relatives: every relative-based line falls back to own_freq
    m = cond.materialise(tables, np.arange(4), cond.WHOLE_GENE + "+" + cond.NO_RELATIVES)
    view, cells, truth = layout.masked_view(m)
    bt = cl.Batch(view, cells, base)
    for name in cl.USES_RELATIVES:
        assert np.allclose(cl.predict(name, bt, p), cl.predict("own_freq", bt, p))
    assert np.allclose(switch.predict(bt, p), cl.predict("own_freq", bt, p))


def test_fit_recovers_the_rates_and_beats_the_floor():
    rates = (0.04, 0.02, 0.3)
    tables, meta = toy_tables(n_tables=60, rows=5, seed=21, rates=rates, spread=(1.0, 1.0))
    ids = np.arange(60)
    params = brun.fit_all(tables, ids, seed=4, chain_tables=24)
    for mask in brun.MASKS:
        p = params[mask]
        assert p["fit"]["cells"] > 0 and abs(sum(p["prior"]) - 1) < 1e-9 and len(p["eps_binned"]) == cl.N_SHARE_BINS + 1
        assert max(abs(a - b) for a, b in zip(p["eps"], rates)) < 0.02, p["eps"]
        assert sorted(p["chains"]) == sorted(switch.variants(mask))
        for ch in p["chains"].values():
            assert ch["fit"]["trace"][-1]["bits"] <= ch["fit"]["trace"][0]["bits"] and ch["fit"]["cells"] > 0
    held, hmeta = toy_tables(n_tables=30, rows=5, seed=22, rates=rates, spread=(1.0, 1.0))
    res = brun.evaluate(held, np.arange(30), params, seed=9)
    assert sorted(res) == sorted(cond.CONDITIONS)
    sc, wg = res[cond.SCATTERED], res[cond.WHOLE_GENE]
    mean = lambda r, n: float(r["bits"][n].mean())                                # noqa: E731
    assert abs(mean(sc, "uniform") - np.log2(65)) < 1e-12
    assert max(mean(sc, "copy_nearest"), mean(sc, "row_average")) < mean(sc, "own_freq") - 1.0 and mean(sc, "own_freq") <= mean(sc, "uniform") + 0.05
    assert mean(sc, switch.NAME) < mean(sc, "own_freq") - 1.0 and mean(wg, "copy_nearest") < mean(wg, "own_freq") - 1.0
    assert mean(sc, switch.NAME) < mean(sc, switch.AHEAD) + 0.02 < mean(sc, switch.PAST) + 0.04      # seeing less never helps
    assert mean(wg, switch.JOINT) < mean(wg, switch.NAME) - 0.2                    # the stretch as a whole finds its source
    assert switch.JOINT not in sc["bits"] and switch.PAST not in wg["bits"]
    assert "copy_nearest" not in res[cond.WHOLE_GENE + "+" + cond.NO_RELATIVES]["bits"]
    assert np.array_equal(res[cond.SCATTERED + "+" + cond.NO_RELATIVES]["bits"]["own_freq"], sc["bits"]["own_freq"])
    rp = res[cond.WHOLE_GENE + "+" + cond.REPAIRED]                               # table 2 has one other row: not re-paired, not scored
    n_other = brun.other_rows(held)
    assert n_other[2] == 1 and 2 not in rp["cell_table"] and set(rp["cell_table"]) == set(np.flatnonzero(n_other >= 2))
    assert np.array_equal(rp["bits"]["row_average"], wg["bits"]["row_average"][wg["cell_table"] != 2])
    assert not np.array_equal(rp["bits"]["copy_nearest"], wg["bits"]["copy_nearest"][wg["cell_table"] != 2])
    head = dict(run="r", commit="c", seed=9, table_set="toy", table_kind=api.TABLE_SYNTHETIC, items_sha256="5" * 64, versions="v")
    rows = brun.result_rows(res, hmeta["group"], hmeta["identity"], params, head, n_groups=3, group_names={0: "zero", 1: "one"}, n_other=n_other)
    same = {(r["condition"], r["variant"]): r for r in rows if r["group"] == "all" and r["stratum"] in ("all", score.OTHER_ROWS[2])
            and (r["condition"], r["stratum"]) in ((cond.WHOLE_GENE, score.OTHER_ROWS[2]), (cond.WHOLE_GENE + "+" + cond.REPAIRED, "all"))}
    a, b = same[(cond.WHOLE_GENE, "row_average")], same[(cond.WHOLE_GENE + "+" + cond.REPAIRED, "row_average")]
    assert a["target_cells"] == b["target_cells"] and abs(a["bits"] - b["bits"]) < 1e-12 and a["n_items"] == b["n_items"] == 29
    assert {r["group"] for r in rows} == {"all", "zero", "one", "2"} and {r["tier"] for r in rows} == {"all"} == {r["item_half"] for r in rows}
    assert {r["stratum"] for r in rows} <= set(score.COARSE + score.FINE + score.OTHER_ROWS + ("all",)) and any(r["stratum"] in score.COARSE for r in rows)
    assert all(tuple(r) == api.RESULT_ROW for r in rows)
    top = [r for r in rows if r["group"] == "all" and r["stratum"] == "all" and r["condition"] == cond.SCATTERED]
    assert sorted(r["variant"] for r in top) == sorted(brun.names_under(cond.SCATTERED))
    own = [r for r in top if r["variant"] == "own_freq"][0]
    cn = [r for r in top if r["variant"] == "copy_nearest"][0]
    assert own["bits_minus_baseline"] == 0 and abs(cn["bits_minus_baseline"] - (cn["bits"] - own["bits"])) < 1e-12 and cn["bits_minus_baseline"] < -1
    assert cn["target_cells"] == sc["cell_table"].shape[0] and cn["items_sha256"] == "5" * 64 and cn["n_items"] == 30
    text = brun.markdown(rows)
    assert "| copy_nearest |" in text and "by the identity of the nearest row" in text and "on the same tables" in text


def test_aggregate_is_the_mean_of_the_group_means():
    bits = np.array([1.0, 3.0, 10.0, 20.0, 30.0, 7.0])
    cell_table = np.array([0, 0, 1, 1, 1, 2])
    group = np.array([0, 1, 1, 2])                                               # table 3 has no cell
    identity = np.array([[1.0, 0.25, np.nan], [1.0, np.nan, np.nan], [1.0, 0.31, 0.95], [1.0, 0.5, 0.5]])
    strata = score.strata_of(identity)
    assert [score.FINE[k] for k in strata[1][1]] == ["nearest_identity=0.2-0.3", "nearest_identity=none", "nearest_identity=0.9-1.0", "nearest_identity=0.5-0.6"]
    assert [score.COARSE[k] for k in strata[0][1]] == ["<0.3", "none", ">=0.9", "0.5-0.7"]
    assert score.fine_stratum(np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 0.3]])).tolist() == [9, 0, 3]
    assert [score.COARSE[k] for k in score.coarse_stratum(np.array([[1.0, 0.3], [1.0, 0.29], [1.0, 0.9], [1.0, 0.7]]))] == ["0.3-0.5", "<0.3", ">=0.9", "0.7-0.9"]
    rows = {(r["group"], r["stratum"]): r for r in score.aggregate(bits, cell_table, group, strata, n_groups=3, group_names={1: "one"})}
    top = rows[("all", "all")]
    assert abs(top["bits"] - (2.0 + (10 + 20 + 30 + 7) / 4.0) / 2) < 1e-12 and top["target_cells"] == 6 and top["n_items"] == 3 and top["n_groups"] == 2
    assert rows[("one", "all")]["bits"] == 67 / 4.0 and rows[("one", "nearest_identity=none")]["bits"] == 20.0 == rows[("one", "none")]["bits"]
    assert rows[("all", "nearest_identity=0.9-1.0")]["bits"] == 7.0 == rows[("all", ">=0.9")]["bits"] and ("2", "all") not in rows and ("0", "<0.3") in rows
    row = score.result_row(run="x", bits=1.5)
    assert tuple(row) == api.RESULT_ROW and row["commit"] == ""
    try:
        score.result_row(bit=1.0)
        raise AssertionError("a column outside the contract must be refused")
    except KeyError:
        pass


def test_result_rows_round_trip(tmp_path):
    rows = [score.result_row(run="r1", variant="own_freq", bits=4.25, target_cells=12, condition=cond.SCATTERED, group="all", stratum="all")]
    path = str(tmp_path / "rows.tsv")
    score.write_rows(path, rows)
    back = score.read_rows(path)
    assert len(back) == 1 and tuple(back[0]) == api.RESULT_ROW and float(back[0]["bits"]) == 4.25 and back[0]["variant"] == "own_freq"


def test_joint_reading_sums_to_one_over_every_stretch():
    """The whole gene hidden and read as one stretch: over every possible stretch of three residues (nine tokens) the
    probabilities add up to one, through the same path the scorer takes."""
    base, length = 2, 3
    nc, V = cl.sizes(base)
    tables, _ = toy_tables(n_tables=3, rows=4, base=base, seed=8)
    s = cond.TARGET_START
    for name in ("tok", "col"):                                                  # cut the target block to three columns
        a = tables[name]
        fill = 0 if name == "tok" else -1
        for t in range(3):
            for r in range(4):
                n = int(tables["n"][t, r, -1])
                keep = np.flatnonzero((tables["col"][t, r, s:s + n] >= 0) & (tables["col"][t, r, s:s + n] < length)) if name == "tok" else None
                if name == "tok":
                    tables.setdefault("_keep", {})[(t, r)] = keep
                keep = tables["_keep"][(t, r)]
                vals = a[t, r, s:s + n][keep].copy()
                a[t, r, s:] = fill
                a[t, r, s:s + len(vals)] = vals
    for (t, r), keep in tables.pop("_keep").items():
        tables["n"][t, r, -1] = len(keep)
    assert (tables["n"][:, 0, -1] == length).all()
    p = _params(base)
    total = np.zeros(3)
    for seq in itertools.product(range(1, V + 1), repeat=length):
        tables["tok"][:, 0, s:s + length] = seq
        view, cells, truth = layout.masked_view(cond.materialise(tables, np.arange(3), cond.WHOLE_GENE))
        bits = switch.joint_bits(cl.Batch(view, cells, base), truth, p)
        total += 2.0 ** -np.bincount(cells[0], weights=bits, minlength=3)
    assert np.abs(total - 1.0).max() < 1e-12
    try:                                                                         # scattered cells are not a stretch
        view, cells, truth = layout.masked_view(cond.materialise(tables, np.arange(3), cond.SCATTERED, rate=0.5))
        switch.joint_bits(cl.Batch(view, cells, base), truth, p)
        raise AssertionError("the joint reading must refuse a stretch with visible cells")
    except ValueError:
        pass


def _chain_case(base=2, C=4, K=2, seed=0):
    rng = np.random.default_rng(seed)
    nc, V = cl.sizes(base)
    Y = rng.integers(1, nc + 1, (1, K, C)).astype(np.uint8)
    Y[0, 0, 1] = api.GAP                                                         # a row without a codon in one column
    q = rng.random((1, V))
    q /= q.sum()
    return Y, q, cl.copy_kernel([0.1, 0.2, 0.3], 0.05, base), 0.15, 0.25


def _brute(X, Y, q, kernel, tau, nosrc, base):
    """The probability of the residues X (C,) by summing over every path of sources."""
    nc = cl.sizes(base)[0]
    K, C = Y.shape[1], Y.shape[2]
    prior = sw.source_prior(Y, nosrc, base)[0]
    total = 0.0
    for path in itertools.product(range(K + 1), repeat=C):
        p = 1.0
        for c, s in enumerate(path):
            p *= prior[s] if c == 0 else ((1 - tau) * (s == path[c - 1]) + tau * prior[s])
            if 1 <= X[c] <= nc + 1:
                y = Y[0, s - 1, c] if s > 0 else api.GAP
                p *= kernel[y - 1, X[c] - 1] if 1 <= y <= nc else q[0, X[c] - 1]
        total += p
    return total


def test_chain_equals_enumeration_and_is_normalised():
    base = 2
    nc, V = cl.sizes(base)
    Y, q, kernel, tau, nosrc = _chain_case(base)
    C = Y.shape[2]
    rng = np.random.default_rng(1)
    for _ in range(5):                                                           # forward = the sum over every path
        X = rng.integers(1, V + 1, (1, C)).astype(np.uint8)
        X[0, rng.integers(0, C)] = api.MASK
        e = sw.emissions(X, Y, q, kernel, base)
        logz = sw.forward(e, sw.source_prior(Y, nosrc, base), tau, np.ones((1, C), bool))[2]
        assert abs(np.exp(logz[0]) - _brute(X[0], Y, q, kernel, tau, nosrc, base)) < 1e-12
    seqs = np.array(list(itertools.product(range(1, V + 1), repeat=3)), dtype=np.uint8)   # every stretch of three residues
    lj = sw.log_joint(seqs, np.repeat(Y[:, :, :3], seqs.shape[0], axis=0), np.repeat(q, seqs.shape[0], axis=0), kernel, tau, nosrc, base)
    assert seqs.shape[0] == V ** 3 and abs(np.exp(lj).sum() - 1.0) < 1e-12
    terms = sw.log_joint(seqs, np.repeat(Y[:, :, :3], seqs.shape[0], axis=0), np.repeat(q, seqs.shape[0], axis=0), kernel, tau, nosrc, base, per_column=True)
    assert terms.shape == (V ** 3, 3) and np.allclose(terms.sum(axis=1), lj, atol=1e-13)   # the chain-rule terms of the stretch
    # a hidden cell: its distribution = the joint with every residue in its place, renormalised; other hidden cells summed out
    X = rng.integers(1, V + 1, (1, C)).astype(np.uint8)
    for hidden in ([1], [0, 2], [C - 1]):
        Xh = X.copy()
        Xh[0, hidden] = api.MASK
        e = sw.emissions(Xh, Y, q, kernel, base)
        prior = sw.source_prior(Y, nosrc, base)
        g = sw.source_posterior(e, prior, tau, np.ones((1, C), bool), "both")
        for c in hidden:
            P = sw.predict_cells(g[:, c], Y[:, :, c], q, kernel, base)[0]
            want = np.empty(V)
            for x in range(1, V + 1):
                Xx = Xh[0].copy()
                Xx[c] = x
                want[x - 1] = _brute(Xx, Y, q, kernel, tau, nosrc, base)
            assert abs(P.sum() - 1) < 1e-12 and np.abs(P - want / want.sum()).max() < 1e-12
            assert abs(sw.truth_probability(g[:, c], Y[:, :, c], q[:, 2], kernel, np.array([3]), base)[0] - P[2]) < 1e-15


def test_chain_modes_and_ragged_rows():
    base = 2
    nc, V = cl.sizes(base)
    Y, q, kernel, tau, nosrc = _chain_case(base, C=6, K=3, seed=4)
    X = np.random.default_rng(2).integers(1, V + 1, (1, 6)).astype(np.uint8)
    X[0, 2] = api.MASK
    e = sw.emissions(X, Y, q, kernel, base)
    prior = sw.source_prior(Y, nosrc, base)
    valid = np.ones((1, 6), bool)
    both = sw.source_posterior(e, prior, tau, valid, "both")
    past = sw.source_posterior(e, prior, tau, valid, "past")
    assert np.allclose(sw.source_posterior(e, prior, tau, valid, ("ahead", 6)), both, atol=1e-14)
    assert np.allclose(sw.source_posterior(e, prior, tau, valid, ("ahead", 0)), past, atol=1e-14)
    assert np.allclose(both[:, -1], past[:, -1], atol=1e-14) and not np.allclose(both[:, 2], past[:, 2])
    ahead1 = sw.source_posterior(e, prior, tau, valid, ("ahead", 1))
    e1 = e.copy()
    e1[:, 4:] = 1.0                                                              # one column ahead of column 2 = columns 4.. unseen
    assert np.allclose(ahead1[:, 2], sw.source_posterior(e1, prior, tau, valid, "both")[:, 2], atol=1e-14)
    # the posterior of a column with its own cell = d log p / d log emission (finite differences)
    _, after, logz = sw.forward(e, prior, tau, valid)
    full = after * sw.backward(e, prior, tau, valid)
    full /= full.sum(axis=2, keepdims=True)
    for c, s in ((0, 1), (3, 0), (5, 2)):
        e2 = e.copy()
        e2[0, c, s] *= 1 + 1e-6
        d = (sw.forward(e2, prior, tau, valid)[2][0] - logz[0]) / 1e-6
        assert abs(d - full[0, c, s]) < 1e-5
    # a shorter row in the same batch: the columns past its end change nothing
    X2 = np.concatenate([X, X], axis=0)
    Y2 = np.concatenate([Y, Y], axis=0)
    X2[1, 4:] = api.GAP
    Xs, Ys, vs = sw.locus(np.concatenate([X2[:, None, :], Y2], axis=1), 0, 6)
    assert np.array_equal(vs[1], [True] * 4 + [False] * 2)
    e2 = sw.emissions(Xs, Ys, np.repeat(q, 2, axis=0), kernel, base)
    g2 = sw.source_posterior(e2, sw.source_prior(Ys, nosrc, base), tau, vs, "both")
    short = sw.source_posterior(sw.emissions(X[:, :4], Y[:, :, :4], q, kernel, base), sw.source_prior(Y, nosrc, base), tau, np.ones((1, 4), bool), "both")
    assert np.allclose(g2[0], both[0], atol=1e-14) and np.allclose(g2[1, :4], short[0], atol=1e-14)
    # no row at all: the chain is the background
    none = sw.source_prior(np.full((2, 3, 6), api.GAP, np.uint8), nosrc, base)
    assert np.array_equal(none, np.array([[1.0, 0, 0, 0]] * 2))
