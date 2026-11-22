"""Implant tables of package W18 (bench/rel/implant.py) on the small synthetic run of W2's tests, whose homolog rows
carry their true correspondences. numpy only (python tests/ci/test_rel_w18_implant.py runs them without pytest)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_rel_w2_tables as W2T  # noqa: E402
from bench.rel import implant as I  # noqa: E402
from seeddrift.train import selfsup as SS  # noqa: E402
from seeddrift.train.rel import api as A  # noqa: E402
from seeddrift.train.rel.readouts import w18_passes as P  # noqa: E402
from seeddrift.train.rel.tables import v0_dist as DI  # noqa: E402

EDGES = [0.05, 0.2, 0.5, 1.0]
_CACHE = {}


def distances(genomes, ids):
    sk = [DI.sketch(genomes[g].codons, genomes[g].coff, 4, 256) for g in ids]
    jac, _ = DI.matrix([s[0] for s in sk], [s[1] for s in sk], 256, 1)
    d = DI.mash(jac, 4)
    np.fill_diagonal(d, 0.0)
    return d


def implanter(attempts=40, **kw):
    d, metas, truth, genomes, ids, train, _ = W2T.run()
    opened = [m for m in metas if m["split"] in ("train", "heldout")]
    dist = distances(genomes, ids)
    eligible = {m["id"] for m in metas if m["split"] == "heldout"}
    imp = I.Implanter(SS.Pairs(d), genomes, train, ids, dist, ids, EDGES, eligible, I.split_units(opened), rows=8, min_codons=20, length_tol=0.5,
                      attempts=attempts, **kw)
    return imp, opened, dist, ids, truth, genomes, train, eligible


def built(seed=5, per_tier=2, cap=30):
    key = (seed, per_tier, cap)
    if key not in _CACHE:
        imp, opened, dist, ids, truth, genomes, train, eligible = implanter()
        out, meta, triples, counts, capacity = imp.build(opened, W2T.GROUPS, seed, per_tier=per_tier, group_tier_cap=cap)
        _CACHE[key] = dict(out=out, meta=meta, triples=triples, counts=counts, capacity=capacity, imp=imp, dist=dist, ids=ids, truth=truth,
                           genomes=genomes, train=train, eligible=eligible, opened=opened)
    return _CACHE[key]


def test_triples_are_sound():
    b = built()
    out, meta, ids = b["out"], b["meta"], b["ids"]
    gix = {g: i for i, g in enumerate(ids)}
    units = I.split_units(b["opened"])
    assert I.verify(out, meta, {gix[g] for g in b["eligible"]}, {gix[g] for g in b["train"]}, {gix[g]: u for g, u in units.items()}) == []
    assert b["counts"]["triples"] >= 6 and len(out["tok"]) == 3 * (b["counts"]["triples"] + b["counts"]["copies"])
    assert sorted(set(meta["variant"].tolist())) == [0, 1, 2] and b["counts"]["phase1"] > 0


def test_target_blocks_hold_the_first_n_codons_of_the_right_genes():
    b = built()
    out, meta, ids, genomes = b["out"], b["meta"], b["ids"], b["genomes"]
    T = I.TARGET
    s0 = I.SLICES[T][1]
    for t in range(len(out["tok"])):
        a, d = ids[out["genome"][t, 0]], ids[meta["donor"][t]]
        v = int(meta["variant"][t])
        owner, gene = [(d, meta["gene_j"][t]), (a, meta["gene_i"][t]), (a, meta["gene_k"][t])][v]
        g = genomes[owner]
        n = int(out["n"][t, 0, T])
        assert n == meta["target_len"][t] and (out["tok"][t, 0, s0:s0 + n] == g.codons[g.coff[gene]:g.coff[gene] + n]).all()
        assert (out["start"][t, :, T] == -1).all() and (out["strand"][t, :, T] == 0).all() and (out["replicon"][t, :, T] == -1).all()
        assert out["gene"][t, T] == (-1 if v == 0 else gene)
        if v == 2:
            assert int(gene) not in out["gene"][t, :T].tolist()       # k is no relation block and no neighbour
        if meta["phase"][t] == 1:
            assert n == 256
        lens = [I.stored_len(genomes[x], int(y)) for x, y in ((d, meta["gene_j"][t]), (a, meta["gene_i"][t]), (a, meta["gene_k"][t]))]
        assert n == min(lens) and (meta["phase"][t] == 1 or max(lens) <= 1.5 * lens[1] + 1e-9)
    for s in np.unique(meta["triple"]):
        ts = np.flatnonzero(meta["triple"] == s)
        assert len({int(out["n"][x, 0, T]) for x in ts}) == 1


def test_rows_cover_every_target_and_are_shared():
    b = built()
    out, meta, imp = b["out"], b["meta"], b["imp"]
    T = I.TARGET
    for s in np.unique(meta["triple"]):
        ts = np.flatnonzero(meta["triple"] == s)
        g = out["genome"][ts]
        assert (g == g[0]).all() and meta["donor"][ts[0]] not in g[0].tolist()
        rows = int((g[0, 1:] >= 0).sum())
        need = imp.need(int(meta["target_len"][ts[0]]))
        for x in ts:
            assert int((out["kind"][x, 1:, T] == A.KIND_PRESENT).sum()) == rows == int(meta["on_target"][x])
            placed = (out["col"][x, 1:, I.SLICES[T][1]:] >= 0).sum(axis=1)
            assert (placed[:rows] >= need).all()                        # every row covers enough of every target
            outside = np.ones(A.ROW_LEN, dtype=bool)
            outside[I.SLICES[T][1]:] = False
            assert (out["tok"][x][:, outside] == out["tok"][ts[0]][:, outside]).all()
            assert (out["col"][x][:, outside] == out["col"][ts[0]][:, outside]).all()


def test_implant_rows_are_placed_by_the_true_alignment_on_the_donor_gene():
    b = built()
    out, meta, ids, truth, genomes = b["out"], b["meta"], b["ids"], b["truth"], b["genomes"]
    s0, s1 = I.SLICES[I.TARGET][1], I.SLICES[I.TARGET][2]
    checked = 0
    for t in np.flatnonzero(meta["variant"] == 0):
        d, j = ids[meta["donor"][t]], int(meta["gene_j"][t])
        for q in range(1, out["tok"].shape[1]):
            if out["genome"][t, q] < 0:
                continue
            pid = ids[out["genome"][t, q]]
            pj, pairs_ = truth[(d, j, pid)]
            assert meta["row_gene"][t, q] == pj
            pc = genomes[pid].codons[genomes[pid].coff[pj]:genomes[pid].coff[pj + 1]]
            want = {x: int(pc[y]) for y, x in pairs_.items()}
            col, tok = out["col"][t, q, s0:s1], out["tok"][t, q, s0:s1]
            for c, k in zip(col[col >= 0], tok[col >= 0]):
                if c in want:
                    assert want[c] == int(k)
                    checked += 1
    assert checked > 50


def test_allocation_caps_and_tiers():
    b = built(per_tier=2, cap=3)
    tr = [t for t in b["triples"] if t["scenario"] == "in_place"]
    by = {}
    for t in tr:
        by.setdefault((t["group"], t["tier"]), []).append(t)
        assert t["tier"] == DI.tier_of(np.array([b["dist"][b["ids"].index(t["a"]), b["ids"].index(t["d"])]]), EDGES)[0]
    assert all(len(v) <= 3 for v in by.values())
    per_a = {}
    for t in tr:
        per_a[(t["group"], t["tier"], t["a"])] = per_a.get((t["group"], t["tier"], t["a"]), 0) + 1
    assert all(v <= 2 for v in per_a.values())
    donors = {}
    for t in tr:
        donors[(t["group"], t["d"])] = donors.get((t["group"], t["d"]), 0) + 1
    assert all(v <= 3 for v in donors.values())
    meta, dist = b["meta"], b["dist"]
    for t in range(len(meta["triple"])):
        a = int(b["out"]["genome"][t, 0])
        for q in range(b["out"]["tok"].shape[1]):
            r = int(b["out"]["genome"][t, q])
            if q == 0 or r < 0:
                assert meta["row_tier"][t, q] == -1
            else:
                assert meta["row_tier"][t, q] == DI.tier_of(np.array([dist[a, r]]), EDGES)[0]


def test_donor_unsampled_copies_drop_the_rows_closer_to_the_donor():
    b = built()
    out, meta, dist = b["out"], b["meta"], b["dist"]
    copies = np.unique(meta["triple"][meta["scenario"] == 1])
    assert len(copies) > 0
    for s in copies:
        x = np.flatnonzero((meta["triple"] == s) & (meta["variant"] == 0))[0]
        p = np.flatnonzero((meta["triple"] == meta["parent"][x]) & (meta["variant"] == 0))[0]
        assert meta["scenario"][p] == 0 and meta["parent"][p] == meta["parent"][x]
        a, d = int(out["genome"][p, 0]), int(meta["donor"][p])
        full = [int(r) for r in out["genome"][p, 1:] if r >= 0]
        kept = [int(r) for r in out["genome"][x, 1:] if r >= 0]
        assert kept == [r for r in full if not dist[r, d] < dist[r, a]] and len(kept) < len(full)
        assert (out["tok"][x, 0] == out["tok"][p, 0]).all()            # the same genes


def test_the_three_tables_of_a_triple_hide_the_same_cells():
    b = built()
    out, meta = b["out"], b["meta"]
    got = {}
    for t in range(len(out["tok"])):
        d = P.mask_digest({k: out[k][t] for k in P.FIELDS}, meta["row_tier"][t], t, int(meta["parent"][t]))
        assert got.setdefault(int(meta["triple"][t]), d) == d


def test_host_counts_leave_out_i_and_k():
    b = built()
    out, meta, ids, genomes = b["out"], b["meta"], b["ids"], b["genomes"]
    t = 0
    g = genomes[ids[out["genome"][t, 0]]]
    want = (g.ccount - g.gene_count(int(meta["gene_i"][t])) - g.gene_count(int(meta["gene_k"][t])))[1:65]
    assert (meta["host_counts"][t] == want).all()


def test_split_units_follow_genus_and_cluster():
    metas = [dict(id="a", group="g", ranks=[1, 2, 3, 4, 50, 500], cluster=""), dict(id="b", group="g", ranks=[1, 2, 3, 4, 50, 501], cluster="c1"),
             dict(id="c", group="g", ranks=[1, 2, 3, 4, 60, 600], cluster="c1"), dict(id="d", group="g", ranks=[1, 2, 3, 4, 0, 700], cluster=""),
             dict(id="e", group="h", ranks=[1, 2, 3, 4, 50, 800], cluster="")]
    u = I.split_units(metas)
    assert u["a"] == u["b"] == u["c"] == "g|c:c1" and u["d"] == "g|s:700" and u["e"] == "h|g:50"


def test_the_build_is_reproducible():
    a = built(seed=9)
    _CACHE.clear()
    b = built(seed=9)
    for part in ("out", "meta"):
        for k in a[part]:
            assert np.array_equal(a[part][k], b[part][k], equal_nan=True), k


def test_a_triple_without_a_common_row_keeps_zero_rows():
    imp, opened, dist, ids, truth, genomes, train, eligible = implanter(attempts=3)
    imp.common_rows = lambda targets, n: []                           # no genome holds all three genes
    out, meta, triples, counts, capacity = imp.build(opened, W2T.GROUPS, 5, per_tier=1, scenarios=False)
    gix = {g: i for i, g in enumerate(ids)}
    assert counts["triples"] == counts["zero_rows"] == counts["phase3"] > 0 and (meta["on_target"] == 0).all() and (out["genome"][:, 1:] == -1).all()
    assert I.verify(out, meta, {gix[g] for g in eligible}, {gix[g] for g in train}) == []
    assert (meta["attempts"] > 6).all()                               # both conditioned phases were tried first


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)


# ---- tables of version 1: the homolog rows of W12's uncapped search (the stand-in DIAMOND of W12's tests) ----
import pytest  # noqa: E402


@pytest.fixture(scope="module")
def v1(tmp_path_factory):
    import json as _json
    import stat
    import test_rel_w12_search as T12
    import test_rel_w2_stage as TS
    from seeddrift.train import source_model as SM
    from seeddrift.train.rel.tables import v1_hits as H
    tmp = tmp_path_factory.mktemp("w18v1")
    src = str(tmp / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    metas = TS.make_source(src)
    T12.add_corpus(src, metas)
    fake = tmp / "diamond"
    fake.write_text(T12.FAKE % dict(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    run_dir, cfg, rep = T12.run_stage(dict(tmp=tmp, src=src, metas=metas, diamond=str(fake)), tmp / "s", "20261123T000010-search")
    with open(os.path.join(run_dir, "data", "rel_w12", "search", "genomes.json")) as f:
        sgen = _json.load(f)["genomes"]
    by = {m["id"]: m for m in metas}
    opened = [by[g["id"]] for g in sgen if g["split"] in ("train", "heldout")]
    arrays = {m["id"]: SM.Genome(m, os.path.join(src, "data", "train_data", m["id"] + ".npz")) for m in opened}
    ids = [m["id"] for m in opened]
    dist = distances(arrays, ids)
    train = [m["id"] for m in opened if m["split"] == "train"]
    eligible = {m["id"] for m in opened if m["split"] == "heldout"}
    rows_of = lambda gid: H.load(os.path.join(run_dir, "data", "rel_w12", "search", gid + ".npz"))      # noqa: E731
    imp = I.ImplanterV1(sgen, arrays, rows_of, train, ids, dist, ids, EDGES, eligible, I.split_units(opened), rows=8, min_codons=20, length_tol=0.6)
    out, meta, triples, counts, capacity = imp.build(opened, sorted(TS.TINY), 5, per_tier=2, group_tier_cap=6)
    return dict(out=out, meta=meta, triples=triples, counts=counts, imp=imp, ids=ids, train=train, eligible=eligible, opened=opened, sgen=sgen,
                rows_of=rows_of, dist=dist)


def test_version_1_triples_are_sound(v1):
    out, meta, ids = v1["out"], v1["meta"], v1["ids"]
    gix = {g: i for i, g in enumerate(ids)}
    units = I.split_units(v1["opened"])
    assert v1["counts"]["triples"] > 0 and I.verify(out, meta, {gix[g] for g in v1["eligible"]}, {gix[g] for g in v1["train"]},
                                                    {gix[g]: u for g, u in units.items()}) == []
    assert {"genome_of_row", "row_gene", "row_first"} <= set(out)
    T = I.TARGET
    assert not (out["kind"] == A.KIND_SELF_COPY).any() and not (out["kind"] == A.KIND_NOT_SEARCHED).any()
    exists = out["genome"] >= 0
    assert ((out["genome_of_row"] >= 0) == exists).all() and (out["genome_of_row"][exists] == np.nonzero(exists)[1]).all()   # one row per genome
    assert (out["tier"] == meta["row_tier"]).all()
    absent = out["kind"][:, 1:, :T] == A.KIND_ABSENT
    assert absent.any()                                                # a row's genome without a homolog of a context gene: searched, absent


def test_version_1_rows_are_the_search_pairs_and_take_tiers_in_turn(v1):
    out, meta, ids, sgen = v1["out"], v1["meta"], v1["ids"], v1["sgen"]
    T = I.TARGET
    sidx = {g["id"]: k for k, g in enumerate(sgen)}
    checked = 0
    for t in range(len(out["tok"])):
        v = int(meta["variant"][t])
        owner = ids[meta["donor"][t]] if v == 0 else ids[out["genome"][t, 0]]
        gene = int([meta["gene_j"][t], meta["gene_i"][t], meta["gene_k"][t]][v])
        a = v1["rows_of"](owner)
        for q in range(1, out["tok"].shape[1]):
            if out["genome"][t, q] < 0:
                continue
            ti = sidx[ids[out["genome"][t, q]]]
            pairs = [p for p in range(int(a["off"][gene]), int(a["off"][gene + 1])) if int(a["tgenome"][p]) == ti]
            assert pairs and out["row_gene"][t, q, T] in {int(a["tgene"][p]) for p in pairs} and out["kind"][t, q, T] == A.KIND_PRESENT
            checked += 1
        tiers = [int(x) for x in meta["row_tier"][t, 1:] if x >= 0]
        k = len(set(tiers))                                            # tiers take turns: the first round holds every tier once, in order
        assert tiers[:k] == sorted(set(tiers))
    assert checked > 10


def test_version_1_masks_are_shared_in_a_triple(v1):
    out, meta = v1["out"], v1["meta"]
    got = {}
    for t in range(len(out["tok"])):
        d = P.mask_digest({k: out[k][t] for k in P.FIELDS}, meta["row_tier"][t], t, int(meta["parent"][t]))
        assert got.setdefault(int(meta["triple"][t]), d) == d
