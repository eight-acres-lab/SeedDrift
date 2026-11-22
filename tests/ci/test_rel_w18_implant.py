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
from seeddrift.train.rel.tables import v0_dist as DI  # noqa: E402

EDGES = [0.05, 0.2, 0.5, 1.0]
_CACHE = {}


def distances(genomes, ids):
    sk = [DI.sketch(genomes[g].codons, genomes[g].coff, 4, 256) for g in ids]
    jac, _ = DI.matrix([s[0] for s in sk], [s[1] for s in sk], 256, 1)
    return DI.mash(jac, 4)


def built(per_group=4, seed=5, rows=8):
    key = (per_group, seed, rows)
    if key not in _CACHE:
        d, metas, truth, genomes, ids, train, _ = W2T.run()
        dist = distances(genomes, ids)
        eligible = {m["id"] for m in metas if m["split"] == "heldout"}
        imp = I.Implanter(SS.Pairs(d), genomes, train, ids, dist, ids, EDGES, eligible, rows=rows, min_codons=20, length_tol=0.5, attempts=40)
        out, meta, triples, counts = imp.build([m for m in metas if m["split"] in ("train", "heldout")], W2T.GROUPS, per_group, seed)
        _CACHE[key] = (out, meta, triples, counts, imp, dist, ids, truth, genomes, metas, train)
    return _CACHE[key]


def test_triples_are_sound():
    out, meta, triples, counts, imp, dist, ids, truth, genomes, metas, train = built()
    gix = {g: i for i, g in enumerate(ids)}
    eligible = {gix[m["id"]] for m in metas if m["split"] == "heldout"}
    assert I.verify(out, meta, eligible, {gix[g] for g in train}) == []
    assert counts["triples"] >= 6 and len(out["tok"]) == 3 * counts["triples"]
    assert sorted(set(meta["variant"].tolist())) == [0, 1, 2]


def test_target_blocks_hold_the_right_genes():
    out, meta, triples, counts, imp, dist, ids, truth, genomes, metas, train = built()
    T = I.TARGET
    s0 = I.SLICES[T][1]
    for t in range(len(out["tok"])):
        a, d = ids[out["genome"][t, 0]], ids[meta["donor"][t]]
        v = int(meta["variant"][t])
        owner, gene = [(d, meta["gene_j"][t]), (a, meta["gene_i"][t]), (a, meta["gene_k"][t])][v]
        g = genomes[owner]
        c = g.codons[g.coff[gene]:g.coff[gene + 1]][:A.LEN_TARGET]
        n = int(out["n"][t, 0, T])
        assert n == len(c) and (out["tok"][t, 0, s0:s0 + n] == c).all()
        assert out["start"][t, 0, T] == -1 and out["strand"][t, 0, T] == 0 and out["replicon"][t, 0, T] == -1
        assert out["gene"][t, T] == (-1 if v == 0 else gene)
        if v == 2:
            assert int(gene) not in out["gene"][t, :T].tolist()       # k is no relation block and no neighbour
        li = I.stored_len(genomes[a], int(meta["gene_i"][t]))
        assert abs(n - li) <= 0.5 * li + 1e-9


def test_rows_are_shared_and_the_donor_is_never_a_row():
    out, meta, triples, counts, imp, dist, ids, truth, genomes, metas, train = built()
    T = I.TARGET
    for s in np.unique(meta["triple"]):
        ts = np.flatnonzero(meta["triple"] == s)
        g = out["genome"][ts]
        assert (g == g[0]).all()
        assert meta["donor"][ts[0]] not in g[0].tolist()
        rows = int((g[0, 1:] >= 0).sum())
        for x in ts:
            assert int((out["kind"][x, 1:, T] == A.KIND_PRESENT).sum()) == rows == int(meta["on_target"][x])
            outside = np.ones(A.ROW_LEN, dtype=bool)
            outside[I.SLICES[T][1]:] = False
            assert (out["tok"][x][:, outside] == out["tok"][ts[0]][:, outside]).all()
            assert (out["col"][x][:, outside] == out["col"][ts[0]][:, outside]).all()


def test_implant_rows_are_placed_by_the_true_alignment_on_the_donor_gene():
    out, meta, triples, counts, imp, dist, ids, truth, genomes, metas, train = built()
    T = I.TARGET
    s0, s1 = I.SLICES[T][1], I.SLICES[T][2]
    checked = 0
    for t in np.flatnonzero(meta["variant"] == 0):
        d, j = ids[meta["donor"][t]], int(meta["gene_j"][t])
        for q in range(1, out["tok"].shape[1]):
            if out["genome"][t, q] < 0:
                continue
            pid = ids[out["genome"][t, q]]
            pj, pairs_ = truth[(d, j, pid)]
            pg = genomes[pid]
            pc = pg.codons[pg.coff[pj]:pg.coff[pj + 1]]
            want = {x: int(pc[y]) for y, x in pairs_.items()}            # column on D:j -> the partner's codon there
            col, tok = out["col"][t, q, s0:s1], out["tok"][t, q, s0:s1]
            for c, k in zip(col[col >= 0], tok[col >= 0]):
                if c in want:
                    assert want[c] == int(k)
                    checked += 1
    assert checked > 50


def test_tiers_and_distances_follow_the_matrix():
    out, meta, triples, counts, imp, dist, ids, truth, genomes, metas, train = built()
    for t in range(len(out["tok"])):
        a = int(out["genome"][t, 0])
        assert np.isclose(meta["donor_dist"][t], dist[a, meta["donor"][t]]) or (np.isinf(meta["donor_dist"][t]) and np.isinf(dist[a, meta["donor"][t]]))
        assert meta["donor_tier"][t] == DI.tier_of(np.array([dist[a, meta["donor"][t]]]), EDGES)[0]
        for q in range(out["tok"].shape[1]):
            r = int(out["genome"][t, q])
            if q == 0 or r < 0:
                assert meta["row_tier"][t, q] == -1
            else:
                assert meta["row_tier"][t, q] == DI.tier_of(np.array([dist[a, r]]), EDGES)[0]


def test_host_counts_leave_out_i_and_k():
    out, meta, triples, counts, imp, dist, ids, truth, genomes, metas, train = built()
    t = 0
    g = genomes[ids[out["genome"][t, 0]]]
    want = (g.ccount - g.gene_count(int(meta["gene_i"][t])) - g.gene_count(int(meta["gene_k"][t])))[1:65]
    assert (meta["host_counts"][t] == want).all()


def test_the_build_is_reproducible():
    a = built(per_group=3, seed=9)
    _CACHE.clear()
    b = built(per_group=3, seed=9)
    for k in a[0]:
        assert np.array_equal(a[0][k], b[0][k], equal_nan=True), k
    for k in a[1]:
        assert np.array_equal(a[1][k], b[1][k], equal_nan=True), k


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
