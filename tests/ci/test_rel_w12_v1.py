"""Tables of version 1 (seeddrift/train/rel/tables/v1_build.py) on the output of stage rel_w12_search over the
synthetic source run, with the stand-in DIAMOND of tests/ci/test_rel_w12_search.py, whose alignments are known: one
ungapped alignment from the first residue over the shorter protein and, for proteins of 40 or more, a second one over
the second half (which overlaps the first and must be dropped)."""
import json
import os

import numpy as np
import pytest

import test_rel_w12_search as T12
import test_rel_w2_stage as TS
from seeddrift.train import source_model as S
from seeddrift.train.rel import api as A
from seeddrift.train.rel.tables import v0_census as C
from seeddrift.train.rel.tables import v1_build as V1
from seeddrift.train.rel.tables import v1_hits as H

GROUPS = sorted(TS.TINY)


def dist(a, b):
    """A distance with every tier occupied: 0 for a genome and itself, inf for some pairs."""
    if a == b:
        return 0.0
    x = (a * 7 + b * 3) % 11
    return float("inf") if x == 10 else x / 6.0


@pytest.fixture(scope="module")
def setup(tmp_path_factory, source_w12):
    run_dir, genomes = source_w12["run_dir"], source_w12["genomes"]
    metas = {m["id"]: m for m in source_w12["metas"]}
    arrays = {g["id"]: S.Genome(metas[g["id"]], os.path.join(source_w12["src"], "data", "train_data", g["id"] + ".npz")) for g in genomes}
    rows_of = lambda gid: H.load(os.path.join(run_dir, "data", "rel_w12", "search", gid + ".npz"))     # noqa: E731
    train = [i for i, g in enumerate(genomes) if g["split"] == "train"]
    b = V1.Builder(genomes, arrays, rows_of, dist, train, rows=32)
    return dict(source_w12, arrays=arrays, rows_of=rows_of, train=train, builder=b)


@pytest.fixture(scope="module")
def source_w12(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w12v1")
    src = str(tmp / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    metas = TS.make_source(src)
    T12.add_corpus(src, metas)
    fake = tmp / "diamond"
    import stat
    import sys
    fake.write_text(T12.FAKE % dict(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    source = dict(tmp=tmp, src=src, metas=metas, diamond=str(fake))
    with pytest.MonkeyPatch.context():
        run_dir, cfg, rep = T12.run_stage(source, tmp / "s", "20261123T000010-search")
    with open(os.path.join(run_dir, "data", "rel_w12", "search", "genomes.json")) as f:
        genomes = json.load(f)["genomes"]
    return dict(source, run_dir=run_dir, genomes=genomes)


def build(setup, split="train", per_group=4, seed=3):
    plan = V1.plan_set(setup["genomes"], GROUPS, split, per_group, seed)
    out = V1.alloc(len(plan[0]), 32)
    cen = V1.Census1(len(GROUPS), 32)
    for t, gid in enumerate(plan[0]):
        setup["builder"].table(out, t, gid, plan[1][t], V1.SPLIT_CODE[split], V1.table_rng(seed, split, t), cen)
    return plan, out, cen


@pytest.fixture(scope="module")
def both(setup):
    return [build(setup, split) for split in ("train", "heldout")]


def test_rows_follow_the_rules_of_version_1(setup, both):
    genomes, gi = setup["genomes"], {g["id"]: i for i, g in enumerate(setup["genomes"])}
    train = set(setup["train"])
    seen_self = seen_absent = dropped = 0
    for plan, z, cen in both:
        dropped += int(cen.a["dropped"].sum())
        seen_self, seen_absent = check_tables(setup, genomes, gi, train, plan, z, seen_self, seen_absent)
    assert seen_self > 0 and seen_absent > 0 and dropped > 0            # self copies (held-out paralogs); absent cells; the overlapping halves


def check_tables(setup, genomes, gi, train, plan, z, seen_self, seen_absent):
    for t, gid in enumerate(plan[0]):
        me = gi[gid]
        a = setup["rows_of"](gid)
        i = int(z["gene"][t, V1.TARGET])
        pairs = {(int(a["tgenome"][p]), int(a["tgene"][p])) for p in range(int(a["off"][i]), int(a["off"][i + 1]))}
        rows = [r for r in range(1, 32) if z["genome"][t, r] >= 0]
        assert rows == list(range(1, len(rows) + 1))
        number = {}
        for r in rows:
            g, j = int(z["genome"][t, r]), int(z["row_gene"][t, r, V1.TARGET])
            assert (g, j) in pairs and (g == me or g in train)              # every row: a pair on the target gene
            if g == me:
                assert z["genome_of_row"][t, r] == 0 and z["kind"][t, r, V1.TARGET] == A.KIND_SELF_COPY and z["tier"][t, r] == 0
                seen_self += 1
            else:
                number.setdefault(g, len(number) + 1)
                assert z["genome_of_row"][t, r] == number[g] and z["kind"][t, r, V1.TARGET] == A.KIND_PRESENT
                assert z["tier"][t, r] == int(V1.D.tier_of([dist(me, g)], V1.TIER_EDGES)[0])
        selfs = [r for r in rows if z["genome"][t, r] == me]
        assert selfs == list(range(1, len(selfs) + 1)) and len(selfs) <= V1.MAX_SELF_COPY_ROWS      # self copies right after row 0
        g_rows = [int(z["genome"][t, r]) for r in rows if z["genome"][t, r] != me]
        assert all(g_rows.count(g) <= V1.MAX_ROWS_PER_GENOME for g in g_rows)
        tiers = [int(z["tier"][t, r]) for r in rows if z["genome"][t, r] != me]
        if len(set(tiers)) > 1:                                         # tiers take turns: no tier twice before every tier with rows once
            first_round = tiers[:len(set(tiers))]
            assert len(set(first_round)) == len(first_round)
        for r in rows:                                                  # cells: the stand-in's ungapped alignment from residue 1
            g = int(z["genome"][t, r])
            for b, (_, s0, s1) in enumerate(V1.SLICES):
                ib = int(z["gene"][t, b])
                if ib < 0:
                    assert z["kind"][t, r, b] == 0
                    continue
                has = [j for (tg, j) in {(int(a["tgenome"][p]), int(a["tgene"][p])) for p in range(int(a["off"][ib]), int(a["off"][ib + 1]))}
                       if tg == g and not (g == me and j == ib)]
                if not has:
                    assert z["kind"][t, r, b] == A.KIND_ABSENT
                    seen_absent += 1
                    continue
                j = int(z["row_gene"][t, r, b])
                assert j in has and z["row_first"][t, r, b] == 0
                partner = setup["arrays"][genomes[g]["id"]]
                cb = partner.codons[partner.coff[j]:partner.coff[j + 1]]
                n = int(z["n"][t, r, b])
                pair = [p for p in range(int(a["off"][ib]), int(a["off"][ib + 1])) if int(a["tgenome"][p]) == g and int(a["tgene"][p]) == j][0]
                full = int(a["qe"][a["hsp_off"][pair]]) - int(a["qs"][a["hsp_off"][pair]]) + 1      # the stand-in's alignment from residue 1
                want = min(full, len(cb), int(z["frame_len"][t, b]), s1 - s0)
                assert n == want and (z["tok"][t, r, s0:s0 + n] == cb[:n]).all() and (z["col"][t, r, s0:s0 + n] == np.arange(n)).all()
    return seen_self, seen_absent


def test_tables_are_reproducible_and_the_census_counts_them(setup):
    p1, z1, c1 = build(setup, seed=5)
    p2, z2, c2 = build(setup, seed=5)
    assert p1 == p2 and all(np.array_equal(z1[f], z2[f], equal_nan=(f == "identity")) for f in z1)
    doc = C.summarise(c1.base, GROUPS)
    assert doc["all"]["tables"] == len(p1[0]) and "absent" in doc["all"]["kinds"]["not_target"]
    assert c1.a["tier_rows"].sum() == int(((z1["genome"][:, 1:] >= 0) & (z1["genome"][:, 1:] != z1["genome"][:, :1])).sum())
    assert c1.a["self_rows"].sum() == int((z1["genome"][:, 1:] == z1["genome"][:, :1]).sum())
    back = V1.Census1.from_arrays(c1.to_arrays(), 32)
    assert all(np.array_equal(back.a[k], c1.a[k]) for k in c1.a)


def test_build_set_writes_sound_shards_in_parallel(setup, tmp_path):
    genomes = setup["genomes"]
    plan = V1.plan_set(genomes, GROUPS, "heldout", 2, 7)
    shards, cen = V1.build_set(setup["builder"], GROUPS, "heldout", plan, str(tmp_path / "a"), shard_tables=10, seed=7, processes=2)
    shards2, _ = V1.build_set(setup["builder"], GROUPS, "heldout", plan, str(tmp_path / "b"), shard_tables=10, seed=7, processes=1)
    for f in V1.fields(32):
        for (name, _), (name2, _) in zip(shards, shards2):
            x, y = np.load(os.path.join(tmp_path, "a", "%s.%s.npy" % (name, f))), np.load(os.path.join(tmp_path, "b", "%s.%s.npy" % (name2, f)))
            assert np.array_equal(x, y, equal_nan=(f == "identity")), f
    examined = {i for i, g in enumerate(genomes) if g["split"] == "heldout"}
    assert V1.verify_set(str(tmp_path / "a"), shards, 32, "heldout", examined, set(setup["train"]), len(GROUPS)) == []
    p = os.path.join(tmp_path, "a", shards[0][0] + ".genome_of_row.npy")
    g = np.load(p)
    g[0, 1] = 0 if g[0, 1] != 0 else 3                                  # a relative marked as the examined genome
    np.save(p, g)
    assert V1.verify_set(str(tmp_path / "a"), shards, 32, "heldout", examined, set(setup["train"]), len(GROUPS)) != []


def test_placement_keeps_collinear_alignments_and_drops_the_rest():
    hits = [(1, 10, 1, 10, b"10M"), (5, 15, 20, 30, b"11M"), (20, 25, 40, 45, b"6M"), (30, 35, 2, 7, b"6M")]
    q, t, dropped = V1.placement(hits, 100, 100)
    assert dropped == 2                          # the longest (5-15) first; 1-10 overlaps it on the query, 30-35 is out of order on the target
    assert q.tolist() == list(range(4, 15)) + list(range(19, 25)) and t.tolist() == list(range(19, 30)) + list(range(39, 45))
    q2, t2, _ = V1.placement(hits, 8, 100)                              # only the query columns inside the frame
    assert q2.tolist() == list(range(4, 8)) and t2.tolist() == list(range(19, 23))


def test_a_gene_is_the_target_again_only_after_every_gene_of_its_genome(setup, tmp_path):
    genomes = setup["genomes"]
    plan = V1.plan_set(genomes, GROUPS, "train", 30, 11)
    assert len(plan) == 4 and all(plan[3][t] == sum(1 for u in range(t) if plan[0][u] == plan[0][t]) for t in range(len(plan[0])))
    shards, _ = V1.build_set(setup["builder"], GROUPS, "train", plan, str(tmp_path / "t"), shard_tables=50, seed=11, processes=2)
    gene = np.concatenate([np.load(os.path.join(tmp_path, "t", "%s.gene.npy" % n))[:, V1.TARGET] for n, _ in shards])
    gid = {g["id"]: i for i, g in enumerate(genomes)}
    for g in set(plan[0]):
        a = setup["arrays"][g]
        ok = {int(j) for j in np.flatnonzero(np.diff(a.coff) > 0)}
        mine = [int(gene[t]) for t in range(len(plan[0])) if plan[0][t] == g]       # in the order of the plan
        n = len(ok)
        for x in range(0, len(mine), n):
            chunk = mine[x:x + n]
            assert len(set(chunk)) == len(chunk) and set(chunk) <= ok              # no repeat inside one pass over the genome
        assert set(mine[:n]) == (ok if len(mine) >= n else set(mine[:n]))
    del gid
    again = V1.build_set(setup["builder"], GROUPS, "train", plan, str(tmp_path / "u"), shard_tables=50, seed=11, processes=1)[0]
    g2 = np.concatenate([np.load(os.path.join(tmp_path, "u", "%s.gene.npy" % n))[:, V1.TARGET] for n, _ in again])
    assert np.array_equal(gene, g2)


def test_a_larger_set_holds_the_smaller_one_table_for_table(setup, tmp_path):
    """Same search, seed and code: the tables of a set with more tables per group begin with those of the smaller set,
    byte for byte (a table depends on its number alone), and the held-out tables are the same."""
    genomes = setup["genomes"]
    small, big = V1.plan_set(genomes, GROUPS, "train", 6, 5), V1.plan_set(genomes, GROUPS, "train", 12, 5)
    n = len(small[0])
    assert [x[:n] for x in (big[0], big[1], big[3])] == [small[0], small[1], small[3]]
    a, _ = V1.build_set(setup["builder"], GROUPS, "train", small, str(tmp_path / "s"), shard_tables=len(GROUPS) * 2, seed=5, processes=2)
    b, _ = V1.build_set(setup["builder"], GROUPS, "train", big, str(tmp_path / "b"), shard_tables=len(GROUPS) * 2, seed=5, processes=2)
    assert [x[0] for x in b[:len(a)]] == [x[0] for x in a]
    for name, _ in a:
        for f in V1.fields(32):
            x, y = (open(os.path.join(tmp_path, d, "%s.%s.npy" % (name, f)), "rb").read() for d in ("s", "b"))
            assert x == y, (name, f)
    h1, h2 = V1.plan_set(genomes, GROUPS, "heldout", 2, 5), V1.plan_set(genomes, GROUPS, "heldout", 2, 5)
    assert h1 == h2
