"""Distances between genomes from shared amino-acid k-mers (seeddrift/train/rel/tables/v0_dist.py) against exact set
arithmetic and against known divergence, and stage rel_w2_dist (ci/stages/rel/w2_dist.py) on the synthetic source run
of the table stages (tests/ci/test_rel_w2_stage.py). Numpy only."""
import json
import math
import os

import numpy as np
import pytest

import rel_fixtures as RF
import test_rel_w2_stage as TS
from ci.gates import GateFailure
from ci.stages.common import Stage
from ci.stages.rel import gates as rel_gates, w2_dist, w2_v0
from seeddrift.train import source_model as S
from seeddrift.train.rel.tables import v0_dist as D

MASK64 = (1 << 64) - 1


def splitmix_ref(x):
    z = (x + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def codon_of(aa):
    """A codon token that codes amino-acid token aa (1..20; 21: a stop codon)."""
    return next(c for c in range(1, 65) if S.CODE[c] == aa)


def genome(rng, genes=30, longest=80):
    lens = rng.integers(20, longest, genes)
    coff = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    return rng.integers(1, 65, coff[-1]).astype(np.uint8), coff


def mutate(rng, codons, rate):
    c = codons.copy()
    hit = rng.random(len(c)) < rate
    c[hit] = rng.integers(1, 65, int(hit.sum()))
    return c


def kmer_set(codons, coff, k):
    return set(D.kmer_codes(codons, coff, k).tolist())


def test_splitmix_matches_the_reference():
    xs = [0, 1, 2, 12345, 2 ** 40 + 7, 20 ** 8 - 1]
    assert D.splitmix64(np.array(xs, dtype=np.int64)).tolist() == [splitmix_ref(x) for x in xs]


def test_kmers_stay_inside_genes_and_skip_stops_and_unknown_codons():
    aa = [1, 2, 3, 4, 5, 6, 21, 7, 8, 9]                              # gene 1: A C D E; gene 2: F G stop H I K
    codons = np.array([codon_of(a) for a in aa], dtype=np.uint8)
    coff = np.array([0, 4, 10])
    want = [(aa[p] - 1) * 20 + aa[p + 1] - 1 for g0, g1 in ((0, 4), (4, 10)) for p in range(g0, g1 - 1) if max(aa[p], aa[p + 1]) <= 20]
    assert D.kmer_codes(codons, coff, 2).tolist() == want and len(want) == 6
    unknown = codons.copy()
    unknown[0] = 65                                                    # a codon with an ambiguous base
    assert D.kmer_codes(unknown, coff, 2).tolist() == want[1:]
    assert len(D.kmer_codes(codons, coff, 5)) == 0                     # no gene holds five residues without a stop
    with pytest.raises(ValueError):
        D.kmer_codes(codons, coff, 13)
    with pytest.raises(ValueError):
        D.kmer_codes(codons, np.array([0, 4, 9]), 2)                  # offsets that do not cover the codons


def test_complete_sketches_give_the_exact_jaccard_and_containment():
    rng = np.random.default_rng(1)
    a, coff = genome(rng)
    b = mutate(rng, a, 0.1)
    c = a[:coff[12]]                                                   # the first 12 genes of a: contained in a
    big = 1 << 16
    for x, y, cy in ((a, b, coff), (a, c, coff[:13])):
        sx, nx = D.sketch(x, coff, 4, big)
        sy, ny = D.sketch(y, cy, 4, big)
        X, Y = kmer_set(x, coff, 4), kmer_set(y, cy, 4)
        assert (nx, ny) == (len(X), len(Y)) == (len(sx), len(sy))
        j, shared, con = D.compare(sx, sy, big, nx, ny)
        small = X if len(X) <= len(Y) else Y
        assert shared == len(X & Y) and abs(j - len(X & Y) / len(X | Y)) < 1e-12 and abs(con - len(X & Y) / len(small)) < 1e-12
    sc, nc = D.sketch(c, coff[:13], 4, big)
    assert D.compare(sc, D.sketch(a, coff, 4, big)[0], big, nc, None)[2] == 1.0


def test_bottom_sketches_estimate_the_jaccard():
    rng = np.random.default_rng(2)
    a, coff = genome(rng, genes=300, longest=300)
    for rate in (0.01, 0.05, 0.2):
        b = mutate(rng, a, rate)
        A, B = kmer_set(a, coff, 5), kmer_set(b, coff, 5)
        exact = len(A & B) / len(A | B)
        sa, na = D.sketch(a, coff, 5, 2048)
        sb, nb = D.sketch(b, coff, 5, 2048)
        assert len(sa) == 2048 < na
        j, _, _ = D.compare(sa, sb, 2048, na, nb)
        assert abs(j - exact) < 4 * math.sqrt(exact * (1 - exact) / 2048) + 1e-3, (rate, j, exact)


def test_distance_grows_with_divergence_and_tiers_follow_it():
    rng = np.random.default_rng(3)
    a, coff = genome(rng, genes=200, longest=250)
    sa, na = D.sketch(a, coff, 6, 4096)
    ds = []
    for rate in (0.0, 0.01, 0.05, 0.15, 0.4):
        sb, nb = D.sketch(mutate(rng, a, rate), coff, 6, 4096)
        ds.append(float(D.mash(D.compare(sa, sb, 4096, na, nb)[0], 6)))
    assert ds[0] == 0.0 and all(x < y for x, y in zip(ds, ds[1:]))
    assert abs(ds[1] - 0.01) < 0.005                                   # about the share of residues that differ
    other, ocoff = genome(np.random.default_rng(99), genes=200, longest=250)
    so, no = D.sketch(other, ocoff, 6, 4096)
    j_other = D.compare(sa, so, 4096, na, no)[0]                       # unrelated: a few 6-mers shared by chance at most
    assert j_other < 2e-3 and D.mash(j_other, 6) > 1.0 and np.isinf(D.mash(0.0, 6))
    assert D.tier_of([0.0, 0.05, 0.1, 0.3, 2.0, np.inf], [0.05, 0.2, 1.0]).tolist() == [0, 1, 1, 2, 3, 4]


def test_matrix_is_symmetric_and_the_same_in_parallel():
    rng = np.random.default_rng(4)
    a, coff = genome(rng, genes=40)
    sk = [D.sketch(mutate(rng, a, r), coff, 5, 512) for r in (0.0, 0.02, 0.1, 0.3, 0.6, 1.0)]
    j1, c1 = D.matrix([s for s, _ in sk], [n for _, n in sk], 512, processes=1)
    j2, c2 = D.matrix([s for s, _ in sk], [n for _, n in sk], 512, processes=2)
    assert np.array_equal(j1, j2) and np.array_equal(c1, c2) and np.array_equal(j1, j1.T) and (np.diag(j1) == 1).all()
    assert j1[0, 1] == D.compare(sk[0][0], sk[1][0], 512, sk[0][1], sk[1][1])[0]
    assert all(j1[0, i] > j1[0, i + 1] for i in range(1, 4))           # farther copies share fewer k-mers


def test_spearman_and_ranks():
    assert w2_dist._ranks([3.0, 1.0, 1.0, np.inf]).tolist() == [2.0, 0.5, 0.5, 3.0]
    x = np.arange(10.0)
    assert abs(w2_dist.spearman(x, x ** 3) - 1.0) < 1e-12 and abs(w2_dist.spearman(x, -x) + 1.0) < 1e-12
    assert w2_dist.spearman([1.0, 2.0], [2.0, 1.0]) is None


@pytest.fixture(scope="module")
def dist(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w2dist")
    src = str(tmp / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    metas, _ = TS.make_source(src)
    with pytest.MonkeyPatch.context() as mp:
        TS.patch(mp, tmp)
        tables, tcfg = RF.rel_run(tmp, stages=("w2_v0",), run_id="20261122T000003-tables",
                                  rel_w2_v0=dict(source=src, expect=TS.SIZES, rows=16, train_per_group=6, heldout_per_group=2, shard_tables=24, processes=1))
        with Stage(tables, "rel_w2_v0") as st:
            w2_v0.build(st, tcfg, tables)
        with open(os.path.join(tables, "stages", "rel_w2_v0.json")) as f:
            stage = json.load(f)
        with open(os.path.join(tables, "record.json"), "w") as f:
            json.dump(dict(run_id=os.path.basename(tables), status="passed", stages=[stage], pipeline="rel_tables"), f)
        run_dir, cfg = RF.rel_run(tmp, stages=("w2_dist",), run_id="20261122T000004-dist",
                                  rel_w2_dist=dict(source=src, expect=TS.SIZES, ks=[4, 5], size=256, edges=[0.1, 0.3, 0.6], processes=2, tables_from=tables))
        with Stage(run_dir, "rel_w2_dist") as st:
            rep = w2_dist.build(st, cfg, run_dir)
    return dict(tmp=tmp, src=src, metas=metas, tables=tables, run_dir=run_dir, cfg=cfg, rep=rep)


def test_stage_writes_distances_of_the_open_genomes(dist):
    run_dir, rep = dist["run_dir"], dist["rep"]
    with open(os.path.join(run_dir, w2_dist.OUT, "genomes.json")) as f:
        genomes = json.load(f)["genomes"]
    split = {m["id"]: m["split"] for m in dist["metas"]}
    ids = [g["id"] for g in genomes]
    assert len(ids) == 36 == rep["genomes"]["n"] and {split[g] for g in ids} == {"train", "heldout"}
    assert not set(ids) & set(TS.SEALED) and TS.LEFT_OUT not in ids
    text = open(os.path.join(run_dir, w2_dist.OUT, "report.json")).read() + open(os.path.join(run_dir, w2_dist.OUT, "report.md")).read()
    assert not any(g in text for g in TS.SEALED) and TS.LEFT_OUT not in text
    for k in (4, 5):
        d = np.load(os.path.join(run_dir, w2_dist.OUT, "distances_k%d.npy" % k))
        assert d.shape == (36, 36) and np.array_equal(d, d.T) and (np.diag(d) == 0).all() and (d >= 0).all()
        assert rep["by_k"][str(k)]["train"]["all"]["genomes"] == 24 and rep["by_k"][str(k)]["heldout"]["all"]["genomes"] == 12
    t = rep["by_k"]["4"]["train"]["all"]["tiers"]
    assert t["names"] == ["<0.10", "0.10-0.30", "0.30-0.60", ">=0.60", "no shared k-mer"]
    assert abs(sum(t["training_genomes_per_genome"].values()) - 23) < 1e-9           # every other training genome is in one tier
    rows = rep["rows"]["parts"]
    z = np.concatenate([np.load(os.path.join(dist["tables"], "data", "rel_w2", "v0", "train", "s%04d.genome.npy" % i)) for i in range(3)])
    assert rows["train"]["rows"] == int((z[:, 1:] >= 0).sum()) == sum(x["rows"] for x in rows["train"]["tiers"].values())
    assert rows["heldout"]["rows"] > 0 and rep["rows"]["tableset_sha256"]
    assert w2_dist.summary(run_dir)["json"]["genomes"] == 36


def test_gate_passes_and_fails_on_a_broken_matrix(dist, monkeypatch, tmp_path):
    import shutil
    TS.patch(monkeypatch, dist["tmp"])
    run_dir, cfg = dist["run_dir"], dist["cfg"]
    with Stage(run_dir, "gates_rel_w2_dist") as st:
        w2_dist.gate(st, cfg, run_dir)
    with open(os.path.join(run_dir, "gates", "G4_rel_w2_dist.json")) as f:
        assert json.load(f)["status"] == "pass"
    with Stage(run_dir, "gates_rel_tables") as st:                       # it writes no table set: the kind gates pass
        rel_gates.kind_phase(st, cfg, run_dir, ["w2_dist"])
    copy = str(tmp_path / "ci-root" / "runs" / os.path.basename(run_dir))
    shutil.copytree(run_dir, copy)
    p = os.path.join(copy, w2_dist.OUT, "distances_k4.npy")
    d = np.load(p)
    d[0, 1] += 0.5                                                       # no longer symmetric
    np.save(p, d)
    with pytest.raises(GateFailure):
        with Stage(copy, "gates_rel_w2_dist") as st:
            w2_dist.gate(st, dict(cfg, run_dir=copy), copy)
    with open(os.path.join(copy, "gates", "G4_rel_w2_dist.json")) as f:
        assert json.load(f)["status"] == "fail"


def test_settings_are_checked():
    for bad in (dict(source="x", colour="blue"), dict(ks=[6]), dict(source="x", ks=[2]), dict(source="x", edges=[0.5, 0.2]), dict(source="x", ks=[6, 6])):
        with pytest.raises(SystemExit):
            w2_dist.settings(dict(rel_w2_dist=bad))
