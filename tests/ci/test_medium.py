"""Medium-scale pieces that run without the GPU: the three-way split with a sealed test part, the construction without
repeated genes, nested training fractions, the whole-genome composition input, and the grid evaluation."""
import json

import numpy as np


def _lin(n, genus_of):
    return {e: dict(genus=genus_of(i), species=1000 + i) for i, e in enumerate(n)}


def test_three_way_split_keeps_genera_and_clusters_in_one_part():
    from ci.stages.train_data import three_way
    ids = ["x%d" % i for i in range(300)]
    groups = {e: ("bacteria" if i % 3 else "viruses.phage") for i, e in enumerate(ids)}
    lin = _lin(ids, lambda i: (i // 2) * 10 if i % 11 else 0)               # genera of two, some genomes without a genus
    cluster = {}
    for c in range(12):                                                    # anchors with three relatives each
        members = ids[200 + 4 * c:204 + 4 * c]
        for e in members:
            cluster[e] = members[0]
    sp = three_way(ids, groups, lin, cluster)
    assert set(sp.values()) == {"train", "heldout", "test"}
    for g in ("bacteria", "viruses.phage"):
        parts = {}
        for e in ids:
            if groups[e] == g and lin[e]["genus"]:
                parts.setdefault(lin[e]["genus"], set()).add(sp[e])
        assert all(len(v) == 1 for v in parts.values())                    # no genus in two parts
        n = sum(1 for e in ids if groups[e] == g)
        share = {k: sum(1 for e in ids if groups[e] == g and sp[e] == k) / n for k in ("train", "heldout", "test")}
        assert share["train"] > 0.45 and share["heldout"] > 0.08 and share["test"] > 0.08, share
    for c in set(cluster.values()):
        assert len({sp[e] for e in ids if cluster.get(e) == c}) == 1          # a cluster stays whole


def test_three_way_leaves_the_development_part_as_split_gives_it():
    from ci.stages.train_data import split, three_way
    ids = ["y%d" % i for i in range(120)]
    groups = {e: "archaea" for e in ids}
    lin = _lin(ids, lambda i: 10 + i // 3)
    two, three = split(ids, groups, lin), three_way(ids, groups, lin)
    assert {e for e in ids if two[e] == "heldout"} == {e for e in ids if three[e] == "heldout"}   # same development genomes
    assert {e for e in ids if three[e] == "test"} <= {e for e in ids if two[e] == "train"}


def _data(S, genomes):
    d = object.__new__(S.Data)
    d.g = {g.meta["id"]: g for g in genomes}
    d.by_split = {}
    for g in genomes:
        d.by_split.setdefault(g.meta["split"], []).append(g.meta["id"])
    return d


def _gen(S, gid, n, ranks, split="heldout", group="bacteria", cluster=""):
    g = object.__new__(S.Genome)
    g.meta = dict(id=gid, group=group, taxid=ranks[5], ranks=ranks, split=split, cluster=cluster)
    g.n, g.ranks = n, np.array(ranks)
    rng = np.random.default_rng(len(gid) * 7 + n)
    g.codons = rng.integers(1, 66, n * 10).astype(np.uint8)
    g.coff = np.arange(0, n * 10 + 1, 10)
    g.ccount = np.bincount(g.codons.astype(np.int64), minlength=66)[:66]
    return g


def test_construction_draws_distinct_genes_and_records_the_depth():
    from seeddrift.train import source_model as S
    small = _gen(S, "small", 3, [1, 2, 3, 4, 5, 6])
    big = _gen(S, "big", 200, [7, 8, 9, 10, 11, 12])
    test = _gen(S, "sealed", 50, [13, 14, 15, 16, 17, 18], split="test")
    d = _data(S, [small, big, test])
    ex = S.constructed(d, 0, "heldout", n_per=120)
    assert all(e[0] != "sealed" and e[2] != "sealed" and e[5] != "sealed" for e in ex)   # the sealed part is never read
    for gid, n in (("small", 3), ("big", 200)):
        nat = [e for e in ex if e[0] == gid and not e[2]]
        rep = [e for e in ex if e[0] == gid and e[2]]
        assert len(nat) == min(60, n) and len(rep) == min(60, n)
        assert len({e[1] for e in nat}) == len(nat) and len({e[1] for e in rep}) == len(rep)   # no repeated position
    assert {e[6] for e in ex} == {"genus"}
    rnd = S.constructed(d, 0, "heldout", n_per=120, depth="random")
    assert {e[6] for e in rnd} <= set(S.DEPTHS) and len({e[6] for e in rnd}) > 1
    assert S.constructed(d, 0, "heldout", n_per=120) == ex                   # fixed by the seed


def test_training_fractions_are_nested_and_move_whole_units():
    from seeddrift.train import source_model as S
    metas = {}
    for i in range(80):
        metas["t%d" % i] = dict(id="t%d" % i, group="bacteria" if i % 2 else "eukaryota.fungi", ranks=[0, 0, 0, 0, 100 + i // 4, 1000 + i],
                                cluster="t0" if i < 4 else "")
    ids = sorted(metas)
    q, h, a = (S.fraction_subset(metas, ids, f) for f in (0.25, 0.5, 1.0))
    assert set(q) <= set(h) <= set(a) and a == ids and 0 < len(q) < len(h) < len(a)
    for sub in (q, h):
        units = {}
        for g in ids:
            units.setdefault(S.unit_of(metas[g]), set()).add(g in sub)
        assert all(len(v) == 1 for v in units.values())                       # a genus or cluster is in or out whole
        assert {metas[g]["group"] for g in sub} == {"bacteria", "eukaryota.fungi"}   # every group keeps genomes


def test_genome_composition_is_the_same_for_a_native_and_a_replacement():
    from seeddrift.train import source_model as S
    A = _gen(S, "A", 6, [1, 0, 0, 0, 100, 10])
    B = _gen(S, "B", 6, [2, 0, 0, 0, 300, 30])
    for g in (A, B):
        g.hoff, g.hprot, g.hqs = np.zeros(g.n + 1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int32)
        g.track, g.toff = np.zeros(0, np.uint8), np.zeros(1, np.int64)
        g.order, g.pos, g.rep = np.arange(g.n), np.arange(g.n), np.zeros(g.n, np.int32)
        g.start = np.arange(g.n) * 100 + 1
        g.end, g.strand, g.rep_len = g.start + 90, np.ones(g.n, np.int8), np.array([1000])
    d = object.__new__(S.Data)
    d.lib_ranks = np.array([[1, 0, 0, 0, 100, 10]])
    d.lib_taxid = np.array([10])
    d.lib_cols = np.concatenate([d.lib_ranks, d.lib_taxid[:, None]], 1)
    d.n_lib, d.maxc, d.kc, d.kn, d.maxt, d.comp = 1, 512, 16, 4, 512, True
    nat = d.window(A, 2, None, "genus", partner=B)[S.WIN]
    rep = d.window(A, 2, (B, 4), "genus")[S.WIN]
    assert np.array_equal(nat["gcomp"], rep["gcomp"])                       # the genome minus the gene at the position
    assert np.allclose(nat["gcomp"], S.frequencies(A.ccount - A.gene_count(2))) and abs(nat["gcomp"].sum() - 1) < 1e-5
    assert np.allclose(rep["ccomp"], S.frequencies(B.gene_count(4))) and np.allclose(nat["ccomp"], S.frequencies(A.gene_count(2)))
    d.comp = False
    assert "gcomp" not in d.window(A, 2, None, "genus", partner=B)[S.WIN]


def test_grid_evaluation_reports_variants_ensembles_balanced_baselines_and_contrasts(tmp_path):
    from ci.stages.eval import cell_name, dev_constructed, paired_contrast
    run = tmp_path
    (run / "data" / "train_data").mkdir(parents=True)
    (run / "data" / "hits").mkdir(parents=True)
    lib_taxid = np.array([10, 20, 30, 40])
    ranks = np.array([[1, 0, 0, 0, 100, 10], [1, 0, 0, 0, 200, 20], [2, 0, 0, 0, 300, 30], [2, 0, 0, 0, 400, 40]])
    rng = np.random.default_rng(0)
    genomes = []
    names = [("A", 10, [1, 0, 0, 0, 100, 10], "bacteria", "heldout"), ("B", 30, [2, 0, 0, 0, 300, 30], "archaea", "heldout"),
             ("C", 20, [1, 0, 0, 0, 200, 20], "bacteria", "train"), ("D", 40, [2, 0, 0, 0, 400, 40], "archaea", "train")]
    for gid, tx, rk, grp, sp in names:
        d = run / "data" / "corpus" / gid
        d.mkdir(parents=True)
        genes = ["g%d" % i for i in range(8)]
        (d / "genes.tsv").write_text("gene_id\treplicon\tstart\tend\tstrand\n" + "".join("%s\tr\t%d\t%d\t+\n" % (g, 100 * i + 1, 100 * i + 90) for i, g in enumerate(genes)))
        gc = 0.3 if grp == "bacteria" else 0.7
        (d / "cds.fna").write_text("".join(">%s__%s\n%s\n" % (gid, g, "".join(rng.choice(list("ACGT"), 90, p=[(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])))
                                           for g in genes))
        np.savez(run / "data" / "hits" / (gid + ".npz"), off=np.arange(0, 8 * 3 + 1, 3), prot=np.tile([0, 1, 2], 8), pid=np.full(24, 800, np.uint16),
                 bits=rng.random(24).astype(np.float32) * 100, sseq_off=np.arange(0, 8 * 3 + 1) * 5)
        genomes.append(dict(id=gid, group=grp, taxid=tx, ranks=rk, split=sp, genes=8))
    (run / "data" / "train_data" / "index.json").write_text(json.dumps(dict(genomes=genomes)))
    model = run / "data" / "model"
    model.mkdir()

    def save(path, rows, score=None):
        cols = dict(recipient=np.array([r[0] for r in rows], dtype="S"), gene=np.array([r[1] for r in rows]), donor=np.array([r[2] for r in rows], dtype="S"),
                    donor_gene=np.array([r[3] for r in rows]), stratum=np.array([5] * len(rows)), partner=np.array([r[4] for r in rows], dtype="S"),
                    depth=np.array([r[5] for r in rows], dtype="S"))
        if score is not None:
            cols["score"] = np.asarray(score, float)
        np.savez(path, **cols)
    dev = [("A", i, "", -1, "B", "genus") for i in range(4)] + [("A", i, "B", i, "B", "genus") for i in range(4, 8)] + \
          [("B", i, "", -1, "A", "genus") for i in range(4)] + [("B", i, "A", i, "A", "genus") for i in range(4, 8)]
    y = np.array([1.0 if r[2] else 0.0 for r in dev])
    train = [("C", i, "", -1, "D", "order") for i in range(4)] + [("C", i, "D", i, "D", "species") for i in range(4, 8)] + \
            [("D", i, "", -1, "C", "family") for i in range(4)] + [("D", i, "C", i, "C", "genus") for i in range(4, 8)]
    save(model / "baseline_train.npz", train)
    save(model / "baseline_train_f0.5.npz", train)
    (model / "fractions.json").write_text(json.dumps({"0.5": ["C", "D"], "1": ["C", "D"]}))
    variants = {}
    for name, f, s in (("f1-w128-s1", 1.0, 1), ("f1-w128-s2", 1.0, 2), ("f0.5-w128-s1", 0.5, 1), ("f0.5-w128-s2", 0.5, 2)):
        (model / name).mkdir()
        save(model / name / "dev_eval.npz", dev, y + rng.normal(0, 0.3, len(y)))
        variants[name] = dict(seed=s, fraction=f, width=128, ablate="", out="data/model/" + name)
    (model / "model.json").write_text(json.dumps(dict(seeds=[], seed_dirs=[], ablations={}, variants=variants)))
    save(model / "dev_eval.npz", dev, y)
    out = dev_constructed(str(run), str(model / "dev_eval.npz"), ranks, lib_taxid)
    m = out["methods"]
    for k in ("model_f1-w128-s1", "model_mean_f1-w128", "model_mean_f0.5-w128", "baseline_mlp_balanced", "baseline_mlp_balanced_mean",
              "baseline_mlp_balanced_seed_2", "baseline_mlp_balanced_composition_only", "baseline_mlp_balanced_hits_only", "baseline_mlp_balanced_no_lineage",
              "baseline_mlp_balanced_mean_f0.5", "baseline_mlp_proportional"):
        assert k in m, k
    assert m["model"]["mean_auroc_within_recipients"] == 1.0
    assert "model_mean_f1-w128 - baseline_mlp_balanced_mean" in out["contrasts"]
    assert "model_mean_f0.5-w128 - baseline_mlp_balanced_mean_f0.5" in out["contrasts"]
    c = out["contrasts"]["model_mean_f1-w128 - baseline_mlp_balanced_mean"]
    assert set(c) <= {"macro_auroc", "within_recipient_auroc", "recall_at_fpr_0_05"} and "macro_auroc" in c
    assert cell_name(dict(fraction=0.25, width=64, ablate="hits")) == "f0.25-w64-no-hits"
    # a contrast of a score with itself is zero with a zero interval
    v = np.arange(len(y), dtype=float) * y
    same = paired_contrast(v, v, y, np.array([r[0] for r in dev]), np.array(["bacteria"] * 8 + ["archaea"] * 8), reps=20)
    assert same["macro_auroc"]["diff"] == 0 and same["macro_auroc"]["ci95"] == [0.0, 0.0]


def test_hiding_depth_of_an_example_reaches_the_comparators():
    from ci.stages.eval import DEPTH_COL
    assert DEPTH_COL["genus"] == 4 and DEPTH_COL["order"] == 2
    from seeddrift.train.source_model import DEPTHS, RANKS
    assert all(DEPTH_COL[d] == RANKS.index(d) for d in DEPTHS)


def test_training_slots_hold_the_parallel_limit_and_run_wide_models_alone(tmp_path):
    import subprocess
    import sys
    from ci.stages.train import run_slots
    jobs = [(dict(width=w), "j%d" % i) for i, w in enumerate([128, 128, 128, 512, 128, 64, 256, 512])]

    def start(c, name):
        logp = tmp_path / (name + ".log")
        lg = open(logp, "w")
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"], stdout=lg, stderr=subprocess.STDOUT), str(logp), lg
    got = dict(run_slots(jobs, start, parallel=3, exclusive_width=512, poll=0.05))
    assert set(got) == {n for _, n in jobs} and max(got.values()) <= 2           # never more than three at once
    assert got["j3"] == 0 and got["j7"] == 0                                     # the wide ones start with nothing running

    def fail(c, name):
        logp = tmp_path / (name + ".err")
        lg = open(logp, "w")
        return subprocess.Popen([sys.executable, "-c", "import sys; print('boom'); sys.exit(2)"], stdout=lg, stderr=subprocess.STDOUT), str(logp), lg
    import pytest
    with pytest.raises(SystemExit, match="boom"):
        run_slots(jobs[:2], fail, parallel=1, poll=0.05)
