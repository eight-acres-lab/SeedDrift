"""The partner's clade hidden at its own depth (train.partner_depths): drawn below the rank the partner shares with the
recipient, the same rule for natives (decoy) and replacements (donor), recorded with each constructed example and read
back by the comparators of the eval stage."""
import json

import numpy as np

# proteomes of the synthetic library: (phylum, class, order, family, genus, species), taxid
LIB = [([1, 11, 111, 1111, 100, 10], 10),    # 0 A itself
       ([1, 11, 111, 1111, 100, 15], 15),    # 1 A's genus
       ([1, 11, 111, 1111, 101, 16], 16),    # 2 A's family, another genus
       ([1, 12, 121, 1211, 300, 30], 30),    # 3 B itself (same phylum as A, another class)
       ([1, 12, 121, 1211, 300, 35], 35),    # 4 B's genus
       ([1, 12, 121, 1222, 310, 36], 36),    # 5 B's order, another family
       ([1, 12, 122, 1233, 320, 37], 37),    # 6 B's class, another order
       ([2, 21, 211, 2111, 400, 40], 40)]    # 7 another phylum
A_RANKS, B_RANKS = LIB[0][0], LIB[3][0]


def _data(S, genomes=()):
    d = object.__new__(S.Data)
    d.lib_ranks = np.array([r for r, _ in LIB])
    d.lib_taxid = np.array([t for _, t in LIB])
    d.lib_cols = np.concatenate([d.lib_ranks, d.lib_taxid[:, None]], 1)
    d.n_lib, d.maxc, d.kc, d.kn, d.maxt, d.comp = len(LIB), 512, 32, 4, 512, False
    d.g = {g.meta["id"]: g for g in genomes}
    d.by_split = {}
    for g in genomes:
        d.by_split.setdefault(g.meta["split"], []).append(g.meta["id"])
    return d


def _genome(S, gid, ranks, n=4, split="heldout", group="bacteria"):
    """n genes, each with one aligned hit to every proteome of the library, nearest first."""
    g = object.__new__(S.Genome)
    g.meta = dict(id=gid, group=group, taxid=ranks[5], ranks=ranks, split=split, cluster="")
    g.ranks = np.array(ranks)
    k = len(LIB)
    g.n = n
    g.codons, g.coff = np.full(n * 5, 2, np.uint8), np.arange(0, n * 5 + 1, 5)
    g.ccount = np.bincount(g.codons.astype(np.int64), minlength=66)[:66]
    g.hoff, g.hprot, g.hqs = np.arange(0, n * k + 1, k), np.tile(np.arange(k), n).astype(np.int32), np.zeros(n * k, np.int32)
    g.track, g.toff = np.ones(n * k * 4, np.uint8), np.arange(0, n * k * 4 + 1, 4)
    g.order, g.pos, g.rep = np.arange(n), np.arange(n), np.zeros(n, np.int32)
    g.start = np.arange(n) * 100 + 1
    g.end, g.strand, g.rep_len = g.start + 90, np.ones(n, np.int8), np.array([1000])
    return g


def test_partner_depth_lies_below_the_shared_rank_and_draws_nothing_when_unset():
    from seeddrift.train import source_model as S
    A, B = _genome(S, "A", A_RANKS), _genome(S, "B", B_RANKS)
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    assert S.partner_depth(A, B, (), rng) is None and rng.bit_generator.state == state           # no draw
    assert S.partner_depth(A, B, ("phylum",), rng) is None and rng.bit_generator.state == state   # A and B share the phylum
    got = {S.partner_depth(A, B, S.PARTNER_DEPTHS, rng) for _ in range(200)}
    assert got == {"species", "genus", "family", "order", "class"}                                # never the shared phylum
    same_genus = _genome(S, "C", [1, 11, 111, 1111, 100, 19])
    assert {S.partner_depth(A, same_genus, S.PARTNER_DEPTHS, rng) for _ in range(50)} == {"species"}
    # a gap in the recipient's lineage (no class, order, family): the partner's clade there may contain it, so only
    # the partner's genus and species are hidden; below a rank where both are defined and differ, everything may go
    gappy = _genome(S, "G", [1, 0, 0, 0, 100, 10])
    assert {S.partner_depth(gappy, B, S.PARTNER_DEPTHS, rng) for _ in range(100)} == {"species", "genus"}
    far = _genome(S, "F", [2, 21, 211, 2111, 400, 40])
    assert {S.partner_depth(gappy, far, S.PARTNER_DEPTHS, rng) for _ in range(300)} == set(S.PARTNER_DEPTHS)


def test_native_and_replacement_lose_the_partner_clade_at_the_partner_depth():
    from seeddrift.train import source_model as S
    A, B = _genome(S, "A", A_RANKS), _genome(S, "B", B_RANKS)
    d = _data(S, [A, B])
    for pd, kept in ((None, [2, 5, 6, 7]), ("genus", [2, 5, 6, 7]), ("family", [2, 5, 6, 7]), ("order", [2, 6, 7]), ("class", [2, 7])):
        native = d.window(A, 1, None, "genus", partner=B, partner_depth=pd)
        repl = d.window(A, 1, (B, 0), "genus", partner_depth=pd)
        for w in (native, repl):
            assert sorted(w[S.WIN]["prots"]) == kept, (pd, w[S.WIN]["prots"])                  # A's genus and B's clade gone
            assert all(sorted(sl["prots"]) == kept for sl in w if sl)                          # in every slot


def test_constructed_examples_record_the_partner_depth():
    from seeddrift.train import source_model as S
    gs = [_genome(S, "A", A_RANKS, n=30), _genome(S, "B", B_RANKS, n=30), _genome(S, "Z", LIB[7][0], n=30, group="archaea")]
    d = _data(S, gs)
    plain = S.constructed(d, 0, "heldout", n_per=20)
    assert all(len(e) == 8 and e[7] == e[6] == "genus" for e in plain)                      # unset: the example's depth
    deep = S.constructed(d, 0, "heldout", n_per=20, partner_depths=S.PARTNER_DEPTHS)
    ranks = {g.meta["id"]: g.meta["ranks"] for g in gs}
    for e in deep:
        shared = max([c for c in range(6) if ranks[e[0]][c] == ranks[e[5]][c]] or [-1])
        assert S.RANKS.index(e[7]) > shared                                                  # never the recipient's own clade
    assert len({e[7] for e in deep}) > 2
    assert S.constructed(d, 0, "heldout", n_per=20, partner_depths=S.PARTNER_DEPTHS) == deep   # fixed by the seed
    d.partner_depths = ("class", "phylum")
    assert {e[7] for e in S.dev_set(d, 0, n_per=20)} <= {"class", "phylum", "genus"}         # genus: nothing below the shared rank


def test_eval_comparators_hide_the_partner_at_the_recorded_depth(tmp_path):
    from ci.stages.eval import dev_constructed
    run = tmp_path
    (run / "data" / "train_data").mkdir(parents=True)
    (run / "data" / "hits").mkdir(parents=True)
    rng = np.random.default_rng(0)
    genomes = []
    for gid, rk, grp in (("A", A_RANKS, "bacteria"), ("B", B_RANKS, "bacteria")):
        d = run / "data" / "corpus" / gid
        d.mkdir(parents=True)
        genes = ["g%d" % i for i in range(6)]
        (d / "genes.tsv").write_text("gene_id\treplicon\tstart\tend\tstrand\n" + "".join("%s\tr\t%d\t%d\t+\n" % (g, 100 * i + 1, 100 * i + 90) for i, g in enumerate(genes)))
        (d / "cds.fna").write_text("".join(">%s__%s\n%s\n" % (gid, g, "".join(rng.choice(list("ACGT"), 90))) for g in genes))
        k = len(LIB)
        np.savez(run / "data" / "hits" / (gid + ".npz"), off=np.arange(0, 6 * k + 1, k), prot=np.tile(np.arange(k), 6), pid=np.full(6 * k, 800, np.uint16),
                 bits=rng.random(6 * k).astype(np.float32) * 100, sseq_off=np.arange(0, 6 * k + 1) * 5)
        genomes.append(dict(id=gid, group=grp, taxid=rk[5], ranks=rk, split="heldout", genes=6))
    (run / "data" / "train_data" / "index.json").write_text(json.dumps(dict(genomes=genomes)))
    ranks = np.array([r for r, _ in LIB])
    lib_taxid = np.array([t for _, t in LIB])
    base = dict(recipient=np.array([b"A", b"A", b"B", b"B"]), gene=np.array([0, 1, 2, 3]), donor=np.array([b"", b"B", b"", b"A"]),
                donor_gene=np.array([-1, 4, -1, 5]), stratum=np.array([3, 3, 3, 3]), partner=np.array([b"B", b"B", b"A", b"A"]),
                depth=np.array([b"genus"] * 4), score=np.array([0.1, 0.9, 0.2, 0.8]))
    vis = {}
    for pd in (None, "genus", "class"):
        p = run / ("dev_%s.npz" % pd)
        np.savez(p, **(base if pd is None else dict(base, partner_depth=np.array([pd.encode()] * 4))))
        vis[pd] = dev_constructed(str(run), str(p), ranks, lib_taxid, k_hits=16)["balance"]["rows_visible"]
    assert vis[None] == vis["genus"]                                         # no column: the example's depth
    # genus: the recipient's genus and the partner's genus hidden, 4 of 8 visible for every example; class: for
    # recipient A the partner B's class goes too (2 left), for recipient B the partner A's class (3 left)
    assert vis["genus"]["mean_natives"] == 4 and vis["genus"]["mean_insertions"] == 4
    assert vis["class"]["mean_natives"] == 2.5 and vis["class"]["mean_insertions"] == 2.5


def test_saved_examples_load_back_as_constructed(tmp_path):
    from seeddrift.train import source_model as S
    gs = [_genome(S, "A", A_RANKS, n=12), _genome(S, "B", B_RANKS, n=12), _genome(S, "Z", LIB[7][0], n=12, group="archaea")]
    d = _data(S, gs)
    d.partner_depths = ("genus", "class", "phylum")
    ex = S.dev_set(d, 3, n_per=10)
    p = tmp_path / "dev_set.npz"
    S.save_examples(str(p), ex)
    assert S.load_examples(str(p)) == ex
    (tmp_path / "data" / "model").mkdir(parents=True)
    S.save_examples(str(tmp_path / "data" / "model" / "dev_set.npz"), ex)
    d.partner_depths = ()                                    # a control variant with its own construction ...
    assert S.run_dev_set(str(tmp_path), d, dict(dev_seed=3)) == ex   # ... is scored on the run's development set


def test_cell_names_keep_control_constructions_apart():
    from ci.stages.eval import cell_name
    assert cell_name(dict(fraction=1.0, width=128)) == "f1-w128"
    assert cell_name(dict(fraction=1.0, width=128, partner_depths=[])) == "f1-w128-pd-recipient"
    assert cell_name(dict(fraction=0.5, width=64, partner_depths=["genus", "class"], ablate="hits")) == "f0.5-w64-pd-genus_class-no-hits"
