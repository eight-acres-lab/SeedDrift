"""Pilot pieces that run without the GPU: alignment tracks and codon tokens of the train_data stage, and the extra
(provisional) exclusion list of pilot runs."""
import os

import pytest

from conftest import ROOT


def test_track_follows_the_cigar_residue_by_residue():
    from ci.stages.train_data import AA, GAP, track
    # 3M: three aligned residues; 2I: two query residues without a target residue; 1M; 1D: a target residue without a
    # query residue (left out); 2M
    t = list(track("3M2I1M1D2M", "ACDEFGH"))
    assert t == [AA["A"], AA["C"], AA["D"], GAP, GAP, AA["E"], AA["G"], AA["H"]]
    with pytest.raises(SystemExit):
        track("3M", "ACDE")                                            # the CIGAR must consume the aligned target exactly


def test_codon_tokens():
    from ci.stages.train_data import codon_tokens
    t = list(codon_tokens("ATGAAANNNTT", 10))
    assert t == [16 * 0 + 4 * 3 + 2 + 1, 1, 65]                         # ATG, AAA, an ambiguous codon; the partial codon is cut
    assert len(codon_tokens("ATG" * 50, 10)) == 10


def test_extra_exclusions_add_to_the_rules_fails(tmp_path, tax):
    from ci.gates.g2_exclusion import Excluder
    plain = Excluder(tax, exclude_path=os.path.join(ROOT, "tests", "ci", "fixtures", "mini_exclude.tsv"))
    t = [x for x in sorted(tax.rank) if tax.rank[x] == "species" and tax.at_rank(x, "genus") and plain.reason(x) is None][0]
    p = tmp_path / "extra.tsv"
    p.write_text("# provisional\ntaxid\trank\tname\tsource\n%d\tgenus\tX\ttest\n" % tax.at_rank(t, "genus"))
    extra = Excluder(tax, exclude_path=os.path.join(ROOT, "tests", "ci", "fixtures", "mini_exclude.tsv"), extra_paths=[str(p)])
    assert extra.reason(t) and "provisional test recipient" in extra.reason(t)[0]       # allowed before, excluded with the list
    missing = Excluder(tax, exclude_path=os.path.join(ROOT, "tests", "ci", "fixtures", "mini_exclude.tsv"), extra_paths=[str(tmp_path / "nope.tsv")])
    assert any("missing" in str(pb) for pb in missing.problems)


def test_split_keeps_every_genus_on_one_side():
    from ci.stages.train_data import split
    ids = ["g%d" % i for i in range(40)]
    groups = {e: ("bacteria" if i % 2 else "archaea") for i, e in enumerate(ids)}
    lin = {e: dict(genus=(i // 3) * 10 if i % 7 else 0, species=1000 + i) for i, e in enumerate(ids)}   # genera of three, some without genus
    sp = split(ids, groups, lin)
    for gr in ("bacteria", "archaea"):
        side = {}
        for e in ids:
            if groups[e] == gr and lin[e]["genus"]:
                side.setdefault(lin[e]["genus"], set()).add(sp[e])
        assert all(len(v) == 1 for v in side.values())
        assert "train" in {sp[e] for e in ids if groups[e] == gr} and "heldout" in {sp[e] for e in ids if groups[e] == gr}


def test_split_keeps_clusters_and_genera_together():
    from ci.stages.train_data import split
    ids = ["a%d" % i for i in range(30)]
    groups = {e: "bacteria" for e in ids}
    lin = {e: dict(genus=100 + i, species=1000 + i) for i, e in enumerate(ids)}
    cluster = {}
    for c in range(5):                                             # five clusters of an anchor and three relatives
        members = ids[4 * c:4 * c + 4]
        for e in members:
            cluster[e] = members[0]
        lin[members[1]]["genus"] = lin[members[0]]["genus"]         # the same-genus relative
    lin[ids[25]]["genus"] = lin[ids[2]]["genus"]                    # a plain genome sharing a relative's genus joins the cluster
    sp = split(ids, groups, lin, cluster)
    for c in range(5):
        assert len({sp[e] for e in ids[4 * c:4 * c + 4]}) == 1
    assert sp[ids[25]] == sp[ids[2]]
    assert {"train", "heldout"} == set(sp.values())


def _genome(S, gid, taxid, ranks, n=3):
    """A Genome of n genes, each with one hit to each of the three proteomes of the synthetic library."""
    import numpy as np
    g = object.__new__(S.Genome)
    g.meta = dict(id=gid, group="bacteria", taxid=taxid, ranks=ranks)
    g.codons, g.coff = np.full(n * 5, 2, np.uint8), np.arange(0, n * 5 + 1, 5)
    g.hoff, g.hprot, g.hqs = np.arange(0, n * 3 + 1, 3), np.tile([0, 1, 2], n).astype(np.int32), np.zeros(n * 3, np.int32)
    g.track, g.toff = np.ones(n * 3 * 4, np.uint8), np.arange(0, n * 3 * 4 + 1, 4)
    g.n, g.order, g.pos, g.rep = n, np.arange(n), np.arange(n), np.zeros(n, np.int32)
    g.start = np.arange(n) * 100 + 1
    g.end, g.strand, g.rep_len, g.ranks = g.start + 90, np.ones(n, np.int8), np.array([1000]), np.array(ranks)
    return g


def test_hiding_is_symmetric_for_natives_and_replacements():
    """A native with a decoy partner and a replacement from that partner lose the same clades from every slot."""
    import numpy as np
    from seeddrift.train import source_model as S
    d = object.__new__(S.Data)
    d.lib_ranks = np.array([[1, 0, 0, 0, 100, 10], [1, 0, 0, 0, 200, 20], [2, 0, 0, 0, 300, 30]])
    d.lib_taxid = np.array([10, 20, 30])
    d.lib_cols = np.concatenate([d.lib_ranks, d.lib_taxid[:, None]], 1)
    d.n_lib, d.maxc, d.kc, d.kn, d.maxt = 3, 512, 32, 4, 512
    A = _genome(S, "A", 10, [1, 0, 0, 0, 100, 10])
    B = _genome(S, "B", 30, [2, 0, 0, 0, 300, 30])
    native = d.window(A, 1, None, "genus", partner=B)
    repl = d.window(A, 1, (B, 0), "genus")
    for w in (native, repl):
        kept = [sorted(sl["prots"]) for sl in w if sl]
        assert kept and all(k == [1] for k in kept)                # A's own and B's genus hidden in every slot
    plain = d.window(A, 1, None, "genus")
    assert sorted(plain[S.WIN]["prots"]) == [1, 2]                 # without a partner only A's own clade goes


def test_near_relatives_take_new_genera_and_skip_shared_anchors():
    from ci.stages.select import near_relatives

    def row(i, genus, family, order, species):
        return dict(id="r%d" % i, genus=genus, family=family, order=order, species_taxid=species, units=1)
    a1, a2, a3 = row(1, 10, 100, 1000, 1), row(2, 20, 200, 2000, 2), row(3, 30, 200, 2000, 3)
    shared = row(4, 30, 200, 2000, 4)                              # a3 shares its genus with another selected genome
    same_genus = row(5, 10, 100, 1000, 5)
    used_genus = row(6, 30, 100, 1000, 6)                          # same family as a1, but genus 30 is selected already
    same_family = row(7, 11, 100, 1000, 7)
    same_order = row(8, 12, 101, 1000, 8)
    chosen = {"bacteria": ([(0, a1), (0, a2), (0, a3), (1, shared)], 4)}
    orders = {"bacteria": [(0, x) for x in (a1, a2, a3, shared, same_genus, used_genus, same_family, same_order)]}
    out = near_relatives(chosen, orders, dict(anchors={"bacteria": 3}))
    got = [(a["id"], rel, r["id"]) for a, rel, r in out["bacteria"]]
    assert ("r1", "genus", "r5") in got and ("r1", "family", "r7") in got and ("r1", "order", "r8") in got
    assert not any(a == "r3" for a, _, _ in got)                   # an anchor whose genus is shared is not used
    assert not any(r == "r6" for _, _, r in got)


def test_search_reuse_per_genome(tmp_path, monkeypatch):
    """Genomes with unchanged inputs take an earlier run's hit file; the others are searched; the check genome is
    searched again and must come out byte for byte the same."""
    import json as _json
    import shutil
    import sys as _sys
    import numpy as np
    import yaml as _yaml
    from ci import manifest as M
    from ci.stages import search, search_or_reuse as R
    from seeddrift.train.logger import save_npz

    def fake_search(run):
        for gid, folder in search.genomes(run, None):
            save_npz(os.path.join(run, "data", "hits", gid + ".npz"), x=np.frombuffer(open(os.path.join(folder, "proteins.faa"), "rb").read(), np.uint8))
        with open(os.path.join(run, "stages", "inner.json"), "w") as f:
            f.write(_json.dumps(sorted(g for g, _ in search.genomes(run, None))))

    def make(name, genomes):
        run = tmp_path / "runs" / name
        for d in ("gates", "stages", "data/hits", "data/reference", "data/eval_inputs"):
            (run / d).mkdir(parents=True, exist_ok=True)
        for g in ("pre", "corpus"):
            (run / "gates" / (g + ".ok")).write_text("")
        ents = []
        for gid, prot in genomes:
            d = run / "data" / "corpus" / gid
            d.mkdir(parents=True)
            (d / "genes.tsv").write_text("gene_id\n" + "".join("g%d\n" % i for i in range(40)))
            (d / "proteins.faa").write_text(prot)
            (d / "cds.fna").write_text(prot.lower())
            ents.append(dict(id=gid, files=[M.file_entry(str(d / n), str(run)) for n in ("genes.tsv", "proteins.faa", "cds.fna")]))
        (run / "data" / "corpus" / "manifest.json").write_text(_json.dumps(dict(sha256="c-" + name, entries=ents)))
        (run / "data" / "eval_inputs" / "manifest.json").write_text(_json.dumps(dict(sha256="e", entries=[])))
        (run / "data" / "reference" / "manifest.json").write_text(_json.dumps(dict(sha256="ref")))
        (run / "record.json").write_text(_json.dumps(dict(env_snapshot={"diamond_version.txt": "2.1"})))
        cfg = dict(search=dict(sensitivity="sensitive", nearest=32, reuse_from=[str(tmp_path / "runs" / "old")], reuse_check=1))
        (run / "config.resolved.yaml").write_text(_yaml.safe_dump(cfg))
        code = tmp_path / "wt" / name / "ci" / "stages"
        code.mkdir(parents=True)
        shutil.copy(R.SEARCH_CODE, code / "search.py")
        return str(run)
    old = make("old", [("W", ">W__g\nMKK\n"), ("X", ">X__g\nMKV\n"), ("Z", ">Z__g\nMAA\n")])
    fake_search(old)
    ents = [dict(id=g, files=[M.file_entry(os.path.join(old, "data", "hits", g + ".npz"), old)]) for g in ("W", "X", "Z")]
    (tmp_path / "runs" / "old" / "data" / "hits" / "manifest.json").write_text(_json.dumps(dict(sha256="h", **M.build("hits", "hits", ents))))
    (tmp_path / "runs" / "old" / "stages" / "search.json").write_text(_json.dumps(dict(status="passed")))
    new = make("new", [("W", ">W__g\nMKK\n"), ("X", ">X__g\nMKV\n"), ("Y", ">Y__g\nMLL\n"), ("Z", ">Z__g\nMAW\n")])   # Z changed, Y new
    monkeypatch.setattr(search, "main", lambda: fake_search(new))
    monkeypatch.setattr(_sys, "argv", ["x", new])
    R.main()
    rep = _json.loads(open(os.path.join(new, "stages", "search.json")).read())
    assert rep["status"] == "passed" and rep["reused_per_genome"] == {"old": 1} and rep["reuse_checked"] == ["W"]
    assert _json.loads(open(os.path.join(new, "stages", "inner.json")).read()) == ["W", "Y", "Z"]   # X linked, W searched again
    assert os.stat(os.path.join(new, "data", "hits", "X.npz")).st_ino == os.stat(os.path.join(old, "data", "hits", "X.npz")).st_ino
    man = _json.loads(open(os.path.join(new, "data", "hits", "manifest.json")).read())
    assert [e["id"] for e in man["entries"]] == ["W", "X", "Y", "Z"]
    # a check genome that comes out different stops the stage
    os.remove(os.path.join(old, "data", "hits", "W.npz"))
    save_npz(os.path.join(old, "data", "hits", "W.npz"), x=np.zeros(3))
    hm = _json.loads(open(os.path.join(old, "data", "hits", "manifest.json")).read())
    hm["entries"][0]["files"][0]["sha256"] = M.file_entry(os.path.join(old, "data", "hits", "W.npz"), old)["sha256"]
    (tmp_path / "runs" / "old" / "data" / "hits" / "manifest.json").write_text(_json.dumps(hm))
    with pytest.raises(SystemExit, match="reuse check"):
        R.main()


def test_dev_constructed_runs_on_a_small_synthetic_run(tmp_path):
    """The development evaluation with comparators (ci/stages/eval.py dev_constructed) on two held-out genomes; natives
    carry a decoy partner of the same stratum as the insertions."""
    import json as _json
    import numpy as np
    from ci.stages.eval import dev_constructed
    run = tmp_path
    (run / "data" / "train_data").mkdir(parents=True)
    (run / "data" / "hits").mkdir(parents=True)
    lib_taxid = np.array([10, 20, 30, 40])
    ranks = np.array([[1, 0, 0, 0, 100, 10], [1, 0, 0, 0, 200, 20], [2, 0, 0, 0, 300, 30], [2, 0, 0, 0, 400, 40]])
    genomes = []
    rng = np.random.default_rng(0)
    for gid, tx, rk, grp in (("A", 10, [1, 0, 0, 0, 100, 10], "bacteria"), ("B", 30, [2, 0, 0, 0, 300, 30], "archaea")):
        d = run / "data" / "corpus" / gid
        d.mkdir(parents=True)
        genes = ["g%d" % i for i in range(6)]
        (d / "genes.tsv").write_text("gene_id\treplicon\tstart\tend\tstrand\n" + "".join("%s\tr\t%d\t%d\t+\n" % (g, 100 * i + 1, 100 * i + 90) for i, g in enumerate(genes)))
        (d / "cds.fna").write_text("".join(">%s__%s\n%s\n" % (gid, g, "".join(rng.choice(list("ACGT"), 90))) for g in genes))
        off = np.arange(0, 6 * 3 + 1, 3)
        prot = np.tile([0, 1, 2], 6)
        so = np.arange(0, 6 * 3 + 1) * 5
        np.savez(run / "data" / "hits" / (gid + ".npz"), off=off, prot=prot, pid=np.full(18, 800, np.uint16), bits=rng.random(18).astype(np.float32) * 100, sseq_off=so)
        genomes.append(dict(id=gid, group=grp, taxid=tx, ranks=rk, split="heldout", genes=6))
    (run / "data" / "train_data" / "index.json").write_text(_json.dumps(dict(genomes=genomes)))
    dp = run / "dev.npz"
    np.savez(dp, recipient=np.array([b"A", b"A", b"B", b"B"]), gene=np.array([0, 1, 2, 3]), donor=np.array([b"", b"B", b"", b"A"]),
             donor_gene=np.array([-1, 4, -1, 5]), stratum=np.array([5, 5, 5, 5]), partner=np.array([b"B", b"B", b"A", b"A"]),
             score=np.array([0.1, 0.9, 0.2, 0.8]))
    (run / "data" / "model").mkdir()
    np.savez(run / "data" / "model" / "baseline_train.npz",
             recipient=np.array([b"A", b"A", b"B", b"B", b"A", b"B"]), gene=np.array([2, 3, 4, 5, 1, 0]), donor=np.array([b"", b"B", b"", b"A", b"B", b""]),
             donor_gene=np.array([-1, 1, -1, 2, 3, -1]), stratum=np.array([5] * 6), partner=np.array([b"B", b"B", b"A", b"A", b"B", b"A"]))
    out = dev_constructed(str(run), str(dp), ranks, lib_taxid)
    assert out["examples"] == 4 and out["insertions"] == 2
    assert out["methods"]["model"]["auroc"] == 1.0
    assert set(out["methods"]) == {"model", "baseline_logistic", "baseline_logistic_composition_only", "alien_index", "best_hit", "counts", "composition",
                                   "gc_difference"}
    st = out["methods"]["model"]["by_stratum"]["other kingdom"]
    assert st["n_pos"] == 2 and st["n_neg"] == 2
    assert len(out["methods"]["model"]["auroc_ci95"]) == 2 and out["methods"]["model"]["macro_auroc_over_groups"] == 1.0
