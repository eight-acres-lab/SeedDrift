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


def test_dev_constructed_runs_on_a_small_synthetic_run(tmp_path):
    """The development evaluation with comparators (ci/stages/eval.py dev_constructed) on two held-out genomes."""
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
             donor_gene=np.array([-1, 4, -1, 5]), stratum=np.array([-1, 5, -1, 5]), score=np.array([0.1, 0.9, 0.2, 0.8]))
    out = dev_constructed(str(run), str(dp), ranks, lib_taxid)
    assert out["examples"] == 4 and out["insertions"] == 2
    assert out["methods"]["model"]["auroc"] == 1.0
    assert set(out["methods"]) == {"model", "alien_index", "best_hit", "counts", "composition"}
    assert out["methods"]["model"]["by_stratum"]["other kingdom"]["n"] == 2
