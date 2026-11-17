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
