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
    t = sorted(x for x in tax.rank if tax.rank[x] == "species")[0]
    genus = tax.at_rank(t, "genus") or t
    p = tmp_path / "extra.tsv"
    p.write_text("# provisional\ntaxid\trank\tname\tsource\n%d\tgenus\tX\ttest\n" % genus)
    plain = Excluder(tax, exclude_path=os.path.join(ROOT, "tests", "ci", "fixtures", "mini_exclude.tsv"))
    extra = Excluder(tax, exclude_path=os.path.join(ROOT, "tests", "ci", "fixtures", "mini_exclude.tsv"), extra_paths=[str(p)])
    assert extra.reason(t) and "provisional test recipient" in extra.reason(t)[0]
    assert plain.reason(t) is None or "provisional" not in plain.reason(t)[0]
    missing = Excluder(tax, exclude_path=os.path.join(ROOT, "tests", "ci", "fixtures", "mini_exclude.tsv"), extra_paths=[str(tmp_path / "nope.tsv")])
    assert any("missing" in str(pb) for pb in missing.problems)
