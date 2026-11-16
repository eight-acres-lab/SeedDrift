import os

import pytest
import yaml

from ci.gates import g1_kingdom
from ci.kingdom import KingdomMap, NoGroup

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALL = {g: {"min_genomes": 1} for g in KingdomMap().names}


def spec(**over):
    s = {"name": "t", "tier": "tiny", "groups": dict(ALL)}
    s.update(over)
    return s


def test_groups_come_from_lineage(tax):
    km = KingdomMap()
    assert km.group(tax, 83333) == "bacteria"
    assert km.group(tax, 2173) == "archaea"
    assert km.group(tax, 9606) == "eukaryota.mammals"
    assert km.group(tax, 7955) == "eukaryota.other_vertebrates"
    assert km.group(tax, 7227) == "eukaryota.invertebrates"
    assert km.group(tax, 3702) == "eukaryota.plants_algae"
    assert km.group(tax, 4932) == "eukaryota.fungi"
    assert km.group(tax, 44689) == "eukaryota.protists"
    assert km.group(tax, 10710) == "viruses.phage"
    assert km.group(tax, 212035) == "viruses.giant"
    assert km.group(tax, 11320) == "viruses.eukaryotic"
    assert km.group(tax, 481146, "plasmid") == "mobile.plasmid"


def test_unknown_taxid_fails_closed(tax):
    with pytest.raises(Exception):
        KingdomMap().group(tax, 424242)


def test_static_pass_with_every_group():
    assert g1_kingdom.static(spec()).status == "pass"


def test_static_fail_when_a_group_is_missing(tmp_path):
    groups = dict(ALL)
    del groups["archaea"]
    r = g1_kingdom.static(spec(groups=groups), waiver_dir=str(tmp_path))
    assert r.status == "fail" and r.evidence["problems"][0]["group"] == "archaea"


def test_static_full_tier_fails_below_the_floors(tmp_path):
    r = g1_kingdom.static(spec(tier="full"), waiver_dir=str(tmp_path))
    assert r.status == "fail"
    floors = yaml.safe_load(open(os.path.join(ROOT, "configs", "corpus", "floors.yaml")))
    full = {g: {"min_genomes": n} for g, n in floors["min_genomes"].items()}
    ok = g1_kingdom.static(spec(tier="full", groups=full, protein_share=floors["protein_share"]), waiver_dir=str(tmp_path))
    assert ok.status == "pass"


def test_waiver_lets_a_missing_group_pass(tmp_path):
    (tmp_path / "w.yml").write_text("gate: G1\nscope: t:archaea\nreason: test\napproved_by: test\ndate: 2026-11-17\nexpires: 2999-01-01\n")
    groups = dict(ALL)
    del groups["archaea"]
    assert g1_kingdom.static(spec(groups=groups), waiver_dir=str(tmp_path)).status == "waived"


def test_expired_waiver_fails(tmp_path):
    from ci.gates import GateFailure
    (tmp_path / "w.yml").write_text("gate: G1\nscope: t:archaea\nreason: test\napproved_by: test\ndate: 2020-01-01\nexpires: 2020-02-01\n")
    groups = dict(ALL)
    del groups["archaea"]
    with pytest.raises(GateFailure):
        g1_kingdom.static(spec(groups=groups), waiver_dir=str(tmp_path))


def entries():
    taxa = {"bacteria": (83333, ""), "archaea": (2173, ""), "eukaryota.mammals": (9606, ""), "eukaryota.other_vertebrates": (7955, ""),
            "eukaryota.invertebrates": (7227, ""), "eukaryota.plants_algae": (3702, ""), "eukaryota.fungi": (4932, ""), "eukaryota.protists": (44689, ""),
            "viruses.phage": (10710, ""), "viruses.eukaryotic": (11320, ""), "viruses.giant": (212035, ""), "mobile.plasmid": (481146, "plasmid")}
    return [{"id": g, "taxid": t, "source_kind": k, "proteins": 10} for g, (t, k) in taxa.items()]


def test_built_pass(tax):
    assert g1_kingdom.built(spec(), entries(), tax).status == "pass"


def test_built_fail_without_eukaryotes(tax, tmp_path):
    es = [e for e in entries() if not e["id"].startswith("eukaryota")]
    r = g1_kingdom.built(spec(), es, tax, waiver_dir=str(tmp_path))
    assert r.status == "fail" and {p["group"] for p in r.evidence["problems"]} >= {"eukaryota.mammals", "eukaryota.fungi"}


def test_built_fail_on_unknown_taxid(tax, tmp_path):
    es = entries() + [{"id": "x", "taxid": 424242, "proteins": 1}]
    assert g1_kingdom.built(spec(), es, tax, waiver_dir=str(tmp_path)).status == "fail"
