"""Package W15: the event registry drafts (bench/rel/registry/registry.py). Standard library only."""
from __future__ import annotations

import csv
import io
import os

from bench.rel.registry import registry as reg

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REG_DIR = os.path.join(ROOT, "bench", "rel", "registry")


def _row(eid, taxon, lineage, **kw):
    r = dict.fromkeys(reg.COLUMNS, "")
    r.update(
        event_id=eid, record_kind="event", claim="c", recipient="r", recipient_group="invertebrates", region="?",
        gene_family="g", donor_claimed="d", donor_group="bacteria", direction="prok>euk", scenario_tags="",
        category="protein_coding", claims="foreign=y;integrated=y;germline=?;donor=y;timing=?",
        exclusion_taxon=taxon, exclusion_lineage=lineage, papers="10.1/%s" % eid, year="2000", db_date="?",
        assembly="?", discovery="S", ev_direct="y: pcr", ev_position="y: flanks", ev_phylogeny="y: ml",
        ev_population="?", position_rule="integration", grade_transfer="A", grade_donor="B", status="standing",
        p_transfer="0.85-0.99", p_donor="0.60-0.95", truth_class="transfer", recheckable="none", later="",
        exposure="L0", in_pretraining="?", sealed_overlap="?", provisional_cover="y", role="unassigned",
        verification="abstract", notes="")
    r.update(kw)
    return r


INSECT = "Eukaryota;Metazoa;Arthropoda;Insecta"


def test_valid_row_passes():
    errors, warnings = reg.check([_row("E1", "Drosophila (genus)", INSECT + ";Diptera;Drosophila")])
    assert errors == [] and warnings == []


def test_intervals_come_from_the_rules_and_never_reach_0_or_1():
    bad = [_row("E1", "Drosophila (genus)", INSECT + ";Drosophila", p_transfer="0.85-1.0"),
           _row("E2", "Aphis (genus)", INSECT + ";Aphis", p_transfer="0.80-0.99")]
    errors, _ = reg.check(bad)
    assert any("E1: p_transfer" in e for e in errors)
    assert any("E2: p_transfer" in e and "0.85-0.99" in e for e in errors)
    over = _row("E3", "Homo (genus)", "Eukaryota;Primates;Homo", status="overturned", p_transfer="0.01-0.30",
                p_donor="0.01-0.30", truth_class="native")
    assert reg.check([over]) == ([], [])
    assert reg.expected_interval("B", "retracted", 0) == (0.40, 0.97)
    assert reg.expected_interval("-", "standing", 1) is None


def test_radius_roles_and_position_rule():
    errors, _ = reg.check([
        _row("E1", "Drosophila ananassae (species)", INSECT + ";Drosophila ananassae"),
        _row("E2", "Aphis (genus)", INSECT + ";Aphis", role="pool"),
        _row("E3", "Aphis (genus)", INSECT + ";Aphis", position_rule="none"),
        _row("E4", "Aphis (genus)", INSECT),
    ])
    assert any(e.startswith("E1: the exclusion radius is at least the genus") for e in errors)
    assert any(e.startswith("E2: role pool") for e in errors)
    assert any(e.startswith("E3: grade A needs position evidence") for e in errors)
    assert any(e.startswith("E4: exclusion_lineage") for e in errors)


def test_grade_warning_when_evidence_implies_another_grade():
    _, warnings = reg.check([_row("E1", "Aphis (genus)", INSECT + ";Aphis", ev_direct="n: none")])
    assert warnings == ["E1: grade_transfer A, the evidence lines imply B"]


def test_groups_join_nested_radii_shared_papers_and_assemblies():
    rows = [
        _row("E1", "Insecta (class)", INSECT),
        _row("E2", "Drosophila (genus)", INSECT + ";Diptera;Drosophila"),
        _row("E3", "Homo (genus)", "Eukaryota;Metazoa;Chordata;Primates;Homo", papers="10.1/x"),
        _row("E4", "Mus (genus)", "Eukaryota;Metazoa;Chordata;Rodentia;Mus", papers="10.1/X"),
        _row("E5", "Thermotoga (genus)", "Bacteria;Thermotogota;Thermotoga", assembly="GCF_1"),
        _row("E6", "Legionella (genus)", "Bacteria;Pseudomonadota;Legionella", assembly="gcf_1"),
        _row("E7", "Neisseria (genus)", "Bacteria;Pseudomonadota;Neisseria"),
    ]
    comps = [[rows[i]["event_id"] for i in m] for m in reg.groups(rows)]
    assert comps == [["E1", "E2"], ["E3", "E4"], ["E5", "E6"], ["E7"]]
    s = reg.summarise(rows)
    assert s["groups"] == 4 and s["records"] == 7


def test_summary_counts_labels_cells_and_recheckable():
    rows = [
        _row("E1", "Drosophila (genus)", INSECT + ";Drosophila", recheckable="reads:SRR1"),
        _row("E2", "Homo (genus)", "Eukaryota;Primates;Homo", status="overturned", p_transfer="0.01-0.30",
             p_donor="0.01-0.30", truth_class="native", recipient_group="mammals"),
        _row("E3", "Aphis (genus)", INSECT + ";Aphis", grade_transfer="C", ev_position="n: none",
             p_transfer="0.40-0.90", grade_donor="-", p_donor="-", recheckable="population:panel"),
    ]
    assert reg.check(rows) == ([], [])
    s = reg.summarise(rows)
    assert s["per_label"]["A"] == 1 and s["per_label"]["overturned"] == 1 and s["per_label"]["C"] == 1
    assert s["cells"]["prok>euk|invertebrates"] == {"A": 1, "C": 1}
    assert s["recheckable"]["records"] == 2 and s["recheckable"]["share"] == round(2 / 3, 3)
    md = reg.markdown(rows, s)
    assert "## 1. Association graph" in md and "## 3. Recheckable evidence" in md


def test_committed_drafts_pass_their_checks():
    drafts = sorted(f for f in os.listdir(REG_DIR) if f.startswith("events_") and f.endswith(".tsv"))
    for name in drafts:
        rows = reg.read(os.path.join(REG_DIR, name))
        errors, _ = reg.check(rows)
        assert errors == [], (name, errors)
        assert all(r["role"] == "unassigned" for r in rows)


def test_drafts_never_sit_at_the_catalogue_path():
    assert not os.path.exists(os.path.join(ROOT, "bench", "catalogue", "events.tsv"))


def test_read_skips_comment_lines(tmp_path):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=reg.COLUMNS, delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerow(_row("E1", "Drosophila (genus)", INSECT + ";Drosophila"))
    p = tmp_path / "r.tsv"
    p.write_text("# a comment\n" + buf.getvalue())
    rows = reg.read(str(p))
    assert [r["event_id"] for r in rows] == ["E1"]


def test_double_registration_agreement(tmp_path):
    p = tmp_path / "d.tsv"
    p.write_text("event_id\tfield\tfirst\tsecond\tnote\n"
                 "E1\tgrade_transfer\tA\tA\t\nE1\tstatus\tstanding\treplicated\tx\nE2\tgrade_transfer\tC\tC\t\n")
    d = reg.agreement(str(p))
    assert d["records"] == 2 and d["cells"] == 3 and d["agree"] == 2
    assert d["per_field"]["status"] == {"agree": 0, "of": 1}
    assert [x["event_id"] for x in d["differ"]] == ["E1"]
    assert "2 of 3 fields agree" in reg.markdown_double(d)


def test_committed_double_registrations_name_registered_records():
    for name in sorted(f for f in os.listdir(REG_DIR) if f.startswith("double_") and f.endswith(".tsv")):
        d = reg.agreement(os.path.join(REG_DIR, name))
        sample = name.replace("double_", "events_")
        ids = {r["event_id"] for r in reg.read(os.path.join(REG_DIR, sample))}
        assert {x["event_id"] for x in d["differ"]} <= ids
