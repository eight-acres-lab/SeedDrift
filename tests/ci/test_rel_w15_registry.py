"""Package W15: the event registry drafts (bench/rel/registry/registry.py, rules version 0.2). Standard library only."""
from __future__ import annotations

import csv
import io
import os

from bench.rel.registry import registry as reg

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REG_DIR = os.path.join(ROOT, "bench", "rel", "registry")
INSECT = "Eukaryota;Metazoa;Arthropoda;Insecta"


def _row(eid, taxon, lineage, **kw):
    r = dict.fromkeys(reg.COLUMNS, "")
    r.update(
        event_id=eid, record_kind="event", claim="c", recipient="r", recipient_group="invertebrates", region="?",
        gene_family="g", donor_claimed="d", donor_group="bacteria", direction="prok>euk", scenario_tags="",
        category="protein_coding", claims="foreign=y;integrated=y;germline=?;donor=y;timing=?",
        exclusion_taxon=taxon, exclusion_lineage=lineage, papers="10.1/%s" % eid, year="2000", db_date="?",
        assembly="?", discovery="S", ev_direct="y: pcr", ev_position="y: flanks", ev_phylogeny="y: ml 100",
        ev_origin="n", ev_population="?", position_rule="integration", grade_transfer="A", grade_donor="B",
        status="standing", p_transfer="0.85-0.99", p_donor="0.60-0.95", truth_class="transfer", recheckable="none",
        later="", full_text="y", exposure="L0", in_pretraining="?", sealed_overlap="?", provisional_cover="y",
        role="unassigned", verification="abstract", notes="")
    r.update(kw)
    return r


def test_valid_row_passes():
    errors, warnings = reg.check([_row("E1", "Drosophila (genus)", INSECT + ";Diptera;Drosophila")])
    assert errors == [] and warnings == []


def test_grade_A_needs_foreign_origin():
    # junction PCR and linkage without a supported tree or donor identity: the overturned tardigrade case
    base = dict(ev_phylogeny="?: not read")
    assert reg.derived_grade(_row("E1", "Hypsibius (genus)", "Tardigrada;Hypsibius", **base)) == "D"
    assert reg.derived_grade(_row("E1", "H (genus)", "H", ev_phylogeny="w: no support")) == "C"
    assert reg.derived_grade(_row("E1", "H (genus)", "H", ev_phylogeny="n", ev_origin="donor_identity: >99% to wRi")) == "A"
    assert reg.derived_grade(_row("E1", "H (genus)", "H", ev_direct="n", ev_phylogeny="y: 100")) == "B"


def test_recombination_test_stands_in_for_the_tree_only_in_replacement():
    rep = dict(ev_direct="?", ev_phylogeny="n", ev_origin="recombination_test: GENECONV", position_rule="replacement")
    assert reg.derived_grade(_row("E1", "Neisseria (genus)", "Neisseria", **rep)) == "B"
    chim = dict(rep, position_rule="virus_chimera", ev_direct="y: pcr across the junction")
    assert reg.derived_grade(_row("E1", "v (no rank)", "Viruses;v", **chim)) == "A"
    chim_nodirect = dict(chim, ev_direct="n")
    assert reg.derived_grade(_row("E1", "v (no rank)", "Viruses;v", **chim_nodirect)) == "C"
    ins = dict(rep, position_rule="integration")
    assert reg.derived_grade(_row("E1", "N (genus)", "N", **ins)) == "D"


def test_donor_structure_is_recorded_but_not_accepted():
    r = _row("E1", "Hominoidea (superfamily)", "Mammalia;Primates;Hominoidea", ev_phylogeny="?",
             ev_origin="donor_structure: proviral LTRs", grade_transfer="D", p_transfer="0.10-0.70",
             grade_donor="A", p_donor="0.80-0.99")
    assert reg.check([r]) == ([], [])
    assert reg.derived_grade(r, reg.ORIGIN_ACCEPTED + ("donor_structure",)) == "A"
    assert reg.summarise([r])["if_donor_structure_accepted"] == [dict(event_id="E1", grade="D", if_accepted="A")]


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


def test_vocabulary_radius_roles_and_position_rule():
    errors, _ = reg.check([
        _row("E1", "Drosophila ananassae (species)", INSECT + ";Drosophila ananassae"),
        _row("E2", "Aphis (genus)", INSECT + ";Aphis", role="pool"),
        _row("E3", "Aphis (genus)", INSECT + ";Aphis", position_rule="none"),
        _row("E4", "Aphis (genus)", INSECT),
        _row("E5", "Aphis (genus)", INSECT + ";Aphis", discovery="X", ev_origin="similarity: high"),
    ])
    assert any(e.startswith("E1: the exclusion radius is at least the genus") for e in errors)
    assert any(e.startswith("E2: role pool") for e in errors)
    assert any(e.startswith("E3: grade A needs position evidence") for e in errors)
    assert any(e.startswith("E4: exclusion_lineage") for e in errors)
    assert any(e.startswith("E5: discovery") for e in errors) and any(e.startswith("E5: ev_origin") for e in errors)
    assert reg.check([_row("E6", "Aphis (genus)", INSECT + ";Aphis", discovery="P")]) == ([], [])


def test_grade_warning_and_truth_class_warnings():
    _, warnings = reg.check([_row("E1", "Aphis (genus)", INSECT + ";Aphis", ev_direct="n: none")])
    assert warnings == ["E1: grade_transfer A, the evidence lines imply B"]
    _, warnings = reg.check([_row("E2", "Aphis (genus)", INSECT + ";Aphis", status="disputed", p_transfer="0.20-0.80",
                                  p_donor="0.10-0.70")])
    assert warnings == ["E2: disputed records are unresolved (RULES.md section 10)"]


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
    assert s["groups"] == 4 and s["records"] == 7 and s["ungrouped"] == []


def test_negatives_grade_D_and_claim_sets_stay_ungrouped():
    v = "Eukaryota;Metazoa;Chordata;Vertebrata"
    rows = [
        _row("E1", "Hominoidea (superfamily)", v + ";Mammalia;Primates;Hominoidea"),
        _row("E2", "Muroidea (superfamily)", v + ";Mammalia;Rodentia;Muroidea"),
        _row("E3", "Vertebrata (subphylum)", v, status="overturned", p_transfer="0.01-0.30", p_donor="0.01-0.30",
             truth_class="native"),
        _row("E4", "Insecta (class)", INSECT, record_kind="claim_set"),
        _row("E5", "Drosophila (genus)", INSECT + ";Drosophila"),
        _row("E6", "Aphis (genus)", INSECT + ";Aphis", grade_transfer="D", ev_phylogeny="n", p_transfer="0.10-0.70"),
    ]
    s = reg.summarise(rows)
    assert s["group_members"] == {"G01": ["E1"], "G02": ["E2"], "G03": ["E5"]}
    assert s["ungrouped"] == ["E3", "E4", "E6"]
    assert s["group_of"]["E3"] == "-"


def test_summary_counts_labels_cells_and_recheckable():
    rows = [
        _row("E1", "Drosophila (genus)", INSECT + ";Drosophila", recheckable="reads:SRR1"),
        _row("E2", "Homo (genus)", "Eukaryota;Primates;Homo", status="overturned", p_transfer="0.01-0.30",
             p_donor="0.01-0.30", truth_class="native", recipient_group="mammals"),
        _row("E3", "Aphis (genus)", INSECT + ";Aphis", grade_transfer="C", ev_position="n: none",
             p_transfer="0.40-0.90", grade_donor="-", p_donor="-", recheckable="population:panel", full_text="n"),
    ]
    assert reg.check(rows) == ([], [])
    s = reg.summarise(rows)
    assert s["per_label"]["A"] == 1 and s["per_label"]["overturned"] == 1 and s["per_label"]["C"] == 1
    assert s["cells"]["prok>euk|invertebrates"] == {"A": 1, "C": 1}
    assert s["recheckable"]["records"] == 2 and s["recheckable"]["share"] == round(2 / 3, 3)
    assert s["full_text_by_label"]["n"] == {"C": 1}
    md = reg.markdown(rows, s)
    assert "## 1. Association graph" in md and "## 3. Recheckable evidence" in md and "Ungrouped: E2" in md


def test_double_registration_agreement(tmp_path):
    p = tmp_path / "d.tsv"
    p.write_text("event_id\tfield\tfirst\tsecond\tnote\n"
                 "E1\tgrade_transfer\tA\tA\t\nE1\tstatus\tstanding\treplicated\tx\nE2\tgrade_transfer\tC\tC\t\n")
    d = reg.agreement(str(p))
    assert d["records"] == 2 and d["cells"] == 3 and d["agree"] == 2
    assert d["per_field"]["status"] == {"agree": 0, "of": 1}
    assert [x["event_id"] for x in d["differ"]] == ["E1"]
    assert "2 of 3 fields agree" in reg.markdown_double(d)


def test_committed_drafts_pass_their_checks():
    drafts = sorted(f for f in os.listdir(REG_DIR) if f.startswith("events_") and f.endswith(".tsv"))
    assert drafts
    for name in drafts:
        rows = reg.read(os.path.join(REG_DIR, name))
        errors, _ = reg.check(rows)
        assert errors == [], (name, errors)
        assert all(r["role"] == "unassigned" for r in rows)


def test_committed_double_registrations_name_registered_records():
    for name in sorted(f for f in os.listdir(REG_DIR) if f.startswith("double_") and f.endswith(".tsv")):
        d = reg.agreement(os.path.join(REG_DIR, name))
        ids = set()
        for sample in os.listdir(REG_DIR):
            if sample.startswith("events_") and sample.endswith(".tsv"):
                ids |= {r["event_id"] for r in reg.read(os.path.join(REG_DIR, sample))}
        assert {x["event_id"] for x in d["differ"]} <= ids


def test_drafts_never_sit_at_the_catalogue_path():
    assert not os.path.exists(os.path.join(ROOT, "bench", "catalogue", "events.tsv"))


def test_read_skips_comment_lines(tmp_path):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=reg.COLUMNS, delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerow(_row("E1", "Drosophila (genus)", INSECT + ";Drosophila"))
    p = tmp_path / "r.tsv"
    p.write_text("# a comment\n" + buf.getvalue())
    assert [r["event_id"] for r in reg.read(str(p))] == ["E1"]


def test_origin_n_or_unknown_may_carry_semicolons_in_its_words():
    r = _row("E1", "Aphis (genus)", INSECT + ";Aphis", ev_origin="?: resembles the donor; identity not read")
    assert reg.origin_kinds(r) == set() and reg.check([r]) == ([], [])


def test_concentration_flags_a_cell_one_source_fills():
    rows = [_row("E%d" % i, "Alloteropsis (genus)", "Poaceae;Alloteropsis", papers="10.1000/d", direction="euk>euk",
                 recipient_group="plants_algae") for i in range(4)]
    rows.append(_row("E9", "Ipomoea (genus)", "Solanales;Ipomoea", papers="10.1000/k", direction="euk>euk", recipient_group="plants_algae"))
    rows.append(_row("E8", "Ipomoea (genus)", "Solanales;Ipomoea", papers="10.1000/k", record_kind="claim_set",
                     direction="euk>euk", recipient_group="plants_algae"))
    c = reg.concentration(rows)
    cell = c["cells"]["euk>euk|plants_algae"]
    assert cell["events"] == 5 and cell["sources"] == 2 and cell["source_share"] == 0.8 and cell["over_cap"]
    assert c["once_per_source"]["euk>euk|plants_algae"] == {"A": 2} and c["claim_sets"] == {"euk>euk|plants_algae": 1}
    s = reg.summarise(rows)
    assert s["largest_group"]["A"][1] == 4
    md = reg.markdown(rows, s)
    assert "Concentration of sources" in md and "**yes**" in md and "largest group" in md


def test_xlsx_sheet_transcribes_values_and_styles_as_they_stand(tmp_path, capsys):
    import zipfile
    from bench.rel.registry import xlsx_sheet
    book = tmp_path / "b.xlsx"
    with zipfile.ZipFile(book, "w") as z:
        z.writestr("xl/sharedStrings.xml", "<sst><si><t>Gene</t></si><si><t>a&amp;b</t></si></sst>")
        z.writestr("xl/worksheets/sheet2.xml",
                   '<worksheet><sheetData><row r="1"><c r="A1" t="s" s="3"><v>0</v></c><c r="B1" s="5"><v>9*</v></c></row>'
                   '<row r="2"><c r="A2" t="s"><v>1</v></c><c r="C2" s="7"/></row></sheetData></worksheet>')
    assert list(xlsx_sheet.rows(str(book), 2)) == [("1", ["A=Gene[3]", "B=9*[5]"]), ("2", ["A=a&b[]", "C=[7]"])]
    xlsx_sheet.main(["x", str(book), "2"])
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("# b.xlsx sheet 2 sha256 ") and out[1] == "1\tA=Gene[3]\tB=9*[5]"
