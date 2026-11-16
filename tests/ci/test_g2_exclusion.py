import os

from ci.gates import g2_exclusion

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
EXC = os.path.join(FIX, "mini_exclude.tsv")


def run(entries, tax, tmp_path, panels=""):
    p = tmp_path / "panels.tsv"
    p.write_text("# comment line, as in the real table\nassembly\tspecies_taxid\trole\tpanel_of\n" + panels)
    return g2_exclusion.entries(entries, tax, exclude_path=EXC, panels_path=str(p), catalogue_path=str(tmp_path / "none.tsv"))


def test_homonym_genus_resolves_to_the_bacterium(tax):
    assert tax.genus_by_name("Bacillus") == [1386]


def test_allowed_genomes_pass(tax, tmp_path):
    r = run([{"id": "paeni", "taxid": 1406}, {"id": "aceto", "taxid": 481146}, {"id": "human", "taxid": 9606}], tax, tmp_path)
    assert r.status == "pass", r.evidence


def test_strain_under_the_development_order_fails(tax, tmp_path):
    r = run([{"id": "k12", "taxid": 83333}], tax, tmp_path)
    assert r.status == "fail" and r.evidence["hits"][0]["node"] == 91347


def test_merged_taxid_is_followed_and_fails(tax, tmp_path):
    assert run([{"id": "old", "taxid": 99999}], tax, tmp_path).status == "fail"


def test_sealed_genus_and_its_family_fail(tax, tmp_path):
    r = run([{"id": "bsub", "taxid": 1423}, {"id": "xyl", "taxid": 2371}], tax, tmp_path)
    assert r.status == "fail" and {h["id"] for h in r.evidence["hits"]} == {"bsub", "xyl"}


def test_unknown_taxid_fails_closed(tax, tmp_path):
    assert run([{"id": "ghost", "taxid": 424242}], tax, tmp_path).status == "fail"


def test_panel_assembly_and_panel_species_fail(tax, tmp_path):
    r = run([{"id": "a", "taxid": 481146, "accession": "GCF_000000001.1"}], tax, tmp_path, panels="GCA_000000001.1\t9999\tother\tNC_x\n")
    assert r.status == "fail" and r.evidence["hits"][0]["reason"] == "panel assembly"
    r = run([{"id": "b", "taxid": 481146}], tax, tmp_path, panels="GCA_000000002.1\t481146\tother\tNC_x\n")
    assert r.status == "fail" and r.evidence["hits"][0]["reason"] == "species of a panel assembly"


def test_exclusion_list_must_only_grow_fails_on_removal():
    old = open(EXC).read()
    new = "\n".join(l for l in old.splitlines() if "Bacillus" not in l) + "\n"
    assert g2_exclusion.append_only(old, new)
    assert g2_exclusion.append_only(old, old + "2026-11-17\tgenus\tVibrio\tx\n") == []


def test_generator_import_in_training_code_fails(tmp_path):
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "x.py").write_text("from bench.hard import construct\n")
    (tmp_path / "research" / "y.py").write_text("import numpy\n")
    bad = g2_exclusion.generator_imports(str(tmp_path), training=("research",))
    assert [b["file"] for b in bad] == [os.path.join("research", "x.py")]
