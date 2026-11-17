"""Pipeline select: catalogue snapshots, the corpus selection, its gates and its budget, on the committed fixtures
(tests/ci/fixtures/select, written by make_select.py from the tiny taxdump)."""
import csv
import json
import os
import subprocess
import sys

import pytest
import yaml

from conftest import ROOT

TINY_TAX = os.path.join(ROOT, "tests", "ci", "fixtures", "tiny", "taxdump")


@pytest.fixture(scope="module")
def ttax():
    from ci.taxonomy import Taxonomy
    return Taxonomy(TINY_TAX)


def _stage(name, run_dir, *args):
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONHASHSEED="0")
    return subprocess.run([sys.executable, "-m", "ci.stages." + name, str(run_dir), *args], cwd=ROOT, env=env, capture_output=True, text=True)


def _run_dir(tmp_path, name="r", spec="configs/select/tiny.yaml", **over):
    d = tmp_path / name
    (d / "gates").mkdir(parents=True)
    (tmp_path / "root").mkdir(exist_ok=True)
    (d / "gates" / "pre.ok").write_text("{}")
    cfg = dict(pipeline="select", tier="tiny", select_spec=spec, taxdump=TINY_TAX, data_root=str(tmp_path / "root"), catalog=dict(mode="lock"),
               calibration=dict(file="configs/select/rates-tiny.yaml"), disk_reserve_gb=1)
    cfg.update(over)
    (d / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    return d


def _select(tmp_path, name="r", **over):
    d = _run_dir(tmp_path, name, **over)
    for st in ("catalog", "select", "budget", "fit"):
        r = _stage(st, d)
        assert r.returncode == 0, (st, r.stderr[-2000:])
    return d


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---- the host decides a virus's group --------------------------------------------------------------------------

def test_virus_groups_follow_the_host_and_refuse_conflicts_fails(ttax):
    from ci.kingdom import GroupConflict, KingdomMap
    k = KingdomMap()
    phage = [t for t in sorted(ttax.rank) if ttax.rank[t] == "species" and 2731619 in ttax.lineage(t)][0]
    giant = [t for t in sorted(ttax.rank) if ttax.rank[t] == "species" and 2732007 in ttax.lineage(t)][0]
    other = [t for t in sorted(ttax.rank) if ttax.rank[t] == "species" and 10239 in ttax.lineage(t)
             and not {2731619, 2732007} & set(ttax.lineage(t))][0]
    bacterium = [t for t in sorted(ttax.rank) if ttax.rank[t] == "species" and 2 in ttax.lineage(t)][0]
    assert k.group(ttax, other, "raw_virus") == "viruses.eukaryotic"                   # no host: the taxon rule
    assert k.group(ttax, other, "raw_virus", bacterium) == "viruses.phage"             # a prokaryote host: a prokaryote virus
    assert k.group(ttax, other, "raw_virus", 9606) == "viruses.eukaryotic"
    assert k.group(ttax, giant, "raw_virus", 9606) == "viruses.giant"                  # giant viruses stay giant
    assert k.group(ttax, phage, "raw_virus", bacterium) == "viruses.phage"
    with pytest.raises(GroupConflict):
        k.group(ttax, phage, "raw_virus", 9606)                                         # a tailed phage on a human host
    with pytest.raises(GroupConflict):
        k.group(ttax, giant, "raw_virus", bacterium)


def test_g2_checks_the_host_of_a_virus_fails(ttax):
    from ci.gates import g2_exclusion
    ent = ttax.taxa_by_name("Enterobacterales", "order")[0]
    host = [t for t in sorted(ttax.rank) if ttax.rank[t] == "species" and ent in ttax.lineage(t)][0]
    phage = [t for t in sorted(ttax.rank) if ttax.rank[t] == "species" and 2731619 in ttax.lineage(t)][0]
    ok = g2_exclusion.entries([dict(id="p", taxid=phage)], ttax)
    assert ok.status == "pass", ok.evidence
    bad = g2_exclusion.entries([dict(id="p", taxid=phage, host_taxid=host)], ttax)
    assert bad.status == "fail" and bad.evidence["hits"][0]["reason"].startswith("host excluded")


# ---- catalogue snapshots ---------------------------------------------------------------------------------------

def test_canonical_refuses_conflicting_duplicates_and_sorts():
    from ci.stages.catalog import canonical
    data, n = canonical([dict(accession="B", x=1), dict(accession="A", x=2), dict(accession="B", x=1)], "accession")
    assert n == 2 and data.decode().splitlines()[0].startswith('{"accession":"A"')
    with pytest.raises(ValueError):
        canonical([dict(accession="A", x=1), dict(accession="A", x=2)], "accession")


def test_paged_reads_until_the_count_matches_and_fails_otherwise(monkeypatch):
    from ci.stages import catalog
    monkeypatch.setattr(catalog, "PAUSE", 0)
    monkeypatch.setattr(catalog.time, "sleep", lambda s: None)
    pages = {"": dict(total_count=3, reports=[{"accession": "A"}, {"accession": "B"}], next_page_token="t"), "t": dict(reports=[{"accession": "C"}])}
    recs, total, n = catalog._paged(lambda tok, size: tok, lambda r: r, "accession", fetch=lambda tok: json.dumps(pages[tok]).encode())
    assert total == 3 and len(recs) == 3 and n == 2
    shifted = {"": dict(total_count=3, reports=[{"accession": "A"}, {"accession": "B"}], next_page_token="t"), "t": dict(reports=[{"accession": "B"}])}
    with pytest.raises(SystemExit, match="consistently"):              # a record repeated across pages, one skipped
        catalog._paged(lambda tok, size: tok, lambda r: r, "accession", fetch=lambda tok: json.dumps(shifted[tok]).encode())


def test_catalog_refuses_a_non_canonical_file_and_a_live_source_without_lock_fails(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"accession":"B"}\n{"accession":"A"}\n')
    spec = yaml.safe_load(open(os.path.join(ROOT, "configs", "select", "tiny.yaml")))
    spec["catalogues"][0]["file"] = str(bad)
    sp = tmp_path / "spec.yaml"
    sp.write_text(yaml.safe_dump(spec))
    r = _stage("catalog", _run_dir(tmp_path, "a", spec=str(sp)))
    assert r.returncode != 0 and "canonical" in r.stderr
    spec = yaml.safe_load(open(os.path.join(ROOT, "configs", "select", "tiny.yaml")))
    spec["catalogues"][0] = dict(id="bacteria", ncbi_genome_reports=dict(taxon=2, params={}))
    sp.write_text(yaml.safe_dump(spec))
    r = _stage("catalog", _run_dir(tmp_path, "b", spec=str(sp)))
    assert r.returncode != 0 and "no entry in the lock" in r.stderr


# ---- the selection ---------------------------------------------------------------------------------------------

def test_selection_covers_every_group_and_records_every_reason(tmp_path, ttax):
    d = _select(tmp_path)
    sel = _rows(d / "data" / "select" / "selected.tsv")
    cand = _rows(d / "data" / "select" / "candidates.tsv")
    from ci.kingdom import KingdomMap
    groups = {r["group"] for r in sel}
    assert groups == set(KingdomMap().names)
    out = {r["id"]: r["out"] for r in cand}
    assert any(v.startswith("excluded: excluded order Enterobacterales") for v in out.values())
    assert any(v.startswith("excluded: sealed genus Pseudomonas") or v.startswith("excluded: family of sealed genus") for v in out.values())
    assert any(v.startswith("another genome of the species") for v in out.values())
    assert any(v.startswith("fewer than 500") for v in out.values())
    assert any(v.startswith("group conflict") for v in out.values())
    assert any(v.startswith("host excluded") for v in out.values())
    bg = json.load(open(d / "data" / "select" / "by_group.json"))
    assert bg["ungrouped"]["viruses"] == {"group conflict": 1}             # the tailed phage on a eukaryote host
    assert bg["groups"]["viruses.eukaryotic"]["out"].get("host unknown") == 1  # placed by its taxon, then refused
    assert "Rejected before a group could be given" in (d / "data" / "select" / "report.md").read_text()
    assert bg["groups"]["mobile.plasmid"]["selected_units"] >= 3
    assert any(v.startswith("no sequence report") for v in out.values())   # plasmids unknown, not guessed
    pl = [r for r in sel if r["group"] == "mobile.plasmid"]
    assert all(len(r["sequences"].split(";")) == int(r["units"]) for r in pl)
    assert any(v.startswith("host unknown") for v in out.values())        # no host record: the host exclusions cannot be checked
    assert bg["groups"]["viruses.eukaryotic"]["selected_genomes"] >= 1
    red = [r for r in cand if r["id"].startswith("plants_algae_") and r["catalogue"] == "ens_protists"]
    assert red, "a red alga listed by Ensembl Protists is placed by its taxid"
    assert any(r["proteins_level"].startswith("estimated") for r in sel + cand if r.get("proteins_level"))
    for r in sel:
        assert not out[r["id"]], r                                     # nothing selected carries a reason
        assert int(r["proteins_low"]) <= int(r["proteins_central"]) <= int(r["proteins_high"])
    for name in ("rates.json", "budget.json", "report.md", "fit.json"):
        assert (d / "data" / "select" / name).exists()


def test_selection_is_deterministic(tmp_path):
    a, b = _select(tmp_path, "a"), _select(tmp_path, "b")
    for name in ("selected.tsv", "candidates.tsv", "by_group.json", "budget.json", "report.md"):
        assert (a / "data" / "select" / name).read_bytes() == (b / "data" / "select" / name).read_bytes(), name


def test_larger_targets_only_add_genomes(tmp_path):
    a = _select(tmp_path, "a")
    spec = yaml.safe_load(open(os.path.join(ROOT, "configs", "select", "tiny.yaml")))
    for g in spec["groups"].values():
        g["target"] = g["target"] + 2
    sp = tmp_path / "bigger.yaml"
    sp.write_text(yaml.safe_dump(spec))
    b = _select(tmp_path, "b", select_spec=str(sp))
    ids_a = {r["id"] for r in _rows(a / "data" / "select" / "selected.tsv")}
    ids_b = {r["id"] for r in _rows(b / "data" / "select" / "selected.tsv")}
    assert ids_a < ids_b


def test_select_gates_pass_and_catch_tampering_fails(tmp_path):
    d = _select(tmp_path)
    r = _stage("gates", d, "select")
    assert r.returncode == 0, r.stderr[-2000:]
    for g in ("G1_select", "G2_select", "G3_select"):
        assert json.load(open(d / "gates" / (g + ".json")))["status"] == "pass", g
    p = d / "data" / "select" / "selected.tsv"
    good = p.read_text()
    from ci.gates.g3_sealed import load_sealed
    sealed_acc = load_sealed()[0][0]
    lines = good.splitlines()
    head, first = lines[0].split("\t"), lines[1].split("\t")
    cases = []
    t = list(first)
    t[head.index("accession")] = sealed_acc                              # a sealed accession in the selection
    cases.append(("G3_select", t))
    t = list(first)
    t[head.index("group")] = "eukaryota.fungi"                           # a group the taxid does not give
    cases.append(("G1_select", t))
    from ci.taxonomy import Taxonomy
    tax = Taxonomy(TINY_TAX)
    ent = tax.taxa_by_name("Enterobacterales", "order")[0]
    t = list(first)
    t[head.index("taxid")] = str([x for x in sorted(tax.rank) if tax.rank[x] == "species" and ent in tax.lineage(x)][0])
    cases.append(("G2_select", t))
    for gate, row in cases:
        p.write_text("\n".join([lines[0], "\t".join(row)] + lines[2:]) + "\n")
        r = _stage("gates", d, "select")
        assert r.returncode != 0 and json.load(open(d / "gates" / (gate + ".json")))["status"] == "fail", gate
    p.write_text(good)


def test_select_spec_check_fails_on_gaps():
    from ci.gates.g1_kingdom import select_static
    spec = yaml.safe_load(open(os.path.join(ROOT, "configs", "select", "tiny.yaml")))
    assert select_static(spec, "tiny").status == "pass"
    v1 = yaml.safe_load(open(os.path.join(ROOT, "configs", "select", "v1.yaml")))
    assert select_static(v1, "full").status == "pass", select_static(v1, "full").evidence
    bad = dict(v1, target_factor=0.5)                                     # below the floors
    assert select_static(bad, "full").status == "fail"
    bad = dict(spec, rules={k: v for k, v in spec["rules"].items() if k != "viruses"})
    assert select_static(bad, "tiny").status == "fail"                    # every kingdom has a rule
    bad = dict(spec, catalogues=[c for c in spec["catalogues"] if c["id"] != "virus_hosts"])
    assert select_static(bad, "tiny").status == "fail"                    # a rule names a missing catalogue
    late = [dict(id="x", ncbi_genome_accessions={"from": ["later"], "field": "a"}), dict(id="later", file="f", **{"as": "ensembl_species"})]
    assert select_static(dict(spec, catalogues=spec["catalogues"] + late), "tiny").status == "fail"


def test_unknown_pipeline_is_refused_fails():
    from ci.__main__ import pipeline
    assert pipeline({})["target"] == "all" and pipeline({"pipeline": "select"})["target"] == "select_all"
    with pytest.raises(SystemExit):
        pipeline({"pipeline": "bogus"})


def test_species_without_genus_are_not_pooled():
    from ci.stages.select import pick_order
    rows = [dict(id="s%d" % i, genus=0, species_taxid=100 + i, family=0, units=1) for i in range(6)]
    rows += [dict(id="g%d" % i, genus=50 + i, species_taxid=200 + i, family=7, units=1) for i in range(2)]
    order = pick_order(rows, "family", 1, "viruses.phage")
    assert all(d == 0 for d, _ in order)                                # every genus-less species is its own genus: depth 0


def test_quality_prefers_zero_contamination(ttax):
    from ci.gates.g2_exclusion import Excluder
    from ci.kingdom import KingdomMap
    from ci.stages import select as S
    A = yaml.safe_load(open(os.path.join(ROOT, "configs", "select", "assumptions.yaml")))
    c = S.Candidates(ttax, KingdomMap(), Excluder(ttax), A)
    recs = [json.loads(l) for l in open(os.path.join(ROOT, "tests", "ci", "fixtures", "select", "bacteria.jsonl"))][:1]
    clean, dirty = dict(recs[0], accession="GCF_900000002.1", checkm_contamination=0.0), dict(recs[0], accession="GCF_900000001.1", checkm_contamination=0.5)
    S.prokaryotes(c, [dirty, clean], "bacteria", dict(min_protein_coding=0))
    win = S.one_per_species([r for r in c.rows if r["group"] == "bacteria"])
    assert [w["accession"] for w in win] == ["GCF_900000002.1"]


def test_targets_are_exact():
    from ci.gates.g1_kingdom import select_targets
    spec = yaml.safe_load(open(os.path.join(ROOT, "configs", "select", "tiny.yaml")))
    spec = dict(spec, groups={g: {"min_genomes": 10} for g in spec["groups"]}, target_factor=1.1)
    assert set(select_targets(spec).values()) == {11}


def test_params_normalised():
    from ci.stages.catalog import params
    assert params({"b": True, "a": ["x", "y"], "c": 3}) == [("a", "x"), ("a", "y"), ("b", "true"), ("c", "3")]
    with pytest.raises(SystemExit):
        params({"a": {"x": 1}})


def test_paged_refuses_empty_and_countless_pages(monkeypatch):
    from ci.stages import catalog
    with pytest.raises(SystemExit, match="no record"):
        catalog._paged(lambda t, size: t, lambda r: r, "accession", fetch=lambda t: json.dumps(dict(total_count=0, reports=[])).encode())
    with pytest.raises(SystemExit, match="total_count"):
        catalog._paged(lambda t, size: t, lambda r: r, "accession", fetch=lambda t: json.dumps(dict(reports=[{"accession": "A"}])).encode())


def test_g3_built_still_fails_closed_on_taxid_zero(ttax):
    from ci.gates import g3_sealed
    from ci.taxonomy import UnknownTaxid
    with pytest.raises(UnknownTaxid):
        g3_sealed.built("/nonexistent", ttax, {"x": [0]}, "tiny")


def test_paged_halves_the_page_after_server_errors(monkeypatch):
    from ci.stages import catalog
    monkeypatch.setattr(catalog, "PAUSE", 0)
    seen = []

    def fetch(u):
        tok, size = u
        seen.append(size)
        if size > 250:
            raise RuntimeError("HTTP Error 500")
        return json.dumps(dict(total_count=2, reports=[{"accession": "A"}, {"accession": "B"}])).encode()
    recs, total, pages = catalog._paged(lambda tok, size: (tok, size), lambda r: r, "accession", fetch=fetch, page_size=1000)
    assert total == 2 and seen == [1000, 500, 250]
    with pytest.raises(RuntimeError):
        catalog._paged(lambda tok, size: (tok, size), lambda r: r, "accession", fetch=lambda u: (_ for _ in ()).throw(RuntimeError("500")), page_size=200)


def test_ensembl_rows_are_keyed_by_production_name():
    from ci.stages.catalog import canonical, reduce_ensembl
    rows = [dict(name="Alternaria arborescens", species="alternaria_arborescens_gca_1", taxonomy_id="156630", assembly_accession="GCA_1.1", genebuild="g", core_db="c"),
            dict(name="Alternaria arborescens", species="alternaria_arborescens_gca_2", taxonomy_id="156630", assembly_accession="GCA_2.1", genebuild="g", core_db="c")]
    data, n = canonical([reduce_ensembl(r, "ensemblgenomes", "fungi", 63) for r in rows], "name")
    assert n == 2 and b"alternaria_arborescens_gca_2" in data


def test_sequence_reports_one_per_assembly_and_missing_ones():
    from ci.stages import catalog

    def fetch(url):
        acc = url.split("/accession/")[1].split("/")[0]
        if acc == "GCF_2":
            raise RuntimeError("https://x: HTTP Error 404: Not Found")
        return json.dumps(dict(reports=[dict(assigned_molecule_location_type="Chromosome", role="assembled-molecule", genbank_accession="X1", length=10),
                                        dict(assigned_molecule_location_type="Plasmid", role="assembled-molecule", genbank_accession="X2", length=5)])).encode()
    recs, missing = catalog.fetch_sequence_reports(["GCF_1", "GCF_2", "GCF_1"], fetch=fetch, rate=1000)
    assert [r["accession"] for r in recs] == ["GCF_1"] and missing == ["GCF_2"]
    assert [x["type"] for x in recs[0]["replicons"]] == ["Chromosome", "Plasmid"]


def test_relaxed_variant_keeps_unknown_hosts_and_strict_refuses_them(tmp_path):
    strict = _select(tmp_path, "s")
    relaxed = _select(tmp_path, "r", select_spec="configs/select/tiny-relaxed.yaml")
    cs = {r["id"]: r for r in _rows(strict / "data" / "select" / "candidates.tsv")}
    cr = {r["id"]: r for r in _rows(relaxed / "data" / "select" / "candidates.tsv")}
    gone = [i for i, r in cs.items() if r["out"].startswith("host unknown")]
    assert gone and all(not cr[i]["out"].startswith("host unknown") for i in gone)
    assert json.load(open(relaxed / "data" / "select" / "by_group.json"))["unknown_host"] == "keep"
    assert "Variant: viruses without a usable host record are kept" in (relaxed / "data" / "select" / "report.md").read_text()


def test_virus_catalogues_merge_by_accession_and_prefer_refseq(tmp_path):
    d = _select(tmp_path)
    cand = _rows(d / "data" / "select" / "candidates.tsv")
    ids = [r["id"] for r in cand]
    assert len(ids) == len(set(ids))
    twin = [r for r in cand if r["accession"] == "GCA_000000400.1"]
    assert twin and twin[0]["out"].startswith("another genome of the species")    # the RefSeq record of the species wins
    est = [r for r in cand if r["accession"] in ("GCA_000000440.1", "GCA_000000441.1")]
    assert est and all(r["proteins_level"] == "estimated from the genome length" and int(r["proteins_central"]) == 63 for r in est)


def test_spec_base_merges_and_detects_loops(tmp_path):
    from ci.stages.common import load_select_spec
    spec = load_select_spec("configs/select/v1-relaxed.yaml")
    assert spec["name"] == "v1-relaxed" and spec["rules"]["viruses"]["unknown_host"] == "keep"
    assert spec["rules"]["viruses"]["catalogues"] == ["viruses", "phages"] and spec["tier"] == "full"
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("base: %s\nname: a\n" % b)
    b.write_text("base: %s\nname: b\n" % a)
    with pytest.raises(SystemExit, match="loop"):
        load_select_spec(str(a))



def test_sequence_report_errors_that_are_not_answers_fail_the_stage():
    from ci.stages import catalog

    def fetch(url):
        raise RuntimeError("https://x: HTTP Error 500: Internal Server Error")
    with pytest.raises(SystemExit, match="not a definite answer"):
        catalog.fetch_sequence_reports(["GCF_1"], fetch=fetch, rate=1000)


def test_species_less_genomes_are_their_own_units():
    from ci.stages.select import one_per_species
    rows = [dict(id="a", species_taxid=0, taxid=11, quality=(0,), out=""), dict(id="b", species_taxid=0, taxid=12, quality=(0,), out=""),
            dict(id="c", species_taxid=5, taxid=13, quality=(1,), out=""), dict(id="d", species_taxid=5, taxid=14, quality=(0,), out="")]
    win = one_per_species(rows)
    assert sorted(r["id"] for r in win) == ["a", "b", "d"] and rows[2]["out"].startswith("another genome")


def test_empty_taxon_inside_a_list_is_allowed(monkeypatch):
    from ci.stages import catalog
    monkeypatch.setattr(catalog, "PAUSE", 0)
    answers = {1: dict(total_count=1, reports=[{"accession": "A", "organism": {"tax_id": 1}}]), 2: dict(total_count=0, reports=[])}
    fetch = lambda url: json.dumps(answers[int(url.split("/taxon/")[1].split("/")[0])]).encode()   # noqa: E731
    recs, total, pages = catalog.fetch_genome_reports({"ncbi_genome_reports": {"taxon": [1, 2]}}, fetch)
    assert [r["accession"] for r in recs] == ["A"] and total == 1
    with pytest.raises(SystemExit, match="no record"):
        catalog.fetch_genome_reports({"ncbi_genome_reports": {"taxon": 2}}, fetch)


def test_g3_static_scans_the_base_of_a_variant(tmp_path):
    from ci.stages.common import load_select_spec
    read = []
    load_select_spec("configs/select/tiny-relaxed.yaml", files=read)
    assert read == ["configs/select/tiny-relaxed.yaml", "configs/select/tiny.yaml"]


def test_select_spec_with_a_null_rule_fails_without_crashing():
    from ci.gates.g1_kingdom import select_static
    spec = yaml.safe_load(open(os.path.join(ROOT, "configs", "select", "tiny.yaml")))
    spec["rules"]["plasmids"] = None
    r = select_static(spec, "tiny")
    assert r.status == "fail" and any(p.get("rule") == "plasmids" for p in r.evidence["problems"])
