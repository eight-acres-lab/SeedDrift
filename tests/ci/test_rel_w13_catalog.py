"""Stage rel_w13_catalog (ci/stages/rel/w13_catalog.py), the metadata snapshot of the ENA portal, with a stand-in for
the portal; and the second, wider count of stage rel_w13_census on such a snapshot (census_wide), on the tiny fixture
taxonomy with one more species of the genus Coxiella, whose relatives and exclusions are known."""
import json
import os
import urllib.parse

import pytest

import rel_fixtures as RF
import test_rel_w13_census as T13
import test_rel_w2_stage as TS
from ci.gates import GateFailure
from ci.stages.common import Stage
from ci.stages.rel import w13_catalog as WK
from ci.stages.rel import w13_census as WC
from ci.taxonomy import Taxonomy

HEAD = ["assembly_accession", "tax_id", "scientific_name", "strain", "assembly_level", "genome_representation", "base_count", "completeness_score",
        "contamination_score", "host_tax_id", "environmental_sample", "version", "last_updated", "accession"]
NEW_SPECIES = 90000001                                                   # a second species of the genus Coxiella (776), added to the fixture


def rec(acc, taxid, host="", rep="full", env="false", bases=2_000_000):
    return dict(assembly_accession=acc, tax_id=str(taxid), scientific_name="x", strain="", assembly_level="complete genome", genome_representation=rep,
                base_count=str(bases), completeness_score="", contamination_score="", host_tax_id=str(host), environmental_sample=env, version="1",
                last_updated="2026-01-01", accession=acc)


# bacteria: the anchor's species twice (one an environmental sample), a partial one, the anchor's own assembly, a sealed
# test genome's assembly, one of the genus, one of the closed development order, an unknown taxid; viruses: the anchor
# phage's species with a bacterial host, with an excluded host, without a host (the selection knows one), with a
# eukaryote host (a conflict), and a eukaryote virus without any host
RECORDS = {
    "bacteria": [rec("GCA_000000101", 227377), rec("GCA_000000102", 777, env="true"), rec("GCA_000000103", 227377, rep="partial"),
                 rec("GCA_000000104", 227377), rec("GCA_000000001", 227377), rec("GCA_000000105", NEW_SPECIES), rec("GCA_000000106", 615),
                 rec("GCA_000000107", 99999999)],
    "viruses": [rec("GCA_000000201", 10702, host=227377, bases=40_000), rec("GCA_000000202", 10702, host=615), rec("GCA_000000203", 10702),
                rec("GCA_000000204", 10702, host=9258), rec("GCA_000000205", 10377)],
}


def portal(records, short=None):
    """A stand-in for fetch.get on the ENA portal: /count announces the records of a query, /search returns them
    (`short`: a catalogue whose answer lacks its last row)."""
    def get(url):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        query = q["query"][0]
        cid = "bacteria" if "tax_tree(2)" in query else "archaea" if "tax_tree(2157)" in query else "viruses"
        rows = records.get(cid, [])
        if "/count?" in url:
            return ("count\n%d\n" % (len(rows) if "AND" in query and "complete" in query else len(rows) * 3)).encode()
        if cid == short:
            rows = rows[:-1]
        fields = q["fields"][0].split(",")
        assert "accession" not in fields                               # the portal appends it
        cols = ["assembly_accession"] + [f for f in fields if f != "assembly_accession"] + ["accession"]
        return ("\t".join(cols) + "\n" + "".join("\t".join(r.get(c, "") for c in cols) + "\n" for r in rows)).encode()
    return get


def catalog_run(tmp, records=RECORDS, short=None, run_id="20261124T000000-cat"):
    run_dir, cfg = RF.rel_run(tmp, kind="rel_corpus", stages=("w13_catalog",), run_id=run_id, data_root=str(tmp / "store"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(WK, "get", portal(records, short))
        with Stage(run_dir, "rel_w13_catalog") as st:
            WK.build(st, cfg, run_dir)
    rp = os.path.join(run_dir, "record.json")
    with open(rp) as f:
        rec_ = json.load(f)
    with open(rp, "w") as f:
        json.dump(dict(rec_, status="passed"), f)
    return run_dir, cfg


def test_the_snapshot_is_canonical_locked_and_read_back(tmp_path):
    run_dir, cfg = catalog_run(tmp_path)
    cats, lock, prov = WK.read(run_dir)
    assert {k: len(v) for k, v in cats.items()} == {"bacteria": 8, "archaea": 0, "viruses": 5}
    assert [r["accession"] for r in cats["bacteria"]] == sorted(r["accession"] for r in RECORDS["bacteria"])
    assert lock["catalogues"]["viruses"]["records"] == lock["catalogues"]["viruses"]["announced"] == 5 and len(lock["catalogues"]["viruses"]["sha256"]) == 64
    assert lock["catalogues"]["bacteria"]["fetched"].endswith("Z") and lock["context"]["2"]["contig"] == 24
    with Stage(run_dir, "gates_rel_w13_catalog") as st:
        WK.gate(st, cfg, run_dir)
    assert json.load(open(os.path.join(run_dir, "gates", "G4_rel_w13_catalog.json")))["status"] == "pass"
    second, _ = catalog_run(tmp_path / "b")                              # the same answer: the same bytes
    assert WK.read(second)[1]["catalogues"]["bacteria"]["sha256"] == lock["catalogues"]["bacteria"]["sha256"]
    p = os.path.join(run_dir, WK.OUT, "viruses.tsv")
    with open(p, "a") as f:
        f.write("GCA_9\t1\n")
    with pytest.raises(SystemExit, match="not the snapshot"):
        WK.read(run_dir)
    with pytest.raises(GateFailure):
        with Stage(run_dir, "gates_rel_w13_catalog") as st:
            WK.gate(st, cfg, run_dir)


def test_an_incomplete_answer_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="not complete"):
        catalog_run(tmp_path, short="viruses")
    with pytest.raises(SystemExit, match="unknown keys"):
        WK.settings(dict(rel_w13_catalog=dict(colour=1)))
    body = ("\t".join(HEAD) + "\n" + "\t".join(rec("GCA_1", 2).get(c, "") for c in HEAD) + "\n" + "\t".join(rec("GCA_1", 3).get(c, "") for c in HEAD) + "\n").encode()
    with pytest.raises(SystemExit, match="twice"):
        WK.canonical(body, WK.FIELDS)


def fixture_taxonomy():
    t = Taxonomy(os.path.join(os.path.dirname(__file__), "fixtures", "tiny", "taxdump"))
    t.parent[NEW_SPECIES], t.rank[NEW_SPECIES], t.sci[NEW_SPECIES] = 776, "species", "Coxiella sp. fixture"
    return t


def test_the_wide_count_follows_the_rules_of_the_selection(tmp_path):
    tax = fixture_taxonomy()
    run_dir, _ = catalog_run(tmp_path)
    cats, _, _ = WK.read(run_dir)
    metas = [dict(id="bacteria_GCF_000000104.1", group="bacteria", split="train", taxid=227377, ranks=[0, 0, 0, 0, 776, 777]),
             dict(id="plasmid_GCF_000000104.1", group="mobile.plasmid", split="train", taxid=227377, ranks=[0, 0, 0, 0, 776, 777]),
             dict(id="phage_a", group="viruses.phage", split="heldout", taxid=10702, ranks=[0, 0, 0, 0, 3044853, 3060797]),
             dict(id="fungi_a", group="eukaryota.fungi", split="train", taxid=4829, ranks=[0, 0, 0, 0, 4828, 4829])]
    entries = [dict(accession="GCF_000000104.1" if m["group"] in ("bacteria", "mobile.plasmid") else "", paired_accession="") for m in metas]
    cands = [dict(id="virus_x", group="viruses.phage", species_taxid="3060797", host_taxid="2320", out="selected", selected="0"),
             dict(id="fungi_b", group="eukaryota.fungi", species_taxid="4829", genus="4828", out="not needed for the target (pick 2)", selected="0", proteins_central="9000")]
    sealed = RF.sealed(test_ids=TS.SEALED)
    s = dict(per_anchor=[1, 3], propose=3, max_genes_euk=1500)
    counts, reasons, kept = WC.census_wide(cats, metas, entries, cands, s, tax, sealed, [])
    b = counts["train"]["bacteria"]
    assert b["new_same_species"]["candidates"] == 2 and b["new_same_genus"]["candidates"] == 1        # 101, 102; 105
    assert counts["train"]["mobile.plasmid"]["new_same_species"]["candidates"] == 2                    # the host species' assemblies, not its own
    assert b["in_corpus"]["same_species"] == 0                                                          # its own assembly is not its relative
    ph = counts["heldout"]["viruses.phage"]["new_same_species"]["candidates"]
    assert ph == 2                                                                                      # 201 (its host), 203 (the selection's host)
    assert counts["train"]["eukaryota.fungi"]["new_same_species"]["candidates"] == 1 and kept == ["eukaryota.fungi"]
    assert counts["train"]["eukaryota.fungi"]["proposals"]["3"]["proteins"] == 1500
    want = {"bacteria | eligible": 2, "bacteria | eligible, an environmental sample": 1, "bacteria | partial genome": 1, "bacteria | already in the corpus": 1,
            "bacteria | refused by the sealed check": 1, "bacteria | excluded by lineage": 1, "bacteria | taxid unknown": 1, "viruses | excluded by a host rule": 1,
            "viruses | group conflict": 1}
    assert all(reasons.get(k) == v for k, v in want.items()), reasons
    assert reasons["viruses | eligible"] == 2
    assert [k for k in reasons if k.startswith("viruses | host unknown")] and sum(reasons.values()) == 13
    text = json.dumps(dict(counts=counts, reasons=reasons))
    assert "GCA_000000001" not in text and not any(x in text for x in TS.SEALED)


def test_the_census_reports_both_counts_side_by_side(tmp_path):
    tmp = tmp_path
    src = str(tmp / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    metas = TS.make_source(src)
    sel, spec, _ = T13.make_selection(tmp, metas)
    cat_dir, _ = catalog_run(tmp / "cat")
    run_dir, cfg = RF.rel_run(tmp, kind="rel_corpus", stages=("w13_census",), run_id="20261124T000001-w13", corpus_spec=spec,
                              rel_w13_census=dict(select_from=sel, source=src, expect=TS.SIZES, per_anchor=[1, 3], propose=3, catalog_from=cat_dir))
    with pytest.MonkeyPatch.context() as mp:
        TS.patch(mp, tmp)
        with Stage(run_dir, "rel_w13_census") as st:
            rep = WC.build(st, cfg, run_dir)
    w = rep["wide"]
    assert set(w["counts"]) == {"train", "heldout"} and w["catalogue"]["catalogues"]["bacteria"]["records"] == 8
    assert w["groups_from_the_selection"] == sorted(g for g in TS.TINY if g.startswith("eukaryota."))
    for sp in ("train", "heldout"):                                     # the eukaryotes are counted on the same candidates both times
        for g in w["groups_from_the_selection"]:
            if g in rep["counts"][sp]:
                assert w["counts"][sp][g]["new_same_species"] == rep["counts"][sp][g]["new_same_species"]
    page = open(os.path.join(run_dir, WC.OUT, "report.md")).read()
    assert "Side by side" in page and "How the two counts differ" in page and not any(x in page for x in TS.SEALED)
    with Stage(run_dir, "gates_rel_w13_census") as st:
        WC.gate(st, cfg, run_dir)
