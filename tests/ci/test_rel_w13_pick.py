"""Stages rel_w13_pick and rel_w13_build (ci/stages/rel/w13_pick.py, w13_build.py): the first batch of close relatives
by name, the ENA lookups that replace a pick without the replicon its group needs, the corpus spec that adds them, and
the checks of the corpus built from it (pinned splits, relatives in their anchor's split, no leak)."""
import json
import os

import pytest
import yaml

import rel_fixtures as RF
import test_rel_w13_catalog as TK
import test_rel_w13_census as T13
import test_rel_w2_stage as TS
from ci.gates import GateFailure
from ci.stages.common import Stage
from ci.stages.rel import w13_build as WB
from ci.stages.rel import w13_census as WC
from ci.stages.rel import w13_pick as WP


def row(acc, group, catalogue="bacteria", version="1", taxid=227377):
    return dict(id=acc, accession=acc, version=version, group=group, catalogue=catalogue, taxid=taxid, proteins_central=100, bases=100_000)


def anchor(aid, group, split, sp=(), ge=()):
    return dict(id=aid, group=group, split=split, new_same_species=list(sp), new_same_genus=list(ge))


def test_picks_are_looked_up_and_replaced_in_order():
    reps = {"GCA_1.1": [("CP1", "Chromosome")], "GCA_2.1": [("CP2", "Chromosome"), ("CP3", "Plasmid")], "GCA_3.1": [],
            "GCA_4.1": [("CP4", "Chromosome")], "GCA_5.1": [("CP5", "Chromosome")], "GCA_6.1": [("CP6", "Plasmid")]}
    asked = []

    def get(a):
        asked.append(a)
        return reps[a]
    euk = dict(row("fungi_x", "eukaryota.fungi", catalogue="ens_fungi"), id="fungi_x")
    anchors = [anchor("b1", "bacteria", "train", sp=[row("GCA_3", "bacteria"), row("GCA_1", "bacteria")], ge=[row("GCA_4", "bacteria"), row("GCA_5", "bacteria")]),
               anchor("p1", "mobile.plasmid", "heldout", sp=[row("GCA_1", "mobile.plasmid"), row("GCA_2", "mobile.plasmid")], ge=[row("GCA_6", "mobile.plasmid")]),
               anchor("f1", "eukaryota.fungi", "train", sp=[euk]),
               anchor("t1", "bacteria", "test", sp=[row("GCA_5", "bacteria")])]
    chosen, seen = WP.choose(anchors, 3, 40, 2, get)
    assert [r["accession"] for r, _ in chosen["b1"]["picks"]] == ["GCA_1", "GCA_4", "GCA_5"] and chosen["b1"]["looked"] == 4   # GCA_3 lists nothing
    assert [rel for _, rel in chosen["b1"]["picks"]] == ["same species", "same genus", "same genus"]
    assert [r["accession"] for r, _ in chosen["p1"]["picks"]] == ["GCA_2", "GCA_6"]           # only assemblies that list a plasmid
    assert [r["id"] for r, _ in chosen["f1"]["picks"]] == ["fungi_x"] and "t1" not in chosen  # no lookup for a eukaryote; no test anchor
    assert len(asked) == len(set(asked)) and "GCA_5.1" in seen
    capped, _ = WP.choose(anchors[:1], 3, 2, 1, get)
    assert [r["accession"] for r, _ in capped["b1"]["picks"]] == ["GCA_1"]                    # at most max_lookups candidates examined


def test_sources_have_the_fields_of_the_corpus_spec():
    b = WP.source_of(row("GCA_9", "bacteria", version="2"), "same species", {}, 1500)
    assert b == dict(id="bacteria_GCA_9.2", source_kind="raw_prokaryote", taxid=227377, table=11, accession="GCA_9.2", paired_accession="",
                     ena_assembly="GCA_9.2", replicon_types=["Chromosome"], relation="same species")
    p = WP.source_of(row("GCA_9", "mobile.plasmid"), "same genus", {}, 1500)
    assert p["id"] == "plasmid_GCA_9.1" and p["source_kind"] == "plasmid" and p["replicon_types"] == ["Plasmid"]
    v = WP.source_of(row("GCA_8", "viruses.phage", catalogue="viruses"), "same species", {}, 1500)
    assert v["id"] == "virus_phage_GCA_8.1" and "replicon_types" not in v and v["source_kind"] == "raw_virus"
    ens = WP.ensembl_fields({"fungi_a_b": dict(group="eukaryota.fungi", catalogue="ens_fungi"), "fungi_zz": dict(group="eukaryota.fungi", catalogue="ens_fungi")},
                            {"ens_fungi": [dict(name="b", site="ensemblgenomes", site_division="fungi", release=63, core_db="b_core_63"),
                                           dict(name="a_b", site="ensemblgenomes", site_division="fungi", release=63, core_db="fungi_asco1_collection_core_63_116_1")]})
    assert ens == {"fungi_a_b": dict(site="ensemblgenomes", division="fungi", species="a_b", release=63, collection="fungi_asco1_collection")}
    e = WP.source_of(dict(row("fungi_a_b", "eukaryota.fungi", catalogue="ens_fungi"), id="fungi_a_b"), "same genus", ens, 1500)
    assert e == dict(id="fungi_a_b", source_kind="annotated_cds", taxid=227377, ensembl=ens["fungi_a_b"], max_genes=1500, relation="same genus")
    with pytest.raises(SystemExit, match="Ensembl"):
        WP.source_of(dict(row("fungi_zz", "eukaryota.fungi"), id="fungi_zz"), "same genus", ens, 1500)


def g(gid, group, split, genus, species, cluster=""):
    return dict(id=gid, source="corpus", group=group, split=split, ranks=[1, 2, 3, 4, genus, species], genes=10, hits=1, cluster=cluster)


def test_the_build_checks_pinned_splits_and_leaks():
    old = [g("a", "bacteria", "train", 10, 100), g("b", "bacteria", "heldout", 20, 200), g("t", "bacteria", "test", 30, 300), g("e", "x", "eval", 0, 0)]
    rel = [dict(id="r1", split="train"), dict(id="r2", split="heldout"), dict(id="r3", split="train")]
    good = old + [g("r1", "bacteria", "train", 10, 101), g("r2", "bacteria", "heldout", 20, 200)]
    problems, counts, totals = WB.check(old, good, rel, {"r3"})
    assert problems == [] and counts == {"train": {"bacteria": dict(genomes=1, genes=10, with_hits=1)}, "heldout": {"bacteria": dict(genomes=1, genes=10, with_hits=1)}}
    assert totals == dict(pinned=4, kept=4, new=2, relatives=3, unavailable=["r3"])
    bad = [dict(x, split="heldout") if x["id"] == "a" else x for x in old] + [g("r1", "bacteria", "train", 20, 201), g("r2", "bacteria", "heldout", 30, 300, cluster="c"),
                                                                          g("zz", "bacteria", "train", 40, 400)]
    problems, _, _ = WB.check(old, bad, rel, set())
    kinds = sorted({(p["check"], p["error"].split(" ")[0]) for p in problems})
    assert ("pinned", "1") in kinds and ("relatives", "new") in kinds and ("relatives", "relatives") in kinds and ("no leak", "new") in kinds
    leak = [p for p in problems if p["check"] == "no leak" and "unit" in p["error"]][0]
    assert sorted(leak["examples"]) == ["r1", "r2"]                       # r1 in a held-out genus, r2 in a test genus
    text = json.dumps(problems)
    assert '"t"' not in text                                               # a test genome is never named


@pytest.fixture(scope="module")
def picked(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w13p")
    src = str(tmp / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    metas = TS.make_source(src)
    sel, spec, _ = T13.make_selection(tmp, metas)
    cat_dir, _ = TK.catalog_run(tmp / "cat")
    census = dict(select_from=sel, source=src, expect=TS.SIZES, per_anchor=[1, 3], propose=3, catalog_from=cat_dir)
    run_c, cfg_c = RF.rel_run(tmp, kind="rel_corpus", stages=("w13_census",), run_id="20261124T000001-w13", corpus_spec=spec, rel_w13_census=census)
    with pytest.MonkeyPatch.context() as mp:
        TS.patch(mp, tmp)
        with Stage(run_c, "rel_w13_census") as st:
            WC.build(st, cfg_c, run_c)
    block = dict(select_from=sel, source=src, expect=TS.SIZES, catalog_from=cat_dir, census_from=run_c, k=3, name="x-k3", threads=2)
    # stage w13_pick is not yet in the registry of kinds (request 20261123-w13-step2 asks for it)
    run_p, cfg_p = RF.rel_run(tmp, kind="rel_corpus", stages=("w13_census",), run_id="20261124T000002-pick", corpus_spec=spec, rel_w13_pick=block)
    with pytest.MonkeyPatch.context() as mp:
        TS.patch(mp, tmp)
        mp.setattr(WP, "replicons", lambda a: [("CP_" + a, "Chromosome"), ("PL_" + a, "Plasmid")])
        with Stage(run_p, "rel_w13_pick") as st:
            rep = WP.build(st, cfg_p, run_p)
    return dict(tmp=tmp, metas=metas, spec=spec, run=run_p, cfg=cfg_p, rep=rep, census=run_c)


def test_the_pick_names_the_census_relatives_and_writes_the_corpus_spec(picked, monkeypatch):
    rep, run = picked["rep"], picked["run"]
    crep = json.load(open(os.path.join(picked["census"], WC.OUT, "report.json")))
    for sp in WC.OPEN:                                                     # the census's first batch, k = 3, by group
        for grp, d in crep["wide"]["counts"][sp].items():
            assert rep["by_split"][sp][grp]["picked"] == d["proposals"]["3"]["genomes"] == rep["by_split"][sp][grp]["census"]
    rel = json.load(open(os.path.join(run, WP.OUT, "relatives.json")))["relatives"]
    assert rel and len(rel) == rep["relatives"] and all(x["split"] in WC.OPEN for x in rel)
    split = {m["id"]: m["split"] for m in picked["metas"]}
    assert all(split[a] == x["split"] for x in rel for a in x["anchors"])
    spec = yaml.safe_load(open(os.path.join(run, WP.OUT, "corpus.yaml")))
    old = yaml.safe_load(open(picked["spec"]))
    assert spec["name"] == "x-k3" and spec["sources"][:len(old["sources"])] == old["sources"] and len(spec["sources"]) == len(old["sources"]) + len(rel)
    head = open(os.path.join(run, WP.OUT, "corpus.yaml")).read(4000)
    assert WC.corpus_header_sha(os.path.join(run, WP.OUT, "corpus.yaml")) == WC.corpus_header_sha(picked["spec"]) and "rel_w13_pick" in head
    new = {x["id"]: x for x in spec["sources"][len(old["sources"]):]}
    assert all(x["source_kind"] == "annotated_cds" and x["max_genes"] == 1500 and x["ensembl"]["release"] == 63 for x in new.values())
    assert {x["relation"] for x in new.values()} == {"same species", "same genus"} and not any("cluster" in x for x in new.values())
    text = open(os.path.join(run, WP.OUT, "corpus.yaml")).read() + json.dumps(rep)
    assert not any(x in text for x in TS.SEALED)
    TS.patch(monkeypatch, picked["tmp"])
    with Stage(run, "gates_rel_w13_pick") as st:
        WP.gate(st, picked["cfg"], run)
    assert json.load(open(os.path.join(run, "gates", "G4_rel_w13_pick.json")))["status"] == "pass"


def test_a_census_that_differs_is_refused(picked, monkeypatch, tmp_path):
    TS.patch(monkeypatch, picked["tmp"])
    p = os.path.join(picked["census"], WC.OUT, "report.json")
    keep = open(p).read()
    rep = json.loads(keep)
    rep["wide"]["counts"]["train"]["all"]["proposals"]["3"]["genomes"] += 1
    try:
        open(p, "w").write(json.dumps(rep))
        with pytest.raises(SystemExit, match="differs from run"):
            with Stage(picked["run"], "rel_w13_pick") as st:
                WP.build(st, picked["cfg"], picked["run"])
    finally:
        open(p, "w").write(keep)
    bad = os.path.join(picked["run"], WP.OUT, "corpus.yaml")
    body = open(bad).read()
    try:
        open(bad, "a").write("# changed\n")
        with pytest.raises(GateFailure):
            with Stage(picked["run"], "gates_rel_w13_pick") as st:
                WP.gate(st, picked["cfg"], picked["run"])
    finally:
        open(bad, "w").write(body)
