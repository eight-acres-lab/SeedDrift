"""Stage rel_w13_supply (ci/stages/rel/w13_supply.py): how many genomes eukaryotic viruses and plants and algae could
still gain, metadata only, on stand-in NCBI genome reports over the fixture taxonomy with two more species (a virus of
the genus of Human herpesvirus 4, a red alga of the genus Chondrus)."""
import json
import os

import pytest

import rel_fixtures as RF
import test_rel_w13_census as T13
import test_rel_w2_stage as TS
from ci.stages.common import Stage
from ci.stages.rel import w13_census as WC
from ci.stages.rel import w13_supply as WS

NEW_VIRUS, NEW_ALGA, HOSTLESS = 90000011, 90000012, 90000013


def taxonomy():
    from ci.taxonomy import Taxonomy
    t = Taxonomy(os.path.join(os.path.dirname(__file__), "fixtures", "tiny", "taxdump"))
    t.parent[NEW_VIRUS], t.rank[NEW_VIRUS], t.sci[NEW_VIRUS] = 10375, "species", "herpesvirus fixture"
    t.parent[HOSTLESS], t.rank[HOSTLESS], t.sci[HOSTLESS] = 10375, "species", "herpesvirus without a host record"
    t.parent[NEW_ALGA], t.rank[NEW_ALGA], t.sci[NEW_ALGA] = 2768, "species", "Chondrus fixture"
    return t


def rec(acc, taxid, genes, level="Complete Genome", length=150000, paired=""):
    return dict(accession=acc, paired_accession=paired, taxid=taxid, organism="x", assembly_level=level, assembly_status="current", refseq_category="",
                length=length, chromosomes=0, components=1, protein_coding=genes, checkm_completeness=None, checkm_contamination=None, source_database="")


VIRUS_RECS = [rec("GCF_000000501.1", NEW_VIRUS, 80, paired="GCA_000000501.1"), rec("GCA_000000501.1", NEW_VIRUS, 80, paired="GCF_000000501.1"),
              rec("GCA_000000502.1", 10377, 0, length=150000), rec("GCA_000000503.1", 10702, 50), rec("GCA_000000504.1", 10280, 200),
              rec("GCA_000000505.1", HOSTLESS, 30)]
PLANT_RECS = [rec("GCA_000000601.1", NEW_ALGA, 9000, level="Scaffold"), rec("GCA_000000602.1", NEW_ALGA, 8000, level="Chromosome"),
              rec("GCA_000000603.1", 2769, 0, level="Contig"), rec("GCA_000000604.1", 10377, 50)]


def test_the_supply_of_the_two_groups_is_counted(tmp_path, monkeypatch):
    from ci.stages import catalog as CAT
    from ci.taxonomy import Taxonomy
    src = str(tmp_path / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    metas = TS.make_source(src)
    sel, spec, _ = T13.make_selection(tmp_path, metas)
    run_dir, cfg = RF.rel_run(tmp_path, kind="rel_corpus", stages=("w13_census",), run_id="20261124T000010-supply", corpus_spec=spec,
                              data_root=str(tmp_path / "store"), rel_w13_supply=dict(select_from=sel, source=src, expect=TS.SIZES))
    TS.patch(monkeypatch, tmp_path)
    ft = taxonomy()
    monkeypatch.setattr(Taxonomy, "__init__", lambda self, folder, names=True: self.__dict__.update(ft.__dict__))
    monkeypatch.setattr(WC, "host_records", lambda run: ([dict(virus_taxid=NEW_VIRUS, host_taxid=9258), dict(virus_taxid=10377, host_taxid=9258),
                                                          dict(virus_taxid=10702, host_taxid=2320)], [dict(id="hosts", records=3, sha256="0" * 64)]))

    def snap(c, done, cas, fetch=None):
        recs = VIRUS_RECS if c["id"] == "viruses" else PLANT_RECS
        data, n = CAT.canonical(recs, "accession")
        return data, n, dict(total=n, pages=1)
    monkeypatch.setattr(CAT, "snapshot", snap)
    with Stage(run_dir, "rel_w13_supply") as st:
        rep = WS.build(st, cfg, run_dir)
    v, p = rep["viruses"], rep["plants"]
    assert v["totals"]["genomes"] == 2 and v["totals"]["genes"] == 80 + 210 and v["totals"]["genes_estimated_genomes"] == 1   # 501 once; 502 by its length
    assert sum(x["genomes"] for x in v["by_split"].values()) == 2 and v["left_out"] == {"host unknown (no host record of the species)": 1}
    assert p["totals"]["species"] == 1 and p["totals"]["genes"] == 8000                    # one per species: the chromosome level first
    assert p["left_out"].get("no protein-coding genes in the annotation") == 1 and p["by_level"] == {"Chromosome": dict(genomes=1, genes=8000)}
    assert p["totals"]["cds_bytes"] == 8000 * 1300
    cur = rep["current"]["viruses.eukaryotic"]
    assert cur["train"]["genomes"] == sum(1 for m in metas if m["group"] == "viruses.eukaryotic" and m["split"] == "train") and cur["train"]["genes"] > 0
    lock = open(os.path.join(run_dir, WS.OUT, "lock.yaml")).read()
    assert "sha256" in lock and "complete_genome" in lock
    text = json.dumps(rep) + open(os.path.join(run_dir, WS.OUT, "report.md")).read()
    assert not any(x in text for x in TS.SEALED)
    with Stage(run_dir, "gates_rel_w13_supply") as st:
        WS.gate(st, cfg, run_dir)
    assert json.load(open(os.path.join(run_dir, "gates", "G4_rel_w13_supply.json")))["status"] == "pass"


def test_a_new_unit_takes_the_split_of_the_hash_rule():
    seen = {WS.unit_split(("g", k)) for k in range(200)}
    assert seen == {"train", "heldout", "test"}
    held = sum(1 for k in range(2000) if WS.unit_split(("g", k)) == "heldout") / 2000
    assert 0.15 < held < 0.25                                             # about one unit in five
    assert WS.place("x", 5, 7, "a", {"x": {("g", 5): "train"}}) == "train"      # an existing genus keeps its split
    with pytest.raises(SystemExit, match="select_from"):
        WS.settings(dict(rel_w13_supply=dict(source="s")))
