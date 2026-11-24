"""Stage rel_w13_xpick (ci/stages/rel/w13_xpick.py): the supplement of eukaryotic viruses and plants and algae, picked
from a stand-in supply run (the records of test_rel_w13_supply and a second alga); the split made again by stage
train_data's three_way(); and the NCBI Datasets path of stages fetch and corpus (a zip file with the coding sequences
and the sequence report of an assembly)."""
import io
import json
import os
import zipfile

import pytest
import yaml

import rel_fixtures as RF
import test_rel_w13_census as T13
import test_rel_w13_supply as T
import test_rel_w2_stage as TS
from ci.stages.common import Stage
from ci.stages.rel import w13_census as WC
from ci.stages.rel import w13_pick as WP
from ci.stages.rel import w13_supply as WS
from ci.stages.rel import w13_xpick as WX

ALGA2 = 90000014
PLANTS = T.PLANT_RECS + [T.rec("GCA_000000605.1", ALGA2, 7000, level="Chromosome")]
V501, V502 = "virus_eukaryotic_GCA_000000501.1", "virus_eukaryotic_GCA_000000502.1"
A602, A605 = "plants_algae_chondrus_fixture_GCA_000000602.1", "plants_algae_chondrus_second_fixture_GCA_000000605.1"


class NoRecord(Exception):
    """Stands in for panel.NoRecord (ENA has no such assembly record)."""


def taxonomy():
    t = T.taxonomy()
    t.parent[ALGA2], t.rank[ALGA2], t.sci[ALGA2] = 2768, "species", "Chondrus second fixture"
    return t


def patch_inputs(monkeypatch, tmp_path):
    from ci.stages import catalog as CAT
    from ci.taxonomy import Taxonomy
    TS.patch(monkeypatch, tmp_path)
    ft = taxonomy()
    monkeypatch.setattr(Taxonomy, "__init__", lambda self, folder, names=True: self.__dict__.update(ft.__dict__))
    monkeypatch.setattr(WC, "host_records", lambda run: ([dict(virus_taxid=T.NEW_VIRUS, host_taxid=9258), dict(virus_taxid=10377, host_taxid=9258),
                                                          dict(virus_taxid=10702, host_taxid=2320)], [dict(id="hosts", records=3, sha256="0" * 64)]))

    def snap(c, done, cas, fetch=None):
        data, n = CAT.canonical(T.VIRUS_RECS if c["id"] == "viruses" else PLANTS, "accession")
        return data, n, dict(total=n, pages=1)
    monkeypatch.setattr(CAT, "snapshot", snap)


@pytest.fixture()
def made(tmp_path, monkeypatch):
    src = str(tmp_path / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    metas = TS.make_source(src)
    sel, _, _ = T13.make_selection(tmp_path, metas)
    spec = tmp_path / "corpus-src.yaml"                                  # the source run's corpus: one source per corpus genome
    spec.write_text("# Corpus src (a fixture).\n" + yaml.safe_dump(dict(name="src", tier="tiny", groups={"bacteria": {"min_genomes": 1}}, sources=[
        dict(id=m["id"], source_kind=TS.TINY[m["group"]][1], taxid=m["taxid"]) for m in metas]), sort_keys=False))
    patch_inputs(monkeypatch, tmp_path)
    sup, scfg = RF.rel_run(tmp_path, kind="rel_corpus", stages=("w13_supply",), run_id="20261124T000010-supply", corpus_spec=str(spec),
                           data_root=str(tmp_path / "store"), rel_w13_supply=dict(select_from=sel, source=src, expect=TS.SIZES))
    with Stage(sup, "rel_w13_supply") as st:
        WS.build(st, scfg, sup)
    rec = json.load(open(os.path.join(sup, "record.json")))
    json.dump(dict(rec, status="passed"), open(os.path.join(sup, "record.json"), "w"))
    fixed = {m["id"]: m["split"] for m in metas}
    want = {V501: "train", V502: "heldout", A602: "test", A605: "train"}
    from ci.stages import train_data as TD
    monkeypatch.setattr(TD, "three_way", lambda entries, groups, lin, cluster=None: {e: fixed.get(e) or want[e] for e in entries})
    return dict(tmp=tmp_path, src=src, metas=metas, sel=sel, spec=str(spec), supply=sup)


def run_xpick(made, monkeypatch, **extra):
    def replicons(acc):
        if acc == "GCA_000000502.1":
            raise NoRecord(acc)
        return [("SEQ%s.1" % acc[-5:-2], "Chromosome")]
    monkeypatch.setattr(WP, "replicons", replicons)
    run_dir, cfg = RF.rel_run(made["tmp"], kind="rel_corpus", stages=("w13_supply",), run_id="20261124T000020-xpick", corpus_spec=made["spec"],
                              rel_w13_xpick=dict(dict(supply_from=made["supply"], select_from=made["sel"], source=made["src"], expect=TS.SIZES, name="src-x"), **extra))
    with Stage(run_dir, "rel_w13_xpick") as st:
        rep = WX.build(st, cfg, run_dir)
    return run_dir, cfg, rep


def test_the_supplement_takes_the_open_units_only(made, monkeypatch):
    run_dir, cfg, rep = run_xpick(made, monkeypatch, max_unavailable=0.6)
    out = os.path.join(run_dir, WX.OUT)
    rel = json.load(open(os.path.join(out, "relatives.json")))
    assert [r["id"] for r in rel["relatives"]] == [A605, V501] and {r["split"] for r in rel["relatives"]} == {"train"}
    assert rel["unavailable"] == [dict(id=V502, accession="GCA_000000502.1", reason="no ENA record")]
    spec = yaml.safe_load(open(os.path.join(out, "corpus.yaml")))
    assert spec["name"] == "src-x" and len(spec["sources"]) == len(made["metas"]) + 2
    by = {x["id"]: x for x in spec["sources"]}
    assert by[V501]["ena_sequences"] == ["SEQ501.1"] and by[V501]["accession"] == "GCA_000000501.1" and by[V501]["paired_accession"] == "GCF_000000501.1"
    assert by[V501]["source_kind"] == "raw_virus" and "ena_assembly" not in by[V501]
    assert by[A605]["ncbi_datasets"] == dict(accession="GCA_000000605.1", include=["CDS_FASTA", "SEQUENCE_REPORT"])
    assert by[A605]["source_kind"] == "annotated_cds" and by[A605]["max_genes"] == 1500 and by[A605]["require_start_stop"] is True
    v, p = rep["counts"]["viruses"], rep["counts"]["plants"]
    assert v["candidates"]["train, annotated"]["genes"] == 80 and v["candidates"]["heldout, not annotated"]["genomes"] == 1
    assert v["picked"] == {"train, annotated": dict(genomes=1, genes=80, bases=150000)}
    assert p["candidates"]["test"]["genomes"] == 1 and p["picked"]["train"]["genes"] == 7000 and p["picked_capped"]["train"]["genes"] == 1500
    assert rep["test_unit_candidates"] == 1 and rep["lookups"]["unavailable"] == 1 and rep["lookups"]["viruses"] == 2
    text = "".join(open(os.path.join(out, n)).read() for n in ("relatives.json", "report.md")) + json.dumps(rep)
    assert "000000602" not in text + open(os.path.join(out, "corpus.yaml")).read()          # a test unit's genome is not named
    assert not any(x in text for x in TS.SEALED)                     # the sealed genomes: only in the corpus spec, as in the source's
    with Stage(run_dir, "gates_rel_w13_xpick") as st:
        WX.gate(st, cfg, run_dir)
    assert json.load(open(os.path.join(run_dir, "gates", "G4_rel_w13_xpick.json")))["status"] == "pass"


def test_too_many_unavailable_viruses_fail_the_gate(made, monkeypatch):
    run_dir, cfg, rep = run_xpick(made, monkeypatch)
    assert rep["lookups"]["unavailable_share"] == 0.5
    from ci.gates import GateFailure
    with Stage(run_dir, "gates_rel_w13_xpick") as st, pytest.raises(GateFailure, match="unavailable"):
        WX.gate(st, cfg, run_dir)


def test_the_split_is_stage_train_datas_own():
    from ci.stages import train_data as TD
    old = [dict(id="v%02d" % k, group="viruses.eukaryotic", ranks=[1, 2, 3, 4, 100 + k // 2, 5000 + k], cluster="", source="corpus") for k in range(40)]
    sp = TD.three_way([o["id"] for o in old], {o["id"]: o["group"] for o in old}, {o["id"]: dict(zip(TD.RANKS, o["ranks"])) for o in old}, {})
    for o in old:
        o["split"] = sp[o["id"]]

    class Tax:
        def at_rank(self, t, r):
            return dict(phylum=1, **{"class": 2}, order=3, family=4, genus=t // 100, species=t)[r]
    cands = [dict(id="same_genus", group="viruses.eukaryotic", taxid=10300 + 7)] + [dict(id="new%d" % k, group="viruses.eukaryotic", taxid=(900 + k) * 100 + 1)
                                                                                     for k in range(30)]
    moved, split = WX.splits_of(old, cands, Tax())
    assert moved == 0 and split["same_genus"] == sp["v06"]                     # genus 103: genomes v06 and v07
    assert all(split["new%d" % k] == WS.unit_split(("g", 900 + k)) for k in range(30))
    assert {split["new%d" % k] for k in range(30)} == {"train", "heldout", "test"}
    wrong = [dict(o, split="train" if o["split"] != "train" else "heldout") if o["id"] == "v00" else o for o in old]
    assert WX.splits_of(wrong, cands, Tax())[0] == 1


# ---- the NCBI Datasets path of stages fetch and corpus -----------------------------------------------------------

def _rec(name, seq, tags):
    return ">lcl|%s %s\n%s\n" % (name, " ".join("[%s=%s]" % kv for kv in tags), seq)


GOOD = "ATG" + "GCT" * 40 + "TAA"
CDS = "".join([
    _rec("NC_1.1_cds_XP_1.1_1", GOOD, [("gene", "A"), ("db_xref", "GeneID:1"), ("protein_id", "XP_1.1"), ("location", "100..225"), ("gbkey", "CDS")]),
    _rec("NC_1.1_cds_XP_2.1_2", "ATG" + "GCT" * 50 + "TAA", [("gene", "A"), ("db_xref", "GeneID:1"), ("location", "join(100..200,300..355)")]),
    _rec("NC_1.1_cds_3", GOOD, [("gene", "B"), ("db_xref", "GeneID:2"), ("pseudo", "true"), ("location", "400..525")]),
    _rec("NC_MT.1_cds_YP_1.1_4", GOOD, [("gene", "C"), ("db_xref", "GeneID:3"), ("location", "10..135"), ("transl_table", "1")]),
    _rec("NC_1.1_cds_XP_5.1_5", GOOD[:-1], [("db_xref", "GeneID:5"), ("location", "600..724")]),
    _rec("NC_1.1_cds_XP_6.1_6", "ATG" + "GCT" * 20 + "TAA" + "GCT" * 20 + "TAA", [("db_xref", "GeneID:6"), ("location", "800..928")]),
    _rec("NC_1.1_cds_XP_7.1_7", "ATG" + "GCT" * 10 + "TAA", [("db_xref", "GeneID:7"), ("location", "1000..1038")]),
    _rec("NC_1.1_cds_XP_8.1_8", "GCT" * 41 + "TAA", [("db_xref", "GeneID:8"), ("location", "1100..1225"), ("partial", "5'")]),
    _rec("NC_1.1_cds_XP_9.1_9", "ATG" + "GCT" * 41, [("db_xref", "GeneID:9"), ("location", "1300..1425"), ("partial", "3'")]),
    _rec("NC_2.1_cds_XP_10.1_10", GOOD, [("db_xref", "GeneID:10"), ("location", "complement(50..175)")]),
    _rec("NC_1.1_cds_XP_11.1_11", GOOD, [("db_xref", "GeneID:11"), ("location", "5000..5125")]),
    _rec("NC_1.1_cds_ABC_12.1_12", GOOD, [("locus_tag", "LT_12"), ("location", "300..425")]),
    _rec("NC_1.1_cds_XP_13.1_13", GOOD, [("db_xref", "GeneID:13"), ("location", "6000..6125"), ("transl_table", "11")]),
])
REPORT = [dict(refseqAccession="NC_1.1", genbankAccession="CM_1.1", assignedMoleculeLocationType="Chromosome", role="assembled-molecule"),
          dict(refseqAccession="NC_2.1", assignedMoleculeLocationType="Chromosome"),
          dict(refseqAccession="NC_MT.1", assignedMoleculeLocationType="Mitochondrion")]


def ncbi_zip(acc, cds=CDS, report=REPORT, parts=("cds", "report")):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", "NCBI Datasets fixture")
        if "cds" in parts:
            z.writestr("ncbi_dataset/data/%s/cds_from_genomic.fna" % acc, cds)
        if "report" in parts:
            z.writestr("ncbi_dataset/data/%s/sequence_report.jsonl" % acc, "".join(json.dumps(r) + "\n" for r in report))
    return buf.getvalue()


def test_fetch_keeps_an_ncbi_datasets_zip_file(tmp_path, monkeypatch):
    from ci.stages import fetch
    acc = "GCA_000000605.1"
    src = dict(id=A605, ncbi_datasets=dict(accession=acc, include=["CDS_FASTA", "SEQUENCE_REPORT"]))
    urls = []
    monkeypatch.setattr(fetch, "get", lambda url, **kw: urls.append(url) or ncbi_zip(acc))
    files, prov, hit = fetch.fetch_source(src, str(tmp_path / "cas"), {})
    assert files[0]["path"].endswith(".ncbi.zip") and files[0]["kind"] == "ncbi.zip" and not hit
    assert urls == ["https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/%s/download?include_annotation_type=CDS_FASTA&include_annotation_type=SEQUENCE_REPORT" % acc]
    assert [m["part"] for m in prov[0]["members"]] == ["CDS_FASTA", "SEQUENCE_REPORT"] and fetch.describe(src) == {"ncbi_datasets": src["ncbi_datasets"]}
    assert fetch.describe(dict(id="e", ensembl=dict(species="x"))) == {"ensembl": dict(species="x")}       # other sources: as before
    monkeypatch.setattr(fetch, "get", lambda url, **kw: ncbi_zip(acc, parts=("report",)))
    with pytest.raises(fetch.NoRecord, match="cds_from_genomic"):
        fetch.fetch_source(src, str(tmp_path / "cas2"), {})
    monkeypatch.setattr(fetch, "get", lambda url, **kw: urls.append(url) or b"<html>busy</html>")
    urls.clear()
    with pytest.raises(RuntimeError, match="zip"):
        fetch.fetch_source(src, str(tmp_path / "cas3"), {})
    assert len(urls) == 3                                                     # a damaged zip file is asked for twice more
    with pytest.raises(SystemExit, match="unknown parts"):
        fetch.fetch_source(dict(src, ncbi_datasets=dict(accession=acc, include=["GENOME_FASTA"])), str(tmp_path / "cas4"), {})


def test_corpus_reads_the_coding_sequences_of_an_ncbi_datasets_file(tmp_path):
    pytest.importorskip("Bio")
    from ci.stages import corpus as C
    p = tmp_path / "x.ncbi.zip"
    p.write_bytes(ncbi_zip("GCA_000000605.1"))
    genes, mode, q = C.ncbi_cds(str(p), 1, max_genes=3)
    assert [g["gene_id"] for g in genes] == ["GeneID_1", "LT_12", "GeneID_11"] and mode == C.NCBI_MODE[True] and "ATG" in mode
    assert len(genes[0]["protein"]) == 51 and genes[0]["start"] == 100 and genes[0]["end"] == 355     # the longest coding sequence of a gene
    assert q["records"] == 13 and q["pseudogene"] == 1 and q["organelle"] == 1 and q["organelle_by_type"] == {"Mitochondrion": 1}
    assert q["another_genetic_code"] == 1 and q["other_isoforms"] == 1 and q["genes"] == 9 and q["partial"] == 2
    assert (q["length_not_multiple_of_3"], q["stop_codon_inside"], q["fewer_than_30_codons"], q["no_atg_start"], q["no_stop_codon"]) == (1, 1, 1, 1, 1)
    assert q["passed"] == 4 and q["kept"] == 3 and q["beyond_max_genes"] == 1
    assert q["left_out_start_stop"] == 2 and q["left_out_start_stop_share"] == round(2 / 9, 4)
    loose, mode2, q2 = C.ncbi_cds(str(p), 1, require_start_stop=False)
    assert mode2 == "annotated (NCBI Datasets)"
    assert q2["passed"] == 6 and q2["no_atg_start"] == 0 and [g["gene_id"] for g in loose][-1] == "GeneID_10"
    assert [g["strand"] for g in loose if g["gene_id"] == "GeneID_10"] == ["-"]
    out = tmp_path / "g"
    genes, _ = C.build_one(dict(id=A605, source_kind="annotated_cds", taxid=ALGA2, files=[str(p)], max_genes=2), str(out))
    assert len(genes) == 2 and json.load(open(out / "cds_quality.json"))["kept"] == 2
    with pytest.raises(SystemExit, match="replicons"):
        C.build_one(dict(id=A605, source_kind="annotated_cds", taxid=ALGA2, files=[str(p)], replicons=["NC_1.1"]), str(tmp_path / "h"))


def test_the_build_names_genomes_that_lose_many_genes_to_the_start_stop_rule(tmp_path):
    from ci.stages.rel import w13_build as WB
    d = tmp_path / "data" / "corpus"
    d.mkdir(parents=True)

    def q(genes, lost):
        return dict(records=genes, genes=genes, kept=min(1500, genes - lost), no_atg_start=lost // 2, no_stop_codon=lost - lost // 2,
                    left_out_start_stop=lost, left_out_start_stop_share=round(lost / genes, 4), organelle_by_type={})
    ents = [dict(id="a", cds_quality=q(1000, 50), gene_calling="x"), dict(id="b", cds_quality=q(1000, 300), gene_calling="x"), dict(id="c")]
    json.dump(dict(entries=ents, meta=dict(cds_rules="the rules")), open(d / "manifest.json", "w"))
    rep = WB.cds_report(str(tmp_path), [dict(id="a"), dict(id="b"), dict(id="c")], 0.2)
    assert rep["genomes"] == 2 and [x["id"] for x in rep["flagged"]] == ["b"] and rep["flagged"][0]["share"] == 0.3
    assert rep["totals"]["no_atg_start"] == 25 + 150 and rep["rules"] == "the rules" and WB.cds_report(str(tmp_path), [dict(id="c")], 0.2) is None
    page = WB.page_of(dict(upto="search", note="n", corpus_spec="s", corpus_spec_sha256="0" * 64, pick=dict(run_id="p"), totals=dict(relatives=3, unavailable=[]),
                           fetch={}, search=None, pinned=None, counts={}, cds=rep, disk=dict(run_gb={}, free_gb=1, total_gb=2), problems=[]))
    assert "| b | 1,000 | 300 | 30.0% | 700 |" in page and "Genomes above 20% (their annotation is doubtful; kept, named here): 1." in page
