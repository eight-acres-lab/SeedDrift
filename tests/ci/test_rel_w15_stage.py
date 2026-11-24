"""Package W15: stage rel_w15_registry (ci/stages/rel/w15_registry.py), with a small taxonomy and no network."""
from __future__ import annotations

import io
import json

import pytest

from ci.stages.rel import w15_registry as W
from tests.ci.test_rel_w15_registry import _row

# root 1 > Eukaryota 2 > Metazoa 3 > Insecta 4 > Drosophila 5 (genus) > D. ananassae 6 ; Insecta 4 > Aphis 7 (genus)
# root 1 > Bacteria 10 > Pseudomonadaceae 11 (family) > Pseudomonas 12 (genus) ; Bacteria 10 > Neisseria 13 (genus)
NODES = {1: (1, "no rank", "root"), 2: (1, "superkingdom", "Eukaryota"), 3: (2, "kingdom", "Metazoa"),
         4: (3, "class", "Insecta"), 5: (4, "genus", "Drosophila"), 6: (5, "species", "Drosophila ananassae"),
         7: (4, "genus", "Aphis"), 10: (1, "superkingdom", "Bacteria"), 11: (10, "family", "Pseudomonadaceae"),
         12: (11, "genus", "Pseudomonas"), 13: (10, "genus", "Neisseria"), 14: (1, "no rank", "Aphis"),
         20: (5, "subgenus", "Sophophora"), 21: (20, "species group", "Drosophila melanogaster group"),
         22: (6, "varietas", "Drosophila ananassae var. pallida"), 23: (12, "species", "Pseudomonas sp. VKM F-3808")}


class Tax:
    def __init__(self):
        self.parent = {t: p for t, (p, _, _) in NODES.items()}
        self.rank = {t: r for t, (_, r, _) in NODES.items()}
        self.sci = {t: n for t, (_, _, n) in NODES.items()}
        self.by_name = {}
        for t, n in self.sci.items():
            self.by_name.setdefault(n, []).append(t)

    def resolve(self, t):
        if int(t) not in self.parent:
            raise KeyError(t)
        return int(t)

    def lineage(self, t):
        t, out = self.resolve(t), []
        while True:
            out.append(t)
            if self.parent[t] == t:
                return out
            t = self.parent[t]

    def name(self, t):
        return self.sci[t]

    def at_rank(self, t, rank):
        return next((x for x in self.lineage(t) if self.rank[x] == rank), 0)

    def genus_by_name(self, name, domains=(10,)):
        return [t for t in self.by_name.get(name, []) if self.rank[t] == "genus" and set(self.lineage(t)) & set(domains)]


def test_taxa_resolve_rank_lineage_and_stated_taxid():
    rows = [_row("E1", "Drosophila (genus)", "Eukaryota;Metazoa;Arthropoda;Insecta;Drosophila", recipient_exclusion_taxid="99"),
            _row("E2", "Aphis (genus)", "Eukaryota;Metazoa;Insecta;Aphis"),
            _row("E3", "Insecta (order)", "Eukaryota;Metazoa;Insecta"),
            _row("E4", "Nothing (genus)", "Eukaryota;Nothing")]
    got = W.taxa(rows, Tax())
    assert got["E1"]["status"] == "ok" and got["E1"]["taxid"] == 5
    assert got["E1"]["names_not_in_ncbi"] == ["Arthropoda"] and got["E1"]["stated_taxid_differs"] == 99
    assert got["E2"]["status"] == "ok" and got["E2"]["taxid"] == 7       # the genus, not the no-rank homonym
    assert got["E3"]["status"] == "rank_differs" and got["E3"]["ncbi_ranks"] == ["class"]
    assert got["E4"]["status"] == "unresolved"


def test_pretraining_reads_only_train_and_heldout():
    found = {"E1": dict(status="ok", taxid=5), "E2": dict(status="ok", taxid=13), "E3": dict(status="unresolved")}
    index = dict(genomes=[dict(split="train", taxid=6), dict(split="heldout", taxid=5), dict(split="test", taxid=6),
                          dict(split="train", taxid=12), dict(split="test", taxid="not a taxid")])
    got = W.pretraining(found, index, Tax())
    assert got["E1"] == dict(train=1, heldout=1, in_pretraining="y", unresolved_genomes=0)
    assert got["E2"]["in_pretraining"] == "n" and "E3" not in got


def test_sealed_genera_and_overlap(tmp_path):
    p = tmp_path / "EXCLUDE.tsv"
    p.write_text("# c\ndate\tkind\tname\treason\n2026-11-16\tgenus\tPseudomonas\tsealed genus of the island anchor (and its family)\n"
                 "2026-11-17\torder\tEnterobacterales\tthe development order\n")
    tax = Tax()
    sealed = W.sealed_genera(tax, str(p))
    assert sealed == {12: "sealed genus Pseudomonas", 11: "family of sealed genus Pseudomonas"}
    got = W.sealed_overlap({"E1": dict(status="ok", taxid=10), "E2": dict(status="ok", taxid=13),
                            "E3": dict(status="ok", taxid=12)}, sealed, tax)
    assert got["E1"]["sealed_overlap"] == "y" and got["E2"]["sealed_overlap"] == "n" and got["E3"]["sealed_overlap"] == "y"


def test_dois_collects_from_papers_and_later():
    rows = [_row("E1", "Aphis (genus)", "Insecta;Aphis", papers="10.1126/science.1142490;10.1186/1471-2164-15-1097",
                 later="support:10.1093/gbe/evv158 (x);challenge:G3 7:2439 (no doi)")]
    assert W.dois(rows) == ["10.1126/science.1142490", "10.1186/1471-2164-15-1097", "10.1093/gbe/evv158"]


EMPTY_LISTING = (b'<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                 b'<KeyCount>0</KeyCount></ListBucketResult>')


class Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_stores_meta_fulltext_and_capped_supplements(tmp_path):
    meta = json.dumps(dict(resultList=dict(result=[dict(pmcid="PMC1", isOpenAccess="Y", pmid="9", title="t", pubYear="2010")])))
    seen = []

    def opener(req, timeout):
        seen.append(req.full_url)
        if req.full_url.startswith(W.S3):
            return Resp(EMPTY_LISTING)
        if "search?" in req.full_url:
            return Resp(meta.encode())
        if req.full_url.endswith("fullTextXML"):
            return Resp(b"<article/>")
        return Resp(b"PK" + b"x" * 2000)

    rec = W.fetch("10.1/x", str(tmp_path / "p"), opener=opener, supplements=True, max_supplement_mb=0.001, pause_s=0,
                  sleep=lambda s: None)
    assert rec["pmcid"] == "PMC1" and rec["fulltext"]["bytes"] == 10 and len(rec["fulltext"]["sha256"]) == 64
    assert "not stored" in rec["supplements"]["error"] and rec["supplements"]["error_type"] == "TooLarge"
    assert W.failures([rec]) == {"supplements": {"TooLarge 200": 1}}
    assert (tmp_path / "p" / "meta.json").exists() and not (tmp_path / "p" / "supplements.zip").exists()
    assert len(seen) == 4 and seen[1].startswith(W.S3)              # metadata, the open-data listing, then Europe PMC


def test_settings_refuse_the_catalogue_path():
    with pytest.raises(SystemExit):
        W.settings({W.STAGE: dict(registry="bench/catalogue/events.tsv")})
    with pytest.raises(SystemExit):
        W.settings({W.STAGE: dict(registry="bench/rel/registry/events_v0.2-sample.tsv", scores="x")})
    s = W.settings({W.STAGE: dict(registry="bench/rel/registry/events_v0.2-sample.tsv", supplements=["10.1/A"])})
    assert s["supplements"] == ["10.1/a"] and s["fetch"] is False and s["registry"] == ["bench/rel/registry/events_v0.2-sample.tsv"]
    with pytest.raises(SystemExit):
        W.settings({W.STAGE: dict(registry=["bench/rel/registry/a.tsv", "bench/catalogue/events.tsv"])})


class Chunked(Resp):
    """A response whose read(-1) returns the chunk framing, as http.client does for a chunked body."""

    def read(self, amt=None):
        body = super().read() if amt is None else super().read(amt)
        if amt is not None and amt < 0:
            return b"1f\r\n" + body + b"\r\n0\r\n\r\n"
        return body


def test_fetch_reads_the_whole_body_and_parses_control_characters(tmp_path):
    meta = '{"resultList":{"result":[{"pmcid":"PMC2","title":"a\tb\nc","isOpenAccess":"N"}]}}'

    def opener(req, timeout):
        return Chunked(meta.encode() if "search?" in req.full_url else b"<article/>")

    rec = W.fetch("10.1/y", str(tmp_path / "q"), opener=opener, pause_s=0, sleep=lambda s: None)
    assert rec.get("found") is True and rec["pmcid"] == "PMC2" and rec["fulltext"]["bytes"] == 10


def test_taxa_keep_the_taxid_when_only_the_rank_differs():
    rows = [_row("E1", "Insecta (order)", "Eukaryota;Metazoa;Insecta")]
    got = W.taxa(rows, Tax())["E1"]
    assert got["status"] == "rank_differs" and got["taxid"] == 4 and got["ncbi_rank"] == "class"
    assert W.pretraining({"E1": got}, dict(genomes=[dict(split="train", taxid=6)]), Tax())["E1"]["in_pretraining"] == "y"


def test_gate_fails_on_a_wholesale_fetch_failure(tmp_path, monkeypatch):
    import ci.stages.rel.w15_registry as mod
    run = tmp_path / "run"
    out = run / mod.OUT
    out.mkdir(parents=True)
    papers = [dict(doi="10.1/%d" % i, meta=dict(error="x")) for i in range(4)] + [dict(doi="10.1/9", found=True)]
    (out / "report.json").write_text(json.dumps(dict(errors=[], registry_sha256={}, papers=papers, records=0, taxa={})))
    seen = []
    monkeypatch.setattr(mod, "require", lambda result, run_dir: seen.append(result))
    mod.gate(None, {}, str(run))
    assert seen[0].status == "fail" and "systematic fetch failure" in json.dumps(seen[0].evidence)


def test_dois_keep_balanced_parentheses():
    rows = [_row("E1", "Aphis (genus)", "Insecta;Aphis", papers="10.1016/S1473-3099(15)00424-7;10.1000/a",
                 later="support:10.1128/aac.00129-17 (from Moraxella);comment:(see 10.1000/b)")]
    assert W.dois(rows) == ["10.1016/s1473-3099(15)00424-7", "10.1000/a", "10.1128/aac.00129-17", "10.1000/b"]


def test_supplement_error_bodies_are_retried_and_never_stored(tmp_path):
    meta = json.dumps(dict(resultList=dict(result=[dict(pmcid="PMC3")])))
    answers = [b"<?xml version='1.0'?><errorBean>Connection refused</errorBean>", b"<errorBean/>", b"PK\x03\x04zip"]
    waited = []

    def opener(req, timeout):
        if req.full_url.startswith(W.S3):
            return Resp(EMPTY_LISTING)
        if "search?" in req.full_url:
            return Resp(meta.encode())
        if req.full_url.endswith("fullTextXML"):
            raise W.urllib.error.HTTPError(req.full_url, 500, "Internal Server Error", {}, None)
        return Resp(answers.pop(0))

    rec = W.fetch("10.1/z", str(tmp_path / "z"), opener=opener, supplements=True, pause_s=0, sleep=waited.append)
    assert rec["supplements"]["bytes"] == 7 and [w for w in waited if w] == [10, 30, 10, 30]  # full text, then supplements
    assert rec["fulltext"]["status"] == 500 and "after retries" in rec["fulltext"]["note"]
    answers[:] = [b"<errorBean/>"] * 3
    rec = W.fetch("10.1/z", str(tmp_path / "z2"), opener=opener, supplements=True, pause_s=0, sleep=lambda s: None)
    assert "not a zip archive" in rec["supplements"]["error"] and not (tmp_path / "z2" / "supplements.zip").exists()


def test_a_stated_taxid_settles_homonyms():
    tax = Tax()
    rows = [_row("E1", "Aphis (no rank)", "root;Aphis", recipient_exclusion_taxid="14")]
    assert W.taxa(rows, tax)["E1"]["taxid"] == 14


def test_fulltext_error_bodies_are_retried(tmp_path):
    meta = json.dumps(dict(resultList=dict(result=[dict(pmcid="PMC4")])))
    answers = [b"<?xml version='1.0'?><ns4:errorBean>refused</ns4:errorBean>", b"<article>ok</article>"]

    def opener(req, timeout):
        return Resp(meta.encode() if "search?" in req.full_url else answers.pop(0))

    rec = W.fetch("10.1/w", str(tmp_path / "w"), opener=opener, pause_s=0, sleep=lambda s: None)
    assert rec["fulltext"]["bytes"] == len(b"<article>ok</article>")


def test_a_body_cut_short_is_retried_and_then_recorded(tmp_path):
    import http.client
    meta = json.dumps(dict(resultList=dict(result=[dict(pmcid="PMC5")])))
    state = dict(cuts=1)

    def opener(req, timeout):
        if "search?" in req.full_url:
            return Resp(meta.encode())
        if req.full_url.endswith("fullTextXML"):
            return Resp(b"<article/>")
        if state["cuts"]:
            state["cuts"] -= 1
            raise http.client.IncompleteRead(b"PK partial", 1000)
        return Resp(b"PK\x03\x04whole")

    rec = W.fetch("10.1/v", str(tmp_path / "v"), opener=opener, supplements=True, pause_s=0, sleep=lambda s: None)
    assert rec["supplements"]["bytes"] == len(b"PK\x03\x04whole")
    state["cuts"] = 99
    rec = W.fetch("10.1/v", str(tmp_path / "v2"), opener=opener, supplements=True, pause_s=0, sleep=lambda s: None)
    assert rec["supplements"]["error_type"] == "IncompleteRead" and rec["supplements"]["status"] is None
    assert rec["fulltext"]["bytes"] == 10


def test_fetch_all_records_an_unexpected_failure_and_goes_on(tmp_path):
    def one(d, dest, **kw):
        if d == "10.1000/bad":
            raise RuntimeError("boom")
        return dict(doi=d, found=True)

    s = dict(supplements=[], max_supplement_mb=1, pause_s=0)
    got = W.fetch_all(["10.1000/a", "10.1000/bad", "10.1000/c"], str(tmp_path), s, fetch_one=one, sleep=lambda x: None)
    assert [g["doi"] for g in got] == ["10.1000/a", "10.1000/bad", "10.1000/c"]
    assert got[1]["fetched"] is False and got[1]["error_type"] == "RuntimeError"


def test_failures_are_counted_by_type_and_last_status(tmp_path):
    meta = json.dumps(dict(resultList=dict(result=[dict(pmcid="PMC6")])))

    def opener(req, timeout):
        if "search?" in req.full_url:
            return Resp(meta.encode())
        raise W.urllib.error.HTTPError(req.full_url, 500, "Internal Server Error", {}, None)

    rec = W.fetch("10.1/u", str(tmp_path / "u"), opener=opener, supplements=True, pause_s=0, sleep=lambda s: None)
    assert rec["fulltext"]["error_type"] == "HTTPError" and rec["fulltext"]["status"] == 500
    papers = [rec, dict(doi="10.1000/x", error_type="RuntimeError", status=None, fetched=False)]
    assert W.failures(papers) == {"fulltext": {"HTTPError 500": 1}, "supplements": {"HTTPError 500": 1},
                                  "paper": {"RuntimeError": 1}}


def test_the_gate_still_fails_when_the_guard_records_every_paper(tmp_path, monkeypatch):
    import ci.stages.rel.w15_registry as mod

    def one(d, dest, **kw):
        raise RuntimeError("down")

    s = dict(supplements=[], max_supplement_mb=1, pause_s=0)
    papers = mod.fetch_all(["10.1000/%d" % i for i in range(5)], str(tmp_path), s, fetch_one=one, sleep=lambda x: None)
    run = tmp_path / "run"
    (run / mod.OUT).mkdir(parents=True)
    (run / mod.OUT / "report.json").write_text(json.dumps(dict(errors=[], registry_sha256={}, papers=papers, records=0, taxa={})))
    seen = []
    monkeypatch.setattr(mod, "require", lambda result, run_dir: seen.append(result))
    mod.gate(None, {}, str(run))
    assert seen[0].status == "fail" and "0 of 5" in json.dumps(seen[0].evidence)


LISTING = (b'<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
           b'<Contents><Key>PMC7.1/PMC7.1.xml</Key><Size>10</Size></Contents>'
           b'<Contents><Key>PMC7.2/PMC7.2.xml</Key><Size>12</Size></Contents>'
           b'<Contents><Key>PMC7.2/PMC7.2.txt</Key><Size>5</Size></Contents>'
           b'<Contents><Key>PMC7.2/PMC7.2.pdf</Key><Size>99</Size></Contents>'
           b'<Contents><Key>PMC7.2/x.sd01.xlsx</Key><Size>4</Size></Contents>'
           b'<Contents><Key>PMC7.2/xfig01.jpg</Key><Size>8</Size></Contents></ListBucketResult>')


def test_the_open_access_dataset_comes_first_latest_version_no_figures(tmp_path):
    meta = json.dumps(dict(resultList=dict(result=[dict(pmcid="PMC7", isOpenAccess="Y")])))
    seen = []

    def opener(req, timeout):
        seen.append(req.full_url)
        if "search?" in req.full_url:
            return Resp(meta.encode())
        if req.full_url.startswith(W.S3 + "?"):
            return Resp(LISTING)
        if req.full_url.startswith(W.S3):
            return Resp(b"<article>" + req.full_url.encode()[-12:] + b"</article>")
        raise AssertionError("Europe PMC asked although the open dataset had everything: " + req.full_url)

    rec = W.fetch("10.1/o", str(tmp_path / "o"), opener=opener, supplements=True, pause_s=0, sleep=lambda s: None)
    assert rec["pmc_oa"] == dict(version=2, files=3)
    assert rec["fulltext"]["source"] == "pmc-oa-opendata" and rec["fulltext"]["file"] == "PMC7.2.xml"
    assert [f["file"] for f in rec["supplements"]["files"]] == ["x.sd01.xlsx"] and rec["supplements"]["sha256"]
    assert not any(u.endswith((".jpg", ".pdf")) for u in seen)
    assert (tmp_path / "o" / "pmc_oa" / "x.sd01.xlsx").exists()


def test_s3_plan_without_the_article_in_the_dataset():
    assert W.s3_plan("PMC8", []) == {}
    assert W.s3_plan("PMC8", [("PMC80.1/PMC80.1.xml", 3)]) == {}           # another article's prefix is not this one


def test_no_supplement_in_the_dataset_falls_back_to_europe_pmc(tmp_path):
    meta = json.dumps(dict(resultList=dict(result=[dict(pmcid="PMC9")])))
    listing = (b'<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
               b'<Contents><Key>PMC9.1/PMC9.1.xml</Key><Size>10</Size></Contents>'
               b'<Contents><Key>PMC9.1/fig1.jpg</Key><Size>8</Size></Contents></ListBucketResult>')
    seen = []

    def opener(req, timeout):
        seen.append(req.full_url)
        if "search?" in req.full_url:
            return Resp(meta.encode())
        if req.full_url.startswith(W.S3 + "?"):
            return Resp(listing)
        if req.full_url.startswith(W.S3):
            return Resp(b"<article/>")
        if req.full_url.endswith("supplementaryFiles"):
            return Resp(b"PK\x03\x04epmc")
        raise AssertionError(req.full_url)

    rec = W.fetch("10.1/n", str(tmp_path / "n"), opener=opener, supplements=True, pause_s=0, sleep=lambda s: None)
    assert rec["fulltext"]["source"] == "pmc-oa-opendata" and rec["supplements"]["source"] == "europepmc"
    assert rec["pmc_oa"]["supplements"] == "none in the open-access dataset"


def test_annotation_lookup_finds_the_assembly_and_matches_locus_tags(tmp_path):
    rows = [_row("E1", "Aphis (genus)", "Insecta;Aphis", region="fragment A: ASEM_C4_12633, ASEM_C4_20551/2",
                 notes="PCK 'ASEM_AUS1_17510 on fragment L'")]
    assert W.registry_ids(rows) == ["ASEM_AUS1_17510", "ASEM_C4_12633", "ASEM_C4_20551"]
    pages = [json.dumps(dict(reports=[dict(annotation=dict(locus_tag="ASEM_AUS1_17510")), dict(annotation=dict(locus_tag="ASEM_AUS1_00001"))],
                             next_page_token="t2")),
             json.dumps(dict(reports=[dict(annotation=dict(locus_tag="ASEM_C4_12633"))]))]
    seen = []

    def opener(req, timeout):
        u = req.full_url
        seen.append(u)
        if "esearch" in u:
            return Resp(json.dumps(dict(esearchresult=dict(idlist=["111"]))).encode())
        if "elink" in u:
            return Resp(json.dumps(dict(linksets=[dict(linksetdbs=[dict(dbto="assembly", links=[222])])])).encode())
        if "esummary" in u:
            return Resp(json.dumps(dict(result={"uids": ["222"], "222": dict(assemblyaccession="GCA_1.1", assemblyname="x")})).encode())
        if "annotation_report" in u:
            return Resp(pages.pop(0).encode())
        raise AssertionError(u)

    rec = W.annotation_lookup(dict(wgs_master="QPGU00000000"), rows, str(tmp_path), opener=opener, sleep=lambda s: None, pause_s=0)
    assert rec["assembly"]["accession"] == "GCA_1.1" and rec["status"] == "annotated" and rec["genes"] == 3
    assert rec["matched"] == ["ASEM_AUS1_17510", "ASEM_C4_12633"] and rec["unmatched"] == ["ASEM_C4_20551"]
    assert rec["locus_tag_prefixes"][0] == ("ASEM_AUS1", 2) and len(rec["pages"]) == 2 and "page_token=t2" in seen[-1]
    assert all("sequence" not in u for u in seen)


def test_annotation_lookup_reports_a_missing_assembly(tmp_path):
    def opener(req, timeout):
        return Resp(json.dumps(dict(esearchresult=dict(idlist=[]))).encode())

    rec = W.annotation_lookup(dict(wgs_master="X"), [], str(tmp_path), opener=opener, sleep=lambda s: None, pause_s=0)
    assert rec["status"] == "no assembly found for the WGS master record"
    with pytest.raises(SystemExit):
        W.settings({W.STAGE: dict(registry="bench/rel/registry/a.tsv", annotation=dict(max_mb=1))})


def _index():
    return dict(genomes=[dict(id="bacteria_GCF_000000123.1", split="train", taxid=12, group="bacteria"),
                         dict(id="insects_drosophila_ananassae", split="heldout", taxid=6, group="eukaryota.invertebrates"),
                         dict(id="bacteria_GCF_000000555.1", split="test", taxid=12, group="bacteria"),
                         dict(id="fungi_hidden_species_gca_000000999", split="test", taxid=6, group="eukaryota.fungi"),
                         dict(id="plants_algae_nicotiana_attenuata", split="test", taxid=6, group="eukaryota.plants_algae")])


def test_latin_names_by_the_reviewers_rule():
    assert W._latin("bdelloid rotifers (Adineta vaga and others)") == ["adineta_vaga"]                     # EV016
    assert W._latin("diatoms (Thalassiosira pseudonana, Phaeodactylum tricornutum and others)") == [
        "thalassiosira_pseudonana", "phaeodactylum_tricornutum"]                                           # EV019
    assert W._latin("Candidatus Pelagibacter ubique") == ["pelagibacter_ubique"]                            # never 'candidatus'
    assert W._latin("Ca. Hodgkinia", single=True) == ["hodgkinia"] and W._latin("Candidatus Hodgkinia") == []
    assert W._latin("Candidatus", single=True) == [] and W._latin("Uncultured bacterium") == []
    for unnamed in ("Pseudomonas sp.", "Chaetothyriales sp. CBS 135597", "Pseudogymnoascus sp. VKM F-3808", "Vibrio spp."):
        assert W._latin(unnamed, single=True) == [], unnamed                    # the assembly numbers cover them
    assert W._latin("Pseudomonas sp. and Vibrio cholerae") == ["vibrio_cholerae"]
    assert W._latin("Wolbachia", single=True) == ["wolbachia"] and W._latin("Wolbachia") == []
    assert W._latin("Klosneuvirus and relatives") == [] and W._latin("Homo sapiens (and 25 other genomes)") == ["homo_sapiens"]
    smelt = "Osmerus mordax (rainbow smelt); also in Japanese smelt (Hypomesus nipponensis)"                 # EV024
    assert W._latin(smelt) == ["osmerus_mordax", "japanese_smelt", "hypomesus_nipponensis"]
    assert W._latin(smelt, genus=lambda g: g in ("Osmerus", "Hypomesus")) == ["osmerus_mordax", "hypomesus_nipponensis"]


def test_nucleotide_accessions_as_the_drafts_cite_them():
    text = "contig JN900499; scaffolds EQ111801, EQ113280; NC_009504.1 and NZ_CP012345.1; WGS ANGL00000000; M35027"
    assert W._nucleotide(text) == {"JN900499", "EQ111801", "EQ113280", "NC_009504", "NZ_CP012345", "ANGL00000000", "M35027"}
    assert W._nucleotide("Pfam PF00729, run EV012, PMC6410850, ST828, ASEM_AUS1_12345, GCA_004135705.1, SOAP18064315") == set()
    assert W._nucleotide("QPGU01000000 and QPGU01000001.1") == {"QPGU01000000", "QPGU01000001"}


def test_unseen_genomes_by_assembly_id_species_words_and_islands():
    u = W.unseen_genomes(_index(), {"NC_000002"})
    assert u["accs"] == {"000000555", "000000999"} and u["nucleotide"] == {"NC_000002"}
    assert ("nicotiana", "attenuata") in u["seqs"] and ("hidden", "species") in u["seqs"]
    assert ("plants", "algae", "nicotiana", "attenuata") in u["seqs"]
    assert not any("drosophila" in x for x in u["seqs"])                 # held-out genomes are not unseen
    names = W.Unseen(u)
    assert names("store/ensembl/Nicotiana_attenuata.IGv2.pep.fa") and names("old/nicotiana_attenuata/genes.gff")
    assert names("old/NicotianaAttenuata.fa") and names("old/x_nicotianaattenuata_v2")                      # joined
    assert names("x/GCA_000000555.2_genomic.fna") and names("islands/NC_000002.1.gbk") and names("islands/nc_000002.gbk")
    assert not names("x/nicotiana_tabacum.faa") and not names("islands/NC_0000021.gbk") and not names("x/NicotianaTabacum")
    assert W._species_words("bacteria_candidatus_pelagibacter_ubique", "x.bacteria") == ("pelagibacter", "ubique")


def _tok(radius=None, draft=None, genomes=0):
    z = lambda d: dict(accessions=set((d or {}).get("accessions", ())), names=set((d or {}).get("names", ())),
                       nucleotide=set((d or {}).get("nucleotide", ())))
    return dict(genomes=genomes, radius=z(radius), draft=z(draft))


def test_exposure_tokens_keep_the_radius_apart_from_the_draft():
    rows = [_row("E1", "Pseudomonas (genus)", "Bacteria;Pseudomonadaceae;Pseudomonas", recipient="Pseudomonas sp. (many strains)"),
            _row("E2", "Drosophila (genus)", "Eukaryota;Metazoa;Insecta;Drosophila", recipient="Drosophila ananassae (lines Hawaii, India)"),
            _row("E3", "Insecta (class)", "Eukaryota;Metazoa;Insecta", recipient="insects (several); also Japanese flies"),
            _row("E4", "Nothing (genus)", "Eukaryota;Nothing", recipient="unknown hosts", region="contig JN900499",
                 assembly="GCA_004135705.1"),
            _row("E5", "Sophophora (subgenus)", "Eukaryota;Metazoa;Insecta;Drosophila;Sophophora", recipient="flies"),
            _row("E6", "Drosophila melanogaster group (species group)", "Eukaryota;Insecta;Drosophila;Sophophora", recipient="flies"),
            _row("E7", "Drosophila ananassae var. pallida (varietas)", "Eukaryota;Insecta;Drosophila", recipient="flies"),
            _row("E8", "Pseudomonas sp. VKM F-3808 (species)", "Bacteria;Pseudomonas", recipient="a strain")]
    tax = Tax()
    found = W.taxa(rows, tax)
    inside = W.radius_genomes(found, _index(), tax)
    assert inside["E1"] == [("bacteria_GCF_000000123.1", 12)]          # the test genome of the same genus is not read
    tok = W.exposure_tokens(rows, found, inside, tax, {"NC_000001.1": 12, "NC_000002.1": 6})
    assert tok["E1"] == _tok(dict(accessions={"000000123"}, nucleotide={"NC_000001"}), dict(names={"pseudomonas"}), 1)
    assert tok["E2"] == _tok(dict(names={"drosophila_ananassae"}, nucleotide={"NC_000002"}),
                             dict(names={"drosophila", "drosophila_ananassae"}), 1)
    assert tok["E3"]["draft"]["names"] == set() and tok["E3"]["radius"]["names"] == {"drosophila_ananassae"}   # no 'japanese_flies'
    assert tok["E4"]["draft"] == dict(accessions={"004135705"}, names=set(), nucleotide={"JN900499"})       # cited by the draft
    assert tok["E5"]["draft"]["names"] == set()                        # below the genus, one word: not a token
    assert tok["E6"]["draft"]["names"] == {"drosophila_melanogaster"}
    assert tok["E7"]["draft"]["names"] == {"drosophila_ananassae"} and tok["E7"]["radius"]["nucleotide"] == set()
    assert tok["E8"]["draft"]["names"] == set()                        # no epithet: no genus either
    rep = W.token_report(tok)
    assert rep["E1"] == dict(genomes=1, n_assemblies=1, n_species=0, n_nucleotide=1, draft_assemblies=[], draft_names=["pseudomonas"],
                             draft_nucleotide=[])


def test_island_taxa_skip_the_sealed_and_the_candidates(tmp_path):
    ref = tmp_path / "v2_reference_set.tsv"
    ref.write_text("accession\torganism\tgenus\tfamily\torder\tpart\tlevel\tstatus\tlevel_from\n"
                   "NC_000001.1\tA b\tA\tF\tO\tdevelopment\tL3\tdevelopment\tx\n"
                   "NC_000002.1\tA c\tA\tF\tO\tsealed_v1\tL0\tsealed_v1\tx\n"
                   "NC_000003.1\tA d\tA\tF\tO\trest\tL1\tcandidate\tx\n"
                   "NC_000004.1\tA e\tA\tF\tO\trest\tL2\tdemoted\tx\n"
                   "NC_000005.1\tA f\tA\tF\tO\tdevelopment\tL3\tdevelopment\tx\n")
    (tmp_path / "panel").mkdir()
    (tmp_path / "panel" / "island_set_taxa.json").write_text(json.dumps({"NC_000001.1": 12, "NC_000002.1": 12, "NC_000003.1": 12,
                                                                        "NC_000004.1": 6}))
    got = W.island_taxa(str(ref), str(tmp_path))
    assert got["taxids"] == {"NC_000001.1": 12, "NC_000004.1": 6} and got["skip"] == {"NC_000002", "NC_000003"}
    assert got["skipped_rows"] == 2 and got["without_taxid"] == 1


def _touch(base, *rel):
    for r in rel:
        p = base / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")


TOKENS = {"E1": _tok(dict(accessions={"000000123"}), dict(names={"pseudomonas"}), 1),
          "E2": _tok(dict(names={"drosophila_ananassae"}), dict(names={"drosophila_ananassae"}), 1),
          "E3": _tok(None, dict(names={"homo"})), "E4": _tok(), "E5": _tok(None, dict(names={"nicotiana"})),
          "E6": _tok(None, dict(nucleotide={"JN900499"})),
          "E7": _tok(dict(names={"echinops_telfairi"}, accessions={"000000777"}, nucleotide={"NC_000009"}), None, 2)}


def test_exposure_scan_levels_by_own_name_prunes_and_skips(tmp_path):
    runs = tmp_path / "runs"
    _touch(runs, "r1/data/corpus/bacteria_GCF_000000123.1/proteins.faa",       # the folder names it (L3); its file does not
           "r2/scores/GCA_000000123.2.tsv",                                    # L4: the GCA spelling of the same assembly
           "r3/eval/drosophila_ananassae_bits.json",                           # L5 by binomial
           "r4/data/corpus/fungi_hidden_species_gca_000000999/pseudomonas.faa",  # a test genome's folder: not entered
           "r5/scores_x/bacteria_GCF_000000555.1.npz",                         # names a test genome: skipped
           "r7/eval/bits_fungi_hidden_species_gca_000000999_pseudomonas.json",  # names it inside a file name: skipped
           "r8/ensembl/Nicotiana_attenuata.IGv2.pep.fa",                        # an unprefixed test genome: skipped
           "r8/ensembl/NicotianaAttenuata_nicotiana.gff",                       # joined: skipped
           "r8/old/nicotiana_attenuata/eval/nicotiana_genes.tsv",              # inside its folder: not entered
           "islands/NC_000002.1/nicotiana.gbk",                                # a sealed island chromosome: not entered
           "cas/ab/GCF_000000123.1",                                           # pruned by relative path
           "own/data/pseudomonas_everything.tsv",                              # this run's own folder: not entered
           "r6/notes/unrelated_file.txt",
           "r9/views/JN900499.1.gb", "r9/raw/AJN900499.gb", "r9/raw/JN9004990.gb")   # scan_names bounds: only the first
    roots = [dict(path=str(runs), prune=["cas", "nothing_here"]), str(tmp_path / "nowhere")]
    unseen = W.unseen_genomes(_index(), {"NC_000002"})
    got = W.exposure_scan(TOKENS, roots, unseen, skip={str(runs / "own")})
    assert got["missing_roots"] == [str(tmp_path / "nowhere")]
    root = got["roots"][0]
    assert root["pruned"] == ["cas"] and root["prune_missing"] == ["nothing_here"] and root["walked"] == got["paths_walked"]
    assert got["events"]["E1"]["level"] == "L4" and got["events"]["E1"]["paths"] == {"L3": 1, "L4": 1}
    assert got["events"]["E1"]["matched"] == [] and got["events"]["E1"]["radius_tokens_matched"] == 1    # counted, not named
    assert got["events"]["E1"]["examples"]["L4"] == ["runs/r2/scores/<assembly>.2.tsv"]
    assert got["events"]["E2"]["level"] == "L5" and got["events"]["E2"]["examples"]["L5"] == ["runs/r3/eval/drosophila_ananassae_bits.json"]
    assert got["events"]["E3"] == dict(level="no_hit") and got["events"]["E4"] == dict(level="no_tokens")
    assert got["events"]["E5"] == dict(level="no_hit")                  # nicotiana appears only in the test genome's paths
    assert got["events"]["E6"]["level"] == "L4" and got["events"]["E6"]["examples"] == {"L4": ["runs/r9/views/JN900499.1.gb"]}
    assert got["skipped_unseen"] == 7 and got["walk_error_count"] == 0
    dump = json.dumps(got)
    assert "attenuata" not in dump.lower() and "hidden" not in dump and "000000555" not in dump and "000000999" not in dump
    assert "NC_000002" not in dump
    again = W.exposure_scan(TOKENS, roots[:1], unseen, skip={str(runs / "own")})
    assert again["roots"][0]["digest"] == root["digest"]
    capped = W.exposure_scan(TOKENS, [str(runs)], max_paths=3)
    assert capped["stopped_at_cap"] and capped["paths_walked"] == 4


def test_no_radius_token_reaches_the_report(tmp_path):
    """With the corpus manifest public, a radius's genomes listed would give the test genomes away by the gaps: the
    report counts them and redacts them in paths; only the draft's own tokens are named."""
    runs = tmp_path / "runs"
    _touch(runs, "r1/data/corpus/mammals_echinops_telfairi/proteins.faa", "r1/data/corpus/bacteria_GCF_000000777.1/x.faa",
           "r2/scores/EchinopsTelfairi.tsv", "r2/raw/NC_000009.1.gbk", "r3/eval/pairs_echinops_telfairi__homo_x.npz")
    got = W.exposure_scan(TOKENS, [str(runs)], W.unseen_genomes(_index()))
    assert got["events"]["E7"]["level"] == "L5" and got["events"]["E7"]["radius_tokens_matched"] == 3
    assert got["events"]["E7"]["matched"] == [] and got["tokens"]["E7"]["n_species"] == 1
    dump = json.dumps(got).lower()
    for word in ("echinops", "telfairi", "000000777", "nc_000009", "echinopstelfairi"):
        assert word not in dump, word
    assert "runs/r3/eval/pairs_<species>__homo_x.npz" in json.dumps(got)


def test_the_last_check_and_the_sorted_walk_are_pinned(tmp_path, monkeypatch):
    """Removing the final filter or the sorting fails here: the walk's own check is blinded until the walk ends, and
    the directory listing comes reversed."""
    import os
    runs = tmp_path / "runs"
    _touch(runs, "r1/mammals_echinops_x_pseudomonas.faa", "r1/a.txt", "r1/b.txt", "r2/c.txt", "r3/mammals_echinops_x_dir/y")
    real_walk, real_scandir, state = os.walk, os.scandir, dict(on=False)
    names = W.Unseen(dict(seqs={("echinops", "x")}))

    def walk(*a, **k):
        yield from real_walk(*a, **k)
        state["on"] = True

    def locked(path):
        if "echinops_x_dir" in str(path):
            raise PermissionError(13, "Permission denied", str(path))
        return real_scandir(path)
    monkeypatch.setattr(os, "walk", walk)
    monkeypatch.setattr(os, "scandir", locked)
    monkeypatch.setattr(W, "Unseen", lambda u: (lambda text: state["on"] and names(text)))
    got = W.exposure_scan(TOKENS, [str(runs)], None)
    assert got["dropped_outputs"] == 2 and "echinops" not in json.dumps(got)      # one example and one walk error
    assert got["events"]["E1"]["paths"] == {"L3": 1} and got["events"]["E1"]["examples"] == {"L3": []}
    assert got["walk_errors"] == [] and got["walk_error_count"] == 1
    monkeypatch.setattr(os, "walk", real_walk)
    monkeypatch.setattr(os, "scandir", real_scandir)
    (runs / "r3" / "mammals_echinops_x_dir" / "y").unlink()
    (runs / "r3" / "mammals_echinops_x_dir").rmdir()
    monkeypatch.setattr(W, "Unseen", lambda u: (lambda text: False))
    forward = W.exposure_scan(TOKENS, [str(runs)], None)["roots"][0]["digest"]
    real_scandir = os.scandir

    class Reversed:
        def __init__(self, path):
            with real_scandir(path) as it:
                self.items = sorted(it, key=lambda e: e.name, reverse=True)

        def __iter__(self):
            return iter(self.items)

        def __next__(self):
            if not self.items:
                raise StopIteration
            return self.items.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def close(self):
            pass
    monkeypatch.setattr(os, "scandir", Reversed)
    assert W.exposure_scan(TOKENS, [str(runs)], None)["roots"][0]["digest"] == forward


def test_the_digest_covers_only_the_paths_kept(tmp_path):
    runs = tmp_path / "runs"
    _touch(runs, "r1/a.txt")
    before = W.exposure_scan(TOKENS, [str(runs)], W.unseen_genomes(_index()))["roots"][0]["digest"]
    _touch(runs, "r1/plants_algae_nicotiana_attenuata.faa")
    after = W.exposure_scan(TOKENS, [str(runs)], W.unseen_genomes(_index()))
    assert after["roots"][0]["digest"] == before and after["skipped_unseen"] == 1


def test_walk_errors_never_name_a_test_genome(tmp_path, monkeypatch):
    import os
    runs = tmp_path / "runs"
    _touch(runs, "r1/plants_algae_nicotiana_attenuata/proteins.faa", "r1/fungi_hidden_species_gca_000000999/x.faa",
           "r2/locked/drosophila_ananassae.faa", "r3/pseudomonas.tsv", "r4/mammals_echinops_telfairi/x")
    real = os.scandir

    def scandir(path):                                   # as a folder without read permission (the scan runs as root too)
        if any(x in str(path) for x in ("locked", "attenuata", "hidden", "echinops")):
            raise PermissionError(13, "Permission denied", str(path))
        return real(path)
    monkeypatch.setattr(os, "scandir", scandir)
    unseen = W.unseen_genomes(_index())
    unseen["seqs"].discard(("nicotiana", "attenuata"))   # as if the species words were missed: the error check must hold
    got = W.exposure_scan(TOKENS, [str(runs)], unseen)
    dump = json.dumps(got)
    assert "attenuata" not in dump and "hidden" not in dump and "000000999" not in dump and "echinops" not in dump
    assert got["walk_errors"] == [dict(root=str(runs), path="r2/locked", errno="EACCES"),
                                  dict(root=str(runs), path="r4/mammals_<species>", errno="EACCES")]
    assert got["walk_error_count"] == 2
    assert got["events"]["E1"]["level"] == "L3"


def test_a_test_genome_error_is_counted_only(tmp_path, monkeypatch):
    """The error check stands on its own: here the walk's own check misses the folder (it only wakes once the error
    happened), and the error is still counted without its path."""
    import os
    runs = tmp_path / "runs"
    _touch(runs, "r1/mammals_echinops_x/proteins.faa", "r1/ok.txt")
    real = os.scandir
    state = dict(on=False)
    names = W.Unseen(dict(accs=set(), seqs={("echinops", "x")}))

    def scandir(path):
        if "echinops" in str(path):
            state["on"] = True
            raise PermissionError(13, "Permission denied", str(path))
        return real(path)
    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(W, "Unseen", lambda u: (lambda text: state["on"] and names(text)))
    got = W.exposure_scan(TOKENS, [str(runs)], dict(ids=set(), accs=set(), seqs=set()))
    assert got["walk_errors"] == [] and got["walk_errors_unseen"] == 1 and got["walk_error_count"] == 1
    assert "echinops" not in json.dumps(got)
    state["on"] = False
    monkeypatch.setattr(W, "Unseen", lambda u: (lambda text: False))
    got = W.exposure_scan(TOKENS, [str(runs)], dict(ids=set(), accs=set(), seqs=set()))
    assert got["walk_errors"] == [dict(root=str(runs), path="r1/mammals_echinops_x", errno="EACCES")]   # not a test genome: kept


def test_exposure_levels_declared_scan_and_unknown_values():
    rows = [_row("E1", "x (genus)", "x", exposure="L4"), _row("E2", "x (genus)", "x", exposure="?"),
            _row("E3", "x (genus)", "x", exposure="?"), _row("E4", "x (genus)", "x", exposure="L1"),
            _row("E5", "x (genus)", "x", exposure="L9")]
    scan = dict(events={"E1": dict(level="L3"), "E2": dict(level="L5"), "E3": dict(level="no_hit"), "E4": dict(level="L4"),
                        "E5": dict(level="L3")})
    got = W.exposure_levels(rows, scan)
    assert got["E1"] == dict(declared="L4", scan="L3", level="L4")
    assert got["E2"] == dict(declared="?", scan="L5", level="L5", above_declared=True)
    assert got["E3"] == dict(declared="?", scan="no_hit", level="?")
    assert got["E4"] == dict(declared="L1", scan="L4", level="L4", above_declared=True)
    assert got["E5"] == dict(declared="L9", scan="L3", level="L3", declared_unknown=True)


def _gate_with(tmp_path, monkeypatch, ex):
    import ci.stages.rel.w15_registry as mod
    run = tmp_path / "run"
    out = run / mod.OUT
    out.mkdir(parents=True, exist_ok=True)
    seen = []
    monkeypatch.setattr(mod, "require", lambda result, run_dir: seen.append(result))
    (out / "report.json").write_text(json.dumps(dict(errors=[], registry_sha256={}, papers=[], records=0, taxa={}, exposure=ex)))
    mod.gate(None, {}, str(run))
    return seen[-1]


def test_gate_fails_when_the_exposure_scan_is_incomplete(tmp_path, monkeypatch):
    root = dict(path="/r", walked=5)
    for ex, why in ((dict(missing_roots=["/x"], roots=[root]), "roots missing"),
                    (dict(missing_roots=[], stopped_at_cap=True, paths_walked=9, roots=[root]), "stopped at 9 paths"),
                    (dict(missing_roots=[], walk_error_count=2, roots=[root]), "2 walk errors"),
                    (dict(missing_roots=[], roots=[root, dict(path="/empty", walked=0)]), "no path walked: ['/empty']")):
        res = _gate_with(tmp_path, monkeypatch, ex)
        assert res.status == "fail" and why in json.dumps(res.evidence)
    assert _gate_with(tmp_path, monkeypatch, dict(missing_roots=[], walk_error_count=0, roots=[root])).status == "pass"


def test_settings_check_the_exposure_roots():
    base = dict(registry="bench/rel/registry/events_v0.2-sample.tsv")
    for bad in (["runs"], [dict(path="/srv/runs", prune=["/abs"])], [dict(path="/srv/runs", prune=["../up"])],
                [dict(path="/srv/runs", other=1)], "/srv/runs"):
        with pytest.raises(SystemExit):
            W.settings(dict(rel_w15_registry=dict(base, exposure_roots=bad)))
    ok = W.settings(dict(rel_w15_registry=dict(base, exposure_roots=["/srv/runs", dict(path="/srv/dev", prune=["cas", "tl"])])))
    assert ok["exposure_roots"][1]["prune"] == ["cas", "tl"]


def test_markdown_shows_the_exposure_scan(tmp_path):
    runs = tmp_path / "runs"
    _touch(runs, "r3/eval/drosophila_ananassae_bits.json", "r1/plain.txt")
    ex = W.exposure_scan(TOKENS, [str(runs)], W.unseen_genomes(_index()))
    rows = [_row(e, "x (genus)", "x", exposure="?") for e in sorted(TOKENS)]
    ex["levels"] = W.exposure_levels(rows, ex)
    ex["above_declared"] = []
    md = W.markdown_server(dict(taxa={}, pretraining={}, sealed_overlap={}, exposure=ex, papers=[]))
    assert ("| E2 | 1 | 0 assemblies, 1 species, 0 island | drosophila_ananassae | ? | L5 | L5 (above the declared level) | "
            "L5 1 | drosophila_ananassae | 0 | runs/r3/eval/") in md
    assert "| E4 | 0 | 0 assemblies, 0 species, 0 island | none | ? | no_tokens | ? |" in md and "| %s | 5 | - | 0 | 0 |" % runs in md
    assert "echinops" not in md


def test_settings_check_the_island_store():
    base = dict(registry="bench/rel/registry/events_v0.2-sample.tsv")
    for bad in ("islandpick", ["/srv/islandpick"]):
        with pytest.raises(SystemExit):
            W.settings(dict(rel_w15_registry=dict(base, exposure_islands=bad)))
    assert W.settings(dict(rel_w15_registry=dict(base, exposure_islands="/srv/islandpick")))["exposure_islands"] == "/srv/islandpick"
