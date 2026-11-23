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
         12: (11, "genus", "Pseudomonas"), 13: (10, "genus", "Neisseria"), 14: (1, "no rank", "Aphis")}


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
    return dict(genomes=[dict(id="bacteria_GCF_000000123.1", split="train", taxid=12),
                         dict(id="insects_drosophila_ananassae", split="heldout", taxid=6),
                         dict(id="bacteria_GCF_000000555.1", split="test", taxid=12),
                         dict(id="fungi_hidden_species_gca_000000999", split="test", taxid=6)])


def test_exposure_tokens_use_only_train_and_heldout_genomes():
    rows = [_row("E1", "Pseudomonas (genus)", "Bacteria;Pseudomonadaceae;Pseudomonas", recipient="Pseudomonas sp. (many strains)"),
            _row("E2", "Drosophila (genus)", "Eukaryota;Metazoa;Insecta;Drosophila", recipient="Drosophila ananassae (lines Hawaii, India)"),
            _row("E3", "Insecta (class)", "Eukaryota;Metazoa;Insecta", recipient="insects (several)")]
    tax = Tax()
    found = W.taxa(rows, tax)
    inside = W.radius_genomes(found, _index(), tax)
    assert inside["E1"] == [("bacteria_GCF_000000123.1", 12)]          # the test genome of the same genus is not read
    tok = W.exposure_tokens(rows, found, inside, tax)
    assert tok["E1"] == ({"000000123"}, {"pseudomonas"})              # 'Pseudomonas sp.' is no binomial
    assert tok["E2"] == (set(), {"drosophila", "drosophila_ananassae"})
    assert tok["E3"] == (set(), {"drosophila_ananassae"})              # a class adds no name of its own
    unseen = W.unseen_genomes(_index())
    assert unseen == dict(ids={"bacteria_gcf_000000555.1", "fungi_hidden_species_gca_000000999"}, accs={"000000555", "000000999"})


def _touch(base, *rel):
    for r in rel:
        p = base / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")


def test_exposure_scan_levels_by_folder_and_skips(tmp_path):
    runs = tmp_path / "runs"
    _touch(runs, "r1/data/corpus/bacteria_GCF_000000123.1/proteins.faa",       # L3 (and its folder)
           "r2/scores/GCA_000000123.2.tsv",                                    # L4: the GCA spelling of the same assembly
           "r3/eval/drosophila_ananassae_bits.json",                           # L5 by binomial
           "r4/data/corpus/fungi_hidden_species_gca_000000999/proteins.faa",   # names a test genome: skipped
           "r5/scores_x/bacteria_GCF_000000555.1.npz",                         # names a test genome: skipped
           "r7/eval/bits_fungi_hidden_species_gca_000000999_pseudomonas.json",  # names it inside a file name: skipped
           "cas/ab/GCF_000000123.1",                                           # content store: not entered
           "own/data/pseudomonas_everything.tsv",                              # this run's own folder: not entered
           "r6/notes/unrelated_file.txt")
    tokens = {"E1": ({"000000123"}, {"pseudomonas"}), "E2": (set(), {"drosophila_ananassae"}), "E3": (set(), {"homo"})}
    got = W.exposure_scan(tokens, [str(runs), str(tmp_path / "nowhere")], W.unseen_genomes(_index()), skip={str(runs / "own")})
    assert got["missing_roots"] == [str(tmp_path / "nowhere")]
    assert got["events"]["E1"]["level"] == "L4" and got["events"]["E1"]["paths"] == {"L3": 2, "L4": 1}
    assert got["events"]["E1"]["matched"] == ["000000123"]
    assert got["events"]["E2"]["level"] == "L5" and got["events"]["E2"]["examples"]["L5"][0].endswith("r3/eval/drosophila_ananassae_bits.json")
    assert "E3" not in got["events"]
    assert got["skipped_unseen"] == 4                                    # the test genome's folder and file, the npz, the json
    assert not any("cas" in p or "own" in p for v in got["events"].values() for ex in v["examples"].values() for p in ex)
    capped = W.exposure_scan(tokens, [str(runs)], max_paths=3)
    assert capped["stopped_at_cap"] and capped["paths_walked"] == 4


def test_exposure_levels_take_the_higher_of_declared_and_scan():
    rows = [_row("E1", "x (genus)", "x", exposure="L4"), _row("E2", "x (genus)", "x", exposure="?"), _row("E3", "x (genus)", "x", exposure="?")]
    scan = dict(events={"E1": dict(level="L3"), "E2": dict(level="L5")})
    got = W.exposure_levels(rows, scan)
    assert got["E1"] == dict(declared="L4", scan="L3", level="L4")
    assert got["E2"]["level"] == "L5" and got["E3"] == dict(declared="?", scan="?", level="?")


def test_gate_fails_when_the_exposure_scan_misses_a_root_or_stops(tmp_path, monkeypatch):
    import ci.stages.rel.w15_registry as mod
    run = tmp_path / "run"
    out = run / mod.OUT
    out.mkdir(parents=True)
    seen = []
    monkeypatch.setattr(mod, "require", lambda result, run_dir: seen.append(result))
    for ex, why in ((dict(missing_roots=["/x"], stopped_at_cap=False), "roots missing"),
                    (dict(missing_roots=[], stopped_at_cap=True, paths_walked=9), "stopped at 9 paths")):
        (out / "report.json").write_text(json.dumps(dict(errors=[], registry_sha256={}, papers=[], records=0, taxa={}, exposure=ex)))
        mod.gate(None, {}, str(run))
        assert seen[-1].status == "fail" and why in json.dumps(seen[-1].evidence)


def test_settings_want_absolute_exposure_roots():
    base = dict(rel_w15_registry=dict(registry="bench/rel/registry/events_v0.2-sample.tsv"))
    with pytest.raises(SystemExit):
        W.settings(dict(rel_w15_registry=dict(base["rel_w15_registry"], exposure_roots=["runs"])))
    assert W.settings(dict(rel_w15_registry=dict(base["rel_w15_registry"], exposure_roots=["/srv/runs"])))["exposure_roots"] == ["/srv/runs"]


def _stored_run(tmp_path):
    """A passed registry run that stored a PDF from the open-access dataset, a supplement archive holding a PDF, and a
    PDF whose bytes changed after it was stored."""
    import hashlib
    import zipfile
    from tests.ci.test_rel_w15_pdf_text import HELV, LINE1, _pages
    run = tmp_path / "20261123T000000-src"
    papers = run / W.OUT / "papers"
    good = _pages((HELV, LINE1, None))
    oa = papers / W._folder("10.1/a") / "pmc_oa"
    oa.mkdir(parents=True)
    (oa / "a.sd02.pdf").write_bytes(good)
    (oa / "a.sd03.pdf").write_bytes(good + b"changed")
    zdir = papers / W._folder("10.1/b")
    zdir.mkdir(parents=True)
    with zipfile.ZipFile(zdir / "supplements.zip", "w") as z:
        z.writestr("supp/b.s1.pdf", good)
        z.writestr("supp/b.s2.xlsx", b"PK")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(papers=[
        dict(doi="10.1/a", supplements=dict(source="pmc-oa-opendata", files=[
            dict(file="a.sd02.pdf", sha256=sha(oa / "a.sd02.pdf")), dict(file="a.sd03.pdf", sha256=hashlib.sha256(good).hexdigest()),
            dict(file="a.sd04.pdf", error_type="TooLarge")])),
        dict(doi="10.1/b", supplements=dict(source="europepmc", file="supplements.zip", sha256=sha(zdir / "supplements.zip")))])
    rp = run / W.OUT / "report.json"
    rp.write_text(json.dumps(report))
    (run / "stages").mkdir()
    (run / "stages" / (W.STAGE + ".json")).write_text(json.dumps(dict(status="passed", outputs=[dict(name="report", sha256=sha(rp))])))
    return run


def test_pdf_texts_read_only_files_the_run_stored_with_their_checksum(tmp_path):
    run = _stored_run(tmp_path)
    got = {f["file"]: f for f in W.pdf_texts(dict(from_run=str(run), files=["*.pdf", "missing.pdf"]), str(tmp_path / "out"))}
    assert sorted(got) == ["a.sd02.pdf", "a.sd03.pdf", "b.s1.pdf", "missing.pdf"]
    assert got["a.sd02.pdf"]["status"] == "ok" and got["a.sd02.pdf"]["per_status"] == {"ok": 1}
    assert got["b.s1.pdf"]["status"] == "ok" and got["b.s1.pdf"]["member"] == "supp/b.s1.pdf"
    assert got["a.sd03.pdf"]["status"] == "checksum_differs" and got["missing.pdf"]["status"] == "not_stored"
    text = (tmp_path / "out" / "a.sd02.pdf.txt").read_text()
    assert "gene ASEM_AUS1_12345" in text and not (tmp_path / "out" / "a.sd03.pdf.txt").exists()


def test_pdf_texts_refuse_a_report_the_run_did_not_record(tmp_path):
    run = _stored_run(tmp_path)
    (run / W.OUT / "report.json").write_text("{}")
    with pytest.raises(SystemExit):
        W.pdf_texts(dict(from_run=str(run), files=["*.pdf"]), str(tmp_path / "out"))


def test_gate_fails_on_a_pdf_not_stored_or_changed(tmp_path, monkeypatch):
    import ci.stages.rel.w15_registry as mod
    run = tmp_path / "run"
    out = run / mod.OUT
    out.mkdir(parents=True)
    seen = []
    monkeypatch.setattr(mod, "require", lambda result, run_dir: seen.append(result))
    files = [dict(file="a.pdf", status="unreadable"), dict(file="b.pdf", status="ok")]
    (out / "report.json").write_text(json.dumps(dict(errors=[], registry_sha256={}, papers=[], records=0, taxa={}, pdf_text=dict(files=files))))
    mod.gate(None, {}, str(run))
    assert seen[-1].status == "pass"                                   # an unreadable PDF is a result, not a failure
    files.append(dict(file="c.pdf", status="checksum_differs"))
    (out / "report.json").write_text(json.dumps(dict(errors=[], registry_sha256={}, papers=[], records=0, taxa={}, pdf_text=dict(files=files))))
    mod.gate(None, {}, str(run))
    assert seen[-1].status == "fail" and "c.pdf: checksum_differs" in json.dumps(seen[-1].evidence)


def test_settings_check_the_pdf_text_block():
    reg_ = dict(registry="bench/rel/registry/events_v0.2-sample.tsv")
    for bad in (dict(from_run="runs/x", files=["a.pdf"]), dict(from_run="/runs/x", files=[]), dict(from_run="/runs/x", files=["a"], other=1)):
        with pytest.raises(SystemExit):
            W.settings(dict(rel_w15_registry=dict(reg_, pdf_text=bad)))
    assert W.settings(dict(rel_w15_registry=dict(reg_, pdf_text=dict(from_run="/runs/x", files=["*.pdf"]))))["pdf_text"]["files"] == ["*.pdf"]
