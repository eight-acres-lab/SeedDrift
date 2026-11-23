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
