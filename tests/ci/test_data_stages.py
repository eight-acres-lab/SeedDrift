"""Data stages without network: the content-addressed fetch store, Ensembl CDS parsing, held-out regions, and the
cached reference tables."""
import gzip
import json
import os

import numpy as np
import pytest

from conftest import FIX


# ---- fetch: content-addressed store and index -------------------------------------------------------------------

def test_fetch_store_reuses_and_refuses_corrupt_files(tmp_path, monkeypatch):
    from ci.stages import fetch
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return b">J02459.1 phage\nACGTACGT\n"
    monkeypatch.setattr(fetch, "get", fake_get)
    cas, index = str(tmp_path / "cas"), {}
    src = dict(id="phage", ena_sequences=["J02459.1"])
    files, prov, hit = fetch.fetch_source(src, cas, index)
    assert not hit and len(calls) == 1 and files[0]["path"].startswith(cas)
    rel = [dict(f, path=os.path.relpath(f["path"], cas)) for f in files]
    index[fetch.sha256_json(fetch.describe(src))] = dict(files=rel, provenance=prov)
    files2, _, hit2 = fetch.fetch_source(src, cas, index)
    assert hit2 and len(calls) == 1 and files2[0]["path"] == files[0]["path"]
    with open(files[0]["path"], "ab") as f:                 # a stored file that no longer matches its name
        f.write(b"N")
    with pytest.raises(SystemExit):
        fetch.fetch_source(src, cas, index)


def test_fetch_refuses_incomplete_downloads_fails(tmp_path, monkeypatch):
    from ci.stages import fetch
    cas = str(tmp_path / "cas")
    monkeypatch.setattr(fetch, "get", lambda url, **kw: b">ENA|J02459|J02459.1 lambda\nACGT\n")
    with pytest.raises(SystemExit):                              # one of two requested sequences came back
        fetch.fetch_source(dict(id="two", ena_sequences=["J02459.1", "AF158101.6"]), cas, {})
    monkeypatch.setattr(fetch, "get", lambda url, **kw: b">ENA|J02459|J02459.1 lambda\n")
    with pytest.raises(SystemExit):                              # an empty record
        fetch.fetch_source(dict(id="one", ena_sequences=["J02459.1"]), cas, {})
    monkeypatch.setattr(fetch, "get", lambda url, **kw: b"<html>busy</html>")
    with pytest.raises(SystemExit):
        fetch.fetch_source(dict(id="one", ena_sequences=["J02459"]), cas, {})
    monkeypatch.setattr(fetch, "get", lambda url, **kw: b">ENA|J02459|J02459.1 lambda\nACGT\n")
    files, _, _ = fetch.fetch_source(dict(id="one", ena_sequences=["J02459"]), cas, {})   # no version: any version
    assert files and os.path.exists(files[0]["path"])
    listing = b'<a href="X.cds.all.fa.gz">x</a>'
    full = gzip.compress(b">t cds x\nATG\n")
    monkeypatch.setattr(fetch, "get", lambda url, **kw: listing if url.endswith("/") else full[:-6])
    with pytest.raises(SystemExit):                              # a truncated gzip stream never enters the store
        fetch.fetch_source(dict(id="e", ensembl=dict(site="ensembl", species="x", release=1)), str(tmp_path / "cas2"), {})
    assert not (tmp_path / "cas2").exists()


def test_fetch_tells_a_missing_record_from_a_network_failure(monkeypatch):
    import urllib.error
    from ci.stages import fetch
    seen = []

    def opener(code):
        def urlopen(req, timeout=0):
            seen.append(code)
            if code:
                raise urllib.error.HTTPError(req.full_url, code, "x", {}, None)
            raise OSError("connection reset")
        return urlopen
    monkeypatch.setattr(fetch.urllib.request, "urlopen", opener(404))
    with pytest.raises(fetch.NoRecord):
        fetch.get("https://example.invalid/a", tries=5, pause=0)
    assert len(seen) == 2                                         # asked twice, then no record
    seen.clear()
    monkeypatch.setattr(fetch.urllib.request, "urlopen", opener(503))
    with pytest.raises(RuntimeError):
        fetch.get("https://example.invalid/b", tries=3, pause=0)
    assert len(seen) == 3                                         # a server error is retried and stays a failure
    monkeypatch.setattr(fetch.urllib.request, "urlopen", opener(0))
    with pytest.raises(RuntimeError):
        fetch.get("https://example.invalid/c", tries=2, pause=0)


def test_corpus_leaves_out_sources_without_a_record():
    from ci.stages.corpus import sources_of
    spec = dict(sources=[dict(id="a", source_kind="raw_virus", taxid=1, ena_sequences=["X1.1"]),
                         dict(id="b", source_kind="raw_virus", taxid=2, ena_sequences=["X2.1"])])
    rows = sources_of(spec, lambda x: x, {"a": ["/store/a.fna"]}, unavailable={"b"})
    assert [r["id"] for r in rows] == ["a"]
    with pytest.raises(SystemExit):                                # a source neither fetched nor listed as unavailable
        sources_of(spec, lambda x: x, {"a": ["/store/a.fna"]})


def test_fetch_ensembl_needs_exactly_one_cds_file_fails(tmp_path, monkeypatch):
    from ci.stages import fetch
    listing = b'<a href="Homo_sapiens.GRCh38.cds.all.fa.gz">x</a> <a href="Homo_sapiens.GRCh38.cds.abinitio.fa.gz">y</a>'
    monkeypatch.setattr(fetch, "get", lambda url, **kw: listing if url.endswith("/") else gzip.compress(b">t cds x\nATG\n"))
    src = dict(id="human", ensembl=dict(site="ensembl", species="homo_sapiens", release=116))
    files, prov, _ = fetch.fetch_source(src, str(tmp_path / "cas"), {})
    assert prov[0]["url"].endswith("/release-116/fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz")
    assert files[0]["path"].endswith(".cds.fa.gz")
    two = listing + b' <a href="Homo_sapiens.other.cds.all.fa.gz">z</a>'
    monkeypatch.setattr(fetch, "get", lambda url, **kw: two)
    with pytest.raises(SystemExit):
        fetch.fetch_source(src, str(tmp_path / "cas2"), {})


# ---- corpus: Ensembl coding sequences, ENA names, held-out regions ----------------------------------------------

def _cds(tag, seq):
    return ">%s\n%s\n" % (tag, seq)


def test_annotated_cds_takes_protein_coding_longest_and_skips_organelles(tmp_path):
    from ci.stages.corpus import annotated_cds
    good = "ATG" + "GCT" * 40 + "TAA"                          # 42 codons with the stop
    longer = "ATG" + "GCT" * 50 + "TAA"
    internal_stop = "ATG" + "GCT" * 20 + "TAA" + "GCT" * 20 + "TAA"
    text = "".join([
        _cds("T1.1 cds chromosome:GRCh38:21:100:225:1 gene:G1.3 gene_biotype:protein_coding transcript_biotype:protein_coding", good),
        _cds("T2.1 cds chromosome:GRCh38:21:100:255:1 gene:G1.3 gene_biotype:protein_coding transcript_biotype:protein_coding", longer),
        _cds("T3.1 cds chromosome:GRCh38:21:900:1025:-1 gene:G2.1 gene_biotype:protein_coding transcript_biotype:nonsense_mediated_decay", good),
        _cds("T4.1 cds chromosome:GRCh38:MT:10:135:1 gene:G3.1 gene_biotype:protein_coding transcript_biotype:protein_coding", good),
        _cds("T5.1 cds chromosome:GRCh38:21:2000:2128:-1 gene:G4.1 gene_biotype:protein_coding transcript_biotype:protein_coding", internal_stop),
        _cds("T6.1 cds chromosome:GRCh38:22:50:175:1 gene:G5.1 gene_biotype:protein_coding transcript_biotype:protein_coding", good),
        _cds("T7.1 cds chromosome:GRCh38:21:3000:3125:-1 gene:G6.1 gene_biotype:protein_coding transcript_biotype:protein_coding", good),
    ])
    p = tmp_path / "x.cds.all.fa.gz"
    with gzip.open(p, "wt") as f:
        f.write(text)
    genes, mode = annotated_cds(str(p), 1)
    ids = [g["gene_id"] for g in genes]
    assert ids == ["G1", "G6", "G5"]                           # genome order; MT, the NMD transcript and the internal stop left out
    assert len(genes[0]["protein"]) == 51 and genes[1]["strand"] == "-"
    only21, _ = annotated_cds(str(p), 1, {"21"}, max_genes=1)
    assert [g["gene_id"] for g in only21] == ["G1"]


def test_cut_regions_and_ena_names():
    from ci.stages.corpus import clean_name, cut_regions
    assert clean_name("ENA|AE016828|AE016828.3") == "AE016828.3" and clean_name("chr1") == "chr1"
    seq = "".join("ACGT"[i % 4] for i in range(100))
    out = cut_regions({"A.1": seq, "B.1": "GG"}, [dict(replicon="A.1", start=11, end=20), dict(replicon="A.1", start=51, end=100)])
    assert out["A.1_p1"] == seq[:10] and out["A.1_p2"] == seq[20:50] and "A.1" not in out and out["B.1"] == "GG"
    with pytest.raises(SystemExit):
        cut_regions({"A.1": seq}, [dict(replicon="C.1", start=1, end=2)])
    for a, b in ((0, 5), (30, 20), (90, 101)):                    # outside the replicon or reversed
        with pytest.raises(SystemExit):
            cut_regions({"A.1": seq}, [dict(replicon="A.1", start=a, end=b)])


# ---- reference: tables kept in the content store ----------------------------------------------------------------

class FakeStage:
    def __init__(self, data_root):
        self.config = dict(data_root=data_root, taxdump=os.path.join(FIX, "mini_taxdump"))

    def repo(self, p):
        return p


def test_reference_tables_cached_and_checked_fail(tmp_path):
    from ci.stages import reference
    taxids = []
    with open(os.path.join(FIX, "mini_taxdump", "nodes.dmp")) as f:
        for line in f:
            taxids.append(int(line.split("|")[0]))
    faa = tmp_path / "ref.faa"
    faa.write_text("".join(">P%03d|%d\nMKV\n" % (i, t) for i, t in enumerate(taxids[:12])))
    st = FakeStage(str(tmp_path / "root"))
    a = tmp_path / "a"
    a.mkdir()
    n, m, note = reference.cached_tables(st, str(faa), str(a / "index.npz"), str(a / "library.npz"))
    assert note.startswith("stored") and n == 12
    b = tmp_path / "b"
    b.mkdir()
    n2, m2, note2 = reference.cached_tables(st, str(faa), str(b / "index.npz"), str(b / "library.npz"))
    assert note2.startswith("reused") and (n2, m2) == (n, m)
    assert (a / "index.npz").read_bytes() == (b / "index.npz").read_bytes()
    assert list(np.load(str(b / "index.npz"))["acc"][:2]) == [b"P000", b"P001"]
    cache_dir = next(p for p in (tmp_path / "root" / "cas" / "reference").iterdir())
    os.chmod(cache_dir / "library.npz", 0o644)
    with open(cache_dir / "library.npz", "ab") as f:            # a cached table changed after it was stored
        f.write(b"x")
    c = tmp_path / "c"
    c.mkdir()
    with pytest.raises(SystemExit):
        reference.cached_tables(st, str(faa), str(c / "index.npz"), str(c / "library.npz"))
    meta = json.loads((cache_dir / "meta.json").read_text())
    assert set(meta["sha256"]) == {"index.npz", "library.npz"}


# ---- released: names and scores of the v0.1.0 screen ------------------------------------------------------------

def test_released_names_and_scores():
    from ci.stages.released import released_name, score_of
    assert released_name("scenario_host_insert__scenario_host_insert_12") == "scenario_host_insert__scenario_host_insert_12"
    assert released_name("a|b c") == "a_b_c" and len(released_name("x" * 80)) == 60
    foreign = score_of(dict(category="prokaryote", AI="12.5"))
    native = score_of(dict(category="native", AI="40.0"))
    orphan = score_of(dict(category="orphan", AI=""))
    assert foreign > native > orphan
