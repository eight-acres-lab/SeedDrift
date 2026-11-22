"""The rows of the uncapped search (seeddrift/train/rel/tables/v1_hits.py) and stage rel_w12_search
(ci/stages/rel/w12_search.py). The stage runs on the synthetic source run of the table stages
(tests/ci/test_rel_w2_stage.py) with its corpus files added, and a stand-in for DIAMOND whose rule is known (every
target protein that starts with the query's first five residues, one or two ungapped alignments), so the rows can be
compared with what that rule gives. The real DIAMOND runs where it is installed (the CI environment)."""
import csv
import json
import os
import shutil
import stat
import sys

import numpy as np
import pytest

import rel_fixtures as RF
import test_rel_w2_stage as TS
from conftest import ROOT
from ci import manifest as M
from ci.gates import GateFailure
from ci.hashing import sha256_file
from ci.stages.common import Stage
from ci.stages.rel import gates as rel_gates, w12_search as WS
from seeddrift.train import source_model as S
from seeddrift.train.rel.tables import v1_hits as H

AA = "ACDEFGHIKLMNPQRSTVWY"

FAKE = r'''#!%(python)s
"""A stand-in for DIAMOND: makedb copies the FASTA; blastp reports, for every query in order, every database protein
that starts with the query's first five residues: one ungapped alignment over the shorter length and, when that is at
least 40 residues, a second one over its second half."""
import shutil
import sys


def fasta(path):
    out, name = [], None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            out.append([line[1:], ""])
        elif line:
            out[-1][1] += line
    return out


a = sys.argv[1:]
if a[0] == "makedb":
    shutil.copy(a[a.index("--in") + 1], a[a.index("-d") + 1] + ".dmnd")
    sys.exit(0)
assert a[0] == "blastp" and a[a.index("-k") + 1] == "0" and a[a.index("--max-hsps") + 1] == "0"
fields = a[a.index("--outfmt") + 2:]
assert fields == ["qseqid", "sseqid", "qstart", "qend", "sstart", "send", "cigar"], fields
db = fasta(a[a.index("-d") + 1])
out = sys.stdout
for qn, qs in fasta(a[a.index("-q") + 1]):
    for tn, ts in db:
        if len(qs) >= 5 and ts[:5] == qs[:5]:
            n = min(len(qs), len(ts))
            out.write("%%s\t%%s\t1\t%%d\t1\t%%d\t%%dM\n" %% (qn, tn, n, n, n))
            if n >= 40:
                h = n // 2
                out.write("%%s\t%%s\t%%d\t%%d\t%%d\t%%d\t%%dM\n" %% (qn, tn, h + 1, n, h + 1, n, n - h))
'''


def protein(codons):
    aa = S.CODE[np.asarray(codons, dtype=np.int64)]
    return "".join(AA[x - 1] if 1 <= x <= 20 else "X" for x in aa)


def add_corpus(src, metas):
    """genes.tsv and proteins.faa of every genome of the source run, from its training arrays, in its corpus manifest.
    A held-out genome's second protein is a copy of its first (a paralog the own-genome search must find)."""
    entries = []
    for m in metas:
        z = np.load(os.path.join(src, "data", "train_data", m["id"] + ".npz"))
        folder = os.path.join(src, "data", "corpus", m["id"])
        os.makedirs(folder, exist_ok=True)
        n = len(z["codon_off"]) - 1
        with open(os.path.join(folder, "genes.tsv"), "w") as f:
            f.write("gene_id\treplicon\tstart\tend\tstrand\tpartial\n")
            for i in range(n):
                f.write("g%03d\tr\t1\t2\t+\t\n" % i)
        prot = [protein(z["codons"][z["codon_off"][i]:z["codon_off"][i + 1]]) for i in range(n)]
        if m["split"] == "heldout" and n >= 2:
            prot[1] = prot[0]
        with open(os.path.join(folder, "proteins.faa"), "w") as f:
            for i in range(n):
                f.write(">%s__g%03d\n%s\n" % (m["id"], i, prot[i]))
        entries.append(dict(id=m["id"], taxid=m["taxid"], source_kind=TS.TINY[m["group"]][1], proteins=n, accession="", paired_accession="",
                            files=[M.file_entry(os.path.join(folder, x), src) for x in ("genes.tsv", "proteins.faa")]))
    man = M.build("fixture", "corpus", entries)
    mp = os.path.join(src, "data", "corpus", "manifest.json")
    with open(mp, "w") as f:
        json.dump(dict(sha256=M.write(man, os.path.join(src, "manifests")), dataset_id=M.dataset_id(man), **man), f)
    with open(os.path.join(src, "stages", "corpus.json"), "w") as f:
        json.dump(dict(name="corpus", status="passed", outputs=[dict(name="corpus_manifest", path="data/corpus/manifest.json",
                                                                     sha256=sha256_file(mp, use_sidecar=False))]), f)


def dev_accessions(n=2):
    with open(os.path.join(ROOT, "bench", "islands", "results", "dev_genomes.tsv")) as f:
        return [r["accession"] for r in csv.DictReader(f, delimiter="\t")][:n]


def add_islands(src, accessions, copy_of):
    """data/islands of a passed island stage: the chromosomes `accessions`, the first one a copy of the proteins of the
    training genome `copy_of`, every other one with its second protein a copy of its first (a paralog)."""
    with open(os.path.join(src, "data", "corpus", copy_of, "proteins.faa")) as f:
        base = [line.strip() for line in f if not line.startswith(">")]
    rng = np.random.default_rng(7)
    genomes, entries = [], []
    for k, acc in enumerate(accessions):
        prot = list(base) if k == 0 else ["M" + "".join(rng.choice(list(AA), int(rng.integers(30, 90)))) for _ in range(8)]
        if k:
            prot[1] = prot[0]
        rel = os.path.join("data", "islands", "inputs", acc)
        os.makedirs(os.path.join(src, rel), exist_ok=True)
        with open(os.path.join(src, rel, "genes.tsv"), "w") as f:
            f.write("gene_id\treplicon\tstart\tend\tstrand\tpartial\n" + "".join("i%03d\tr\t1\t2\t+\t\n" % i for i in range(len(prot))))
        with open(os.path.join(src, rel, "proteins.faa"), "w") as f:
            f.write("".join(">%s__i%03d\n%s\n" % (acc, i, p) for i, p in enumerate(prot)))
        genomes.append(dict(id=acc, folder=rel, group="bacteria", genes=len(prot), taxid=562, ranks=[1224, 1236, 91347, 543, 561, 562], genus="Escherichia"))
        entries.append(dict(id=acc, files=[M.file_entry(os.path.join(src, rel, n), src) for n in ("genes.tsv", "proteins.faa")]))
    man = M.build("islands", "islands", entries)
    with open(os.path.join(src, "data", "islands", "manifest.json"), "w") as f:
        json.dump(dict(sha256=M.write(man, os.path.join(src, "manifests")), dataset_id=M.dataset_id(man), **man), f)
    ip = os.path.join(src, "data", "islands", "index.json")
    with open(ip, "w") as f:
        json.dump(dict(dataset_id="ds-islands-test", genomes=genomes), f)
    with open(os.path.join(src, "stages", "islands_data.json"), "w") as f:
        json.dump(dict(name="islands_data", status="passed", outputs=[dict(name="islands_index", path="data/islands/index.json", sha256=sha256_file(ip, use_sidecar=False))]), f)
    return genomes


def expected(src, metas, searched=None, islands=()):
    """{(query genome, query gene): {(target genome, target gene): alignments}} by the stand-in's rule."""
    open_ = [m for m in metas if m["split"] in ("train", "heldout") and m["id"] != TS.LEFT_OUT]     # the sealed check leaves that one out
    open_ += [dict(id=g["id"], split="island", folder=g["folder"]) for g in islands]
    prot = {}
    for m in open_:                                                      # the proteins, as the stage reads them
        with open(os.path.join(src, m.get("folder") or os.path.join("data", "corpus", m["id"]), "proteins.faa")) as f:
            prot[m["id"]] = [line.strip() for line in f if not line.startswith(">")]
    train = [m["id"] for m in open_ if m["split"] == "train"]
    out = {}
    for m in open_:
        g = m["id"]
        targets = train if m["split"] == "train" else train + [g]
        for i, q in enumerate(prot[g]):
            if searched is not None and not searched[g][i]:
                continue
            for t in targets:
                if m["split"] != "train" and t != g and t not in train:
                    continue
                for j, p in enumerate(prot[t]):
                    if (t, j) == (g, i) or len(q) < 5 or p[:5] != q[:5]:
                        continue
                    out.setdefault((g, i), {})[(t, j)] = 2 if min(len(q), len(p)) >= 40 else 1
    return out


def got_pairs(run_dir):
    with open(os.path.join(run_dir, WS.OUT, "genomes.json")) as f:
        genomes = json.load(f)["genomes"]
    ids = [g["id"] for g in genomes]
    out = {}
    for g in genomes:
        a = H.load(os.path.join(run_dir, WS.OUT, g["id"] + ".npz"))
        for i in range(g["genes"]):
            for p in range(int(a["off"][i]), int(a["off"][i + 1])):
                out.setdefault((g["id"], i), {})[(ids[a["tgenome"][p]], int(a["tgene"][p]))] = int(a["hsp_off"][p + 1] - a["hsp_off"][p])
    return out, genomes


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w12")
    src = str(tmp / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    metas = TS.make_source(src)
    add_corpus(src, metas)
    fake = tmp / "diamond"
    fake.write_text(FAKE % dict(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return dict(tmp=tmp, src=src, metas=metas, diamond=str(fake))


def run_stage(source, tmp, run_id, **block):
    run_dir, cfg = RF.rel_run(tmp, stages=("w12_search",), run_id=run_id, tools=dict(diamond=source["diamond"], python=sys.executable),
                              rel_w12_search=dict(dict(source=source["src"], expect=TS.SIZES, threads=1, chunk_proteins=150, self_batch_proteins=40), **block))
    with pytest.MonkeyPatch.context() as mp:
        TS.patch(mp, source["tmp"])
        with Stage(run_dir, "rel_w12_search") as st:
            rep = WS.build(st, cfg, run_dir)
    return run_dir, cfg, rep


# --- the rows -----------------------------------------------------------------------------------------------------

def test_collector_groups_pairs_and_orders_them_without_scores():
    c = H.Collector(3)
    c.add(2, 5, 1, 10, 20, 3, 13, "11M")
    c.add(0, 7, 4, 30, 40, 1, 11, "11M")
    c.add(0, 2, 9, 1, 10, 1, 10, "10M")
    c.add(0, 7, 4, 1, 12, 1, 10, "5M2I5M")
    a = c.arrays()
    assert a["off"].tolist() == [0, 2, 2, 3] and a["tgenome"].tolist() == [2, 7, 5] and a["tgene"].tolist() == [9, 4, 1]
    assert a["hsp_off"].tolist() == [0, 1, 3, 4] and a["qs"].tolist() == [1, 1, 30, 10]                 # by query start inside a pair
    assert bytes(a["cigar"][a["cigar_off"][1]:a["cigar_off"][2]]) == b"5M2I5M" and H.check(a, 8) == []
    assert set(a) == set(H.KEYS) and not any("bit" in k or "score" in k or "evalue" in k for k in a)


def test_stream_parses_filters_and_refuses_rows_out_of_order():
    gi = {"A": 0, "B": 1}
    idx = {"A": {"x": 0, "y": 1}, "B": {"u": 0}}
    rows = ["A__x\tB__u\t1\t10\t1\t10\t10M\n", "A__x\tA__x\t1\t10\t1\t10\t10M\n", "A__x\tA__y\t1\t9\t2\t11\t4M1D5M\n", "B__u\tA__y\t1\t5\t1\t5\t5M\n"]
    got = {}
    kept = H.stream(rows, gi, idx, lambda qg, qi, tg, tj: not (qg == tg and qi == tj), lambda g, a: got.setdefault(g, a), {"A": 2, "B": 1})
    assert kept == {"A": 2, "B": 1} and got["A"]["tgenome"].tolist() == [0, 1] and got["A"]["tgene"].tolist() == [1, 0]
    with pytest.raises(ValueError, match="contiguous"):
        H.stream(rows + ["A__y\tB__u\t1\t5\t1\t5\t5M\n"], gi, idx, lambda *x: True, lambda g, a: None, {"A": 2, "B": 1})
    with pytest.raises(ValueError, match="fields"):
        H.stream(["A__x\tB__u\t1\t10\n"], gi, idx, lambda *x: True, lambda g, a: None, {"A": 2, "B": 1})


def test_merge_columns_spans_and_save():
    a = H.Collector(2)
    a.add(0, 1, 3, 1, 10, 1, 10, "10M")
    b = H.Collector(2)
    b.add(0, 1, 3, 1, 10, 1, 10, "10M")                                 # the same alignment again
    b.add(1, 0, 0, 2, 6, 4, 9, "2M1D3M")
    m = H.merge(a.arrays(), b.arrays())
    assert len(m["qs"]) == 2 and m["off"].tolist() == [0, 1, 2] and H.check(m, 2) == []
    q, t = H.columns(b"2M1I1M2D3M", 5, 10)
    assert q.tolist() == [4, 5, 7, 8, 9, 10] and t.tolist() == [9, 10, 11, 14, 15, 16] and H.spans(b"2M1I1M2D3M") == (7, 8)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        H.save(os.path.join(d, "x.npz"), m)
        back = H.load(os.path.join(d, "x.npz"))
        assert all(np.array_equal(back[k], m[k]) for k in H.KEYS)


def test_check_fails_on_broken_rows():
    c = H.Collector(2, searched=[1, 0])
    c.add(0, 1, 3, 1, 10, 1, 10, "10M")
    good = c.arrays()
    assert H.check(good, 2) == []
    for key, change in (("tgenome", lambda a: a.__setitem__("tgenome", np.array([5], np.int32))),
                        ("cigar", lambda a: a.__setitem__("cigar", np.frombuffer(b"9M", np.uint8).copy()) or a.__setitem__("cigar_off", np.array([0, 2]))),
                        ("searched", lambda a: a.__setitem__("off", np.array([0, 0, 1])))):
        bad = {k: v.copy() for k, v in good.items()}
        change(bad)
        status = "fail" if H.check(bad, 2) else "pass"
        assert status == "fail", key


# --- the stage ------------------------------------------------------------------------------------------------------

def test_full_search_gives_every_pair_of_the_rule(source, tmp_path, monkeypatch):
    run_dir, cfg, rep = run_stage(source, tmp_path, "20261123T000001-full")
    got, genomes = got_pairs(run_dir)
    want = expected(source["src"], source["metas"])
    assert got == want and len(want) > 20
    assert {g["split"] for g in genomes} == {"train", "heldout"} and not {g["id"] for g in genomes} & set(TS.SEALED) and TS.LEFT_OUT not in {g["id"] for g in genomes}
    assert rep["database"]["genomes"] == 24 and rep["queries"]["genomes"] == 36 and "extrapolation" not in rep
    held = [g for g in genomes if g["split"] == "heldout"]
    assert sum(g["self_pairs"] for g in held) > 0 and len(rep["timing"]["self_s"]) >= 2 and len(rep["timing"]["main_s"]) >= 2    # several calls
    assert rep["rows"]["hsps"] == sum(sum(v.values()) for v in want.values())
    assert not os.path.exists(os.path.join(run_dir, WS.OUT, "parts"))
    text = open(os.path.join(run_dir, WS.OUT, "manifest.json")).read() + open(os.path.join(run_dir, WS.OUT, "report.md")).read()
    assert not any(x in text for x in TS.SEALED) and TS.LEFT_OUT not in text
    TS.patch(monkeypatch, source["tmp"])
    with Stage(run_dir, "gates_rel_w12_search") as st:
        WS.gate(st, cfg, run_dir)
    assert json.load(open(os.path.join(run_dir, "gates", "G4_rel_w12_search.json")))["status"] == "pass"
    with Stage(run_dir, "gates_rel_tables") as st:
        rel_gates.kind_phase(st, cfg, run_dir, ["w12_search"])
    p = os.path.join(run_dir, WS.OUT, genomes[0]["id"] + ".npz")
    a = H.load(p)
    a["tgenome"][:1] = 99                                                 # a row to a genome outside the list
    H.save(p, a)
    with pytest.raises(GateFailure):
        with Stage(run_dir, "gates_rel_w12_search") as st:
            WS.gate(st, cfg, run_dir)


def test_sample_run_searches_a_share_per_group_and_extrapolates(source, tmp_path):
    run_dir, cfg, rep = run_stage(source, tmp_path, "20261123T000002-sample", sample=0.3, probe_proteins=3)
    got, genomes = got_pairs(run_dir)
    names = {g["id"]: g["genes"] for g in genomes}
    searched = {}
    for g in genomes:
        searched[g["id"]] = H.load(os.path.join(run_dir, WS.OUT, g["id"] + ".npz"))["searched"]
    assert got == expected(source["src"], source["metas"], searched)
    for grp in {g["group"] for g in genomes}:                             # every group sampled, about 30% of its proteins
        n = sum(names[g["id"]] for g in genomes if g["group"] == grp)
        k = sum(int(searched[g["id"]].sum()) for g in genomes if g["group"] == grp)
        assert k == max(1, int(round(0.3 * n)))
    e = rep["extrapolation"]
    assert e["factor_letters"] > 2 and e["hsps"] > rep["rows"]["hsps"] and e["total_h"] >= 0 and rep["timing"]["probe_proteins"] == 3
    page = open(os.path.join(run_dir, WS.OUT, "report.md")).read()
    assert "extrapolated linearly" in page and "segments of at most 3 h" in page


def test_a_stopped_run_resumes_from_its_finished_calls(source, tmp_path, monkeypatch):
    run_dir, cfg = RF.rel_run(tmp_path, stages=("w12_search",), run_id="20261123T000003-resume", tools=dict(diamond=source["diamond"], python=sys.executable),
                              rel_w12_search=dict(source=source["src"], expect=TS.SIZES, threads=1, chunk_proteins=150, self_batch_proteins=40))
    TS.patch(monkeypatch, source["tmp"])
    real, made = WS.diamond_search, []

    def killed_at_the_third(*a, **k):
        if len(made) == 2:
            raise RuntimeError("killed")
        made.append(1)
        return real(*a, **k)
    monkeypatch.setattr(WS, "diamond_search", killed_at_the_third)
    with pytest.raises(RuntimeError):
        with Stage(run_dir, "rel_w12_search") as st:
            WS.build(st, cfg, run_dir)
    calls = []
    monkeypatch.setattr(WS, "diamond_search", lambda *a, **k: calls.append(1) or real(*a, **k))
    with Stage(run_dir, "rel_w12_search") as st:
        rep = WS.build(st, cfg, run_dir)
    got, _ = got_pairs(run_dir)
    assert got == expected(source["src"], source["metas"]) and len(calls) == len(rep["timing"]["main_s"]) + len(rep["timing"]["self_s"]) - 2
    stop = dict(cfg, rel_w12_search=dict(cfg["rel_w12_search"], wall_s=1e-9))
    run2, _ = RF.rel_run(tmp_path, stages=("w12_search",), run_id="20261123T000004-stop")
    with pytest.raises(SystemExit, match="resume the run"):
        with Stage(run2, "rel_w12_search") as st:
            WS.build(st, dict(stop, run_dir=run2), run2)


def test_development_chromosomes_are_extra_queries(source, tmp_path, monkeypatch):
    import shutil as sh
    src = str(tmp_path / "ci-root" / "runs" / "20261120T000000-src")
    sh.copytree(source["src"], src)
    accs = dev_accessions(2)
    isl = add_islands(src, accs, [m["id"] for m in source["metas"] if m["split"] == "train"][0])
    s2 = dict(source, src=src)
    run_dir, cfg, rep = run_stage(s2, tmp_path, "20261123T000005-islands", islands=True)
    got, genomes = got_pairs(run_dir)
    assert got == expected(src, source["metas"], islands=isl)
    split = {g["id"]: g["split"] for g in genomes}
    assert [a for a in accs if split.get(a) == "island"] == accs and rep["queries"]["per_split"]["island"] == 2 and rep["islands"]["chromosomes"] == 2
    assert any(k[0] == accs[0] for k in got) and any(t[0] == accs[1] for k, v in got.items() if k[0] == accs[1] for t in v)   # hits, and a paralog
    TS.patch(monkeypatch, source["tmp"])
    with Stage(run_dir, "gates_rel_w12_search") as st:
        WS.gate(st, cfg, run_dir)
    with open(os.path.join(src, "data", "islands", "index.json")) as f:
        idx = json.load(f)
    idx["genomes"][0]["id"] = "NC_999999.1"                              # not a development chromosome
    ip = os.path.join(src, "data", "islands", "index.json")
    with open(ip, "w") as f:
        json.dump(idx, f)
    with open(os.path.join(src, "stages", "islands_data.json"), "w") as f:  # the index as its stage recorded it
        json.dump(dict(name="islands_data", status="passed", outputs=[dict(name="islands_index", path="data/islands/index.json", sha256=sha256_file(ip, use_sidecar=False))]), f)
    with pytest.raises(SystemExit, match="not development chromosomes"):
        run_stage(s2, tmp_path, "20261123T000006-islands-bad", islands=True)


def test_settings_are_checked():
    for bad in (dict(source="x", colour="red"), dict(sample=0.5), dict(source="x", sample=0), dict(source="x", sensitivity="loose")):
        with pytest.raises(SystemExit):
            WS.settings(dict(rel_w12_search=bad))


@pytest.mark.skipif(shutil.which("diamond") is None, reason="DIAMOND is not installed here (it is in the CI environment)")
def test_the_real_diamond_reports_every_alignment(tmp_path):
    rng = np.random.default_rng(5)
    a = "".join(rng.choice(list(AA), 300))
    b = "".join(c if rng.random() > 0.1 else rng.choice(list(AA)) for c in a)
    c = "".join(rng.choice(list(AA), 300))
    db = str(tmp_path / "db.dmnd")
    WS.makedb("diamond", [("T__b", b), ("T__c", c), ("T__a2", a[150:] + a[:150])], str(tmp_path / "db.faa"), db)    # a2: a rotated
    q = tmp_path / "q.faa"
    q.write_text(">Q__a\n%s\n" % a)
    s = WS.settings(dict(rel_w12_search=dict(source="x", threads=1, block_size=0.1)))
    got = {}
    sec, kept = WS.diamond_search("diamond", str(q), db, s, lambda lines: H.stream(lines, {"Q": 0, "T": 1}, {"Q": {"a": 0}, "T": {"b": 0, "c": 1, "a2": 2}},
                                                                                   lambda *x: True, lambda g, arr: got.setdefault(g, arr), {"Q": 1}))
    arr = got["Q"]
    assert kept["Q"] >= 2 and H.check(arr, 2) == [] and 0 in arr["tgene"].tolist()
    assert (np.diff(arr["hsp_off"]) >= 2).any()                           # the rotated copy: two alignments of one pair
