"""The extension of an earlier search (stage rel_w12_search, rel_w12_search.extend_from): on the synthetic source run of
test_rel_w12_search and an earlier corpus without three of its genomes (two training, one held out), the search of the
earlier corpus extended to the whole one must be the whole one's search made in full with the same fixed settings
(dbsize, algo 0, one held-out genome per own-protein search): every genome's rows array by array, the genome list and
the lookup index, byte for byte. With the stand-in for DIAMOND everywhere; with the real DIAMOND where it is installed
(the CI environment). Also the vectorized remaking of rows (v1_hits.flat, combine) against the Collector."""
import json
import os
import shutil
import stat
import sys

import numpy as np
import pytest

import rel_fixtures as RF
import test_rel_w12_search as T
import test_rel_w2_stage as TS
from ci.hashing import sha256_file
from ci.stages.common import Stage
from ci.stages.rel import w12_search as WS
from seeddrift.train.rel.tables import v1_hits as H


def test_combine_is_the_collector():
    rng = np.random.default_rng(3)
    c, rows = H.Collector(6), []
    for _ in range(60):
        r = (int(rng.integers(0, 6)), int(rng.integers(0, 4)), int(rng.integers(0, 5)), int(rng.integers(1, 30)))
        rows.append(r + (r[3] + 9, 1, 10, "%dM" % 10))
        c.add(*rows[-1])
    a = c.arrays()
    b = H.combine([H.flat(a)], 6, a["searched"])
    assert all(np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in H.KEYS)
    tmap = np.array([3, 0, 2, 1])                                          # renumbered target genomes, genome 2 (new numbering) left out
    c2 = H.Collector(6)
    for r in rows:
        if tmap[r[1]] != 2:
            c2.add(r[0], int(tmap[r[1]]), *r[2:])
    want, got = c2.arrays(), H.combine([H.flat(a, tmap, drop=2)], 6, a["searched"])
    assert all(np.array_equal(want[k], got[k]) for k in H.KEYS)
    half = H.Collector(6)                                                 # two parts make the arrays of all their alignments
    for r in rows[:30]:
        half.add(*r)
    rest = H.Collector(6)
    for r in rows[30:]:
        rest.add(*r)
    two = H.combine([H.flat(half.arrays()), H.flat(rest.arrays())], 6, a["searched"])
    assert all(np.array_equal(a[k], two[k]) for k in H.KEYS) and H.combine([], 2, np.ones(2, np.uint8))["off"].tolist() == [0, 0, 0]
    assert H.gene_rows(a, rows[0][0]) and H.gene_rows(a, rows[0][0], drop=rows[0][1]) != H.gene_rows(a, rows[0][0])


@pytest.fixture(scope="module")
def run_source(tmp_path_factory):
    """The synthetic source run with its corpus files and the stand-in for DIAMOND (as test_rel_w12_search's)."""
    tmp = tmp_path_factory.mktemp("w12x")
    src = str(tmp / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    metas = TS.make_source(src)
    T.add_corpus(src, metas)
    fake = tmp / "diamond"
    fake.write_text(T.FAKE % dict(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return dict(tmp=tmp, src=src, metas=metas, diamond=str(fake))


DROP_SPLITS = ("train", "train", "heldout")


def earlier(src, metas):
    """A copy of the source run without three genomes of three groups (two training, one held out): (its folder, the
    expected split sizes, the ids left out)."""
    gone, groups = [], set()
    for want in DROP_SPLITS:
        m = next(m for m in metas if m["split"] == want and m["group"] not in groups and m["id"] != TS.LEFT_OUT and m["group"] != "mobile.plasmid")
        gone.append(m["id"])
        groups.add(m["group"])
    old = src.replace("-src", "-old")
    shutil.rmtree(old, ignore_errors=True)
    shutil.copytree(src, old)
    ip = os.path.join(old, "data", "train_data", "index.json")
    index = json.load(open(ip))
    json.dump(dict(index, genomes=[g for g in index["genomes"] if g["id"] not in gone]), open(ip, "w"))
    cp = os.path.join(old, "data", "corpus", "manifest.json")
    man = json.load(open(cp))
    json.dump(dict(man, entries=[e for e in man["entries"] if e["id"] not in gone]), open(cp, "w"))
    for stage, name, path in (("train_data", "train_data_index", ip), ("corpus", "corpus_manifest", cp)):
        sp = os.path.join(old, "stages", stage + ".json")
        rec = json.load(open(sp))
        for o in rec["outputs"]:
            if o["name"] == name:
                o["sha256"] = sha256_file(path, use_sidecar=False)
        json.dump(rec, open(sp, "w"))
    rec = json.load(open(os.path.join(old, "record.json")))
    json.dump(dict(rec, run_id=os.path.basename(old)), open(os.path.join(old, "record.json"), "w"))
    sizes = dict(TS.SIZES)
    for sp in DROP_SPLITS:
        sizes[sp] -= 1
    return old, sizes, gone


def search(source, tmp, run_id, src, expect, diamond, **block):
    run_dir, cfg = RF.rel_run(tmp, stages=("w12_search",), run_id=run_id, tools=dict(diamond=diamond, python=sys.executable),
                              rel_w12_search=dict(dict(source=src, expect=expect, threads=1, chunk_proteins=150, self_batch_proteins=40), **block))
    with pytest.MonkeyPatch.context() as mp:
        TS.patch(mp, source["tmp"])
        with Stage(run_dir, "rel_w12_search") as st:
            rep = WS.build(st, cfg, run_dir)
        rec = json.load(open(os.path.join(run_dir, "record.json")))
        json.dump(dict(rec, status="passed"), open(os.path.join(run_dir, "record.json"), "w"))
    return run_dir, rep


def same_search(a, b):
    """Problems: where the rows, genome lists or lookup indexes of two search runs differ."""
    out = []
    ga = json.load(open(os.path.join(a, WS.OUT, "genomes.json")))["genomes"]
    gb = json.load(open(os.path.join(b, WS.OUT, "genomes.json")))["genomes"]
    if ga != gb:
        out.append("genome lists")
    for g in ga:
        x, y = H.load(os.path.join(a, WS.OUT, g["id"] + ".npz")), H.load(os.path.join(b, WS.OUT, g["id"] + ".npz"))
        out += ["%s %s" % (g["id"], k) for k in H.KEYS if not np.array_equal(x[k], y[k]) or x[k].dtype != y[k].dtype]
    for k in H.INDEX_FIELDS:
        if sha256_file(os.path.join(a, WS.OUT, "index", k + ".npy"), use_sidecar=False) != sha256_file(os.path.join(b, WS.OUT, "index", k + ".npy"), use_sidecar=False):
            out.append("index %s" % k)
    return out


def extended_equals_full(source, tmp, diamond, verify):
    old, sizes, gone = earlier(source["src"], source["metas"])
    e_dir, e_rep = search(source, tmp, "20261124T000001-old", old, sizes, diamond)
    x_dir, x_rep = search(source, tmp, "20261124T000002-ext", source["src"], TS.SIZES, diamond, extend_from=e_dir, verify_proteins=verify)
    f_dir, f_rep = search(source, tmp, "20261124T000003-full", source["src"], TS.SIZES, diamond, dbsize=e_rep["database"]["letters"], algo=0, self_batch_proteins=1)
    return e_rep, x_rep, f_rep, same_search(x_dir, f_dir), gone


def test_an_extended_search_is_the_full_search(run_source, tmp_path):
    source = run_source
    e_rep, x_rep, f_rep, problems, gone = extended_equals_full(source, tmp_path, source["diamond"], verify=60)
    assert problems == []
    x = x_rep["extension"]
    assert x["genomes_before"] == e_rep["queries"]["genomes"] and x["new_genomes"] == 3 and x["new_training_genomes"] == 2
    assert x["dbsize"] == e_rep["database"]["letters"] and x_rep["settings"]["dbsize"] == x["dbsize"] and x_rep["settings"]["algo"] == 0
    assert x["verify"]["proteins"] > 0 and x["verify"]["early_mismatches"] == 0 and x["verify"]["late_mismatches"] == 0
    assert x_rep["rows"]["hsps"] == f_rep["rows"]["hsps"] > e_rep["rows"]["hsps"]
    assert len(x_rep["timing"]["main_s"]) < len(f_rep["timing"]["main_s"]) and x_rep["timing"]["xold_s"]     # the earlier genomes not searched again in full


@pytest.mark.skipif(shutil.which("diamond") is None, reason="DIAMOND is not installed here (it is in the CI environment)")
def test_an_extended_search_is_the_full_search_with_diamond(run_source, tmp_path):
    """With DIAMOND: the extension of an earlier search made with algo 0 (and without, as the k3 search was: DIAMOND
    chooses the seed search algorithm; recorded, not required) equals the full search; the full search does not
    depend on how the queries are cut into calls. The message names every outcome."""
    source, d, out = run_source, shutil.which("diamond"), {}
    old, sizes, _ = earlier(source["src"], source["metas"])
    letters = None
    for k, (label, block) in enumerate((("earlier search with DIAMOND's choice of algorithm", {}), ("earlier search with algo 0", dict(algo=0)),
                                        ("earlier search with algo 0 and dbsize", "dbsize"))):
        if block == "dbsize":
            block = dict(algo=0, dbsize=letters)
        e_dir, e_rep = search(source, tmp_path, "20261124T00001%d-old" % k, old, sizes, d, **block)
        letters = e_rep["database"]["letters"]
        try:
            x_dir, _ = search(source, tmp_path, "20261124T00002%d-ext" % k, source["src"], TS.SIZES, d, extend_from=e_dir, verify_proteins=100000)
        except SystemExit as err:
            out[label] = "the extension stopped: %s" % str(err)[:1500]
            continue
        if "full" not in out:
            f_dir, f_rep = search(source, tmp_path, "20261124T000030-full", source["src"], TS.SIZES, d, dbsize=letters, algo=0, self_batch_proteins=1)
            g_dir, _ = search(source, tmp_path, "20261124T000031-full", source["src"], TS.SIZES, d, dbsize=letters, algo=0, self_batch_proteins=1,
                              chunk_proteins=37)
            out["full"] = "made"
            out["full search cut into other calls"] = same_search(f_dir, g_dir) or "the same"
        out[label] = same_search(x_dir, f_dir) or "the same"
    msg = json.dumps(out, indent=1)
    assert out["earlier search with algo 0"] == "the same" and out["earlier search with algo 0 and dbsize"] == "the same", msg
    assert out["full search cut into other calls"] == "the same", msg


def test_an_extension_that_differs_stops_before_the_long_calls(run_source, tmp_path, monkeypatch):
    source = run_source
    old, sizes, _ = earlier(source["src"], source["metas"])
    e_dir, _ = search(source, tmp_path, "20261124T000001-old", old, sizes, source["diamond"])
    real = WS.diamond_search

    def no_rows_from_the_new_genomes(diamond, queries, db, s, out_lines):
        return real(diamond, queries, db, s, lambda lines: out_lines(x for x in lines if not db.endswith("db_new.dmnd")))
    monkeypatch.setattr(WS, "diamond_search", no_rows_from_the_new_genomes)
    with pytest.raises(SystemExit, match="differs from the search done again"):
        search(source, tmp_path, "20261124T000002-ext", source["src"], TS.SIZES, source["diamond"], extend_from=e_dir, verify_proteins=10000)
    with pytest.raises(SystemExit, match="extend_from extends a full search"):
        WS.settings(dict(rel_w12_search=dict(source="s", extend_from="e", sample=0.5)))


def test_an_extension_refuses_changed_corpus_files(run_source, tmp_path):
    source = run_source
    old, sizes, _ = earlier(source["src"], source["metas"])
    e_dir, _ = search(source, tmp_path, "20261124T000001-old", old, sizes, source["diamond"])
    m = json.load(open(os.path.join(e_dir, WS.OUT, "manifest.json")))
    gid = json.load(open(os.path.join(e_dir, WS.OUT, "genomes.json")))["genomes"][0]["id"]
    p = os.path.join(e_dir, WS.OUT, gid + ".npz")
    a = H.load(p)
    H.save(p, dict(a, searched=np.zeros_like(a["searched"])))              # a rows file that is not the one its manifest names
    with pytest.raises(SystemExit, match="rows files of the extended search differ"):
        search(source, tmp_path, "20261124T000002-ext", source["src"], TS.SIZES, source["diamond"], extend_from=e_dir)
    assert m["settings"].get("dbsize") is None
