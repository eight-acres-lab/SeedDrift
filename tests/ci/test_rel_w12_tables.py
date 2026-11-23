"""Stages rel_w12_v1 and rel_w12_islands (ci/stages/rel/w12_v1.py, w12_islands.py): the tables of version 1 from a full
run of stage rel_w12_search (the stand-in DIAMOND of tests/ci/test_rel_w12_search.py) over the synthetic source run,
with two development chromosomes; each table set passes the hand-off check, the soundness check of version 1 and its
gate, and the development chromosomes' set is marked as never for training."""
import json
import os
import stat
import sys

import numpy as np
import pytest

import rel_fixtures as RF
import test_rel_w12_search as T12
import test_rel_w2_stage as TS
from ci.gates import GateFailure
from ci.stages.common import Stage
from ci.stages.rel import w12_islands, w12_v1 as WV
from seeddrift.train.rel import api, handoff
from seeddrift.train.rel.tables import v1_build as V1


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w12t")
    src = str(tmp / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    metas = TS.make_source(src)
    T12.add_corpus(src, metas)
    accs = T12.dev_accessions(2)
    T12.add_islands(src, accs, [m["id"] for m in metas if m["split"] == "train"][0])
    fake = tmp / "diamond"
    fake.write_text(T12.FAKE % dict(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    source = dict(tmp=tmp, src=src, metas=metas, diamond=str(fake))
    search, _, _ = T12.run_stage(source, tmp, "20261123T000020-search", islands=True)
    with open(os.path.join(search, "stages", "rel_w12_search.json")) as f:
        stage = json.load(f)
    with open(os.path.join(search, "record.json"), "w") as f:
        json.dump(dict(run_id=os.path.basename(search), status="passed", stages=[stage], pipeline="rel_tables", git=dict(sha="3" * 40)), f)
    block = dict(search_from=search, expect=TS.SIZES, rows=32, train_per_group=3, heldout_per_group=2, per_chromosome=4, shard_tables=20, processes=2,
                 dist_k=4, dist_size=256)
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        TS.patch(mp, tmp)
        for which, run_id in (("v1", "20261123T000021-v1"), ("islands", "20261123T000022-islands")):
            run_dir, cfg = RF.rel_run(tmp, stages=("w12_v1",), run_id=run_id, **{WV.WHICH[which]["block"]: block})
            with Stage(run_dir, "rel_w12_" + which) as st:
                out[which] = dict(run_dir=run_dir, cfg=cfg, doc=WV.build(st, cfg, run_dir, which))
    return dict(tmp=tmp, src=src, metas=metas, accs=accs, search=search, out=out)


def test_tables_of_version_1(runs, monkeypatch):
    TS.patch(monkeypatch, runs["tmp"])
    r = runs["out"]["v1"]
    doc, run_dir = r["doc"], r["run_dir"]
    assert handoff.tableset_problems(doc, run_dir) == [] and doc["table_version"] == V1.TABLE_VERSION and doc["sealed_check"]["passed"]
    assert {k: v["tables"] for k, v in doc["parts"].items()} == dict(train=36, heldout=24) and "not_for_training" not in doc
    t = doc["parts"]["train"]["targets"]
    assert t["tables"] == 36 and 0 < t["distinct"] <= 36 and t["max_times"] >= 1
    roles = {os.path.basename(f["path"]).split(".")[1]: f["role"] for f in doc["files"] if f["path"].endswith(".npy")}
    assert roles["genome_of_row"] == "shard" and roles["row_gene"] == "meta" and roles["row_first"] == "meta" and roles["tok"] == "shard"
    with open(os.path.join(run_dir, doc["index"])) as f:
        idx = json.load(f)
    with open(os.path.join(runs["search"], "data", "rel_w12", "search", "genomes.json")) as f:
        assert [g["id"] for g in idx["genomes"]] == [g["id"] for g in json.load(f)["genomes"]]       # the search's numbering
    train = {i for i, g in enumerate(idx["genomes"]) if g["split"] == "train"}
    for part in ("train", "heldout"):
        folder, shards = os.path.join(run_dir, doc["parts"][part]["folder"]), [(x["name"], x["tables"]) for x in doc["parts"][part]["shards"]]
        examined = {i for i, g in enumerate(idx["genomes"]) if g["split"] == part}
        assert V1.verify_set(folder, shards, 32, part, examined, train, len(idx["groups"])) == []
        kind = np.concatenate([np.load(os.path.join(folder, "%s.kind.npy" % n)) for n, _ in shards])
        assert (kind == api.KIND_ABSENT).any()
    text = json.dumps(doc) + open(os.path.join(run_dir, WV.WHICH["v1"]["out"], "census.md")).read()
    assert not any(x in text for x in TS.SEALED) and not any(a in text for a in runs["accs"])
    with Stage(run_dir, "gates_rel_w12_v1") as st:
        WV.gate(st, r["cfg"], run_dir)
    assert json.load(open(os.path.join(run_dir, "gates", "G4_rel_w12_v1.json")))["status"] == "pass"


def test_development_chromosomes_have_their_own_table_set_never_for_training(runs, monkeypatch):
    TS.patch(monkeypatch, runs["tmp"])
    r = runs["out"]["islands"]
    doc, run_dir = r["doc"], r["run_dir"]
    assert handoff.tableset_problems(doc, run_dir) == [] and doc["purpose"] == "heldout" and doc["not_for_training"] is True
    assert list(doc["parts"]) == ["island"] and doc["parts"]["island"]["tables"] == 8
    splits = {g["id"]: g["split"] for g in doc["genomes"]}
    assert {splits[a] for a in runs["accs"]} == {"island"} and set(splits.values()) <= {"island", "train"}
    folder = os.path.join(run_dir, doc["parts"]["island"]["folder"])
    g = np.concatenate([np.load(os.path.join(folder, "%s.genome.npy" % x["name"])) for x in doc["parts"]["island"]["shards"]])
    with open(os.path.join(run_dir, doc["index"])) as f:
        ids = [x["id"] for x in json.load(f)["genomes"]]
    assert {ids[i] for i in g[:, 0]} == set(runs["accs"]) and (g[:, 1:] >= 0).any()
    rec = doc["sealed_check"]                                           # the id check covers the development chromosomes too
    assert rec["ids"]["distinct"] >= len(doc["genomes"]) and "island" in rec["ids"]["allowed_splits"] and rec["islands"]["checked"] == len(runs["accs"])
    assert "island" not in runs["out"]["v1"]["doc"]["sealed_check"]["ids"]["allowed_splits"]        # the tables of version 1 never load them
    with Stage(run_dir, "gates_rel_w12_islands") as st:
        w12_islands.gate(st, r["cfg"], run_dir)
    bad = dict(doc)
    bad.pop("not_for_training")
    p = os.path.join(run_dir, WV.WHICH["islands"]["out"], "tableset.json")
    json.dump(handoff.finish(bad), open(p, "w"))
    with pytest.raises(GateFailure):
        with Stage(run_dir, "gates_rel_w12_islands") as st:
            w12_islands.gate(st, r["cfg"], run_dir)


def test_a_sample_search_is_refused(runs, tmp_path):
    import shutil
    copy = str(tmp_path / "search")
    shutil.copytree(runs["search"], copy)
    mp = os.path.join(copy, "data", "rel_w12", "search", "manifest.json")
    man = json.load(open(mp))
    man["settings"]["sample"] = 0.01
    json.dump(man, open(mp, "w"))
    with pytest.raises(SystemExit):
        WV.search_run(copy)                                              # no longer the output its stage recorded, and a sample


def test_the_frame_index_answers_the_online_layout_as_the_whole_index(runs, monkeypatch):
    """With rel_w12_v1.frame_index the table set carries the pairs of the search's index that the online frame layout
    reads, so a machine that trains on the tables needs that subset, not the whole index."""
    from seeddrift.train.rel.tables import align
    from seeddrift.train.rel.tables import v1_hits as H
    from seeddrift.train.rel import api
    TS.patch(monkeypatch, runs["tmp"])
    block = dict(search_from=runs["search"], expect=TS.SIZES, rows=32, train_per_group=3, heldout_per_group=2, shard_tables=20, processes=2,
                 dist_k=4, dist_size=256, frame_index="all")
    run_dir, cfg = RF.rel_run(runs["tmp"], stages=("w12_v1",), run_id="20261123T000023-v1f", rel_w12_v1=block)
    with Stage(run_dir, "rel_w12_v1") as st:
        doc = WV.build(st, cfg, run_dir, "v1")
    assert handoff.tableset_problems(doc, run_dir) == [] and doc["frame_index"]["pairs"] > 0
    full = H.Index(os.path.join(runs["search"], "data", "rel_w12", "search", "index"))
    sub = H.Index(os.path.join(run_dir, doc["frame_index"]["folder"]))
    assert doc["frame_index"]["pairs"] < len(full.a["tgenome"])
    seen = 0
    for part in ("train", "heldout"):
        folder = os.path.join(run_dir, doc["parts"][part]["folder"])
        for x in doc["parts"][part]["shards"]:
            z = {f: np.load(os.path.join(folder, "%s.%s.npy" % (x["name"], f))) for f in V1.fields(32)}
            for t in range(x["tables"]):
                tab = {f: z[f][t] for f in z}
                meta = dict(genome=tab["genome"], row_gene=tab["row_gene"], row_first=tab["row_first"])
                for b in range(api.N_BLOCKS):
                    one, two = align.frame_layout_online(tab, meta, full, b), align.frame_layout_online(tab, meta, sub, b)
                    assert all(np.array_equal(u, v) for u, v in zip(one[:3], two[:3])) and one[3].keys() == two[3].keys()
                    seen += len(one[3])
    assert seen > 0                                                       # rows were placed through the index


def test_training_targets_drawn_over_the_group_and_the_same_held_out_tables(runs, monkeypatch):
    """rel_w12_v1.train_sampling gene (the programme lead, 2026-11-23): the training targets are drawn over all genes of a
    group without replacement, so no target is used more than once above the group's mean; the held-out tables stay
    those of the other way, byte for byte."""
    TS.patch(monkeypatch, runs["tmp"])
    block = dict(search_from=runs["search"], expect=TS.SIZES, rows=32, train_per_group=40, heldout_per_group=2, shard_tables=20, processes=2,
                 dist_k=4, dist_size=256, train_sampling="gene", max_uses_per_target=1)
    run_dir, cfg = RF.rel_run(runs["tmp"], stages=("w12_v1",), run_id="20261123T000024-v1g", rel_w12_v1=block)
    with Stage(run_dir, "rel_w12_v1") as st:
        doc = WV.build(st, cfg, run_dir, "v1")
    assert handoff.tableset_problems(doc, run_dir) == []
    assert doc["parts"]["heldout"]["sha256"] == runs["out"]["v1"]["doc"]["parts"]["heldout"]["sha256"]
    t = doc["parts"]["train"]["targets"]
    assert t["sampling"] == "gene" and runs["out"]["v1"]["doc"]["parts"]["train"]["targets"]["sampling"] == "genome"
    for g, d in t["by_group"].items():
        if g == "all":
            continue
        assert d["max_uses"] == 1 and d["tables"] == min(40, d["targetable_genes"]) == t["tables_by_group"][g], (g, d)   # at most one use each
    assert t["max_uses_per_target"] == 1 and t["by_group"]["all"]["tables"] == sum(t["tables_by_group"].values()) == doc["parts"]["train"]["tables"]
    with pytest.raises(SystemExit, match="max_uses_per_target"):
        WV.settings(dict(rel_w12_v1=dict(search_from="x", max_uses_per_target=5)))


def test_the_use_of_target_genes_is_reported_per_group(runs, monkeypatch, tmp_path):
    from ci.stages.rel import w12_tstats as WT
    TS.patch(monkeypatch, runs["tmp"])
    v1 = runs["out"]["v1"]["run_dir"]
    rp = os.path.join(v1, "record.json")
    json.dump(dict(json.load(open(rp)), status="passed"), open(rp, "w"))
    run_dir, cfg = RF.rel_run(runs["tmp"], stages=("w12_v1",), run_id="20261123T000025-tstats", rel_w12_tstats=dict(tables_from=v1))
    with Stage(run_dir, "rel_w12_tstats") as st:
        rep = WT.build(st, cfg, run_dir)
    doc = runs["out"]["v1"]["doc"]
    for part in ("train", "heldout"):
        d = rep["parts"][part]
        assert d["tables"] == doc["parts"][part]["tables"] and d["by_group"]["all"]["distinct"] == doc["parts"][part]["targets"]["distinct"]
        assert d["by_group"]["all"]["max_uses"] == doc["parts"][part]["targets"]["max_times"] and d["by_group"]["all"]["targetable_genes"] > 0
    with Stage(run_dir, "gates_rel_w12_tstats") as st:
        WT.gate(st, cfg, run_dir)
    assert json.load(open(os.path.join(run_dir, "gates", "G4_rel_w12_tstats.json")))["status"] == "pass"


def test_the_held_out_tables_must_be_the_reference_s_byte_for_byte(runs, monkeypatch):
    """rel_w12_v1.heldout_equals (the programme lead, 2026-11-23): the gate compares every held-out file of the set,
    hashed again, with the reference set's; one byte changed fails it and names the file."""
    TS.patch(monkeypatch, runs["tmp"])
    ref = runs["out"]["v1"]
    block = dict(search_from=runs["search"], expect=TS.SIZES, rows=32, train_per_group=40, heldout_per_group=2, shard_tables=20, processes=2,
                 dist_k=4, dist_size=256, train_sampling="gene", heldout_equals=dict(run=ref["run_dir"], sha256=ref["doc"]["sha256"]))
    run_dir, cfg = RF.rel_run(runs["tmp"], stages=("w12_v1",), run_id="20261123T000026-v1he", rel_w12_v1=block)
    with Stage(run_dir, "rel_w12_v1") as st:
        WV.build(st, cfg, run_dir, "v1")
    with Stage(run_dir, "gates_rel_w12_v1") as st:
        WV.gate(st, cfg, run_dir)
    assert json.load(open(os.path.join(run_dir, "gates", "G4_rel_w12_v1.json")))["status"] == "pass"
    folder = os.path.join(run_dir, "data", "rel_w12", "v1", "heldout")
    name = sorted(n for n in os.listdir(folder) if n.endswith(".tok.npy"))[0]
    p = os.path.join(folder, name)
    body = bytearray(open(p, "rb").read())
    body[-1] ^= 1
    open(p, "wb").write(bytes(body))
    with pytest.raises(GateFailure):
        with Stage(run_dir, "gates_rel_w12_v1") as st:
            WV.gate(st, cfg, run_dir)
    ev = json.load(open(os.path.join(run_dir, "gates", "G4_rel_w12_v1.json")))
    he = [x for x in ev["evidence"]["problems"] if x.get("check") == "heldout_equals"]
    assert he and he[0]["files"] == [name]
    body[-1] ^= 1
    open(p, "wb").write(bytes(body))
    bad = dict(cfg, rel_w12_v1=dict(block, heldout_equals=dict(run=ref["run_dir"], sha256="0" * 64)))
    with pytest.raises(GateFailure):                                      # another reference than the one named
        with Stage(run_dir, "gates_rel_w12_v1") as st:
            WV.gate(st, bad, run_dir)
    with pytest.raises(SystemExit, match="heldout_equals"):
        WV.settings(dict(rel_w12_islands=dict(search_from="x", heldout_equals=dict(run="r", sha256="s"))), "islands")
