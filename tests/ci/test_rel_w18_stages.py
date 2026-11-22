"""The stages of package W18 on a small synthetic source run (the one of test_rel_w2_stage.py): rel_w18_rulers
(ci/stages/rel/w18_rulers.py) takes a passed run's training arrays and homolog rows and a passed distance run by their
recorded checksums, leaves the sealed test split unopened, and writes a ruler set of implant tables that the hand-off
checks accept; rel_w18_readout (ci/stages/rel/w18_readout.py) turns per-cell bits into the readouts and their AUCs.
The stages are not yet in configs/rel/kinds.yaml (a request asks the coordinator), so the tests call build and gate
with the plain Stage."""
import json
import os

import numpy as np
import pytest

import rel_fixtures as RF
import test_rel_w2_stage as W2S
from ci.hashing import sha256_file
from ci.stages.common import Stage
from ci.stages.rel import w2_v0, w18_rulers
from seeddrift.train.rel import api, handoff
from seeddrift.train.rel.evalh import shards
from seeddrift.train.rel.readouts import w18_passes as P
from seeddrift.train.rel.tables import v0_build as V
from seeddrift.train.rel.tables import v0_dist as DI


def make_dist(run, src):
    """A passed run of stage w2_dist on the source run src: its genome list, distances (k = 4 here) and report."""
    s = w2_v0.source(src)
    metas = [m for m in s["index"]["genomes"] if m["split"] in V.OPEN_SPLITS]
    folder = os.path.join(run, "data", "rel_w2", "dist")
    os.makedirs(folder)
    os.makedirs(os.path.join(run, "stages"))
    sk = []
    for m in metas:
        with np.load(os.path.join(src, "data", "train_data", m["id"] + ".npz")) as z:
            sk.append(DI.sketch(z["codons"], z["codon_off"], 4, 256))
    jac, _ = DI.matrix([x[0] for x in sk], [x[1] for x in sk], 256, 1)
    dist = DI.mash(jac, 4)
    np.fill_diagonal(dist, 0.0)
    files = []
    gp = os.path.join(folder, "genomes.json")
    with open(gp, "w") as f:
        json.dump(dict(genomes=[dict(id=m["id"], group=m["group"], split=m["split"]) for m in metas], ks=[4]), f)
    dp = os.path.join(folder, "distances_k4.npy")
    np.save(dp, dist.astype(np.float32))
    for p in (gp, dp):
        files.append(dict(path=os.path.relpath(p, run), sha256=sha256_file(p, use_sidecar=False), bytes=os.path.getsize(p)))
    rp = os.path.join(folder, "report.json")
    with open(rp, "w") as f:
        json.dump(dict(format="rel-w2-dist", ks=[4], edges=[0.05, 0.2, 0.5, 1.0], source=s["provenance"], files=files), f)
    with open(os.path.join(run, "stages", "rel_w2_dist.json"), "w") as f:
        json.dump(dict(name="rel_w2_dist", status="passed", outputs=[dict(name="dist_report", path="data/rel_w2/dist/report.json",
                                                                          sha256=sha256_file(rp, use_sidecar=False))]), f)
    with open(os.path.join(run, "record.json"), "w") as f:
        json.dump(dict(run_id=os.path.basename(run), status="passed"), f)
    return metas, dist


def patch(mp, tmp):
    W2S.patch(mp, tmp)
    mp.setattr(w18_rulers, "declared_tables", lambda path: {})
    mp.setattr(w18_rulers, "nuclear_codes", lambda taxdump, taxids, tax=None: {int(t): 11 for t in taxids})


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w18")
    src = str(tmp / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    metas, truth = W2S.make_source(src, with_truth=True)
    dist_run = str(tmp / "ci-root" / "runs" / "20261122T000000-dist")
    os.makedirs(dist_run)
    dmetas, dist = make_dist(dist_run, src)
    run_dir, cfg = RF.rel_run(tmp)
    cfg = dict(cfg, rel=dict(machine="i1", stages=["w18_rulers"]),
               rel_w18_rulers=dict(source=src, dist=dist_run, expect=W2S.SIZES, rows=8, triples_per_group=3, k=4, min_codons=20, length_tol=0.6,
                                   shard_tables=9, processes=1))
    with pytest.MonkeyPatch.context() as mp:
        patch(mp, tmp)
        mp.setattr(w18_rulers, "declared_tables", lambda path: {m["id"]: 11 for m in metas})
        with Stage(run_dir, "rel_w18_rulers") as st:
            doc = w18_rulers.build(st, cfg, run_dir)
    return dict(tmp=tmp, src=src, metas=metas, truth=truth, run_dir=run_dir, cfg=cfg, doc=doc, dist=dist, dmetas=dmetas)


def _gate(run_dir, name):
    with open(os.path.join(run_dir, "gates", name + ".json")) as f:
        return json.load(f)


def test_rulers_stage_writes_a_ruler_set(built):
    run_dir, doc = built["run_dir"], built["doc"]
    assert handoff.tableset_problems(doc, run_dir) == []
    assert doc["kind"] == "ruler" and doc["purpose"] == "ruler" and doc["producer"]["stage"] == "w18_rulers" and doc["sealed_check"]["passed"] is True
    assert doc["tables"] % 3 == 0 and doc["tables"] >= 9
    text = json.dumps(doc) + open(os.path.join(run_dir, w18_rulers.OUT, "index.json")).read()
    assert not any(g in text for g in W2S.SEALED)                     # the sealed genomes are named nowhere
    split = {m["id"]: m["split"] for m in built["metas"]}
    assert {g["split"] for g in doc["genomes"]} <= {"train", "heldout"} and all(split[g["id"]] == g["split"] for g in doc["genomes"])


def test_the_evaluator_reads_the_ruler_set_and_the_meta_file(built):
    run_dir, doc = built["run_dir"], built["doc"]
    sets = shards.by_split(run_dir, [doc])
    assert list(sets) == [1]
    ts = sets[1]
    meta = P.load_meta(run_dir, doc)
    assert ts.tables == doc["tables"] == len(meta["triple"])
    got = list(P.tables_of(ts, meta))
    assert [g[0] for g in got] == list(range(ts.tables))
    k = ts.load(0, ts.tables, ("kind", "genome"))
    for tid, tab, tier in got:
        exists = (k["kind"][tid] != api.KIND_NONE).any(axis=1)
        assert ((tier >= 0) == (exists & (np.arange(len(tier)) > 0))).all()


def test_rulers_gate_passes_and_catches_a_broken_triple(built, monkeypatch):
    run_dir, cfg = built["run_dir"], built["cfg"]
    patch(monkeypatch, built["tmp"])
    with Stage(run_dir, "gates_rel_w18_rulers") as st:
        w18_rulers.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w18_rulers")["status"] == "pass"
    out = {"genome": np.zeros((3, 8), np.int32), "kind": np.zeros((3, 8, api.N_BLOCKS), np.uint8), "n": np.zeros((3, 8, api.N_BLOCKS), np.int16),
           "split": np.ones(3, np.int8)}
    out["genome"][:, 1:] = -1
    meta = dict(triple=np.zeros(3, np.int32), variant=np.array([0, 1, 1], np.int8), donor=np.ones(3, np.int32))
    from bench.rel import implant as I
    assert any("three tables" in p for p in I.verify(out, meta, {0, 1}, set()))
    assert w18_rulers.summary(run_dir)["json"]["tables"] == built["doc"]["tables"]


def test_the_meta_file_is_checked_against_the_manifest(built, tmp_path):
    doc = json.loads(json.dumps(built["doc"]))
    for e in doc["files"]:
        if e["path"].endswith(P.META_FILE):
            e["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        P.load_meta(built["run_dir"], doc)


def test_eligibility_needs_the_standard_code_twice(tmp_path):
    metas = [dict(id="a", split="heldout", group="g", taxid=1), dict(id="b", split="heldout", group="g", taxid=2), dict(id="c", split="heldout", group="g", taxid=3),
             dict(id="d", split="train", group="g", taxid=1)]
    ok, left = w18_rulers.eligibility(metas, dict(a=11, b=11, c=4, d=11), {1: 1, 2: 4, 3: 11})
    assert ok == {"a"} and sorted(x["id"] for x in left["g"]) == ["b", "c"]
    nodes = tmp_path / "nodes.dmp"
    nodes.write_text("1\t|\t1\t|\tno rank\t|\t\t|\t8\t|\t0\t|\t1\t|\t0\t|\t0\t|\n2\t|\t1\t|\tgenus\t|\t\t|\t0\t|\t1\t|\t4\t|\t0\t|\t0\t|\n3\t|\t1\t|\tgenus\t|\n")
    assert w18_rulers.nuclear_codes(str(tmp_path), [1, 2, 3]) == {1: 1, 2: 4, 3: -1}


def w6_form(examples):
    """A scorer of the evaluation stage's form, from the toy model of test_rel_w18_readouts: flat cells with their example."""
    import test_rel_w18_readouts as RT
    got = RT.toy_score([(e["grid"], e["truth"]) for e in examples])
    return dict(bits=np.concatenate([g[0] for g in got]), aa_bits=np.concatenate([g[1] for g in got]), entropy=np.concatenate([g[2] for g in got]),
                example=np.concatenate([np.full(len(g[0]), k) for k, g in enumerate(got)]))


def test_run_ruler_with_a_scorer_of_the_evaluation_stage(built, tmp_path):
    from seeddrift.train.rel.data import tables as T
    run_dir, doc = built["run_dir"], built["doc"]
    tabs = T.open_split(run_dir, doc, 1)
    info = dict(arm="toy", weights="untrained", relatives=True, layout="stored", render_version=T.RENDER_VERSION, run="r", commit="0" * 40, root=run_dir)
    head = P.run_ruler(w6_form, tabs, info, str(tmp_path), presence=True)
    cells, h = P.load_cells(str(tmp_path / P.RULER_CELLS))
    meta = P.load_meta(run_dir, doc)
    assert h == head and h["tableset_sha256"] == doc["sha256"] and h["presence"] is True
    assert cells["digest"] == P.plan_digest(P.tables_of(tabs.ts, meta), True, True)[0]
    norel = P.run_ruler(w6_form, tabs, dict(info, relatives=False), str(tmp_path / "n"), presence=True)
    assert norel["presence"] is False and norel["pass_names"][P.PASS_ID["N"]] == "N" and norel["passes"] == doc["tables"]
    with pytest.raises(ValueError):
        P.run_ruler(w6_form, tabs, dict(info, layout="leakfree"), str(tmp_path / "x"))
