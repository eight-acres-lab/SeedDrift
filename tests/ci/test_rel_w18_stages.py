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
    from ci.stages.rel import kinds
    W2S.patch(mp, tmp)
    mp.setattr(w18_rulers, "declared_tables", lambda path: {})
    mp.setattr(w18_rulers, "nuclear_codes", lambda taxdump, taxids, tax=None: {int(t): 11 for t in taxids})
    real = kinds.reads
    mp.setattr(kinds, "reads", lambda name, *a: ("ruler",) if name in ("w18_readout", "w18_offline") else real(name, *a))   # until registered


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
               rel_w18_rulers=dict(source=src, dist=dist_run, expect=W2S.SIZES, rows=8, per_tier=2, group_tier_cap=4, k=4, min_codons=20, length_tol=0.6,
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
    with open(os.path.join(run_dir, w18_rulers.OUT, "index.json")) as f:
        index = json.load(f)
    assert index["tier_expectation"]["0"] == index["tier_expectation"]["1"] == "expected too few" and index["units"]
    assert all(t["mask_sha256"] for t in index["triples"]) and {t["scenario"] for t in index["triples"]} <= {"in_place", "donor_unsampled"}
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
    for tid, tab, tier, mid in got:
        assert mid == meta["parent"][tid]
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
    out["start"], out["strand"], out["replicon"] = (np.full((3, 8, api.N_BLOCKS), v) for v in (-1, 0, -1))
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
    inplace = P.run_ruler(w6_form, tabs, dict(info, relatives=False), str(tmp_path / "p"), scenarios=[0])
    assert inplace["scenarios"] == [0] and inplace["passes"] == int((meta["scenario"] == 0).sum()) < doc["tables"]
    c0, _ = P.load_cells(str(tmp_path / "p" / P.RULER_CELLS))
    assert set(meta["scenario"][np.unique(c0["table"])].tolist()) == {0}
    with pytest.raises(ValueError):
        P.run_ruler(w6_form, tabs, dict(info, layout="leakfree"), str(tmp_path / "x"))


def uniform_form(examples):
    """An untrained stand-in: the uniform distribution over tokens 1..65."""
    import test_rel_w18_readouts as RT
    cls = RT.CLS
    pc = np.bincount(cls, minlength=23) / 65.0
    out = dict(bits=[], aa_bits=[], entropy=[], example=[])
    for k, e in enumerate(examples):
        t = e["truth"][np.nonzero(e["truth"])].astype(np.int64)
        out["bits"].append(np.full(len(t), np.log2(65)))
        out["aa_bits"].append(-np.log2(pc[cls[t - 1]]))
        out["entropy"].append(np.full(len(t), np.log2(65)))
        out["example"].append(np.full(len(t), k))
    return {k: np.concatenate(v) for k, v in out.items()}


@pytest.fixture(scope="module")
def evaluated(built, tmp_path_factory):
    """The ruler run marked passed, and an evaluation run with the per-cell files of three arms, listed in its record of
    stage rel_w6_eval."""
    from seeddrift.train.rel.data import tables as T
    run_dir, doc = built["run_dir"], built["doc"]
    with open(os.path.join(run_dir, "stages", "rel_w18_rulers.json")) as f:
        stage = json.load(f)
    with open(os.path.join(run_dir, "record.json"), "w") as f:
        json.dump(dict(run_id=os.path.basename(run_dir), status="passed", stages=[stage]), f)
    tmp = tmp_path_factory.mktemp("w18eval")
    ev, cfg = RF.rel_run(tmp, kind="rel_traineval", stages=("w6_eval",), run_id="20261126T000000-exp-rel-00000000-eval")
    tabs = T.open_split(run_dir, doc, 1)
    base = dict(weights="final", relatives=True, layout="stored", render_version=T.RENDER_VERSION, run=os.path.basename(ev), commit="0" * 40, root=run_dir)
    with Stage(ev, "rel_w6_eval") as st:
        for folder, scorer, info, presence, scen, psc in (("toy", w6_form, dict(base, arm="toy"), True, None, [0]),
                                                          ("toy-norel", w6_form, dict(base, arm="toy-norel", relatives=False), False, None, None),
                                                          ("toy.untrained", uniform_form, dict(base, arm="toy", weights="untrained"), False, [0], None)):
            out = os.path.join(ev, "data", "rel_w6", "eval", folder)
            P.run_ruler(scorer, tabs, info, out, presence=presence, scenarios=scen, presence_scenarios=psc)
            st.output("ruler_cells_" + folder, os.path.join(out, P.RULER_CELLS))
    cfg = dict(cfg, rel=dict(machine="i1", stages=["w6_eval", "w18_readout"]),
               rel_w18_readout=dict(tables=dict(run=run_dir, path=os.path.join("data", "rel_w18", "rulers", "tableset.json")), reps=50))
    return dict(ev=ev, cfg=cfg, tabs=tabs)


def test_readout_stage_reads_every_arm_and_writes_the_aucs(built, evaluated, monkeypatch):
    from ci.stages.rel import w18_readout
    from seeddrift.train.rel.readouts import w18_readouts as R
    ev, cfg = evaluated["ev"], evaluated["cfg"]
    patch(monkeypatch, built["tmp"])
    monkeypatch.setattr(R, "MIN_GROUP", 1)                             # the fixture has a few triples per group; let them enter
    with Stage(ev, "rel_w18_readout") as st:
        rep = w18_readout.build(st, cfg, ev)
    assert sorted((a["arm"], a["weights"]) for a in rep["arms"]) == [("toy", "final"), ("toy", "untrained"), ("toy-norel", "final")]
    assert [a["scenarios"] for a in rep["arms"] if a["weights"] == "untrained"] == [[0]]
    assert [a["presence_scenarios"] for a in rep["arms"] if a["presence"]] == [[0]]
    assert rep["tables"]["sha256"] == built["doc"]["sha256"] and rep["tables"]["tables"] == built["doc"]["tables"]
    import csv
    with open(os.path.join(ev, w18_readout.OUT, "stats.tsv")) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    labels = {(r["label"], r["score"]) for r in rows}
    assert {("toy", "source_shift_aa"), ("toy", "gain_aa"), ("toy", "gain_rel_aa"), ("presence:toy", "gain"), ("untrained:toy", "gain"),
            ("toy-norel", "composition"), ("control", "best_similarity"), ("control", "coverage_shift"), ("control", "host_codon")} <= labels
    assert all(0 <= float(r["value"]) <= 1 for r in rows if r["value"])
    assert {r["scenario"] for r in rows} == {"in_place", "donor_unsampled"} and {r["rule9"] for r in rows} == {"on", "off"}
    assert {r["statistic"] for r in rows} == {"win", "pooled"}
    assert not any(r["readable"] == "True" for r in rows)              # the fixture is far too small to read anything
    g = [r for r in rows if r["label"] == "toy" and r["score"] == "gain" and r["stratum"] == "all" and r["comparison"] == "own" and r["groups"] != "0"
         and r["scenario"] == "in_place"]
    assert g and json.loads(g[0]["vs_untrained"])["reference"] is not None and json.loads(g[0]["vs_best_similarity"])["value"] is not None
    c = [r for r in rows if r["label"] == "toy-norel" and r["score"] == "composition" and r["groups"] != "0"]
    assert c and json.loads(c[0]["vs_host_codon"])["diff"] is not None
    with open(os.path.join(ev, w18_readout.OUT, "defined.tsv")) as f:
        defined = list(csv.DictReader(f, delimiter="\t"))
    keys = [(d["label"], d["of"], d["scenario"], d["rule9"], d["block"]) for d in defined]
    assert len(keys) == len(set(keys)) and {d["of"] for d in defined} == {"source_shift", "source_shift_aa"}   # once per readout, no repeats
    with open(os.path.join(ev, w18_readout.OUT, "coverage.tsv")) as f:
        cover = list(csv.DictReader(f, delimiter="\t"))
    assert len(cover) == 7 * 12 and {c["mark"] for c in cover} <= {"construct only", "not evaluated"}
    with open(os.path.join(ev, w18_readout.OUT, "scores.tsv")) as f:
        assert len(f.read().strip().splitlines()) == built["doc"]["tables"] + 1
    page = open(os.path.join(ev, w18_readout.OUT, "report.md")).read()
    assert "Coverage (charter rule 7)" in page and "Exploratory, outside the pre-registered protocol" in page
    assert {r["stratum_kind"] for r in rows} >= {"x_implant_gc", "x_gc_minus_own", "x_aa_distance"} and ("control", "host_aa") in labels
    with Stage(ev, "gates_rel_w18_readout") as st:
        w18_readout.gate(st, cfg, ev)
    assert _gate(ev, "G4_rel_w18_readout")["status"] == "pass"
    assert "too few" in open(os.path.join(ev, w18_readout.OUT, "report.md")).read()
    assert w18_readout.summary(ev)["json"]["tables"]["name"] == built["doc"]["name"]


def test_readout_stage_refuses_cells_of_another_plan(built, evaluated, tmp_path, monkeypatch):
    from ci.stages.rel import w18_readout
    from seeddrift.train.rel.data import tables as T
    ev, cfg = evaluated["ev"], evaluated["cfg"]
    patch(monkeypatch, built["tmp"])
    meta = P.load_meta(built["run_dir"], built["doc"])
    cells = P.run(P.tables_of(evaluated["tabs"].ts, meta), T.render, P.flat_scorer(w6_form), relatives=True, presence=False)
    bad = os.path.join(ev, "data", "rel_w6", "eval", "toy", P.RULER_CELLS)
    P.save_cells(bad, cells, arm="toy", weights="final", relatives=True, presence=True, layout="stored", render_version=T.RENDER_VERSION,
                 tableset_sha256=built["doc"]["sha256"], run="r", commit="0" * 40)
    with pytest.raises(SystemExit):                                    # not the file the evaluation's record lists
        with Stage(ev, "rel_w18_readout") as st:
            w18_readout.build(st, cfg, ev)
    rec = os.path.join(ev, "stages", "rel_w6_eval.json")
    with open(rec) as f:
        stage = json.load(f)
    for o in stage["outputs"]:
        if o["path"].endswith(os.path.join("toy", P.RULER_CELLS)):
            o["sha256"] = sha256_file(bad, use_sidecar=False)
    with open(rec, "w") as f:
        json.dump(stage, f)
    with pytest.raises(SystemExit, match="another plan"):              # listed now, but it scored the plan without presence passes
        with Stage(ev, "rel_w18_readout") as st:
            w18_readout.build(st, cfg, ev)


def test_readout_stage_reads_cells_from_several_evaluation_runs(built, evaluated, tmp_path, monkeypatch):
    """Arms evaluated on two machines: eval_run lists both runs; an arm folder in both is refused."""
    import shutil
    from ci.stages.rel import w18_readout
    ev, cfg = evaluated["ev"], evaluated["cfg"]
    patch(monkeypatch, built["tmp"])
    other = str(tmp_path / "20261126T000001-exp-rel-00000000-eval-i2")
    os.makedirs(os.path.join(other, "stages"))
    src = os.path.join(ev, "data", "rel_w6", "eval", "toy-norel")
    dst = os.path.join(other, "data", "rel_w6", "eval", "toy-norel")
    shutil.copytree(src, dst)
    p = os.path.join(dst, P.RULER_CELLS)
    with open(os.path.join(other, "stages", "rel_w6_eval.json"), "w") as f:
        json.dump(dict(name="rel_w6_eval", status="passed", outputs=[dict(name="ruler_cells_toy-norel", path=os.path.relpath(p, other),
                                                                          sha256=sha256_file(p, use_sidecar=False))]), f)
    s = w18_readout.settings(dict(cfg, rel_w18_readout=dict(cfg["rel_w18_readout"], eval_run=[ev, other])))
    with pytest.raises(SystemExit, match="two evaluation runs"):
        w18_readout.cell_files(s, ev)
    s = w18_readout.settings(dict(cfg, rel_w18_readout=dict(cfg["rel_w18_readout"], eval_run=[ev, other], arms=["toy", "toy.untrained"])))
    assert [a for a, _, _ in w18_readout.cell_files(s, ev)] == ["toy", "toy.untrained"]
    shutil.rmtree(src)
    try:
        s = w18_readout.settings(dict(cfg, rel_w18_readout=dict(cfg["rel_w18_readout"], eval_run=[ev, other])))
        got = w18_readout.cell_files(s, ev)
        assert sorted(a for a, _, _ in got) == ["toy", "toy-norel", "toy.untrained"] and [x[1] for x in got if x[0] == "toy-norel"] == [p]
    finally:
        shutil.copytree(dst, src)


@pytest.fixture(scope="module")
def v1stage(tmp_path_factory):
    """A ruler set of version 1 built by the stage from a passed full search run (W12's stand-in DIAMOND)."""
    import stat
    import sys
    import test_rel_w12_search as T12
    tmp = tmp_path_factory.mktemp("w18v1stage")
    src = str(tmp / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    metas = W2S.make_source(src)
    T12.add_corpus(src, metas)
    fake = tmp / "diamond"
    fake.write_text(T12.FAKE % dict(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    search_dir, _, _ = T12.run_stage(dict(tmp=tmp, src=src, metas=metas, diamond=str(fake)), tmp / "s", "20261123T000010-search")
    with open(os.path.join(search_dir, "record.json")) as f:
        rec = json.load(f)
    if rec.get("status") != "passed":                                  # the search stage's own test leaves the record to the launcher
        stages = []
        for name in sorted(os.listdir(os.path.join(search_dir, "stages"))):
            with open(os.path.join(search_dir, "stages", name)) as f:
                stages.append(json.load(f))
        with open(os.path.join(search_dir, "record.json"), "w") as f:
            json.dump(dict(rec, status="passed", stages=stages), f)
    dist_run = str(tmp / "ci-root" / "runs" / "20261122T000000-dist")
    os.makedirs(dist_run)
    make_dist(dist_run, src)
    run_dir, cfg = RF.rel_run(tmp)
    cfg = dict(cfg, rel=dict(machine="i1", stages=["w18_rulers"]),
               rel_w18_rulers=dict(version=1, search_from=search_dir, expect=W2S.SIZES, rows=8, per_tier=2, group_tier_cap=4, k=4,
                                   min_codons=20, length_tol=0.6, shard_tables=9, processes=1))     # the source and the distances from the search
    with pytest.MonkeyPatch.context() as mp:
        patch(mp, tmp)
        mp.setattr(w18_rulers, "declared_tables", lambda path: {m["id"]: 11 for m in metas})
        with Stage(run_dir, "rel_w18_rulers") as st:
            doc = w18_rulers.build(st, cfg, run_dir)
    return dict(tmp=tmp, run_dir=run_dir, cfg=cfg, doc=doc, dist_run=dist_run, metas=metas)


def test_rulers_stage_builds_tables_of_version_1(v1stage, monkeypatch):
    """Version 1: the homolog rows of a passed full run of stage w12_search on the same source, the fields of version
    1, every rows file checked against the search's manifest."""
    tmp, run_dir, cfg, doc, dist_run = v1stage["tmp"], v1stage["run_dir"], v1stage["cfg"], v1stage["doc"], v1stage["dist_run"]
    patch(monkeypatch, tmp)
    assert handoff.tableset_problems(doc, run_dir) == [] and doc["tables_of_version"] == 1 and doc["name"].startswith("w18-implant-v1")
    with open(os.path.join(run_dir, w18_rulers.OUT, "index.json")) as f:
        assert json.load(f)["distance_run"]["computed"]
    bad = dict(cfg, rel_w18_rulers=dict(cfg["rel_w18_rulers"], source=dist_run))
    with pytest.raises(SystemExit, match="another source"):
        w18_rulers.source_of(w18_rulers.settings(bad))
    with Stage(run_dir, "gates_rel_w18_rulers") as st:
        w18_rulers.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w18_rulers")["status"] == "pass"
    ts = shards.by_split(run_dir, [doc])[1]
    got = ts.load(0, ts.tables, ("kind", "genome_of_row", "tier"))
    assert (got["kind"] == api.KIND_ABSENT).any() and not (got["kind"] == api.KIND_NOT_SEARCHED).any()
    meta = P.load_meta(run_dir, doc)
    assert (got["tier"] == meta["row_tier"]).all() and (got["genome_of_row"][:, 0] == 0).all()


def test_arms_of_version_0_read_adapted_tables_of_version_1(v1stage, tmp_path_factory, monkeypatch):
    """The programme lead's adaptation (adapt_v0): the arms of tables of version 0 read the implant tables of version 1
    with fewer rows taken tier by tier and KIND_ABSENT read as KIND_NOT_SEARCHED; the readout stage adapts the same way
    and refuses a reading that mixes adapted and plain arms."""
    from ci.stages.rel import w18_readout
    from seeddrift.train.rel.data import tables as T
    tmp, run_dir, doc = v1stage["tmp"], v1stage["run_dir"], v1stage["doc"]
    patch(monkeypatch, tmp)
    with open(os.path.join(run_dir, "stages", "rel_w18_rulers.json")) as f:
        stage = json.load(f)
    with open(os.path.join(run_dir, "record.json"), "w") as f:
        json.dump(dict(run_id=os.path.basename(run_dir), status="passed", stages=[stage]), f)
    ev, cfg = RF.rel_run(tmp_path_factory.mktemp("w18adapt"), kind="rel_traineval", stages=("w6_eval",), run_id="20261126T000002-exp-rel-00000000-eval")
    tabs = T.open_split(run_dir, doc, 1)
    base = dict(weights="final", relatives=True, layout="stored", render_version=T.RENDER_VERSION, run=os.path.basename(ev), commit="0" * 40, root=run_dir)
    seen = []

    def watch(examples):
        seen.extend(int(e["grid"].shape[0]) for e in examples)
        assert not any((e["grid"] == api.ABSENT).any() for e in examples)        # the adapted input never shows ABSENT
        return w6_form(examples)
    with Stage(ev, "rel_w6_eval") as st:
        for folder, info in (("toy", dict(base, arm="toy")), ("toy-norel", dict(base, arm="toy-norel", relatives=False))):
            out = os.path.join(ev, "data", "rel_w6", "eval", folder)
            head = P.run_ruler(watch, tabs, info, out, adapt_v0=5)
            assert head["adapt_v0"] == 5
            st.output("ruler_cells_" + folder, os.path.join(out, P.RULER_CELLS))
    assert max(seen) <= 5
    cfg = dict(cfg, rel=dict(machine="i1", stages=["w6_eval", "w18_readout"]),
               rel_w18_readout=dict(tables=dict(run=run_dir, path=os.path.join("data", "rel_w18", "rulers", "tableset.json")), reps=20))
    with Stage(ev, "rel_w18_readout") as st:
        rep = w18_readout.build(st, cfg, ev)
    assert rep["adapt_v0"] == 5 and rep["status"].startswith("ADAPTED")
    page = open(os.path.join(ev, w18_readout.OUT, "report.md")).read()
    assert page.count("ADAPTED") >= 1 and "Near-tier donors without near reference rows (construct only)" in page   # tables of version 1
    plain = os.path.join(ev, "data", "rel_w6", "eval", "toy-plain")
    P.run_ruler(w6_form, tabs, dict(base, arm="toy-plain"), plain)
    rec = os.path.join(ev, "stages", "rel_w6_eval.json")
    with open(rec) as f:
        st_rec = json.load(f)
    st_rec["outputs"].append(dict(name="ruler_cells_toy-plain", path=os.path.relpath(os.path.join(plain, P.RULER_CELLS), ev),
                                  sha256=sha256_file(os.path.join(plain, P.RULER_CELLS), use_sidecar=False)))
    with open(rec, "w") as f:
        json.dump(st_rec, f)
    with pytest.raises(SystemExit, match="adapted in different ways"):
        with Stage(ev, "rel_w18_readout") as st:
            w18_readout.build(st, cfg, ev)


def test_offline_stage_and_a_leak_free_reading(v1stage, tmp_path_factory, monkeypatch):
    """Protocol w18-v3: stage rel_w18_offline builds the strict placements of every table and mask family with W3's
    aligner and passes the fourth test; run_ruler renders a leak-free reading on them (and only on them); the readout
    stage reads it and names the placements."""
    from ci.stages.rel import w18_offline, w18_readout
    from seeddrift.train.rel.data import tables as T
    tmp, run_dir, doc = v1stage["tmp"], v1stage["run_dir"], v1stage["doc"]
    patch(monkeypatch, tmp)
    with open(os.path.join(run_dir, "stages", "rel_w18_rulers.json")) as f:
        stage = json.load(f)
    with open(os.path.join(run_dir, "record.json"), "w") as f:
        json.dump(dict(run_id=os.path.basename(run_dir), status="passed", stages=[stage]), f)
    off, cfg = RF.rel_run(tmp_path_factory.mktemp("w18off"), kind="rel_tables", stages=("w18_rulers",), run_id="20261124T000001-exp-rel-00000000-offline")
    path = os.path.join("data", "rel_w18", "rulers", "tableset.json")
    cfg = dict(cfg, rel=dict(machine="i1", stages=["w18_offline"]), rel_w18_offline=dict(tables=dict(run=run_dir, path=path), processes=2, chunk=4))
    with Stage(off, "rel_w18_offline") as st:
        w18_offline.build(st, cfg, off)
    with Stage(off, "gates_rel_w18_offline") as st:
        w18_offline.gate(st, cfg, off)
    assert _gate(off, "G4_rel_w18_offline")["status"] == "pass"
    head = json.load(open(os.path.join(off, P.OFFLINE_DIR, P.OFFLINE_FILE)))
    assert head["fourth_test"]["failures"] == 0 and head["fourth_test"]["checked"] >= head["ruler"]["tables"] * 2
    assert "W18 strict offline rendering" in w18_offline.summary(off)["md"][0] and head["validity_limits"] == w18_offline.VALIDITY_LIMITS
    _, ts_, meta_ = w18_offline.ruler(w18_offline.settings(cfg))
    has = np.load(os.path.join(off, P.OFFLINE_DIR, "has.npy"))
    assert w18_offline.missing_families(ts_, meta_, has) == []
    cut = has.copy()
    c = P.FAMILIES.index("C")
    t_c = int(np.flatnonzero(cut[:, c])[0])
    cut[t_c, c] = False
    assert w18_offline.missing_families(ts_, meta_, cut) == [(t_c, "C")]                  # a family the plan uses, its placement missing
    with open(os.path.join(off, "stages", "rel_w18_offline.json")) as f:
        off_stage = json.load(f)
    with open(os.path.join(off, "record.json"), "w") as f:                      # the launcher marks the run passed
        json.dump(dict(run_id=os.path.basename(off), status="passed", stages=[off_stage]), f)
    ev, ecfg = RF.rel_run(tmp_path_factory.mktemp("w18lf"), kind="rel_traineval", stages=("w6_eval",), run_id="20261124T000002-exp-rel-00000000-eval")
    tabs = T.open_split(run_dir, doc, 1)
    base = dict(weights="final", relatives=True, layout="leakfree", render_version="leakfree-1", run=os.path.basename(ev), commit="0" * 40, root=run_dir)
    with pytest.raises(ValueError, match="strictly offline"):
        P.run_ruler(w6_form, tabs, dict(base, arm="x"), str(tmp / "x"))
    with pytest.raises(ValueError, match="leak-free layout"):
        P.run_ruler(w6_form, tabs, dict(base, arm="x", layout="stored"), str(tmp / "x"), offline=off)
    seen = []

    def watch(examples):
        seen.append(sum(int(e["ext"].sum()) for e in examples if e.get("ext") is not None))
        return w6_form(examples)
    with Stage(ev, "rel_w6_eval") as st:
        for folder, info in (("lf", dict(base, arm="lf")), ("lf-norel", dict(base, arm="lf-norel", relatives=False))):
            out = os.path.join(ev, "data", "rel_w6", "eval", folder)
            h = P.run_ruler(watch, tabs, info, out, presence=True, offline=off)
            assert h["offline"] == os.path.basename(off) and len(h["offline_sha256"]) == 64
            st.output("ruler_cells_" + folder, os.path.join(out, P.RULER_CELLS))
    assert sum(seen) > 0                                               # cells laid by extension reach the scorer
    ecfg = dict(ecfg, rel=dict(machine="i1", stages=["w6_eval", "w18_readout"]),
                rel_w18_readout=dict(tables=dict(run=run_dir, path=path), reps=20))
    with Stage(ev, "rel_w18_readout") as st:
        rep = w18_readout.build(st, ecfg, ev)
    assert rep["layout"] == dict(layout="leakfree", offline=os.path.basename(off), offline_sha256=h["offline_sha256"])
    # the 2 x 2 table: the same arms read under the stored layout in another run, compared
    ev2, ecfg2 = RF.rel_run(tmp_path_factory.mktemp("w18st"), kind="rel_traineval", stages=("w6_eval",), run_id="20261124T000003-exp-rel-00000000-eval")
    with Stage(ev2, "rel_w6_eval") as st:
        for folder, info in (("lf", dict(base, arm="lf", layout="stored", render_version=T.RENDER_VERSION)),):
            out = os.path.join(ev2, "data", "rel_w6", "eval", folder)
            P.run_ruler(w6_form, tabs, info, out, presence=True)
            st.output("ruler_cells_" + folder, os.path.join(out, P.RULER_CELLS))
    ecfg2 = dict(ecfg2, rel=dict(machine="i1", stages=["w6_eval", "w18_readout"]),
                 rel_w18_readout=dict(tables=dict(run=run_dir, path=path), reps=20, compare=[ev]))
    with Stage(ev2, "rel_w18_readout") as st:
        rep2 = w18_readout.build(st, ecfg2, ev2)
    cells = {(r["training"], r["evaluation"]) for r in rep2["two_by_two"]}
    assert cells == {(w18_readout.TRAIN_ROWS[0], w18_readout.EVAL_COLS[0]), (w18_readout.TRAIN_ROWS[0], w18_readout.EVAL_COLS[1])}
    assert {(c["training"], c["evaluation"]) for c in rep2["not_read"]} == {(w18_readout.TRAIN_ROWS[1], c) for c in w18_readout.EVAL_COLS}
    assert "Training layout x evaluation layout" in open(os.path.join(ev2, w18_readout.OUT, "report.md")).read()


def test_rulers_with_self_copies_run_the_donor_search(v1stage, tmp_path_factory, monkeypatch):
    """Protocol w18-v3: with self_copies the stage searches every implanted D:j against A's own proteins (W12's DIAMOND,
    here its stand-in) before drawing the self-copy rows and reading rule 9; the pairs reach held-out A only."""
    import stat
    import sys
    import test_rel_w12_search as T12
    tmp, base_cfg = v1stage["tmp"], v1stage["cfg"]
    patch(monkeypatch, tmp)
    fake = tmp / "diamond2"
    fake.write_text(T12.FAKE % dict(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    run_dir, cfg = RF.rel_run(tmp_path_factory.mktemp("w18self"), run_id="20261124T000004-exp-rel-00000000-rulers")
    cfg = dict(cfg, tools=dict(diamond=str(fake)), rel=dict(machine="i1", stages=["w18_rulers"]),
               rel_w18_rulers=dict(base_cfg["rel_w18_rulers"], self_copies=2))
    monkeypatch.setattr(w18_rulers, "declared_tables", lambda path: {m["id"]: 11 for m in v1stage["metas"]})
    with Stage(run_dir, "rel_w18_rulers") as st:
        doc = w18_rulers.build(st, cfg, run_dir)
    assert doc["name"].startswith("w18-implant-v1") and "w18-implant-3" in doc["table_version"]
    with open(os.path.join(run_dir, w18_rulers.OUT, "index.json")) as f:
        index = json.load(f)
    rec = index["donor_search"]
    assert rec["queries"] > 0 and rec["queries"] == index["counts"]["donor_queries"] and index["rule9_bit1"].startswith("read")
    assert not os.path.exists(os.path.join(run_dir, w18_rulers.OUT, "donor_search.tmp", "donor_queries.faa"))
    with Stage(run_dir, "gates_rel_w18_rulers") as st:
        w18_rulers.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w18_rulers")["status"] == "pass"
    meta = P.load_meta(run_dir, doc)
    assert "self_rows" in meta and (meta["self_rows"] <= 2).all()
    off = dict(cfg, rel_w18_rulers=dict(cfg["rel_w18_rulers"], self_copies=0))
    assert w18_rulers.settings(off)["donor_search"] is False
    with pytest.raises(SystemExit, match="version 1"):
        w18_rulers.settings(dict(cfg, rel_w18_rulers=dict(source="x", dist="y", donor_search=True)))
