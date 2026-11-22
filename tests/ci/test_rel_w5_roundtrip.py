"""The tables package W2 builds, read back by the evaluator (the round trip of the day's protocol, addendum 3, item 5):
stage rel_w2_v0 builds a small real-kind table set from a synthetic source run (the fixture of test_rel_w2_stage.py);
evalh/shards.py reads it through its manifest exactly as W2's own reader does, evalh/layout.py places its rows where
W2's stored grid places them, and stage rel_w5_baselines runs on that tables run and passes its gate."""
import json
import os
import sys

import numpy as np
import pytest

import rel_fixtures as RF
import test_rel_w2_stage as W2S
from ci.hashing import sha256_file
from ci.stages.common import Stage
from ci.stages.rel import common, w2_v0, w5_baselines
from seeddrift.train.rel import api, handoff
from seeddrift.train.rel import sealed as S
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import layout, shards
from seeddrift.train.rel.tables import v0_io as IO

MANIFEST = "data/rel_w2/v0/tableset.json"


def _sealed(cfg, spec=None):
    """The sealed sources of the fixture, with a machine registry named (what a manifest's sealed record must show)."""
    return S.SealedSet.from_parts(W2S.SEALED, (), ("NC_000913.3",), dict(kind="pinned", registry=dict(path="host/sealed_registry.json")))


@pytest.fixture(scope="module")
def tables_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w5rt")
    src = str(tmp / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    W2S.make_source(src)
    run_dir, cfg = RF.rel_run(tmp, stages=("w2_v0",), run_id="20261122T000000-exp-rel-00000000-w2tables",
                              rel_w2_v0=dict(source=src, expect=W2S.SIZES, rows=16, train_per_group=6, heldout_per_group=2, shard_tables=24, processes=1))
    with pytest.MonkeyPatch.context() as mp:
        W2S.patch(mp, tmp)
        mp.setattr(common, "sealed_set", _sealed)
        with Stage(run_dir, "rel_w2_v0") as st:
            doc = w2_v0.build(st, cfg, run_dir)
    with open(os.path.join(run_dir, "record.json"), "w") as f:               # the run passed and its record lists the manifest
        json.dump(dict(run_id=os.path.basename(run_dir), status="passed", pipeline="rel_tables",
                       stages=[dict(name="rel_w2_v0", status="passed", outputs=[dict(name="tableset", path=MANIFEST,
                                                                                   sha256=sha256_file(os.path.join(run_dir, MANIFEST), use_sidecar=False))])]), f)
    return dict(tmp=tmp, run_dir=run_dir, doc=doc)


def test_the_reader_reads_what_the_builder_wrote(tables_run):
    run_dir, doc = tables_run["run_dir"], tables_run["doc"]
    assert handoff.from_run(run_dir, MANIFEST, doc["sha256"])["sha256"] == doc["sha256"]
    sets = shards.by_split(run_dir, [doc])
    parts, _ = IO.open_parts(run_dir)
    for split, name in ((0, "train"), (1, "heldout")):
        mine, theirs = sets[split], parts[name]
        assert mine.folder == doc["parts"][name]["folder"] and mine.tables == theirs.tables == doc["parts"][name]["tables"]
        fields = sorted(api.SHARD_FIELDS) + sorted(api.META_FIELDS)
        got, want = mine.load(0, mine.tables, fields), theirs.take(np.arange(theirs.tables), fields)
        for f in fields:
            assert got[f].dtype == want[f].dtype and np.array_equal(got[f], want[f], equal_nan=f == "identity"), (name, f)
        assert len(mine.verified) == len(mine.shards) * len(fields)              # every file read was checked against the manifest
    assert shards.group_names(run_dir, doc) == json.load(open(os.path.join(run_dir, doc["index"])))["groups"] and len(shards.group_names(run_dir, doc)) == 12


def test_place_is_the_builders_stored_grid(tables_run):
    held = shards.by_split(tables_run["run_dir"], [tables_run["doc"]])[1]
    t = held.load(0, held.tables, ("tok", "n", "col", "kind"))
    mine = layout.place(t["tok"], t["n"], t["col"], api.GAP)
    theirs = IO.stored_grid(t["tok"], t["col"], api.GAP)
    assert np.array_equal(mine, theirs)
    present = cond.present_rows(t["kind"])
    assert present.sum() > held.tables and not (t["n"][~present] > 0).any()        # rows that do not exist hold nothing
    m = cond.materialise(t, np.arange(held.tables), cond.SCATTERED)
    view, cells, truth = layout.masked_view(m)
    assert truth.shape[0] >= held.tables and (truth >= api.TARGET_MIN).all() and (truth <= api.TARGET_MAX).all()


def test_baselines_stage_on_the_tables_run(tables_run, monkeypatch):
    src, doc, tmp = tables_run["run_dir"], tables_run["doc"], tables_run["tmp"]
    block = dict(tables_from=src, tablesets=[dict(path=MANIFEST, sha256=doc["sha256"])], fit_tables=72, chain_tables=24)
    run_dir, cfg = RF.rel_run(tmp / "b", stages=["w5_baselines"], rel_w5_baselines=block)
    W2S.patch(monkeypatch, tmp)
    monkeypatch.setattr(common, "sealed_set", _sealed)
    monkeypatch.setattr(common, "check_machine", lambda cfg, gpus=None: dict(machine="i1", note="test"))
    monkeypatch.setattr(sys, "argv", ["w5_baselines", run_dir])
    w5_baselines.main()
    res = json.load(open(os.path.join(run_dir, w5_baselines.OUT, "results.json")))
    assert res["heldout"]["tables"] == 24 and res["heldout"]["kind"] == api.TABLE_REAL and res["n_groups"] == 12
    assert res["sealed"]["ids"]["checked"] == len(doc["genomes"]) and res["groups"] == shards.group_names(src, doc)
    with Stage(run_dir, "gates_rel_w5_baselines", requires=("pre",)) as st:
        w5_baselines.gate(st, cfg, run_dir)
    assert json.load(open(os.path.join(run_dir, "gates", "G4_rel_w5_baselines.json")))["status"] == "pass"
