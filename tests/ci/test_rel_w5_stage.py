"""Stage rel_w5_baselines end to end on a tiny tables run made here: the stage, its gate and its summary; and what
it refuses (a tables run that has not passed, a manifest that is not the pinned one, a table set of a kind the
profile does not accept, a held-out set that holds training tables)."""
import json
import os
import sys

import numpy as np
import pytest

import rel_fixtures as RF
from test_rel_w5_baselines import manifest, toy_tables, write_set
from ci.gates import GateFailure
from ci.hashing import sha256_file
from ci.stages.common import Stage
from ci.stages.rel import common, kinds, w5_baselines
from seeddrift.train.rel import api, handoff
from seeddrift.train.rel.evalh import conditions as cond

TABLESET = "data/rel_w2/v0/tableset.json"


def tables_run(tmp_path, status="passed", kind=api.TABLE_SYNTHETIC, heldout_split=1, run_id="20261122T010000-exp-rel-00000000-tables"):
    """A run directory that looks like a passed tables run: shards of a training and a held-out set, the manifest,
    and a record that lists the manifest."""
    run_dir = tmp_path / "ci-root" / "runs" / run_id
    os.makedirs(run_dir / "data" / "rel_w2" / "v0")
    train, tmeta = toy_tables(n_tables=36, rows=5, seed=41)
    held, hmeta = toy_tables(n_tables=12, rows=5, seed=42)
    files = write_set(run_dir, "data/rel_w2/v0/train", train, tmeta, 0, per_shard=20) + write_set(run_dir, "data/rel_w2/v0/heldout", held, hmeta, heldout_split, per_shard=20)
    doc = manifest(files, name="toy-v0", kind=kind, tables=48)
    doc["producer"] = dict(stage="w2_v0", package="W2", run_id=run_id, commit="0" * 40)
    doc = handoff.finish(doc)
    with open(run_dir / TABLESET, "w") as f:
        json.dump(doc, f)
    rec = dict(run_id=run_id, status=status, pipeline="rel_tables",
               stages=[dict(name="rel_w2_v0", status="passed", outputs=[dict(name="tableset", path=TABLESET, sha256=sha256_file(str(run_dir / TABLESET), use_sidecar=False))])])
    with open(run_dir / "record.json", "w") as f:
        json.dump(rec, f)
    return str(run_dir), doc


def baselines_run(tmp_path, monkeypatch, src, doc, **over):
    block = dict(tables_from=src, tablesets=[dict(path=TABLESET, sha256=doc["sha256"])], fit_tables=30, chain_tables=12, allow_kinds=[api.TABLE_SYNTHETIC])
    block.update(over)
    run_dir, cfg = RF.rel_run(tmp_path, stages=["w5_baselines"], rel_w5_baselines=block)
    monkeypatch.setattr(common, "check_machine", lambda cfg, gpus=None: dict(machine="i1", note="test"))
    monkeypatch.setattr(sys, "argv", ["w5_baselines", run_dir])
    return run_dir, cfg


def test_the_stage_is_registered_as_written():
    d = kinds.stage("w5_baselines")
    assert d["kind"] == "rel_tables" and kinds.load()["kinds"]["rel_tables"]["lock"] == "select" and not d.get("stub")
    assert d["module"] == w5_baselines.__name__ and d["rule"] == w5_baselines.STAGE and d["gates"] == ["G4_rel_w5_baselines"]
    assert os.path.normpath(d["output"]) == os.path.join(w5_baselines.OUT, "results.json")


def test_stage_gate_and_summary_on_a_tiny_tables_run(tmp_path, monkeypatch):
    src, doc = tables_run(tmp_path)
    run_dir, cfg = baselines_run(tmp_path, monkeypatch, src, doc)
    w5_baselines.main()
    out = os.path.join(run_dir, w5_baselines.OUT)
    res = json.load(open(os.path.join(out, "results.json")))
    assert res["conditions"] == list(cond.CONDITIONS) and res["heldout"]["tables"] == 12 and res["fit"]["tables"] == 30
    assert res["tables_run"]["tablesets"][0]["sha256"] == doc["sha256"] and res["selfcheck"] == cond.SELFCHECK
    bits = res["headline"]["bits"]
    assert abs(bits[cond.SCATTERED]["uniform"] - np.log2(65)) < 1e-6
    assert bits[cond.SCATTERED]["copy_nearest"] < bits[cond.SCATTERED]["own_freq"] - 1 and "copy_nearest" not in bits[cond.SCATTERED + "+" + cond.NO_RELATIVES]
    assert res["items"][cond.SCATTERED]["cells"] == res["items"][cond.SCATTERED + "+" + cond.NO_RELATIVES]["cells"] > 0
    rep = json.load(open(os.path.join(run_dir, "stages", "rel_w5_baselines.json")))
    assert rep["status"] == "passed" and {o["name"] for o in rep["outputs"]} == {"rows", "params", "report", "results"}
    with Stage(run_dir, "gates_rel_w5_baselines", requires=("pre",)) as st:
        key = w5_baselines.gate(st, cfg, run_dir)
    gate = json.load(open(os.path.join(run_dir, "gates", "G4_rel_w5_baselines.json")))
    assert gate["status"] == "pass" and len(key) == 64 and gate["evidence"]["problems"] == []
    s = w5_baselines.summary(run_dir)
    assert s["json"]["headline"]["bits"] == bits and any("copy_nearest" in line for line in s["md"]) and len(json.dumps(s["json"])) < 20000
    text = open(os.path.join(out, "report.md")).read()
    assert "upper bound" in text and "/" + "Users" + "/" not in text
    # a broken result fails the gate: a row without cells, then a missing condition
    rows = open(os.path.join(out, "rows.tsv")).read()
    with open(os.path.join(out, "rows.tsv"), "w") as f:
        f.write(rows.replace("\tok\t", "\tfailed\t", 1))
    with pytest.raises(GateFailure):
        with Stage(run_dir, "gates_rel_w5_baselines", requires=("pre",)) as st:
            w5_baselines.gate(st, cfg, run_dir)
    assert json.load(open(os.path.join(run_dir, "gates", "G4_rel_w5_baselines.json")))["status"] == "fail"


def test_stage_refuses_wrong_inputs_fails(tmp_path, monkeypatch):
    src, doc = tables_run(tmp_path)
    for over in (dict(tablesets=[dict(path=TABLESET, sha256="0" * 64)]),            # not the pinned manifest
                 dict(allow_kinds=[api.TABLE_REAL]),                                 # a synthetic set where real tables are asked for
                 dict(tables_from=os.path.join(os.path.dirname(src), "nothing"))):   # no such run
        run_dir, _ = baselines_run(tmp_path / str(len(over)) / next(iter(over)), monkeypatch, src, doc, **over)
        with pytest.raises(SystemExit):
            w5_baselines.main()
        assert json.load(open(os.path.join(run_dir, "stages", "rel_w5_baselines.json")))["status"] == "failed"
    src2, doc2 = tables_run(tmp_path / "notpassed", status="failed")
    run_dir, _ = baselines_run(tmp_path / "np", monkeypatch, src2, doc2)
    with pytest.raises(SystemExit):
        w5_baselines.main()
    src3, doc3 = tables_run(tmp_path / "leak", heldout_split=0)                      # two training sets, no held-out set
    run_dir, _ = baselines_run(tmp_path / "lk", monkeypatch, src3, doc3)
    with pytest.raises(SystemExit):
        w5_baselines.main()
