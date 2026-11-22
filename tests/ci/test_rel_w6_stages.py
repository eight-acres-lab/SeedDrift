"""The stages of package W6 in the CI environment (no torch): the variant files, the profile block of the training
stage, the model index and gate of a finished training, the report of curves beside baselines, the result rows."""
import json
import os

import numpy as np
import pytest
import yaml

import rel_fixtures as RF
from conftest import ROOT
from ci.stages.common import Stage
from ci.stages.rel import kinds, w6_fit, w6_report, w6_selftest
from seeddrift.train.rel import api, handoff, registry
from seeddrift.train.rel.data import heldout_plan as H
from seeddrift.train.rel.trainer import rows as W

VARIANTS = ("w6-ref10m", "w6-ref30m", "w6-ref30m-norel", "w6-ref100m")
STAGES = ["w6_selftest", "w6_fit", "w6_report"]


def _gate(run_dir, gate):
    with open(os.path.join(run_dir, "gates", gate + ".json")) as f:
        return json.load(f)


def test_stages_are_registered_as_written():
    for s, side in (("w6_selftest", "train"), ("w6_fit", "train"), ("w6_report", "eval")):
        d = kinds.stage(s)
        assert d["kind"] == "rel_traineval" and d["module"] == "ci.stages.rel." + s and d["side"] == side and not d.get("stub")
    assert kinds.stage("w6_fit")["output"] == os.path.join(w6_fit.OUT, "models.json") and kinds.stage("w6_fit").get("produces") == "models"
    assert kinds.stage("w6_report")["output"] == os.path.join(w6_report.OUT, "report.json")
    assert kinds.stage("w6_selftest")["output"] == os.path.join(w6_selftest.OUT, "report.json")


def test_variant_files_follow_the_contract():
    st = Stage.__new__(Stage)
    for name in VARIANTS:
        v = w6_fit.load_variant(st, "configs/rel/variants/%s.yaml" % name)
        assert v["name"] == name and v["author"] == "w6" and v["api"] == 1 and v["layout"] == api.LAYOUT_STORED and v["columns"] == api.ROW_LEN
        assert v["model"] in registry.available("model") and v["objective"] in registry.available("objective")
        assert all(m in registry.available("mask") for m in v["masks"]) and v["rows"] == [4, 16] and v["tables"] == api.TABLE_REAL
    sizes = {n: yaml.safe_load(open(os.path.join(ROOT, "configs", "rel", "variants", n + ".yaml")))["size"] for n in VARIANTS}
    assert sizes == {"w6-ref10m": "10m", "w6-ref30m": "30m", "w6-ref30m-norel": "30m", "w6-ref100m": "100m"}


def test_a_changed_variant_file_fails(tmp_path):
    st = Stage.__new__(Stage)
    v = yaml.safe_load(open(os.path.join(ROOT, "configs", "rel", "variants", "w6-ref30m.yaml")))
    for change in (dict(api=2), dict(optimizer=dict(lr=1e-3, betas=[0.9, 0.9])), dict(name="w6-other"), dict(extra=1)):
        p = tmp_path / "w6-ref30m.yaml"
        p.write_text(yaml.safe_dump(dict(v, **change)))
        with pytest.raises(SystemExit):
            w6_fit.load_variant(st, str(p))


def _block(**over):
    block = dict(tables=dict(train=dict(run="/x/runs/a", path="data/rel_w2/v0/tableset.json", split=0),
                             heldout=dict(run="/x/runs/a", path="data/rel_w2/v0/tableset.json", split=1)),
                 arms=[dict(variant="configs/rel/variants/w6-ref30m.yaml", seed=1), dict(variant="configs/rel/variants/w6-ref10m.yaml", seed=1, budget_cells=1000)])
    block.update(over)
    return block


def test_profile_block():
    st = Stage.__new__(Stage)
    s = w6_fit.settings(st, dict(rel_w6_fit=_block(wall_s=100, eval_per_group=2)))
    assert [a["name"] for a in s["arms"]] == ["w6-ref30m-s1", "w6-ref10m-s1"] and s["arms"][1]["budget_cells"] == 1000 and s["arms"][0]["budget_cells"] is None
    assert s["wall_s"] == 100 and s["passed_on"] == dict(eval_per_group=2) and s["probe"] is False and s["tables"]["heldout"]["split"] == 1
    conf = w6_fit.trainer_conf(dict(train=dict(device="cpu"), git=dict(sha="c" * 40)), s, s["arms"][0], None, "/out/x", "/x/runs/20261122T000000-r")
    assert conf["train_tables"] == "/x/runs/a/data/rel_w2/v0/tableset.json" and conf["train_root"] == "/x/runs/a" and conf["heldout_split"] == 1
    assert conf["device"] == "cpu" and conf["lr"] == 3e-4 and conf["relatives"] is True and conf["run"] == "20261122T000000-r" and conf["eval_per_group"] == 2


@pytest.mark.parametrize("change", [dict(arms=[]), dict(nonsense=1), dict(tables=dict(train=dict(run="/x", path="p"))),
                                    dict(arms=[dict(variant="configs/rel/variants/w6-ref30m.yaml", seed=1)] * 2),
                                    dict(arms=[dict(variant="configs/rel/variants/w6-ref30m.yaml")]),
                                    dict(tables=dict(train=dict(run="/x", path="p", split=2), heldout=dict(run="/x", path="p")))])
def test_bad_profile_blocks_fail(change):
    with pytest.raises(SystemExit):
        w6_fit.settings(Stage.__new__(Stage), dict(rel_w6_fit=_block(**change)))


def _rows(run, arm, points, final, conditions, params=100):
    """Result rows of an arm: `points` [(cells_seen, bits)] scored on the subset, then the final weights on all items."""
    out = []
    for weights, half, cells, bits in [("raw", "subset", c, b) for c, b in points] + [("final", "all", points[-1][0], final)]:
        fixed = dict(run=run, commit="c", variant=arm, author="w6", module="axial", params=params, seed=1, table_set="t", table_kind="real",
                     items_sha256="0" * 64, item_half=half, layout="stored", budget="cells=none;wall_s=10", weights=weights, cells_seen=cells,
                     target_cells=cells // 8, cells_per_s="10.0", peak_mb="5", status="ok", versions="v")
        for k, cond in enumerate(conditions):
            agg = [("bacteria", "all", bits + k, 3), ("fungi", "all", bits + k + 0.5, 3), ("all", "all", bits + k + 0.25, 6),
                   ("all", "<0.3", bits + k + 1, 4), ("all", ">=0.9", bits + k - 1, 2)]
            out += W.result_rows(fixed, cond, agg)
    return out


def _finished_fit(tmp_path, statuses=("ok", "failed")):
    """A run directory in which rel_w6_fit has just trained (or failed) its arms; returns run_dir, cfg, the model index."""
    run_dir, cfg = RF.rel_run(tmp_path, kind="rel_traineval", stages=STAGES)
    st = Stage(run_dir, "rel_w6_fit")
    arms = {}
    for name, status in zip(("w6-ref30m-s1", "w6-ref30m-norel-s1"), statuses):
        folder = os.path.join(run_dir, w6_fit.OUT, name)
        os.makedirs(folder)
        conds = list(H.CONDITIONS) if "norel" not in name else [c for c in H.CONDITIONS if "no_relatives" in c]
        summ = dict(status="ok", steps=10, cells_seen=4000, target_cells=500, cells_per_s=12.5, peak_mb=10, reserved_mb=12, ended_by="wall",
                    conditions=conds, final={c: 5.0 for c in conds}, best=dict(step=5, cells_seen=2000, bits=5.1, condition=conds[0]), versions="v")
        if status == "ok":
            W.write_rows(os.path.join(folder, "results.tsv"), _rows(os.path.basename(run_dir), name, [(0, 6.0), (2000, 5.5), (4000, 5.2)], 5.0, conds), header=True)
            for which in ("final", "best"):
                with open(os.path.join(folder, which + ".pt"), "wb") as f:
                    f.write(which.encode())
            with open(os.path.join(folder, "summary.json"), "w") as f:
                json.dump(summ, f)
        variant = yaml.safe_load(open(os.path.join(ROOT, "configs", "rel", "variants", name[:-3] + ".yaml")))
        arms[name] = dict(arm=dict(name=name, file="configs/rel/variants/%s.yaml" % name[:-3], sha256="1" * 64, seed=1, budget_cells=None, variant=variant),
                          folder=folder, preflight=dict(ok=True, params=28_000_000, micro=4, peak_mb=9, reserved_mb=11, cells_per_s=20.0, flops_per_cell=1),
                          status=status, summary=summ if status == "ok" else {}, error=None if status == "ok" else "training (exit 1): out of memory")
    s = dict(tables=dict(train=dict(run="/x/runs/tables", split=0), heldout=dict(run="/x/runs/tables", split=1)), probe=False, wall_s=10)
    sets = {k: dict(doc=dict(name="v0", sha256="2" * 64)) for k in ("train", "heldout")}
    doc = w6_fit.finish(st, cfg, run_dir, s, sets, arms, [dict(arm="w6-ref30m-s1", beside=0, started_s=1)], 60000.0, dict(python="py"))
    return run_dir, cfg, doc, st


def test_model_index_and_gate_of_a_finished_training(tmp_path):
    run_dir, cfg, doc, st = _finished_fit(tmp_path)
    assert handoff.models_problems(doc, run_dir) == [] and [v["status"] for v in doc["variants"]] == ["ok", "failed"]
    assert [c["which"] for c in doc["variants"][0]["checkpoints"]] == ["final", "best"] and doc["variants"][1]["checkpoints"] == []
    assert "out of memory" in doc["variants"][1]["error"] and doc["tablesets"][0]["run_id"] == "tables"
    assert {o["name"] for o in st.outputs} >= {"models", "results", "summary_w6-ref30m-s1"}
    w6_fit.gate(st, cfg, run_dir)                                       # one arm trained, one failed: recorded, not a failure
    g = _gate(run_dir, "G4_rel_w6_fit")
    assert g["status"] == "pass" and g["evidence"]["trained"] == ["w6-ref30m-s1"] and "w6-ref30m-norel-s1" in g["evidence"]["failed"]
    part = w6_fit.summary(run_dir)
    assert part["json"]["arms"]["w6-ref30m-s1"]["cells_per_s"] == 12.5 and any("failed" in line for line in part["md"])
    assert len(json.dumps(part["json"])) < 20000


def test_training_gate_fails_without_a_trained_arm_or_its_final_rows(tmp_path):
    from ci.gates import GateFailure
    run_dir, cfg, doc, st = _finished_fit(tmp_path / "a", statuses=("failed", "failed"))
    with pytest.raises(GateFailure):
        w6_fit.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w6_fit")["status"] == "fail"
    run_dir, cfg, doc, st = _finished_fit(tmp_path / "b", statuses=("ok", "ok"))
    p = os.path.join(run_dir, w6_fit.OUT, "results.tsv")
    lines = [line for line in open(p).read().splitlines() if "\tfinal\t" not in line or "w6-ref30m-norel-s1" not in line]
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    with pytest.raises(GateFailure):
        w6_fit.gate(st, cfg, run_dir)
    assert "no final row" in json.dumps(_gate(run_dir, "G4_rel_w6_fit")["evidence"])


def test_report_sets_curves_beside_baselines(tmp_path):
    run_dir, cfg, doc, _ = _finished_fit(tmp_path, statuses=("ok", "ok"))
    base_run = os.path.join(os.path.dirname(run_dir), "20261122T000001-baselines")
    os.makedirs(os.path.join(base_run, "data", "rel_w5"))
    with open(os.path.join(base_run, "record.json"), "w") as f:
        json.dump(dict(run_id="b", status="passed"), f)
    brows = []
    for name, bits in (("copy_nearest", 5.4), ("own_freq", 5.9)):
        for r in _rows("b", name, [(0, bits)], bits, list(H.CONDITIONS), params=3):
            if r[api.RESULT_ROW.index("weights")] == "final":
                brows.append(r)
    W.write_rows(os.path.join(base_run, "data", "rel_w5", "results.tsv"), brows, header=True)
    cfg = dict(cfg, rel_w6_report=dict(baselines=[dict(run=base_run, path="data/rel_w5/results.tsv")]))
    st = Stage(run_dir, "rel_w6_report")
    rep = w6_report.build(st, cfg, run_dir)
    assert rep["by_condition"]["columns"] == ["condition", "w6-ref30m-norel-s1", "w6-ref30m-s1", "copy_nearest", "own_freq"]
    row = {r[0]: r[1:] for r in rep["by_condition"]["rows"]}
    assert row["scattered"] == [None, 5.25, 5.65, 6.15] and row["scattered+no_relatives"][0] == 5.25
    assert rep["aligned"]["columns"] == ["cells_seen", "w6-ref30m-norel-s1", "w6-ref30m-s1"]
    assert [r[0] for r in rep["aligned"]["rows"]] == [0, 2000, 4000] and rep["aligned"]["rows"][1][2] == 5.75
    assert rep["arms"]["w6-ref30m-s1"]["evaluations"] == 4 and rep["baselines"]["own_freq"]["whole_gene"] == 7.15
    assert any(r[:2] == ["scattered", ">=0.9"] for r in rep["by_stratum"]["rows"]) and len(rep["caveats"]) >= 4
    text = open(os.path.join(run_dir, w6_report.OUT, "report.md")).read()
    assert "upper bound" in text and "copy_nearest" in text and "cells/s" in text
    w6_report.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w6_report")["status"] == "pass"
    part = w6_report.summary(run_dir)
    assert len(json.dumps(part["json"])) < 20000 and part["json"]["baselines"]["copy_nearest"]["scattered"] == 5.65


def test_report_gate_fails_on_a_curve_of_one_point(tmp_path):
    from ci.gates import GateFailure
    run_dir, cfg, doc, _ = _finished_fit(tmp_path, statuses=("ok", "ok"))
    p = os.path.join(run_dir, w6_fit.OUT, "results.tsv")
    lines = [line for line in open(p).read().splitlines() if "\traw\t" not in line or "w6-ref30m-s1" not in line]
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    st = Stage(run_dir, "rel_w6_report")
    w6_report.build(st, cfg, run_dir)
    with pytest.raises(GateFailure):
        w6_report.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w6_report")["status"] == "fail"


def test_selftest_verdict():
    good = dict(exit=0, gpu_hidden=True, modules={"m": dict(tests=3, passed=3, failed={})})
    assert w6_selftest.problems(good, ["m"]) == []
    assert w6_selftest.problems(dict(good, gpu_hidden=False), ["m"]) != []
    assert w6_selftest.problems(good, ["m", "other"]) != []
    assert w6_selftest.problems(dict(good, modules={"m": dict(tests=3, passed=2, failed={"test_x": "boom"})}), ["m"]) != []
    assert w6_selftest.problems(dict(good, exit=1), ["m"]) != [] and w6_selftest.problems(dict(good, modules={"m": dict(tests=0, passed=0, failed={})}), ["m"]) != []
    with pytest.raises(SystemExit):
        w6_selftest.settings(dict(rel_w6_selftest=dict(modules=["/abs/x.py"])))


def test_result_rows_round_trip_and_aggregate(tmp_path):
    bits, cells = np.array([10.0, 20.0, 0.0, 6.0]), np.array([2, 4, 0, 3])
    agg = W.aggregate(bits, cells, np.array([0, 0, 1, 1]), np.array(["none", "<0.3", "none", "<0.3"]), {0: "bacteria"})
    assert ("bacteria", "all", 5.0, 2) in agg and ("1", "all", 2.0, 1) in agg and ("all", "all", 3.5, 3) in agg and ("all", "none", 5.0, 1) in agg
    fixed = {c: "x" for c in api.RESULT_ROW if c not in ("condition", "group", "tier", "stratum", "passes", "bits", "baseline", "bits_minus_baseline", "n_items")}
    p = str(tmp_path / "r.tsv")
    W.write_rows(p, W.result_rows(fixed, "scattered", agg), header=True)
    back = W.read_rows(p)
    assert len(back) == len(agg) and back[0]["condition"] == "scattered" and back[0]["passes"] == "1" and float(back[2]["bits"]) == 3.5
    with pytest.raises(KeyError):
        W.result_rows(dict(run="r"), "scattered", agg)
    with pytest.raises(ValueError):
        W.write_rows(p, [["a\tb"] * len(api.RESULT_ROW)])
    with open(p, "a") as f:
        f.write("too\tfew\n")
    with pytest.raises(ValueError):
        W.read_rows(p)
