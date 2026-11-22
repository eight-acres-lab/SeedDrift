"""The stages of package W6 in the CI environment (no torch): the variant files, the profile block of the training
stage, the model index and gate of a finished training, the report of the curves on the axis of tables seen beside
the classical baselines, and the self-test verdict."""
import json
import os

import pytest
import yaml

import rel_fixtures as RF
from conftest import ROOT
from ci.stages.common import Stage
from ci.stages.rel import kinds, w6_fit, w6_report, w6_selftest
from seeddrift.train.rel import api, handoff, registry
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import score as SC

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
    lr = {}
    for name in VARIANTS:
        v = w6_fit.load_variant(st, "configs/rel/variants/%s.yaml" % name)
        assert v["name"] == name and v["author"] == "w6" and v["api"] == 1 and v["layout"] == api.LAYOUT_STORED and v["columns"] == api.ROW_LEN
        assert v["model"] in registry.available("model") and v["objective"] in registry.available("objective")
        assert all(m in registry.available("mask") for m in v["masks"]) and v["rows"] == [4, 16] and v["tables"] == api.TABLE_REAL
        lr[name] = v["optimizer"]["lr"]
    sizes = {n: yaml.safe_load(open(os.path.join(ROOT, "configs", "rel", "variants", n + ".yaml")))["size"] for n in VARIANTS}
    assert sizes == {"w6-ref10m": "10m", "w6-ref30m": "30m", "w6-ref30m-norel": "30m", "w6-ref100m": "100m"}
    assert lr == {"w6-ref10m": 4e-4, "w6-ref30m": 3e-4, "w6-ref30m-norel": 3e-4, "w6-ref100m": 2e-4}


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
                 arms=[dict(variant="configs/rel/variants/w6-ref30m.yaml", seed=1), dict(variant="configs/rel/variants/w6-ref10m.yaml", seed=1, lr=1e-4)])
    block.update(over)
    return block


def test_profile_block():
    st = Stage.__new__(Stage)
    s = w6_fit.settings(st, dict(rel_w6_fit=_block(wall_s=100, eval_per_group=25, warmup_steps=500)))
    assert [a["name"] for a in s["arms"]] == ["w6-ref30m-s1", "w6-ref10m-s1"] and [a["lr"] for a in s["arms"]] == [3e-4, 1e-4]
    assert s["wall_s"] == 100 and s["passed_on"] == dict(eval_per_group=25, warmup_steps=500) and s["probe"] is False and s["tables"]["heldout"]["split"] == 1
    sets = dict(heldout=dict(row_groups={0: "bacteria"}))
    conf = w6_fit.trainer_conf(dict(train=dict(device="cpu"), git=dict(sha="c" * 40)), s, s["arms"][0], sets, "/out/x", "/x/runs/20261122T000000-r")
    assert conf["train_tables"] == "/x/runs/a/data/rel_w2/v0/tableset.json" and conf["train_root"] == "/x/runs/a" and conf["heldout_split"] == 1
    assert conf["device"] == "cpu" and conf["lr"] == 3e-4 and conf["relatives"] is True and conf["run"] == "20261122T000000-r" and conf["eval_per_group"] == 25
    assert conf["group_names"] == {"0": "bacteria"} and conf["warmup_steps"] == 500


@pytest.mark.parametrize("change", [dict(arms=[]), dict(nonsense=1), dict(tables=dict(train=dict(run="/x", path="p"))),
                                    dict(arms=[dict(variant="configs/rel/variants/w6-ref30m.yaml", seed=1)] * 2),
                                    dict(arms=[dict(variant="configs/rel/variants/w6-ref30m.yaml")]),
                                    dict(arms=[dict(variant="configs/rel/variants/w6-ref30m.yaml", seed=1, lr=1.0)]),
                                    dict(tables=dict(train=dict(run="/x", path="p", split=2), heldout=dict(run="/x", path="p")))])
def test_bad_profile_blocks_fail(change):
    with pytest.raises(SystemExit):
        w6_fit.settings(Stage.__new__(Stage), dict(rel_w6_fit=_block(**change)))


def test_table_sets_come_only_from_passed_runs_with_their_sealed_record(tmp_path):
    from ci.record import stage_report
    run_dir, cfg = RF.rel_run(tmp_path, kind="rel_tables", stages=["w2_v0"], run_id="20261122T000000-exp-rel-00000000-tables")
    doc = RF.write_tableset(run_dir, stage="w2_v0", name="v0", kind="real", genomes=[dict(id="g1", taxid=562, split="train")], package="W2")
    path = kinds.stage("w2_v0")["output"]
    stage_report(run_dir, "rel_w2_v0", "", 0.0, [dict(name="tableset", path=path, sha256=handoff._file_sha(os.path.join(run_dir, path)), bytes=1)])
    from ci.record import Record
    rec = Record.open(run_dir)
    rec.merge_stages()
    rec.update(status="passed")
    got = w6_fit.table_set(cfg, dict(run=run_dir, path=path, sha256=doc["sha256"], split=1), verify=True)
    assert got["doc"]["name"] == "v0" and got["sealed"] is True and got["files_verified"]
    with pytest.raises(SystemExit):                                     # another manifest than the one pinned
        w6_fit.table_set(cfg, dict(run=run_dir, path=path, sha256="f" * 64, split=1), verify=True)
    rec.update(status="failed")
    with pytest.raises(SystemExit):                                     # a run that did not pass
        w6_fit.table_set(cfg, dict(run=run_dir, path=path, sha256=None, split=1), verify=True)


GROUPS = ("bacteria", "eukaryota.fungi")


def _rows(run, arm, points, final, conditions, params=100, module="models.axial", reads="ema", best=None):
    """Result rows of an arm: `points` [(tables, cells, bits)] at the nodes (the subset of tables), then the final
    evaluation on all items, and optionally (best: (tables, bits)) the best node read on all items."""
    out = []
    nodes = [("%s@tables=%d" % (reads, t), "subset", c, b, 6, t) for t, c, b in points] + [("%s@final" % reads, "all", points[-1][1], final, 12, points[-1][0])]
    if best:
        nodes.append(("%s@best" % reads, "all", 0, best[1], 12, best[0]))
    for weights, half, cells, bits, items, tables in nodes:
        for k, c in enumerate(conditions):
            for group, stratum, x, n in ((GROUPS[0], SC.ALL, bits + k, items // 2), (GROUPS[1], SC.ALL, bits + k + 0.5, items // 2), (SC.ALL, SC.ALL, bits + k + 0.25, items),
                                          (SC.ALL, SC.FINE[2], bits + k + 1, 4), (SC.ALL, SC.FINE[9], bits + k - 1, 2), (SC.ALL, SC.FINE[-1], bits + k + 2, 1),
                                          (SC.ALL, SC.OTHER_ROWS[2], bits + k, 3), (SC.ALL, SC.OTHER_ROWS[0], bits + 0.125, 1),
                                          (GROUPS[0], SC.RELATIVES[0], bits + k + 0.0625, 4), (GROUPS[1], SC.RELATIVES[0], bits + k + 0.5625, 4)):
                out.append(SC.result_row(run=run, commit="c", variant=arm, author="w6", module=module, params=params, seed=1, table_set="t", table_kind="real",
                                         items_sha256="5" * 64, item_half=half, layout="stored", budget="tables=%d;wall_s=10" % tables, weights=weights,
                                         cells_seen=cells, target_cells=40 * n, cells_per_s=10.0, peak_mb=5, condition=c, group=group, tier=SC.TIER,
                                         stratum=stratum, passes=1, bits=x, n_items=n, status="ok", versions="v"))
    return out


ARMS = (("w6-ref30m-s1", 28_000_000, 5.0, [(0, 0, 6.0), (1000, 10000, 5.5), (2000, 20000, 5.2)], None),
        ("w6-ref30m-norel-s1", 28_000_000, 5.5, [(0, 0, 6.0), (1000, 10000, 5.6), (2000, 20000, 5.45)], (1000, 5.4)),
        ("w6-ref10m-s1", 8_400_000, 5.1, [(0, 0, 6.0), (1000, 10000, 5.6), (2000, 20000, 5.3)], None))


def _finished_fit(tmp_path, statuses=("ok", "failed", "ok")):
    """A run directory in which rel_w6_fit has just trained (or failed) its arms; returns run_dir, cfg, the model index."""
    run_dir, cfg = RF.rel_run(tmp_path, kind="rel_traineval", stages=STAGES)
    st = Stage(run_dir, "rel_w6_fit")
    arms = {}
    for (name, params, final, points, best), status in zip(ARMS, statuses):
        folder = os.path.join(run_dir, w6_fit.OUT, name)
        os.makedirs(folder)
        conds = list(cond.CONDITIONS) if "norel" not in name else [c for c in cond.CONDITIONS if "no_relatives" in c]
        summ = dict(status="ok", steps=10, tables_seen=4000, cells_seen=40000, target_cells=500, cells_per_s=12.5, tables_per_s=1.25, peak_mb=10, reserved_mb=12,
                    ended_by="wall", conditions=conds, final={c: 5.0 for c in conds}, final_items={c: "digest-" + c for c in conds}, eval_subset=600,
                    eval_final=3000, weights_read="ema", eval_overhead=0.05, versions="v",
                    best=dict(step=5, tables_seen=best[0] if best else 2000, bits=5.1, condition=conds[0]))
        if status == "ok":
            SC.write_rows(os.path.join(folder, "results.tsv"), _rows(os.path.basename(run_dir), name, points, final, conds, best=best))
            for which in ("final", "best"):
                with open(os.path.join(folder, which + ".pt"), "wb") as f:
                    f.write(which.encode())
            with open(os.path.join(folder, "summary.json"), "w") as f:
                json.dump(summ, f)
            with open(os.path.join(folder, "preflight.json"), "w") as f:
                json.dump(dict(ok=True, train_tables=100800), f)
        variant = yaml.safe_load(open(os.path.join(ROOT, "configs", "rel", "variants", name[:-3] + ".yaml")))
        arms[name] = dict(arm=dict(name=name, file="configs/rel/variants/%s.yaml" % name[:-3], sha256="1" * 64, seed=1, budget_cells=None, lr=3e-4, variant=variant),
                          folder=folder, preflight=dict(ok=True, params=params, micro=4, peak_mb=9, reserved_mb=11, cells_per_s=20.0, flops_per_cell=1,
                                                        lr=3e-4, lr_reduced=False, guard=dict(failed=False)),
                          status=status, summary=summ if status == "ok" else {}, error=None if status == "ok" else "training (exit 1): out of memory")
    s = dict(tables=dict(train=dict(run="/x/runs/tables", split=0), heldout=dict(run="/x/runs/tables", split=1)), probe=False, wall_s=10)
    sets = {k: dict(doc=dict(name="v0", sha256="2" * 64), groups=list(GROUPS), row_groups=dict(enumerate(GROUPS))) for k in ("train", "heldout")}
    doc = w6_fit.finish(st, cfg, run_dir, s, sets, arms, [dict(arm="w6-ref30m-s1", beside=0, started_s=1)], 60000.0, dict(python="py"))
    return run_dir, cfg, doc, st


def test_model_index_and_gate_of_a_finished_training(tmp_path):
    run_dir, cfg, doc, st = _finished_fit(tmp_path)
    assert handoff.models_problems(doc, run_dir) == [] and [v["status"] for v in doc["variants"]] == ["ok", "failed", "ok"]
    assert [c["which"] for c in doc["variants"][0]["checkpoints"]] == ["final", "best"] and doc["variants"][1]["checkpoints"] == []
    assert "out of memory" in doc["variants"][1]["error"] and doc["tablesets"][0]["run_id"] == "tables" and doc["groups"] == list(GROUPS)
    assert doc["variants"][0]["info"]["tables_seen"] == 4000 and doc["variants"][0]["info"]["final_items"]["scattered"] == "digest-scattered"
    assert {o["name"] for o in st.outputs} >= {"models", "results", "summary_w6-ref30m-s1"}
    w6_fit.gate(st, cfg, run_dir)                                       # one arm trained, one failed: recorded, not a failure
    g = _gate(run_dir, "G4_rel_w6_fit")
    assert g["status"] == "pass" and g["evidence"]["trained"] == ["w6-ref30m-s1", "w6-ref10m-s1"] and "w6-ref30m-norel-s1" in g["evidence"]["failed"]
    part = w6_fit.summary(run_dir)
    assert part["json"]["arms"]["w6-ref30m-s1"]["cells_per_s"] == 12.5 and part["json"]["arms"]["w6-ref30m-s1"]["tables_seen"] == 4000
    assert any("failed" in line for line in part["md"]) and len(json.dumps(part["json"])) < 20000


def test_training_gate_fails_without_a_trained_arm_or_its_final_rows(tmp_path):
    from ci.gates import GateFailure
    run_dir, cfg, doc, st = _finished_fit(tmp_path / "a", statuses=("failed", "failed", "failed"))
    with pytest.raises(GateFailure):
        w6_fit.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w6_fit")["status"] == "fail"
    run_dir, cfg, doc, st = _finished_fit(tmp_path / "b", statuses=("ok", "ok", "ok"))
    p = os.path.join(run_dir, w6_fit.OUT, "results.tsv")
    SC.write_rows(p, [r for r in SC.read_rows(p) if not (r["weights"] == "ema@final" and r["variant"] == "w6-ref30m-norel-s1")])
    with pytest.raises(GateFailure):
        w6_fit.gate(st, cfg, run_dir)
    assert "no final row" in json.dumps(_gate(run_dir, "G4_rel_w6_fit")["evidence"])


def _baselines(run_dir, digest=True):
    """A passed run beside run_dir with the rows of two classical baselines and their results.json (as package W5 writes them)."""
    base_run = os.path.join(os.path.dirname(run_dir), "20261122T000001-baselines")
    folder = os.path.join(base_run, "data", "rel_w5", "baselines")
    os.makedirs(folder)
    with open(os.path.join(base_run, "record.json"), "w") as f:
        json.dump(dict(run_id="b", status="passed"), f)
    rows = []
    for name, bits in (("copy_nearest", 5.4), ("own_freq", 5.9)):
        for r in _rows("b", name, [(0, 0, bits)], bits, list(cond.CONDITIONS), params=3, module="baselines.classical"):
            if r["weights"].endswith("@final"):
                rows.append(dict(r, weights="", budget=0))
    SC.write_rows(os.path.join(folder, "rows.tsv"), rows)
    with open(os.path.join(folder, "results.json"), "w") as f:
        json.dump(dict(items={c: dict(sha256="digest-" + c if digest else "other", cells=10, tables=12) for c in cond.CONDITIONS}), f)
    return dict(run=base_run, path="data/rel_w5/baselines/rows.tsv")


def test_report_readings_beside_baselines(tmp_path):
    run_dir, cfg, doc, _ = _finished_fit(tmp_path, statuses=("ok", "ok", "ok"))
    cfg = dict(cfg, rel_w6_report=dict(baselines=[_baselines(run_dir)]))
    st = Stage(run_dir, "rel_w6_report")
    rep = w6_report.build(st, cfg, run_dir)
    cols = ["w6-ref10m-s1", "w6-ref30m-norel-s1", "w6-ref30m-s1", "copy_nearest", "own_freq"]
    assert rep["headline"]["columns"] == ["condition", "stratum"] + cols
    assert all(r[1] != "none" for r in rep["headline"]["rows"])                                       # the bin without other rows is no headline
    assert ["scattered", "0.9-1.0", 4.1, None, 4.0, 4.4, 4.9] in rep["headline"]["rows"]
    blocks = {(r[0], r[1].split(" (")[0]): r[2:] for r in rep["blocks"]["rows"]}
    assert blocks[("scattered", "eukaryotes")] == [5.6, None, 5.5, 5.9, 6.4] and blocks[("scattered", "prokaryotes, plasmids, viruses")][2] == 5.0
    worth = {r["condition"]: r for r in rep["relatives"]}
    assert worth["scattered"]["without_at"].startswith("best node") and worth["scattered"]["bits_without"] == 5.65      # 5.4 at its best node
    assert abs(worth["scattered"]["worth_all_tables"] - 0.4) < 1e-9 and worth["scattered"]["node"] == 2000
    assert abs(worth["scattered"]["worth_at_node"] - 0.25) < 1e-9 and [p[0] for p in worth["scattered"]["per_node"]] == [1000, 2000]
    assert rep["scale"]["verdict"] == "bigger is better" and rep["scale"]["node"] == 2000 and rep["scale"]["arms"] == ["w6-ref10m-s1", "w6-ref30m-s1"]
    repair = {r[0]: r[1:] for r in rep["repair"]["rows"]}
    assert repair["w6-ref30m-s1"] == [6.0, 9.0, 3.0] and "copy_nearest" in repair
    assert all(abs(r[4]) < 1e-9 for r in rep["check"]["rows"])
    assert [r[0] for r in rep["by_group"]["rows"]] == list(GROUPS) and rep["by_group"]["rows"][1][1] == "eukaryotes"
    lead = {(r[0], r[1].split(" (")[0]): r[2:] for r in rep["lead"]["rows"]}
    assert lead[("scattered", "eukaryotes")] == [5.6625, None, 5.5625, 5.9625, 6.4625] and lead[("whole_gene", "prokaryotes, plasmids, viruses")][2] == 6.0625
    assert rep["aligned"]["columns"] == ["tables_seen", "w6-ref10m-s1 (scattered)", "w6-ref30m-norel-s1 (scattered+no_relatives)", "w6-ref30m-s1 (scattered)"]
    assert rep["aligned"]["rows"][1] == [1000, 5.85, 5.85, 5.75] and rep["same_items"] == {c: True for c in cond.CONDITIONS}
    text = open(os.path.join(run_dir, w6_report.OUT, "report.md")).read()
    assert "upper bound" not in text and "optimistic" in text and "bigger is better" in text and "passes over the training tables" in text
    assert "copy_nearest" in text and " NO" not in text
    curves = open(os.path.join(run_dir, w6_report.OUT, "curves.tsv")).read().splitlines()
    assert curves[0].split("\t")[:3] == ["arm", "point", "tables_seen"] and any(line.split("\t")[1:3] == ["final", "4000"] for line in curves)
    w6_report.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w6_report")["status"] == "pass"
    part = w6_report.summary(run_dir)
    assert len(json.dumps(part["json"])) < 20000 and part["json"]["baselines"]["copy_nearest"]["scattered"] == 5.65


def test_report_without_baselines_passes(tmp_path):
    run_dir, cfg, doc, _ = _finished_fit(tmp_path, statuses=("ok", "failed", "ok"))
    st = Stage(run_dir, "rel_w6_report")
    rep = w6_report.build(st, cfg, run_dir)
    assert rep["baselines"] == {} and rep["same_items"] == {} and rep["arms"]["w6-ref30m-norel-s1"]["status"] == "failed" and rep["relatives"] == []
    w6_report.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w6_report")["status"] == "pass"


def test_report_gate_fails_on_a_curve_of_one_point_or_other_cells(tmp_path):
    from ci.gates import GateFailure
    run_dir, cfg, doc, _ = _finished_fit(tmp_path / "a", statuses=("ok", "ok", "ok"))
    p = os.path.join(run_dir, w6_fit.OUT, "results.tsv")
    SC.write_rows(p, [r for r in SC.read_rows(p) if not (r["weights"].startswith("ema@tables") and r["variant"] == "w6-ref30m-s1")])
    st = Stage(run_dir, "rel_w6_report")
    w6_report.build(st, cfg, run_dir)
    with pytest.raises(GateFailure):
        w6_report.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w6_report")["status"] == "fail"
    run_dir, cfg, doc, _ = _finished_fit(tmp_path / "b", statuses=("ok", "ok", "ok"))      # baselines scored on other cells
    cfg = dict(cfg, rel_w6_report=dict(baselines=[_baselines(run_dir, digest=False)]))
    st = Stage(run_dir, "rel_w6_report")
    assert not any(w6_report.build(st, cfg, run_dir)["same_items"].values())
    with pytest.raises(GateFailure):
        w6_report.gate(st, cfg, run_dir)


def test_selftest_verdict():
    good = dict(exit=0, gpu_hidden=True, modules={"m": dict(tests=3, passed=3, failed={})})
    assert w6_selftest.problems(good, ["m"]) == []
    assert w6_selftest.problems(dict(good, gpu_hidden=False), ["m"]) != []
    assert w6_selftest.problems(good, ["m", "other"]) != []
    assert w6_selftest.problems(dict(good, modules={"m": dict(tests=3, passed=2, failed={"test_x": "boom"})}), ["m"]) != []
    assert w6_selftest.problems(dict(good, exit=1), ["m"]) != [] and w6_selftest.problems(dict(good, modules={"m": dict(tests=0, passed=0, failed={})}), ["m"]) != []
    with pytest.raises(SystemExit):
        w6_selftest.settings(dict(rel_w6_selftest=dict(modules=["/abs/x.py"])))
