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


def test_every_variant_file_loads_and_round_1a_changes_one_key():
    st = Stage.__new__(Stage)
    folder = os.path.join(ROOT, "configs", "rel", "variants")
    names = sorted(f[:-5] for f in os.listdir(folder) if f.startswith("w6-") and f.endswith(".yaml"))
    for n in names:
        assert w6_fit.load_variant(st, "configs/rel/variants/%s.yaml" % n)["name"] == n
    load = lambda n: yaml.safe_load(open(os.path.join(folder, n + ".yaml")))                      # noqa: E731
    rest = lambda v, *keys: {k: x for k, x in v.items() if k not in ("name",) + keys}              # noqa: E731
    for n, base, lr in (("w6-ref30m-lr1.5e-4", "w6-ref30m", 1.5e-4), ("w6-ref30m-lr6e-4", "w6-ref30m", 6e-4),
                        ("w6-ref30m-leakfree-lr1.5e-4", "w6-ref30m-leakfree", 1.5e-4),
                        ("w6-ref100m-lr8e-4", "w6-ref100m", 8e-4), ("w6-ref100m-lr1e-4", "w6-ref100m", 1e-4)):
        assert load(n)["optimizer"] == {"lr": lr} and rest(load(n), "optimizer") == rest(load(base), "optimizer"), n
    assert load("w6-ref30m-hier")["size"] == {"layers": 12, "width": 384, "head": "hierarchical"}
    assert rest(load("w6-ref30m-hier"), "size") == rest(load("w6-ref30m"), "size")
    assert load("w6-ref30m-leakfree-hier")["size"] == {"layers": 12, "width": 384, "head": "hierarchical"}
    assert rest(load("w6-ref30m-leakfree-hier"), "size") == rest(load("w6-ref30m-leakfree"), "size")
    assert rest(load("w6-ref30m-leakfree"), "layout", "whole_gene_layout") == rest(load("w6-ref30m"), "layout")


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
                                          (GROUPS[0], SC.RELATIVES[0], bits + k + 0.0625, 4), (GROUPS[1], SC.RELATIVES[0], bits + k + 0.5625, 4),
                                          (SC.ALL, SC.RELATIVES[0], bits + k + 0.25, 11), (SC.ALL, SC.RELATIVES[1], bits + k + 2, 1)):
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
            with open(os.path.join(folder, "evals.jsonl"), "w") as f:
                for t, _, b in points:
                    f.write(json.dumps(dict(tables_seen=t, bits={c: b for c in conds}, items={c: "subset-" + c for c in conds})) + "\n")
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
    SC.write_rows(os.path.join(folder, "rows_subset.tsv"), [dict(r, item_half="subset", bits=float(r["bits"]) + 0.01) for r in rows])
    with open(os.path.join(folder, "results.json"), "w") as f:
        json.dump(dict(items={c: dict(sha256="digest-" + c if digest else "other", cells=10, tables=12) for c in cond.CONDITIONS},
                       subset=dict(items={c: dict(sha256="subset-" + c, cells=5, tables=6) for c in cond.CONDITIONS})), f)
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
    assert abs(worth["scattered"]["worth"] - 0.4) < 1e-9 and worth["scattered"]["node"] == 2000 and worth["scattered"]["tables"] == SC.RELATIVES[0]
    assert [b[0] for b in worth["scattered"]["by_bin"]] == ["0.2-0.3", "0.9-1.0"] and abs(worth["scattered"]["by_bin"][1][3] - 0.4) < 1e-9
    assert abs(rep["heldout_alone"] - 1 / 12) < 1e-9 and any("drew tables without other rows" in c for c in rep["caveats"])
    assert abs(worth["scattered"]["worth_at_node"] - 0.25) < 1e-9 and [p[0] for p in worth["scattered"]["per_node"]] == [1000, 2000]
    assert rep["scale"]["verdict"] == "bigger is better" and rep["scale"]["node"] == 2000 and rep["scale"]["arms"] == ["w6-ref10m-s1", "w6-ref30m-s1"]
    assert [(p["arms"], p["node"], p["verdict"]) for p in rep["scale_pairs"]] == [(["w6-ref10m-s1", "w6-ref30m-s1"], 2000, "bigger is better")]
    repair = {r[0]: r[1:] for r in rep["repair"]["rows"]}
    assert repair["w6-ref30m-s1"] == [6.0, 9.0, 3.0] and "copy_nearest" in repair
    assert all(abs(r[4]) < 1e-9 for r in rep["check"]["rows"])
    assert [r[0] for r in rep["by_group"]["rows"]] == list(GROUPS) and rep["by_group"]["rows"][1][1] == "eukaryotes"
    lead = {(r[0], r[1].split(" (")[0]): r[2:] for r in rep["lead"]["rows"]}
    assert lead[("scattered", "eukaryotes")] == [5.6625, None, 5.5625, 5.9625, 6.4625] and lead[("whole_gene", "prokaryotes, plasmids, viruses")][2] == 6.0625
    assert rep["aligned"]["columns"] == ["tables_seen", "w6-ref10m-s1 (scattered)", "w6-ref30m-norel-s1 (scattered+no_relatives)", "w6-ref30m-s1 (scattered)"]
    assert rep["aligned"]["rows"][1] == [1000, 5.85, 5.85, 5.75] and rep["same_items"] == {c: True for c in cond.CONDITIONS}
    assert rep["same_subset"] == {c: True for c in cond.CONDITIONS} and rep["lines"]["columns"] == ["condition", "copy_nearest", "own_freq"]
    assert ["scattered", 5.66, 6.16] in rep["lines"]["rows"]
    svg = open(os.path.join(run_dir, w6_report.OUT, "curves.svg")).read()
    assert svg.startswith("<svg") and svg.count("<polyline") == 3 and "own_freq, scattered" in svg
    text = open(os.path.join(run_dir, w6_report.OUT, "report.md")).read()
    assert "upper bound" not in text and "optimistic" in text and "bigger is better" in text and "passes over the training tables" in text
    assert "Every two arms with relatives at the last node both reached" in text and "a slight selection on the test set" in text
    assert "copy_nearest" in text and " NO" not in text
    curves = open(os.path.join(run_dir, w6_report.OUT, "curves.tsv")).read().splitlines()
    assert curves[0].split("\t")[:3] == ["arm", "point", "tables_seen"] and any(line.split("\t")[1:3] == ["final", "4000"] for line in curves)
    w6_report.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w6_report")["status"] == "pass"
    part = w6_report.summary(run_dir)
    assert len(json.dumps(part["json"])) < 20000 and part["json"]["baselines"]["copy_nearest"]["scattered"] == 5.65
    # the arm without relatives ends below its best node: its end is the conservative reading
    p = os.path.join(run_dir, w6_fit.OUT, "results.tsv")
    SC.write_rows(p, [dict(r, bits=float(r["bits"]) + 1.0) if r["weights"] == "ema@best" else r for r in SC.read_rows(p)])
    worth = {r["condition"]: r for r in w6_report.build(st, cfg, run_dir)["relatives"]}
    assert worth["scattered"]["without_at"].startswith("end (below its best node") and worth["scattered"]["bits_without"] == 5.75
    assert worth["scattered"]["bits_without_best_node"] == 6.65 and abs(worth["scattered"]["worth"] - 0.5) < 1e-9


def test_scale_at_the_last_node_of_all_arms_and_of_every_two():
    nodes = {"s": [0, 1000, 2000, 4000], "m": [0, 1000, 2000], "b": [0, 1000, 2000, 4000]}
    at = {("s", 2000): 5.0, ("m", 2000): 4.9, ("b", 2000): 4.95, ("s", 4000): 4.8, ("b", 4000): 4.7}

    def N(v, t, c):
        x = at.get((v, t))
        return None if x is None else x + (0.5 if c == cond.WHOLE_GENE else 0.0)
    scale, pairs = w6_report.scale_readings(["b", "s", "m"], {"s": 10, "m": 30, "b": 100}, nodes, N)
    assert scale["node"] == 2000 and scale["arms"] == ["s", "m", "b"] and scale["verdict"] == "cannot be read"     # the biggest above the middle
    assert {tuple(p["arms"]): (p["node"], p["verdict"]) for p in pairs} == {("s", "m"): (2000, "bigger is better"), ("s", "b"): (4000, "bigger is better"),
                                                                            ("m", "b"): (2000, "cannot be read")}
    assert pairs[1]["bits"]["b"] == {cond.SCATTERED: 4.7, cond.WHOLE_GENE: 5.2}
    assert w6_report.scale_readings(["s"], {"s": 10}, nodes, N) == (None, [])
    assert w6_report.scale_readings(["s", "x"], {"s": 10, "x": 20}, dict(nodes, x=[0]), N) == (None, [])        # no node besides step 0 in common


def test_report_reads_the_whole_gene_in_two_layouts(tmp_path):
    """An arm trained leak-free reads the whole gene leak-free and also in the stored layout it was trained in: the
    stored rows come under "<condition>@stored", the others keep their condition, and a caveat says so."""
    run_dir, cfg, doc, _ = _finished_fit(tmp_path, statuses=("ok", "ok", "ok"))
    p = os.path.join(run_dir, w6_fit.OUT, "results.tsv")
    rows = SC.read_rows(p)
    wg = [cond.WHOLE_GENE, cond.WHOLE_GENE + "+" + cond.REPAIRED]
    lf = [dict(r, layout="leakfree", bits=float(r["bits"]) + 0.5) for r in rows if r["variant"] == "w6-ref30m-s1" and r["condition"] in wg]
    SC.write_rows(p, rows + lf)
    rep = w6_report.build(Stage(run_dir, "rel_w6_report"), dict(cfg, rel_w6_report={}), run_dir)
    fin = rep["arms"]["w6-ref30m-s1"]["final"]
    assert abs(fin["whole_gene"] - fin["whole_gene@stored"] - 0.5) < 1e-9 and "whole_gene+repaired@stored" in rep["conditions"]
    assert "whole_gene@stored" not in rep["arms"]["w6-ref10m-s1"]["final"] or rep["arms"]["w6-ref10m-s1"]["final"]["whole_gene@stored"] is None
    assert any("trained stored, evaluated leak-free" in c for c in rep["caveats"])


def test_report_reads_the_leak_of_the_stored_layout(tmp_path):
    """leak_pairs: the leak-free arm's bits minus the stored arm's, under scattered and suffix, on all tables and on the
    tables with a relative on the target block; suffix rows appear among the report's conditions."""
    run_dir, cfg, doc, _ = _finished_fit(tmp_path, statuses=("ok", "ok", "ok"))
    p = os.path.join(run_dir, w6_fit.OUT, "results.tsv")
    rows = SC.read_rows(p)
    extra = []
    for arm, bits in (("w6-ref30m-s1", 4.0), ("w6-ref10m-s1", 4.3)):
        extra += [r for r in _rows(os.path.basename(run_dir), arm, [(2000, 20000, bits)], bits, [cond.SUFFIX]) if r["weights"] == "ema@final"]
    SC.write_rows(p, rows + extra)
    rep = w6_report.build(Stage(run_dir, "rel_w6_report"), dict(cfg, rel_w6_report=dict(leak_pairs=[["w6-ref10m-s1", "w6-ref30m-s1"]],
                                                                                      pairs=[dict(name="head", first="w6-ref30m-s1", second="w6-ref10m-s1")])), run_dir)
    at = {(r["node"], r["tables"]): r for r in rep["leak_nodes"]}
    assert sorted({k[0] for k in at}) == [1000, 2000] and abs(at[(2000, SC.ALL)]["leak"] - 0.1) < 1e-9
    q = rep["pairs"][0]
    assert [r[0] for r in q["rows"]] == [1000, 2000, "end"] and abs(q["rows"][-1][3] - 0.1) < 1e-9 and abs(q["rows"][-1][11] - 0.3) < 1e-9
    got = {(r["condition"], r["tables"]): r for r in rep["leak"]}
    assert abs(got[(cond.SUFFIX, SC.ALL)]["leak"] - 0.3) < 1e-9 and abs(got[(cond.SCATTERED, SC.ALL)]["leak"] - 0.1) < 1e-9
    assert got[(cond.SUFFIX, SC.RELATIVES[0])]["bits_stored"] == 4.25 and cond.SUFFIX in rep["conditions"]
    text = open(os.path.join(run_dir, w6_report.OUT, "report.md")).read()
    assert "## The leak of the stored layout" in text and "Scattered at every node both reached" in text and "## head: w6-ref10m-s1 against w6-ref30m-s1" in text
    with pytest.raises(SystemExit):
        w6_report.build(Stage(run_dir, "rel_w6_report"), dict(cfg, rel_w6_report=dict(leak_pairs=[["nobody", "w6-ref30m-s1"]])), run_dir)


def test_report_reads_learning_rate_sweeps_and_the_curve_over_passes(tmp_path):
    run_dir, cfg, doc, _ = _finished_fit(tmp_path, statuses=("ok", "ok", "ok"))
    folder = os.path.join(run_dir, w6_fit.OUT, "w6-ref30m-s1")
    with open(os.path.join(folder, "steps.csv"), "w") as f:
        f.write("step,tables_seen,cells_seen,target_cells,alone_tables,loss_bits,lr,cells_per_s,peak_mb,grad_norm,wall_s,bits_scattered\n")
        for k in range(1, 501):
            f.write("%d,%d,0,0,0,%.3f,0,0,0,0,0,\n" % (k, 4 * k, 6.0 - k / 250.0))
    sc = dict(lr_sweeps=[dict(name="fixture", arms=["w6-ref10m-s1", ["w6-ref30m-s1"]])], passes=["w6-ref30m-s1"], notes=["A note of the profile."])
    rep = w6_report.build(Stage(run_dir, "rel_w6_report"), dict(cfg, rel_w6_report=sc), run_dir)
    w = rep["lr_sweeps"][0]
    assert w["node"] == 2000 and [r[0] for r in w["rows"]] == [1000, 2000] and w["best"][cond.SCATTERED]["entry"] == "w6-ref30m-s1"
    assert abs(w["best"][cond.SCATTERED]["margin"] - 0.1) < 1e-9 and w["best"][cond.SCATTERED]["reading"] is None   # no seeds: no threshold
    o = rep["passes"][0]
    assert [r[0] for r in o["rows"]] == [1000, 2000] and abs(o["rows"][1][1] - 2000 / 100800) < 1e-12
    assert abs(o["rows"][1][5] - 0.3) < 1e-9                                       # the fall of scattered from 5.75 to 5.45
    assert abs(o["rows"][1][7] - (6.0 - 375.5 / 250.0)) < 1e-6                      # the training loss of steps 251..500 (tables 1,004..2,000)
    text = open(os.path.join(run_dir, w6_report.OUT, "report.md")).read()
    assert "## Learning rates of fixture" in text and "over passes of the training tables" in text and "- A note of the profile." in text
    with pytest.raises(SystemExit):
        w6_report.build(Stage(run_dir, "rel_w6_report"), dict(cfg, rel_w6_report=dict(passes=["nobody"])), run_dir)


def test_learning_rate_check_has_three_conclusions():
    def reading(bits, noise=lambda t, c: dict(bits=0.05, node=32000), diverged=False):
        return w6_report.lr_reading("s", "l", "h", 64000, lambda a, t, c: bits[a][0 if c == cond.SCATTERED else 1], noise, diverged)
    small, low = (4.15, 4.28), (4.25, 4.34)
    r = reading(dict(s=small, l=low, h=(4.16, 4.30)))                                                 # above the small arm by no more than 0.05
    assert r["conclusion"] == "caught up" and r["passed"] is False
    r = reading(dict(s=small, l=low, h=(4.05, 4.20)))                                                 # below it by more than 0.05 under both
    assert r["conclusion"] == "caught up" and r["passed"] is True
    assert reading(dict(s=small, l=low, h=(4.05, 4.25)))["passed"] is False                           # whole_gene only 0.03 below
    assert reading(dict(s=small, l=(4.30, 4.40), h=(4.21, 4.34)))["conclusion"] == "part"            # below the lower rate by more than 0.05
    assert reading(dict(s=small, l=low, h=(4.26, 4.35)))["conclusion"] == "not the rate"
    r = reading(dict(s=small, l=low, h=(4.16, 4.36)))                                                 # scattered caught up, whole_gene not the rate
    assert r["conclusion"] is None and "disagree" in r["why"]
    assert reading(dict(s=small, l=low, h=(4.16, 4.30)), diverged=True)["conclusion"] == "not the rate"
    assert reading(dict(s=small, l=low, h=(4.16, 4.30)), noise=None)["conclusion"] == "part"           # threshold zero: 0.01 above the small arm
    assert abs(reading(dict(s=small, l=low, h=(4.16, 4.30)))["by_condition"][cond.SCATTERED]["high_minus_small"] - 0.01) < 1e-9


def test_report_reads_the_learning_rate_check(tmp_path):
    run_dir, cfg, doc, _ = _finished_fit(tmp_path / "a", statuses=("ok", "ok", "ok"))
    second, _, _, _ = _finished_fit(tmp_path / "b", statuses=("ok", "failed", "failed"))
    _passed(second, os.path.join(w6_fit.OUT, "models.json"))
    p = os.path.join(second, w6_fit.OUT, "results.tsv")                 # the arm at the higher rate: 0.3 bits lower at every node
    SC.write_rows(p, [dict(r, bits=float(r["bits"]) - 0.3) if r["weights"].startswith("ema@tables=") else r for r in SC.read_rows(p)])
    high = "w6-ref30m-s1 @lr"
    check = dict(small="w6-ref10m-s1", low="w6-ref30m-s1", high=high, note="The two big arms trained on different machines.")
    cfg = dict(cfg, rel_w6_report=dict(fits=[dict(run=second, label="lr")], readings_exclude=[high], lr_check=check))
    st = Stage(run_dir, "rel_w6_report")
    rep = w6_report.build(st, cfg, run_dir)
    lr = rep["lr_check"]
    assert lr["at_node"]["node"] == 2000 and lr["at_node"]["conclusion"] == "caught up" and lr["conclusion"].startswith("the big arm at the higher rate caught up")
    assert lr["at_end"]["conclusion"] == "caught up" and [r[0] for r in lr["curves"]] == [1000, 2000] and lr["final"][0][0] == "end"
    assert abs(lr["curves"][1][3] - 5.15) < 1e-9 and rep["scale"]["arms"] == ["w6-ref10m-s1", "w6-ref30m-s1"]       # the excluded arm is not in the scale
    text = open(os.path.join(run_dir, w6_report.OUT, "report.md")).read()
    assert "## The learning rate" in text and "The two big arms trained on different machines." in text
    assert lr["passed"] is True and lr["sign"] == w6_report.SCALING_SIGN
    head = text.split("What the numbers do not show")[0]
    assert "Key points:" in head and w6_report.SCALING_SIGN in head and "Scale, at the last node" in head
    w6_report.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w6_report")["status"] == "pass"
    with pytest.raises(SystemExit):
        w6_report.build(st, dict(cfg, rel_w6_report=dict(fits=[dict(run=second, label="lr")], lr_check=dict(check, high="nobody"))), run_dir)
    with pytest.raises(SystemExit):
        w6_report.settings(dict(rel_w6_report=dict(lr_check=dict(small="a", low="b"))))


def _passed(run_dir, rel):
    """Record the hand-off file rel of a fixture run in a stage record and mark the run passed."""
    from ci.record import Record, stage_report
    p = os.path.join(run_dir, rel)
    stage_report(run_dir, "rel_w6_fit" if "/fit/" in rel else "rel_w6_eval", "", 0.0, [dict(name="models", path=rel, sha256=handoff._file_sha(p), bytes=1)])
    rec = Record.open(run_dir)
    rec.merge_stages()
    rec.update(status="passed")


def test_report_of_two_runs_with_labels_exclusions_and_a_pair_of_runs(tmp_path):
    run_dir, cfg, doc, _ = _finished_fit(tmp_path / "a", statuses=("ok", "ok", "ok"))
    second, _, _, _ = _finished_fit(tmp_path / "b", statuses=("ok", "failed", "failed"))
    _passed(second, os.path.join(w6_fit.OUT, "models.json"))
    cfg = dict(cfg, rel_w6_report=dict(fits=[dict(run=second, label="i2")], readings_exclude=["w6-ref30m-s1"],
                                       seeds=[["w6-ref30m-s1", "w6-ref30m-s1 @i2"]]))
    st = Stage(run_dir, "rel_w6_report")
    rep = w6_report.build(st, cfg, run_dir)
    assert {"w6-ref30m-s1", "w6-ref30m-s1 @i2", "w6-ref10m-s1 @i2"} <= set(rep["arms"]) and rep["arms"]["w6-ref10m-s1 @i2"]["status"] == "failed"
    assert rep["seeds"][0]["rows"] == [[1000, 5.75, 5.75, 0.0, 6.75, 6.75, 0.0], [2000, 5.45, 5.45, 0.0, 6.45, 6.45, 0.0]]
    assert rep["scale"]["arms"] == ["w6-ref10m-s1", "w6-ref30m-s1 @i2"] and rep["scale"]["verdict"] == "bigger is better"     # the excluded arm is no level
    assert [r["with_relatives"] for r in rep["relatives"]] == ["w6-ref30m-s1 @i2", "w6-ref30m-s1 @i2"]
    text = open(os.path.join(run_dir, w6_report.OUT, "report.md")).read()
    assert "Runs of one seed of one arm" in text and "Left out of the readings" in text
    w6_report.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w6_report")["status"] == "pass"


def test_report_with_two_seeds_reads_against_their_difference(tmp_path):
    run_dir, cfg, doc, _ = _finished_fit(tmp_path / "a", statuses=("ok", "ok", "ok"))
    second, _, _, _ = _finished_fit(tmp_path / "b", statuses=("ok", "failed", "failed"))
    _passed(second, os.path.join(w6_fit.OUT, "models.json"))
    conds = list(cond.CONDITIONS)
    # the second run of the 30 M arm: 0.2 bits above the first at every node, and one node more (4,000 tables), which the
    # 10 M arm of the first run also reached
    p = os.path.join(second, w6_fit.OUT, "results.tsv")
    later = [r for r in _rows(os.path.basename(second), "w6-ref30m-s1", [(4000, 40000, 4.5)], 4.5, conds) if r["weights"] == "ema@tables=4000"]
    SC.write_rows(p, [dict(r, bits=float(r["bits"]) + 0.2) if r["weights"].startswith("ema@tables=") else r for r in SC.read_rows(p)] + later)
    cfg = dict(cfg, rel_w6_report=dict(fits=[dict(run=second, label="i2")], seeds=[["w6-ref30m-s1", "w6-ref30m-s1 @i2"]]))
    st = Stage(run_dir, "rel_w6_report")
    level = "w6-ref30m-s1 + w6-ref30m-s1 @i2"
    # the levels at 2,000 tables, the last node both reached: 30 M is the mean of its two seeds (5.45 and 5.65), not below 10 M (5.55)
    rep = w6_report.build(st, cfg, run_dir)
    assert rep["scale"]["node"] == 2000 and abs(rep["scale"]["bits"][level][cond.SCATTERED] - 5.55) < 1e-9 and rep["scale"]["steps"] == ["not lower"]
    assert rep["scale"]["seeds_read"] == {level: ["w6-ref30m-s1", "w6-ref30m-s1 @i2"]} and rep["scale"]["verdict"] == "cannot be read"
    rows = rep["seeds"][0]["rows"]
    assert [r[0] for r in rows] == [1000, 2000] and abs(rows[0][3] - 0.2) < 1e-9 and abs(rows[1][6] - 0.2) < 1e-9
    noise = {(n["node"], n["condition"], n["stratum"]): n["bits"] for n in rep["noise"]}
    assert abs(noise[(2000, cond.SCATTERED, SC.ALL)] - 0.2) < 1e-9 and abs(noise[(2000, cond.WHOLE_GENE, SC.RELATIVES[0])] - 0.2) < 1e-9
    # the 10 M arm reaches 4,000 tables too, where only the second seed of 30 M did: 10 M 5.25, 30 M 4.75 (scattered), more than 0.2 apart
    q = os.path.join(run_dir, w6_fit.OUT, "results.tsv")
    SC.write_rows(q, SC.read_rows(q) + [r for r in _rows(os.path.basename(run_dir), "w6-ref10m-s1", [(4000, 40000, 5.0)], 5.0, conds)
                                        if r["weights"] == "ema@tables=4000"])
    rep = w6_report.build(st, cfg, run_dir)
    assert rep["scale"]["arms"] == ["w6-ref10m-s1", level] and rep["scale"]["node"] == 4000 and rep["scale"]["verdict"] == "bigger is better"
    assert rep["scale"]["seeds_read"] == {level: ["w6-ref30m-s1 @i2"]} and rep["scale"]["noise"][cond.SCATTERED]["node"] == 2000
    assert abs(rep["scale"]["bits"][level][cond.SCATTERED] - 4.75) < 1e-9
    # a step in size lower by no more than the seeds' difference cannot be read
    vals = {"small": {cond.SCATTERED: 5.55, cond.WHOLE_GENE: 6.55}, "big": {cond.SCATTERED: 5.5, cond.WHOLE_GENE: 6.3}}
    scale, _ = w6_report.scale_readings(["big", "small"], {"small": 1, "big": 2}, {"small": [2000], "big": [2000]}, lambda v, t, c: vals[v][c],
                                        lambda t, c: dict(bits=0.2, node=2000))
    assert scale["verdict"] == "cannot be read" and scale["steps"] == ["lower by no more than the seeds' difference"]
    worth = {(r["with_relatives"], r["condition"]): r for r in rep["relatives"]}
    one, two = worth[("w6-ref30m-s1", cond.SCATTERED)], worth[("w6-ref30m-s1 @i2", cond.SCATTERED)]
    assert abs(one["worth_at_node"] - 0.25) < 1e-9 and one["reading_at_node"] == "beyond the seeds' difference"
    assert abs(two["worth_at_node"] - 0.05) < 1e-9 and two["reading_at_node"] == "cannot be read" and two["noise_at_node"]["node"] == 2000
    assert one["reading_end"] == "beyond the seeds' difference" and one["noise_end"]["node"] == 2000
    text = open(os.path.join(run_dir, w6_report.OUT, "report.md")).read()
    assert "Against the seeds' difference" in text and "only " + "w6-ref30m-s1 @i2" in text and "the only noise estimate" in text
    assert "by the difference at 2,000 tables is only an approximation" in text
    assert any("whose difference (seeds and machines at once)" in c for c in rep["caveats"])
    w6_report.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w6_report")["status"] == "pass"


def test_report_takes_the_final_evaluation_of_an_evaluation_run(tmp_path):
    from ci.stages.common import write_json
    run_dir, cfg, doc, _ = _finished_fit(tmp_path, statuses=("ok", "ok", "ok"))
    folder = os.path.join(run_dir, "data", "rel_w6", "eval")
    os.makedirs(os.path.join(folder, "w6-ref30m-s1"))
    rows = [dict(r, bits=float(r["bits"]) - 1.0) for r in _rows(os.path.basename(run_dir), "w6-ref30m-s1", [(0, 0, 6.0)], 5.0, list(cond.CONDITIONS))
            if r["weights"] == "ema@final"]
    SC.write_rows(os.path.join(folder, "results.tsv"), rows)
    v = dict(next(x for x in doc["variants"] if x["name"] == "w6-ref30m-s1"))
    v["info"] = dict(v["info"], folder=os.path.relpath(os.path.join(folder, "w6-ref30m-s1"), run_dir), final_items={c: "again-" + c for c in cond.CONDITIONS})
    index = handoff.finish(dict(format="rel-models", format_version=1, tablesets=doc["tablesets"], variants=[v], probe=False, groups=list(GROUPS),
                                producer=dict(stage="w6_eval", package="W6", run_id=os.path.basename(run_dir), commit="c")))
    write_json(os.path.join(folder, "models.json"), index)
    _passed(run_dir, os.path.join("data", "rel_w6", "eval", "models.json"))
    cfg = dict(cfg, rel_w6_report=dict(fits=[dict(run=run_dir, folder="data/rel_w6/eval")]))
    rep = w6_report.build(Stage(run_dir, "rel_w6_report"), cfg, run_dir)
    assert rep["arms"]["w6-ref30m-s1"]["final"]["scattered"] == 4.25 and rep["arms"]["w6-ref10m-s1"]["final"]["scattered"] == 5.35
    assert rep["arms"]["w6-ref30m-s1"]["info"]["final_items"]["scattered"] == "again-scattered" and rep["arms"]["w6-ref30m-s1"]["evaluations"] == 4


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


def test_evaluation_again_settings_and_source():
    from ci.stages.rel import w6_eval
    good = dict(rel_w6_eval={"from": "/x/runs/t", "tables": {"heldout": dict(run="/x/runs/tables", path="data/rel_w2/v0/tableset.json", split=1)}})
    s = w6_eval.settings(good)
    assert s["source"] == "/x/runs/t" and s["best"] is True and s["final_per_group"] == 0 and s["heldout"]["split"] == 1
    for bad in ({}, {"from": "/x"}, dict(good["rel_w6_eval"], nonsense=1),
                {"from": "/x", "tables": {"heldout": dict(run="/x", path="p", split="one")}},
                dict(good["rel_w6_eval"], arms=[dict(name="w6-ref30m-s1", untrained=True)]),                 # untrained without a readout
                dict(good["rel_w6_eval"], readout=dict(presence_arms=["w6-ref30m"])),                        # a readout without its module
                dict(good["rel_w6_eval"], arms=["w6-ref30m-s1", "w6-ref30m-s1"])):
        with pytest.raises(SystemExit):
            w6_eval.settings(dict(rel_w6_eval=bad))
    ro = w6_eval.settings(dict(rel_w6_eval=dict(good["rel_w6_eval"], **{"from": ["/x/runs/a", "/x/runs/b"]},
                                                arms=["w6-ref30m-s1", dict(name="w6-ref30m-s1", untrained=True)],
                                                readout=dict(module="seeddrift.train.rel.readouts.w18_passes", presence_arms=["w6-ref30m"]))))
    assert ro["sources"] == ["/x/runs/a", "/x/runs/b"] and ro["arms"][1] == dict(name="w6-ref30m-s1", untrained=True)
    assert ro["readout"] == dict(module="seeddrift.train.rel.readouts.w18_passes", presence_arms=["w6-ref30m"], max_rows=32, layout="stored",
                                 in_place_only=False)
    assert w6_eval.variant_of_arm("w6-ref100m-lr4e-4-s1") == "w6-ref100m-lr4e-4" and w6_eval.variant_of_arm("w6-ref30m-s12") == "w6-ref30m"


def test_evaluation_again_reads_ruler_sets_and_training_never(tmp_path):
    from ci.record import Record, stage_report
    from ci.stages.rel import w6_eval
    sets = {}
    for kind in ("real", "ruler"):
        run_dir, cfg = RF.rel_run(tmp_path / kind, kind="rel_tables", stages=["w2_v0"], run_id="20261122T000000-exp-rel-00000000-" + kind)
        doc = RF.write_tableset(run_dir, stage="w2_v0", name="v0-" + kind, kind=kind, genomes=[dict(id="g1", taxid=562, split="heldout")], package="W2")
        path = kinds.stage("w2_v0")["output"]
        stage_report(run_dir, "rel_w2_v0", "", 0.0, [dict(name="tableset", path=path, sha256=handoff._file_sha(os.path.join(run_dir, path)), bytes=1)])
        rec = Record.open(run_dir)
        rec.merge_stages()
        rec.update(status="passed")
        sets[kind] = dict(run=run_dir, path=path, sha256=doc["sha256"])
    assert w6_eval.heldout_set(dict(sets["real"], split=1), verify=True)["kind"] == "real"
    assert w6_eval.heldout_set(dict(sets["ruler"], split=0), verify=True)["kind"] == "ruler"      # a ruler set: the split its builder set
    with pytest.raises(SystemExit):                                     # a real held-out set is split 1
        w6_eval.heldout_set(dict(sets["real"], split=0), verify=True)
    with pytest.raises(SystemExit):                                     # the training stage never reads a ruler set
        w6_fit.table_set(cfg, dict(sets["ruler"], split=1), verify=True)
    # a real set marked not for training (package W12's development sets): read by the evaluation, never by the training
    run_dir = sets["real"]["run"]
    p = os.path.join(run_dir, sets["real"]["path"])
    with open(p) as f:
        doc = json.load(f)
    doc = handoff.finish(dict({k: v for k, v in doc.items() if k != "sha256"}, not_for_training=True))
    with open(p, "w") as f:
        json.dump(doc, f)
    stage_report(run_dir, "rel_w2_v0", "", 0.0, [dict(name="tableset", path=sets["real"]["path"], sha256=handoff._file_sha(p), bytes=1)])
    rec = Record.open(run_dir)
    rec.merge_stages()
    rec.update(status="passed")
    assert w6_eval.heldout_set(dict(sets["real"], sha256=doc["sha256"], split=1), verify=True)["kind"] == "real"
    with pytest.raises(SystemExit):
        w6_fit.table_set(cfg, dict(sets["real"], sha256=doc["sha256"], split=1), verify=True)


def test_evaluation_again_reads_only_a_recorded_model_index(tmp_path):
    from ci.record import Record, stage_report
    from ci.stages.rel import w6_eval
    run_dir, cfg, doc, st = _finished_fit(tmp_path, statuses=("ok", "failed", "ok"))
    with pytest.raises(SystemExit):                                     # the stage record does not list it yet
        w6_eval.source_index(run_dir)
    p = os.path.join(run_dir, w6_fit.OUT, "models.json")
    stage_report(run_dir, "rel_w6_fit", "", 0.0, [dict(name="models", path=os.path.join(w6_fit.OUT, "models.json"), sha256=handoff._file_sha(p), bytes=1)])
    rec = Record.open(run_dir)
    rec.merge_stages()
    rec.update(status="gate_failed")                                    # a run that failed still serves its index
    got, record = w6_eval.source_index(run_dir)
    assert got["sha256"] == doc["sha256"] and record["status"] == "gate_failed"
    with open(p, "a") as f:
        f.write(" ")
    with pytest.raises(SystemExit):                                     # changed after its stage recorded it
        w6_eval.source_index(run_dir)


def test_selftest_verdict():
    good = dict(exit=0, gpu_hidden=True, modules={"m": dict(tests=3, passed=3, failed={})})
    assert w6_selftest.problems(good, ["m"]) == []
    assert w6_selftest.problems(dict(good, gpu_hidden=False), ["m"]) != []
    assert w6_selftest.problems(good, ["m", "other"]) != []
    assert w6_selftest.problems(dict(good, modules={"m": dict(tests=3, passed=2, failed={"test_x": "boom"})}), ["m"]) != []
    assert w6_selftest.problems(dict(good, exit=1), ["m"]) != [] and w6_selftest.problems(dict(good, modules={"m": dict(tests=0, passed=0, failed={})}), ["m"]) != []
    with pytest.raises(SystemExit):
        w6_selftest.settings(dict(rel_w6_selftest=dict(modules=["/abs/x.py"])))
