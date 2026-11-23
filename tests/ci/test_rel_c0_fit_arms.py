"""Gate G4_rel_w6_fit holds a run of several arms to all of them (the programme lead's rule of 2026-11-23): an arm
refused for GPU memory, failed, or ended before its budget fails the run, so a reading of several arms never goes on
without one. The training stage runs here with its trainer replaced by a stand-in that writes what the trainer writes
(no torch): the scheduler, the model index and the gate are the stage's own."""
import json
import os

import pytest

import rel_fixtures as RF
from ci.gates import GateFailure
from ci.stages.common import Stage
from ci.stages.rel import w6_fit
from seeddrift.train.rel.evalh import score as SC

GROUPS = ("bacteria", "eukaryota.fungi")
STAGES = ["w6_selftest", "w6_fit", "w6_report"]
ARMS = [dict(variant="configs/rel/variants/w6-ref30m.yaml", seed=1), dict(variant="configs/rel/variants/w6-ref10m.yaml", seed=1)]
RESERVED = {"w6-ref30m-s1": 18928, "w6-ref10m-s1": 8000}          # 110619's reference arm: 18,928 MB reserved, 21,321 MB needed


class _Proc:
    def __init__(self, code):
        self.returncode = code

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


class _Log:
    def close(self):
        pass


def _trainer(ended_by="max_steps", preflight_fails=()):
    """A stand-in for the trainer's two modes: the files a preflight and a training write, as the trainer writes them."""
    def start(py, mode, conf, log, env):
        with open(conf) as f:
            c = json.load(f)
        name, folder = c["name"], c["out"]
        if mode == "preflight":
            ok = name not in preflight_fails
            pre = dict(ok=ok, params=1000, micro=int(c["micro"]), lr=c["lr"], peak_mb=RESERVED[name] - 400, reserved_mb=RESERVED[name], cells_per_s=10.0,
                       flops_per_cell=1, guard=dict(failed=not ok), error=None if ok else "the loss did not fall")
            with open(os.path.join(folder, "preflight.json"), "w") as f:
                json.dump(pre, f)
            return _Proc(0 if ok else 1), _Log()
        conds = ["scattered"]
        SC.write_rows(os.path.join(folder, "results.tsv"), [SC.result_row(
            run=c["run"], commit="c", variant=name, author="w6", module="models.axial", params=1000, seed=1, table_set="t", table_kind="real",
            items_sha256="5" * 64, item_half="all", layout="stored", budget="tables=40;wall_s=10", weights="ema@final", cells_seen=400, target_cells=40,
            cells_per_s=10.0, peak_mb=5, condition=k, group=SC.ALL, tier=SC.TIER, stratum=SC.ALL, passes=1, bits=5.0, n_items=12, status="ok",
            versions="v") for k in conds])
        for which in ("final", "best"):
            with open(os.path.join(folder, which + ".pt"), "wb") as f:
                f.write(which.encode())
        with open(os.path.join(folder, "summary.json"), "w") as f:
            json.dump(dict(status="ok", ended_by=ended_by, steps=10, tables_seen=40, cells_seen=400, conditions=conds, final={k: 5.0 for k in conds}), f)
        return _Proc(0), _Log()
    return start


def _fit(tmp_path, monkeypatch, free_mb, **over):
    """rel_w6_fit's build with two arms and the stand-in trainer, beside `free_mb` of free GPU memory; then its gate."""
    block = dict(tables=dict(train=dict(run="/x/runs/a", path="data/rel_w2/v0/tableset.json", split=0),
                             heldout=dict(run="/x/runs/a", path="data/rel_w2/v0/tableset.json", split=1)),
                 arms=ARMS, max_steps=10, wall_s=600)
    block.update(over.pop("block", {}))
    run_dir, cfg = RF.rel_run(tmp_path, kind="rel_traineval", stages=STAGES)
    cfg = dict(cfg, rel_w6_fit=block, train=dict(gpu_lease=False, device="auto"))
    st = Stage(run_dir, "rel_w6_fit")
    sets = dict(doc=dict(kind="real", name="v0", sha256="2" * 64, tables=100), groups=list(GROUPS), row_groups=dict(enumerate(GROUPS)), sealed=True)
    monkeypatch.setattr(w6_fit, "table_set", lambda cfg, spec, verify: dict(sets))
    monkeypatch.setattr(w6_fit, "POLL_S", 0)
    monkeypatch.setattr(w6_fit.w6_tools, "interpreter", lambda st, cfg: ("py", "lock", {"torch": "2"}))
    monkeypatch.setattr(w6_fit.w6_tools, "gpu_free_mb", lambda: float(free_mb))
    monkeypatch.setattr(w6_fit.w6_tools, "start", _trainer(**over))
    doc = w6_fit.build(st, cfg, run_dir)
    try:
        w6_fit.gate(st, cfg, run_dir)
    except GateFailure:
        pass
    with open(os.path.join(run_dir, "gates", "G4_rel_w6_fit.json")) as f:
        return doc, json.load(f)


def test_an_arm_refused_for_gpu_memory_fails_the_run(tmp_path, monkeypatch):
    doc, g = _fit(tmp_path, monkeypatch, free_mb=20918)                 # 110619: 20,918 MB free, the reference arm needs 21,321
    assert {v["name"]: v["status"] for v in doc["variants"]} == {"w6-ref30m-s1": "failed", "w6-ref10m-s1": "ok"}
    assert g["status"] == "fail" and g["evidence"]["trained"] == ["w6-ref10m-s1"]
    assert any(e.get("arm") == "w6-ref30m-s1" and "not trained" in e["error"] and "not enough GPU memory" in e["error"] for e in g["evidence"]["problems"])


def test_every_arm_trained_to_its_budget_passes(tmp_path, monkeypatch):
    doc, g = _fit(tmp_path, monkeypatch, free_mb=80000)
    assert [v["status"] for v in doc["variants"]] == ["ok", "ok"] and g["status"] == "pass" and g["evidence"]["problems"] == []


def test_an_arm_ended_before_its_budget_fails_the_run(tmp_path, monkeypatch):
    _, g = _fit(tmp_path / "steps", monkeypatch, free_mb=80000, ended_by="wall")    # max_steps set: the wall limit came first
    assert g["status"] == "fail" and any("ended by wall, not at its budget (max_steps)" in e["error"] for e in g["evidence"]["problems"])
    _, g = _fit(tmp_path / "cells", monkeypatch, free_mb=80000, ended_by="wall",
                block=dict(max_steps=None, arms=[dict(a, budget_cells=4000) for a in ARMS]))
    assert g["status"] == "fail" and any("not at its budget (budget)" in e["error"] for e in g["evidence"]["problems"])
    _, g = _fit(tmp_path / "wall", monkeypatch, free_mb=80000, ended_by="wall", block=dict(max_steps=None))
    assert g["status"] == "pass"                                        # no budget in cells or steps: the wall limit is the budget


def test_a_preflight_only_run_with_an_arm_not_preflighted_fails(tmp_path, monkeypatch):
    doc, g = _fit(tmp_path, monkeypatch, free_mb=80000, preflight_fails=("w6-ref10m-s1",), block=dict(preflight_only=True))
    assert doc["preflight_only"] is True and {v["name"]: v["status"] for v in doc["variants"]} == {"w6-ref30m-s1": "ok", "w6-ref10m-s1": "failed"}
    assert g["status"] == "fail" and any(e.get("arm") == "w6-ref10m-s1" and "not preflighted" in e["error"] for e in g["evidence"]["problems"])


@pytest.mark.parametrize("ended_by,budget,max_steps,ok", [("max_steps", None, 10, True), ("budget", 4000, None, True), ("budget", 4000, 10, True),
                                                          ("max_steps", 4000, 10, True), ("wall", 4000, 10, False), ("wall", None, None, True),
                                                          ("max_steps", None, None, False), (None, None, 10, False)])
def test_the_budget_an_arm_must_reach(ended_by, budget, max_steps, ok):
    doc = dict(variants=[dict(name="a", status="ok", info=dict(ended_by=ended_by, budget_cells=budget))])
    assert (w6_fit.arm_problems(doc, dict(rel_w6_fit=dict(max_steps=max_steps))) == []) is ok


def test_an_arm_may_lower_its_micro_batch(tmp_path, monkeypatch):
    """rel_w6_fit.arms[].micro: the micro-batch an arm starts its preflight with, lowered from the variant's (the
    programme lead's margin rule on a shared GPU); the trainer's configuration and the index carry it, and a micro-batch
    above the variant's, below 1 or not a whole number is refused."""
    from ci.stages.common import Stage as St
    st0 = St.__new__(St)
    block = dict(tables=dict(train=dict(run="/x/runs/a", path="data/rel_w2/v0/tableset.json", split=0),
                             heldout=dict(run="/x/runs/a", path="data/rel_w2/v0/tableset.json", split=1)),
                 arms=[dict(ARMS[0], micro=1), ARMS[1]])
    s = w6_fit.settings(st0, dict(rel_w6_fit=block))
    assert [a["micro"] for a in s["arms"]] == [1, None]
    conf = w6_fit.trainer_conf(dict(git=dict(sha="0" * 40)), s, s["arms"][0], None, "/x", "/x/runs/r")
    assert conf["micro"] == 1
    assert w6_fit.trainer_conf(dict(git=dict(sha="0" * 40)), s, s["arms"][1], None, "/x", "/x/runs/r")["micro"] == int(s["arms"][1]["variant"].get("micro", 4))
    for bad in (0, 8, 1.5, True, "1"):
        with pytest.raises(SystemExit):
            w6_fit.settings(st0, dict(rel_w6_fit=dict(block, arms=[dict(ARMS[0], micro=bad)])))
    doc, g = _fit(tmp_path, monkeypatch, free_mb=80000, block=dict(arms=[dict(ARMS[0], micro=1), ARMS[1]]))
    by = {v["name"]: v for v in doc["variants"]}
    assert by["w6-ref30m-s1"]["info"]["micro"] == 1 and by["w6-ref30m-s1"]["info"]["micro_profile"] == 1
    assert by["w6-ref10m-s1"]["info"]["micro_profile"] is None and g["status"] == "pass"
