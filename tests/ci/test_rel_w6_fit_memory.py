"""The training stage's scheduler on a shared GPU (run 20261123T110619: 36,100 MB free at the preflights, 20,918 MB when
the trainings began, another user's process having grown in between): the free memory is measured again when each arm
is to start, and an arm that no longer fits is refused (the run then fails at its gate) or, with the profile's
micro_fallback, preflighted again within what is free then, its micro-batch halved. The coordinator's stand-in trainer
(tests/ci/test_rel_c0_fit_arms.py) writes what the trainer writes; no torch."""
import json
import os

import rel_fixtures as RF
import test_rel_c0_fit_arms as C0T
from ci.gates import GateFailure
from ci.stages.common import Stage
from ci.stages.rel import w6_fit

REF = "w6-ref30m-s1"


def _trainer():
    """The stand-in's training, and a preflight that halves the micro-batch while its reserve exceeds the allowance
    (gpu_mb), as the trainer's preflight does."""
    base = C0T._trainer()

    def start(py, mode, conf, log, env):
        if mode != "preflight":
            return base(py, mode, conf, log, env)
        with open(conf) as f:
            c = json.load(f)
        micro, reserved = int(c["micro"]), float(C0T.RESERVED[c["name"]])
        while reserved > float(c["gpu_mb"]) and micro > 1:
            micro, reserved = micro // 2, reserved * 0.6
        ok = reserved <= float(c["gpu_mb"])                            # one table that does not fit: the preflight fails
        pre = dict(ok=ok, params=1000, micro=micro, lr=c["lr"], peak_mb=reserved - 400, reserved_mb=reserved, cells_per_s=10.0, flops_per_cell=1,
                   guard=dict(failed=False), error=None if ok else "one table of 16 rows does not fit")
        with open(os.path.join(c["out"], "preflight.json"), "w") as f:
            json.dump(pre, f)
        return C0T._Proc(0 if ok else 1), C0T._Log()
    return start


def _fit(tmp_path, monkeypatch, free, **block):
    """rel_w6_fit's build with the coordinator's two arms; `free` the free GPU memory at each measurement in turn (the
    last repeated). Returns the model index, the gate's record and the run directory."""
    b = dict(tables=dict(train=dict(run="/x/runs/a", path="data/rel_w2/v0/tableset.json", split=0),
                         heldout=dict(run="/x/runs/a", path="data/rel_w2/v0/tableset.json", split=1)),
             arms=C0T.ARMS, max_steps=10, wall_s=600)
    b.update(block)
    run_dir, cfg = RF.rel_run(tmp_path, kind="rel_traineval", stages=C0T.STAGES)
    cfg = dict(cfg, rel_w6_fit=b, train=dict(gpu_lease=False, device="auto"))
    st = Stage(run_dir, "rel_w6_fit")
    sets = dict(doc=dict(kind="real", name="v0", sha256="2" * 64, tables=100), groups=list(C0T.GROUPS), row_groups=dict(enumerate(C0T.GROUPS)), sealed=True)
    seen = list(free)
    monkeypatch.setattr(w6_fit, "table_set", lambda cfg, spec, verify: dict(sets))
    monkeypatch.setattr(w6_fit, "POLL_S", 0)
    monkeypatch.setattr(w6_fit.w6_tools, "interpreter", lambda st, cfg: ("py", "lock", {"torch": "2"}))
    monkeypatch.setattr(w6_fit.w6_tools, "gpu_free_mb", lambda: float(seen.pop(0) if len(seen) > 1 else seen[0]))
    monkeypatch.setattr(w6_fit.w6_tools, "start", _trainer())
    doc = w6_fit.build(st, cfg, run_dir)
    try:
        w6_fit.gate(st, cfg, run_dir)
    except GateFailure:
        pass
    with open(os.path.join(run_dir, "gates", "G4_rel_w6_fit.json")) as f:
        return doc, json.load(f), run_dir


def test_the_free_memory_is_measured_again_when_an_arm_is_to_start(tmp_path, monkeypatch):
    """110619's numbers: enough at the preflights and when the trainings began, 20,918 MB when the reference arm is to
    start: it is refused (it would have been started on the memory measured at the beginning), the other arm trains,
    and the run fails at its gate."""
    doc, g, _ = _fit(tmp_path, monkeypatch, free=[36100, 36100, 20918])
    by = {v["name"]: v for v in doc["variants"]}
    assert by[REF]["status"] == "failed" and by["w6-ref10m-s1"]["status"] == "ok"
    assert "not enough GPU memory: 20918 MB free now" in by[REF]["error"] and by[REF]["info"]["micro_fallback"] is None
    assert g["status"] == "fail"


def test_micro_fallback_halves_the_micro_batch_within_the_memory_free_then(tmp_path, monkeypatch):
    """With micro_fallback the refused arm is preflighted again within what is free then ((20,918 - 500) / 1.1 MB):
    its micro-batch halves from 4 to 2, it fits and trains with it (the training's configuration says so, the first
    preflight is kept beside), and the index records the fallback."""
    doc, g, run_dir = _fit(tmp_path, monkeypatch, free=[36100, 36100, 20918], micro_fallback=True)
    by = {v["name"]: v for v in doc["variants"]}
    assert by[REF]["status"] == "ok" and by["w6-ref10m-s1"]["status"] == "ok" and g["status"] == "pass"
    fb = by[REF]["info"]["micro_fallback"]
    assert fb == dict(free_mb=20918, needed_mb=21321, gpu_mb=int((20918 - 500) / 1.1), micro_from=4, micro_to=2, ok=True)
    folder = os.path.join(run_dir, by[REF]["info"]["folder"])
    with open(os.path.join(folder, "conf.json")) as f:
        conf = json.load(f)
    assert conf["micro"] == 2 and conf["gpu_mb"] == fb["gpu_mb"] and by[REF]["info"]["micro"] == 2
    with open(os.path.join(folder, "preflight_first.json")) as f:
        assert json.load(f)["micro"] == 4                              # the first preflight kept beside the second
    assert by["w6-ref10m-s1"]["info"]["micro_fallback"] is None


def test_micro_fallback_refuses_an_arm_that_still_does_not_fit(tmp_path, monkeypatch):
    """Too little free even at a micro-batch of one table: the second preflight fails, the arm is refused once, with
    the fallback in its error."""
    doc, g, _ = _fit(tmp_path, monkeypatch, free=[36100, 36100, 7000], micro_fallback=True)
    by = {v["name"]: v for v in doc["variants"]}
    fb = by[REF]["info"]["micro_fallback"]
    assert by[REF]["status"] == "failed" and "micro_fallback" in by[REF]["error"] and fb["ok"] is False and fb["micro_to"] is None
    assert g["status"] == "fail"
