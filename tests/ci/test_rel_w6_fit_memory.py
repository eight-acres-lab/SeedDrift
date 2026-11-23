"""The training stage's scheduler on a shared GPU (run 20261123T110619: 36,100 MB free at the preflights, 20,918 MB when
the trainings began, another user's process having grown in between): the free memory is measured again when each arm
is to start, and an arm that no longer fits is refused (the run then fails at its gate) or, with the profile's
micro_fallback, preflighted again within what is free then, its micro-batch halved. The coordinator's stand-in trainer
(tests/ci/test_rel_c0_fit_arms.py) writes what the trainer writes; no torch."""
import json
import os

import pytest

import rel_fixtures as RF
import test_rel_c0_fit_arms as C0T
from ci.gates import GateFailure
from ci.stages.common import Stage
from ci.stages.rel import w6_fit

REF = "w6-ref30m-s1"


class _Slow(C0T._Proc):
    """A training that runs for a few of the scheduler's polls before it ends; at its end it records the micro-batch
    its configuration then says (events: ("end", arm, micro))."""

    def __init__(self, code, conf, name, events, polls=3):
        super().__init__(code)
        self.left, self.conf, self.name, self.events = polls, conf, name, events

    def poll(self):
        self.left -= 1
        if self.left > 0:
            return None
        if self.left == 0:
            with open(self.conf) as f:
                self.events.append(("end", self.name, json.load(f)["micro"]))
        return self.returncode


def _trainer(started=None, events=None):
    """The stand-in's training (each running for a few polls; `started` collects the configuration every training
    started with), and a preflight that halves the micro-batch while its reserve exceeds the allowance (gpu_mb), as the
    trainer's preflight does. events: every call in order, ("preflight", arm), ("fit", arm, micro), ("end", arm, micro)."""
    base = C0T._trainer()
    events = [] if events is None else events

    def start(py, mode, conf, log, env):
        with open(conf) as f:
            c = json.load(f)
        if mode != "preflight":
            proc, lg = base(py, mode, conf, log, env)
            (started if started is not None else []).append(c)
            events.append(("fit", c["name"], c["micro"]))
            return _Slow(proc.returncode, conf, c["name"], events), lg
        events.append(("preflight", c["name"]))
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


def _fit(tmp_path, monkeypatch, free, parallel=1, started=None, events=None, **block):
    """rel_w6_fit's build with the coordinator's two arms, `parallel` at once; `free` the free GPU memory at each
    measurement in turn (the last repeated; or a function giving it). Returns the model index, the gate's record and the
    run directory."""
    b = dict(tables=dict(train=dict(run="/x/runs/a", path="data/rel_w2/v0/tableset.json", split=0),
                         heldout=dict(run="/x/runs/a", path="data/rel_w2/v0/tableset.json", split=1)),
             arms=C0T.ARMS, max_steps=10, wall_s=600)
    b.update(block)
    run_dir, cfg = RF.rel_run(tmp_path, kind="rel_traineval", stages=C0T.STAGES)
    cfg = dict(cfg, rel_w6_fit=b, train=dict(gpu_lease=False, device="auto"), rel=dict(cfg.get("rel") or {}, parallel=parallel))
    st = Stage(run_dir, "rel_w6_fit")
    sets = dict(doc=dict(kind="real", name="v0", sha256="2" * 64, tables=100), groups=list(C0T.GROUPS), row_groups=dict(enumerate(C0T.GROUPS)), sealed=True)
    seen = [] if callable(free) else list(free)
    monkeypatch.setattr(w6_fit, "table_set", lambda cfg, spec, verify: dict(sets))
    monkeypatch.setattr(w6_fit, "POLL_S", 0)
    monkeypatch.setattr(w6_fit.w6_tools, "interpreter", lambda st, cfg: ("py", "lock", {"torch": "2"}))
    monkeypatch.setattr(w6_fit.w6_tools, "gpu_free_mb", free if callable(free) else lambda: float(seen.pop(0) if len(seen) > 1 else seen[0]))
    monkeypatch.setattr(w6_fit.w6_tools, "start", _trainer(started, events))
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
    doc, g, run_dir = _fit(tmp_path, monkeypatch, free=[36100, 36100, 20918], micro_fallback=True, arms=C0T.ARMS[:1])
    by = {v["name"]: v for v in doc["variants"]}
    assert by[REF]["status"] == "ok" and g["status"] == "pass"
    fb = by[REF]["info"]["micro_fallback"]
    assert fb == dict(free_mb=20918, needed_mb=21321, gpu_mb=int((20918 - 500) / 1.1), micro_from=4, micro_to=2, ok=True)
    folder = os.path.join(run_dir, by[REF]["info"]["folder"])
    with open(os.path.join(folder, "conf.json")) as f:
        conf = json.load(f)
    assert conf["micro"] == 2 and conf["gpu_mb"] == fb["gpu_mb"] and by[REF]["info"]["micro"] == 2
    with open(os.path.join(folder, "preflight_first.json")) as f:
        assert json.load(f)["micro"] == 4                              # the first preflight kept beside the second


def test_micro_fallback_refuses_an_arm_that_still_does_not_fit(tmp_path, monkeypatch):
    """Too little free even at a micro-batch of one table: the second preflight fails, the arm is refused once, with
    the fallback in its error."""
    doc, g, _ = _fit(tmp_path, monkeypatch, free=[36100, 36100, 7000], micro_fallback=True, arms=C0T.ARMS[:1])
    by = {v["name"]: v for v in doc["variants"]}
    fb = by[REF]["info"]["micro_fallback"]
    assert by[REF]["status"] == "failed" and "micro_fallback" in by[REF]["error"] and fb["ok"] is False and fb["micro_to"] is None
    assert g["status"] == "fail"


def _never_during_a_training(events):
    """Every arm: no preflight once its training started, and the micro-batch its configuration says at the end of the
    training equal to the one it started with."""
    for k, e in enumerate(events):
        if e[0] == "fit":
            assert ("preflight", e[1]) not in events[k + 1:], (e, events)
            end = [x for x in events[k + 1:] if x[0] == "end" and x[1] == e[1]]
            assert end and end[0][2] == e[2], (e, end)


def test_micro_fallback_acts_before_a_training_starts_never_during_one(tmp_path, monkeypatch):
    """micro_fallback acts in the scheduling, before a training starts, never during a training. One arm with the
    switch on: preflighted again before it starts (the memory dropped since its preflight), then started at 2; the
    memory drops again while it trains and nothing is preflighted or changed. Two arms at once (the switch cannot be on
    there): the memory drops while the first trains, the second waits and starts once the first has ended; neither arm
    is preflighted after its training started, and each ends with the micro-batch it started with."""
    events = []
    doc, g, run_dir = _fit(tmp_path / "one", monkeypatch, free=[36100, 36100, 20918, 20918, 9000], events=events, micro_fallback=True,
                           arms=C0T.ARMS[:1])
    assert [e[0] for e in events] == ["preflight", "preflight", "fit", "end"] and events[2][2] == 2 and g["status"] == "pass"
    _never_during_a_training(events)
    events = []

    def free():                                                       # 36,100 MB until a training runs, 9,000 while one does, 20,000 after
        fits, ends = sum(e[0] == "fit" for e in events), sum(e[0] == "end" for e in events)
        return 36100.0 if not fits else (9000.0 if fits > ends else 20000.0)
    doc, g, run_dir = _fit(tmp_path / "two", monkeypatch, free=free, parallel=2, events=events)
    assert [e[:2] for e in events if e[0] != "preflight"] == [("fit", REF), ("end", REF), ("fit", "w6-ref10m-s1"), ("end", "w6-ref10m-s1")]
    assert [x["beside"] for x in doc["schedule"]] == [0, 0] and g["status"] == "pass"
    _never_during_a_training(events)
    for v in doc["variants"]:
        with open(os.path.join(run_dir, v["info"]["folder"], "conf.json")) as f:
            assert json.load(f)["micro"] == v["info"]["micro"] == 4
    with pytest.raises(SystemExit):                                   # arms compared in one run train under one micro-batch
        _fit(tmp_path / "refused", monkeypatch, free=[36100], micro_fallback=True)
