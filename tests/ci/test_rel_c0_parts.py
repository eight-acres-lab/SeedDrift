"""The shared parts of the relational-model line's foundation: the registry of arm parts and the stub arm, the gate
dispatch (a stage's phase and the kind's G3_rel and G4_rel_handoff), the stub stage, the run-summary hook and
`ci compare --expect within_tolerance` with the tolerance function at its fixed path."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import rel_fixtures as RF
from conftest import ROOT
from ci.stages.rel import common, gates as rel_gates
from seeddrift.train.rel import registry


def test_registry_finds_the_stub_arm_and_its_interface():
    assert "uniform" in registry.available("model") and "common" not in registry.available("model")
    m = registry.load("model", "uniform")
    target = np.zeros((4, 3, 8), dtype=bool)
    target[0, 0, :5] = True
    lp = m.logp(None, target)
    assert lp.shape == (5, 65) and np.allclose(np.exp(lp).sum(1), 1.0)
    assert abs(-lp[0, 0] / np.log(2) - np.log2(65)) < 1e-12 and m.info["parameters"] == 0


@pytest.mark.parametrize("cat,name", [("model", "Uniform"), ("model", "../sealed"), ("model", "construct"), ("model", "absent"),
                                      ("mask", "uniform"), ("nothing", "uniform")])
def test_registry_refusals_fail(cat, name):
    with pytest.raises(registry.RegistryError):
        registry.load(cat, name)


def test_registry_checks_what_the_contract_asks(tmp_path, monkeypatch):
    with pytest.raises(registry.RegistryError):
        registry.compose(model="uniform", objective="none_yet", mask="none_yet")


def _gate_json(run_dir, gate):
    with open(os.path.join(run_dir, "gates", gate + ".json")) as f:
        return json.load(f)


def _stage(run_dir, name):
    from ci.stages.common import Stage
    return Stage(run_dir, name)


def test_kind_phase_passes_a_synthetic_tableset(tmp_path, monkeypatch):
    run_dir, cfg = RF.rel_run(tmp_path)
    RF.write_tableset(run_dir)
    monkeypatch.setattr(common, "sealed_set", lambda cfg, spec=None: RF.sealed())
    with _stage(run_dir, "gates_rel_tables") as st:
        rel_gates.phase(st, cfg, run_dir, "rel_tables")
    assert _gate_json(run_dir, "G4_rel_handoff")["status"] == "pass" and _gate_json(run_dir, "G3_rel")["status"] == "pass"


def test_kind_phase_fails_on_a_sealed_genome_or_a_broken_manifest(tmp_path, monkeypatch):
    from ci.gates import GateFailure
    run_dir, cfg = RF.rel_run(tmp_path)
    RF.write_tableset(run_dir, kind="real", genomes=[dict(id="medium_test_GCF_000000001.1", taxid=562, split="train")])
    monkeypatch.setattr(common, "sealed_set", lambda cfg, spec=None: RF.sealed())
    monkeypatch.setattr(common, "check_lineage", lambda *a, **k: dict(checked=1))
    with pytest.raises(GateFailure):
        with _stage(run_dir, "gates_rel_tables") as st:
            rel_gates.phase(st, cfg, run_dir, "rel_tables")
    assert _gate_json(run_dir, "G3_rel")["status"] == "fail" and _gate_json(run_dir, "G4_rel_handoff")["status"] == "pass"
    out = os.path.join(run_dir, "data", "rel_c0", "synth", "tableset.json")
    doc = json.load(open(out))
    doc["tables"] = 0
    json.dump(doc, open(out, "w"))
    with pytest.raises(GateFailure):
        with _stage(run_dir, "gates_rel_tables") as st:
            rel_gates.phase(st, cfg, run_dir, "rel_tables")
    assert _gate_json(run_dir, "G4_rel_handoff")["status"] == "fail"


def test_stage_phases_refuse_stubs_other_kinds_and_stages_not_run_fails(tmp_path):
    run_dir, cfg = RF.rel_run(tmp_path, stages=["c0_synth", "w2_dist"])
    for phase in ("rel_w2_dist", "rel_w2_v0", "rel_w4_train", "rel_traineval", "rel_nothing"):
        with pytest.raises(SystemExit):
            with _stage(run_dir, "gates_" + phase) as st:
                rel_gates.phase(st, cfg, run_dir, phase)


def test_a_stub_stage_fails_with_its_report(tmp_path):
    run_dir, _ = RF.rel_run(tmp_path, stages=["w2_dist"])
    r = subprocess.run([sys.executable, "-m", "ci.stages.rel.stub", run_dir, "w2_dist"], cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=ROOT))
    assert r.returncode != 0 and "stub" in r.stderr
    rep = json.load(open(os.path.join(run_dir, "stages", "rel_w2_dist.json")))
    assert rep["status"] == "failed" and rep["stub"] is True


def test_summary_hook_lists_every_stage_and_never_raises(tmp_path):
    from ci.stages.rel import summary
    run_dir, _ = RF.rel_run(tmp_path, stages=["c0_synth", "w2_dist"])
    js, md = summary.hook(run_dir, dict(pipeline="rel_tables", stages=[dict(name="rel_c0_synth", status="failed")]))
    assert js["stages"]["c0_synth"]["status"] == "failed" and js["stages"]["w2_dist"]["status"] == "not run"
    assert any("w2_dist" in line for line in md)
    os.remove(os.path.join(run_dir, "config.resolved.yaml"))
    js, md = summary.hook(run_dir, dict(pipeline="rel_tables"))
    assert "error" in js


def test_compare_within_tolerance_hands_training_outputs_to_the_tolerance_function(tmp_path):
    import argparse
    from ci.__main__ import _compare_tolerance
    from ci.record import Record
    ra, rb = [Record.create(str(tmp_path / x), run_id=x, pipeline="rel_traineval") for x in ("a", "b")]
    for r in (ra, rb):
        r.data["stages"] = [dict(name="rel_w4_train", outputs=[dict(name="models", path="data/rel_w4/train/models.json")])]
    a = argparse.Namespace(a=str(tmp_path / "a"), b=str(tmp_path / "b"), protocol="")
    diffs = [dict(stage="rel_w4_train", output="models", a="1", b="2")]
    assert _compare_tolerance(a, ra, rb, diffs) == 1                          # the stub function says undecided
    assert rb.data["reproduces"]["verdict"] == "undecided"
    assert _compare_tolerance(a, ra, rb, diffs + [dict(stage="gates_pre", field="status", a="passed", b="failed")]) == 1
    assert rb.data["reproduces"]["verdict"] == "diverged"
    assert _compare_tolerance(a, ra, rb, []) == 0 and rb.data["reproduces"]["verdict"] == "exact"
