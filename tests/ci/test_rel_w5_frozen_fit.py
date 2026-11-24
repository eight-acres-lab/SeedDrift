"""Exercise frozen fits against the real stage's output on tiny training tables."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ci.hashing import sha256_file
from ci.record import Record
from ci.stages.common import Stage
from ci.stages.rel import w5_baselines, w5_frozen_fit
from seeddrift.train.rel import handoff
from test_rel_w5_stage import TABLESET, baselines_run, tables_run


@pytest.fixture
def passed_fit(tmp_path, monkeypatch):
    source, manifest = tables_run(tmp_path)
    run, cfg = baselines_run(tmp_path, monkeypatch, source, manifest, chain=False)
    w5_baselines.main()
    with Stage(run, "gates_rel_w5_baselines", requires=("pre",)) as stage:
        w5_baselines.gate(stage, cfg, run)
    record = Record.open(run)
    record.update(git={"sha": "a" * 40})
    record.finish("passed")
    fit = next(s for s in record.data["stages"] if s["name"] == w5_frozen_fit.STAGE)
    outputs = {o["name"]: o["sha256"] for o in fit["outputs"]}
    spec = dict(run=run, commit=record.data["git"]["sha"], fit_tables=30, fit_seed=w5_baselines.cond.SEED + 1,
                train_tableset_sha256=manifest["sha256"], params_sha256=outputs["params"], results_sha256=outputs["results"])
    return spec


def test_loads_exact_bytes_and_training_provenance_without_refitting(passed_fit, monkeypatch):
    spec = passed_fit
    run = Path(spec["run"])
    wanted = (run / w5_baselines.OUT / "params.json").read_bytes()

    def refit_called(*args, **kwargs):
        pytest.fail("unexpected training fit")

    monkeypatch.setattr(w5_baselines.brun, "fit_all", refit_called)
    params, raw, origin = w5_frozen_fit.load_frozen_fit(spec)
    assert raw == wanted and params == json.loads(wanted)
    assert origin["train_tableset_sha256"] == spec["train_tableset_sha256"]
    assert origin["run_id"] == run.name and origin["folder"].endswith("/train")
    assert origin["commit"] == spec["commit"]
    assert origin["files_read"] > 0 and origin["fit_tables"] == 30
    assert origin["params_sha256"] == hashlib.sha256(raw).hexdigest()
    scored = json.loads((run / w5_baselines.OUT / "results.json").read_bytes())
    assert origin["original_items"] == scored["items"] and origin["original_layouts"] == scored["layouts"]
    assert all(row["cells"] > 0 and len(row["sha256"]) == 64 for row in origin["original_items"].values())

    for key in ("params_sha256", "results_sha256", "train_tableset_sha256", "fit_seed", "fit_tables"):
        with pytest.raises(ValueError):
            w5_frozen_fit.load_frozen_fit(dict(spec, **{key: 0}))

    record_path = run / "record.json"
    original = record_path.read_text()
    for changed in (dict(run_id="other-run"), dict(git={"sha": "b" * 40})):
        record_path.write_text(json.dumps(dict(json.loads(original), **changed)))
        with pytest.raises(ValueError, match="run ID or Git commit"):
            w5_frozen_fit.load_frozen_fit(spec)
    for target in ("run", "stage", "gate"):
        record = json.loads(original)
        if target == "stage":
            next(s for s in record["stages"] if s["name"] == w5_frozen_fit.STAGE)["status"] = "failed"
        elif target == "gate":
            record["gates"]["G4_rel_w5_baselines"] = "waived"
        else:
            record["status"] = "failed"
        record_path.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="must have passed"):
            w5_frozen_fit.load_frozen_fit(spec)


def test_refuses_mislabeled_fit_metadata_even_if_output_is_pinned(passed_fit):
    spec = passed_fit
    run = Path(spec["run"])
    record_path = run / "record.json"
    record = json.loads(record_path.read_bytes())
    stage = next(s for s in record["stages"] if s["name"] == w5_frozen_fit.STAGE)
    entry = next(o for o in stage["outputs"] if o["name"] == "results")
    path = run / entry["path"]
    doc = json.loads(path.read_bytes())
    doc["fit"]["folder"] = doc["heldout"]["folder"]
    path.write_text(json.dumps(doc))
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="separate training input"):
        w5_frozen_fit.load_frozen_fit(dict(spec, results_sha256=entry["sha256"]))


def test_scores_paired_heldout_tables_without_refitting(tmp_path, monkeypatch, passed_fit):
    src, manifest = tables_run(tmp_path / "new")
    manifest.update(files=[f for f in manifest["files"] if "/train/" not in f["path"]],
                    purpose="heldout", not_for_training=True, tables=12)
    manifest = handoff.finish(manifest)
    path = Path(src) / TABLESET
    path.write_text(json.dumps(manifest))
    record_path = Path(src) / "record.json"
    record = json.loads(record_path.read_bytes())
    record["stages"][0]["outputs"][0]["sha256"] = sha256_file(str(path), use_sidecar=False)
    record_path.write_text(json.dumps(record))
    run, cfg = baselines_run(tmp_path / "scoring", monkeypatch, src, manifest, frozen_fit=passed_fit, chain=False)
    monkeypatch.setattr(w5_baselines.brun, "fit_all", lambda *a, **kw: pytest.fail("scoring unexpectedly refitted"))
    w5_baselines.main()
    with Stage(run, "gates_rel_w5_baselines", requires=("pre",)) as audit:
        w5_baselines.gate(audit, cfg, run)
    out = Path(run) / w5_baselines.OUT
    original = Path(passed_fit["run"]) / w5_baselines.OUT
    assert (out / "params.json").read_bytes() == (original / "params.json").read_bytes()
    assert json.loads((out / "results.json").read_bytes())["fit"]["source"]["commit"] == passed_fit["commit"]
    with np.load(out / "cells.npz") as new, np.load(original / "cells.npz") as old:
        assert new.files == old.files
        for k in new.files:
            np.testing.assert_array_equal(new[k], old[k])
