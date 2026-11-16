import json
import os

import pytest

from ci.record import Record, stage_report, status_of


def test_create_update_finish(tmp_path):
    r = Record.create(str(tmp_path / "run"), run_id="run")
    r.update(metrics={"a": 1})
    stage_report(str(tmp_path / "run"), "corpus", "k", 0.0, [{"name": "m", "path": "x", "sha256": "0" * 64, "bytes": 1}])
    r.finish("passed")
    d = json.load(open(tmp_path / "run" / "record.json"))
    assert d["status"] == "passed" and d["stages"][0]["name"] == "corpus" and d["ended"]
    with pytest.raises(FileExistsError):
        Record.create(str(tmp_path / "run"), run_id="run")


def test_stale_heartbeat_fails_as_interrupted(tmp_path):
    r = Record.create(str(tmp_path / "run"), run_id="run")
    r.update(heartbeat="2000-01-01T00:00:00Z")
    assert status_of(str(tmp_path / "run")) == "interrupted"
    assert os.path.exists(tmp_path / "run" / "record.json")


def test_stage_extra_cannot_overwrite_status(tmp_path):
    with pytest.raises(ValueError):
        stage_report(str(tmp_path), "train", "", 0.0, [], status="diverged")
