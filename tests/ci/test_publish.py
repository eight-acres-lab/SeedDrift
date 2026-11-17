import json
import os
import subprocess

import pytest

from conftest import ROOT

import importlib.util

_spec = importlib.util.spec_from_file_location("publish", os.path.join(ROOT, "ci", "publish.py"))
publish = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(publish)

SHA = "a" * 40
HOME_A, HOME_B = "/" + "home/someone/x", "/" + "Users/other/y"           # built here so that this file passes hygiene
NOTES = "strat" + "egy/charter.md"


def _files(**over):
    s = dict(run_id="r1", sha=SHA, profile="smoke", status="passed", ci_version="0.1", gates={"G1_corpus": "pass"})
    f = {"summary.json": json.dumps(s).encode(), "summary.md": b"# Run r1\n", "curve.csv": b"step,loss\n1,0.5\n"}
    f.update(over)
    return lambda name: f[name]


def test_clean_summary_is_collected_and_home_paths_are_replaced():
    files, s = publish.collect(_files(**{"summary.md": ("log at %s and %s\n" % (HOME_A, HOME_B)).encode()}), "r1")
    assert s["sha"] == SHA and b"someone" not in files["summary.md"] and files["summary.md"].count(b"<home>") == 2


def test_internal_notes_sealed_accessions_and_large_files_fail():
    with pytest.raises(publish.Refused):
        publish.collect(_files(**{"summary.md": ("see %s\n" % NOTES).encode()}), "r1")
    with pytest.raises(publish.Refused):
        publish.collect(_files(**{"summary.md": b"scored NC_010410.1\n"}), "r1")
    with pytest.raises(publish.Refused):
        publish.collect(_files(**{"curve.csv": b"x" * (publish.LIMIT + 1)}), "r1")
    with pytest.raises(publish.Refused):
        publish.collect(_files(**{"curve.csv": b"\0\1"}), "r1")


def test_summary_of_another_run_or_unsafe_id_fails():
    with pytest.raises(publish.Refused):
        publish.collect(_files(), "r2")
    with pytest.raises(publish.Refused):
        publish.collect(_files(), "../r1")
    with pytest.raises(publish.Refused):
        publish.collect(_files(**{"summary.json": b"{"}), "r1")


def test_publish_commits_on_a_run_branch(tmp_path, monkeypatch):
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    run = lambda *a, cwd=work: subprocess.run(["git", "-c", "user.email=ci@example.invalid", "-c", "user.name=ci", *a], cwd=cwd, capture_output=True, text=True, check=True)
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True, capture_output=True)
    (work / "runs-index").mkdir()
    (work / "runs-index" / "INDEX.tsv").write_text("run_id\tci_version\tsha\tprofile\tstatus\tsummary\n")
    run("add", "-A")
    run("commit", "-q", "-m", "x")
    run("push", "-q", "origin", "HEAD:main")
    sha = run("rev-parse", "HEAD").stdout.strip()
    monkeypatch.setattr(publish, "ROOT", str(work))
    for k in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(k, "ci")
    for k in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(k, "ci@example.invalid")
    files, s = publish.collect(_files(**{"summary.json": json.dumps(dict(run_id="r1", sha=sha, profile="smoke", status="passed", ci_version="0.1", gates={})).encode()}), "r1", root=ROOT)
    branch = publish.publish("r1", files, s)
    assert branch == "runs/r1"
    shown = subprocess.run(["git", "--git-dir", str(origin), "show", "runs/r1:runs-index/INDEX.tsv"], capture_output=True, text=True, check=True).stdout
    assert shown.splitlines()[-1].startswith("r1\t0.1\t%s\tsmoke\tpassed" % sha)
    with pytest.raises(publish.Refused):                                # published once only
        publish.publish("r1", files, s)
