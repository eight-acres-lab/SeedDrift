"""The launchers refuse what they must, without a server or GitHub: a fake `gh` answers the check-run query, a fake
ssh records what it is asked to run, and server_launch.sh runs against a scratch CI root."""
import fcntl
import os
import re
import shutil
import subprocess

import pytest

from conftest import ROOT


def _run(cmd, cwd, env=None, stdin=None):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, stdin=stdin or subprocess.DEVNULL)


@pytest.fixture
def dev(tmp_path):
    """A developer clone with the launch scripts and one profile, pushed to a bare origin; fake gh and ssh on PATH."""
    origin, work, fake = tmp_path / "origin.git", tmp_path / "work", tmp_path / "bin"
    _run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], tmp_path)
    _run(["git", "clone", "-q", str(origin), str(work)], tmp_path)
    for rel in ("ci/launch.sh", "ci/server_launch.sh", "ci/publish.py"):
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), work / rel)
    (work / "ci" / "profiles" / "smoke").mkdir(parents=True)
    (work / "ci" / "profiles" / "smoke" / "config.yaml").write_text("tier: smoke\n")
    g = lambda *a: _run(["git", "-c", "user.email=ci@example.invalid", "-c", "user.name=ci", *a], work)
    g("checkout", "-q", "-b", "work")
    g("add", "-A")
    g("commit", "-q", "-m", "x")
    assert g("push", "-q", "origin", "work").returncode == 0
    fake.mkdir()
    (fake / "gh").write_text("#!/bin/sh\necho \"${FAKE_GH:-false}\"\n")
    (fake / "ssh").write_text("#!/bin/sh\necho \"$@\" >> \"$FAKE_LOG\"\ncat > /dev/null\nexit 0\n")
    for f in ("gh", "ssh"):
        os.chmod(fake / f, 0o755)
    env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ["PATH"], CI_SSH=str(fake / "ssh"), FAKE_LOG=str(tmp_path / "ssh.log"),
               CI_REPO="owner/name", GIT_AUTHOR_NAME="ci", GIT_AUTHOR_EMAIL="ci@example.invalid", GIT_COMMITTER_NAME="ci", GIT_COMMITTER_EMAIL="ci@example.invalid")
    return work, env, tmp_path / "ssh.log", g


def test_launch_refuses_unsafe_names_fails(dev):
    work, env, _, _ = dev
    for args in (["HEAD", "../smoke"], ["HEAD", "smoke", "--parent", "a;b"], ["status", "a/b"], ["publish", "x..y"], ["compare", "a", "$(id)"], ["HEAD", "smoke", "--force"]):
        r = _run(["bash", "ci/launch.sh", *args], work, env)
        assert r.returncode == 2, (args, r.stderr)


def test_launch_refuses_an_unpushed_commit_fails(dev):
    work, env, log, g = dev
    (work / "new.txt").write_text("y\n")
    g("add", "new.txt")
    g("commit", "-q", "-m", "local only")
    r = _run(["bash", "ci/launch.sh", "HEAD", "smoke"], work, dict(env, FAKE_GH="true"))
    assert r.returncode == 1 and "not on any branch of origin" in r.stderr
    assert not log.exists()                                         # nothing was sent to the server


def test_launch_refuses_failing_checks_and_missing_profiles_fails(dev):
    work, env, log, _ = dev
    r = _run(["bash", "ci/launch.sh", "HEAD", "smoke"], work, dict(env, FAKE_GH="false"))
    assert r.returncode == 1 and "has not passed" in r.stderr
    r = _run(["bash", "ci/launch.sh", "HEAD", "nosuch"], work, dict(env, FAKE_GH="true"))
    assert r.returncode == 1 and "no profile" in r.stderr
    assert not log.exists()


def test_launch_starts_a_pushed_commit_with_passing_checks(dev):
    work, env, log, g = dev
    r = _run(["bash", "ci/launch.sh", "HEAD", "smoke"], work, dict(env, FAKE_GH="true"))
    assert r.returncode == 0, r.stderr
    sha = g("rev-parse", "HEAD").stdout.strip()
    run_id = r.stdout.strip().splitlines()[-1]
    assert re.fullmatch(r"\d{8}T\d{6}-work-%s-smoke" % sha[:8], run_id)
    sent = log.read_text()
    assert "bash -s -- '%s'" % sha in sent and "'smoke' '%s' 'github' 'work'" % run_id in sent


# ---- server side ------------------------------------------------------------------------------------------------

@pytest.fixture
def server(dev, tmp_path):
    work, env, _, g = dev
    root = tmp_path / "ciroot"
    root.mkdir()
    _run(["git", "clone", "-q", "--bare", str(work), str(root / "repo.git")], tmp_path)
    sha = g("rev-parse", "HEAD").stdout.strip()
    tree = g("rev-parse", "HEAD^{tree}").stdout.strip()
    env = dict(env, MICROMAMBA=shutil.which("true"))
    return root, env, sha, tree, os.path.join(ROOT, "ci", "server_launch.sh")


def test_server_refuses_bad_arguments_fails(server):
    root, env, sha, tree, script = server
    assert _run(["bash", script, sha, tree, "smoke", "a;b", "github", "work"], root, env).returncode == 2
    assert _run(["bash", script, "abc", tree, "smoke", "r1", "github", "work"], root, env).returncode == 2
    r = _run(["bash", script, sha, "0" * 40, "smoke", "r1", "github", "work"], root, env)
    assert r.returncode == 1 and "tree" in r.stderr
    r = _run(["bash", script, sha, tree, "nosuch", "r1", "github", "work"], root, env)
    assert r.returncode == 1 and "no profile" in r.stderr
    (root / "runs" / "r1").mkdir(parents=True)
    r = _run(["bash", script, sha, tree, "smoke", "r1", "github", "work"], root, env)
    assert r.returncode == 1 and "exists" in r.stderr


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock(1) is not installed here (it is on the server and GitHub runners)")
def test_server_refuses_when_the_gpu_lock_is_held_fails(server):
    root, env, sha, tree, script = server
    (root / "locks").mkdir()
    with open(root / "locks" / "gpu.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = _run(["bash", script, sha, tree, "smoke", "r2", "github", "work"], root, env)
    assert r.returncode == 3 and "GPU lock" in r.stderr
    assert not (root / "wt" / "r2").exists()


def test_launch_refuses_a_hostile_branch_name_fails(dev):
    work, env, log, g = dev
    g("checkout", "-q", "-b", "x';touch${IFS}pwned;'y")
    (work / "b.txt").write_text("b\n")
    g("add", "b.txt")
    g("commit", "-q", "-m", "on a hostile branch only")
    assert g("push", "-q", "origin", "HEAD").returncode == 0
    r = _run(["bash", "ci/launch.sh", "HEAD", "smoke"], work, dict(env, FAKE_GH="true"))
    assert r.returncode == 2 and "unsafe branch name" in r.stderr
    assert not log.exists() and not (work / "pwned").exists()
