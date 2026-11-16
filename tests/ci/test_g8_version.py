from ci.gates import g8_version


def test_bootstrap_without_tag_passes(git_repo):
    assert g8_version.check(str(git_repo)).status == "pass"


def test_bootstrap_with_other_version_fails(git_repo):
    (git_repo / "ci" / "VERSION").write_text("0.2\n")
    git_repo.git("commit", "-qam", "v")
    assert g8_version.check(str(git_repo)).status == "fail"


def test_ci_change_without_bump_fails(git_repo):
    git_repo.git("tag", "ci-v0.1")
    (git_repo / "ci" / "gate.py").write_text("x = 1\n")
    git_repo.git("add", "-A")
    git_repo.git("commit", "-qm", "change")
    assert g8_version.check(str(git_repo)).status == "fail"
    (git_repo / "ci" / "VERSION").write_text("0.2\n")
    git_repo.git("commit", "-qam", "bump")
    assert g8_version.check(str(git_repo)).status == "pass"


def test_tag_not_an_ancestor_fails(git_repo):
    git_repo.git("checkout", "-q", "-b", "side")
    (git_repo / "x").write_text("1\n")
    git_repo.git("add", "-A")
    git_repo.git("commit", "-qm", "side")
    git_repo.git("tag", "ci-v0.1")
    git_repo.git("checkout", "-q", "main")
    assert g8_version.check(str(git_repo)).status == "fail"
