from ci.gates import g6_clean


def test_clean_commit_passes(git_repo):
    sha = git_repo.git("rev-parse", "HEAD")
    assert g6_clean.check(str(git_repo), sha).status == "pass"


def test_modified_or_untracked_file_fails(git_repo):
    sha = git_repo.git("rev-parse", "HEAD")
    (git_repo / "README").write_text("changed\n")
    assert g6_clean.check(str(git_repo), sha).status == "fail"
    git_repo.git("checkout", "README")
    (git_repo / "new.py").write_text("x = 1\n")
    assert g6_clean.check(str(git_repo), sha).status == "fail"


def test_other_commit_fails(git_repo):
    sha = git_repo.git("rev-parse", "HEAD")
    (git_repo / "README").write_text("v2\n")
    git_repo.git("commit", "-qam", "v2")
    assert g6_clean.check(str(git_repo), sha).status == "fail"


def test_worktree_outside_the_ci_root_fails(git_repo, tmp_path_factory):
    sha = git_repo.git("rev-parse", "HEAD")
    root = tmp_path_factory.mktemp("ciroot")
    assert g6_clean.check(str(git_repo), sha, ci_root=str(root)).status == "fail"
