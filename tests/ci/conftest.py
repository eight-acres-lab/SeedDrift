import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


@pytest.fixture(scope="session")
def tax():
    from ci.taxonomy import Taxonomy
    return Taxonomy(os.path.join(FIX, "mini_taxdump"))


class Repo:
    """A temporary git repository: str(repo), repo / "file" and repo.git(*args)."""

    def __init__(self, path, git):
        self.path, self.git = path, git

    def __truediv__(self, other):
        return self.path / other

    def __str__(self):
        return str(self.path)

    def __fspath__(self):
        return str(self.path)


@pytest.fixture
def git_repo(tmp_path):
    def git(*a):
        return subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q", "-b", "main")
    git("config", "user.email", "ci@example.invalid")
    git("config", "user.name", "ci test")
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci" / "VERSION").write_text("0.1\n")
    (tmp_path / "README").write_text("x\n")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    return Repo(tmp_path, git)
