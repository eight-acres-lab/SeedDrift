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
    tmp_path.git = git
    return tmp_path
