"""ci publish: bring the summary of a server run back into git (run on the developer machine; standard library only).

  ci/launch.sh publish RUN_ID        (or: python3 ci/publish.py RUN_ID)

1. fetch runs/<RUN_ID>/{summary.json, summary.md, curve.csv} from the server over ssh (CI_SSH, CI_REMOTE_ROOT);
2. check them: each file and all together at most 200 KB, nothing else; summary.json parses and names this run, a
   commit, a profile and a status; the run's commit is on a branch of origin; home paths are replaced by <home>;
   any other match of ci/denylist.txt or any sealed accession (bench/hard/results/sealed.tsv) refuses the publish;
3. commit them on a new branch runs/<RUN_ID>, based on the run's commit, as runs-index/<RUN_ID>/ with one more row in
   runs-index/INDEX.tsv, and push that branch. A run already published (the branch exists on origin) is refused.
A failed publish changes nothing on the server and can be retried.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ("summary.json", "summary.md", "curve.csv")
LIMIT = 200_000
HOME_RX = re.compile(r"/(Users|home)/[A-Za-z0-9_.-]+")
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class Refused(Exception):
    pass


def denylist(root=ROOT):
    with open(os.path.join(root, "ci", "denylist.txt")) as f:
        return [re.compile(l.strip()) for l in f if l.strip() and not l.startswith("#")]


def sealed_pattern(root=ROOT):
    with open(os.path.join(root, "bench", "hard", "results", "sealed.tsv")) as f:
        stems = sorted({r["accession"].split(".")[0] for r in csv.DictReader(f, delimiter="\t")})
    return re.compile(r"(?<![A-Za-z0-9])(%s)(\.\d+)?(?![0-9])" % "|".join(re.escape(s) for s in stems), re.IGNORECASE)


def scrub(name: str, data: bytes, deny, sealed) -> bytes:
    """The file with home paths replaced; raises Refused on anything else that must not be published."""
    if len(data) > LIMIT:
        raise Refused("%s is %d bytes (limit %d)" % (name, len(data), LIMIT))
    if b"\0" in data:
        raise Refused("%s is binary" % name)
    text = HOME_RX.sub("<home>", data.decode("utf-8"))
    for rx in deny:
        if rx.search(text):
            raise Refused("%s matches the denylist pattern %s" % (name, rx.pattern))
    m = sealed.search(text)
    if m:
        raise Refused("%s names the sealed accession %s" % (name, m.group(0)))
    return text.encode("utf-8")


def check_summary(data: bytes, run_id: str) -> dict:
    try:
        s = json.loads(data)
    except ValueError as e:
        raise Refused("summary.json does not parse: %s" % e)
    for k in ("run_id", "sha", "profile", "status", "ci_version", "gates"):
        if k not in s:
            raise Refused("summary.json has no %s" % k)
    if s["run_id"] != run_id:
        raise Refused("summary.json is of run %s, not %s" % (s["run_id"], run_id))
    if not re.fullmatch(r"[0-9a-f]{40}", s["sha"]):
        raise Refused("summary.json names no full commit")
    for k in ("profile", "status"):                              # these go into a row of the public index
        if not isinstance(s[k], str) or not SAFE_ID.match(s[k]):
            raise Refused("summary.json field %s is not a plain name: %r" % (k, s[k]))
    if not re.fullmatch(r"\d+\.\d+", str(s["ci_version"])):
        raise Refused("summary.json ci_version is not MAJOR.MINOR")
    return s


def collect(fetch, run_id, root=ROOT):
    """{file: bytes} checked and scrubbed; fetch(name) returns the bytes of runs/<run_id>/<name>."""
    if not SAFE_ID.match(run_id) or ".." in run_id:
        raise Refused("unsafe run id %r" % run_id)
    deny, sealed = denylist(root), sealed_pattern(root)
    out = {n: scrub(n, fetch(n), deny, sealed) for n in FILES}
    if sum(len(v) for v in out.values()) > LIMIT:
        raise Refused("the summary files together exceed %d bytes" % LIMIT)
    return out, check_summary(out["summary.json"], run_id)


def _git(*args, cwd=None, check=True):
    return subprocess.run(["git", *args], cwd=cwd or ROOT, capture_output=True, text=True, check=check)


def publish(run_id, files, summary, push=True, remote="origin"):
    sha = summary["sha"]
    if _git("ls-remote", "--exit-code", "--heads", remote, "runs/" + run_id, check=False).returncode == 0:
        raise Refused("runs/%s already exists on %s" % (run_id, remote))
    _git("fetch", "-q", remote)
    if not _git("branch", "-r", "--contains", sha, check=False).stdout.strip():
        raise Refused("the run's commit %s is not on a branch of %s" % (sha[:12], remote))
    branch = "runs/" + run_id
    _git("branch", "-D", branch, check=False)                       # a local leftover of a failed attempt
    with tempfile.TemporaryDirectory() as tmp:
        wt = os.path.join(tmp, "wt")
        _git("worktree", "add", "-q", "-b", branch, wt, sha)
        try:
            d = os.path.join(wt, "runs-index", run_id)
            os.makedirs(d)
            for n, b in files.items():
                with open(os.path.join(d, n), "wb") as f:
                    f.write(b)
            idx = os.path.join(wt, "runs-index", "INDEX.tsv")
            with open(idx, "a") as f:
                f.write("\t".join([run_id, str(summary["ci_version"]), sha, summary["profile"], summary["status"], "runs-index/%s/summary.md" % run_id]) + "\n")
            _git("add", "runs-index", cwd=wt)
            msg = "run summary %s (%s, %s)\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" % (run_id, summary["profile"], summary["status"])
            _git("commit", "-q", "-m", msg, cwd=wt)
            if push:
                _git("push", "-q", remote, branch, cwd=wt)
        finally:
            _git("worktree", "remove", "--force", wt, check=False)
    return branch


def main(argv):
    if len(argv) != 1:
        raise SystemExit(__doc__)
    run_id = argv[0]
    ssh = shlex.split(os.environ.get("CI_SSH") or "")
    if not ssh:
        raise SystemExit("set CI_SSH to the ssh command of the server")
    root = os.environ.get("CI_REMOTE_ROOT", "/hy-tmp/seeddrift-ci")

    def fetch(name):
        r = subprocess.run(ssh + ["cat %s/runs/%s/%s" % (shlex.quote(root), run_id, name)], stdin=subprocess.DEVNULL, capture_output=True)
        if r.returncode != 0:
            raise Refused("could not read %s of run %s on the server" % (name, run_id))
        return r.stdout
    try:
        files, summary = collect(fetch, run_id)
        branch = publish(run_id, files, summary)
    except Refused as e:
        raise SystemExit("refused: %s" % e)
    print("published %s on branch %s" % (run_id, branch))


if __name__ == "__main__":
    main(sys.argv[1:])
