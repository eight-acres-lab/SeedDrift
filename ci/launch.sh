#!/usr/bin/env bash
# Launch a T2/T3 run on the project server from a clean, pushed commit (run on the developer machine).
#
#   ci/launch.sh <commit-or-ref> <profile> [--parent RUN_ID]
#
# Refuses unless: the commit is on a branch of the GitHub remote, and the T0 job of the GitHub workflow passed on it.
# The server then checks the commit out read-only into its own worktree and runs `python -m ci run` there (see
# ci/server_launch.sh); nothing on the server is edited by hand. Prints the run id; `ci/launch.sh status RUN_ID`
# shows its state.
#
# Configuration (environment): CI_SSH (the ssh command for the server, e.g. "ssh -p 22 user@host"),
# CI_REMOTE_ROOT (default /hy-tmp/seeddrift-ci), CI_REPO (owner/name on GitHub; default from the origin remote).
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"
: "${CI_SSH:?set CI_SSH to the ssh command of the server}"
CI_REMOTE_ROOT=${CI_REMOTE_ROOT:-/hy-tmp/seeddrift-ci}
CI_REPO=${CI_REPO:-$(git remote get-url origin | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')}

if [ "${1:-}" = status ]; then
  $CI_SSH "cd $CI_REMOTE_ROOT && envs/current/bin/python -c 'import sys; sys.path.insert(0, \"wt/$2\"); from ci.record import status_of; print(status_of(\"runs/$2\"))'" < /dev/null
  exit 0
fi
ref=${1:?commit or ref}; profile=${2:?profile}; shift 2
parent=""
while [ $# -gt 0 ]; do case "$1" in --parent) parent=$2; shift 2;; *) echo "unknown option $1" >&2; exit 2;; esac; done

git fetch -q origin
sha=$(git rev-parse --verify "$ref^{commit}")
if [ -z "$(git branch -r --contains "$sha")" ]; then
  echo "refused: $sha is not on any branch of origin (push it first)" >&2; exit 1
fi
t0=$(gh api "repos/$CI_REPO/commits/$sha/check-runs" --jq '[.check_runs[] | select(.name == "T0 static checks")] | .[0].conclusion' 2>/dev/null || true)
if [ "$t0" != "success" ]; then
  echo "refused: T0 has not passed on $sha (found: ${t0:-none})" >&2; exit 1
fi
tree=$(git rev-parse "$sha^{tree}")
branch=$(git branch -r --contains "$sha" | head -n 1 | sed 's#^ *origin/##')
run_id="$(date -u +%Y%m%dT%H%M%S)-$(echo "$branch" | tr '/' '-' | tr -cd 'A-Za-z0-9._-')-${sha:0:8}-$profile"

# make the commit available on the server: fetch from GitHub there, or send a bundle over ssh
if ! $CI_SSH "cd $CI_REMOTE_ROOT && git -C repo.git -c http.version=HTTP/1.1 fetch -q origin $sha" < /dev/null; then
  echo "server could not fetch from GitHub; sending a bundle" >&2
  base=$($CI_SSH "git -C $CI_REMOTE_ROOT/repo.git rev-parse --verify -q main 2>/dev/null || true" < /dev/null)
  if [ -n "$base" ] && git cat-file -e "$base" 2>/dev/null; then range="$base..$sha"; else range="$sha"; fi
  git bundle create - "$range" | $CI_SSH "cd $CI_REMOTE_ROOT && cat > tmp/launch.bundle && git -C repo.git fetch -q $CI_REMOTE_ROOT/tmp/launch.bundle '+$sha:refs/ci/$run_id' && rm tmp/launch.bundle"
  provenance=bundle
else
  provenance=github
fi
$CI_SSH "cd $CI_REMOTE_ROOT && bash -s -- '$sha' '$tree' '$profile' '$run_id' '$provenance' '$parent'" < ci/server_launch.sh
echo "$run_id"
