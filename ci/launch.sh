#!/usr/bin/env bash
# Launch a T2/T3 run on the project server from a clean, pushed commit (run on the developer machine).
#
#   ci/launch.sh <commit-or-ref> <profile> [--parent RUN_ID]
#   ci/launch.sh status <RUN_ID>
#   ci/launch.sh reproduce <RUN_ID>     a new run of the same commit and profile, with RUN_ID as its parent
#   ci/launch.sh compare <RUN_A> <RUN_B> the reproduction verdict of B against A (written into B's record)
#   ci/launch.sh publish <RUN_ID>       the run's summary into git (ci/publish.py: branch runs/<RUN_ID>)
#   ci/launch.sh promote <RUN_ID>       print the catalogue lock of a passed select refresh run (redirect it into
#                                       configs/select/<spec>.lock.yaml and commit it as a new CI version)
#
# Refuses unless the commit is on a branch of the GitHub remote and every GitHub Actions run of both T0 and T1 on it
# passed. The server checks the commit out read-only into its own worktree and runs `python -m ci run` there (see
# ci/server_launch.sh); nothing on the server is edited by hand. Prints the run id.
#
# Configuration (environment): CI_SSH (the ssh command for the server, e.g. "ssh -p 22 user@host"),
# CI_REMOTE_ROOT (default /hy-tmp/seeddrift-ci), CI_REPO (owner/name on GitHub; default from the origin remote).
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"
: "${CI_SSH:?set CI_SSH to the ssh command of the server}"
CI_REMOTE_ROOT=${CI_REMOTE_ROOT:-/hy-tmp/seeddrift-ci}
CI_REPO=${CI_REPO:-$(git remote get-url origin | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')}
safe() { [[ "$1" =~ ^[A-Za-z0-9._-]+$ && "$1" != *..* ]] || { echo "refused: unsafe name '$1'" >&2; exit 2; }; }

if [ "${1:-}" = status ]; then
  safe "${2:?run id}"
  $CI_SSH "cd $CI_REMOTE_ROOT && envs/current/bin/python -c 'import sys; sys.path.insert(0, \"wt/$2\"); from ci.record import status_of; print(status_of(\"runs/$2\"))'" < /dev/null
  exit 0
fi
if [ "${1:-}" = compare ]; then
  a=${2:?run A}; b=${3:?run B}; safe "$a"; safe "$b"
  $CI_SSH "cd $CI_REMOTE_ROOT/wt/$b && $CI_REMOTE_ROOT/envs/current/bin/python -m ci compare $CI_REMOTE_ROOT/runs/$a $CI_REMOTE_ROOT/runs/$b" < /dev/null
  exit $?
fi
if [ "${1:-}" = promote ]; then
  safe "${2:?run id}"
  $CI_SSH "cd $CI_REMOTE_ROOT/wt/$2 && $CI_REMOTE_ROOT/envs/current/bin/python -m ci promote $CI_REMOTE_ROOT/runs/$2" < /dev/null
  exit $?
fi
if [ "${1:-}" = publish ]; then
  safe "${2:?run id}"
  exec python3 "$here/ci/publish.py" "$2"
fi
if [ "${1:-}" = reproduce ]; then
  safe "${2:?run id}"
  line=$($CI_SSH "cd $CI_REMOTE_ROOT && python3 -c 'import json; r=json.load(open(\"runs/$2/record.json\")); print(r[\"git\"][\"sha\"], r[\"profile\"])'" < /dev/null)
  read -r rsha rprofile <<< "$line"
  [[ "$rsha" =~ ^[0-9a-f]{40}$ ]] || { echo "refused: no commit in the record of $2" >&2; exit 1; }
  exec bash "$0" "$rsha" "$rprofile" --parent "$2"          # bash: the script need not be executable
fi
ref=${1:?commit or ref}; profile=${2:?profile}; shift 2
safe "$profile"
parent=""
while [ $# -gt 0 ]; do case "$1" in --parent) parent=$2; safe "$parent"; shift 2;; *) echo "unknown option $1" >&2; exit 2;; esac; done

git fetch -q origin --tags
sha=$(git rev-parse --verify "$ref^{commit}")
git cat-file -e "$sha:ci/profiles/$profile/config.yaml" 2>/dev/null || { echo "refused: no profile $profile in $sha" >&2; exit 1; }
branch=$(git branch -r --contains "$sha" --format='%(refname:short)' | { grep -v '/HEAD$' || true; } | head -n 1 | sed 's#^origin/##')
[ -n "$branch" ] || { echo "refused: $sha is not on any branch of origin (push it first)" >&2; exit 1; }
# the branch name goes into the server command line: only plain ref characters
[[ "$branch" =~ ^[A-Za-z0-9._/-]+$ && "$branch" != *..* ]] || { echo "refused: unsafe branch name '$branch'" >&2; exit 2; }
for job in "T0 static checks" "T1 tests and the ci-tiny pipeline twice"; do
  # one conclusion per line over all pages (--jq runs per page); at least one, and every one a success
  concl=$(gh api --paginate "repos/$CI_REPO/commits/$sha/check-runs" --jq ".check_runs[] | select(.name == \"$job\" and .app.slug == \"github-actions\") | .conclusion" 2>/dev/null || true)
  [ -n "$concl" ] && [ -z "$(printf '%s\n' "$concl" | grep -vx success || true)" ] || { echo "refused: '$job' has not passed on $sha" >&2; exit 1; }
done
tree=$(git rev-parse "$sha^{tree}")
run_id="$(date -u +%Y%m%dT%H%M%S)-$(echo "$branch" | tr '/' '-' | tr -cd 'A-Za-z0-9._-')-${sha:0:8}-$profile"

# make the commit and the ci-v* tags available on the server: fetch from GitHub there, or send a bundle over ssh
$CI_SSH "mkdir -p $CI_REMOTE_ROOT/tmp" < /dev/null
# (the release tags v* travel too: stage released recomputes the released screen from its tag)
if $CI_SSH "cd $CI_REMOTE_ROOT && git -C repo.git -c http.version=HTTP/1.1 fetch -q origin $sha 'refs/tags/ci-v*:refs/tags/ci-v*' 'refs/tags/v*:refs/tags/v*'" < /dev/null; then
  provenance=github
else
  echo "server could not fetch from GitHub; sending a bundle" >&2
  git update-ref "refs/ci-launch/$run_id" "$sha"
  tags=$(git tag -l 'ci-v*' 'v*')
  git bundle create - "refs/ci-launch/$run_id" $tags | $CI_SSH "cd $CI_REMOTE_ROOT && cat > tmp/$run_id.bundle && git -C repo.git fetch -q $CI_REMOTE_ROOT/tmp/$run_id.bundle 'refs/ci-launch/$run_id:refs/ci/$run_id' 'refs/tags/ci-v*:refs/tags/ci-v*' 'refs/tags/v*:refs/tags/v*' && rm tmp/$run_id.bundle"
  git update-ref -d "refs/ci-launch/$run_id"
  provenance=bundle
fi
$CI_SSH "cd $CI_REMOTE_ROOT && bash -s -- '$sha' '$tree' '$profile' '$run_id' '$provenance' '$branch' '$parent'" < ci/server_launch.sh
echo "$run_id"
