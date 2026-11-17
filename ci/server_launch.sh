#!/usr/bin/env bash
# Server side of ci/launch.sh (read from stdin over ssh; run in the CI root). Arguments: sha tree profile run_id
# provenance branch [parent]. Creates a read-only worktree at the commit and the locked environment if it does not
# exist yet (built at its final path, named by the lock file's hash, complete only with its marker; never changed), takes the GPU
# lock in the foreground (a busy GPU refuses the launch), and starts the run in the background; it reports success
# only once the run record exists.
set -euo pipefail
sha=$1; tree=$2; profile=$3; run_id=$4; provenance=$5; branch=$6; parent=${7:-}
for x in "$profile" "$run_id" "$provenance" "$branch" ${parent:+"$parent"}; do
  [[ "$x" =~ ^[A-Za-z0-9._/-]+$ && "$x" != *..* ]] || { echo "refused: unsafe argument '$x'" >&2; exit 2; }
done
[[ "$sha" =~ ^[0-9a-f]{40}$ && "$tree" =~ ^[0-9a-f]{40}$ ]] || { echo "refused: bad sha" >&2; exit 2; }
root=$(pwd)
mkdir -p wt envs runs tmp cache locks
MM=${MICROMAMBA:-$(command -v micromamba || echo /hy-tmp/hgtscan/bin/micromamba)}
[ -x "$MM" ] || { echo "refused: micromamba not found (set MICROMAMBA)" >&2; exit 1; }
[ "$(git -C repo.git rev-parse "$sha^{tree}")" = "$tree" ] || { echo "tree of $sha differs from the developer's" >&2; exit 1; }
git -C repo.git cat-file -e "$sha:ci/profiles/$profile/config.yaml" || { echo "refused: no profile $profile in $sha" >&2; exit 1; }
[ ! -e "runs/$run_id" ] && [ ! -e "wt/$run_id" ] || { echo "refused: run id $run_id exists" >&2; exit 1; }

exec 9> locks/gpu.lock
flock -n 9 || { echo "refused: another run holds the GPU lock" >&2; exit 3; }

cleanup() { chmod -R u+w "wt/$run_id" 2>/dev/null || true; git -C repo.git worktree remove --force "$root/wt/$run_id" 2>/dev/null || true; }
trap 'cleanup' ERR
git -C repo.git worktree add -q --detach "$root/wt/$run_id" "$sha"
[ -z "$(git -C "wt/$run_id" status --porcelain --untracked-files=all --ignored)" ] || { echo "worktree not clean" >&2; exit 1; }
# a conda environment cannot be moved (paths such as the OpenSSL certificate file are compiled in), so it is built at
# its final path; .ci-env-complete is written only after a complete build, and a prefix without it is built again
build_env() {   # lock file, prefix
  if [ ! -f "$2/.ci-env-complete" ]; then
    T=${CI_CONDA_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud}
    rm -rf "$2"
    MAMBA_ROOT_PREFIX="$root/mamba" "$MM" create -y -q -p "$root/$2" -f "$1" --override-channels -c "$T/conda-forge" -c "$T/bioconda" > "tmp/env-$(basename "$2").log" 2>&1
    sha256sum "$1" > "$2/.ci-env-complete"
  fi
}
lock="wt/$run_id/ci/envs/ci.lock.yml"
h=$(sha256sum "$lock" | cut -c1-8)
env="envs/sd-ci-$h"
build_env "$lock" "$env"
# the released screen's own environment, for profiles that recompute it (stage released)
rlock="wt/$run_id/ci/envs/released-v0.1.0.lock.yml"
if grep -q "^released:" "wt/$run_id/ci/profiles/$profile/config.yaml"; then
  build_env "$rlock" "envs/sd-released-$(sha256sum "$rlock" | cut -c1-8)"
fi
ln -sfn "sd-ci-$h" envs/current
chmod -R a-w "wt/$run_id"
trap - ERR

export TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache" PIP_CACHE_DIR="$root/cache/pip" CI_PROVENANCE="$provenance" CI_BRANCH="$branch"
export CI_EXPECT_SHA="$sha" CI_ROOT="$root" MAMBA_EXE="$MM" PATH="$root/$env/bin:$PATH" CONDA_PREFIX="$root/$env"
cd "wt/$run_id"
# the background run inherits descriptor 9 and so holds the GPU lock until it ends
setsid nohup "$root/$env/bin/python" -m ci run --profile "$profile" --run-dir "$root/runs/$run_id" ${parent:+--parent "$parent"} \
  > "$root/runs/$run_id.launch.log" 2>&1 < /dev/null &
for _ in $(seq 1 60); do
  [ -s "$root/runs/$run_id/record.json" ] && { echo "started $run_id"; exit 0; }
  kill -0 $! 2>/dev/null || break
  sleep 2
done
echo "the run did not start; see runs/$run_id.launch.log" >&2
tail -n 20 "$root/runs/$run_id.launch.log" >&2 || true
exit 1
