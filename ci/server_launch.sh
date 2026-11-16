#!/usr/bin/env bash
# Server side of ci/launch.sh (read from stdin over ssh; run in the CI root). Arguments: sha tree profile run_id
# provenance [parent]. Creates a read-only worktree at the commit, the locked environment if it does not exist yet
# (named by the lock file's hash; never changed in place), and starts the run in the background under the GPU lock.
set -euo pipefail
sha=$1; tree=$2; profile=$3; run_id=$4; provenance=$5; parent=${6:-}
root=$(pwd)
[ "$(git -C repo.git rev-parse "$sha^{tree}")" = "$tree" ] || { echo "tree of $sha differs from the developer's" >&2; exit 1; }
git -C repo.git worktree add -q --detach "wt/$run_id" "$sha"
[ -z "$(git -C "wt/$run_id" status --porcelain --untracked-files=all)" ] || { echo "worktree not clean" >&2; exit 1; }
chmod -R a-w "wt/$run_id"
lock="wt/$run_id/ci/envs/ci.lock.yml"
h=$(sha256sum "$lock" | cut -c1-8)
env="envs/sd-ci-$h"
if [ ! -x "$env/bin/python" ]; then
  T=${CI_CONDA_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud}
  MAMBA_ROOT_PREFIX="$root/mamba" "${MICROMAMBA:-micromamba}" create -y -q -p "$env" -f "$lock" --override-channels -c "$T/conda-forge" -c "$T/bioconda" > "tmp/env-$h.log" 2>&1
fi
ln -sfn "sd-ci-$h" envs/current
mkdir -p runs tmp cache
export TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache" PIP_CACHE_DIR="$root/cache/pip" CI_PROVENANCE="$provenance" CONDA_PREFIX="$root/$env"
export PATH="$root/$env/bin:$PATH" MAMBA_EXE="${MICROMAMBA:-$(command -v micromamba || true)}"
cd "wt/$run_id"
setsid nohup flock -n "$root/locks/gpu.lock" "$root/$env/bin/python" -m ci run --profile "$profile" --run-dir "$root/runs/$run_id" \
  ${parent:+--parent "$parent"} > "$root/runs/$run_id.launch.log" 2>&1 < /dev/null &
echo "started $run_id"
