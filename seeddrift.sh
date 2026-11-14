#!/usr/bin/env bash
# SeedDrift: one entry point to install, fetch data, run and stop the services, and open the web page.
#
#   ./seeddrift.sh up                  install + data + start      (a fresh GPU machine, one command)
#   ./seeddrift.sh install             micromamba environments for the three services
#   ./seeddrift.sh data [names]        fetch from the original sources and index (resumable)
#   ./seeddrift.sh start|stop|restart|status [api|tools|esmfold]
#   ./seeddrift.sh logs [service]
#   ./seeddrift.sh test                unit tests
#   ./seeddrift.sh web                 the web page on this computer (http://localhost:8700)
#   ./seeddrift.sh tunnel -p PORT user@host   forward the remote API to localhost:8600
set -euo pipefail
SD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SD_ROOT
CONF="$SD_ROOT/deploy/seeddrift.env"
[ -f "$CONF" ] || cp "$SD_ROOT/deploy/seeddrift.env.example" "$CONF"
set -a; eval "$(sed -e 's/#.*$//' "$CONF" | grep -E '^[A-Za-z_][A-Za-z0-9_]*=')"; set +a
: "${SEEDDRIFT_DATA:=$SD_ROOT/store}" "${SD_MAMBA_ROOT:=$SD_ROOT/.mamba}" "${SD_API_PORT:=8600}"
: "${SD_TOOLS_PORT:=8601}" "${SD_ESMFOLD_PORT:=8602}" "${SD_API_HOST:=127.0.0.1}" "${SD_ESMFOLD_TORCH:=conda}"
RUN="$SD_ROOT/.run"; mkdir -p "$RUN"
export PYTHONPATH="$SD_ROOT${PYTHONPATH:+:$PYTHONPATH}" SEEDDRIFT_DATA
ENVS="$SD_MAMBA_ROOT/envs"
say() { printf '\033[32m[seeddrift]\033[0m %s\n' "$*"; }

mamba_exe() {
  if [ -n "${MAMBA_EXE:-}" ] && [ -x "$MAMBA_EXE" ]; then echo "$MAMBA_EXE"; return; fi
  if command -v micromamba >/dev/null; then command -v micromamba; return; fi
  local m="$SD_MAMBA_ROOT/bin/micromamba"
  if [ ! -x "$m" ]; then
    local os arch; os=$(uname -s); arch=$(uname -m)
    case "$os-$arch" in
      Linux-x86_64) p=linux-64;; Linux-aarch64) p=linux-aarch64;; Darwin-arm64) p=osx-arm64;; Darwin-x86_64) p=osx-64;;
      *) echo "unsupported platform $os-$arch" >&2; exit 1;;
    esac
    say "downloading micromamba ($p)"
    mkdir -p "$SD_MAMBA_ROOT/bin"
    curl -fsSL "https://micro.mamba.pm/api/micromamba/$p/latest" | tar -xj -C "$SD_MAMBA_ROOT" bin/micromamba
  fi
  echo "$m"
}

channels_args() {
  if [ "${SD_CONDA_MIRROR:-}" = tuna ]; then
    local t=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    echo "--override-channels -c $t/conda-forge -c $t/bioconda"
  fi
}

install() {
  local mm; mm=$(mamba_exe)
  for e in app tools; do
    if [ -x "$ENVS/sd-$e/bin/python" ]; then say "env sd-$e present"; continue; fi
    say "creating env sd-$e"
    # shellcheck disable=SC2046
    MAMBA_ROOT_PREFIX="$SD_MAMBA_ROOT" "$mm" create -y -f "$SD_ROOT/deploy/envs/$e.yml" $(channels_args)
  done
  if [ "$SD_ESMFOLD_TORCH" = system ]; then
    say "ESMFold on the system python3 ($(python3 -c 'import torch; print("torch", torch.__version__)'))"
    python3 -m pip install -q --target "$SD_ROOT/.pylibs/esmfold" transformers==4.49.0 "fastapi>=0.110" "uvicorn>=0.29"
  elif [ ! -x "$ENVS/sd-esmfold/bin/python" ]; then
    say "creating env sd-esmfold (torch for CUDA 12)"
    # shellcheck disable=SC2046
    MAMBA_ROOT_PREFIX="$SD_MAMBA_ROOT" "$mm" create -y -f "$SD_ROOT/deploy/envs/esmfold.yml" $(channels_args)
  fi
  say "environments ready"
}

tool_env() {   # binaries of the tools env
  export SEEDDRIFT_DIAMOND="${SEEDDRIFT_DIAMOND:-$ENVS/sd-tools/bin/diamond}"
  export SEEDDRIFT_FOLDSEEK="${SEEDDRIFT_FOLDSEEK:-$ENVS/sd-tools/bin/foldseek}"
  export SEEDDRIFT_NCOILS="${SEEDDRIFT_NCOILS:-$ENVS/sd-tools/bin/ncoils}"
  export SEEDDRIFT_MAFFT="${SEEDDRIFT_MAFFT:-$ENVS/sd-tools/bin/mafft}"
  export SEEDDRIFT_FASTTREE="${SEEDDRIFT_FASTTREE:-$ENVS/sd-tools/bin/FastTree}"
  export SEEDDRIFT_METAEUK="${SEEDDRIFT_METAEUK:-$ENVS/sd-tools/bin/metaeuk}"
}

data() {
  tool_env
  "$ENVS/sd-tools/bin/python" -m seeddrift data fetch "$@"
  "$ENVS/sd-tools/bin/python" -m seeddrift data build "$@"
  "$ENVS/sd-tools/bin/python" -m seeddrift data status
}

pyfor() {
  case $1 in
    api) echo "$ENVS/sd-app/bin/python";;
    tools) echo "$ENVS/sd-tools/bin/python";;
    esmfold) if [ "$SD_ESMFOLD_TORCH" = system ]; then echo python3; else echo "$ENVS/sd-esmfold/bin/python"; fi;;
  esac
}

running() { [ -f "$RUN/$1.pid" ] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }

start_one() {
  local s=$1 port host=127.0.0.1
  if running "$s"; then say "$s already running (pid $(cat "$RUN/$s.pid"))"; return; fi
  case $s in
    api) port=$SD_API_PORT; host=$SD_API_HOST
         export SEEDDRIFT_TOOLS_URL="http://127.0.0.1:$SD_TOOLS_PORT" SEEDDRIFT_ESMFOLD_URL="http://127.0.0.1:$SD_ESMFOLD_PORT";;
    tools) port=$SD_TOOLS_PORT; tool_env;;
    esmfold) port=$SD_ESMFOLD_PORT
             [ "$SD_ESMFOLD_TORCH" = system ] && export PYTHONPATH="$SD_ROOT/.pylibs/esmfold:$PYTHONPATH";;
  esac
  ( cd "$SD_ROOT" && nohup "$(pyfor "$s")" -m uvicorn "services.$s.app:app" --host "$host" --port "$port" \
      > "$RUN/$s.log" 2>&1 < /dev/null & echo $! > "$RUN/$s.pid" )
  say "$s started on $host:$port (log: .run/$s.log)"
}

stop_one() {
  if running "$1"; then kill "$(cat "$RUN/$1.pid")"; say "$1 stopped"; else say "$1 not running"; fi
  rm -f "$RUN/$1.pid"
}

status() {
  for s in tools esmfold api; do
    if running "$s"; then say "$s: running (pid $(cat "$RUN/$s.pid"))"; else say "$s: stopped"; fi
  done
  curl -fsS "http://127.0.0.1:$SD_API_PORT/api/health" 2>/dev/null | "$(pyfor api)" -m json.tool 2>/dev/null || true
}

svc_list() { if [ $# -gt 0 ]; then echo "$@"; else echo tools esmfold api; fi; }

cmd=${1:-help}; shift || true
case $cmd in
  install) install;;
  data) data "$@";;
  start) for s in $(svc_list "$@"); do start_one "$s"; done;;
  stop) for s in $(svc_list "$@"); do stop_one "$s"; done;;
  restart) for s in $(svc_list "$@"); do stop_one "$s"; sleep 1; start_one "$s"; done;;
  status) status;;
  logs) tail -n 50 -f "$RUN/${1:-api}.log";;
  up) install; data; for s in tools esmfold api; do start_one "$s"; done; status;;
  test) cd "$SD_ROOT" && "$(pyfor api)" -m unittest discover -s tests;;
  web) cd "$SD_ROOT" && exec python3 -m seeddrift web "$@";;
  tunnel) say "forwarding localhost:$SD_API_PORT -> remote 127.0.0.1:$SD_API_PORT (Ctrl-C to stop)"
          exec ssh -N -L "$SD_API_PORT:127.0.0.1:$SD_API_PORT" "$@";;
  *) sed -n '2,11p' "$0";;
esac
