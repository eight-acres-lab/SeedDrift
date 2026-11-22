"""Helpers of the relational-model line's stages (CI environment).

  RelStage(run_dir, name)        the Stage of ci/stages/common.py for rel stages: it refuses a run that is not a rel
                                 kind or whose gate phase pre has not passed, and checks that this machine is the one
                                 the profile is written for (check_machine) before the stage does anything; the
                                 machine's facts go into the stage report
  machines()                     configs/rel/machines.json
  check_machine(cfg, gpus=None)  "this configuration belongs to this machine": rel.machine names a machine of
                                 configs/rel/machines.json, nvidia-smi shows exactly its GPU (name and total memory),
                                 the run's CI root is the machine's and the run directory lies under it, and
                                 rel.parallel does not exceed the machine's parallel_max; raises SystemExit otherwise
  sealed_set(cfg)                the SealedSet of this machine (seeddrift/train/rel/sealed.py, from the run's CI root)
  check_lineage(entries, cfg)    the exclusions by full NCBI lineage (gate G2's rules: bench/hard/EXCLUDE.tsv, the
                                 lists of configs/rel/sealed.json extra_exclusions and the profile's extra_exclusions)
                                 on [{id, taxid, accession?}]; raises SystemExit on any hit
  sealed_check(cfg, genomes, splits=None)
                                 the whole sealed check of the CI environment on [{id, taxid}]: check_ids, then
                                 check_lineage. Every rel stage that builds tables from genomes calls it; it needs the
                                 machine's sealed registry, so tables are built on machine i-1
  read_tableset(stage, run_dir, path, sha256=None)
                                 a table set of another run for stage `stage`, through handoff.from_run with the kinds
                                 the stage may read (configs/rel/kinds.yaml field reads): an evaluation-only stage may
                                 read ruler sets, a training stage never; raises SystemExit otherwise
  sealed_record(cfg, genomes, splits=None, allowed=...)
                                 sealed_check, returned as the record a table-set manifest carries (sealed_check:
                                 passed, ids, lineage; seeddrift/train/rel/handoff.py). Stages that read table sets
                                 take them through handoff.from_run, which refuses a set without a passed record.
"""
from __future__ import annotations

import json
import os
import subprocess

from ci import ROOT
from ci.stages.common import Stage
from ci.stages.rel import kinds
from seeddrift.train.rel import sealed as S

MACHINES = os.path.join(ROOT, "configs", "rel", "machines.json")


def machines(path=MACHINES) -> dict:
    with open(path) as f:
        return json.load(f)["machines"]


def gpu_query():
    """[(name, total MiB)] of the GPUs nvidia-smi sees, or None without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in out.strip().splitlines():
        name, _, mem = line.rpartition(",")
        rows.append((name.strip(), int(float(mem))))
    return rows


def check_machine(cfg, gpus=None, path=MACHINES) -> dict:
    rc = cfg.get("rel") or {}
    name = rc.get("machine")
    known = machines(path)
    if name not in known:
        raise SystemExit("rel.machine %r is not a machine of configs/rel/machines.json (%s)" % (name, ", ".join(sorted(known))))
    m = known[name]
    seen = gpu_query() if gpus is None else gpus
    problems = []
    if not seen:
        problems.append("no GPU visible (nvidia-smi)")
    elif len(seen) != 1 or seen[0] != (m["gpu_name"], int(m["gpu_mem_mb"])):
        problems.append("the GPUs here are %s, machine %s has one %s with %d MiB" % (seen, name, m["gpu_name"], m["gpu_mem_mb"]))
    root = cfg.get("ci_root")
    if not root or os.path.realpath(root) != os.path.realpath(m["ci_root"]):
        problems.append("the run's CI root is %r, machine %s's is %s" % (root, name, m["ci_root"]))
    run_dir = os.path.realpath(cfg.get("run_dir", ""))
    if not run_dir.startswith(os.path.realpath(m["ci_root"]) + os.sep):
        problems.append("the run directory %s is not under %s" % (run_dir, m["ci_root"]))
    par = rc.get("parallel", 1)
    if not isinstance(par, int) or isinstance(par, bool) or not 1 <= par <= int(m["parallel_max"]):
        problems.append("rel.parallel %r: between 1 and machine %s's parallel_max %s" % (par, name, m["parallel_max"]))
    if problems:
        raise SystemExit("this configuration is not for this machine: " + "; ".join(problems))
    return dict(machine=name, gpus=[list(g) for g in seen], ci_root=root, parallel=par)


class RelStage(Stage):
    """with RelStage(run_dir, "rel_c0_synth") as st: ...; requires gate phase pre (and any other phases named)."""

    def __init__(self, run_dir: str, name: str, key: str = "", requires=("pre",), gpus=None):
        super().__init__(run_dir, name, key, requires=tuple(dict.fromkeys(("pre",) + tuple(requires))))
        self._gpus = gpus

    def __enter__(self):
        super().__enter__()
        try:
            if not kinds.is_kind(self.config.get("pipeline")):
                raise SystemExit("stage %s runs in the pipelines rel_* only, not %r" % (self.name, self.config.get("pipeline")))
            kinds.stages_of(self.config)
            self.extra["machine"] = check_machine(self.config, self._gpus)
        except BaseException:
            self.__exit__(SystemExit, None, None)            # the stage report records the failure
            raise
        return self


def sealed_set(cfg, spec_path=S.SPEC) -> S.SealedSet:
    root = cfg.get("ci_root")
    if not root:
        raise SystemExit("the sealed check needs the run's CI root (ci_root, set by the server launcher)")
    try:
        return S.SealedSet.load(root, spec_path)
    except S.SealedError as err:
        raise SystemExit("sealed check: %s" % err)


def check_lineage(entries, cfg, tax=None, spec_path=S.SPEC, **g2_paths) -> dict:
    """g2_paths: exclude_path, panels_path, catalogue_path of gate G2 (tests only; the defaults are the repository's)."""
    from ci.gates import g2_exclusion
    from ci.taxonomy import Taxonomy
    with open(spec_path) as f:
        spec = json.load(f)
    extra = list(dict.fromkeys(list(spec.get("extra_exclusions") or []) + list(cfg.get("extra_exclusions") or [])))
    repo = lambda rel: rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
    tax = tax or Taxonomy(repo(cfg["taxdump"]))
    res = g2_exclusion.entries(list(entries), tax, extra_paths=[repo(x) for x in extra], **g2_paths)
    ev = res.evidence
    if res.status != "pass":
        raise SystemExit("lineage check failed: %s" % json.dumps(dict(problems=ev.get("problems"), hits=ev.get("hits")))[:2000])
    return dict(checked=ev.get("checked"), excluded_nodes=ev.get("excluded_nodes"), extra_exclusions=extra)


def sealed_check(cfg, genomes, splits=None, allowed=S.ALLOWED, sealed=None, tax=None, spec_path=S.SPEC, **g2_paths) -> dict:
    genomes = list(genomes)
    try:
        ids = S.check_ids([g["id"] for g in genomes], sealed or sealed_set(cfg, spec_path), splits, allowed)
    except S.SealedError as err:
        raise SystemExit(str(err))
    lin = check_lineage(genomes, cfg, tax, spec_path, **g2_paths) if genomes else dict(checked=0, note="no genomes")
    return dict(ids=ids, lineage=lin)


def read_tableset(stage, run_dir, path, sha256=None) -> dict:
    from seeddrift.train.rel import handoff
    try:
        return handoff.from_run(run_dir, path, sha256, kinds=kinds.reads(stage))
    except handoff.HandoffError as err:
        raise SystemExit("stage %s: %s" % (stage, err))


def sealed_record(cfg, genomes, splits=None, allowed=S.ALLOWED, **kw) -> dict:
    return dict(passed=True, **sealed_check(cfg, genomes, splits, allowed, **kw))
