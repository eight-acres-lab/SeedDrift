"""Fixtures of the relational-model line for every package's tests (owned by the coordinator). Test side only: no
train-side module imports this file (tests/ci/test_rel_c0_imports.py).

  synthetic(n=4, seed=1, **params)      tables of the synthetic generator (ci/stages/rel/synth.py)
  sealed(test_ids=..., chromosomes=...) a SealedSet made from given parts
  gpus(machine)                         what nvidia-smi reports on a machine of configs/rel/machines.json
  rel_run(tmp_path, kind, stages, ...)  a run directory of a rel kind: config.resolved.yaml (CI root, machine, stages),
                                        gates/pre.ok, an empty record; returns (run_dir, cfg)
  write_tableset(run_dir, stage, ...)   a valid table-set manifest with one shard file (and its meta), for gate tests
"""
from __future__ import annotations

import json
import os

import yaml

from ci.stages.rel import common, kinds, synth
from seeddrift.train.rel import handoff, sealed as S


def synthetic(n=4, seed=1, **params):
    return list(synth.tables(n, seed, **params))


def sealed(test_ids=("medium_test_GCF_000000001.1",), chromosomes=("NC_000913.3",), aliases=()):
    return S.SealedSet.from_parts(test_ids, aliases, chromosomes)


def gpus(machine="i1"):
    m = common.machines()[machine]
    return [(m["gpu_name"], int(m["gpu_mem_mb"]))]


def rel_run(tmp_path, kind="rel_tables", stages=("c0_synth",), machine="i1", run_id="20261122T000000-exp-rel-00000000-test", **extra):
    """A run directory under a scratch CI root that stands in for the machine's (cfg ci_root is the machine's path, so
    check_machine passes only where a test also gives run_dir under it; tests of other checks pass gpus and patch)."""
    root = tmp_path / "ci-root"
    run_dir = root / "runs" / run_id
    (run_dir / "gates").mkdir(parents=True)
    (run_dir / "stages").mkdir()
    cfg = dict(pipeline=kind, tier="tiny", seed=1, corpus_spec="configs/corpus/ci-tiny.yaml", taxdump="tests/ci/fixtures/tiny/taxdump",
               rel=dict(machine=machine, stages=list(stages)), train=dict(gpu_lease=False, device="cpu"), ci_root=str(root), run_dir=str(run_dir),
               git=dict(sha="0" * 40, branch="exp/rel"), config_sha256="0" * 64)
    cfg.update(extra)
    kinds.stages_of(cfg)
    with open(run_dir / "config.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    (run_dir / "gates" / "pre.ok").write_text(json.dumps(dict(phase="pre", passed=True, inputs="x")))
    with open(run_dir / "record.json", "w") as f:
        json.dump(dict(run_id=run_id, status="running", stages=[], pipeline=kind), f)
    return str(run_dir), cfg


def write_tableset(run_dir, stage="c0_synth", name="t", kind="synthetic", genomes=(), package="C0"):
    out = kinds.stage(stage)["output"]
    folder = os.path.dirname(os.path.join(run_dir, out))
    os.makedirs(folder, exist_ok=True)
    files = []
    for role, fn, body in (("shard", "shard-00000.bin", b"\x01\x02\x03"), ("meta", "meta.json", b"{}")):
        p = os.path.join(folder, fn)
        with open(p, "wb") as f:
            f.write(body)
        files.append(dict(path=os.path.relpath(p, run_dir), sha256=handoff._file_sha(p), bytes=len(body), role=role))
    doc = dict(format="rel-tableset", format_version=1, name=name, kind=kind, api_version=1, table_version="test",
               coverage=dict(groups=[], scenarios=[], not_covered=["everything: a test fixture"], note="test"),
               tables=1, files=files, genomes=list(genomes),
               producer=dict(stage=stage, package=package, run_id=os.path.basename(run_dir), commit="0" * 40))
    if kind != "synthetic":                          # the record a table-building stage writes (common.sealed_record)
        doc["sealed_check"] = dict(passed=True, ids=dict(checked=len(doc["genomes"]), distinct=len({g["id"] for g in doc["genomes"]}),
                                                         source=dict(kind="pinned", registry=dict(path="host/sealed_registry.json"))))
    doc = handoff.finish(doc)
    with open(os.path.join(run_dir, out), "w") as f:
        json.dump(doc, f)
    return doc
