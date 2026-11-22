"""Gate phases of the relational-model line; ci/stages/gates.py hands every phase rel_<stage> and every kind here.

  rel_<stage>  the stage's own gates (configs/rel/kinds.yaml): the function gate(st, cfg, run_dir) of the stage's module
               decides each of them (with ci.gates.require or run) and returns the hash of what it read; a gate the
               registry names and the function did not decide fails the phase
  <kind>       after every stage of the run (gates/<kind>.ok):
               G4_rel_handoff  every table-set manifest and model index the run wrote is complete and matches its files
                               (seeddrift/train/rel/handoff.py)
               G3_rel          the sealed check (seeddrift/train/rel/sealed.py, then the lineage check of
                               ci/stages/rel/common.py) on every genome of every table set the run wrote, each in an
                               allowed split (train only for a set of purpose train; train or heldout otherwise),
                               and the set's own sealed record present. A run that wrote no table set has nothing to
                               check here (its stages read table sets through handoff.from_run, which refuses any
                               set without a passed record), so it passes on every machine.
Both kind gates are decided and recorded before a failure stops the run.
"""
from __future__ import annotations

import importlib
import json
import os

from ci.gates import GateFailure, GateResult, run
from ci.hashing import sha256_file, sha256_json
from ci.stages.rel import common, kinds
from seeddrift.train.rel import handoff, sealed as S


def phase(st, cfg, run_dir, name):
    on = kinds.stages_of(cfg)
    if kinds.is_kind(name):
        if name != cfg["pipeline"]:
            raise SystemExit("gate phase %s belongs to another kind than this run's %s" % (name, cfg["pipeline"]))
        return kind_phase(st, cfg, run_dir, on)
    stage = name[4:] if name.startswith("rel_") else ""
    if stage not in kinds.all_stages():
        raise SystemExit("unknown gate phase %s" % name)
    if stage not in on:
        raise SystemExit("gate phase %s: stage %s is not run by this profile (rel.stages)" % (name, stage))
    d = kinds.stage(stage)
    if d.get("stub"):
        raise SystemExit("stage %s is still a stub (configs/rel/kinds.yaml)" % stage)
    mod = importlib.import_module(d["module"])
    key = mod.gate(st, cfg, run_dir)
    missing = [g for g in d["gates"] if not os.path.exists(os.path.join(run_dir, "gates", g + ".json"))]
    if missing:
        raise SystemExit("gate phase %s did not decide %s" % (name, ", ".join(missing)))
    return key


def kind_phase(st, cfg, run_dir, on):
    hand, seal, keys = [], [], []
    counts = dict(tablesets=0, model_indexes=0, genomes=0)
    sealed = None
    for s in on:
        d = kinds.stage(s)
        if not d.get("produces"):
            continue
        p = os.path.join(run_dir, d["output"])
        try:
            with open(p) as f:
                doc = json.load(f)
        except (OSError, ValueError) as err:
            hand.append(dict(stage=s, error="cannot read %s: %s" % (d["output"], err)))
            continue
        keys.append(sha256_file(p, use_sidecar=False))
        if d["produces"] == "models":
            counts["model_indexes"] += 1
            hand += [dict(e, stage=s) for e in handoff.models_problems(doc, run_dir)]
            continue
        counts["tablesets"] += 1
        hand += [dict(e, stage=s) for e in handoff.tableset_problems(doc, run_dir)]
        if (doc.get("producer") or {}).get("stage") != s:
            hand.append(dict(stage=s, error="the manifest names producer stage %r" % (doc.get("producer") or {}).get("stage")))
        genomes = [g for g in doc.get("genomes") or [] if isinstance(g, dict)]
        counts["genomes"] += len(genomes)
        allowed = ("train",) if doc.get("purpose") == "train" else S.ALLOWED
        island = [g for g in genomes if g.get("split") == "island"]
        rest = [g for g in genomes if g.get("split") != "island"]
        try:
            sealed = sealed or common.sealed_set(cfg)
            if island:                                 # the island set's development chromosomes (8.8): not_for_training only,
                if not doc.get("not_for_training"):     # never sealed, and not in the lineage check (their order is excluded
                    raise SystemExit("genomes of split island in a set not marked not_for_training")   # from training only)
                common.development_chromosomes([g["id"] for g in island])
                S.check_ids([g["id"] for g in island], sealed, splits={g["id"]: "island" for g in island}, allowed=("island",))
            ev = common.sealed_check(cfg, rest, splits={g["id"]: g.get("split") for g in rest}, allowed=allowed, sealed=sealed)
            if island:
                ev["islands"] = dict(checked=len(island), development_chromosomes=True)
            seal.append(dict(stage=s, tableset=doc.get("name"), kind=doc.get("kind"), passed=True, evidence=ev))
        except SystemExit as err:
            seal.append(dict(stage=s, tableset=doc.get("name"), kind=doc.get("kind"), passed=False, error=str(err)[:1500]))
    if counts["tablesets"] == 0:
        seal.append(dict(passed=True, note="no table set written by this run; its stages read table sets through handoff.from_run"))
    results = [run(GateResult("G4_rel_handoff", "fail" if hand else "pass", dict(problems=hand[:50], counts=counts)), run_dir),
               run(GateResult("G3_rel", "fail" if any(not x.get("passed") for x in seal) else "pass",
                              dict(checks=seal, source=sealed.source if sealed is not None else None)), run_dir)]
    bad = [r.gate for r in results if r.status == "fail"]
    if bad:
        raise GateFailure("gates failed: %s" % ", ".join(bad))
    return sha256_json(keys)
