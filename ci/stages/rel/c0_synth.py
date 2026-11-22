"""Stage rel_c0_synth (package C0; pipeline rel_tables): synthetic tables (ci/stages/rel/synth.py) written as an
ordinary table set, so that readers, renderers, trainers and evaluators can be run on a machine before any real table
exists. `python -m ci.stages.rel.c0_synth RUN_DIR`.

Configuration (profile block rel_c0_synth): tables (the count), seed, per_shard (tables per shard file), params
(overrides of synth.DEFAULTS). Outputs under data/rel_c0/synth/:
  shards/      the tables in the shard format of the contract (seeddrift/train/rel/api.py)
  meta.json    what models never read: per table its id, task, number of rows, block lengths, synthetic group
  truth.json   the correct sources of every table (synth.py `truth`), for tests and capability checks only
  tableset.json  the table-set manifest (seeddrift/train/rel/handoff.py), kind synthetic, written last
The stage reads no genome; it still loads this machine's sealed sources (the sealed check of every data stage), so a
machine without them fails here first. Gate G4_rel_c0_synth (gate()): the manifest is of kind synthetic, has the
configured number of tables, every file it lists is there, the truth covers every table, and no genome is listed.
"""
from __future__ import annotations

import json
import os

from ci.hashing import sha256_file, sha256_json
from ci.gates import GateResult, require
from ci.stages.common import main_args, write_json
from ci.stages.rel import common, synth
from seeddrift.train.rel import api, handoff

OUT = os.path.join("data", "rel_c0", "synth")
TABLE_VERSION = "c0-synth-1"
GROUPS = 12                    # synthetic group labels syn01..syn12, so per-group sampling and logging have groups to work on


def settings(cfg) -> dict:
    sc = cfg.get("rel_c0_synth") or {}
    unknown = set(sc) - {"tables", "seed", "per_shard", "params"}
    if unknown:
        raise SystemExit("rel_c0_synth: unknown keys %s" % sorted(unknown))
    out = dict(tables=int(sc.get("tables", 256)), seed=int(sc.get("seed", 1)), per_shard=int(sc.get("per_shard", 256)), params=synth.params(**(sc.get("params") or {})))
    if not 1 <= out["tables"] <= 100_000 or not 1 <= out["per_shard"] <= 4096:
        raise SystemExit("rel_c0_synth: tables 1..100000, per_shard 1..4096: %r" % out)
    return out


def write_shards(tables, folder, per_shard):
    """[(path, role)] of the shard files of these tables, in the contract's shard format."""
    raise SystemExit("rel_c0_synth: the shard format waits for version 1 of the contract (seeddrift/train/rel/api.py)")


def build(st, cfg, run_dir):
    s = settings(cfg)
    st.extra["sealed_check"] = common.sealed_check(cfg, [])
    tables = list(synth.tables(s["tables"], s["seed"], **s["params"]))
    folder = os.path.join(run_dir, OUT)
    os.makedirs(os.path.join(folder, "shards"), exist_ok=True)
    files = list(write_shards(tables, os.path.join(folder, "shards"), s["per_shard"]))
    meta = dict(tables=[dict(id=t["id"], task=t["task"], rows=int(t["blocks"][0].shape[0]), lengths=[int(b.shape[1]) for b in t["blocks"]],
                             group="syn%02d" % (1 + i % GROUPS)) for i, t in enumerate(tables)], roles=tables[0]["roles"], settings=s)
    truth = dict(note="the correct sources of every synthetic table: for tests and capability checks only, never a model input",
                 tables={t["id"]: t["truth"] for t in tables})
    for name, obj, role in (("meta.json", meta, "meta"), ("truth.json", truth, "truth")):
        write_json(os.path.join(folder, name), obj)
        files.append((os.path.join(folder, name), role))
    entries = [dict(path=os.path.relpath(p, run_dir), sha256=sha256_file(p, use_sidecar=False), bytes=os.path.getsize(p), role=role) for p, role in files]
    tasks = {k: sum(1 for t in tables if t["task"] == k) for k in ("locus", "switch")}
    doc = handoff.finish(dict(format="rel-tableset", format_version=1, name="c0-synth-s%d" % s["seed"], kind="synthetic", api_version=api.API_VERSION,
                              table_version=TABLE_VERSION,
                              coverage=dict(groups=["syn%02d" % (i + 1) for i in range(GROUPS)], scenarios=["synthetic copying with substitutions"],
                                            not_covered=["every real genome, group and scenario: these tables are synthetic",
                                                         "insertions and deletions", "transfer readouts"],
                                            note="synthetic tables for probes, tests and capability checks only; never training data of a real model"),
                              tables=len(tables), tasks=tasks, files=entries, genomes=[],
                              producer=dict(stage="c0_synth", package="C0", run_id=os.path.basename(run_dir), commit=cfg["git"]["sha"])))
    p = os.path.join(folder, "tableset.json")
    write_json(p, doc)
    st.output("tableset", p)
    st.extra.update(tables=len(tables), tasks=tasks, bytes=sum(e["bytes"] for e in entries), tableset_sha256=doc["sha256"])
    return doc


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "tableset.json")
    with open(p) as f:
        doc = json.load(f)
    s = settings(cfg)
    problems = [dict(e, part="manifest") for e in handoff.tableset_problems(doc, run_dir)]
    if doc.get("kind") != "synthetic" or doc.get("genomes"):
        problems.append(dict(error="not a synthetic table set without genomes"))
    if doc.get("tables") != s["tables"]:
        problems.append(dict(error="%s tables, %d configured" % (doc.get("tables"), s["tables"])))
    truth = [f for f in doc.get("files") or [] if f.get("role") == "truth"]
    if len(truth) != 1:
        problems.append(dict(error="exactly one truth file"))
    else:
        with open(os.path.join(run_dir, truth[0]["path"])) as f:
            n = len(json.load(f).get("tables") or {})
        if n != doc.get("tables"):
            problems.append(dict(error="the truth covers %d of %s tables" % (n, doc.get("tables"))))
    require(GateResult("G4_rel_c0_synth", "fail" if problems else "pass", dict(problems=problems[:50], tables=doc.get("tables"), tasks=doc.get("tasks"))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "tableset.json")) as f:
        doc = json.load(f)
    size = sum(x["bytes"] for x in doc["files"])
    return dict(json=dict(tables=doc["tables"], tasks=doc.get("tasks"), bytes=size, tableset_sha256=doc["sha256"]),
                md=["%d synthetic tables (%s), %.1f MB, table-set manifest %s" % (doc["tables"], ", ".join("%s %d" % kv for kv in sorted((doc.get("tasks") or {}).items())),
                                                                                    size / 1e6, doc["sha256"][:16])])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_c0_synth") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
