"""Stage rel_w12_tstats (package W12; pipeline rel_tables): how often the target genes of a built set of tables of
version 1 are used, per group (the programme lead, 2026-11-23, for round 1's report: the share of tables whose target is
used ten times or more, and the genomes they come from). `python -m ci.stages.rel.w12_tstats RUN_DIR`. Reads the
table set's manifest, its index and the target and group fields of its shards, the training data of the search's
source (only the codon offsets, for the genes that can be a target) and the relation of every genome; writes no table.

Configuration (profile block rel_w12_tstats): tables_from (a passed run of stage w12_v1, w12_islands or w2_v0), which
(v1, islands or v0, default v1; v0: the tables of version 0, package W2, whose source run is named by their index). Per
group it also gives the genomes of the part (the examined genomes, row 0) and how many of them the tables use. Outputs under data/rel_w12/tstats/: report.json (written last) and report.md. Gate
G4_rel_w12_tstats: every part's counts add up to its tables, and the set is the one the report names.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from ci.stages.rel import w2_v0 as W
from ci.stages.rel import w12_v1 as WV
from seeddrift.train.rel import handoff
from seeddrift.train.rel.tables import v0_io as IO
from seeddrift.train.rel.tables import v1_build as V1

OUT = os.path.join("data", "rel_w12", "tstats")
KEYS = {"tables_from", "which"}


SETS = dict(WV.WHICH, v0=dict(stage="w2_v0", out=os.path.join("data", "rel_w2", "v0")))


def settings(cfg) -> dict:
    c = cfg.get("rel_w12_tstats") or {}
    unknown = set(c) - KEYS
    if unknown or not c.get("tables_from") or str(c.get("which", "v1")) not in SETS:
        raise SystemExit("rel_w12_tstats: needs tables_from, which in %s; unknown keys %s" % (sorted(SETS), sorted(unknown)))
    return dict(tables_from=str(c["tables_from"]), which=str(c.get("which", "v1")))


def build(st, cfg, run_dir):
    s = settings(cfg)
    w = SETS[s["which"]]
    run = s["tables_from"]
    rec = json.load(open(os.path.join(run, "record.json")))
    if rec.get("status") != "passed":
        raise SystemExit("rel_w12_tstats: %s is %s, not passed" % (rec.get("run_id"), rec.get("status")))
    tp = os.path.join(run, w["out"], "tableset.json")
    W._output(run, "rel_" + w["stage"], "tableset", tp)
    doc = json.load(open(tp))
    bad = handoff.tableset_problems(doc, run)
    if bad:
        raise SystemExit("rel_w12_tstats: the table set of %s is not sound: %s" % (rec.get("run_id"), bad[:3]))
    index = json.load(open(os.path.join(run, w["out"], "index.json")))
    genomes, groups = index["genomes"], index["groups"]
    runs = os.path.dirname(os.path.normpath(run))
    if s["which"] == "v0":                                          # version 0: the source run its index names
        prov = index.get("source")
        src = W.source(os.path.join(runs, prov["run_id"]), pairs=False)
    else:
        man, _, _, prov = WV.search_run(os.path.join(runs, doc["search"]["run_id"]))
        src = W.source(man["settings"]["source"], pairs=False)
    rel_of = {g["id"]: g.get("relation", "") for g in src["index"]["genomes"] if g.get("relation")}
    st.extra["sealed_check"] = common.sealed_check(cfg, [dict(id=g["id"], taxid=int(g["taxid"])) for g in doc["genomes"] if g["split"] != "island"],
                                                   splits={g["id"]: g["split"] for g in doc["genomes"] if g["split"] != "island"})
    parts = {}
    for part, p in doc["parts"].items():
        folder = os.path.join(run, p["folder"])
        names = [x["name"] for x in p["shards"]]
        tgt = np.concatenate([np.load(IO.shard_path(folder, n, "gene"), mmap_mode="r")[:, V1.TARGET] for n in names]).astype(np.int64)
        gen = np.concatenate([np.load(IO.shard_path(folder, n, "genome"), mmap_mode="r")[:, 0] for n in names]).astype(np.int64)
        grp = np.concatenate([np.load(IO.shard_path(folder, n, "group"), mmap_mode="r") for n in names]).astype(np.int64)
        plan = ([genomes[i]["id"] for i in gen], [int(x) for x in grp])
        mine = [g for g in genomes if g["split"] == part]
        valid = {}
        for g in mine:
            path = os.path.join(src["run"], "data", "train_data", g["id"] + ".npz")
            if os.path.exists(path):
                with np.load(path) as z:
                    valid[g["id"]] = np.flatnonzero(np.diff(z["codon_off"]) > 0)
        stats = V1.target_stats(plan, groups, tgt, rel_of, valid, {g["id"]: g["group"] for g in mine})
        used = set(plan[0])
        for g, x in stats.items():
            gs = [m["id"] for m in mine if g == "all" or m["group"] == g]
            x.update(genomes=len(gs), genomes_examined=sum(1 for m in gs if m in used))
        parts[part] = dict(tables=int(len(tgt)), sampling=(p.get("targets") or {}).get("sampling", "genome"), by_group=stats)
    rep = dict(format="rel-w12-tstats", version=1, note=__doc__.strip().splitlines()[0], tables=dict(run=rec.get("run_id"), which=s["which"],
               tableset_sha256=doc.get("sha256"), name=doc.get("name")), search=prov, parts=parts)
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    page = os.path.join(folder, "report.md")
    with open(page, "w") as f:
        f.write(page_of(rep))
    st.output("tstats_page", page)
    p = os.path.join(folder, "report.json")
    write_json(p, rep)
    st.output("tstats_report", p)
    st.extra.update(parts={k: v["by_group"].get("all") for k, v in parts.items()})
    return rep


def page_of(rep):
    t = rep["tables"]
    L = ["# How often the target genes are used (tables of version 1)", "", rep["note"], "",
         "Table set `%s` of run `%s` (sha256 %s)." % (t["name"], t["run"], (t["tableset_sha256"] or "")[:12]), ""]
    for part, d in rep["parts"].items():
        L += ["## %s: %s tables (targets drawn %s)" % (part, "{:,}".format(d["tables"]), "over each group's genes" if d["sampling"] == "gene" else "per genome"), "",
              "| group | tables | genomes (examined) | genes that can be a target | distinct targets | mean uses | most uses | tables whose target is used 10+ times | genomes of those |",
              "|---|---|---|---|---|---|---|---|---|"]
        for g, x in d["by_group"].items():
            L.append("| %s | %s | %s (%s) | %s | %s | %s | %d | %s (%.1f%%) | %d |" % (g, "{:,}".format(x["tables"]), x.get("genomes"), x.get("genomes_examined"),
                                                                    "{:,}".format(x.get("targetable_genes") or 0),
                                                                    "{:,}".format(x["distinct"]), x.get("mean_uses"), x["max_uses"],
                                                                    "{:,}".format(x["tables_target_used_10_plus"]), 100 * x["share_target_used_10_plus"],
                                                                    x["genomes_of_those"]))
        L.append("")
    return "\n".join(L)


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "report.json")
    rep = json.load(open(p))
    problems = []
    for part, d in (rep.get("parts") or {}).items():
        groups = {g: x for g, x in d["by_group"].items() if g != "all"}
        if sum(x["tables"] for x in groups.values()) != d["tables"] or d["by_group"]["all"]["tables"] != d["tables"]:
            problems.append(dict(part=part, error="the groups' tables do not add up"))
    s = settings(cfg)
    tp = os.path.join(s["tables_from"], SETS[s["which"]]["out"], "tableset.json")
    if json.load(open(tp)).get("sha256") != (rep.get("tables") or {}).get("tableset_sha256"):
        problems.append(dict(error="the table set is not the one the report names"))
    require(GateResult("G4_rel_w12_tstats", "fail" if problems else "pass", dict(problems=problems[:50])), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    rep = json.load(open(os.path.join(run_dir, OUT, "report.json")))
    return dict(json={k: v["by_group"].get("all") for k, v in rep["parts"].items()}, md=page_of(rep).splitlines()[2:40])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w12_tstats") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
