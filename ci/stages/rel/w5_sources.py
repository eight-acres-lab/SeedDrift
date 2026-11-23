"""Stage rel_w5_sources (package W5; pipeline rel_traineval, run on the CPU with the GPUs hidden): what arms S1 and S2
of decision D7 take from their sources, read with the final weights of a passed training on the held-out tables of
its final evaluation, beside the reference arm's gain from relatives per stratum. A diagnosis only: nothing is
trained, no model code is changed, no arm is chosen by it. `python -m ci.stages.rel.w5_sources RUN_DIR`.

The model interpreter runs seeddrift/train/rel/readouts/w5_sources.py (its docstring names what is read per cell).
This stage then sums the cells by stratum: every cell, the bins of the nearest identity (the evaluator's FINE), the
two blocks (eukaryotes; prokaryotes, plasmids and viruses), with and without other rows; and sets beside it, from the
result rows of the trainings' final evaluations, the bits each arm gains from relatives (no relatives minus with
relatives, per bin of the nearest identity) and the share of the reference arm's gain each S arm reaches.

Profile block rel_w5_sources: arms [{name, run, folder}] (a passed training run and the arm's folder under
data/rel_w6/fit/), reference {name, run, folder} (its result rows only), conditions ({condition: layout}, default
scattered leakfree and whole_gene stored), per_group (50), max_rows (32), threads (4), wall_s (7200).
Outputs under data/rel_w5/sources/: <arm>.<condition>.npz (per cell), <arm>.json, done.json, readout.log,
report.json, report.md. Gate G4_rel_w5_sources: every arm read under every condition with cells, the cells read
are those of the training's final evaluation (the evaluator's digest), the GPUs hidden, every mean finite.
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time

import numpy as np

from ci import ROOT
from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common, w6_tools
from seeddrift.train.rel.evalh import score as SC
from seeddrift.train.rel.evalh import shards

STAGE = "rel_w5_sources"
OUT = os.path.join("data", "rel_w5", "sources")
FIT = os.path.join("data", "rel_w6", "fit")
KEYS = {"arms", "reference", "conditions", "per_group", "max_rows", "threads", "wall_s"}
DEFAULT_CONDITIONS = {"scattered": "leakfree", "whole_gene": "stored"}
ENTRY = "seeddrift.train.rel.readouts.w5_sources"
csv.field_size_limit(10 ** 9)


def settings(cfg) -> dict:
    c = dict(cfg.get(STAGE) or {})
    unknown = set(c) - KEYS
    if unknown or not c.get("arms") or not c.get("reference"):
        raise SystemExit("%s: arms and reference are needed (keys: %s); unknown %s" % (STAGE, sorted(KEYS), sorted(unknown)))

    def arm(a):
        if set(a) - {"name", "run", "folder"} or not all(a.get(k) for k in ("name", "run", "folder")):
            raise SystemExit("%s: an arm is {name, run, folder}: %r" % (STAGE, a))
        return dict(name=str(a["name"]), run=str(a["run"]), folder=str(a["folder"]), fit=os.path.join(str(a["run"]), FIT, str(a["folder"])))
    conds = dict(c.get("conditions") or DEFAULT_CONDITIONS)
    if any(v not in ("leakfree", "stored") for v in conds.values()):
        raise SystemExit("%s: a condition is read in the layout leakfree or stored: %r" % (STAGE, conds))
    return dict(arms=[arm(a) for a in c["arms"]], reference=arm(c["reference"]), conditions=conds, per_group=int(c.get("per_group", 50)),
                max_rows=int(c.get("max_rows", 32)), threads=int(c.get("threads", 4)), wall_s=int(c.get("wall_s", 7200)))


def passed(run):
    with open(os.path.join(run, "record.json")) as f:
        rec = json.load(f)
    return rec.get("status") == "passed"


def final_rows(fit):
    """{(condition, layout, stratum): bits} of the result rows of the final evaluation, group all."""
    out = {}
    with open(os.path.join(fit, "results.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["weights"] == "ema@final" and r["group"] == "all":
                out[(r["condition"], r["layout"], r["stratum"])] = float(r["bits"])
    return out


def gains(rows, layout="leakfree"):
    """{stratum: bits gained from relatives}: scattered+no_relatives minus scattered, per stratum both have."""
    out = {}
    for (c, lay, s), b in rows.items():
        if c == "scattered" and lay == layout and ("scattered+no_relatives", "stored", s) in rows:
            out[s] = rows[("scattered+no_relatives", "stored", s)] - b
    return out


def heldout_meta(fit):
    """identity (N, R) and group names (N,) of the held-out set the arm was scored on."""
    with open(os.path.join(fit, "conf.json")) as f:
        cfg = json.load(f)
    with open(cfg["heldout_tables"]) as f:
        doc = json.load(f)
    held = shards.by_split(cfg["heldout_root"], [doc])[shards.SPLIT_HELDOUT]
    got = held.load(0, held.tables, ("identity", "group"))
    names = {int(k): str(v) for k, v in (cfg.get("group_names") or {}).items()}
    return got["identity"], np.array([names.get(int(g), str(int(g))) for g in got["group"]])


def strata(identity, group):
    """[(name, labels, index of every table)]."""
    blocks = np.array([0 if SC.block_of(g) == SC.BLOCKS[0] else 1 for g in group], dtype=np.int64)
    has = ~np.isnan(SC.nearest_identity(identity))
    return [("nearest_identity", SC.FINE, SC.fine_stratum(identity)), ("block", SC.BLOCKS, blocks),
            ("relatives_on_target", SC.RELATIVES, (~has).astype(np.int64))]


def summarise(cells, identity, group):
    """{stratum label: means of the cells of its tables} over every cell and every stratum."""
    keys = [k for k in ("none", "entropy", "top", "bits", "none_given_truth") if k in cells]
    def means(m):
        if not m.any():
            return None
        out = {k: float(np.mean(cells[k][m])) for k in keys}
        out["lam"] = 1.0 - out["none"]
        out["level"] = [float(x) for x in np.mean(cells["level"][m], axis=0)]
        tid = cells["top_identity"][m]
        out["top_identity"] = float(np.nanmean(tid)) if np.isfinite(tid).any() else None
        out["top_is_nearest"] = float(np.mean(cells["top_is_nearest"][m]))
        out.update(cells=int(m.sum()), tables=int(len(np.unique(cells["table"][m]))))
        return out
    t = cells["table"]
    res = {"all": means(np.ones(len(t), dtype=bool))}
    for _, labels, index in strata(identity, group):
        for k, lab in enumerate(labels):
            got = means(index[t] == k)
            if got is not None:
                res[lab] = got
    return res


def build(st, cfg, run_dir):
    s, t0 = settings(cfg), time.time()
    for a in s["arms"] + [s["reference"]]:
        if not passed(a["run"]):
            raise SystemExit("%s: run %s has not passed" % (STAGE, a["run"]))
        need = ["conf.json", "results.tsv", "summary.json"] + (["final.pt"] if a is not s["reference"] else [])
        missing = [n for n in need if not os.path.exists(os.path.join(a["fit"], n))]
        if missing:
            raise SystemExit("%s: arm %s lacks %s" % (STAGE, a["name"], missing))
    py, lock, have = w6_tools.interpreter(st, cfg)
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    conf, log = os.path.join(folder, "readout.conf.json"), os.path.join(folder, "readout.log")
    write_json(conf, dict(arms=[dict(name=a["name"], fit=a["fit"]) for a in s["arms"]], conditions=s["conditions"], per_group=s["per_group"],
                          max_rows=s["max_rows"], threads=s["threads"], out=folder))
    lg = open(log, "a")
    proc = subprocess.Popen([py, "-m", ENTRY, conf], cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT, env=w6_tools.environment(s["threads"], hide_gpu=True))
    try:
        code = proc.wait(timeout=s["wall_s"])
    except subprocess.TimeoutExpired:
        code = "timeout after %d s" % s["wall_s"]
    w6_tools.stop(proc, lg)
    if code != 0 or not os.path.exists(os.path.join(folder, "done.json")):
        raise SystemExit("%s: the readout ended with %r: %s" % (STAGE, code, w6_tools.tail(log)))
    with open(os.path.join(folder, "done.json")) as f:
        done = json.load(f)
    ref_rows = final_rows(s["reference"]["fit"])
    ref_gain = gains(ref_rows)
    report = dict(version=1, python=py, python_lock=lock, model_env_sha256=sha256_json(have), torch=done.get("torch"), gpu_hidden=done.get("gpu_hidden"),
                  per_group=s["per_group"], conditions=s["conditions"], reference=dict(name=s["reference"]["name"], run=os.path.basename(s["reference"]["run"]),
                  gain=ref_gain), arms={})
    for a in s["arms"]:
        head = done["arms"][a["name"]]
        with open(os.path.join(a["fit"], "summary.json")) as f:
            final_items = json.load(f).get("final_items") or {}
        identity, group = heldout_meta(a["fit"])
        with open(os.path.join(a["fit"], "conf.json")) as f:
            trained_layout = json.load(f).get("layout")
        rows = final_rows(a["fit"])
        g = gains(rows)
        entry = dict(run=os.path.basename(a["run"]), folder=a["folder"], arm=head["arm"], variant=head["variant"], tables_seen=head["tables_seen"],
                     gain=g, share_of_reference_gain={k: (v / ref_gain[k]) if ref_gain.get(k) else None for k, v in g.items()}, conditions={})
        for c, layout in s["conditions"].items():
            h = head["conditions"][c]
            key = "%s@stored" % c if layout == "stored" and trained_layout != "stored" else c      # the final evaluation's own key
            with np.load(os.path.join(folder, "%s.%s.npz" % (a["name"], c.replace("+", "__")))) as z:
                cells = {k: z[k] for k in z.files}
            entry["conditions"][c] = dict(layout=layout, tables=h["tables"], cells=h["cells"], items_sha256=h["items_sha256"],
                                          same_cells_as_final=h["items_sha256"] == final_items.get(key), seconds=h["seconds"],
                                          strata=summarise(cells, identity, group))
        report["arms"][a["name"]] = entry
    report["seconds"] = round(time.time() - t0, 1)
    p = os.path.join(folder, "report.json")
    write_json(p, report)
    st.output("report", p)
    md = os.path.join(folder, "report.md")
    with open(md, "w") as f:
        f.write("\n".join(markdown(report)) + "\n")
    st.output("report_md", md)
    st.extra.update(arms=sorted(report["arms"]), seconds=report["seconds"])


def markdown(report):
    out = ["# What arms S1 and S2 take from their sources (package W5, decision D7; a diagnosis, never a selection)", "",
           "Final weights (moving average), the held-out tables of the final evaluation (%d per group), read on the CPU." % report["per_group"], ""]
    for name, a in report["arms"].items():
        out += ["## %s (arm %s, %s tables seen)" % (name, a["arm"], a["tables_seen"]), ""]
        for c, x in a["conditions"].items():
            out += ["### %s, layout %s: %d tables, %d cells; the cells of the final evaluation: %s" % (c, x["layout"], x["tables"], x["cells"], x["same_cells_as_final"]), "",
                    "| stratum | tables | cells | bits | lambda | none | none given truth | entropy | top weight | top identity | top is nearest | levels |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for lab, v in x["strata"].items():
                out.append("| %s | %d | %d | %.3f | %.3f | %.3f | %s | %.3f | %.3f | %s | %.3f | %s |" % (
                    lab, v["tables"], v["cells"], v["bits"], v["lam"], v["none"], "%.3f" % v["none_given_truth"] if "none_given_truth" in v else "-",
                    v["entropy"], v["top"], "%.3f" % v["top_identity"] if v["top_identity"] is not None else "-", v["top_is_nearest"],
                    "/".join("%.2f" % q for q in v["level"])))
            out.append("")
        out += ["### bits gained from relatives (scattered, final; no relatives minus with), beside the reference arm (%s)" % report["reference"]["name"], "",
                "| stratum | reference | this arm | share |", "|---|---|---|---|"]
        for lab, gr in report["reference"]["gain"].items():
            if lab in a["gain"]:
                sh = a["share_of_reference_gain"].get(lab)
                out.append("| %s | %.3f | %.3f | %s |" % (lab, gr, a["gain"][lab], "%.2f" % sh if sh is not None else "-"))
        out.append("")
    return out


def gate(st, cfg, run_dir):
    s = settings(cfg)
    p = os.path.join(run_dir, OUT, "report.json")
    with open(p) as f:
        report = json.load(f)
    problems = []
    if report.get("gpu_hidden") is not True:
        problems.append(dict(error="the GPUs were not hidden from the readout"))
    for a in s["arms"]:
        r = (report.get("arms") or {}).get(a["name"])
        if r is None:
            problems.append(dict(arm=a["name"], error="not read"))
            continue
        for c in s["conditions"]:
            x = (r.get("conditions") or {}).get(c)
            if not x or not x.get("cells"):
                problems.append(dict(arm=a["name"], condition=c, error="no cell read"))
                continue
            if not x.get("same_cells_as_final"):
                problems.append(dict(arm=a["name"], condition=c, error="the cells read are not those of the training's final evaluation"))
            bad = [lab for lab, v in x["strata"].items() if not all(math.isfinite(v[k]) for k in ("bits", "none", "entropy", "top"))]
            if bad:
                problems.append(dict(arm=a["name"], condition=c, error="a mean is not finite", strata=bad[:5]))
    require(GateResult("G4_rel_w5_sources", "fail" if problems else "pass", dict(problems=problems[:20], arms=sorted(report.get("arms") or {}))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        report = json.load(f)
    js = {name: {c: {k: round(x["strata"]["all"][k], 4) for k in ("bits", "lam", "none", "entropy", "top")} for c, x in a["conditions"].items()}
          for name, a in report["arms"].items()}
    with open(os.path.join(run_dir, OUT, "report.md")) as f:
        md = f.read().splitlines()
    return dict(json=js, md=md[:120])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, STAGE) as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
