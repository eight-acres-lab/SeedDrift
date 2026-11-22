"""Stage rel_w6_report (package W6; pipeline rel_traineval): the learning curves of the arms set side by side, aligned
by cells seen, next to the classical baselines scored on the same held-out tables. `python -m ci.stages.rel.w6_report
RUN_DIR`. CPU only, numpy and the standard library; it reads result rows (the columns of the contract) and nothing else of a model.

Profile block rel_w6_report (all optional):
  fits       [run directory, ...]: passed runs of this machine whose stage rel_w6_fit trained arms; their rows are set
             beside those of this run (which comes first when its own rel_w6_fit ran)
  baselines  [{run, path}]: result tables (the same columns; one `variant` per baseline, cells_seen 0) of runs of this
             machine, e.g. the classical baselines of package W5
  main       the condition of the headline curve (default scattered)
Outputs under data/rel_w6/report/:
  curves.tsv    every evaluation of every arm: cells seen, target cells, bits under each condition (all groups, all strata)
  aligned.tsv   the curves of the main condition on one grid of cells seen (linear interpolation between an arm's
                evaluations; empty beyond its last one), one column per arm
  (the result rows are the evaluator's, seeddrift/train/rel/evalh/score.py: groups by their index, the strata the
  identity of the nearest row in ten bins; an arm and a baseline scored the same cells when their rows of a condition
  carry the same items_sha256, which the report states and its gate requires)
  final.tsv     the final weights of every arm and every baseline: bits by condition, group and stratum
  report.md     the tables a reader starts from; report.json the same as data, written last
Every number is a development diagnosis under the stored placement: a hidden whole gene still has its relatives placed
by the hidden residues, so "whole_gene" rows are upper bounds; rows were chosen among genomes that have a cell on the
target gene, which favours relatives; one seed per arm, no noise floor.
Gate G4_rel_w6_report: every trained arm has a curve of at least two evaluations and final bits under its conditions.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from seeddrift.train.rel import handoff
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import score as SC

OUT = os.path.join("data", "rel_w6", "report")
FIT = os.path.join("data", "rel_w6", "fit")
CAVEATS = [
    "stored placement: a hidden whole gene keeps its relatives placed by the hidden residues, so whole_gene rows are upper bounds; scattered is the main reading",
    "rows were drawn among training genomes that have a cell on the target gene, which similarity decided: what relatives are worth is overestimated",
    "one seed per arm and no noise floor: only large effects (relatives or not, the trend over sizes) are read",
    "development diagnosis, not a round of the selection; curves end where the learning rate reaches zero, so their last stretch is shaped by the schedule",
]


def settings(cfg) -> dict:
    sc = cfg.get("rel_w6_report") or {}
    unknown = set(sc) - {"fits", "baselines", "main"}
    if unknown:
        raise SystemExit("rel_w6_report: unknown keys %s" % sorted(unknown))
    base = [dict(run=str(b["run"]), path=str(b["path"])) for b in sc.get("baselines") or []]
    return dict(fits=[str(x) for x in sc.get("fits") or []], baselines=base, main=str(sc.get("main", cond.SCATTERED)))


def number(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def collect(run_dir, cfg, s):
    """Rows and model indexes of this run's fit stage and of the other runs named; baseline rows."""
    fits = ([run_dir] if os.path.exists(os.path.join(run_dir, FIT, "models.json")) else []) + [f for f in s["fits"] if os.path.realpath(f) != os.path.realpath(run_dir)]
    if not fits:
        raise SystemExit("rel_w6_report: no fit to report (this run trained nothing and rel_w6_report.fits names no run)")
    rows, arms, sources, groups = [], {}, [], []
    for f in fits:
        if f == run_dir:
            with open(os.path.join(f, FIT, "models.json")) as fh:
                doc = json.load(fh)
        else:
            try:
                doc = handoff.from_run(f, os.path.join(FIT, "models.json"))
            except (handoff.HandoffError, OSError, ValueError) as err:
                raise SystemExit("rel_w6_report: %s: %s" % (f, err))
        for v in doc["variants"]:
            if v["name"] in arms:
                raise SystemExit("rel_w6_report: arm %s is in two of the runs reported" % v["name"])
            arms[v["name"]] = dict(v, run=os.path.basename(f))
        rows += SC.read_rows(os.path.join(f, FIT, "results.tsv"))
        groups = groups or list(doc.get("groups") or [])
        sources.append(dict(run=os.path.basename(f), models_sha256=doc["sha256"], tablesets=doc["tablesets"]))
    base = []
    for b in s["baselines"]:
        p = os.path.join(b["run"], b["path"])
        rec = os.path.join(b["run"], "record.json")
        status = json.load(open(rec)).get("status") if os.path.exists(rec) else None
        if status != "passed":
            raise SystemExit("rel_w6_report: the baselines' run %s is %s, not passed" % (b["run"], status))
        got = SC.read_rows(p)
        base += got
        sources.append(dict(run=os.path.basename(b["run"]), baselines=b["path"], sha256=sha256_file(p, use_sidecar=False), rows=len(got)))
    return rows, arms, base, sources, groups


def same_items(rows, base, arms):
    """{condition: whether the final rows of the arms and the baseline rows carry one digest of what was scored}; a
    condition without baseline rows is not listed. False means the bits beside each other are of different cells."""
    out = {}
    for c in cond.CONDITIONS:
        mine = {r["items_sha256"] for r in rows if r["weights"] == "final" and r["condition"] == c and r["variant"] in arms}
        theirs = {r["items_sha256"] for r in base if r["condition"] == c}
        if mine and theirs:
            out[c] = len(mine | theirs) == 1
    return out


def curves(rows, arms):
    """{arm: [{cells_seen, target_cells, weights, condition: bits}]} over the evaluations (all groups, all strata), by cells seen."""
    out = {}
    for r in rows:
        if r["group"] == SC.ALL and r["stratum"] == SC.ALL and r["variant"] in arms:
            key = (int(r["cells_seen"]), r["weights"] == "final")
            point = out.setdefault(r["variant"], {}).setdefault(key, dict(cells_seen=int(r["cells_seen"]), scored_cells=int(r["target_cells"]),
                                                                       weights=r["weights"], items=int(r["n_items"])))
            point[r["condition"]] = float(r["bits"])
    return {a: [pts[k] for k in sorted(pts)] for a, pts in out.items()}


def aligned(cv, main):
    """The main condition of every arm on one grid of cells seen (for an arm trained without relatives the same
    condition without relatives, named in its column). The subset points make the curve (the final point is scored on
    all tables, another set of items, and is listed in final.tsv)."""
    alone = main + "+" + cond.NO_RELATIVES                       # the reading of an arm trained without relatives
    series = {}
    for a, pts in cv.items():
        c = main if any(main in p for p in pts) else alone
        x = [(p["cells_seen"], p[c]) for p in pts if p["weights"] != "final" and c in p]
        if len(x) >= 2:
            series[a if c == main else "%s (%s)" % (a, c)] = x
    if not series:
        return [], []
    steps = [np.median(np.diff([c for c, _ in x])) for x in series.values()]
    step = float(max(steps))
    top = max(x[-1][0] for x in series.values())
    grid = np.arange(0, top + step / 2, step)
    table = []
    for g in grid:
        row = [int(g)]
        for a in sorted(series):
            xs, ys = zip(*series[a])
            row.append(float(np.interp(g, xs, ys)) if g <= xs[-1] else None)
        table.append(row)
    return ["cells_seen"] + sorted(series), table


def final_table(rows, base, arms):
    """[(kind, variant, condition, group, stratum, bits, n_items)] of the final weights and of the baselines."""
    out = []
    for r in rows:
        if r["weights"] == "final" and r["variant"] in arms:
            out.append(("model", r["variant"], r["condition"], r["group"], r["stratum"], float(r["bits"]), int(r["n_items"])))
    for r in base:
        if number(r["bits"]) is not None:
            out.append(("baseline", r["variant"], r["condition"], r["group"], r["stratum"], float(r["bits"]), int(number(r["n_items"]) or 0)))
    return out


def tsv(path, header, table):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for row in table:
            f.write("\t".join("" if x is None else ("%.6f" % x if isinstance(x, float) else str(x)) for x in row) + "\n")


def md_table(header, table):
    fmt = lambda x: "" if x is None else ("%.3f" % x if isinstance(x, float) else ("{:,}".format(x) if isinstance(x, int) else str(x)))
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(fmt(x) for x in row) + " |" for row in table]


def build(st, cfg, run_dir):
    s = settings(cfg)
    rows, arms, base, sources, groups = collect(run_dir, cfg, s)
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    cv = curves(rows, arms)
    conds = [c for c in cond.CONDITIONS if any(c in p for pts in cv.values() for p in pts)]
    tsv(os.path.join(folder, "curves.tsv"), ["arm", "weights", "tables_scored", "cells_seen"] + conds,
        [[a, p["weights"], p["items"], p["cells_seen"]] + [p.get(c) for c in conds] for a in sorted(cv) for p in cv[a]])
    head, grid = aligned(cv, s["main"])
    tsv(os.path.join(folder, "aligned.tsv"), head or ["cells_seen"], grid)
    fin = final_table(rows, base, arms)
    tsv(os.path.join(folder, "final.tsv"), ["kind", "variant", "condition", "group", "stratum", "bits", "n_items"], fin)
    names = sorted(arms)
    bnames = sorted({r[1] for r in fin if r[0] == "baseline"})
    cell = {(r[1], r[2], r[3], r[4]): r[5] for r in fin}
    arm_rows = []
    for a in names:
        i = arms[a]["info"]
        arm_rows.append([a, arms[a]["status"], i.get("params"), i.get("steps"), i.get("cells_seen"), i.get("target_cells"), number(i.get("cells_per_s")),
                         number((i.get("preflight") or {}).get("cells_per_s")), i.get("reserved_mb"), i.get("peak_mb"), i.get("micro"), i.get("ended_by"), i.get("wall_s")])
    by_cond = [[c] + [cell.get((v, c, SC.ALL, SC.ALL)) for v in names + bnames] for c in conds]
    strata = [[c, x.split("=")[-1]] + [cell.get((v, c, SC.ALL, x)) for v in names + bnames] for c in conds for x in SC.STRATA
              if any((v, c, SC.ALL, x) in cell for v in names + bnames)]
    gname = lambda g: groups[int(g)] if g.isdigit() and int(g) < len(groups) else g                # noqa: E731
    by_group = [[gname(g)] + [cell.get((v, s["main"], g, SC.ALL)) for v in names + bnames]
                for g in sorted({r[3] for r in fin if r[3] != SC.ALL}, key=lambda g: (not g.isdigit(), int(g) if g.isdigit() else 0, g))]
    same = same_items(rows, base, arms)
    md = ["# Relational line: learning curves of the reference trunk", "", "Bits per codon of the examined row's target block on held-out tables. " +
          "Development diagnosis:", ""] + ["- " + c for c in CAVEATS] + ["", "## Arms: throughput and memory as measured", ""]
    md += md_table(["arm", "status", "parameters", "steps", "cells seen", "target cells", "cells/s (training)", "cells/s (preflight, alone)", "GPU MB reserved",
                    "GPU MB allocated", "tables per micro-batch", "ended by", "wall s"], arm_rows)
    md += ["", "## Final weights, all held-out tables: bits per codon by condition (mean of the groups)", ""] + md_table(["condition"] + names + bnames, by_cond)
    if bnames:
        md += ["", "The arms and the baselines scored the same cells (one digest of the scored cells per condition): %s." % (
            ", ".join("%s %s" % (c, "yes" if ok else "NO") for c, ok in same.items()) or "no condition in common")]
    md += ["", "## By the identity of the nearest row (target block)", ""] + md_table(["condition", "nearest identity"] + names + bnames, strata)
    md += ["", "## By group, condition %s" % s["main"], ""] + md_table(["group"] + names + bnames, by_group)
    md += ["", "## Curves of condition %s, aligned by cells seen (the subset of held-out tables scored on the way)" % s["main"], ""] + (md_table(head, grid) if head else ["no curve"])
    for a in names:
        if arms[a]["status"] != "ok":
            md += ["", "%s failed: `%s`" % (a, " ".join((arms[a].get("error") or "").split())[-600:])]
    with open(os.path.join(folder, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    report = dict(main=s["main"], caveats=CAVEATS, sources=sources, conditions=conds, groups=groups, same_items=same,
                  arms={a: dict(status=arms[a]["status"], run=arms[a]["run"], info=arms[a]["info"], evaluations=len(cv.get(a, [])),
                                final={c: cell.get((a, c, SC.ALL, SC.ALL)) for c in conds}) for a in names},
                  baselines={b: {c: cell.get((b, c, SC.ALL, SC.ALL)) for c in conds} for b in bnames},
                  aligned=dict(columns=head, rows=grid), by_condition=dict(columns=["condition"] + names + bnames, rows=by_cond),
                  by_stratum=dict(columns=["condition", "stratum"] + names + bnames, rows=strata), by_group=dict(columns=["group"] + names + bnames, rows=by_group))
    for name in ("curves.tsv", "aligned.tsv", "final.tsv", "report.md"):
        st.output(name.replace(".", "_"), os.path.join(folder, name))
    p = os.path.join(folder, "report.json")
    write_json(p, report)                                                 # the rule's target last
    st.output("report", p)
    st.extra.update(arms=len(names), baselines=bnames, conditions=conds)
    return report


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "report.json")
    with open(p) as f:
        rep = json.load(f)
    problems = []
    trained = {a: x for a, x in rep["arms"].items() if x["status"] == "ok"}
    if not trained:
        problems.append(dict(error="no trained arm in the report"))
    for a, x in trained.items():
        if x["evaluations"] < 2:
            problems.append(dict(arm=a, error="a curve of %d evaluations" % x["evaluations"]))
        if not any(v is not None for v in x["final"].values()):
            problems.append(dict(arm=a, error="no final bits"))
    if len(rep.get("caveats") or []) < 4:
        problems.append(dict(error="the report does not state what it does not show"))
    if not all((rep.get("same_items") or {}).values()):
        problems.append(dict(error="arms and baselines were not scored on the same cells", same_items=rep.get("same_items")))
    require(GateResult("G4_rel_w6_report", "fail" if problems else "pass",
                       dict(problems=problems[:20], arms={a: x["final"] for a, x in rep["arms"].items()}, baselines=sorted(rep["baselines"]))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        rep = json.load(f)
    with open(os.path.join(run_dir, OUT, "report.md")) as f:
        md = f.read().splitlines()
    small = dict(main=rep["main"], caveats=rep["caveats"], by_condition=rep["by_condition"], by_stratum=rep["by_stratum"], aligned=rep["aligned"],
                 same_items=rep.get("same_items"),
                 arms={a: dict(status=x["status"], final=x["final"], cells_seen=x["info"].get("cells_seen"), cells_per_s=x["info"].get("cells_per_s"),
                               reserved_mb=x["info"].get("reserved_mb"), params=x["info"].get("params")) for a, x in rep["arms"].items()},
                 baselines=rep["baselines"])
    return dict(json=small, md=["####" + line[2:] if line.startswith("## ") else line for line in md[1:]])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w6_report") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
