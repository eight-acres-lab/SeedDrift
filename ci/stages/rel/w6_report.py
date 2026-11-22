"""Stage rel_w6_report (package W6; pipeline rel_traineval): the learning curves of the arms side by side, on the axis of
tables seen, next to the classical baselines scored on the same held-out tables. `python -m ci.stages.rel.w6_report
RUN_DIR`. CPU only, numpy and the standard library; it reads result rows (the columns of the contract) and the model
index, nothing of a model.

Profile block rel_w6_report (all optional):
  fits       [run directory, ...]: passed runs of this machine whose stage rel_w6_fit trained arms; their rows are set
             beside those of this run (which comes first when its own rel_w6_fit ran)
  baselines  [{run, path}]: result tables of runs of this machine (the classical baselines of package W5: path
             data/rel_w5/baselines/rows.tsv; its results.json beside it, when there, gives the digest of the cells it
             scored under every condition)
  main       the condition of the headline curve (default scattered; an arm trained without relatives is read under
             the same condition without relatives)
Outputs under data/rel_w6/report/:
  curves.tsv    every evaluation of every arm: tables, cells and target cells seen, bits under each condition (all
                groups, all strata); the nodes are the same for every arm (1,000, 2,000, 4,000 ... tables seen)
  aligned.tsv   the main condition of every arm at every node, one column per arm
  final.tsv     the final evaluation of every arm and every baseline: bits by condition, group and stratum
  report.md     the tables a reader starts from; report.json the same as data, written last
The rows are the evaluator's (seeddrift/train/rel/evalh/score.py): groups by their index (names from the table set's
index), strata of the nearest row's identity (ten bins and coarse ones, and the number of other rows). An arm and a
baseline scored the same cells when the digests of their final evaluation (conditions.ItemsDigest) are equal, which
the report states and its gate requires wherever both are known.
Every number is a development diagnosis under the stored placement (see CAVEATS).
Gate G4_rel_w6_report: every trained arm has a curve of at least two evaluations and final bits under its conditions,
and no digest comparison failed.
"""
from __future__ import annotations

import json
import os
import re

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from seeddrift.train.rel import handoff
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import score as SC

OUT = os.path.join("data", "rel_w6", "report")
FIT = os.path.join("data", "rel_w6", "fit")
NODE_RX = re.compile(r"^(ema|raw)@tables=(\d+)$")
CAVEATS = [
    "stored placement: a hidden whole gene keeps its relatives placed by the hidden residues, so whole_gene rows are upper bounds; scattered is the main reading",
    "rows were drawn among training genomes that have a cell on the target gene, which similarity decided: what relatives are worth is overestimated",
    "one seed per arm and no noise floor: only large effects (relatives or not, the trend over sizes) are read",
    "learning rates were set by width, not tuned (warm-up over 1,000 steps, then constant); evaluations read the moving average of the weights",
    "development diagnosis, not a round of the selection; arms saw different numbers of tables in the same time, so curves are compared at equal tables seen",
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
    """Rows and model indexes of this run's fit stage and of the other runs named; baseline rows and digests."""
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
    base, digests = [], {}
    for b in s["baselines"]:
        p = os.path.join(b["run"], b["path"])
        rec = os.path.join(b["run"], "record.json")
        status = json.load(open(rec)).get("status") if os.path.exists(rec) else None
        if status != "passed":
            raise SystemExit("rel_w6_report: the baselines' run %s is %s, not passed" % (b["run"], status))
        got = SC.read_rows(p)
        base += got
        res = os.path.join(os.path.dirname(p), "results.json")
        if os.path.exists(res):
            with open(res) as fh:
                items = json.load(fh).get("items") or {}
            digests.update({c: x.get("sha256") for c, x in items.items() if isinstance(x, dict)})
        sources.append(dict(run=os.path.basename(b["run"]), baselines=b["path"], sha256=sha256_file(p, use_sidecar=False), rows=len(got)))
    return rows, arms, base, digests, sources, groups


def same_items(arms, digests):
    """{condition: whether every trained arm scored, in its final evaluation, the cells the baselines scored}; only
    conditions for which both digests are known."""
    out = {}
    for c in cond.CONDITIONS:
        mine = {(a["info"].get("final_items") or {}).get(c) for a in arms.values() if a["status"] == "ok"} - {None}
        if mine and digests.get(c):
            out[c] = mine == {digests[c]}
    return out


def curves(rows, arms):
    """{arm: [{tables_seen, cells_seen, scored_cells, items, condition: bits}]} over the node evaluations (all groups,
    all strata), by tables seen; the final evaluation is the last point, tables seen from the model index."""
    out = {}
    for r in rows:
        if r["group"] != SC.ALL or r["stratum"] != SC.ALL or r["variant"] not in arms:
            continue
        m = NODE_RX.match(r["weights"])
        if m:
            key, tables = (0, int(m.group(2))), int(m.group(2))
        elif r["weights"].endswith("@final"):
            key, tables = (1, 0), arms[r["variant"]]["info"].get("tables_seen")
        else:
            continue
        point = out.setdefault(r["variant"], {}).setdefault(key, dict(tables_seen=tables, cells_seen=int(r["cells_seen"]), final=key[0] == 1,
                                                                   scored_cells=int(r["target_cells"]), items=int(r["n_items"])))
        point[r["condition"]] = float(r["bits"])
    return {a: [pts[k] for k in sorted(pts)] for a, pts in out.items()}


def reading(arm_points, main):
    """The condition an arm's curve is read under: main, or main without relatives for an arm trained without them."""
    return main if any(main in p for p in arm_points) else main + "+" + cond.NO_RELATIVES


def aligned(cv, main):
    """The main reading of every arm at every node of tables seen (the nodes are the same for every arm)."""
    cols = {a: reading(pts, main) for a, pts in cv.items()}
    nodes = sorted({p["tables_seen"] for pts in cv.values() for p in pts if not p["final"]})
    head = ["tables_seen"] + ["%s (%s)" % (a, cols[a]) if cols[a] != main else a for a in sorted(cv)]
    table = []
    for t in nodes:
        row = [t]
        for a in sorted(cv):
            got = [p.get(cols[a]) for p in cv[a] if not p["final"] and p["tables_seen"] == t]
            row.append(got[0] if got else None)
        table.append(row)
    return head, table


def final_table(rows, base, arms):
    """[(kind, variant, condition, group, stratum, bits, n_items)] of the arms' final evaluation and of the baselines."""
    out = []
    for r in rows:
        if r["weights"].endswith("@final") and r["variant"] in arms:
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
    fmt = lambda x: "" if x is None else ("%.3f" % x if isinstance(x, float) else ("{:,}".format(x) if isinstance(x, int) else str(x)))      # noqa: E731
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(fmt(x) for x in row) + " |" for row in table]


def build(st, cfg, run_dir):
    s = settings(cfg)
    rows, arms, base, digests, sources, groups = collect(run_dir, cfg, s)
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    cv = curves(rows, arms)
    conds = [c for c in cond.CONDITIONS if any(c in p for pts in cv.values() for p in pts)]
    tsv(os.path.join(folder, "curves.tsv"), ["arm", "point", "tables_seen", "cells_seen", "tables_scored"] + conds,
        [[a, "final" if p["final"] else "node", p["tables_seen"], p["cells_seen"], p["items"]] + [p.get(c) for c in conds] for a in sorted(cv) for p in cv[a]])
    head, grid = aligned(cv, s["main"])
    tsv(os.path.join(folder, "aligned.tsv"), head, grid)
    fin = final_table(rows, base, arms)
    tsv(os.path.join(folder, "final.tsv"), ["kind", "variant", "condition", "group", "stratum", "bits", "n_items"], fin)
    names = sorted(arms)
    bnames = sorted({r[1] for r in fin if r[0] == "baseline"})
    cell = {(r[1], r[2], r[3], r[4]): r[5] for r in fin}
    same = same_items(arms, digests)
    arm_rows = []
    for a in names:
        i = arms[a]["info"]
        arm_rows.append([a, arms[a]["status"], i.get("params"), i.get("lr"), "yes" if i.get("lr_reduced") else "no", i.get("steps"), i.get("tables_seen"),
                         i.get("cells_seen"), number(i.get("cells_per_s")), number(i.get("tables_per_s")), number((i.get("preflight") or {}).get("cells_per_s")),
                         i.get("reserved_mb"), i.get("peak_mb"), i.get("micro"), number(i.get("eval_overhead")), i.get("ended_by"), i.get("wall_s")])
    by_cond = [[c] + [cell.get((v, c, SC.ALL, SC.ALL)) for v in names + bnames] for c in conds]
    strata = {key: [[c, x.replace(SC.FINE_PREFIX, "")] + [cell.get((v, c, SC.ALL, x)) for v in names + bnames] for c in conds for x in labels
                    if any((v, c, SC.ALL, x) in cell for v in names + bnames)] for key, labels in (("fine", SC.FINE), ("other_rows", SC.OTHER_ROWS))}
    gname = lambda g: groups[int(g)] if g.isdigit() and int(g) < len(groups) else g                # noqa: E731
    by_group = [[gname(g)] + [cell.get((v, c, g, SC.ALL)) for v in names + bnames]
                for c in [s["main"]] for g in sorted({r[3] for r in fin if r[3] != SC.ALL}, key=lambda g: (not g.isdigit(), int(g) if g.isdigit() else 0, g))]
    md = ["# Relational line: learning curves of the reference trunk", "",
          "Bits per codon of the hidden residues of the examined row's target block, on held-out tables. Development diagnosis:", ""]
    md += ["- " + c for c in CAVEATS]
    md += ["", "## Arms: throughput and memory as measured", ""]
    md += md_table(["arm", "status", "parameters", "peak lr", "lr reduced by the guard", "steps", "tables seen", "cells seen", "cells/s", "tables/s",
                    "cells/s (preflight, alone)", "GPU MB reserved", "GPU MB allocated", "tables per micro-batch", "evaluation share of training time",
                    "ended by", "wall s"], arm_rows)
    md += ["", "## Final evaluation, all held-out tables: bits per codon by condition (mean of the groups)", ""] + md_table(["condition"] + names + bnames, by_cond)
    if bnames:
        md += ["", "Arms and baselines scored the same cells (digests of the scored cells): %s." % (
            ", ".join("%s %s" % (c, "yes" if ok else "NO") for c, ok in same.items()) or "not known (no digests beside the baseline rows)")]
    md += ["", "## By the identity of the nearest row at the target locus (ten bins)", ""] + md_table(["condition", "nearest identity"] + names + bnames, strata["fine"])
    md += ["", "## By the number of other rows of the table", ""] + md_table(["condition", "other rows"] + names + bnames, strata["other_rows"])
    md += ["", "## By group, condition %s" % s["main"], ""] + md_table(["group"] + names + bnames, by_group)
    md += ["", "## Curves at equal tables seen (condition %s; the first %s held-out tables of every group at every node)" % (
        s["main"], "/".join(sorted({str(arms[a]["info"].get("eval_subset") or "") for a in names} - {""})) or "?"), ""] + (md_table(head, grid) if grid else ["no curve"])
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
                  by_stratum={k: dict(columns=["condition", "stratum"] + names + bnames, rows=v) for k, v in strata.items()},
                  by_group=dict(columns=["group"] + names + bnames, rows=by_group))
    for name in ("curves.tsv", "aligned.tsv", "final.tsv", "report.md"):
        st.output(name.replace(".", "_"), os.path.join(folder, name))
    p = os.path.join(folder, "report.json")
    write_json(p, report)                                                 # the rule's target last
    st.output("report", p)
    st.extra.update(arms=len(names), baselines=bnames, conditions=conds, same_items=same)
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
                       dict(problems=problems[:20], arms={a: x["final"] for a, x in rep["arms"].items()}, baselines=sorted(rep["baselines"]),
                            same_items=rep.get("same_items"))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        rep = json.load(f)
    with open(os.path.join(run_dir, OUT, "report.md")) as f:
        md = f.read().splitlines()
    small = dict(main=rep["main"], caveats=rep["caveats"], by_condition=rep["by_condition"], aligned=rep["aligned"], same_items=rep.get("same_items"),
                 arms={a: dict(status=x["status"], final=x["final"], tables_seen=x["info"].get("tables_seen"), cells_seen=x["info"].get("cells_seen"),
                               cells_per_s=x["info"].get("cells_per_s"), reserved_mb=x["info"].get("reserved_mb"), params=x["info"].get("params"),
                               lr=x["info"].get("lr")) for a, x in rep["arms"].items()},
                 baselines=rep["baselines"])
    return dict(json=small, md=["####" + line[2:] if line.startswith("## ") else line for line in md[1:]])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w6_report") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
