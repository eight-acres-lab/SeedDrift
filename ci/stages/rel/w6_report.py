"""Stage rel_w6_report (package W6; pipeline rel_traineval): the learning curves of the arms on the axis of tables seen,
the readings today's ruling asks for, and the classical baselines on the same held-out tables. `python -m
ci.stages.rel.w6_report RUN_DIR`. CPU only, numpy and the standard library; it reads result rows (the columns of the
contract), the model index and each arm's preflight record, nothing of a model.

Profile block rel_w6_report (all optional):
  fits       [run directory, ...]: passed runs whose stage rel_w6_fit trained arms (of this machine, or copied to it
             with record.json, stages/ and data/rel_w6/fit/); their rows are set beside those of this run (which comes
             first when its own rel_w6_fit ran)
  baselines  [{run, path}]: result tables of passed runs (the classical baselines of package W5: path
             data/rel_w5/baselines/rows.tsv; its results.json beside it gives the digest of the cells it scored)
  main       the condition of the headline curve (default scattered; an arm trained without relatives is read under
             the same condition without relatives)
Readings (each also in report.json):
  arms          parameters, learning rate, steps, tables seen, passes over the training tables, throughput, memory
  headline      the final evaluation by the identity of the nearest row (ten bins, the bin without other rows left
                out), every condition, arms and baselines; then the eukaryotic groups apart from the others
  relatives     what relatives are worth, twice: an arm without relatives at its best node against the arm of the
                same size with relatives at its end, on all held-out tables; and the two at the same nodes, on the
                tables of the nodes
  scale         the arms with relatives at the last node all of them reached: "bigger is better" when the bits fall
                with the size under both scattered and whole_gene, else "cannot be read"
  re-pairing    whole_gene against whole_gene+repaired on the tables that can be re-paired (paired)
  check         on the tables without other rows an arm's scattered and scattered+no_relatives bits must agree, and
                its whole_gene and whole_gene+no_relatives bits
  curves        every evaluation of every arm at the nodes of tables seen (the same nodes for every arm)
Outputs under data/rel_w6/report/: curves.tsv, aligned.tsv, final.tsv, report.md and, written last, report.json.
Gate G4_rel_w6_report: every trained arm has a curve of at least two evaluations and final bits under its conditions,
the report states its caveats, and no digest comparison with the baselines failed.
"""
from __future__ import annotations

import json
import os
import re

import yaml

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
NO_REL = "+" + cond.NO_RELATIVES
NONE_BIN = SC.FINE[-1]
WITH_RELATIVES = getattr(SC, "RELATIVES", (None,))[0]               # the stratum of the tables with a relative on the target
WITHOUT_RELATIVES = getattr(SC, "RELATIVES", (None, None))[1]
BLOCK_OF = getattr(SC, "block_of", lambda g: "eukaryotes" if str(g).startswith("eukaryota.") else "prokaryotes, plasmids, viruses")
BLOCKS = getattr(SC, "BLOCKS", ("eukaryotes", "prokaryotes, plasmids, viruses"))
CAVEATS = [
    "Lower bits are better. Under the stored placement the relatives of a hidden whole gene are placed by its hidden residues, so "
    "the whole_gene bits are biased low (optimistic), the value of relatives under that condition is overestimated and the cost of "
    "re-pairing underestimated; the scattered bits are biased low only slightly (not measured); the two conditions without "
    "relatives have no such leak.",
    "The direction of the remaining bias is unknown: rows were drawn by similarity, which raises what relatives are worth; whole "
    "held-out clusters have no close relatives and the 15 other rows are drawn at random, which lowers it; close relatives are not "
    "covered at all.",
    "The strata are the identity of the nearest row over the aligned part of the target block: short, very similar fragments enter "
    "the top bins, so a flat top bin does not show that close relatives are of no use.",
    "The classical baselines are weak (no codon substitution model, no amino-acid model): they are a lower bound on what relatives "
    "are worth, not an estimate of it.",
    "Training masks: the three schemes (scattered 15% on every row, one span of one row, the whole target gene of row 0) are drawn "
    "with equal chances; the loss is the mean over every hidden cell of the four tables of a step. About half of the loss of the arm "
    "without relatives falls on whole genes and about a sixth of that of the arms with relatives: they optimise different mixtures.",
    "One seed per arm, no noise floor, learning rates set by width and not tuned (a warm-up over 1,000 steps, then constant): only "
    "large effects in one direction count. Throughput and memory were measured on GPUs that other users' processes shared.",
    "The eukaryotic groups hold only the first 1,500 genes of each genome; they are reported apart from the prokaryotes, plasmids "
    "and viruses, and the mean over all groups is not the headline.",
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


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def variant_of(st, path):
    """relatives and size of an arm's variant file (from this commit; the model index names it and its hash)."""
    try:
        with open(st.repo(path)) as f:
            v = yaml.safe_load(f)
        return dict(relatives=bool(v.get("relatives", True)), size=v.get("size"))
    except (OSError, ValueError, AttributeError):
        return dict(relatives=True, size=None)


def collect(st, run_dir, s):
    """Rows, arms (model index entries with their run and variant), baseline rows and digests, sources, group names."""
    fits = ([run_dir] if os.path.exists(os.path.join(run_dir, FIT, "models.json")) else []) + [f for f in s["fits"] if os.path.realpath(f) != os.path.realpath(run_dir)]
    if not fits:
        raise SystemExit("rel_w6_report: no fit to report (this run trained nothing and rel_w6_report.fits names no run)")
    rows, arms, sources, groups = [], {}, [], []
    for f in fits:
        if f == run_dir:
            doc = read_json(os.path.join(f, FIT, "models.json"))
        else:
            try:
                doc = handoff.from_run(f, os.path.join(FIT, "models.json"))
            except (handoff.HandoffError, OSError, ValueError) as err:
                raise SystemExit("rel_w6_report: %s: %s" % (f, err))
        for v in doc["variants"]:
            if v["name"] in arms:
                raise SystemExit("rel_w6_report: arm %s is in two of the runs reported" % v["name"])
            pre = read_json(os.path.join(f, v["info"]["folder"], "preflight.json")) or {}
            arms[v["name"]] = dict(v, run=os.path.basename(f), variant=variant_of(st, v["variant_file"]), train_tables=pre.get("train_tables"))
        rows += SC.read_rows(os.path.join(f, FIT, "results.tsv"))
        groups = groups or list(doc.get("groups") or [])
        sources.append(dict(run=os.path.basename(f), models_sha256=doc["sha256"], tablesets=doc["tablesets"]))
    base, digests = [], {}
    for b in s["baselines"]:
        p = os.path.join(b["run"], b["path"])
        status = (read_json(os.path.join(b["run"], "record.json")) or {}).get("status")
        if status != "passed":
            raise SystemExit("rel_w6_report: the baselines' run %s is %s, not passed" % (b["run"], status))
        got = SC.read_rows(p)
        base += got
        items = (read_json(os.path.join(os.path.dirname(p), "results.json")) or {}).get("items") or {}
        digests.update({c: x.get("sha256") for c, x in items.items() if isinstance(x, dict)})
        sources.append(dict(run=os.path.basename(b["run"]), baselines=b["path"], sha256=sha256_file(p, use_sidecar=False), rows=len(got)))
    return rows, arms, base, digests, sources, groups


def point_of(r):
    """('node', tables) / ('final', None) / ('best', tables) of a result row of an arm; None for another row."""
    m = NODE_RX.match(r["weights"])
    if m:
        return "node", int(m.group(2))
    if r["weights"].endswith("@final"):
        return "final", None
    if r["weights"].endswith("@best"):
        t = re.match(r"^tables=(\d+)", r.get("budget") or "")
        return "best", int(t.group(1)) if t else None
    return None


def index_rows(rows, arms, base):
    """{(arm, point, tables, condition, group, stratum): bits}; baselines under point 'final'."""
    out = {}
    for r in rows:
        if r["variant"] not in arms:
            continue
        p = point_of(r)
        if p is not None:
            out[(r["variant"], p[0], p[1], r["condition"], r["group"], r["stratum"])] = float(r["bits"])
    for r in base:
        if number(r["bits"]) is not None:
            out[(r["variant"], "final", None, r["condition"], r["group"], r["stratum"])] = float(r["bits"])
    return out


def same_items(arms, digests):
    """{condition: whether every trained arm scored, in its final evaluation, the cells the baselines scored}; only
    conditions for which both digests are known."""
    out = {}
    for c in cond.CONDITIONS:
        mine = {(a["info"].get("final_items") or {}).get(c) for a in arms.values() if a["status"] == "ok"} - {None}
        if mine and digests.get(c):
            out[c] = mine == {digests[c]}
    return out


def tsv(path, header, table):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for row in table:
            f.write("\t".join("" if x is None else ("%.6f" % x if isinstance(x, float) else str(x)) for x in row) + "\n")


def md_table(header, table):
    fmt = lambda x: "" if x is None else ("%.3f" % x if isinstance(x, float) else ("{:,}".format(x) if isinstance(x, int) else str(x)))      # noqa: E731
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(fmt(x) for x in row) + " |" for row in table]


def diff(a, b):
    return None if a is None or b is None else a - b


def build(st, cfg, run_dir):
    s = settings(cfg)
    rows, arms, base, digests, sources, groups = collect(st, run_dir, s)
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    bits = index_rows(rows, arms, base)
    ok = sorted(a for a in arms if arms[a]["status"] == "ok")
    rel = [a for a in ok if arms[a]["variant"]["relatives"]]
    norel = [a for a in ok if not arms[a]["variant"]["relatives"]]
    bnames = sorted({r["variant"] for r in base if number(r["bits"]) is not None})
    names = sorted(arms)
    cols = names + bnames
    F = lambda v, c, g=SC.ALL, x=SC.ALL: bits.get((v, "final", None, c, g, x))                     # noqa: E731
    N = lambda v, t, c, g=SC.ALL, x=SC.ALL: bits.get((v, "node", t, c, g, x))                       # noqa: E731
    conds = [c for c in cond.CONDITIONS if any(F(v, c) is not None for v in cols)]
    reading = {a: (s["main"] if a in rel else s["main"] + NO_REL) for a in names}
    nodes = {a: sorted({k[2] for k in bits if k[0] == a and k[1] == "node"}) for a in names}
    same = same_items(arms, digests)

    # arms
    arm_rows = []
    for a in names:
        i = arms[a]["info"]
        passes = (i.get("tables_seen") or 0) / arms[a]["train_tables"] if arms[a]["train_tables"] else None
        arm_rows.append([a, arms[a]["run"], arms[a]["status"], "yes" if arms[a]["variant"]["relatives"] else "no", i.get("params"), i.get("lr"),
                         "yes" if i.get("lr_reduced") else "no", i.get("steps"), i.get("tables_seen"), passes, number(i.get("cells_per_s")),
                         number(i.get("tables_per_s")), i.get("reserved_mb"), i.get("micro"), number(i.get("eval_overhead")), i.get("ended_by"),
                         (i.get("row_key_mask") or {}).get("state"), number(i.get("alone_share_seen"))])
    # the share of held-out tables without other rows (from the rows of a final evaluation)
    n_of = {(r["stratum"]): int(r["n_items"]) for r in rows if r["weights"].endswith("@final") and r["condition"] == cond.WHOLE_GENE and r["group"] == SC.ALL}
    held_alone = n_of[WITHOUT_RELATIVES] / n_of[SC.ALL] if WITHOUT_RELATIVES in n_of and n_of.get(SC.ALL) else None
    trained_alone = {a: number(arms[a]["info"].get("alone_share_seen")) for a in rel}
    caveats = CAVEATS + ["The arms with relatives drew tables without other rows at most at 5%% of a group's draws (drawn: %s); %s of the held-out "
                         "tables have none. On the bin without other rows and under the two conditions without relatives these arms are therefore "
                         "read on inputs they saw little of; what relatives are worth is reported by bin, without that bin." % (
                             ", ".join("%s %s" % (a, "%.3f" % x if x is not None else "?") for a, x in trained_alone.items()) or "no arm",
                             "%.1f%%" % (100 * held_alone) if held_alone is not None else "an unknown share")]
    # headline: by stratum (ten bins, without the bin of tables without other rows), then the eukaryotic groups apart
    headline = [[c, x.replace(SC.FINE_PREFIX, "")] + [F(v, c, SC.ALL, x) for v in cols] for c in conds for x in SC.FINE
                if x != NONE_BIN and any(F(v, c, SC.ALL, x) is not None for v in cols)]
    group_set = sorted({k[4] for k in bits if k[4] != SC.ALL})
    in_block = {b: [g for g in group_set if BLOCK_OF(g) == b] for b in BLOCKS}

    def mean_of(v, c, gs, x=SC.ALL):
        xs = [F(v, c, g, x) for g in gs]
        xs = [y for y in xs if y is not None]
        return sum(xs) / len(xs) if xs else None
    # the headline: the tables with a relative on the target block, the two blocks apart (mean of the groups of a block)
    lead = [[c, "%s (%d groups)" % (b, len(gs))] + [mean_of(v, c, gs, WITH_RELATIVES) for v in cols] for c in conds for b, gs in in_block.items()
            if gs and WITH_RELATIVES is not None]
    blocks = [[c, "%s (%d groups)" % (b, len(gs))] + [mean_of(v, c, gs) for v in cols] for c in conds for b, gs in in_block.items() if gs]
    by_group = [[g, BLOCK_OF(g)] + [F(v, s["main"] if v not in norel else s["main"] + NO_REL, g) for v in cols] for g in group_set]

    # what relatives are worth, twice
    relatives = []
    for b in norel:
        size = arms[b]["variant"]["size"]
        partners = [a for a in rel if arms[a]["variant"]["size"] == size] or rel
        best = arms[b]["info"].get("best") or {}
        has_best = any(k[0] == b and k[1] == "best" for k in bits)
        x0 = WITH_RELATIVES or SC.ALL                                  # the tables with a relative on the target block
        for a in partners:
            for c in (cond.SCATTERED, cond.WHOLE_GENE):
                def without(x, c=c):
                    got = bits.get((b, "best", best.get("tables_seen"), c + NO_REL, SC.ALL, x)) if has_best else None
                    return got if got is not None else F(b, c + NO_REL, SC.ALL, x)
                common_nodes = sorted(set(nodes.get(a, [])) & set(nodes.get(b, [])) - {0})
                last = common_nodes[-1] if common_nodes else None
                relatives.append(dict(with_relatives=a, without=b, condition=c, tables=x0,
                                      without_at="best node (tables %s)" % best.get("tables_seen") if has_best else "end",
                                      bits_without=without(x0), bits_with_end=F(a, c, SC.ALL, x0), worth=diff(without(x0), F(a, c, SC.ALL, x0)),
                                      same_model_rows_removed=diff(F(a, c + NO_REL, SC.ALL, x0), F(a, c, SC.ALL, x0)), node=last,
                                      worth_at_node=diff(N(b, last, c + NO_REL, SC.ALL, x0), N(a, last, c, SC.ALL, x0)) if last else None,
                                      per_node=[[t, N(b, t, c + NO_REL, SC.ALL, x0), N(a, t, c, SC.ALL, x0), diff(N(b, t, c + NO_REL, SC.ALL, x0), N(a, t, c, SC.ALL, x0))]
                                                for t in common_nodes],
                                      by_bin=[[x.replace(SC.FINE_PREFIX, ""), without(x), F(a, c, SC.ALL, x), diff(without(x), F(a, c, SC.ALL, x))]
                                              for x in SC.FINE if x != NONE_BIN and (without(x) is not None or F(a, c, SC.ALL, x) is not None)]))
    # scale: the arms with relatives at the last node all of them reached
    scale = None
    if len(rel) >= 2:
        common_nodes = sorted(set.intersection(*[set(nodes.get(a, [])) for a in rel]) - {0})
        if common_nodes:
            t = common_nodes[-1]
            order = sorted(rel, key=lambda a: arms[a]["info"].get("params") or 0)
            falls = all(all(N(order[j + 1], t, c) is not None and N(order[j], t, c) is not None and N(order[j + 1], t, c) < N(order[j], t, c)
                            for j in range(len(order) - 1)) for c in (cond.SCATTERED, cond.WHOLE_GENE))
            scale = dict(node=t, arms=order, bits={a: {c: N(a, t, c) for c in (cond.SCATTERED, cond.WHOLE_GENE)} for a in order},
                         verdict="bigger is better" if falls else "cannot be read")
    # re-pairing, on the tables that can be re-paired
    two = SC.OTHER_ROWS[2]
    repair = [[v, F(v, cond.WHOLE_GENE, SC.ALL, two), F(v, cond.WHOLE_GENE + "+" + cond.REPAIRED, SC.ALL, two),
               diff(F(v, cond.WHOLE_GENE + "+" + cond.REPAIRED, SC.ALL, two), F(v, cond.WHOLE_GENE, SC.ALL, two))] for v in rel + bnames
              if F(v, cond.WHOLE_GENE + "+" + cond.REPAIRED, SC.ALL, two) is not None]
    # the free check: tables without other rows
    zero = SC.OTHER_ROWS[0]
    check = [[a, c, F(a, c, SC.ALL, zero), F(a, c + NO_REL, SC.ALL, zero), diff(F(a, c, SC.ALL, zero), F(a, c + NO_REL, SC.ALL, zero))]
             for a in rel for c in (cond.SCATTERED, cond.WHOLE_GENE)]
    # curves
    curve_rows, all_nodes = [], sorted({t for a in names for t in nodes.get(a, [])})
    for a in names:
        for t in nodes.get(a, []):
            curve_rows.append([a, "node", t] + [N(a, t, c) for c in conds])
        curve_rows.append([a, "final", arms[a]["info"].get("tables_seen")] + [F(a, c) for c in conds])
    tsv(os.path.join(folder, "curves.tsv"), ["arm", "point", "tables_seen"] + conds, curve_rows)
    head = ["tables_seen"] + ["%s (%s)" % (a, reading[a]) for a in names]
    grid = [[t] + [N(a, t, reading[a]) for a in names] for t in all_nodes]
    tsv(os.path.join(folder, "aligned.tsv"), head, grid)
    fin = [["model" if v in arms else "baseline", v, c, g, x, bits[(v, p, t, c, g, x)]] for (v, p, t, c, g, x) in sorted(bits, key=str) if p == "final"]
    tsv(os.path.join(folder, "final.tsv"), ["kind", "variant", "condition", "group", "stratum", "bits"], fin)

    md = ["# Relational line: the reference trunk's first curves", "",
          "Bits per codon of the hidden residues of the examined row's target block, on held-out tables (lower is better). "
          "Development diagnosis. What the numbers do not show:", ""] + ["- " + c for c in caveats]
    md += ["", "## Arms", ""] + md_table(["arm", "run", "status", "relatives", "parameters", "peak lr", "lr reduced by the preflight", "steps", "tables seen",
                                         "passes over the training tables", "cells/s", "tables/s", "GPU MB reserved", "tables per micro-batch",
                                         "evaluation share of training time", "ended by", "masked row attention",
                                         "share of training tables without other rows"], arm_rows)
    md += ["", "## Headline: final evaluation, the held-out tables with a relative on the target block, the two blocks apart (mean of the groups)", ""]
    md += md_table(["condition", "block"] + cols, lead) if lead else ["(the rows carry no stratum of tables with relatives)"]
    md += ["", "## By the identity of the nearest row (mean of the groups; the bin without other rows left out)", ""]
    md += md_table(["condition", "nearest identity"] + cols, headline)
    if bnames:
        md += ["", "Arms and baselines scored the same cells (digests of the scored cells): %s." % (
            ", ".join("%s %s" % (c, "yes" if v else "NO") for c, v in same.items()) or "not known (no digests beside the baseline rows)")]
    md += ["", "### The two blocks over all their tables (mean of the groups, all strata)", ""] + md_table(["condition", "block"] + cols, blocks)
    md += ["", "## What relatives are worth", "",
           "Bits without relatives minus bits with them (positive: relatives help), on the held-out tables with a relative on the target block. "
           "First at the end, on all such tables: the arm without relatives at its best node against the arm of the same size with relatives at "
           "its end (conservative while the curve with relatives still falls); then at the same node, on the tables of the nodes. The column "
           "'same model' removes the rows from the arm with relatives at evaluation. Then by bin of the nearest identity, without the bin of no "
           "other rows.", ""]
    md += md_table(["with", "without", "condition", "without read at", "bits without", "bits with (end)", "worth", "same model, rows removed",
                    "last common node", "worth at that node"],
                   [[r["with_relatives"], r["without"], r["condition"], r["without_at"], r["bits_without"], r["bits_with_end"], r["worth"],
                     r["same_model_rows_removed"], r["node"], r["worth_at_node"]] for r in relatives])
    for r in relatives:
        md += ["", "%s against %s, %s, by bin (end):" % (r["without"], r["with_relatives"], r["condition"]), ""]
        md += md_table(["nearest identity", "bits without", "bits with", "worth"], r["by_bin"])
    md += ["", "## Scale", ""]
    if scale:
        md += ["At the last node every arm with relatives reached (%s tables seen): %s." % ("{:,}".format(scale["node"]), scale["verdict"]), ""]
        md += md_table(["arm", "parameters", "scattered", "whole_gene"], [[a, arms[a]["info"].get("params"), scale["bits"][a][cond.SCATTERED],
                                                                            scale["bits"][a][cond.WHOLE_GENE]] for a in scale["arms"]])
        md += ["", "The rule: bigger is better only when the bits fall with every step in size under both conditions; otherwise the scale cannot be read."]
    else:
        md += ["Fewer than two arms with relatives, or no node all of them reached."]
    md += ["", "## The cost of re-pairing, on the tables that can be re-paired (two or more other rows), paired", ""]
    md += md_table(["arm or baseline", "whole_gene", "whole_gene+repaired", "cost"], repair)
    md += ["", "No difference would not show that the relation blocks go unused."]
    md += ["", "## A check: tables without other rows", "",
           "There an arm reads the same table with and without relatives, so the two columns must agree (up to rounding).", ""]
    md += md_table(["arm", "condition", "with relatives", "without", "difference"], check)
    md += ["", "## By group (final evaluation; condition %s, an arm without relatives under %s)" % (s["main"], s["main"] + NO_REL), ""]
    md += md_table(["group", "block"] + cols, by_group)
    md += ["", "## Curves at equal tables seen (the first held-out tables of every group at every node)", ""] + (md_table(head, grid) if grid else ["no curve"])
    for a in names:
        if arms[a]["status"] != "ok":
            md += ["", "%s failed: `%s`" % (a, " ".join((arms[a].get("error") or "").split())[-600:])]
    with open(os.path.join(folder, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    report = dict(main=s["main"], caveats=caveats, sources=sources, conditions=conds, groups=groups, same_items=same, heldout_alone=held_alone,
                  trained_alone=trained_alone,
                  arms={a: dict(status=arms[a]["status"], run=arms[a]["run"], relatives=arms[a]["variant"]["relatives"], info=arms[a]["info"],
                                evaluations=len(nodes.get(a, [])) + (1 if any(F(a, c) is not None for c in conds) else 0),
                                final={c: F(a, c) for c in conds}) for a in names},
                  baselines={b: {c: F(b, c) for c in conds} for b in bnames},
                  lead=dict(columns=["condition", "block"] + cols, rows=lead), headline=dict(columns=["condition", "stratum"] + cols, rows=headline),
                  blocks=dict(columns=["condition", "block"] + cols, rows=blocks),
                  relatives=relatives, scale=scale, repair=dict(columns=["variant", "whole_gene", "whole_gene+repaired", "cost"], rows=repair),
                  check=dict(columns=["arm", "condition", "with", "without", "difference"], rows=check),
                  aligned=dict(columns=head, rows=grid), by_group=dict(columns=["group", "block"] + cols, rows=by_group))
    for name in ("curves.tsv", "aligned.tsv", "final.tsv", "report.md"):
        st.output(name.replace(".", "_"), os.path.join(folder, name))
    p = os.path.join(folder, "report.json")
    write_json(p, report)                                                 # the rule's target last
    st.output("report", p)
    st.extra.update(arms=len(names), baselines=bnames, conditions=conds, same_items=same, scale=(scale or {}).get("verdict"))
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
    if len(rep.get("caveats") or []) < 5:
        problems.append(dict(error="the report does not state what it does not show"))
    if not all((rep.get("same_items") or {}).values()):
        problems.append(dict(error="arms and baselines were not scored on the same cells", same_items=rep.get("same_items")))
    require(GateResult("G4_rel_w6_report", "fail" if problems else "pass",
                       dict(problems=problems[:20], arms={a: x["final"] for a, x in rep["arms"].items()}, baselines=sorted(rep["baselines"]),
                            same_items=rep.get("same_items"), scale=(rep.get("scale") or {}).get("verdict"))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        rep = json.load(f)
    with open(os.path.join(run_dir, OUT, "report.md")) as f:
        md = f.read().splitlines()
    small = dict(main=rep["main"], same_items=rep.get("same_items"), scale=rep.get("scale"), lead=rep.get("lead"), headline=rep["headline"],
                 relatives=[{k: r[k] for k in r if k not in ("per_node", "by_bin")} for r in rep["relatives"]],
                 arms={a: dict(status=x["status"], final=x["final"], tables_seen=x["info"].get("tables_seen"), cells_seen=x["info"].get("cells_seen"),
                               cells_per_s=x["info"].get("cells_per_s"), reserved_mb=x["info"].get("reserved_mb"), params=x["info"].get("params"),
                               lr=x["info"].get("lr")) for a, x in rep["arms"].items()},
                 baselines=rep["baselines"])
    if len(json.dumps(small, default=str)) > 18000:
        small.pop("headline")
    return dict(json=small, md=["####" + line[2:] if line.startswith("## ") else line for line in md[1:]])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w6_report") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
