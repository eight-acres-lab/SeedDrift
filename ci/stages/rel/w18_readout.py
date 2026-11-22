"""Stage rel_w18_readout (package W18; pipeline rel_traineval, after rel_w6_eval): the readouts of the HGT capability
check and their AUCs (protocol w18-v0), from the per-cell files the evaluation of a ruler set wrote. `python -m
ci.stages.rel.w18_readout RUN_DIR`. CPU only, nothing is trained, no model is run.

Inputs: the ruler set (rel_w18_readout.tables: {run, path, sha256}, taken through handoff.from_run, so the run passed and
the set carries its sealed record) and the per-cell files of the evaluation (data/rel_w6/eval/<arm>/ruler_cells.npz of
rel_w18_readout.eval_run, default this run; each file listed with its sha256 in that run's record of stage rel_w6_eval).
Every file must name the ruler set's manifest, and the digest of what it scored must equal the plan of
seeddrift/train/rel/readouts/w18_passes.py recomputed here without a model: a file that scored another plan is refused.

Arms by their files: an arm with relatives and trained weights gives source shift, its undefined-at-top version, the
two-block gain (with its calibration) and abstention; with presence passes the same again on them; an arm without
relatives gives the composition readout; weights "untrained" mark the untrained control of the same readouts. The
classical controls come from the tables. AUCs: seeddrift/train/rel/readouts/w18_readouts.py (implant against own, the
primary comparison, and against original; per group, then the readable groups of a block; genus bootstrap), with the
paired difference of every model readout against the best-similarity rule, the untrained model and the presence passes
of the same readout, and of composition against the host codon model.

Profile block rel_w18_readout: tables, eval_run (optional), arms (optional: the arms to read, default every file
found), reps (2,000), seed (20261124).
Outputs under data/rel_w18/readout/: scores.tsv (every score of every table), auc.tsv (the AUC rows), groups.tsv (the
all-stratum AUC per group), report.md, report.json (written last).
Gate G4_rel_w18_readout: every per-cell file read was checked, at least one arm with relatives was read, every AUC is in
[0, 1], and the report names the run and the tables of every number.
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from seeddrift.train.rel import handoff
from seeddrift.train.rel.evalh import score as SC
from seeddrift.train.rel.evalh import shards
from seeddrift.train.rel.readouts import w18_passes as P
from seeddrift.train.rel.readouts import w18_readouts as R

OUT = os.path.join("data", "rel_w18", "readout")
EVAL = os.path.join("data", "rel_w6", "eval")
KEYS = {"tables", "eval_run", "arms", "reps", "seed"}
STATUS = ("preliminary: tables of version 0 (rows chosen among the 64 nearest genomes of a gene) and the stored layout, which places relatives by the "
          "hidden residues; the counting reading is on tables of version 1, the leak-free reading with a model retrained leak-free (protocol w18-v0). "
          "A ruler, not a transfer label: these numbers choose no arm, head, threshold or readout.")


def settings(cfg) -> dict:
    c = cfg.get("rel_w18_readout") or {}
    unknown = set(c) - KEYS
    t = c.get("tables") or {}
    if unknown or not t.get("run") or not t.get("path"):
        raise SystemExit("rel_w18_readout: needs tables {run, path[, sha256]}; unknown keys %s" % sorted(unknown))
    return dict(tables=dict(run=str(t["run"]), path=str(t["path"]), sha256=t.get("sha256")), eval_run=c.get("eval_run"),
                arms=[str(a) for a in c.get("arms") or []], reps=int(c.get("reps", R.BOOT)), seed=int(c.get("seed", R.BOOT_SEED)))


def ruler(s):
    """(manifest, TableSet, W18 meta, table fields, group names) of the ruler set."""
    doc = handoff.from_run(s["tables"]["run"], s["tables"]["path"], s["tables"]["sha256"])
    if doc.get("kind") != "ruler":
        raise SystemExit("rel_w18_readout: the table set is kind %r, not ruler" % doc.get("kind"))
    sets = shards.by_split(s["tables"]["run"], [doc])
    if list(sets) != [1]:
        raise SystemExit("rel_w18_readout: a ruler set of held-out tables only (splits %s)" % sorted(sets))
    ts = sets[1]
    meta = P.load_meta(s["tables"]["run"], doc)
    fields = ts.load(0, ts.tables, ("tok", "n", "col", "kind", "frame_len", "group", "identity", "genome"))
    return doc, ts, meta, fields, shards.group_names(s["tables"]["run"], doc)


def cell_files(s, run_dir):
    """[(arm, path, sha256)] of the per-cell files of the evaluation run, each listed in its record of stage rel_w6_eval."""
    ev = s["eval_run"] or run_dir
    root = os.path.join(ev, EVAL)
    rec = os.path.join(ev, "stages", "rel_w6_eval.json")
    listed = {}
    if os.path.exists(rec):
        with open(rec) as f:
            for o in json.load(f).get("outputs") or []:
                listed[os.path.normpath(os.path.join(ev, o["path"]) if not os.path.isabs(o["path"]) else o["path"])] = o.get("sha256")
    out = []
    for arm in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        p = os.path.join(root, arm, P.RULER_CELLS)
        if not os.path.exists(p) or (s["arms"] and arm not in s["arms"]):
            continue
        sha = sha256_file(p, use_sidecar=False)
        if listed.get(os.path.normpath(p)) != sha:
            raise SystemExit("rel_w18_readout: %s is not an output the evaluation run's record of stage rel_w6_eval lists with this content" % p)
        out.append((arm, p, sha))
    missing = sorted(set(s["arms"]) - {a for a, _, _ in out})
    if missing or not out:
        raise SystemExit("rel_w18_readout: no per-cell file for arms %s under %s" % (missing or "any", root))
    return out


def donor_block_same(meta, fields, groups, genomes):
    a = np.array([SC.block_of(groups[int(g)]) for g in fields["group"]])
    d = np.array([SC.block_of(genomes[int(x)]["group"]) for x in meta["donor"]])
    return a == d


def build(st, cfg, run_dir):
    s, t0 = settings(cfg), time.time()
    say = lambda text: print("rel_w18_readout: " + text, flush=True)          # noqa: E731
    doc, ts, meta, fields, groups = ruler(s)
    with open(os.path.join(s["tables"]["run"], doc["index"])) as f:
        index = json.load(f)
    genomes = index["genomes"]
    n = ts.tables
    arms, scores, labels = [], {}, {}
    plans = {}
    for arm, path, sha in cell_files(s, run_dir):
        cells, head = P.load_cells(path)
        if head["tableset_sha256"] != doc["sha256"]:
            raise SystemExit("rel_w18_readout: %s scored another table set (%s)" % (path, head["tableset_sha256"]))
        key = (bool(head["relatives"]), bool(head["presence"]))
        if key not in plans:
            plans[key] = P.plan_digest(P.tables_of(ts, meta), *key)
        if cells["digest"] != plans[key][0]:
            raise SystemExit("rel_w18_readout: %s scored another plan than the passes of this commit" % path)
        untrained = head["weights"] == "untrained"
        tag = ("untrained:" if untrained else "") + head["arm"]
        info = dict(arm=head["arm"], folder=arm, file=os.path.relpath(path, run_dir) if path.startswith(run_dir) else path, sha256=sha, weights=head["weights"],
                    relatives=head["relatives"], presence=head["presence"], layout=head["layout"], render_version=head["render_version"],
                    eval_run=head["run"], eval_commit=head["commit"], cells=int(len(cells["bits"])), passes=head["passes"])
        if head["relatives"]:
            r = R.readouts(cells, meta)
            info["calibration"] = R.calibration(cells, meta)
            info["abstain"] = {v: float(np.nanmean(r["abstain"][meta["variant"] == k])) for k, v in enumerate(("implant", "original", "own"))}
            info["source_shift_defined"] = {v: float(np.isfinite(r["source_shift"][meta["variant"] == k]).mean()) for k, v in enumerate(("implant", "original", "own"))}
            for k in ("source_shift", "source_shift_top", "gain"):
                scores[(tag, k)] = r[k]
            if head["presence"]:
                rp = R.readouts(cells, meta, prefix=P.PRESENCE)
                for k in ("source_shift", "source_shift_top", "gain"):
                    scores[("presence:" + tag, k)] = rp[k]
        else:
            c = R.composition(cells, meta)
            for k in ("composition", "composition_rel"):
                scores[(tag, k)] = c[k]
        labels[tag] = info
        arms.append(info)
        say("%s (%s): %d cells read" % (arm, head["weights"], info["cells"]))
    ctl = R.controls(fields, meta)
    for k, v in ctl.items():
        scores[("control", k)] = v
    strata = R.stratum_masks(meta, fields["identity"], donor_block_same(meta, fields, groups, genomes))
    boot = R.Bootstrap(fields["group"], meta["genus_a"], reps=s["reps"], seed=s["seed"])
    gtab = fields["group"].astype(np.int64)
    rows, per_group = [], []
    for (label, name), v in sorted(scores.items()):
        ref = {}
        if label != "control":
            base = label.split(":", 1)[-1]
            ref["best_similarity"] = ctl["best_similarity"]
            if name.startswith("composition"):
                ref["host_codon"] = ctl["host_codon_rel" if name.endswith("_rel") else "host_codon"]
            for other in ("untrained:" + base, "presence:" + base):
                if (other, name) in scores and other != label:
                    ref[other.split(":")[0]] = scores[(other, name)]
        rows += R.auc_rows({name: v}, meta, gtab, groups, strata, boot, label=label, reference=ref)
        per_group += R.per_group_rows({name: v}, meta, gtab, groups, label=label)
    say("%d AUC rows, %.0f s" % (len(rows), time.time() - t0))
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    sp = os.path.join(folder, "scores.tsv")
    with open(sp, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        keys = sorted(scores)
        w.writerow(["table", "triple", "variant", "group", "donor_tier", "on_target"] + ["%s|%s" % k for k in keys])
        for t in range(n):
            w.writerow([t, int(meta["triple"][t]), int(meta["variant"][t]), groups[int(gtab[t])], int(meta["donor_tier"][t]), int(meta["on_target"][t])]
                       + ["%.6g" % scores[k][t] if np.isfinite(scores[k][t]) else ("inf" if np.isinf(scores[k][t]) else "") for k in keys])
    ap = os.path.join(folder, "auc.tsv")
    write_rows(ap, rows)
    gp = os.path.join(folder, "groups.tsv")
    write_rows(gp, per_group)
    rep = dict(format="rel-w18-readout", version=1, protocol="w18-v0", status=STATUS, run=os.path.basename(run_dir), commit=cfg["git"]["sha"],
               tables=dict(run=os.path.basename(os.path.normpath(s["tables"]["run"])), name=doc["name"], sha256=doc["sha256"], tables=n,
                           triples=int(len(np.unique(meta["triple"]))), layout=doc.get("layout")),
               arms=arms, groups=groups, tiers=index.get("tiers"), reps=s["reps"], seed=s["seed"], passes_version=P.VERSION, readouts_version=R.VERSION,
               readable=dict(min_group=R.MIN_GROUP, min_groups=R.MIN_GROUPS, min_implants=R.MIN_IMPLANTS, min_genera=R.MIN_GENERA),
               headline=headline(rows), too_few=sorted({(r["block"], r["stratum_kind"], r["stratum"]) for r in rows if not r["readable"]}),
               files={os.path.relpath(p, run_dir): sha256_file(p, use_sidecar=False) for p in (sp, ap, gp)})
    rep["too_few"] = [list(x) for x in rep["too_few"]]
    page = os.path.join(folder, "report.md")
    with open(page + ".part", "w") as f:
        f.write(page_of(rep, rows))
    os.replace(page + ".part", page)
    st.output("w18_readout_page", page)
    for name, p in (("w18_scores", sp), ("w18_auc", ap), ("w18_groups", gp)):
        st.output(name, p)
    p = os.path.join(folder, "report.json")
    write_json(p, rep)
    st.output("w18_readout_report", p)
    st.extra.update(arms=[a["arm"] + ":" + a["weights"] for a in arms], rows=len(rows), seconds=round(time.time() - t0, 1))
    return rep


def write_rows(path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path + ".part", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(keys)
        for r in rows:
            w.writerow([json.dumps(r[k], sort_keys=True) if isinstance(r.get(k), dict) else ("" if r.get(k) is None else r[k]) for k in keys])
    os.replace(path + ".part", path)


def headline(rows):
    """The primary comparison (implant against own) by donor tier and in all, for every score, per block."""
    out = []
    for r in rows:
        if r["comparison"] == "own" and r["stratum_kind"] in ("all", "donor_tier"):
            out.append({k: r.get(k) for k in ("label", "score", "block", "stratum", "auc", "lo", "hi", "implants", "groups", "genera", "readable")})
    return out


def _f(x):
    return "" if x is None else "%.3f" % x


def page_of(rep, rows):
    md = ["# W18 HGT capability check: readouts on implant tables", "",
          "Run `%s` (commit %s); ruler set `%s` of run `%s` (manifest `%s`, %d tables in %d triples, layout %s)." % (
              rep["run"], rep["commit"][:12], rep["tables"]["name"], rep["tables"]["run"], rep["tables"]["sha256"][:16], rep["tables"]["tables"],
              rep["tables"]["triples"], rep["tables"]["layout"]),
          "", "**Status.** " + rep["status"], "",
          "AUC = P(implant scores higher than control) + 1/2 P(tie), per group of A, then the mean over the readable groups of a block; "
          "95%% intervals from %d genus-bootstrap replicates. A stratum is read with at least %d groups of %d implant and %d control tables, %d implants "
          "and %d genera of A; otherwise it says \"too few\"." % (rep["reps"], R.MIN_GROUPS, R.MIN_GROUP, R.MIN_GROUP, R.MIN_IMPLANTS, R.MIN_GENERA), "",
          "Arms: " + "; ".join("%s (%s, %s%s)" % (a["arm"], a["weights"], "relatives" if a["relatives"] else "no relatives", ", presence passes" if a["presence"] else "")
                               for a in rep["arms"]), ""]
    tiers = rep.get("tiers") or []
    for block in SC.BLOCKS:
        md += ["## %s: implant against another gene of A (primary)" % block, "",
               "| readout | " + " | ".join(["all"] + ["tier %s (%s)" % (t, tiers[t] if t < len(tiers) else "") for t in range(len(tiers))]) + " |",
               "|---|" + "---|" * (len(tiers) + 1)]
        by = {}
        for r in rows:
            if r["block"] == block and r["comparison"] == "own" and r["stratum_kind"] in ("all", "donor_tier"):
                by.setdefault((r["label"], r["score"]), {})[r["stratum"]] = r
        for (label, score), cells in sorted(by.items()):
            line = []
            for st in ["all"] + ["tier=%d" % t for t in range(len(tiers))]:
                r = cells.get(st)
                if r is None or r["auc"] is None:
                    line.append("")
                elif not r["readable"]:
                    line.append("too few (%d)" % r["implants"])
                else:
                    line.append("%s [%s, %s] n=%d" % (_f(r["auc"]), _f(r["lo"]), _f(r["hi"]), r["implants"]))
            md.append("| %s %s | %s |" % (label, score, " | ".join(line)))
        md.append("")
    md += ["## Abstention and the defined share of the source shift", "", "| arm | abstain: implant / original / own | source shift defined: implant / original / own | calibration error (own) |",
           "|---|---|---|---|"]
    for a in rep["arms"]:
        if a["relatives"]:
            ab, de = a["abstain"], a["source_shift_defined"]
            md.append("| %s (%s) | %s / %s / %s | %s / %s / %s | %s%s |" % (a["arm"], a["weights"], _f(ab["implant"]), _f(ab["original"]), _f(ab["own"]),
                                                                         _f(de["implant"]), _f(de["original"]), _f(de["own"]), _f(a["calibration"]["own"]["error"]),
                                                                         " (uncalibrated)" if a["calibration"]["uncalibrated"] else ""))
    md += ["", "The full tables (every stratum, both comparisons, the paired differences against the controls) are in auc.tsv; every table's scores in scores.tsv."]
    return "\n".join(md) + "\n"


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "report.json")
    with open(p) as f:
        rep = json.load(f)
    problems = []
    if not any(a.get("relatives") and a.get("weights") != "untrained" for a in rep.get("arms") or []):
        problems.append(dict(error="no trained arm with relatives was read"))
    for name, sha in (rep.get("files") or {}).items():
        if sha256_file(os.path.join(run_dir, name), use_sidecar=False) != sha:
            problems.append(dict(file=name, error="not the file the report names"))
    with open(os.path.join(run_dir, OUT, "auc.tsv")) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    bad = [r for r in rows if r["auc"] not in ("", None) and not 0.0 <= float(r["auc"]) <= 1.0]
    if not rows or bad:
        problems.append(dict(error="no AUC row, or an AUC outside [0, 1]", n=len(bad)))
    if not rep.get("tables", {}).get("sha256") or not rep.get("run"):
        problems.append(dict(error="the report does not name its run and tables"))
    require(GateResult("G4_rel_w18_readout", "fail" if problems else "pass", dict(problems=problems[:20], arms=[a["arm"] + ":" + a["weights"] for a in rep.get("arms") or []],
                                                                              rows=len(rows))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        rep = json.load(f)
    js = dict(tables=rep["tables"], arms=[a["arm"] + ":" + a["weights"] for a in rep["arms"]],
              headline=[h for h in rep["headline"] if h["stratum"] == "all"])
    md = ["W18 readouts on `%s` (%d tables): %s." % (rep["tables"]["name"], rep["tables"]["tables"], ", ".join(js["arms"])),
          "Implant against own, all tiers: " + "; ".join("%s %s %s %s" % (h["label"], h["score"], h["block"][:10], _f(h["auc"]) if h["readable"] else "too few")
                                                         for h in js["headline"] if h["score"] in ("source_shift", "gain", "composition", "best_similarity", "host_codon"))]
    return dict(json=js, md=md)


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w18_readout") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
