"""Stage rel_w18_readout (package W18; pipeline rel_traineval, after rel_w6_eval): the readouts of the HGT capability
check, their controls and statistics (protocol w18-v0), from the per-cell files the evaluation of a ruler set wrote.
`python -m ci.stages.rel.w18_readout RUN_DIR`. CPU only, nothing is trained, no model is run. Every result is
"construct only".

Inputs: the ruler set (rel_w18_readout.tables: {run, path, sha256}, read through common.read_tableset, so the run
passed, the kind is one this stage may read and the set carries its sealed record) and the per-cell files of the
evaluation (data/rel_w6/eval/<arm>/ruler_cells.npz of rel_w18_readout.eval_run: one run directory or a list of them,
default this run; each file listed with its sha256 in its run's record of stage rel_w6_eval; an arm folder in one run
only). Every file must name the ruler set's manifest, and the digest of
what it scored must equal the plan of seeddrift/train/rel/readouts/w18_passes.py recomputed here without a model.

Arms by their files: an arm with relatives and trained weights gives the source shift, the two-block gain, the gain
without relatives and their difference, each from the bits and from the amino-acid bits alone (the lineage readings:
source_shift_aa, gain_aa, gain_rel_aa), abstention and the calibration; with presence passes the same on them; an arm
without relatives gives the composition readout; weights "untrained" mark the untrained control. The model-free controls
come from the tables (seeddrift/train/rel/readouts/w18_readouts.py).

Populations: scenario in place or the donor's lineage unsampled, each with rule 9 on (the triples whose donor gene has a
homolog in A's context left out) and off. The source shift is read on the triples whose three tables define it and
whose rows span two tiers or more; its defined share (implant minus own) is a readout of its own. Statistics: the
paired win rate of implant against own (primary) and against original (the position control) in every stratum; the
pooled AUC in all, by donor tier and by quartile of the host codon model; paired differences against the best-similarity
rule, the coverage shift, the untrained model, the presence passes and the host codon model.

Adapted readings: when the per-cell files say their arms read the implant tables adapted (adapt_v0, the rows kept: arms
trained on tables of version 0 reading implant tables of version 1), the stage adapts the tables the same way
(w18_passes.adapt_arrays) before every readout, control and plan digest, and the page says so first; every arm of one
reading must have read the tables the same way.

Protocol w18-v3, version 4 (a leak-free reading, rendered on the strict offline placements of stage rel_w18_offline): one rendering
per reading (the per-cell files' layout and placements, recorded as `layout`); arms labelled by their weights
(arm@node@N beside final); every reading also per relatives layer (other-genome rows on the target, self copies only,
none); the criterion "the readouts see the relatives" (defined share of source_shift_aa above the untrained model's and
the arm's presence passes', abstention below both, the win rate of gain_rel_aa above the untrained model's; paired
differences, 95% intervals) and the second verdict (the win rate of source_shift_aa above both), per arm with relatives
and block, in place, rule 9 on, the layer of other-genome rows, near-tier donors (tiers 0 and 1) left out (criterion.tsv,
report.json verdicts). The second verdict (report.json second_verdicts): source_shift_top_aa (an undefined shift
ranked the most foreign, a table that could not be computed left out and counted) on a population the ruler fixes (no
self-copy rows, every table with other-genome rows of two tiers or more on the target), decomposed per triple into a
shift term and a definedness term, against the model-free coverage_shift counted on the cells of passes T and C as
rendered (setting offline: the run of the placements, for a strict reading); labels holds / through definedness /
not shown / too few to read; the untrained model, the presence passes, best_similarity, the primary subset, the
strata of coverage ratio and host codon quartiles beside it; both verdicts counted per reading ("k of the m readable
verdicts hold"). The untrained control is the reading's one untrained model with relatives, mapped to every arm
(setting untrained names it when there are several), never paired by name.

Profile block rel_w18_readout: tables, eval_run (optional; a run directory or a list), arms (optional: the arms to read, default every file
found), reps (2,000), seed (20261124), compare (other readings, the 2 x 2 table), untrained (the untrained control's
name, when there are several), offline (the run of stage rel_w18_offline of a strict reading).
Outputs under data/rel_w18/readout/: scores.tsv (every score of every table), stats.tsv (the win rates and AUCs),
defined.tsv (the defined share of the source shift, implant minus own), coverage.tsv (charter rule 7's table),
criterion.tsv (protocol w18-v3's paired share differences), report.md, report.json (written last).
Gate G4_rel_w18_readout: every per-cell file read was checked, at least one trained arm with relatives was read, every
value is in [0, 1], no coverage cell is marked evaluated, and the report names the run and the tables of every number.
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
from seeddrift.train.rel.evalh import score as SC
from seeddrift.train.rel.evalh import shards
from seeddrift.train.rel.readouts import w18_passes as P
from seeddrift.train.rel.readouts import w18_readouts as R

OUT = os.path.join("data", "rel_w18", "readout")
EVAL = os.path.join("data", "rel_w6", "eval")
KEYS = {"tables", "eval_run", "arms", "reps", "seed", "compare", "untrained", "offline"}
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
                arms=[str(a) for a in c.get("arms") or []], reps=int(c.get("reps", R.BOOT)), seed=int(c.get("seed", R.BOOT_SEED)),
                compare=[str(x) for x in c.get("compare") or []], untrained=c.get("untrained"), offline=c.get("offline"))


TRAIN_ROWS = ("leak-free, tables of version 1", "stored, tables of version 0, adapted")
EVAL_COLS = ("strict offline (leak-free)", "stored")
TWO_SCORES = ("gain_rel_aa", "source_shift_aa", "gain_aa")


TAIL = "A ruler, not a transfer label: these numbers choose no arm, head, threshold, readout or node."
STORED_V1 = ("implant tables of version 1 under the stored layout, which places relatives by the hidden residues: a diagnosis or the "
             "stored column of protocol w18-v3's 2 x 2 table, never the counting reading. Construct only.")


def status_of(strict, adapt, version):
    """The page's status line: the counting reading (strict), an adapted diagnosis, the stored layout on tables of
    version 1, or the preliminary reading of version 0."""
    if strict:
        return STATUS_V3
    if adapt:
        return (ADAPTED % int(adapt)) + " " + ((STORED_V1 + " " + TAIL) if version else STATUS)
    return (STORED_V1 + " " + TAIL) if version else STATUS


def cell_of(rep):
    """(training layout, evaluation layout) of a reading: adapted arms were trained on tables of version 0 under the
    stored layout, the others are round 1's (version 1, leak-free); the evaluation layout is the reading's rendering."""
    lay = (rep.get("layout") or {}).get("layout") or (rep.get("arms") or [{}])[0].get("layout")
    return TRAIN_ROWS[1] if rep.get("adapt_v0") else TRAIN_ROWS[0], EVAL_COLS[0] if lay == "leakfree" else EVAL_COLS[1]


def compared(runs):
    """[report] of other readings of stage rel_w18_readout (compare: their run directories), each file the one its run's
    stage record lists."""
    out = []
    for run in runs:
        p = os.path.join(run, OUT, "report.json")
        rec = os.path.join(run, "stages", "rel_w18_readout.json")
        with open(rec) as f:
            listed = {os.path.normpath(os.path.join(run, o["path"]) if not os.path.isabs(o["path"]) else o["path"]): o.get("sha256")
                      for o in json.load(f).get("outputs") or []}
        if listed.get(os.path.normpath(p)) != sha256_file(p, use_sidecar=False):
            raise SystemExit("rel_w18_readout: %s is not the report its run's record of stage rel_w18_readout lists" % p)
        with open(p) as f:
            out.append(json.load(f))
    return out


def two_by_two(reps):
    """Protocol w18-v3's 2 x 2 table (training layout x evaluation layout): per cell, arm and block the paired win rate
    against own (in place, rule 9 on, all tiers) of the lineage readings, from this reading and the compared ones; a
    cell no reading fills is "not read"."""
    rows = []
    for rep in reps:
        tr_, ev_ = cell_of(rep)
        for h in rep.get("headline") or []:
            if h["stratum"] == "all" and h["score"] in TWO_SCORES and h["label"] != "control" and not h["label"].startswith(("untrained:", "presence:")):
                rows.append(dict(training=tr_, evaluation=ev_, run=rep["run"], arm=h["label"], score=h["score"], block=h["block"], value=h["value"],
                                 lo=h["lo"], hi=h["hi"], triples=h["triples"], readable=h["readable"]))
    filled = {(r["training"], r["evaluation"]) for r in rows}
    return rows, [dict(training=a, evaluation=b) for a in TRAIN_ROWS for b in EVAL_COLS if (a, b) not in filled]


def ruler(s):
    """(manifest, TableSet, W18 meta, table fields, group names, index) of the ruler set, read through the registry's
    reader (common.read_tableset: the kinds this stage may read, the sealed record)."""
    doc = common.read_tableset("w18_readout", s["tables"]["run"], s["tables"]["path"], s["tables"]["sha256"])
    if doc.get("kind") != "ruler":
        raise SystemExit("rel_w18_readout: the table set is kind %r, not ruler" % doc.get("kind"))
    sets = shards.by_split(s["tables"]["run"], [doc])
    if list(sets) != [1]:
        raise SystemExit("rel_w18_readout: a ruler set of held-out tables only (splits %s)" % sorted(sets))
    ts = sets[1]
    meta = P.load_meta(s["tables"]["run"], doc)
    fields = ts.load(0, ts.tables, ("tok", "n", "col", "kind", "frame_len", "group", "identity", "genome"))
    with open(os.path.join(s["tables"]["run"], doc["index"])) as f:
        index = json.load(f)
    return doc, ts, meta, fields, shards.group_names(s["tables"]["run"], doc), index


def cell_files(s, run_dir):
    """[(arm, path, sha256)] of the per-cell files of the evaluation runs (eval_run: one run directory or a list; default
    this run), each listed in its run's record of stage rel_w6_eval; an arm folder appears in one run only."""
    runs = s["eval_run"] or run_dir
    runs = [runs] if isinstance(runs, str) else list(runs)
    out, seen = [], {}
    for ev in runs:
        root = os.path.join(ev, EVAL)
        rec = os.path.join(ev, "stages", "rel_w6_eval.json")
        listed = {}
        if os.path.exists(rec):
            with open(rec) as f:
                for o in json.load(f).get("outputs") or []:
                    listed[os.path.normpath(os.path.join(ev, o["path"]) if not os.path.isabs(o["path"]) else o["path"])] = o.get("sha256")
        for arm in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            p = os.path.join(root, arm, P.RULER_CELLS)
            if not os.path.exists(p) or (s["arms"] and arm not in s["arms"]):
                continue
            if arm in seen:
                raise SystemExit("rel_w18_readout: arm folder %s is in two evaluation runs (%s and %s)" % (arm, seen[arm], ev))
            sha = sha256_file(p, use_sidecar=False)
            if listed.get(os.path.normpath(p)) != sha:
                raise SystemExit("rel_w18_readout: %s is not an output the evaluation run's record of stage rel_w6_eval lists with this content" % p)
            seen[arm] = ev
            out.append((arm, p, sha))
    missing = sorted(set(s["arms"]) - {a for a, _, _ in out})
    if missing or not out:
        raise SystemExit("rel_w18_readout: no per-cell file for arms %s under %s" % (missing or "any", runs))
    return out


RULE9_UNSET = ("its first bit (A's homolog of D:j among the context genes) was never set on this ruler set: the search holds no pair of a donor "
               "gene with A's genes, so the triples left out by rule 9 are those of its second bit alone (a row's homolog of D:j is its homolog on a "
               "context block)")
STATUS_V3 = ("the counting reading (protocol w18-v3): round 1's arms, trained on tables of version 1 under the leak-free layout, on implant tables "
             "of version 1 rendered strictly offline (W3's aligner, the fourth validity test passed on every table and mask). Construct only. "
             "A ruler, not a transfer label: these numbers choose no arm, head, threshold, readout or node.")
ADAPTED = ("ADAPTED: the arms were trained on tables of version 0 and read implant tables of version 1 adapted for them "
           "(%d rows taken tier by tier, KIND_ABSENT read as KIND_NOT_SEARCHED; the programme lead's diagnosis of 2026-11-23), "
           "never the counting reading.")
LINEAGE = ("source_shift_aa", "gain_aa", "gain_rel_aa")          # the lineage readings (amino-acid bits alone): primary
POOLED_STRATA = ("all", "donor_tier", "host_codon_quartile")


def duplicate_cells(cells):
    """A problem of one per-cell file, or None: a (table, pass, slot) scored twice, or a pass name twice (the presence
    passes and the plain ones are told apart by their pass names only)."""
    names = [str(x) for x in cells["pass_names"]]
    if len(set(names)) != len(names):
        return "pass names repeated"
    key = (np.asarray(cells["table"], np.int64) * (len(names) + 1) + np.asarray(cells["pass_id"], np.int64)) * (P.L + 1) + np.asarray(cells["slot"], np.int64)
    dup = int(len(key) - len(np.unique(key)))
    return "%d cells repeated" % dup if dup else None


STAT_KEY = ("label", "score", "statistic", "comparison", "block", "stratum_kind", "stratum", "scenario", "rule9")
CRITERION_KEY = ("label", "score", "reference", "block", "scenario", "rule9", "variant")


def duplicate_keys(rows, key):
    """[key] that occur twice among rows (a result written twice would hide one of them)."""
    seen, dup = set(), []
    for r in rows:
        k = tuple(str(r.get(x)) for x in key)
        if k in seen:
            dup.append(k)
        seen.add(k)
    return dup


def untrained_control(arms, named=None):
    """The label of the reading's one untrained model with relatives, the control of every arm (protocol w18-v3 version
    4: the ref30m variant with seed 1, mapped here, never paired by name); `named` (setting untrained: its arm name or
    label) chooses among several; None without one."""
    labels = [a["label"] for a in arms if a["weights"] == "untrained" and a["relatives"]]
    if named:
        got = [x for x in labels if x in (str(named), "untrained:" + str(named))]
        if len(got) != 1:
            raise SystemExit("rel_w18_readout: the untrained control %r is not one untrained arm of this reading (%s)" % (named, labels))
        return got[0]
    if len(labels) > 1:
        raise SystemExit("rel_w18_readout: %d untrained arms (%s); name the control of every arm (setting untrained)" % (len(labels), labels))
    return labels[0] if labels else None


def criterion_rows(scores, trained, untrained_of, tr, crit, full, boot):
    """The rows the first verdict reads its gain_rel_aa part from: the paired win rate against own on the criterion's
    population (in place, rule 9 on, other-genome rows, no near-tier donor), against the untrained control."""
    rows = []
    for label in trained:
        ref = {}
        u = untrained_of(label)
        if u and (u, "gain_rel_aa") in scores:
            ref["untrained"] = scores[(u, "gain_rel_aa")]
        rows += R.stat_rows({"gain_rel_aa": scores[(label, "gain_rel_aa")]}, tr, crit, {R.CRITERION_STRATUM: full[R.CRITERION_STRATUM]}, boot, label=label,
                            reference=ref, stats=("win",), tag=dict(R.CRITERION_POPULATION))
    return rows


def strict_shown(s, doc, lay, ts, meta, fields, say):
    """(on_t, on_c) [N, R] of the strict reading: the cells every row shows under passes T and C on the offline
    placements the arms were rendered on (setting offline: the run of stage rel_w18_offline; its offline.json must be the
    one the per-cell files name). Adapted tables are never strict."""
    if not s["offline"]:
        raise SystemExit("rel_w18_readout: a strict reading needs the run of its offline placements (setting offline) for coverage_shift")
    arrays, _, sha = P.load_offline(s["offline"], doc["sha256"])
    if sha != lay["offline_sha256"]:
        raise SystemExit("rel_w18_readout: the offline placements given are not the ones the arms were rendered on")
    N, R = fields["tok"].shape[:2]
    on_t, on_c = np.zeros((N, R), dtype=np.int64), np.zeros((N, R), dtype=np.int64)
    for t in range(N):
        tab = {f: fields[f][t] for f in P.FIELDS}
        on_t[t], on_c[t] = P.shown_cells(tab, meta["row_tier"][t], t, int(meta["parent"][t]), P.offline_place(arrays, t, tab))
    say("coverage_shift on the strict placements of %s" % os.path.basename(os.path.normpath(s["offline"])))
    return (on_t, on_c), os.path.basename(os.path.normpath(s["offline"]))


def build(st, cfg, run_dir):
    s, t0 = settings(cfg), time.time()
    say = lambda text: print("rel_w18_readout: " + text, flush=True)          # noqa: E731
    doc, ts, meta, fields, groups, index = ruler(s)
    others = compared(s["compare"])
    if any((o.get("tables") or {}).get("sha256") != doc["sha256"] for o in others):
        raise SystemExit("rel_w18_readout: a compared reading read another ruler set")
    genomes = index["genomes"]
    n = ts.tables
    loaded = [(arm, path, sha) + P.load_cells(path) for arm, path, sha in cell_files(s, run_dir)]
    for arm, path, _, cells, _ in loaded:
        bad = duplicate_cells(cells)
        if bad:
            raise SystemExit("rel_w18_readout: %s holds a cell twice or a pass name twice (%s)" % (path, bad))
    adapts = {head.get("adapt_v0") for _, _, _, _, head in loaded}
    if len(adapts) != 1:
        raise SystemExit("rel_w18_readout: the arms read the tables adapted in different ways (%s); one reading reads one way" % sorted(map(str, adapts)))
    adapt = adapts.pop()
    layouts = {(head.get("layout"), head.get("offline"), head.get("offline_sha256")) for _, _, _, _, head in loaded}
    if len(layouts) != 1:
        raise SystemExit("rel_w18_readout: the arms were rendered in different ways (%s); one reading renders one way" % sorted(map(str, layouts)))
    lay = dict(zip(("layout", "offline", "offline_sha256"), layouts.pop()))
    strict = lay["layout"] == "leakfree"
    if strict and not lay["offline_sha256"]:
        raise SystemExit("rel_w18_readout: a leak-free reading without the strict offline placements (protocol w18-v3)")
    meta_stored = meta                                              # the plan digest adapts the stored tables itself (tables_of)
    if adapt:                                                       # arms of tables of version 0 on implant tables of version 1
        fields, meta = P.adapt_arrays(fields, meta, int(adapt))
    n0 = fields["n"][:, 0, :]
    tr = R.Triples(meta, fields["group"], fields["identity"], [genomes[int(d)]["group"] for d in meta["donor"]], groups)
    boot = R.Bootstrap(tr, reps=s["reps"], seed=s["seed"])
    arms, scores, plans = [], {}, {}
    for arm, path, sha, cells, head in loaded:
        if head["tableset_sha256"] != doc["sha256"]:
            raise SystemExit("rel_w18_readout: %s scored another table set (%s)" % (path, head["tableset_sha256"]))
        scen, psc = head.get("scenarios"), head.get("presence_scenarios")
        key = (bool(head["relatives"]), bool(head["presence"]), None if scen is None else tuple(scen), None if psc is None else tuple(psc))
        if key not in plans:
            plans[key] = P.plan_digest(P.tables_of(ts, meta_stored, scenarios=key[2], presence=key[1], presence_scenarios=key[3], adapt=adapt), key[0], key[1])
        if cells["digest"] != plans[key][0]:
            raise SystemExit("rel_w18_readout: %s scored another plan than the passes of this commit" % path)
        tag = ("untrained:" if head["weights"] == "untrained" else "") + head["arm"] + ("" if head["weights"] in ("final", "untrained") else "@" + str(head["weights"]))
        if any(a["label"] == tag for a in arms):
            raise SystemExit("rel_w18_readout: arm %s at weights %s read twice" % (head["arm"], head["weights"]))
        info = dict(arm=head["arm"], label=tag, folder=arm, file=os.path.relpath(path, run_dir) if path.startswith(run_dir) else path, sha256=sha, weights=head["weights"],
                    relatives=head["relatives"], presence=head["presence"], layout=head["layout"], render_version=head["render_version"],
                    eval_run=head["run"], eval_commit=head["commit"], cells=int(len(cells["bits"])), passes=head["passes"], scenarios=scen, presence_scenarios=psc)
        if head["relatives"]:
            r = R.readouts(cells, meta, n0)
            info["calibration"] = R.calibration(cells, meta, n0)
            info["abstain"] = {v: _mean(r["abstain"][meta["variant"] == k]) for k, v in enumerate(("implant", "original", "own"))}
            for k, v in r.items():
                if k != "abstain":
                    scores[(tag, k)] = v
            scores[(tag, "abstain")] = r["abstain"]
            if head["presence"]:
                for k, v in R.readouts(cells, meta, n0, prefix=P.PRESENCE).items():
                    scores[("presence:" + tag, k)] = v
        else:
            for k, v in R.composition(cells, meta, n0).items():
                scores[(tag, k)] = v
        arms.append(info)
        say("%s (%s): %d cells read" % (arm, head["weights"], info["cells"]))
    shown, off_run = None, None
    if strict:                                                      # coverage_shift on the cells of T and C as rendered strictly
        shown, off_run = strict_shown(s, doc, lay, ts, meta_stored, fields, say)
    ctl = R.controls(fields, meta, shown=shown)
    for k, v in ctl.items():
        if not k.startswith("x_"):                                    # x_gc, x_aa_distance: strata, not scores
            scores[("control", k)] = v
    has_rows = (tr.phase != 3) | (tr.self_rows > 0)                 # phase 3: no common row of another genome (self copies may remain)
    control = untrained_control(arms, s["untrained"])

    def untrained_of(base):
        """The untrained control of every arm (protocol w18-v3 version 4: one untrained model, mapped to every arm here,
        never paired by name)."""
        return control
    rule9 = tr.context == 0
    pops = {("in_place", "on"): (tr.scenario == 0) & rule9, ("in_place", "off"): tr.scenario == 0,
            ("donor_unsampled", "on"): (tr.scenario == 1) & rule9, ("donor_unsampled", "off"): tr.scenario == 1}
    full = tr.strata()
    full.update(tr.host_quartiles(ctl["host_codon"]))
    full.update(tr.exploratory(ctl["x_gc"], ctl["x_aa_distance"]))      # exploratory strata (kinds x_...), outside the protocol
    short = {k: v for k, v in full.items() if k[0] in ("all", "donor_tier", "relatives_layer")}
    expected = tuple("tier=%s" % t for t, e in (index.get("tier_expectation") or {}).items() if e == "expected too few")
    rows, defined = [], []
    span = R.tier_span(meta, fields["kind"])[tr.table].all(axis=1) & (tr.self_rows == 0)      # the ruler's: every table spans two tiers
    for (label, name), v in sorted(scores.items()):
        if name == "abstain" or name.startswith("source_shift_defined"):
            continue
        base = label.split(":", 1)[-1]
        ref = {}
        if label != "control":
            ref["best_similarity"] = ctl["best_similarity"]
            if name.startswith("source_shift"):
                ref["coverage_shift"] = ctl["coverage_shift"]
            if name.startswith("composition"):
                ref["host_codon"] = ctl["host_codon_rel" if name.endswith("_rel") else "host_codon"]
            if name.startswith("gain") and name.endswith("_aa"):
                ref["host_aa"] = ctl["host_aa"]                             # exploratory: is gain_aa only "unlike A's amino acids"?
            for other in (untrained_of(base), "presence:" + base):
                if other and (other, name) in scores and other != label and not label.startswith("untrained:"):
                    ref[other.split(":")[0]] = scores[(other, name)]
        for (scen, r9), pop in pops.items():
            pop = pop & (has_rows if not name.startswith(("composition", "host_codon")) else True)
            if name.startswith("source_shift_top") and label != "control":
                pop = pop & span                                            # w18-v3 version 4: the ruler decides, never the definedness
            elif name.startswith("source_shift") and label != "control":
                d = np.isfinite(v)
                if name in ("source_shift", "source_shift_aa"):
                    defined += R.defined_share_rows(v, tr, pop, boot, label=label, tag=dict(scenario=scen, rule9=r9, of=name))
                pop = pop & d[tr.table].all(axis=1) & (tr.tiers_present >= 2)
            strata = full if (scen, r9) == ("in_place", "on") else short
            tag = dict(scenario=scen, rule9=r9)
            rows += R.stat_rows({name: v}, tr, pop, strata, boot, label=label, reference=ref, stats=("win",), expected_too_few=expected, tag=tag)
            rows += R.stat_rows({name: v}, tr, pop, {k: m for k, m in strata.items() if k[0] in POOLED_STRATA}, boot, label=label, reference=ref,
                                stats=("pooled",), expected_too_few=expected, tag=tag)
    shares, trained = [], [a["label"] for a in arms if a["relatives"] and a["weights"] != "untrained"]
    near = np.isin(tr.tier, R.NEAR_TIERS)
    crit = pops[("in_place", "on")] & has_rows & full[R.CRITERION_STRATUM] & ~near     # near-tier donors: the near-tier section only
    for label in trained:
        controls_ = dict(untrained=untrained_of(label), presence="presence:" + label)
        for score, kind, pop in (("source_shift_aa_defined", "source_shift_defined_aa", crit & (tr.tiers_present >= 2)), ("abstain", "abstain", crit)):
            x = scores[(label, kind)]
            for ref, other in controls_.items():
                if other is None or (other, kind) not in scores:
                    continue
                shares += R.paired_share_rows(score, x, scores[(other, kind)], tr, pop, boot, label=label, reference=ref, tag=dict(R.CRITERION_POPULATION))
    verdict = R.verdicts(criterion_rows(scores, trained, untrained_of, tr, crit, full, boot), shares, trained)
    pop2 = R.second_population(tr, meta, fields["kind"], crit)
    strata2 = {"coverage ratio 0.8-1.25": (tr.coverage_ratio >= 0.8) & (tr.coverage_ratio < 1.25)}
    strata2.update({"host codon %s" % q: m for (k, q), m in full.items() if k == "host_codon_quartile"})
    second = []
    for label in trained:
        cons = {k: scores[(v, "source_shift_top_aa")] for k, v in (("untrained", untrained_of(label)), ("presence", "presence:" + label)) if v and (v, "source_shift_top_aa") in scores}
        for block in SC.BLOCKS:
            second.append(dict(arm=label, **R.second_verdict(scores[(label, "source_shift_top_aa")], ctl["coverage_shift"], tr, pop2, boot, block,
                                                             controls=cons, best=ctl["best_similarity"], strata=strata2)))
    second_pop = {block: dict(triples=int((pop2 & (tr.block == block)).sum()),
                              by_group={tr.group_names[g]: int((pop2 & (tr.group == g)).sum()) for g in sorted(set(tr.group[pop2 & (tr.block == block)].tolist()))})
                  for block in SC.BLOCKS}
    counted = dict(first=R.multiplicity(verdict, len(trained), key="verdict", holds=(R.SEE,), too_few=()),
                   second=R.multiplicity(second, len(trained)))
    counted["first"] = dict(counted["first"], too_few=sum(1 for v in verdict if not v["readable"]), readable=sum(1 for v in verdict if v["readable"]))
    counted["first"]["text"] = "%d of the %d readable verdicts hold, %d too few to read" % (
        sum(1 for v in verdict if v["readable"] and v["verdict"] == R.SEE), counted["first"]["readable"], counted["first"]["too_few"])
    say("%d result rows, %d criterion rows, %.0f s" % (len(rows), len(shares), time.time() - t0))
    cover = R.coverage_table(tr, pops[("in_place", "on")] & has_rows)
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    sp = os.path.join(folder, "scores.tsv")
    keys = sorted(scores)
    with open(sp + ".part", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["table", "triple", "parent", "scenario", "variant", "group", "donor_tier", "on_target", "context_homolog"] + ["%s|%s" % k for k in keys])
        for t in range(n):
            w.writerow([t, int(meta["triple"][t]), int(meta["parent"][t]), int(meta["scenario"][t]), int(meta["variant"][t]), groups[int(fields["group"][t])],
                        int(meta["donor_tier"][t]), int(meta["on_target"][t]), int(meta["context_homolog"][t])]
                       + ["%.6g" % scores[k][t] if np.isfinite(scores[k][t]) else ("inf" if np.isinf(scores[k][t]) else "") for k in keys])
    os.replace(sp + ".part", sp)
    ap, dp, cp, kp = (os.path.join(folder, x) for x in ("stats.tsv", "defined.tsv", "coverage.tsv", "criterion.tsv"))
    write_rows(ap, rows)
    write_rows(dp, defined)
    write_rows(cp, cover)
    write_rows(kp, shares)
    status = status_of(strict, adapt, int(doc.get("tables_of_version", 0)))
    rep = dict(format="rel-w18-readout", version=3, protocol="w18-v3" if strict else "w18-v0", status=status, adapt_v0=adapt, layout=lay,
               run=os.path.basename(run_dir), commit=cfg["git"]["sha"], verdicts=verdict, second_verdicts=second, second_population=second_pop,
               multiplicity=counted, untrained_control=control, coverage_shift_cells=("strict offline placements of %s" % off_run) if off_run else "stored layout",
               layers=list(R.LAYERS),
               layer_triples={name: int(((tr.layer == name) & pops[("in_place", "on")]).sum()) for name in R.LAYERS},
               rule9=index.get("rule9_bit1") or RULE9_UNSET,
               tables=dict(run=os.path.basename(os.path.normpath(s["tables"]["run"])), name=doc["name"], sha256=doc["sha256"], tables=n,
                           tables_of_version=int(doc.get("tables_of_version", 0)),
                           triples=int(len(tr.ids)), in_place=int((tr.scenario == 0).sum()), donor_unsampled=int((tr.scenario == 1).sum()), layout=doc.get("layout")),
               arms=arms, groups=groups, tiers=index.get("tiers"), tier_expectation=index.get("tier_expectation"), reps=s["reps"], seed=s["seed"],
               passes_version=P.VERSION, readouts_version=R.VERSION, lineage=list(LINEAGE),
               readable=dict(min_group=R.MIN_GROUP, min_groups=R.MIN_GROUPS, min_triples=R.MIN_TRIPLES, min_units=R.MIN_UNITS),
               headline=headline(rows), coverage=cover, defined=defined, compared=[dict(run=r["run"], cell=list(cell_of(r))) for r in others],
               files={os.path.relpath(p, run_dir): sha256_file(p, use_sidecar=False) for p in (sp, ap, dp, cp, kp)})
    rep["two_by_two"], rep["not_read"] = two_by_two([rep] + others)
    page = os.path.join(folder, "report.md")
    with open(page + ".part", "w") as f:
        f.write(page_of(rep, rows))
    os.replace(page + ".part", page)
    st.output("w18_readout_page", page)
    for name, p in (("w18_scores", sp), ("w18_stats", ap), ("w18_defined", dp), ("w18_coverage", cp), ("w18_criterion", kp)):
        st.output(name, p)
    p = os.path.join(folder, "report.json")
    write_json(p, rep)
    st.output("w18_readout_report", p)
    st.extra.update(arms=[a["arm"] + ":" + a["weights"] for a in arms], rows=len(rows), seconds=round(time.time() - t0, 1))
    return rep


def _mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def write_rows(path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path + ".part", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(keys)
        for r in rows:
            w.writerow([json.dumps(r[k], sort_keys=True) if isinstance(r.get(k), dict) else ("" if r.get(k) is None else r[k]) for k in keys])
    os.replace(path + ".part", path)


HEAD_SCORES = ("source_shift_aa", "source_shift", "gain_aa", "gain", "gain_rel_aa", "composition", "best_similarity", "coverage_shift", "host_codon",
               "rows_on_target", "covered_share")


def headline(rows):
    """The primary reading: scenario in place, rule 9 on, the paired win rate of implant against own, by donor tier and
    in all, per block."""
    keys = ("label", "score", "block", "stratum", "value", "lo", "hi", "triples", "groups", "units", "readable", "expected_too_few")
    return [{k: r.get(k) for k in keys} for r in rows if r.get("scenario") == "in_place" and r.get("rule9") == "on" and r["statistic"] == "win"
            and r["comparison"] == "own" and r["stratum_kind"] in ("all", "donor_tier")]


def _f(x):
    return "" if x is None else "%.3f" % x


def _cell(r):
    if r is None or r.get("value") is None:
        return ""
    if not r["readable"]:
        return "too few (%d)" % r["triples"]
    return "%s [%s, %s] n=%d" % (_f(r["value"]), _f(r["lo"]), _f(r["hi"]), r["triples"])


def _table(rows, block, scenario, rule9, statistic, comparison, tiers, expectation, scores=HEAD_SCORES):
    by = {}
    for r in rows:
        if (r["block"], r.get("scenario"), r.get("rule9"), r["statistic"], r["comparison"]) == (block, scenario, rule9, statistic, comparison) \
                and r["stratum_kind"] in ("all", "donor_tier") and r["score"] in scores:
            by.setdefault((r["label"], r["score"]), {})[r["stratum"]] = r
    head = ["all"] + ["tier %d (%s)%s" % (t, tiers[t] if t < len(tiers) else "", " *" if (expectation or {}).get(str(t)) == "expected too few" else "")
                      for t in range(len(tiers))]
    md = ["| readout | " + " | ".join(head) + " |", "|---|" + "---|" * len(head)]
    order = {s: k for k, s in enumerate(scores)}
    for (label, score), cells in sorted(by.items(), key=lambda kv: (kv[0][0] != "control", order.get(kv[0][1], 99), kv[0][0])):
        md.append("| %s %s | %s |" % (label, score, " | ".join(_cell(cells.get(x)) for x in ["all"] + ["tier=%d" % t for t in range(len(tiers))])))
    return md


def _ci(m):
    """value [lo, hi] (triples, groups) of a triple_mean, or its counts when it is not readable."""
    if not m or m.get("value") is None:
        return "none (%d triples)" % ((m or {}).get("triples") or 0)
    txt = "%s [%s, %s] n=%d, %d groups" % (_f(m["value"]), _f(m["lo"]), _f(m["hi"]), m["triples"], m["groups"])
    return txt if m.get("readable") else "too few (%s)" % txt


def criterion_md(rep):
    """Protocol w18-v3's two verdicts, per arm with relatives and block (never pooled)."""
    lay = rep.get("layout") or {}
    v3 = rep.get("protocol") == "w18-v3"
    head = ("## Criterion (protocol w18-v3): do the readouts see the relatives?" if v3 else
            "## Protocol w18-v3's criterion applied to this reading (a diagnosis, not the counting reading)")
    mult = rep.get("multiplicity") or {}
    md = [head, "",
          "Read once, in place, rule 9 on, the triples with other-genome rows on the target, without the triples whose donor lies in tier 0 or 1 "
          "(those are read in the near-tier section). Every part compares the arm with controls that share its tables, masks and cells: the "
          "untrained model (`%s`, the control of every arm; no learning, no content) and the arm's own presence passes (the relatives' pattern "
          "without their content). Parts: the defined share of the source shift (amino-acid bits; a table that could not be computed left out) on "
          "the own-control tables whose rows span two tiers or more, above both; the abstention on the own-control tables, below both; the paired "
          "win rate of gain_rel_aa (implant against own) above the untrained model's. Each holds when the 95%% interval of its paired difference "
          "(the same bootstrap replicates) excludes 0 on the right side. Rendering: %s%s. A verdict describes the ruler; it selects nothing." % (
              rep.get("untrained_control"), lay.get("layout"),
              ", offline placements of run `%s` (offline.json %s)" % (lay["offline"], str(lay["offline_sha256"])[:16]) if lay.get("offline") else ""),
          "", "What the strict rendering does not rule out: W3's fourth test covers the placement only; which rows a block holds, their kinds "
          "and stretch lengths come from the search, which was run with the hidden gene, and only the presence passes measure what they "
          "carry (protocol w3-offline-v0 version 3, validity_limits).",
          "", "First verdict, counted: %s." % ((mult.get("first") or {}).get("text") or "none"), "",
          "| arm | block | verdict | defined share | abstention | gain_rel_aa |", "|---|---|---|---|---|---|"]
    for v in rep.get("verdicts") or []:
        cells = [("%s%s" % ("holds: " if v["parts"][p]["holds"] else "fails: ", v["parts"][p]["detail"])) for p in R.CRITERION]
        md.append("| %s | %s | %s%s | %s |" % (v["arm"], v["block"], v["verdict"], "" if v["readable"] else " (a part too few to read)", " | ".join(cells)))
    md += [""] + second_md(rep)
    return md + [""]


def second_md(rep):
    """The second verdict (protocol w18-v3 version 4): its labels, decomposition, coverage bar and the rows beside it."""
    pop = rep.get("second_population") or {}
    mult = rep.get("multiplicity") or {}
    md = ["### Second verdict: the source shift detects implants beyond chance and beyond the ruler's coverage pattern", "",
          "Score source_shift_top_aa (w18-v0's secondary ordering: a table whose shift is undefined by the threshold ranked as the most foreign; a "
          "table that could not be computed is not ranked, and its triple is left out of that model's reading). Population, fixed by the ruler: the "
          "criterion's, without self-copy rows, every table carrying other-genome rows of two tiers or more on the target (%s). Per triple, "
          "win - 1/2 = s - u/2 with s = 1[both tables defined] (win - 1/2) and u = defined(implant) - defined(own): the shift term mean(s) and "
          "the definedness term -(1/2) mean(u). It holds when the shift term and, on the same triples, 1[both defined] (win - win of "
          "coverage_shift) both have 95%% intervals above 0; coverage_shift counts the cells of passes T and C as rendered (%s). Counted: %s. "
          "At these counts \"not shown\" is not evidence of absence." % (
              "; ".join("%s %d triples" % (b, x["triples"]) for b, x in pop.items()), rep.get("coverage_shift_cells"),
              (mult.get("second") or {}).get("text") or "none"),
          "", "| arm | block | label | win_top | shift term | definedness term | beats coverage_shift | left out (failed / coverage undefined) |",
          "|---|---|---|---|---|---|---|---|"]
    for v in rep.get("second_verdicts") or []:
        lab = v["label"] + (" (%s)" % ", ".join(v["failed"]) if v["label"] == R.NOT_SHOWN and v["failed"] else "")
        md.append("| %s | %s | %s | %s | %s | %s | %s | %d / %d |" % (v["arm"], v["block"], lab, _ci(v["win_top"]), _ci(v["shift_term"]), _ci(v["definedness_term"]),
                                                                   _ci(v["coverage_bar"]), v["left_out_failed"], v["coverage_bar_left_out"]))
    md += ["", "Beside the verdict (descriptions; they decide nothing):", "",
           "| arm | block | untrained: win_top, defined implant/own | presence: win_top, defined implant/own | best_similarity | win_top - coverage_shift "
           "| both defined; win on them (selected by the arm's own definedness) | only implant / only own undefined | undefined as no shift (sensitivity) "
           "| A units per entering group |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v in rep.get("second_verdicts") or []:
        cons = []
        for name in ("untrained", "presence"):
            c = (v.get("controls") or {}).get(name)
            cons.append("not read" if c is None else "%s; %s / %s (about 1/2 through ties where undefined)" % (
                _ci(c["win_top"]), _f(c["defined_implant"]), _f(c["defined_own"])))
        md.append("| %s | %s | %s | %s | %s | %s | %s; %s | %s / %s | %s | %s |" % (
            v["arm"], v["block"], cons[0], cons[1], _ci(v.get("best_similarity")), _ci(v["win_top_minus_coverage"]), _f(v["both_defined"]),
            _f(v["win_both_defined"]), _f(v["only_implant_undefined"]), _f(v["only_own_undefined"]), _ci(v["sensitivity_undefined_as_zero"]),
            ", ".join("%s %d" % kv for kv in (v["shift_term"].get("units_by_group") or {}).items()) or "none"))
    md += ["", "| arm | block | stratum | win_top | shift term |", "|---|---|---|---|---|"]
    for v in rep.get("second_verdicts") or []:
        for name, x in (v.get("strata") or {}).items():
            md.append("| %s | %s | %s | %s | %s |" % (v["arm"], v["block"], name, _ci(x["win_top"]), _ci(x["shift_term"])))
    return md


def two_by_two_md(rep):
    md = ["## Training layout x evaluation layout (2 x 2)", "",
          "The paired win rate against another gene of A (in place, rule 9 on, all tiers) of the lineage readings in every cell this reading and the "
          "readings it compares fill (%s). The adapted row is the diagnosis of 2026-11-23, labelled as such; it is never the counting reading." % (
              ", ".join("`%s`" % c["run"] for c in rep.get("compared") or []) or "no other reading"),
          "", "| training | evaluation | run | arm | readout | block | win rate |", "|---|---|---|---|---|---|---|"]
    for r in rep.get("two_by_two") or []:
        md.append("| %s | %s | %s | %s | %s | %s | %s |" % (r["training"], r["evaluation"], r["run"], r["arm"], r["score"], r["block"], _cell(r) or "none"))
    for c in rep.get("not_read") or []:
        md.append("| %s | %s | - | - | - | - | not read |" % (c["training"], c["evaluation"]))
    return md + [""]


def layers_md(rep, rows):
    """The headline readouts per relatives layer (protocol w18-v3: every reading per layer)."""
    md = ["## By relatives layer, in place, rule 9 on (implant against another gene of A)", "",
          "Layers: " + "; ".join("%s %d triples" % (k, n) for k, n in (rep.get("layer_triples") or {}).items()), "",
          "| readout | block | " + " | ".join(R.LAYERS) + " |", "|---|---|" + "---|" * len(R.LAYERS)]
    by = {}
    for r in rows:
        if r.get("scenario") == "in_place" and r.get("rule9") == "on" and r["statistic"] == "win" and r["comparison"] == "own" \
                and r["stratum_kind"] == "relatives_layer" and r["score"] in HEAD_SCORES:
            by.setdefault((r["label"], r["score"], r["block"]), {})[r["stratum"]] = r
    order = {x: k for k, x in enumerate(HEAD_SCORES)}
    for (label, score, block), cells in sorted(by.items(), key=lambda kv: (kv[0][2], kv[0][0] != "control", order.get(kv[0][1], 99), kv[0][0])):
        md.append("| %s %s | %s | %s |" % (label, score, block, " | ".join(_cell(cells.get(k)) or "none" for k in R.LAYERS)))
    return md + [""]


def page_of(rep, rows):
    tiers, exp = rep.get("tiers") or [], rep.get("tier_expectation") or {}
    md = ["# W18 HGT capability check: readouts on implant tables (construct only)", "",
          "Run `%s` (commit %s); ruler set `%s` of run `%s` (manifest `%s`): %d tables, %d triples in place and %d with the donor's lineage "
          "unsampled; layout %s." % (rep["run"], rep["commit"][:12], rep["tables"]["name"], rep["tables"]["run"], rep["tables"]["sha256"][:16],
                                     rep["tables"]["tables"], rep["tables"]["in_place"], rep["tables"]["donor_unsampled"], rep["tables"]["layout"]),
          "", "**Status.** " + rep["status"], "",
          "**Rule 9.** " + (rep.get("rule9") or RULE9_UNSET) + ".", "",
          "Primary statistic: the paired win rate inside a triple, P(implant > another gene of A) + 1/2 P(tie), per group of A, then the mean over the "
          "readable groups of a block; 95%% intervals from %d replicates of a two-level bootstrap (A's split units inside each group, D's split units). "
          "A group enters with at least %d triples; a block's stratum is read with %d groups, %d triples and %d units of A, otherwise \"too few\". "
          "Lineage readings use the amino-acid bits alone (%s). Tiers marked * were pre-declared \"expected too few\"." % (
              rep["reps"], R.MIN_GROUP, R.MIN_GROUPS, R.MIN_TRIPLES, R.MIN_UNITS, ", ".join(rep["lineage"])), "",
          "Arms: " + "; ".join("%s (%s, %s%s)" % (a.get("label", a["arm"]), a["weights"], "relatives" if a["relatives"] else "no relatives", ", presence passes" if a["presence"] else "")
                               for a in rep["arms"]), ""]
    if rep.get("protocol") == "w18-v3" or rep.get("verdicts"):
        md += criterion_md(rep) + layers_md(rep, rows)
    if rep.get("compared") or rep.get("protocol") == "w18-v3":
        md += two_by_two_md(rep)
    for block in SC.BLOCKS:
        md += ["## %s: implant against another gene of A, in place, rule 9 on" % block, ""] + _table(rows, block, "in_place", "on", "win", "own", tiers, exp) + [""]
        md += ["## %s: the donor's lineage unsampled, rule 9 on" % block, ""] + _table(rows, block, "donor_unsampled", "on", "win", "own", tiers, exp,
                                                                                         scores=("source_shift_aa", "gain_aa", "gain_rel_aa", "composition",
                                                                                                 "best_similarity")) + [""]
    near = [r for r in rows if r.get("scenario") == "in_place" and r.get("rule9") == "on" and r["statistic"] == "win" and r["comparison"] == "own"
            and r["stratum_kind"] == "donor_tier" and r["stratum"] in ("tier=0", "tier=1")]
    if rep["tables"].get("tables_of_version") or any(r["triples"] for r in near):
        md += ["## Near-tier donors without near reference rows (construct only)", "",
               "Donor tiers 0 and 1: D close to A, often of A's own split unit (the search of version 1 read the corpus with the close "
               "relatives, and a held-out genome's close relatives are held out with it). The other rows leave out the split units of A "
               "and D, so no row is near either of them: seen from afar the two lineages are one, and only the gene's mismatch with A's "
               "own context can tell them apart, with no relative as a reference. This section is not the close-transfer scenario (that "
               "one needs near reference rows, which the context rule D5 does not give a held-out A). The protocol pre-declared these "
               "tiers \"expected too few\"; they are read here as their counts allow.", "",
               "| readout | block | tier 0 | tier 1 |", "|---|---|---|---|"]
        by = {}
        for r in near:
            if r["score"] in HEAD_SCORES:
                by.setdefault((r["label"], r["score"], r["block"]), {})[r["stratum"]] = r
        order = {x: k for k, x in enumerate(HEAD_SCORES)}
        for (label, score, block), cells in sorted(by.items(), key=lambda kv: (kv[0][2], kv[0][0] != "control", order.get(kv[0][1], 99), kv[0][0])):
            md.append("| %s %s | %s | %s | %s |" % (label, score, block, _cell(cells.get("tier=0")) or "none", _cell(cells.get("tier=1")) or "none"))
        md.append("")
    md += ["## Exploratory, outside the pre-registered protocol", "",
           "The programme lead's additions of 2026-11-23, never to change a conclusion or choose anything: the host amino-acid "
           "model (control host_aa) and the strata by the implant's own composition (x_implant_gc, x_gc_minus_own, x_aa_distance; "
           "quartiles per block), read here for the 30 M arm's gain_aa, the untrained model's gain_aa and host_aa (win rate against own, "
           "in place, rule 9 on). Every stratum and arm is in stats.tsv.", ""]
    for block in SC.BLOCKS:
        md += ["### %s" % block, "", "| stratum | " + " | ".join("%s q%d" % (k[2:], q) for k in ("x_implant_gc", "x_gc_minus_own", "x_aa_distance") for q in range(1, 5)) + " |",
               "|---|" + "---|" * 12]
        by = {}
        for r in rows:
            if r["block"] == block and r.get("scenario") == "in_place" and r.get("rule9") == "on" and r["statistic"] == "win" and r["comparison"] == "own" \
                    and r["stratum_kind"].startswith("x_"):
                by.setdefault((r["label"], r["score"]), {})[(r["stratum_kind"], r["stratum"])] = r
        for (label, score), cells in sorted(by.items()):
            if score not in ("gain_aa", "host_aa") or label.startswith("presence:"):
                continue
            md.append("| %s %s | %s |" % (label, score, " | ".join(_cell(cells.get((k, "q%d" % q))) for k in ("x_implant_gc", "x_gc_minus_own", "x_aa_distance")
                                                                    for q in range(1, 5))))
        md.append("")
    md += ["## Rule 9 off (viruses and plasmids read both ways)", "",
           "The same readings with the triples whose donor gene has a homolog in A's context kept: stats.tsv, rows with rule9 = off.", "",
           "## Abstention, defined share of the source shift, calibration", "",
           "| arm | abstain: implant / original / own | calibration error (own) |", "|---|---|---|"]
    for a in rep["arms"]:
        if a["relatives"]:
            ab = a["abstain"]
            md.append("| %s (%s) | %s / %s / %s | %s%s |" % (a["arm"], a["weights"], _f(ab["implant"]), _f(ab["original"]), _f(ab["own"]),
                                                          _f(a["calibration"]["own"]["error"]), " (uncalibrated)" if a["calibration"]["uncalibrated"] else ""))
    md += ["", "| label | of | scenario | rule 9 | block | defined: implant / own | implant minus own [95%] |", "|---|---|---|---|---|---|---|"]
    for d in rep["defined"]:
        if d["score"] == "source_shift_defined" and d["label"] and not d["label"].startswith(("presence:", "untrained:")) and d.get("rule9") == "on":
            md.append("| %s | %s | %s | %s | %s | %s / %s | %s [%s, %s] |" % (d["label"], d.get("of", ""), d["scenario"], d["rule9"], d["block"],
                                                                            _f(d["implant_defined"]), _f(d["own_defined"]), _f(d["value"]), _f(d["lo"]), _f(d["hi"])))
    groups = [g for g in rep["groups"]]
    md += ["", "## Coverage (charter rule 7): direction x recipient group", "",
           "A ruler marks a cell \"construct only\" (at least %d triples in place, rule 9 on) or \"not evaluated\"; no cell here is \"evaluated\", so "
           "every cell reads unverified." % R.MIN_GROUP, "",
           "| direction | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    by = {(c["direction"], c["recipient"]): c for c in rep["coverage"]}
    for dname in R.DIRECTIONS:
        md.append("| %s | %s |" % (dname, " | ".join("%s (%d)" % ("construct" if by[(dname, g)]["mark"] == "construct only" else "-", by[(dname, g)]["triples"])
                                                    for g in groups)))
    md += ["", "Every stratum, both statistics, both comparisons (original is the position control) and the paired differences against the controls "
           "(best similarity, coverage shift, the untrained model, the presence passes, the host codon model) are in stats.tsv; every table's scores in scores.tsv."]
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
    with open(os.path.join(run_dir, OUT, "stats.tsv")) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    bad = [r for r in rows if r["value"] not in ("", None) and not 0.0 <= float(r["value"]) <= 1.0]
    if not rows or bad:
        problems.append(dict(error="no result row, or a win rate or AUC outside [0, 1]", n=len(bad)))
    if any(c.get("mark") == "evaluated" for c in rep.get("coverage") or []):
        problems.append(dict(error="a ruler never marks a cell evaluated"))
    dup = duplicate_keys(rows, STAT_KEY)
    if dup:                                                         # one result row per key: nothing overwrites another
        problems.append(dict(error="a result row twice in stats.tsv", n=len(dup), first=list(dup[0])))
    cp = os.path.join(run_dir, OUT, "criterion.tsv")
    if os.path.exists(cp):
        with open(cp) as f:
            crit = list(csv.DictReader(f, delimiter="\t"))
        if duplicate_keys(crit, CRITERION_KEY):
            problems.append(dict(error="a criterion row twice in criterion.tsv"))
    for name, key in (("verdicts", ("arm", "block")), ("second_verdicts", ("arm", "block")), ("arms", ("label",))):
        if duplicate_keys(rep.get(name) or [], key):
            problems.append(dict(error="%s holds one key twice" % name))
    with open(os.path.join(run_dir, OUT, "scores.tsv")) as f:
        header = f.readline().rstrip("\n").split("\t")
    if len(set(header)) != len(header):
        problems.append(dict(error="a score column twice in scores.tsv"))
    if not rep.get("tables", {}).get("sha256") or not rep.get("run"):
        problems.append(dict(error="the report does not name its run and tables"))
    require(GateResult("G4_rel_w18_readout", "fail" if problems else "pass", dict(problems=problems[:20], arms=[a["arm"] + ":" + a["weights"] for a in rep.get("arms") or []],
                                                                              rows=len(rows))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        rep = json.load(f)
    js = dict(tables=rep["tables"], arms=[a["arm"] + ":" + a["weights"] for a in rep["arms"]], headline=[h for h in rep["headline"] if h["stratum"] == "all"])
    md = ["W18 readouts on `%s` (%d tables, construct only): %s." % (rep["tables"]["name"], rep["tables"]["tables"], ", ".join(js["arms"])),
          "Win rate against another gene of A, in place, all tiers: " + "; ".join(
              "%s %s %s %s" % (h["label"], h["score"], h["block"][:10], _f(h["value"]) if h["readable"] else "too few")
              for h in js["headline"] if h["score"] in ("source_shift_aa", "gain_aa", "composition", "best_similarity", "host_codon"))]
    return dict(json=js, md=md)


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w18_readout") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
