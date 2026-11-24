"""Stage rel_w6_report (package W6; pipeline rel_traineval): the learning curves of the arms on the axis of tables seen,
the readings today's ruling asks for, and the classical baselines on the same held-out tables. `python -m
ci.stages.rel.w6_report RUN_DIR`. CPU only, numpy and the standard library; it reads result rows (the columns of the
contract), the model index and each arm's preflight record, nothing of a model.

Profile block rel_w6_report (all optional):
  fits       [run directory | {run, folder, label, accept}, ...]: runs whose stage rel_w6_fit trained arms (folder
             data/rel_w6/fit, the default) or whose stage rel_w6_eval evaluated their weights again (folder
             data/rel_w6/eval), of this machine or copied to it with record.json, stages/ and the folder. A run must
             have passed unless accept names its status (e.g. [gate_failed]); its index must be the file its stage
             record lists either way. label tells apart arms of one name from two runs (the arm is "<name> @<label>");
             a later entry with the same arms adds to and replaces what an earlier one gave (its final evaluation over
             a training run's, say). This run comes first when its own rel_w6_fit ran.
  readings_exclude  [arm, ...]: arms shown in the curves and tables but left out of the readings "relatives" and
             "scale" (an arm whose run was slowed, with another run of the same arm as the reading)
  scale_exclude [arm, ...]: arms left out of the reading "scale" only (arms of another layout, read for relatives)
  seeds      [[arm, arm, ...], ...]: arms of one variant with different seeds (or from different runs): their
             difference at every node all of them reached is the noise estimate; a difference of scale or of what
             relatives are worth no larger than it "cannot be read"; in the scale they are one level (their mean at a
             node all of them reached, the one that reached it otherwise, named in the report)
  lr_check   {small, low, high, note}: a big arm at a higher peak learning rate (high) against the same arm at the
             lower rate (low) and a small arm (small), at the last node all three reached and at the end, against the
             seeds' difference; one of three conclusions (caught up: the rate; better than the lower rate but not caught
             up: part; diverged or no better: not the rate); note is printed with it (the machines, say)
  pairs      [{name, first, second}]: two arms side by side (the second minus the first) at every node both reached and
             at the end, under scattered, whole_gene and suffix, against the seeds' difference
  notes      [text, ...]: what these runs do not show, added to the report's caveats (the machines they ran on, say)
  baselines  [{run, path}]: result tables of passed runs (the classical baselines of package W5: path
             data/rel_w5/baselines/rows.tsv; its results.json beside it gives the digest of the cells it scored)
  main       the condition of the headline curve (default scattered; an arm trained without relatives is read under
             the same condition without relatives)
Readings (each also in report.json):
  arms          parameters, learning rate, steps, tables seen, passes over the training tables, throughput, memory
  headline      the final evaluation by the identity of the nearest row (ten bins, the bin without other rows left
                out), every condition, arms and baselines; then the eukaryotic groups apart from the others
  relatives     what relatives are worth, twice: an arm without relatives at the lower of its best node and its end
                against the arm of the same size with relatives at its end, on all held-out tables; and the two at the
                same nodes, on the tables of the nodes; each against the seeds' difference where seeds are named
  scale         the arms with relatives at the last node all of them reached, and every two of them at the last node
                both reached: "bigger is better" when the bits fall with the size under both scattered and whole_gene
                (by more than the seeds' difference where it is known), else "cannot be read"
  seeds         the difference of the seeds at every node they all reached
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

import numpy as np
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
EVAL = os.path.join("data", "rel_w6", "eval")
NODE_RX = re.compile(r"^(ema|raw)@tables=(\d+)$")
NO_REL = "+" + cond.NO_RELATIVES
NONE_BIN = SC.FINE[-1]
WITH_RELATIVES = getattr(SC, "RELATIVES", (None,))[0]               # the stratum of the tables with a relative on the target
WITHOUT_RELATIVES = getattr(SC, "RELATIVES", (None, None))[1]
GROUP_FLOOR = 0.02                    # the least threshold of a group's own reading (the programme lead, 2026-11-23)
NODE_CELLS = "node_cells.npz"         # the trainer's bits per held-out table at every node


def held_units(conf, source):
    """(group name, split unit) of every held-out table of an arm (its training configuration conf: heldout_tables,
    heldout_root): the examined genome (row 0) of every table, by the genome list of the table set's index, and its
    split unit as stage train_data forms them (bench/rel/implant.split_units, package W18's copy of its rule) on the
    training-data index of the source run; units never cross the splits, so the held-out genomes alone give them."""
    from bench.rel import implant as I
    from ci.stages.rel import w2_v0
    from seeddrift.train.rel.data import tables as T
    held = T.open_split(conf["heldout_root"], conf["heldout_tables"], 1)
    doc = T.read_manifest(conf["heldout_tables"])
    idx = next((e for e in doc.get("files") or [] if e.get("role") == "index" and os.path.basename(str(e.get("path"))) == "index.json"), None)
    if idx is None:
        raise ValueError("the table set lists no index.json")
    with open(os.path.join(conf["heldout_root"], idx["path"])) as f:
        index = json.load(f)
    listed = index.get("genomes") or []
    names = [str(g.get("name")) if isinstance(g, dict) else str(g) for g in index.get("groups") or []]
    ids = [str(g["id"]) if isinstance(g, dict) else str(g) for g in listed]
    rows0 = np.asarray(held.load(("genome",))["genome"])[:, 0].astype(np.int64)
    mine = {ids[k] for k in rows0 if 0 <= k < len(ids)}
    metas = [m for m in w2_v0.source(source, pairs=False)["index"]["genomes"] if m["id"] in mine]
    units = I.split_units(metas)
    group = [names[int(g)] if int(g) < len(names) else str(int(g)) for g in np.asarray(held.group)]
    unit = [units.get(ids[k], ids[k]) if 0 <= k < len(ids) else "row0-%d" % i for i, k in enumerate(rows0)]
    return group, unit


def held_strata(conf):
    """{stratum label: bool per held-out table}: the evaluator's strata of the arm's held-out set (score.strata_of, with
    the kinds of the rows where it reads them), as the trainer's evaluation forms them."""
    import inspect
    from seeddrift.train.rel.data import tables as T
    held = T.open_split(conf["heldout_root"], conf["heldout_tables"], 1)
    got = held.load(("kind", "identity"))
    kind, identity = np.asarray(got["kind"]), np.asarray(got["identity"])
    other = cond.present_rows(kind)[:, 1:].sum(axis=1)
    strata = SC.strata_of(identity, other, kind) if "kind" in inspect.signature(SC.strata_of).parameters else SC.strata_of(identity, other)
    out = {}
    for labels, index in strata:
        for k, lab in enumerate(labels):
            out[lab] = np.asarray(index) == k
    return out
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
    "The six eukaryotic groups (fungi, invertebrates, mammals, other vertebrates, plants and algae, protists) hold only the first "
    "1,500 genes of each genome; they are reported apart from the six groups of prokaryotes, plasmids and viruses, and the mean over "
    "all twelve groups is not the headline.",
]


def settings(cfg) -> dict:
    sc = cfg.get("rel_w6_report") or {}
    unknown = set(sc) - {"fits", "baselines", "main", "readings_exclude", "seeds", "lr_check", "leak_pairs", "lr_sweeps", "passes", "notes", "pairs",
                         "scale_exclude", "units_source", "threshold_floor", "worth_without"}
    if unknown:
        raise SystemExit("rel_w6_report: unknown keys %s" % sorted(unknown))
    if str(sc.get("worth_without") or "lower") not in ("lower", "end"):
        raise SystemExit("rel_w6_report.worth_without: lower (the arm without relatives at the lower of its best node and its end) or end "
                         "(end against end, round 1's protocol), not %r" % sc.get("worth_without"))
    lr = sc.get("lr_check")
    if lr is not None and (set(lr) - {"small", "low", "high", "note"} or not all(lr.get(k) for k in ("small", "low", "high"))):
        raise SystemExit("rel_w6_report.lr_check: {small, low, high: arms; note: text}: %r" % (lr,))
    seeds = [[str(a) for a in g] for g in sc.get("seeds") or []]
    if any(len(g) < 2 or len(set(g)) != len(g) for g in seeds):
        raise SystemExit("rel_w6_report.seeds: groups of two or more different arms of one variant: %r" % (seeds,))
    base = [dict(run=str(b["run"]), path=str(b["path"])) for b in sc.get("baselines") or []]
    fits = []
    for x in sc.get("fits") or []:
        x = dict(run=x) if isinstance(x, str) else dict(x)
        if set(x) - {"run", "folder", "label", "accept"} or not x.get("run") or x.get("folder", FIT) not in (FIT, EVAL):
            raise SystemExit("rel_w6_report.fits: a run directory or {run, folder (%s or %s), label, accept}: %r" % (FIT, EVAL, x))
        fits.append(dict(run=str(x["run"]), folder=str(x.get("folder", FIT)), label=str(x.get("label") or ""), accept=[str(a) for a in x.get("accept") or []]))
    return dict(fits=fits, baselines=base, main=str(sc.get("main", cond.SCATTERED)), exclude=[str(a) for a in sc.get("readings_exclude") or []],
                seeds=seeds, lr_check={k: str(v) for k, v in lr.items()} if lr else None,
                leak_pairs=[[str(a), str(b)] for a, b in sc.get("leak_pairs") or []],
                lr_sweeps=[dict(name=str(w["name"]), arms=[[str(x) for x in e] if isinstance(e, list) else [str(e)] for e in w["arms"]])
                           for w in sc.get("lr_sweeps") or []],
                passes=[str(a) for a in sc.get("passes") or []], notes=[str(x) for x in sc.get("notes") or []],
                pairs=[dict(name=str(q["name"]), first=str(q["first"]), second=str(q["second"])) for q in sc.get("pairs") or []],
                scale_exclude=[str(a) for a in sc.get("scale_exclude") or []],
                units_source=str(sc["units_source"]) if sc.get("units_source") else None,
                threshold_floor=float(sc.get("threshold_floor") or 0.0), worth_without=str(sc.get("worth_without") or "lower"))


LR_CONCLUSIONS = {
    "caught up": "the big arm at the higher rate caught up with or passed the small arm: tonight's 'smaller is better' comes mainly from the "
                 "learning rate",
    "part": "the big arm at the higher rate is better than at the lower rate but still worse than the small arm: the learning rate is part of "
            "the cause, and scale still cannot be read",
    "not the rate": "the big arm at the higher rate diverged or is no better than at the lower rate: the learning rate is not the cause",
}


SCALING_SIGN = "At the same learning rate the bigger arm is better: the only sign of scaling today."


def lr_reading(small, low, high, t, N, noise=None, diverged=False):
    """The learning-rate check at tables seen t (None: no node of all three): a big arm at a higher peak rate (high) against
    the same arm at the lower rate (low) and a small arm (small). Under each of scattered and whole_gene: 'caught up' when
    high is above small by no more than the noise; else 'part' when high is below low by more than the noise; else 'not
    the rate'. The conclusion stands only when both conditions agree. N(arm, t, condition); noise(t, condition): None or
    {bits, node}; diverged: the high arm diverged or stopped at its preflight (then 'not the rate')."""
    both = (cond.SCATTERED, cond.WHOLE_GENE)
    if diverged:
        return dict(node=t, by_condition={}, conclusion="not the rate", why="the arm at the higher rate diverged", passed=False)
    by = {}
    for c in both:
        h, lo, s = (N(a, t, c) if t is not None else None for a in (high, low, small))
        n = noise(t, c) if noise else None
        thr = n["bits"] if n else 0.0
        if h is None or lo is None or s is None:
            by[c] = dict(kind=None, noise=n)
            continue
        kind = "caught up" if h - s <= thr else ("part" if lo - h > thr else "not the rate")
        by[c] = dict(kind=kind, high=h, low=lo, small=s, high_minus_small=h - s, low_minus_high=lo - h, noise=n, passed=s - h > thr)
    kinds = {x["kind"] for x in by.values()}
    if None in kinds or not by:
        return dict(node=t, by_condition=by, conclusion=None, why="a reading is missing", passed=False)
    if len(kinds) > 1:
        return dict(node=t, by_condition=by, conclusion=None, why="the two conditions disagree: cannot be read", passed=False)
    kind = kinds.pop()
    # passed: below the small arm by more than the noise under both conditions, the only sign of scaling of the day
    return dict(node=t, by_condition=by, conclusion=kind, why=None, passed=kind == "caught up" and all(x["passed"] for x in by.values()))


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


def index_of(run, rel, accept):
    """A model index of another run, checked against that run's stage record: from a passed run through
    handoff.from_run; from a run whose status `accept` names, the same check without the status."""
    rec = read_json(os.path.join(run, "record.json")) or {}
    if rec.get("status") == "passed":
        try:
            return handoff.from_run(run, rel)
        except (handoff.HandoffError, OSError, ValueError) as err:
            raise SystemExit("rel_w6_report: %s: %s" % (run, err))
    if rec.get("status") not in accept:
        raise SystemExit("rel_w6_report: run %s is %s, which the profile does not accept" % (run, rec.get("status")))
    p = os.path.join(run, rel)
    listed = [o for x in rec.get("stages") or [] for o in x.get("outputs") or [] if o.get("path") == rel]
    doc = read_json(p)
    if doc is None or not listed or any(o.get("sha256") != sha256_file(p, use_sidecar=False) for o in listed) or doc.get("sha256") != handoff.digest(doc):
        raise SystemExit("rel_w6_report: %s of run %s is not the file its stage record lists" % (rel, run))
    return doc


def collect(st, run_dir, s):
    """Rows, arms (model index entries with their run and variant, keyed by name and label), baseline rows and digests,
    sources, group names."""
    own = [dict(run=run_dir, folder=FIT, label="", accept=[])] if os.path.exists(os.path.join(run_dir, FIT, "models.json")) else []
    fits = own + [f for f in s["fits"] if not (os.path.realpath(f["run"]) == os.path.realpath(run_dir) and f["folder"] == FIT)]
    if not fits:
        raise SystemExit("rel_w6_report: no fit to report (this run trained nothing and rel_w6_report.fits names no run)")
    rows, arms, sources, groups, online = [], {}, [], [], {}
    for spec in fits:
        f, folder, label = spec["run"], spec["folder"], spec["label"]
        key = lambda name: name + (" @" + label if label else "")                                   # noqa: E731
        if f == run_dir and folder == FIT:
            doc = read_json(os.path.join(f, FIT, "models.json"))
        else:
            doc = index_of(f, os.path.join(folder, "models.json"), spec["accept"])
        for v in doc["variants"]:
            if (v.get("info") or {}).get("render") == "online":          # the same final weights read online (w6_eval online_too)
                if v.get("status") == "ok":
                    online[key(v["name"][:-len(".online")])] = dict(folder=os.path.join(f, v["info"]["folder"]), run=os.path.basename(f))
                continue
            pre = read_json(os.path.join(f, v["info"]["folder"], "preflight.json")) or {}
            node_items = {}
            try:                                                       # the digest of the cells the nodes scored (the same subset at every node)
                with open(os.path.join(f, v["info"]["folder"], "evals.jsonl")) as fh:
                    for line in fh:
                        e = json.loads(line)
                        if not e.get("final") and e.get("items"):
                            node_items = e["items"]
                            break
            except (OSError, ValueError):
                pass
            entry = dict(v, name=key(v["name"]), run=os.path.basename(f), run_dir=f, variant=variant_of(st, v["variant_file"]), train_tables=pre.get("train_tables"),
                         node_items=node_items)
            old = arms.get(entry["name"])
            if old is not None:                                        # a later entry adds to and replaces an earlier one
                entry["info"] = dict(old["info"], **{k: x for k, x in v["info"].items() if x is not None})
                for k in ("final", "final_items"):                      # an evaluation again of some conditions adds to the others
                    if isinstance(old["info"].get(k), dict) and isinstance(v["info"].get(k), dict):
                        entry["info"][k] = dict(old["info"][k], **v["info"][k])
                entry["train_tables"] = entry["train_tables"] or old.get("train_tables")
                entry["node_items"] = entry["node_items"] or old.get("node_items")
                entry["run"] = "%s + %s" % (old["run"], entry["run"]) if old["run"] != entry["run"] else entry["run"]
            arms[entry["name"]] = entry
        names = {v["name"] for v in doc["variants"]}
        for r in SC.read_rows(os.path.join(f, folder, "results.tsv")):
            m = NODE_RX.match(r["weights"]) if folder == EVAL else None
            if m and r["variant"] + ".node@" + m.group(2) in names:
                # an evaluation again of a node's moving average (w6_eval weights node@<n>, its rows under the arm's own name):
                # the entry <arm>.node@<n> of its own, never a node of the training's curve
                r = dict(r, variant=r["variant"] + ".node@" + m.group(2))
            rows.append(dict(dict(r, variant=key(r["variant"])) if r["variant"] in names else r, _source=len(sources)))
        groups = groups or list(doc.get("groups") or [])
        sources.append(dict(run=os.path.basename(f), folder=folder, label=label, models_sha256=doc["sha256"], tablesets=doc["tablesets"]))
    base, digests, sub = [], {}, dict(rows=[], digests={})
    for b in s["baselines"]:
        p = os.path.join(b["run"], b["path"])
        status = (read_json(os.path.join(b["run"], "record.json")) or {}).get("status")
        if status != "passed":
            raise SystemExit("rel_w6_report: the baselines' run %s is %s, not passed" % (b["run"], status))
        got = SC.read_rows(p)
        base += got
        res = read_json(os.path.join(os.path.dirname(p), "results.json")) or {}
        digests.update({c: x.get("sha256") for c, x in (res.get("items") or {}).items() if isinstance(x, dict)})
        # the baselines on the subset of tables the nodes score (rows_subset.tsv beside the rows; package W5)
        q = os.path.join(os.path.dirname(p), "rows_subset.tsv")
        if os.path.exists(q):
            sub["rows"] += SC.read_rows(q)
            sub["digests"].update({c: x.get("sha256") for c, x in ((res.get("subset") or {}).get("items") or {}).items() if isinstance(x, dict)})
        sources.append(dict(run=os.path.basename(b["run"]), baselines=b["path"], sha256=sha256_file(p, use_sidecar=False), rows=len(got),
                            subset_rows=os.path.exists(q)))
    return rows, arms, base, digests, sources, groups, sub, online


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


def condition_labels(rows):
    """{(arm, condition, layout): label}: an arm whose rows read one condition in two layouts (an arm trained leak-free
    reads the whole gene also in the stored layout it was trained in) has the stored rows under "<condition>@stored";
    the rows of a control of the renderer (layout "<layout>+<control>") are "<condition>@<control>", never the
    condition's own; every other row keeps its condition."""
    seen = {}
    for r in rows:
        seen.setdefault((r["variant"], r["condition"]), set()).add(r.get("layout") or "stored")

    def label(c, x, xs):
        if "+" in x:                                                   # a control of the renderer (leakfree+shuffle): "<condition>@shuffle"
            return "%s@%s" % (c, x.rsplit("+", 1)[1])
        return "%s@%s" % (c, x) if len([y for y in xs if "+" not in y]) > 1 and x == "stored" else c
    return {(v, c, x): label(c, x, xs) for (v, c), xs in seen.items() for x in xs}


def index_rows(rows, arms, base):
    """{(arm, point, tables, condition label, group, stratum): bits}; baselines under point 'final'. Two rows of one
    source (one run's result table) that come to one key are refused: a reading would take one of them for the other
    (round 1's first report read a control as its plain condition); a later source replaces an earlier one's key."""
    out, seen = {}, {}
    label = condition_labels(rows)
    for r in rows:
        if r["variant"] not in arms:
            continue
        p = point_of(r)
        if p is not None:
            c = label.get((r["variant"], r["condition"], r.get("layout") or "stored"), r["condition"])
            k = (r["variant"], p[0], p[1], c, r["group"], r["stratum"])
            src = r.get("_source")
            if (src, k) in seen and seen[(src, k)] != r.get("layout"):
                raise SystemExit("rel_w6_report: two rows of one result table come to one reading %s (layouts %s and %s): the report would "
                                 "read one for the other" % (k, seen[(src, k)], r.get("layout")))
            if (src, k) in seen:
                raise SystemExit("rel_w6_report: a result table holds the reading %s twice (layout %s)" % (k, r.get("layout")))
            seen[(src, k)] = r.get("layout")
            out[k] = float(r["bits"])
    for r in base:
        if number(r["bits"]) is not None:
            out[(r["variant"], "final", None, r["condition"], r["group"], r["stratum"])] = float(r["bits"])
    return out


def same_items(arms, digests):
    """{condition: whether every trained arm scored, in its final evaluation, the cells the baselines scored}; only
    conditions for which both digests are known."""
    out = {}
    for c in list(cond.CONDITIONS) + list(getattr(cond, "EXTRA_CONDITIONS", ())):
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


def scale_readings(rel, params, nodes, N, noise=None):
    """(scale, pairs): the arms (or levels) with relatives, ordered by parameters, at the last node all of them reached,
    with the verdict 'bigger is better' only when the bits fall with every step in size under both conditions by more
    than the noise (where noise gives one), else 'cannot be read' (None without two arms or such a node); and every two
    of them at the last node both reached, under the same rule. nodes: {arm: [tables seen at its nodes]};
    N(arm, tables, condition): the bits of the node, or None; noise(tables, condition): None or {bits, node}."""
    both = (cond.SCATTERED, cond.WHOLE_GENE)
    order = sorted(rel, key=lambda a: params.get(a) or 0)

    def last(xs):
        common = sorted(set.intersection(*[set(nodes.get(a, [])) for a in xs]) - {0})
        return common[-1] if common else None

    def step(big, small, t):
        for c in both:
            b, s = N(big, t, c), N(small, t, c)
            if b is None or s is None:
                return "a reading is missing"
            if not b < s:
                return "not lower"
            n = noise(t, c) if noise else None
            if n is not None and s - b <= n["bits"]:
                return "lower by no more than the seeds' difference"
        return "lower"

    def reading(xs, t):
        steps = [step(xs[j + 1], xs[j], t) for j in range(len(xs) - 1)]
        return dict(node=t, arms=list(xs), bits={a: {c: N(a, t, c) for c in both} for a in xs}, steps=steps,
                    noise={c: noise(t, c) for c in both} if noise else None,
                    verdict="bigger is better" if all(x == "lower" for x in steps) else "cannot be read")
    pairs = [reading([small, big], last([small, big])) for i, small in enumerate(order) for big in order[i + 1:] if last([small, big]) is not None]
    t = last(order) if len(order) >= 2 else None
    return (reading(order, t) if t is not None else None), pairs


def build(st, cfg, run_dir):
    s = settings(cfg)
    rows, arms, base, digests, sources, groups, sub, online = collect(st, run_dir, s)
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    bits = index_rows(rows, arms, base)
    ok = sorted(a for a in arms if arms[a]["status"] == "ok")
    rel = [a for a in ok if arms[a]["variant"]["relatives"] and a not in s["exclude"]]
    norel = [a for a in ok if not arms[a]["variant"]["relatives"] and a not in s["exclude"]]
    bnames = sorted({r["variant"] for r in base if number(r["bits"]) is not None})
    names = sorted(arms)
    cols = names + bnames
    F = lambda v, c, g=SC.ALL, x=SC.ALL: bits.get((v, "final", None, c, g, x))                     # noqa: E731
    N = lambda v, t, c, g=SC.ALL, x=SC.ALL: bits.get((v, "node", t, c, g, x))                       # noqa: E731
    extra = sorted({k[3] for k in bits if "@" in k[3]})                # a condition read in a second layout (condition_labels)
    conds = [c for c in list(cond.CONDITIONS) + list(getattr(cond, "EXTRA_CONDITIONS", ())) + extra if any(F(v, c) is not None for v in cols)]
    reading = {a: (s["main"] if arms[a]["variant"]["relatives"] else s["main"] + NO_REL) for a in names}
    nodes = {a: sorted({k[2] for k in bits if k[0] == a and k[1] == "node"}) for a in names}
    same = same_items(arms, digests)
    same_subset = {}
    for c in cond.CONDITIONS:                                         # the nodes against the baselines' subset
        mine = {a["node_items"].get(c) for a in arms.values() if a["status"] == "ok"} - {None}
        if mine and sub["digests"].get(c):
            same_subset[c] = mine == {sub["digests"][c]}
    lines = {}                                                        # {condition: {baseline: bits}} on the subset, over the groups
    for r in sub["rows"]:
        if r["group"] == SC.ALL and r["stratum"] == SC.ALL and number(r["bits"]) is not None:
            lines.setdefault(r["condition"], {})[r["variant"]] = float(r["bits"])

    # seeds: arms of one variant with different seeds (or runs); their difference at every node they all reached is the
    # only noise estimate; a reading of scale or of what relatives are worth within it cannot be read
    x0 = WITH_RELATIVES or SC.ALL                                      # the tables with a relative on the target block
    both = (cond.SCATTERED, cond.WHOLE_GENE)
    seed_groups, seeds_out, spread = [], [], {}
    for g in s["seeds"]:
        have = [a for a in g if a in arms and arms[a]["status"] == "ok"]
        files = {arms[a]["variant_file"] for a in g if a in arms}
        if len(files) > 1:
            raise SystemExit("rel_w6_report.seeds: %s are not one variant (%s)" % (g, sorted(files)))
        if len(have) < 2:
            seeds_out.append(dict(arms=g, trained=have, rows=[], note="fewer than two of these arms trained"))
            continue
        seed_groups.append(have)
        common = sorted(set.intersection(*[set(nodes.get(a, [])) for a in have]) - {0})
        table = []
        for t in common:
            row = [t]
            for c in both:
                xs = [N(a, t, c) for a in have]
                d = None if any(x is None for x in xs) else (xs[-1] - xs[0] if len(xs) == 2 else max(xs) - min(xs))
                row += xs + [d]
                for x in (SC.ALL, x0):
                    ys = [N(a, t, c, SC.ALL, x) for a in have]
                    if all(y is not None for y in ys):
                        spread[(t, c, x)] = max(spread.get((t, c, x), 0.0), max(ys) - min(ys))
            for c in [c for c in cond.CONDITIONS if c not in both]:  # the conditions the round-1a readings add (no relatives)
                ys = [N(a, t, c) for a in have]
                if all(y is not None for y in ys):
                    spread[(t, c, SC.ALL)] = max(spread.get((t, c, SC.ALL), 0.0), max(ys) - min(ys))
            table.append(row)
        seeds_out.append(dict(arms=have, seeds=[arms[a]["info"].get("seed") for a in have], runs=[arms[a]["run"] for a in have], rows=table))

    # the seeds' largest spread per group under scattered at the last node they all reached (the noise of a group's
    # reading: a group is read on its own 50 tables of the nodes, so its spread is larger than that of the mean)
    group_spread = {}                                                  # {(group, condition): the seeds' spread}
    for have in seed_groups:
        common = sorted(set.intersection(*[set(nodes.get(a, [])) for a in have]) - {0})
        if not common:
            continue
        for g in sorted({k[4] for k in bits if k[4] != SC.ALL}):
            for c in (cond.SCATTERED, cond.WHOLE_GENE):
                ys = [N(a, common[-1], c, g) for a in have]
                if all(y is not None for y in ys):
                    d = max(ys) - min(ys)
                    if (g, c) not in group_spread or d > group_spread[(g, c)]["bits"]:
                        group_spread[(g, c)] = dict(bits=d, node=common[-1])

    def noise(t, c, x=SC.ALL):
        """{bits, node}: the largest spread of the seeds under condition c at node t, on the stratum x of the nodes'
        tables; at a node they did not all reach (or at the end, t None), at the last node they all reached."""
        at = sorted(u for (u, cc, xx) in spread if cc == c and xx == x)
        if not at:
            return None
        u = at[-1] if t is None or t > at[-1] else max([v for v in at if v <= t] or at[:1])
        return dict(bits=max(spread[(u, c, x)], s["threshold_floor"]), node=u, seeds=spread[(u, c, x)])
    beyond = lambda d, n: None if d is None or n is None else ("beyond the seeds' difference" if abs(d) > n["bits"] else "cannot be read")   # noqa: E731
    fmt_noise = lambda n: ("%.3f (at %s tables)" % (n["bits"], "{:,}".format(n["node"])) if n.get("seeds", n["bits"]) >= n["bits"] else      # noqa: E731
                           "%.3f (the floor; the seeds' %.3f at %s tables)" % (n["bits"], n["seeds"], "{:,}".format(n["node"]))) if n else "none"
    # the cells of an arm per held-out table, for paired intervals (the evaluator's intervals.paired_bootstrap)
    cache = {}

    def folder_with(a, name):
        """The folder of arm a that holds `name`: its own (the run that read it last), else its training run's."""
        base = os.path.basename(arms[a]["info"]["folder"])
        for f in [os.path.join(arms[a]["run_dir"], arms[a]["info"]["folder"])] + [os.path.join(x["run"], FIT, base) for x in s["fits"]] + \
                [os.path.join(run_dir, FIT, base)]:
            if os.path.exists(os.path.join(f, name)):
                return f
        return None

    def training_summary(a):
        """The summary the arm's training wrote (its mask schemes, frame relatives, ...): the first summary.json of its
        folders that has them; {} without."""
        base = os.path.basename(arms[a]["info"]["folder"])
        for f in [os.path.join(arms[a]["run_dir"], arms[a]["info"]["folder"])] + [os.path.join(x["run"], FIT, base) for x in s["fits"]] + \
                [os.path.join(run_dir, FIT, base)]:
            sm = read_json(os.path.join(f, "summary.json")) or {}
            if sm.get("schemes") or sm.get("frame_tiers"):
                return sm
        return {}

    def held_of(a):
        """(group, unit, strata {label: bool per table}) of the held-out set arm a was read on; None without units_source."""
        f = folder_with(a, "conf.json")
        conf = read_json(os.path.join(f, "conf.json")) if f else None
        if not conf or not s["units_source"]:
            return None
        key = conf.get("heldout_tables")
        if key not in cache:
            try:
                group, unit = held_units(conf, s["units_source"])
                cache[key] = (group, unit, held_strata(conf))
            except (OSError, ValueError, KeyError, SystemExit):
                cache[key] = None
        return cache[key]

    def table_cells(a, which, c):
        """(bits summed per held-out table, cells per table) of arm a under condition c: which "final" or "best" (the
        cells of its last evaluation, heldout_<which>.npz) or a node of tables seen (node_cells.npz)."""
        cc = c.replace("+", "__")
        if isinstance(which, int):
            f = folder_with(a, NODE_CELLS)
            if f is None:
                return None
            z = np.load(os.path.join(f, NODE_CELLS))
            k = "t%d_%s" % (which, cc)
            if k + "_bits" not in z:
                return None
            n = int(np.max(z["tables"])) + 1 if len(z["tables"]) else 0
            bits_t, cells_t = np.zeros(n), np.zeros(n)
            bits_t[z["tables"]], cells_t[z["tables"]] = z[k + "_bits"], z[k + "_cells"]
            return bits_t, cells_t
        f = folder_with(a, "heldout_%s.npz" % which)
        if f is None:
            return None
        z = np.load(os.path.join(f, "heldout_%s.npz" % which))
        if "bits_" + cc not in z:
            return None
        t = np.asarray(z["table_" + cc]).astype(np.int64)
        n = int(t.max()) + 1 if len(t) else 0
        return np.bincount(t, weights=np.asarray(z["bits_" + cc], dtype=np.float64), minlength=n), np.bincount(t, minlength=n).astype(np.float64)

    def paired(a, wa, ca, b, wb, cb, x, only_group=None):
        """The paired bootstrap of (b under cb at wb) minus (a under ca at wa) on the tables of stratum x both scored
        on the same cells: {"all", block: dict(estimate, low, high, excludes_zero, ...)}, or None (and why)."""
        try:
            from seeddrift.train.rel.evalh import intervals as IV
        except ImportError:
            return None, "no intervals module of the evaluator"
        h = held_of(a)
        if h is None:
            return None, "no split units (units_source)"
        A, B = table_cells(a, wa, ca), table_cells(b, wb, cb)
        if A is None or B is None:
            return None, "no cells per table"
        group, unit, strata = h
        n = min(len(A[0]), len(B[0]), len(group))
        keep = (A[1][:n] > 0) & (B[1][:n] > 0) & np.asarray(strata.get(x, np.ones(len(group), bool)))[:n]
        if only_group is not None:
            keep &= np.array([gname == only_group for gname in group[:n]])
        if not keep.any():
            return None, "no tables"
        if not np.array_equal(A[1][:n][keep], B[1][:n][keep]):
            return None, "not the same cells"
        idx = np.flatnonzero(keep)
        return IV.paired_bootstrap((B[0][:n] - A[0][:n])[idx], A[1][:n][idx], [group[k] for k in idx], [unit[k] for k in idx]), None
    fmt_iv = lambda r: "%.3f to %.3f" % (r["low"], r["high"]) if r else ""                                                                     # noqa: E731

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
    caveats = list(CAVEATS) + list(s["notes"])                         # the profile's own notes on these runs
    if extra:                                                          # arms trained leak-free on scattered and span, stored on the whole gene
        caveats.append("Arms trained leak-free saw the whole gene in the stored layout: their %s rows are read leak-free on the frame relative, "
                       "a layout they were not trained on (trained stored, evaluated leak-free; a reference only), and the rows %s in the "
                       "stored layout they were trained in." % (" and ".join(x.split("@")[0] for x in extra), ", ".join(extra)))
    if seed_groups:                                                    # one arm has two seeds: their difference is the noise estimate
        caveats[5] = caveats[5].replace("One seed per arm, no noise floor", "One seed per arm except %s, whose difference (seeds and machines at once) is "
                                        "the only noise estimate" % "; ".join(" and ".join(g) for g in seed_groups))
    caveats += ["The arms with relatives drew tables without other rows at most at 5%% of a group's draws (drawn: %s); %s of the held-out "
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
        for a in partners:
            for c in (cond.SCATTERED, cond.WHOLE_GENE):
                # the arm without relatives at its best on all tables: its best node, or its end where the end is lower
                # (the best node is chosen on the nodes' tables, the end is evaluated on all tables too)
                at_best = lambda x, c=c: bits.get((b, "best", best.get("tables_seen"), c + NO_REL, SC.ALL, x)) if has_best else None     # noqa: E731
                at_end = lambda x, c=c: F(b, c + NO_REL, SC.ALL, x)                                                                      # noqa: E731
                use_best = at_best(x0) is not None and (at_end(x0) is None or at_best(x0) <= at_end(x0)) and s["worth_without"] != "end"
                without = at_best if use_best else at_end
                common_nodes = sorted(set(nodes.get(a, [])) & set(nodes.get(b, [])) - {0})
                last = common_nodes[-1] if common_nodes else None
                worth_end = diff(without(x0), F(a, c, SC.ALL, x0))
                worth_node = diff(N(b, last, c + NO_REL, SC.ALL, x0), N(a, last, c, SC.ALL, x0)) if last else None
                n_end, n_node = noise(None, c, x0), (noise(last, c, x0) if last else None)
                iv_end, why_end = paired(a, "final", c, b, ("best" if use_best else "final"), c + NO_REL, x0)
                iv_node, why_node = paired(a, last, c, b, last, c + NO_REL, x0) if last else (None, "no common node")

                def counts(worth, n, iv):
                    if worth is None or n is None:
                        return None
                    above, excl = abs(worth) > n["bits"], bool(iv and iv["all"]["excludes_zero"])
                    return ("counts: beyond the threshold, its interval excludes 0" if above and excl else
                            "beyond the threshold only" if above else "its interval only" if excl else "cannot be read")
                relatives.append(dict(with_relatives=a, without=b, condition=c, tables=x0, noise_end=n_end, reading_end=beyond(worth_end, n_end),
                                      interval_end=iv_end, interval_end_why=why_end, interval_node=iv_node, interval_node_why=why_node,
                                      counts_end=counts(worth_end, n_end, iv_end), counts_node=counts(worth_node, n_node, iv_node),
                                      noise_at_node=n_node, reading_at_node=beyond(worth_node, n_node),
                                      without_at=("best node (tables %s)" % best.get("tables_seen")) if use_best else
                                      ("end (end against end)" if s["worth_without"] == "end" else
                                       "end (below its best node at tables %s)" % best.get("tables_seen") if has_best else "end"),
                                      bits_without_best_node=at_best(x0), bits_without_end=at_end(x0),
                                      bits_without=without(x0), bits_with_end=F(a, c, SC.ALL, x0), worth=worth_end,
                                      same_model_rows_removed=diff(F(a, c + NO_REL, SC.ALL, x0), F(a, c, SC.ALL, x0)), node=last,
                                      worth_at_node=worth_node,
                                      per_node=[[t, N(b, t, c + NO_REL, SC.ALL, x0), N(a, t, c, SC.ALL, x0), diff(N(b, t, c + NO_REL, SC.ALL, x0), N(a, t, c, SC.ALL, x0))]
                                                for t in common_nodes],
                                      by_bin=[[x.replace(SC.FINE_PREFIX, ""), without(x), F(a, c, SC.ALL, x), diff(without(x), F(a, c, SC.ALL, x))]
                                              for x in SC.FINE if x != NONE_BIN and (without(x) is not None or F(a, c, SC.ALL, x) is not None)]))
    # scale: the arms with relatives at the last node all of them reached; every two of them at the last node both reached.
    # The seeds of one variant are one level: their mean at a node all of them reached, the one that reached it otherwise.
    rel_scale = [a for a in rel if a not in s["scale_exclude"]]         # e.g. arms of another layout: no step in size
    level_of = {a: " + ".join(g) for g in seed_groups if all(x in rel_scale for x in g) for a in g}
    levels = {}
    for a in rel_scale:
        levels.setdefault(level_of.get(a, a), []).append(a)

    def NL(level, t, c, x=SC.ALL):
        xs = [N(a, t, c, SC.ALL, x) for a in levels[level]]
        xs = [y for y in xs if y is not None]
        return sum(xs) / len(xs) if xs else None
    scale, scale_pairs = scale_readings(list(levels), {k: arms[v[0]]["info"].get("params") for k, v in levels.items()},
                                        {k: sorted(set().union(*[set(nodes.get(a, [])) for a in v])) for k, v in levels.items()}, NL,
                                        noise if spread else None)
    for r in ([scale] if scale else []) + scale_pairs:
        r["seeds_read"] = {k: [a for a in levels[k] if N(a, r["node"], cond.SCATTERED) is not None] for k in r["arms"] if len(levels[k]) > 1}
    # the learning-rate check: a big arm at a higher peak rate against the same arm at the lower rate and a small arm
    lr_check = None
    if s["lr_check"]:
        trio = [s["lr_check"][k] for k in ("small", "low", "high")]
        lack = [a for a in trio if a not in arms]
        if lack:
            raise SystemExit("rel_w6_report.lr_check: %s are no arms of the report (its arms: %s)" % (lack, names))
        small_arm, low_arm, high_arm = trio
        trained = [a for a in trio if arms[a]["status"] == "ok"]
        err = " ".join((arms[high_arm].get("error") or "").split()).lower()
        # diverged in training, or stopped by the preflight's guard (a loss that was not finite or did not fall)
        diverged = arms[high_arm]["status"] != "ok" and ("diverg" in err or "did not train" in err
                                                          or bool((arms[high_arm]["info"].get("guard") or {}).get("failed")))
        common = sorted(set.intersection(*[set(nodes.get(a, [])) for a in trained]) - {0}) if len(trained) == 3 else []
        last_t = common[-1] if common else None
        at_node = lr_reading(small_arm, low_arm, high_arm, last_t, N, noise if spread else None, diverged)
        at_end = lr_reading(small_arm, low_arm, high_arm, "end", lambda v, t, c: F(v, c), (lambda t, c: noise(None, c)) if spread else None, diverged)
        if arms[high_arm]["status"] != "ok" and not diverged:
            at_node = dict(at_node, conclusion=None, why="the arm at the higher rate did not train (%s)" % err[-200:])
            at_end = dict(at_end, conclusion=None, why=at_node["why"])
        lr_check = dict(arms=dict(small=small_arm, low=low_arm, high=high_arm), lr={a: arms[a]["info"].get("lr") for a in trio},
                        runs={a: arms[a]["run"] for a in trio}, tables_seen={a: arms[a]["info"].get("tables_seen") for a in trio},
                        note=s["lr_check"].get("note"), at_node=at_node, at_end=at_end,
                        conclusion=LR_CONCLUSIONS.get(at_node["conclusion"]) if at_node["conclusion"] else None,
                        passed=bool(at_node.get("passed")), sign=SCALING_SIGN if at_node.get("passed") else None,
                        curves=[[t] + [N(a, t, c) for c in both for a in trio] for t in sorted(set().union(*[set(nodes.get(a, [])) for a in trio]) - {0})],
                        final=[["end"] + [F(a, c) for c in both for a in trio]])
    # the leak of the stored layout: an arm trained and read leak-free against the same arm trained and read stored
    leak = []
    for lf, st_arm in s["leak_pairs"]:
        if lf not in arms or st_arm not in arms:
            raise SystemExit("rel_w6_report.leak_pairs: %s are no arms of the report" % [x for x in (lf, st_arm) if x not in arms])
        for c in (cond.SCATTERED, getattr(cond, "SUFFIX", "suffix")):
            for x in (SC.ALL, x0):
                a, b = F(lf, c, SC.ALL, x), F(st_arm, c, SC.ALL, x)
                d = diff(a, b)
                n = noise(None, c, x) if spread else None
                leak.append(dict(leakfree=lf, stored=st_arm, condition=c, tables=x, bits_leakfree=a, bits_stored=b, leak=d, noise=n,
                                 reading=beyond(d, n) if n else None))
    leak_nodes = []                                                    # at every node both reached: equal tables seen
    for lf, st_arm in s["leak_pairs"]:
        common = sorted(set(nodes.get(lf, [])) & set(nodes.get(st_arm, [])) - {0})
        for t in common:
            for x in (SC.ALL, x0):
                a, b = N(lf, t, cond.SCATTERED, SC.ALL, x), N(st_arm, t, cond.SCATTERED, SC.ALL, x)
                n = noise(t, cond.SCATTERED, x) if spread else None
                leak_nodes.append(dict(leakfree=lf, stored=st_arm, node=t, tables=x, bits_leakfree=a, bits_stored=b, leak=diff(a, b), noise=n,
                                       reading=beyond(diff(a, b), n) if n else None))
    # pairs of arms: the second minus the first, at every node both reached and at the end, against the seeds' difference
    pairs_out = []
    for q in s["pairs"]:
        if q["first"] not in arms or q["second"] not in arms:
            raise SystemExit("rel_w6_report.pairs %s: %s are no arms of the report" % (q["name"], [x for x in (q["first"], q["second"]) if x not in arms]))
        a1, a2 = q["first"], q["second"]
        common = sorted(set(nodes.get(a1, [])) & set(nodes.get(a2, [])) - {0})
        prow = []
        for t in common + ["end"]:
            row = [t]
            for c in (cond.SCATTERED, cond.WHOLE_GENE, getattr(cond, "SUFFIX", "suffix")):
                u, v = (F(a1, c), F(a2, c)) if t == "end" else (N(a1, t, c), N(a2, t, c))
                n = noise(None if t == "end" else t, c) if spread else None
                row += [u, v, diff(v, u), beyond(diff(v, u), n) if n else None]
            prow.append(row)
        pairs_out.append(dict(name=q["name"], first=a1, second=a2, rows=prow,
                              tables_seen={a: arms[a]["info"].get("tables_seen") for a in (a1, a2)}))
    # learning-rate sweeps: arms of one size at several peak rates (a seed group as one entry: its mean where all of it
    # reached a node, the one that reached it otherwise) at every node all entries reached, the best against the next
    sweep_conds = [c for c in (cond.SCATTERED, cond.SCATTERED + NO_REL, cond.WHOLE_GENE + NO_REL)]
    sweeps = []
    for w in s["lr_sweeps"]:
        lack = [a for g in w["arms"] for a in g if a not in arms]
        if lack:
            raise SystemExit("rel_w6_report.lr_sweeps %s: %s are no arms of the report" % (w["name"], lack))
        ents = [(" + ".join(g), g) for g in w["arms"]]
        NG = lambda g, t, c: (lambda xs: sum(xs) / len(xs) if xs else None)([y for y in (N(a, t, c) for a in g) if y is not None])  # noqa: E731
        FG = lambda g, c: (lambda xs: sum(xs) / len(xs) if xs else None)([y for y in (F(a, c) for a in g) if y is not None])        # noqa: E731
        reach = [set().union(*[set(nodes.get(a, [])) for a in g]) for _, g in ents]
        common = sorted(set.intersection(*reach) - {0}) if reach else []
        last = common[-1] if common else None
        best = {}
        for c in sweep_conds:
            vals = [(NG(g, last, c), lab) for lab, g in ents] if last else []
            vals = sorted(v for v in vals if v[0] is not None)
            if len(vals) >= 2:
                n = noise(last, c) if spread else None
                margin = vals[1][0] - vals[0][0]
                best[c] = dict(entry=vals[0][1], bits=vals[0][0], next=vals[1][1], margin=margin, noise=n,
                               reading=beyond(margin, n) if n else None)
        sweeps.append(dict(name=w["name"], entries=[lab for lab, _ in ents], lr={lab: arms[g[0]]["info"].get("lr") for lab, g in ents},
                           tables_seen={lab: [arms[a]["info"].get("tables_seen") for a in g] for lab, g in ents}, node=last, best=best,
                           conditions=sweep_conds, rows=[[t] + [NG(g, t, c) for c in sweep_conds for _, g in ents] for t in common],
                           final=[[FG(g, c) for c in sweep_conds for _, g in ents]]))
    # the curve over passes of the training tables: held-out bits at every node, the fall from the node before, and the
    # training loss of the 1,000 tables before the node (a fall of the training loss beside a rise of the held-out bits
    # is overfitting)
    over, iv_why, iv_cache = [], {}, {}

    def interval(a, g, t_lo, t_hi, c=cond.SCATTERED):
        """The paired bootstrap (the evaluator's intervals.paired_bootstrap: split units inside the group, 2,000 replicates,
        seed 20261123) of the change of arm a's scattered bits in group g from node t_lo to node t_hi, on the nodes' tables;
        None (the reason in iv_why) without the trainer's node_cells.npz, the units' source or the evaluator's intervals."""
        if a not in iv_cache:
            iv_cache[a] = None
            folder = folder_with(a, NODE_CELLS)                        # the training run's folder (an evaluation again has none)
            if folder is None:
                iv_why[a] = "no bits per table at the nodes (trained before node_cells.npz)"
                return None
            if not s["units_source"]:
                iv_why[a] = "no split units (units_source not given)"
                return None
            try:
                from seeddrift.train.rel.evalh import intervals as IV
            except ImportError:
                iv_why[a] = "no intervals module of the evaluator"
                return None
            try:
                group, unit = held_units(read_json(os.path.join(folder, "conf.json")) or {}, s["units_source"])
            except (OSError, ValueError, KeyError, SystemExit) as err:
                iv_why[a] = "no split units (%s)" % str(err)[:200]
                return None
            iv_cache[a] = (IV, np.load(os.path.join(folder, NODE_CELLS)), group, unit)
        if iv_cache[a] is None:
            return None
        IV, nc, group, unit = iv_cache[a]
        lo, hi = ("t%d_%s" % (t, c.replace("+", "__")) for t in (t_lo, t_hi))
        if lo + "_bits" not in nc or hi + "_bits" not in nc or not np.array_equal(nc[lo + "_cells"], nc[hi + "_cells"]):
            return None
        tabs = np.asarray(nc["tables"]).astype(np.int64)
        m = np.array([group[k] == g for k in tabs])
        if not m.any():
            return None
        got = IV.paired_bootstrap((nc[hi + "_bits"] - nc[lo + "_bits"])[m], nc[hi + "_cells"][m], [group[k] for k in tabs[m]],
                                  [unit[k] for k in tabs[m]], blocks=False)
        return got.get("all")

    for a in s["passes"]:
        if a not in arms:
            raise SystemExit("rel_w6_report.passes: %s is no arm of the report" % a)
        tt = arms[a]["train_tables"] or 0
        steps = []
        try:
            with open(os.path.join(folder_with(a, "steps.csv") or os.devnull + ".none", "steps.csv")) as fh:
                head = fh.readline().strip().split(",")
                ti, li = head.index("tables_seen"), head.index("loss_bits")
                steps = [(int(x[ti]), float(x[li])) for x in (line.strip().split(",") for line in fh) if len(x) > li]
        except (OSError, ValueError):
            steps = []
        rows_p, prev = [], None
        for t in [u for u in nodes.get(a, []) if u > 0]:
            near = [lo for tab, lo in steps if t - 1000 < tab <= t]
            cur = N(a, t, cond.SCATTERED)
            fall = diff(prev, cur)
            n = noise(t, cond.SCATTERED) if spread else None
            rows_p.append([t, t / tt if tt else None, cur, N(a, t, cond.SCATTERED + NO_REL), N(a, t, cond.WHOLE_GENE + NO_REL), fall,
                           beyond(fall, n) if (fall is not None and n) else None, sum(near) / len(near) if near else None])
            prev = cur
        # by group: the node of its lowest held-out bits, and how far they rose after it (the programme lead's ruling of
        # 2026-11-23: the over-fitting of round 1 is read group by group; the mean over the groups hides a group that
        # over-fits first, the eukaryotic viruses of version 1 say)
        by_g = []
        for g, gc in [(g, c) for c in ((cond.SCATTERED, cond.WHOLE_GENE) if arms[a]["variant"]["relatives"] else (cond.SCATTERED + NO_REL,))
                      for g in group_set]:
            pts = [(t, N(a, t, gc, g)) for t in nodes.get(a, []) if t > 0]
            pts = [(t, y) for t, y in pts if y is not None]
            if not pts:
                continue
            k = min(range(len(pts)), key=lambda i: pts[i][1])
            t0, y0 = pts[k]
            nxt = pts[k + 1] if k + 1 < len(pts) else None
            rise = pts[-1][1] - y0 if k + 1 < len(pts) else None
            gn = group_spread.get((g, gc))
            gn = dict(gn, bits=max(gn["bits"], GROUP_FLOOR)) if gn else None        # a pair of seeds can differ by 0 in a group by chance
            above = rise is not None and gn is not None and rise > gn["bits"]
            iv = interval(a, g, t0, pts[-1][0], gc) if rise is not None else None   # paired, the same tables at the two nodes
            excl = bool(iv and iv["excludes_zero"] and iv["low"] > 0)
            said = ("no rise after it (the last node)" if rise is None else
                    "over-fits: beyond the threshold, its interval above 0" if above and excl else
                    "beyond the threshold only" if above else "its interval above 0 only" if excl else
                    ("within the threshold" if gn else "no seeds' difference of the group"))
            by_g.append([g, BLOCK_OF(g), t0, t0 / tt if tt else None, y0, nxt[0] if nxt else None, diff(nxt[1], y0) if nxt else None,
                         pts[-1][0], rise, fmt_noise(gn) if gn else "none",
                         ("%.3f to %.3f (%d units)" % (iv["low"], iv["high"], iv["units"])) if iv else ("" if rise is None else iv_why.get(a, "none")),
                         said,
                         F(a, gc, g), gc])
        over.append(dict(arm=a, train_tables=tt, rows=rows_p, groups=by_g, final=dict(tables_seen=arms[a]["info"].get("tables_seen"),
                                                                           passes=(arms[a]["info"].get("tables_seen") or 0) / tt if tt else None,
                                                                           bits={c: F(a, c) for c in sweep_conds})))
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
    lbase = sorted({v for d in lines.values() for v in d})
    line_rows = [[c] + [lines[c].get(v) for v in lbase] for c in conds if c in lines]
    with open(os.path.join(folder, "curves.svg"), "w") as f:
        f.write(svg_curves({a: [(t, N(a, t, reading[a])) for t in nodes.get(a, []) if t > 0] for a in names}, reading, lines))
    fin = [["model" if v in arms else "baseline", v, c, g, x, bits[(v, p, t, c, g, x)]] for (v, p, t, c, g, x) in sorted(bits, key=str) if p == "final"]
    tsv(os.path.join(folder, "final.tsv"), ["kind", "variant", "condition", "group", "stratum", "bits"], fin)

    key = []                                                           # the key points, first
    for o in over:                                                     # a group of a curve over passes that over-fits
        risen = [r for r in o.get("groups") or [] if r[11] == "over-fits: beyond the threshold, its interval above 0"]
        if risen:
            key.append("Over-fitting by group (%s): %s." % (o["arm"], "; ".join(
                "%s under %s lowest at %s tables (%.2f passes), then +%.3f by %s tables" % (r[0], r[13], "{:,}".format(r[2]), r[3] or 0, r[8],
                                                                                         "{:,}".format(r[7]))
                for r in risen)))
    if scale:
        key.append("Scale, at the last node every arm with relatives reached (%s tables seen): %s." % ("{:,}".format(scale["node"]), scale["verdict"]))
    if lr_check:
        key.append("The learning rate: %s." % (lr_check["conclusion"] or "no conclusion: " + (lr_check["at_node"]["why"] or "")))
        if lr_check["sign"]:
            key.append("%s (%s below %s at %s tables seen under both conditions, by more than the seeds' difference)" % (
                lr_check["sign"], lr_check["arms"]["high"], lr_check["arms"]["small"], "{:,}".format(lr_check["at_node"]["node"])))
    md = ["# Relational line: the reference trunk's first curves", ""]
    if key:
        md += ["Key points:", ""] + ["- " + x for x in key] + [""]
    md += ["Bits per codon of the hidden residues of the examined row's target block, on held-out tables (lower is better). "
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
           "First at the end, on all such tables: the arm without relatives at its best node (or at its end, where its end is lower: both are "
           "evaluated on all tables) against the arm of the same size with relatives at its end (conservative while the curve with relatives "
           "still falls; the lower of the two candidates is picked on the same held-out tables it is then read on, a slight selection on the "
           "test set, one of two, whose effect is small); then at the same node, on the tables of the nodes. The column "
           "'same model' removes the rows from the arm with relatives at evaluation. Then by bin of the nearest identity, without the bin of no "
           "other rows.", ""]
    if s["worth_without"] == "end":
        md += ["End against end (the protocol of round 1): the arm without relatives is read at its end; its best node is listed beside, "
               "never read.", ""]
    md += md_table(["with", "without", "condition", "without read at", "bits without", "bits with (end)", "worth", "same model, rows removed",
                    "last common node", "worth at that node", "without at its best node (beside)"],
                   [[r["with_relatives"], r["without"], r["condition"], r["without_at"], r["bits_without"], r["bits_with_end"], r["worth"],
                     r["same_model_rows_removed"], r["node"], r["worth_at_node"], r["bits_without_best_node"]] for r in relatives])
    if spread:
        md += ["", "Against the seeds' difference (on the same tables with a relative on the target block; at the end, the difference at the last "
                   "node the seeds all reached):", ""]
        md += md_table(["with", "without", "condition", "worth (end)", "seeds' difference", "reading", "worth at the node", "seeds' difference there", "reading"],
                       [[r["with_relatives"], r["without"], r["condition"], r["worth"], fmt_noise(r["noise_end"]), r["reading_end"], r["worth_at_node"],
                         fmt_noise(r["noise_at_node"]), r["reading_at_node"]] for r in relatives])
    if any(r.get("interval_end") or r.get("interval_node") for r in relatives) or any(r.get("interval_end_why") for r in relatives):
        md += ["", "With intervals (the protocol's rule: a difference counts only beyond its threshold and with its 95% interval excluding 0; a "
                   "paired bootstrap over the held-out split units inside each group, 2,000 replicates, seed 20261123, on the cells both arms "
                   "scored, the tables with a relative on the target block; the blocks apart):", ""]
        rows_iv = []
        for r in relatives:
            ie, inn = r.get("interval_end") or {}, r.get("interval_node") or {}
            rows_iv.append([r["with_relatives"], r["without"], r["condition"], r["worth"], fmt_iv(ie.get("all")) or r.get("interval_end_why") or "",
                            "; ".join("%s %s" % (k, fmt_iv(v)) for k, v in ie.items() if k != "all"), r.get("counts_end") or "",
                            r["worth_at_node"], fmt_iv(inn.get("all")) or r.get("interval_node_why") or "", r.get("counts_node") or ""])
        md += md_table(["with", "without", "condition", "worth (end)", "95% interval (all)", "by block", "reading", "worth at the node",
                        "95% interval there", "reading there"], rows_iv)
    for r in relatives:
        md += ["", "%s against %s, %s, by bin (end):" % (r["without"], r["with_relatives"], r["condition"]), ""]
        md += md_table(["nearest identity", "bits without", "bits with", "worth"], r["by_bin"])
    # the online rendering of the nodes against the strict offline one: the same final weights, the same tables (w6_eval
    # online_too), the difference and its paired interval, by the three relatives strata (the programme lead's ruling)
    online_rows = []
    for a, o in sorted(online.items()):
        if a not in arms:
            continue
        sm = read_json(os.path.join(o["folder"], "summary.json")) or {}
        h = held_of(a)
        for c in sm.get("conditions") or []:
            off_c, on_c = table_cells(a, "final", c), None
            p_on = os.path.join(o["folder"], "heldout_final.npz")
            if os.path.exists(p_on):
                z = np.load(p_on)
                cc = c.replace("+", "__")
                if "bits_" + cc in z:
                    t = np.asarray(z["table_" + cc]).astype(np.int64)
                    n = int(t.max()) + 1 if len(t) else 0
                    on_c = (np.bincount(t, weights=np.asarray(z["bits_" + cc], dtype=np.float64), minlength=n), np.bincount(t, minlength=n).astype(np.float64))
            for x in [None] + list(getattr(SC, "RELATIVE_KINDS", ())):
                why, iv = None, None
                try:
                    from seeddrift.train.rel.evalh import intervals as IV
                except ImportError:
                    IV, why = None, "no intervals module of the evaluator"
                if off_c is None or on_c is None:
                    why = why or "no cells per table"
                elif h is None:
                    why = why or "no split units (units_source)"
                elif IV is not None:
                    group, unit, strata = h
                    n = min(len(off_c[0]), len(on_c[0]), len(group))
                    keep = (off_c[1][:n] > 0) & (on_c[1][:n] > 0)
                    if x is not None:
                        keep &= np.asarray(strata.get(x, np.zeros(len(group), bool)))[:n]
                    if not keep.any():
                        why = "no tables"
                    elif not np.array_equal(off_c[1][:n][keep], on_c[1][:n][keep]):
                        why = "not the same cells"
                    else:
                        idx = np.flatnonzero(keep)
                        iv = IV.paired_bootstrap((on_c[0][:n] - off_c[0][:n])[idx], off_c[1][:n][idx], [group[k] for k in idx], [unit[k] for k in idx])
                xs = x or SC.ALL
                online_rows.append([a, c, xs, F(a, c, SC.ALL, xs), (sm.get("final") or {}).get(c) if x is None else None,
                                    iv["all"]["estimate"] if iv else None, fmt_iv(iv["all"]) if iv else why or "",
                                    "; ".join("%s %s" % (k, fmt_iv(v)) for k, v in (iv or {}).items() if k != "all")])
    if online_rows:
        md += ["", "## The online rendering of the nodes against the strict offline rendering", "",
               "The same final weights on the same held-out tables, rendered online (the leak-free renderer of the training and of the "
               "nodes: the anchors of scattered and span are the stored pairs, aligned while the hidden residues were visible) and "
               "strictly offline (package W3; the rendering every scored reading uses): online minus offline (negative: the online "
               "rendering reads too low), with its paired interval, over all tables and by the relatives strata.", ""]
        md += md_table(["arm", "condition", "tables", "bits offline", "bits online (all)", "online minus offline", "95% interval", "by block"],
                       online_rows)
    # the controls of the renderer: every arm with relatives under shuffle and presence against its plain reading, at the
    # end on the tables with a relative on the target block, paired (protocol round1, readings.controls: a model that uses
    # the correspondence of the rows loses bits under the shuffle)
    control_rows = []
    for a in rel:
        for c in (cond.SCATTERED, cond.WHOLE_GENE):
            plain = F(a, c, SC.ALL, x0)
            for ctl in ("shuffle", "presence"):
                k = "%s@%s" % (c, ctl)
                got = F(a, k, SC.ALL, x0)
                if got is None:
                    continue
                iv, why = paired(a, "final", c, a, "final", k, x0)
                n_end = noise(None, c, x0)
                d = diff(got, plain)
                said = None
                if d is not None and n_end is not None:
                    above, excl = abs(d) > n_end["bits"], bool(iv and iv["all"]["excludes_zero"])
                    said = ("counts: beyond the threshold, its interval excludes 0" if above and excl else
                            "beyond the threshold only" if above else "its interval only" if excl else "cannot be read")
                control_rows.append([a, c, ctl, plain, got, d, fmt_iv((iv or {}).get("all")) or why or "", fmt_noise(n_end), said or ""])
    if control_rows:
        md += ["", "## The controls of the renderer", "",
               "Every arm with relatives at its end under W3's two render-mode controls against its plain reading, on the tables with a "
               "relative on the target block: shuffle (the other rows' correspondence permuted by column) and presence (only which cells "
               "exist). The control minus the plain reading (positive: the control loses bits), paired on the same cells.", ""]
        md += md_table(["arm", "condition", "control", "bits plain", "bits under the control", "control minus plain", "95% interval", "threshold",
                        "reading"], control_rows)
    # a node's moving average read again on all held-out tables (w6_eval weights node@<n>: the entry <arm>.node@<n>)
    # against the arm's end on the same cells: the end minus the node (positive: worse at the end), over all tables and by
    # group, paired; beside it the arm's own node at its last node on the nodes' tables (the programme lead's ruling for
    # round 1's long arm: the eukaryotic viruses' whole gene at its lowest node, strictly offline)
    again_rows = []
    for p_arm in sorted(arms):
        m = re.match(r"^(.*)\.node@(\d+)$", p_arm)
        if not m or m.group(1) not in arms:
            continue
        base, t_n = m.group(1), int(m.group(2))
        last = max([u for u in nodes.get(base, []) if u > 0] or [0]) or None
        for c in (cond.SCATTERED, cond.WHOLE_GENE, getattr(cond, "SUFFIX", "suffix")):
            for g in [SC.ALL] + group_set:
                at_node = N(p_arm, t_n, c, g)
                end = F(base, c, g)
                if at_node is None or end is None:
                    continue
                iv, why = paired(p_arm, "final", c, base, "final", c, SC.ALL, only_group=None if g == SC.ALL else g)
                n_end = group_spread.get((g, c)) if g != SC.ALL else noise(None, c)
                if n_end and g != SC.ALL:
                    n_end = dict(n_end, bits=max(n_end["bits"], GROUP_FLOOR))
                d = diff(end, at_node)
                above, excl = d is not None and n_end is not None and abs(d) > n_end["bits"], bool(iv and iv["all"]["excludes_zero"])
                said = ("counts: beyond the threshold, its interval excludes 0" if above and excl else "beyond the threshold only" if above else
                        "its interval only" if excl else "cannot be read") if d is not None and n_end is not None else ""
                again_rows.append([base, "{:,}".format(t_n), c, g, at_node, N(base, last, c, g) if last else None, "{:,}".format(last) if last else "",
                                   end, d, fmt_iv((iv or {}).get("all")) or why or "", fmt_noise(n_end) if n_end else "none", said])
    if again_rows:
        md += ["", "## Node weights read again on all held-out tables", "",
               "The moving average an arm kept at a node, read again on all held-out tables strictly offline (stage w6_eval, weights "
               "node@<tables>), against the arm's end on the same cells: the end minus the node (positive: the end is worse, the arm "
               "over-fitted after the node), over all tables and by group, with its paired interval over the split units; beside it the "
               "arm's own reading at its last node on the nodes' tables (online for scattered and span).", ""]
        md += md_table(["arm", "node read again", "condition", "group", "bits at that node (all tables)", "bits at its last node (nodes' tables)",
                        "last node", "bits at the end", "end minus the node", "95% interval", "threshold", "reading"], again_rows)
    # what each arm optimised: the realised share of every mask scheme in the target cells and in the loss (round 1)
    shares_rows, tiers = [], {}
    for a in names:
        sm = training_summary(a)                                       # the training's own summary (an evaluation again writes another)
        for m, x in (sm.get("schemes") or {}).items():
            shares_rows.append([a, m, x.get("tables"), x.get("target_cells"), x.get("cell_share"), x.get("loss_share"), x.get("bits_per_cell")])
        if sm.get("frame_tiers"):
            tiers[a] = sm["frame_tiers"]
    if shares_rows:
        md += ["", "## The mask schemes each arm trained on", "",
               "Their realised shares of the target cells and of the training loss (the arms with and without relatives optimise different "
               "mixtures), from every step of the training.", ""]
        md += md_table(["arm", "scheme", "tables", "target cells", "share of the cells", "share of the loss", "bits per cell"], shares_rows)
    if tiers:
        labels = sorted({k for t in tiers.values() for w in t.values() for k in w}, key=lambda k: (k != "self_copy", k == "none", k))
        md += ["", "## The whole gene's frame relative, by distance", "",
               "The row the whole gene was laid on: in training, the frame relative of the rows drawn; in the evaluation, that of every "
               "held-out table laid whole (self_copy: a row of the examined genome's own copies; tier_<t>: the row's distance tier; none: no "
               "other row has a stretch in the target block).", ""]
        md += md_table(["arm", "tables of"] + labels, [[a, w] + [tiers[a][w].get(k, 0) for k in labels] for a in tiers for w in ("train", "heldout")
                                                       if w in tiers[a]])
    md += ["", "## Scale", ""]
    read_by = lambda r: "; ".join("%s: %s" % (k, " + ".join(v) if len(v) == len(levels[k]) and len(v) > 1 else "only " + ", ".join(v))   # noqa: E731
                                  for k, v in r["seeds_read"].items()) or ""
    if scale:
        md += ["At the last node every arm with relatives reached (%s tables seen): %s." % ("{:,}".format(scale["node"]), scale["verdict"]), ""]
        md += md_table(["arm", "parameters", "scattered", "whole_gene"], [[a, arms[levels[a][0]]["info"].get("params"), scale["bits"][a][cond.SCATTERED],
                                                                            scale["bits"][a][cond.WHOLE_GENE]] for a in scale["arms"]])
        md += ["", "Each step in size: %s." % "; ".join("%s to %s %s" % (scale["arms"][j], scale["arms"][j + 1], x) for j, x in enumerate(scale["steps"]))]
        if scale["noise"]:
            md += ["The seeds' difference there: %s." % "; ".join("%s %s" % (c, fmt_noise(n)) for c, n in scale["noise"].items())]
        if scale["seeds_read"]:
            md += ["Seeds read at that node (their mean where more than one): %s." % read_by(scale)]
        md += ["", "The rule: bigger is better only when the bits fall with every step in size under both conditions, by more than the seeds' "
                   "difference where it is known; otherwise the scale cannot be read."]
    else:
        md += ["Fewer than two arms with relatives, or no node all of them reached."]
    if scale_pairs:
        md += ["", "Every two arms with relatives at the last node both reached (the same rule):", ""]
        md += md_table(["smaller", "bigger", "tables seen", "scattered, smaller", "scattered, bigger", "whole_gene, smaller", "whole_gene, bigger",
                        "seeds' difference (scattered; whole_gene)", "seeds read", "verdict"],
                       [[p["arms"][0], p["arms"][1], p["node"]] + [p["bits"][a][c] for c in both for a in p["arms"]]
                        + ["; ".join(fmt_noise(p["noise"][c]) for c in both) if p["noise"] else "none", read_by(p), p["verdict"]] for p in scale_pairs])
    if lr_check:
        k = lr_check["arms"]
        trio = [k["small"], k["low"], k["high"]]
        md += ["", "## The learning rate: %s at peak rate %s against itself at %s and %s at %s" % (
            k["high"], lr_check["lr"][k["high"]], lr_check["lr"][k["low"]], k["small"], lr_check["lr"][k["small"]]), ""]
        if lr_check["note"]:
            md += [lr_check["note"], ""]
        a = lr_check["at_node"]
        md += ["At the last node all three reached (%s tables seen): %s.%s" % ("{:,}".format(a["node"]) if isinstance(a["node"], int) else "none",
                                                                              lr_check["conclusion"] or ("no conclusion: " + (a["why"] or "")),
                                                                              " " + lr_check["sign"] if lr_check["sign"] else ""),
               "At the end (all held-out tables; the arms saw %s tables): %s." % (
                   ", ".join("{:,}".format(lr_check["tables_seen"][x] or 0) for x in trio),
                   LR_CONCLUSIONS.get(lr_check["at_end"]["conclusion"]) or "no conclusion: " + (lr_check["at_end"]["why"] or "")), "",
               "The rule, under each of scattered and whole_gene (both must agree): caught up when the higher rate is above the small arm by no "
               "more than the seeds' difference; else part of the cause when the higher rate is below the lower rate by more than it; else not "
               "the cause. Without a seeds' difference the threshold is zero.", ""]
        md += md_table(["tables seen"] + ["%s, %s" % (c, x) for c in both for x in trio], lr_check["curves"] + lr_check["final"])
        md += ["", "By condition at that node:", ""]
        md += md_table(["condition", "higher rate minus small arm", "lower rate minus higher rate", "seeds' difference", "reading"],
                       [[c, x.get("high_minus_small"), x.get("low_minus_high"), fmt_noise(x.get("noise")), x.get("kind")] for c, x in a["by_condition"].items()])
    if leak:
        md += ["", "## The leak of the stored layout", "",
               "The same arm trained and read leak-free against it trained and read in the stored layout, at the end, on all held-out tables "
               "and on those with a relative on the target block: the leak is the bits leak-free minus the bits stored (positive: the stored "
               "layout read too low). Scattered for the scattered mask; suffix (the second half of the target hidden) for the span. Suffix is "
               "read end against end only (the stored arms were evaluated on it at their end): the two arms saw different numbers of tables "
               "(%s), so the suffix pair is a side reading; the main reading of the leak is scattered at the common nodes below." % "; ".join(
                   "%s %s against %s %s" % (lf, "{:,}".format(arms[lf]["info"].get("tables_seen") or 0), st_arm,
                                            "{:,}".format(arms[st_arm]["info"].get("tables_seen") or 0)) for lf, st_arm in s["leak_pairs"]), ""]
        md += md_table(["leak-free arm", "stored arm", "condition", "tables", "bits leak-free", "bits stored", "leak", "seeds' difference", "reading"],
                       [[r["leakfree"], r["stored"], r["condition"], r["tables"], r["bits_leakfree"], r["bits_stored"], r["leak"], fmt_noise(r["noise"]),
                         r["reading"] or ""] for r in leak])
        if leak_nodes:
            md += ["", "Scattered at every node both reached (equal tables seen; 600 tables):", ""]
            md += md_table(["leak-free arm", "stored arm", "tables seen", "tables", "bits leak-free", "bits stored", "leak", "seeds' difference", "reading"],
                           [[r["leakfree"], r["stored"], r["node"], r["tables"], r["bits_leakfree"], r["bits_stored"], r["leak"], fmt_noise(r["noise"]),
                             r["reading"] or ""] for r in leak_nodes])
    for q in pairs_out:
        md += ["", "## %s: %s against %s" % (q["name"], q["second"], q["first"]), "",
               "The second minus the first (negative: the second is lower), at every node both reached (600 tables) and at the end (all tables; "
               "tables seen %s), against the seeds' difference." % ", ".join("%s %s" % (a, t) for a, t in q["tables_seen"].items()), ""]
        md += md_table(["tables seen"] + ["%s, %s" % (c, w) for c in (cond.SCATTERED, cond.WHOLE_GENE, getattr(cond, "SUFFIX", "suffix"))
                                          for w in ("first", "second", "difference", "reading")], q["rows"])
    for w in sweeps:
        md += ["", "## Learning rates of %s: %s" % (w["name"], ", ".join("%s at %s" % (e, w["lr"][e]) for e in w["entries"])), "",
               "Bits at every node all of them reached (600 tables; a group of seeds: its mean), then at the end (all tables; each at its own "
               "tables seen: %s). At the last common node (%s tables), the best against the next: a lead no larger than the seeds' difference "
               "cannot be read." % ("; ".join("%s %s" % (e, w["tables_seen"][e]) for e in w["entries"]), "{:,}".format(w["node"]) if w["node"] else "none"), ""]
        md += md_table(["tables seen"] + ["%s, %s" % (c, e) for c in w["conditions"] for e in w["entries"]], w["rows"] + [["end"] + w["final"][0]])
        md += ["", "Best at that node:", ""]
        md += md_table(["condition", "best", "bits", "next", "lead", "seeds' difference", "reading"],
                       [[c, b["entry"], b["bits"], b["next"], b["margin"], fmt_noise(b["noise"]), b["reading"] or ""] for c, b in w["best"].items()])
    for o in over:
        md += ["", "## %s over passes of the training tables (%s tables)" % (o["arm"], "{:,}".format(o["train_tables"]) if o["train_tables"] else "?"), "",
               "Held-out bits at every node (600 tables), the fall of scattered from the node before (against the seeds' difference), and the "
               "training loss (bits per hidden cell, all mask schemes, mean of the 1,000 tables before the node). The training loss falling "
               "while the held-out bits rise would be overfitting.", ""]
        md += md_table(["tables seen", "passes", "scattered", "scattered+no_relatives", "whole_gene+no_relatives", "fall of scattered", "reading",
                        "training loss"], o["rows"])
        f = o["final"]
        md += ["", "End: %s tables (%s passes): %s." % ("{:,}".format(f["tables_seen"] or 0), "%.2f" % f["passes"] if f["passes"] else "?",
                                                      ", ".join("%s %.3f" % (c, v) for c, v in f["bits"].items() if v is not None))]
        if o.get("groups"):
            md += ["", "By group (scattered and whole_gene; each group on its own first held-out tables at the nodes, so a group's reading is noisier than "
                   "the mean's; its threshold is the seeds' largest difference in that group at the last node they all reached, at least "
                   "0.02 bits): the node of "
                   "the group's lowest held-out bits, the change to the next node and to the last node. A rise beyond the group's seeds' "
                   "difference whose paired interval (split units inside the group, 2,000 replicates, seed 20261123, 95%) lies above 0 is "
                   "the group's over-fitting (a key point); one of the two alone is listed only. The end is read on all held-out tables "
                   "of the group (other tables than the nodes').", ""]
            md += md_table(["condition", "group", "block", "lowest at (tables)", "passes there", "bits there", "next node", "change to it",
                            "last node", "change to the last", "threshold (group's seeds' difference, at least 0.02)", "95% interval of that change",
                            "reading", "end (all tables)"], [[r[13]] + r[:13] for r in o["groups"]])
    md += ["", "## The cost of re-pairing, on the tables that can be re-paired (two or more other rows), paired", ""]
    md += md_table(["arm or baseline", "whole_gene", "whole_gene+repaired", "cost"], repair)
    md += ["", "No difference would not show that the relation blocks go unused."]
    for g in seeds_out:
        if not g["rows"]:
            md += ["", "## Seeds: %s" % ", ".join(g["arms"]), "", g.get("note") or "no node all of them reached"]
            continue
        title = "Two seeds" if len(set(g["seeds"])) == len(g["seeds"]) == 2 else ("Seeds" if len(set(g["seeds"])) == len(g["seeds"]) else "Runs of one seed")
        md += ["", "## %s of one arm: %s" % (title, ", ".join("%s (seed %s, run %s)" % (a, sd, r) for a, sd, r in zip(g["arms"], g["seeds"], g["runs"]))), "",
               "Their difference at every node all of them reached, on the nodes' tables (%s; more than two: the largest minus the smallest). "
               "It is the only noise estimate of the day, and it holds two differences at once: that of the seeds and that of the machines "
               "(the runs trained on different GPUs, which other processes shared). A difference of scale or of what relatives are worth that "
               "is not larger than it cannot be read; at a node the seeds did not all reach, and at the end, the difference at the last node "
               "they all reached stands in. That difference is one reading of one pair of seeds at one node and is itself very imprecise; later "
               "in training the seeds may come closer or drift apart, so judging later nodes by the difference at %s tables is only an "
               "approximation." % ("the second minus the first" if len(g["arms"]) == 2 else "spread", "{:,}".format(g["rows"][-1][0])), ""]
        md += md_table(["tables seen"] + ["%s, %s" % (c, a) for c in both for a in g["arms"] + ["difference"]], g["rows"])
    if s["exclude"]:
        md += ["", "Left out of the readings of relatives and scale (shown in the curves only): %s." % ", ".join(s["exclude"])]
    md += ["", "## A check: tables without other rows", "",
           "There an arm reads the same table with and without relatives, so the two columns must agree (up to rounding).", ""]
    md += md_table(["arm", "condition", "with relatives", "without", "difference"], check)
    md += ["", "## By group (final evaluation; condition %s, an arm without relatives under %s)" % (s["main"], s["main"] + NO_REL), ""]
    md += md_table(["group", "block"] + cols, by_group)
    md += ["", "## Curves at equal tables seen (the first held-out tables of every group at every node)", "", "![curves](curves.svg)", ""]
    md += md_table(head, grid) if grid else ["no curve"]
    if line_rows:
        md += ["", "The classical baselines on the same tables (the horizontal lines of the figure; mean of the groups):", ""]
        md += md_table(["condition"] + lbase, line_rows)
        md += ["", "The nodes and the baselines scored the same cells: %s." % (", ".join("%s %s" % (c, "yes" if v else "NO") for c, v in same_subset.items())
                                                                          or "not known (no digests of the subset)")]
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
                  relatives=relatives, scale=scale, scale_pairs=scale_pairs, repair=dict(columns=["variant", "whole_gene", "whole_gene+repaired", "cost"], rows=repair),
                  check=dict(columns=["arm", "condition", "with", "without", "difference"], rows=check),
                  aligned=dict(columns=head, rows=grid), by_group=dict(columns=["group", "block"] + cols, rows=by_group),
                  lines=dict(columns=["condition"] + lbase, rows=line_rows), same_subset=same_subset, seeds=seeds_out, lr_check=lr_check, leak=leak, leak_nodes=leak_nodes, pairs=pairs_out, lr_sweeps=sweeps,
                  passes=over,
                  noise=[dict(node=t, condition=c, stratum=x, bits=v) for (t, c, x), v in sorted(spread.items())], readings_exclude=s["exclude"])
    for name in ("curves.tsv", "aligned.tsv", "final.tsv", "curves.svg", "report.md"):
        st.output(name.replace(".", "_"), os.path.join(folder, name))
    p = os.path.join(folder, "report.json")
    write_json(p, report)                                                 # the rule's target last
    st.output("report", p)
    st.extra.update(arms=len(names), baselines=bnames, conditions=conds, same_items=same, scale=(scale or {}).get("verdict"))
    return report


PALETTE = ("#2f6db5", "#c4512b", "#2e8b57", "#8a4fb0", "#b8860b", "#4a4a4a")
LINE_BASELINES = ("own_freq", "row_average", "copy_nearest_binned", "switch_copy")


def svg_curves(series, reading, lines, width=760, height=440):
    """A figure of the curves: bits against tables seen (log scale) for every arm, and the baselines on the same tables
    as horizontal lines (dashed) under the condition of the arms with relatives (and of the arm without, when it differs)."""
    import math
    pts = [(t, b) for xs in series.values() for t, b in xs if b is not None and t > 0]
    hlines = [(c, v, x) for c in sorted({reading[a] for a in series}) for v, x in (lines.get(c) or {}).items() if v in LINE_BASELINES]
    if not pts:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="60"><text x="10" y="30">no curve</text></svg>\n' % width
    left, right, top, bottom = 64, 230, 20, 46
    lo_t, hi_t = min(t for t, _ in pts), max(t for t, _ in pts)
    ys = [b for _, b in pts] + [x for _, _, x in hlines]
    lo_b, hi_b = min(ys) - 0.05, max(ys) + 0.05
    span_t = max(math.log2(hi_t) - math.log2(lo_t), 1.0)
    X = lambda t: left + (math.log2(t) - math.log2(lo_t)) / span_t * (width - left - right)      # noqa: E731
    Y = lambda b: top + (hi_b - b) / (hi_b - lo_b) * (height - top - bottom)                       # noqa: E731
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="sans-serif" font-size="11">' % (width, height),
           '<rect width="100%" height="100%" fill="white"/>',
           '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#888"/>' % (left, height - bottom, width - right, height - bottom),
           '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#888"/>' % (left, top, left, height - bottom)]
    t = 1000
    while t <= hi_t * 1.0001:
        if t >= lo_t / 1.0001:
            out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#eee"/><text x="%.1f" y="%d" text-anchor="middle">%s</text>'
                       % (X(t), top, X(t), height - bottom, X(t), height - bottom + 14, "{:,}".format(t)))
        t *= 2
    step = 0.1 if hi_b - lo_b < 1.5 else 0.25
    b = math.ceil(lo_b / step) * step
    while b <= hi_b:
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#eee"/><text x="%d" y="%.1f" text-anchor="end">%.2f</text>'
                   % (left, Y(b), width - right, Y(b), left - 4, Y(b) + 4, b))
        b += step
    out.append('<text x="%d" y="%d" text-anchor="middle">tables seen</text>' % ((left + width - right) // 2, height - 8))
    out.append('<text x="14" y="%d" transform="rotate(-90 14 %d)" text-anchor="middle">bits per codon (lower is better)</text>'
               % ((top + height - bottom) // 2, (top + height - bottom) // 2))
    for k, (c, v, x) in enumerate(hlines):
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#999" stroke-dasharray="5,4"/>' % (left, Y(x), width - right, Y(x)))
        out.append('<text x="%d" y="%.1f" fill="#666">%s, %s %.3f</text>' % (width - right + 6, Y(x) + 4, v, c, x))
    for k, a in enumerate(sorted(series)):
        xs = [(t, b) for t, b in series[a] if b is not None and t > 0]
        if not xs:
            continue
        colour = PALETTE[k % len(PALETTE)]
        out.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (colour, " ".join("%.1f,%.1f" % (X(t), Y(b)) for t, b in xs)))
        out += ['<circle cx="%.1f" cy="%.1f" r="2.5" fill="%s"/>' % (X(t), Y(b), colour) for t, b in xs]
        out.append('<text x="%.1f" y="%.1f" fill="%s">%s</text>' % (min(X(xs[-1][0]) + 4, width - right - 4), Y(xs[-1][1]) - 4, colour, a))
    out.append("</svg>")
    return "\n".join(out) + "\n"


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
    if not all((rep.get("same_subset") or {}).values()):
        problems.append(dict(error="the nodes and the baselines' subset were not scored on the same cells", same_subset=rep.get("same_subset")))
    require(GateResult("G4_rel_w6_report", "fail" if problems else "pass",
                       dict(problems=problems[:20], arms={a: x["final"] for a, x in rep["arms"].items()}, baselines=sorted(rep["baselines"]),
                            same_items=rep.get("same_items"), scale=(rep.get("scale") or {}).get("verdict"))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        rep = json.load(f)
    with open(os.path.join(run_dir, OUT, "report.md")) as f:
        md = f.read().splitlines()
    small = dict(main=rep["main"], same_items=rep.get("same_items"), scale=rep.get("scale"), scale_pairs=rep.get("scale_pairs"),
                 lr_check={k: (rep.get("lr_check") or {}).get(k) for k in ("conclusion", "sign")}, lead=rep.get("lead"),
                 headline=rep["headline"],
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
