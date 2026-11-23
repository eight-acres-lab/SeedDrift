"""Stage rel_w6_eval (package W6; pipeline rel_traineval): the saved weights of a training run's arms evaluated again on
the held-out tables, for arms whose final evaluation the wall limit cut short. `python -m ci.stages.rel.w6_eval
RUN_DIR`. Nothing is trained; the weights are read, never changed.

Profile block rel_w6_eval:
  from       the directory of the training run (stage rel_w6_fit) on this machine; it need not have passed (an arm
             stopped at its wall limit makes the run fail) but its stage rel_w6_fit must have written its model index
             (several runs as a list: their arms by name, each name once)
  arms       the arms to evaluate again (default: every arm of those runs with a final.pt); an entry {name, untrained:
             true} reads the arm's model as built with its seed and never trained (a readout's control; untrained.pt);
             {name, weights} reads the weights named: final (the default), best (best.pt) or node@<tables> (the moving
             average the training kept at that node, ema_tables<tables>.pt, keep_node_weights), into the folder
             <arm> for final and <arm>.<weights> otherwise (rows ema@best, ema@tables=<n>; a readout's info.weights is
             the name given); an arm may be listed several times with different weights
  readout    {module, presence_arms, max_rows, layout} (with a ruler set only, and a ruler set only with it): instead
             of rows, the passes of the module (package W18: seeddrift.train.rel.readouts.w18_passes, run_ruler) scored
             by every arm, written to data/rel_w6/eval/<arm>/ruler_cells.npz (<arm>.untrained/ for an untrained arm);
             presence_arms (arm or variant names) also run the presence passes; layout stored or leakfree;
             in_place_only (false): the untrained arms, and the presence passes, on the in-place implants only;
             adapt_v0_arms (false): every arm (the untrained too) read through the module's adaptation of tables of
             version 1 to arms trained on version 0 (adapt_v0=True)
  tables     heldout_offline (optional): {run, path, sha256}: package W3's strict offline rendering of that held-out
             set (layout leakfree, its parts), read by the arms trained leak-free in place of the training's own; never
             with a ruler set (a readout renders its passes itself)
             heldout: {run, path, sha256, split}: a real held-out set (split 1) or a ruler set of package W18 (the
             split its builder sets; this stage reads the kinds configs/rel/kinds.yaml gives it, read_tableset), as in
             rel_w6_fit otherwise: the held-out table set, a hand-off file of a
             passed run, every file it lists with its recorded hash; the arm's own training configuration names the
             training tables, which are opened (not read) only to build the arm
  best       true (default): an arm trained without relatives is read at its best node too (best.pt, rows ema@best);
             all: every arm (round 1: the final and best weights of every arm on all held-out tables)
  end_controls  [shuffle, presence]: the controls of package W3's renderer under scattered and whole_gene of the arms
             trained leak-free (keys <condition>@<control>), in place of the training's own
  optional   final_per_group (0: every held-out table), eval_max_rows, threads, wall_s (of one arm's evaluation,
             default 7200), verify_files (true); extra_conditions (evaluator conditions besides today's five, e.g.
             [suffix]) or only_conditions (these alone: an arm evaluated again on one more condition, its earlier rows
             kept by the report)
For every arm, in the model interpreter: the moving average of final.pt (and of best.pt) under the arm's conditions,
rows "ema@final" ("ema@best") with the tables, cells and wall time the training had reached, as the training's own
final evaluation writes them. The checkpoints are hard links into this run (data/rel_w6/eval/<arm>/), their hashes
recorded; an arm that fails is recorded and does not stop the others.
Outputs under data/rel_w6/eval/: <arm>/ (conf.json, eval.log, results.tsv, summary.json, heldout_final.npz, the
linked checkpoints), results.tsv (every arm), models.json (the model index of the hand-off, the format of rel_w6_fit:
each arm's training information with its final evaluation replaced by this one), written last.
Gate G4_rel_w6_eval: the index is well-formed, at least one arm was evaluated, and every evaluated arm has its final
rows under each of its conditions.
"""
from __future__ import annotations

import json
import os
import re
import subprocess

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common, w6_fit, w6_tools
from seeddrift.train.rel import handoff
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import score as SC
from seeddrift.train.rel.evalh import shards

OUT = os.path.join("data", "rel_w6", "eval")
FIT = w6_fit.OUT
KEYS = {"from", "arms", "tables", "best", "final_per_group", "eval_max_rows", "threads", "wall_s", "verify_files", "readout", "extra_conditions",
        "only_conditions", "end_controls"}
CONTROLS = ("shuffle", "presence")
NODE_WEIGHTS = w6_fit.NODE_WEIGHTS
READOUT_KEYS = {"module", "presence_arms", "max_rows", "layout", "in_place_only", "adapt_v0_arms"}
UNTRAINED = "untrained"
RULER_CELLS = "ruler_cells.npz"
WEIGHTS_RX = re.compile(r"^(final|best|node@[1-9][0-9]*)$")


def weights_label(weights):
    """The result rows' weights column (after "ema@") of the weights an entry reads: final, best, tables=<n>."""
    return "tables=%s" % weights.split("@")[1] if weights.startswith("node@") else weights


def settings(cfg) -> dict:
    sc = cfg.get("rel_w6_eval") or {}
    unknown = set(sc) - KEYS
    if unknown or not sc.get("from") or not (sc.get("tables") or {}).get("heldout"):
        raise SystemExit("rel_w6_eval: from and tables.heldout are needed; unknown keys %s" % sorted(unknown))
    h = sc["tables"]["heldout"]
    if set(h) - {"run", "path", "sha256", "split"} or not h.get("run") or not h.get("path") or not isinstance(h.get("split", 1), int):
        raise SystemExit("rel_w6_eval.tables.heldout: run, path, optionally sha256 and split (1 for a real set): %r" % (h,))
    if set(sc["tables"]) - {"heldout", "heldout_offline"}:
        raise SystemExit("rel_w6_eval.tables: heldout and optionally heldout_offline: %r" % sorted(sc["tables"]))
    off = sc["tables"].get("heldout_offline")
    if off is not None and (set(off) - {"run", "path", "sha256"} or not off.get("run") or not off.get("path")):
        raise SystemExit("rel_w6_eval.tables.heldout_offline: run, path and optionally sha256: %r" % (off,))
    arms = []
    for a in sc.get("arms") or []:
        a = dict(name=a) if isinstance(a, str) else dict(a)
        w = str(a.get("weights", "final"))
        if set(a) - {"name", "untrained", "weights"} or not a.get("name") or not WEIGHTS_RX.match(w) or (a.get("untrained") and "weights" in a):
            raise SystemExit("rel_w6_eval.arms: an arm's name, {name, untrained: true} or {name, weights: final | best | node@<tables>}: %r" % (a,))
        arms.append(dict(name=str(a["name"]), untrained=bool(a.get("untrained", False)), weights=w))
    if len({(a["name"], a["untrained"], a["weights"]) for a in arms}) != len(arms):
        raise SystemExit("rel_w6_eval.arms: an arm is listed twice with the same weights")
    best = sc.get("best", True)
    if best not in (True, False, "all"):
        raise SystemExit("rel_w6_eval.best: true, false or all, not %r" % (best,))
    ctl = sc.get("end_controls")
    if ctl is not None and (not isinstance(ctl, list) or set(ctl) - set(CONTROLS)):
        raise SystemExit("rel_w6_eval.end_controls: a list among %s, not %r" % (CONTROLS, ctl))
    ro = sc.get("readout")
    if ro is not None and (set(ro) - READOUT_KEYS or not ro.get("module")):
        raise SystemExit("rel_w6_eval.readout: {module, presence_arms, max_rows, layout, in_place_only, adapt_v0_arms}: %r" % (ro,))
    if any(a["untrained"] for a in arms) and ro is None:
        raise SystemExit("rel_w6_eval: an untrained arm is read by a readout only (rel_w6_eval.readout)")
    sources = sc["from"] if isinstance(sc["from"], list) else [sc["from"]]
    return dict(sources=[str(x) for x in sources], source=str(sources[0]), arms=arms,
                heldout=dict(run=str(h["run"]), path=str(h["path"]), sha256=h.get("sha256"), split=int(h.get("split", 1))),
                heldout_offline=None if off is None else dict(run=str(off["run"]), path=str(off["path"]), sha256=off.get("sha256")),
                best=best, end_controls=None if ctl is None else [str(c) for c in ctl], final_per_group=int(sc.get("final_per_group", 0)), eval_max_rows=int(sc.get("eval_max_rows", 32)),
                threads=int(sc.get("threads", 4)), wall_s=int(sc.get("wall_s", 7200)), verify_files=bool(sc.get("verify_files", True)),
                extra_conditions=[str(c) for c in sc.get("extra_conditions") or []], only_conditions=[str(c) for c in sc.get("only_conditions") or []] or None,
                readout=None if ro is None else dict(module=str(ro["module"]), presence_arms=[str(x) for x in ro.get("presence_arms") or []],
                                                     max_rows=int(ro.get("max_rows", 32)), layout=str(ro.get("layout") or "stored"),
                                                     in_place_only=bool(ro.get("in_place_only", False)), adapt_v0_arms=bool(ro.get("adapt_v0_arms", False))))


def source_index(source):
    """The model index the training run's stage rel_w6_fit wrote, checked against that run's stage record (the run itself
    may have failed: an arm stopped at its wall limit fails it)."""
    p = os.path.join(source, FIT, "models.json")
    rec = w6_fit.read_json(os.path.join(source, "record.json")) or {}
    listed = [o for s in rec.get("stages") or [] for o in s.get("outputs") or [] if o.get("path") == os.path.join(FIT, "models.json")]
    if not os.path.exists(p) or not listed or any(o.get("sha256") != sha256_file(p, use_sidecar=False) for o in listed):
        raise SystemExit("rel_w6_eval: %s holds no model index its stage record lists" % source)
    with open(p) as f:
        doc = json.load(f)
    if doc.get("sha256") != handoff.digest(doc):
        raise SystemExit("rel_w6_eval: the model index of %s does not match its own sha256" % source)
    return doc, rec


def heldout_set(spec, verify) -> dict:
    """The table set the weights are read on: a hand-off file of a passed run of a kind this stage reads (configs/rel/
    kinds.yaml: real, or a ruler set of package W18), every file it lists with its recorded hash. A real set is read as
    split 1; a ruler set as the split its profile names (its builder sets it)."""
    doc = common.read_tableset("w6_eval", spec["run"], spec["path"], spec["sha256"])
    if doc.get("kind") == "real" and spec["split"] != 1:
        raise SystemExit("rel_w6_eval: a real held-out table set is read as split 1, not %d" % spec["split"])
    bad = handoff.tableset_problems(doc, spec["run"] if verify else "")
    if bad:
        raise SystemExit("rel_w6_eval: table set %s of %s: %s" % (spec["path"], spec["run"], bad[:5]))
    groups = shards.group_names(spec["run"], doc)
    return dict(doc=doc, groups=groups, row_groups=dict(enumerate(groups)), kind=doc.get("kind"))


def link(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:                                                   # another file system: a copy
        import shutil
        shutil.copyfile(src, dst)
    return dict(path=dst, sha256=sha256_file(dst, use_sidecar=False), bytes=os.path.getsize(dst))


def variant_of_arm(name):
    """The variant name of an arm (<variant>-s<seed>)."""
    head, _, seed = name.rpartition("-s")
    return head if head and seed.isdigit() else name


def build(st, cfg, run_dir):
    s = settings(cfg)
    py, lock, have = w6_tools.interpreter(st, cfg)
    held = heldout_set(s["heldout"], s["verify_files"])
    ro = s["readout"]
    if s["heldout_offline"] is not None:                              # package W3's strict offline rendering of that held-out set
        if ro is not None:
            raise SystemExit("rel_w6_eval: a readout renders its passes itself; no tables.heldout_offline with it")
        off = heldout_set(dict(s["heldout_offline"], split=1), s["verify_files"])["doc"]
        if off.get("layout") != "leakfree" or not set(off.get("parts") or {}) >= {"scattered", "whole_gene"}:
            raise SystemExit("rel_w6_eval.tables.heldout_offline: the strict offline rendering (layout leakfree, parts scattered and whole_gene), "
                             "not %s with parts %s" % (off.get("layout"), sorted(off.get("parts") or {})))
        if int(off.get("tables") or 0) and held["doc"].get("tables") and int(off["tables"]) != int(held["doc"]["tables"]):
            raise SystemExit("rel_w6_eval: the offline rendering holds %s tables, the held-out set %s" % (off.get("tables"), held["doc"].get("tables")))
        st.extra.update(heldout_offline=dict(name=off.get("name"), sha256=off.get("sha256"), run=os.path.basename(s["heldout_offline"]["run"])))
    if (held["kind"] == "ruler") != (ro is not None):
        raise SystemExit("rel_w6_eval: a ruler set is read through a readout (rel_w6_eval.readout) and a readout reads a ruler set (given a %s set%s)"
                         % (held["kind"], " and a readout" if ro else ""))
    variants, docs = {}, []
    for source in s["sources"]:                                       # the arms of every training run named, by name
        doc, rec = source_index(source)
        docs.append(doc)
        for v in doc["variants"]:
            if v["name"] in variants:
                raise SystemExit("rel_w6_eval: arm %s is an arm of two of the runs named in from" % v["name"])
            variants[v["name"]] = (source, rec, v)
    trained = lambda n: os.path.exists(os.path.join(variants[n][0], variants[n][2]["info"]["folder"], "final.pt"))    # noqa: E731
    wanted = s["arms"] or [dict(name=n, untrained=False, weights="final") for n in sorted(variants) if trained(n)]
    missing = [a["name"] for a in wanted if a["name"] not in variants]
    if missing or not wanted:
        raise SystemExit("rel_w6_eval: arms %s are not arms of %s (their arms: %s)" % (missing or wanted, s["sources"], sorted(variants)))
    root = os.path.join(run_dir, OUT)
    os.makedirs(root, exist_ok=True)
    env = w6_tools.environment(s["threads"], hide_gpu=(cfg.get("train") or {}).get("device") == "cpu")
    done = {}
    for want in wanted:
        a, untrained, weights = want["name"], want["untrained"], want.get("weights", "final")
        source, rec, v = variants[a]
        key = a + ("." + UNTRAINED if untrained else ("" if weights == "final" else "." + weights))
        src = os.path.join(source, v["info"]["folder"])
        folder = os.path.join(root, key)
        os.makedirs(folder, exist_ok=True)
        summ = w6_fit.read_json(os.path.join(src, "summary.json")) or {}
        conf = w6_fit.read_json(os.path.join(src, "conf.json")) or {}
        cks, error, point = [], None, {}
        try:
            if not conf:
                raise RuntimeError("no conf.json of the arm in %s" % src)
            read = None                                               # the weights read: (file, which, budget, tables, cells, wall s)
            best = summ.get("best") or {}
            if untrained:
                pass
            elif weights == "final":
                if not os.path.exists(os.path.join(src, "final.pt")) or not summ.get("tables_seen"):
                    raise RuntimeError("no final.pt or no record of the tables the training saw")
                read = ("final.pt", "final", None, summ["tables_seen"], summ.get("cells_seen", 0), float(summ.get("train_s", 0)) + float(summ.get("eval_s", 0)))
            elif weights == "best":
                if not os.path.exists(os.path.join(src, "best.pt")) or not best.get("tables_seen"):
                    raise RuntimeError("no best.pt or no record of the arm's best node")
                read = ("best.pt", "best", None, best["tables_seen"], best.get("cells_seen", 0), best.get("wall_s", 0))
            else:
                n = int(weights.split("@")[1])
                node = next((e for e in summ.get("evals") or [] if int(e.get("tables_seen", -1)) == n), None)
                if node is None or not os.path.exists(os.path.join(src, NODE_WEIGHTS % n)):
                    raise RuntimeError("no node at %d tables with its weights %s (the training's keep_node_weights)" % (n, NODE_WEIGHTS % n))
                read = (NODE_WEIGHTS % n, "ema", "tables=%d" % n, n, node.get("cells_seen", 0), node.get("wall_s", 0))
            if read is not None:
                cks.append(dict(which=read[1], budget=read[2], **link(os.path.join(src, read[0]), os.path.join(folder, read[0]))))
                point = dict(tables=read[3], cells=read[4], wall_s=read[5])
            best_path = os.path.join(src, "best.pt")
            with_best = weights == "final" and not untrained and ro is None and (s["best"] == "all" or (s["best"] and not conf.get("relatives", True))) and \
                best.get("tables_seen") and best["tables_seen"] < summ["tables_seen"] and os.path.exists(best_path)
            if with_best:
                cks.append(dict(which="best", budget=None, **link(best_path, os.path.join(folder, "best.pt"))))
            ec = dict(conf, out=folder, final=UNTRAINED if untrained else os.path.join(folder, read[0]), weights=UNTRAINED if untrained else weights,
                      best=os.path.join(folder, "best.pt") if with_best else None,
                      final_tables=point.get("tables", 0), final_cells=point.get("cells", 0), final_wall_s=float(point.get("wall_s", 0)),
                      best_tables=best.get("tables_seen", 0), best_cells=best.get("cells_seen", 0), best_wall_s=best.get("wall_s", 0),
                      heldout_tables=os.path.join(s["heldout"]["run"], s["heldout"]["path"]), heldout_root=s["heldout"]["run"], heldout_split=s["heldout"]["split"],
                      final_per_group=s["final_per_group"], eval_max_rows=s["eval_max_rows"], threads=s["threads"], run=os.path.basename(run_dir),
                      commit=cfg["git"]["sha"], extra_conditions=s["extra_conditions"] or conf.get("extra_conditions") or [],
                      only_conditions=s["only_conditions"])
            if s["end_controls"] is not None:
                ec["end_controls"] = s["end_controls"]
            if s["heldout_offline"] is not None:
                ec.update(heldout_offline_tables=os.path.join(s["heldout_offline"]["run"], s["heldout_offline"]["path"]),
                          heldout_offline_root=s["heldout_offline"]["run"])
            if ro is not None:                                         # a ruler set: no offline rendering of the held-out set, no controls
                for k in ("heldout_offline_tables", "heldout_offline_root"):
                    ec.pop(k, None)
                ec["end_controls"] = []
            if ro is not None:                                         # the passes of the readout: no rows, a per-cell file
                ec["readout"] = dict(module=ro["module"], max_rows=ro["max_rows"], layout=ro["layout"], in_place_only=ro["in_place_only"], untrained=untrained,
                                     adapt_v0=ro["adapt_v0_arms"],
                                     presence=not untrained and (a in ro["presence_arms"] or variant_of_arm(a) in ro["presence_arms"]))
            cp = os.path.join(folder, "conf.json")
            write_json(cp, ec)
            proc, lg = w6_tools.start(py, "evaluate", cp, os.path.join(folder, "eval.log"), env)
            try:
                code = proc.wait(timeout=s["wall_s"])
            except subprocess.TimeoutExpired:
                code = "over its limit of %d s" % s["wall_s"]
            w6_tools.stop(proc, lg)
            got = w6_fit.read_json(os.path.join(folder, "summary.json")) or {}
            if code != 0 or got.get("status") != "ok":
                raise RuntimeError("evaluation (%s, state %s): %s" % (code, got.get("status"), w6_tools.tail(os.path.join(folder, "eval.log"))))
            if ro is not None:
                if not os.path.exists(os.path.join(folder, RULER_CELLS)):
                    raise RuntimeError("the readout wrote no %s" % RULER_CELLS)
                if untrained:
                    cks.append(dict(which="ema", budget=None, path=os.path.join(folder, "untrained.pt"),
                                    sha256=sha256_file(os.path.join(folder, "untrained.pt"), use_sidecar=False),
                                    bytes=os.path.getsize(os.path.join(folder, "untrained.pt"))))
        except RuntimeError as err:
            error = str(err)[-1500:]
            got = {}
        info = dict(v["info"], final=got.get("final"), final_items=got.get("final_items"), eval_final=got.get("eval_final"),
                    weights=UNTRAINED if untrained else weights,
                    tables_seen=point.get("tables", summ.get("tables_seen", v["info"].get("tables_seen"))) if not untrained else 0,
                    steps=summ.get("steps", v["info"].get("steps")) if not untrained and weights == "final" else (0 if untrained else None),
                    cells_seen=point.get("cells", summ.get("cells_seen", v["info"].get("cells_seen"))) if not untrained else 0, final_best=got.get("final_best"),
                    evaluated_again=dict(source_run=os.path.basename(source), source_status=rec.get("status"), source_arm_status=v["status"],
                                         final_eval_s=got.get("final_eval_s"), checkpoints={c["which"]: c["sha256"] for c in cks}),
                    folder=os.path.relpath(folder, run_dir))
        if ro is not None:
            info.update(final=None, final_items=None, eval_final=None, weights_read=UNTRAINED if untrained else "ema",
                        readout=dict(module=ro["module"], layout=ro["layout"], presence=(ec.get("readout") or {}).get("presence") if error is None else None,
                                     cells=os.path.relpath(os.path.join(folder, RULER_CELLS), run_dir), head=got.get("ruler")))
        done[key] = dict(name=key, variant_file=v["variant_file"], variant_sha256=v["variant_sha256"], api_version=v["api_version"],
                         status="ok" if error is None else "failed", error=error, info=info,
                         checkpoints=[dict(which=c["which"], budget=c.get("budget"), path=os.path.relpath(c["path"], run_dir), sha256=c["sha256"],
                                           bytes=c["bytes"]) for c in cks] if error is None else [])
        if ro is not None and error is None:
            st.output("ruler_cells_" + key, os.path.join(folder, RULER_CELLS))
    results = os.path.join(root, "results.tsv")
    SC.write_rows(results, [r for a in done if os.path.exists(os.path.join(root, a, "results.tsv")) for r in SC.read_rows(os.path.join(root, a, "results.tsv"))])
    tablesets = list({t["sha256"]: t for d in docs for t in d["tablesets"]}.values())
    index = handoff.finish(dict(format="rel-models", format_version=1, tablesets=tablesets, variants=list(done.values()), probe=False,
                                groups=docs[0].get("groups") or held["groups"], row_groups=docs[0].get("row_groups") or {},
                                evaluated_again_from=", ".join(os.path.basename(x) for x in s["sources"]),
                                evaluated_on=dict(run_id=os.path.basename(s["heldout"]["run"]), name=held["doc"].get("name"), sha256=held["doc"].get("sha256"),
                                                  kind=held["kind"], split=s["heldout"]["split"]),
                                readout=ro, model_env=dict(python=py, python_lock=lock, model_env_sha256=sha256_json(have)),
                                producer=dict(stage="w6_eval", package="W6", run_id=os.path.basename(run_dir), commit=cfg["git"]["sha"])))
    st.output("results", results)
    p = os.path.join(root, "models.json")
    write_json(p, index)                                              # the rule's target last
    st.output("models", p)
    st.extra.update(source=", ".join(os.path.basename(x) for x in s["sources"]), arms={a: d["status"] for a, d in done.items()},
                    failed={a: d["error"] for a, d in done.items() if d["status"] != "ok"}, readout=(ro or {}).get("module"))
    return index


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "models.json")
    with open(p) as f:
        doc = json.load(f)
    problems = list(handoff.models_problems(doc, run_dir))
    ok = [v for v in doc.get("variants") or [] if v.get("status") == "ok"]
    if not ok:
        problems.append(dict(error="no arm was evaluated"))
    final = {}
    if doc.get("readout"):                                            # a ruler set through a readout: a per-cell file per arm, no rows
        for v in ok:
            if not os.path.exists(os.path.join(run_dir, v["info"]["folder"], RULER_CELLS)):
                problems.append(dict(arm=v["name"], error="no %s" % RULER_CELLS))
    else:
        for v in ok:                                                  # the rows of the weights each entry read (ema@final, ema@best, ema@tables=n)
            label = "ema@" + weights_label(v["info"].get("weights") or "final")
            p_rows = os.path.join(run_dir, v["info"]["folder"], "results.tsv")
            for r in SC.read_rows(p_rows) if os.path.exists(p_rows) else []:
                if r["weights"] == label and r["group"] == SC.ALL and r["stratum"] == SC.ALL:
                    final.setdefault(v["name"], {})[r["condition"]] = float(r["bits"])
            want = (w6_fit.read_json(os.path.join(run_dir, v["info"]["folder"], "summary.json")) or {}).get("conditions") or list(cond.CONDITIONS)
            lack = [c for c in want if not 0 < final.get(v["name"], {}).get(c, 0) < 20]
            if lack:
                problems.append(dict(arm=v["name"], error="no final row under %s" % ", ".join(lack)))
    require(GateResult("G4_rel_w6_eval", "fail" if problems else "pass", dict(problems=problems[:20], evaluated=[v["name"] for v in ok], final=final,
                                                                          failed={v["name"]: (v.get("error") or "")[-600:] for v in doc.get("variants") or []
                                                                                  if v.get("status") != "ok"})), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "models.json")) as f:
        doc = json.load(f)
    arms = {v["name"]: dict(status=v["status"], tables_seen=v["info"].get("tables_seen"), final=v["info"].get("final"),
                            final_eval_s=(v["info"].get("evaluated_again") or {}).get("final_eval_s"), error=(v.get("error") or "")[-600:] or None)
            for v in doc["variants"]}
    md = ["Saved weights of run %s evaluated again on the held-out tables:" % doc.get("evaluated_again_from"), "",
          "| arm | status | tables seen | " + " | ".join(cond.CONDITIONS) + " |", "|---|---|---|" + "---|" * len(cond.CONDITIONS)]
    for a, x in arms.items():
        md.append("| %s | %s | %s | %s |" % (a, x["status"], x["tables_seen"], " | ".join("%.3f" % x["final"][c] if x["final"] and x["final"].get(c) is not None else ""
                                                                                    for c in cond.CONDITIONS)))
    return dict(json=dict(arms=arms, source=doc.get("evaluated_again_from")), md=md)


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w6_eval") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
