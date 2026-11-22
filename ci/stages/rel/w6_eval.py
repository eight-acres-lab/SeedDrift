"""Stage rel_w6_eval (package W6; pipeline rel_traineval): the saved weights of a training run's arms evaluated again on
the held-out tables, for arms whose final evaluation the wall limit cut short. `python -m ci.stages.rel.w6_eval
RUN_DIR`. Nothing is trained; the weights are read, never changed.

Profile block rel_w6_eval:
  from       the directory of the training run (stage rel_w6_fit) on this machine; it need not have passed (an arm
             stopped at its wall limit makes the run fail) but its stage rel_w6_fit must have written its model index
             (several runs as a list: their arms by name, each name once)
  arms       the arms to evaluate again (default: every arm of those runs with a final.pt); an entry {name, untrained:
             true} reads the arm's model as built with its seed and never trained (a readout's control; untrained.pt)
  readout    {module, presence_arms, max_rows, layout} (with a ruler set only, and a ruler set only with it): instead
             of rows, the passes of the module (package W18: seeddrift.train.rel.readouts.w18_passes, run_ruler) scored
             by every arm, written to data/rel_w6/eval/<arm>/ruler_cells.npz (<arm>.untrained/ for an untrained arm);
             presence_arms (arm or variant names) also run the presence passes; layout stored or leakfree;
             in_place_only (false): the untrained arms, and the presence passes, on the in-place implants only
  tables     heldout: {run, path, sha256, split}: a real held-out set (split 1) or a ruler set of package W18 (the
             split its builder sets; this stage reads the kinds configs/rel/kinds.yaml gives it, read_tableset), as in
             rel_w6_fit otherwise: the held-out table set, a hand-off file of a
             passed run, every file it lists with its recorded hash; the arm's own training configuration names the
             training tables, which are opened (not read) only to build the arm
  best       true (default): an arm trained without relatives is read at its best node too (best.pt, rows ema@best)
  optional   final_per_group (0: every held-out table), eval_max_rows, threads, wall_s (of one arm's evaluation,
             default 7200), verify_files (true)
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
KEYS = {"from", "arms", "tables", "best", "final_per_group", "eval_max_rows", "threads", "wall_s", "verify_files", "readout"}
READOUT_KEYS = {"module", "presence_arms", "max_rows", "layout", "in_place_only"}
UNTRAINED = "untrained"
RULER_CELLS = "ruler_cells.npz"


def settings(cfg) -> dict:
    sc = cfg.get("rel_w6_eval") or {}
    unknown = set(sc) - KEYS
    if unknown or not sc.get("from") or not (sc.get("tables") or {}).get("heldout"):
        raise SystemExit("rel_w6_eval: from and tables.heldout are needed; unknown keys %s" % sorted(unknown))
    h = sc["tables"]["heldout"]
    if set(h) - {"run", "path", "sha256", "split"} or not h.get("run") or not h.get("path") or not isinstance(h.get("split", 1), int):
        raise SystemExit("rel_w6_eval.tables.heldout: run, path, optionally sha256 and split (1 for a real set): %r" % (h,))
    arms = []
    for a in sc.get("arms") or []:
        a = dict(name=a) if isinstance(a, str) else dict(a)
        if set(a) - {"name", "untrained"} or not a.get("name"):
            raise SystemExit("rel_w6_eval.arms: an arm's name, or {name, untrained: true}: %r" % (a,))
        arms.append(dict(name=str(a["name"]), untrained=bool(a.get("untrained", False))))
    if len({(a["name"], a["untrained"]) for a in arms}) != len(arms):
        raise SystemExit("rel_w6_eval.arms: an arm is listed twice")
    ro = sc.get("readout")
    if ro is not None and (set(ro) - READOUT_KEYS or not ro.get("module")):
        raise SystemExit("rel_w6_eval.readout: {module, presence_arms, max_rows, layout, in_place_only}: %r" % (ro,))
    if any(a["untrained"] for a in arms) and ro is None:
        raise SystemExit("rel_w6_eval: an untrained arm is read by a readout only (rel_w6_eval.readout)")
    sources = sc["from"] if isinstance(sc["from"], list) else [sc["from"]]
    return dict(sources=[str(x) for x in sources], source=str(sources[0]), arms=arms,
                heldout=dict(run=str(h["run"]), path=str(h["path"]), sha256=h.get("sha256"), split=int(h.get("split", 1))),
                best=bool(sc.get("best", True)), final_per_group=int(sc.get("final_per_group", 0)), eval_max_rows=int(sc.get("eval_max_rows", 32)),
                threads=int(sc.get("threads", 4)), wall_s=int(sc.get("wall_s", 7200)), verify_files=bool(sc.get("verify_files", True)),
                readout=None if ro is None else dict(module=str(ro["module"]), presence_arms=[str(x) for x in ro.get("presence_arms") or []],
                                                     max_rows=int(ro.get("max_rows", 32)), layout=str(ro.get("layout") or "stored"),
                                                     in_place_only=bool(ro.get("in_place_only", False))))


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
    wanted = s["arms"] or [dict(name=n, untrained=False) for n in sorted(variants) if trained(n)]
    missing = [a["name"] for a in wanted if a["name"] not in variants]
    if missing or not wanted:
        raise SystemExit("rel_w6_eval: arms %s are not arms of %s (their arms: %s)" % (missing or wanted, s["sources"], sorted(variants)))
    root = os.path.join(run_dir, OUT)
    os.makedirs(root, exist_ok=True)
    env = w6_tools.environment(s["threads"], hide_gpu=(cfg.get("train") or {}).get("device") == "cpu")
    done = {}
    for want in wanted:
        a, untrained = want["name"], want["untrained"]
        source, rec, v = variants[a]
        key = a + ("." + UNTRAINED if untrained else "")
        src = os.path.join(source, v["info"]["folder"])
        folder = os.path.join(root, key)
        os.makedirs(folder, exist_ok=True)
        summ = w6_fit.read_json(os.path.join(src, "summary.json")) or {}
        conf = w6_fit.read_json(os.path.join(src, "conf.json")) or {}
        final = os.path.join(src, "final.pt")
        cks, error = [], None
        try:
            if not conf:
                raise RuntimeError("no conf.json of the arm in %s" % src)
            if not untrained and (not os.path.exists(final) or not summ.get("tables_seen")):
                raise RuntimeError("no final.pt or no record of the tables the training saw")
            if not untrained:
                cks.append(dict(which="final", budget=None, **link(final, os.path.join(folder, "final.pt"))))
            best = summ.get("best") or {}
            best_path = os.path.join(src, "best.pt")
            with_best = ro is None and s["best"] and not conf.get("relatives", True) and best.get("tables_seen") and \
                best["tables_seen"] < summ["tables_seen"] and os.path.exists(best_path)
            if with_best:
                cks.append(dict(which="best", budget=None, **link(best_path, os.path.join(folder, "best.pt"))))
            ec = dict(conf, out=folder, final=UNTRAINED if untrained else os.path.join(folder, "final.pt"),
                      best=os.path.join(folder, "best.pt") if with_best else None,
                      final_tables=summ.get("tables_seen", 0), final_cells=summ.get("cells_seen", 0),
                      final_wall_s=float(summ.get("train_s", 0)) + float(summ.get("eval_s", 0)),
                      best_tables=best.get("tables_seen", 0), best_cells=best.get("cells_seen", 0), best_wall_s=best.get("wall_s", 0),
                      heldout_tables=os.path.join(s["heldout"]["run"], s["heldout"]["path"]), heldout_root=s["heldout"]["run"], heldout_split=s["heldout"]["split"],
                      final_per_group=s["final_per_group"], eval_max_rows=s["eval_max_rows"], threads=s["threads"], run=os.path.basename(run_dir),
                      commit=cfg["git"]["sha"])
            if ro is not None:                                         # the passes of the readout: no rows, a per-cell file
                ec["readout"] = dict(module=ro["module"], max_rows=ro["max_rows"], layout=ro["layout"], in_place_only=ro["in_place_only"], untrained=untrained,
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
                    tables_seen=summ.get("tables_seen", v["info"].get("tables_seen")) if not untrained else 0,
                    steps=summ.get("steps", v["info"].get("steps")) if not untrained else 0,
                    cells_seen=summ.get("cells_seen", v["info"].get("cells_seen")) if not untrained else 0, final_best=got.get("final_best"),
                    evaluated_again=dict(source_run=os.path.basename(source), source_status=rec.get("status"), source_arm_status=v["status"],
                                         final_eval_s=got.get("final_eval_s"), checkpoints={c["which"]: c["sha256"] for c in cks}),
                    folder=os.path.relpath(folder, run_dir))
        if ro is not None:
            info.update(final=None, final_items=None, eval_final=None, weights_read=UNTRAINED if untrained else "ema",
                        readout=dict(module=ro["module"], layout=ro["layout"], presence=(ec.get("readout") or {}).get("presence") if error is None else None,
                                     cells=os.path.relpath(os.path.join(folder, RULER_CELLS), run_dir), head=got.get("ruler")))
        done[key] = dict(name=key, variant_file=v["variant_file"], variant_sha256=v["variant_sha256"], api_version=v["api_version"],
                         status="ok" if error is None else "failed", error=error, info=info,
                         checkpoints=[dict(which=c["which"], budget=None, path=os.path.relpath(c["path"], run_dir), sha256=c["sha256"], bytes=c["bytes"])
                                      for c in cks] if error is None else [])
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
        for r in SC.read_rows(os.path.join(run_dir, OUT, "results.tsv")):
            if r["weights"] == "ema@final" and r["group"] == SC.ALL and r["stratum"] == SC.ALL:
                final.setdefault(r["variant"], {})[r["condition"]] = float(r["bits"])
        for v in ok:
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
