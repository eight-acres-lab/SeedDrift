"""Stage rel_w6_fit (package W6 with the minimal trainer of W4; pipeline rel_traineval): the arms of a profile trained
by the relational line's trainer in the model interpreter, each with its held-out evaluation on the way and at the
end. `python -m ci.stages.rel.w6_fit RUN_DIR`. Nothing of the old line's training stage is changed or extended; only
its environment check is imported, and the GPU lease never.

Profile block rel_w6_fit:
  tables     train and heldout, each {run, path, sha256, split}: a table-set manifest (path, relative) of a passed run
             of this machine (run, its directory), the manifest's sha256 when it is to be pinned, and the metadata
             split to take (0 train, 1 held out; omitted: every table). Both may name the same manifest.
  arms       [{variant, seed, budget_cells, lr}]: a variant file (configs/rel/variants/<author>-<name>.yaml), a seed, and
             optionally a budget in cells seen (without it the wall limit is the budget) and a peak learning rate that
             replaces the variant's. The arm is <name>-s<seed>.
  wall_s     wall-clock limit of one training, its evaluations included (default 16200); a training ends in the
             normal way at its budget or at this limit, whichever comes first
  probe      true: only tables that are not real, no checkpoint kept (default false: only real tables)
  preflight_only  true: every arm preflighted on the real tables and nothing trained (round 1's probes: preflight_steps
             steps, the loss, the memory, tables per second, the sampler's cost); the index lists the arms with their
             preflight and no checkpoint
  optional   eval_per_group (50), eval_first (1000 tables), final_per_group (0: all), eval_max_rows, warmup_steps (1000),
             ema (true), preflight_steps (200), preflight_guard (stop), max_steps, threads, preflight_wall_s, verify_files (default true: every
             file of the table sets is hashed against its manifest before any training starts), extra_conditions (evaluator
             conditions scored besides today's five, e.g. [suffix]), references_at_nodes (false: the extra conditions
             and a leak-free arm's stored readings at the final evaluation only), eval_online (true: a leak-free arm
             without a strict offline rendering is scored online, the anchors the stored pairs), end_controls ([shuffle,
             presence]: package W3's controls of the renderer, at the final evaluation, scattered and whole_gene),
             keep_node_weights (false; true: the moving average of every node kept, ema_tables<tables seen>.pt, for
             readings of two arms at a node both reached), only_conditions (the conditions alone, e.g. round 1's four in
             the loop), nodes_every and nodes_every_from (nodes also every nodes_every tables from nodes_every_from on,
             default 100000), data_workers (0: the examples drawn in the training's process; n: by n background processes)
  tables     ... pair_index {run, path}: the pair index of the uncapped search (package W12, a folder of a passed run),
             for arms that lay the whole gene on its frame relative online by the stored pairs (whole_gene_layout
             leakfree); whole_gene_layout frame (W3's strict aligner) needs none
The number of trainings at once is rel.parallel (never above the machine's parallel_max, configs/rel/machines.json).

What the stage does:
  1. the model interpreter's packages equal their lock; the table sets are hand-off files of passed runs (which for
     a real table set includes the record of its passed sealed check, seeddrift/train/rel/handoff.py from_run) and
     every file they list has its recorded hash;
  2. every arm is preflighted alone, one after the other: the model built on the real tables, parameters counted, the
     largest micro-batch run (with fewer tables per micro-batch when it does not fit the arm's memory allowance, the
     variant's gpu_mb), then 200 steps on a fresh model as the divergence guard (a loss that is not finite or does not
     fall: the preflight fails and the arm is not trained; the learning rate is never lowered on its own unless the
     profile sets preflight_guard: reduce), 72 held-out tables scored (an evaluation batch at its largest);
  3. the arms that passed are trained, largest declared memory first, at most rel.parallel at once and only while the
     memory their preflights measured fits beside what was free at the start;
  4. an arm that fails (preflight, training, its wall limit) is recorded with the end of its log and does not stop
     the others; the run then fails at its gate (below), its files kept for reading.
Outputs under data/rel_w6/fit/: <arm>/ (preflight.json, steps.csv, results.tsv, summary.json, heldout_final.npz, the
checkpoints final.pt and best.pt, logs), results.tsv (the result rows of every arm, api.RESULT_ROW), models.json (the
model index of the hand-off, written last). Gate G4_rel_w6_fit: the index is well-formed; every arm trained (a
preflight-only run: preflighted) and ended at its budget (arm_problems: its budget_cells where it has one, max_steps
where the profile sets it, otherwise the wall limit), so a run with an arm refused for memory, failed or ended early
fails and hands nothing off; and every trained arm has its final rows under each of its conditions.
"""
from __future__ import annotations

import json
import os
import re
import time

import yaml

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common, w6_tools
from seeddrift.train.rel import api, handoff
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import score as SC
from seeddrift.train.rel.evalh import shards

OUT = os.path.join("data", "rel_w6", "fit")
PASS_ON = ("eval_per_group", "eval_first", "final_per_group", "eval_max_rows", "warmup_steps", "ema", "preflight_steps", "preflight_guard", "max_steps",
           "extra_conditions", "only_conditions", "references_at_nodes", "eval_online", "end_controls", "keep_node_weights", "nodes_every",
           "nodes_every_from", "data_workers")
BLOCK_KEYS = {"tables", "arms", "wall_s", "probe", "threads", "preflight_wall_s", "verify_files", "preflight_only"} | set(PASS_ON)
FRAME = "frame"                                                   # whole_gene_layout: W3's strict aligner (tables/align.py frame_layout)
NODE_WEIGHTS = "ema_tables%d.pt"                                  # the trainer's keep_node_weights files
GRACE_S, POLL_S = 900, 5
VARIANT_NEED = ("name", "author", "api", "model", "size", "objective", "masks", "layout", "rows", "columns", "relatives", "optimizer", "gpu_mb", "tables")
VARIANT_MAY = ("mask_weights", "micro", "note", "alone_share", "whole_gene_layout")
API_VERSION = getattr(api, "API", None) or getattr(api, "API_VERSION", None)
NAME_RX = re.compile(r"^[A-Za-z0-9._-]+$")


def load_variant(st, path) -> dict:
    with open(st.repo(path)) as f:
        v = yaml.safe_load(f)
    missing, unknown = [k for k in VARIANT_NEED if k not in v], sorted(set(v) - set(VARIANT_NEED) - set(VARIANT_MAY))
    if missing or unknown:
        raise SystemExit("variant %s: missing keys %s, unknown keys %s" % (path, missing, unknown))
    if not NAME_RX.match(str(v["name"])) or os.path.basename(path) != "%s.yaml" % v["name"] or not str(v["name"]).startswith(str(v["author"]) + "-"):
        raise SystemExit("variant %s: the file is <author>-<name>.yaml and its name key says the same" % path)
    if v["api"] != API_VERSION:
        raise SystemExit("variant %s is written for contract version %s, the contract here is version %s" % (path, v["api"], API_VERSION))
    if set(v["optimizer"]) != {"lr"}:
        raise SystemExit("variant %s: the fixed protocol leaves the optimizer one key, lr" % path)
    return v


def settings(st, cfg) -> dict:
    sc = cfg.get("rel_w6_fit") or {}
    unknown = set(sc) - BLOCK_KEYS
    if unknown or not sc.get("tables") or not sc.get("arms"):
        raise SystemExit("rel_w6_fit: tables and arms are needed; unknown keys %s" % sorted(unknown))
    out = dict(wall_s=int(sc.get("wall_s", 16200)), probe=bool(sc.get("probe", False)), threads=int(sc.get("threads", 4)),
               preflight_only=bool(sc.get("preflight_only", False)),
               preflight_wall_s=int(sc.get("preflight_wall_s", 900)), verify_files=bool(sc.get("verify_files", True)),
               passed_on={k: sc[k] for k in PASS_ON if sc.get(k) is not None}, tables={}, arms=[])
    if set(sc["tables"]) - {"train", "heldout", "heldout_offline", "pair_index"}:
        raise SystemExit("rel_w6_fit.tables: train, heldout and optionally heldout_offline and pair_index: %r" % sorted(sc["tables"]))
    pi = sc["tables"].get("pair_index")
    if pi is not None:                                                # the pair index of W12's uncapped search (a folder of a passed run)
        if set(pi) - {"run", "path"} or not pi.get("run") or not pi.get("path"):
            raise SystemExit("rel_w6_fit.tables.pair_index: {run, path}: %r" % (pi,))
        out["pair_index"] = dict(run=str(pi["run"]), path=str(pi["path"]))
    for which in ("train", "heldout", "heldout_offline"):
        t = sc["tables"].get(which) or {}
        if which == "heldout_offline" and not t:
            continue                                                  # package W3's strict offline rendering: leak-free arms only
        if set(t) - {"run", "path", "sha256", "split"} or not t.get("run") or not t.get("path") or t.get("split") not in (None, 0, 1):
            raise SystemExit("rel_w6_fit.tables.%s: run, path, and optionally sha256 and split (0 or 1): %r" % (which, t))
        out["tables"][which] = dict(run=str(t["run"]), path=str(t["path"]), sha256=t.get("sha256"), split=t.get("split"))
    for a in sc["arms"]:
        if set(a) - {"variant", "seed", "budget_cells", "lr"} or not a.get("variant") or "seed" not in a:
            raise SystemExit("rel_w6_fit.arms: each has a variant file and a seed, and optionally budget_cells and lr: %r" % (a,))
        v = load_variant(st, a["variant"])
        lr = float(a["lr"]) if a.get("lr") is not None else float(v["optimizer"]["lr"])
        if not 0 < lr < 0.1:
            raise SystemExit("rel_w6_fit.arms: a peak learning rate in (0, 0.1): %r" % (a,))
        out["arms"].append(dict(name="%s-s%d" % (v["name"], int(a["seed"])), file=str(a["variant"]), sha256=sha256_file(st.repo(a["variant"]), use_sidecar=False),
                                seed=int(a["seed"]), budget_cells=int(a["budget_cells"]) if a.get("budget_cells") else None, lr=lr, variant=v))
    names = [a["name"] for a in out["arms"]]
    if len(set(names)) != len(names):
        raise SystemExit("rel_w6_fit.arms: an arm (variant and seed) is listed twice: %s" % names)
    return out


def table_set(cfg, spec, verify) -> dict:
    """The manifest of a table set of another run (of this machine, or copied to it): a hand-off file of a passed run,
    with the record of the sealed check made where the tables were built (handoff.from_run refuses a real set without
    it; the check itself needs the sealed registry, which only the machine that builds tables has), every file it
    lists with its recorded hash."""
    probe = bool((cfg.get("rel_w6_fit") or {}).get("probe", False))
    try:
        if probe:                                                     # a probe reads synthetic sets only
            doc = handoff.from_run(spec["run"], spec["path"], spec["sha256"], kinds=("synthetic",))
        else:                                                         # the kinds the registry gives the stage; never a set not for training
            doc = common.read_tableset("w6_fit", spec["run"], spec["path"], spec["sha256"])
    except (handoff.HandoffError, OSError, ValueError) as err:
        raise SystemExit("rel_w6_fit: table set %s of %s: %s" % (spec["path"], spec["run"], err))
    if doc.get("kind") not in ("real", "synthetic"):                  # a ruler set is read by evaluations only, never trained on
        raise SystemExit("rel_w6_fit: table set %s of %s is of kind %s; training reads real (or, probing, synthetic) sets only"
                         % (spec["path"], spec["run"], doc.get("kind")))
    bad = handoff.tableset_problems(doc, spec["run"] if verify else "")
    if bad:
        raise SystemExit("rel_w6_fit: table set %s of %s: %s" % (spec["path"], spec["run"], bad[:5]))
    groups = shards.group_names(spec["run"], doc)                     # the names the group column carries, as the baselines' rows
    return dict(doc=doc, groups=groups, row_groups=dict(enumerate(groups)),
                sealed=(doc.get("sealed_check") or {}).get("passed"), files_verified=bool(verify))


def trainer_conf(cfg, s, arm, sets, folder, run_dir) -> dict:
    v = arm["variant"]
    tc = cfg.get("train") or {}
    conf = dict(name=arm["name"], author=v["author"], api=v["api"], model=v["model"], size=v["size"], objective=v["objective"], masks=v["masks"],
                mask_weights=v.get("mask_weights"), layout=v["layout"], whole_gene_layout=v.get("whole_gene_layout"), rows=v["rows"], columns=v["columns"], relatives=bool(v["relatives"]),
                lr=arm["lr"], gpu_mb=v["gpu_mb"], micro=int(v.get("micro", 4)), alone_share=v.get("alone_share"), seed=arm["seed"], budget_cells=arm["budget_cells"],
                wall_s=s["wall_s"], probe=s["probe"], threads=s["threads"], device="cpu" if tc.get("device") == "cpu" else "auto",
                out=folder, run=os.path.basename(run_dir), commit=cfg["git"]["sha"],
                group_names={str(k): v for k, v in ((sets or {}).get("heldout") or {}).get("row_groups", {}).items()})
    for which in ("train", "heldout"):
        conf.update({which + "_tables": os.path.join(s["tables"][which]["run"], s["tables"][which]["path"]), which + "_root": s["tables"][which]["run"],
                     which + "_split": s["tables"][which]["split"]})
    off = s["tables"].get("heldout_offline")
    if off and v["layout"] != api.LAYOUT_STORED:                      # a leak-free arm is scored on the strict offline rendering
        conf.update(heldout_offline_tables=os.path.join(off["run"], off["path"]), heldout_offline_root=off["run"])
    if s.get("pair_index") and (v.get("whole_gene_layout") or v["layout"]) not in (api.LAYOUT_STORED, FRAME):
        conf.update(pair_index=os.path.join(s["pair_index"]["run"], s["pair_index"]["path"]))
    conf.update(s["passed_on"])
    return conf


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def build(st, cfg, run_dir):
    s = settings(st, cfg)
    py, lock, have = w6_tools.interpreter(st, cfg)
    cpu = (cfg.get("train") or {}).get("device") == "cpu"
    parallel = int((cfg.get("rel") or {}).get("parallel", 1))
    sets = {which: table_set(cfg, s["tables"][which], s["verify_files"]) for which in ("train", "heldout")}
    for which, t in sets.items():
        if s["probe"] == (t["doc"]["kind"] == api.TABLE_REAL):
            raise SystemExit("rel_w6_fit: the %s tables are of kind %s; a probe takes only tables that are not real, every other run only real ones"
                             % (which, t["doc"]["kind"]))
    if any(v["variant"]["tables"] != sets["train"]["doc"]["kind"] for v in s["arms"]):
        raise SystemExit("rel_w6_fit: a variant is written for another kind of tables than %s" % sets["train"]["doc"]["kind"])
    leakfree = [a["name"] for a in s["arms"] if a["variant"]["layout"] != api.LAYOUT_STORED]
    framed = [a["name"] for a in s["arms"] if (a["variant"].get("whole_gene_layout") or a["variant"]["layout"]) not in (api.LAYOUT_STORED, FRAME)]
    if framed and not s["probe"]:                                     # the whole gene on its frame relative: W12's pair index
        pi = s.get("pair_index")
        rec = read_json(os.path.join(pi["run"], "record.json")) if pi else None
        if not pi or (rec or {}).get("status") != "passed" or not os.path.isdir(os.path.join(pi["run"], pi["path"])):
            raise SystemExit("rel_w6_fit: the arms %s lay the whole gene on its frame relative: tables.pair_index, a folder of a passed run "
                             "(given %s)" % (framed, pi))
        st.extra.update(pair_index=dict(run=os.path.basename(pi["run"]), path=pi["path"]))
    if s["tables"].get("heldout_offline"):
        off = table_set(cfg, s["tables"]["heldout_offline"], s["verify_files"])
        if off["doc"].get("layout") != api.LAYOUT_LEAKFREE or set(off["doc"].get("parts") or {}) < {"scattered", "whole_gene"}:
            raise SystemExit("rel_w6_fit.tables.heldout_offline: the strict offline rendering (layout leakfree, parts scattered and whole_gene), "
                             "not %s with parts %s" % (off["doc"].get("layout"), sorted(off["doc"].get("parts") or {})))
        st.extra.update(heldout_offline=dict(name=off["doc"]["name"], sha256=off["doc"]["sha256"], run=os.path.basename(s["tables"]["heldout_offline"]["run"])))
    elif leakfree and not s["probe"] and not s["passed_on"].get("eval_online"):
        raise SystemExit("rel_w6_fit: the leak-free arms %s are scored on the strict offline rendering (tables.heldout_offline), or online "
                         "where the profile says so (eval_online: true)" % leakfree)
    st.extra.update(sealed_record={k: t["sealed"] for k, t in sets.items()}, tablesets={k: dict(name=t["doc"]["name"], sha256=t["doc"]["sha256"],
                    kind=t["doc"]["kind"], tables=t["doc"]["tables"], run=os.path.basename(s["tables"][k]["run"])) for k, t in sets.items()},
                    model_env=py, parallel=parallel, device="cpu" if cpu else "auto")
    root = os.path.join(run_dir, OUT)
    os.makedirs(root, exist_ok=True)
    env = w6_tools.environment(s["threads"], hide_gpu=cpu)
    arms = {}
    # the memory allowance of a preflight: the variant's, or less where the GPU has less free than the training of the
    # arm would need beside nothing else (the scheduler below asks 1.1 times the measured reserve and 500 MB more), so
    # that an arm preflighted at a micro-batch is never refused its training for memory (i-2 beside its vLLM service)
    free_pre = None if cpu else w6_tools.gpu_free_mb()
    cap = None if free_pre is None else int((float(free_pre) - 500.0) / 1.1)
    st.extra.update(gpu_free_mb_at_preflight=free_pre, gpu_mb_cap=cap)
    for a in s["arms"]:                                                # every arm alone, before any training starts
        folder = os.path.join(root, a["name"])
        os.makedirs(folder, exist_ok=True)
        conf = os.path.join(folder, "conf.json")
        c = trainer_conf(cfg, s, a, sets, folder, run_dir)
        if cap is not None and cap < float(c["gpu_mb"]):
            c.update(gpu_mb=cap, gpu_mb_variant=c["gpu_mb"])
        write_json(conf, c)
        log = os.path.join(folder, "preflight.log")
        proc, lg = w6_tools.start(py, "preflight", conf, log, env)
        try:
            code = proc.wait(timeout=s["preflight_wall_s"])
        except Exception:  # noqa: BLE001 - a preflight over its limit is a failed preflight
            code = "over its limit of %d s" % s["preflight_wall_s"]
        w6_tools.stop(proc, lg)
        pre = read_json(os.path.join(folder, "preflight.json")) or {}
        ok = code == 0 and pre.get("ok") is True
        arms[a["name"]] = dict(arm=a, folder=folder, conf=conf, preflight=pre, status="preflighted" if ok else "failed",
                               error=None if ok else "preflight (%s): %s" % (code, pre.get("error") or w6_tools.tail(log)))
    if s["preflight_only"]:                                           # the probes: nothing is trained
        for x in arms.values():
            x.update(status="ok" if x["status"] == "preflighted" else "failed")
        return finish(st, cfg, run_dir, s, sets, arms, [], None if cpu else w6_tools.gpu_free_mb(),
                      dict(python=py, python_lock=lock, model_env_sha256=sha256_json(have)))
    # the trainings: largest declared memory first, at most `parallel` at once, each beside what the others were measured to take
    free0 = None if cpu else w6_tools.gpu_free_mb()
    need = {n: float(x["preflight"].get("reserved_mb") or 0) * 1.1 + 500 for n, x in arms.items()}
    pending = sorted((n for n, x in arms.items() if x["status"] == "preflighted"), key=lambda n: -float(arms[n]["arm"]["variant"]["gpu_mb"]))
    running, schedule = {}, []
    while pending or running:
        for n in list(running):
            proc, lg, deadline, started = running[n]
            late = time.time() > deadline
            if proc.poll() is None and not late:
                continue
            w6_tools.stop(proc, lg)
            del running[n]
            x = arms[n]
            summ = read_json(os.path.join(x["folder"], "summary.json")) or {}
            if proc.returncode == 0 and summ.get("status") == "ok" and not late:
                x.update(status="ok", summary=summ)
            else:
                x.update(status="failed", summary=summ, error="training (exit %s%s, state %s): %s" % (
                    proc.returncode, ", stopped %d s after its wall limit" % GRACE_S if late else "", summ.get("status"), w6_tools.tail(os.path.join(x["folder"], "fit.log"))))
            x["wall_s"] = round(time.time() - started)
        while pending and len(running) < parallel:
            n = pending[0]
            taken = sum(need[m] for m in running)
            if free0 is not None and free0 - taken < need[n]:
                if running:
                    break                                               # wait for memory
                pending.pop(0)
                arms[n].update(status="failed", error="not enough GPU memory: %.0f MB free at the start, %.0f MB needed" % (free0, need[n]))
                continue
            pending.pop(0)
            x = arms[n]
            conf = dict(read_json(x["conf"]), micro=int(x["preflight"]["micro"]), lr=float(x["preflight"]["lr"]))
            write_json(x["conf"], conf)
            proc, lg = w6_tools.start(py, "fit", x["conf"], os.path.join(x["folder"], "fit.log"), env)
            running[n] = (proc, lg, time.time() + s["wall_s"] + GRACE_S, time.time())
            schedule.append(dict(arm=n, beside=len(running) - 1, started_s=round(time.time() - st.t0)))
        time.sleep(POLL_S)
    return finish(st, cfg, run_dir, s, sets, arms, schedule, free0, dict(python=py, python_lock=lock, model_env_sha256=sha256_json(have)))


def finish(st, cfg, run_dir, s, sets, arms, schedule, free0, env_info):
    root = os.path.join(run_dir, OUT)
    results = os.path.join(root, "results.tsv")
    SC.write_rows(results, [r for x in arms.values() if os.path.exists(os.path.join(x["folder"], "results.tsv"))
                            for r in SC.read_rows(os.path.join(x["folder"], "results.tsv"))])
    variants = []
    for n, x in arms.items():
        a, pre, summ = x["arm"], x["preflight"], x.get("summary") or {}
        info = dict(seed=a["seed"], budget_cells=a["budget_cells"], params=pre.get("params"), flops_per_cell=pre.get("flops_per_cell"), micro=pre.get("micro"),
                    lr_peak=a["lr"], lr=pre.get("lr"), lr_reduced=pre.get("lr_reduced"),
                    preflight=dict(peak_mb=pre.get("peak_mb"), reserved_mb=pre.get("reserved_mb"), cells_per_s=pre.get("cells_per_s"), memory=pre.get("memory"),
                                   guard=pre.get("guard"), steps_per_s=pre.get("steps_per_s"), tables_per_s=pre.get("tables_per_s"),
                                   data_wait_s_per_step=pre.get("data_wait_s_per_step"), sampler=pre.get("sampler"), gpu=pre.get("gpu")),
                    schemes=summ.get("schemes"), data_wait_s=summ.get("data_wait_s"),
                    steps=summ.get("steps"), tables_seen=summ.get("tables_seen"), cells_seen=summ.get("cells_seen"), target_cells=summ.get("target_cells"),
                    cells_per_s=summ.get("cells_per_s"), tables_per_s=summ.get("tables_per_s"), train_s=summ.get("train_s"), eval_s=summ.get("eval_s"),
                    eval_overhead=summ.get("eval_overhead"), peak_mb=summ.get("peak_mb"), reserved_mb=summ.get("reserved_mb"), ended_by=summ.get("ended_by"),
                    weights_read=summ.get("weights_read"), eval_subset=summ.get("eval_subset"), eval_final=summ.get("eval_final"),
                    row_key_mask=summ.get("row_key_mask") or pre.get("row_key_mask"), alone_share=summ.get("alone_share"),
                    alone_share_seen=summ.get("alone_share_seen"),
                    final=summ.get("final"), final_items=summ.get("final_items"), best=summ.get("best"),
                    versions=summ.get("versions"), wall_s=x.get("wall_s"), folder=os.path.relpath(x["folder"], run_dir))
        cks = []
        for which in ("final", "best"):
            p = os.path.join(x["folder"], which + ".pt")
            if x["status"] == "ok" and os.path.exists(p):
                cks.append(dict(which=which, budget=None, path=os.path.relpath(p, run_dir), sha256=sha256_file(p, use_sidecar=False), bytes=os.path.getsize(p)))
        for t in (summ.get("node_weights") or []) if x["status"] == "ok" else []:    # keep_node_weights: the moving average of every node
            p = os.path.join(x["folder"], NODE_WEIGHTS % int(t))
            if os.path.exists(p):
                cks.append(dict(which="ema", budget="tables=%d" % int(t), path=os.path.relpath(p, run_dir), sha256=sha256_file(p, use_sidecar=False),
                                bytes=os.path.getsize(p)))
        variants.append(dict(name=n, variant_file=a["file"], variant_sha256=a["sha256"], api_version=a["variant"]["api"],
                             status="ok" if x["status"] == "ok" else "failed", error=x.get("error"), info=info, checkpoints=cks))
    doc = handoff.finish(dict(format="rel-models", format_version=1,
                              tablesets=[dict(run_id=os.path.basename(s["tables"][k]["run"]), name=t["doc"]["name"], sha256=t["doc"]["sha256"], use=k,
                                              split=s["tables"][k]["split"]) for k, t in sets.items()],
                              variants=variants, probe=s["probe"], preflight_only=bool(s.get("preflight_only")), wall_s=s["wall_s"], schedule=schedule,
                              gpu_free_mb_at_start=free0, model_env=env_info,
                              groups=sets["heldout"].get("groups") or [], row_groups={str(k): v for k, v in (sets["heldout"].get("row_groups") or {}).items()},
                              producer=dict(stage="w6_fit", package="W6", run_id=os.path.basename(run_dir), commit=cfg["git"]["sha"])))
    st.output("results", results)
    for v in variants:
        for name in ("summary.json", "preflight.json"):
            p = os.path.join(run_dir, v["info"]["folder"], name)
            if os.path.exists(p):
                st.output("%s_%s" % (name.split(".")[0], v["name"]), p)
    p = os.path.join(root, "models.json")
    write_json(p, doc)                                                  # the rule's target last
    st.output("models", p)
    st.extra.update(arms={v["name"]: v["status"] for v in variants}, schedule=schedule,
                    failed={v["name"]: (v["error"] or "")[-1500:] for v in variants if v["status"] != "ok"})
    return doc


def final_rows(run_dir):
    """{arm: {condition: bits}} of the rows of the final evaluation (all groups, all strata) of the stage's result table."""
    out = {}
    for r in SC.read_rows(os.path.join(run_dir, OUT, "results.tsv")):
        if r["weights"].endswith("@final") and r["group"] == SC.ALL and r["stratum"] == SC.ALL:
            out.setdefault(r["variant"], {})[r["condition"]] = float(r["bits"])
    return out


def arm_problems(doc, cfg) -> list:
    """Every arm of the run trained (a preflight-only run: preflighted) and, trained, ended at its budget: its budget in
    cells where it has one, max_steps where the profile sets it, otherwise the wall limit. An arm that did not (refused
    for GPU memory, failed, ended by its wall limit before its budget) fails the run: a reading of several arms, whose
    differences are the thresholds, never goes on without one of them."""
    out, only = [], bool(doc.get("preflight_only"))
    max_steps = (cfg.get("rel_w6_fit") or {}).get("max_steps")
    for v in doc.get("variants") or []:
        if v.get("status") != "ok":
            out.append(dict(arm=v.get("name"), error="the arm was not %s: %s" % ("preflighted" if only else "trained",
                                                                                 " ".join(str(v.get("error") or "no reason recorded").split())[-400:])))
        elif not only:
            info = v.get("info") or {}
            want = (["budget"] if info.get("budget_cells") else []) + (["max_steps"] if max_steps else []) or ["wall"]
            if info.get("ended_by") not in want:
                out.append(dict(arm=v.get("name"), error="the arm ended by %s, not at its budget (%s)" % (info.get("ended_by"), " or ".join(want))))
    return out


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "models.json")
    with open(p) as f:
        doc = json.load(f)
    probe, only = bool(doc.get("probe")), bool(doc.get("preflight_only"))
    problems = [e for e in handoff.models_problems(doc, run_dir) if not ((probe or only) and "checkpoints are" in str(e.get("error")))]
    ok = [v for v in doc.get("variants") or [] if v.get("status") == "ok"]
    if not ok:
        problems.append(dict(error="no arm was %s" % ("preflighted" if only else "trained")))
    problems += arm_problems(doc, cfg)
    final = {} if only else final_rows(run_dir)
    for v in [] if only else ok:
        want = (read_json(os.path.join(run_dir, v["info"]["folder"], "summary.json")) or {}).get("conditions") or list(cond.CONDITIONS)
        have = final.get(v["name"], {})
        lack = [c for c in want if c not in have or not 0 < have[c] < 20]
        if lack:
            problems.append(dict(arm=v["name"], error="no final row under %s" % ", ".join(lack)))
    failed = {v["name"]: (v.get("error") or "")[-600:] for v in doc.get("variants") or [] if v.get("status") != "ok"}
    require(GateResult("G4_rel_w6_fit", "fail" if problems else "pass", dict(problems=problems[:20], trained=[v["name"] for v in ok], failed=failed, final=final)), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False), sha256_file(os.path.join(run_dir, OUT, "results.tsv"), use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "models.json")) as f:
        doc = json.load(f)
    head = ["arm", "status", "parameters", "peak lr (used)", "steps", "tables seen", "cells seen", "cells/s", "GPU MB reserved", "tables per micro-batch",
            "evaluation share", "ended by"]
    arms, md = {}, ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    fmt = lambda x: "-" if x is None else ("{:,}".format(x) if isinstance(x, int) else str(x))        # noqa: E731
    for v in doc["variants"]:
        i = v["info"]
        arms[v["name"]] = dict(status=v["status"], params=i["params"], lr=i.get("lr"), lr_reduced=i.get("lr_reduced"), steps=i["steps"],
                               tables_seen=i.get("tables_seen"), cells_seen=i["cells_seen"], target_cells=i["target_cells"], cells_per_s=i["cells_per_s"],
                               tables_per_s=i.get("tables_per_s"), peak_mb=i["peak_mb"], reserved_mb=i["reserved_mb"], micro=i["micro"],
                               eval_overhead=i.get("eval_overhead"), ended_by=i["ended_by"], final=i["final"], best=i["best"],
                               preflight_cells_per_s=i["preflight"]["cells_per_s"], error=(v.get("error") or "")[-800:] or None)
        md.append("| " + " | ".join(fmt(x) for x in (v["name"], v["status"], i["params"], "%s%s" % (i.get("lr"), " (reduced)" if i.get("lr_reduced") else ""),
                                                         i["steps"], i.get("tables_seen"), i["cells_seen"], i["cells_per_s"], i["reserved_mb"], i["micro"],
                                                         i.get("eval_overhead"), i["ended_by"])) + " |")
    for v in doc["variants"]:
        if v["status"] != "ok":
            md += ["", "%s failed: `%s`" % (v["name"], " ".join((v.get("error") or "").split())[-600:])]
    return dict(json=dict(arms=arms, tablesets=doc["tablesets"], wall_s=doc.get("wall_s"), probe=doc.get("probe")), md=md)


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w6_fit") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
