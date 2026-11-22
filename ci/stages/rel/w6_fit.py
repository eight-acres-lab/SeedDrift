"""Stage rel_w6_fit (package W6 with the minimal trainer of W4; pipeline rel_traineval): the arms of a profile trained
by the relational line's trainer in the model interpreter, each with its held-out evaluation on the way and at the
end. `python -m ci.stages.rel.w6_fit RUN_DIR`. Nothing of the old line's training stage is changed or extended; only
its environment check is imported, and the GPU lease never.

Profile block rel_w6_fit:
  tables     train and heldout, each {run, path, sha256, split}: a table-set manifest (path, relative) of a passed run
             of this machine (run, its directory), the manifest's sha256 when it is to be pinned, and the metadata
             split to take (0 train, 1 held out; omitted: every table). Both may name the same manifest.
  arms       [{variant, seed, budget_cells}]: a variant file (configs/rel/variants/<author>-<name>.yaml), a seed, and
             optionally a budget in cells seen (without it the wall limit is the budget). The arm is <name>-s<seed>.
  wall_s     wall-clock limit of one training, its evaluations included (default 16200); a training ends in the
             normal way at its budget or at this limit, whichever comes first
  probe      true: only tables that are not real, no checkpoint kept (default false: only real tables)
  optional   eval_per_group, eval_points, eval_every_cells, final_per_group, eval_max_rows, max_steps, threads,
             preflight_wall_s, verify_files (default true: every file of the table sets is hashed against its manifest)
The number of trainings at once is rel.parallel (never above the machine's parallel_max, configs/rel/machines.json).

What the stage does:
  1. the model interpreter's packages equal their lock; the table sets are complete and pass the sealed check;
  2. every arm is preflighted alone, one after the other: the model built on the real tables, parameters counted, the
     largest micro-batch run (with fewer tables per micro-batch when it does not fit the arm's memory allowance, the
     variant's gpu_mb), a few steps timed, two held-out tables scored;
  3. the arms that passed are trained, largest declared memory first, at most rel.parallel at once and only while the
     memory their preflights measured fits beside what was free at the start;
  4. an arm that fails (preflight, training, its wall limit) is recorded with the end of its log and does not stop
     the others.
Outputs under data/rel_w6/fit/: <arm>/ (preflight.json, steps.csv, results.tsv, summary.json, heldout_final.npz, the
checkpoints final.pt and best.pt, logs), results.tsv (the result rows of every arm, api.RESULT_ROW), models.json (the
model index of the hand-off, written last). Gate G4_rel_w6_fit: the index is well-formed, at least one arm trained,
and every trained arm has its final rows under each of its conditions.
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

OUT = os.path.join("data", "rel_w6", "fit")
BLOCK_KEYS = {"tables", "arms", "wall_s", "probe", "eval_per_group", "eval_points", "eval_every_cells", "final_per_group", "eval_max_rows", "max_steps",
              "threads", "preflight_wall_s", "verify_files"}
VARIANT_NEED = ("name", "author", "api", "model", "size", "objective", "masks", "layout", "rows", "columns", "relatives", "optimizer", "gpu_mb", "tables")
VARIANT_MAY = ("mask_weights", "micro", "note")
PASS_ON = ("eval_per_group", "eval_points", "eval_every_cells", "final_per_group", "eval_max_rows", "max_steps")
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
               preflight_wall_s=int(sc.get("preflight_wall_s", 900)), verify_files=bool(sc.get("verify_files", True)),
               passed_on={k: sc[k] for k in PASS_ON if sc.get(k) is not None}, tables={}, arms=[])
    for which in ("train", "heldout"):
        t = sc["tables"].get(which) or {}
        if set(t) - {"run", "path", "sha256", "split"} or not t.get("run") or not t.get("path") or t.get("split") not in (None, 0, 1):
            raise SystemExit("rel_w6_fit.tables.%s: run, path, and optionally sha256 and split (0 or 1): %r" % (which, t))
        out["tables"][which] = dict(run=str(t["run"]), path=str(t["path"]), sha256=t.get("sha256"), split=t.get("split"))
    for a in sc["arms"]:
        if set(a) - {"variant", "seed", "budget_cells"} or not a.get("variant") or "seed" not in a:
            raise SystemExit("rel_w6_fit.arms: each has a variant file and a seed, and optionally budget_cells: %r" % (a,))
        v = load_variant(st, a["variant"])
        out["arms"].append(dict(name="%s-s%d" % (v["name"], int(a["seed"])), file=str(a["variant"]), sha256=sha256_file(st.repo(a["variant"]), use_sidecar=False),
                                seed=int(a["seed"]), budget_cells=int(a["budget_cells"]) if a.get("budget_cells") else None, variant=v))
    names = [a["name"] for a in out["arms"]]
    if len(set(names)) != len(names):
        raise SystemExit("rel_w6_fit.arms: an arm (variant and seed) is listed twice: %s" % names)
    return out


def table_set(cfg, spec, verify) -> dict:
    """The manifest of a table set of another run of this machine, complete and through the sealed check."""
    try:
        doc = handoff.from_run(spec["run"], spec["path"], spec["sha256"])
    except (handoff.HandoffError, OSError, ValueError) as err:
        raise SystemExit("rel_w6_fit: table set %s of %s: %s" % (spec["path"], spec["run"], err))
    bad = handoff.tableset_problems(doc, spec["run"] if verify else "")
    if bad:
        raise SystemExit("rel_w6_fit: table set %s of %s: %s" % (spec["path"], spec["run"], bad[:5]))
    genomes = list(doc.get("genomes") or [])
    seal = common.sealed_check(cfg, genomes, splits={g["id"]: g.get("split") for g in genomes})
    return dict(doc=doc, sealed=dict(genomes=len(genomes), ids=seal.get("ids"), lineage=seal.get("lineage")))


def trainer_conf(cfg, s, arm, sets, folder, run_dir) -> dict:
    v = arm["variant"]
    tc = cfg.get("train") or {}
    conf = dict(name=arm["name"], author=v["author"], api=v["api"], model=v["model"], size=v["size"], objective=v["objective"], masks=v["masks"],
                mask_weights=v.get("mask_weights"), layout=v["layout"], rows=v["rows"], columns=v["columns"], relatives=bool(v["relatives"]),
                lr=float(v["optimizer"]["lr"]), gpu_mb=v["gpu_mb"], micro=int(v.get("micro", 4)), seed=arm["seed"], budget_cells=arm["budget_cells"],
                wall_s=s["wall_s"], probe=s["probe"], threads=s["threads"], device="cpu" if tc.get("device") == "cpu" else "auto",
                out=folder, run=os.path.basename(run_dir), commit=cfg["git"]["sha"])
    for which in ("train", "heldout"):
        conf.update({which + "_tables": os.path.join(s["tables"][which]["run"], s["tables"][which]["path"]), which + "_root": s["tables"][which]["run"],
                     which + "_split": s["tables"][which]["split"]})
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
    st.extra.update(sealed_check={k: t["sealed"] for k, t in sets.items()}, tablesets={k: dict(name=t["doc"]["name"], sha256=t["doc"]["sha256"],
                    kind=t["doc"]["kind"], tables=t["doc"]["tables"], run=os.path.basename(s["tables"][k]["run"])) for k, t in sets.items()},
                    model_env=py, parallel=parallel, device="cpu" if cpu else "auto")
    root = os.path.join(run_dir, OUT)
    os.makedirs(root, exist_ok=True)
    env = w6_tools.environment(s["threads"], hide_gpu=cpu)
    arms = {}
    for a in s["arms"]:                                                # every arm alone, before any training starts
        folder = os.path.join(root, a["name"])
        os.makedirs(folder, exist_ok=True)
        conf = os.path.join(folder, "conf.json")
        write_json(conf, trainer_conf(cfg, s, a, sets, folder, run_dir))
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
            conf = dict(read_json(x["conf"]), micro=int(x["preflight"]["micro"]))
            write_json(x["conf"], conf)
            proc, lg = w6_tools.start(py, "fit", x["conf"], os.path.join(x["folder"], "fit.log"), env)
            running[n] = (proc, lg, time.time() + s["wall_s"] + GRACE_S, time.time())
            schedule.append(dict(arm=n, beside=len(running) - 1, started_s=round(time.time() - st.t0)))
        time.sleep(POLL_S)
    return finish(st, cfg, run_dir, s, sets, arms, schedule, free0, dict(python=py, python_lock=lock, model_env_sha256=sha256_json(have)))


GRACE_S, POLL_S = 900, 5


def finish(st, cfg, run_dir, s, sets, arms, schedule, free0, env_info):
    root = os.path.join(run_dir, OUT)
    results = os.path.join(root, "results.tsv")
    with open(results + ".part", "w") as out:
        out.write("\t".join(api.RESULT_ROW) + "\n")
        for n, x in arms.items():
            p = os.path.join(x["folder"], "results.tsv")
            if os.path.exists(p):
                with open(p) as f:
                    out.writelines(f.readlines()[1:])
    os.replace(results + ".part", results)
    variants = []
    for n, x in arms.items():
        a, pre, summ = x["arm"], x["preflight"], x.get("summary") or {}
        info = dict(seed=a["seed"], budget_cells=a["budget_cells"], params=pre.get("params"), flops_per_cell=pre.get("flops_per_cell"), micro=pre.get("micro"),
                    preflight=dict(peak_mb=pre.get("peak_mb"), reserved_mb=pre.get("reserved_mb"), cells_per_s=pre.get("cells_per_s"), memory=pre.get("memory")),
                    steps=summ.get("steps"), cells_seen=summ.get("cells_seen"), target_cells=summ.get("target_cells"), cells_per_s=summ.get("cells_per_s"),
                    peak_mb=summ.get("peak_mb"), reserved_mb=summ.get("reserved_mb"), ended_by=summ.get("ended_by"), final=summ.get("final"),
                    best=summ.get("best"), versions=summ.get("versions"), wall_s=x.get("wall_s"), folder=os.path.relpath(x["folder"], run_dir))
        cks = []
        for which in ("final", "best"):
            p = os.path.join(x["folder"], which + ".pt")
            if x["status"] == "ok" and os.path.exists(p):
                cks.append(dict(which=which, budget=None, path=os.path.relpath(p, run_dir), sha256=sha256_file(p, use_sidecar=False), bytes=os.path.getsize(p)))
        variants.append(dict(name=n, variant_file=a["file"], variant_sha256=a["sha256"], api_version=a["variant"]["api"],
                             status="ok" if x["status"] == "ok" else "failed", error=x.get("error"), info=info, checkpoints=cks))
    doc = handoff.finish(dict(format="rel-models", format_version=1,
                              tablesets=[dict(run_id=os.path.basename(s["tables"][k]["run"]), name=t["doc"]["name"], sha256=t["doc"]["sha256"], use=k,
                                              split=s["tables"][k]["split"]) for k, t in sets.items()],
                              variants=variants, probe=s["probe"], wall_s=s["wall_s"], schedule=schedule, gpu_free_mb_at_start=free0, model_env=env_info,
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
    """{arm: {condition: bits}} of the rows (final weights, all groups, all strata) of the stage's result table."""
    out = {}
    with open(os.path.join(run_dir, OUT, "results.tsv")) as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(head, line.rstrip("\n").split("\t")))
            if r["weights"] == "final" and r["group"] == "all" and r["stratum"] == "all":
                out.setdefault(r["variant"], {})[r["condition"]] = float(r["bits"])
    return out


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "models.json")
    with open(p) as f:
        doc = json.load(f)
    probe = bool(doc.get("probe"))
    problems = [e for e in handoff.models_problems(doc, run_dir) if not (probe and "checkpoints are" in str(e.get("error")))]
    ok = [v for v in doc.get("variants") or [] if v.get("status") == "ok"]
    if not ok:
        problems.append(dict(error="no arm was trained"))
    final = final_rows(run_dir)
    for v in ok:
        want = (read_json(os.path.join(run_dir, v["info"]["folder"], "summary.json")) or {}).get("conditions") or ["?"]
        have = final.get(v["name"], {})
        lack = [c for c in want if c not in have or not 0 < have[c] < 20]
        # a table set without two other rows anywhere has no "repaired" condition: its row is then absent, not wrong
        lack = [c for c in lack if not (c.endswith(api.COND_REPAIRED) and (v["info"].get("final") or {}).get(c, 0) is None)]
        if lack:
            problems.append(dict(arm=v["name"], error="no final row under %s" % ", ".join(lack)))
    failed = {v["name"]: (v.get("error") or "")[-600:] for v in doc.get("variants") or [] if v.get("status") != "ok"}
    require(GateResult("G4_rel_w6_fit", "fail" if problems else "pass", dict(problems=problems[:20], trained=[v["name"] for v in ok], failed=failed, final=final)), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False), sha256_file(os.path.join(run_dir, OUT, "results.tsv"), use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "models.json")) as f:
        doc = json.load(f)
    arms, md = {}, ["| arm | status | parameters | steps | cells seen | cells/s | GPU MB (reserved) | tables per micro-batch | ended by |", "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda x: "-" if x is None else ("{:,}".format(x) if isinstance(x, int) else str(x))
    for v in doc["variants"]:
        i = v["info"]
        arms[v["name"]] = dict(status=v["status"], params=i["params"], steps=i["steps"], cells_seen=i["cells_seen"], target_cells=i["target_cells"],
                               cells_per_s=i["cells_per_s"], peak_mb=i["peak_mb"], reserved_mb=i["reserved_mb"], micro=i["micro"], ended_by=i["ended_by"],
                               final=i["final"], best=i["best"], preflight_cells_per_s=i["preflight"]["cells_per_s"], error=(v.get("error") or "")[-800:] or None)
        md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (v["name"], v["status"], fmt(i["params"]), fmt(i["steps"]), fmt(i["cells_seen"]), fmt(i["cells_per_s"]),
                                                                 fmt(i["reserved_mb"]), fmt(i["micro"]), fmt(i["ended_by"])))
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
