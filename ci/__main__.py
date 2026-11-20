"""python -m ci <command>

  check                       T0: repository hygiene, G1 on every corpus spec, G2 static, G8, Snakemake dry runs
  run --profile P --run-dir D start a recorded run of profile P (ci/profiles/P/config.yaml) in the new directory D
      [--resume]              continue an interrupted run in D (Snakemake reruns only incomplete rules; training
                              resumes from its last checkpoint)
      [--set key=value ...]   override profile values for this run (recorded in the config hash)
  compare A B [--expect exact|within_tolerance]
                              reproduction verdict of run B against run A (written into B's record)
  status D                    status of a run (a running record without heartbeat for 10 minutes is "interrupted")
  promote D                   print the catalogue lock of a passed refresh run of pipeline select (with its provenance),
                              to be committed as configs/select/<spec>.lock.yaml (ci/launch.sh promote RUN_ID)

Profiles choose a pipeline (`pipeline:`): `train` (default: corpus, search, training, evaluation), `select` (catalogue
snapshots, the multi-kingdom corpus selection and its cost), `search` (corpus and search only, with their gates: a long
search runs once, and train runs take it whole through `search.reuse_from`) or `selfsup` (the self-supervised model of the
truth-labels line with its own report and gates). Each pipeline has its Snakemake target, its required gates
and the gate phases a resume decides again (PIPELINES).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile

import yaml

from ci import ROOT, version
from ci.hashing import pipeline_hash, sha256_file, sha256_json, taxdump_hash
from ci.record import Record, atomic_write_json, status_of, validate

LAUNCHER_VERSION = "0.1"
# --set may change only these (prefixes): never a gate input such as the corpus spec, taxonomy, disk or GPU limits,
# the CI root or the read-only requirement
SETTABLE = ("train.", "search.threads", "search.block_size", "cores", "blas_threads")
# a run passes only when every one of these gates is present and passed (or waived)
REQUIRED_GATES = ("G6", "G7", "G7_prefix", "G7_resources", "G8", "G1_spec", "G2_static", "G3_static", "G1_corpus", "G2_corpus", "G1_train", "G2_train",
                  "G2_holdout", "G3", "G4")
SEARCH_GATES = ("G6", "G7", "G7_prefix", "G7_resources", "G8", "G1_spec", "G2_static", "G3_static", "G1_corpus", "G2_corpus", "G2_holdout", "G3")
SELECT_GATES = ("G6", "G7", "G7_prefix", "G7_resources", "G8", "G1_select_spec", "G2_static", "G3_static", "G1_select", "G2_select", "G3_select")
PIPELINES = {
    "train": dict(target="all", required=REQUIRED_GATES,
                  resume=(("pre", None), ("corpus", "data/corpus/manifest.json"), ("train", "data/train/ids.json"), ("eval", "data/eval_inputs/manifest.json"),
                          ("report", "data/eval/eval_report.json"))),
    "select": dict(target="select_all", required=SELECT_GATES, resume=(("pre", None), ("select", "data/select/manifest.json"))),   # written last
    "search": dict(target="search_all", required=SEARCH_GATES,
                   resume=(("pre", None), ("corpus", "data/corpus/manifest.json"), ("eval", "data/eval_inputs/manifest.json"))),
    # the truth-labels line: training targets are observed facts (seeddrift/train/selfsup.py); judged by its own report
    # (held-out diagnosis, development island chromosomes), not by the scenario report of pipeline train (G4)
    "selfsup": dict(target="selfsup_all", required=tuple(g for g in REQUIRED_GATES if g != "G4") + ("G3_islands", "G4_selfsup"),
                    resume=(("pre", None), ("corpus", "data/corpus/manifest.json"), ("train", "data/train/ids.json"), ("eval", "data/eval_inputs/manifest.json"),
                            ("islands", "data/islands/index.json"), ("selfsup", "data/selfsup/report.json"))),
    # the codon-usage study of the truth-labels line (ci/stages/codon.py): no corpus, search or training of its own; the
    # inputs are passed runs' outputs, checked by sha256; judged by its own gates
    "codon": dict(target="codon_all", required=("G6", "G7", "G7_prefix", "G7_resources", "G8", "G1_spec", "G2_static", "G3_static", "G3_codon", "G4_codon"),
                  resume=(("pre", None), ("codon", "data/codon/report.json"))),
    # the register of the island reference set for benchmark v2 (ci/stages/register.py; rule bench/hard/V2_REGISTER.md):
    # metadata only, no GPU (profiles set needs_gpu: false)
    "register": dict(target="register_all", required=("G6", "G7", "G7_prefix", "G7_resources", "G8", "G1_spec", "G2_static", "G3_static", "G3_register"),
                     resume=(("pre", None), ("register", "data/register/report.json"))),
}
DATASETS = ("catalog", "select", "corpus", "reference", "eval_inputs", "hits")


def pipeline(cfg):
    p = cfg.get("pipeline", "train")
    if p not in PIPELINES:
        raise SystemExit("unknown pipeline %r (known: %s)" % (p, ", ".join(sorted(PIPELINES))))
    return PIPELINES[p]


def _git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def git_info():
    sha = _git("rev-parse", "HEAD")
    branch = os.environ.get("CI_BRANCH") or os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME") or _git("branch", "--show-current") or "detached"
    clean = not _git("status", "--porcelain", "--untracked-files=all")
    return dict(sha=sha, tree_sha=_git("rev-parse", "HEAD^{tree}"), branch=branch, describe=_git("describe", "--tags", "--always"),
                nearest_ci_tag=_git("describe", "--tags", "--abbrev=0", "--match", "ci-v*") or None, clean=clean,
                provenance="github" if os.environ.get("GITHUB_ACTIONS") else os.environ.get("CI_PROVENANCE", "local"))


def load_profile(name, overrides=()):
    path = os.path.join(ROOT, "ci", "profiles", name, "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for kv in overrides:
        key, _, val = kv.partition("=")
        if not key.startswith(SETTABLE):
            raise SystemExit("--set %s: only %s may be overridden" % (key, ", ".join(SETTABLE)))
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def hardware():
    gpu = None
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        gpu = q.stdout.strip() or None
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        apps = None
    quota = None
    for p in ("/sys/fs/cgroup/cpu.max",):
        if os.path.exists(p):
            quota = open(p).read().strip()
    return dict(machine=platform.machine(), system=platform.platform(), python=sys.version.split()[0], cpu_count=os.cpu_count(), cgroup_cpu_max=quota,
                gpu=gpu, gpu_processes=apps)


def env_snapshot(run_dir, cfg):
    d = os.path.join(run_dir, "env")
    os.makedirs(d, exist_ok=True)
    out = {}
    for name, cmd in (("pip_freeze.txt", [sys.executable, "-m", "pip", "freeze"]), ("diamond_version.txt", [cfg["tools"]["diamond"], "version"]),
                      ("nvidia_smi.txt", ["nvidia-smi", "-q"])):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            text = r.stdout
        except (OSError, subprocess.SubprocessError) as err:
            text = "unavailable: %s\n" % err
        with open(os.path.join(d, name), "w") as f:
            f.write(text)
        out[name] = sha256_file(os.path.join(d, name), use_sidecar=False)
    mm = os.environ.get("MAMBA_EXE") or shutil.which("micromamba") or ""
    if mm:
        r = subprocess.run([mm, "env", "export", "-p", sys.prefix, "--no-md5"], capture_output=True, text=True)     # the interpreter running the pipeline
        if r.returncode == 0:
            p = os.path.join(d, "env_export.yml")
            with open(p, "w") as f:
                f.write("\n".join(l for l in r.stdout.splitlines() if not l.startswith("prefix:")) + "\n")
            out["env_export.yml"] = sha256_file(p, use_sidecar=False)
            cfg["env_export"] = p
    return out


def resolve(profile, run_dir, overrides=()):
    cfg = load_profile(profile, overrides)
    cfg["profile"] = profile
    pipeline(cfg)
    cfg.setdefault("require_lock", True)
    cfg["config_sha256"] = sha256_json(cfg)
    cfg["git"] = git_info()
    cfg["expect_sha"] = os.environ.get("CI_EXPECT_SHA") or os.environ.get("GITHUB_SHA") or ""
    if os.environ.get("CI_ROOT"):
        cfg["ci_root"] = os.environ["CI_ROOT"]
        cfg["require_readonly"] = True
        cfg["data_root"] = os.environ["CI_ROOT"]
    cfg["tools"] = dict(python=sys.executable, diamond=shutil.which("diamond") or "diamond")
    cfg.setdefault("data_root", os.path.dirname(os.path.abspath(run_dir)))
    cfg["run_dir"] = os.path.abspath(run_dir)
    return cfg


def summary(run_dir, rec):
    d = rec.data
    out = dict(run_id=d["run_id"], ci_version=d["ci_version"], sha=d["git"]["sha"], profile=d["profile"], status=d["status"], config_sha256=d["config_sha256"],
               pipeline_hash=d["pipeline_hash"], env_lock_sha256=d["env_lock_sha256"], seeds=d["seeds"], gates=d["gates"], waivers_used=d.get("waivers_used", []),
               stages=[dict(name=s["name"], status=s["status"], wall_s=s["wall_s"], outputs={o["name"]: o["sha256"] for o in s.get("outputs", [])}) for s in d["stages"]],
               dataset_ids=d.get("dataset_ids", []), model_ids=d.get("model_ids", []), metrics=d.get("metrics", {}))
    atomic_write_json(os.path.join(run_dir, "summary.json"), out)
    rows = []
    mpath = os.path.join(run_dir, "data", "train", "logs", "metrics.jsonl")
    if os.path.exists(mpath):
        with open(mpath) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    step = max(1, len(rows) // 2000)
    with open(os.path.join(run_dir, "curve.csv"), "w") as f:
        f.write("step,loss\n")
        for r in rows[::step]:
            f.write("%s,%s\n" % (r["step"], r.get("loss")))
    lines = ["# Run %s" % d["run_id"], "", "- status: **%s**" % d["status"], "- CI %s, commit %s (%s), profile %s" % (d["ci_version"], d["git"]["sha"][:10], d["git"]["branch"], d["profile"]),
             "- gates: " + ", ".join("%s %s" % kv for kv in sorted(d["gates"].items())), ""]
    sel = os.path.join(run_dir, "data", "select", "report.md")
    if os.path.exists(sel):                          # pipeline select: the selection report, then the fit of this disk
        with open(sel) as f:
            lines += [l.replace("# Corpus selection", "## Corpus selection", 1) if l.startswith("# ") else l for l in f.read().splitlines()]
        fp = os.path.join(run_dir, "data", "select", "fit.json")
        if os.path.exists(fp):
            with open(fp) as f:
                fit = json.load(f)
            lines += ["", "## Fit on this data disk (%.0f GB free, %.0f GB reserve)" % (fit["data_root_free_gb"], fit["reserve_gb"]), "",
                      "| case | total GB | margin GB | verdict | uniform fraction that fits | smallest share of a floor kept | search hours |",
                      "|---|---|---|---|---|---|---|"]
            for c, x in fit["cases"].items():
                lines.append("| %s | %.1f | %.1f | %s | %.3f | %.3f%s | %.0f |" % (c, x["total_gb"], x["margin_gb"], x["verdict"], x["uniform_fraction_that_fits"],
                                                                         x["smallest_fraction_of_a_floor"], " (below half: no training)" if x["below_half_of_floors"] else "",
                                                                         x["search_hours"]))
            if fit.get("groups_short_now"):
                lines += ["", "Already below their floors before any scaling: %s." % ", ".join(fit["groups_short_now"])]
            lines += ["", "A uniform scale-down is information, not a plan: disk, network and compute limits are the user's decision."]
        lines.append("")
    rep = os.path.join(run_dir, "data", "eval", "eval_report.json")
    if os.path.exists(rep):
        with open(rep) as f:
            r = json.load(f)
        lines += ["**%s**" % r["catalogue_judge"]["note"], ""]
        for sc in r["scenarios"]:
            lines += ["## %s (%s; %d positives, %d negatives)" % (sc["name"], sc["kind"], sc["n_pos"], sc["n_neg"]), "", "| method | AUROC | recall at the lowest FPR | status |", "|---|---|---|---|"]
            for row in sc["rows"]:
                fmt = lambda x: "—" if x is None else "%.3f" % x
                lines.append("| %s | %s | %s | %s |" % (row["method"], fmt(row["auroc"]), fmt(row["recall_at_lowest_fpr"]), row["status"]))
            lines.append("")
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _resume_check(run_dir, a):
    """A resume continues the same run: same commit, clean tree, same pipeline, configuration and environment lock."""
    if a.set:
        raise SystemExit("--set cannot be combined with --resume (the configuration of a run is fixed)")
    rec = Record.open(run_dir)
    if status_of(run_dir) == "running":
        raise SystemExit("the run is still running (fresh heartbeat); not resuming")
    with open(os.path.join(run_dir, "config.resolved.yaml")) as f:
        cfg = yaml.safe_load(f)
    now_git = git_info()
    problems = []
    if now_git["sha"] != rec.data["git"]["sha"]:
        problems.append("HEAD %s is not the run's commit %s" % (now_git["sha"], rec.data["git"]["sha"]))
    if not now_git["clean"]:
        problems.append("the worktree is not clean")
    if pipeline_hash(ROOT) != rec.data["pipeline_hash"]:
        problems.append("pipeline hash changed")
    if sha256_file(os.path.join(ROOT, "ci", "envs", "ci.lock.yml"), use_sidecar=False) != rec.data["env_lock_sha256"]:
        problems.append("environment lock changed")
    if a.profile != rec.data["profile"]:
        problems.append("profile %s is not the run's profile %s" % (a.profile, rec.data["profile"]))
    if problems:
        raise SystemExit("refusing to resume: " + "; ".join(problems))
    # every gate is decided again; sentinels whose verdict does not change keep their time stamps (gates.py)
    for name in os.listdir(os.path.join(run_dir, "gates")):
        if name.endswith(".json"):
            os.remove(os.path.join(run_dir, "gates", name))
    cfg.setdefault("train", {})["resume"] = True
    env_snapshot(run_dir, cfg)
    with open(os.path.join(run_dir, "config.resolved.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    rec.update(status="running", resumed=(rec.data.get("resumed") or 0) + 1)
    return rec, cfg


def cmd_run(a):
    run_dir = os.path.abspath(a.run_dir)
    if a.resume:
        if not os.path.exists(os.path.join(run_dir, "record.json")):
            raise SystemExit("nothing to resume in %s" % run_dir)
        rec, cfg = _resume_check(run_dir, a)
    else:
        if os.path.exists(os.path.join(run_dir, "record.json")):
            raise SystemExit("%s already holds a run; a run id is used once (use --resume to continue it)" % run_dir)
        os.makedirs(run_dir, exist_ok=True)
        cfg = resolve(a.profile, run_dir, a.set)
        env_hashes = env_snapshot(run_dir, cfg)
        with open(os.path.join(run_dir, "config.resolved.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)
        seeds = dict(master=cfg.get("seed", 1), train=cfg.get("train", {}).get("seed"), numpy="per stage from master", torch=None, deterministic=True)
        atomic_write_json(os.path.join(run_dir, "seeds.json"), seeds)
        hw = hardware()
        atomic_write_json(os.path.join(run_dir, "hardware.json"), hw)
        rec = Record.create(run_dir, run_id=os.path.basename(run_dir), ci_version=version(), launcher_version=LAUNCHER_VERSION, pipeline_hash=pipeline_hash(ROOT),
                            profile=a.profile, pipeline=cfg.get("pipeline", "train"), tier=cfg.get("tier", ""), parent_run=a.parent or None, git=cfg["git"], expect_sha=cfg["expect_sha"],
                            config_sha256=cfg["config_sha256"], env_lock_sha256=sha256_file(os.path.join(ROOT, "ci", "envs", "ci.lock.yml"), use_sidecar=False),
                            env_snapshot=env_hashes, taxdump_sha256=taxdump_hash(cfg["taxdump"]), seeds=seeds, hardware=hw)
    rec.start_heartbeat()
    pipe = pipeline(cfg)
    status, rc = "failed", None
    try:
        env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONHASHSEED="0")
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[k] = str(cfg.get("blas_threads", 1))
        root = os.path.realpath(cfg["data_root"])
        for var, default in (("TMPDIR", os.path.join(root, "tmp")), ("XDG_CACHE_HOME", os.path.join(root, "cache")), ("PIP_CACHE_DIR", os.path.join(root, "cache", "pip"))):
            if not os.path.realpath(env.get(var, "") or "/nonexistent").startswith(root + os.sep):
                env[var] = default                     # temporary files and caches stay on the run's data disk (G7)
            os.makedirs(env[var], exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        cmd = [sys.executable, "-m", "snakemake", "-s", os.path.join(ROOT, "ci", "Snakefile"), "--directory", run_dir, "--configfile",
               os.path.join(run_dir, "config.resolved.yaml"), "--cores", str(cfg.get("cores", 2)), "--resources", "gpu=1", "diamond=1",
               "--rerun-triggers", "mtime", "--printshellcmds", "--nocolor", "--keep-going", pipe["target"]]
        if a.resume:
            # re-decide every gate whose inputs exist before anything continues; sentinels are left as they are
            for phase, need in pipe["resume"]:
                if need and not os.path.exists(os.path.join(run_dir, need)):
                    continue
                if phase == "eval" and not os.path.exists(os.path.join(run_dir, "data", "corpus", "manifest.json")):
                    continue
                r = subprocess.run([sys.executable, "-m", "ci.stages.gates", run_dir, phase, "--check-only"], env=env, capture_output=True, text=True)
                if r.returncode != 0:
                    rec.merge_stages()
                    status = "gate_failed"
                    raise SystemExit("resume refused, gate phase %s failed: %s" % (phase, (r.stderr or r.stdout)[-1500:]))
            cmd += ["--rerun-incomplete"]
        with open(os.path.join(run_dir, "logs", "snakemake.log"), "a") as log:
            rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        rec.merge_stages()
        gates = rec.data["gates"]
        missing = [g for g in pipe["required"] if g not in gates]
        failed = [g for g, v in gates.items() if v == "fail"]
        diverged = any(s.get("status_note") == "diverged" for s in rec.data["stages"])
        if rc == 0 and not missing and not failed:
            status = "passed"
        elif failed:
            status = "gate_failed"
        elif diverged:
            status = "diverged"
        ids, mids, metrics = [], [], {}
        for name in DATASETS:
            mp = os.path.join(run_dir, "data", name, "manifest.json")
            if os.path.exists(mp):
                with open(mp) as f:
                    ids.append(json.load(f)["dataset_id"])
        mj = os.path.join(run_dir, "data", "model", "model.json")
        if os.path.exists(mj):
            with open(mj) as f:
                mids.append(json.load(f)["model_id"])
        rep = os.path.join(run_dir, "data", "eval", "eval_report.json")
        if os.path.exists(rep):
            with open(rep) as f:
                r = json.load(f)
            metrics = {sc["name"]: {row["method"]: dict(auroc=row["auroc"], recall_at_lowest_fpr=row["recall_at_lowest_fpr"], status=row["status"]) for row in sc["rows"]}
                       for sc in r["scenarios"]}
        used = {}
        for name in sorted(os.listdir(os.path.join(run_dir, "gates"))):
            if name.endswith(".json"):
                with open(os.path.join(run_dir, "gates", name)) as f:
                    for w in json.load(f).get("waivers_matched", []):
                        used[(w["gate"], w["scope"])] = w
        rec.update(dataset_ids=ids, model_ids=mids, metrics=metrics, waivers_used=list(used.values()), snakemake_exit=rc, missing_gates=missing)
    except BaseException as err:  # noqa: BLE001 - the record must never stay "running"
        rec.update(deviations=rec.data.get("deviations", []) + ["driver error: %s: %s" % (type(err).__name__, str(err)[:500])])
        status = "failed"
        raise
    finally:
        rec.finish(status)
        try:
            summary(run_dir, rec)
        finally:
            validate(rec.data)
    print(json.dumps(dict(run_id=rec.data["run_id"], status=status, gates=rec.data["gates"], missing_gates=rec.data.get("missing_gates")), indent=1))
    return 0 if status == "passed" else 1


def _content_hash(path):
    """Hash of a stage output that ignores what may differ between identical runs (npz archive layout, JSON key
    order); everything else must match byte for byte."""
    import numpy as np
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        return sha256_json([[k, str(z[k].dtype), list(z[k].shape), sha256_file_bytes(z[k].tobytes())] for k in sorted(z.files)])
    if path.endswith(".json"):
        with open(path) as f:
            return sha256_json(json.load(f))
    return sha256_file(path, use_sidecar=False)


def sha256_file_bytes(b):
    import hashlib
    return hashlib.sha256(b).hexdigest()


VOLATILE = {("reference", "database"),          # tool-built binaries; their inputs are compared instead
            ("fit", "fit")}                      # the budget against the free disk of the moment (pipeline select)


def cmd_compare(a):
    ra, rb = Record.open(a.a), Record.open(a.b)
    diffs = []
    for r in (ra, rb):
        if r.data["status"] != "passed":
            diffs.append(dict(run=r.data["run_id"], error="status is %s, not passed" % r.data["status"]))
    for key in ("ci_version", "pipeline_hash", "config_sha256", "env_lock_sha256", "taxdump_sha256"):
        if ra.data.get(key) != rb.data.get(key):
            diffs.append(dict(field=key, a=ra.data.get(key), b=rb.data.get(key)))
    if ra.data["git"]["sha"] != rb.data["git"]["sha"]:
        diffs.append(dict(field="git.sha", a=ra.data["git"]["sha"], b=rb.data["git"]["sha"]))
    if ra.data["gates"] != rb.data["gates"]:
        diffs.append(dict(field="gates", a=ra.data["gates"], b=rb.data["gates"]))
    for key in ("pip_freeze.txt", "diamond_version.txt"):
        if ra.data.get("env_snapshot", {}).get(key) != rb.data.get("env_snapshot", {}).get(key):
            diffs.append(dict(field="env_snapshot." + key))
    from ci.gates.g7_env import packages
    pa, pb = [packages(open(os.path.join(x, "env", "env_export.yml")).read()) if os.path.exists(os.path.join(x, "env", "env_export.yml")) else None for x in (a.a, a.b)]
    if pa != pb:
        diffs.append(dict(field="environment packages"))
    sa = {s["name"]: s for s in ra.data["stages"] if not s["name"].endswith("_recheck")}
    sb = {s["name"]: s for s in rb.data["stages"] if not s["name"].endswith("_recheck")}
    if set(sa) != set(sb) or not sa:
        diffs.append(dict(field="stages", a=sorted(sa), b=sorted(sb)))
    for name in sorted(set(sa) & set(sb)):
        if sa[name]["status"] != sb[name]["status"]:
            diffs.append(dict(stage=name, field="status", a=sa[name]["status"], b=sb[name]["status"]))
    for name in sorted(set(sa) | set(sb)):
        oa = {o["name"]: o["path"] for o in sa.get(name, {}).get("outputs", [])}
        ob = {o["name"]: o["path"] for o in sb.get(name, {}).get("outputs", [])}
        for out in sorted(set(oa) | set(ob)):
            if (name, out) in VOLATILE:
                continue
            if out not in oa or out not in ob:
                diffs.append(dict(stage=name, output=out, error="missing in one run"))
                continue
            ha, hb = _content_hash(os.path.join(a.a, oa[out])), _content_hash(os.path.join(a.b, ob[out]))
            if ha != hb:
                diffs.append(dict(stage=name, output=out, a=ha, b=hb))
    verdict = "exact" if not diffs else "diverged"
    rb.update(reproduces=dict(of_run_id=ra.data["run_id"], verdict=verdict, diffs=diffs[:100]))
    print(json.dumps(dict(verdict=verdict, diffs=diffs[:20]), indent=1))
    return 0 if verdict == "exact" else 1


def cmd_check(a):
    from ci import hygiene
    from ci.gates import g1_kingdom, g2_exclusion, g8_version
    report = {}
    report["hygiene"] = hygiene.check(ROOT)
    g1 = {}
    for p in sorted(os.listdir(os.path.join(ROOT, "configs", "corpus"))):
        if p.endswith(".yaml") and p != "floors.yaml":
            with open(os.path.join(ROOT, "configs", "corpus", p)) as f:
                spec = yaml.safe_load(f)
            r = g1_kingdom.static(spec, spec.get("tier"))
            g1[p] = r.status if r.status != "fail" else r.evidence
    report["G1_static"] = g1
    gs = {}
    sdir = os.path.join(ROOT, "configs", "select")
    for p in sorted(os.listdir(sdir)) if os.path.isdir(sdir) else []:
        if p.endswith(".yaml") and not p.endswith((".lock.yaml", "assumptions.yaml")) and not p.startswith("rates"):
            from ci.stages.common import load_select_spec
            spec = load_select_spec(os.path.join(sdir, p))
            r = g1_kingdom.select_static(spec, spec.get("tier"))
            gs[p] = r.status if r.status != "fail" else r.evidence
    report["G1_select_spec"] = gs
    g2 = g2_exclusion.static(ROOT)
    report["G2_static"] = g2.status if g2.status != "fail" else g2.evidence
    g8 = g8_version.check(ROOT)
    report["G8"] = g8.status if g8.status != "fail" else g8.evidence
    dry = {}
    for prof in sorted(os.listdir(os.path.join(ROOT, "ci", "profiles"))):
        with tempfile.TemporaryDirectory() as d:
            cfg = resolve(prof, d)
            with open(os.path.join(d, "config.resolved.yaml"), "w") as f:
                yaml.safe_dump(cfg, f)
            r = subprocess.run([sys.executable, "-m", "snakemake", "-n", "-q", "-s", os.path.join(ROOT, "ci", "Snakefile"), "--directory", d,
                                "--configfile", os.path.join(d, "config.resolved.yaml"), "--cores", "1", pipeline(cfg)["target"]], capture_output=True, text=True)
            dry[prof] = "ok" if r.returncode == 0 else (r.stdout + r.stderr)[-1500:]
    report["snakemake_dry_run"] = dry
    bad = bool(report["hygiene"]) or any(v != "pass" for v in g1.values()) or any(v != "pass" for v in gs.values()) or report["G2_static"] != "pass" \
        or report["G8"] != "pass" or any(v != "ok" for v in dry.values())
    print(json.dumps(report, indent=1, default=str))
    return 1 if bad else 0


def cmd_status(a):
    print(status_of(a.run_dir))
    return 0


def cmd_emit_corpus(a):
    """A corpus spec (YAML on stdout) from the selected table of a passed select run: one source per selected genome, in
    the fields the fetch and corpus stages read; the header names the run, so the spec is tied to the selection."""
    import csv
    rec = Record.open(a.run_dir)
    if rec.data["status"] not in ("passed",) + (("gate_failed",) if a.allow_gate_failed else ()):
        raise SystemExit("run %s is %s" % (rec.data["run_id"], rec.data["status"]))
    p = os.path.join(a.run_dir, "data", "select", "selected.tsv")
    with open(p) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    sources, groups = [], {}
    for r in rows:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
        src = dict(id=r["id"], source_kind=r["source_kind"], taxid=int(r["taxid"]))
        if r["source_kind"] in ("raw_prokaryote", "plasmid", "raw_virus"):
            if not r["ena_assembly"]:
                raise SystemExit("%s has no INSDC assembly accession to fetch from ENA" % r["id"])
            src.update(table=int(r["table"] or 11), accession=r["accession"], paired_accession=r["paired_accession"], ena_assembly=r["ena_assembly"])
            if r["replicon_types"]:
                src["replicon_types"] = [r["replicon_types"]]
        elif r["source_kind"] == "annotated_cds":
            e = dict(site=r["ensembl_site"], division=r["ensembl_division"], species=r["ensembl_species"], release=int(r["ensembl_release"]))
            if "_collection_" in r["ensembl_core_db"]:
                e["collection"] = r["ensembl_core_db"].split("_core_")[0]
            src.update(ensembl=e)
            if a.max_genes:
                src["max_genes"] = int(a.max_genes)
        else:
            raise SystemExit("unknown source kind %s" % r["source_kind"])
        if r.get("cluster"):                                   # near relatives (select spec near_relatives)
            src.update(cluster=r["cluster"], relation=r["relation"])
            if a.near_max_genes and r["relation"] != "anchor":
                src["max_genes"] = int(a.near_max_genes)
        sources.append(src)
    spec = dict(name=a.name, tier=a.tier, groups={g: dict(min_genomes=1) for g in sorted(groups)}, sources=sources)
    sys.stdout.write("# Corpus %s, emitted by `python -m ci emit-corpus` from the selection of run %s (CI %s, commit %s, profile %s;\n"
                     "# selected.tsv sha256 %s). Edit the selection, not this file.\n" % (a.name, rec.data["run_id"], rec.data["ci_version"], rec.data["git"]["sha"],
                                                                                   rec.data["profile"], sha256_file(p, use_sidecar=False)))
    sys.stdout.write(yaml.safe_dump(spec, sort_keys=False, width=160))
    return 0


def cmd_promote(a):
    """The catalogue lock of a passed refresh run, with the run that wrote it, ready to be committed."""
    rec = Record.open(a.run_dir)
    if rec.data["status"] != "passed":
        raise SystemExit("run %s is %s; only a passed run's lock may be promoted" % (rec.data["run_id"], rec.data["status"]))
    lp = os.path.join(a.run_dir, "data", "catalog", "lock.yaml")
    if not os.path.exists(lp):
        raise SystemExit("run %s wrote no catalogue lock (not a refresh run of pipeline select)" % rec.data["run_id"])
    out = [s for s in rec.data["stages"] if s["name"] == "catalog"]
    want = {o["name"]: o["sha256"] for o in out[0]["outputs"]} if out else {}
    if want.get("catalog_lock") != sha256_file(lp, use_sidecar=False):
        raise SystemExit("the lock file of %s does not match its record" % rec.data["run_id"])
    with open(lp) as f:
        body = f.read()
    sys.stdout.write("# Catalogue lock written by run %s (CI %s, commit %s, profile %s); promoted with `python -m ci promote`.\n"
                     "# Selection runs in lock mode take exactly these snapshots. Replace it only by promoting a newer refresh run.\n%s"
                     % (rec.data["run_id"], rec.data["ci_version"], rec.data["git"]["sha"], rec.data["profile"], body))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ci", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("--profile", required=True)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--parent", default="")
    r.add_argument("--set", nargs="*", default=[])
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--expect", default="exact", choices=["exact"])
    s = sub.add_parser("status")
    s.add_argument("run_dir")
    pr = sub.add_parser("promote")
    pr.add_argument("run_dir")
    em = sub.add_parser("emit-corpus")
    em.add_argument("run_dir")
    em.add_argument("--name", required=True)
    em.add_argument("--tier", required=True)
    em.add_argument("--max-genes", default=0, help="eukaryotes (annotated_cds): at most this many genes in genome order")
    em.add_argument("--near-max-genes", default=0, help="near relatives (not anchors), any source kind: at most this many genes")
    em.add_argument("--allow-gate-failed", action="store_true")
    a = ap.parse_args(argv)
    return dict(check=cmd_check, run=cmd_run, compare=cmd_compare, status=cmd_status, promote=cmd_promote, **{"emit-corpus": cmd_emit_corpus})[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
