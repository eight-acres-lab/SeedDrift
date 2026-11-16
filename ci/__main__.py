"""python -m ci <command>

  check                       T0: repository hygiene, G1 on every corpus spec, G2 static, G8, Snakemake dry runs
  run --profile P --run-dir D start a recorded run of profile P (ci/profiles/P/config.yaml) in the new directory D
      [--resume]              continue an interrupted run in D (Snakemake reruns only incomplete rules; training
                              resumes from its last checkpoint)
      [--set key=value ...]   override profile values for this run (recorded in the config hash)
  compare A B [--expect exact|within_tolerance]
                              reproduction verdict of run B against run A (written into B's record)
  status D                    status of a run (a running record without heartbeat for 10 minutes is "interrupted")
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
from ci.hashing import canonical_json, files_hash, pipeline_hash, sha256_file, sha256_json, walk
from ci.record import Record, atomic_write_json, status_of, validate

LAUNCHER_VERSION = "0.1"


def _git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def git_info():
    sha = _git("rev-parse", "HEAD")
    branch = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME") or _git("branch", "--show-current") or "detached"
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
    mm = shutil.which("micromamba") or os.environ.get("MAMBA_EXE", "")
    prefix = os.environ.get("CONDA_PREFIX", "")
    if mm and prefix:
        r = subprocess.run([mm, "env", "export", "-p", prefix, "--no-md5"], capture_output=True, text=True)
        if r.returncode == 0:
            p = os.path.join(d, "env_export.yml")
            with open(p, "w") as f:
                f.write(r.stdout)
            out["env_export.yml"] = sha256_file(p, use_sidecar=False)
            cfg["env_export"] = p
    return out


def resolve(profile, run_dir, overrides=()):
    cfg = load_profile(profile, overrides)
    cfg["profile"] = profile
    cfg["config_sha256"] = sha256_json(cfg)
    cfg["git"] = git_info()
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


def cmd_run(a):
    run_dir = os.path.abspath(a.run_dir)
    if os.path.exists(os.path.join(run_dir, "record.json")) and not a.resume:
        raise SystemExit("%s already holds a run; a run id is used once (use --resume to continue it)" % run_dir)
    os.makedirs(run_dir, exist_ok=True)
    cfg = resolve(a.profile, run_dir, a.set)
    if a.resume:
        cfg.setdefault("train", {})["resume"] = True
    env_hashes = env_snapshot(run_dir, cfg)
    with open(os.path.join(run_dir, "config.resolved.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    seeds = dict(master=cfg.get("seed", 1), train=cfg.get("train", {}).get("seed"), numpy="per stage from master", torch=None, deterministic=True)
    atomic_write_json(os.path.join(run_dir, "seeds.json"), seeds)
    atomic_write_json(os.path.join(run_dir, "hardware.json"), hardware())
    lock = os.path.join(ROOT, "ci", "envs", "ci.lock.yml")
    taxdump = cfg.get("taxdump")
    tax_hash = files_hash(ROOT, walk(ROOT, [taxdump])) if taxdump and not os.path.isabs(taxdump) else (taxdump or "")
    if a.resume and os.path.exists(os.path.join(run_dir, "record.json")):
        rec = Record.open(run_dir)
        rec.update(status="running", resumed=True)
    else:
        rec = Record.create(run_dir, run_id=os.path.basename(run_dir), ci_version=version(), launcher_version=LAUNCHER_VERSION, pipeline_hash=pipeline_hash(ROOT),
                            profile=a.profile, tier=cfg.get("tier", ""), parent_run=a.parent or None, git=cfg["git"], config_sha256=cfg["config_sha256"],
                            env_lock_sha256=sha256_file(lock, use_sidecar=False), env_snapshot=env_hashes, taxdump_sha256=tax_hash, seeds=seeds,
                            hardware=hardware())
    rec.start_heartbeat()
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONHASHSEED="0")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = str(cfg.get("blas_threads", 1))
    root = os.path.realpath(cfg["data_root"])
    for var, default in (("TMPDIR", os.path.join(root, "tmp")), ("XDG_CACHE_HOME", os.path.join(root, "cache")), ("PIP_CACHE_DIR", os.path.join(root, "cache", "pip"))):
        if not os.path.realpath(env.get(var, "") or "/nonexistent").startswith(root + os.sep):
            env[var] = default                     # temporary files and caches stay on the run's data disk (G7)
        os.makedirs(env[var], exist_ok=True)
    os.environ.update({k: env[k] for k in ("TMPDIR", "XDG_CACHE_HOME", "PIP_CACHE_DIR")})
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    cmd = [sys.executable, "-m", "snakemake", "-s", os.path.join(ROOT, "ci", "Snakefile"), "--directory", run_dir, "--configfile",
           os.path.join(run_dir, "config.resolved.yaml"), "--cores", str(cfg.get("cores", 2)), "--resources", "gpu=1", "diamond=1",
           "--rerun-triggers", "mtime", "--printshellcmds", "--nocolor", "--keep-going"]
    if a.resume:
        cmd += ["--rerun-incomplete"]
    with open(os.path.join(run_dir, "logs", "snakemake.log"), "a") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    rec.merge_stages()
    gates_failed = any(v == "fail" for v in rec.data["gates"].values())
    diverged = any(s.get("status_note") == "diverged" or s.get("status") == "diverged" for s in rec.data["stages"])
    status = "passed" if rc == 0 else ("gate_failed" if gates_failed else ("diverged" if diverged else "failed"))
    ids, mids, metrics = [], [], {}
    for name in ("corpus", "reference", "eval_inputs", "hits"):
        p = os.path.join(run_dir, "data", name, "manifest.json")
        if os.path.exists(p):
            with open(p) as f:
                ids.append(json.load(f)["dataset_id"])
    mj = os.path.join(run_dir, "data", "model", "model.json")
    if os.path.exists(mj):
        with open(mj) as f:
            mids.append(json.load(f)["model_id"])
    rep = os.path.join(run_dir, "data", "eval", "eval_report.json")
    if os.path.exists(rep):
        with open(rep) as f:
            r = json.load(f)
        metrics = {sc["name"]: {row["method"]: dict(auroc=row["auroc"], recall_at_lowest_fpr=row["recall_at_lowest_fpr"]) for row in sc["rows"]} for sc in r["scenarios"]}
    from ci.gates import waivers
    used = [dict(gate=w["gate"], scope=w["scope"], expires=str(w["expires"]), file=w["file"]) for w in waivers()]
    rec.update(dataset_ids=ids, model_ids=mids, metrics=metrics, waivers_used=used, snakemake_exit=rc)
    rec.finish(status)
    summary(run_dir, rec)
    validate(rec.data)
    print(json.dumps(dict(run_id=rec.data["run_id"], status=status, gates=rec.data["gates"]), indent=1))
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


VOLATILE = {("reference", "database")}          # tool-built binaries; their inputs are compared instead


def cmd_compare(a):
    ra, rb = Record.open(a.a), Record.open(a.b)
    diffs = []
    for key in ("ci_version", "pipeline_hash", "config_sha256", "env_lock_sha256", "taxdump_sha256"):
        if ra.data.get(key) != rb.data.get(key):
            diffs.append(dict(field=key, a=ra.data.get(key), b=rb.data.get(key)))
    if ra.data["git"]["sha"] != rb.data["git"]["sha"]:
        diffs.append(dict(field="git.sha", a=ra.data["git"]["sha"], b=rb.data["git"]["sha"]))
    if ra.data["gates"] != rb.data["gates"]:
        diffs.append(dict(field="gates", a=ra.data["gates"], b=rb.data["gates"]))
    sa = {s["name"]: s for s in ra.data["stages"]}
    sb = {s["name"]: s for s in rb.data["stages"]}
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
    if a.expect == "exact" and verdict != "exact":
        return 1
    return 0


def cmd_check(a):
    from ci import hygiene
    from ci.gates import g1_kingdom, g2_exclusion, g8_version
    report = {}
    report["hygiene"] = hygiene.check(ROOT)
    g1 = {}
    for p in sorted(os.listdir(os.path.join(ROOT, "configs", "corpus"))):
        if p.endswith(".yaml") and p != "floors.yaml":
            with open(os.path.join(ROOT, "configs", "corpus", p)) as f:
                r = g1_kingdom.static(yaml.safe_load(f))
            g1[p] = r.status if r.status != "fail" else r.evidence
    report["G1_static"] = g1
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
                                "--configfile", os.path.join(d, "config.resolved.yaml"), "--cores", "1"], capture_output=True, text=True)
            dry[prof] = "ok" if r.returncode == 0 else (r.stdout + r.stderr)[-1500:]
    report["snakemake_dry_run"] = dry
    bad = bool(report["hygiene"]) or any(v != "pass" for v in g1.values()) or report["G2_static"] != "pass" or report["G8"] != "pass" or any(v != "ok" for v in dry.values())
    print(json.dumps(report, indent=1, default=str))
    return 1 if bad else 0


def cmd_status(a):
    print(status_of(a.run_dir))
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
    c.add_argument("--expect", default="exact")
    s = sub.add_parser("status")
    s.add_argument("run_dir")
    a = ap.parse_args(argv)
    return dict(check=cmd_check, run=cmd_run, compare=cmd_compare, status=cmd_status)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
