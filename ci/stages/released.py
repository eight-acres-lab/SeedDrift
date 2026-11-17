"""Stage released: the released single-genome screen (tag v0.1.0), recomputed in this run on the evaluation genes.

Nothing is taken from earlier tables. The stage
  - unpacks the tag with `git archive` into data/released/src (commit and tree recorded);
  - uses the tag's own environment, built by the launcher from ci/envs/released-v0.1.0.lock.yml at
    <env_root>/sd-released-<lock hash> (complete only with its marker) and checked against the lock again here;
  - gives it a data folder of its own (data/released/store): the shared reference data as read-only links, an empty
    cache and job folder, no tools or folding service (SEEDDRIFT_TOOLS_URL and SEEDDRIFT_ESMFOLD_URL empty), so
    the production services are never called;
  - runs `python -m seeddrift genome proteins.faa --self-taxon <host> --no-deep --no-trees` per evaluation genome
    and writes data/released/<id>/scores.tsv keyed by the CI's gene ids (the screen renames proteins; the renaming
    of v0.1.0 is repeated here and checked to be one-to-one).
The score of a gene: genes the screen calls foreign rank first, each group by the screen's Alien Index (genes without
one last). A profile without `released:` must set `comparators: {released_v0.1.0: skip}` (only tier tiny may;
gate G4); the stage then writes a manifest that says so.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import tarfile
import io

from ci import ROOT, manifest as M
from ci.gates import g7_env
from ci.stages.common import Stage, main_args, read_fasta, write_json

FOREIGN_RELEASED = ("mammal", "prokaryote", "eukaryote", "virus")       # seeddrift/evidence.py FOREIGN at v0.1.0


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def released_name(k):
    """The protein name v0.1.0 gives a FASTA name (seeddrift/pipeline.py clean_proteins at the tag)."""
    return re.sub(r"[^\w.:-]", "_", k)[:60] or "gene"


def score_of(row):
    ai = row.get("AI", "")
    try:
        a = float(ai)
    except ValueError:
        a = -1000.0
    return (10000.0 if row.get("category") in FOREIGN_RELEASED else 0.0) + a


def main():
    run_dir = main_args()
    with Stage(run_dir, "released", requires=("pre", "eval")) as st:
        cfg = st.config
        rc = cfg.get("released")
        mp = st.path("released", "manifest.json")
        if not rc:
            if (cfg.get("comparators") or {}).get("released_v0.1.0") != "skip":
                raise SystemExit("the profile has neither `released:` nor `comparators: {released_v0.1.0: skip}`")
            write_json(mp, dict(status="skipped", reason="profile %s skips the released screen (tier %s)" % (cfg.get("profile"), cfg.get("tier")), entries=[]))
            st.output("released_manifest", mp)
            st.extra["released"] = "skipped"
            return
        tag = rc["tag"]
        commit, tree = _git("rev-parse", tag + "^{commit}"), _git("rev-parse", tag + "^{tree}")
        src = st.path("released", "src", "x")[:-2]
        tar = subprocess.run(["git", "archive", "--format=tar", commit], cwd=ROOT, capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
            tf.extractall(src, filter="data")
        lock = st.repo(rc["lock"])
        with open(lock, "rb") as f:
            lock_sha = hashlib.sha256(f.read()).hexdigest()
        env = os.path.join(rc["env_root"], "sd-released-%s" % lock_sha[:8])
        if not os.path.exists(os.path.join(env, ".ci-env-complete")):
            raise SystemExit("the released environment %s is not built (ci/server_launch.sh builds it from %s)" % (env, rc["lock"]))
        mm = os.environ.get("MAMBA_EXE") or "micromamba"
        exported = g7_env.export_env(env, mm)
        lk = g7_env.lock_matches(exported, lock)
        if lk.status != "pass":
            raise SystemExit("the released environment differs from its lock: %s" % json.dumps(lk.evidence)[:1500])
        store = st.path("released", "store", "x")[:-2]
        links = {}
        for name in rc.get("links", ["reference", "pfam", "sprot", "afdb"]):
            target = os.path.join(rc["store"], name)
            if not os.path.exists(target):
                raise SystemExit("released data %s is missing" % target)
            dst = os.path.join(store, name)
            if not os.path.islink(dst):
                os.symlink(target, dst)
            links[name] = os.path.realpath(target)
        for name in ("cache", "jobs"):
            os.makedirs(os.path.join(store, name), exist_ok=True)
        py = os.path.join(env, "bin", "python")
        # no SEEDDRIFT_* setting of the calling shell reaches the screen (a production env file could point its jobs,
        # binaries or services elsewhere); only these are set, and the tools are the locked environment's
        envv = {k: v for k, v in os.environ.items() if not k.startswith(("SEEDDRIFT_", "PYTHON", "CONDA"))}
        envv.update(SEEDDRIFT_DATA=store, SEEDDRIFT_JOBS=os.path.join(store, "jobs"), SEEDDRIFT_TOOLS_URL="", SEEDDRIFT_ESMFOLD_URL="",
                    SEEDDRIFT_THREADS=str(rc.get("threads", 12)), SEEDDRIFT_CACHE="1", PYTHONNOUSERSITE="1", CONDA_PREFIX=env,
                    PATH=os.path.join(env, "bin") + os.pathsep + "/usr/bin:/bin")
        where = subprocess.run([py, "-c", "import seeddrift; print(seeddrift.__file__)"], cwd=src, env=envv, capture_output=True, text=True, check=True).stdout.strip()
        if not os.path.realpath(where).startswith(os.path.realpath(src) + os.sep):
            raise SystemExit("the released code was imported from %s, not from the unpacked tag" % where)
        with open(os.path.join(run_dir, "data", "eval_inputs", "manifest.json")) as f:
            evals = json.load(f)["entries"]
        entries = []
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        for e in evals:
            folder = os.path.join(run_dir, "data", "eval_inputs", e["id"])
            faa = os.path.join(folder, "proteins.faa")
            out = st.path("released", e["id"], "job", "x")[:-2]
            cmd = [py, "-m", "seeddrift", "genome", faa, "--out", out, "--self-taxon", str(e["host_taxid"]), "--no-deep", "--no-trees"]
            with open(os.path.join(run_dir, "logs", "released-%s.log" % e["id"]), "w") as log:
                subprocess.run(cmd, cwd=src, env=envv, stdout=log, stderr=subprocess.STDOUT, check=True)
            names = list(read_fasta(faa))
            back = {}
            for n in names:
                r = released_name(n)
                if r in back:
                    raise SystemExit("two proteins get the released name %s" % r)
                back[r] = n.split("__", 1)[1]
            with open(os.path.join(out, "genes.tsv")) as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            sp = st.path("released", e["id"], "scores.tsv")
            with open(sp + ".part", "w") as f:
                f.write("gene_id\tcategory\tAI\tscore\n")
                for r in rows:
                    if r["gene_id"] not in back:
                        raise SystemExit("the released screen reports an unknown gene %s" % r["gene_id"])
                    f.write("%s\t%s\t%s\t%.3f\n" % (back[r["gene_id"]], r["category"], r.get("AI", ""), score_of(r)))
            os.replace(sp + ".part", sp)
            entries.append(dict(id=e["id"], scored=len(rows), genes=len(names), files=[M.file_entry(sp, run_dir), M.file_entry(os.path.join(out, "genes.tsv"), run_dir)]))
        prov = dict(tag=tag, commit=commit, tree=tree, lock=rc["lock"], lock_sha256=lock_sha, env=env, data=links,
                    command="python -m seeddrift genome proteins.faa --out DIR --self-taxon HOST --no-deep --no-trees",
                    services="none (SEEDDRIFT_TOOLS_URL and SEEDDRIFT_ESMFOLD_URL empty: tools run in the process)")
        man = M.build("released", "released", entries, meta=prov)
        sha = M.write(man, os.path.join(run_dir, "manifests"))
        write_json(mp, dict(sha256=sha, dataset_id=M.dataset_id(man), status="computed", provenance=prov, **man))
        st.output("released_manifest", mp)
        st.extra.update(released="computed", commit=commit[:12], genomes=len(entries))


if __name__ == "__main__":
    main()
