"""Stage gates: `python -m ci.stages.gates RUN_DIR pre|corpus|train|eval|report|select [--check-only]`.

pre      before any data step: G6 clean committed code (against the commit the launcher asked for), G7 environment
         lock, prefix (not moved) and resources, G8 CI version, G1 on the spec (tier must equal the profile's), G2 static,
         G3 static (no sealed accession in the configuration, the corpus spec or the evaluation tables)
corpus   on the built corpus: G1 group floors from taxids, G2 exclusions by full lineage on every genome
train    on the exact id list the training loader will read: non-empty, no duplicates, a subset of the corpus manifest
         and of the same corpus dataset; then G1 and G2 again on exactly those genomes
eval     G2_holdout: no evaluation gene (protein or coding DNA) occurs in the training corpus; G3: no evaluation genome
         under a sealed genus and no sealed accession in any manifest of the run
report   G4 on the evaluation report: six rows per scenario on one gene set, the released screen recomputed from its tag
select   (pipeline select) G1_select, G2_select, G3_select on the selected corpus table; the pre phase of that pipeline
         checks the select spec (G1_select_spec) instead of a corpus spec
Each gate writes gates/<gate>.json. A failure exits non-zero, so no rule that takes the sentinel gates/<phase>.ok as
input can run. The sentinel holds the hash of the phase's inputs and is not rewritten when its content would not
change. --check-only decides the gates without touching the sentinel (used by a resume before Snakemake starts).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import yaml

from ci import ROOT
from ci.gates import GateFailure, GateResult, find_waiver, require, run
from ci.gates import g1_kingdom, g2_exclusion, g3_sealed, g4_report, g6_clean, g7_env, g8_version
from ci.hashing import sha256_file, sha256_json
from ci.kingdom import KingdomMap
from ci.record import atomic_write_json
from ci.stages.common import Stage, read_fasta
from ci.taxonomy import Taxonomy


def _manifest(run_dir, name):
    with open(os.path.join(run_dir, "data", name, "manifest.json")) as f:
        return json.load(f)


def _seq_hashes(folder, names=("proteins.faa", "cds.fna")):
    out = set()
    for n in names:
        for s in read_fasta(os.path.join(folder, n)).values():
            out.add((n, hashlib.sha256(s.upper().rstrip("*").encode()).hexdigest()))
    return out


def phase_pre(st, cfg, run_dir):
    require(g6_clean.check(ROOT, cfg.get("expect_sha") or cfg["git"]["sha"], ci_root=cfg.get("ci_root", ""), require_readonly=cfg.get("require_readonly", False)), run_dir)
    if cfg.get("env_export"):
        require(g7_env.lock_matches(open(cfg["env_export"]).read()), run_dir)
    elif cfg.get("require_lock", True):
        require(GateResult("G7", "fail", dict(error="the environment could not be exported (micromamba not found), so the lock cannot be verified")), run_dir)
    else:
        run(GateResult("G7", "waived", dict(note="profile sets require_lock: false")), run_dir)
    require(g7_env.relocation(), run_dir)
    res = g7_env.resources(cfg["data_root"], float(cfg.get("disk_gb", 1)), float(cfg.get("disk_reserve_gb", 30)), gpu_mem_min_mb=int(cfg.get("gpu_mem_min_mb", 0)))
    res.gate = "G7_resources"
    require(res, run_dir)
    require(g8_version.check(ROOT), run_dir)
    if pipeline_of(cfg) == "select":
        with open(st.repo(cfg["select_spec"])) as f:
            spec = yaml.safe_load(f)
        require(g1_kingdom.select_static(spec, cfg.get("tier")), run_dir)
    else:
        with open(st.repo(cfg["corpus_spec"])) as f:
            spec = yaml.safe_load(f)
        g1 = g1_kingdom.static(spec, cfg.get("tier"))
        g1.gate = "G1_spec"
        require(g1, run_dir)
    g2 = g2_exclusion.static(ROOT)
    g2.gate = "G2_static"
    require(g2, run_dir)
    require(g3_sealed.static(_g3_files(st, cfg, run_dir), cfg.get("tier")), run_dir)
    return sha256_json([cfg.get("expect_sha") or cfg["git"]["sha"], cfg["config_sha256"]])


def pipeline_of(cfg):
    p = cfg.get("pipeline", "train")
    if p not in ("train", "select"):
        raise SystemExit("unknown pipeline %r" % p)
    return p


def _g3_files(st, cfg, run_dir):
    if pipeline_of(cfg) == "select":
        with open(st.repo(cfg["select_spec"])) as f:
            spec = yaml.safe_load(f)
        lock = (cfg.get("catalog") or {}).get("lock")
        return [os.path.join(run_dir, "config.resolved.yaml"), st.repo(cfg["select_spec"]), st.repo(spec["assumptions"])] + \
            ([st.repo(lock)] if lock and os.path.exists(st.repo(lock)) else [])
    sc = cfg.get("scenario") or {}
    return [os.path.join(run_dir, "config.resolved.yaml"), st.repo(cfg["corpus_spec"])] + [st.repo(sc[k]) for k in ("table", "truth") if sc.get(k)]


def phase_select(st, cfg, run_dir):
    """G1_select, G2_select and G3_select on the selected table; all three are decided and recorded before a failure stops
    the run (the report of the selection is written by the budget stage whatever they say)."""
    import csv
    with open(st.repo(cfg["select_spec"])) as f:
        spec = yaml.safe_load(f)
    d = os.path.join(run_dir, "data", "select")
    with open(os.path.join(d, "selected.tsv")) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    results = [run(g1_kingdom.selection(spec, rows, tax, cfg.get("tier")), run_dir)]
    def hosts(r):
        return sorted({int(h) for h in (r.get("hosts") or "").split(";") if h} | ({int(r["host_taxid"])} if r.get("host_taxid") not in ("", "0", None) else set()))
    ent = []
    for r in rows:                                     # one entry per host, so every host of a virus is checked by lineage
        base = dict(id=r["id"], taxid=int(r["taxid"]), accession=r["accession"], paired_accession=r["paired_accession"])
        ent += [dict(base, host_taxid=h) for h in hosts(r)] or [base]
    g2 = g2_exclusion.entries(ent, tax)
    g2.gate = "G2_select"
    results.append(run(g2, run_dir))
    files = [os.path.join(d, n) for n in ("selected.tsv", "by_group.json", "manifest.json")]
    results.append(run(g3_sealed.selection(files, tax, {r["id"]: [int(r["taxid"])] + hosts(r) for r in rows}, cfg.get("tier")), run_dir))
    bad = [r.gate for r in results if r.status == "fail"]
    if bad:
        raise GateFailure("gates failed: %s" % ", ".join(bad))
    return sha256_file(os.path.join(d, "selected.tsv"), use_sidecar=False)


def phase_built(st, cfg, run_dir, phase):
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    man = _manifest(run_dir, "corpus")
    entries = man["entries"]
    inputs = [man["sha256"]]
    if phase == "train":
        p = os.path.join(run_dir, "data", "train", "ids.json")
        with open(p) as f:
            doc = json.load(f)
        ids = doc.get("genomes") or []
        known = {e["id"] for e in entries}
        problems = []
        if not ids:
            problems.append("the training id list is empty")
        if len(ids) != len(set(ids)):
            problems.append("the training id list has duplicates")
        if set(ids) - known:
            problems.append("training ids not in the corpus manifest: %s" % sorted(set(ids) - known)[:5])
        if doc.get("corpus_dataset") != man["dataset_id"]:
            problems.append("the id list names corpus %s, the manifest is %s" % (doc.get("corpus_dataset"), man["dataset_id"]))
        require(GateResult("G2_train_ids", "fail" if problems else "pass", dict(problems=problems, n=len(ids))), run_dir)
        entries = [e for e in entries if e["id"] in set(ids)]
        inputs.append(sha256_file(p, use_sidecar=False))
    with open(st.repo(cfg["corpus_spec"])) as f:
        spec = yaml.safe_load(f)
    g1 = g1_kingdom.built(spec, entries, tax, cfg.get("tier"), KingdomMap())
    g1.gate = "G1_" + phase
    require(g1, run_dir)
    g2 = g2_exclusion.entries(entries, tax)
    g2.gate = "G2_" + phase
    require(g2, run_dir)
    if find_waiver("G5", "*"):
        st.extra["g5"] = "waived: no public-event catalogue yet"
    return sha256_json(inputs)


def phase_eval(st, cfg, run_dir):
    corpus = _manifest(run_dir, "corpus")
    evals = _manifest(run_dir, "eval_inputs")
    train = set()
    for e in corpus["entries"]:
        train |= _seq_hashes(os.path.join(run_dir, "data", "corpus", e["id"]))
    shared = {}
    for e in evals["entries"]:
        n = len(_seq_hashes(os.path.join(run_dir, "data", "eval_inputs", e["id"])) & train)
        if n:
            shared[e["id"]] = n
    ids = {e["id"] for e in corpus["entries"]} & {e["id"] for e in evals["entries"]}
    problems = [dict(error="evaluation genome is a corpus genome", ids=sorted(ids))] if ids else []
    if shared:
        problems.append(dict(error="evaluation sequences found in the training corpus", per_genome=shared))
    require(GateResult("G2_holdout", "fail" if problems else "pass", dict(problems=problems, eval_genomes=len(evals["entries"]))), run_dir)
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    eval_taxids = {}
    truth = []
    sc = cfg.get("scenario") or {}
    if sc.get("truth"):
        import csv
        with open(st.repo(sc["truth"])) as f:
            truth = list(csv.DictReader(f, delimiter="\t"))
    for e in evals["entries"]:
        eval_taxids[e["id"]] = [int(e["host_taxid"])] + [int(t["donor_taxid"]) for t in truth if t.get("scenario") == e["id"] and t.get("donor_taxid")]
    require(g3_sealed.built(run_dir, tax, eval_taxids, cfg.get("tier"), extra_files=_g3_files(st, cfg, run_dir)), run_dir)
    return sha256_json([corpus["sha256"], evals["sha256"]])


def phase_report(st, cfg, run_dir):
    import subprocess
    p = os.path.join(run_dir, "data", "eval", "eval_report.json")
    with open(p) as f:
        report = json.load(f)
    r = subprocess.run(["git", "rev-parse", g4_report.TAG + "^{commit}"], cwd=ROOT, capture_output=True, text=True)
    tag_commit = r.stdout.strip() if r.returncode == 0 else ""
    skips = [k for k, v in (cfg.get("comparators") or {}).items() if v == "skip"]
    require(g4_report.check(report, cfg.get("tier"), skips, tag_commit), run_dir)
    return sha256_file(p, use_sidecar=False)


def main():
    run_dir, phase = sys.argv[1], sys.argv[2]
    check_only = "--check-only" in sys.argv[3:]
    with Stage(run_dir, "gates_" + phase + ("_recheck" if check_only else "")) as st:
        cfg = st.config
        try:
            if phase == "pre":
                key = phase_pre(st, cfg, run_dir)
            elif phase in ("corpus", "train"):
                key = phase_built(st, cfg, run_dir, phase)
            elif phase == "eval":
                key = phase_eval(st, cfg, run_dir)
            elif phase == "report":
                key = phase_report(st, cfg, run_dir)
            elif phase == "select":
                key = phase_select(st, cfg, run_dir)
            else:
                raise SystemExit("unknown gate phase %s" % phase)
        except GateFailure as err:
            st.extra["gate_failure"] = str(err)[:2000]
            raise SystemExit(str(err)[:2000])
        except Exception as err:  # noqa: BLE001 - a gate that cannot run has failed (fail closed)
            run(GateResult("gates_%s_error" % phase, "fail", dict(error="%s: %s" % (type(err).__name__, err))), run_dir)
            st.extra["gate_failure"] = "%s: %s" % (type(err).__name__, err)
            raise
        if check_only:
            return
        sentinel = os.path.join(run_dir, "gates", "%s.ok" % phase)
        content = {"phase": phase, "passed": True, "inputs": key}
        old = None
        if os.path.exists(sentinel):
            with open(sentinel) as f:
                old = json.load(f)
        if old != content:
            atomic_write_json(sentinel, content)
        st.output("sentinel", sentinel)


if __name__ == "__main__":
    main()
