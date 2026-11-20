"""Stage train_data, or its outputs taken over whole from an earlier run with exactly the same inputs (profile
`train_data.reuse_from`; 2026-11-20, the side session's suggestion passed on with the user's consent: rounds on the
same corpus need not rebuild their training data, and a machine without the corpus hit files can still train).

train_data.py is deterministic for given inputs; the key holds every one of them:
  corpus, eval_inputs, reference, hits   the sha256 fields of the four datasets' manifest.json (the corpus and
                                          evaluation-input manifests name every input file, the hits manifest every
                                          hit file, by sha256)
  taxdump                                the run record's taxdump_sha256 (lineages, groups and the split)
  settings                               the train_data settings without the reuse keys
  code                                   sha256 of ci/stages/train_data.py, ci/kingdom.py, ci/taxonomy.py,
                                          configs/kingdom_map.yml (groups, so the split) and ci/manifest.py, whole; and of
                                          the source of the helpers in shared modules that shape the written bytes
                                          (read_fasta and write_json of ci/stages/common.py, save_npz of
                                          seeddrift/train/logger.py, canonical_json and sha256_bytes of ci/hashing.py),
                                          function by function, so that other changes to those modules do not end reuse;
                                          the candidate's from its worktree
A candidate is a run whose train_data stage passed with the index it reported and whose train-data manifest is the
one the index names (manifests/<sha>.json). Every file of that manifest is linked (copied across file systems) and
checked against its sha256; the index is copied byte for byte. train_data.reuse_required: true fails when no candidate
matches (always set with search.reuse_manifest_only, where the hit files are absent). Otherwise train_data.py runs as
usual. This wrapper lives in its own module so that train_data.py, whose checksum is part of the key, stays unchanged.
"""
from __future__ import annotations

import ast
import json
import os
import shutil

import yaml

from ci import ROOT
from ci import manifest as M
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import Stage, main_args

CODE = ("ci/stages/train_data.py", "ci/kingdom.py", "ci/taxonomy.py", "configs/kingdom_map.yml", "ci/manifest.py")
FUNCS = (("ci/stages/common.py", "read_fasta"), ("ci/stages/common.py", "write_json"), ("seeddrift/train/logger.py", "save_npz"),
         ("ci/hashing.py", "canonical_json"), ("ci/hashing.py", "sha256_bytes"))
REUSE_KEYS = ("reuse_from", "reuse_required")
DATASETS = ("corpus", "eval_inputs", "reference", "hits")


def _man_sha(run, name):
    p = os.path.join(run, "data", name, "manifest.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f).get("sha256")


def function_source(path, name):
    """The source text of the top-level function `name` of the Python file `path`, or None."""
    with open(path) as f:
        src = f.read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def key(run, cfg, code_root):
    """The reuse key of `run` with configuration `cfg`, its code read under code_root; None when a part is missing."""
    rp = os.path.join(run, "record.json")
    if not os.path.exists(rp):
        return None
    with open(rp) as f:
        rec = json.load(f)
    paths = [os.path.join(code_root, c) for c in CODE]
    if not all(os.path.exists(p) for p in paths + [os.path.join(code_root, f) for f, _ in FUNCS]):
        return None
    helpers = {}
    for f, name in FUNCS:
        src = function_source(os.path.join(code_root, f), name)
        if src is None:
            return None
        helpers["%s:%s" % (f, name)] = sha256_json(src)
    k = {name: _man_sha(run, name) for name in DATASETS}
    k.update(taxdump=rec.get("taxdump_sha256"), settings=sha256_json({x: v for x, v in (cfg.get("train_data") or {}).items() if x not in REUSE_KEYS}),
             code={c: sha256_file(p, use_sidecar=False) for c, p in zip(CODE, paths)}, helpers=helpers)
    return k if all(k[x] for x in DATASETS + ("taxdump",)) else None


def candidate_ok(cand):
    """The candidate's train_data stage passed with the index it reported, and the manifest that index names exists
    under the candidate's manifests/ with a matching digest. Returns (index, manifest) or None."""
    sp, ip = os.path.join(cand, "stages", "train_data.json"), os.path.join(cand, "data", "train_data", "index.json")
    if not os.path.exists(sp) or not os.path.exists(ip):
        return None
    with open(sp) as f:
        rep = json.load(f)
    out = {o["name"]: o["sha256"] for o in rep.get("outputs") or []}
    if rep.get("status") != "passed" or out.get("train_data_index") != sha256_file(ip, use_sidecar=False):
        return None
    with open(ip) as f:
        idx = json.load(f)
    mp = os.path.join(cand, "manifests", "%s.json" % idx.get("sha256"))
    if not os.path.exists(mp):
        return None
    try:
        man = M.load(mp)
    except M.ManifestError:
        return None
    if M.digest(man) != idx["sha256"]:
        return None
    return idx, man


def link(src, dst, sha):
    """Hard link (a copy across file systems) of src at dst, checked against sha."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
    if sha256_file(dst, use_sidecar=False) != sha:
        raise SystemExit("reused training-data file %s does not match its manifest" % dst)


def candidates(cfg):
    for c in ((cfg.get("train_data") or {}).get("reuse_from") or []):
        cand = c if os.path.isabs(c) else os.path.join(ROOT, c)
        ok = candidate_ok(cand)
        if not ok:
            yield cand, None, None
            continue
        with open(os.path.join(cand, "config.resolved.yaml")) as f:
            ccfg = yaml.safe_load(f)
        code_root = os.path.join(os.path.dirname(os.path.dirname(cand)), "wt", os.path.basename(cand))
        yield cand, key(cand, ccfg, code_root), ok


def main():
    run_dir = main_args()
    with open(os.path.join(run_dir, "config.resolved.yaml")) as f:
        cfg = yaml.safe_load(f)
    dc = cfg.get("train_data") or {}
    ours = key(run_dir, cfg, ROOT)
    cands = list(candidates(cfg)) if dc.get("reuse_from") else []
    for cand, ck, ok in cands:
        if ours is None or ck != ours:
            continue
        idx, man = ok
        with Stage(run_dir, "train_data", requires=("pre", "corpus")) as st:
            for e in man["entries"]:
                for fe in e["files"]:
                    link(os.path.join(cand, fe["path"]), os.path.join(run_dir, fe["path"]), fe["sha256"])
            M.write(man, os.path.join(run_dir, "manifests"))                 # content-addressed copy, same digest
            ip = st.path("train_data", "index.json")
            shutil.copyfile(os.path.join(cand, "data", "train_data", "index.json"), ip)
            st.output("train_data_index", ip)
            st.extra.update(reused_from=os.path.basename(cand), reuse_key=ours, genomes=len(idx["genomes"]),
                            with_hits=sum(1 for g in idx["genomes"] if g["hits"]))
        return
    if dc.get("reuse_required"):
        diffs = {os.path.basename(c): ("not a candidate (stage, index or manifest)" if ok is None else
                                       sorted(x for x in (ours or {}) if (ck or {}).get(x) != ours[x]) if ours else "this run's key is incomplete")
                 for c, ck, ok in cands}
        raise SystemExit("train_data.reuse_required: no run of train_data.reuse_from has this key; per candidate: %s" % (diffs or "no candidate"))
    with open(os.path.join(run_dir, "stages", "search.json")) as f:
        if json.load(f).get("manifest_only"):
            raise SystemExit("the hit files are absent (search.reuse_manifest_only): training data can only be reused here")
    from ci.stages import train_data
    train_data.main()


if __name__ == "__main__":
    main()
