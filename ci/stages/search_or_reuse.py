"""Stage search, or its outputs taken over from earlier runs with exactly the same inputs (profile `search.reuse_from`).

The search is deterministic: the same proteins, reference, search settings, search code and DIAMOND version give the
same hit file for a genome, whatever other genomes are searched with it (each query's hits depend only on the query and
the database). Two levels of reuse, tried in this order:
  whole run   a run named in `reuse_from` whose corpus, evaluation inputs, reference, settings, code and DIAMOND
              version are all equal and whose search stage passed: every hit file is linked
  per genome  runs with equal reference, settings, code and DIAMOND version: a genome whose input files (genes.tsv,
              proteins.faa, cds.fna; sha256 in the corpus or evaluation-input manifest) are equal takes that run's hit
              file; ci/stages/search.py searches only the others, plus `reuse_check` (default 2) of the reused genomes
              (the smallest), whose new hit files must equal the reused ones byte for byte, or the stage fails
Every linked file is checked against its sha256 in the source run's manifest. The stage report records where each part
came from. Otherwise (no candidate, or any difference) search.py runs as usual; this wrapper lives in its own module so
that search.py, whose checksum is part of the key, stays unchanged.
"""
from __future__ import annotations

import json
import os

import yaml

from ci import ROOT
from ci.hashing import sha256_file, sha256_json
from ci.stages import search
from ci.stages.common import Stage, main_args, write_json

SEARCH_CODE = os.path.join(ROOT, "ci", "stages", "search.py")


def _sha(run, name):
    with open(os.path.join(run, "data", name, "manifest.json")) as f:
        return json.load(f)["sha256"]


def key(run, search_cfg, code_path):
    with open(os.path.join(run, "record.json")) as f:
        rec = json.load(f)
    sc = {k: v for k, v in (search_cfg or {}).items() if k not in ("reuse_from", "reuse_check")}    # how to reuse, not what to search
    return dict(corpus=_sha(run, "corpus"), eval_inputs=_sha(run, "eval_inputs"), reference=_sha(run, "reference"), search=sha256_json(sc),
                code=sha256_file(code_path, use_sidecar=False), diamond=(rec.get("env_snapshot") or {}).get("diamond_version.txt"))


def candidate_ok(cand):
    """A run whose search stage passed and whose hit manifest is the one that stage reported and covers every genome
    of its corpus and evaluation inputs (a run killed between the partial search and the full manifest of the
    per-genome path below leaves a passed partial report: not a candidate)."""
    p, mp = os.path.join(cand, "stages", "search.json"), os.path.join(cand, "data", "hits", "manifest.json")
    if not os.path.exists(p) or not os.path.exists(mp):
        return False
    with open(p) as f:
        rep = json.load(f)
    if rep.get("status") != "passed":
        return False
    out = {o["name"]: o["sha256"] for o in rep.get("outputs") or []}
    if out.get("hits_manifest") != sha256_file(mp, use_sidecar=False):
        return False
    with open(mp) as f:
        have = {e["id"] for e in json.load(f)["entries"]}
    return have == set(inputs(cand))


def inputs(run):
    """{genome id: its input files as sorted (name, sha256)} from the corpus and evaluation-input manifests."""
    out = {}
    for name in ("corpus", "eval_inputs"):
        p = os.path.join(run, "data", name, "manifest.json")
        if os.path.exists(p):
            with open(p) as f:
                for e in json.load(f)["entries"]:
                    out[e["id"]] = sorted((os.path.basename(x["path"]), x["sha256"]) for x in e["files"])
    return out


def link(src, dst, sha):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    os.link(src, dst)
    if sha256_file(dst, use_sidecar=False) != sha:
        raise SystemExit("reused hit file %s does not match its manifest" % dst)


def candidates(cfg):
    for c in (cfg["search"].get("reuse_from") or []):
        cand = c if os.path.isabs(c) else os.path.join(ROOT, c)
        if not candidate_ok(cand):
            continue
        with open(os.path.join(cand, "config.resolved.yaml")) as f:
            ccfg = yaml.safe_load(f)
        code = os.path.join(os.path.dirname(os.path.dirname(cand)), "wt", os.path.basename(cand), "ci", "stages", "search.py")
        if os.path.exists(code):
            yield cand, key(cand, ccfg["search"], code)


def main():
    run_dir = main_args()
    with open(os.path.join(run_dir, "config.resolved.yaml")) as f:
        cfg = yaml.safe_load(f)
    ours = key(run_dir, cfg["search"], SEARCH_CODE)
    cands = list(candidates(cfg))
    from ci import manifest as M
    for cand, ck in cands:                                        # whole run
        if ck != ours:
            continue
        with Stage(run_dir, "search", requires=("pre", "corpus")) as st:
            with open(os.path.join(cand, "data", "hits", "manifest.json")) as f:
                man = json.load(f)
            for e in man["entries"]:
                for fe in e["files"]:
                    link(os.path.join(cand, fe["path"]), os.path.join(run_dir, fe["path"]), fe["sha256"])
            M.write({k: man[k] for k in ("name", "kind", "entries", "meta")}, os.path.join(run_dir, "manifests"))   # content-addressed copy
            mp = st.path("hits", "manifest.json")
            write_json(mp, man)
            st.output("hits_manifest", mp)
            st.extra.update(reused_from=os.path.basename(cand), reuse_key=ours, genomes=len(man["entries"]))
        return
    glob = {k: v for k, v in ours.items() if k not in ("corpus", "eval_inputs")}
    mine = inputs(run_dir)
    reuse = {}                                                   # genome -> (run, file entry)
    for cand, ck in cands:                                        # per genome
        if {k: ck[k] for k in glob} != glob:
            continue
        theirs = inputs(cand)
        with open(os.path.join(cand, "data", "hits", "manifest.json")) as f:
            hm = {e["id"]: e["files"][0] for e in json.load(f)["entries"]}
        for gid, files in mine.items():
            if gid not in reuse and gid in hm and theirs.get(gid) == files:
                reuse[gid] = (cand, hm[gid])
    if not reuse:
        search.main()
        return
    n_check = int(cfg["search"].get("reuse_check", 2))
    size = {}
    for g in reuse:
        d = "corpus" if os.path.isdir(os.path.join(run_dir, "data", "corpus", g)) else "eval_inputs"
        with open(os.path.join(run_dir, "data", d, g, "genes.tsv")) as f:
            size[g] = sum(1 for _ in f) - 1
    # the smallest genomes with 30 genes or more: cheap to search again, and not empty
    check = sorted((g for g in reuse if size[g] >= 30), key=lambda g: (size[g], g))[:n_check]
    todo = [(g, f) for g, f in search.genomes(run_dir, cfg) if g not in reuse or g in check]
    t_search = 0.0
    if todo:
        import time
        t0 = time.time()
        full = search.genomes
        search.genomes = lambda rd, c: todo                      # search.py searches only these (its report is replaced below)
        try:
            search.main()
        finally:
            search.genomes = full
        t_search = round(time.time() - t0, 1)
    with Stage(run_dir, "search", requires=("pre", "corpus")) as st:
        hits_dir = st.path("hits", "x")[:-2]
        for g in check:
            got = sha256_file(os.path.join(hits_dir, g + ".npz"), use_sidecar=False)
            if got != reuse[g][1]["sha256"]:
                raise SystemExit("reuse check: %s searched again differs from %s (per-genome reuse is not safe)" % (g, os.path.basename(reuse[g][0])))
        for g, (cand, fe) in sorted(reuse.items()):
            if g not in check:
                link(os.path.join(cand, fe["path"]), os.path.join(run_dir, fe["path"]), fe["sha256"])
        allg = search.genomes(run_dir, cfg)
        entries = [dict(id=g, files=[M.file_entry(os.path.join(hits_dir, g + ".npz"), run_dir)]) for g, _ in allg]
        sc = cfg["search"]
        man = M.build("hits", "hits", entries, meta=dict(nearest=sc.get("nearest", 32), max_target_seqs=sc.get("max_target_seqs", 1000)))
        sha = M.write(man, os.path.join(run_dir, "manifests"))
        mp = st.path("hits", "manifest.json")
        write_json(mp, dict(sha256=sha, dataset_id=M.dataset_id(man), **man))
        st.output("hits_manifest", mp)
        by_run = {}
        for g, (cand, _) in reuse.items():
            if g not in check:
                by_run[os.path.basename(cand)] = by_run.get(os.path.basename(cand), 0) + 1
        st.extra.update(genomes=len(allg), reused_per_genome=by_run, searched=len(todo) - len(check), reuse_checked=check, search_wall_s=t_search,
                        reuse_key={k: v for k, v in glob.items()})


if __name__ == "__main__":
    main()
