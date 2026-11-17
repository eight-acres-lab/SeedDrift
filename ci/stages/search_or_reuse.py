"""Stage search, or its outputs taken over from an earlier run with exactly the same inputs (profile `search.reuse_from`).

The search is deterministic: the same corpus, evaluation inputs, reference, search settings, search code and DIAMOND
version give the same hit files. A run named in `reuse_from` is used only when all six are equal and its search stage
passed; each hit file is linked and checked against its sha256 in that run's manifest. The stage report records the run
it came from and the key. Otherwise (no candidate, or any difference) ci/stages/search.py runs as usual; this wrapper
lives in its own module so that search.py, whose checksum is part of the key, stays unchanged.
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
    sc = {k: v for k, v in (search_cfg or {}).items() if k != "reuse_from"}
    return dict(corpus=_sha(run, "corpus"), eval_inputs=_sha(run, "eval_inputs"), reference=_sha(run, "reference"), search=sha256_json(sc),
                code=sha256_file(code_path, use_sidecar=False), diamond=(rec.get("env_snapshot") or {}).get("diamond_version.txt"))


def candidate_ok(cand):
    p = os.path.join(cand, "stages", "search.json")
    if not os.path.exists(p) or not os.path.exists(os.path.join(cand, "data", "hits", "manifest.json")):
        return False
    with open(p) as f:
        return json.load(f).get("status") == "passed"


def main():
    run_dir = main_args()
    with open(os.path.join(run_dir, "config.resolved.yaml")) as f:
        cfg = yaml.safe_load(f)
    ours = key(run_dir, cfg["search"], SEARCH_CODE)
    for c in (cfg["search"].get("reuse_from") or []):
        cand = c if os.path.isabs(c) else os.path.join(ROOT, c)
        if not candidate_ok(cand):
            continue
        with open(os.path.join(cand, "config.resolved.yaml")) as f:
            ccfg = yaml.safe_load(f)
        code = os.path.join(os.path.dirname(os.path.dirname(cand)), "wt", os.path.basename(cand), "ci", "stages", "search.py")
        if not os.path.exists(code) or key(cand, ccfg["search"], code) != ours:
            continue
        with Stage(run_dir, "search", requires=("pre", "corpus")) as st:
            with open(os.path.join(cand, "data", "hits", "manifest.json")) as f:
                man = json.load(f)
            for e in man["entries"]:
                for fe in e["files"]:
                    src, dst = os.path.join(cand, fe["path"]), os.path.join(run_dir, fe["path"])
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.link(src, dst)
                    if sha256_file(dst, use_sidecar=False) != fe["sha256"]:
                        raise SystemExit("reused hit file %s does not match its manifest" % fe["path"])
            from ci import manifest as M
            M.write({k: man[k] for k in ("name", "kind", "entries", "meta")}, os.path.join(run_dir, "manifests"))   # content-addressed copy
            mp = st.path("hits", "manifest.json")
            write_json(mp, man)
            st.output("hits_manifest", mp)
            st.extra.update(reused_from=os.path.basename(cand), reuse_key=ours, genomes=len(man["entries"]))
        return
    search.main()


if __name__ == "__main__":
    main()
