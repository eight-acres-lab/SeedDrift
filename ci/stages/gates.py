"""Stage gates: `python -m ci.stages.gates RUN_DIR pre|corpus|train`.

pre      before any data step: G6 clean committed code, G7 environment and resources, G8 CI version, G1 on the spec,
         G2 static (EXCLUDE.tsv parses and only grows, no generator imports in training code)
corpus   on the built corpus: G1 group floors from taxids, G2 exclusions by full lineage on every genome
train    on the exact id list the training loader will read (written by the train stage before its first step)
Each writes gates/<gate>.json; a failure exits non-zero, so every rule that takes the sentinel as input cannot run.
"""
from __future__ import annotations

import json
import os
import sys

import yaml

from ci import ROOT
from ci.gates import GateFailure, find_waiver, require
from ci.gates import g1_kingdom, g2_exclusion, g6_clean, g7_env, g8_version
from ci.kingdom import KingdomMap
from ci.record import atomic_write_json
from ci.stages.common import Stage
from ci.taxonomy import Taxonomy


def main():
    run_dir, phase = sys.argv[1], sys.argv[2]
    with Stage(run_dir, "gates_" + phase) as st:
        cfg = st.config
        try:
            if phase == "pre":
                require(g6_clean.check(ROOT, cfg["git"]["sha"], ci_root=cfg.get("ci_root", ""), require_readonly=cfg.get("require_readonly", False)), run_dir)
                if cfg.get("env_export"):
                    require(g7_env.lock_matches(open(cfg["env_export"]).read()), run_dir)
                res = g7_env.resources(cfg["data_root"], float(cfg.get("disk_gb", 1)), float(cfg.get("disk_reserve_gb", 30)), gpu_mem_min_mb=int(cfg.get("gpu_mem_min_mb", 0)))
                res.gate = "G7_resources"
                require(res, run_dir)
                require(g8_version.check(ROOT), run_dir)
                with open(st.repo(cfg["corpus_spec"])) as f:
                    spec = yaml.safe_load(f)
                g1 = g1_kingdom.static(spec)
                g1.gate = "G1_spec"
                require(g1, run_dir)
                g2 = g2_exclusion.static(ROOT)
                g2.gate = "G2_static"
                require(g2, run_dir)
            elif phase in ("corpus", "train"):
                tax = Taxonomy(st.repo(cfg["taxdump"]))
                with open(os.path.join(run_dir, "data", "corpus", "manifest.json")) as f:
                    man = json.load(f)
                entries = man["entries"]
                if phase == "train":
                    with open(os.path.join(run_dir, "data", "train", "ids.json")) as f:
                        ids = set(json.load(f)["genomes"])
                    entries = [e for e in entries if e["id"] in ids]
                with open(st.repo(cfg["corpus_spec"])) as f:
                    spec = yaml.safe_load(f)
                g1 = g1_kingdom.built(spec, entries, tax, KingdomMap())
                g1.gate = "G1_" + phase
                require(g1, run_dir)
                g2 = g2_exclusion.entries(entries, tax)
                g2.gate = "G2_" + phase
                require(g2, run_dir)
                g5 = find_waiver("G5", "*")
                st.extra["g5"] = "waived until %s: no public-event catalogue yet" % g5["expires"] if g5 else "no waiver"
            else:
                raise SystemExit("unknown gate phase %s" % phase)
        except GateFailure as err:
            st.extra["gate_failure"] = str(err)[:2000]
            raise SystemExit(str(err)[:2000])
        except Exception as err:  # noqa: BLE001 - a gate that cannot run has failed (fail closed)
            from ci.gates import GateResult, run
            run(GateResult("gates_%s_error" % phase, "fail", dict(error="%s: %s" % (type(err).__name__, err))), run_dir)
            st.extra["gate_failure"] = "%s: %s" % (type(err).__name__, err)
            raise
        sentinel = os.path.join(run_dir, "gates", "%s.ok" % phase)
        atomic_write_json(sentinel, {"phase": phase, "passed": True})
        st.output("sentinel", sentinel)


if __name__ == "__main__":
    main()
