"""Stage rel_w5_items (package W5; pipeline rel_tables): the evaluation items: which stretch of which held-out genome is
scored, and in which half (seeddrift/train/rel/evalh/items.py). `python -m ci.stages.rel.w5_items RUN_DIR`.

The held-out genomes and their genes come from the training-data arrays of a passed selfsup run (the source the tables
of version 0 are built from), the run and every file checked against its records and manifests as stage rel_w2_v0
checks them (ci/stages/rel/w2_v0.py source). Only genomes of the held-out split are opened, after the sealed check on
all of them; only gene lengths and coordinates are read, and the genus from each genome's metadata. The tables of the
items (context rows, relation loci) are built by package W2 from this file.

Profile block rel_w5_items: source (the run directory of the passed selfsup run), seed, per_group (100), max_codons
(128), min_codons (16).
Output data/rel_w5/items/items.json: format rel-w5-items, version, the rules, the items, the groups short of items,
counts per group and half, the held-out genomes, the source's provenance and the sha256 of the items.
Gate G4_rel_w5_items: every group of the held-out split has its items, both halves in every group, no genus in both
halves, every stretch inside the stored part of its gene, the sha256 matches.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from ci.gates import GateResult, require
from ci.hashing import sha256_file
from ci.stages.common import main_args, write_json
from ci.stages.rel import common, w2_v0
from seeddrift.train.rel.evalh import items as I

STAGE = "rel_w5_items"
OUT = os.path.join("data", "rel_w5", "items")
KEYS = {"source", "seed", "per_group", "max_codons", "min_codons"}


def settings(cfg) -> dict:
    c = dict(cfg.get(STAGE) or {})
    unknown = set(c) - KEYS
    if unknown or not c.get("source"):
        raise SystemExit("%s: the profile names source (and only %s): %s" % (STAGE, sorted(KEYS), sorted(unknown)))
    out = dict(source=str(c["source"]), seed=int(c.get("seed", I.SEED)), per_group=int(c.get("per_group", I.PER_GROUP)),
               max_codons=int(c.get("max_codons", I.MAX_CODONS)), min_codons=int(c.get("min_codons", I.MIN_CODONS)))
    if not (1 <= out["per_group"] <= 10_000 and 1 <= out["min_codons"] <= out["max_codons"] <= I.MAX_CODONS):
        raise SystemExit("%s: per_group 1..10000 and 1 <= min_codons <= max_codons <= %d: %r" % (STAGE, I.MAX_CODONS, out))
    return out


def build(st, cfg, run_dir):
    s = settings(cfg)
    t0 = time.time()
    src = w2_v0.source(s["source"])
    metas = [m for m in src["index"]["genomes"] if m.get("split") == "heldout"]
    groups = sorted({m["group"] for m in src["index"]["genomes"]})
    if not metas:
        raise SystemExit("%s: the source has no held-out genome" % STAGE)
    # the sealed check on every genome that will be opened, before any is
    st.extra["sealed_check"] = common.sealed_check(cfg, [dict(id=m["id"], taxid=int(m["taxid"])) for m in metas], splits={m["id"]: m["split"] for m in metas})
    genomes, lengths = [], {}
    for m in metas:
        rel = os.path.join("data", "train_data", m["id"] + ".npz")
        full = os.path.join(src["run"], rel)
        if src["files"].get(rel) != sha256_file(full, use_sidecar=False):
            raise SystemExit("%s: %s of the source is not the file its manifest names" % (STAGE, rel))
        with np.load(full) as z:
            n = np.diff(z["codon_off"]).astype(np.int64)
            genomes.append(dict(id=m["id"], group=m["group"], genus=I.genus_of(m), lengths=n, start=np.array(z["start"]), end=np.array(z["end"]),
                                strand=np.array(z["strand"]), replicon=np.array(z["replicon"])))
        lengths.update({(m["id"], j): int(x) for j, x in enumerate(n)})
    items, short = I.plan(genomes, s["seed"], s["per_group"], s["max_codons"], s["min_codons"])
    bad = I.problems(items, short, s["per_group"], groups, lengths)
    counts = {g: {h: sum(1 for it in items if it["group"] == g and it["half"] == h) for h in I.HALVES} for g in groups}
    doc = dict(format="rel-w5-items", version=I.VERSION, seed=s["seed"],
               rules=dict(per_group=s["per_group"], max_codons=s["max_codons"], min_codons=s["min_codons"], halves=list(I.HALVES),
                          note="per group the held-out genomes take turns, one unused gene each, the stretch inside the stored part of the "
                               "gene; halves by genus; random choices by a 64-bit hash (seeddrift/train/rel/evalh/items.py)"),
               groups=groups, items=items, short=short, counts=counts,
               genomes=sorted({it["genome"] for it in items}), heldout_genomes=len(metas), genera=len({it["genus"] for it in items}),
               source=src["provenance"], problems=bad, sha256=I.digest(items), seconds=round(time.time() - t0, 1))
    out = st.path("rel_w5", "items", "items.json")
    write_json(out, doc)
    st.output("items", out)
    st.extra.update(items=len(items), short=short, problems=len(bad))
    return doc


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "items.json")
    with open(p) as f:
        doc = json.load(f)
    items = doc.get("items") or []
    bad = I.problems(items, doc.get("short") or {}, int((doc.get("rules") or {}).get("per_group", I.PER_GROUP)), doc.get("groups"))
    if I.digest(items) != doc.get("sha256"):
        bad.append(dict(error="the sha256 of the items is not the one recorded"))
    bad += [dict(e, recorded=True) for e in doc.get("problems") or []]
    require(GateResult("G4_rel_w5_items", "fail" if bad else "pass", dict(problems=bad[:50], items=len(items), counts=doc.get("counts"),
                                                                            genera=doc.get("genera"), sha256=doc.get("sha256"))), run_dir)
    return sha256_file(p, use_sidecar=False)


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "items.json")) as f:
        doc = json.load(f)
    md = ["| group | screen | confirm |", "|---|---|---|"] + ["| %s | %d | %d |" % (g, c["screen"], c["confirm"]) for g, c in sorted(doc["counts"].items())]
    return dict(json=dict(items=len(doc["items"]), sha256=doc["sha256"], genera=doc["genera"], short=doc["short"], counts=doc["counts"]), md=md)


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, STAGE) as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
