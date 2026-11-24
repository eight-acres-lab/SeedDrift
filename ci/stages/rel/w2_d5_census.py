"""D5 phase 1 census using five pinned round-1 inputs. Only row zero enters the plan."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common, d5_census

TABLE_RUN = "20261123T122205-exp-rel-c76dd7cd-rel-dev-w12-v1-k3-i1"
SEARCH_RUN = "20261123T061001-exp-rel-c8f59d82-rel-dev-w12-full-k3-i1"
V1 = Path("data/rel_w12/v1")
SEARCH_ROSTER = Path("data/rel_w12/search/genomes.json")
TABLESET_SHA = "dba2eb115052b8f1e10be9e6b72e391328d21e301332bfe454a1ed0b018b89bd"
INDEX_SHA = "c6a995ac613b89dee95889313f2976ef7c0744da8aefe62932d630425f562445"
SEARCH_SHA = "6cfc8f3eb6ebb89e3c949e6a334255eaea0265c9be3cb0d5a96b2521f631bd37"
ROW0_SHA = "33afc60b16d6ab204b0b385876eab873bc30570fc6607db306649bea305bfbe8"
SHARDS = (("s0000", 2400), ("s0001", 600))
OUT = Path("data/rel_w2/d5_census/report.json")


def _checked_path(path):
    if path.resolve() != path:
        raise ValueError("D5 requires canonical inputs without symbolic links")
    return path


def _json_input(path, digest):
    raw = _checked_path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"D5 checksum mismatch for {path.name}")
    return json.loads(raw)


def _row0_plan(run, doc, index):
    part = index["parts"]["heldout"]
    shards = [(s["name"], s["tables"]) for s in part["shards"]]
    if tuple(shards) != SHARDS or part["tables"] != 3000 or index["rows"] != 32:
        raise ValueError("D5 requires original heldout shards of 2,400 and 600 tables")
    files = {f["path"]: f["sha256"] for f in doc["files"]}
    declared = {}
    plan = []
    roster = index["genomes"]
    for name, n in SHARDS:
        rel = V1 / "heldout" / f"{name}.genome.npy"
        path = _checked_path(run / rel)
        declared[str(rel)] = files[str(rel)]
        rows = np.load(path, mmap_mode="r", allow_pickle=False)
        if rows.shape != (n, 32) or rows.dtype != np.dtype("int32"):
            raise ValueError("D5 unexpected heldout genome metadata shape or dtype")
        row0 = rows[:, 0]
        if np.any(row0 < 0) or np.any(row0 >= len(roster)):
            raise ValueError("D5 target index outside the baseline roster")
        for i in row0:
            if roster[i]["split"] != "heldout":
                raise ValueError("D5 target outside the heldout split")
            plan.append(roster[i]["id"])
    digest = sha256_json(plan)
    if digest != ROW0_SHA:
        raise ValueError("D5 ordered row-zero target checksum mismatch")
    return plan, digest, declared


def inventory(cfg):
    s = cfg["rel_w2_d5_census"]
    if set(s) != {"tables_from"}:
        raise ValueError("D5 requires the pinned tables_from input only")
    run = Path(s["tables_from"])
    if not run.is_absolute() or run.name != TABLE_RUN:
        raise ValueError("D5 requires the round-1 baseline run 122205")

    doc = _json_input(run / V1 / "tableset.json", TABLESET_SHA)
    index = _json_input(run / V1 / "index.json", INDEX_SHA)
    search = _json_input(run.parent / SEARCH_RUN / SEARCH_ROSTER, SEARCH_SHA)
    if (doc["producer"]["run_id"] != TABLE_RUN or doc["search"] != index["search"]
            or index["search"]["run_id"] != SEARCH_RUN or index["search"]["genomes_sha256"] != SEARCH_SHA):
        raise ValueError("D5 table and search provenance disagree")

    keys = ("id", "group", "split", "taxid")
    roster = index["genomes"]
    ordered = [tuple(g[k] for k in keys) for g in roster]
    if ordered != [tuple(g[k] for k in keys) for g in search["genomes"]]:
        raise ValueError("D5 ordered table and search rosters differ")
    if len({g["id"] for g in roster}) != len(roster):
        raise ValueError("D5 duplicate genome in baseline roster")
    if any(g["split"] not in ("train", "heldout", "island") for g in roster):
        raise ValueError("D5 unexpected split in open roster")
    metas = [g for g in roster if g["split"] != "island"]
    counts = Counter(g["split"] for g in metas)
    if counts != {"train": 1093, "heldout": 422} or len({g["group"] for g in metas}) != 12:
        raise ValueError("D5 open roster counts differ from the original corpus")

    plan, plan_digest, declared_shas = _row0_plan(run, doc, index)
    return dict(format="rel-w2-d5-census", version=1, tables_run=TABLE_RUN, search_run=SEARCH_RUN,
                tableset_file_sha256=TABLESET_SHA, index_sha256=INDEX_SHA, search_genomes_sha256=SEARCH_SHA,
                heldout_tables=len(plan), row0_plan_sha256=plan_digest, shard_file_sha256_as_recorded=declared_shas,
                census=d5_census.open_inventory(metas, plan))


def main():
    run = main_args()
    with common.RelStage(run, "rel_w2_d5_census") as stage:
        result = inventory(stage.config)
        path = Path(run) / OUT
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(str(path), result)
        stage.output("d5_census_report", str(path))


def gate(st, cfg, run_dir):
    path = Path(run_dir) / OUT
    reproduced = sha256_json(json.loads(path.read_bytes())) == sha256_json(inventory(cfg))
    require(GateResult("G4_rel_w2_d5_census", "pass" if reproduced else "fail", dict(reproduced=reproduced)), run_dir)
    return sha256_json([sha256_file(str(path), use_sidecar=False)])


def summary(run_dir):
    report = json.loads((Path(run_dir) / OUT).read_bytes())
    n = report["heldout_tables"]
    return dict(json=dict(heldout_tables=n, distinct_row0=report["census"]["distinct_row0"]),
                md=[f"D5 metadata census of {n} original held-out tables."])


if __name__ == "__main__":
    main()
