"""Open-split inventory for D5. Row zero belongs to the original held-out table plan.

Genome-level opportunities are upper bounds: a homolog on the target and a slot in
the final table are both required before an opportunity becomes an evaluation row.
"""
from collections import Counter, defaultdict
from numbers import Integral

import numpy as np

from bench.rel.implant import split_units
from seeddrift.train.rel.api import TIER_EDGES
from seeddrift.train.rel.tables.v0_dist import tier_of


def unit_map(genomes):
    ids = set()
    species_splits = {}
    for g in genomes:
        gid = g["id"]
        if not isinstance(gid, str) or not gid or gid in ids:
            raise ValueError("missing or repeated genome ID")
        ids.add(gid)
        if g["split"] not in ("train", "heldout") or g.get("source", "corpus") != "corpus":
            raise ValueError("D5 accepts only open corpus genomes")
        ranks = g.get("ranks")
        if not isinstance(ranks, (list, tuple)) or len(ranks) != 6 or any(
            isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 0 for rank in ranks
        ):
            raise ValueError("expected six nonnegative taxonomy rank IDs for " + gid)
        if not isinstance(g.get("group"), str) or not g["group"] or not isinstance(g.get("cluster"), str):
            raise ValueError("expected group and cluster fields for " + gid)
        species_key = (g["group"], ranks[5])
        if ranks[5] and species_splits.setdefault(species_key, g["split"]) != g["split"]:
            raise ValueError("species crosses training and held-out sets: " + gid)
    units = split_units(genomes)
    owners = {}
    for g in genomes:
        if owners.setdefault(units[g["id"]], g["split"]) != g["split"]:
            raise ValueError("split unit crosses training and held-out sets: " + units[g["id"]])
    return units


def census(genomes, row0_ids, distance=None):
    """Count ordered anchor-to-context opportunities and unique units of planned tables."""
    units = unit_map(genomes)
    ids = [g["id"] for g in genomes]
    split = np.asarray([g["split"] for g in genomes])
    unit = np.asarray([units[gid] for gid in ids])
    plan = Counter(row0_ids)
    if not plan or set(plan).difference(gid for gid, sp in zip(ids, split) if sp == "heldout"):
        raise ValueError("the reference plan must contain only known held-out anchors")
    if distance is not None:
        if np.ma.is_masked(distance):
            raise ValueError("masked distance has no measured tier")
        distance = np.asarray(distance)
        if distance.shape != (len(ids), len(ids)) or distance.dtype.kind != "f":
            raise ValueError("expected a floating square open-genome distance matrix")
        if (not np.array_equal(distance, distance.T) or (distance < 0).any()
                or np.diag(distance).any() or np.isnan(distance).any()):
            raise ValueError("invalid distances: expected symmetric, nonnegative values and a zero diagonal")

    records = {}
    by_group = defaultdict(dict)
    for i in np.flatnonzero(split == "heldout"):
        gid = ids[i]
        same = unit == unit[i]
        same[i] = False
        other = (split == "heldout") & (unit != unit[i])
        places = {"same_unit": same, "other_heldout": other, "training": split == "train"}
        strata = {}
        tiers = tier_of(distance[i], TIER_EDGES) if distance is not None else None
        for location, mask in places.items():
            counts = {"pairs": int(mask.sum()), "tier0": None, "tier1": None, "near_clone": None}
            if distance is not None:
                counts.update(tier0=int(np.count_nonzero(mask & (tiers == 0))),
                              tier1=int(np.count_nonzero(mask & (tiers == 1))),
                              near_clone=int(np.count_nonzero(mask & (distance[i] < 0.001))))
            counts["tier0_or_1"] = counts["tier0"] + counts["tier1"] if distance is not None else None
            strata[location] = counts
        records[gid] = {"unit": units[gid], "tables": plan[gid], "locations": strata}
        group = genomes[i]["group"]
        unit_type = "cluster" if units[gid].split("|", 1)[1].startswith("c:") else "plain"
        for label in ("all", unit_type):
            by_group[group].setdefault(label, []).append(records[gid])

    summaries = {group: {label: _summarize(rows, distance is not None) for label, rows in subsets.items()}
                 for group, subsets in sorted(by_group.items())}
    return {"units": units, "per_heldout": records, "by_group": summaries,
            "all": _summarize(list(records.values()), distance is not None),
            "distance_edges": list(TIER_EDGES), "near_clone_below": 0.001,
            "sealed_distance_coverage": None, "observed_tables_with_close_target_rows": None,
            "paired_cell_coverage": {"status": "requires_built_tables_and_paired_cells",
                                     "distinct_units_with_added_near_rows_and_paired_cells": None,
                                     "tables_with_added_near_rows_and_paired_cells": None}}


def _summarize(rows, distances_known):
    def opportunity(metric):
        eligible = [r for r in rows if r["locations"]["same_unit"][metric]]
        units = {r["unit"] for r in eligible if r["tables"]}
        return {"eligible_anchors": len(eligible), "planned_table_upper_bound": sum(r["tables"] for r in eligible),
                "represented_unit_upper_bound": len(units)}

    pairs = {}
    for place in ("same_unit", "other_heldout", "training"):
        pairs[place] = {key: sum(r["locations"][place][key] for r in rows) if key == "pairs" or distances_known else None
                        for key in ("pairs", "tier0", "tier1", "tier0_or_1", "near_clone")}
    return {"heldout_genomes": len(rows), "heldout_units": len({r["unit"] for r in rows}),
            "planned_tables": sum(r["tables"] for r in rows), "ordered_pairs": pairs,
            "same_unit_opportunities": {k: opportunity(k) if k == "pairs" or distances_known else None
                                        for k in ("pairs", "tier0", "tier1", "tier0_or_1", "near_clone")}}


def open_inventory(genomes, row0_ids):
    """Count original anchors using the open roster, even without its unit or distance annotations."""
    ids = [g["id"] for g in genomes]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("open roster IDs must be present and unique")
    if any(g["split"] not in ("train", "heldout") or not g.get("group") for g in genomes):
        raise ValueError("unknown group or split in the open roster")
    plan = Counter(row0_ids)
    held = {g["id"] for g in genomes if g["split"] == "heldout"}
    if not plan or plan.keys() - held:
        raise ValueError("the reference plan must contain only known held-out anchors")
    counts = {}
    for g in genomes:
        n = counts.setdefault(g["group"], dict(train=0, heldout=0, distinct_row0=0, planned_tables=0,
                                               with_rank_metadata=0, with_cluster_metadata=0))
        n[g["split"]] += 1
        n["distinct_row0"] += int(g["id"] in plan)
        n["planned_tables"] += plan[g["id"]]
        n["with_rank_metadata"] += int(g.get("ranks") is not None)
        n["with_cluster_metadata"] += int(g.get("cluster") is not None)
    if all(g.get("ranks") is not None and g.get("cluster") is not None for g in genomes):
        result = census(genomes, row0_ids)
    else:
        result = dict(units=None, per_heldout=None, all=None, by_group=None, sealed_distance_coverage=None,
                      observed_tables_with_close_target_rows=None,
                      paired_cell_coverage=dict(status="requires_built_tables_and_paired_cells",
                                                distinct_units_with_added_near_rows_and_paired_cells=None,
                                                tables_with_added_near_rows_and_paired_cells=None))
    result["open_counts_by_group"] = counts
    result["heldout_tables"] = sum(plan.values())
    result["distinct_row0"] = len(plan)
    return result
