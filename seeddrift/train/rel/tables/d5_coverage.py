"""Build-time observed coverage for one paired D5 phase-1 evaluation condition."""

from collections import Counter
from dataclasses import dataclass, field
from numbers import Integral

from ci.hashing import sha256_json
from seeddrift.train.rel import api
from seeddrift.train.rel.evalh.score import BLOCKS, FINE_PREFIX, block_of

CONDITIONS = ("scattered", "suffix", "whole_gene")
IDENTITIES = tuple(
    FINE_PREFIX + f"{n / 10:.1f}-{(n + 1) / 10:.1f}" for n in range(10)
) + (
    FINE_PREFIX + "short_fragments_only",
    FINE_PREFIX + "none",
)
TIERS = tuple(range(api.TIER_NO_SHARED + 1))
STRATA = {
    "nearest_tier": (*TIERS, "none"),
    "identity_v5": IDENTITIES,
    "unit_type": ("cluster", "plain"),
}


@dataclass(frozen=True)
class Table:
    """Metadata from a materialized table and its strict mask for the specified condition.

    table_id is the global zero-based ID in the original plan. cells are scored
    row-zero positions relative to this target block. context holds actual
    other-genome rows in table order as (genome ID, pinned distance tier),
    including each paralog row and excluding padding and self copies. self_copies lists their original locus IDs in table order.
    identity_v5 is the close table's evaluator-v5 nearest-row stratum.
    """

    condition: str
    table_id: int
    row0: str
    locus: int
    self_copies: tuple
    cells: tuple
    context: tuple
    identity_v5: str = ""


def _is_int(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _index(tables, expected):
    indexed = {}
    for t in tables:
        if not _is_int(t.table_id) or t.table_id in indexed:
            raise ValueError(f"invalid or duplicate table ID {t.table_id}")
        indexed[t.table_id] = t
    if indexed.keys() != expected:
        raise ValueError("table roster differs from the original plan")
    return indexed


def _cells(t):
    ids = set(t.cells)
    if len(ids) != len(t.cells) or any(
        not _is_int(c) or not 0 <= c < api.LEN_TARGET for c in ids
    ):
        raise ValueError(
            f"duplicate or invalid scored target cell in table {t.table_id}"
        )
    return ids


def _rows(t, units):
    counts = Counter()
    tiers = {}
    rows = []
    for gid, tier in t.context:
        counts[gid] += 1
        if gid not in units or gid == t.row0 or counts[gid] > api.MAX_ROWS_PER_GENOME:
            raise ValueError(
                f"invalid materialized context roster in table {t.table_id}"
            )
        if (
            not _is_int(tier)
            or tier not in TIERS
            or tiers.setdefault(gid, tier) != tier
        ):
            raise ValueError(
                f"invalid or inconsistent context tier in table {t.table_id}"
            )
        rows.append((gid, tier))
    return rows


@dataclass
class _Counts:
    tables: int = 0
    scored_cells: int = 0
    zero_target_tables: int = 0
    tables_with_added_near_rows: int = 0
    qualifying_tables: int = 0
    qualifying_cells: int = 0
    groups: set = field(default_factory=set)
    units: set = field(default_factory=set)

    def add(self, cells, added_near, group, unit):
        self.tables += 1
        self.scored_cells += cells
        self.zero_target_tables += int(cells == 0)
        self.tables_with_added_near_rows += int(added_near)
        if cells and added_near:
            self.qualifying_tables += 1
            self.qualifying_cells += cells
            self.groups.add(group)
            self.units.add(unit)

    def report(self):
        return {
            "paired_tables": self.tables,
            "paired_cells": self.scored_cells,
            "zero_target_tables": self.zero_target_tables,
            "tables_with_added_near_rows": self.tables_with_added_near_rows,
            "qualifying_tables": self.qualifying_tables,
            "qualifying_cells": self.qualifying_cells,
            "eligible_groups": sorted(self.groups),
            "distinct_units": len(self.units),
            "meets_coverage_minimum": self.qualifying_tables >= 30
            and len(self.units) >= 8,
        }


def coverage(step0, condition, original, close):
    """Count observed rows and paired cells per block and separate breakdown.

    Supply the pinned step-0 report plus one Table per original table ID on
    each side. An invalid pair raises and blocks the reading for this condition.
    This metadata check precedes the strict-render, no-relatives and training
    invariance checks required for a scored D5 result.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown D5 condition {condition}")
    n = step0["heldout_tables"]
    if not n:
        raise ValueError("empty original table plan")
    expected = set(range(n))
    old, new = _index(original, expected), _index(close, expected)
    plan = [old[i].row0 for i in range(n)]
    if sha256_json(plan) != step0["row0_plan_sha256"]:
        raise ValueError("row-zero plan differs from step 0")
    units = step0["census"]["units"]
    if units is None:
        raise ValueError("D5 coverage requires a complete step-0 unit map")
    held = step0["census"]["per_heldout"]
    expected_counts = {g: r["tables"] for g, r in held.items() if r["tables"]}
    if Counter(plan) != expected_counts:
        raise ValueError("planned anchors differ from the step-0 held-out roster")

    bins = {
        (block, name, label): _Counts()
        for block in BLOCKS
        for name, labels in STRATA.items()
        for label in labels
    }
    for i in range(n):
        a, b = old[i], new[i]
        if a.condition != condition or b.condition != condition:
            raise ValueError(f"condition differs in table {i}")
        if not _is_int(a.locus) or a.locus < 0 or not _is_int(b.locus):
            raise ValueError(f"invalid target locus in table {i}")
        if a.row0 != b.row0 or a.locus != b.locus or a.self_copies != b.self_copies:
            raise ValueError(f"row zero, locus or self copies changed in table {i}")
        cell_ids = _cells(a)
        if cell_ids != _cells(b):
            raise ValueError(f"scored cells differ in table {i}")
        before, after = _rows(a, units), _rows(b, units)
        if any(g in held for g, _ in before):
            raise ValueError(f"original other-genome row is held out in table {i}")
        unit = units[a.row0]
        added = [(g, tier) for g, tier in after if g in held]
        if any(units[g] != unit for g, _ in added):
            raise ValueError(f"new context outside the held-out unit in table {i}")
        retained = [(g, tier) for g, tier in after if g not in held]
        dropped = len(before) - len(retained)
        if retained != before[: len(retained)] or dropped > len(added):
            raise ValueError(f"original context changed unexpectedly in table {i}")
        if b.identity_v5 not in IDENTITIES:
            raise ValueError(
                f"missing or invalid evaluator-v5 identity stratum in table {i}"
            )
        added_near = any(tier in (0, 1) for _, tier in added)
        group, key = unit.split("|", 1)
        labels = {
            "nearest_tier": min((tier for _, tier in after), default="none"),
            "identity_v5": b.identity_v5,
            "unit_type": "cluster" if key.startswith("c:") else "plain",
        }
        for name, label in labels.items():
            bins[block_of(group), name, label].add(
                len(cell_ids), added_near, group, unit
            )
    strata = [
        dict(block=block, breakdown=name, stratum=label, **counts.report())
        for (block, name, label), counts in bins.items()
    ]
    return {
        "condition": condition,
        "strata": strata,
        "any_stratum_meets_minimum": any(s["meets_coverage_minimum"] for s in strata),
    }
