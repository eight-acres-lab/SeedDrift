from dataclasses import replace

import pytest

from ci.hashing import sha256_json
from ci.stages.rel.d5_census import census
from seeddrift.train.rel.evalh.score import BLOCKS
from seeddrift.train.rel.tables.d5_coverage import IDENTITIES, Table, coverage


def paired(count=30, n_units=8, mixed_blocks=False, condition="scattered"):
    genomes = []
    before, after = [], []
    for i in range(n_units):
        group = "eukaryota.fungi" if mixed_blocks and i % 2 else "bacteria"
        for suffix in ("target", "relative"):
            genomes.append(
                {
                    "id": f"{i}.{suffix}",
                    "group": group,
                    "split": "heldout",
                    "ranks": [0, 0, 0, 0, i + 1, 0],
                    "cluster": "",
                }
            )
    genomes.append(
        {
            "id": "trained",
            "group": "bacteria",
            "split": "train",
            "ranks": [0, 0, 0, 0, 99, 0],
            "cluster": "",
        }
    )
    for i in range(count):
        unit = i % n_units
        t = Table(condition, i, f"{unit}.target", i, (51,), (1, 7), (("trained", 1),))
        before.append(t)
        after.append(
            replace(
                t,
                cells=(7, 1),
                context=t.context + ((f"{unit}.relative", 0),),
                identity_v5=IDENTITIES[6],
            )
        )
    plan = [t.row0 for t in before]
    step0 = {
        "heldout_tables": count,
        "row0_plan_sha256": sha256_json(plan),
        "census": census(genomes, plan),
    }
    return step0, before, after


def tier_report(report):
    return next(
        s
        for s in report["strata"]
        if (s["block"], s["breakdown"], s["stratum"]) == (BLOCKS[1], "nearest_tier", 0)
    )


@pytest.mark.parametrize(
    ("n", "units", "meets"), [(30, 8, True), (29, 8, False), (30, 7, False)]
)
def test_minima_are_independent(n, units, meets):
    step0, before, after = paired(n, units)
    result = coverage(step0, "scattered", reversed(before), after)
    row = tier_report(result)
    assert (
        row["distinct_units"] == units
        and row["qualifying_tables"] == n
        and row["qualifying_cells"] == 2 * n
    )
    assert (
        row["meets_coverage_minimum"] is meets
        and result["any_stratum_meets_minimum"] is meets
    )
    assert row["eligible_groups"] == ["bacteria"]


def test_blocks_and_breakdowns_do_not_pool():
    step0, before, after = paired(32, 8, mixed_blocks=True)
    result = coverage(step0, "scattered", before, after)
    assert not result["any_stratum_meets_minimum"]
    for block in BLOCKS:
        for name in ("identity_v5", "nearest_tier", "unit_type"):
            selected = [
                s
                for s in result["strata"]
                if s["breakdown"] == name and s["block"] == block
            ]
            assert sum(s["paired_tables"] for s in selected) == 16
            assert sum(s["qualifying_tables"] for s in selected) == 16


def test_added_rows_and_nonempty_matching_cells_are_both_required():
    step0, before, after = paired(32)
    # One unchanged table already has a near training row; another only receives a far relative.
    after[0] = replace(after[0], context=before[0].context)
    after[1] = replace(after[1], context=before[1].context + (("1.relative", 2),))
    # A third table has an added near row but neither side has scored targets.
    before[2] = replace(before[2], cells=())
    after[2] = replace(after[2], cells=())
    result = coverage(step0, "scattered", before, after)
    row = next(
        s
        for s in result["strata"]
        if s["breakdown"] == "identity_v5" and s["paired_tables"]
    )
    assert (
        row["zero_target_tables"],
        row["tables_with_added_near_rows"],
        row["qualifying_tables"],
    ) == (1, 30, 29)
    assert not result["any_stratum_meets_minimum"]


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"row0": "1.target"}, "row zero"),
        ({"locus": 10}, "locus"),
        ({"self_copies": (7,)}, "self copies"),
        ({"cells": (1, 9)}, "scored cells"),
        ({"cells": (1, 1)}, "duplicate or invalid"),
        ({"condition": "suffix"}, "condition"),
        ({"context": (("1.relative", 0),)}, "outside the held-out unit"),
        ({"context": (("0.relative", 0),) * 3}, "context roster"),
        (
            {"context": (("0.relative", 0), ("0.relative", 1))},
            "inconsistent context tier",
        ),
        ({"context": (("trained", 0),)}, "context changed"),
        ({"identity_v5": "not a bin"}, "identity stratum"),
    ],
)
def test_bad_pairs_fail_closed(change, error):
    step0, before, after = paired()
    after[0] = replace(after[0], **change)
    with pytest.raises(ValueError, match=error):
        coverage(step0, "scattered", before, after)


def test_missing_duplicate_and_changed_plan_fail_closed():
    step0, before, after = paired()
    for invalid in (after[1:], after + after[:1]):
        with pytest.raises(ValueError, match="table (ID|roster)"):
            coverage(step0, "scattered", before, invalid)
    with pytest.raises(ValueError, match="row-zero plan"):
        coverage(dict(step0, row0_plan_sha256="wrong"), "scattered", before, after)


def test_short_fragments_and_empty_strata_remain_visible():
    step0, before, after = paired(condition="suffix")
    after[0] = replace(after[0], identity_v5=IDENTITIES[-2])
    result = coverage(step0, "suffix", before, after)
    observed = {
        s["stratum"]: s["paired_tables"]
        for s in result["strata"]
        if s["block"] == BLOCKS[1] and s["breakdown"] == "identity_v5"
    }
    assert observed[IDENTITIES[-2]] == 1 and observed[IDENTITIES[-1]] == 0
    assert observed[IDENTITIES[6]] == 29


def test_two_paralog_rows_count_once_and_displace_only_a_suffix():
    step0, before, after = paired(count=1, n_units=1)
    step0["census"]["units"]["second_train"] = "bacteria|g:100"
    train, second, near = ("trained", 1), ("second_train", 2), ("0.relative", 0)
    before[0] = replace(before[0], context=(train, second, train))
    after[0] = replace(after[0], context=(near, train, near, second))
    assert (
        tier_report(coverage(step0, "scattered", before, after))["qualifying_tables"]
        == 1
    )
    after[0] = replace(after[0], context=(train, near, second, train))
    assert (
        tier_report(coverage(step0, "scattered", before, after))["qualifying_tables"]
        == 1
    )

    for invalid in (
        (near, second, train),
        (near, train, train),
        (train, second),
        (second, train, train),
        (near, train),
        (near, train, train, train),
    ):
        after[0] = replace(after[0], context=invalid)
        with pytest.raises(ValueError, match="original context changed|context roster"):
            coverage(step0, "scattered", before, after)


def test_partial_inventory_cannot_supply_coverage_units():
    step0, before, after = paired()
    step0["census"]["units"] = None
    with pytest.raises(ValueError, match="complete step-0 unit map"):
        coverage(step0, "scattered", before, after)
