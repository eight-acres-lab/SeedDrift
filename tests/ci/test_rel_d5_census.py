import numpy as np
import pytest

from ci.stages.rel.d5_census import census


def genome(gid, genus=1, cluster="", group="bacteria", split="heldout"):
    return dict(id=gid, group=group, split=split, ranks=[0, 0, 0, 0, genus, 0], cluster=cluster)


def test_full_open_map_counts_bridges_and_zero_opportunities():
    gs = [genome("A"), genome("B", cluster="K"), genome("C", 2, "K"),
          genome("D", 3), genome("E", 1, "K", "archaea", "train")]
    r = census(gs, ["A", "A", "C", "D"])
    assert r["units"]["A"] == r["units"]["C"] != r["units"]["E"]
    assert r["per_heldout"]["B"]["tables"] == 0
    a = r["all"]
    assert (a["heldout_genomes"], a["heldout_units"], a["planned_tables"]) == (4, 2, 4)
    assert a["same_unit_opportunities"]["pairs"] == dict(eligible_anchors=3, planned_table_upper_bound=3,
                                                        represented_unit_upper_bound=1)
    assert a["ordered_pairs"]["same_unit"] == dict(pairs=6, tier0=None, tier1=None, tier0_or_1=None, near_clone=None)
    assert a["ordered_pairs"]["training"]["pairs"] == 4
    assert r["sealed_distance_coverage"] is None


def test_distance_edges_and_distinct_directed_pairs():
    gs = [genome(x) for x in "ABCD"]
    d = np.full((4, 4), np.inf)
    np.fill_diagonal(d, 0)
    for i, value in enumerate((0.0009, 0.001, 0.05), 1):
        d[0, i] = d[i, 0] = value
    d[1, 2] = d[2, 1] = 0.2
    r = census(gs, ["A", "A"], d)
    assert r["all"]["ordered_pairs"]["same_unit"] == dict(pairs=12, tier0=4, tier1=2, tier0_or_1=6, near_clone=2)
    assert r["all"]["same_unit_opportunities"]["near_clone"]["planned_table_upper_bound"] == 2
    assert r["observed_tables_with_close_target_rows"] is None


@pytest.mark.parametrize("gs, plan", [
    ([genome("A"), genome("A")], ["A"]),
    ([genome("A"), genome("T", split="train")], ["A"]),
    ([genome("A"), genome("T", 2, split="train")], ["A", "T"]),
    ([genome("A"), genome("S", split="test")], ["A"]),
    ([genome("A", 0), {"id": "B", "group": "archaea", "split": "train"}], ["A"]),
    ([genome("A")], ["missing"]),
])
def test_rejects_inconsistent_scope(gs, plan):
    with pytest.raises(ValueError):
        census(gs, plan)


@pytest.mark.parametrize("matrix", [np.ones((2, 2)), np.array([[0., 1.], [2., 0.]]),
                                    np.array([[0., np.nan], [np.nan, 0.]]), np.zeros((1, 1))])
def test_rejects_invalid_distance_matrix(matrix):
    with pytest.raises(ValueError):
        census([genome("A"), genome("B")], ["A"], matrix)


def test_missing_distance_cannot_be_read_as_zero_distance():
    d = np.ma.array(np.zeros((2, 2)), mask=[[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="masked distance"):
        census([genome("A"), genome("B")], ["A"], d)


def test_rank_ids_cannot_create_spurious_units():
    g = genome("B", 0)
    g["ranks"][-1] = "0"
    with pytest.raises(ValueError, match="taxonomy"):
        census([genome("A", 0), g], ["A"])


def test_rejects_species_on_both_sides_even_with_different_genus_ids():
    gs = [genome("A"), genome("B", 2, split="train")]
    for g in gs:
        g["ranks"][-1] = 55
    with pytest.raises(ValueError, match="species crosses"):
        census(gs, ["A"])


def test_many_known_close_opportunities_leave_actual_paired_cell_coverage_unknown():
    gs = [genome(f"g{i}", i // 2 + 1) for i in range(16)]
    d = np.zeros((16, 16))
    r = census(gs, [g["id"] for g in gs] * 3, d)
    assert r["all"]["same_unit_opportunities"]["tier0_or_1"] == dict(
        eligible_anchors=16, planned_table_upper_bound=48, represented_unit_upper_bound=8)
    assert r["paired_cell_coverage"]["tables_with_added_near_rows_and_paired_cells"] is None
    assert r["paired_cell_coverage"]["distinct_units_with_added_near_rows_and_paired_cells"] is None


def test_open_inventory_with_missing_cluster_reports_counts_only():
    from ci.stages.rel.d5_census import open_inventory
    gs = [genome("A"), genome("B"), genome("T", 3, split="train")]
    del gs[1]["cluster"]
    r = open_inventory(gs, ["A", "A"])
    assert r["units"] is None and r["all"] is None
    assert r["distinct_row0"] == 1
    assert r["open_counts_by_group"]["bacteria"] == dict(train=1, heldout=2, distinct_row0=1, planned_tables=2,
                                                         with_rank_metadata=3, with_cluster_metadata=2)
    gs[1]["cluster"] = ""
    assert open_inventory(gs, ["A", "A"])["all"]["heldout_units"] == 1
