"""The evaluator's paired bootstrap over split units inside each group (seeddrift/train/rel/evalh/intervals.py)."""
import numpy as np
import pytest

from seeddrift.train.rel.evalh import intervals as IV
from seeddrift.train.rel.evalh import score as SC


def _data(seed=0, T=60):
    rng = np.random.default_rng(seed)
    groups = np.array(["bacteria"] * 24 + ["eukaryota.fungi"] * 21 + ["viruses.phage"] * 15)[:T]
    units = np.array(["u%d" % (k // 3) for k in range(T)])                      # three tables per unit, units inside a group
    cells = rng.integers(5, 40, T).astype(float)
    return groups, units, cells, rng


def test_the_estimate_is_the_evaluators_aggregate():
    groups, units, cells, rng = _data()
    per_cell = rng.normal(0.1, 1.0, int(cells.sum()))
    cell_table = np.repeat(np.arange(len(cells)), cells.astype(int))
    diff_sum, n = IV.table_sums(per_cell, cell_table, len(cells))
    out = IV.paired_bootstrap(diff_sum, n, groups, units, replicates=200)
    names = {k: g for k, g in enumerate(sorted(set(groups)))}
    code = np.array([sorted(set(groups)).index(g) for g in groups])
    agg = {(r["group"], r["stratum"]): r["bits"] for r in SC.aggregate(per_cell, cell_table, code, [], len(names), names)}
    assert out["all"]["estimate"] == pytest.approx(agg[("all", "all")])
    assert out["eukaryotes"]["groups"] == 1 and out["prokaryotes, plasmids, viruses"]["groups"] == 2
    assert out["all"]["low"] <= out["all"]["estimate"] <= out["all"]["high"]


def test_zero_and_constant_differences():
    groups, units, cells, _ = _data()
    z = IV.paired_bootstrap(np.zeros(len(cells)), cells, groups, units, replicates=100)["all"]
    assert z["estimate"] == 0.0 and z["low"] == 0.0 and z["high"] == 0.0 and not z["excludes_zero"]
    c = IV.paired_bootstrap(0.05 * cells, cells, groups, units, replicates=100)["all"]
    assert c["low"] == pytest.approx(0.05) and c["high"] == pytest.approx(0.05) and c["excludes_zero"]


def test_units_move_together_inside_their_group():
    groups, units, cells, _ = _data()
    gcode = np.array([sorted(set(groups)).index(g) for g in groups])
    ucode = np.array([sorted(set(units)).index(u) for u in units])
    w = IV.replicate_weights(gcode, ucode, replicates=300, seed=7)
    for u in np.unique(ucode):
        tabs = np.flatnonzero(ucode == u)
        assert (w[:, tabs] == w[:, tabs[:1]]).all()                               # a unit's tables share their weight
    for g in np.unique(gcode):
        tabs = np.flatnonzero(gcode == g)
        n_units = len(np.unique(ucode[tabs]))
        first = [np.flatnonzero(ucode == u)[0] for u in np.unique(ucode[tabs])]
        assert (w[:, first].sum(axis=1) == n_units).all()                         # as many draws as units, inside the group
    assert np.array_equal(w, IV.replicate_weights(gcode, ucode, replicates=300, seed=7))
    assert not np.array_equal(w, IV.replicate_weights(gcode, ucode, replicates=300, seed=8))


def test_a_unit_effect_widens_the_interval_and_the_replicates_are_pinned():
    groups, units, cells, rng = _data(1)
    unit_effect = {u: rng.normal(0, 0.5) for u in set(units)}
    diff = np.array([0.1 + unit_effect[u] for u in units]) * cells
    wide = IV.paired_bootstrap(diff, cells, groups, units, replicates=500)["all"]
    alone = IV.paired_bootstrap(diff, cells, groups, ["t%d" % k for k in range(len(cells))], replicates=500)["all"]
    assert (wide["high"] - wide["low"]) > (alone["high"] - alone["low"])         # tables of a unit are not independent
    again = IV.paired_bootstrap(diff, cells, groups, units, replicates=500)["all"]
    assert again == wide and wide["replicates"] == 500 and wide["seed"] == IV.SEED and wide["units"] == 20
    with pytest.raises(ValueError):
        IV.paired_bootstrap(diff[:-1], cells, groups, units)
