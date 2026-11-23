"""Intervals of the evaluator (package W5): a paired bootstrap over the held-out split units inside each group, as the
protocols of the relational line ask (round1 evaluation.intervals: 2,000 replicates, seed 20261123). Numpy only, so it
runs in the CI environment and in the model interpreter, and gives the same replicates in every numpy: the units a
replicate draws come from the evaluator's 64-bit hash (conditions.hash_u01), not from a generator of numpy.

Two readings of the same cells (two arms, or one arm at two nodes) are compared by their difference. The statistic is
the evaluator's aggregate (score.aggregate): inside a group, the difference summed over its cells divided by its cells;
then the mean over the groups that have cells (over a block's groups for the block). A replicate draws, inside every
group, as many of its split units as it has, with replacement, and weights every table by how often its unit was
drawn; the interval is the percentile interval of the replicates.

  paired_bootstrap(diff_sum, cells, group, unit, replicates=2000, seed=20261123, level=0.95, blocks=True)
      diff_sum (T,): per table, bits of reading A minus bits of reading B summed over the table's scored cells (the
      same cells in both); cells (T,): the scored cells of every table; group (T,): its group name; unit (T,): the split
      unit of its examined genome (row 0). Returns {"all": ..., "eukaryotes": ..., "prokaryotes, plasmids, viruses": ...}
      each dict(estimate, low, high, excludes_zero, groups, units, tables, replicates, seed, level).

A table's split unit is that of its examined genome, as stage train_data forms them (inside a group, the genomes linked
by a shared genus or near-relative cluster); the tables carry no unit, so the caller supplies it (an evaluation stage
may take bench/rel/implant.split_units, package W18's copy of train_data's rule, on the genomes' metadata).
"""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import score as SC

VERSION = 1
REPLICATES = 2000
SEED = 20261123
STREAM_BOOTSTRAP = 51


def _index(values):
    """(codes, labels): an integer code per entry, labels in sorted order."""
    labels = sorted({str(v) for v in values})
    at = {v: k for k, v in enumerate(labels)}
    return np.array([at[str(v)] for v in values], dtype=np.int64), labels


def replicate_weights(group_code, unit_code, replicates=REPLICATES, seed=SEED):
    """(replicates, T) float: how often every table's unit is drawn in every replicate, units drawn inside their group
    with replacement, as many draws as the group has units. Replicate 0 is not special: every replicate is a draw."""
    T = len(unit_code)
    w = np.zeros((int(replicates), T))
    for g in np.unique(group_code):
        tabs = np.flatnonzero(group_code == g)
        units = np.unique(unit_code[tabs])
        k = len(units)
        u = cond.hash_u01(int(seed), STREAM_BOOTSTRAP, np.arange(int(replicates))[:, None], int(g) * 1_000_003 + np.arange(k)[None, :])
        drawn = np.minimum((u * k).astype(np.int64), k - 1)                      # (replicates, k) indices into units
        counts = np.zeros((int(replicates), k))
        np.add.at(counts, (np.repeat(np.arange(int(replicates)), k), drawn.ravel()), 1.0)
        pos = np.searchsorted(units, unit_code[tabs])
        w[:, tabs] = counts[:, pos]
    return w


def _statistic(diff_sum, cells, group_code, weights, members):
    """(replicates,) the mean over the groups of `members` of the weighted difference per cell; groups without cells
    in a replicate are left out of its mean."""
    stats = []
    for g in members:
        m = group_code == g
        d = weights[:, m] @ diff_sum[m]
        n = weights[:, m] @ cells[m]
        stats.append(np.where(n > 0, d / np.where(n > 0, n, 1.0), np.nan))
    s = np.array(stats)                                                          # (groups, replicates)
    return np.nanmean(s, axis=0) if len(s) else np.full(weights.shape[0], np.nan)


def paired_bootstrap(diff_sum, cells, group, unit, replicates=REPLICATES, seed=SEED, level=0.95, blocks=True):
    diff_sum, cells = np.asarray(diff_sum, dtype=np.float64), np.asarray(cells, dtype=np.float64)
    group, unit = [str(x) for x in group], [str(x) for x in unit]
    if not (len(diff_sum) == len(cells) == len(group) == len(unit)):
        raise ValueError("diff_sum, cells, group and unit have one entry per table")
    keep = cells > 0
    diff_sum, cells = diff_sum[keep], cells[keep]
    unit = ["%s|%s" % (g, u) for g, u, k in zip(group, unit, keep) if k]          # a unit belongs to its group
    group = [g for g, k in zip(group, keep) if k]
    gcode, glabels = _index(group)
    ucode, _ = _index(unit)
    w = replicate_weights(gcode, ucode, replicates, seed)
    ones = np.ones((1, len(diff_sum)))
    sets = {"all": list(range(len(glabels)))}
    if blocks:
        for b in SC.BLOCKS:
            sets[b] = [k for k, g in enumerate(glabels) if SC.block_of(g) == b]
    out = {}
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    for name, members in sets.items():
        if not members:
            continue
        est = float(_statistic(diff_sum, cells, gcode, ones, members)[0])
        reps = _statistic(diff_sum, cells, gcode, w, members)
        reps = reps[np.isfinite(reps)]
        low, high = (float(np.percentile(reps, lo_q)), float(np.percentile(reps, hi_q))) if len(reps) else (float("nan"), float("nan"))
        m = np.isin(gcode, members)
        out[name] = dict(estimate=est, low=low, high=high, excludes_zero=bool(low > 0 or high < 0), groups=len(members),
                         units=int(len(np.unique(ucode[m]))), tables=int(m.sum()), replicates=int(replicates), seed=int(seed), level=float(level))
    return out


def table_sums(bits, cell_table, n_tables):
    """(sum of bits per table, cells per table) from per-cell bits and the table of every cell."""
    t = np.asarray(cell_table).astype(np.int64)
    return np.bincount(t, weights=np.asarray(bits, dtype=np.float64), minlength=int(n_tables)), np.bincount(t, minlength=int(n_tables)).astype(np.float64)
