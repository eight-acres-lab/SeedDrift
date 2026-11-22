"""Synthetic tables of the relational-model line (numpy; CI environment; owned by the coordinator).

A generator only. No train-side module imports it (tests/ci/test_rel_c0_imports.py), and its tables are never training
data of a real model: they serve the unit tests, round 0's probes of speed and memory, and each architecture's
capability check before it enters a round (a throw-away model, no checkpoint, nothing in the results table).

A table is rows (genomes) x blocks. Blocks, in this order: target (64 to 256 symbols), left and right neighbours (32 to
128 each), and `relations` relation sites (32 each); every row of a block has the block's length (copies with
substitutions only, no insertions or deletions), symbols 1..64 (the codon ids of the arrays). A table has 16 to 32 rows,
all made by copying, so the relations between rows are known:
  - a founder row of random symbols (uniform, independent);
  - each further row copies a parent among the rows made before it, every block, with its own substitution rate drawn
    log-uniformly in [rate_min, rate_max] (default 0.1% to 30%): each position is replaced, with that probability, by a
    different symbol drawn uniformly;
  - the examined row (row 0 of the table as stored) is made last from a source row, so no row copies it; the other rows
    are stored in random order.
The examined row poses one of the two capability tasks:
  locus    it copies every block of its source at its own rate. A check masks the whole target block of the examined
           row, so the source can only be found elsewhere (the relation sites, the neighbours).
  switch   it copies its source A everywhere except the target block from position k on, which it copies from a second
           row B (k uniform in the middle half of the block): a switch of source inside the locus.
Truth (`truth` of each table, for tests and capability checks only, never an input of a model): every row's parent
(-1 for the founder) and rate; the examined row's source, second source and switch point; and the expected divergence
(probability that a position differs) between the examined row and every row along the copy tree.

  generate(index, seed, **params)   one table, a dict: id, task, roles, blocks (list of uint8 arrays rows x length),
                                    truth; reproducible from (seed, index) alone
  tables(n, seed, **params)         the first n tables of a seed
  expected_divergence(parent, rate, a, b)
"""
from __future__ import annotations

import numpy as np

SYMBOLS = 64
DEFAULTS = dict(rows_min=16, rows_max=32, rate_min=0.001, rate_max=0.3, relations=16, target_len=(64, 256), neighbour_len=(32, 128),
                relation_len=32, switch_share=0.5)


def params(**over) -> dict:
    p = dict(DEFAULTS)
    unknown = set(over) - set(p)
    if unknown:
        raise ValueError("unknown synthetic-table parameters %s" % sorted(unknown))
    p.update(over)
    p["target_len"], p["neighbour_len"] = tuple(p["target_len"]), tuple(p["neighbour_len"])
    ok = (2 <= p["rows_min"] <= p["rows_max"] and 0 < p["rate_min"] <= p["rate_max"] < 1 and p["relations"] >= 0 and p["relation_len"] >= 1
          and 4 <= p["target_len"][0] <= p["target_len"][1] and 1 <= p["neighbour_len"][0] <= p["neighbour_len"][1] and 0 <= p["switch_share"] <= 1)
    if not ok:
        raise ValueError("synthetic-table parameters out of range: %r" % p)
    if p["switch_share"] > 0 and p["rows_min"] < 3:
        raise ValueError("the switch task needs at least 3 rows")
    return p


def roles(relations: int) -> list:
    return ["target", "left", "right"] + ["rel%02d" % (i + 1) for i in range(relations)]


def _mutate(rng, x, rate):
    hit = rng.random(x.shape) < rate
    out = x.copy()
    shift = rng.integers(1, SYMBOLS, size=int(hit.sum()))
    out[hit] = ((x[hit].astype(np.int64) - 1 + shift) % SYMBOLS + 1).astype(x.dtype)
    return out


def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def expected_divergence(parent, rate, a, b) -> float:
    """P(a position differs) between rows a and b: along the path through their last common ancestor each edge keeps
    a position with eigenvalue 1 - rate * K / (K - 1) (K = 64 symbols, a substitution draws a different symbol)."""
    def up(x):
        path = [x]
        while parent[path[-1]] >= 0:
            path.append(parent[path[-1]])
        return path
    pa, pb = up(a), up(b)
    common = next(x for x in pa if x in set(pb))
    lam = 1.0
    for path in (pa, pb):
        for x in path[:path.index(common)]:
            lam *= 1.0 - rate[x] * SYMBOLS / (SYMBOLS - 1)
    return (SYMBOLS - 1) / SYMBOLS * (1.0 - lam)


def generate(index: int, seed: int, **over) -> dict:
    p = params(**over)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(index)]))
    n = int(rng.integers(p["rows_min"], p["rows_max"] + 1))
    lengths = [int(rng.integers(p["target_len"][0], p["target_len"][1] + 1))] + \
        [int(rng.integers(p["neighbour_len"][0], p["neighbour_len"][1] + 1)) for _ in range(2)] + [int(p["relation_len"])] * int(p["relations"])
    task = "switch" if rng.random() < p["switch_share"] else "locus"
    # generation order: 0 the founder, 1..n-2 copies, n-1 the examined row
    par, rate = [-1], [0.0]
    made = [[rng.integers(1, SYMBOLS + 1, size=L).astype(np.uint8) for L in lengths]]
    for j in range(1, n - 1):
        q, r = int(rng.integers(0, j)), _loguniform(rng, p["rate_min"], p["rate_max"])
        made.append([_mutate(rng, b, r) for b in made[q]])
        par.append(q)
        rate.append(r)
    src, r0 = int(rng.integers(0, n - 1)), _loguniform(rng, p["rate_min"], p["rate_max"])
    ex = [_mutate(rng, b, r0) for b in made[src]]
    second, cut = -1, -1
    if task == "switch":
        second = int(rng.choice([x for x in range(n - 1) if x != src]))
        L = lengths[0]
        cut = int(rng.integers(L // 4, 3 * L // 4 + 1))
        tail = _mutate(rng, made[second][0], r0)
        ex[0] = np.concatenate([ex[0][:cut], tail[cut:]])
    made.append(ex)
    par.append(src)
    rate.append(r0)
    # stored order: the examined row first, the others shuffled
    order = [n - 1] + [int(x) for x in rng.permutation(n - 1)]
    pos = {g: i for i, g in enumerate(order)}
    blocks = [np.stack([made[g][b] for g in order]) for b in range(len(lengths))]
    parent = [pos[par[g]] if par[g] >= 0 else -1 for g in order]
    rates = [rate[g] for g in order]
    truth = dict(examined=0, parent=parent, rate=rates, source=pos[src], second=pos[second] if second >= 0 else -1, switch_at=cut,
                 divergence=[expected_divergence(parent, rates, 0, i) if i else 0.0 for i in range(n)])
    return dict(id="syn-%d-%06d" % (int(seed), int(index)), task=task, roles=roles(int(p["relations"])), blocks=blocks, truth=truth, params=p)


def tables(n: int, seed: int, **over):
    for i in range(int(n)):
        yield generate(i, seed, **over)
