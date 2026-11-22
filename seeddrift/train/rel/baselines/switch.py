"""The copying chain with switching as a baseline: chain.switching on the target locus of every table.

The chain runs inside the target block only. Variants (each with its own fitted numbers):
  switch_copy          every hidden cell on its own, from the visible cells on both sides of it. With the whole gene
                       hidden the chain has nothing to read and this is the plain mixture of the rows under its prior.
  switch_copy.past     scattered cells only: from the visible cells to the left of the cell (what a model that reads
                       left to right sees)
  switch_copy.ahead4   scattered cells only: the cells to the left and four columns to the right
  switch_copy.joint    the whole gene only: the hidden stretch as one stretch, cell after cell, each cell given the
                       true residues to its left (the chain rule; one pass). Its bits per codon against switch_copy's
                       are what reading a stretch as a whole is worth to this model.
Carrying the source across loci is the next step.

Fitted on training tables by direct search on what is scored, the bits of the hidden residues: the three error
rates, the switch rate tau and the weight nosrc of copying from no row. rho, prior and alpha come from fit.fit.
"""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel.baselines import classical as cl
from seeddrift.train.rel.baselines import fit as fitting
from seeddrift.train.rel.chain import switching as sw
from seeddrift.train.rel.evalh import conditions as cond

NAME = "switch_copy"
PAST, AHEAD, JOINT = NAME + ".past", NAME + ".ahead4", NAME + ".joint"
MODES = {NAME: "both", PAST: "past", AHEAD: ("ahead", 4), JOINT: "joint"}
NAMES = (NAME, PAST, AHEAD, JOINT)
N_PARAMS = 65 + 1 + 5                      # prior, alpha, rho; three error rates, tau, nosrc
START = dict(tau=0.02, nosrc=0.05)
FACTORS = (0.5, 0.8, 1.25, 2.0)
BOUNDS = dict(eps=(1e-4, 0.9), tau=(1e-5, 1.0), nosrc=(1e-5, 0.999))


def variants(mask):
    """The variants that mean something under a kind of mask."""
    return (NAME, PAST, AHEAD) if mask == cond.SCATTERED else (NAME, JOINT)


def _parts(view, params, chain, base):
    X, Y, valid = sw.locus(view, cond.TARGET_START, cond.TARGET_STOP)
    q = cl.shrunk(cl.own_counts(view, base), np.asarray(params["prior"]), params["alpha"])
    return X, Y, valid, q, cl.copy_kernel(chain["eps"], params["rho"], base)


def _gamma(view, params, chain, base, mode):
    X, Y, valid, q, kernel = _parts(view, params, chain, base)
    e = sw.emissions(X, Y, q, kernel, base)
    return sw.source_posterior(e, sw.source_prior(Y, chain["nosrc"], base), chain["tau"], valid, mode), q, kernel


def _in_target(cells):
    s = cells[1]
    if s.size and (s.min() < cond.TARGET_START or s.max() >= cond.TARGET_STOP):
        raise ValueError("the chain scores cells of the target block only")
    return s - cond.TARGET_START


def predict(batch, params, name=NAME):
    """(C, V) for the scored cells of a classical.Batch, every cell on its own (not for the joint variant)."""
    if MODES[name] == "joint":
        raise ValueError("%s scores a stretch as a whole: joint_bits" % name)
    t = batch.cells[0]
    gamma, q, kernel = _gamma(batch.view, params, params["chains"][name], batch.base, MODES[name])
    return sw.predict_cells(gamma[t, _in_target(batch.cells)], batch.Y, q[t], kernel, batch.base)


def _joint(view, cells, truth, params, chain, base):
    """Bits of every scored cell given the true residues to its left; every cell of the stretch must be scored."""
    X, Y, valid, q, kernel = _parts(view, params, chain, base)
    t, c = cells[0], _in_target(cells)
    scored = np.zeros(X.shape[0], dtype=bool)
    scored[t] = True                                           # a table without the condition has no scored cell
    if ((X >= 1) & (X <= cl.sizes(base)[1]))[scored].any():
        raise ValueError("the joint reading is for a stretch that is hidden as a whole")
    X = X.copy()
    X[t, c] = truth
    return -sw.log_joint(X, Y, q, kernel, chain["tau"], chain["nosrc"], base, per_column=True)[t, c] / np.log(2.0)


def joint_bits(batch, truth, params, name=JOINT):
    """(C,) bits of the scored cells of a classical.Batch read as one stretch (the whole gene hidden)."""
    return _joint(batch.view, batch.cells, truth, params, params["chains"][name], batch.base)


def mean_bits(chunks, params, chain, name, base=4):
    """Bits per scored cell over prepared chunks (view, cells, truth, Y) under a variant."""
    total, count = 0.0, 0
    for view, cells, truth, Y in chunks:
        if MODES[name] == "joint":
            total += float(_joint(view, cells, truth, params, chain, base).sum())
        else:
            t = cells[0]
            gamma, q, kernel = _gamma(view, params, chain, base, MODES[name])
            p = sw.truth_probability(gamma[t, _in_target(cells)], Y, q[t, truth - 1], kernel, truth, base)
            total += float(-np.log2(p).sum())
        count += int(truth.shape[0])
    return total / max(count, 1)


def search(chunks, params, name, base=4, rounds=6, tol=1e-4):
    """The chain's numbers for one variant: coordinate search from the copy_nearest rates, each number scaled by
    FACTORS in turn while the bits fall."""
    cur = dict(eps=list(params["eps"]), tau=START["tau"], nosrc=START["nosrc"])
    best = mean_bits(chunks, params, cur, name, base)
    trace = [dict(step="start", bits=best)]
    keys = [("eps", 0), ("eps", 1), ("eps", 2), ("tau", None), ("nosrc", None)]
    for r in range(rounds):
        before = best
        for key, k in keys:
            lo, hi = BOUNDS[key]
            for f in FACTORS:
                trial = dict(cur, eps=list(cur["eps"]))
                old = trial[key] if k is None else trial[key][k]
                new = float(min(max(old * f, lo), hi))
                if new == old:
                    continue
                if k is None:
                    trial[key] = new
                else:
                    trial[key][k] = new
                b = mean_bits(chunks, params, trial, name, base)
                if b < best - 1e-9:
                    best, cur = b, trial
        trace.append(dict(step="round %d" % (r + 1), bits=best))
        if before - best < tol:
            break
    return dict(cur, fit=dict(bits=best, trace=trace, mode=str(MODES[name])))


def fit(tables, table_ids, mask, seed, params, base=4):
    """{variant: its numbers} for the variants of a kind of mask. params: the output of fit.fit for the same mask."""
    chunks = [(bt.view, bt.cells, truth, bt.Y) for _, _, _, bt, truth in fitting.batches(tables, table_ids, mask, seed, base)]
    cells = int(sum(c[2].shape[0] for c in chunks))
    out = {}
    for name in variants(mask):
        out[name] = search(chunks, params, name, base)
        out[name]["fit"].update(mask=mask, seed=int(seed), tables=int(np.asarray(table_ids).shape[0]), cells=cells)
    return out
