"""The copying chain with switching as a baseline: chain.switching on the target locus of every table.

The chain runs inside the target block only. With scattered cells hidden it reads the visible cells on both sides of
every hidden cell; with the whole gene hidden it has nothing to read and predicts the plain mixture of the rows under
its prior. (Carrying the source across loci, and the probability of the stretch as a whole, are the next steps.)

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
N_PARAMS = 65 + 1 + 5                      # prior, alpha, rho; three error rates, tau, nosrc
START = dict(tau=0.02, nosrc=0.05)
FACTORS = (0.5, 0.8, 1.25, 2.0)
BOUNDS = dict(eps=(1e-4, 0.9), tau=(1e-5, 1.0), nosrc=(1e-5, 0.999))


def _gamma(view, params, base, mode):
    X, Y, valid = sw.locus(view, cond.TARGET_START, cond.TARGET_STOP)
    q = cl.shrunk(cl.own_counts(view, base), np.asarray(params["prior"]), params["alpha"])
    kernel = cl.copy_kernel(params["chain"]["eps"], params["rho"], base)
    e = sw.emissions(X, Y, q, kernel, base)
    prior = sw.source_prior(Y, params["chain"]["nosrc"], base)
    return sw.source_posterior(e, prior, params["chain"]["tau"], valid, mode), q, kernel


def _in_target(cells):
    s = cells[1]
    if s.size and (s.min() < cond.TARGET_START or s.max() >= cond.TARGET_STOP):
        raise ValueError("the chain scores cells of the target block only")
    return s - cond.TARGET_START


def predict(batch, params, mode="both"):
    """(C, V) for the scored cells of a classical.Batch."""
    t = batch.cells[0]
    c = _in_target(batch.cells)
    gamma, q, kernel = _gamma(batch.view, params, batch.base, mode)
    return sw.predict_cells(gamma[t, c], batch.Y, q[t], kernel, batch.base)


def mean_bits(chunks, params, base=4, mode="both"):
    """Bits per scored cell over prepared chunks (view, cells, truth)."""
    total, count = 0.0, 0
    for view, cells, truth, Y in chunks:
        t = cells[0]
        gamma, q, kernel = _gamma(view, params, base, mode)
        p = sw.truth_probability(gamma[t, _in_target(cells)], Y, q[t, truth - 1], kernel, truth, base)
        total += float(-np.log2(p).sum())
        count += int(truth.shape[0])
    return total / max(count, 1)


def fit(tables, table_ids, mask, seed, params, base=4, rounds=4, tol=1e-4, mode="both"):
    """params: the output of fit.fit for the same mask. Returns the chain's numbers and the trace of the search."""
    chunks = [(bt.view, bt.cells, truth, bt.Y) for _, _, _, bt, truth in fitting.batches(tables, table_ids, mask, seed, base)]
    cur = dict(params, chain=dict(eps=list(params["eps"]), tau=START["tau"], nosrc=START["nosrc"]))
    best = mean_bits(chunks, cur, base, mode)
    trace = [dict(step="start", bits=best)]
    keys = [("eps", 0), ("eps", 1), ("eps", 2), ("tau", None), ("nosrc", None)]
    for r in range(rounds):
        before = best
        for name, k in keys:
            lo, hi = BOUNDS[name]
            for f in FACTORS:
                trial = dict(cur, chain=dict(cur["chain"], eps=list(cur["chain"]["eps"])))
                old = trial["chain"][name] if k is None else trial["chain"][name][k]
                new = float(min(max(old * f, lo), hi))
                if new == old:
                    continue
                if k is None:
                    trial["chain"][name] = new
                else:
                    trial["chain"][name][k] = new
                b = mean_bits(chunks, trial, base, mode)
                if b < best - 1e-9:
                    best, cur = b, trial
        trace.append(dict(step="round %d" % (r + 1), bits=best))
        if before - best < tol:
            break
    out = dict(cur["chain"])
    out["fit"] = dict(mask=mask, seed=int(seed), tables=int(np.asarray(table_ids).shape[0]), cells=int(sum(c[2].shape[0] for c in chunks)),
                      bits=best, trace=trace, mode=str(mode))
    return out
