#!/usr/bin/env python
"""Metrics of the hard benchmark. Every function takes plain arrays; higher score = more foreign.

Ties are scored at their expectation: a group of equal scores is a straight segment of the ROC curve, so a method
cannot gain or lose from the order in which equal scores happen to be listed.

  roc(pos, neg)                 the ROC curve as points (false-positive rate, recall), tie groups as single steps
  recall_at(pos, neg, fpr)      recall where the curve crosses a false-positive rate (linear inside a tie group)
  mean_recall(pos, neg, fpr)    mean recall over false-positive rates from 0 to fpr (partial area / fpr)
  auroc(pos, neg)               area under the whole curve
  threshold_at(neg, fpr)        the score above which at most this share of the negatives lies (for frozen thresholds)
  rank_window(v, contig, k)     mean, over a gene and its k neighbours on each side, of the within-chromosome rank
  region_units(...)             one score per labelled region: the median over its genes
  regions_found(...)            share of positive regions above the (allowed + 1)-th best negative region
  shift_null(...)               the same share after moving all labels together by random offsets along the chromosome
"""
from __future__ import annotations

import numpy as np


def roc(pos, neg):
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    s = np.r_[pos, neg]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    last = np.r_[s[1:] != s[:-1], True]                       # last element of every tie group
    tp, fp = np.cumsum(y)[last], np.cumsum(1 - y)[last]
    return np.r_[0.0, fp / max(len(neg), 1)], np.r_[0.0, tp / max(len(pos), 1)]


def recall_at(pos, neg, fpr):
    x, y = roc(pos, neg)
    return float(np.interp(fpr, x, y))


def mean_recall(pos, neg, fpr):
    x, y = roc(pos, neg)
    cut = np.interp(fpr, x, y)
    keep = x < fpr
    xs, ys = np.r_[x[keep], fpr], np.r_[y[keep], cut]
    return float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2) / fpr)


def auroc(pos, neg):
    x, y = roc(pos, neg)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def threshold_at(neg, fpr):
    """Smallest threshold t such that the share of negatives with a score above t is at most fpr."""
    neg = np.sort(np.asarray(neg, dtype=float))[::-1]
    k = int(np.floor(fpr * len(neg)))
    return float(neg[k]) if k < len(neg) else -np.inf


def avg_rank(v):
    """Ranks from 1 (lowest), ties at their average, divided by the number of values."""
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    r[order] = np.arange(1, len(v) + 1)
    sv = v[order]
    grp = np.cumsum(np.r_[True, sv[1:] != sv[:-1]]) - 1
    r[order] = (np.bincount(grp, weights=r[order]) / np.bincount(grp))[grp]
    return r / len(v)


def rank_window(v, contig, k=5, circular=True):
    """v: one score per gene in chromosome order (missing values already set below every real value).
    Returns the mean within-chromosome rank over each gene and its k neighbours on each side, inside its contig."""
    r = avg_rank(v)
    if k == 0:
        return r
    out = np.empty(len(r))
    contig = np.asarray(contig)
    for c in np.unique(contig):
        ix = np.flatnonzero(contig == c)
        n = len(ix)
        if circular:
            w = min(k, (n - 1) // 2)
            nb = (np.arange(n)[:, None] + np.arange(-w, w + 1)[None, :]) % n
            out[ix] = r[ix][nb].mean(1)
        else:
            c1 = np.r_[0.0, np.cumsum(r[ix])]
            lo, hi = np.maximum(np.arange(n) - k, 0), np.minimum(np.arange(n) + k + 1, n)
            out[ix] = (c1[hi] - c1[lo]) / (hi - lo)
    return out


def region_units(v, region, label):
    """Median score per region. region: region id per gene ('' = none); label: 'island' / 'backbone' / other per gene.
    Returns (scores of island regions, scores of backbone regions)."""
    v, region, label = np.asarray(v, dtype=float), np.asarray(region), np.asarray(label)
    out = {"island": [], "backbone": []}
    for kind in out:
        m = (label == kind) & (region != "")                  # a labelled gene outside every unit belongs to no region
        for r in np.unique(region[m]):
            out[kind].append(float(np.median(v[m & (region == r)])))
    return np.array(out["island"]), np.array(out["backbone"])


def regions_found(pos, neg, allowed=1):
    """(found, total): positive regions strictly above the (allowed + 1)-th highest negative region."""
    if len(neg) <= allowed:
        return int(len(pos)), int(len(pos))
    thr = np.sort(neg)[::-1][allowed]
    return int(np.sum(pos > thr)), int(len(pos))


def region_index(region, label):
    """Gene positions of every labelled region as one padded matrix (-1 = padding), and which rows are island regions."""
    region, label = np.asarray(region), np.asarray(label)
    rows, isl = [], []
    for kind in ("island", "backbone"):
        m = (label == kind) & (region != "")
        for r in np.unique(region[m]):
            rows.append(np.flatnonzero(m & (region == r))); isl.append(kind == "island")
    idx = np.full((len(rows), max((len(r) for r in rows), default=1)), -1, dtype=np.int64)
    for i, r in enumerate(rows):
        idx[i, :len(r)] = r
    return idx, np.array(isl, dtype=bool)


def shift_null(v, region, label, allowed=1, n=1000, seed=0):
    """Share of island regions found (regions_found) when all labels are moved together by random offsets along the
    chromosome: the block structure of the labels is kept, their position is not."""
    v = np.asarray(v, dtype=float)
    idx, isl = region_index(region, label)
    if not isl.any() or isl.all():
        return np.full(n, np.nan)
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        k = int(rng.integers(1, len(v)))
        med = np.nanmedian(np.where(idx >= 0, v[(idx - k) % len(v)], np.nan), axis=1)     # moving the labels by k = reading the scores k genes back
        f, t = regions_found(med[isl], med[~isl], allowed)
        out[i] = f / t
    return out
