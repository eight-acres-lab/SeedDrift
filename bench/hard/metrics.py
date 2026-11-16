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


# ---- quantiles, windows over scored genes, fusion (PROTOCOL.md section 5) ---------------------------------------
def quantile(v):
    """q = (rank - 0.5) / n among the scored genes (finite values), ties at the average rank; nan where unscored."""
    v = np.asarray(v, dtype=float)
    q = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    n = int(ok.sum())
    if n:
        q[ok] = (avg_rank(v[ok]) * n - 0.5) / n
    return q


def window_scored(q, contig, k=5, closed=True, min_scored=3):
    """Mean of q over the scored genes among each gene and its k neighbours on each side (same contig; the window wraps
    only on a closed chromosome). An unscored gene takes the mean of its scored neighbours; a window with fewer than
    min_scored scored genes is unscored (nan)."""
    q = np.asarray(q, dtype=float)
    out = np.full(len(q), np.nan)
    contig = np.asarray(contig)
    for c in np.unique(contig):
        ix = np.flatnonzero(contig == c)
        n = len(ix)
        x = q[ix]
        ok = np.isfinite(x).astype(float)
        val = np.where(ok > 0, x, 0.0)
        if closed and n > 2 * k:
            nb = (np.arange(n)[:, None] + np.arange(-k, k + 1)[None, :]) % n
            s, m = val[nb].sum(1), ok[nb].sum(1)
        else:                                                  # a linear contig, or a contig shorter than one window
            c1, c2 = np.r_[0.0, np.cumsum(val)], np.r_[0.0, np.cumsum(ok)]
            lo, hi = np.maximum(np.arange(n) - k, 0), np.minimum(np.arange(n) + k + 1, n)
            s, m = c1[hi] - c1[lo], c2[hi] - c2[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[ix] = np.where(m >= min_scored, s / m, np.nan)
    return out


def _normal_score(q):
    """Inverse normal CDF (Acklam's rational approximation, relative error below 1.2e-9)."""
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    out = np.empty(len(q))
    lo, hi = q < 0.02425, q > 1 - 0.02425
    mid = ~(lo | hi)
    r = q[mid] - 0.5
    t = r * r
    out[mid] = (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5]) * r / (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1)
    for m, sign in ((lo, 1.0), (hi, -1.0)):
        p = q[m] if sign > 0 else 1 - q[m]
        u = np.sqrt(-2 * np.log(p))
        out[m] = sign * (((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]) / ((((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1)
    return out


FUSIONS = ("max", "mean_1_1", "mean_3_1", "mean_1_3")       # candidates in the order of the protocol; weights are first:second


def fuse(qa, qb, candidate):
    """Fusion of two quantile vectors (nan = unscored). On the genes both score: the candidate statistic, expressed as a
    quantile among those genes. A gene scored by one component takes that component's q; by neither: nan."""
    qa, qb = np.asarray(qa, dtype=float), np.asarray(qb, dtype=float)
    both = np.isfinite(qa) & np.isfinite(qb)
    out = np.where(np.isfinite(qa), qa, qb)
    if both.any():
        if candidate == "max":
            f = np.maximum(qa[both], qb[both])
        else:
            wa, wb = {"mean_1_1": (1, 1), "mean_3_1": (3, 1), "mean_1_3": (1, 3)}[candidate]
            f = (wa * _normal_score(qa[both]) + wb * _normal_score(qb[both])) / (wa + wb)
        out = out.copy()
        out[both] = quantile(f)
    return out


# ---- inference over units (PROTOCOL.md section 8) -----------------------------------------------------------------
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t_interval(d):
    """(mean, lower, upper) of the two-sided 95% t-interval of the mean of d; (mean, nan, nan) with fewer than 2 values."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(d.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    t = _T975[n - 2] if n - 1 <= len(_T975) else 1.96 + 2.4 / (n - 1)
    h = t * float(d.std(ddof=1)) / np.sqrt(n)
    return m, m - h, m + h


def sign_flip_p(d, max_exact=20, draws=200000, seed=0):
    """Two-sided exact sign-flip permutation p of the mean of d (all 2^n sign patterns up to max_exact units, else random
    patterns). The smallest attainable value is 2 / 2^n."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return float("nan")
    obs = abs(d.sum())
    if n <= max_exact:
        signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1)
        tot = np.abs(signs @ d)
        return float(np.mean(tot >= obs - 1e-12))
    rng = np.random.default_rng(seed)
    tot = np.abs(rng.choice([-1.0, 1.0], size=(draws, n)) @ d)
    return float((np.sum(tot >= obs - 1e-12) + 1) / (draws + 1))


def holm(ps):
    """Holm-adjusted p-values (nan stays nan and is not counted in the family)."""
    ps = np.asarray(ps, dtype=float)
    out = np.full(len(ps), np.nan)
    ok = np.flatnonzero(np.isfinite(ps))
    run = 0.0
    for i, j in enumerate(ok[np.argsort(ps[ok])]):
        run = max(run, min(1.0, (len(ok) - i) * ps[j]))
        out[j] = run
    return out
