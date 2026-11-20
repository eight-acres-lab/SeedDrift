"""The shared test protocol of every score of the truth-labels line on the island chromosomes (2026-11-20; the user
approved the proposal of the objectives review: "我觉得合理，你整理一下告诉一下主会话，让他加上"). Development diagnosis
only: the island set is biased (genus-accessory against genus-core genes), so no score is kept or dropped on it alone.

For every score s (higher = more foreign; NaN = not scored) on the gated labels (island against native genes):
  auroc            AUROC per chromosome (NaN ranked lowest, as the benchmark's evaluator does), mean over units
  coverage         share of island and of native genes the score scores
  residual_auroc   s with the known features regressed out within each chromosome (GC3, codon-usage distance, k-mer
                   distance, log tracked hits, log length; least squares on labelled genes), AUROC of the residual
  combined_gain    leave one unit out: a logistic regression on the known features and on the known features plus s,
                   fitted on the other units, AUROC on the left-out unit; the gain per unit and its mean
  strata_auroc     AUROC within deciles (per chromosome) of the tracked hits and of the length, mean over the strata
                   with both classes
  gain_over_best   per chromosome the AUROC of s minus the best comparator's, mean over units
  spearman         rank correlation with every other score within each chromosome (ties averaged), median over
                   chromosomes; above 0.9 with a known feature, A, B1 or C marks a collapsed score
  matched_auroc    natives reweighted to the islands' joint distribution of length and homolog count (coarsened exact
                   matching per chromosome): a homolog-counting or length score falls to about 0.5 (benchmark v2,
                   family A); matched_positives_kept is the share of islands with a matching native cell
  accessory_auroc  islands against genus-accessory genes outside islands (orthologue share <= 0.5, published "none";
                   before 2026-11-20 evening the published-backbone and "both" genes of that share were let in too):
                   a noisy negative (some are unlabelled transfers or losses), for sensitivity only
"""
from __future__ import annotations

import os
import sys

import numpy as np

KNOWN = ("cmp_gc3", "cmp_codon", "cmp_kmer", "log_hits", "log_length")
# parts of the proposal's protocol this module does not compute (named in every report, not silently missing)
NOT_IMPLEMENTED = ("bootstrap intervals for the metrics other than the AUROC (paired_delta: AUROC and AUROC differences only, for the scores a report asks)",
                   "the placebo gate (relocated and sham-ameliorated natives at AUROC 0.45-0.55): needs the constructed controls of benchmark v2",
                   "a partner-count-matched null for scores that take a maximum over partners",
                   "the hiding-depth curve as a test (the depth readouts are reported as scores; compare them with the default A and B1)")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "bench", "hard"))
import metrics as BM  # noqa: E402


def avg_rank(a):
    """Ranks 0..n-1 with ties at their average rank."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    sv = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def within_quantile(v):
    """A column as its quantile within one chromosome (ties averaged), NaN kept."""
    v = np.asarray(v, float)
    out = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    if ok.sum() > 1:
        out[ok] = avg_rank(v[ok]) / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 0.5
    return out


def auroc(pos, neg):
    """AUROC with ties counted one half; NaN scores ranked below every scored one (as unscored genes)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    lo = np.nanmin(np.concatenate([pos, neg])) if np.isfinite(np.concatenate([pos, neg])).any() else 0.0
    pos, neg = np.where(np.isfinite(pos), pos, lo - 1.0), np.where(np.isfinite(neg), neg, lo - 1.0)
    v = np.concatenate([pos, neg])
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def weighted_auroc(pos, neg, w):
    """AUROC of positives against negatives weighted by w (ties one half; NaN ranked lowest)."""
    pos, neg, w = np.asarray(pos, float), np.asarray(neg, float), np.asarray(w, float)
    keep = w > 0
    neg, w = neg[keep], w[keep]
    if not len(pos) or not len(neg) or w.sum() <= 0:
        return None
    allv = np.concatenate([pos, neg])
    lo = np.nanmin(allv) - 1.0 if np.isfinite(allv).any() else -1.0
    pos, neg = np.where(np.isfinite(pos), pos, lo), np.where(np.isfinite(neg), neg, lo)
    order = np.argsort(neg, kind="mergesort")
    sn, sw = neg[order], w[order]
    cw = np.concatenate([[0.0], np.cumsum(sw)])
    below = cw[np.searchsorted(sn, pos, side="left")]
    equal = cw[np.searchsorted(sn, pos, side="right")] - below
    return float(np.sum(below + 0.5 * equal) / (len(pos) * sw.sum()))


def matching_weights(y, covs, bins=(10, 5)):
    """Coarsened exact matching of the negatives to the positives of one chromosome: each covariate cut at quantiles of
    the labelled genes (len(bins) covariates, bins[k] cells each); a negative weighs n_pos / n_neg of its cell, zero in
    cells without positives; positives in cells without negatives are dropped. Returns (weights for the negatives,
    mask of the positives kept)."""
    lab = y >= 0
    cell = np.zeros(len(y), np.int64)
    for c, nb in zip(covs, bins):
        c = np.asarray(c, float)
        edges = np.unique(np.nanquantile(c[lab & np.isfinite(c)], np.linspace(0, 1, nb + 1)[1:-1])) if (lab & np.isfinite(c)).any() else []
        cell = cell * (nb + 1) + np.digitize(np.where(np.isfinite(c), c, -np.inf), edges)
    w = np.zeros(len(y))
    keep_pos = np.zeros(len(y), bool)
    for k in np.unique(cell[lab]):
        p, n = (cell == k) & (y == 1), (cell == k) & (y == 0)
        if p.any() and n.any():
            w[n] = p.sum() / n.sum()
            keep_pos[p] = True
    return w[y == 0], keep_pos[y == 1]


def known_features(cols):
    """The known-feature matrix of one chromosome from its comparator columns (missing values set to the column mean)."""
    hits = np.log1p(np.maximum(-np.asarray(cols["cmp_hits"], float), 0))
    length = np.log(np.maximum(-np.asarray(cols["cmp_short"], float), 1))
    X = np.stack([np.asarray(cols["cmp_gc3"], float), np.asarray(cols["cmp_codon"], float),
                  np.asarray(cols.get("cmp_kmer", np.zeros(len(hits))), float), hits, length], 1)
    for k in range(X.shape[1]):
        bad = ~np.isfinite(X[:, k])
        if bad.any():
            X[bad, k] = np.nanmean(X[~bad, k]) if (~bad).any() else 0.0
    return X


def residual(s, X, ok):
    """s minus its least-squares fit on X (with intercept) over the genes `ok` where s is finite; NaN elsewhere."""
    out = np.full(len(s), np.nan)
    m = ok & np.isfinite(s)
    if m.sum() <= X.shape[1] + 1:
        return out
    A = np.concatenate([X[m], np.ones((m.sum(), 1))], 1)
    coef, *_ = np.linalg.lstsq(A, s[m], rcond=None)
    out[m] = s[m] - A @ coef
    return out


def logistic_fit(X, y, l2=1e-2, iters=40):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
    w = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], l2 * len(X))
    reg[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + reg * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg + 1e-9)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return mu, sd, w


def logistic_score(fit, X):
    mu, sd, w = fit
    return np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1) @ w


def evaluate(table, units, scores, comparators):
    """table: {chromosome: dict(y=labels 1/0/-1, cols={name: values}, unit=unit)}; scores: the names to assess;
    comparators: the no-model names (for gain_over_best). Returns {score: metrics} and the Spearman matrix."""
    out = {}
    for s in scores:
        per_unit = {}
        cov = [0, 0, 0, 0]
        res_u, strata_u, gain_best_u, best_names, match_u, kept_u, acc_u = {}, {}, {}, {}, {}, {}, {}
        for acc, t in table.items():
            y, v = t["y"], np.asarray(t["cols"][s], float)
            lab = y >= 0
            pos, neg = v[y == 1], v[y == 0]
            a = auroc(pos, neg)
            if a is None:
                continue
            per_unit.setdefault(t["unit"], []).append(a)
            cov[0] += int(np.isfinite(pos).sum()); cov[1] += len(pos); cov[2] += int(np.isfinite(neg).sum()); cov[3] += len(neg)
            X = known_features(t["cols"])
            r = residual(v, X, lab)
            ra = auroc(r[y == 1], r[y == 0])
            if ra is not None:
                res_u.setdefault(t["unit"], []).append(ra)
            for key in ("cmp_hits", "cmp_short"):
                q = np.asarray(t["cols"][key], float)
                edges = np.unique(np.nanquantile(q[lab], np.linspace(0, 1, 11)[1:-1])) if lab.any() else []
                b = np.digitize(q, edges)
                vals = [auroc(v[(b == k) & (y == 1)], v[(b == k) & (y == 0)]) for k in range(len(edges) + 1)]
                vals = [x for x in vals if x is not None]
                if vals:
                    strata_u.setdefault(key, {}).setdefault(t["unit"], []).append(float(np.mean(vals)))
            cands = []
            for c in comparators:
                cv = np.asarray(t["cols"].get(c, []), float)
                if c == s or not len(cv) or not np.isfinite(cv[lab]).any():
                    continue
                ca = auroc(cv[y == 1], cv[y == 0])
                if ca is not None:
                    cands.append((ca, c))
            if cands:
                best, bname = max(cands)
                gain_best_u.setdefault(t["unit"], []).append(a - best)
                best_names[bname] = best_names.get(bname, 0) + 1
            # benchmark v2, family A first views: natives matched to the islands on length and homolog count (the
            # homolog-counting scores can no longer win by construction), and islands against genus-accessory genes
            # outside islands (accessory, not island; a noisy negative: some are unlabelled transfers or losses)
            if "match" in t:
                wn, kp = matching_weights(y, t["match"])
                ma = weighted_auroc(pos[kp], neg, wn)
                if ma is not None:
                    match_u.setdefault(t["unit"], []).append(ma)
                    kept_u.setdefault(t["unit"], []).append(float(kp.mean()) if len(kp) else 0.0)
            if "accessory" in t and t["accessory"].any():
                aa = auroc(pos, v[t["accessory"]])
                if aa is not None:
                    acc_u.setdefault(t["unit"], []).append(aa)
        if not per_unit:
            continue
        umean = lambda d: float(np.mean([np.mean(v) for v in d.values()])) if d else None
        row = dict(auroc=umean(per_unit), auroc_per_unit={u: round(float(np.mean(v)), 4) for u, v in per_unit.items()},
                   coverage_island=cov[0] / max(cov[1], 1), coverage_native=cov[2] / max(cov[3], 1),
                   residual_auroc=umean(res_u), gain_over_best_comparator=umean(gain_best_u), best_comparator_counts=best_names,
                   matched_auroc=umean(match_u), matched_positives_kept=umean(kept_u), accessory_auroc=umean(acc_u))
        for key, name in (("cmp_hits", "strata_auroc_hits_deciles"), ("cmp_short", "strata_auroc_length_deciles")):
            if strata_u.get(key):
                row[name] = umean(strata_u[key])
        row.update(combined_gain(table, units, s))
        out[s] = row
    return out


def _features(t, s=None, flag=True):
    """Known features (and the score) of one chromosome as within-chromosome quantiles; a missing value is 0.5 with a
    missing flag for the score (flag=False: without the flag)."""
    X = known_features(t["cols"])
    F = [within_quantile(X[:, k]) for k in range(X.shape[1])]
    if s is not None:
        q = within_quantile(t["cols"][s])
        miss = ~np.isfinite(q)
        F.append(np.where(miss, 0.5, q))
        if flag:
            F.append(miss.astype(float))
    F = np.stack(F, 1)
    return np.where(np.isfinite(F), F, 0.5)


def combined_gain(table, units, s):
    """Leave one unit out: AUROC on the left-out unit of a logistic regression fitted on the other units' labelled genes,
    with the known features alone and with the score added (within-chromosome quantiles; with and without a missing
    flag); the gain per unit and a t-interval over units."""
    out = {}
    for flag, key in ((True, "combined_gain"), (False, "combined_gain_without_missing_flag")):
        gains = {}
        for u in units:
            tr = [t for t in table.values() if t["unit"] != u]
            te = [t for t in table.values() if t["unit"] == u]
            if not tr or not te:
                continue
            res = []
            for with_s in (False, True):
                X = np.concatenate([_features(t, s if with_s else None, flag)[t["y"] >= 0] for t in tr])
                Y = np.concatenate([t["y"][t["y"] >= 0].astype(float) for t in tr])
                if not (0 < Y.sum() < len(Y)):
                    break
                fit = logistic_fit(X, Y)
                per = []
                for t in te:
                    lab = t["y"] >= 0
                    sc = logistic_score(fit, _features(t, s if with_s else None, flag)[lab])
                    yy = t["y"][lab]
                    a = auroc(sc[yy == 1], sc[yy == 0])
                    if a is not None:
                        per.append(a)
                res.append(float(np.mean(per)) if per else None)
            if len(res) == 2 and None not in res:
                gains[u] = res[1] - res[0]
        if gains:
            g = list(gains.values())
            m, lo, hi = BM.t_interval(g) if len(g) > 1 else (g[0], float("nan"), float("nan"))
            out[key] = float(np.mean(g))
            out[key + "_per_unit"] = {u: round(x, 4) for u, x in gains.items()}
            out[key + "_t95"] = [lo, hi]
    return out


def spearman_matrix(table, names):
    """Rank correlation (ties averaged) of every pair of scores within each chromosome over its labelled genes (NaN
    genes dropped pairwise), the median over chromosomes; the collapse test asks whether a score reproduces another's
    ranking of a genome's genes, so the genomes are not pooled."""
    per = {}
    for t in table.values():
        lab = t["y"] >= 0
        cols = {n: np.asarray(t["cols"][n], float)[lab] for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                m = np.isfinite(cols[a]) & np.isfinite(cols[b])
                if m.sum() > 10 and np.ptp(cols[a][m]) > 0 and np.ptp(cols[b][m]) > 0:
                    per.setdefault("%s ~ %s" % (a, b), []).append(float(np.corrcoef(avg_rank(cols[a][m]), avg_rank(cols[b][m]))[0, 1]))
    return {k: round(float(np.median(v)), 4) for k, v in per.items()}


def collapsed(spear, names, against=("cmp_hits", "cmp_gc3", "cmp_codon", "cmp_kmer")):
    """Scores whose Spearman with a known feature (or any column named in `against`) exceeds 0.9."""
    out = {}
    for n in names:
        for a in against:
            r = spear.get("%s ~ %s" % (n, a), spear.get("%s ~ %s" % (a, n)))
            if r is not None and abs(r) > 0.9 and n != a:
                out.setdefault(n, []).append(a)
    return out


# --- paired intervals (2026-11-20, the codon study; strategy: "判决改用成对差的自助区间") ------------------------------

DECIDE = 0.05      # a difference inside +-0.05 is reported as small: with 3 units the set cannot resolve less


def clusters_of(y, start, block=10):
    """The resampling clusters of one chromosome's labelled genes in genome order (by start): a run of island genes not
    broken by a native gene is one cluster (unlabelled genes do not break a run); native genes are cut into blocks of
    `block` consecutive natives. Returns cluster ids per gene (-1: unlabelled) and whether each cluster is an island."""
    st = np.asarray(start, float)
    order = np.argsort(np.where(np.isfinite(st), st, np.inf), kind="mergesort")
    cid = np.full(len(y), -1)
    flag, cur, cur_y, count = [], -1, None, 0
    for i in order:
        if y[i] < 0:
            continue
        if y[i] != cur_y or (y[i] == 0 and count >= block):
            cur, cur_y, count = cur + 1, y[i], 0
            flag.append(bool(y[i] == 1))
        cid[i] = cur
        count += 1
    return cid, np.array(flag, bool)


def _kernel(v, y, cid, pos_cl, neg_cl):
    """K[p, q] = sum over island genes of island cluster p and native genes of native cluster q of 1[s_i > s_j] + 1/2
    1[s_i = s_j], NaN ranked lowest (as auroc)."""
    v = np.asarray(v, float)
    fin = np.isfinite(v)
    lo = np.nanmin(v[fin]) if fin.any() else 0.0
    v = np.where(fin, v, lo - 1.0)
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    G = (v[ip][:, None] > v[ineg][None, :]).astype(float) + 0.5 * (v[ip][:, None] == v[ineg][None, :])
    Ap = np.zeros((len(ip), len(pos_cl)))
    Ap[np.arange(len(ip)), np.searchsorted(pos_cl, cid[ip])] = 1.0
    An = np.zeros((len(ineg), len(neg_cl)))
    An[np.arange(len(ineg)), np.searchsorted(neg_cl, cid[ineg])] = 1.0
    return Ap.T @ G @ An


def paired_delta(table, units, pairs, n_boot=1000, seed=0, block=10):
    """Bootstrap intervals of AUROC differences, paired (the same draws for both scores). The AUROC is the protocol's
    (per chromosome, mean over the chromosomes of a unit, mean over units). A draw resamples the chromosomes of each unit
    (multinomial) and weights every cluster of a chromosome (clusters_of: islands, blocks of natives) by Poisson(1).
    Strains of one genus share accessory families, which the clusters do not join: the intervals are narrower than the
    truth by an unknown amount. pairs: [(a, b)] with b None for the interval of a alone. Returns one row per pair:
    delta (the observed difference), lo and hi (2.5 and 97.5 percent of the draws), p_le0 (share of draws with
    difference <= 0), verdict (a>b when lo > 0, a<b when hi < 0, else undecided) and small (|delta| < DECIDE)."""
    rng = np.random.default_rng([int(seed), 13])
    chroms = sorted(table)
    prep = {}
    for acc in chroms:
        t = table[acc]
        y = np.asarray(t["y"])
        cid, flag = clusters_of(y, t.get("start", np.arange(len(y))), block)
        cl = np.arange(len(flag))
        pos_cl, neg_cl = cl[flag], cl[~flag]
        if not len(pos_cl) or not len(neg_cl):
            continue
        npos = np.bincount(cid[y == 1], minlength=len(flag))[pos_cl].astype(float)
        nneg = np.bincount(cid[y == 0], minlength=len(flag))[neg_cl].astype(float)
        prep[acc] = dict(y=y, cid=cid, pos_cl=pos_cl, neg_cl=neg_cl, npos=npos, nneg=nneg,
                         wp=rng.poisson(1.0, (n_boot, len(pos_cl))).astype(float), wn=rng.poisson(1.0, (n_boot, len(neg_cl))).astype(float))
    by_unit = {u: [a for a in chroms if a in prep and table[a]["unit"] == u] for u in units}
    by_unit = {u: v for u, v in by_unit.items() if v}
    mult = {u: rng.multinomial(len(v), [1.0 / len(v)] * len(v), size=n_boot).astype(float) for u, v in by_unit.items()}
    cache = {}

    def draws(s):
        if s in cache:
            return cache[s]
        point, boot = [], np.zeros(n_boot)
        for u, accs in by_unit.items():
            pu, bu = [], np.zeros((n_boot, len(accs)))
            for j, acc in enumerate(accs):
                p = prep[acc]
                K = _kernel(table[acc]["cols"][s], p["y"], p["cid"], p["pos_cl"], p["neg_cl"])
                pu.append(K.sum() / (p["npos"].sum() * p["nneg"].sum()))
                den = (p["wp"] @ p["npos"]) * (p["wn"] @ p["nneg"])
                with np.errstate(invalid="ignore", divide="ignore"):
                    bu[:, j] = np.where(den > 0, ((p["wp"] @ K) * p["wn"]).sum(1) / den, np.nan)
            point.append(float(np.mean(pu)))
            m = mult[u] * np.isfinite(bu)
            with np.errstate(invalid="ignore", divide="ignore"):
                boot += np.nansum(m * np.nan_to_num(bu), 1) / m.sum(1)
        cache[s] = (float(np.mean(point)), boot / len(by_unit))
        return cache[s]
    out = []
    for a, b in pairs:
        pa, ba = draws(a)
        if b is None:
            d, bd = pa, ba
        else:
            pb, bb = draws(b)
            d, bd = pa - pb, ba - bb
        bd = bd[np.isfinite(bd)]
        lo, hi = (float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))) if len(bd) else (float("nan"), float("nan"))
        row = dict(a=a, b=b, delta=d, lo=lo, hi=hi, n_boot=int(len(bd)))
        if b is not None:
            row.update(p_le0=float((bd <= 0).mean()) if len(bd) else None, small=bool(abs(d) < DECIDE),
                       verdict="a>b" if lo > 0 else "a<b" if hi < 0 else "undecided")
        out.append(row)
    return out
