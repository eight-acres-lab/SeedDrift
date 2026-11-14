"""Evidence combination: logistic model for P(HGT), softmax for donor ranking."""
from __future__ import annotations

import json
import math

import numpy as np

# Three kinds of candidates, each with its own logistic model:
#   tree  -- a transfer found by DTL reconciliation (topological incongruence)
#   clock -- a speciation node whose two sides are much younger than the species split
#            (molecular-clock anomaly; topologically invisible sister-lineage transfer)
#   gain  -- a family confined to a non-root clade: acquired from a lineage outside the
#            dataset (unsampled donor), versus vertical descent with many losses, a de novo /
#            spurious ORF, or an orthogroup split
FEATURES = ["f_T", "f_opt", "dcost", "near", "depth", "ins", "ins_n", "ctx", "donor_syn", "cotx",
            "me", "recency", "contam", "comp", "alt", "copies"]
TREE_FEATURES = FEATURES
CLOCK_FEATURES = ["clock", "dir", "ins", "ins_n", "ctx", "cotx", "me", "depth", "dcost", "contam"]
# no `depth` here: on simulated (Yule) trees it is collinear with g_alt and would transfer badly
# to unbalanced real trees, where a strain can sit a dozen branches below the root
GAIN_FEATURES = ["g_alt", "g_patchy", "g_size", "site", "site_n", "cotx", "me", "comp", "g_len", "contam",
                 "g_para"]
ALL_FEATURES = list(dict.fromkeys(TREE_FEATURES + CLOCK_FEATURES + GAIN_FEATURES))
LOG_FEATURES = ("ins_n", "site_n", "g_alt", "alt")      # used as log1p

FEATURE_DOC = {
    "clock": "-z of log(gene divergence / expected) across the speciation node, z from the genome-wide "
             "distribution at that node (median/MAD); large = the two lineages exchanged the gene after splitting",
    "dir": "outgroup-only insertion score of this side minus that of the other side: >0 = this side holds "
           "the new (inserted) copy, i.e. is the recipient; 0 = unresolved (no outgroup / no synteny)",
    "f_T": "fraction of reconciliations (tree samples x tie-breaks) with a transfer into the recipient branch",
    "dcost": "log1p(cost without transfers - cost with transfers), ML tree",
    "near": "donor probability mass inside the recipient's sister clade (tree error / ILS risk)",
    "f_opt": "fraction of gene trees where a transfer into the recipient is part of SOME optimal "
             "reconciliation (ties with duplication+loss resolved toward transfer)",
    "depth": "log2(number of genomes under the recipient branch); 0 = single genome",
    "ins": "recipient-side synteny: (empty sites - conserved sites)/informative relatives, in [-1, 1]",
    "ins_n": "number of informative relatives for `ins` (model uses log1p)",
    "ctx": "donor-side context shift: genome-pair synteny background minus best Jaccard with donor copies",
    "donor_syn": "co-transfer: neighbours shared with a donor copy but absent from the ancestral locus, min(n,3)/3",
    "cotx": "fraction of recipient copies whose neighbours were also transferred into the same branch",
    "me": "mobile-element proximity score minus genome background",
    "recency": "-log(gene distance / expected distance); >0 means more similar than the genomes are",
    "contam": "1 if eukaryotic leaf recipient copy sits on a contig with < 2 native genes",
    "alt": "extra DTL cost of the ML gene tree when no transfer may land in the recipient clade "
           "(model uses log1p); ~0-1 = duplication + losses explains the incongruence about as well",
    "copies": "mean copies per genome in the family (hidden-paralogy risk)",
    "comp": "composition outlier: mean robust |z| of GC and GC3 against the genome, times "
            "exp(-age/0.3) of the recipient branch (old acquisitions ameliorate; 0 without --composition)",
    "g_alt": "gain: losses the vertical alternative needs (branches between the root and the clade, "
             "each lacking the family); model uses log1p",
    "g_patchy": "gain: fraction of genomes inside the clade that lack the family",
    "g_size": "gain: log2(copies per genome that carries the family); >0 = expanded after the gain",
    "site": "gain: (clean empty sites - occupied sites)/informative relatives; occupied = other genes "
            "sit between the flanks (replacement or orthogroup split)",
    "site_n": "gain: informative relatives for `site` (model uses log1p)",
    "g_para": "gain: share of gaining genomes that already carry a sister subfamily (a paralog, after "
              "--split-paralogs): origin by duplication rather than acquisition",
    "g_len": "gain: log2(gene length / median gene length of the genome); short = spurious ORF risk",
}

def _plain(features, coef, intercept):
    return {"features": list(features), "mean": [0.0] * len(features), "std": [1.0] * len(features),
            "coef": list(coef), "intercept": intercept}


# Hand-set starting weights; replace with `seeddrift calibrate` output.
DEFAULT_MODEL = {
    "source": "default hand-set weights (uncalibrated)",
    "models": {
        "tree": _plain(TREE_FEATURES, [4.0, 1.0, 0.6, -2.0, 0.0, 1.5, 0.0, 1.5, 1.0, 1.0, 1.5, 0.5, -3.0, 0.5,
                                       1.0, -0.5], -3.5),
        "clock": _plain(CLOCK_FEATURES, [0.5, 1.0, 1.5, 0.0, 1.0, 0.5, 1.0, 0.0, 0.0, -3.0], -4.5),
        "gain": _plain(GAIN_FEATURES, [1.0, -1.0, -0.5, 0.8, 0.0, 2.0, 1.5, 0.7, 0.5, -3.0, -1.5], -2.5),
    },
}

DONOR_GAMMA = {"syn": 3.0, "cotx": 1.5, "recency": 0.5, "me_shared": 0.5}


def load_model(path):
    if not path:
        return DEFAULT_MODEL
    with open(path) as fh:
        return json.load(fh)


def save_model(model, path):
    with open(path, "w") as fh:
        json.dump(model, fh, indent=2)


def _value(row, f):
    v = float(row[f])
    return math.log1p(max(0.0, v)) if f in LOG_FEATURES else v


def matrix(rows, features):
    return np.array([[_value(r, f) for f in features] for r in rows], dtype=float).reshape(len(rows), len(features))


def _predict_one(model, rows):
    if not rows:
        return np.zeros(0)
    X = matrix(rows, model["features"])
    Z = (X - np.asarray(model["mean"])) / np.asarray(model["std"])
    return 1.0 / (1.0 + np.exp(-(Z @ np.asarray(model["coef"]) + model["intercept"])))


def predict(model, rows):
    """P(HGT) per row; rows carry `source` (tree|clock|gain) when the model has sub-models.
    A source missing from the model file (older calibrations) falls back to the default weights."""
    if "models" not in model:          # single model (v0.1/v0.2 files)
        return _predict_one(model, rows)
    out = np.zeros(len(rows))
    subs = dict(DEFAULT_MODEL["models"])
    subs.update(model["models"])
    for src, sub in subs.items():
        idx = [i for i, r in enumerate(rows) if r.get("source", "tree") == src]
        if idx:
            out[idx] = _predict_one(sub, [rows[i] for i in idx])
    return out


def fit_logistic(rows, labels, features=FEATURES, l2=1.0, iters=4000, lr=0.5):
    X = matrix(rows, features)
    y = np.asarray(labels, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    n = len(y)
    p0 = min(max(y.mean(), 1e-3), 1 - 1e-3)
    w = np.zeros(Z.shape[1])
    b = math.log(p0 / (1 - p0))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        w -= lr * (Z.T @ (p - y) / n + l2 * w / n)
        b -= lr * float(np.mean(p - y))
    return {"source": "calibrated logistic regression (n=%d, positives=%d)" % (n, int(y.sum())),
            "features": list(features), "mean": mu.tolist(), "std": sd.tolist(),
            "coef": w.tolist(), "intercept": b}


def tier(p):
    if p >= 0.9:
        return "high"
    if p >= 0.6:
        return "medium"
    if p >= 0.3:
        return "low"
    return "unlikely"


def softmax_donors(scores):
    """scores: {donor: log-score} -> {donor: prob}."""
    if not scores:
        return {}
    m = max(scores.values())
    ex = {d: math.exp(s - m) for d, s in scores.items()}
    z = sum(ex.values())
    return {d: v / z for d, v in ex.items()}


def auc(scores, labels):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))
