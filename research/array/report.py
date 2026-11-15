#!/usr/bin/env python
"""Write RESULTS.md for the array model from the files of the runs (nothing is typed in by hand).

Trials are identical for every run (fixed seeds), so arms are compared trial by trial. Intervals resample the
held-out classes (the unit that was kept out of training), with all seeds of an arm averaged inside each
resample.

usage:
  python research/array/report.py --runs DIR/run2 --profiles DIR/profiles.npz --library LIB --out research/array/RESULTS.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

BANDS = (("near", 0.0, 0.8), ("mid", 0.8, 1.3), ("far", 1.3, 99.0))
ARMS = [("none_anchor", "only which cells exist"), ("none_pairs", "only which cells exist, with the pair machinery"),
        ("blind", "numbers, all pairs, query hidden"),
        ("num_anchor", "numbers, gene-member"), ("num_pairs", "numbers, all pairs"),
        ("num_pairs_comp", "numbers, all pairs, plus composition"),
        ("msa_anchor", "residues, gene-member"), ("msa_pairs", "residues, all pairs")]
CONTRASTS = [("msa_pairs", "num_pairs", "residues minus numbers (all pairs)"),
             ("msa_anchor", "num_anchor", "residues minus numbers (gene-member)"),
             ("num_pairs", "num_anchor", "all pairs minus gene-member (numbers)"),
             ("msa_pairs", "msa_anchor", "all pairs minus gene-member (residues)"),
             ("num_pairs_comp", "num_pairs", "composition added (numbers, all pairs)"),
             ("num_anchor", "none_anchor", "numbers minus presence only (gene-member)"),
             ("num_pairs", "none_pairs", "numbers minus presence only (all pairs)"),
             ("none_pairs", "none_anchor", "pair machinery without any relation"),
             ("msa_pairs", "rule_ai", "residues, all pairs minus the fixed rule"),
             ("num_anchor", "rule_ai", "numbers, gene-member minus the fixed rule")]


def auroc(score, y):
    y = y.astype(bool)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(score, kind="stable")
    s = score[order]
    r = np.arange(1, len(s) + 1, dtype=np.float64)
    new = np.r_[True, s[1:] != s[:-1]]
    grp = np.cumsum(new) - 1
    r = (np.bincount(grp, weights=r) / np.bincount(grp))[grp]      # average ranks over ties
    rank = np.empty(len(s)); rank[order] = r
    return float((rank[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recall_at(score, y, fpr=0.05):
    y = y.astype(bool)
    if y.all() or not y.any():
        return np.nan
    neg = np.sort(score[~y])[::-1]
    return float((score[y] > neg[min(len(neg) - 1, int(fpr * len(neg)))]).mean())


def load(runs, which):
    """{arm: [scores per seed]} and the shared trial table."""
    table, scores = None, {}
    for arm, _ in ARMS:
        for d in sorted(glob.glob(os.path.join(runs, arm)) + glob.glob(os.path.join(runs, arm + "_s[0-9]"))):
            f = os.path.join(d, "trials_%s.npz" % which)
            if not os.path.exists(f):
                continue
            z = dict(np.load(f))
            if table is None:
                table = z
            elif not (np.array_equal(table["x"], z["x"]) and np.array_equal(table["label"], z["label"]) and np.array_equal(table["donor"], z["donor"])):
                raise SystemExit("%s: trials differ from the first run" % d)
            scores.setdefault(arm, []).append(z["logit"].astype(np.float64))
    if table is not None:
        for k in ("rule_ai", "rule_ctx"):
            scores[k] = [table[k].astype(np.float64)]
    return table, scores


def masks(t):
    m = {"all": np.ones(len(t["label"]), dtype=bool)}
    for name, lo, hi in BANDS:
        m[name] = (t["d_gt"] >= lo) & (t["d_gt"] < hi)
    m["near, nothing hidden"] = m["near"] & (t["k_g"] == 0) & (t["k_t"] == 0)
    m["near, 16 rows hidden around the genome"] = m["near"] & (t["k_g"] == 16)
    m["near, 16 rows hidden around the donor"] = m["near"] & (t["k_t"] == 16)
    m["near, clean context"] = m["near"] & (t["n_c"] == 0)
    m["near, 3 genes from elsewhere in the context"] = m["near"] & (t["n_c"] == 3)
    return m


def mean_auroc(seeds, y, m):
    return float(np.mean([auroc(s[m], y[m]) for s in seeds]))


def boot(scores, y, m, unit, a, b, n=1000, seed=0):
    """Interval of mean AUROC(a) - mean AUROC(b) on mask m, resampling units (classes) with replacement."""
    rng = np.random.default_rng(seed)
    units = np.unique(unit)
    idx = {u: np.flatnonzero((unit == u) & m) for u in units}
    out = []
    for _ in range(n):
        pick = np.concatenate([idx[u] for u in rng.choice(units, len(units))])
        if len(pick) == 0:
            continue
        yy = y[pick]
        d = np.mean([auroc(s[pick], yy) for s in scores[a]]) - np.mean([auroc(s[pick], yy) for s in scores[b]])
        if np.isfinite(d):
            out.append(d)
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True); ap.add_argument("--profiles", required=True); ap.add_argument("--library", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    prof, lib = np.load(a.profiles), np.load(a.library)
    r = lib["ranks"][np.maximum(prof["own"], 0)]
    cls = np.where(r[:, 2] != 0, r[:, 2], np.where(r[:, 1] != 0, r[:, 1], r[:, 0]))     # the unit held out of training
    L = ["# Array model: results", "",
         "Generated by `research/array/report.py` from the run files. Do not edit by hand.", ""]
    for which, title in (("unseen", "Genomes of classes never trained on, donors from such classes too"),
                         ("validation", "Validation genomes (classes seen in training; used to choose the checkpoint)")):
        t, scores = load(a.runs, which)
        if t is None:
            continue
        y, unit, M = t["label"] > 0.5, cls[t["g"]], masks(t)
        L += ["## " + title, "",
              "- Trials: %d (%d native, %d foreign) on %d genomes of %d classes; every run scores the same trials." % (
                  len(y), int((~y).sum()), int(y.sum()), len(set(t["g"].tolist())), len(set(unit.tolist()))),
              "- Donor distance classes by the measured distance between the genome and the donor: near below 0.8 "
              "(typically the same order or family), mid 0.8 to 1.3 (same class or phylum), far above (other phyla and domains, or no measurable distance).",
              "- Trials per class: " + ", ".join("%s %d" % (k, int(M[k].sum())) for k in ("near", "mid", "far")) + ".",
              "- Query has at least one member among the rows shown: %.0f%% of native queries, %.0f%% of foreign ones." % (
                  100 * (t["n_present"][~y] > 0).mean(), 100 * (t["n_present"][y] > 0).mean()), "",
              "### AUROC, native against foreign queries", "",
              "| input | seeds | all | near | mid | far | recall at 5% false positives, near |", "|---|---|---|---|---|---|---|"]
        rows = [(k, d) for k, d in ARMS if k in scores] + [("rule_ai", "fixed rule: best similarity outside the genome's nearest rows minus best inside"),
                                                           ("rule_ctx", "fixed rule: largest excess over the genome's genes")]
        for k, d in rows:
            L.append("| %s | %d | %s | %.3f |" % (d, len(scores[k]), " | ".join("%.3f" % mean_auroc(scores[k], y, M[b]) for b in ("all", "near", "mid", "far")),
                                                  np.mean([recall_at(s[M["near"]], y[M["near"]]) for s in scores[k]])))
        if which == "unseen":
            L += ["", "### Differences between inputs (AUROC, 95% interval over held-out classes)", "",
                  "| comparison | all | near | mid | far |", "|---|---|---|---|---|"]
            for x, z, d in CONTRASTS:
                if x in scores and z in scores:
                    cells = []
                    for b in ("all", "near", "mid", "far"):
                        diff = mean_auroc(scores[x], y, M[b]) - mean_auroc(scores[z], y, M[b])
                        lo, hi = boot(scores, y, M[b], unit, x, z, a.boot)
                        cells.append("%+.3f (%+.3f to %+.3f)" % (diff, lo, hi))
                    L.append("| %s | %s |" % (d, " | ".join(cells)))
            L += ["", "### Near donors in detail (AUROC)", "", "| input | " + " | ".join(k for k in M if k.startswith("near,")) + " |",
                  "|---|" + "---|" * sum(k.startswith("near,") for k in M)]
            for k, d in rows:
                L.append("| %s | %s |" % (d, " | ".join("%.3f" % mean_auroc(scores[k], y, M[b]) for b in M if b.startswith("near,"))))
            L += ["", "- Trials in these cells: " + ", ".join("%s %d" % (k[6:], int(M[k].sum())) for k in M if k.startswith("near,")) + "."]
            res = {}
            for k, _ in ARMS:
                f = os.path.join(a.runs, k, "result.json")
                if os.path.exists(f):
                    res[k] = json.load(open(f))
            if res:
                L += ["", "### Pointer to the nearest row of the true origin (foreign queries)", "", "| input | model | most similar row | chance |", "|---|---|---|---|"]
                for k, d in ARMS:
                    if k in res and "pointer" in res[k]["unseen"]:
                        p = res[k]["unseen"]["pointer"]
                        L.append("| %s | %.3f | %.3f | %.3f |" % (d, p["model"], p["most_similar_row"], p["chance"]))
        L.append("")
    f = os.path.join(a.runs, "bench.json")
    if os.path.exists(f):
        B = json.load(open(f))
        L += ["## Benchmark genomes (a floor check, not a ranking)", "",
              "- All benchmark donors are from other phyla or domains, most positives carry simulated random substitutions, and the negatives "
              "contain the recipient's real transfers. These numbers show whether a score fails grossly; they cannot rank the inputs.",
              "- A gene without any member in the rows shown ranks last, as genes without evidence do for the earlier scores. "
              "The array model reads a quarter of the reference proteomes; the earlier scores read all of them.", ""]
        keys = ["rules", "ai_all", "place.node", "num_anchor.rule_ai"] + ["%s.array" % k for k, _ in ARMS]
        names = dict([("rules", "pipeline rules"), ("ai_all", "Alien Index"), ("place.node", "placement model (taxonomy)"),
                      ("num_anchor.rule_ai", "fixed rule on the array rows")] + [("%s.array" % k, d) for k, d in ARMS])
        for ds, r in B.items():
            L += ["### %s (%d positives, %d core genes, %d other genes)" % (ds, r["n_pos"], r["n_core"], r["n_neg"]), "",
                  "| score | AP against core genes | AP against all genes | recall at the high tier's false positives |", "|---|---|---|---|"]
            for k in keys:
                m = r.get("scores", {}).get(k)
                if m:
                    L.append("| %s | %.3f | %.3f | %.3f |" % (names[k], m["ap_core"], m["ap_all"], m["recall_at_high_fp"]))
            L.append("")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
