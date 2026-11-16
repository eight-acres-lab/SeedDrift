#!/usr/bin/env python
"""Write RESULTS.md for the array model from the files of the runs (nothing is typed in by hand).

All runs score the same queries (fixed seeds), so inputs are compared trial by trial. An interval combines
two sources of variation: the held-out classes (resampled with replacement, all seeds of an input averaged
inside each resample) and the training seed (variance of the seed mean; an input with one seed borrows the
median seed spread of the inputs that have several, and is marked).

usage:
  python research/array/report.py --runs DIR/run2 DIR/run3 --ceiling DIR/ceiling_unseen.npz --profiles DIR/profiles.npz \
      --library LIB --out research/array/RESULTS.md
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
        ("msa_anchor", "residues, gene-member"), ("msa_pairs", "residues, all pairs"),
        ("plm_anchor", "residues with language-model sites, gene-member"), ("plm_pairs", "residues with language-model sites, all pairs"),
        ("wide_panel", "wide table: 64 genes, every gene judged, panel rows"),
        ("wide_panel_show13", "wide table control: only 13 of the 64 genes shown, panel rows"),
        ("wide_panel_blank", "wide table control: only which cells exist, panel rows"),
        ("wide_all", "wide table: 64 genes, every gene judged, all reference proteomes")]
NAME = dict(ARMS)
SECOND = [("msa_pairs", "num_pairs", "residues minus numbers (all pairs)"),
          ("msa_anchor", "num_anchor", "residues minus numbers (gene-member)"),
          ("num_pairs", "num_anchor", "all pairs minus gene-member (numbers)"),
          ("msa_pairs", "msa_anchor", "all pairs minus gene-member (residues)")]
THIRD = [("wide_panel", "num_anchor", "wide table (panel rows) minus numbers, gene-member"),
         ("wide_panel", "wide_panel_show13", "wide table (panel rows): all 64 genes shown minus 13 shown"),
         ("wide_panel", "wide_panel_blank", "wide table (panel rows): numbers in the cells minus presence only"),
         ("wide_all", "wide_panel", "wide table: all reference proteomes minus panel rows")]
OTHER = [("plm_pairs", "msa_pairs", "language-model sites added (all pairs)"),
         ("plm_anchor", "msa_anchor", "language-model sites added (gene-member)"),
         ("num_pairs_comp", "num_pairs", "composition added (numbers, all pairs)"),
         ("num_anchor", "none_anchor", "numbers minus presence only (gene-member)"),
         ("num_pairs", "none_pairs", "numbers minus presence only (all pairs)"),
         ("none_pairs", "none_anchor", "pair machinery without any relation"),
         ("num_anchor", "rule_ai", "numbers, gene-member minus the fixed rule on its rows"),
         ("msa_pairs", "rule_ai", "residues, all pairs minus the fixed rule on its rows"),
         ("wide_panel", "rule_ai@wide_panel", "wide table (panel rows) minus the fixed rule on its own rows"),
         ("wide_all", "rule_ai@wide_all", "wide table (all reference proteomes) minus the fixed rule on its own rows"),
         ("wide_panel", "msa_pairs", "wide table (panel rows) minus residues, all pairs")]


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
    """{input: [scores per seed]}, the shared trial table (of the first second-version run found) and the result files."""
    table, scores, info = None, {}, {}
    for arm, _ in ARMS:
        seeds = {}
        for d in sorted(sum((glob.glob(os.path.join(r, arm)) + glob.glob(os.path.join(r, arm + "_s[0-9]*")) for r in runs), [])):
            f, rj = os.path.join(d, "trials_%s.npz" % which), os.path.join(d, "result.json")
            if not (os.path.exists(f) and os.path.exists(rj)):
                continue
            res = json.load(open(rj))
            seed = int(res["args"]["seed"])
            if seed in seeds:
                raise SystemExit("%s: seed %d appears twice (%s and %s)" % (arm, seed, seeds[seed], d))
            seeds[seed] = d
            z = dict(np.load(f))
            if table is None:
                if arm.startswith("wide"):
                    raise SystemExit("the trial table has to come from a second-version run")
                table = z
            elif not all(np.array_equal(table[k], z[k]) for k in ("x", "label", "donor", "k_g", "k_t")):
                raise SystemExit("%s: trials differ from the first run" % d)
            scores.setdefault(arm, []).append(z["logit"].astype(np.float64))
            info.setdefault(arm, []).append(res)
            if "wide_rule_ai" in z and "rule_ai@" + arm not in scores:      # the same for every seed of the input: its rows do not depend on training
                scores["rule_ai@" + arm] = [z["wide_rule_ai"].astype(np.float64)]
                NAME["rule_ai@" + arm] = "fixed rule on the rows of: " + NAME[arm]
    if table is not None:
        for k in ("rule_ai", "rule_ctx"):
            scores[k] = [table[k].astype(np.float64)]
    return table, scores, info


def masks(t, sep=None):
    m = {"all": np.ones(len(t["label"]), dtype=bool)}
    if sep is not None:
        m["separable"], m["not separable"] = sep, ~sep
    for name, lo, hi in BANDS:
        m[name] = (t["d_gt"] >= lo) & (t["d_gt"] < hi)
    m["near, nothing hidden"] = m["near"] & (t["k_g"] == 0) & (t["k_t"] == 0)
    m["near, 16 rows hidden around the genome"] = m["near"] & (t["k_g"] == 16)
    m["near, 16 rows hidden around the donor"] = m["near"] & (t["k_t"] == 16)
    m["near, clean context"] = m["near"] & (t["n_c"] == 0)
    m["near, mixed context (3 of 12 genes from elsewhere; 16 of 63 in the wide table)"] = m["near"] & (t["n_c"] == 3)
    return m


def per_seed(seeds, y, m):
    return np.array([auroc(s[m], y[m]) for s in seeds])


def mean_auroc(seeds, y, m):
    return float(np.mean(per_seed(seeds, y, m)))


class Stats:
    def __init__(self, scores, y, unit, n_boot):
        self.scores, self.y, self.unit, self.n = scores, y, unit, n_boot

    def pooled_sd(self, m):
        """Median seed-to-seed SD of the AUROC on mask m over the inputs with several seeds."""
        sd = [np.std(per_seed(v, self.y, m), ddof=1) for v in self.scores.values() if len(v) > 1]
        return float(np.median(sd)) if sd else float("nan")

    def seed_var(self, k, m):
        """Variance of the seed mean of input k; (value, borrowed?)."""
        v = self.scores[k]
        if k.startswith("rule"):
            return 0.0, False
        if len(v) > 1:
            return float(np.var(per_seed(v, self.y, m), ddof=1) / len(v)), False
        sd = self.pooled_sd(m)
        return (sd * sd if np.isfinite(sd) else 0.0), True

    def diff(self, a, b, m, seed=0):
        """AUROC(a) - AUROC(b) on mask m, its 95% interval, and whether a seed spread was borrowed."""
        rng = np.random.default_rng(seed)
        units = np.unique(self.unit)
        idx = {u: np.flatnonzero((self.unit == u) & m) for u in units}
        d0 = mean_auroc(self.scores[a], self.y, m) - mean_auroc(self.scores[b], self.y, m)
        out = []
        for _ in range(self.n):
            pick = np.concatenate([idx[u] for u in rng.choice(units, len(units))])
            if len(pick) == 0:
                continue
            yy = self.y[pick]
            d = np.mean([auroc(s[pick], yy) for s in self.scores[a]]) - np.mean([auroc(s[pick], yy) for s in self.scores[b]])
            if np.isfinite(d):
                out.append(d)
        if not out:
            return d0, np.nan, np.nan, False
        (va, ba), (vb, bb) = self.seed_var(a, m), self.seed_var(b, m)
        hw = 1.96 * float(np.sqrt(np.var(out) + va + vb))
        return d0, d0 - hw, d0 + hw, ba or bb

    def cell(self, a, b, m):
        d, lo, hi, borrowed = self.diff(a, b, m)
        return "%+.3f (%+.3f to %+.3f)%s" % (d, lo, hi, " †" if borrowed else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--profiles", required=True); ap.add_argument("--library", required=True)
    ap.add_argument("--ceiling", default="", help="the .npz written by ceiling.py for the unseen-class trials (adds the split by separability)")
    ap.add_argument("--which", nargs="+", default=["unseen", "validation"], help="trial sets to report (file names trials_<which>.npz)")
    ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    prof, lib = np.load(a.profiles), np.load(a.library)
    r = lib["ranks"][np.maximum(prof["own"], 0)]
    cls = np.where(r[:, 2] != 0, r[:, 2], np.where(r[:, 1] != 0, r[:, 1], r[:, 0]))     # the unit held out of training
    titles = {"unseen": "Genomes of classes never trained on, donors from such classes too",
              "fresh": "The same genomes, a fresh draw of trials (made after the third version was designed)",
              "validation": "Validation genomes (classes seen in training; used to choose the checkpoint)"}
    L = ["# Array model: results", "",
         "Generated by `research/array/report.py` from the run files. Do not edit by hand.", "",
         "- † marks an interval where an input had a single training seed and borrowed the median seed spread of the inputs with several.",
         "- Intervals are normal intervals: spread over held-out classes (bootstrap) combined with the spread of the seed mean. "
         "With about two dozen classes they are somewhat too narrow.",
         "- Inputs marked \"added later\" are not part of the protocol fixed before the second version's results.", ""]
    for which in a.which:
        t, scores, info = load(a.runs, which)
        if t is None:
            continue
        sep = sep_all = None
        if which == "unseen" and a.ceiling:
            z = np.load(a.ceiling)
            st, names = z["stats"], [str(x) for x in z["names"]]
            if not (np.array_equal(st[:, 0], t["x"]) and np.array_equal(st[:, 3], t["label"])):
                raise SystemExit("the ceiling file is not for these trials")
            ds, gs = st[:, 4 + names.index("donor_side")], st[:, 4 + names.index("genome_side")]
            sep = ~((ds <= 0.05) & (gs <= 0.05))                   # an undefined side (the two genomes have no measurable distance) counts as separable
            if "donor_side_all" in names:
                sep_all = ~((st[:, 4 + names.index("donor_side_all")] <= 0.05) & (st[:, 4 + names.index("genome_side_all")] <= 0.05))
        y, unit, M = t["label"] > 0.5, cls[t["g"]], masks(t, sep)
        S = Stats(scores, y, unit, a.boot)
        n_cls = len(set(unit.tolist()))
        big = max(np.bincount(np.unique(unit, return_inverse=True)[1])) / len(unit)
        L += ["## " + titles.get(which, which), "",
              "- Trials: %d (%d native, %d foreign) on %d genomes of %d classes (the largest class holds %.0f%% of the trials). Every run scores the same "
              "queries with the same genome, donor and hiding level; the wide table draws its own other genes and rows." % (
                  len(y), int((~y).sum()), int(y.sum()), len(set(t["g"].tolist())), n_cls, 100 * big),
              "- Donor distance classes by the measured distance between the genome and the donor: near below 0.8 "
              "(typically the same order or family), mid 0.8 to 1.3 (same class or phylum), far above (other phyla and domains, or no measurable distance).",
              "- Trials per class: " + ", ".join("%s %d" % (k, int(M[k].sum())) for k in ("near", "mid", "far")) + ".",
              "- Query has at least one member among the 32 rows of the second version: %.0f%% of native queries, %.0f%% of foreign ones." % (
                  100 * (t["n_present"][~y] > 0).mean(), 100 * (t["n_present"][y] > 0).mean())]
        for k in ("wide_panel", "wide_all"):
            q = info[k][0].get(which, {}).get("query_has_rows") if k in info else None
            if q:
                L.append("- Among the rows of the %s: %.0f%% of native queries, %.0f%% of foreign ones." % (NAME[k], 100 * q["share_native"], 100 * q["share_foreign"]))
        L += ["", "### AUROC, native against foreign queries", "",
              "| input | seeds | all | near | mid | far | seed SD (all) | recall at 5% false positives, all |", "|---|---|---|---|---|---|---|---|"]
        later = {k for k, _ in ARMS if k.startswith(("plm", "wide"))}
        rows = [(k, NAME[k] + (" (added later)" if k in later else "")) for k, _ in ARMS if k in scores]
        rows += [("rule_ai", "fixed rule on the second version's 32 rows: best similarity outside the genome's nearest rows minus best inside"),
                 ("rule_ctx", "fixed rule on the second version's 32 rows: largest excess over the genome's genes")]
        rows += [(k, NAME[k]) for k in scores if k.startswith("rule_ai@")]
        for k, d in rows:
            ps = per_seed(scores[k], y, M["all"])
            L.append("| %s | %d | %s | %s | %.3f |" % (d, len(scores[k]), " | ".join("%.3f" % mean_auroc(scores[k], y, M[b]) for b in ("all", "near", "mid", "far")),
                                                       "%.3f" % np.std(ps, ddof=1) if len(ps) > 1 else "-",
                                                       np.mean([recall_at(s, y) for s in scores[k]])))
        if which != "validation":
            for title, group in (("Comparisons fixed before the second version's results", SECOND),
                                 ("Comparisons of the third version (fixed before its results)", THIRD),
                                 ("Other comparisons (exploratory)", OTHER)):
                have = [(x, z, d) for x, z, d in group if x in scores and z in scores]
                if not have:
                    continue
                cols = ("all", "near", "mid", "far") + (("separable",) if sep is not None else ())
                L += ["", "### " + title + ": AUROC difference (95% interval)", "", "| comparison | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
                for x, z, d in have:
                    L.append("| %s | %s |" % (d, " | ".join(S.cell(x, z, M[b]) for b in cols)))
            if sep is not None:
                L += ["", "### By whether any visible panel row can tell the donor from the genome", "",
                      "- Split by the two genome profiles only, on the panel rows left visible in the second version's trial (ceiling.py); no query is looked at. "
                      "\"Not separable\": the nearest visible panel row of the donor is at most 0.05 nearer to it than the genome is, and the same from the "
                      "genome's side, so every panel row is about equally far from both.",
                      "- The split and its 0.05 tolerance were chosen after the first results of the second version were seen. Trials: separable %d, not separable %d "
                      "(%.0f%% of the near donors, %.0f%% of mid, %.0f%% of far are not separable)." % (
                          int(sep.sum()), int((~sep).sum()), 100 * (~sep & M["near"]).sum() / max(1, M["near"].sum()),
                          100 * (~sep & M["mid"]).sum() / max(1, M["mid"].sum()), 100 * (~sep & M["far"]).sum() / max(1, M["far"].sum())),
                      "- The wide table on all reference proteomes reads rows outside the panel, so the split does not bound it: its \"not separable\" value can be above 0.5.", "",
                      "| input | separable | not separable |", "|---|---|---|"]
                for k, d in rows:
                    L.append("| %s | %.3f | %.3f |" % (d, mean_auroc(scores[k], y, M["separable"]), mean_auroc(scores[k], y, M["not separable"])))
            if sep_all is not None and any(k.startswith("wide_all") for k in scores):
                L += ["", "The same split with every reference proteome as a possible row and hiding by radius, as the wide table on all reference proteomes sees it "
                      "(separable %d, not separable %d):" % (int(sep_all.sum()), int((~sep_all).sum())), "",
                      "| input | separable | not separable |", "|---|---|---|"]
                for k, d in rows:
                    if "wide_all" in k:
                        L.append("| %s | %.3f | %.3f |" % (d, mean_auroc(scores[k], y, sep_all), mean_auroc(scores[k], y, ~sep_all)))
            near = [k for k in M if k.startswith("near,")]
            L += ["", "### Near donors in detail (AUROC)", "", "| input | " + " | ".join(k[6:] for k in near) + " |", "|---|" + "---|" * len(near)]
            for k, d in rows:
                L.append("| %s | %s |" % (d, " | ".join("%.3f" % mean_auroc(scores[k], y, M[b]) for b in near)))
            L += ["", "- Trials in these cells: " + ", ".join("%s %d" % (k[6:], int(M[k].sum())) for k in near) + "."]
            pt = [(k, [x[which]["pointer"] for x in info[k] if "pointer" in x.get(which, {})]) for k, _ in ARMS if k in info]
            pt = [(k, v) for k, v in pt if v]
            if pt:
                L += ["", "### Pointer to the nearest row of the true origin (foreign queries; mean over seeds)", "",
                      "| input | model | most similar row | chance |", "|---|---|---|---|"]
                for k, v in pt:
                    L.append("| %s | %.3f | %.3f | %.3f |" % (NAME[k], np.mean([p["model"] for p in v]), np.mean([p["most_similar_row"] for p in v]),
                                                             np.mean([p["chance"] for p in v])))
                L += ["", "- Each input is scored against its own rows; the wide table has other and more rows than the second version, so the columns are "
                      "comparable within a line, not between the two versions."]
        L.append("")
    B = {}
    for r in a.runs:
        f = os.path.join(r, "bench.json")
        if os.path.exists(f):
            for ds, v in json.load(open(f)).items():
                if ds in B:
                    B[ds]["scores"].update(v.get("scores", {}))
                else:
                    B[ds] = v
    if B:
        L += ["## Benchmark genomes (a floor check, not a ranking)", "",
              "- All benchmark donors are from other phyla or domains, most positives carry simulated random substitutions, and the negatives "
              "contain the recipient's real transfers. These numbers show whether a score fails grossly; they cannot rank the inputs.",
              "- A gene without any member in the rows shown ranks last, as genes without evidence do for the earlier scores. "
              "The second version reads a quarter of the reference proteomes; the earlier scores and the wide table on all reference proteomes read all of them.",
              "- Seed 0 of every input.", ""]
        keys = ["rules", "ai_all", "place.node", "num_anchor.rule_ai"] + ["%s.array" % k for k, _ in ARMS] + [
            "wide_all.wide_rule_ai"] + ["%s.wide" % k for k, _ in ARMS]
        names = dict([("rules", "pipeline rules"), ("ai_all", "Alien Index"), ("place.node", "placement model (taxonomy)"),
                      ("num_anchor.rule_ai", "fixed rule on the second version's rows"), ("wide_all.wide_rule_ai", "fixed rule on the wide table's rows")] +
                     [("%s.array" % k, d) for k, d in ARMS] + [("%s.wide" % k, d) for k, d in ARMS])
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
