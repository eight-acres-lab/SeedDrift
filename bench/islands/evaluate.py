#!/usr/bin/env python
"""Island genes against backbone genes, per chromosome, for any number of gene scores (protocol: README.md).

  labels   <labels>/<accession>.tsv from labels.py (island / backbone / none); unlabelled genes take no part
  scores   folders with scores_<accession>.tsv (gene_id plus score columns, higher = more foreign); a gene that a
           file does not list ranks last for the scores of that file
  metrics  per chromosome: AUROC (average ranks for ties); recall of island genes when at most 5% / 1% of the
           backbone genes pass (ties count against the score), with the share of backbone genes that actually pass
  summary  over the chromosomes with at least --min-genes island and backbone genes: mean and median, and for
           every --contrast a-b the mean paired difference with a bootstrap interval over chromosomes
  tiers    for every --tiers label=DIR (score files with the columns tier and category, from run.py scores): the share
           of backbone genes and of island genes that are called at the high tier, at high or medium, at any tier;
           counts summed over all chromosomes, exact binomial intervals; island calls by category

usage:
  python bench/islands/evaluate.py --labels DIR/labels --genomes results/genomes.tsv \\
      --scores comp=DIR/composition wide=DIR/wide_all:wide --contrast comp.kmer_nb-comp.gc3 --out results/name
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np

PHAGE = ("phage", "integrase", "transposase", "terminase", "capsid", "tail", "portal", "holin", "lysozyme", "recombinase",
         "insertion sequence", "is element", "conjug", "plasmid")


def auroc(pos, neg):
    s = np.r_[pos, neg]
    order = np.argsort(s, kind="stable")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    ss = s[order]
    new = np.r_[True, ss[1:] != ss[:-1]]
    grp = np.cumsum(new) - 1
    r[order] = (np.bincount(grp, weights=r[order]) / np.bincount(grp))[grp]      # average ranks over ties
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_at(pos, neg, share):
    """Recall when at most `share` of the negatives pass; a positive passes only if it scores above the threshold."""
    k = int(np.floor(share * len(neg)))
    srt = np.sort(neg)[::-1]
    thr = srt[k] if k < len(srt) else -np.inf            # the (k+1)-th highest negative must not pass
    return float(np.mean(pos > thr)), float(np.mean(neg > thr))


def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if p <= 0.0 or k >= n:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq) for i in range(k + 1)))


def exact_interval(k, n, alpha=0.05):
    """Clopper-Pearson interval of a share k / n."""
    def solve(f):                                            # f is decreasing in p; the p with f(p) = 0
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        return (lo + hi) / 2
    if n == 0:
        return (0.0, 1.0)
    lower = 0.0 if k == 0 else solve(lambda p: alpha / 2 - (1 - binom_cdf(k - 1, n, p)))
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) - alpha / 2)
    return (lower, upper)


def read_tiers(spec, acc):
    label, d = spec.split("=", 1)
    path = os.path.join(d, "scores_%s.tsv" % acc)
    if not os.path.exists(path):
        return label, None
    out = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        gid = r["gene_id"]
        out[gid[len(acc) + 2:] if gid.startswith(acc + "__") else gid] = (int(r["tier"]), r.get("category") or "")
    return label, out


def read_scores(spec, acc):
    label, rest = spec.split("=", 1)
    d, _, cols = rest.partition(":")
    path = os.path.join(d, "scores_%s.tsv" % acc)
    if not os.path.exists(path):
        return {}
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    want = cols.split(",") if cols else [c for c in rows[0] if c != "gene_id"]
    out = {}
    for c in want:
        m = {}
        for r in rows:
            gid = r["gene_id"]
            if gid.startswith(acc + "__"):
                gid = gid[len(acc) + 2:]
            try:
                m[gid] = float(r[c])
            except (ValueError, TypeError):
                pass
        out["%s.%s" % (label, c)] = m
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True); ap.add_argument("--genomes", required=True)
    ap.add_argument("--scores", nargs="+", required=True, help="label=DIR[:col,col]")
    ap.add_argument("--contrast", nargs="*", default=[], help="a-b pairs of score names")
    ap.add_argument("--tiers", nargs="*", default=[], help="label=DIR of score files with the columns tier and category")
    ap.add_argument("--min-genes", type=int, default=50); ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    genomes = list(csv.DictReader(open(a.genomes), delimiter="\t"))
    per, names = [], []
    tiers = {}
    for g in genomes:
        acc = g["accession"]
        lab = list(csv.DictReader(open(os.path.join(a.labels, acc + ".tsv")), delimiter="\t"))
        isl = [r["gene_id"] for r in lab if r["label"] == "island"]
        bb = [r["gene_id"] for r in lab if r["label"] == "backbone"]
        mobile = {r["gene_id"] for r in lab if any(w in r["product"].lower() for w in PHAGE)}
        row = dict(accession=acc, genus=g["genus"], organism=g["organism"], n_island=len(isl), n_backbone=len(bb),
                   island_mobile_words=round(float(np.mean([x in mobile for x in isl])), 3) if isl else None,
                   backbone_mobile_words=round(float(np.mean([x in mobile for x in bb])), 3) if bb else None,
                   in_summary=len(isl) >= a.min_genes and len(bb) >= a.min_genes, scores={})
        for spec in a.scores:
            for name, m in read_scores(spec, acc).items():
                if name not in names:
                    names.append(name)
                if not isl or not bb:
                    continue
                low = min(m.values()) - 1.0 if m else 0.0
                p = np.array([m.get(x, low) for x in isl]); n = np.array([m.get(x, low) for x in bb])
                r5, f5 = recall_at(p, n, 0.05); r1, f1 = recall_at(p, n, 0.01)
                row["scores"][name] = dict(auroc=round(auroc(p, n), 4), recall_5=round(r5, 4), fp_5=round(f5, 4), recall_1=round(r1, 4),
                                           fp_1=round(f1, 4), listed=round(float(np.mean([x in m for x in isl + bb])), 3))
        for spec in a.tiers:
            label, m = read_tiers(spec, acc)
            if m is None:
                continue
            t = tiers.setdefault(label, dict(chromosomes=0, backbone=[0, 0, 0, 0], island=[0, 0, 0, 0], island_category={}, per_genome={}))
            t["chromosomes"] += 1
            pg = dict(backbone=[0, 0, 0, 0], island=[0, 0, 0, 0])
            for kind, ids in (("backbone", bb), ("island", isl)):
                for x in ids:
                    tier, cat = m.get(x, (0, ""))
                    pg[kind][0] += 1
                    for lvl, cut in ((1, 3), (2, 2), (3, 1)):                  # high; high or medium; any tier
                        pg[kind][lvl] += tier >= cut
                    if kind == "island" and tier >= 1:
                        t["island_category"][cat] = t["island_category"].get(cat, 0) + 1
                t[kind] = [u + v for u, v in zip(t[kind], pg[kind])]
            t["per_genome"][acc] = pg
        per.append(row)
    for t in tiers.values():
        for kind in ("backbone", "island"):
            n = t[kind][0]
            t[kind + "_share"] = {name: dict(k=t[kind][lvl], n=n, share=round(t[kind][lvl] / n, 5) if n else None,
                                             interval=[round(v, 5) for v in exact_interval(t[kind][lvl], n)])
                                  for lvl, name in ((1, "high"), (2, "high_or_medium"), (3, "any"))}
    use = [r for r in per if r["in_summary"]]
    rng = np.random.default_rng(0)
    summ = {}
    for k in names:
        v = {m: np.array([r["scores"][k][m] for r in use if k in r["scores"]]) for m in ("auroc", "recall_5", "recall_1")}
        summ[k] = {m: dict(mean=round(float(x.mean()), 4), median=round(float(np.median(x)), 4), n=len(x)) for m, x in v.items() if len(x)}
    cons = {}
    for c in a.contrast:
        x, _, y = c.partition("-")
        rows = [r for r in use if x in r["scores"] and y in r["scores"]]
        cons[c] = {}
        for m in ("auroc", "recall_5", "recall_1"):
            d = np.array([r["scores"][x][m] - r["scores"][y][m] for r in rows])
            if not len(d):
                continue
            bs = d[rng.integers(0, len(d), (a.boot, len(d)))].mean(1)
            cons[c][m] = dict(mean=round(float(d.mean()), 4), lo=round(float(np.quantile(bs, 0.025)), 4), hi=round(float(np.quantile(bs, 0.975)), 4),
                              n=len(d), wins=int((d > 0).sum()))
    json.dump(dict(per_genome=per, summary=summ, contrasts=cons, tiers=tiers, min_genes=a.min_genes), open(a.out + ".json", "w"), indent=1)
    L = ["Chromosomes: %d, of which %d have at least %d island and %d backbone genes and enter the summary." % (len(per), len(use), a.min_genes, a.min_genes), "",
         "| score | AUROC mean | AUROC median | recall at 5% backbone FP | recall at 1% backbone FP | chromosomes |", "|---|---|---|---|---|---|"]
    for k in names:
        s = summ.get(k, {})
        if "auroc" in s:
            L.append("| %s | %.3f | %.3f | %.3f | %.3f | %d |" % (k, s["auroc"]["mean"], s["auroc"]["median"], s["recall_5"]["mean"], s["recall_1"]["mean"], s["auroc"]["n"]))
    if cons:
        L += ["", "| contrast | AUROC difference (95%) | recall at 5% FP difference (95%) | recall at 1% FP difference (95%) | chromosomes won |", "|---|---|---|---|---|"]
        for c, v in cons.items():
            f = lambda m: "%+.3f (%+.3f to %+.3f)" % (v[m]["mean"], v[m]["lo"], v[m]["hi"]) if m in v else ""
            L.append("| %s | %s | %s | %s | %s |" % (c, f("auroc"), f("recall_5"), f("recall_1"), "%d of %d" % (v["auroc"]["wins"], v["auroc"]["n"]) if "auroc" in v else ""))
    for label, t in tiers.items():
        L += ["", "Calls of `%s` (%d chromosomes, counts summed; exact 95%% intervals):" % (label, t["chromosomes"]), "",
              "| genes | called at | called | of | share (95%) |", "|---|---|---|---|---|"]
        for kind in ("backbone", "island"):
            for name, v in t[kind + "_share"].items():
                L.append("| %s | %s | %d | %d | %s |" % (kind, name.replace("_", " "), v["k"], v["n"],
                         "" if v["share"] is None else "%.2f%% (%.2f%% to %.2f%%)" % (100 * v["share"], 100 * v["interval"][0], 100 * v["interval"][1])))
        L += ["", "Island genes called, by category: " + (", ".join("%s %d" % kv for kv in sorted(t["island_category"].items(), key=lambda kv: -kv[1])) or "none") + "."]
    L += ["", "Per chromosome (AUROC):", "", "| chromosome | organism | island genes | backbone genes | island genes with mobile-element words | " + " | ".join(names) + " |",
          "|---|---|---|---|---|" + "---|" * len(names)]
    for r in per:
        L.append("| %s | %s | %d | %d | %s | %s |" % (r["accession"], r["organism"][:45], r["n_island"], r["n_backbone"],
                 "" if r["island_mobile_words"] is None else "%.2f" % r["island_mobile_words"],
                 " | ".join("%.3f" % r["scores"][k]["auroc"] if k in r["scores"] else "" for k in names)))
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
