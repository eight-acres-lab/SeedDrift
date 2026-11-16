#!/usr/bin/env python
"""Island genes against backbone genes, by genus, at strict thresholds, gene by gene and region by region.

  unit       the genus (all its chromosomes pooled): strains and species of one genus are not independent
  window     every score is evaluated twice, with the same rule for all scores:
               w1   the gene's own score
               w11  the mean, over the gene and its 5 neighbours on each side (same contig), of the score's rank
                    within the chromosome
  genes      recall at 1% backbone false positives (ties at their expectation), mean recall over 0-5%, AUROC.
             A gene without a score ranks last, is never called, and stays in every count
  regions    (w11 only) one value per published region, the median over its genes. Share of island regions above
             the second best backbone region of the same chromosome (one backbone region is allowed to pass),
             and the same share when all labels are moved together along the chromosome (what block structure
             alone gives)
  frozen     --freeze FILE stores, per score and window, the threshold that passes 1% of the pooled backbone genes
             of this run; --frozen FILE applies stored thresholds unchanged: realised backbone false-positive rate
             per genus, recall, and the precision at a prevalence of 5%
  summary    mean over the eligible genera, 95% intervals and paired contrasts by bootstrap over genera;
             Holm-adjusted p-values over the contrasts of one metric

Eligibility (from label counts only): gene metrics need at least --min-backbone backbone and --min-island island
genes in the genus; region metrics need a chromosome with at least --min-backbone-regions backbone and
--min-island-regions island regions. Other genera are listed but enter no mean.

usage:
  python bench/hard/evaluate.py --labels DIR/labels --genomes results/dev_genomes.tsv \\
      --scores comp=DIR/dev_composition:codon,kmer table=DIR/dev_views/genus:ai_all,ai_narrow \\
      --contrast comp.codon-table.ai_all --out results/name
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as M  # noqa: E402

GENE_METRICS = ("recall_1", "mean_recall_5", "auroc")
WINDOWS = ("w1", "w11")


def read_scores(spec, acc):
    """{score name: {gene: value}} from DIR/scores_<acc>.tsv; label=DIR[:col,col]."""
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
                v = float(r[c])
            except (ValueError, TypeError, KeyError):
                continue
            if v == v:
                m[gid] = v
        out["%s.%s" % (label, c)] = m
    return out


def chromosome(labels_dir, acc):
    rows = list(csv.DictReader(open(os.path.join(labels_dir, acc + ".tsv")), delimiter="\t"))
    rows.sort(key=lambda r: (r["contig"], int(r["start"]), int(r["end"])))
    return dict(gene=[r["gene_id"] for r in rows], contig=np.array([r["contig"] for r in rows]),
                label=np.array([r["label"] for r in rows], dtype=object), region=np.array([r["region"] for r in rows], dtype=object))


def boot_mean(x, rng, n):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return None
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return dict(mean=round(float(x.mean()), 4), lo=round(float(np.quantile(bs, 0.025)), 4), hi=round(float(np.quantile(bs, 0.975)), 4), n=len(x))


def boot_diff(d, rng, n):
    d = np.asarray(d, dtype=float)
    if not len(d):
        return None
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    p = min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)) + 1.0 / n)
    return dict(mean=round(float(d.mean()), 4), lo=round(float(np.quantile(bs, 0.025)), 4), hi=round(float(np.quantile(bs, 0.975)), 4),
                n=len(d), wins=int((d > 0).sum()), p=round(float(p), 4))


def holm(ps):
    order = np.argsort(ps)
    out, run = np.empty(len(ps)), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        out[j] = run
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True); ap.add_argument("--genomes", required=True, help="table with the columns genus and accession")
    ap.add_argument("--scores", nargs="+", required=True, help="label=DIR[:col,col]")
    ap.add_argument("--contrast", nargs="*", default=[], help="a-b pairs of score names (both windows and every metric are compared)")
    ap.add_argument("--neighbours", type=int, default=5)
    ap.add_argument("--min-backbone", type=int, default=300); ap.add_argument("--min-island", type=int, default=50)
    ap.add_argument("--min-backbone-regions", type=int, default=20); ap.add_argument("--min-island-regions", type=int, default=5)
    ap.add_argument("--shifts", type=int, default=1000); ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--freeze", default="", help="write the thresholds of this run"); ap.add_argument("--frozen", default="", help="apply stored thresholds")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    genomes = list(csv.DictReader(open(a.genomes), delimiter="\t"))
    names, G = [], {}
    for g in genomes:
        acc = g["accession"]
        ch = chromosome(a.labels, acc)
        isl, bb = ch["label"] == "island", ch["label"] == "backbone"
        n_reg = {k: len(set(ch["region"][ch["label"] == k]) - {""}) for k in ("island", "backbone")}
        x = G.setdefault(g["genus"], dict(chromosomes=[], n_island=0, n_backbone=0, region_ok=False, pos={}, neg={}, scored={}, found={}, null={}))
        x["chromosomes"].append(acc); x["n_island"] += int(isl.sum()); x["n_backbone"] += int(bb.sum())
        reg_ok = n_reg["backbone"] >= a.min_backbone_regions and n_reg["island"] >= a.min_island_regions
        x["region_ok"] |= reg_ok
        for spec in a.scores:
            for name, m in read_scores(spec, acc).items():
                if name not in names:
                    names.append(name)
                has = np.array([q in m for q in ch["gene"]])
                low = (min(m.values()) if m else 0.0) - 1.0
                v = np.array([m.get(q, low) for q in ch["gene"]])
                x["scored"].setdefault(name, []).append((int(has[isl].sum()), int(has[bb].sum())))
                w11 = M.rank_window(v, ch["contig"], a.neighbours)
                raw = np.where(has, v, -np.inf)                       # a gene without a score is below every threshold
                for win, val in (("w1", raw), ("w11", w11)):
                    x["pos"].setdefault((name, win), []).append(val[isl]); x["neg"].setdefault((name, win), []).append(val[bb])
                if reg_ok:
                    p, q = M.region_units(w11, ch["region"], ch["label"])
                    f, n = M.regions_found(p, q, 1)
                    null = M.shift_null(w11, ch["region"], ch["label"], 1, a.shifts, seed=1) if a.shifts else np.array([np.nan])
                    x["found"].setdefault(name, []).append((f, n)); x["null"].setdefault(name, []).append((float(np.nanmean(null)), n))
    frozen = json.load(open(a.frozen)) if a.frozen else None
    res, thr_out = {}, {}
    for genus, x in sorted(G.items()):
        gene_ok = x["n_backbone"] >= a.min_backbone and x["n_island"] >= a.min_island
        r = dict(chromosomes=x["chromosomes"], n_island=x["n_island"], n_backbone=x["n_backbone"], gene_ok=gene_ok, region_ok=bool(x["region_ok"]), scores={})
        for name in names:
            if (name, "w1") not in x["pos"]:
                continue
            s = {}
            for win in WINDOWS:
                pos, neg = np.concatenate(x["pos"][(name, win)]), np.concatenate(x["neg"][(name, win)])
                if not len(pos) or not len(neg):
                    continue
                fin = lambda z: np.where(np.isfinite(z), z, np.nanmin(np.r_[pos[np.isfinite(pos)], neg[np.isfinite(neg)], 0.0]) - 1.0)
                p2, n2 = fin(pos), fin(neg)
                s[win] = dict(recall_1=round(M.recall_at(p2, n2, 0.01), 4), mean_recall_5=round(M.mean_recall(p2, n2, 0.05), 4), auroc=round(M.auroc(p2, n2), 4))
                if frozen and "%s@%s" % (name, win) in frozen:
                    t = frozen["%s@%s" % (name, win)]
                    rec, fpr = float(np.mean(pos > t)), float(np.mean(neg > t))
                    s[win].update(frozen_recall=round(rec, 4), frozen_fpr=round(fpr, 4),
                                  frozen_precision_5=round(0.05 * rec / (0.05 * rec + 0.95 * fpr), 4) if rec + fpr > 0 else None)
            sc = np.array(x["scored"][name])
            s["scored_island"] = round(float(sc[:, 0].sum() / max(x["n_island"], 1)), 4)
            s["scored_backbone"] = round(float(sc[:, 1].sum() / max(x["n_backbone"], 1)), 4)
            if name in x["found"]:
                f = np.array(x["found"][name]); nl = np.array(x["null"][name])
                s["regions"] = dict(found=int(f[:, 0].sum()), total=int(f[:, 1].sum()), share=round(float(f[:, 0].sum() / f[:, 1].sum()), 4),
                                    shifted=round(float((nl[:, 0] * nl[:, 1]).sum() / nl[:, 1].sum()), 4))
            r["scores"][name] = s
        res[genus] = r
    if a.freeze:                                                  # thresholds on the pooled backbone genes of all genera of this run
        for name in names:
            for win in WINDOWS:
                neg = [np.concatenate(x["neg"][(name, win)]) for x in G.values() if (name, win) in x["neg"]]
                if neg:
                    thr_out["%s@%s" % (name, win)] = M.threshold_at(np.concatenate(neg), 0.01)
        json.dump(thr_out, open(a.freeze, "w"), indent=1)
    ok_g = [g for g, r in res.items() if r["gene_ok"]]
    ok_r = [g for g, r in res.items() if r["region_ok"]]
    summ, cons = {}, {}
    val = lambda g, name, win, m: res[g]["scores"].get(name, {}).get(win, {}).get(m)
    reg = lambda g, name, m: res[g]["scores"].get(name, {}).get("regions", {}).get(m)
    for name in names:
        summ[name] = {}
        for win in WINDOWS:
            for m in GENE_METRICS + (("frozen_recall", "frozen_fpr") if frozen else ()):
                v = [val(g, name, win, m) for g in ok_g]
                summ[name]["%s.%s" % (win, m)] = boot_mean([z for z in v if z is not None], rng, a.boot)
        for m in ("share", "shifted"):
            v = [reg(g, name, m) for g in ok_r]
            summ[name]["regions." + m] = boot_mean([z for z in v if z is not None], rng, a.boot)
    for c in a.contrast:
        x, _, y = c.partition("-")
        cons[c] = {}
        for win in WINDOWS:
            for m in GENE_METRICS:
                d = [val(g, x, win, m) - val(g, y, win, m) for g in ok_g if val(g, x, win, m) is not None and val(g, y, win, m) is not None]
                cons[c]["%s.%s" % (win, m)] = boot_diff(d, rng, a.boot)
        d = [reg(g, x, "share") - reg(g, y, "share") for g in ok_r if reg(g, x, "share") is not None and reg(g, y, "share") is not None]
        cons[c]["regions.share"] = boot_diff(d, rng, a.boot)
    keys = sorted({k for v in cons.values() for k in v})
    for k in keys:                                                # Holm over the contrasts of one metric
        cs = [c for c in cons if cons[c].get(k)]
        if cs:
            for c, p in zip(cs, holm(np.array([cons[c][k]["p"] for c in cs]))):
                cons[c][k]["p_holm"] = round(float(p), 4)
    json.dump(dict(genera=res, summary=summ, contrasts=cons, eligible_gene=ok_g, eligible_region=ok_r, args=vars(a)), open(a.out + ".json", "w"), indent=1)
    f3 = lambda s: "" if not s else "%.3f (%.3f to %.3f)" % (s["mean"], s["lo"], s["hi"])
    L = ["Genera: %d; %d enter the gene-level means (at least %d backbone and %d island genes), %d the region-level means." % (
            len(res), len(ok_g), a.min_backbone, a.min_island, len(ok_r)), "",
         "Gene level, mean over genera (95%% interval by bootstrap over genera). w1 = the gene's own score; w11 = mean rank over the gene and %d neighbours on each side." % a.neighbours, "",
         "| score | recall at 1% backbone FP, w1 | mean recall 0-5%, w1 | AUROC, w1 | recall at 1%, w11 | AUROC, w11 | island / backbone genes scored |", "|---|---|---|---|---|---|---|"]
    for name in names:
        s = summ[name]
        cov = [(res[g]["scores"][name]["scored_island"], res[g]["scores"][name]["scored_backbone"]) for g in res if name in res[g]["scores"]]
        L.append("| %s | %s | %s | %s | %s | %s | %.2f / %.2f |" % (name, f3(s["w1.recall_1"]), f3(s["w1.mean_recall_5"]), f3(s["w1.auroc"]), f3(s["w11.recall_1"]),
                 f3(s["w11.auroc"]), np.mean([c[0] for c in cov]), np.mean([c[1] for c in cov])))
    if ok_r:
        L += ["", "Region level (w11; one value per published region; one backbone region per chromosome may pass):", "",
              "| score | island regions found | the same with all labels shifted along the chromosome |", "|---|---|---|"]
        for name in names:
            L.append("| %s | %s | %s |" % (name, f3(summ[name]["regions.share"]), f3(summ[name]["regions.shifted"])))
    if frozen:
        L += ["", "Thresholds frozen elsewhere (1% of the pooled backbone genes there), applied unchanged:", "",
              "| score | window | backbone false-positive rate, mean over genera | highest genus | recall |", "|---|---|---|---|---|"]
        for name in names:
            for win in WINDOWS:
                v = [val(g, name, win, "frozen_fpr") for g in ok_g]
                v = [z for z in v if z is not None]
                if v:
                    L.append("| %s | %s | %s | %.3f | %s |" % (name, win, f3(summ[name]["%s.frozen_fpr" % win]), max(v), f3(summ[name]["%s.frozen_recall" % win])))
    if cons:
        L += ["", "Paired contrasts over genera (difference, 95% interval, genera won, Holm-adjusted p over the contrasts of the column):", "",
              "| contrast | recall at 1%, w1 | recall at 1%, w11 | AUROC, w1 | island regions found |", "|---|---|---|---|---|"]
        fc = lambda v: "" if not v else "%+.3f (%+.3f to %+.3f), %d of %d, p %.3f" % (v["mean"], v["lo"], v["hi"], v["wins"], v["n"], v.get("p_holm", v["p"]))
        for c, v in cons.items():
            L.append("| %s | %s | %s | %s | %s |" % (c, fc(v.get("w1.recall_1")), fc(v.get("w11.recall_1")), fc(v.get("w1.auroc")), fc(v.get("regions.share"))))
    L += ["", "Per genus, recall at 1% backbone false positives (w1 / w11):", "", "| genus | chromosomes | island genes | backbone genes | in means | " + " | ".join(names) + " |",
          "|---|---|---|---|---|" + "---|" * len(names)]
    for g, r in res.items():
        cell = lambda n: "" if n not in r["scores"] or "w1" not in r["scores"][n] else "%.3f / %.3f" % (r["scores"][n]["w1"]["recall_1"], r["scores"][n]["w11"]["recall_1"])
        L.append("| %s | %d | %d | %d | %s | %s |" % (g, len(r["chromosomes"]), r["n_island"], r["n_backbone"],
                 ("genes" if r["gene_ok"] else "") + (" regions" if r["region_ok"] else ""), " | ".join(cell(n) for n in names)))
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
