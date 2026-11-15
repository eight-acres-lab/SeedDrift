#!/usr/bin/env python
"""Score files (gene_id + any score columns, higher = more foreign) against the spike-in benchmark, with the
same tables as bench/fusion_probe.py. Genes missing from a score file rank last for that score.

usage (repo root; needs scikit-learn and the job folders):
  python research/lineage/eval_bench.py --scores lineage=RUN [place=DIR ...] --out RUN/bench.json
  (each folder holds scores_<dataset>.tsv)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "bench"))
import fusion_probe as fp  # noqa: E402
from seeddrift.pipeline import get_taxonomy  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", nargs="+", required=True, help="label=DIR")
    ap.add_argument("--columns", default="", help="comma-separated score columns to use (default: all)")
    ap.add_argument("--results", default="bench/results/results.json")
    ap.add_argument("--config", default="bench/benchmark.json")
    ap.add_argument("--label", default="broad")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = json.load(open(a.results))
    tax = get_taxonomy()
    want = set(a.columns.split(",")) if a.columns else None
    res = {}
    for ds in json.load(open(a.config)):
        run = runs.get(ds["name"], {}).get("full@" + a.label)
        if ds["kind"] not in ("spikein", "transplant") or not run or run.get("status") != "done":
            continue
        rows = fp.load(ds, run["job"], tax)
        scores = fp.base_scores(rows)
        scores["ai_all"] = np.array([r["f"]["AI"] if fp.has_evidence(r) else -1e9 for r in rows])
        for spec in a.scores:
            label, d = spec.split("=", 1)
            path = os.path.join(d, "scores_%s.tsv" % ds["name"])
            if not os.path.exists(path):
                continue
            sc = {r["gene_id"]: r for r in csv.DictReader(open(path), delimiter="\t")}
            for col in next(iter(sc.values())):
                if col != "gene_id" and (want is None or col in want):
                    try:
                        scores["%s.%s" % (label, col)] = np.array([float(sc[r["gene"]][col]) if r["gene"] in sc else -1e9 for r in rows])
                    except ValueError:
                        pass
        r = fp.evaluate(rows, scores)
        y = np.array([x["y"] for x in rows]); div = np.array([x["div"] for x in rows])
        for k in scores:
            thr = fp.recall_at_fp(scores[k], y, r["high_tier_fp"])[1]
            r["by_divergence"][k] = {d: round(float(np.mean(scores[k][(y == 1) & (div == d)] > thr)), 3) for d in sorted(set(div[y == 1]), key=float)}
        r["top_negatives"] = {k: fp.top_negatives(rows, scores[k], 20) for k in scores if "." in k}
        res[ds["name"]] = r
        cols = list(scores)
        print("\n### %s  (%d positives, %d core, %d negatives; high tier: recall %.3f with %d false positives)" % (
            ds["name"], r["n_pos"], r["n_core"], r["n_neg"], r["high_tier_recall"], r["high_tier_fp"]))
        print("\n".join(fp.table(r, cols)))
        print("\n".join(fp.core_table(r, cols)))
        for k in cols:
            print("- recall at high-tier FP by divergence, %s: %s" % (k, ", ".join("%s: %.2f" % kv for kv in r["by_divergence"][k].items())))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
