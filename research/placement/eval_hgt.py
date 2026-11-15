#!/usr/bin/env python
"""Compare the placement model's zero-shot scores with the Alien Index, the confidence rules and the
feature model of bench/fusion_probe.py on the benchmark runs. Same protocol and metrics as the probe.

The placement model has never seen a spiked gene, so nothing is left out for it. The feature model is
trained on the other recipients, as in the probe. Genes without any hit rank last for every score.

usage (repo root; needs scikit-learn and the job folders):
  python research/placement/eval_hgt.py --scores DIR --label broad --out DIR/hgt_eval.json
  (DIR holds scores_<dataset>.tsv written by score.py)
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

PLACE = ["disc", "node", "p1", "p2", "top", "mean5"]
TOKEN_RULES = ["ai_pipe", "hidx_pipe", "ai_tok", "hidx_tok", "lca_tok", "ai_near", "hidx_near", "lca_adapt"]   # rules computed by score.py on the model's tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--results", default="bench/results/results.json")
    ap.add_argument("--config", default="bench/benchmark.json")
    ap.add_argument("--label", default="broad")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="name=DIR: disc and node of other models (ablations, untrained) from their score files")
    a = ap.parse_args()
    runs = json.load(open(a.results))
    tax = get_taxonomy()
    data = {}
    for ds in json.load(open(a.config)):
        run = runs.get(ds["name"], {}).get("full@" + a.label)
        if ds["kind"] in ("spikein", "transplant") and run and run.get("status") == "done":
            data[ds["name"]] = fp.load(ds, run["job"], tax)
    res = {}
    for test, rows in data.items():
        if not os.path.exists(os.path.join(a.scores, "scores_%s.tsv" % test)):
            print("no placement scores for %s: skipped" % test, file=sys.stderr)
            continue
        scores = fp.base_scores(rows)
        donors = {r["donor"] for r in rows if r["y"]}
        train = [r for n in data if n != test for r in data[n] if not (r["y"] and r["donor"] in donors)]
        scores["gbm_hits"] = fp.predict(fp.fit("gbm", fp.HITS, train), rows, fp.HITS)
        sc = {r["gene_id"]: r for r in csv.DictReader(open(os.path.join(a.scores, "scores_%s.tsv" % test)), delimiter="\t")}
        for k in PLACE:
            scores["place_" + k] = np.array([float(sc[r["gene"]][k]) if r["gene"] in sc else -1e9 for r in rows])
        for k in TOKEN_RULES:
            scores["rule_" + k] = np.array([float(sc[r["gene"]][k]) if r["gene"] in sc else -1e9 for r in rows])
        # the Alien Index of the pipeline for every gene that has one, without the cut that turns uncalled genes into ties;
        # and the token Alien Index with its ties (E-value floor) broken by the bit-score difference
        scores["ai_all"] = np.array([r["f"]["AI"] if fp.has_evidence(r) else -1e9 for r in rows])
        scores["rule_ai_pipe_tb"] = scores["rule_ai_pipe"] + 1e-3 * np.where(scores["rule_hidx_pipe"] > -1e8, scores["rule_hidx_pipe"], 0.0)
        extra = []
        for spec in a.extra:
            name, d = spec.split("=", 1)
            sx = {r["gene_id"]: r for r in csv.DictReader(open(os.path.join(d, "scores_%s.tsv" % test)), delimiter="\t")}
            for k in ("disc", "node"):
                scores["%s_%s" % (name, k)] = np.array([float(sx[r["gene"]][k]) if r["gene"] in sx else -1e9 for r in rows])
                extra.append("%s_%s" % (name, k))
        r = fp.evaluate(rows, scores)
        y = np.array([x["y"] for x in rows])
        div = np.array([x["div"] for x in rows])
        k_high = r["high_tier_fp"]
        for k in ("place_disc", "place_node", "rule_ai_tok", "rule_ai_near"):
            thr = fp.recall_at_fp(scores[k], y, k_high)[1]
            r["by_divergence"][k] = {d: round(float(np.mean(scores[k][(y == 1) & (div == d)] > thr)), 3)
                                     for d in sorted(set(div[y == 1]), key=float)}
        r["scored_by_placement"] = dict(positives=int(sum(x["gene"] in sc for x in rows if x["y"])),
                                        negatives=int(sum(x["gene"] in sc for x in rows if not x["y"])))
        # the same comparison on the genes whose token set is exact (the others are scored from a cut-off list,
        # a state the model never saw in training) plus the genes without any hit, which rank last for every score
        keep = [i for i, x in enumerate(rows) if x["gene"] not in sc or sc[x["gene"]]["exact"] == "1"]
        sub = fp.evaluate([rows[i] for i in keep], {k: v[keep] for k, v in scores.items()})
        r["exact_only"] = dict(n_pos=sub["n_pos"], n_neg=sub["n_neg"], high_tier_fp=sub["high_tier_fp"], high_tier_recall=sub["high_tier_recall"],
                               scores={k: {m: v[m] for m in ("ap_core", "ap_all", "recall_at_high_fp", "recall_at_1pct_fp")} for k, v in sub["scores"].items()})
        r["top_negatives"] = {k: fp.top_negatives(rows, scores[k], 25) for k in ("rules", "rule_ai_near", "place_disc", "place_node")}
        res[test] = r
        print("\n### %s  (%d positives, %d core, %d negatives; high tier: recall %.3f with %d false positives)" % (
            test, r["n_pos"], r["n_core"], r["n_neg"], r["high_tier_recall"], r["high_tier_fp"]))
        cols = ["ai", "ai_all", "rules", "gbm_hits", "rule_ai_pipe_tb"] + ["rule_" + k for k in TOKEN_RULES] + ["place_" + k for k in PLACE] + extra
        print("\n".join(fp.table(r, cols)))
        print("\n".join(fp.core_table(r, cols)))
        for k, v in r["by_divergence"].items():
            print("- recall at high-tier FP by divergence, %s: %s" % (k, ", ".join("%s: %.2f" % kv for kv in v.items())))
        e = r["exact_only"]
        print("- genes with an exact token set (plus genes without hits): %d positives, %d negatives; AP vs all / recall at high-tier FP: %s" % (
            e["n_pos"], e["n_neg"], ", ".join("%s %.3f / %.3f" % (k, e["scores"][k]["ap_all"], e["scores"][k]["recall_at_high_fp"])
                                              for k in ("ai", "rules", "gbm_hits", "rule_ai_tok", "rule_ai_near", "place_disc", "place_node"))))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
