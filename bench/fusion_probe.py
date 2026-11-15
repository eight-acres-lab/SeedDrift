#!/usr/bin/env python
"""Probe: can a learned fusion of the per-gene evidence rank foreign genes better than the hand-set
confidence rules?

Reads the benchmark runs that already exist (bench/results/results.json -> job folders with genes.tsv
and hits.tsv) and their truth tables; nothing is searched again.

  scores     ai         Alien Index of the called genes (the score behind the benchmark's AUPRC)
             rules      confidence tier (high > medium > low > no call), Alien Index within a tier
             lr_rule    logistic regression on the five quantities the rules use
             gbm_rule   gradient-boosted trees on the same five quantities
             gbm_table  + the other evidence columns of genes.tsv
             gbm_hits   + the shape of the hit list (hits.tsv)
  protocol   leave one dataset out: train on the other recipients, test on the held-out one; training
             positives from a donor species that also occurs in the test set are dropped.
             Same-recipient cross-validation (donors kept apart) is reported as an optimistic bound.
  metrics    average precision against core genes and against all other genes; recall at the number of
             false positives that the "high" tier (and "high or medium") makes; paired bootstrap intervals

Features that would only recognise the simulation are left out: gene ids, gene length, coordinates and
hits inside the recipient's own genus (a spiked gene never has any). Genes without any outside homolog
are not trained on and always rank last.

usage (repo root, on the machine that holds the job folders; needs scikit-learn):
  python bench/fusion_probe.py --results bench/results/results.json --label broad --out bench/results
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))
from run_benchmark import FOREIGN, match_transplant  # noqa: E402
from seeddrift import config  # noqa: E402
from seeddrift.pipeline import get_taxonomy  # noqa: E402
from seeddrift.single import MIN_QCOV, home_group  # noqa: E402

NAN = float("nan")
TIER = {"high": 3, "medium": 2, "low": 1}
DOMAINS = (2, 2157, 2759, 10239)                     # Bacteria, Archaea, Eukaryota, Viruses
RULE = ["AI", "support", "tree", "gap", "has_home"]
TABLE = RULE + ["donor_share", "n_outside", "repeat_frac", "best_pident", "best_home_pident", "log_evalue",
                "tree_frac", "deep"]
HITS = TABLE + ["h_bits_res", "h_qcov", "h_tcov", "h_len_ratio", "h_away_frac10", "h_first_home", "h_home_n",
                "h_bits_gap", "h_groups10", "h_species10", "h_pid_drop", "h_bits_drop", "h_cross_domain"]
CONTEXT = HITS + ["comp_abs", "island"]              # DNA input only
MODELS = [("lr_rule", "lr", RULE), ("gbm_rule", "gbm", RULE), ("gbm_table", "gbm", TABLE), ("gbm_hits", "gbm", HITS)]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# -- features ----------------------------------------------------------------------------------------
def table_features(r, has_coords):
    ai, bp, hp = num(r.get("AI")), num(r.get("best_pident")), num(r.get("best_home_pident"))
    k, _, n = (r.get("tree_outside") or "").partition("/")
    nout, ev, z = num(r.get("n_outside")), num(r.get("best_evalue")), num(r.get("comp_z"))
    return dict(AI=ai, support=num(r.get("support")),
                tree={"outside": 2.0, "mixed": 1.0, "home": 0.0}.get(r.get("tree") or "", NAN),
                gap=bp - hp, has_home=NAN if math.isnan(ai) else float(not math.isnan(hp)),
                donor_share=num(r.get("donor_share")), n_outside=nout,
                repeat_frac=num(r.get("repeat_hits")) / nout if nout > 0 else NAN,
                best_pident=bp, best_home_pident=hp,
                log_evalue=-math.log10(max(ev, 1e-200)) if not math.isnan(ev) else NAN,
                tree_frac=int(k) / int(n) if n else NAN, deep=float((r.get("search") or "") == "deep"),
                comp_abs=abs(z), island=(num(r.get("island")) if r.get("island") else 0.0) if has_coords else NAN)


def hit_features(path, tax, self_taxid, home_t):
    """Shape of the hit list of every query: hits with enough coverage, outside the recipient's own lineage
    (as in single.classify_gene; the dedicated home-group search is not part of hits.tsv)."""
    self_dom = next((d for d in DOMAINS if d in tax.lineage(self_taxid)), 0)
    info, by = {}, {}
    with open(path) as fh:
        ix = {k: i for i, k in enumerate(fh.readline().rstrip("\n").split("\t"))}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            qlen, tlen = int(f[ix["qlen"]]), int(f[ix["tlen"]])
            qcov = (int(f[ix["qend"]]) - int(f[ix["qstart"]]) + 1) / qlen
            if qcov < MIN_QCOV:
                continue
            t = int(f[ix["taxid"]])
            if t not in info:
                lin = set(tax.lineage(t))
                info[t] = (self_taxid in lin, home_t in lin, tax.group(t), tax.at_rank(t, "species") or t,
                           next((d for d in DOMAINS if d in lin), 0))
            if info[t][0]:
                continue
            by.setdefault(f[ix["query"]], []).append(
                (float(f[ix["bits"]]), t, float(f[ix["pident"]]), qcov, int(f[ix["alnlen"]]) / tlen, tlen / qlen, qlen))
    out = {}
    for q, hs in by.items():
        hs.sort(key=lambda h: -h[0])
        b, qlen = hs[0], hs[0][6]
        home = [h for h in hs if info[h[1]][1]]
        away = [h for h in hs if not info[h[1]][1]]
        f = dict(h_bits_res=b[0] / qlen, h_qcov=b[3], h_tcov=b[4], h_len_ratio=b[5],
                 h_away_frac10=sum(not info[h[1]][1] for h in hs[:10]) / len(hs[:10]),
                 h_first_home=next((i + 1.0 for i, h in enumerate(hs) if info[h[1]][1]), NAN), h_home_n=float(len(home)),
                 h_bits_gap=(away[0][0] - home[0][0]) / qlen if away and home else NAN)
        if away:
            a = away[:10]
            f.update(h_groups10=float(len({info[h[1]][2] for h in a})), h_species10=float(len({info[h[1]][3] for h in a})),
                     h_pid_drop=a[0][2] - a[-1][2], h_bits_drop=(a[0][0] - away[min(4, len(away) - 1)][0]) / a[0][0],
                     h_cross_domain=float(info[a[0][1]][4] != self_dom))
        out[q] = f
    return out


def load(ds, job, tax):
    jd = os.path.join(config.JOBS, job)
    pred, truth = read_tsv(os.path.join(jd, "genes.tsv")), read_tsv(os.path.join(ROOT, ds["dir"], "truth.tsv"))
    if ds["kind"] == "transplant":
        pred, truth = match_transplant(pred, truth)
    summ = json.load(open(os.path.join(jd, "summary.json")))
    home_t, _ = home_group(tax, summ["self_taxid"], summ.get("params", {}).get("home_rank", "auto"))
    hf = hit_features(os.path.join(jd, "hits.tsv"), tax, summ["self_taxid"], home_t)
    by = {r["gene_id"]: r for r in pred}
    rows = []
    for t in truth:
        r = by.get(t["gene_id"])
        if r is None:
            continue
        f = table_features(r, bool(summ.get("has_coords")))
        f.update(hf.get(r["gene_id"], {}))
        rows.append(dict(y=int(t["label"] == "spike"), kind=t["label"], donor=t.get("donor_name") or "",
                         div=t.get("divergence") or "", conf=r.get("confidence") or "", cat=r.get("category") or "", f=f,
                         gene=r["gene_id"], best=r.get("best_taxon") or "", group=r.get("donor_group") or ""))
    return rows


# -- scores and metrics ------------------------------------------------------------------------------
def matrix(rows, cols):
    return np.array([[r["f"].get(c, NAN) for c in cols] for r in rows], dtype=float)


def has_evidence(r):
    """An Alien Index exists: the gene has a homolog outside its own lineage and is not repeat-driven."""
    return not math.isnan(r["f"]["AI"])


def predict(model, rows, cols):
    """Model score; genes without evidence are ranked last, as the rules do. Otherwise "no hit at all" becomes
    a predictor of the spiked genes (strongly diverged spikes lose all their hits; real orphans are just orphans)."""
    p = model.predict_proba(matrix(rows, cols))[:, 1]
    return np.where([has_evidence(r) for r in rows], p, -1.0)


def base_scores(rows):
    ai = np.array([r["f"]["AI"] if r["cat"] in FOREIGN and not math.isnan(r["f"]["AI"]) else -1e9 for r in rows])
    tier = np.array([TIER.get(r["conf"], 0) for r in rows])
    return dict(ai=ai, rules=np.where(ai > -1e8, tier * 1e4 + np.clip(ai, 0, 9999), -1e9))


def fit(kind, cols, train):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    train = [r for r in train if has_evidence(r)]
    y = np.array([r["y"] for r in train])
    if kind == "lr":
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=5000))
    else:
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=8, min_samples_leaf=25,
                                           l2_regularization=1.0, early_stopping=False, random_state=0)
    return m.fit(matrix(train, cols), y)


def recall_at_fp(s, y, k):
    """Recall when the threshold is set so that at most k negatives pass (ties count against the score)."""
    neg = np.sort(s[y == 0])[::-1]
    thr = neg[k] if k < len(neg) else -np.inf
    return float(np.mean(s[y == 1] > thr)), thr


def metrics(s, y, core, k_high, k_hm):
    from sklearn.metrics import average_precision_score as ap
    m = (y == 1) | core
    return dict(ap_core=round(float(ap(y[m], s[m])), 3), ap_all=round(float(ap(y, s)), 3),
                recall_at_high_fp=round(recall_at_fp(s, y, k_high)[0], 3),
                recall_at_high_or_medium_fp=round(recall_at_fp(s, y, k_hm)[0], 3),
                recall_at_1pct_fp=round(recall_at_fp(s, y, int(0.01 * int((y == 0).sum())))[0], 3))


def boot_diff(a, b, y, k_high, n=300, seed=0):
    """Paired bootstrap (resampling genes within each class) of score a minus score b."""
    from sklearn.metrics import average_precision_score as ap
    rnd = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d_ap, d_r = [], []
    for _ in range(n):
        i = np.concatenate([rnd.choice(pos, len(pos)), rnd.choice(neg, len(neg))])
        d_ap.append(ap(y[i], a[i]) - ap(y[i], b[i]))
        d_r.append(recall_at_fp(a[i], y[i], k_high)[0] - recall_at_fp(b[i], y[i], k_high)[0])
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]  # noqa: E731
    return dict(ap_all=q(d_ap), recall_at_high_fp=q(d_r))


def evaluate(rows, scores):
    y = np.array([r["y"] for r in rows])
    core = np.array([r["kind"] == "core" for r in rows])
    conf = np.array([r["conf"] for r in rows])
    k_high = int(((y == 0) & (conf == "high")).sum())
    k_hm = int(((y == 0) & np.isin(conf, ("high", "medium"))).sum())
    out = dict(n_pos=int(y.sum()), n_core=int(core.sum()), n_neg=int((y == 0).sum()), high_tier_fp=k_high,
               high_tier_recall=round(float(np.mean(conf[y == 1] == "high")), 3), high_or_medium_fp=k_hm,
               high_or_medium_recall=round(float(np.mean(np.isin(conf[y == 1], ("high", "medium")))), 3),
               scores={k: metrics(s, y, core, k_high, k_hm) for k, s in scores.items()}, by_divergence={})
    for k in scores:
        if k not in ("ai", "rules"):
            out["scores"][k]["minus_rules_ci95"] = boot_diff(scores[k], scores["rules"], y, k_high)
    # the same comparison against core genes only (clean negatives, but few): each rule tier fixes how many
    # core genes may pass, every score is read at that number
    ci = np.where((y == 1) | core)[0]
    out["core_matched"] = {}
    for name, t in (("high", ("high",)), ("high_or_medium", ("high", "medium")), ("any_call", ("high", "medium", "low"))):
        k = int((core & np.isin(conf, t)).sum())
        out["core_matched"][name] = dict(core_fp=k, tier_recall=round(float(np.mean(np.isin(conf[y == 1], t))), 3),
                                         recall={s: round(recall_at_fp(v[ci], y[ci], k)[0], 3) for s, v in scores.items()})
    div = np.array([r["div"] for r in rows])
    for k in ("rules", "gbm_hits", "gbm_context"):
        if k in scores:
            thr = recall_at_fp(scores[k], y, k_high)[1]
            out["by_divergence"][k] = {d: round(float(np.mean(scores[k][(y == 1) & (div == d)] > thr)), 3)
                                       for d in sorted(set(div[y == 1]), key=float)}
    return out


def importance(model, rows, cols, top=6):
    from sklearn.inspection import permutation_importance
    y = np.array([r["y"] for r in rows])
    pi = permutation_importance(model, matrix(rows, cols), y, scoring="average_precision", n_repeats=5, random_state=0)
    return [[cols[i], round(float(pi.importances_mean[i]), 3)] for i in np.argsort(-pi.importances_mean)[:top]]


def top_negatives(rows, s, n=30):
    """The negatives a score ranks highest: unlabelled real transfers of the recipient, or mistakes."""
    order = np.argsort(-s)
    out, pos = [], 0
    for i in order:
        r = rows[i]
        if r["y"]:
            pos += 1
            continue
        f = r["f"]
        out.append(dict(gene=r["gene"], kind=r["kind"], name=r["donor"], category=r["cat"], confidence=r["conf"],
                        AI=None if math.isnan(f["AI"]) else f["AI"], gap=None if math.isnan(f["gap"]) else round(f["gap"], 1),
                        donor_group=r["group"], best_taxon=r["best"], score=round(float(s[i]), 3), positives_above=pos))
        if len(out) == n:
            break
    return out


def cross_validate(rows, sets, folds=5, seed=0):
    """Same-recipient cross-validation: donors are kept apart, the recipient's own genes are not."""
    rnd = np.random.default_rng(seed)
    donors = sorted({r["donor"] for r in rows if r["y"]})
    dfold = {d: i % folds for i, d in enumerate(rnd.permutation(donors))}
    fold = np.array([dfold[r["donor"]] if r["y"] else int(rnd.integers(folds)) for r in rows])
    scores = base_scores(rows)
    for name, cols in sets.items():
        s = np.zeros(len(rows))
        for k in range(folds):
            te = np.where(fold == k)[0]
            m = fit("gbm", cols, [rows[i] for i in np.where(fold != k)[0]])
            s[te] = predict(m, [rows[i] for i in te], cols)
        scores[name] = s
    return evaluate(rows, scores)


# -- report ------------------------------------------------------------------------------------------
def table(res, keys):
    cols = ["ap_core", "ap_all", "recall_at_high_fp", "recall_at_high_or_medium_fp", "recall_at_1pct_fp"]
    out = ["| score | AP vs core | AP vs all | recall at high-tier FP | recall at high+medium FP | recall at 1% FP | AP vs all minus rules (95%) | recall at high-tier FP minus rules (95%) |",
           "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        if k not in res["scores"]:
            continue
        m = res["scores"][k]
        ci = m.get("minus_rules_ci95")
        out.append("| %s | %s | %s | %s |" % (k, " | ".join("%.3f" % m[c] for c in cols),
                                             "%+.3f … %+.3f" % tuple(ci["ap_all"]) if ci else "",
                                             "%+.3f … %+.3f" % tuple(ci["recall_at_high_fp"]) if ci else ""))
    return out


def core_table(res, keys):
    keys = [k for k in keys if k in res["scores"]]
    out = ["| core genes allowed through | rule tier recall | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for name, m in res["core_matched"].items():
        out.append("| %d of %d (%s) | %.3f | %s |" % (m["core_fp"], res["n_core"], name.replace("_", " "), m["tier_recall"],
                                                     " | ".join("%.3f" % m["recall"][k] for k in keys)))
    return out


def report(res):
    keys = ["ai", "rules"] + [m[0] for m in MODELS] + ["gbm_context"]
    L = ["# Evidence-fusion probe", "",
         "Generated by `bench/fusion_probe.py` from the benchmark runs in `bench/results/results.json` (no new searches).",
         "Question: does a learned combination of the per-gene evidence rank foreign genes better than the hand-set",
         "confidence rules? All positives are synthetic (spike-ins, DNA transplant), so this measures headroom in the",
         "features, not performance on real transfers.", "",
         "- `ai`: Alien Index of the called genes. `rules`: confidence tier, then Alien Index.",
         "- `lr_rule` / `gbm_rule`: logistic regression / gradient-boosted trees on the five quantities the rules use",
         "  (%s)." % ", ".join(RULE),
         "- `gbm_table`: plus the other evidence columns of the gene table. `gbm_hits`: plus the shape of the hit list.",
         "- Recall is read at the number of false positives that the rule tiers make on the same genes, so the",
         "  columns compare like with like. \"All\" negatives include unlabelled real transfers of the recipient",
         "  (prophage genes of E. coli K-12, bacterial-type metabolic enzymes of Entamoeba: see `top_negatives` in",
         "  `bench/results/fusion_probe.json`), and these sit at the very top of every ranking. The tables against",
         "  core genes use clean negatives, but only a few dozen of them.", "",
         "## Leave one recipient out", ""]
    for name, r in res["leave_one_out"].items():
        L += ["### test: %s (trained on %s)" % (name, ", ".join(r["train"])), "",
              "%d positives, %d core genes, %d negatives in all. High tier: recall %.3f with %d false positives; "
              "high or medium: recall %.3f with %d." % (r["n_pos"], r["n_core"], r["n_neg"], r["high_tier_recall"],
                                                        r["high_tier_fp"], r["high_or_medium_recall"], r["high_or_medium_fp"]), ""]
        L += table(r, keys) + ["", "Recall when only as many core genes pass as in each rule tier:", ""] + core_table(r, keys) + [""]
        for k, v in r["by_divergence"].items():
            L.append("- recall at high-tier FP by divergence, %s: %s" % (k, ", ".join("%s: %.2f" % kv for kv in v.items())))
        L += ["- most important features of gbm_hits (drop in AP when shuffled): "
              + ", ".join("%s %.3f" % tuple(x) for x in r["importance"]), ""]
    L += ["## Same recipient, donors kept apart (optimistic bound)", ""]
    for name, r in res["same_recipient_cv"].items():
        L += ["### %s" % name, ""] + table(r, keys) + [""] + core_table(r, keys) + [""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="bench/results/results.json")
    ap.add_argument("--config", default="bench/benchmark.json")
    ap.add_argument("--label", default="broad")
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--report", default="docs/fusion-probe.md")
    a = ap.parse_args()
    runs = json.load(open(a.results))
    tax = get_taxonomy()
    data = {}
    for ds in json.load(open(a.config)):
        run = runs.get(ds["name"], {}).get("full@" + a.label if a.label else "full")
        if ds["kind"] in ("spikein", "transplant") and run and run.get("status") == "done":
            data[ds["name"]] = load(ds, run["job"], tax)
            print("[probe] %s: %d genes, %d positives" % (ds["name"], len(data[ds["name"]]),
                                                         sum(r["y"] for r in data[ds["name"]])), file=sys.stderr)
    res = dict(features=dict(rule=RULE, table=TABLE, hits=HITS, context=CONTEXT), leave_one_out={}, same_recipient_cv={})
    for test, rows in data.items():
        donors = {r["donor"] for r in rows if r["y"]}
        others = [n for n in data if n != test]
        train = [r for n in others for r in data[n] if not (r["y"] and r["donor"] in donors)]
        scores, models = base_scores(rows), {}
        for name, kind, cols in MODELS:
            models[name] = fit(kind, cols, train)
            scores[name] = predict(models[name], rows, cols)
        r = evaluate(rows, scores)
        r.update(train=others, n_train=len(train), n_train_pos=sum(x["y"] for x in train),
                 importance=importance(models["gbm_hits"], rows, HITS),
                 top_negatives={k: top_negatives(rows, scores[k]) for k in ("rules", "gbm_hits")})
        res["leave_one_out"][test] = r
        sets = dict(gbm_hits=HITS)
        if any(not math.isnan(x["f"]["island"]) for x in rows):
            sets["gbm_context"] = CONTEXT
        res["same_recipient_cv"][test] = cross_validate(rows, sets)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "fusion_probe.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    md = report(res)
    if a.report:
        with open(a.report, "w") as fh:
            fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
