"""Benchmarking against simulated truth, calibration and rescoring."""
from __future__ import annotations

import csv
import os

import numpy as np

from .model import (CLOCK_FEATURES, DEFAULT_MODEL, GAIN_FEATURES, TREE_FEATURES, auc, fit_logistic, load_model,
                    predict, save_model, tier)
from .tree import SpeciesTree, parse_newick


def _read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _node_name(label):
    return label.split("{", 1)[0] if label else ""


def load_truth(path):
    truth = []
    for r in _read_tsv(path):
        r["genes"] = set(x for x in r["recipient_genes"].split(",") if x)
        r["detectable"] = r.get("detectable", "1") == "1"
        truth.append(r)
    return truth


def match(events, truth, min_jaccard=0.5):
    """Label each predicted event 1 if it recovers a true transfer (same family, gene overlap)."""
    by_fam = {}
    for i, t in enumerate(truth):
        by_fam.setdefault(t["family"], []).append(i)
    labels, hits = [], []
    for ev in events:
        genes = set(x for x in ev["recipient_genes"].split(",") if x)
        best, bj = None, 0.0
        for i in by_fam.get(_base_family(ev["family"]), []):
            tg = truth[i]["genes"]
            j = len(genes & tg) / float(len(genes | tg)) if genes | tg else 0.0
            if j > bj:
                best, bj = i, j
        ok = best is not None and bj >= min_jaccard
        labels.append(1 if ok else 0)
        hits.append(best if ok else None)
    return labels, hits


def _base_family(fam):
    """Subfamilies from --split-paralogs are named <family>.<k>; truth uses the family."""
    head, _, tail = fam.rpartition(".")
    return head if head and tail.isdigit() else fam


def _genes(txt):
    return set(x for x in (txt or "").split(",") if x)


def match_any(events, truth, min_jaccard=0.5):
    """Like match(), but a clock 'exchange' event may also match on its alternative side.

    Returns (labels, hits, direct): direct[i] = matched on the stated recipient side.
    """
    by_fam = {}
    for i, t in enumerate(truth):
        by_fam.setdefault(t["family"], []).append(i)
    labels, hits, direct = [], [], []
    for ev in events:
        best, bj, bd = None, 0.0, False
        for side, is_main in ((_genes(ev["recipient_genes"]), True), (_genes(ev.get("alt_genes")), False)):
            for i in by_fam.get(_base_family(ev["family"]), []) if side else ():
                tg = truth[i]["genes"]
                j = len(side & tg) / float(len(side | tg))
                if j > bj or (j == bj and is_main and not bd):
                    best, bj, bd = i, j, is_main
        ok = best is not None and bj >= min_jaccard
        labels.append(1 if ok else 0)
        hits.append(best if ok else None)
        direct.append(bd if ok else False)
    return labels, hits, direct


def evaluate(events_path, truth_path, species_tree, genes_path=None, contam_path=None, thr=0.6,
             oracle_path=None):
    events = _read_tsv(events_path)
    truth = load_truth(truth_path)
    with open(species_tree) as fh:
        sp = SpeciesTree(parse_newick(fh.read()))
    labels, hits, direct = match_any(events, truth)
    P = np.array([float(e["P_HGT"]) for e in events]) if events else np.zeros(0)
    fT = np.array([float(e["f_T"]) for e in events]) if events else np.zeros(0)
    y = np.array(labels)
    res = {"n_events": len(events), "n_true_transfers": len(truth),
           "n_true_detectable": sum(t["detectable"] for t in truth)}
    res["event_auc_P"] = auc(P, y)
    res["event_auc_fT_only"] = auc(fT, y)
    called = P >= thr
    tp = int((called & (y == 1)).sum())
    res["threshold"] = thr
    res["called"] = int(called.sum())
    res["precision"] = tp / float(called.sum()) if called.sum() else float("nan")
    # found = right recipient (direction correct); found_any also counts a clock exchange
    # whose recipient is the stated alternative side
    found = set(h for h, c, d in zip(hits, called, direct) if h is not None and c and d)
    found_any = set(h for h, c in zip(hits, called) if h is not None and c)
    det = [i for i, t in enumerate(truth) if t["detectable"]]
    res["recall_detectable"] = len(found & set(det)) / float(len(det)) if det else float("nan")
    res["recall_all"] = len(found) / float(len(truth)) if truth else float("nan")
    by_src = {}
    for e, h, c in zip(events, hits, called):
        if c:
            src = e.get("source", "tree")
            tp_fp = by_src.setdefault(src, [0, 0])
            tp_fp[0 if h is not None else 1] += 1
    res["called_by_source"] = by_src

    # recall split by recipient type: single genome (recent) vs ancestral branch (ancient);
    # "identifiable" = recovered (any P) by a scan of the true, noise-free gene trees
    ident = None
    if oracle_path and os.path.exists(oracle_path):
        _, oh, _ = match_any(_read_tsv(oracle_path), truth)
        ident = set(h for h in oh if h is not None)
    res["recall_by_type"] = {}
    for kind in ("leaf", "internal"):
        idx = [i for i in det if sp.is_leaf[sp.index[truth[i]["recipient_node"]]] == (kind == "leaf")]
        row = {"n": len(idx), "recall": len(found & set(idx)) / float(len(idx)) if idx else float("nan"),
               "recall_any_direction": len(found_any & set(idx)) / float(len(idx)) if idx else float("nan")}
        if ident is not None:
            iid = [i for i in idx if i in ident]
            row["n_identifiable"] = len(iid)
            row["recall_identifiable"] = len(found & set(iid)) / float(len(iid)) if iid else float("nan")
        res["recall_by_type"][kind] = row

    # recall by donor kind: sampled lineage vs. outside the dataset
    res["recall_by_donor_kind"] = {}
    for kind in ("sampled", "unsampled"):
        idx = [i for i in det if truth[i].get("donor_kind", "sampled") == kind]
        if idx:
            res["recall_by_donor_kind"][kind] = {"n": len(idx), "recall": len(found & set(idx)) / float(len(idx))}

    # donor accuracy on correctly called events
    exact = lineage = top3 = n = 0
    for e, h, c, d in zip(events, hits, called, direct):
        if h is None or not c or not d:
            continue
        true_d = sp.index.get(truth[h]["donor_node"])
        cands = [sp.index.get(_node_name(e["donor%d" % k])) for k in (1, 2, 3)]
        if true_d is None or cands[0] is None:      # unsampled donor (truth or gain call)
            continue
        n += 1
        exact += cands[0] == true_d
        lineage += sp.related(cands[0], true_d)
        top3 += true_d in cands
    res["donor_top1_exact"] = exact / float(n) if n else float("nan")
    res["donor_top1_lineage"] = lineage / float(n) if n else float("nan")
    res["donor_top3_exact"] = top3 / float(n) if n else float("nan")

    # calibration table
    bins = [0.0, 0.3, 0.6, 0.9, 1.01]
    res["calibration"] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (P >= lo) & (P < hi)
        if m.sum():
            res["calibration"].append((lo, min(hi, 1.0), int(m.sum()), float(y[m].mean())))

    # gene level
    if genes_path:
        gP = {}
        for e in events:
            pdir = float(e.get("p_direction") or 1.0)
            for genes, w in ((e["recipient_genes"], pdir), (e.get("alt_genes"), 1.0 - pdir)):
                for g in _genes(genes):
                    gP[g] = max(gP.get(g, 0.0), float(e["P_HGT"]) * w)
        true_genes = set().union(*[t["genes"] for t in truth]) if truth else set()
        all_genes = [r["gene_id"] for r in _read_tsv(genes_path)]
        s = [gP.get(g, 0.0) for g in all_genes]
        lab = [1 if g in true_genes else 0 for g in all_genes]
        res["gene_auc"] = auc(s, lab)
        res["gene_true_hgt"] = sum(lab)
        callg = [g for g in all_genes if gP.get(g, 0.0) >= thr]
        res["gene_precision"] = (sum(1 for g in callg if g in true_genes) / float(len(callg))) if callg else float("nan")
        res["gene_recall"] = (sum(1 for g in callg if g in true_genes) / float(sum(lab))) if sum(lab) else float("nan")

    # contamination
    if contam_path and os.path.exists(contam_path):
        cg = set(r["gene_id"] for r in _read_tsv(contam_path))
        flagged = called_c = 0
        for e in events:
            genes = set(e["recipient_genes"].split(","))
            if genes & cg:
                flagged += "possible_contamination" in e["flags"]
                called_c += float(e["P_HGT"]) >= thr
        res["contaminant_genes"] = len(cg)
        res["contaminant_events"] = sum(1 for e in events if set(e["recipient_genes"].split(",")) & cg)
        res["contaminant_events_flagged"] = flagged
        res["contaminant_events_called_hgt"] = called_c
    return res


def print_report(res):
    def f(x):
        return "nan" if x != x else ("%.3f" % x if isinstance(x, float) else str(x))
    print("== event level ==")
    print("candidate events: %s | true transfers: %s (detectable: %s)"
          % (res["n_events"], res["n_true_transfers"], res["n_true_detectable"]))
    print("AUC  P_HGT: %s   (tree-only f_T: %s)" % (f(res["event_auc_P"]), f(res["event_auc_fT_only"])))
    print("at P >= %.2f: called %d, precision %s, recall(detectable) %s, recall(all) %s"
          % (res["threshold"], res["called"], f(res["precision"]), f(res["recall_detectable"]), f(res["recall_all"])))
    if res.get("called_by_source"):
        print("  calls by source: " + ", ".join("%s %d (true %d)" % (k, v[0] + v[1], v[0])
                                               for k, v in sorted(res["called_by_source"].items())))
    for kind, label in (("leaf", "into one genome (recent)"), ("internal", "into ancestral branch (ancient)")):
        r = res["recall_by_type"][kind]
        line = "  recall %-32s n=%-4d %s" % (label, r["n"], f(r["recall"]))
        if r["recall_any_direction"] != r["recall"]:
            line += " (exchange detected, any direction: %s)" % f(r["recall_any_direction"])
        if "recall_identifiable" in r:
            line += " | identifiable from true trees: n=%-4d recall %s" % (r["n_identifiable"], f(r["recall_identifiable"]))
        print(line)
    for kind, r in sorted(res.get("recall_by_donor_kind", {}).items()):
        print("  recall, donor %-28s n=%-4d %s" % ("in the dataset" if kind == "sampled" else "outside the dataset",
                                                  r["n"], f(r["recall"])))
    print("donor (correct calls): top1 exact %s, top1 same lineage %s, top3 exact %s"
          % (f(res["donor_top1_exact"]), f(res["donor_top1_lineage"]), f(res["donor_top3_exact"])))
    print("calibration (P bin -> n, observed fraction true):")
    for lo, hi, n, obs in res["calibration"]:
        print("  [%.1f, %.1f)  n=%-5d true=%.2f" % (lo, hi, n, obs))
    if "gene_auc" in res:
        print("== gene level ==")
        print("true HGT-derived genes: %d | AUC %s | precision %s | recall %s"
              % (res["gene_true_hgt"], f(res["gene_auc"]), f(res["gene_precision"]), f(res["gene_recall"])))
    if "contaminant_genes" in res:
        print("== contamination (eukaryote) ==")
        print("contaminant genes: %d | events touching them: %d | flagged: %d | still called HGT: %d"
              % (res["contaminant_genes"], res["contaminant_events"],
                 res["contaminant_events_flagged"], res["contaminant_events_called_hgt"]))


def calibrate(events_paths, truth_paths, out_path, l2=1.0):
    rows, labels = [], []
    for ep, tp in zip(events_paths, truth_paths):
        ev = _read_tsv(ep)
        lab, _, _ = match_any(ev, load_truth(tp))
        rows += ev
        labels += lab
    models = {}
    for src, feats in (("tree", TREE_FEATURES), ("clock", CLOCK_FEATURES), ("gain", GAIN_FEATURES)):
        idx = [i for i, r in enumerate(rows) if r.get("source", "tree") == src]
        y = [labels[i] for i in idx]
        if sum(y) < 5 or len(y) - sum(y) < 5:
            models[src] = DEFAULT_MODEL["models"][src]
            print("[%s] only %d events / %d true: keeping default weights" % (src, len(y), sum(y)))
            continue
        models[src] = fit_logistic([rows[i] for i in idx], y, feats, l2=l2)
        print("[%s] calibrated on %d events (%d true). standardized coefficients:" % (src, len(y), sum(y)))
        for f, c in sorted(zip(feats, models[src]["coef"]), key=lambda x: -abs(x[1])):
            print("  %-10s %+.2f" % (f, c))
        print("  intercept  %+.2f" % models[src]["intercept"])
    model = {"source": "calibrated logistic regression per candidate source", "models": models}
    save_model(model, out_path)
    print("-> %s" % out_path)
    return model


def rescore(events_path, model_path, out_path):
    with open(events_path) as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        cols = rd.fieldnames
        rows = list(rd)
    model = load_model(model_path)
    P = predict(model, rows)
    for r, p in zip(rows, P):
        r["P_HGT"] = "%.3f" % p
        r["tier"] = tier(float(p))
    rows.sort(key=lambda r: -float(r["P_HGT"]))
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("rescored %d events with %s -> %s" % (len(rows), model.get("source"), out_path))
