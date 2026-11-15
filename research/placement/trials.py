#!/usr/bin/env python
"""Transfer trials without a simulator (protocol in README.md).

A protein from proteome A of a held-out lineage is searched once. A decoy proteome B from another class
plays the host. The lineages of A and of B are both hidden to the same level, giving one set of tokens.
  native trial   the protein sits in A: scored against A's lineage
  foreign trial  the same protein, same tokens, sits in B: scored against B's lineage
Model and rule baselines read exactly the same tokens.

Scores (higher = more foreign), G = the genome the trial is scored against, s_i = depth G shares with hit i:
  disc       max_i ( E[depth the model says the protein shares with hit i] - s_i )          (registered primary)
  node       log odds that the protein attaches at a node that is not an ancestor of G      (added later: exploratory)
  p1, p2, top, mean5   other read-outs of the per-hit output
  ai, hidx   Alien Index / bits-per-residue difference, home = hits sharing the phylum with G (registered; as the pipeline does)
  lca        LCA of the hits within 10% of the best bit score: depth of that clade minus the depth G shares with it (registered)
  *_adapt    home = the nearest group of G that the library still samples, at most the phylum      (exploratory)
  *_near     the same without the phylum cap: home can be as narrow as G's order or family          (exploratory)
  lca_adapt  the LCA rule with the clade depth capped at the nearest group of G still sampled       (exploratory)

Statistics per stratum:
  auroc      foreign trials of the stratum against clean native trials (marker proteins)
  recall     share of the foreign trials above the score that 1% of those natives exceed (ties count against;
             the realised false-positive rate is stored)
  paired     over the token sets of marker proteins in the stratum: share in which the foreign trial scores
             above the native trial of the SAME tokens (ties half). Nothing but the reference genome differs.

usage (needs torch):
  python research/placement/trials.py --library L.npz --data train_a.npz --markers markers.txt \
      --models full=RUN_full/model.pt nopair=RUN_nopair/model.pt mlp=RUN_mlp/model.pt --out trials.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from placenet import LEVELS, Library, Runner, discordance, load_model, make_example, offpath, pack  # noqa: E402
from train import BENCH, LIST_LIMIT, queries, split_of, stored_limit  # noqa: E402

LN10 = float(np.log(10))
RULES = ["ai", "hidx", "lca", "ai_adapt", "hidx_adapt", "ai_near", "hidx_near", "lca_adapt"]
MODEL_SCORES = ["disc", "node", "p1", "p2", "top", "mean5"]
DECOY = {0: "another domain", 1: "another phylum", 2: "another class"}


def auroc(pos, neg):
    """Probability that a positive scores above a negative (ties count half)."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    x = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    rank = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def recall_at(pos, neg, fpr):
    """(share of positives above the score that the given share of negatives exceeds, realised share of negatives above it)."""
    if len(pos) == 0 or len(neg) == 0:
        return None, None
    k = int(fpr * len(neg))
    thr = np.sort(neg)[::-1][k] if k < len(neg) else -np.inf
    return float(np.mean(pos > thr)), float(np.mean(neg > thr))


def rule_scores(f, s, near, top, clade):
    """Rule baselines on one token set against one genome. f: token features; s: depth the genome shares with each
    token; near: deepest depth at which the genome still has a relative in the library; top: tokens within 10% of
    the best bit score; clade: depth of their LCA."""
    loge, bres = f[:, 5].astype(np.float64) * 200.0, f[:, 0].astype(np.float64)
    out = {}
    for name, h in (("", 2), ("_adapt", min(2, near)), ("_near", near)):
        home, away = s >= h, s < h
        out["ai" + name] = LN10 * ((loge[away].max() if away.any() else 0.0) - (loge[home].max() if home.any() else 0.0))
        out["hidx" + name] = (bres[away].max() if away.any() else 0.0) - (bres[home].max() if home.any() else 0.0)
    out["lca"] = float(clade - s[top].min())
    out["lca_adapt"] = float(min(clade, near) - s[top].min())
    return out


def stats(scores, pos_sel, neg_sel, mk):
    out = {}
    pair = pos_sel & mk
    for k, v in scores.items():
        pos, neg = v["for"][pos_sel], v["nat"][neg_sel]
        r1, f1 = recall_at(pos, neg, 0.01)
        d = v["for"][pair] - v["nat"][pair]
        out[k] = dict(auroc=auroc(pos, neg), recall_fpr1=r1, realised_fpr1=f1, recall_fpr5=recall_at(pos, neg, 0.05)[0],
                      paired=float(np.mean(d > 0) + 0.5 * np.mean(d == 0)) if len(d) else None,
                      auroc_marker_positives=auroc(v["for"][pair], neg))
    return dict(n_foreign=int(pos_sel.sum()), n_clean_native=int(neg_sel.sum()), n_pairs=int(pair.sum()), scores=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="name=path; the first is the primary model")
    ap.add_argument("--markers", required=True, help="file with the ids of queries from marker families (clean natives)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--list-limit", type=int, default=LIST_LIMIT)
    ap.add_argument("--max-queries", type=int, default=0, help="per split, for a quick run")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lib = Library(a.library)
    split = split_of(lib)
    r = lib.ranks
    class_key = np.where(r[:, 2] != 0, r[:, 2], np.where(r[:, 1] != 0, r[:, 1], r[:, 0]))
    lineage = np.full(lib.P, "", dtype=object)
    for name, (lin, depth) in BENCH.items():
        lineage[lib.relation_to(lin) >= depth] = name
    markers = set(open(a.markers).read().split())
    limit = stored_limit(a.data, a.list_limit)
    by_depth = {}                                           # decoy candidates per proteome, by shared depth
    ex, meta, n_q = [], [], {2: 0, 3: 0}
    for name, A, hp, d, qlen in queries(a.data, lib):
        sp = int(split[A])
        if sp not in (2, 3) or (a.max_queries and n_q[sp] >= a.max_queries):
            continue
        n_q[sp] += 1
        relA = lib.S[A]
        if A not in by_depth:
            by_depth[A] = [np.flatnonzero(relA == t) for t in (0, 1, 2)]
        for m in LEVELS:
            for t in (0, 1, 2):
                cand = by_depth[A][t]
                if len(cand) == 0:
                    continue
                B = int(cand[zlib.crc32(("%s|%d|%d" % (name, m, t)).encode()) % len(cand)])
                relB = lib.S[B]
                hidden = (relA >= m) | (relB >= m)
                e = make_example(lib, hp, d, qlen, hidden, limit)
                if e is None:
                    continue
                f, tp = e
                sA, sB = relA[tp].astype(int), relB[tp].astype(int)
                left = ~hidden
                row = dict(split=sp, A=A, B=B, level=m, decoy_depth=t, oracle=int((sA - sB).max()), marker=name in markers)
                top = f[:, 1] >= 0.9                          # hits within 10% of the best bit score
                sub = lib.S[np.ix_(tp[top], tp[top])].astype(int)
                clade = int(sub[~np.eye(len(sub), dtype=bool)].min()) if len(sub) > 1 else 6
                for g, s, rel in (("nat", sA, relA), ("for", sB, relB)):
                    row[g] = rule_scores(f, s, int(rel[left].max()) if left.any() else 0, top, clade)
                ex.append(e); meta.append(row)
    X, P, NID = pack(ex, lib)
    print("trials: %d token sets from %s queries (unseen classes, benchmark lineages); marker ids %d" % (len(ex), [n_q[2], n_q[3]], len(markers)), file=sys.stderr)
    run = Runner(lib, dev)
    Pc = np.clip(P, 0, None)
    A_ix, B_ix = np.array([x["A"] for x in meta]), np.array([x["B"] for x in meta])
    sA, sB = lib.S[A_ix[:, None], Pc].astype(np.float32), lib.S[B_ix[:, None], Pc].astype(np.float32)
    scores = {}                                             # name -> {"nat": [E], "for": [E]}
    for spec in a.models:
        mname, path = spec.split("=", 1)
        prob, node = run.predict(load_model(path, dev), X, P, NID)
        for g, s in (("nat", sA), ("for", sB)):
            res = discordance(prob, P, s)
            res["node"] = offpath(node, NID, P, s)
            for k in MODEL_SCORES:
                scores.setdefault("%s.%s" % (mname, k), {})[g] = res[k]
    for k in RULES:
        scores[k] = {g: np.array([x[g][k] for x in meta]) for g in ("nat", "for")}
    sp = np.array([x["split"] for x in meta]); lv = np.array([x["level"] for x in meta]); dd = np.array([x["decoy_depth"] for x in meta])
    oracle = np.array([x["oracle"] for x in meta]); mk = np.array([x["marker"] for x in meta])
    first = a.models[0].split("=", 1)[0]
    primary = first + ".disc"
    others = [k for k in scores if k.endswith(".disc") and k != primary]
    shown = [primary, first + ".node"] + others + RULES
    out = dict(n_token_sets=len(meta), primary=primary, markers_file=a.markers, n_marker_ids=len(markers),
               registered=[primary] + others + ["ai", "hidx", "lca"], splits={})
    rnd = np.random.default_rng(0)
    for s, sname in ((2, "unseen_class"), (3, "bench_lineage")):
        in_s = sp == s
        det, clean = in_s & (oracle >= 1), in_s & mk
        strata = [("all levels, oracle >= 1", det, clean)] + \
                 [("oracle = %d" % o, in_s & (oracle == o), clean) for o in (1, 2, 3)] + \
                 [("oracle >= 4", in_s & (oracle >= 4), clean), ("oracle <= 0: no signal in principle (null)", in_s & (oracle <= 0), clean)] + \
                 [("hide >= %d, oracle >= 1, natives of the same level" % m, det & (lv == m), clean & (lv == m)) for m in LEVELS] + \
                 [("decoy in %s, oracle >= 1" % DECOY[t], det & (dd == t), clean) for t in (0, 1, 2)]
        if s == 3:
            strata += [("%s lineage, oracle >= 1, its own natives" % n, det & (lineage[A_ix] == n), clean & (lineage[A_ix] == n)) for n in BENCH]
        res = dict(n_token_sets=int(in_s.sum()), n_source_proteomes=len(set(A_ix[in_s])), n_marker_proteomes=len(set(A_ix[clean])),
                   strata={label: stats(scores, p, n, mk) for label, p, n in strata})
        if s == 2 and a.boot:
            # cluster bootstrap over the held-out classes: primary minus every other score, on the detectable trials
            keys = class_key[A_ix]
            groups = np.unique(keys[in_s])
            ip = {g: np.flatnonzero(det & (keys == g)) for g in groups}
            ineg = {g: np.flatnonzero(clean & (keys == g)) for g in groups}
            diff = {k: dict(auroc=[], recall_fpr1=[]) for k in shown if k != primary}
            for _ in range(a.boot):
                pick = rnd.choice(groups, len(groups))
                p, n = np.concatenate([ip[g] for g in pick]), np.concatenate([ineg[g] for g in pick])
                if len(p) == 0 or len(n) == 0:
                    continue
                base = (auroc(scores[primary]["for"][p], scores[primary]["nat"][n]), recall_at(scores[primary]["for"][p], scores[primary]["nat"][n], 0.01)[0])
                for k in diff:
                    diff[k]["auroc"].append(base[0] - auroc(scores[k]["for"][p], scores[k]["nat"][n]))
                    diff[k]["recall_fpr1"].append(base[1] - recall_at(scores[k]["for"][p], scores[k]["nat"][n], 0.01)[0])
            res["primary_minus_other_ci95"] = {k: {m: [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])] for m, v in d.items() if v}
                                               for k, d in diff.items()}
            res["n_classes"] = int(len(groups))
        out["splits"][sname] = res
        print("\n## %s: %d token sets, %d source proteomes (%d with marker proteins)" % (sname, res["n_token_sets"], res["n_source_proteomes"], res["n_marker_proteomes"]))
        print("| stratum | foreign | clean natives | pairs | " + " | ".join(shown) + " |")
        print("|---|---|---|---|" + "---|" * len(shown))
        fmt = lambda x: "–" if x is None else "%.2f" % x  # noqa: E731
        for label, row in res["strata"].items():
            cells = ["%s / %s / %s" % ("–" if row["scores"][k]["auroc"] is None else "%.3f" % row["scores"][k]["auroc"],
                                       fmt(row["scores"][k]["recall_fpr1"]), fmt(row["scores"][k]["paired"])) for k in shown]
            print("| %s | %d | %d | %d | %s |" % (label, row["n_foreign"], row["n_clean_native"], row["n_pairs"], " | ".join(cells)))
        print("cells: AUROC against clean natives / recall at 1% false positives / share of identical token sets in which the foreign trial scores higher.")
        if "primary_minus_other_ci95" in res:
            print("%s minus other, 95%% interval over %d held-out classes: %s" % (primary, res["n_classes"], json.dumps(res["primary_minus_other_ci95"])))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
