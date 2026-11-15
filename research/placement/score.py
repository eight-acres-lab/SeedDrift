#!/usr/bin/env python
"""Zero-shot foreign-gene scores for one genome from the placement model.

For every gene the model says where it attaches among its selected hits. The genome's own lineage says how
deep a rank the GENOME shares with each hit. A gene placed closer to a lineage than its genome is to that
lineage is a foreign candidate; the lineage is the donor. Scores as in trials.py (disc, node, p1, p2, top, mean5).

`exact` is 1 when the gene's token set is the one a search of the library without the genome's own genus
would select (see placenet.make_example); other genes are still scored, from what the stored list holds.

usage (needs torch):
  python research/placement/score.py --library L.npz --model RUN/model.pt --job job_x.npz --out scores_x.tsv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from placenet import COLS, Library, Runner, discordance, load_model, make_example, offpath, pack  # noqa: E402
from trials import rule_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hide", type=int, default=6, help="hide library proteomes sharing at least this rank with the genome (6 = own genus)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lib, z = Library(a.library), np.load(a.job)
    limit = int(z["limit"])
    g_rel = lib.relation_to(z["genome_ranks"])
    hidden = g_rel >= a.hide
    own = lib.pidx.get(int(z["self_taxid"]))
    if own is not None:
        hidden[own] = True
    if not hidden.any():
        sys.exit("nothing to hide: the genome's lineage has no rank %d and its taxon is not a library proteome; "
                 "training always hides the query's own proteome, so scores would not be comparable" % a.hide)
    off, cols = z["off"], {k: z["h_" + k] for k in COLS}
    hp_all, q_len, q_id = z["h_p"], z["q_len"], z["q_id"]
    ex, genes, exact = [], [], []
    for q in range(len(q_id)):
        s, e = off[q], off[q + 1]
        if e == s:
            continue
        hp, d, qlen = hp_all[s:e], {k: v[s:e] for k, v in cols.items()}, float(q_len[q])
        x = make_example(lib, hp, d, qlen, hidden, limit)
        ok = x is not None
        x = x or make_example(lib, hp, d, qlen, hidden, limit, strict=False)
        if x is None:
            continue
        ex.append(x); genes.append(q); exact.append(ok)
    X, P, NID = pack(ex, lib)
    prob, node = Runner(lib, dev).predict(load_model(a.model, dev), X, P, NID)
    pad, Pc = P < 0, np.clip(P, 0, None)
    s_tax = g_rel[Pc].astype(np.float32)
    r = discordance(prob, P, s_tax)
    r["node"] = offpath(node, NID, P, s_tax)
    best = r["exc"].argmax(1)
    # rule baselines on the same tokens (trials.py): home = the nearest group of the genome that the library still samples
    near = int(g_rel[~hidden].max())
    rules = []
    for f, tp in ex:
        top = f[:, 1] >= 0.9
        sub = lib.S[np.ix_(tp[top], tp[top])].astype(int)
        clade = int(sub[~np.eye(len(sub), dtype=bool)].min()) if len(sub) > 1 else 6
        rules.append(rule_scores(f, g_rel[tp].astype(int), near, top, clade))
    with open(a.out, "w") as fo:
        fo.write("gene_id\tn_hits\texact\tdisc\tnode\tp1\tp2\ttop\tmean5\tai_near\thidx_near\tlca_adapt\tdonor_taxid\tdonor_pred_depth\tdonor_genome_depth\n")
        for g, q in enumerate(genes):
            b = best[g]
            fo.write("%s\t%d\t%d\t%.3f\t%.3f\t%.4f\t%.4f\t%.3f\t%.3f\t%.2f\t%.4f\t%.1f\t%d\t%.2f\t%d\n" % (
                q_id[q], int((~pad[g]).sum()), exact[g], r["disc"][g], r["node"][g], r["p1"][g], r["p2"][g], r["top"][g], r["mean5"][g],
                rules[g]["ai_near"], rules[g]["hidx_near"], rules[g]["lca_adapt"], lib.taxid[Pc[g, b]], r["ev"][g, b], int(s_tax[g, b])))
    print("genes scored %d of %d (%d with an exact token set); library proteomes hidden %d; stored list limit %d; "
          "nearest group of the genome still in the library: depth %d" % (len(genes), len(q_id), sum(exact), int(hidden.sum()), limit, near), file=sys.stderr)


if __name__ == "__main__":
    main()
