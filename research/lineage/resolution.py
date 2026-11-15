#!/usr/bin/env python
"""How finely does a pooled language-model state resolve sequence similarity?

Every query has homologs at several score ranks, each with the identity and bit score of its alignment to the
query. For two homologs of the same query, the question is whether the state says the same one is closer as
the alignment does. Reported by the size of the gap between the two alignments, for the raw cosine of each
layer and for a small learned comparator (trained on other queries).

usage:
  python research/lineage/resolution.py --queries emb_queries.npz --homologs emb_homologs.npz --pairs homologs.tsv
"""
from __future__ import annotations

import argparse
import csv
import zlib

import numpy as np
import torch
import torch.nn as nn

GAPS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 101)]        # identity points
RGAPS = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 9)]   # difference of bits per residue


class Comparator(nn.Module):
    def __init__(self, d_in, d=256):
        super().__init__()
        self.p = nn.Linear(d_in, d)
        self.m = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, 2))

    def forward(self, a, b):
        a, b = self.p(a), self.p(b)
        return self.m(torch.cat([a * b, (a - b).abs()], -1))


def order_table(score, truth, qix, gaps, test):
    """Share of homolog pairs of one query that the score orders as the alignment does, by size of the gap."""
    out, by = [], {}
    for i in np.flatnonzero(test):
        by.setdefault(qix[i], []).append(i)
    ia, ib = [], []
    for v in by.values():
        for x in range(len(v)):
            for y in range(x + 1, len(v)):
                ia.append(v[x]); ib.append(v[y])
    ia, ib = np.array(ia), np.array(ib)
    gap = np.abs(truth[ia] - truth[ib])
    ok = np.sign(score[ia] - score[ib]) == np.sign(truth[ia] - truth[ib])
    for lo, hi in gaps:
        m = (gap > lo) & (gap <= hi) if lo else (gap > 0) & (gap <= hi)
        out.append((int(m.sum()), float(ok[m].mean()) if m.any() else float("nan")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True); ap.add_argument("--homologs", required=True); ap.add_argument("--pairs", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    zq, zh = np.load(a.queries), np.load(a.homologs)
    qi = {k: i for i, k in enumerate(zq["ids"])}; hi = {k: i for i, k in enumerate(zh["ids"])}
    qlen = zq["length"]
    rows = [r for r in csv.DictReader(open(a.pairs), delimiter="\t") if r["query"] in qi and r["homolog"] in hi]
    Q = np.array([qi[r["query"]] for r in rows]); Hm = np.array([hi[r["homolog"]] for r in rows])
    ident = np.array([float(r["identity"]) for r in rows]); bpr = np.array([float(r["bits"]) for r in rows]) / qlen[Q]
    test = np.array([zlib.crc32(r["query"].encode()) % 5 == 0 for r in rows])
    print("pairs %d (test %d), queries %d" % (len(rows), test.sum(), len(set(Q.tolist()))))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eq = torch.tensor(zq["emb"][Q].astype(np.float32)); eh = torch.tensor(zh["emb"][Hm].astype(np.float32))
    scores = {}
    for l, name in enumerate(("layer 6", "layer 20", "layer 33")):
        scores["cosine, " + name] = torch.cosine_similarity(eq[:, l], eh[:, l], dim=-1).numpy()
    xa, xb = eq.flatten(1), eh.flatten(1)
    mu, sd = xa[~test].mean(0), xa[~test].std(0) + 1e-6
    xa, xb = ((xa - mu) / sd).to(dev), ((xb - mu) / sd).to(dev)
    y = torch.tensor(np.stack([ident / 100.0, np.minimum(bpr, 2.5) / 2.5], 1), dtype=torch.float32, device=dev)
    tr = torch.tensor(np.flatnonzero(~test), device=dev)
    torch.manual_seed(0)
    net = Comparator(xa.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    for ep in range(a.epochs):
        perm = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(perm), 1024):
            b = perm[i:i + 1024]
            loss = nn.functional.mse_loss(net(xa[b], xb[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(xa[i:i + 8192], xb[i:i + 8192]) for i in range(0, len(xa), 8192)]).cpu().numpy()
    scores["learned comparator (identity)"] = pred[:, 0]; scores["learned comparator (bits)"] = pred[:, 1]
    r = np.corrcoef(pred[test, 0], ident[test] / 100)[0, 1]
    print("learned comparator on held-out queries: correlation with identity %.3f, mean absolute error %.1f points" % (
        r, 100 * np.abs(pred[test, 0] - ident[test] / 100).mean()))
    for truth, gaps, title, fmt in ((ident, GAPS, "gap in identity (points)", "%g-%g"), (bpr, RGAPS, "gap in bits per residue", "%g-%g")):
        print("\n| which of two homologs is closer: agreement with the alignment, by %s | %s |" % (title, " | ".join(fmt % g for g in gaps)))
        print("|---|" + "---|" * len(gaps))
        n = None
        for k, s in scores.items():
            t = order_table(s, truth, Q, gaps, test)
            n = n or t
            print("| %s | %s |" % (k, " | ".join("%.3f" % v for _, v in t)))
        print("| pairs | %s |" % " | ".join(str(c) for c, _ in n))


if __name__ == "__main__":
    main()
