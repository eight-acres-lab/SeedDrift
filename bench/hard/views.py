#!/usr/bin/env python
"""Views of the deep hit table of one chromosome, and the scores that need nothing but the table.

A view hides reference proteomes that are close to the genome before any method reads the table:

  genus    proteomes of the genome's own genus (how every method is deployed: the genome's own taxon is its genus),
           and every proteome that is measurably as close as that: closeness s(r) of a reference proteome r = the
           median, over the genes with a hit covering at least 70% of the gene in r, of the best such identity
           (defined when at least 30% of the genes have such a hit). Hidden are the own genus, every r with s(r) at
           least that of the closest own-genus proteome, and every r with s(r) >= 90. A genus name alone would
           leave Shigella visible to an Escherichia genome. --by-name switches the distance part off.
  family   the same plus the proteomes of its family (the same genes as if the lineage were poorly sampled)
  order    the same plus the proteomes of its order

Hiding is by filtering the stored list. A gene whose stored list is full (--limit hits) may have lost visible hits
that a search without the hidden taxa would have returned; such genes are flagged (full = 1) and can be searched
again with the hidden taxa excluded.

Scores, one row per gene of --genes (higher = more foreign; empty = the gene has no value and ranks last):

  ai_all     Alien Index with the phylum as home group: ln of the best home E-value minus ln of the best outside
             E-value (hits covering at least 40% of the gene; a side without a hit counts as E = 1)
  hidx       the same on bit scores per residue (no E-value floor, no ties at the floor)
  ai_narrow  Alien Index with the narrowest home group among family, order and class that still holds at least
             10 visible genera other than the genome's own, else the phylum
  nohit      1 when no visible hit is left (a control: it knows nothing else)
  n_rows     minus the number of visible proteomes with a covering hit
  best_id    minus the best visible identity
  sim_anom   distance of the gene's best visible bit score per residue from the genome's median
  viral      share of viral proteomes among the 10 best visible hits
  copies     stored hits per visible proteome with a hit

usage:
  python bench/hard/views.py --hits hits_ACC.tsv --library library.npz --lineage job_ACC.npz --genes labels/ACC.tsv \\
      --hide genus --out DIR/scores_ACC.tsv
"""
from __future__ import annotations

import argparse
import array
import csv
import math
import os

import numpy as np

LEVEL = {"genus": 6, "family": 5, "order": 4}          # shared depth: 1 domain, 2 phylum, 3 class, 4 order, 5 family, 6 genus
MIN_QCOV = 0.4
VIRUSES = 10239
COLS = ["full", "ai_all", "hidx", "ai_narrow", "nohit", "n_rows", "best_id", "sim_anom", "viral", "copies"]


def shared_depth(lineage, ranks):
    """Deepest rank (1 = domain .. 6 = genus) at which a lineage and every reference proteome have the same node."""
    lineage = np.asarray(lineage)
    same = (ranks == lineage[None, :]) & (lineage[None, :] != 0)
    depth = np.zeros(len(ranks), dtype=np.int64)
    for i in range(6):
        depth[same[:, i]] = i + 1
    return depth


def read_hits(path, taxid_index):
    """Hits of one file as arrays (query number, proteome, identity, E-value, bit score, covered share, query length)."""
    qid, qno = {}, array.array("i")
    p, pid, bits, cov, qlen = array.array("i"), array.array("f"), array.array("f"), array.array("f"), array.array("i")
    loge = array.array("f")
    unknown = 0
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            tt = f[1].rpartition("|")[2]
            pi = taxid_index.get(int(tt)) if tt.isdigit() else None
            if pi is None:
                unknown += 1
                continue
            name = f[0].partition("__")[2] or f[0]
            qno.append(qid.setdefault(name, len(qid)))
            ql = int(f[10])
            p.append(pi); pid.append(float(f[2])); bits.append(float(f[5])); qlen.append(ql)
            cov.append((int(f[7]) - int(f[6]) + 1) / ql)
            loge.append(min(200.0, -math.log10(max(float(f[4]), 1e-200))))
    arr = lambda a, t: np.frombuffer(a, dtype=t) if len(a) else np.zeros(0, dtype=t)
    return list(qid), dict(q=arr(qno, np.int32), p=arr(p, np.int32), pid=arr(pid, np.float32), bits=arr(bits, np.float32),
                           cov=arr(cov, np.float32), qlen=arr(qlen, np.int32), loge=arr(loge, np.float32)), unknown


def closeness(h, n_genes, n_prot, min_cov=0.7, min_share=0.3):
    """s(r) per reference proteome (nan where fewer than min_share of the genes have a covering hit)."""
    out = np.full(n_prot, np.nan)
    ok = h["cov"] >= min_cov
    if not ok.any():
        return out
    q, p, pid = h["q"][ok].astype(np.int64), h["p"][ok].astype(np.int64), h["pid"][ok]
    order = np.lexsort((-pid, q * n_prot + p))
    key = (q * n_prot + p)[order]
    first = np.r_[True, key[1:] != key[:-1]]                  # best identity of every gene in every proteome
    bp, bid = p[order][first], pid[order][first]
    o2 = np.argsort(bp, kind="stable")
    bp, bid = bp[o2], bid[o2]
    cuts = np.r_[0, np.flatnonzero(np.diff(bp)) + 1, len(bp)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= min_share * n_genes:
            out[bp[a]] = float(np.median(bid[a:b]))
    return out


def hidden_set(depth, close, hide, by_distance=True):
    """Rows hidden in a view, and why: own genus (or family / order), as close as the own genus, identity >= 90."""
    hidden = depth >= LEVEL[hide]
    info = dict(by_rank=int(hidden.sum()), as_close_as_genus=0, identity_90=0, closest_own_genus=None)
    if by_distance:
        own = depth >= 6
        with np.errstate(invalid="ignore"):
            if own.any() and np.isfinite(close[own]).any():
                ref = float(np.nanmax(close[own]))
                add = (close >= ref) & ~hidden
                info.update(as_close_as_genus=int(add.sum()), closest_own_genus=round(ref, 2))
                hidden = hidden | add
            add = (close >= 90.0) & ~hidden
        info["identity_90"] = int(add.sum())
        hidden = hidden | add
    return hidden, info


def narrow_home(depth, hidden, genus_of, own_genus, need=10):
    """Narrowest home depth among family, order, class with at least `need` visible genera besides the genome's own; else phylum."""
    for d in (5, 4, 3):
        g = set(genus_of[(depth >= d) & ~hidden].tolist()) - {0, own_genus}
        if len(g) >= need:
            return d
    return 2


def alien(loge, home):
    away = ~home
    return math.log(10.0) * ((loge[away].max() if away.any() else 0.0) - (loge[home].max() if home.any() else 0.0))


def view_scores(names, h, ranks, lineage, hide, limit=1000, n_genes=0, by_distance=True):
    """{gene: {column: value}} for the genes that have a stored hit."""
    depth = shared_depth(lineage, ranks)
    close = closeness(h, n_genes or len(names), len(ranks))
    hidden, hinfo = hidden_set(depth, close, hide, by_distance)
    viral = ranks[:, 0] == VIRUSES
    home_narrow = narrow_home(depth, hidden, ranks[:, 5], int(lineage[5]))
    order = np.lexsort((-h["bits"], h["q"]))
    q = h["q"][order]
    starts = np.r_[0, np.flatnonzero(np.diff(q)) + 1, len(q)] if len(q) else np.zeros(1, dtype=np.int64)
    out = {}
    for a, b in zip(starts[:-1], starts[1:]):
        ix = order[a:b]
        name = names[int(q[a])]
        row = dict(full=float(b - a >= limit))
        vis = ix[~hidden[h["p"][ix]]]
        if not len(vis):
            row["nohit"] = 1.0
            out[name] = row
            continue
        p, bits, ql = h["p"][vis], h["bits"][vis].astype(np.float64), float(h["qlen"][vis[0]])
        ok = h["cov"][vis] >= MIN_QCOV
        row.update(nohit=0.0, best_id=-float(h["pid"][vis].max()), bres=float(bits.max() / ql),
                   viral=float(viral[p[:10]].mean()), copies=float(len(p) / len(np.unique(p))))
        row["n_rows"] = float(-len(np.unique(p[ok])))
        if ok.any():
            d, le, br = depth[p[ok]], h["loge"][vis][ok].astype(np.float64), bits[ok] / ql
            row["ai_all"] = alien(le, d >= 2)
            row["ai_narrow"] = alien(le, d >= home_narrow)
            home = d >= 2
            row["hidx"] = float((br[~home].max() if (~home).any() else 0.0) - (br[home].max() if home.any() else 0.0))
        out[name] = row
    med = np.median([r["bres"] for r in out.values() if "bres" in r]) if any("bres" in r for r in out.values()) else 0.0
    for r in out.values():
        if "bres" in r:
            r["sim_anom"] = abs(r.pop("bres") - med)
    return out, dict(hidden=int(hidden.sum()), home_narrow_depth=int(home_narrow), genes_with_hits=len(out),
                     genes_full=int(sum(r["full"] for r in out.values())), genes_no_visible_hit=int(sum(r.get("nohit", 0) for r in out.values())),
                     hidden_rows=np.flatnonzero(hidden), closeness=close, depth=depth, **hinfo)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hits", required=True, help="deep hits of one chromosome (12 DIAMOND columns, targets named accession|taxid)")
    ap.add_argument("--library", required=True)
    ap.add_argument("--lineage", required=True, help="npz with genome_ranks (domain, phylum, class, order, family, genus)")
    ap.add_argument("--genes", required=True, help="table with a gene_id column: every gene of the chromosome")
    ap.add_argument("--hide", default="genus", choices=list(LEVEL))
    ap.add_argument("--limit", type=int, default=1000, help="--max-target-seqs of the search")
    ap.add_argument("--by-name", action="store_true", help="hide by taxonomic rank only, not also by measured closeness")
    ap.add_argument("--hidden-out", default="", help="write the hidden reference proteomes (taxid, shared depth, closeness) to this table")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lib = np.load(a.library)
    taxid_index = {int(t): i for i, t in reversed(list(enumerate(lib["taxid"])))}
    names, h, unknown = read_hits(a.hits, taxid_index)
    lineage = np.load(a.lineage)["genome_ranks"]
    genes = [r["gene_id"] for r in csv.DictReader(open(a.genes), delimiter="\t")]
    sc, info = view_scores(names, h, lib["ranks"], lineage, a.hide, a.limit, len(genes), not a.by_name)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.hidden_out:
        with open(a.hidden_out, "w") as fo:
            fo.write("taxid\tshared_depth\tcloseness\n")
            for i in info["hidden_rows"]:
                fo.write("%d\t%d\t%s\n" % (int(lib["taxid"][i]), int(info["depth"][i]), "" if info["closeness"][i] != info["closeness"][i] else "%.2f" % info["closeness"][i]))
    with open(a.out, "w") as fo:
        fo.write("gene_id\t" + "\t".join(COLS) + "\n")
        for g in genes:
            r = sc.get(g, dict(full=0.0, nohit=1.0))
            fo.write(g + "\t" + "\t".join("" if c not in r else "%.4f" % r[c] for c in COLS) + "\n")
    missing = len(set(sc) - set(genes))
    print("%s hide=%s: genes %d, with stored hits %d (%d not in the gene table), full lists %d, no visible hit %d of those with hits; "
          "hidden proteomes %d (by rank %d, as close as the own genus [%s] %d, identity >= 90 %d); narrow home depth %d; hits to taxa outside the library %d" % (
              os.path.basename(a.hits), a.hide, len(genes), info["genes_with_hits"], missing, info["genes_full"], info["genes_no_visible_hit"],
              info["hidden"], info["by_rank"], info["closest_own_genus"], info["as_close_as_genus"], info["identity_90"], info["home_narrow_depth"], unknown), flush=True)


if __name__ == "__main__":
    main()
