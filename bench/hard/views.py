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
that a search without the hidden taxa would have returned. Such genes are searched once more to --deep-limit hits
(--deep-hits); their deeper list replaces the stored one. A view keeps the first --limit visible hits of a gene and
is exact (exact = 1) when that many visible hits remain or when the stored list was not full.

Scores, one row per gene of --genes (higher = more foreign; empty = the gene has no value and ranks last):

  ai_all     Alien Index with the phylum as home group: ln of the best home E-value minus ln of the best outside
             E-value (hits covering at least 40% of the gene; a side without a hit counts as E = 1)
  hidx       the same on bit scores per residue (no E-value floor, no ties at the floor)
  ai_narrow  Alien Index with the narrowest home group among family, order and class that still holds at least
             10 visible genera other than the genome's own, else the phylum
  ai_gene    Alien Index with the home rank chosen per gene: the narrowest of family, order, class, phylum in which
             the gene has a visible covering hit
  nohit      1 when no visible hit is left (a control: it knows nothing else)
  n_rows     minus the number of visible proteomes with a covering hit (0, its highest value, for a gene without one)
  absent_near  1 minus the best bit score per residue (relative to 2 bits per residue, clipped to 0-1) among the 12
             visible proteomes that are closest to the genome; 1 for a gene without a hit there
  best_id    minus the best visible identity
  sim_anom   distance of the gene's best visible bit score per residue from the genome's median
  viral      share of viral proteomes among the 10 best visible hits
  copies     visible hits per visible proteome with a hit
  random     a number from SHA-256("<gene id>|random-v1")

--hits-out writes the visible hits themselves (the same 12 columns), for methods that build their own tables.

usage:
  python bench/hard/views.py --hits hits_ACC.tsv --library library.npz --lineage job_ACC.npz --genes genes/ACC.tsv --prefix ACC__ \\
      --hide genus --out DIR/scores_ACC.tsv
"""
from __future__ import annotations

import argparse
import array
import csv
import hashlib
import math
import os
import sys

import numpy as np

LEVEL = {"genus": 6, "family": 5, "order": 4}          # shared depth: 1 domain, 2 phylum, 3 class, 4 order, 5 family, 6 genus
MIN_QCOV = 0.4
VIRUSES = 10239
COLS = ["full", "exact", "ai_all", "hidx", "ai_narrow", "ai_gene", "nohit", "n_rows", "absent_near", "best_id", "sim_anom", "viral", "copies", "random"]
NEAR_ROWS = 12


def shared_depth(lineage, ranks):
    """Deepest rank (1 = domain .. 6 = genus) at which a lineage and every reference proteome have the same node."""
    lineage = np.asarray(lineage)
    same = (ranks == lineage[None, :]) & (lineage[None, :] != 0)
    depth = np.zeros(len(ranks), dtype=np.int64)
    for i in range(6):
        depth[same[:, i]] = i + 1
    return depth


def read_hits(path, taxid_index, deep_path="", prefix=""):
    """Hits as arrays (query number, proteome, identity, E-value, bit score, covered share, query length, source file,
    line), and the number of stored lines per gene before any taxon filter. Genes that the deeper file lists take their
    hits from it. prefix: the "<accession>__" that the search put before the gene ids."""
    qid, qno, stored = {}, array.array("i"), {}
    p, pid, bits, cov, qlen = array.array("i"), array.array("f"), array.array("f"), array.array("f"), array.array("i")
    loge, src, line_no = array.array("f"), array.array("b"), array.array("i")
    unknown, deep = 0, set()

    def read(fp, source, skip):
        nonlocal unknown
        with open(fp) as fh:
            for i, line in enumerate(fh):
                f = line.rstrip("\n").split("\t")
                name = f[0][len(prefix):] if prefix and f[0].startswith(prefix) else f[0]
                if name in skip:
                    continue
                if source == 1:
                    deep.add(name)
                stored[name] = stored.get(name, 0) + 1
                tt = f[1].rpartition("|")[2]
                pi = taxid_index.get(int(tt)) if tt.isdigit() else None
                if pi is None:
                    unknown += 1
                    continue
                qno.append(qid.setdefault(name, len(qid)))
                ql = int(f[10])
                p.append(pi); pid.append(float(f[2])); bits.append(float(f[5])); qlen.append(ql)
                cov.append((int(f[7]) - int(f[6]) + 1) / ql)
                loge.append(min(200.0, -math.log10(max(float(f[4]), 1e-200))))
                src.append(source); line_no.append(i)

    if deep_path:
        read(deep_path, 1, set())
    read(path, 0, deep)
    arr = lambda a, t: np.frombuffer(a, dtype=t) if len(a) else np.zeros(0, dtype=t)
    return list(qid), dict(q=arr(qno, np.int32), p=arr(p, np.int32), pid=arr(pid, np.float32), bits=arr(bits, np.float32),
                           cov=arr(cov, np.float32), qlen=arr(qlen, np.int32), loge=arr(loge, np.float32), src=arr(src, np.int8),
                           line=arr(line_no, np.int32)), unknown, deep, stored


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
        with np.errstate(invalid="ignore"):
            own = (depth >= 6) & (close < 99.0)               # the genome's own strain may or may not be a library proteome: rows at 99 or more set no reference
            if own.any():
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


def gene_random(name):
    return int.from_bytes(hashlib.sha256(("%s|random-v1" % name).encode()).digest()[:8], "big") / 2.0 ** 64


def view_scores(names, h, ranks, lineage, hide, limit=1000, n_genes=0, by_distance=True, deep=(), deep_limit=4000, taxid=None, stored=None):
    """({gene: {column: value}} for the genes that have a stored hit, information on the view, mask of the hits kept)."""
    depth = shared_depth(lineage, ranks)
    close = closeness(h, n_genes or len(names), len(ranks))
    hidden, hinfo = hidden_set(depth, close, hide, by_distance)
    viral = ranks[:, 0] == VIRUSES
    home_narrow = narrow_home(depth, hidden, ranks[:, 5], int(lineage[5]))
    with np.errstate(invalid="ignore"):
        cand = np.flatnonzero(~hidden & np.isfinite(close))
    near = np.zeros(len(ranks), dtype=bool)
    tie = taxid[cand] if taxid is not None else cand
    near[cand[np.lexsort((tie, -close[cand]))[:NEAR_ROWS]]] = True               # the visible proteomes closest to the genome; ties to the lower taxid
    order = np.lexsort((-h["bits"], h["q"]))
    q = h["q"][order]
    starts = np.r_[0, np.flatnonzero(np.diff(q)) + 1, len(q)] if len(q) else np.zeros(1, dtype=np.int64)
    kept = np.zeros(len(h["q"]), dtype=bool)
    out = {}
    for a, b in zip(starts[:-1], starts[1:]):
        ix = order[a:b]
        name = names[int(q[a])]
        cap = deep_limit if name in deep else limit
        n_stored = stored.get(name, b - a) if stored else b - a   # lines of the search output, counted before any taxon filter
        vis_all = ix[~hidden[h["p"][ix]]]
        vis = vis_all[:limit]
        kept[vis] = True
        row = dict(full=float(n_stored >= cap), exact=float(len(vis_all) >= limit or n_stored < cap), absent_near=1.0, n_rows=0.0)
        if not len(vis):
            row["nohit"] = 1.0
            out[name] = row
            continue
        p, bits, ql = h["p"][vis], h["bits"][vis].astype(np.float64), float(h["qlen"][vis[0]])
        ok = h["cov"][vis] >= MIN_QCOV
        row.update(nohit=0.0, best_id=-float(h["pid"][vis].max()), bres=float(bits.max() / ql),
                   viral=float(viral[p[:10]].mean()), copies=float(len(p) / len(np.unique(p))))
        row["n_rows"] = float(-len(np.unique(p[ok])))
        nr = near[p]
        if nr.any():
            row["absent_near"] = 1.0 - float(np.clip(bits[nr].max() / (2.0 * ql), 0.0, 1.0))
        if ok.any():
            d, le, br = depth[p[ok]], h["loge"][vis][ok].astype(np.float64), bits[ok] / ql
            row["ai_all"] = alien(le, d >= 2)
            row["ai_narrow"] = alien(le, d >= home_narrow)
            row["ai_gene"] = alien(le, d >= next((r for r in (5, 4, 3) if (d >= r).any()), 2))
            home = d >= 2
            row["hidx"] = float((br[~home].max() if (~home).any() else 0.0) - (br[home].max() if home.any() else 0.0))
        out[name] = row
    med = np.median([r["bres"] for r in out.values() if "bres" in r]) if any("bres" in r for r in out.values()) else 0.0
    for r in out.values():
        if "bres" in r:
            r["sim_anom"] = abs(r.pop("bres") - med)
    return out, dict(hidden=int(hidden.sum()), home_narrow_depth=int(home_narrow), genes_with_hits=len(out),
                     genes_full=int(sum(r["full"] for r in out.values())), genes_exact=int(sum(r["exact"] for r in out.values())),
                     genes_no_visible_hit=int(sum(r.get("nohit", 0) for r in out.values())),
                     hidden_rows=np.flatnonzero(hidden), closeness=close, depth=depth, near_rows=int(near.sum()), **hinfo), kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hits", required=True, help="deep hits of one chromosome (12 DIAMOND columns, targets named accession|taxid)")
    ap.add_argument("--library", required=True)
    ap.add_argument("--lineage", required=True, help="npz with genome_ranks (domain, phylum, class, order, family, genus)")
    ap.add_argument("--genes", required=True, help="the label-free gene table of the chromosome (genes/<accession>.tsv of bench/islands/labels.py)")
    ap.add_argument("--prefix", default="", help="what the search put before the gene ids, e.g. NC_000000.1__")
    ap.add_argument("--hide", default="genus", choices=list(LEVEL))
    ap.add_argument("--limit", type=int, default=1000, help="--max-target-seqs of the search, and the number of visible hits a view keeps")
    ap.add_argument("--deep-hits", default="", help="the deeper search of the genes whose list was full (same format)")
    ap.add_argument("--deep-limit", type=int, default=4000, help="--max-target-seqs of the deeper search")
    ap.add_argument("--hits-out", default="", help="write the visible hits (the 12 columns) to this file")
    ap.add_argument("--by-name", action="store_true", help="hide by taxonomic rank only, not also by measured closeness")
    ap.add_argument("--hidden-out", default="", help="write the hidden reference proteomes (taxid, shared depth, closeness) to this table")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lib = np.load(a.library)
    taxid_index = {int(t): i for i, t in reversed(list(enumerate(lib["taxid"])))}
    table = list(csv.DictReader(open(a.genes), delimiter="\t"))
    if table and ("label" in table[0] or "product" in table[0] or "class" in table[0]):
        sys.exit("--genes must be the label-free gene table (genes/<accession>.tsv), not a label file")
    genes = [r["gene_id"] for r in table]
    names, h, unknown, deep, stored = read_hits(a.hits, taxid_index, a.deep_hits, a.prefix)
    lineage = np.load(a.lineage)["genome_ranks"]
    if not int(lineage[5]):
        sys.exit("the lineage gives no genus: nothing of the genome's own lineage would be hidden")
    strangers = set(names) - set(genes)
    if strangers:
        sys.exit("%d query ids of the hit file are not in the gene table (for example %s); is --prefix right?" % (len(strangers), sorted(strangers)[0]))
    sc, info, kept = view_scores(names, h, lib["ranks"], lineage, a.hide, a.limit, len(genes), not a.by_name, deep, a.deep_limit, lib["taxid"], stored)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.hidden_out:
        with open(a.hidden_out, "w") as fo:
            fo.write("taxid\tshared_depth\tcloseness\n")
            for i in info["hidden_rows"]:
                fo.write("%d\t%d\t%s\n" % (int(lib["taxid"][i]), int(info["depth"][i]), "" if info["closeness"][i] != info["closeness"][i] else "%.2f" % info["closeness"][i]))
    with open(a.out, "w") as fo:
        fo.write("gene_id\t" + "\t".join(COLS) + "\n")
        for g in genes:
            r = dict(sc.get(g, dict(full=0.0, exact=1.0, nohit=1.0, absent_near=1.0, n_rows=0.0)), random=gene_random(g))
            fo.write(g + "\t" + "\t".join("" if c not in r else "%.4f" % r[c] for c in COLS) + "\n")
    if a.hits_out:
        with open(a.hits_out, "w") as fo:
            for source, path in ((1, a.deep_hits), (0, a.hits)):
                if not path:
                    continue
                want = set(h["line"][kept & (h["src"] == source)].tolist())
                with open(path) as fh:
                    for i, line in enumerate(fh):
                        if i in want:
                            fo.write(line)
    missing = len(set(sc) - set(genes))
    print("%s hide=%s: genes %d, with stored hits %d (%d not in the gene table), searched deeper %d, full lists %d, exact views %d, no visible hit %d of those with hits; "
          "hidden proteomes %d (by rank %d, as close as the own genus [%s] %d, identity >= 90 %d); narrow home depth %d; hits to taxa outside the library %d" % (
              os.path.basename(a.hits), a.hide, len(genes), info["genes_with_hits"], missing, len(deep), info["genes_full"], info["genes_exact"], info["genes_no_visible_hit"],
              info["hidden"], info["by_rank"], info["closest_own_genus"], info["as_close_as_genus"], info["identity_90"], info["home_narrow_depth"], unknown), flush=True)


if __name__ == "__main__":
    main()
