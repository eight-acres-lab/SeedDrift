"""Extra evidence for foreign-gene candidates, beyond best hits, and a confidence level.

  phylogeny     MAFFT + FastTree of the gene, its closest clean outside hits and >= 2 closest home hits;
                the 5 references nearest to the gene (patristic) decide: outside (>= 4 outside the home
                group), home (>= 3 inside) or mixed
  composition   GC and GC3 of the coding sequence vs the genome (robust z; genes >= 300 bp)
  context       clusters of foreign candidates along a contig (gaps of <= 3 genes): islands
  contamination eukaryote assemblies: contigs on which nearly all classified genes look prokaryotic
                are more likely bacterial contamination than transfer
  confidence    high: AI >= 45, >= 2 donor species, and either no relative at all in the home group
                (even after the home-group search) or the phylogeny outside with an identity gap
                (outside - home) >= 10 points; medium: AI >= 45 or phylogeny outside, phylogeny not
                home, and no home relative or a gap >= 5 points; low: the other candidates and those
                on contaminant contigs. Tuned on the spike-in sets; check on held-out data.
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict

from .reconcile import tree_distances
from .tree import parse_newick, unrooted_adjacency

FOREIGN = ("mammal", "prokaryote", "eukaryote", "virus")
QUERY = "QUERY"


def tree_jobs(rows, by_q, tseqs, prots, tax, self_taxid, home_t, treg_rep, n_out=20, n_home=10, per_species=2):
    """{gene: {label: protein}} for the candidates; labels of references are 'acc|taxid'."""
    jobs = {}
    for r in rows:
        if r["category"] not in FOREIGN:
            continue
        g = r["gene_id"]
        out, home, seen = [], [], Counter()
        for h in sorted(by_q.get(g, []), key=lambda h: -h["bits"]):
            if h["qcov"] < 0.4 or h["target"] not in tseqs or (g, h["target"]) in treg_rep:
                continue
            lin = tax.lineage(h["taxid"])
            if self_taxid and self_taxid in lin:
                continue
            sp = tax.at_rank(h["taxid"], "species") or h["taxid"]
            if seen[sp] >= per_species:
                continue
            bucket = home if home_t in lin else out
            cap = n_home if bucket is home else n_out
            if len(bucket) < cap:
                bucket.append(h)
                seen[sp] += 1
        if len(out) + len(home) < 3 or len(home) < 2:     # without home relatives the tree cannot test anything
            continue
        seqs = {QUERY: prots[g]}
        for h in out + home:
            seqs["%s|%d" % (h["target"], h["taxid"])] = tseqs[h["target"]]
        jobs[g] = seqs
    return jobs


def tree_verdict(newick, tax, home_t, k=5):
    empty = dict(tree="na", tree_group="", tree_outside="", tree_taxa="")
    if not newick:
        return empty
    try:
        adj, names = unrooted_adjacency(parse_newick(newick))
    except Exception:                          # noqa: BLE001 - malformed tree
        return empty
    q = next((i for i, n in enumerate(names) if n == QUERY), None)
    refs = [i for i, n in enumerate(names) if n and n != QUERY]
    if q is None or len(refs) < 3:
        return empty
    d = tree_distances(adj, q)
    near = sorted(refs, key=lambda j: d.get(j, 1e9))[:k]
    tx = [int(names[j].rsplit("|", 1)[1]) for j in near]
    n_out = sum(1 for t in tx if home_t not in tax.lineage(t))
    verdict = "outside" if n_out >= max(3, len(near) - 1) else ("home" if len(near) - n_out >= 3 else "mixed")
    grp = Counter(tax.group(t) for t in tx).most_common(1)[0][0]
    return dict(tree=verdict, tree_group=grp, tree_outside="%d/%d" % (n_out, len(near)),
                tree_taxa="; ".join(tax.name.get(t, str(t)) for t in tx))


def composition(cds, min_len=300):
    """{gene: (gc, gc3, z)}: z = the larger |robust z| of GC and GC3, signed (negative: AT-richer)."""
    from .inputs import gc_stats
    st = {g: gc_stats(s) for g, s in cds.items() if len(s) >= min_len}
    if len(st) < 30:
        return {}
    out = {}
    for i in (0, 1):
        vals = [v[i] for v in st.values()]
        med = statistics.median(vals)
        mad = statistics.median(abs(x - med) for x in vals) * 1.4826 or 1e-6
        for g, v in st.items():
            z = (v[i] - med) / mad
            if g not in out or abs(z) > abs(out[g][2]):
                out[g] = (v[0], v[1], z)
    return out


def islands(rows, coords, gap=3):
    """{gene: size of its cluster of foreign candidates} along contigs (gene order by start)."""
    foreign = {r["gene_id"] for r in rows if r["category"] in FOREIGN}
    by_contig = defaultdict(list)
    for g, c in coords.items():
        by_contig[c[0]].append((c[1], g))
    size = {}
    for contig, genes in by_contig.items():
        genes.sort()
        order = [g for _, g in genes]
        idx = [i for i, g in enumerate(order) if g in foreign]
        cluster = []
        for i in idx + [None]:
            if cluster and (i is None or i - cluster[-1] > gap + 1):
                for j in cluster:
                    size[order[j]] = len(cluster)
                cluster = []
            if i is not None:
                cluster.append(i)
    return size


def contaminant_contigs(rows, coords, min_genes=3, frac=0.8):
    per = defaultdict(Counter)
    for r in rows:
        c = coords.get(r["gene_id"])
        if c and r["category"] not in ("orphan", "repeat"):
            per[c[0]][r["category"] == "prokaryote"] += 1
    return {c for c, n in per.items() if n[True] >= min_genes and n[True] >= frac * (n[True] + n[False])}


def confidence(r):
    """high / medium / low for a candidate (see the module docstring); '' for non-candidates."""
    if r["category"] not in FOREIGN:
        return ""
    if r.get("contig_flag"):
        return "low"
    ai = float(r["AI"] or 0)
    tree = r.get("tree") or "na"
    no_home = not r.get("best_home_pident")              # not even the home-group search found a relative
    gap = 100.0 if no_home else float(r["best_pident"] or 0) - float(r["best_home_pident"])
    if ai >= 45 and int(r.get("support") or 0) >= 2 and (no_home or (tree == "outside" and gap >= 10)):
        return "high"
    if (ai >= 45 or tree == "outside") and tree != "home" and (no_home or gap >= 5):
        return "medium"
    return "low"
