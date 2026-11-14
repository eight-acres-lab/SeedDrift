"""Check hgtscan calls against well-documented horizontally acquired islands.

Islands are defined by coordinates in a reference genome (gene symbols in the Ensembl/Prokka
annotation are too sparse: most SPI-2 and LEE genes have no name). Other genomes' copies are
found by tree orthology: a gene belongs to the island if its nearest reference-genome gene in
the family's gene tree lies in the island (OrthoFinder puts the homologous T3SS parts of SPI-1,
SPI-2, LEE, ETT2 and Ysa into the same orthogroups, so family membership alone is not enough).

Expectations from the literature:
  SPI-1  invasion T3SS, avrA..invH (S. Typhimurium). In all Salmonella incl. S. bongori
         -> acquired by the Salmonella ancestor.
  SPI-2  T3SS part, ssrA..ssaU (S. Typhimurium). In S. enterica, absent from S. bongori
         -> acquired by the S. enterica ancestor after the split from S. bongori.
         (The tetrathionate-reductase half of SPI-2, ttrRSBCA, is older and also in
         S. bongori, so it is left out.)
  LEE    locus of enterocyte effacement, espF..ler (EHEC O157:H7). In O157 and E. albertii,
         absent from K-12, CFT073, Shigella, E. fergusonii -> acquired by their common ancestor
         with later losses (the parsimonious DTL answer) or independently by both.

Usage: python3 check_known_islands.py <workdir> <scan dir> [P threshold]
"""
import csv
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from hgtscan.reconcile import tree_distances  # noqa: E402
from hgtscan.tree import SpeciesTree, parse_newick, unrooted_adjacency  # noqa: E402

ISLANDS = [
    # name, reference genome, contig, span, genomes expected to carry it
    ("SPI-1", "STyphimurium", "Chromosome", (3009900, 3045300),
     ["STyphimurium", "STyphi", "Sarizonae", "Sbongori"]),
    ("SPI-2", "STyphimurium", "Chromosome", (1476000, 1501000),
     ["STyphimurium", "STyphi", "Sarizonae"]),
    ("LEE", "Ecoli_O157", "Chromosome", (4589000, 4622500),
     ["Ecoli_O157", "Ealbertii"]),
]
GAP = 20000          # max distance between consecutive island genes in one cluster
MIN_CLUSTER = 4      # genes needed to call the island present in a genome


def load(workdir, scandir):
    genes = {}
    with open(os.path.join(workdir, "genes.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            genes[r["gene_id"]] = (r["species"], r["family"], r["contig"], int(r["start"]))
    calls = {}
    with open(os.path.join(scandir, "hgt_genes.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            calls[r["gene_id"]] = r
    with open(os.path.join(workdir, "species_tree.nwk")) as fh:
        sp = SpeciesTree(parse_newick(fh.read()))
    return genes, calls, sp


def island_orthologs(workdir, fam, members, ref, island_ref):
    """Genes of `fam` whose nearest `ref` gene in the gene tree is an island gene."""
    path = os.path.join(workdir, "gene_trees", fam + ".treefile")
    if not os.path.exists(path):
        return set(members)                      # small family without a tree
    with open(path) as fh:
        adj, names = unrooted_adjacency(parse_newick(fh.read().strip()))
    idx = {nm: i for i, nm in enumerate(names) if nm}
    refs = [g for g in members if g in idx and g.startswith(ref + "__")]
    out = set(g for g in refs if g in island_ref)
    for g in members:
        if g not in idx or g.startswith(ref + "__") or not refs:
            continue
        d = tree_distances([[(v, L + 1e-9) for v, L in nb] for nb in adj], idx[g])
        near = min(refs, key=lambda r: d.get(idx[r], 1e9))
        if near in island_ref:
            out.add(g)
    return out


def clusters(members):
    """Split [(contig, start, gene)] into runs of nearby genes; largest first."""
    out, cur = [], []
    for c, s, g in sorted(members):
        if cur and (c != cur[-1][0] or s - cur[-1][1] > GAP):
            out.append(cur)
            cur = []
        cur.append((c, s, g))
    if cur:
        out.append(cur)
    return sorted(out, key=len, reverse=True)


def mrca(sp, names):
    idx = [sp.names.index(n) for n in names]
    best = None
    for i in range(sp.n):
        if all(sp.desc[i, j] for j in idx) and (best is None or len(sp.leafsets[i]) < len(sp.leafsets[best])):
            best = i
    return best


def node_set(sp, label):
    name = label.split("{")[0]
    return frozenset(sp.leafsets[sp.names.index(name)]) if name in sp.names else None


def placement(sp, label, exp_node):
    name = label.split("{")[0]
    if name not in sp.names:
        return "none"
    r = sp.names.index(name)
    if r == exp_node:
        return "expected node"
    if sp.desc[exp_node, r]:
        return "later (inside clade)"
    if sp.desc[r, exp_node]:
        return "deeper"
    return "other lineage"


def main(workdir, scandir, thr=0.6):
    genes, calls, sp = load(workdir, scandir)
    by_fam = defaultdict(list)
    for g, (s, f, c, st) in genes.items():
        by_fam[f].append(g)
    print("scan: %s   (gene P_HGT >= %.1f counts as a call)" % (scandir, thr))
    totals = Counter()
    for name, ref, contig, (a, b), expected in ISLANDS:
        island_ref = {g for g, (s, f, c, st) in genes.items() if s == ref and c == contig and a <= st <= b}
        fams = {genes[g][1] for g in island_ref}
        exp_node = mrca(sp, expected)
        clade = set(sp.leafsets[exp_node])
        orth = set()
        for f in fams:
            orth |= island_orthologs(workdir, f, by_fam[f], ref, island_ref)
        per_sp = defaultdict(list)
        for g in orth:
            s, f, c, st = genes[g]
            per_sp[s].append((c, st, g))
        island = {}
        for s, mem in per_sp.items():
            cl = clusters(mem)
            if cl and len(cl[0]) >= MIN_CLUSTER:
                island[s] = [g for _, _, g in cl[0]]
        # families present only inside the expected recipient clade: no sampled donor exists,
        # so a DTL reconciliation sees a gene origin, not a transfer
        restricted = {f for f in fams if all(genes[g][0] in clade for g in by_fam[f])}
        no_tree = {f for f in fams if not os.path.exists(os.path.join(workdir, "gene_trees", f + ".treefile"))}
        print("\n== %s: %d genes in %s, expected in %s; expected recipient %s"
              % (name, len(island_ref), ref, ", ".join(expected), sp.names[exp_node]))
        print("  families: %d | clade-restricted %d | without gene tree (<4 genes) %d"
              % (len(fams), len(restricted), len(no_tree)))
        missing = [s for s in expected if s not in island]
        extra = sorted(s for s in island if s not in expected)
        if missing:
            print("  island cluster MISSING in: " + ", ".join(missing))
        if extra:
            print("  related cluster (homologous system) also in: "
                  + ", ".join("%s(%d)" % (s, len(island[s])) for s in extra))
        for s in expected:
            mem = island.get(s, [])
            hit = [calls[g] for g in mem if g in calls and float(calls[g]["P_HGT"]) >= thr]
            weak = sum(1 for g in mem if g in calls and 0.3 <= float(calls[g]["P_HGT"]) < thr)
            restr = sum(1 for g in mem if genes[g][1] in restricted)
            rec = Counter(h["recipient"].split("{")[0] for h in hit)
            don = Counter(h["donor1"].split("{")[0] for h in hit)
            kinds = Counter(placement(sp, h["recipient"], exp_node) for h in hit)
            totals[name, "genes"] += len(mem)
            totals[name, "hit"] += len(hit)
            totals[name, "restricted"] += restr
            print("  %-13s %3d genes (%2d clade-restricted) | called %3d (%3.0f%%), weak %2d | recipient %-24s"
                  " | donor %-28s | %s"
                  % (s, len(mem), restr, len(hit), 100.0 * len(hit) / max(1, len(mem)), weak,
                     ", ".join("%s x%d" % kv for kv in rec.most_common(3)) or "-",
                     ", ".join("%s x%d" % kv for kv in don.most_common(2)) or "-",
                     ", ".join("%s %d" % kv for kv in kinds.most_common()) or ""))

    # genome-wide: all calls, and calls whose recipient clade holds < half of the genomes
    # (deep recipients are where hidden paralogy and long-branch artefacts concentrate)
    half = len(sp.leaf_index) / 2.0
    per = defaultdict(lambda: [0, 0, Counter(), 0])
    for g, (s, _, _, _) in genes.items():
        per[s][1] += 1
        c = calls.get(g)
        if c is not None and float(c["P_HGT"]) >= thr:
            per[s][0] += 1
            rec = node_set(sp, c["recipient"])
            if rec is not None and len(rec) < half:
                per[s][3] += 1
                per[s][2][c["source"]] += 1
    print("\n== genes with P_HGT >= %.1f per genome (denominator: all protein-coding genes)" % thr)
    print("  %-16s %16s   %-40s" % ("", "all calls", "recipient clade < half of the genomes (by source)"))
    for s, (n, tot, src, nr) in sorted(per.items(), key=lambda x: -x[1][3] / x[1][1]):
        print("  %-16s %5d (%4.1f%%) | %5d (%4.1f%%)  %s" % (
            s, n, 100.0 * n / tot, nr, 100.0 * nr / tot,
            ", ".join("%s %d" % kv for kv in sorted(src.items()))))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.6)
