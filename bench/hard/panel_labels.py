#!/usr/bin/env python
"""Gene classes of a benchmark chromosome from its genus panel (panel.py) and the published island regions.

Nothing here reads a score, the reference library or sequence composition. The evidence is gene order in genomes of
the chromosome's own genus, which every method under test has hidden.

  hit of gene g in panel genome P   best alignment of g in P with at least 40% identity over at least 70% of g
  positional orthologue             g has a hit h in P, and at least one of the 2 genes before g and at least one of
                                    the 2 genes after g has its own hit in P within 5 genes of h on the same replicon
  f                                 share of the panel genomes of OTHER species of the genus with a positional
                                    orthologue (needs at least 3 such genomes)
  single copy                       no other gene of the chromosome aligns with at least 40% identity over at least
                                    80% of both genes

  classes   island      published island gene with f <= 0.5                      (positive)
            native      not a published island gene, f >= 0.8, single copy, and not within 5 genes of a published
                        island gene                                              (negative, "positional core")
            conflict    published island gene with f >= 0.8 (the other species keep it in place)
            edge        positional core within 5 genes of a published island gene
            unresolved  everything else: never counted as native or as transferred
  units     island region   a published island region with at least 3 labelled genes, at least half of them class island
            native block    11 consecutive genes, none a published island gene or within 5 genes of one, at least 9
                            of them class native; tiled without overlap
  flags     absent      island-class gene without any alignment (40% identity over 50% of the gene) in any other species
            pool        island-class gene with a 70% / 70% alignment outside its position in another species
            deep        native-class gene with a hit in at least 2 sibling-genus genomes
            fast        native-class gene in the lowest tenth of mean identity to its positional orthologues

Writes <out>/<accession>.tsv (the columns of bench/islands/labels.py with label = island / backbone / none for the
classes island / native / other, region = the unit, plus the class, f and the flags) and <out>/<accession>.json
(counts, panel sizes, agreement with the published labels).

usage (repo root; needs the seeddrift package, Biopython and DIAMOND):
  python bench/hard/panel_labels.py --accessions NC_... --genbank DIR/genomes --labels DIR/labels --panel DIR/panel --out DIR/labels_panel
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

MIN_ID, MIN_COV = 40.0, 0.7
NEAR = 5                    # genes: neighbourhood of a hit, and the margin around published island genes
BLOCK = 11


def panel_genome(path):
    """Proteins of one panel assembly in replicon order: [(id, replicon, index on the replicon, sequence)]."""
    from seeddrift.inputs import from_genbank
    g = from_genbank(gzip.open(path, "rt").read(), fmt="embl")
    by = {}
    for gid, seq in g.proteins.items():
        contig, s, e, _, _ = g.coords[gid]
        by.setdefault(contig, []).append((s, e, gid, seq))
    out = []
    for contig, rows in by.items():
        for i, (_, _, gid, seq) in enumerate(sorted(rows)):
            out.append((gid, contig, i, seq))
    return out


def diamond(exe, query, db_fasta, out, threads, tmp, k=500):
    subprocess.run([exe, "makedb", "--in", db_fasta, "-d", db_fasta + ".dmnd", "--quiet"], check=True)
    subprocess.run([exe, "blastp", "-q", query, "-d", db_fasta + ".dmnd", "-o", out, "--sensitive", "--max-target-seqs", str(k), "-e", "1e-5",
                    "--threads", str(threads), "--tmpdir", tmp, "--quiet",
                    "--outfmt", "6", "qseqid", "sseqid", "pident", "qstart", "qend", "qlen", "sstart", "send", "slen", "bitscore"], check=True)


def classify(genes, contig, published, hits, panel, self_multi):
    """genes: ids in chromosome order; contig: their contig; published: their published label; hits: {(gene, genome): (bits,
    identity, replicon, index, coverage)} best covering hit; panel: {genome: role}; self_multi: set of multi-copy genes."""
    n = len(genes)
    ix = {g: i for i, g in enumerate(genes)}
    contig = np.asarray(contig)
    start = {c: int(np.flatnonzero(contig == c)[0]) for c in np.unique(contig)}
    size = {c: int((contig == c).sum()) for c in np.unique(contig)}

    def neighbour(i, d):                                      # the gene d places away on the same contig (circular)
        c = contig[i]
        return start[c] + (i - start[c] + d) % size[c]

    others = [p for p, r in panel.items() if r == "other"]
    sibs = [p for p, r in panel.items() if r == "sibling"]
    same = [p for p, r in panel.items() if r == "same"]

    def positional(i, p):
        h = hits.get((genes[i], p))
        if h is None:
            return False
        ok = []
        for side in ((-1, -2), (1, 2)):
            good = False
            for d in side:
                j = neighbour(i, d)
                if j == i:
                    continue
                hj = hits.get((genes[j], p))
                if hj is not None and hj[2] == h[2] and abs(hj[3] - h[3]) <= NEAR:
                    good = True
            ok.append(good)
        return all(ok)

    isl = np.array([x == "island" for x in published])
    near_isl = isl.copy()
    for i in np.flatnonzero(isl):
        for d in range(-NEAR, NEAR + 1):
            near_isl[neighbour(int(i), d)] = True
    rows = []
    for i, g in enumerate(genes):
        pos_o = [p for p in others if positional(i, p)]
        f = len(pos_o) / len(others) if len(others) >= 3 else float("nan")
        f_same = float(np.mean([positional(i, p) for p in same])) if same else float("nan")
        ident = [hits[(g, p)][1] for p in pos_o]
        any_other = any((g, p) in hits for p in others)
        rows.append(dict(f=f, f_same=f_same, n_other=len(others), pos_other=len(pos_o), hit_other=sum((g, p) in hits for p in others),
                         hit_sibling=sum((g, p) in hits for p in sibs), identity=float(np.mean(ident)) if ident else float("nan"),
                         multi=g in self_multi, any_other=any_other, pos_list=set(pos_o)))
    cls = []
    for i, r in enumerate(rows):
        f = r["f"]
        if f != f:
            c = "unresolved"
        elif isl[i]:
            c = "island" if f <= 0.5 else "conflict" if f >= 0.8 else "unresolved"
        elif f >= 0.8 and not r["multi"]:
            c = "edge" if near_isl[i] else "native"
        else:
            c = "unresolved"
        cls.append(c)
    cls = np.array(cls, dtype=object)
    # native blocks: tiled windows of BLOCK consecutive genes without a published island gene or its margin
    unit = np.array([""] * n, dtype=object)
    b = 0
    for c in np.unique(contig):
        i, end = start[c], start[c] + size[c]
        while i + BLOCK <= end:
            w = slice(i, i + BLOCK)
            if not near_isl[w].any() and (cls[w] == "native").sum() >= BLOCK - 2:
                unit[w] = "NB%d" % b
                b += 1
                i += BLOCK
            else:
                i += 1
    return rows, cls, unit, near_isl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accessions", nargs="+", required=True)
    ap.add_argument("--genbank", nargs="+", required=True, help="folders with <accession>.gb (searched in order)")
    ap.add_argument("--labels", required=True, help="folder of bench/islands/labels.py (published labels)")
    ap.add_argument("--panel", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--diamond", default="diamond"); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    from seeddrift.inputs import from_genbank
    from seeddrift.pipeline import clean_proteins
    os.makedirs(a.out, exist_ok=True)
    for acc in a.accessions:
        pdir = os.path.join(a.panel, acc)
        panel_rows = list(csv.DictReader(open(os.path.join(pdir, "panel.tsv")), delimiter="\t"))
        work = os.path.join(pdir, "work")
        os.makedirs(work, exist_ok=True)
        gb = next(os.path.join(d, acc + ".gb") for d in a.genbank if os.path.exists(os.path.join(d, acc + ".gb")))
        g = from_genbank(open(gb).read())
        prots, ren = clean_proteins(g.proteins)
        lab = {r["gene_id"]: r for r in csv.DictReader(open(os.path.join(a.labels, acc + ".tsv")), delimiter="\t")}
        order = sorted(ren, key=lambda k: (g.coords[k][0], g.coords[k][1], g.coords[k][2]))
        genes = [ren[k] for k in order]
        contig = [g.coords[k][0] for k in order]
        with open(os.path.join(work, "query.faa"), "w") as fo:
            for k in genes:
                fo.write(">%s\n%s\n" % (k, prots[k]))
        where, role = {}, {}
        with open(os.path.join(work, "panel.faa"), "w") as fo:
            for pi, pr in enumerate(panel_rows):
                role[pi] = pr["role"]
                for j, (gid, rep, idx, seq) in enumerate(panel_genome(os.path.join(a.panel, "assemblies", pr["assembly"] + ".embl.gz"))):
                    name = "P%d_%d" % (pi, j)
                    where[name] = (pi, rep, idx)
                    fo.write(">%s\n%s\n" % (name, seq))
        diamond(a.diamond, os.path.join(work, "query.faa"), os.path.join(work, "panel.faa"), os.path.join(work, "panel_hits.tsv"), a.threads, work)
        diamond(a.diamond, os.path.join(work, "query.faa"), os.path.join(work, "query.faa"), os.path.join(work, "self_hits.tsv"), a.threads, work, k=50)
        hits, loose, strong = {}, set(), set()
        for line in open(os.path.join(work, "panel_hits.tsv")):
            q, s, pid, qs, qe, ql, _, _, _, bits = line.rstrip("\n").split("\t")
            pid, cov, bits = float(pid), (int(qe) - int(qs) + 1) / int(ql), float(bits)
            pi, rep, idx = where[s]
            if pid >= MIN_ID and cov >= 0.5:
                loose.add((q, pi))
            if pid >= 70.0 and cov >= 0.7:
                strong.add((q, pi))
            if pid >= MIN_ID and cov >= MIN_COV and ((q, pi) not in hits or bits > hits[(q, pi)][0]):
                hits[(q, pi)] = (bits, pid, rep, idx, cov)
        multi = set()
        for line in open(os.path.join(work, "self_hits.tsv")):
            q, s, pid, qs, qe, ql, ss, se, sl, _ = line.rstrip("\n").split("\t")
            if q != s and float(pid) >= MIN_ID and (int(qe) - int(qs) + 1) / int(ql) >= 0.8 and (int(se) - int(ss) + 1) / int(sl) >= 0.8:
                multi.add(q)
        published = [lab[k]["label"] if k in lab else "none" for k in genes]
        rows, cls, unit, near_isl = classify(genes, contig, published, hits, role, multi)
        others = [p for p, r in role.items() if r == "other"]
        sibs = [p for p, r in role.items() if r == "sibling"]
        # island regions as units: published regions with at least 3 labelled genes, at least half of them class island
        reg = {}
        for i, k in enumerate(genes):
            if published[i] == "island":
                reg.setdefault(lab[k]["region"], []).append(i)
        for rid, idx in reg.items():
            if len(idx) >= 3 and np.mean(cls[idx] == "island") >= 0.5:
                for i in idx:
                    if cls[i] == "island":
                        unit[i] = rid
        nat = np.flatnonzero(cls == "native")
        ident = np.array([rows[i]["identity"] for i in nat])
        fast = set(nat[ident <= np.nanquantile(ident, 0.1)].tolist()) if len(nat) else set()
        with open(os.path.join(a.out, acc + ".tsv"), "w") as fo:
            fo.write("gene_id\tcontig\tstart\tend\tstrand\tlabel\tregion\tproduct\tclass\tpublished\tf\tf_same\tpos_other\tn_other\thit_sibling\tidentity\tcopies\tflags\n")
            for i, k in enumerate(genes):
                r, c = rows[i], cls[i]
                flags = []
                if c == "island":
                    if not any((k, p) in loose for p in others):
                        flags.append("absent")
                    if any((k, p) in strong and p not in r["pos_list"] for p in others):
                        flags.append("pool")
                if c == "native":
                    if r["hit_sibling"] >= 2:
                        flags.append("deep")
                    if i in fast:
                        flags.append("fast")
                contig_i, s, e, strand, _ = g.coords[order[i]]
                fo.write("\t".join(str(x) for x in (
                    k, contig_i, s, e, strand, "island" if c == "island" else "backbone" if c == "native" else "none", unit[i],
                    "", c, published[i], "" if r["f"] != r["f"] else "%.2f" % r["f"], "" if r["f_same"] != r["f_same"] else "%.2f" % r["f_same"],
                    r["pos_other"], r["n_other"], r["hit_sibling"], "" if r["identity"] != r["identity"] else "%.1f" % r["identity"],
                    "multi" if r["multi"] else "single", ",".join(flags))) + "\n")
        pub = np.array(published, dtype=object)
        single = np.array([not r["multi"] for r in rows])
        pc = np.array([c in ("native", "edge") for c in cls])
        bb_single = (pub == "backbone") & single
        info = dict(accession=acc, genes=len(genes), panel={k: sum(1 for r in role.values() if r == k) for k in ("same", "other", "sibling")},
                    classes={k: int((cls == k).sum()) for k in ("island", "native", "conflict", "edge", "unresolved")},
                    published={k: int((pub == k).sum()) for k in ("island", "backbone", "none")},
                    native_blocks=len({u for u in unit if u.startswith("NB")}), island_regions=len({u for u in unit if u and not u.startswith("NB")}),
                    published_island_regions=len(reg),
                    backbone_single_copy_positional_core=round(float(pc[bb_single].mean()), 4) if bb_single.any() else None,
                    island_genes_conflict=round(float((cls[pub == "island"] == "conflict").mean()), 4) if (pub == "island").any() else None,
                    island_genes_gated=round(float((cls[pub == "island"] == "island").mean()), 4) if (pub == "island").any() else None,
                    native_deep=round(float(np.mean([rows[i]["hit_sibling"] >= 2 for i in nat])), 4) if len(nat) else None,
                    median_identity_to_other_species=round(float(np.nanmedian(ident)), 1) if len(nat) else None)
        json.dump(info, open(os.path.join(a.out, acc + ".json"), "w"), indent=1)
        print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
