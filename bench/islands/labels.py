#!/usr/bin/env python
"""Gene labels of the test chromosomes from the genomic-island reference set (protocol: README.md).

A gene is "island" when at least half of its coding region lies in one island region, "backbone" when at least half
lies in one backbone region, otherwise unlabelled ("none"). A gene with at least half in both (the two region sets
come from different published runs and can overlap) is written as "both": unlabelled in every comparison, but never
a native and never inside a native block. No score of any method is read here.

Writes, per chromosome, labels/<accession>.tsv (gene_id as the pipeline names it, contig, start, end, strand, label,
region, product), genes/<accession>.tsv (the same genes without label, region and product: the only gene table a
scoring script may read) and results/label_counts.tsv.

usage (repo root, on the machine with the data; needs the seeddrift package and Biopython):
  python bench/islands/labels.py [--dir $SEEDDRIFT_DATA/evalsets/islandpick] [--accessions NC_... ...]
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))


def read_regions(path):
    """{accession: [(start, end, id)]} from one of the supplementary tables (old Mac line ends)."""
    txt = open(path, newline="").read().replace("\r\n", "\n").replace("\r", "\n")
    out = {}
    for r in csv.DictReader(io.StringIO(txt), delimiter="\t"):
        rid = r.get("GI_ID") or r.get("Neg_ID") or r.get("Lit_ID") or ""
        out.setdefault(r["Accession"].strip(), []).append((int(r["Start"]), int(r["End"]), rid))
    return out


def best_region(start, end, regions):
    """The region covering the largest share of [start, end] and that share."""
    best, share = "", 0.0
    for a, b, rid in regions:
        s = max(0, min(end, b) - max(start, a) + 1) / (end - start + 1)
        if s > share:
            best, share = rid, s
    return best, share


def products(path):
    from Bio import SeqIO
    out = {}
    for rec in SeqIO.parse(path, "genbank"):
        n = 0
        for f in rec.features:
            if f.type != "CDS" or "pseudo" in f.qualifiers:
                continue
            q = f.qualifiers
            n += 1
            gid = (q.get("locus_tag") or q.get("protein_id") or q.get("gene") or ["%s_%d" % (rec.id, n)])[0]
            out[gid] = q.get("product", [""])[0]
    return out


def label_genome(gb_path, islands, backbone):
    """Rows (pipeline gene id, contig, start, end, strand, label, region, product) for one GenBank record."""
    from seeddrift.inputs import from_genbank
    from seeddrift.pipeline import clean_proteins
    g = from_genbank(open(gb_path).read())
    _, ren = clean_proteins(g.proteins)
    prod = products(gb_path)
    rows = []
    for gid, new in ren.items():
        contig, s, e, strand, _ = g.coords[gid]
        ri, fi = best_region(s, e, islands)
        rb, fb = best_region(s, e, backbone)
        lab, reg = ("both", "") if fi >= 0.5 and fb >= 0.5 else ("island", ri) if fi >= 0.5 else ("backbone", rb) if fb >= 0.5 else ("none", "")
        rows.append((new, contig, s, e, strand, lab, reg, prod.get(gid, "")))
    return rows, len(g.proteins) - len(ren)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(os.environ.get("SEEDDRIFT_DATA", "."), "evalsets", "islandpick"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--accessions", nargs="*", default=[], help="default: the chromosomes of results/genomes.tsv")
    ap.add_argument("--genbank", default="", help="folder with <accession>.gb (default: <dir>/genomes)")
    a = ap.parse_args()
    isl, back = read_regions(os.path.join(a.dir, "SupTable1.txt")), read_regions(os.path.join(a.dir, "SupTable2.txt"))
    accs = a.accessions or [r["accession"] for r in csv.DictReader(open(os.path.join(a.out, "genomes.tsv")), delimiter="\t")]
    os.makedirs(os.path.join(a.dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(a.dir, "genes"), exist_ok=True)
    counts = []
    for acc in accs:
        rows, dropped = label_genome(os.path.join(a.genbank or os.path.join(a.dir, "genomes"), acc + ".gb"), isl.get(acc, []), back.get(acc, []))
        n = {k: sum(1 for r in rows if r[5] == k) for k in ("island", "backbone", "none", "both")}
        with open(os.path.join(a.dir, "labels", acc + ".tsv"), "w") as fo, open(os.path.join(a.dir, "genes", acc + ".tsv"), "w") as fg:
            fo.write("gene_id\tcontig\tstart\tend\tstrand\tlabel\tregion\tproduct\n")
            fg.write("gene_id\tcontig\tstart\tend\tstrand\n")
            for r in rows:
                fo.write("\t".join(str(x) for x in r) + "\n")
                fg.write("\t".join(str(x) for x in r[:5]) + "\n")
        counts.append((acc, len(rows), n["island"], n["backbone"], n["none"] + n["both"], n["both"], dropped))
        print("%s genes %d island %d backbone %d unlabelled %d (of which in both region sets %d); coding sequences without a usable protein %d" % counts[-1], flush=True)
    if not a.accessions:
        with open(os.path.join(a.out, "label_counts.tsv"), "w") as fo:
            fo.write("accession\tgenes\tisland\tbackbone\tunlabelled\tin_both_region_sets\tcds_without_usable_protein\n")
            for c in counts:
                fo.write("\t".join(str(x) for x in c) + "\n")


if __name__ == "__main__":
    main()
