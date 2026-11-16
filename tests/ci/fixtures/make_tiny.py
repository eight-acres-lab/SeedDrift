#!/usr/bin/env python
"""Builds tests/ci/fixtures/tiny/: the small fixed data of the ci-tiny profile (T1), from public data on the server.

Run once on the server (it reads the corpus, the 40M-protein reference and the NCBI taxdump there); the output, under
2 MB, is committed. Real parts: two 40 kb chromosome segments and one plasmid segment of corpus genomes (a bacterium
and an archaeon; INSDC sequences), the proteins of one reference proteome per eukaryote and virus group, the reference
proteins that those queries hit, and the NCBI lineages of every taxid involved. Synthetic parts (marked as such in
sources.tsv): the coding DNA of the eukaryote and virus groups (back-translated with a fixed codon table) and the
spacers between those genes, and the scenario genome (the bacterial segment with three genes of a Bacillota protein
set spliced in at an intergenic position; its truth table is scenario_truth.tsv).

usage: python tests/ci/fixtures/make_tiny.py --out tests/ci/fixtures/tiny [paths of the server data, see --help]
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import random
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from ci.hashing import sha256_file  # noqa: E402
from ci.kingdom import KingdomMap  # noqa: E402
from ci.taxonomy import Taxonomy  # noqa: E402

CODON = dict(A="GCG", R="CGT", N="AAC", D="GAT", C="TGC", Q="CAG", E="GAA", G="GGC", H="CAT", I="ATT", L="CTG", K="AAA", M="ATG",
             F="TTT", P="CCG", S="AGC", T="ACC", W="TGG", Y="TAT", V="GTG")
COMP = str.maketrans("ACGT", "TGCA")


def revcomp(s):
    return s.translate(COMP)[::-1]


def read_fasta_gz(path):
    out, name, buf = {}, None, []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if name:
                    out[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name:
        out[name] = "".join(buf)
    return out


def call(seq):
    import pyrodigal
    gf = pyrodigal.GeneFinder(meta=True)
    return [(g.begin, g.end, g.strand, g.translate().rstrip("*")) for g in gf.find_genes(seq.encode())]


def back_translate(prot):
    return "ATG" + "".join(CODON.get(a, "GCG") for a in prot[1:]) + "TAA"


def write_fasta(path, records):
    with open(path, "w") as f:
        for name, seq in records:
            f.write(">%s\n" % name)
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


def proteins_of(reference, taxids, per, lo=100, hi=600):
    """First `per` proteins (length lo..hi, standard residues) of each taxid, scanning the reference once."""
    want = {str(t): [] for t in taxids}
    with open(reference) as f:
        head = None
        for line in f:
            if line.startswith(">"):
                head = line[1:].strip()
                continue
            t = head.rsplit("|", 1)[-1]
            if t in want and len(want[t]) < per:
                s = line.strip()
                if lo <= len(s) <= hi and set(s) <= set("ACDEFGHIKLMNPQRSTVWY") and s[0] == "M":
                    want[t].append((head.split("|")[0], s))
                if all(len(v) >= per for v in want.values()):
                    break
    return want


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="/hy-tmp/seeddrift-store/corpus/v1")
    ap.add_argument("--reference", default="/hy-tmp/seeddrift-store/reference/reference.faa")
    ap.add_argument("--sdx", default="/hy-tmp/seeddrift-store/reference/reference.faa.sdx.npz")
    ap.add_argument("--ref-dmnd", default="/hy-tmp/seeddrift-store/reference/ref.dmnd")
    ap.add_argument("--library", default="/hy-tmp/seeddrift-store/placement/library.npz")
    ap.add_argument("--taxdump", default="/hy-tmp/seeddrift-store/reference/taxdump")
    ap.add_argument("--panels", default="/hy-tmp/seeddrift-store/evalsets/islandpick/panel")
    ap.add_argument("--exclude", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "bench", "hard", "EXCLUDE.tsv"))
    ap.add_argument("--diamond", default="diamond")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--panel-table", default="", help="also write the panel assembly table here (configs/exclusions/panel_assemblies.tsv)")
    a = ap.parse_args()
    rnd = random.Random(20261117)
    out = a.out
    for d in ("genomes", "annotated"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    tax = Taxonomy(a.taxdump)
    kmap = KingdomMap()
    sel = list(csv.DictReader(open(os.path.join(a.corpus, "selected.tsv")), delimiter="\t"))
    ok = {m["accession"]: m for m in (json.load(open(p)) for p in glob.glob(os.path.join(a.corpus, "meta", "*.json"))) if m["status"] == "ok"}
    sources, queries = [], []

    # real prokaryote segments
    bac = next(r for r in sorted(sel, key=lambda r: r["accession"]) if r["domain"] == "bacteria" and r["accession"] in ok and "Plasmid" in ok[r["accession"]]["types"])
    arc = next(r for r in sorted(sel, key=lambda r: r["accession"]) if r["domain"] == "archaea" and r["accession"] in ok)
    for r, label in ((bac, "bacterium"), (arc, "archaeon")):
        seqs = read_fasta_gz(os.path.join(a.corpus, "genomes", r["accession"] + ".fna.gz"))
        chrom = max(seqs.items(), key=lambda kv: len(kv[1]))
        seg = chrom[1][100000:140000]
        name = "%s_%s" % (label, r["accession"])
        write_fasta(os.path.join(out, "genomes", name + ".fna"), [("%s:100001-140000" % chrom[0], seg)])
        sources.append(dict(id=name, source_kind="raw_prokaryote", taxid=r["taxid"], files="genomes/%s.fna" % name, synthetic="no",
                            accession=r["accession"], paired_accession=ok[r["accession"]].get("gca", ""),
                            origin="%s %s, %s positions 100001-140000" % (r["accession"], r["species"], chrom[0])))
        queries += [("%s_%d" % (name, i), p) for i, (_, _, _, p) in enumerate(call(seg), 1) if len(p) >= 60]
        if label == "bacterium":
            bac_name, bac_chrom = name, chrom
            scen_seg = chrom[1][160000:200000]               # evaluation region: no overlap with the training segment
            ptypes = ok[r["accession"]]["types"]
            plas = [k for k, t in zip(seqs, ptypes) if t == "Plasmid"]
            pseq = seqs[plas[0]][:20000]
            pname = "plasmid_%s" % plas[0].replace(".", "_")
            write_fasta(os.path.join(out, "genomes", pname + ".fna"), [(plas[0] + ":1-%d" % len(pseq), pseq)])
            sources.append(dict(id=pname, source_kind="plasmid", taxid=r["taxid"], files="genomes/%s.fna" % pname, synthetic="no",
                                accession=r["accession"], paired_accession=ok[r["accession"]].get("gca", ""),
                                origin="plasmid %s of %s, first %d bp" % (plas[0], r["accession"], len(pseq))))
            queries += [("%s_%d" % (pname, i), p) for i, (_, _, _, p) in enumerate(call(pseq), 1) if len(p) >= 60]

    # one reference proteome per eukaryote and virus group: real proteins, synthetic coding DNA
    lib = np.load(a.library)
    lib_taxids = sorted(int(t) for t in lib["taxid"])
    pick = {}
    for t in lib_taxids:
        try:
            g = kmap.group(tax, t)
        except Exception:  # noqa: BLE001
            continue
        if (g.startswith("eukaryota.") or g.startswith("viruses.")) and g not in pick:
            pick[g] = t
    prots = proteins_of(a.reference, list(pick.values()), per=8)
    for g, t in sorted(pick.items()):
        ps = prots[str(t)]
        contig, genes, faa, pos = [], [], [], 0
        name = "%s_%d" % (g.replace(".", "_"), t)
        for i, (acc, p) in enumerate(ps, 1):
            spacer = "".join(rnd.choice("ACGT") for _ in range(150))
            cds = back_translate(p)
            strand = "+" if i % 2 else "-"
            contig.append(spacer)
            pos += len(spacer)
            contig.append(cds if strand == "+" else revcomp(cds))
            genes.append(dict(gene_id="%s_g%d" % (name, i), replicon="%s_contig1" % name, start=pos + 1, end=pos + len(cds), strand=strand, protein_source=acc))
            faa.append(("%s_g%d" % (name, i), p))
            pos += len(cds)
        contig.append("".join(rnd.choice("ACGT") for _ in range(150)))
        write_fasta(os.path.join(out, "annotated", name + ".fna"), [("%s_contig1" % name, "".join(contig))])
        with open(os.path.join(out, "annotated", name + ".genes.tsv"), "w") as f:
            f.write("gene_id\treplicon\tstart\tend\tstrand\tprotein_source\n")
            for x in genes:
                f.write("%(gene_id)s\t%(replicon)s\t%(start)d\t%(end)d\t%(strand)s\t%(protein_source)s\n" % x)
        write_fasta(os.path.join(out, "annotated", name + ".faa"), faa)
        sources.append(dict(id=name, source_kind="annotated", taxid=t, files="annotated/%s.fna;annotated/%s.genes.tsv;annotated/%s.faa" % (name, name, name),
                            synthetic="coding DNA and spacers", accession="", paired_accession="", origin="proteins of reference proteome taxid %d (%s)" % (t, tax.name(t))))
        queries += faa

    # scenario: three consecutive real genes of a Bacillota corpus genome spliced into the bacterial segment
    calls = call(scen_seg)
    gaps = [(calls[i][1], calls[i + 1][0]) for i in range(len(calls) - 1) if calls[i + 1][0] - calls[i][1] > 30]
    cut_a, cut_b = min(gaps, key=lambda g: abs((g[0] + g[1]) // 2 - 20000))
    cut = (cut_a + cut_b) // 2
    don = next(r for r in sorted(sel, key=lambda r: r["accession"]) if r["accession"] in ok and 1239 in tax.lineage(int(r["taxid"])))
    donor_taxid = int(don["taxid"])
    dseqs = read_fasta_gz(os.path.join(a.corpus, "genomes", don["accession"] + ".fna.gz"))
    dchrom = max(dseqs.values(), key=len)
    window = dchrom[200000:260000]
    dcalls = [c for c in call(window) if c[1] - c[0] + 1 >= 300]
    i = len(dcalls) // 2
    trio = dcalls[i:i + 3]
    seg_lo, seg_hi = trio[0][0] - 1 - 20, trio[-1][1] + 20
    segment = window[seg_lo:seg_hi]
    host = scen_seg[:cut] + segment + scen_seg[cut:]
    truth = [dict(start=cut + (b - 1 - seg_lo) + 1, end=cut + (e - seg_lo), strand="+" if st == 1 else "-", donor_taxid=donor_taxid,
                  protein_source="%s:%d-%d" % (don["accession"], 200000 + b, 200000 + e)) for b, e, st, _ in trio]
    write_fasta(os.path.join(out, "genomes", "scenario_host_insert.fna"), [("scenario_host_insert", host)])
    with open(os.path.join(out, "scenario_truth.tsv"), "w") as f:
        f.write("scenario\treplicon\tstart\tend\tstrand\tclass\tdonor_taxid\tprotein_source\n")
        for x in truth:
            f.write("scenario_host_insert\tscenario_host_insert\t%(start)d\t%(end)d\t%(strand)s\tinsert\t%(donor_taxid)d\t%(protein_source)s\n" % x)
    queries += [("donor_%d" % k, p) for k, (_, _, _, p) in enumerate(trio, 1)]
    with open(os.path.join(out, "scenario.tsv"), "w") as f:
        f.write("id\thost\thost_taxid\tinsert_after\tfiles\tnote\n")
        f.write("scenario_host_insert\t%s\t%s\t%d\tgenomes/scenario_host_insert.fna\treal DNA by construction: positions 160001-200000 of %s %s (not in the training corpus) with three consecutive genes of %s (%s) spliced in\n"
                % (bac_name, bac["taxid"], cut, bac["accession"], bac_chrom[0], don["accession"], don["species"]))

    # mini reference: the targets these queries hit (closest 20 plus the best of every genus), at most 2,000 proteins
    work = os.path.join(out, ".work")
    os.makedirs(work, exist_ok=True)
    write_fasta(os.path.join(work, "q.faa"), queries)
    hits = os.path.join(work, "hits.tsv")
    if not os.path.exists(hits):
        subprocess.run([a.diamond, "blastp", "-q", os.path.join(work, "q.faa"), "-d", a.ref_dmnd, "-o", hits, "--sensitive", "-k", "200", "-e", "1e-5",
                        "--threads", str(a.threads), "-b", "4", "-c", "1", "--quiet", "--outfmt", "6", "qseqid", "sseqid", "bitscore"], check=True)
    genus_of = {int(t): int(r[5]) for t, r in zip(lib["taxid"], lib["ranks"])}
    keep, per_q, best_genus = [], {}, {}
    for line in open(hits):
        q, s, b = line.split("\t")
        acc, t = s.split("|")
        n = per_q.setdefault(q, 0)
        g = genus_of.get(int(t), 0)
        if n < 20:
            keep.append((float(b), acc, int(t)))
        elif (q, g) not in best_genus:
            keep.append((float(b) - 1e6, acc, int(t)))
        best_genus.setdefault((q, g), acc)
        per_q[q] = n + 1
    chosen, seen = [], set()
    for _, acc, t in sorted(keep, key=lambda x: -x[0]):
        if acc not in seen:
            seen.add(acc)
            chosen.append((acc, t))
        if len(chosen) >= 2000:
            break
    sdx = np.load(a.sdx)
    accs = sdx["acc"]
    order = np.argsort(accs) if not np.all(accs[:-1] <= accs[1:]) else None
    ref = []
    with open(a.reference, "rb") as f:
        for acc, t in sorted(chosen):
            key = acc.encode()
            i = int(np.searchsorted(accs, key)) if order is None else int(order[np.searchsorted(accs[order], key)])
            if accs[i] != key:
                raise SystemExit("accession %s not in the sequence index" % acc)
            f.seek(int(sdx["off"][i]))
            head = f.readline().decode().strip()
            seq = f.readline().decode().strip()
            if head[1:].split("|")[0] != acc:
                raise SystemExit("index offset does not point at %s" % acc)
            ref.append(("%s|%d" % (acc, t), seq))
    write_fasta(os.path.join(out, "reference.faa"), ref)

    # taxdump subset: lineages of every taxid used, the groups of the kingdom map, and the names of EXCLUDE.tsv
    need = {int(s["taxid"]) for s in sources} | {t for _, t in chosen} | {donor_taxid}
    for g in kmap.groups:
        need |= set(g["include"]) | set(g.get("exclude", []))
    for r in csv.DictReader((l for l in open(a.exclude) if not l.startswith("#")), delimiter="\t"):
        if r["kind"] == "genus":
            need |= set(tax.genus_by_name(r["name"]))
            need |= set(tax.by_name.get(r["name"], []))
        elif r["kind"] == "order":
            need |= set(tax.taxa_by_name(r["name"], "order"))
    closure = set()
    for t in need:
        closure |= set(tax.lineage(t))
    td = os.path.join(out, "taxdump")
    os.makedirs(td, exist_ok=True)
    with open(os.path.join(td, "nodes.dmp"), "w") as f:
        for t in sorted(closure):
            f.write("%d\t|\t%d\t|\t%s\t|\n" % (t, tax.parent[t], tax.rank[t]))
    with open(os.path.join(td, "names.dmp"), "w") as f:
        for t in sorted(closure):
            f.write("%d\t|\t%s\t|\t\t|\tscientific name\t|\n" % (t, tax.sci.get(t, str(t))))

    with open(os.path.join(out, "sources.tsv"), "w") as f:
        cols = ["id", "source_kind", "taxid", "accession", "paired_accession", "files", "synthetic", "origin"]
        f.write("\t".join(cols) + "\n")
        for s in sources:
            f.write("\t".join(str(s[c]) for c in cols) + "\n")
    if a.panel_table:
        rows = set()
        for p in sorted(glob.glob(os.path.join(a.panels, "*", "panel.tsv"))):
            for r in csv.DictReader(open(p), delimiter="\t"):
                sp = tax.at_rank(int(r["tax_id"]), "species") or int(r.get("species") or 0)
                rows.add((r["assembly"], sp, r["role"], os.path.basename(os.path.dirname(p))))
        with open(a.panel_table, "w") as f:
            f.write("# Assemblies of the genus panels of bench/hard (development and sealed chromosomes) and their species.\n")
            f.write("# Excluded from every training corpus by the 'panel genomes' rule of bench/hard/EXCLUDE.tsv (gate G2).\n")
            f.write("assembly\tspecies_taxid\trole\tpanel_of\n")
            for r in sorted(rows):
                f.write("%s\t%d\t%s\t%s\n" % r)
    files = sorted(os.path.relpath(os.path.join(d, x), out) for d, _, fs in os.walk(out) for x in fs if ".work" not in d and x != "MANIFEST.sha256")
    with open(os.path.join(out, "MANIFEST.sha256"), "w") as f:
        for p in files:
            f.write("%s  %s\n" % (sha256_file(os.path.join(out, p), use_sidecar=False), p))
    size = sum(os.path.getsize(os.path.join(out, p)) for p in files)
    print("fixture written: %d files, %.2f MB; groups %s; donor taxid %d; reference proteins %d; taxdump nodes %d"
          % (len(files), size / 1e6, sorted(pick), donor_taxid, len(ref), len(closure)))


if __name__ == "__main__":
    main()
