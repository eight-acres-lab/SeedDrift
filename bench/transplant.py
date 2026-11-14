"""Gene-transplant benchmark data (DNA level): real donor CDS inserted into a real recipient genome.

Donor CDS come from Ensembl (complete ORFs of species whose genus is absent from the reference
library). They are inserted into intergenic gaps of the recipient, singly or as clusters of up to four
genes (islands), on either strand. Amelioration: a fraction of codons is replaced by synonymous codons
drawn from the recipient's codon usage (0 = donor composition kept, 1 = fully recipient-like), which
tests the composition evidence. The whole genome then goes through gene calling, so the benchmark
also measures whether the inserted genes are found at all.

Truth (truth.tsv): one row per inserted gene and per recipient core gene (ribosomal proteins, RNA
polymerase subunits, identified from the recipient's UniProt proteome), with coordinates in the new
genome; run_benchmark.py matches predicted genes to these intervals.

usage:  python bench/transplant.py --out bench/work/transplant_legionella
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seeddrift import config  # noqa: E402
from seeddrift.donors import taxonomy  # noqa: E402
from seeddrift.inputs import call_prokaryote, read_fasta_text, revcomp, translate  # noqa: E402
from spikein import CORE, CORE_EXCLUDE, fetch_fasta  # noqa: E402

E63 = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/release-63"
RECIPIENT = dict(
    dna=E63 + "/bacteria/fasta/bacteria_16_collection/legionella_pneumophila_subsp_pneumophila_str_philadelphia_1_gca_000008485/"
              "dna/Legionella_pneumophila_subsp_pneumophila_str_philadelphia_1_gca_000008485.ASM848v1_.dna.toplevel.fa.gz",
    proteome="https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Bacteria/"
             "UP000000609/UP000000609_272624.fasta.gz")
DONORS = [   # class, taxon name (genus absent from the reference library), Ensembl CDS
    ("mammal", "Dasypus novemcinctus", "https://ftp.ensembl.org/pub/release-116/fasta/dasypus_novemcinctus/cds/Dasypus_novemcinctus.Dasnov3.0.cds.all.fa.gz"),
    ("mammal", "Ochotona princeps", "https://ftp.ensembl.org/pub/release-116/fasta/ochotona_princeps/cds/Ochotona_princeps.OchPri2.0-Ens.cds.all.fa.gz"),
    ("mammal", "Sorex araneus", "https://ftp.ensembl.org/pub/release-116/fasta/sorex_araneus/cds/Sorex_araneus.COMMON_SHREW1.cds.all.fa.gz"),
    ("eukaryote", "Acidomyces richmondensis", E63 + "/fungi/fasta/fungi_ascomycota3_collection/acidomyces_richmondensis_bfw_gca_001592465/cds/Acidomyces_richmondensis_bfw_gca_001592465.Acidomyces_richmondensis_BFW_v1.0.cds.all.fa.gz"),
    ("eukaryote", "Agaricus bisporus", E63 + "/fungi/fasta/fungi_basidiomycota1_collection/agaricus_bisporus_var_burnettii_jb137_s8_gca_000300555/cds/Agaricus_bisporus_var_burnettii_jb137_s8_gca_000300555.Agabi_varbur_1.cds.all.fa.gz"),
    ("eukaryote", "Arabidopsis lyrata", E63 + "/plants/fasta/arabidopsis_lyrata/cds/Arabidopsis_lyrata.v.1.0.cds.all.fa.gz"),
    ("prokaryote", "Agriterribacter soli", E63 + "/bacteria/fasta/bacteria_87_collection/agriterribacter_soli_gca_003075435/cds/Agriterribacter_soli_gca_003075435.ASM307543v1.cds.all.fa.gz"),
    ("prokaryote", "Alkalinema", E63 + "/bacteria/fasta/bacteria_91_collection/alkalinema_sp_caciam_70d_gca_002148405/cds/Alkalinema_sp_caciam_70d_gca_002148405.ASM214840v1.cds.all.fa.gz"),
    ("prokaryote", "Alloactinosynnema", E63 + "/bacteria/fasta/bacteria_14_collection/alloactinosynnema_sp_l_07_gca_900070365/cds/Alloactinosynnema_sp_l_07_gca_900070365.Alloactinosynnema_sp._L-07.cds.all.fa.gz"),
    ("prokaryote", "Atopobacter", E63 + "/bacteria/fasta/bacteria_87_collection/atopobacter_sp_ah10_gca_003664595/cds/Atopobacter_sp_ah10_gca_003664595.ASM366459v1.cds.all.fa.gz"),
]
STOPS = ("TAA", "TAG", "TGA")


def codon_usage(cds):
    from Bio.Data.CodonTable import unambiguous_dna_by_id
    fw = unambiguous_dna_by_id[11].forward_table
    use = {}
    for s in cds:
        for i in range(3, len(s) - 3, 3):
            c = s[i:i + 3]
            if c in fw:
                use.setdefault(fw[c], {}).setdefault(c, 0)
                use[fw[c]][c] += 1
    return fw, use


def ameliorate(s, a, fw, use, rnd):
    if a <= 0:
        return s
    out = [s[:3]]
    for i in range(3, len(s) - 3, 3):
        c = s[i:i + 3]
        if c in fw and rnd.random() < a:
            syn = use.get(fw[c])
            if syn:
                c = rnd.choices(list(syn), list(syn.values()))[0]
        out.append(c)
    out.append(s[-3:])
    return "".join(out)


def good_cds(seqs, rnd, n):
    ok = []
    for s in seqs:
        s = s.upper()
        if 300 <= len(s) <= 3000 and len(s) % 3 == 0 and s[:3] == "ATG" and s[-3:] in STOPS \
                and not set(s) - set("ACGT") and "*" not in translate(s[:-3], 1):
            ok.append(s)
    return rnd.sample(ok, min(n, len(ok)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-donor", type=int, default=24)
    ap.add_argument("--amelioration", default="0,0.5,1")
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    cache = os.path.join(a.out, "cache")
    os.makedirs(cache, exist_ok=True)
    tax = taxonomy(config.REFERENCE)
    names = {v.lower(): k for k, v in tax.name.items()}
    contigs = {}
    for n, s in fetch_fasta(RECIPIENT["dna"], os.path.join(cache, "recipient.fa.gz")):
        contigs[n.split()[0]] = s.upper()
    rec = call_prokaryote(contigs)
    fw, use = codon_usage(rec.cds.values())
    # recipient core genes: pyrodigal proteins that match UniProt ribosomal proteins / RNAP subunits
    core_seq = {}
    for n, s in fetch_fasta(RECIPIENT["proteome"], os.path.join(cache, "recipient_proteome.fa.gz")):
        desc = n.split(" ", 1)[1].split(" OS=")[0] if " " in n else ""
        if CORE.search(desc) and not CORE_EXCLUDE.search(desc):
            core_seq[s[1:]] = desc
    levels = [float(x) for x in a.amelioration.split(",")]
    # insertion sites: gaps between consecutive recipient genes
    sites = {}
    for g, (c, s, e, st, _) in rec.coords.items():
        sites.setdefault(c, []).append((s, e))
    gaps = []
    for c, iv in sites.items():
        iv.sort()
        gaps += [(c, iv[i][1], iv[i + 1][0]) for i in range(len(iv) - 1) if iv[i + 1][0] - iv[i][1] > 30]
    islands = []
    for cls, name, url in DONORS:
        t = names.get(name.lower()) or names.get(name.split()[0].lower())      # species, else genus
        group = tax.group(t) if t else ""
        try:
            seqs = [s for _, s in fetch_fasta(url, os.path.join(cache, os.path.basename(url)))]
        except Exception as e:                    # noqa: BLE001
            print("skip %s: %s" % (name, e), file=sys.stderr)
            continue
        genes = good_cds(seqs, rnd, a.per_donor)
        k = 0
        while k < len(genes):                     # clusters of 1-4 genes, each cluster one amelioration level
            size = min(len(genes) - k, rnd.choice((1, 1, 2, 3, 4)))
            lev = levels[len(islands) % len(levels)]
            islands.append(dict(cls=cls, name=name, taxid=t or "", group=group, level=lev,
                                genes=[ameliorate(s, lev, fw, use, rnd) for s in genes[k:k + size]]))
            k += size
        print("  %-10s %-26s %-26s %d genes" % (cls, name, group, len(genes)), file=sys.stderr)
    rnd.shuffle(islands)
    chosen = rnd.sample(gaps, min(len(islands), len(gaps)))
    by_contig = {}
    for (c, after, before), isl in zip(chosen, islands):
        by_contig.setdefault(c, []).append((after + (before - after) // 2, isl))
    truth = []
    gc = sum(x.count("G") + x.count("C") for x in contigs.values()) / sum(map(len, contigs.values()))
    out_fa, inserted = [], {}
    for c, seq in contigs.items():
        ins = sorted(by_contig.get(c, []), key=lambda x: x[0])
        parts, pos, shift = [], 0, 0
        for at, isl in ins:
            parts.append(seq[pos:at])
            cur = at + shift
            added = 0
            for gseq in isl["genes"]:
                spacer = "".join(rnd.choice("GC" if rnd.random() < gc else "AT") for _ in range(60))
                strand = rnd.choice("+-")
                block = spacer + (gseq if strand == "+" else revcomp(gseq))
                start = cur + added + len(spacer) + 1
                truth.append(dict(contig=c, start=start, end=start + len(gseq) - 1, strand=strand, label="spike",
                                  true_class=isl["cls"], divergence=isl["level"], donor_taxid=isl["taxid"],
                                  donor_group=isl["group"], donor_name=isl["name"], island_size=len(isl["genes"])))
                parts.append(block)
                added += len(block)
            inserted.setdefault(c, []).append((at, added))      # (original position, inserted length)
            shift += added
            pos = at
        parts.append(seq[pos:])
        out_fa.append((c, "".join(parts)))
    for g, (c, st0, en0, strand, _) in rec.coords.items():
        desc = core_seq.get(rec.proteins[g][1:])
        if not desc:
            continue
        shift = sum(n for at, n in inserted.get(c, []) if at < st0)
        truth.append(dict(contig=c, start=st0 + shift, end=en0 + shift, strand=strand, label="core", true_class="native",
                          divergence="", donor_taxid="", donor_group="", donor_name=desc[:60], island_size=""))
    with open(os.path.join(a.out, "genome.fa"), "w") as fh:
        for c, s in out_fa:
            fh.write(">%s\n" % c)
            for i in range(0, len(s), 80):
                fh.write(s[i:i + 80] + "\n")
    keys = ["contig", "start", "end", "strand", "label", "true_class", "divergence", "donor_taxid", "donor_group",
            "donor_name", "island_size"]
    with open(os.path.join(a.out, "truth.tsv"), "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for t in truth:
            fh.write("\t".join(str(t[k]) for k in keys) + "\n")
    print("%d genes inserted in %d clusters; %d core genes; genome %.2f Mb -> %s" % (
        sum(t["label"] == "spike" for t in truth), len(islands), sum(t["label"] == "core" for t in truth),
        sum(len(s) for _, s in out_fa) / 1e6, a.out), file=sys.stderr)


if __name__ == "__main__":
    main()
