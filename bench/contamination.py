"""Contamination benchmark data: a eukaryote assembly with bacterial contigs mixed in.

The eukaryote contigs (Encephalitozoon cuniculi GB-M1) are real; N fragments of a bacterial genome
(Legionella pneumophila Philadelphia 1) are added as extra contigs named contam_*. A good screen
flags the contam_* contigs as contamination and does not report their genes as confident transfers.

usage:  python bench/contamination.py --out bench/work/contam_ecuniculi [--n 6 --size 60000]
"""
from __future__ import annotations

import argparse
import gzip
import os
import random
import urllib.request

EUK = ("https://ftp.ensemblgenomes.ebi.ac.uk/pub/release-63/fungi/fasta/fungi_microsporidia1_collection/"
       "encephalitozoon_cuniculi_gb_m1_gca_000091225/dna/Encephalitozoon_cuniculi_gb_m1_gca_000091225.ASM9122v2.dna.toplevel.fa.gz")
BAC = ("https://ftp.ensemblgenomes.ebi.ac.uk/pub/release-63/bacteria/fasta/bacteria_16_collection/"
       "legionella_pneumophila_subsp_pneumophila_str_philadelphia_1_gca_000008485/dna/"
       "Legionella_pneumophila_subsp_pneumophila_str_philadelphia_1_gca_000008485.ASM848v1_.dna.toplevel.fa.gz")


def fasta(url, cache):
    if not os.path.exists(cache):
        with urllib.request.urlopen(url, timeout=600) as r, open(cache, "wb") as fh:
            fh.write(r.read())
    out, name, buf = {}, None, []
    for line in gzip.open(cache, "rt"):
        line = line.strip()
        if line.startswith(">"):
            if name:
                out[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line)
    if name:
        out[name] = "".join(buf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--size", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rnd = random.Random(a.seed)
    euk = fasta(EUK, os.path.join(a.out, "euk.fa.gz"))
    bac = fasta(BAC, os.path.join(a.out, "bac.fa.gz"))
    chrom = max(bac.values(), key=len)
    with open(os.path.join(a.out, "genome.fa"), "w") as fh, open(os.path.join(a.out, "truth.tsv"), "w") as tf:
        tf.write("contig\tlabel\n")
        for k, v in euk.items():
            fh.write(">%s\n%s\n" % (k, v))
            tf.write("%s\teukaryote\n" % k)
        for i in range(a.n):
            s = rnd.randrange(0, len(chrom) - a.size)
            fh.write(">contam_%d\n%s\n" % (i + 1, chrom[s:s + a.size]))
            tf.write("contam_%d\tcontamination\n" % (i + 1))
    print("%d eukaryote contigs (%.1f Mb) + %d bacterial fragments of %d kb" %
          (len(euk), sum(map(len, euk.values())) / 1e6, a.n, a.size // 1000))


if __name__ == "__main__":
    main()
