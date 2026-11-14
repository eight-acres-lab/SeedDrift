import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from seeddrift.inputs import (detect, from_genbank, from_gff, load, looks_like_cds, parse_metaeuk,  # noqa: E402
                              read_fasta_text, revcomp, translate)

rnd = random.Random(5)
CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT" if a + b + c not in ("TAA", "TAG", "TGA")]


def orf(n_codons):
    return "ATG" + "".join(rnd.choice(CODONS) for _ in range(n_codons)) + "TAA"


class TestInputs(unittest.TestCase):
    def test_detect(self):
        self.assertEqual(detect(">a\nMKV\n"), "fasta")
        self.assertEqual(detect("LOCUS       X\n"), "genbank")
        self.assertEqual(detect("##gff-version 3\nc1\t.\tCDS\t1\t9\t.\t+\t0\tID=x\n"), "gff")
        self.assertEqual(detect("c1\tsrc\tgene\t1\t9\t.\t+\t.\tID=g\n"), "gff")

    def test_protein_and_cds_fasta(self):
        g = load([b">p1\nMKVLAAGIEERSTQWL\n>p2\nMSTNPKPQRKTKRNTNRRPQDVKFPGG\n"])
        self.assertEqual(g.kind, "protein")
        cds = {"c%d" % i: orf(60) for i in range(20)}
        self.assertTrue(looks_like_cds(cds))
        g = load([("".join(">%s\n%s\n" % kv for kv in cds.items())).encode()])
        self.assertEqual(g.kind, "cds")
        self.assertEqual(g.proteins["c0"], translate(cds["c0"]))

    def test_gff_minus_strand_with_intron(self):
        exon1, intron, exon2 = orf(30)[:-3], "GTAAGT" + "A" * 40 + "TTTCAG", "GCTGCT" + "TAA"
        gene = exon1 + intron + exon2                 # gene on the plus strand ...
        contig = "C" * 50 + revcomp(gene) + "G" * 50  # ... placed on the minus strand
        L = len(gene)
        a2, b2 = 51, 51 + len(exon2) - 1              # exon2 comes first on the contig (minus strand)
        a1, b1 = 51 + len(exon2) + len(intron), 50 + L
        gff = ("##gff-version 3\n"
               "c1\tt\tgene\t51\t%d\t.\t-\t.\tID=gene1\n" % (50 + L) +
               "c1\tt\tmRNA\t51\t%d\t.\t-\t.\tID=tx1;Parent=gene1\n" % (50 + L) +
               "c1\tt\tCDS\t%d\t%d\t.\t-\t0\tID=cds1;Parent=tx1\n" % (a1, b1) +
               "c1\tt\tCDS\t%d\t%d\t.\t-\t0\tID=cds1;Parent=tx1\n" % (a2, b2) +
               "##FASTA\n>c1\n%s\n" % contig)
        g = from_gff(gff)
        self.assertEqual(list(g.proteins), ["gene1"])
        self.assertEqual(g.proteins["gene1"], translate(exon1 + exon2, 1))
        self.assertEqual(g.coords["gene1"][3:], ("-", 2))
        # the same with the FASTA as a second file, in either order
        head = gff.split("##FASTA")[0]
        g2 = load([(">c1\n%s\n" % contig).encode(), head.encode()])
        self.assertEqual(g2.proteins["gene1"], g.proteins["gene1"])

    def test_genbank(self):
        dna = "TT" + orf(40) + "AA"
        gb = ("LOCUS       ctg1 %d bp    DNA     linear   BCT 01-JAN-2026\n" % len(dna) +
              "FEATURES             Location/Qualifiers\n"
              "     CDS             3..%d\n" % (2 + len(orf(40))) +
              "                     /locus_tag=\"T_0001\"\n"
              "ORIGIN\n" +
              "".join("%9d %s\n" % (i + 1, dna[i:i + 60].lower()) for i in range(0, len(dna), 60)) + "//\n")
        g = from_genbank(gb)
        self.assertEqual(list(g.proteins), ["T_0001"])
        self.assertTrue(g.proteins["T_0001"].startswith("M"))
        self.assertEqual(g.coords["T_0001"][:4], ("ctg1", 3, 2 + len(orf(40)), "+"))

    def test_metaeuk_header(self):
        contigs = {"chr1": "A" * 2000}
        faa = (">P12345|chr1|+|250|1e-60|2|99|1199|99[99]:399[399]:301[301]|899[899]:1199[1199]:301[301]\nMKV\n"
               ">Q9|chr1|+|120|1e-20|2|99|1199|99[99]:399[399]:301[301]|899[899]:1199[1199]:301[301]\nMKA\n")
        g = parse_metaeuk(faa, contigs)
        self.assertEqual(len(g.proteins), 1)                      # one locus, the best-scoring model
        gid = list(g.proteins)[0]
        self.assertEqual(g.proteins[gid], "MKV")
        self.assertEqual(g.coords[gid][4], 2)
        self.assertEqual(len(g.cds[gid]), 602)

    def test_read_fasta(self):
        self.assertEqual(read_fasta_text(">a x\nAC\nGT\n>b\nT\n"), {"a": "ACGT", "b": "T"})


if __name__ == "__main__":
    unittest.main()
