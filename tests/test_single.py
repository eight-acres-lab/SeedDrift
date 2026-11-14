import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from seeddrift.donors import Taxonomy  # noqa: E402
from seeddrift.single import classify_gene, home_group, infer_self  # noqa: E402

NODES = [  # taxid, parent, rank, name
    (1, 1, "no rank", "root"), (2, 1, "superkingdom", "Bacteria"), (1224, 2, "phylum", "Pseudomonadota"),
    (1236, 1224, "class", "Gammaproteobacteria"), (91347, 1236, "order", "Enterobacterales"),
    (543, 91347, "family", "Enterobacteriaceae"), (561, 543, "genus", "Escherichia"),
    (562, 561, "species", "Escherichia coli"), (590, 543, "genus", "Salmonella"),
    (28901, 590, "species", "Salmonella enterica"), (1239, 2, "phylum", "Bacillota"),
    (1386, 1239, "genus", "Bacillus"), (1423, 1386, "species", "Bacillus subtilis"),
    (2759, 1, "superkingdom", "Eukaryota"), (33208, 2759, "kingdom", "Metazoa"),
    (7742, 33208, "clade", "Vertebrata"), (40674, 7742, "class", "Mammalia"), (9605, 40674, "genus", "Homo"),
    (9606, 9605, "species", "Homo sapiens"), (6231, 33208, "phylum", "Nematoda"),
    (6237, 6231, "genus", "Caenorhabditis"), (6239, 6237, "species", "Caenorhabditis elegans"),
    (4751, 2759, "kingdom", "Fungi"), (4932, 4751, "species", "Saccharomyces cerevisiae"),
]


def hit(q, t, taxid, e, bits, pident=50.0, qcov=0.9, qs=1, qe=100, ts=1, te=100):
    return dict(query=q, target=t, taxid=taxid, evalue=e, bits=bits, pident=pident, qcov=qcov,
                qstart=qs, qend=qe, tstart=ts, tend=te)


class TestSingleGenome(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.TemporaryDirectory()
        with open(os.path.join(cls.d.name, "taxonomy.tsv"), "w") as fo:
            for t, p, r, n in NODES:
                fo.write("%d\t%d\t%s\t%s\n" % (t, p, r, n))
        cls.tax = Taxonomy(cls.d.name)

    @classmethod
    def tearDownClass(cls):
        cls.d.cleanup()

    def run1(self, hits, self_taxid=561, qreg=None):
        return classify_gene("g", 100, hits, self.tax, self_taxid, home_group(self.tax, self_taxid), qreg=qreg)

    def test_home_group(self):
        self.assertEqual(home_group(self.tax, 561)[0], 1224)        # prokaryote: its phylum
        self.assertEqual(home_group(self.tax, 6239)[0], 33208)      # animal: Metazoa

    def test_native(self):
        r = self.run1([hit("g", "s1", 28901, 1e-80, 300), hit("g", "b1", 1423, 1e-10, 60)])
        self.assertEqual(r["category"], "native")
        self.assertLess(float(r["AI"]), 0)

    def test_mammal_in_bacterium(self):
        r = self.run1([hit("g", "h1", 9606, 1e-50, 200), hit("g", "b1", 1423, 1e-5, 40)])
        self.assertEqual(r["category"], "mammal")
        self.assertEqual(r["strength"], "strong")
        self.assertEqual(r["support"], 1)                         # one species only: flagged in summaries

    def test_other_prokaryote_in_bacterium(self):
        r = self.run1([hit("g", "b1", 1423, 1e-40, 180), hit("g", "s1", 28901, 1e-10, 60)])
        self.assertEqual(r["category"], "prokaryote")
        self.assertAlmostEqual(float(r["AI"]), 69.1, places=0)

    def test_mammal_in_nematode_uses_non_mammal_animals(self):
        r = self.run1([hit("g", "h1", 9606, 1e-60, 250), hit("g", "e1", 562, 1e-20, 90)], self_taxid=6237)
        self.assertEqual(r["category"], "mammal")

    def test_prokaryote_in_nematode(self):
        r = self.run1([hit("g", "e1", 562, 1e-80, 300), hit("g", "h1", 9606, 1e-10, 60)], self_taxid=6237)
        self.assertEqual(r["category"], "prokaryote")
        self.assertEqual(r["strength"], "strong")

    def test_self_hits_only_is_orphan(self):
        self.assertEqual(self.run1([hit("g", "e1", 562, 1e-90, 400, pident=100)])["category"], "orphan")

    def test_low_coverage_ignored(self):
        self.assertEqual(self.run1([hit("g", "h1", 9606, 1e-30, 120, qcov=0.2)])["category"], "orphan")

    def test_repeat_driven(self):
        hs = [hit("g", "h1", 9606, 1e-30, 150, pident=22.0, qs=10, qe=90),
              hit("g", "h2", 9606, 1e-25, 130, pident=21.0, qs=10, qe=90)]
        self.assertEqual(self.run1(hs, qreg=[(1, 100)])["category"], "repeat")
        # the same hits at 60% identity are real homologs and are kept
        hs = [dict(h, pident=60.0) for h in hs]
        self.assertEqual(self.run1(hs, qreg=[(1, 100)])["category"], "mammal")

    def test_infer_self(self):
        hs = [hit("g%d" % i, "e%d" % i, 562, 1e-90, 400, pident=99.5) for i in range(10)]
        hs += [hit("x", "s", 28901, 1e-90, 400, pident=99.0)]
        t, rank, n = infer_self(hs, self.tax)
        self.assertEqual((t, rank, n), (561, "genus", 10))


if __name__ == "__main__":
    unittest.main()
