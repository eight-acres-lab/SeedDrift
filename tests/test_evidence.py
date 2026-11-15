import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from seeddrift.donors import Taxonomy  # noqa: E402
from seeddrift.evidence import composition, confidence, islands, tree_verdict  # noqa: E402
from test_single import NODES  # noqa: E402


class TestEvidence(unittest.TestCase):
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

    def test_tree_outside(self):
        # the gene sits among Bacillus sequences, far from the Salmonella ones: outside the Pseudomonadota
        nwk = "((QUERY:0.1,(b1|1423:0.1,b2|1423:0.1):0.1):0.2,(b3|1423:0.2,b4|1423:0.2):0.1,(s1|28901:0.5,s2|28901:0.5):0.5);"
        v = tree_verdict(nwk, self.tax, 1224)
        self.assertEqual(v["tree"], "outside")
        self.assertEqual(v["tree_group"], "Bacillota")

    def test_tree_home(self):
        nwk = "((QUERY:0.1,(s1|28901:0.1,s2|28901:0.1):0.1):0.2,(s3|28901:0.2,b4|1423:0.9):0.1,(b1|1423:0.5,b2|1423:0.5):0.5);"
        self.assertEqual(tree_verdict(nwk, self.tax, 1224)["tree"], "home")

    def test_tree_missing(self):
        self.assertEqual(tree_verdict("", self.tax, 1224)["tree"], "na")
        self.assertEqual(tree_verdict("(QUERY:1,a|1423:1);", self.tax, 1224)["tree"], "na")

    def test_islands(self):
        rows = [dict(gene_id="g%d" % i, category="prokaryote" if i in (3, 4, 6, 20) else "native") for i in range(30)]
        coords = {"g%d" % i: ("c1", i * 1000, i * 1000 + 900, "+", 1) for i in range(30)}
        size = islands(rows, coords)
        self.assertEqual(size["g3"], 3)                 # 3, 4 and 6 (gap of one gene)
        self.assertEqual(size["g20"], 1)

    def test_composition(self):
        rnd = random.Random(1)
        cds = {"g%d" % i: "".join(rnd.choice("ACGT") for _ in range(900)) for i in range(60)}
        cds["at"] = "".join(rnd.choice("AATTTAGC") for _ in range(900))
        comp = composition(cds)
        self.assertLess(comp["at"][2], -3)
        self.assertLess(abs(comp["g1"][2]), 3)

    def test_confidence(self):
        no_home = dict(category="mammal", AI="80", support="3", tree="na", best_pident="70", best_home_pident="")
        self.assertEqual(confidence(no_home), "high")                       # no relative in the home group
        self.assertEqual(confidence(dict(no_home, support="1")), "medium")   # a single donor species
        near = dict(no_home, best_home_pident="62", tree="outside")          # gap 8: not enough for high
        self.assertEqual(confidence(near), "medium")
        self.assertEqual(confidence(dict(near, best_home_pident="55")), "high")  # gap 15 + phylogeny
        self.assertEqual(confidence(dict(near, best_home_pident="68")), "low")   # near-tie
        self.assertEqual(confidence(dict(near, tree="home")), "low")
        self.assertEqual(confidence(dict(no_home, contig_flag="prokaryote-like contig")), "low")
        self.assertEqual(confidence(dict(no_home, category="native")), "")


if __name__ == "__main__":
    unittest.main()
