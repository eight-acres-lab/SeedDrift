import gzip
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from seeddrift.structure import _is_repeat, _overlap, _pfam_meta, coil_regions, tandem_regions  # noqa: E402

DAT = """# STOCKHOLM 1.0
#=GF ID   LRR_8
#=GF AC   PF13855.10
#=GF TP   Repeat
#=GF CL   CL0022
//
#=GF ID   LRR_x
#=GF AC   PF99991.1
#=GF TP   Family
#=GF CL   CL0022
//
#=GF ID   Pkinase
#=GF AC   PF00069.29
#=GF TP   Domain
#=GF CL   CL0016
//
#=GF ID   Kdo
#=GF AC   PF06293.18
#=GF TP   Family
#=GF CL   CL0016
//
#=GF ID   CC_x
#=GF AC   PF99992.1
#=GF TP   Coiled-coil
//
"""


class TestRepeatFilter(unittest.TestCase):
    def test_overlap(self):
        self.assertAlmostEqual(_overlap([(1, 10), (20, 30)], 5, 24), 11 / 20.0)
        self.assertEqual(_overlap([(1, 10), (5, 15)], 1, 15), 1.0)       # overlapping regions counted once
        self.assertEqual(_overlap([], 1, 5), 0.0)
        self.assertEqual(_overlap([(100, 200)], 1, 50), 0.0)

    def test_repeat_either_side(self):
        f = ["q", "t", "0", "0", "0", "0", "0", "0", "200", "240", "10", "60"]   # qstart qend tstart tend
        self.assertTrue(_is_repeat([], [(1, 100)], f, 8, 10))       # target segment on an ankyrin array
        self.assertTrue(_is_repeat([(190, 260)], [], f, 8, 10))     # query segment on a repeat
        self.assertFalse(_is_repeat([(1, 50)], [(70, 90)], f, 8, 10))
        self.assertFalse(_is_repeat([(190, 260)], [], f[:8], 8, 10))  # old tables without coordinates

    def test_tandem_repeats(self):
        import random
        unit = "WESAHLAGRRCRVEIRPPCRAPVPCGNPPTLPRAGAV"           # 37-aa unit of a repeat-derived ORF
        seq = "MSTKLLDEAQRTVLSGHIK" + unit * 3 + "KLIVDNEQAGHMKRTPLSAAGHVILKRTE"
        reg = tandem_regions(seq)
        self.assertEqual(len(reg), 1)
        self.assertLessEqual(reg[0][0], 25)
        self.assertGreaterEqual(reg[0][1], 19 + 3 * len(unit) - 5)
        rnd = random.Random(3)
        for _ in range(20):                                   # random sequences: no repeat
            self.assertEqual(tandem_regions("".join(rnd.choice("ARNDCQEGHILKMFPSTWYV") for _ in range(300))), [])
        self.assertEqual(tandem_regions("MKV"), [])

    def test_coils_missing_binary(self):
        self.assertEqual(coil_regions({"a": "MKV" * 20}, "no-such-ncoils-binary"), {})

    def test_pfam_meta(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "Pfam-A.hmm.dat.gz")
            with gzip.open(p, "wt") as fh:
                fh.write(DAT)
            m = _pfam_meta(p)
        self.assertTrue(m["PF13855"][3])          # Repeat type
        self.assertTrue(m["PF99991"][3])          # Family in a clan that is half Repeat
        self.assertTrue(m["PF99992"][3])          # Coiled-coil type
        self.assertFalse(m["PF00069"][3])         # kinase: a real domain, kept
        self.assertEqual(m["PF00069"][0], "Pkinase")


if __name__ == "__main__":
    unittest.main()
