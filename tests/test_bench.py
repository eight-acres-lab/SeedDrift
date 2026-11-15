import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
from run_benchmark import auprc, eval_spikein, match_transplant  # noqa: E402


class TestBenchMetrics(unittest.TestCase):
    def test_auprc(self):
        self.assertEqual(auprc([10, 9], [1, 0]), 1.0)               # perfect ranking
        self.assertLess(auprc([1, 0], [10, 9]), 0.5)
        # one spike called, one not: the uncalled spike shares the lowest score with three uncalled natives
        self.assertEqual(auprc([5, -1e9], [-1e9, -1e9, -1e9]), 0.7)

    def test_match_transplant(self):
        pred = [dict(gene_id="a", contig="c", start="100", end="400", category="mammal", confidence="high",
                     donor_group="Mammalia", AI="80", comp_z="3.1", island="2"),
                dict(gene_id="b", contig="c", start="1000", end="1300", category="native", confidence="",
                     donor_group="", AI="-5", comp_z="0.2", island="")]
        truth = [dict(contig="c", start="110", end="400", label="spike", true_class="mammal", divergence="0",
                      donor_group="Mammalia", island_size="2"),
                 dict(contig="c", start="5000", end="5300", label="spike", true_class="mammal", divergence="0",
                      donor_group="Mammalia", island_size="1")]
        rows, labeled = match_transplant(pred, truth)
        lab = {t["gene_id"]: t["label"] for t in labeled}
        self.assertEqual(lab["a"], "spike")
        self.assertEqual(lab["b"], "other")
        self.assertIn("MISSED_1", lab)                              # not found by gene calling
        m = eval_spikein(rows, labeled)
        self.assertEqual(m["classes"]["mammal"]["0"]["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
