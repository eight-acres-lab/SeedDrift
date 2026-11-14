import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from seeddrift.reconcile import DTLModel, GeneTreeDTL, prepare_tree  # noqa: E402
from seeddrift.tree import SpeciesTree, parse_newick, to_newick  # noqa: E402

SP6 = "(((A,B)AB,C)ABC,((D,E)DE,F)DEF)root;"


def setup(sp_txt, gene_txt, mapping):
    sp = SpeciesTree(parse_newick(sp_txt))
    model = DTLModel(sp)
    gene_sp = {g: sp.leaf_index[s] for g, s in mapping.items()}
    adj, names = prepare_tree(gene_txt, gene_sp, random.Random(0))
    leaf_sp = {i: gene_sp[nm] for i, nm in enumerate(names) if nm is not None}
    return sp, model, GeneTreeDTL(model, adj, leaf_sp), names


class TestNewick(unittest.TestCase):
    def test_roundtrip(self):
        t = parse_newick("((a:1,'b c':2)95:0.5,d:3);")
        self.assertEqual(to_newick(t), "((a:1,'b c':2):0.5,d:3);")
        self.assertEqual(t.children[0].support, 95.0)


class TestDTL(unittest.TestCase):
    ident = {c: c.upper() for c in "abcdef"}

    def test_congruent_rooted_and_unrooted(self):
        for g in ["(((a,b),c),((d,e),f));", "((a,b),c,((d,e),f));"]:
            _, _, rec, _ = setup(SP6, g, self.ident)
            self.assertEqual(rec.best_cost, 0.0)
            self.assertEqual(rec.sample(random.Random(1))["transfers"], [])

    def test_single_loss(self):
        m = {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}
        _, _, rec, _ = setup(SP6, "(((a,b),c),(d,e));", m)
        self.assertEqual(rec.best_cost, 1.0)  # F lost

    def test_transfer_F_to_A(self):
        m = dict(self.ident, x="A")
        sp, _, rec, names = setup(SP6, "(((a,b),c),((d,e),(f,x)));", m)
        self.assertEqual(rec.best_cost, 3.0)
        for seed in range(20):
            tr = rec.sample(random.Random(seed))["transfers"]
            self.assertEqual(len(tr), 1)
            self.assertEqual(sp.names[tr[0]["donor"]], "F")
            self.assertEqual(sp.names[tr[0]["recipient"]], "A")
            moved = {names[i] for i in rec.leaves(tr[0]["moved"])}
            self.assertEqual(moved, {"x"})

    def test_transfer_unrooted_input(self):
        m = dict(self.ident, x="A")
        sp, _, rec, _ = setup(SP6, "((a,b),c,((d,e),(f,x)));", m)
        self.assertEqual(rec.best_cost, 3.0)

    def test_dl_only_cost_higher(self):
        sp = SpeciesTree(parse_newick(SP6))
        model = DTLModel(sp)
        m = dict(self.ident, x="A")
        gene_sp = {g: sp.leaf_index[s] for g, s in m.items()}
        adj, names = prepare_tree("(((a,b),c),((d,e),(f,x)));", gene_sp, random.Random(0))
        leaf_sp = {i: gene_sp[nm] for i, nm in enumerate(names) if nm is not None}
        dl = GeneTreeDTL(model, adj, leaf_sp, allow_transfer=False)
        self.assertGreater(dl.best_cost, 3.0)


class TestCollapse(unittest.TestCase):
    """Unsupported incongruence disappears after collapsing; supported incongruence stays."""

    def cost(self, support, collapse):
        sp = SpeciesTree(parse_newick(SP6))
        model = DTLModel(sp)
        gene_sp = {g: sp.leaf_index[g.upper()] for g in "abcdef"}
        txt = "(((a:1,b:1)100:1,(c:1,f:1)%d:1)100:1,(d:1,e:1)100:1);" % support
        adj, names = prepare_tree(txt, gene_sp, random.Random(0), sp, collapse)
        leaf_sp = {i: gene_sp[nm] for i, nm in enumerate(names) if nm is not None}
        return GeneTreeDTL(model, adj, leaf_sp).best_cost

    def test_weak_branch_resolved_along_species_tree(self):
        self.assertEqual(self.cost(30, 95), 0)

    def test_supported_branch_kept(self):
        self.assertGreater(self.cost(100, 95), 0)

    def test_off_keeps_ml_topology(self):
        self.assertGreater(self.cost(30, 0), 0)


class TestParalogSplit(unittest.TestCase):
    """Ancient duplications (species overlap) split a family; a single extra copy does not."""

    def split(self, txt, mapping, **kw):
        from seeddrift.reconcile import split_paralogs
        sp = SpeciesTree(parse_newick(SP6))
        gene_sp = {g: sp.leaf_index[s] for g, s in mapping.items()}
        return split_paralogs("F", txt, gene_sp, DTLModel(sp), **kw)

    def test_ancient_duplication_is_split(self):
        # paralogs a* and b* in every species (2 copies per genome)
        txt = "((((a1,b1),c1),((d1,e1),f1)),(((a2,b2),c2),((d2,e2),f2)));"
        mapping = {g: g[0].upper() for g in "a1 b1 c1 d1 e1 f1 a2 b2 c2 d2 e2 f2".split()}
        groups = self.split(txt, mapping)
        self.assertEqual(sorted(map(sorted, groups)),
                         [["a1", "b1", "c1", "d1", "e1", "f1"], ["a2", "b2", "c2", "d2", "e2", "f2"]])

    def test_single_extra_copy_not_split(self):
        txt = "((((a1,b1),c1),((d1,e1),f1)),a2);"
        mapping = {g: g[0].upper() for g in "a1 b1 c1 d1 e1 f1 a2".split()}
        self.assertEqual(len(self.split(txt, mapping, min_copies=1.0)), 1)

    def test_low_copy_family_left_alone(self):
        txt = "((((a1,b1),c1),((d1,e1),f1)),(a2,b2));"
        mapping = {g: g[0].upper() for g in "a1 b1 c1 d1 e1 f1 a2 b2".split()}
        self.assertEqual(len(self.split(txt, mapping)), 1)          # 8 genes / 6 genomes < 1.5


class TestForbidClade(unittest.TestCase):
    def test_transfer_needed(self):
        sp, model, rec, _ = setup(SP6, "(((a,f),b),c,((d,e)));",
                                  {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E", "f": "F"})
        alt = GeneTreeDTL(model.forbid_clade(sp.index["F"]), rec.adj, rec.leaf_sp)
        self.assertGreater(alt.best_cost, rec.best_cost)


class TestGainParsimony(unittest.TestCase):
    """Gain nodes for families with no donor in the data (gain cost 3, loss cost 1)."""

    def gains(self, present):
        from seeddrift.scan import gain_nodes
        sp = SpeciesTree(parse_newick(SP6))
        out = gain_nodes(sp, set(sp.leaf_index[x] for x in present))
        return {sp.names[k]: sorted(sp.names[x] for x in v) for k, v in out.items()}

    def test_clade_restricted(self):
        self.assertEqual(self.gains("AB"), {"AB": ["A", "B"]})

    def test_one_gain_with_loss_beats_two_gains(self):
        self.assertEqual(self.gains("AC"), {"ABC": ["A", "C"]})     # 3 + 1 loss < 3 + 3

    def test_distant_patchy_family_gets_independent_gains(self):
        # root gain + losses in B, C, E, F = 7 > two leaf gains = 6
        self.assertEqual(self.gains("AD"), {"A": ["A"], "D": ["D"]})

    def test_everywhere_is_ancestral(self):
        self.assertEqual(self.gains("ABCDEF"), {})


if __name__ == "__main__":
    unittest.main()
