"""Benchmark v2 family A re-cut, increment 1 (ci/stages/family_a.py): the GTDB tree and taxonomy, the panel re-cut, the
outgroup choice and the genome checks, on synthetic inputs (no network)."""
import gzip
import hashlib
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from ci.stages import family_a as FA  # noqa: E402

# a GTDB-like tree: ((Esch reps),(Citro)) inside Enterobacteriaceae, then Klebsiella, then a far clade (Serratia and a
# forbidden genus)
TREE = ("((((RS_GCF_000000001.1:0.01,RS_GCF_000000002.1:0.02)'100.0:g__Escherichia':0.05,"
        "(RS_GCF_000000003.1:0.01,RS_GCF_000000004.1:0.01)'g__Citrobacter':0.04)'99.0':0.03,"
        "(RS_GCF_000000005.1:0.02)'g__Klebsiella':0.06)'f__Enterobacteriaceae':0.1,"
        "((RS_GCF_000000006.1:0.02)'g__Serratia':0.05,(RS_GCF_000000007.1:0.03)'g__Vibrio':0.04):0.2);")
GT = {"00000000%d" % i: dict(genus=g, species=s) for i, g, s in (
    (1, "Escherichia", "Escherichia coli"), (2, "Escherichia", "Escherichia albertii"), (3, "Citrobacter", "Citrobacter freundii"),
    (4, "Citrobacter", "Citrobacter braakii"), (5, "Klebsiella", "Klebsiella pneumoniae"), (6, "Serratia", "Serratia marcescens"),
    (7, "Vibrio", "Vibrio cholerae"))}


def test_tree_parser_mrca_and_sister_clades():
    t = FA.read_tree(TREE)
    leaves = [i for i, n in enumerate(t["name"]) if n]
    assert sorted(t["name"][i] for i in leaves) == ["RS_GCF_00000000%d.1" % i for i in range(1, 8)]
    esch = [i for i in leaves if t["name"][i] in ("RS_GCF_000000001.1", "RS_GCF_000000002.1")]
    m = FA.mrca(t, esch)
    assert ("genus", "Escherichia") in t["taxa"][m] and abs(t["length"][m] - 0.05) < 1e-12
    clades = FA.sister_clades(t, m)
    names = [sorted(t["name"][x] for r in roots for x in FA.leaves_under(t, r)) for roots, _ in clades]
    assert names[0] == ["RS_GCF_000000003.1", "RS_GCF_000000004.1"] and names[1] == ["RS_GCF_000000005.1"]
    assert names[2] == ["RS_GCF_000000006.1", "RS_GCF_000000007.1"]
    assert [round(d, 6) for _, d in clades] == [0.05, 0.08, 0.18]
    assert FA.gtdb_key("GB_GCA_000000003.2") == FA.gtdb_key("GCF_000000003.1") == "000000003"


def test_gtdb_taxonomy_and_files(tmp_path):
    p = tmp_path / "tax.tsv.gz"
    with gzip.open(p, "wt") as f:
        f.write("RS_GCF_000000001.1\td__Bacteria;p__Pseudomonadota;o__Enterobacterales;g__Escherichia;s__Escherichia coli\n")
    assert FA.gtdb_taxonomy(str(p)) == {"000000001": dict(domain="Bacteria", phylum="Pseudomonadota", order="Enterobacterales",
                                                             genus="Escherichia", species="Escherichia coli")}
    data = {"https://x/r/a.tsv.gz": b"AAA", "https://x/r/MD5SUM.txt": ("%s  ./a.tsv.gz\n" % hashlib.md5(b"AAA").hexdigest()).encode()}
    stored = {}

    def store(cas, b, suffix):
        stored[suffix] = b
        return "/cas/x" + suffix, hashlib.sha256(b).hexdigest()
    files, md5 = FA.gtdb_files(dict(base="https://x/r", files=["a.tsv.gz"]), "/cas", data.__getitem__, store)
    assert files["a.tsv.gz"]["md5"] == hashlib.md5(b"AAA").hexdigest() and md5["sha256"]
    data["https://x/r/a.tsv.gz"] = b"AAB"                                  # a changed file fails its md5
    with pytest.raises(SystemExit, match="md5"):
        FA.gtdb_files(dict(base="https://x/r", files=["a.tsv.gz"]), "/cas", data.__getitem__, store)


def test_recut_panel_quotas_and_order():
    pool = [dict(assembly="GCA_%09d.1" % i, gtdb=dict(genus="Escherichia", species=sp)) for i, sp in enumerate(
        ["Escherichia coli"] * 8 + ["Escherichia albertii"] * 2 + ["Escherichia fergusonii", "Escherichia marmotae"] + ["Citrobacter x"])]
    pool[-1]["gtdb"]["genus"] = "Citrobacter"
    same, other = FA.recut_panel("Escherichia coli", "Escherichia", pool, 5, 5, "panel-v2")
    key = lambda r: FA.order_key("%s|panel-v2" % r["assembly"])
    assert len(same) == 5 and same == sorted(same, key=key) and all(r["gtdb"]["species"] == "Escherichia coli" for r in same)
    assert sorted(r["gtdb"]["species"] for r in other) == ["Escherichia albertii", "Escherichia fergusonii", "Escherichia marmotae"]


class _Tax:
    """A taxonomy with genera under orders: Enterobacterales (91347) holds Escherichia, Citrobacter, Klebsiella,
    Serratia; Vibrionales holds Vibrio; species taxids 1000+genus."""
    def __init__(self):
        self.genus = {"Escherichia": 561, "Citrobacter": 544, "Klebsiella": 570, "Serratia": 613, "Vibrio": 662}
        self.family = {561: 543, 544: 543, 570: 543, 613: 1903411, 662: 641}
        self.order = {543: 91347, 1903411: 91347, 641: 135623}
        self.by_name = {k: [v] for k, v in self.genus.items()}
        self.rank = {v: "genus" for v in self.genus.values()}

    def at_rank(self, t, rank):
        g = t - 1000 if t > 1000 else t
        if rank == "genus":
            return g
        if rank == "family":
            return self.family[g]
        if rank == "order":
            return self.order[self.family[g]]
        return 0


def test_choose_outgroups_graded_eligible_and_reserved():
    tax = _Tax()
    tree = FA.read_tree(TREE)
    gt = dict(GT)
    rows_of = {}
    for g, n in (("Citrobacter", 6), ("Klebsiella", 1), ("Serratia", 4), ("Vibrio", 5)):
        rows_of[tax.genus[g]] = [dict(assembly="GCA_9%02d%06d.1" % (tax.genus[g] % 100, i), tax_id=str(1000 + tax.genus[g])) for i in range(n)]
        for i, r in enumerate(rows_of[tax.genus[g]]):
            gt[FA.gtdb_key(r["assembly"])] = dict(genus=g, species="%s sp%d" % (g, i % 2))
    forb = dict(genera={tax.genus["Vibrio"]}, families=set(), island_genera={561})
    ogs = FA.choose_outgroups(tree, gt, "Escherichia", lambda t: rows_of.get(t, []), tax, forb, 91347, set(), set(), k=3, n=2, min_asm=2, salt="outgroup-v1")
    assert [o["genus"] for o in ogs] == ["Citrobacter", "Serratia"]          # Klebsiella has 1 genome; Vibrio is forbidden
    assert ogs[0]["distance"] < ogs[1]["distance"]
    two = ogs[0]["genomes"][:2]
    assert len({g["gtdb"]["species"] for g in two}) == 2 and len(ogs[0]["genomes"]) == 6


def test_longest_n_run(tmp_path):
    fa = tmp_path / "a.fna"
    fa.write_text(">c1\nACGTNNNNACGT\nACGN\n>c2\nAC" + "N" * 120 + "GT\n")
    assert FA.longest_n_run(str(fa)) == 120
    gb = tmp_path / "b.gb.gz"
    with gzip.open(gb, "wt") as f:
        f.write("LOCUS x\nFEATURES\nORIGIN\n        1 acgtnnnnna cgt\n//\nLOCUS y\nORIGIN\n        1 acgtacgt\n//\n")
    assert FA.longest_n_run(str(gb)) == 5


def test_forbidden_genera_from_the_register_and_exclude():
    from ci.taxonomy import Taxonomy
    tax = Taxonomy(os.path.join(ROOT, "tests", "ci", "fixtures", "tiny", "taxdump"))
    forb = FA.forbidden(lambda rel: os.path.join(ROOT, rel), tax)
    assert "Pseudomonas" in forb["names"] and "Vibrio" in forb["names"] and "Escherichia" not in forb["names"]


def test_label_code_imports_no_score_or_search_module():
    import re
    from ci.stages.gates import FAMA_BANNED
    src = open(os.path.join(ROOT, "ci", "stages", "family_a.py")).read()
    for m in FAMA_BANNED:
        assert not re.search(r"from ci\.stages(\.%s| import [^\n]*\b%s\b)" % (m, m), src), m
