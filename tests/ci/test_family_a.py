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


# the amendment of 2026-11-21: a placeholder genus whose representative is an NCBI E. coli genome (ECMA, the sister of
# Escherichia) and one placed far off next to Klebsiella (Gk)
TREE2 = ("(((((RS_GCF_000000001.1:0.01,RS_GCF_000000002.1:0.01,RS_GCF_000000010.1:0.01)'g__Escherichia':0.001,"
         "GB_GCA_000000008.1:0.024):0.001,(RS_GCF_000000003.1:0.01,RS_GCF_000000004.1:0.01)'g__Citrobacter':0.04):0.03,"
         "(RS_GCF_000000005.1:0.02,GB_GCA_000000009.1:0.024):0.06):0.1,(RS_GCF_000000006.1:0.02,RS_GCF_000000007.1:0.03):0.2);")
GT2 = {"%09d" % i: dict(genus=g, species=sp) for i, g, sp in (
    (1, "Escherichia", "Escherichia coli"), (2, "Escherichia", "Escherichia albertii"), (10, "Escherichia", "Escherichia fergusonii"),
    (8, "ECMA", "ECMA sp1"), (3, "Citrobacter", "Citrobacter freundii"), (4, "Citrobacter", "Citrobacter braakii"),
    (5, "Klebsiella", "Klebsiella pneumoniae"), (9, "Gk", "Gk sp1"), (6, "Serratia", "Serratia marcescens"), (7, "Vibrio", "Vibrio cholerae"))}


class _SpTax:
    def at_rank(self, t, rank):
        return t


def _pool2():
    rows = []
    for i, (n, tax, g, sp) in enumerate([(6, 562, "Escherichia", "Escherichia coli"), (2, 562, "ECMA", "ECMA sp1"), (1, 562, "Gk", "Gk sp1"),
                                         (2, 564, "Escherichia", "Escherichia albertii"), (1, 565, "Escherichia", "Escherichia fergusonii")]):
        rows += [dict(assembly="GCA_5%02d%06d.1" % (i, k), tax_id=str(tax), gtdb=dict(genus=g, species=sp)) for k in range(n)]
    rows += [dict(assembly="GCA_000000008.1", tax_id="562", gtdb=dict(genus="ECMA", species="ECMA sp1")),     # the representatives
             dict(assembly="GCA_000000009.1", tax_id="562", gtdb=dict(genus="Gk", species="Gk sp1"))]
    return rows


def test_walk_panel_anchors_at_the_majority_genus():
    t, pool = FA.read_tree(TREE2), _pool2()
    esch = FA.genus_mrca(t, GT2, "Escherichia")
    for sp, g in (("ECMA sp1", "ECMA"), ("Gk sp1", "Gk")):
        node, species, rec = FA.walk_panel(t, GT2, sp, g, pool, _SpTax(), 3)
        assert node == esch and rec["start_genus"] == "Escherichia" and rec["steps"] == 0 and rec["majority_share"] == round(6 / 11, 4)
        assert species == {"Escherichia coli", "Escherichia albertii", "Escherichia fergusonii"}
        same, other = FA.recut_panel(sp, g, pool, 5, 5, "panel-v2", species)
        assert {r["gtdb"]["species"] for r in other} == species and {r["gtdb"]["species"] for r in same} == {sp}
    # no majority (6 of 11 is not above 0.6): the walk starts at the genus's own MRCA and climbs one node
    node, species, rec = FA.walk_panel(t, GT2, "ECMA sp1", "ECMA", pool, _SpTax(), 3, majority=0.6)
    assert rec["start_genus"] == "ECMA" and rec["steps"] == 1 and len(species) == 3
    # Gk without the anchor climbs to the clade over Klebsiella's and takes ECMA as an other species
    node, species, rec = FA.walk_panel(t, GT2, "Gk sp1", "Gk", pool, _SpTax(), 3, majority=0.6)
    assert rec["steps"] == 2 and "ECMA sp1" in species
    # a species of a GTDB genus named after a forbidden genus never counts: with Escherichia forbidden the walk climbs to Gk
    _, species, rec = FA.walk_panel(t, GT2, "ECMA sp1", "ECMA", pool, _SpTax(), 1, banned={"Escherichia"})
    assert species == {"Gk sp1"} and rec["steps"] == 3


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
    # amendment 2026-11-21: a walked unit takes its outgroups above its node: Gk from Escherichia's MRCA gets Citrobacter
    # first, from its own place Klebsiella
    t2 = FA.read_tree(TREE2)
    gt2 = dict(GT2)
    rows2 = {}
    for g in ("Citrobacter", "Klebsiella", "Serratia"):
        rows2[tax.genus[g]] = [dict(assembly="GCA_7%02d%06d.1" % (tax.genus[g] % 100, i), tax_id=str(1000 + tax.genus[g])) for i in range(2)]
        for i, r in enumerate(rows2[tax.genus[g]]):
            gt2[FA.gtdb_key(r["assembly"])] = dict(genus=g, species="%s sp%d" % (g, i))
    args = (gt2, "Gk", lambda x: rows2.get(x, []), tax, forb, 91347, set(), set())
    walked = FA.choose_outgroups(t2, *args, k=3, n=2, min_asm=2, salt="outgroup-v1", start=FA.genus_mrca(t2, GT2, "Escherichia"))
    own = FA.choose_outgroups(t2, *args, k=3, n=2, min_asm=2, salt="outgroup-v1")
    assert [o["genus"] for o in walked] == ["Citrobacter", "Klebsiella", "Serratia"] and [o["genus"] for o in own][0] == "Klebsiella"
    # amendment 2026-11-21: a genus of reduced genomes is not eligible
    for r in rows_of[tax.genus["Citrobacter"]]:
        r["base_count"] = "600000"
    for r in rows_of[tax.genus["Serratia"]]:
        r["base_count"] = "5000000"
    ogs = FA.choose_outgroups(tree, gt, "Escherichia", lambda t: rows_of.get(t, []), tax, forb, 91347, set(), set(), k=3, n=2, min_asm=2,
                              salt="outgroup-v1", min_bases=2_000_000)
    assert [o["genus"] for o in ogs] == ["Serratia"] and ogs[0]["median_bases"] == 5_000_000
    assert FA.median_bases([dict(base_count="1"), dict(base_count="3"), dict(base_count="")]) == 2.0


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


# --- increment 2: presence and labels (bench/hard/V2_FAMILY_A.md part 2) ---------------------------------------------

def _loc(t, start, end, ident=90.0, cov=0.95, bits=100.0, rep="r1", strand="+"):
    return dict(target=t, rep=rep, strand=strand, start=start, end=end, ident=ident, cov=cov, bits=bits)


def test_parse_hits_chunks_and_merge():
    ch = FA.chunks({"r1": ("A" * 450_000, True)}, size=200_000, overlap=30_000)
    assert [(c[2], len(c[3])) for c in ch] == [(0, 200_000), (170_000, 200_000), (340_000, 110_000)]
    offs = {c[0]: (c[1], c[2]) for c in ch}
    lines = ["r1|0\tg1\t90\t100\t180001\t180300\t1\t100\t200\t150\n",         # in the overlap: seen twice
             "r1|170000\tg1\t90\t100\t10001\t10300\t1\t100\t200\t150\n",
             "r1|170000\tg1\t80\t100\t11000\t10701\t101\t200\t200\t120\n",      # minus strand: its own locus
             "r1|340000\tg1\t70\t90\t1\t270\t110\t200\t200\t90\n"]
    hits = FA.parse_hits(lines, offs)
    assert len(hits) == 3 and hits[1]["strand"] == "-" and hits[1]["start"] == 180_701
    loci = FA.merge_loci(hits)
    plus = sorted((x for x in loci if x["strand"] == "+"), key=lambda x: x["start"])
    assert len(plus) == 2 and plus[0]["cov"] == 0.5                         # 340 kb is far from 180 kb: two loci


def test_owned_and_calls_for_pos():
    names = ["a", "b", "c", "d"]
    order = [0, 1, 2, 3]
    loci = [_loc("a", 1000, 2000), _loc("b", 3000, 4000), _loc("c", 5000, 6000), _loc("d", 7000, 8000),
            _loc("x", 5100, 6000, bits=500.0),                              # another gene owns c's place
            _loc("b", 90000, 91000, ident=45.0, cov=0.6, bits=50.0)]        # a weaker homolog of b elsewhere
    own = FA.owned(loci, {"a", "b", "d", "x"})
    assert {x["target"] for x in own} == {"a", "b", "d", "x"}
    own = [x for x in own if x["target"] in names]
    bits, best = FA.calls_for(names, loci, own, False, order, near_bp=10_000)
    O, H, P = FA.CALL["ortho"], FA.CALL["hom"], FA.CALL["pos"]
    assert [bool(b & O) for b in bits] == [True, True, False, True]
    assert all(b & H for b in bits)                                        # c's locus is a homolog, not owned
    # pos needs an ortho neighbour on both sides: a is at the open end, c has no ortho, b and d see a and d / b
    assert [bool(b & P) for b in bits] == [False, True, False, False]
    bits_c, _ = FA.calls_for(names, loci, own, True, order, near_bp=10_000)
    assert bits_c[0] & P                                                   # closed: a's left neighbour is d, 5 kb away
    og_bits, og_best = FA.calls_for(names, [], [_loc("a", 1, 900, ident=33.0, cov=0.6)], False, order, best_at="og")
    assert og_bits[0] & FA.CALL["og"] and not og_bits[0] & O and og_best[0] is not None


def test_runs_bridge_wrap_and_floors():
    import numpy as np
    n = 12
    start = np.arange(n) * 1500 + 1
    end = start + 1200
    absent = np.zeros(n, bool)
    absent[[2, 3, 5, 6]] = True                                            # gene 4 is a one-gene gap
    r = FA.runs(absent, list(range(n)), start, end, False, 4000, 3, 1)
    assert r == [[2, 3, 4, 5, 6]]
    absent[:] = False
    absent[[0, 1, 10, 11]] = True
    assert FA.runs(absent, list(range(n)), start, end, False, 4000, 3, 1) == []           # two short pieces
    assert FA.runs(absent, list(range(n)), start, end, True, 1000, 3, 1) == [[10, 11, 0, 1]]  # joined over the origin


def test_site_states_and_n_calibration():
    L = dict(rep="r", start=1000, end=2000, strand="+")
    R = dict(rep="r", start=4000, end=5000, strand="+")
    assert FA.site_state(L, R, "+", "+", 10_000, False) == ("empty", 1999)
    assert FA.site_state(L, dict(R, start=60_000, end=61_000), "+", "+", 0, False)[0] == "replaced"
    assert FA.site_state(L, dict(R, rep="q"), "+", "+", 0, False)[0] == "broken"
    assert FA.site_state(L, dict(R, strand="-"), "+", "+", 0, False)[0] == "inverted"
    assert FA.site_state(None, R, "+", "+", 0, False)[0] == "flank_lost"
    # on a circular replicon the gap is the shorter way round
    assert FA.site_state(dict(L, start=10, end=500), dict(R, start=999_000, end=999_800), "+", "+", 1_000_000, True)[0] == "empty"
    lit = [("c", "L", 10_000 + i) for i in range(12)]
    own = [("c", "L", 10_000 + 500 * k) for k in range(12)]
    cal = FA.calibrate_n(lit, own)
    assert cal["calibrated"] and 1 <= cal["n_kb"] <= 10
    assert FA.calibrate_n(lit[:3], own) == dict(n_kb=5, calibrated=False, matched=3)


def _chromosome(n=40, seg=range(15, 21)):
    """A synthetic chromosome: n genes of 1 kb with 300 bp spacing; 4 other-species genomes holding every gene but the
    segment, in place and with the segment's flanks adjacent; 2 outgroup genera of 2 genomes holding none of the
    segment."""
    import numpy as np
    names = ["g%02d" % i for i in range(n)]
    coords = [("chr", 1 + 1300 * i, 1300 * i + 1000, "+") for i in range(n)]
    g = dict(names=names, ids=names, coords=coords, flags={x: [] for x in names}, closed=True)
    others = ["O%d" % k for k in range(4)]
    ogs = ["X1a", "X1b", "X2a", "X2b"]
    genomes = others + ogs
    calls = np.zeros((n, len(genomes)), np.uint8)
    loci = {}
    for c, a in enumerate(genomes):
        lp, x = [None] * n, 1
        for i in range(n):
            if i in seg:
                continue
            calls[i, c] = (FA.CALL["ortho"] | FA.CALL["hom"] | FA.CALL["og"] | FA.CALL["pos"]) if a in others else FA.CALL["og"]
            lp[i] = ["chr", x, x + 999, "+", 10_000_000, True]
            x += 1300
        loci[a] = lp
    spec = dict(labels=dict(other_min=3, band_share=0.5, min_bp=4000, min_genes=3, gap=1, core=0.8, margin_genes=2, outgroup_genera_min=2,
                            lacking=1.0, flank_genes=3, empty_bp=5000, max_bp=100000, flank_empty=1, flank_empty_or_replaced=2))
    return g, calls, genomes, others, [(a, 0 if a.startswith("X1") else 1) for a in ogs], loci, spec


def test_classify_insertion_core_and_exclusions():
    g, calls, genomes, others, ogs, loci, spec = _chromosome()
    cols, segs = FA.classify(g, calls, genomes, others, [], ogs, [False] * 40, spec, 1000, loci)
    cls = list(cols["class_"])
    assert len(segs) == 1 and segs[0]["passes"] and segs[0]["empty"] == 8
    assert cls[15] == cls[20] == "edge_in" and cls[16:20] == ["insertion"] * 4   # 1 kb margin removes the end genes
    assert cls[13] == cls[22] == "edge" and cls[12] == "core" and cls[0] == "core"
    # an outgroup that holds the segment takes the polarity away
    calls[16:20, genomes.index("X1a")] = FA.CALL["og"]
    cols, _ = FA.classify(g, calls, genomes, others, [], ogs, [False] * 40, spec, 1000, loci)
    assert list(cols["class_"])[16:20] == ["no_polarity"] * 4
    # flanks far apart in every genome: no empty site, the run fails
    g, calls, genomes, others, ogs, loci, spec = _chromosome()
    for a in genomes:
        for i in range(21, 40):
            loci[a][i][1] += 200_000
            loci[a][i][2] += 200_000
    cols, segs = FA.classify(g, calls, genomes, others, [], ogs, [False] * 40, spec, 1000, loci)
    assert not segs[0]["passes"] and list(cols["class_"])[16:20] == ["no_flank"] * 4
    # QC, multi-copy and a thin panel
    g, calls, genomes, others, ogs, loci, spec = _chromosome()
    g["flags"]["g17"] = ["frame"]
    multi = [False] * 40
    multi[5] = True
    cols, _ = FA.classify(g, calls, genomes, others, [], ogs, multi, spec, 1000, loci)
    assert cols["class_"][17] == "qc_fail" and cols["class_"][5] == "unresolved"
    g, calls, genomes, others, ogs, loci, spec = _chromosome()
    cols, _ = FA.classify(g, calls, genomes, others[:2], [], ogs, [False] * 40, spec, 1000, loci)
    assert set(cols["class_"]) == {"no_panel"}


def test_classify_run_over_the_origin_keeps_the_core():
    """A segment over the origin of a circular chromosome is measured along it: the genes far from it stay core."""
    g, calls, genomes, others, ogs, loci, spec = _chromosome(seg=(37, 38, 39, 0, 1, 2))
    g["length"] = 40 * 1300
    for a in genomes:                                       # the other genomes' loci: the segment's flanks adjacent
        lp, x = [None] * 40, 1
        for i in list(range(3, 37)):
            lp[i] = ["chr", x, x + 999, "+", 34 * 1300, True]  # a circular genome without the segment: 36 and 3 adjacent
            x += 1300
        loci[a] = lp
    cols, segs = FA.classify(g, calls, genomes, others, [], ogs, [False] * 40, spec, 1000, loci)
    cls = list(cols["class_"])
    assert len(segs) == 1 and segs[0]["passes"] and segs[0]["bp"] == 6 * 1300 - 300
    assert cls[38] == cls[39] == cls[0] == cls[1] == "insertion" and cls[37] == cls[2] == "edge_in"
    assert cls[20] == "core" and cls[4] == "edge"


def test_label_gate_rechecks_the_rules(tmp_path):
    """G4_famA_labels refuses an insertion that breaks its rule as written in the label table."""
    import json
    import numpy as np
    import yaml
    from ci.stages import gates as G
    run = tmp_path / "run"
    base = run / "data" / "famA"
    (base / "labels" / "segments").mkdir(parents=True)
    (base / "presence").mkdir(parents=True)
    (run / "gates").mkdir()
    (base / "taxa.json").write_text(json.dumps(dict(placement={"NC_1": {}})))
    rows = [("g1", "core", "0.0000", "1.0000", "-1", "False", "False", "", "", "False", "False", "8000.0"),
            ("g2", "insertion", "0.0000", "0.0000", "0", "True", "False", "", "9000.0", "True", "True", "0.0")]
    with open(base / "labels" / "NC_1.tsv", "w") as f:
        f.write("gene_id\tclass\tf_ortho\tf_pos\trun\tog_absent\tmulti\tqc\tdist_edge\tin_u\tnear_u\tdist_u\tband\n")
        for r in rows:
            f.write("\t".join(r) + "\tFalse\n")
    (base / "labels" / "segments" / "NC_1.json").write_text(json.dumps([dict(segment=0, passes=True)]))
    (base / "labels" / "index.json").write_text(json.dumps(dict(calibration=dict(n_kb=5, calibrated=False), counts={"NC_1": {}})))
    np.savez_compressed(base / "presence" / "NC_1.npz", names=np.array(["g1", "g2"]))
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump(dict(labels=dict(core=0.8))))

    class St:
        def repo(self, rel):
            return str(spec) if rel == "SPEC" else os.path.join(ROOT, rel)
    cfg = dict(famA=dict(spec="SPEC"))
    G.phase_famA_labels(St(), cfg, str(run))
    assert json.load(open(run / "gates" / "G4_famA_labels.json"))["status"] == "pass"
    (base / "labels" / "segments" / "NC_1.json").write_text(json.dumps([dict(segment=0, passes=False)]))
    with pytest.raises(Exception):
        G.phase_famA_labels(St(), cfg, str(run))
    # a core gene nearer than N to a U run breaks its rule too
    (base / "labels" / "segments" / "NC_1.json").write_text(json.dumps([dict(segment=0, passes=True)]))
    txt = (base / "labels" / "NC_1.tsv").read_text().replace("\t8000.0\t", "\t3000.0\t")
    (base / "labels" / "NC_1.tsv").write_text(txt)
    with pytest.raises(Exception):
        G.phase_famA_labels(St(), cfg, str(run))


def test_review_fixes_loci_owned_pos_and_arc():
    import numpy as np
    # alternative HSPs on the protein do not add up; disjoint pieces of one protein do
    h = lambda a, b, s0, s1, bits: dict(target="g", rep="r", strand="+", start=a, end=b, pident=80.0, length=b - a + 1, sstart=s0, send=s1, slen=400, bits=bits)
    loc = FA.merge_loci([h(1000, 1720, 1, 240, 300.0), h(3000, 3480, 241, 400, 200.0), h(1010, 1700, 5, 235, 250.0)])
    assert len(loc) == 1 and loc[0]["cov"] == 1.0 and loc[0]["bits"] == 500.0
    # owned: a place on the other strand is still one place
    own = FA.owned([_loc("A", 1000, 1900, bits=300.0), _loc("B", 1200, 1800, bits=80.0, strand="-")], {"A", "B"})
    assert [x["target"] for x in own] == ["A"]
    # pos: any ortho locus of a neighbour counts, and around the origin of a circular target replicon
    names, order = ["a", "b", "c"], [0, 1, 2]
    own = [_loc("a", 4_999_000, 4_999_900), _loc("b", 100, 1000), _loc("c", 2000, 3000), _loc("c", 900_000, 901_000, bits=500.0)]
    bits, _ = FA.calls_for(names, own, own, True, order, 10_000, rep_info={"r1": (5_000_000, True)})
    # a and b: near around the origin, and b sees c's in-place copy though c's best copy is far; c's own best is far
    assert [bool(b & FA.CALL["pos"]) for b in bits] == [True, True, False]
    bits, _ = FA.calls_for(names, own, own, True, order, 10_000, rep_info={"r1": (5_000_000, False)})
    assert not bits[0] & FA.CALL["pos"] and bits[1] & FA.CALL["pos"]
    # a run's extent covers a nested gene; over the origin it splits at the wrap
    start, end = np.array([1000, 2300, 3600, 3700]), np.array([2000, 3300, 12000, 4500])
    assert FA.arc([0, 1, 2, 3], start, end, 50_000) == (1000, 12000, 11001)
    start, end = np.array([100, 48000, 49500]), np.array([900, 49000, 50500])
    assert FA.arc([1, 2, 0], start, end, 51_000) == (48000, 900, 51_000 - 48000 + 1 + 900)


def test_record_genes_names_and_qc(tmp_path):
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
    from Bio.SeqRecord import SeqRecord
    body = "ATG" + "AAA" * 30 + "TAA"
    double = "ATG" + "AAA" * 29 + "TAATAA"                  # a stop before the last codon
    seq = "C" * 50 + body + "G" * 50 + double + "T" * 300 + "ATGAAACCC"
    L = len(seq) + 12
    seq = seq + "GGGAAATAAGGG"                              # the tail and head of a CDS joined over the origin
    rec = SeqRecord(Seq(seq), id="NC_X.1", name="NC_X", annotations=dict(molecule_type="DNA", topology="circular"))
    a = 50
    rec.features = [SeqFeature(FeatureLocation(a, a + len(body), 1), type="CDS", qualifiers=dict(locus_tag=["T/1"])),
                    SeqFeature(FeatureLocation(a + len(body) + 50, a + len(body) + 50 + len(double), 1), type="CDS", qualifiers=dict(locus_tag=["T2"])),
                    SeqFeature(CompoundLocation([FeatureLocation(L - 9, L, 1), FeatureLocation(0, 3, 1)]), type="CDS",
                               qualifiers=dict(locus_tag=["T3"], translation=["M" + "K" * 40]))]
    p = tmp_path / "x.gb"
    SeqIO.write(rec, str(p), "genbank")
    g = FA.record_genes(str(p))
    assert g["ids"] == ["T/1", "T2", "T3"] and g["names"][0] != "T/1"           # the cleaned name differs from the record id
    assert g["flags"][g["names"][0]] == [] and "frame" in g["flags"][g["names"][1]] and "origin" in g["flags"][g["names"][2]]
    assert g["coords"][2][1:3] == (L - 8, L) and g["records"] == 1
