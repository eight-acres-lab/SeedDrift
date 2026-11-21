"""Benchmark v2 family A, part 3 (ci/stages/family_a_eval.py and the unit functions of ci/stages/family_a.py): element
links and the merge rule, the 1:1 matching, the hit covariates and lk_og, the AUROC aggregation, the arm tables and the
evaluation gate, on synthetic inputs (no network, no DIAMOND)."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from ci.stages import family_a as FA  # noqa: E402
from ci.stages import family_a_eval as FE  # noqa: E402


def test_segment_links_is_like_and_merge_units():
    # chromosome X segment 0 (4 genes) shares 3 genes with Y segment 1 (5 genes): linked (3 >= max(3, 0.5 x 4));
    # an IS-like gene x9 hits 3 segments of Z and links nothing
    seg_of, size_of, hits = {}, {}, []
    for c, k, n in (("X", 0, 4), ("Y", 1, 5), ("Z", 0, 3), ("Z", 1, 3), ("Z", 2, 3), ("X", 9, 1)):
        size_of[(c, k)] = n
        for i in range(n):
            seg_of["%s|%d|g%d" % (c, k, i)] = (c, k)
    for i in range(3):
        hits += [("X|0|g%d" % i, "Y|1|g%d" % i), ("Y|1|g%d" % i, "X|0|g%d" % i)]
    for k in range(3):
        hits += [("X|9|g0", "Z|%d|g0" % k), ("Z|%d|g0" % k, "X|9|g0")]
    hits += [("X|0|g3", "Y|1|g4")]                                       # one way only: not reciprocal
    links = FA.segment_links(hits, seg_of, size_of, 3, 0.5, 3)
    assert links == [(("X", 0), ("Y", 1), 3)]
    comps = FA.components(sorted(size_of), [(a, b) for a, b, _ in links])
    assert [("X", 0), ("Y", 1)] in comps and len(comps) == 5
    units, cross = FA.merge_units([("elA_1", [("X", 0), ("Y", 1)]), ("elA_2", [("Z", 0)])], {"X": "Escherichia", "Y": "Yersinia", "Z": "Salmonella"},
                                  {"Escherichia", "Yersinia", "Salmonella"})
    assert units == {"Escherichia+Yersinia": ["Escherichia", "Yersinia"], "Salmonella": ["Salmonella"]}
    assert cross == [dict(cluster="elA_1", genera=["Escherichia", "Yersinia"], segments=["X:0", "Y:1"])]


def test_insertion_segments_keep_qc_pass_single_copy_genes(tmp_path):
    p = tmp_path / "c.tsv"
    rows = [("a", "insertion", 0, "", "False"), ("b", "edge_in", 0, "", "False"), ("c", "insertion", 0, "frame", "False"),
            ("d", "insertion", 0, "", "True"), ("e", "core", -1, "", "False"), ("f", "no_flank", 1, "", "False")]
    with open(p, "w") as f:
        f.write("gene_id\tclass\trun\tqc\tmulti\tpublished\tstart\tend\n")
        for k, (g, c, r, q, m) in enumerate(rows):
            f.write("%s\t%s\t%d\t%s\t%s\tnone\t%d\t%d\n" % (g, c, r, q, m, 1 + 1000 * k, 900 + 1000 * k))
    tab = FA.read_label_table(str(p))
    assert FA.insertion_segments(tab) == {0: [0, 1]}                    # run 1 has no insertion; c fails QC, d is multi


def test_matching_removes_a_confounded_covariate():
    rng = np.random.default_rng(3)
    n_pos, n_neg = 60, 600
    length = np.concatenate([rng.normal(1.0, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)])      # positives are longer
    names = ["g%d" % i for i in range(n_pos + n_neg)]
    pos, neg = np.arange(n_pos), np.arange(n_pos, n_pos + n_neg)
    lab = np.concatenate([pos, neg])
    Q = FE.quantiles(length, lab)[:, None]
    m = FE.match_pairs("X", names, pos, neg, Q, np.zeros((len(names), 1)), 0.1)
    assert len(m) >= 0.8 * n_pos and len(set(m.values())) == len(m)                        # 1:1 without replacement
    from ci.stages import protocol as PR
    raw = PR.auroc(length[pos], length[neg])
    after = PR.auroc(length[list(m)], length[list(m.values())])
    assert raw > 0.7 and abs(after - 0.5) < 0.08
    qa, qb = Q[list(m), 0], Q[list(m.values()), 0]
    assert abs(FE.smd(qa, qb)) < 0.1
    assert FE.match_pairs("X", names, pos, neg, Q, np.zeros((len(names), 1)), 0.1) == m     # deterministic
    # an exact field that differs never matches
    exact = np.zeros((len(names), 1))
    exact[pos] = 1
    assert FE.match_pairs("X", names, pos, neg, Q, exact, 0.1) == {}


def test_hit_covariates_and_outgroup_leak():
    h = dict(off=np.array([0, 0, 3, 4]), prot=np.array([0, 1, 2, 1]), pid=np.array([900, 550, 400, 700]))
    rows, best = FE.hit_covariates(h)
    assert list(rows) == [0, 3, 1] and list(best) == [0.0, 90.0, 70.0]
    genus_rank = np.array([10, 20, 30])                                 # proteome -> NCBI genus taxid
    count, ident = FE.outgroup_leak(h, genus_rank, [{20}, {30, 31}, set()])
    assert list(count) == [0, 2, 1] and list(ident) == [0.0, 55.0, 70.0]


def test_arm_auroc_means_over_chromosomes_then_units():
    t = lambda y, v, u: dict(y=np.array(y), cols=dict(s=np.array(v, float)), unit=u)
    table = {"A": t([1, 0, 0], [3, 1, 2], "U1"), "B": t([1, 0], [0, 1], "U1"), "C": t([1, 0], [1, 0], "U2")}
    a, um, per = FE.arm_auroc(table, ["U1", "U2"], "s")
    assert per == {"A": 1.0, "B": 0.0, "C": 1.0} and um == {"U1": 0.5, "U2": 1.0} and a == 0.75


def _T(n=400, n_pos=30, seed=0):
    rng = np.random.default_rng(seed)
    cls = np.array(["core"] * n, dtype=object)
    cls[:n_pos] = "insertion"
    cls[n_pos:n_pos + 20] = "accessory"
    cols = dict(cov_len=rng.normal(size=n), cov_hom=rng.normal(size=n), cov_ident=rng.uniform(30, 90, n), tv_cai=rng.normal(size=n),
                tv_win_gc=rng.normal(size=n), tv_win_dev=rng.normal(size=n), s=rng.normal(size=n))
    return dict(names=["g%03d" % i for i in range(n)], cls=cls, published=np.array(["none"] * n, dtype=object), start=np.arange(n) * 1000.0,
                cols=cols, has_hit=np.ones(n), cat=np.array(["other"] * n, dtype=object), unit="U")


def test_arm_tables_eligibility_ladder_and_arms():
    spec = dict(eval=dict(min_pos=10, min_neg=100, caliper=0.1))
    T = {"A": _T(), "B": _T(n_pos=5, seed=1)}
    tables, record, ladder = FE.arm_tables(T, spec)
    assert record["B"]["evaluated"] is False and set(tables["core"]) == {"A"}
    t = tables["main"]["A"]
    assert (t["y"] == 1).sum() == (t["y"] == 0).sum() == len(t["pairs"]) == record["A"]["kept"]
    assert (tables["noise"]["A"]["y"] == 0).sum() == 20 and (tables["core"]["A"]["y"] == 0).sum() == 350
    assert list(ladder) == ["len", "len+hom", "len+hom+ident", "+cai", "+neighbourhood", "+category"]
    assert ladder["len+hom+ident"]["A"]["kept"] == record["A"]["kept"] and ladder["len"]["A"]["positives"] == 30


def _report(smd=0.02, diff=0.0, shuffle=0.5):
    arms = {a: {s: dict(auroc=0.6) for s in ("cmp_codon", "cov_len", "cov_hom", "cov_ident", "lk_og")} for a in ("main", "core", "noise")}
    arms["v1"] = {"cmp_codon": dict(auroc=0.7, audit=0.7, diff=diff)}
    return dict(chromosomes={"NC_002695.1": dict(evaluated=True)}, scores=["cmp_codon", "cov_len", "cov_hom", "cov_ident", "lk_og"], arms=arms,
                intervals=dict(main={c: dict(lo=0.45, hi=0.55, n_boot=1000) for c in FE.COVS}), balance=[dict(unit="U", covariate="cov_len", smd_after=smd, kept=30)],
                shuffle={"cmp_codon": shuffle}, lk_og=dict(auroc_main=0.55, flag=False), evaluated_units=["U"],
                ladder={"len+hom+ident": dict(kept=30, positives=40)})


@pytest.mark.parametrize("kw,ok", [({}, True), (dict(smd=0.3), False), (dict(diff=1e-6), False), (dict(shuffle=0.53), False), (dict(smd=None), False)])
def test_eval_gate(tmp_path, kw, ok):
    import yaml
    from ci.stages import gates as G
    run = tmp_path / "run"
    (run / "data" / "famA" / "eval").mkdir(parents=True)
    (run / "gates").mkdir()
    (run / "data" / "famA" / "eval" / "report.json").write_text(json.dumps(_report(**kw)))
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump(dict(eval=dict(smd_max=0.1, shuffle_tolerance=0.01, reproduce=1e-9, boot=1000))))

    class St:
        def repo(self, rel):
            return str(spec) if rel == "SPEC" else os.path.join(ROOT, rel)
    cfg = dict(famAe=dict(spec="SPEC"))
    if ok:
        G.phase_famA_eval(St(), cfg, str(run))
        assert json.load(open(run / "gates" / "G4_famA_eval.json"))["status"] == "pass"
    else:
        with pytest.raises(Exception):
            G.phase_famA_eval(St(), cfg, str(run))


def test_smd_and_tie_breaking():
    assert FE.smd([1.0, 1.0], [1.0, 1.0]) == 0.0
    assert FE.smd([1.0, 1.0], [2.0, 2.0]) == float("inf") and FE.smd([1.0], [2.0, 3.0]) is None
    # two negatives at the same distance: the one first in key order wins, whatever the float rounding
    names = ["p", "n1", "n2"]
    Q = np.array([[0.5], [0.4], [0.6]])
    m = FE.match_pairs("X", names, [0], [1, 2], Q, np.zeros((3, 1)), 0.2)
    first = min([1, 2], key=lambda i: FE.order_key("X|%s|match-v1" % names[i]))
    assert m == {0: first}


def test_link_threshold_counts_the_whole_run():
    seg_of, hits = {}, []
    for c, k in (("X", 0), ("Y", 0)):
        for i in range(3):
            seg_of["%s|%d|g%d" % (c, k, i)] = (c, k)
    for i in range(3):
        hits += [("X|0|g%d" % i, "Y|0|g%d" % i), ("Y|0|g%d" % i, "X|0|g%d" % i)]
    assert FA.segment_links(hits, seg_of, {("X", 0): 3, ("Y", 0): 3}) == [(("X", 0), ("Y", 0), 3)]
    assert FA.segment_links(hits, seg_of, {("X", 0): 12, ("Y", 0): 12}) == []          # 3 < max(3, 6): runs of 12 genes
