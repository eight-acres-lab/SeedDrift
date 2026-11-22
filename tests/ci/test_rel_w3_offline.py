"""The strict offline rendering: the aligner (seeddrift/train/rel/tables/align.py) against enumeration, its path against
its score, the realignment independent of hidden residues, and stage rel_w3_offline on a real-kind tables run built by
the tables package's own code (the fixture of test_rel_w2_stage.py): one part per condition, the fourth test on every
table, the scored cells those of the stored tables, the gate."""
import itertools
import json
import os

import numpy as np
import pytest

import rel_fixtures as RF
import test_rel_w2_stage as W2S
import test_rel_w5_roundtrip as RT
from ci.gates import GateFailure
from ci.stages.common import Stage
from ci.stages.rel import common, w3_offline
from seeddrift.train.rel import api as A
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import shards
from seeddrift.train.rel.tables import align as AL
from seeddrift.train.rel.tables import render as RD


def brute(f, q):
    """The best semi-global score by enumeration of the monotone pairings (tiny sequences)."""
    n, m = len(f), len(q)
    best = None
    for r in range(0, min(n, m) + 1):
        for qi in itertools.combinations(range(m), r):
            for fi in itertools.combinations(range(n), r):
                score = sum(int(AL.SCORE[f[a], q[b]]) for a, b in zip(fi, qi))
                cuts, fcuts = [-1] + list(qi) + [m], [None] + list(fi) + [None]
                for t in range(len(cuts) - 1):
                    u = cuts[t + 1] - cuts[t] - 1
                    v = fcuts[t + 1] - fcuts[t] - 1 if fcuts[t] is not None and fcuts[t + 1] is not None else 0
                    score -= (AL.GAP_OPEN + AL.GAP_EXTEND * u) if u > 0 else 0
                    score -= (AL.GAP_OPEN + AL.GAP_EXTEND * v) if v > 0 else 0
                best = score if best is None or score > best else best
    return best


def path_score(f, q, col):
    """The score of the pairing an alignment returned (pairs, gaps between them, unpaired residues of the other row)."""
    pairs = [(int(c), j) for j, c in enumerate(col) if c >= 0]
    score = sum(int(AL.SCORE[f[c], q[j]]) for c, j in pairs)
    cuts, fcuts = [-1] + [j for _, j in pairs] + [len(q)], [None] + [c for c, _ in pairs] + [None]
    for t in range(len(cuts) - 1):
        u = cuts[t + 1] - cuts[t] - 1
        v = fcuts[t + 1] - fcuts[t] - 1 if fcuts[t] is not None and fcuts[t + 1] is not None else 0
        score -= (AL.GAP_OPEN + AL.GAP_EXTEND * u) if u > 0 else 0
        score -= (AL.GAP_OPEN + AL.GAP_EXTEND * v) if v > 0 else 0
    return score


def test_the_matrix_and_the_aligner_against_enumeration():
    B = AL.BLOSUM62
    assert (B == B.T).all() and [int(B[i, i]) for i in range(20)] == [4, 5, 6, 6, 9, 5, 5, 6, 8, 4, 4, 5, 5, 6, 7, 4, 5, 11, 7, 4]
    assert (AL.SCORE[AL.X] == AL.UNKNOWN).all() and (AL.SCORE[:, AL.X] == AL.UNKNOWN).all()
    rng = np.random.default_rng(0)
    for _ in range(120):
        n, m = int(rng.integers(1, 6)), int(rng.integers(1, 5))
        f, q = rng.integers(0, 22, n), rng.integers(0, 22, m)
        cols, scores = AL.align(f, [q])
        assert scores[0] == brute(list(f), list(q)) == path_score(f, q, cols[0])
        v = cols[0][cols[0] >= 0]
        assert (np.diff(v) > 0).all() and (v < n).all()
    f = rng.integers(0, 20, 120)                                                 # a batch of rows of different lengths, against one by one
    others = [rng.integers(0, 20, int(rng.integers(1, 110))) for _ in range(6)]
    batch, scores = AL.align(f, others)
    for o, c, sc in zip(others, batch, scores):
        alone, s1 = AL.align(f, [o])
        assert np.array_equal(alone[0], c) and s1[0] == sc == path_score(f, o, c)
    same = AL.align(f, [f[10:90].copy()])[0][0]                                 # an exact piece aligns to where it came from
    assert np.array_equal(same, np.arange(10, 90))


def test_the_frame_relative_layout():
    from test_rel_w5_baselines import toy_tables
    tables, _ = toy_tables(n_tables=3, rows=5, seed=83)
    tb = slice(RD.SLICES[-1][1], RD.SLICES[-1][2])
    for t in range(3):
        tab = {k: tables[k][t] for k in ("tok", "n", "col", "kind")}
        tab["frame_len"], tab["frame_row"] = tables["n"][t, 0], np.zeros(A.N_BLOCKS, np.int8)
        fr = AL.frame_relative(tab)
        rows = [r for r in range(1, 5) if tab["n"][r, -1] > 0]
        assert fr == (rows[0] if rows else None)
        if fr is None:
            continue
        col, frame_row, frame_len = AL.frame_layout(tab)
        nf, n0 = int(tab["n"][fr, -1]), int(tab["n"][0, -1])
        assert frame_row[-1] == fr and frame_len[-1] == max(nf, n0) and (frame_row[:-1] == 0).all()
        assert np.array_equal(col[fr, tb][:nf], np.arange(nf)) and np.array_equal(col[0, tb][:n0], np.arange(n0))
        assert np.array_equal(col[:, :tb.start], tab["col"][:, :tb.start])
        other = dict(tab, tok=np.array(tab["tok"], copy=True), col=np.full_like(tab["col"], -1))
        other["tok"][0, tb] = 1 + (other["tok"][0, tb] % 64)                     # the examined row's content and every position
        assert all(np.array_equal(a, b) for a, b in zip(AL.frame_layout(other), (np.where(np.arange(A.ROW_LEN)[None, :] < tb.start, other["col"], col), frame_row, frame_len)))
    copy = {k: np.array(v, copy=True) for k, v in tab.items()}                    # a row that is the frame row's copy aligns one to one
    copy["tok"][2, tb], copy["n"][2, -1] = copy["tok"][fr, tb], copy["n"][fr, -1]
    if fr != 2:
        col, _, _ = AL.frame_layout(copy)
        assert np.array_equal(col[2, tb][:int(copy["n"][fr, -1])], np.arange(int(copy["n"][fr, -1])))


def test_strict_realignment_ignores_what_is_hidden():
    rng = np.random.default_rng(3)
    from test_rel_w5_baselines import toy_tables
    tables, _ = toy_tables(n_tables=4, rows=5, seed=81)
    for t in range(4):
        tab = {k: tables[k][t] for k in ("tok", "n", "col", "kind")}
        tab["frame_len"] = tables["n"][t, 0]
        for c in (cond.SCATTERED, cond.SUFFIX):
            m = cond.materialise({k: tables[k][t:t + 1] for k in ("tok", "n", "col", "kind")}, [t], c)
            vis, hide = m["visible"][0], m["target"][0]
            col = AL.strict(tab, vis)
            gone = np.array(tab["col"], copy=True)
            gone[1:] = -1                                                        # every stored position of the other rows erased
            other = dict(tab, tok=np.where(hide, rng.integers(1, 66, tab["tok"].shape), tab["tok"]).astype(np.uint8), col=gone)
            tb = slice(RD.SLICES[-1][1], RD.SLICES[-1][2])
            assert np.array_equal(AL.strict(other, vis)[:, tb], col[:, tb])           # the target block: aligned the same
            assert np.array_equal(col[:, :tb.start], tab["col"][:, :tb.start])        # blocks without hidden residues: stored
            assert np.array_equal(AL.strict(other, vis)[0], tab["col"][0])              # the frame row keeps its own columns
            g1 = RD.render(dict(tab, col=col), vis, hide)
            g2 = RD.render(dict(other, col=AL.strict(other, vis)), vis, hide)
            assert np.array_equal(g1[0][:, tb], g2[0][:, tb]) and np.array_equal(g1[2][:, tb], g2[2][:, tb])


def _tables_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w3off")
    src = str(tmp / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    W2S.make_source(src)
    run_dir, cfg = RF.rel_run(tmp, stages=("w2_v0",), run_id="20261122T000000-exp-rel-00000000-w2tables",
                              rel_w2_v0=dict(source=src, expect=W2S.SIZES, rows=16, train_per_group=6, heldout_per_group=2, shard_tables=24, processes=1))
    with pytest.MonkeyPatch.context() as mp:
        W2S.patch(mp, tmp)
        mp.setattr(common, "sealed_set", RT._sealed)
        with Stage(run_dir, "rel_w2_v0") as st:
            doc = W2S.w2_v0.build(st, cfg, run_dir)
    with open(os.path.join(run_dir, "record.json"), "w") as f:
        json.dump(dict(run_id=os.path.basename(run_dir), status="passed", pipeline="rel_tables",
                       stages=[dict(name="rel_w2_v0", status="passed", outputs=[dict(name="tableset", path=RT.MANIFEST,
                                                                                   sha256=W2S.sha256_file(os.path.join(run_dir, RT.MANIFEST), use_sidecar=False))])]), f)
    return dict(tmp=tmp, run_dir=run_dir, doc=doc)


def test_stage_renders_every_held_out_table_strictly(tmp_path_factory, monkeypatch):
    built = _tables_run(tmp_path_factory)
    src, doc, tmp = built["run_dir"], built["doc"], built["tmp"]
    block = dict(tables_from=src, tablesets=[dict(path=RT.MANIFEST, sha256=doc["sha256"])], shard_tables=10)
    run_dir, cfg = RF.rel_run(tmp / "w3", stages=["w3_offline"], rel_w3_offline=block)
    W2S.patch(monkeypatch, tmp)
    monkeypatch.setattr(common, "sealed_set", RT._sealed)
    with Stage(run_dir, w3_offline.STAGE) as st:
        out = w3_offline.build(st, cfg, run_dir)
    held = doc["parts"]["heldout"]["tables"]
    assert sorted(out["parts"]) == [cond.SCATTERED, cond.SUFFIX, cond.WHOLE_GENE] and out["kind"] == A.TABLE_REAL and out["layout"] == A.LAYOUT_LEAKFREE
    rep = json.load(open(os.path.join(run_dir, w3_offline.OUT, "report.json")))
    for c, r in rep["conditions"].items():
        assert r["tables"] == held and r["fourth_test_failures"] == 0 and r["dropped_targets"] == 0 and r["stored_anchors"] > 0
    sets = shards.parts(run_dir, out)
    old = shards.by_split(src, [doc])[shards.SPLIT_HELDOUT].load(0, held, ("tok", "n", "kind", "col"))
    for c in (cond.SCATTERED, cond.SUFFIX, cond.WHOLE_GENE):
        new = sets[c].load(0, held, ("tok", "n", "kind", "col"))
        assert all(np.array_equal(new[f], old[f]) for f in ("tok", "n", "kind")) and not np.array_equal(new["col"], old["col"])
        tb = slice(RD.SLICES[-1][1], RD.SLICES[-1][2])
        assert np.array_equal(new["col"][:, :, :tb.start], old["col"][:, :, :tb.start])   # only the target block is aligned again
    wg = sets[cond.WHOLE_GENE].load(0, held, ("n", "frame_row", "frame_len"))
    for t in range(held):                                                        # the frame relative: the first row with a stretch there
        rows = [r for r in range(1, wg["n"].shape[1]) if wg["n"][t, r, -1] > 0]
        assert wg["frame_row"][t, -1] == (rows[0] if rows else 0) and (wg["frame_row"][t, :-1] == 0).all()
    with Stage(run_dir, "gates_rel_w3_offline", requires=("pre",)) as st:
        w3_offline.gate(st, cfg, run_dir)
    assert json.load(open(os.path.join(run_dir, "gates", "G4_rel_w3_offline.json")))["status"] == "pass"
    assert w3_offline.summary(run_dir)["json"][cond.SCATTERED]["fourth_test_failures"] == 0
    rp = os.path.join(run_dir, w3_offline.OUT, "report.json")                    # a failed fourth test fails the gate
    r = json.load(open(rp))
    r["conditions"][cond.SUFFIX]["fourth_test_failures"] = 1
    json.dump(r, open(rp, "w"))
    with pytest.raises(GateFailure):
        with Stage(run_dir, "gates_rel_w3_offline", requires=("pre",)) as st:
            w3_offline.gate(st, cfg, run_dir)
    with pytest.raises(SystemExit):
        w3_offline.settings(dict(rel_w3_offline=dict(block, conditions=[cond.WHOLE_GENE + "+" + cond.REPAIRED])))


class FakeIndex:
    """The interface of package W2's index of the uncapped search, holding given alignments."""

    def __init__(self):
        self.store, self.t = {}, []

    def add(self, qg, tg, qi, ti, hsps):
        self.store.setdefault((qg, tg, qi), []).append(len(self.t))
        self.t.append((ti, hsps))

    def pairs(self, qg, tg, qi):
        return list(self.store.get((qg, tg, qi), []))

    def tgene(self, p):
        return self.t[p][0]

    def alignments(self, p):
        return self.t[p][1]


def fake_columns(cigar, qs, ss):
    """The CIGAR reading of the index (M both, I query, D target), 0-based."""
    import re
    q, t, oq, ot = int(qs) - 1, int(ss) - 1, [], []
    for k, op in re.findall(rb"(\d+)([MID])", bytes(cigar)):
        k = int(k)
        if op == b"M":
            oq.append(np.arange(q, q + k))
            ot.append(np.arange(t, t + k))
            q, t = q + k, t + k
        elif op == b"I":
            q += k
        else:
            t += k
    return (np.concatenate(oq), np.concatenate(ot)) if oq else (np.zeros(0, int), np.zeros(0, int))


def to_hsp(col, qf, tf):
    """One alignment (qs, qe, ss, se, cigar) of the pairs col[j] = frame position (-1 none), shifted by the stretches'
    first codons."""
    pairs = [(int(c), j) for j, c in enumerate(col) if c >= 0]
    ops, (pq, pt) = [], pairs[0]
    run = 1
    for q, t in pairs[1:]:
        dq, dt = q - pq - 1, t - pt - 1
        if dq or dt:
            ops.append(b"%dM" % run)
            run = 0
            if dq:
                ops.append(b"%dI" % dq)
            if dt:
                ops.append(b"%dD" % dt)
        run += 1
        pq, pt = q, t
    ops.append(b"%dM" % run)
    return (pairs[0][0] + qf + 1, pq + qf + 1, pairs[0][1] + tf + 1, pt + tf + 1, b"".join(ops))


def test_online_frame_layout_from_the_search_index_is_the_offline_one():
    from test_rel_w5_baselines import toy_tables
    tables, _ = toy_tables(n_tables=4, rows=5, seed=85)
    rng = np.random.default_rng(2)
    for t in range(4):
        tab = {k: tables[k][t] for k in ("tok", "n", "col", "kind")}
        tab["frame_len"], tab["frame_row"] = tables["n"][t, 0], np.zeros(A.N_BLOCKS, np.int8)
        R = tab["tok"].shape[0]
        meta = dict(genome=np.arange(R) + 100, row_gene=rng.integers(0, 50, (R, A.N_BLOCKS)), row_first=rng.integers(0, 9, (R, A.N_BLOCKS)))
        off_col, off_fr, off_len = AL.frame_layout(tab)
        fr = AL.frame_relative(tab)
        if fr is None:
            continue
        ix = FakeIndex()
        s = RD.SLICES[-1][1]
        for r in range(1, R):
            m = int(tab["n"][r, -1])
            c = off_col[r, s:s + m]
            if r == fr or m == 0 or (c < 0).all():
                continue
            ix.add(int(meta["genome"][fr]), int(meta["genome"][r]), int(meta["row_gene"][fr, -1]), int(meta["row_gene"][r, -1]),
                   [to_hsp(c, int(meta["row_first"][fr, -1]), int(meta["row_first"][r, -1]))])
            ix.add(int(meta["genome"][fr]), int(meta["genome"][r]), int(meta["row_gene"][fr, -1]), int(meta["row_gene"][r, -1]) + 1000,
                   [to_hsp(np.arange(m) % 3, 0, 0)])                       # a paralogue of the same genome: another gene, ignored
        col, frame_row, frame_len, placed = AL.frame_layout_online(tab, meta, ix, columns=fake_columns)
        assert np.array_equal(col, off_col) and np.array_equal(frame_row, off_fr) and np.array_equal(frame_len, off_len)
        assert set(placed) == {r for r in range(1, R) if r != fr and tab["n"][r, -1] > 0}
        miss = dict(meta, row_gene=np.full_like(meta["row_gene"], 7))           # no stored pair: the row is not shown
        col2, _, _, placed2 = AL.frame_layout_online(tab, miss, ix, columns=fake_columns)
        assert all(v == 0 for v in placed2.values()) and all((col2[r, s:s + int(tab["n"][r, -1])] == -1).all() for r in placed2)
