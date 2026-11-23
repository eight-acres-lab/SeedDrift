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


def test_online_frame_layout_reads_the_other_direction_when_the_target_is_not_in_the_library():
    """The search's targets are the training genomes only (every genome is a query, and each genome was also searched
    against itself). A self-copy row of a held-out row 0 laid on a training frame relative is found only as the query:
    the other direction, its sides swapped. Two rows of one non-training genome: the genome against itself. Two rows of
    two different non-training genomes: no pair, the row is not shown."""
    from test_rel_w5_baselines import toy_tables
    TRAIN, HELD, OTHER = set(range(100, 120)), 900, 901
    tables, _ = toy_tables(n_tables=8, rows=6, seed=91)
    rng = np.random.default_rng(5)
    used = {"forward": 0, "reverse": 0, "self": 0, "none": 0}
    s = RD.SLICES[-1][1]
    for t in range(8):
        tab = {k: tables[k][t] for k in ("tok", "n", "col", "kind")}
        tab["frame_len"], tab["frame_row"] = tables["n"][t, 0], np.zeros(A.N_BLOCKS, np.int8)
        R = tab["tok"].shape[0]
        fr = AL.frame_relative(tab)
        if fr is None:
            continue
        genome = np.array([HELD] + [int(g) for g in rng.choice([HELD, OTHER, 100, 101, 102], R - 1)])
        genome[fr] = HELD if t % 2 == 0 else 100 + t                             # a self-copy frame, or a training one
        meta =dict(genome=genome, row_gene=rng.integers(0, 50, (R, A.N_BLOCKS)), row_first=rng.integers(0, 9, (R, A.N_BLOCKS)))
        gene, first = meta["row_gene"][:, -1], meta["row_first"][:, -1]
        off_col, off_fr, off_len = AL.frame_layout(tab)
        nf = int(tab["n"][fr, -1])
        ix, want = FakeIndex(), off_col.copy()
        for r in range(1, R):
            m = int(tab["n"][r, -1])
            c = off_col[r, s:s + m].astype(np.int64)
            if r == fr or m == 0 or (c < 0).all():
                continue
            F, G = int(genome[fr]), int(genome[r])
            if G in TRAIN or F == G:                                             # stored with the frame relative as the query
                ix.add(F, G, int(gene[fr]), int(gene[r]), [to_hsp(c, int(first[fr]), int(first[r]))])
                used["self" if F == G and G not in TRAIN else "forward"] += 1
            if F in TRAIN or F == G:                                             # stored with the row as the query
                inv = np.full(nf, -1, np.int64)
                inv[c[c >= 0]] = np.flatnonzero(c >= 0)
                ix.add(G, F, int(gene[r]), int(gene[fr]), [to_hsp(inv, int(first[r]), int(first[fr]))])
                if G not in TRAIN and F != G:
                    used["reverse"] += 1
            if G not in TRAIN and F not in TRAIN and F != G:                     # no direction: the row is not shown
                want[r, s:s + m] = -1
                used["none"] += 1
        col, frame_row, frame_len, placed = AL.frame_layout_online(tab, meta, ix, columns=fake_columns)
        assert np.array_equal(col, want) and np.array_equal(frame_row, off_fr) and np.array_equal(frame_len, off_len)
    assert all(v > 0 for v in used.values()), used


V1_MANIFEST = os.path.join("data", "rel_w12", "v1", "tableset.json")


def _v1_like(built, tmp, name, kind=A.TABLE_REAL, purpose=None, extra=False, part="heldout"):
    """A tables run of the shape of version 1 made from the fixture's held-out tables: 32 rows (the fixture's rows and
    copies of them), rows 1 and 2 self-copies, ABSENT and TRUNCATED blocks, the fields the contract adds, and for a
    ruler set a meta file of its own and an index, as W18's implant set has (this stage refuses such a set)."""
    from ci.hashing import sha256_file
    from seeddrift.train.rel import handoff
    from seeddrift.train.rel.tables import v0_io as IO
    src, doc = built["run_dir"], built["doc"]
    held = shards.by_split(src, [doc])[shards.SPLIT_HELDOUT]
    a = held.load(0, held.tables, list(held.fields))
    N, r0 = a["tok"].shape[:2]
    R = 32
    out = IO.alloc(N, R, version=1)
    spec = IO.fields(r0, 0)
    for f, x in a.items():
        if spec[f][1][:1] == (r0,):
            out[f][:, :r0] = x
            out[f][:, r0:2 * r0 - 1] = x[:, 1:]                                 # more rows: copies of rows 1..
        else:
            out[f][:] = x
    rng = np.random.default_rng(5)
    exists = (out["kind"] != A.KIND_NONE).any(axis=2)
    out["genome_of_row"][:] = np.where(exists, np.arange(R)[None, :] - 2, -1)
    out["genome_of_row"][:, :3] = np.where(exists[:, :3], 0, -1)
    for r in (1, 2):
        out["kind"][:, r] = np.where(out["kind"][:, r] == A.KIND_PRESENT, A.KIND_SELF_COPY, out["kind"][:, r])
    has = out["n"] > 0
    out["row_gene"][:] = np.where(has, rng.integers(0, 500, has.shape), -1)
    out["row_first"][:] = np.where(has, 0, -1)
    for r, b, k in ((3, 0, A.KIND_ABSENT), (4, 2, A.KIND_ABSENT), (5, 5, A.KIND_TRUNCATED), (6, A.N_BLOCKS - 1, A.KIND_ABSENT)):
        s0, e0 = RD.SLICES[b][1], RD.SLICES[b][2]
        live = out["kind"][:, r, b] != A.KIND_NONE
        out["n"][live, r, b], out["kind"][live, r, b] = 0, k
        out["tok"][live, r, s0:e0], out["col"][live, r, s0:e0] = A.PAD, -1
        out["row_gene"][live, r, b], out["row_first"][live, r, b] = -1, -1
    if part == "train":
        out["split"][:] = shards.SPLIT_TRAIN                                     # the same tables as a training part
    run_dir = str(tmp / "ci-root" / "runs" / ("20261123T000000-exp-rel-00000000-" + name))
    folder = os.path.join(run_dir, "data", "rel_w12", "v1", part)
    written = IO.write(folder, out, 10, version=1)
    files = [dict(path=os.path.relpath(IO.shard_path(folder, sh, f), run_dir), sha256=sha256_file(IO.shard_path(folder, sh, f), use_sidecar=False),
                  bytes=os.path.getsize(IO.shard_path(folder, sh, f)), role="shard" if f in shards.SHARD_KNOWN else "meta")
             for sh, _ in written for f in out]
    if extra:
        for fname, role, write in (("w18_meta.npz", "meta", lambda p: np.savez(p, case=np.arange(N, dtype=np.int32))),
                                   ("index.json", "index", lambda p: json.dump(dict(triples=N // 3), open(p, "w")))):
            p = os.path.join(run_dir, "data", "rel_w12", "v1", fname)
            write(p)
            files.append(dict(path=os.path.relpath(p, run_dir), sha256=sha256_file(p, use_sidecar=False), bytes=os.path.getsize(p), role=role))
    body = {k: v for k, v in doc.items() if k not in ("sha256", "parts", "files", "tables", "name", "kind", "purpose")}
    new = handoff.finish(dict(body, name=name, kind=kind, rows=R, tables=N, files=files, table_version="w12-v1-1",
                              parts={part: dict(split=shards.SPLIT_HELDOUT if part == "heldout" else shards.SPLIT_TRAIN, tables=N, folder=os.path.relpath(folder, run_dir),
                                                      shards=[dict(name=sh, tables=m) for sh, m in written])},
                              **({"purpose": purpose} if purpose else {})))
    mp = os.path.join(run_dir, V1_MANIFEST)
    with open(mp, "w") as f:
        json.dump(new, f)
    with open(os.path.join(run_dir, "record.json"), "w") as f:
        json.dump(dict(run_id=os.path.basename(run_dir), status="passed", pipeline="rel_tables",
                       stages=[dict(name="rel_w12_v1", status="passed", outputs=[dict(name="tableset", path=V1_MANIFEST, sha256=sha256_file(mp, use_sidecar=False))])]), f)
    return run_dir, new, out


def test_stage_renders_tables_of_version_1_and_refuses_ruler_sets(tmp_path_factory, monkeypatch):
    built = _tables_run(tmp_path_factory)
    tmp = built["tmp"]
    W2S.patch(monkeypatch, tmp)
    monkeypatch.setattr(common, "sealed_set", RT._sealed)
    src, doc, arrays = _v1_like(built, tmp, "w12-v1like-r32")
    outs = {}
    for processes in (1, 2):
        block = dict(tables_from=src, tablesets=[dict(path=V1_MANIFEST, sha256=doc["sha256"])], shard_tables=7, processes=processes)
        run_dir, cfg = RF.rel_run(tmp / ("w3v1-%d" % processes), stages=["w3_offline"], rel_w3_offline=block)
        with Stage(run_dir, w3_offline.STAGE) as st:
            outs[processes] = (w3_offline.build(st, cfg, run_dir), run_dir, cfg)
    out, run_dir, cfg = outs[1]
    assert out["name"] == "w3-strict-w12-v1like-r32-s%d" % cond.SEED and out["kind"] == A.TABLE_REAL and out["rows"] == 32
    assert out["source_table_version"] == 1 and "purpose" not in out and out["source_files"] == []
    rep = json.load(open(os.path.join(run_dir, w3_offline.OUT, "report.json")))
    rep2 = json.load(open(os.path.join(outs[2][1], w3_offline.OUT, "report.json")))
    N = arrays["tok"].shape[0]
    for c in (cond.SCATTERED, cond.SUFFIX, cond.WHOLE_GENE):
        r = rep["conditions"][c]
        assert r["tables"] == N and r["fourth_test_failures"] == 0 and r["dropped_targets"] == 0, (c, r)
        assert {k: v for k, v in r.items() if k != "seconds"} == {k: v for k, v in rep2["conditions"][c].items() if k != "seconds"}, c
        assert out["parts"][c]["items_sha256"] == outs[2][0]["parts"][c]["items_sha256"]
    sets, sets2 = shards.parts(run_dir, out), shards.parts(outs[2][1], outs[2][0])
    for c in (cond.SCATTERED, cond.SUFFIX, cond.WHOLE_GENE):
        new, new2 = sets[c].load(0, N, sets[c].fields), sets2[c].load(0, N, sets2[c].fields)
        assert sorted(new) == sorted(arrays) and all(np.array_equal(new[f], new2[f], equal_nan=True) for f in new), c   # processes: the same
        for f in ("tok", "n", "kind", "genome_of_row", "row_gene", "row_first", "identity"):
            assert np.array_equal(new[f], arrays[f], equal_nan=True), (c, f)
        for t in range(N):                                                       # absent blocks stay absent in the grid
            tab = w3_offline.one(new, t)
            m = cond.materialise({f: new[f][t:t + 1] for f in ("tok", "n", "col", "kind")}, [t], c)
            grid = RD.render(tab, m["visible"][0], m["target"][0])[0]
            for r, b in ((3, 0), (4, 2), (6, A.N_BLOCKS - 1)):
                f0 = int(tab["frame_len"][b])
                if tab["kind"][r, b] == A.KIND_ABSENT and f0:
                    assert (grid[r, RD.SLICES[b][1]:RD.SLICES[b][1] + f0] == A.ABSENT).all()
    wg = sets[cond.WHOLE_GENE].load(0, N, ("n", "kind", "frame_row"))
    seen_self = 0
    for t in range(N):                                                           # the frame relative: the first row with a stretch, a self-copy too
        rows = [r for r in range(1, 32) if wg["n"][t, r, -1] > 0]
        assert wg["frame_row"][t, -1] == (rows[0] if rows else 0)
        seen_self += bool(rows) and wg["kind"][t, rows[0], -1] == A.KIND_SELF_COPY
    assert seen_self > 0
    with Stage(run_dir, "gates_rel_w3_offline", requires=("pre",)) as st:
        w3_offline.gate(st, cfg, run_dir)
    assert json.load(open(os.path.join(run_dir, "gates", "G4_rel_w3_offline.json")))["status"] == "pass"
    rsrc, rdoc, _ = _v1_like(built, tmp, "w18-implant-v1like-r32", kind=A.TABLE_RULER, purpose="ruler", extra=True)
    block = dict(tables_from=rsrc, tablesets=[dict(path=V1_MANIFEST, sha256=rdoc["sha256"])], shard_tables=7, conditions=[cond.WHOLE_GENE])
    rrun, rcfg = RF.rel_run(tmp / "w3ruler", stages=["w3_offline"], rel_w3_offline=block)
    with pytest.raises(SystemExit) as err:                                       # a ruler set is W18's stage's, refused here
        with Stage(rrun, w3_offline.STAGE) as st:
            w3_offline.build(st, rcfg, rrun)
    assert "rel_w18_offline" in str(err.value)
    for kinds in ([A.TABLE_RULER], [A.TABLE_REAL, A.TABLE_RULER]):              # and the profile cannot ask for one
        with pytest.raises(SystemExit) as err:
            w3_offline.settings(dict(rel_w3_offline=dict(block, allow_kinds=kinds)))
        assert "rel_w18_offline" in str(err.value)
    assert w3_offline.settings(dict(rel_w3_offline=dict(block, allow_kinds=[A.TABLE_SYNTHETIC])))["allow_kinds"] == [A.TABLE_SYNTHETIC]


def test_the_training_check_stage(tmp_path_factory, monkeypatch):
    """Stage rel_w3_traincheck on a training part of the shape of version 1: every mask scheme checked on every table
    drawn, no failure, the gate passes; a failure fails it."""
    from ci.stages.rel import w3_traincheck as TC
    built = _tables_run(tmp_path_factory)
    tmp = built["tmp"]
    W2S.patch(monkeypatch, tmp)
    monkeypatch.setattr(common, "sealed_set", RT._sealed)
    src, doc, arrays = _v1_like(built, tmp, "w12-v1like-train", part="train")
    block = dict(tables_from=src, tablesets=[dict(path=V1_MANIFEST, sha256=doc["sha256"])], tables=12, processes=2, rows=[4, 32])
    run_dir, cfg = RF.rel_run(tmp / "tc", stages=["w3_traincheck"], rel_w3_traincheck=block)
    with Stage(run_dir, TC.STAGE) as st:
        TC.build(st, cfg, run_dir)
    rep = json.load(open(os.path.join(run_dir, TC.OUT, "report.json")))
    assert rep["tables"] == 12 and set(rep["masks"]) == {"scattered", "span", "whole_gene"}
    assert all(r["samples"] == 12 and r["failures"] == 0 and r["cells"] > 0 and r["ms_per_rendering"] > 0 for r in rep["masks"].values())
    with Stage(run_dir, "gates_rel_w3_traincheck", requires=("pre",)) as st:
        TC.gate(st, cfg, run_dir)
    assert json.load(open(os.path.join(run_dir, "gates", "G4_rel_w3_traincheck.json")))["status"] == "pass"
    keep = TC.rows_of({k: arrays[k][0] for k in ("tok", "n", "col", "kind", "frame_len", "frame_row")}, 0, [4, 32], 1)
    assert keep[0] == 0 and 1 <= len(keep) <= 32 and len(set(keep.tolist())) == len(keep)
    assert np.array_equal(TC.drawn(100, 12, 5), TC.drawn(100, 12, 5)) and len(TC.drawn(100, 12, 5)) == 12
    rep["masks"]["span"]["failures"] = 1
    json.dump(rep, open(os.path.join(run_dir, TC.OUT, "report.json"), "w"))
    with pytest.raises(GateFailure):
        with Stage(run_dir, "gates_rel_w3_traincheck", requires=("pre",)) as st:
            TC.gate(st, cfg, run_dir)


def test_the_strict_rendering_online_is_the_offline_part_cell_for_cell(tmp_path_factory, monkeypatch):
    """Under the evaluator's plans (only row 0 hidden) the strict rendering of the stored tables (tables/strict.py, as
    the training and its nodes render) is the leak-free rendering of the offline part (W3's strict offline rendering,
    as the final evaluation reads it), grid, truth and cells laid by extension, table for table and condition for
    condition: aligner version 2 is version 1 there, so the two readings are one layout."""
    from seeddrift.train.rel.tables import strict as ST
    built = _tables_run(tmp_path_factory)
    src, doc, tmp = built["run_dir"], built["doc"], built["tmp"]
    block = dict(tables_from=src, tablesets=[dict(path=RT.MANIFEST, sha256=doc["sha256"])], shard_tables=10)
    run_dir, cfg = RF.rel_run(tmp / "same", stages=["w3_offline"], rel_w3_offline=block)
    W2S.patch(monkeypatch, tmp)
    monkeypatch.setattr(common, "sealed_set", RT._sealed)
    with Stage(run_dir, w3_offline.STAGE) as st:
        out = w3_offline.build(st, cfg, run_dir)
    sets = shards.parts(run_dir, out)
    held = shards.by_split(src, [doc])[shards.SPLIT_HELDOUT]
    fields = ("tok", "n", "col", "kind", "frame_len", "frame_row")
    stored = held.load(0, held.tables, fields)
    for c in (cond.SCATTERED, cond.SUFFIX, cond.WHOLE_GENE):
        part = sets[c].load(0, held.tables, fields)
        plan = cond.materialise({f: stored[f] for f in ("tok", "n", "col", "kind")}, np.arange(held.tables), c)
        for t in range(held.tables):
            vis, tgt = plan["visible"][t], plan["target"][t]
            online = ST.render({f: stored[f][t] for f in fields}, vis, tgt, whole_gene=c == cond.WHOLE_GENE)[:3]
            offline = RD.render({f: part[f][t] for f in fields}, vis, tgt, RD.LEAKFREE)
            assert all(np.array_equal(a, b) for a, b in zip(online, offline)), (c, t)
