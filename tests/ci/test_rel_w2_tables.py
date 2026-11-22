"""Tables of version 0 (seeddrift/train/rel/tables/v0_build.py, v0_census.py) on a small synthetic run.

The fixture evolves every genome's genes from common ancestors with substitutions, insertions and deletions, writes
them in the layout of stages train_data and pairs, and keeps the true correspondence of every homolog row. The tests
compare the tables with that truth, not with the code that reads the alignments. Numpy only (python
tests/ci/test_rel_w2_tables.py runs them without pytest).
"""
import atexit
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from seeddrift.train import selfsup as SS  # noqa: E402
from seeddrift.train import source_model as S  # noqa: E402
from seeddrift.train.rel import api as A  # noqa: E402
from seeddrift.train.rel.tables import v0_build as V  # noqa: E402
from seeddrift.train.rel.tables import v0_census as C  # noqa: E402
from seeddrift.train.rel.tables import v0_io as IO  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
GROUPS = ["archaea", "bacteria", "viruses.phage"]
#          id, group, split, genes kept
GENOMES = ([("b%d" % k, "bacteria", "train") for k in range(6)] + [("bh%d" % k, "bacteria", "heldout") for k in range(2)] + [("bt0", "bacteria", "test")]
           + [("a%d" % k, "archaea", "train") for k in range(3)] + [("ah0", "archaea", "heldout"), ("at0", "archaea", "test")]
           + [("v%d" % k, "viruses.phage", "train") for k in range(3)] + [("vh0", "viruses.phage", "heldout")])
KEEP = 6                                                             # homolog rows kept per gene (the cap of stage pairs, small)


def _evolve(rng, anc):
    """A descendant of an ancestral gene: (codons, ancestral position of each codon or -1 for an inserted one)."""
    pos = np.arange(len(anc))
    if rng.random() < 0.5 and len(anc) > 12:                         # a deletion
        a = int(rng.integers(0, len(anc) - 6))
        pos = np.concatenate([pos[:a], pos[a + int(rng.integers(1, 6)):]])
    cod = anc[pos].copy()
    sub = rng.random(len(cod)) < rng.choice([0.02, 0.2, 0.5])
    cod[sub] = rng.integers(1, 65, size=int(sub.sum()))
    for _ in range(int(rng.integers(0, 3))):                         # insertions
        a, n = int(rng.integers(0, len(cod) + 1)), int(rng.integers(1, 7))
        cod = np.concatenate([cod[:a], rng.integers(1, 65, size=n), cod[a:]])
        pos = np.concatenate([pos[:a], np.full(n, -1), pos[a:]])
    return cod.astype(np.uint8), pos


def _align(pa, pb):
    """Alignment columns of two descendants from their ancestral positions: a list of (op, x, y)."""
    cols, i, j = [], 0, 0
    while i < len(pa) or j < len(pb):
        if i < len(pa) and pa[i] < 0:
            cols.append(("I", i, -1))
            i += 1
        elif j < len(pb) and pb[j] < 0:
            cols.append(("D", -1, j))
            j += 1
        elif i < len(pa) and j < len(pb) and pa[i] == pb[j]:
            cols.append(("M", i, j))
            i, j = i + 1, j + 1
        elif j >= len(pb) or (i < len(pa) and pa[i] < pb[j]):
            cols.append(("I", i, -1))
            i += 1
        else:
            cols.append(("D", -1, j))
            j += 1
    return cols


def make_run(root, seed=0, shuffle_rows=False, empty_pairs=False, genomes=None, lengths=None, taxids=None):
    """A run directory with data/train_data and data/pairs; returns (metas, truth) with truth[(A, gene, P)] = (gene of P,
    {codon of P: codon of A} over the matched columns of the stored alignment). genomes: [(id, group, split)] (default
    GENOMES); lengths: codons of the ancestral genes; taxids: {group: taxid}."""
    rng, other = np.random.default_rng(seed), np.random.default_rng(seed + 1000)    # `other`: draws that leave the data as it is
    os.makedirs(os.path.join(root, "data", "train_data"), exist_ok=True)
    os.makedirs(os.path.join(root, "data", "pairs"), exist_ok=True)
    lengths = lengths or [int(x) for x in rng.integers(14, 60, size=34)] + [130, 150, 190, 262, 280, 300]
    ancestors = [rng.integers(1, 65, size=n) for n in lengths]
    genes, metas = {}, []
    for k, (gid, group, split) in enumerate(genomes or GENOMES):
        share = 0.25 if group.startswith("viruses") else 0.85
        have = [a for a in range(len(ancestors)) if rng.random() < share] or [0]
        rng.shuffle(have)
        gl = [(a,) + _evolve(rng, ancestors[a]) for a in have]
        genes[gid] = gl
        n = len(gl)
        rep = np.sort(rng.integers(0, 2, size=n)).astype(np.int32) if n > 3 else np.zeros(n, np.int32)
        start, end, at = np.zeros(n, np.int64), np.zeros(n, np.int64), {0: 1, 1: 1}
        for i, (_, cod, _) in enumerate(gl):
            start[i] = at[int(rep[i])] + int(rng.integers(0, 50))
            end[i] = start[i] + 3 * len(cod) - 1
            at[int(rep[i])] = end[i] + 1
        perm = rng.permutation(n)                                    # the order of the gene table is not the genome order
        gl = [gl[int(x)] for x in perm]
        genes[gid] = gl
        rep, start, end = rep[perm], start[perm], end[perm]
        coff = np.concatenate([[0], np.cumsum([len(c) for _, c, _ in gl])]).astype(np.int64)
        np.savez(os.path.join(root, "data", "train_data", gid + ".npz"), codons=np.concatenate([c for _, c, _ in gl]), codon_off=coff, replicon=rep,
                 start=start, end=end, strand=rng.choice([-1, 1], size=n).astype(np.int8), replicon_len=np.array([int(end.max()) + 10] * (int(rep.max()) + 1), np.int64),
                 hit_off=np.zeros(n + 1, np.int64), hit_prot=np.zeros(0, np.int32), hit_qs=np.zeros(0, np.int32), track=np.zeros(0, np.uint8),
                 track_off=np.zeros(1, np.int64))
        metas.append(dict(id=gid, source="corpus", group=group, taxid=(taxids or {}).get(group, 1000 + k), ranks=[1, 2, 3, 4, 5 + k, 100 + k], genes=n, hits=0, split=split,
                          cluster="", relation=""))
    with open(os.path.join(root, "data", "train_data", "index.json"), "w") as f:
        json.dump(dict(dataset_id="ds-test", genomes=metas, library="data/reference/library.npz", max_codons=1000), f)
    order = [m["id"] for m in metas]
    with open(os.path.join(root, "data", "pairs", "genomes.json"), "w") as f:
        json.dump(dict(genomes=order), f)
    truth = {}
    for a, gid in enumerate(order):
        off, tg, tj, qs, ss, bits, cig, cig_off, sq, sq_off = [0], [], [], [], [], [], bytearray(), [0], bytearray(), [0]
        for i, (anc, cod, pos) in enumerate(genes[gid]):
            rows = []
            for b, pid in enumerate(order):
                if pid == gid or empty_pairs:
                    continue
                for j, (anc2, cod2, pos2) in enumerate(genes[pid]):
                    if anc2 != anc:
                        continue
                    cols = _align(pos, pos2)
                    ms = [c for c, (op, _, _) in enumerate(cols) if op == "M"]
                    if not ms:
                        continue
                    lo = ms[int(rng.integers(0, max(1, len(ms) // 2)))] if rng.random() < 0.3 else ms[0]   # some alignments start late
                    cols = cols[lo:ms[-1] + 1]
                    run, ops = [], [c[0] for c in cols]
                    for op in ops:
                        if run and run[-1][1] == op:
                            run[-1][0] += 1
                        else:
                            run.append([1, op])
                    pairs_ = {y: x for op, x, y in cols if op == "M"}
                    score = sum(1 for y, x in pairs_.items() if cod2[y] == cod[x]) + 0.001 * b
                    rows.append((score, b, j, cols[0][1] + 1, cols[0][2] + 1, "".join("%d%s" % (n_, op) for n_, op in run),
                                 "".join(AA[S.CODE[int(cod2[y])] - 1] if S.CODE[int(cod2[y])] <= 20 else "*" for op, _, y in cols if op != "I"), pairs_))
            rows.sort(key=lambda r: -r[0])
            rows = rows[:KEEP]                                       # the cap: nearest genomes only
            if shuffle_rows:
                rows = [rows[int(x)] for x in other.permutation(len(rows))]
            for score, b, j, q0, s0, cigar, sseq, pairs_ in rows:
                truth[(gid, i, order[b])] = (j, pairs_)
                tg.append(b)
                tj.append(j)
                qs.append(q0)
                ss.append(s0)
                bits.append(float(other.random()) if shuffle_rows else score)
                cig += cigar.encode()
                cig_off.append(len(cig))
                sq += sseq.encode()
                sq_off.append(len(sq))
            off.append(len(tg))
        np.savez(os.path.join(root, "data", "pairs", gid + ".npz"), off=np.array(off, np.int32), tgenome=np.array(tg, np.int32), tgene=np.array(tj, np.int32),
                 qs=np.array(qs, np.uint32), ss=np.array(ss, np.uint32), bits=np.array(bits, np.float32), cigar=np.frombuffer(bytes(cig), np.uint8),
                 cigar_off=np.array(cig_off, np.int64), sseq=np.frombuffer(bytes(sq), np.uint8), sseq_off=np.array(sq_off, np.int64))
    return metas, truth


_CACHE = {}


def run():
    """The fixture run, made once: (folder, metas, truth, {id: Genome} of the open splits, ids, training ids)."""
    if "run" not in _CACHE:
        d = tempfile.mkdtemp(prefix="rel-w2-")
        atexit.register(shutil.rmtree, d, ignore_errors=True)
        metas, truth = make_run(d)
        opened = [m for m in metas if m["split"] in V.OPEN_SPLITS]
        genomes, stats = V.load_genomes(os.path.join(d, "data", "train_data"), opened)
        _CACHE["run"] = (d, metas, truth, genomes, [m["id"] for m in opened], [m["id"] for m in metas if m["split"] == "train"], stats)
    return _CACHE["run"]


def build(split="train", per_group=20, rows=16, seed=3, folder=None, processes=1, shard_tables=24):
    d, metas, truth, genomes, ids, train, _ = run()
    out = tempfile.mkdtemp(prefix="rel-w2-set-")
    atexit.register(shutil.rmtree, out, ignore_errors=True)
    plan = V.plan_set(metas, GROUPS, split, per_group, seed)
    shards, census = V.build_set(SS.Pairs(folder or d), genomes, train, ids, GROUPS, split, plan, out, rows=rows, shard_tables=shard_tables, seed=seed,
                                 processes=processes)
    return out, plan, shards, census


def load(out, shards, rows=16):
    return {f: np.concatenate([np.load(os.path.join(out, "%s.%s.npy" % (name, f))) for name, _ in shards]) for f in V.fields(rows)}


def tables():
    if "tables" not in _CACHE:
        out, plan, shards, census = build()
        _CACHE["tables"] = (out, plan, shards, census, load(out, shards))
    return _CACHE["tables"]


def test_fields_follow_the_contract():
    out, plan, shards, census, z = tables()
    n = len(plan[0])
    assert n == 60 and [m for _, m in shards] == [24, 24, 12]
    dims = dict(N=n, R=16, ROW_LEN=A.ROW_LEN, B=A.N_BLOCKS)
    for name, (dtype, shape, _) in list(A.SHARD_FIELDS.items()) + list(A.META_FIELDS.items()):
        assert z[name].dtype == np.dtype(dtype) and z[name].shape == tuple(dims[s] for s in shape.split()), name
    assert sorted(os.listdir(out)) == sorted("%s.%s.npy" % (s, f) for s, _ in shards for f in V.fields(16))
    assert (z["frame_row"] == 0).all() and (z["tier"] == -1).all() and (z["split"] == 0).all()
    assert (np.bincount(z["group"], minlength=3) == 20).all()        # the groups hold equally many tables
    assert set(np.unique(z["kind"])) <= {0, A.KIND_PRESENT, A.KIND_NOT_SEARCHED, A.KIND_TRUNCATED}
    assert {A.KIND_PRESENT, A.KIND_NOT_SEARCHED, A.KIND_TRUNCATED} <= set(np.unique(z["kind"][:, 1:]))   # the fixture shows every kind
    assert z["tok"].max() <= A.TARGET_MAX


def test_examined_row_is_the_genome_itself():
    d, metas, truth, genomes, ids, train, _ = run()
    out, plan, shards, census, z = tables()
    short = 0
    for t, gid in enumerate(plan[0]):
        gen = genomes[gid]
        assert ids[z["genome"][t, 0]] == gid and gen.meta["split"] == "train" and GROUPS[z["group"][t]] == gen.meta["group"]
        tgt = int(z["gene"][t, V.TARGET])
        p = int(gen.pos[tgt])
        for b, x in ((V.LEFT, p - 1), (V.RIGHT, p + 1)):             # the neighbours: next on the replicon, in coordinate order
            j = int(gen.order[x]) if 0 <= x < gen.n and gen.rep[gen.order[x]] == gen.rep[tgt] else -1
            assert z["gene"][t, b] == j
            if j >= 0:
                assert (gen.start[j] < gen.start[tgt]) == (b == V.LEFT)
        used = [int(j) for j in z["gene"][t] if j >= 0]
        assert len(used) == len(set(used)) == min(gen.n, A.N_BLOCKS) - sum(1 for b in (V.LEFT, V.RIGHT) if z["gene"][t, b] < 0 and gen.n >= A.N_BLOCKS)
        short += gen.n < A.N_BLOCKS
        for b, (role, s0, s1) in enumerate(V.SLICES):
            j = int(z["gene"][t, b])
            if j < 0:
                assert z["kind"][t, 0, b] == 0 and z["n"][t, 0, b] == 0 and (z["tok"][t, :, s0:s1] == 0).all() and (z["kind"][t, :, b] == 0).all()
                continue
            c = gen.codons[gen.coff[j]:gen.coff[j + 1]][:s1 - s0]
            n = z["n"][t, 0, b]
            assert n == len(c) == z["frame_len"][t, b] and (z["tok"][t, 0, s0:s0 + n] == c).all() and (z["tok"][t, 0, s0 + n:s1] == A.PAD).all()
            assert (z["col"][t, 0, s0:s0 + n] == np.arange(n)).all() and (z["col"][t, 0, s0 + n:s1] == -1).all()
            assert z["strand"][t, 0, b] == gen.strand[j] and z["replicon"][t, 0, b] == gen.rep[j]
            assert z["start"][t, 0, b] == (gen.start[j] if gen.strand[j] > 0 else gen.end[j])
    assert short > 0                                                 # the small genomes (fewer than 19 genes) were covered
    assert (z["n"][:, 0, V.TARGET] == 256).any() and (z["n"][:, 0, V.LEFT] == 128).any()   # and blocks cut at their length


def test_other_rows_match_the_true_correspondence():
    d, metas, truth, genomes, ids, train, _ = run()
    out, plan, shards, census, z = tables()
    seen = {k: 0 for k in ("present", "not_searched", "truncated", "cut", "gap", "extra")}
    for t, gid in enumerate(plan[0]):
        rows = [q for q in range(1, 16) if z["genome"][t, q] >= 0]
        assert rows == list(range(1, len(rows) + 1))                 # packed at the front
        for q in range(1, 16):
            if q not in rows:
                assert (z["kind"][t, q] == 0).all() and (z["tok"][t, q] == 0).all() and (z["col"][t, q] == -1).all() and np.isnan(z["identity"][t, q])
                continue
            pid = ids[z["genome"][t, q]]
            part = genomes[pid]
            assert part.meta["split"] == "train" and pid != gid
            for b, (role, s0, s1) in enumerate(V.SLICES):
                i, flen, kind = int(z["gene"][t, b]), int(z["frame_len"][t, b]), int(z["kind"][t, q, b])
                if i < 0:
                    assert kind == 0
                    continue
                row = truth.get((gid, i, pid))
                under = {y: x for y, x in row[1].items() if x < flen and y < part.coff[row[0] + 1] - part.coff[row[0]]} if row else {}
                if row is None:
                    assert kind == A.KIND_NOT_SEARCHED and z["n"][t, q, b] == 0
                    seen["not_searched"] += 1
                    continue
                if not under:
                    assert kind == A.KIND_TRUNCATED and z["n"][t, q, b] == 0
                    seen["truncated"] += 1
                    continue
                assert kind == A.KIND_PRESENT
                seen["present"] += 1
                j, first, n = row[0], min(under), int(z["n"][t, q, b])
                last = min(max(under), first + (s1 - s0) - 1)        # cut at the block length
                seen["cut"] += max(under) > last
                want = part.codons[part.coff[j]:part.coff[j + 1]][first:last + 1]
                while len(want) and (first + len(want) - 1) not in under:   # the stretch ends at its last matched codon
                    want = want[:-1]
                assert n == len(want) and (z["tok"][t, q, s0:s0 + n] == want).all() and (z["tok"][t, q, s0 + n:s1] == A.PAD).all()
                col = z["col"][t, q, s0:s0 + n]
                assert [int(c) for c in col] == [under.get(first + c, -1) for c in range(n)] and (z["col"][t, q, s0 + n:s1] == -1).all()
                seen["extra"] += int((col < 0).sum())
                seen["gap"] += int(len(set(range(int(col[col >= 0].min()), int(col.max()) + 1)) - set(int(c) for c in col)))
                assert z["strand"][t, q, b] == part.strand[j] and z["replicon"][t, q, b] == part.rep[j]
                assert z["start"][t, q, b] == (part.start[j] + 3 * first if part.strand[j] > 0 else part.end[j] - 3 * first)
                m = col >= 0
                share = float((z["tok"][t, q, s0:s0 + n][m] == z["tok"][t, 0, s0:s1][col[m]]).mean())
                if b == V.TARGET:
                    assert abs(z["identity"][t, q] - share) < 1e-6
            assert z["kind"][t, q, V.TARGET] == A.KIND_PRESENT       # a row is in the table because of its target cell
    assert all(v > 0 for v in seen.values()), seen                   # every case occurred in the fixture
    assert (z["identity"][:, 0] == 1.0).all()


def test_no_sealed_genome_and_only_training_rows():
    d, metas, truth, genomes, ids, train, _ = run()
    split = {m["id"]: m["split"] for m in metas}
    assert all(split[g] != "test" for g in ids) and all(split[g] != "test" for g in genomes)
    for name in ("train", "heldout"):
        out, plan, shards, census = build(name, per_group=10)
        z = load(out, shards)
        assert {split[ids[g]] for g in z["genome"][:, 0]} == {name} and (z["split"] == V.SPLITS[name]).all()
        assert {split[ids[g]] for g in z["genome"][:, 1:].ravel() if g >= 0} == {"train"}
        assert not V.verify_set(out, shards, 16, name, {i for i, g in enumerate(ids) if split[g] == name}, {i for i, g in enumerate(ids) if split[g] == "train"}, 3)
    # the fixture's homolog rows do point at sealed genomes: they are dropped before any alignment is read
    order = SS.Pairs(d).order
    assert any(split[order[int(x)]] == "test" for g in train for x in np.load(os.path.join(d, "data", "pairs", g + ".npz"))["tgenome"])


def test_sealed_genomes_are_refused():
    d, metas, truth, genomes, ids, train, _ = run()
    sealed = [m for m in metas if m["split"] == "test"]
    for call in (lambda: V.load_genomes(os.path.join(d, "data", "train_data"), sealed[:1]),
                 lambda: V.Builder(SS.Pairs(d), dict(genomes, x=S.Genome(sealed[0], os.path.join(d, "data", "train_data", sealed[0]["id"] + ".npz"))), train, ids),
                 lambda: V.Builder(SS.Pairs(d), genomes, train + [m["id"] for m in metas if m["split"] == "heldout"][:1], ids)):
        try:
            call()
        except SystemExit:
            continue
        raise AssertionError("a sealed or held-out genome was accepted")


def test_verify_fails_on_a_wrong_row():
    d, metas, truth, genomes, ids, train, _ = run()
    split = {m["id"]: m["split"] for m in metas}
    out, plan, shards, census = build("train", per_group=10)
    args = (out, shards, 16, "train", {i for i, g in enumerate(ids) if split[g] == "train"}, {i for i, g in enumerate(ids) if split[g] == "train"}, 3)
    assert V.verify_set(*args) == []
    p = os.path.join(out, "s0000.genome.npy")
    g = np.load(p)
    t = int(np.flatnonzero(g[:, 1] >= 0)[0])
    g[t, 1] = [i for i, x in enumerate(ids) if split[x] == "heldout"][0]
    np.save(p, g)
    status = "fail" if V.verify_set(*args) else "pass"
    assert status == "fail"
    p = os.path.join(out, "s0000.tok.npy")
    tok = np.load(p)
    tok[0, 0, -1] = 7                                                # a codon after the end of a stretch
    np.save(p, tok)
    assert any("tokens and lengths" in x for x in V.verify_set(*args))


def test_same_tables_whatever_the_processes_and_the_row_order():
    d, metas, truth, genomes, ids, train, _ = run()
    out, plan, shards, census, z = tables()
    out2, plan2, shards2, census2 = build(processes=2)
    z2 = load(out2, shards2)
    assert plan2 == plan and all(np.array_equal(z[f], z2[f], equal_nan=(f == "identity")) for f in z)
    assert all(np.array_equal(census.a[k], census2.a[k]) for k in census.a)
    assert V.hash_shards(out, shards, 16) == V.hash_shards(out2, shards2, 16, processes=2)
    # the same data with the homolog rows of every gene in another order and other bit scores: the same tables
    with tempfile.TemporaryDirectory() as e:
        make_run(e, shuffle_rows=True)
        out3, plan3, shards3, census3 = build(folder=e)
        z3 = load(out3, shards3)
        assert all(np.array_equal(z[f], z3[f], equal_nan=(f == "identity")) for f in z)
    b = V.Builder(SS.Pairs(d), genomes, train, ids)
    assert "bits" not in b.state(plan[0][0])["z"]                     # dropped before anything is read


def test_loci_do_not_depend_on_other_genomes():
    out, plan, shards, census, z = tables()
    with tempfile.TemporaryDirectory() as e:
        make_run(e, empty_pairs=True)                                # the same genomes, no homolog row at all
        out2, plan2, shards2, census2 = build(folder=e)
        z2 = load(out2, shards2)
    assert plan2 == plan and np.array_equal(z["gene"], z2["gene"]) and np.array_equal(z["tok"][:, 0], z2["tok"][:, 0])
    assert (z2["genome"][:, 1:] == -1).all() and (z2["kind"][:, 1:] == 0).all()
    assert census2.a["relatives"][:, 0].sum() == len(plan[0])


def test_plan_is_balanced():
    d, metas, truth, genomes, ids, train, _ = run()
    genome_of, group_of, missing = V.plan_set(metas, GROUPS, "train", 12, 5)
    assert missing == [] and len(genome_of) == 36 and group_of[:6] == [0, 1, 2, 0, 1, 2]
    for g, want in (("b", 2), ("a", 4), ("v", 4)):                   # 6, 3 and 3 training genomes: each examined equally often
        counts = [sum(1 for x in genome_of if x == m["id"]) for m in metas if m["split"] == "train" and m["id"].startswith(g)]
        assert set(counts) == {want}
    only = [dict(m, split="train" if m["group"] != "archaea" else "test") for m in metas]
    genome_of, group_of, missing = V.plan_set(only, GROUPS, "train", 5, 5)
    assert missing == ["archaea"] and len(genome_of) == 10 and set(group_of) == {1, 2}


def test_census_counts_the_tables():
    d, metas, truth, genomes, ids, train, stats = run()
    out, plan, shards, census, z = tables()
    a = census.a
    has = z["genome"][:, 1:] >= 0
    assert a["tables"].tolist() == [20, 20, 20]
    assert a["relatives"].sum(0).tolist() == np.bincount(has.sum(1), minlength=16).tolist()
    role = V.ROLE_OF_BLOCK
    for r in range(4):
        k = z["kind"][:, 1:][:, :, role == r]
        exists = (z["kind"][:, :1][:, :, role == r] > 0) & has[:, :, None]
        assert a["cells"].sum(0)[r].tolist() == np.bincount(k[exists], minlength=6).tolist()
        assert a["blocks"].sum(0)[r] == (z["kind"][:, 0][:, role == r] > 0).sum()
        assert a["codons"].sum(0)[r].tolist() == [int(z["n"][:, 0][:, role == r].sum()), int(z["n"][:, 1:][:, :, role == r].sum())]
    assert a["nearest"].sum(0)[3].sum() == 60 and a["nearest"].sum(0)[3, C.N_BINS] == (~has.any(1)).sum()
    best = np.nanmax(np.where(has, z["identity"][:, 1:], -1.0), 1)
    assert a["nearest"].sum(0)[3, :C.N_BINS].tolist() == np.bincount(C.bin_of(best[best >= 0]), minlength=C.N_BINS).tolist()
    # rows drawn at random: per block, the training genomes with a homolog row starting inside it, over all the others
    want = np.zeros(4)
    for t, gid in enumerate(plan[0]):
        others = [g for g in train if g != gid]
        for b in range(A.N_BLOCKS):
            i, flen = int(z["gene"][t, b]), int(z["frame_len"][t, b])
            if i >= 0:
                want[role[b]] += sum(1 for g in others if (gid, i, g) in truth and min(truth[(gid, i, g)][1].values()) < flen) / len(others)
    assert np.allclose(a["random"].sum(0), want)
    assert a["misc"].sum(0)[C.MISC.index("checked")] == a["misc"].sum(0)[C.MISC.index("agree")] > 0      # the codons code the aligned residues
    assert a["misc"].sum(0)[C.MISC.index("cut_cells")] > 0 and a["misc"].sum(0)[C.MISC.index("duplicate_rows")] == 0
    doc = C.summarise(census, GROUPS)
    assert set(doc["per_group"]) == set(GROUPS) and doc["all"]["tables"] == 60
    f = doc["all"]["filled"]["not_target"]
    assert abs(f["share"] - f["with_codons"] / f["cells"]) < 1e-12 and f["share_of_row_slots"] <= f["share"] <= 1
    assert doc["all"]["filled"]["target"]["share"] == 1.0            # rows are chosen by their target cell
    assert doc["all"]["random_rows_filled"]["target"]["share"] < 1
    corpus = C.corpus_counts(stats, GROUPS)
    assert corpus["train"]["all"]["genomes"] == 12 and corpus["heldout"]["all"]["genomes"] == 4 and "test" not in corpus
    assert corpus["train"]["bacteria"]["codons"] == sum(len(genomes[g].codons) for g in train if g.startswith("b"))
    text = C.page(dict(note="n", source_run="r", coverage="c", max_codons=1000, groups=GROUPS, rows=16, corpus=corpus, caveats=["x"],
                       sets=dict(train=dict(tables=60, census=doc))))
    assert "bacteria" in text and "## Table set `train`" in text


def test_reader_and_writer_agree():
    out, plan, shards, census, z = tables()
    part = IO.Part(out, shards, 16)
    ids = np.array([59, 0, 24, 23, 47, 5, 5])                         # any order, across shards, with a repeat
    got = part.take(ids, IO.MODEL_FIELDS + IO.META_FIELDS)
    assert part.tables == 60 and all(np.array_equal(got[f], z[f][ids], equal_nan=(f == "identity")) for f in got)
    assert np.array_equal(part.field("kind"), z["kind"]) and set(part.shard(1)) == set(A.SHARD_FIELDS) and part.shard(2)["tok"].shape[0] == 12
    assert V.KIND_NONE == A.KIND_NONE == 0 and V.API_VERSION == A.API
    with tempfile.TemporaryDirectory() as e:
        sh = IO.write(e, z, 25)
        again = IO.Part(e, sh, 16)
        assert [m for _, m in sh] == [25, 25, 10] and all(np.array_equal(again.field(f), z[f], equal_nan=(f == "identity")) for f in z)
        for bad in ({k: v for k, v in z.items() if k != "col"}, dict(z, extra=z["n"]), dict(z, n=z["n"][:, :, :3])):
            try:
                IO.write(e, bad, 25)
            except ValueError:
                continue
            raise AssertionError("a part without the contract's fields was written")
    try:
        part.take([60])
    except IndexError:
        pass
    else:
        raise AssertionError("a table outside the part was read")


def test_stored_grid_places_rows_in_the_examined_coordinates():
    out, plan, shards, census, z = tables()
    grid = IO.stored_grid(z["tok"], z["col"])
    want = np.full(z["tok"].shape, A.GAP, dtype=np.uint8)
    for t in range(0, 60, 7):
        for q in range(16):
            for b, (_, s0, s1) in enumerate(V.SLICES):
                for k in range(int(z["n"][t, q, b])):
                    if z["col"][t, q, s0 + k] >= 0:
                        want[t, q, s0 + z["col"][t, q, s0 + k]] = z["tok"][t, q, s0 + k]
        assert np.array_equal(grid[t], want[t])
        own = z["tok"][t, 0] != A.PAD
        assert np.array_equal(grid[t, 0][own], z["tok"][t, 0][own]) and (grid[t, 0][~own] == A.GAP).all()
        same = (grid[t, 1:] == grid[t, :1]) & (grid[t, 1:] != A.GAP)       # identity read from the grid is the meta field's
        s0, s1 = V.SLICES[V.TARGET][1:]
        for q in range(1, 16):
            if z["genome"][t, q] >= 0:
                filled = (grid[t, q, s0:s1] != A.GAP).sum()
                assert abs(same[q - 1, s0:s1].sum() / filled - z["identity"][t, q]) < 1e-6


def test_identity_bins():
    assert C.bin_of([0.0, 0.049, 0.05, 0.949, 0.95, 0.989, 0.99, 0.999, 1.0]).tolist() == [0, 0, 1, 18, 19, 19, 20, 20, 21]
    assert len(C.BIN_LABELS) == C.N_BINS == 22


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
