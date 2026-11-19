"""The self-supervised model of the truth-labels line (seeddrift/train/selfsup.py) on synthetic genomes.

The numpy parts run everywhere; the torch parts are skipped where torch is not installed (the CI environment) and run
on the GPU machine with the model's Python (python tests/ci/test_selfsup.py)."""
import json
import os
import tempfile

import numpy as np

try:
    import pytest
except ImportError:                                                    # the model's Python has no pytest
    pytest = None

from seeddrift.train import selfsup as SS
from seeddrift.train import source_model as S

N_LIB = 12


def _lib():
    taxid = np.arange(100, 100 + N_LIB, dtype=np.int64)
    ranks = np.zeros((N_LIB, 6), dtype=np.int64)
    ranks[:, 0] = 1                                                     # one phylum
    ranks[:, 2] = 30 + np.arange(N_LIB) // 6                            # two orders
    ranks[:, 3] = 50 + np.arange(N_LIB) // 4                            # three families
    ranks[:, 4] = 70 + np.arange(N_LIB) // 2                            # six genera
    ranks[:, 5] = taxid
    return taxid, ranks


def make_run(root, n_genomes=10, seed=0):
    """A run directory with data/reference/library.npz and data/train_data (index.json and one npz per genome) in the
    layout of ci/stages/train_data.py. Genome k is library row k % N_LIB; genomes 0..6 train, 7..9 held out."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "data", "train_data"), exist_ok=True)
    os.makedirs(os.path.join(root, "data", "reference"), exist_ok=True)
    taxid, ranks = _lib()
    np.savez(os.path.join(root, "data", "reference", "library.npz"), taxid=taxid, ranks=ranks)
    metas = []
    for k in range(n_genomes):
        row = k % N_LIB
        bias = rng.dirichlet(np.ones(64) * 0.5)                         # each genome its own codon usage
        n = int(rng.integers(8, 16))
        cod, coff, hoff, hprot, hqs, tr, toff = [], [0], [0], [], [], [], [0]
        for i in range(n):
            L = int(rng.integers(12, 40))
            c = (rng.choice(64, size=L, p=bias) + 1).astype(np.uint8)
            cod.append(c)
            coff.append(coff[-1] + L)
            nh = int(rng.integers(0, 5))
            for _ in range(nh):
                qs = int(rng.integers(0, 4))
                t = rng.integers(1, 23, size=int(rng.integers(1, L - qs))).astype(np.uint8)
                hprot.append(int(rng.integers(0, N_LIB)))
                hqs.append(qs)
                tr.append(t)
                toff.append(toff[-1] + len(t))
            hoff.append(hoff[-1] + nh)
        start = np.arange(n, dtype=np.int64) * 1000 + 1
        np.savez(os.path.join(root, "data", "train_data", "g%d.npz" % k), codons=np.concatenate(cod), codon_off=np.array(coff, dtype=np.int64),
                 replicon=np.zeros(n, dtype=np.int32), start=start, end=start + 900, strand=np.ones(n, dtype=np.int8),
                 replicon_len=np.array([n * 1000], dtype=np.int64), hit_off=np.array(hoff, dtype=np.int64), hit_prot=np.array(hprot, dtype=np.int32),
                 hit_qs=np.array(hqs, dtype=np.int32), track=np.concatenate(tr) if tr else np.zeros(0, np.uint8), track_off=np.array(toff, dtype=np.int64))
        metas.append(dict(id="g%d" % k, source="corpus", group="bacteria" if k % 3 else "viruses.phage", taxid=int(taxid[row]),
                          ranks=[int(x) for x in ranks[row]], genes=n, hits=hoff[-1], split="train" if k < 7 else "heldout", cluster="", relation=""))
    with open(os.path.join(root, "data", "train_data", "index.json"), "w") as f:
        json.dump(dict(dataset_id="ds-test", genomes=metas, library="data/reference/library.npz"), f)
    make_pairs(root, metas, rng)
    return root


def make_pairs(root, metas, rng):
    """data/pairs in the layout of ci/stages/pairs.py: per gene up to three homologs in other genomes, ungapped or with
    one insertion and one deletion, starting at the first residues."""
    d = os.path.join(root, "data", "pairs")
    os.makedirs(d, exist_ok=True)
    order = [m["id"] for m in metas]
    with open(os.path.join(d, "genomes.json"), "w") as f:
        json.dump(dict(genomes=order), f)
    for a, m in enumerate(metas):
        off, tg, tj, qs, ss, cig, cig_off, sq, sq_off = [0], [], [], [], [], bytearray(), [0], bytearray(), [0]
        for i in range(m["genes"]):
            for b in rng.choice([x for x in range(len(order)) if x != a], size=int(rng.integers(0, 4)), replace=False):
                n = int(rng.integers(6, 12))
                c = "%dM" % n if rng.random() < 0.5 else "3M1I%dM1D2M" % (n - 6)
                n_t = sum(int(k) for k, op in __import__("re").findall(r"(\d+)([MID])", c) if op in "MD")
                tg.append(int(b))
                tj.append(int(rng.integers(0, metas[b]["genes"])))
                qs.append(1)
                ss.append(1)
                cig += c.encode()
                cig_off.append(len(cig))
                sq += "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), n_t)).encode()
                sq_off.append(len(sq))
            off.append(len(tg))
        np.savez(os.path.join(d, m["id"] + ".npz"), off=np.array(off, np.int32), tgenome=np.array(tg, np.int32), tgene=np.array(tj, np.int32),
                 qs=np.array(qs, np.uint32), ss=np.array(ss, np.uint32), bits=np.ones(len(tg), np.float32), cigar=np.frombuffer(bytes(cig), np.uint8),
                 cigar_off=np.array(cig_off, np.int64), sseq=np.frombuffer(bytes(sq), np.uint8), sseq_off=np.array(sq_off, np.int64))


CFG = dict(seed=1, lr=1e-3, steps=3, batch=4, width=8, eval_every=2, ckpt_every=2, source_genes=4, source_codons=64, source_hits=4, b2_hits=2,
           held_batches=1, held_genes=5, hits_centre=4, max_codons=64, max_track=64, dev_seed=0)


def test_synonym_table():
    assert SS.SYN[:, 1:65].sum(0).tolist() == [1] * 64                  # every codon codes one amino acid (or stop)
    assert not SS.SYN[:, 0].any() and not SS.SYN[:, 65].any()           # padding and unknown codons code nothing
    aa = {a: i + 1 for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    assert SS.N_SYN[aa["M"]] == 1 and SS.N_SYN[aa["W"]] == 1 and SS.N_SYN[aa["L"]] == 6 and SS.N_SYN[21] == 3


def test_pair_track_follows_the_cigar():
    t, pos = SS.pair_track(b"2M1I1M2D1M", b"ACDEFG", 5)
    assert t.tolist() == [1, 2, 22, 3, 6] and pos.tolist() == [4, 5, -1, 6, 9]


def test_c_sampler_and_example():
    with tempfile.TemporaryDirectory() as d:
        data = S.Data(make_run(d), CFG)
        pairs = SS.Pairs(d)
        train = data.by_split["train"]
        cs = SS.CSampler(data, pairs, train, train)
        items, groups, depth = cs.batch(np.random.default_rng(0), 4, CFG)
        assert 0 < len(items) <= 4 and depth in SS.DEPTHS
        for e in items:
            assert len(e["t"]) == len(e["tcod"]) and e["qs"] == 0
            ok = (e["tcod"] > 0)
            assert ((e["t"][ok] == 22) | (S.CODE[e["tcod"][ok]] > 0)).all()
            assert e["partner_gene"] not in e["B"]["idx"]               # the partner gene does not describe its genome


def test_objectives_validated():
    assert SS.objectives_of({}) == ["A", "B1", "B2"]
    assert SS.objectives_of({"objectives": ["C", "A"]}) == ["A", "C"]
    assert SS.objectives_of({"objectives": ["B2", "A"]}) == ["A", "B2"]
    for bad in (["D"], [], ["A", "A"]):
        try:
            SS.objectives_of({"objectives": bad})
        except SystemExit:
            continue
        raise AssertionError("objectives %r accepted" % bad)


def test_hidden_rows_union_of_clades_and_own_proteomes():
    with tempfile.TemporaryDirectory() as d:
        data = S.Data(make_run(d), CFG)
        g0, g3 = data.g["g0"], data.g["g3"]                             # library rows 0 (genus 70) and 3 (genus 71)
        m = SS.hidden_rows(data, [g0, g3], "genus")
        assert m.tolist() == [True] * 4 + [False] * (N_LIB - 4)
        m = SS.hidden_rows(data, [g0], "family", taxids=[111])
        assert m[:4].all() and not m[4:11].any() and m[11]
        m = SS.hidden_rows(data, [g0], "species")
        assert m.tolist() == [True] + [False] * (N_LIB - 1)


def test_example_hides_and_excludes_the_gene():
    with tempfile.TemporaryDirectory() as d:
        data = S.Data(make_run(d), CFG)
        gen = data.g["g1"]
        hidden = np.zeros(N_LIB, bool)
        hidden[::2] = True
        rng = np.random.default_rng(0)
        for i in range(gen.n):
            e = SS.example(data, gen, i, hidden, rng, CFG)
            assert all(not hidden[p] for p in e["prots"])
            assert all(not hidden[p] for ps in e["src_prots"] for p in ps)
            assert len(e["src"]) == min(CFG["source_genes"], gen.n - 1)
            assert np.allclose(e["gcomp"], S.frequencies(gen.ccount - gen.gene_count(i)))
            for t, q, qs, tg in zip(e["tracks"], e["prots"], e["qs"], e["tgts"]):
                assert np.array_equal(t, S.CODE[e["c"]][qs:qs + len(t)] * 23 + tg)


def test_sampler_distinct_genomes_with_near_pairs():
    with tempfile.TemporaryDirectory() as d:
        data = S.Data(make_run(d), CFG)
        ids = data.by_split["train"]
        sm = SS.Sampler(data, ids)
        for s in range(20):
            rng = np.random.default_rng(s)
            g = sm.genomes(rng, 6)
            assert len(g) == len(set(g)) == 6 and set(g) <= set(ids)
            g = sm.genomes(rng, 50)                                    # more than there are: every genome once
            assert sorted(g) == sorted(ids)
        assert "g0" in sm.near["g1"]                                   # same genus (library rows 0 and 1)


def test_panel_round_robin_and_fixed():
    with tempfile.TemporaryDirectory() as d:
        data = S.Data(make_run(d), CFG)
        ids = data.by_split["train"]
        p = SS.panel_of(data, ids, 4)
        assert p == SS.panel_of(data, list(reversed(ids)), 4) and len(set(p)) == 4
        assert {data.g[g].meta["group"] for g in p[:2]} == {"bacteria", "viruses.phage"}


def test_combine_is_mean_of_genome_z_scores():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([4.0, np.nan, 2.0, 1.0])
    c = SS.combine([a, b])
    za = (a - a.mean()) / a.std()
    ok = np.isfinite(b)
    zb = np.where(ok, (b - b[ok].mean()) / b[ok].std(), np.nan)
    assert np.allclose(c, [(za[0] + zb[0]) / 2, za[1], (za[2] + zb[2]) / 2, (za[3] + zb[3]) / 2])


def _torch():
    return pytest.importorskip("torch") if pytest else __import__("torch")


def test_b_objectives_do_not_read_their_targets():
    """B2 never reads the hit tracks (its target), and B1 reads the gene's amino acids, not its codons (its target):
    changing a gene's tracks and its codons synonymously leaves both predictions unchanged."""
    torch = _torch()
    with tempfile.TemporaryDirectory() as d:
        data = S.Data(make_run(d), CFG)
        rng = np.random.default_rng(3)
        gens = [data.g[g] for g in ("g0", "g4", "g5")]
        hidden = SS.hidden_rows(data, gens, "genus")
        items = [SS.example(data, g, 0, hidden, rng, CFG) for g in gens]
        torch.manual_seed(0)
        model = SS.build(torch, data.n_lib, d=8).eval()
        x = SS.collate(items, torch, 2)
        y = {k: v.clone() for k, v in x.items()}
        y["tracks"] = torch.where(y["tracks"] > 0, (y["tracks"] % 500) + 1, y["tracks"])
        syn = {}
        for c in range(1, 65):
            syn.setdefault(int(S.CODE[c]), []).append(c)
        cod = y["codons"].clone()
        for b in range(cod.shape[0]):
            for p in range(cod.shape[1]):
                c = int(cod[b, p])
                if 0 < c < 65 and len(syn[int(S.CODE[c])]) > 1:
                    alt = syn[int(S.CODE[c])]
                    cod[b, p] = alt[(alt.index(c) + 1) % len(alt)]
        y["codons"] = cod
        with torch.no_grad():
            G1, G2 = model.genome_rep(x), model.genome_rep(y)
            assert torch.equal(G1, G2)
            h1, h2 = model.aa_states(x["aa"]), model.aa_states(y["aa"])
            assert torch.equal(h1, h2)
            # B2 logits use h, the hit's proteome and G only: compare the aware likelihood of the same targets
            assert torch.allclose(model.b2(x, h1, G1)[0], model.b2(dict(y, b2_t=x["b2_t"]), h2, G2)[0])


def test_fit_and_score_end_to_end_on_cpu():
    _torch()
    with tempfile.TemporaryDirectory() as d:
        make_run(d)
        cfg = dict(CFG, logs="data/train/logs/t", out="data/model/t", objectives=["A", "B1", "B2", "C"], c_batch=4, c_partners=2)
        SS.fit(d, cfg)
        out = os.path.join(d, "data", "model", "t")
        z = np.load(os.path.join(out, "heldout_scores.npz"))
        summ = json.load(open(os.path.join(out, "heldout_summary.json")))
        assert summ["genomes"] == 3 and len(z["gene"]) == summ["genes"] > 0
        assert np.isfinite(z["A"]).all() and np.isfinite(z["B1"]).all() and np.isfinite(z["comb"]).all()
        assert np.isfinite(z["C"]).any() and (z["c_partner"][np.isfinite(z["C"])] >= 0).all()
        assert all(np.isfinite(v) for v in summ.values() if isinstance(v, float))
        rows = [json.loads(line) for line in open(os.path.join(d, "data", "train", "logs", "t", "metrics.jsonl"))]
        assert [r["step"] for r in rows] == [1, 2, 3] and all(np.isfinite(r["loss"]) for r in rows)
        # an external genome is scored gene by gene, in its order, with its hidden proteomes
        idx = dict(genomes=[dict(id="x", path="data/train_data/g8.npz", taxid=108, ranks=[1, 0, 31, 52, 74, 108], group="bacteria", hidden_taxids=[109],
                                 pairs="data/pairs/g8.npz")])
        with open(os.path.join(d, "ext.json"), "w") as f:
            json.dump(idx, f)
        SS.score_genomes(d, dict(cfg, genomes_index="ext.json", genomes_out="data/ext"))
        r = np.load(os.path.join(d, "data", "ext", "x.npz"))
        assert np.array_equal(r["gene"], np.arange(S.Data(d, cfg).g["g8"].n)) and np.isfinite(r["A"]).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
