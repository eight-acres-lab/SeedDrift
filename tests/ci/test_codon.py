"""The codon-usage study (seeddrift/train/codon_study.py, ci/stages/codon.py) and the protocol's paired intervals."""
import json
import os
import sys

import numpy as np

try:
    import pytest
except ImportError:                       # the model interpreter of the servers has torch but no pytest: run this file
    import contextlib                     # directly (python tests/ci/test_codon.py) with this stand-in
    import importlib

    class pytest:  # noqa: N801
        importorskip = staticmethod(importlib.import_module)

        @staticmethod
        @contextlib.contextmanager
        def raises(exc):
            try:
                yield
            except exc:
                return
            raise AssertionError("did not raise %s" % exc)

        class MonkeyPatch:
            def __init__(self):
                self._undo = []

            def setattr(self, obj, name, value):
                self._undo.append((obj, name, getattr(obj, name)))
                setattr(obj, name, value)

            def undo(self):
                for obj, name, value in reversed(self._undo):
                    setattr(obj, name, value)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from ci.stages import protocol as PR  # noqa: E402
from seeddrift.train import codon_study as CS  # noqa: E402


def _codon(s):
    return 16 * "ACGT".index(s[0]) + 4 * "ACGT".index(s[1]) + "ACGT".index(s[2])


def test_genetic_code_classes():
    aa = CS.amino_acids()
    names = sorted(set(CS.TCAG))
    assert names[aa[_codon("ATG")]] == "M" and names[aa[_codon("TGG")]] == "W"
    assert {names[aa[_codon(c)]] for c in ("TAA", "TAG", "TGA")} == {"*"}
    sizes = np.bincount(aa, minlength=21)
    assert sizes[names.index("L")] == 6 and sizes[names.index("S")] == 6 and sizes[names.index("R")] == 6 and sizes[names.index("*")] == 3
    assert len(names) == 21 and sizes.sum() == 64


def _write_genome(path, rng, n_genes, bias, length=(60, 300), places=True):
    """A genome npz as train_data.one_genome writes it (the arrays the study reads): codon tokens drawn per gene from
    `bias` (a function gene index -> 64 probabilities), 65 sprinkled in."""
    cod, off = [], [0]
    for i in range(n_genes):
        L = int(rng.integers(*length))
        t = rng.choice(64, size=L, p=bias(i)) + 1
        t[rng.random(L) < 0.01] = 65
        cod.append(t.astype(np.uint8))
        off.append(off[-1] + L)
    arrays = dict(codons=np.concatenate(cod), codon_off=np.array(off, np.int64))
    if places:
        start = np.arange(n_genes) * 1000 + 1
        arrays.update(replicon=np.zeros(n_genes, np.int64), start=start, end=start + 899)
    np.savez(path, **arrays)


def _profiles(rng, k):
    return [rng.dirichlet(np.ones(64) * 5) for _ in range(k)]


def _setup(tmp_path, seed=0, n=60, foreign=range(20, 30)):
    rng = np.random.default_rng(seed)
    host, donor, *others = _profiles(rng, 6)
    gp = tmp_path / "chrom.npz"
    _write_genome(gp, rng, n, lambda i: donor if i in foreign else host)
    gen = CS.Light(dict(id="C1", ranks=[1, 2, 3, 4, 5, 6]), str(gp), places=True)
    prof = {}
    for j, q in enumerate([donor] + others):
        p = tmp_path / ("g%d.npz" % j)
        _write_genome(p, rng, 30, lambda i, q=q: q, places=False)
        prof["g%d" % j] = CS.Light(dict(id="g%d" % j), str(p)).profile()
    return gen, prof


def test_cu_reproduces_the_readout_and_its_parts_add_up(tmp_path):
    torch = pytest.importorskip("torch")
    gen, prof = _setup(tmp_path)
    members = sorted(prof)
    spec = dict(name="p", scores=list(CS.SCORES))
    cols, diag = CS.score_chromosome(torch, torch.device("cpu"), gen, [(spec, members)], prof, {g: np.zeros(256) for g in members}, [], {})
    # selfsup.readouts' R_CU: KL of the gene's smoothed frequencies to its genome without it, minus the lowest KL to a panel genome
    N = gen.gene_counts()
    F = (N + 0.5) / (N + 0.5).sum(1, keepdims=True)
    rest = N.sum(0)[None, :] - N + 0.5
    kl_own = (F * (np.log(F) - np.log(rest / rest.sum(1, keepdims=True)))).sum(1)
    logp = np.log(np.stack([(prof[g] + 0.5) / (prof[g] + 0.5).sum() for g in members]))
    kl_pan = (F * np.log(F)).sum(1, keepdims=True) - F @ logp.T
    assert np.allclose(cols["cd_cu_p"], kl_own - kl_pan.min(1), atol=1e-10)
    # with one panel genome the amino-acid and synonymous parts add up to CU exactly
    one, _ = CS.score_chromosome(torch, torch.device("cpu"), gen, [(dict(name="q", scores=["cu", "aa", "syn"]), members[:1])], prof, {}, [], {})
    assert np.allclose(one["cd_aa_q"] + one["cd_syn_q"], one["cd_cu_q"], atol=1e-10)
    # relaxation contains CU (rate 0), the donor's genes win, the host variants are finite
    assert np.all(cols["cd_relax_p"] >= cols["cd_cun_p"] - 1e-9)
    for c in ("cd_hin_p", "cd_trim_p", "cd_block_p", "cd_mix3_p", "cd_lse_p", "cd_std_p", "cd_tot_p", "cd_m1_p"):
        assert np.isfinite(cols[c]).all(), c
    y = np.zeros(gen.n, int)
    y[20:30] = 1
    for c in ("cd_cu_p", "cd_cun_p", "cd_relax_p", "cd_trim_p", "cd_block_p"):
        assert PR.auroc(cols[c][y == 1], cols[c][y == 0]) > 0.9, c
    assert (diag["win_p"][20:30] == members.index("g0")).mean() > 0.8          # the donor is the best panel genome
    assert diag["relax_la_p"].shape == (gen.n,)
    # a filled and an emptied panel (its whole clade hidden) write the same columns: CS.panel_columns
    assert sorted(c for c in cols if c.endswith("_p")) == sorted(CS.panel_columns(set(CS.SCORES), "p"))
    empty, ediag = CS.score_chromosome(torch, torch.device("cpu"), gen, [(spec, [])], prof, {}, [], {})
    assert sorted(empty) == sorted(CS.panel_columns(set(CS.SCORES), "p")) and all(np.isnan(v).all() for v in empty.values())
    assert (ediag["win_p"] == -1).all()


def test_block_host_drops_the_neighbourhood(tmp_path):
    gen, _ = _setup(tmp_path, n=10)
    N = gen.gene_counts()
    H = CS.block_hosts(gen, N, 1500.0)                  # middles 1000 apart: the gene and its two neighbours go
    assert np.allclose(H[5], N.sum(0) - N[4:7].sum(0))
    assert np.allclose(H[0], N.sum(0) - N[0:2].sum(0))


def test_panels_hash_order_hiding_and_controls():
    from seeddrift.train import selfsup as SS
    meta = {"g%02d" % i: dict(id="g%02d" % i, split="train" if i < 16 else "heldout", group="bacteria" if i % 3 else "viruses.phage", genes=10,
                              ranks=[1, 2, 3, 40 + (i % 2), 50 + i, 60 + i]) for i in range(24)}
    corpora = dict(c=meta)

    class _D:
        g = {k: type("G", (), dict(meta=v)) for k, v in meta.items()}
    ids = sorted(g for g, m in meta.items() if m["split"] == "train")
    assert CS.panel_of(ids, {g: meta[g]["group"] for g in ids}, 7) == SS.panel_of(_D, ids, 7)
    base = CS.base_members(dict(name="a", corpus="c", splits=["train"], size=7), corpora, {})
    assert base == SS.panel_of(_D, ids, 7)
    allb = CS.base_members(dict(name="b", corpus="c", splits=["train", "heldout"], groups=["bacteria"]), corpora, {})
    assert len(allb) == 16 and all(meta[g]["group"] == "bacteria" for g in allb)
    with pytest.raises(SystemExit):
        CS.base_members(dict(name="x", corpus="c", splits=["test"]), corpora, {})
    chrom = dict(id="C", ranks=[1, 2, 3, 41, 999, 9999])
    hid = CS.members_for(dict(name="h", hide="family"), allb, chrom, meta.__getitem__)
    assert hid == [g for g in allb if meta[g]["ranks"][3] != 41]
    ctl = CS.members_for(dict(name="r", hide="family", control="random", seed=1), allb, chrom, meta.__getitem__)
    assert len(ctl) == len(hid) and set(ctl) <= set(allb)
    assert CS.members_for(dict(name="r", hide="family", control="random", seed=1), allb, chrom, meta.__getitem__) == ctl     # seeded
    curve = CS.expand_curve([dict(name="b", corpus="c", sizes=[2, 4], seeds=[1, 2])])
    assert [s["name"] for s in curve] == ["rc_b_2s1", "rc_b_2s2", "rc_b_4s1", "rc_b_4s2", "rc_b_all"]


def test_manifold_finds_its_dimension_and_positions():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    d, G = 2, 160
    W = rng.normal(size=(d, 64)) * 0.6
    b = rng.normal(size=64) * 0.5
    Z = rng.normal(size=(G, d))
    q = np.exp(b + Z @ W)
    q /= q.sum(1, keepdims=True)
    C = np.stack([rng.multinomial(200000, p) for p in q]).astype(float)
    rep = CS.manifold_curve(torch, torch.device("cpu"), C, ["g%d" % i for i in range(G)], [1, 2, 4], 1500, 0.05, 1)
    de = {r["d"]: r for r in rep["curve"]}
    assert de[2]["deviance_explained_train"] > 0.97 and de[1]["deviance_explained_train"] < de[2]["deviance_explained_train"] - 0.05
    assert de[2]["deviance_explained_heldout"] > 0.95
    bf, Wf, Zf, _ = CS.fit_manifold(torch, torch.device("cpu"), C, d, 1500, 0.05, 1)
    mu, S = Zf.mean(0), np.cov(Zf.T) + 1e-6 * np.eye(d)
    t = lambda x: torch.tensor(x, dtype=torch.float64)
    # a long gene drawn from genome 0's profile sits at genome 0's place; its Bayes factor against genome 0 is small
    N = t(rng.multinomial(20000, q[0])[None, :].astype(float))
    z, ph, ll, ld = CS.map_positions(torch, N, t(bf), t(Wf), t(mu), t(np.linalg.inv(S)), t(mu[None, :]))
    assert np.linalg.norm(z.numpy()[0] - Zf[0]) < 0.1 * np.linalg.norm(Zf[0] - mu) + 0.05


def test_paired_delta_is_zero_on_itself_and_decides_a_clear_gap():
    rng = np.random.default_rng(5)
    table = {}
    for k, unit in enumerate(["u1", "u1", "u2", "u3"]):
        n = 400
        y = np.zeros(n, int)
        y[100:130] = 1
        y[250:262] = 1
        y[rng.random(n) < 0.05] = -1
        good = y + rng.normal(size=n) * 0.5
        bad = rng.normal(size=n)
        good[3] = np.nan
        table["c%d" % k] = dict(y=y, unit=unit, cols=dict(good=good, bad=bad), start=np.arange(n) * 1000.0)
    units = ["u1", "u2", "u3"]
    rows = PR.paired_delta(table, units, [("good", "good"), ("good", "bad"), ("good", None)], n_boot=300, seed=1)
    same, gap, alone = rows
    assert same["delta"] == 0 and same["lo"] == 0 and same["hi"] == 0 and same["verdict"] == "undecided" and same["small"]
    assert gap["verdict"] == "a>b" and gap["lo"] > 0.2 and not gap["small"]
    # the point estimate is the protocol's AUROC (per chromosome, mean over a unit's chromosomes, mean over units)
    per = {}
    for t in table.values():
        per.setdefault(t["unit"], []).append(PR.auroc(t["cols"]["good"][t["y"] == 1], t["cols"]["good"][t["y"] == 0]))
    assert abs(alone["delta"] - np.mean([np.mean(v) for v in per.values()])) < 1e-12
    assert alone["lo"] <= alone["delta"] <= alone["hi"]
    cid, flag = PR.clusters_of(np.array([0, 0, 0, 1, 1, -1, 1, 0, 0]), np.arange(9), block=2)
    assert list(cid) == [0, 0, 1, 2, 2, -1, 2, 3, 3] and list(flag) == [False, False, True, False]


def test_codon_pairs():
    from ci.stages import codon as CO
    cols = ["cd_cu_a", "cd_cun_a", "cd_trim_a", "cd_aa_a", "cd_cu_a_hfam", "cd_cu_a_rfam1", "cd_mfa4"]
    pairs = CO.default_pairs(cols, "cd_cu_a", ["cmp_codon"])
    assert ("cd_trim_a", "cd_cun_a") in pairs and ("cd_aa_a", "cd_cu_a") in pairs and ("cmp_codon", "cd_cu_a") in pairs and ("cd_cu_a", None) in pairs
    specs = [dict(name="a_hfam", base="a", hide="family"), dict(name="a_rfam1", base="a", hide="family", control="random", seed=1)]
    assert CO.panel_contrasts(cols, specs) == [("cd_cu_a_hfam", "cd_cu_a"), ("cd_cu_a_hfam", "cd_cu_a_rfam1"), ("cd_cu_a_rfam1", "cd_cu_a")]


def _stage_json(run, stage, outputs):
    from ci.hashing import sha256_file
    (run / "stages").mkdir(exist_ok=True)
    (run / "stages" / (stage + ".json")).write_text(json.dumps(dict(status="passed", outputs=[
        dict(name=n, path=os.path.relpath(p, run), sha256=sha256_file(str(p), use_sidecar=False)) for n, p in outputs])))


def _manifest(run, name, paths):
    from ci import manifest as M
    man = M.build(name, name, [dict(id=p, files=[M.file_entry(str(run / p), str(run))]) for p in paths])
    return M.write(man, str(run / "manifests"))


def test_codon_stage_end_to_end(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")  # noqa: F841
    import yaml
    from ci.stages import codon as CO
    from ci.stages import gates as GA
    from ci.stages import train as TR
    rng = np.random.default_rng(11)
    host, donor, *others = _profiles(rng, 8)
    # the island run: two chromosomes of two units, their arrays, comparators, evaluator output, gates
    isl = tmp_path / "runs" / "isl"
    (isl / "data" / "islands" / "train_data").mkdir(parents=True)
    (isl / "gates").mkdir()
    store = tmp_path / "store"
    (store / "labels_panel").mkdir(parents=True)
    genomes, arrays = [], []
    res = dict(per_chromosome={}, eligibility={}, units={})
    for k, (acc, genus) in enumerate([("NC_1", "Escherichia"), ("NC_2", "Salmonella")]):
        n = 80
        p = isl / "data" / "islands" / "train_data" / (acc + ".npz")
        _write_genome(p, rng, n, lambda i: donor if 30 <= i < 42 else host)
        folder = isl / "data" / "islands" / "inputs" / acc
        folder.mkdir(parents=True)
        names = ["%s_g%d" % (acc, i) for i in range(n)]
        (folder / "genes.tsv").write_text("gene_id\n" + "\n".join(names) + "\n")
        rows = ["gene_id\tclass\tstart\tend\tf\tlabel"] + ["%s\t%s\t%d\t%d\t%s\tnone" % (g, "island" if 30 <= i < 42 else "native", i * 1000 + 1, i * 1000 + 900,
                                                                                           "0.2" if 30 <= i < 42 else "1.0") for i, g in enumerate(names)]
        (store / "labels_panel" / (acc + ".tsv")).write_text("\n".join(rows) + "\n")
        cdir = isl / "data" / "islands" / "scores" / "comparators"
        cdir.mkdir(parents=True, exist_ok=True)
        from ci.stages.islands import COMPARATORS
        with open(cdir / ("scores_%s.tsv" % acc), "w") as f:
            f.write("gene_id\t" + "\t".join(COMPARATORS) + "\n")
            for i, g in enumerate(names):
                f.write(g + "\t" + "\t".join(repr(float(x)) for x in rng.normal(size=len(COMPARATORS))) + "\n")
        genomes.append(dict(id=acc, genus=genus, taxid=100 + k, ranks=[1, 2, 3, 4, 10 + k, 100 + k], hidden_taxids=[], group="bacteria",
                            path=os.path.relpath(p, isl), folder=os.path.relpath(folder, isl)))
        arrays += [os.path.relpath(p, isl), os.path.relpath(folder / "genes.tsv", isl)]
        res["per_chromosome"][acc] = dict(v1=dict(gated=dict(auroc=0.5)))
        res["eligibility"][acc] = dict(dev_gene=True)
        res["units"][acc] = genus
    msha = _manifest(isl, "islands", arrays)
    (isl / "data" / "islands" / "manifest.json").write_text(json.dumps(dict(sha256=msha)))
    (isl / "data" / "islands" / "index.json").write_text(json.dumps(dict(genomes=genomes)))
    (isl / "data" / "islands" / "eval").mkdir()
    (isl / "data" / "islands" / "eval" / "dev.json").write_text(json.dumps(res))
    (isl / "data" / "islands" / "scores.json").write_text("{}")
    (isl / "config.resolved.yaml").write_text(yaml.safe_dump(dict(islands=dict(store=str(store)))))
    _stage_json(isl, "islands_data", [("islands_index", isl / "data" / "islands" / "index.json")])
    _stage_json(isl, "islands_report", [("islands_eval", isl / "data" / "islands" / "eval" / "dev.json")])
    _stage_json(isl, "islands_score", [("islands_scores", isl / "data" / "islands" / "scores.json")])
    (isl / "gates" / "G3_islands.json").write_text(json.dumps(dict(status="pass")))
    # a corpus run: twelve genomes of other genera, one of them the donor's composition
    cor = tmp_path / "runs" / "cor"
    (cor / "data" / "train_data").mkdir(parents=True)
    meta, paths = [], []
    for j in range(12):
        gid = "G%02d" % j
        p = cor / "data" / "train_data" / (gid + ".npz")
        q = donor if j == 0 else others[j % len(others)]
        _write_genome(p, rng, 40, lambda i, q=q: q, places=False)
        meta.append(dict(id=gid, split="train" if j < 9 else "heldout", group="bacteria", taxid=1000 + j, ranks=[1, 2, 3, 40 + j % 2, 500 + j, 1000 + j], genes=40))
        paths.append(os.path.relpath(p, cor))
    tsha = _manifest(cor, "train-data", paths)
    (cor / "data" / "train_data" / "index.json").write_text(json.dumps(dict(sha256=tsha, genomes=meta)))
    _stage_json(cor, "train_data", [("train_data_index", cor / "data" / "train_data" / "index.json")])
    # a taxonomy: genus nodes of the chromosomes and of the corpus genomes
    tax = tmp_path / "tax"
    tax.mkdir()
    nodes = ["1\t|\t1\t|\tno rank\t|"] + ["%d\t|\t1\t|\tgenus\t|" % g for g in [10, 11] + [500 + j for j in range(12)]] + \
            ["%d\t|\t%d\t|\tspecies\t|" % (100 + k, 10 + k) for k in range(2)] + ["%d\t|\t%d\t|\tspecies\t|" % (1000 + j, 500 + j) for j in range(12)]
    names = ["10\t|\tEscherichia\t|\t\t|\tscientific name\t|", "11\t|\tSalmonella\t|\t\t|\tscientific name\t|"] + \
            ["%d\t|\tGenus%d\t|\t\t|\tscientific name\t|" % (500 + j, j) for j in range(12)]
    (tax / "nodes.dmp").write_text("\n".join(nodes) + "\n")
    (tax / "names.dmp").write_text("\n".join(names) + "\n")
    # our run
    run = tmp_path / "runs" / "ours"
    (run / "gates").mkdir(parents=True)
    (run / "gates" / "pre.ok").write_text("{}")
    cfg = dict(pipeline="codon", seed=1, taxdump=str(tax), train=dict(python=sys.executable, gpu_lease=False, device="cpu"),
               codon=dict(islands_from=str(isl), corpora=dict(c=str(cor)), baseline="cd_cu_a", boot=50,
                          panels=[dict(name="a", corpus="c", splits=["train"], size=63, scores=["cu", "aa", "syn", "cun", "host", "relax"]),
                                  dict(name="a_hfam", base="a", hide="family"), dict(name="a_rfam1", base="a", hide="family", control="random", seed=1)],
                          curves=[dict(name="b", corpus="c", splits=["train", "heldout"], sizes=[2, 4], seeds=[1, 2])],
                          manifolds=[dict(name="m", corpus="c", splits=["train", "heldout"], dims_curve=[1, 2], dims_score=[2], steps=200)]))
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(TR, "model_env", lambda st, py, lock=None: [])
    CO.do_study(str(run))
    assert (run / "data" / "codon" / "scores" / "scores_NC_1.tsv").exists()
    CO.do_report(str(run))
    rep = json.loads((run / "data" / "codon" / "report.json").read_text())
    assert rep["scores"]["cd_cu_a"]["auroc"] > 0.9 and rep["chromosomes"] == ["NC_1", "NC_2"]
    assert set(rep["panel_size_curve"]["b"]) == {"2", "4", "all"}
    assert any(r["a"] == "cd_trim_a" and r["b"] == "cd_cun_a" for r in rep["intervals"])
    assert "m" in rep["manifolds"] and rep["winners"]["a"]["island"]
    # the gates pass; a corpus genome of an island chromosome's genus fails G3_codon
    st = type("St", (), dict(repo=staticmethod(lambda rel: rel if os.path.isabs(rel) else os.path.join(ROOT, rel))))
    GA.phase_codon(st, cfg, str(run))
    assert json.loads((run / "gates" / "G3_codon.json").read_text())["status"] == "pass"
    assert json.loads((run / "gates" / "G4_codon.json").read_text())["status"] == "pass"
    # a second configured corpus with a panel genome in its test part fails G3_codon (the sealed part of any corpus)
    cor2 = tmp_path / "runs" / "cor2"
    (cor2 / "data" / "train_data").mkdir(parents=True)
    (cor2 / "data" / "train_data" / "index.json").write_text(json.dumps(dict(sha256="x", genomes=[dict(meta[0], split="test")])))
    cfg2 = dict(cfg, codon=dict(cfg["codon"], corpora=dict(c=str(cor), d=str(cor2))))
    with pytest.raises(Exception):
        GA.phase_codon(st, cfg2, str(run))
    g3 = json.loads((run / "gates" / "G3_codon.json").read_text())
    assert g3["status"] == "fail" and any(p.get("split") == "test" for p in g3["evidence"]["problems"]), g3
    (tax / "nodes.dmp").write_text("\n".join(nodes).replace("1000\t|\t500\t|", "1000\t|\t10\t|") + "\n")
    with pytest.raises(Exception):
        GA.phase_codon(st, cfg, str(run))
    assert json.loads((run / "gates" / "G3_codon.json").read_text())["status"] == "fail"


if __name__ == "__main__":
    import tempfile
    import pathlib
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            args = fn.__code__.co_varnames[:fn.__code__.co_argcount]
            kw = {}
            if "tmp_path" in args:
                kw["tmp_path"] = pathlib.Path(tempfile.mkdtemp())
            if "monkeypatch" in args:
                kw["monkeypatch"] = pytest.MonkeyPatch()
            fn(**kw)
            print("ok", name)
