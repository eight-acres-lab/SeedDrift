"""Helpers of the island stage of pipeline selfsup (ci/stages/islands.py): sealed or non-development chromosomes are
refused, gene names follow the benchmark's, the comparators point the right way, and the contrasts over units."""
import os

import numpy as np

from ci.stages import islands as I
from seeddrift.train import source_model as S

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _St:
    def repo(self, rel):
        return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)


def test_development_chromosomes_accepted():
    accs = I.accessions(_St(), {})
    assert len(accs) == 24 and all(a.startswith("NC_") for _, a in accs)


def test_sealed_or_unlisted_chromosome_fails(tmp_path):
    sealed = open(os.path.join(ROOT, "bench", "hard", "results", "sealed.tsv")).read().split("\n")[1].split("\t")[0]
    for acc in (sealed, "NC_999999.1"):
        p = tmp_path / ("g_%s.tsv" % acc)
        p.write_text("genus\taccession\nX\t%s\n" % acc)
        try:
            I.accessions(_St(), {"genomes": str(p)})
        except SystemExit as err:
            assert acc in str(err)
            continue
        raise AssertionError("chromosome %s was accepted" % acc)


def test_clean_names_match_the_pipeline():
    seqs = {"a b": "MKV" * 20, "a_b": "MKV" * 20, "short": "MKV", "x|y": "MK*V" * 10}
    out, ren = I.clean_names(seqs)
    assert ren == {"a b": "a_b", "a_b": "a_b_", "x|y": "x_y"} and out["x_y"] == "MKV" * 10
    try:
        from seeddrift.pipeline import clean_proteins
    except Exception:                                                   # noqa: BLE001 - optional cross-check
        return
    assert clean_proteins(seqs) == (out, ren)


def _genome(tmp_path):
    """Ten genes with the same codon usage and one (gene 3) with another, fewer hits and shorter."""
    rng = np.random.default_rng(0)
    base, other = rng.dirichlet(np.ones(64)), rng.dirichlet(np.ones(64))
    cod, coff, hoff = [], [0], [0]
    for i in range(10):
        L = 60 if i == 3 else 200
        c = rng.choice(64, size=L, p=other if i == 3 else base) + 1
        cod.append(c.astype(np.uint8))
        coff.append(coff[-1] + L)
        hoff.append(hoff[-1] + (0 if i == 3 else 3))
    n_h = hoff[-1]
    p = tmp_path / "g.npz"
    np.savez(p, codons=np.concatenate(cod), codon_off=np.array(coff), hit_off=np.array(hoff), hit_prot=np.zeros(n_h, np.int32),
             hit_qs=np.zeros(n_h, np.int32), track=np.ones(n_h, np.uint8), track_off=np.arange(n_h + 1), replicon=np.zeros(10, np.int32),
             start=np.arange(10) * 1000, end=np.arange(10) * 1000 + 600, strand=np.ones(10, np.int8), replicon_len=np.array([10000]))
    return S.Genome(dict(id="g", taxid=1, ranks=[0] * 6, group="bacteria"), str(p))


def test_comparators_point_to_the_odd_gene(tmp_path):
    c = I.comparators(_genome(tmp_path))
    for k in ("cmp_codon", "cmp_hits", "cmp_short"):
        assert int(np.argmax(c[k])) == 3, k
    assert all(np.isfinite(c[k]).all() for k in I.COMPARATORS if k not in ("cmp_pairnull", "cmp_triplet"))   # those need partner rows
    assert np.isnan(c["cmp_pairnull"]).all() and np.isnan(c["cmp_triplet"]).all()


def test_contrasts_over_units():
    per = {"a1": dict(_unit="U1", m=dict(r1=0.5, auroc=0.8), c=dict(r1=0.2, auroc=0.7)),
           "a2": dict(_unit="U1", m=dict(r1=0.3, auroc=0.8), c=dict(r1=0.2, auroc=0.7)),
           "b1": dict(_unit="U2", m=dict(r1=0.4, auroc=0.9), c=dict(r1=0.1, auroc=0.6))}
    rows = {(r["metric"], r["a"], r["b"]): r for r in I.contrasts(per, ["U1", "U2"], ["m"], ["c"])}
    r = rows[("r1", "m", "c")]
    assert r["n_units"] == 2 and np.allclose(r["per_unit"], [0.2, 0.3]) and np.isclose(r["mean"], 0.25)
    assert r["p_signflip"] == 0.5                                        # 2 of the 4 sign patterns are as extreme


def test_keep_bounds_rows_without_changing_the_selection():
    """Keep() retains what search.select() picks from a full list streamed in non-increasing bit score."""
    from ci.stages.search import select
    rng = np.random.default_rng(1)
    genus = {p: int(p // 7) for p in range(400)}
    for trial in range(20):
        n = int(rng.integers(50, 600))
        bits = np.sort(np.round(rng.uniform(30, 900, n)))[::-1]          # ties included
        rows = [(int(t), int(rng.integers(0, 400)), 0, 0, 0, 0, 0, 0, 0, float(b), "5M", "AAAAA") for t, b in zip(rng.permutation(n), bits)]
        k = I.Keep(8, genus)
        for r in rows:
            k.add(r)
        want = [r[:2] for r in select(list(rows), genus, 8)]
        got = [r[:2] for r in select(list(k.rows), genus, 8)]
        assert got == want and len(k.rows) <= len(rows)
        assert all(r[11] for r in select(list(k.rows), genus, 8)[:8])      # the nearest keep their residues



def _store(tmp_path, accs):
    store = tmp_path / "store"
    for sub in ("dev_genomes", "genes", "dev24_v/genus"):
        (store / sub).mkdir(parents=True, exist_ok=True)
    for _, acc in accs:
        (store / "dev_genomes" / (acc + ".gb")).write_text("LOCUS " + acc)
        (store / "genes" / (acc + ".tsv")).write_text("gene_id\n" + acc)
        (store / "dev24_v" / "genus" / ("hidden_%s.tsv" % acc)).write_text("taxid\n1\n")
    return store


def _island_run(root, name, store, accs, cfg_extra=None, version=1, islands_code=None, proteins=None):
    """A run folder with what search_candidates reads: record, resolved config, reference manifest, a passed island
    data stage (extras at the top of its report, as Stage writes them) with its manifest, index, input proteins and hit
    files, and the code of its worktree."""
    import json
    import shutil
    import yaml
    from ci.hashing import sha256_file
    run = root / "runs" / name
    (run / "data" / "reference").mkdir(parents=True)
    (run / "data" / "islands" / "hits").mkdir(parents=True)
    (run / "stages").mkdir()
    (run / "record.json").write_text(json.dumps(dict(env_snapshot={"diamond_version.txt": "d1"}, taxdump_sha256="tax1")))
    cfg = dict(islands=dict(store=str(store), max_target_seqs=4000, threads=12), search=dict(sensitivity="sensitive", evalue=1e-5, block_size=4, nearest=32,
                                                                                         keep_sseq=32, threads=12))
    for k, v in (cfg_extra or {}).items():
        cfg[k].update(v)
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    (run / "data" / "reference" / "manifest.json").write_text(json.dumps(dict(sha256="ref1")))
    entries, genomes, folders = [], [], {}
    for genus, acc in accs:
        folder = run / "data" / "islands" / "inputs" / acc
        folder.mkdir(parents=True)
        (folder / "proteins.faa").write_text((proteins or {}).get(acc, ">%s__g1\nMK\n" % acc))
        hp = run / "data" / "islands" / "hits" / (acc + ".npz")
        np.savez(hp, off=np.array([0, 1]), nraw=np.array([1]))
        entries.append(dict(id=acc, files=[dict(path="data/islands/inputs/%s/proteins.faa" % acc, sha256=sha256_file(str(folder / "proteins.faa"), use_sidecar=False)),
                                           dict(path="data/islands/hits/%s.npz" % acc, sha256=sha256_file(str(hp), use_sidecar=False))]))
        genomes.append(dict(id=acc, genus=genus, taxid=562, ranks=[1, 2, 3, 4, 561, 562], hidden_taxids=[1], hidden_rows_dropped=3))
        folders[acc] = str(folder)
    mp, ip = run / "data" / "islands" / "manifest.json", run / "data" / "islands" / "index.json"
    mp.write_text(json.dumps(dict(entries=entries)))
    ip.write_text(json.dumps(dict(genomes=genomes)))
    rep = dict(status="passed", outputs=[dict(name="islands_manifest", sha256=sha256_file(str(mp), use_sidecar=False)),
                                         dict(name="islands_index", sha256=sha256_file(str(ip), use_sidecar=False))])
    if version is not None:
        rep["search_version"] = version
    (run / "stages" / "islands_data.json").write_text(json.dumps(rep))
    code = root / "wt" / name / "ci" / "stages"
    code.mkdir(parents=True)
    shutil.copy(I.SEARCH_CODE, code / "search.py")
    (code / "islands.py").write_text(islands_code if islands_code is not None else "# islands.py of the candidate\n")
    return run, cfg, genomes, folders


def test_island_search_reuse_matches_only_an_equal_key_and_inputs(tmp_path):
    import json
    accs = [("Escherichia", "NC_1"), ("Salmonella", "NC_2")]
    store = _store(tmp_path, accs)
    cand, _, _, _ = _island_run(tmp_path, "cand", store, accs)
    ours, cfg, genomes, folders = _island_run(tmp_path, "ours", store, accs, version=None)
    cfg["islands"]["reuse_search_from"] = [str(cand)]
    key, inp = I.search_key(str(ours), cfg, accs, I.ISLAND_SEARCH_VERSION, I.SEARCH_CODE), I.search_inputs(genomes, folders)
    found = list(I.search_candidates(str(ours), cfg))
    assert [c for c, k, ki, _, _ in found if k == key and ki == inp] == [str(cand)]
    # the reused hit files are linked and checked; the dropped rows come with them
    man = json.loads((cand / "data" / "islands" / "manifest.json").read_text())
    idx = json.loads((cand / "data" / "islands" / "index.json").read_text())
    new = tmp_path / "new"
    dropped = I.link_reused_hits(str(new), str(cand), man, idx, ["NC_1", "NC_2"])
    assert dropped == {"NC_1": 3, "NC_2": 3} and (new / "data" / "islands" / "hits" / "NC_1.npz").exists()
    try:
        I.link_reused_hits(str(tmp_path / "new2"), str(cand), man, idx, ["NC_1"])
        raise AssertionError("other chromosomes accepted")
    except SystemExit:
        pass
    # another setting (threads included), another order of the queries, or other query proteins: no reuse
    for name, extra in (("other", dict(islands=dict(max_target_seqs=1000))), ("threads", dict(islands=dict(threads=6)))):
        other, _, _, _ = _island_run(tmp_path, name, store, accs, cfg_extra=extra)
        cfg["islands"]["reuse_search_from"] = [str(other)]
        assert all(k != key for _, k, _, _, _ in I.search_candidates(str(ours), cfg))
    flipped, _, _, _ = _island_run(tmp_path, "flipped", store, accs[::-1])
    cfg["islands"]["reuse_search_from"] = [str(flipped)]
    assert all(k != key for _, k, _, _, _ in I.search_candidates(str(ours), cfg))
    changed, _, _, _ = _island_run(tmp_path, "changed", store, accs, proteins={"NC_2": ">NC_2__g1\nMKV\n"})
    cfg["islands"]["reuse_search_from"] = [str(changed)]
    assert all(not (k == key and ki == inp) for _, k, ki, _, _ in I.search_candidates(str(ours), cfg))
    # a candidate with a file missing is skipped, not a crash
    (changed / "record.json").unlink()
    assert list(I.search_candidates(str(ours), cfg)) == []


def test_island_search_reuse_needs_a_known_search_version(tmp_path):
    import hashlib
    accs = [("Yersinia", "NC_3")]
    store = _store(tmp_path, accs)
    unknown, cfg, _, _ = _island_run(tmp_path, "unknown", store, accs, version=None)    # no version, code not version 1
    cfg["islands"]["reuse_search_from"] = [str(unknown)]
    assert list(I.search_candidates(str(unknown), cfg)) == []
    old, cfg2, _, _ = _island_run(tmp_path, "old", store, accs, version=None, islands_code="# v1\n")
    h = hashlib.sha256(b"# v1\n").hexdigest()
    I.SEARCH_V1_FILES.add(h)
    try:
        cfg2["islands"]["reuse_search_from"] = [str(old)]
        assert [k["version"] for _, k, _, _, _ in I.search_candidates(str(old), cfg2)] == [1]
    finally:
        I.SEARCH_V1_FILES.discard(h)



def test_kmer_and_pair_comparators(tmp_path):
    """cmp_kmer points to the gene of an odd composition; cmp_pairnull to a gene far closer to a partner genome than
    the chromosome's genes usually are; cmp_triplet to a gene whose partner order contradicts the chromosome's."""
    g = object.__new__(S.Genome)
    rng = np.random.default_rng(0)
    n = 30
    cod = [rng.integers(1, 33, 60) for _ in range(n)]                 # codons of half the table (A and C first bases)
    cod[7] = rng.integers(33, 65, 60)                                  # an odd gene (G and T first bases)
    g.codons = np.concatenate(cod).astype(np.uint8)
    g.coff = np.concatenate([[0], np.cumsum([len(c) for c in cod])])
    g.n = n
    k = I.kmer_distance(g)
    assert int(np.nanargmax(k)) == 7
    # partners: genome 0 is closer to the chromosome than genome 1 for every gene but gene 3 (closest to genome 1) and
    # gene 5 (far closer to genome 1 than any other gene)
    off, tg, bits = [0], [], []
    for i in range(n):
        s0, s1 = 1.0 + 0.01 * i, 0.5 + 0.01 * (i % 5)
        if i == 3:
            s0, s1 = 0.6, 0.9
        if i == 5:
            s1 = 3.0
        tg += [0, 1]
        bits += [s0 * 60, s1 * 60]
        off.append(len(tg))
    np.savez(tmp_path / "p.npz", off=np.array(off), tgenome=np.array(tg), bits=np.array(bits, np.float32))
    pc = I.pair_comparators(str(tmp_path / "p.npz"), g)
    assert int(np.nanargmax(pc["cmp_pairnull"])) == 5
    assert pc["cmp_triplet"][3] == 1.0 and pc["cmp_triplet"][0] == 0.0


def test_protocol_measures_what_known_features_leave():
    """A score that only repeats a known feature has no residual signal and no combined gain; an independent signal
    keeps both."""
    from ci.stages import protocol as PR
    rng = np.random.default_rng(1)
    table = {}
    for c, unit in enumerate(["U1", "U1", "U2", "U3"]):
        m = 600
        y = (rng.random(m) < 0.1).astype(int)
        gc3 = rng.normal(0, 1, m) + 1.0 * y
        sig = rng.normal(0, 1, m) + 1.5 * y
        cols = dict(cmp_gc3=gc3, cmp_codon=rng.normal(0, 1, m), cmp_kmer=rng.normal(0, 1, m), cmp_hits=-rng.integers(0, 30, m).astype(float),
                    cmp_short=-rng.integers(50, 500, m).astype(float), copy=gc3 + rng.normal(0, 0.01, m), new=sig, cmp_pairnull=np.full(m, np.nan),
                    cmp_triplet=np.full(m, np.nan))
        table["c%d" % c] = dict(y=y, cols=cols, unit=unit)
    ev = PR.evaluate(table, ["U1", "U2", "U3"], ["copy", "new"], ["cmp_gc3", "cmp_codon", "cmp_kmer", "cmp_hits", "cmp_short"])
    assert ev["copy"]["auroc"] > 0.7 and abs(ev["copy"]["residual_auroc"] - 0.5) < 0.08 and abs(ev["copy"]["combined_gain"]) < 0.02
    assert ev["new"]["residual_auroc"] > 0.75 and ev["new"]["combined_gain"] > 0.05
    sp = PR.spearman_matrix(table, ["copy", "new", "cmp_gc3"])
    assert PR.collapsed(sp, ["copy", "new"], ["cmp_gc3"]) == {"copy": ["cmp_gc3"]}
    assert PR.auroc([1.0, np.nan], [0.5]) == 0.5                      # an unscored positive ranks below the negatives
